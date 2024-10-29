"""Code for MCF and MCNF."""

import itertools
import gudhi as gd
import numpy as np

from tqdm import tqdm

from mcf.measures import (
    compute_bettis,
    compute_partition_size,
    compute_persistent_hierarchy,
    compute_persistent_conflict,
)
from mcf.utils import node_id_to_dict
from mcf.plotting import plot_sankey, plot_pd


class MCF:
    """Main class to construct MCF from a sequence of
    partitions and analyse its persistent homology."""

    def __init__(self):

        # initialise sequence of partitions
        self.partitions = []
        self.filtration_indices = []
        self.max_dim = 3

        # initialise for gudhi
        self.filtration_gudhi = None

        # initialise persistent homology attribute
        self.persistence = []

    @property
    def n_partitions(self):
        """ "Computes number of partitions in sequence."""
        return len(self.partitions)

    def load_data(self, partitions, filtration_indices=[]):
        """Method to load sequence of partitions and
        filtration indices."""
        self.partitions = partitions

        if len(filtration_indices) > 0:
            self.filtration_indices = filtration_indices
        else:
            # if no filtration indices are given use enumeration
            self.filtration_indices = np.arange(1, self.n_partitions + 1)

    def build_filtration(self, max_dim=3, tqdm_disable=False):
        """Build MCF filtration."""

        # define max_dim of filtration
        self.max_dim = min(3, max_dim)

        # initialise simplex tree
        self.filtration_gudhi = gd.SimplexTree()

        # store all communities to later avoid repetitious computations
        all_communities = set()

        for t in tqdm(range(len(self.filtration_indices)), disable=tqdm_disable):

            # continue if partition at scale t has appeared before
            is_repetition = False
            for s in range(t - 1, -1, -1):
                if np.array_equal(self.partitions[s], self.partitions[t]):
                    is_repetition = True
                    break
            if is_repetition:
                continue

            # add communities at scale t as simplices to tree
            for community in node_id_to_dict(self.partitions[t]).values():
                # continue if community has been added before
                if community in all_communities:
                    continue
                # add community to set of all communities
                else:
                    all_communities.add(community)
                # compute size of community
                s_community = len(community)
                # cover community by max_dim-simplices when community is larger than max_dim
                for face in itertools.combinations(
                    community, min(self.max_dim + 1, s_community)
                ):
                    self.filtration_gudhi.insert(
                        list(face), filtration=self.filtration_indices[t]
                    )

    def compute_persistence(self):
        """Compute persistent homology of MCF."""

        # compute persistence with GUDHI (over 2 element field)
        self.filtration_gudhi.persistence(homology_coeff_field=2)

        # summarise persistence
        self.persistence = []
        for i in range(self.max_dim):
            PD = self.filtration_gudhi.persistence_intervals_in_dimension(i)
            self.persistence.append(PD)

    def plot_pd(self, alpha=0.5):
        """Plot MCF persistence diagram."""
        return plot_pd(self, alpha)

    def plot_sankey(self, step=1, color=True, alpha=0.5):
        """Plot Sankey diagram of partitions."""
        return plot_sankey(self, step, color, alpha)

    def compute_bettis(self):
        """Compute Betti curves of MCF."""
        betti_0, betti_1, betti_2 = compute_bettis(self)
        return betti_0, betti_1, betti_2

    def compute_partition_size(self):
        """Compute size of partitions."""
        s_partitions = compute_partition_size(self)
        return s_partitions

    def compute_persistent_hierarchy(self):
        """Compute persistent hierarchy."""
        h, h_bar = compute_persistent_hierarchy(self)
        return h, h_bar

    def compute_persistent_conflict(self):
        """Compute persistent conflict."""
        c_1, c_2, c = compute_persistent_conflict(self)
        return c_1, c_2, c


class MCNF(MCF):
    """Class to construct MCNF from a sequence of partitions using equivalent
    nerve-based construction and analyse its persistent homology. The only
    difference from MCF class is the construction of filtration."""

    def build_filtration(self, max_dim=3, tqdm_disable=False):
        """Build MCNF filtration."""

        # define max_dim of filtration
        self.max_dim = min(3, max_dim)

        # initialise simplex tree
        self.filtration_gudhi = gd.SimplexTree()

        # we store cluster indices of new clusters per partition
        partitions_c_ind = []
        # and a dictionary that maps cluster indices to sets
        ind_to_c = {}

        # store clusters seen already
        all_clusters = set()
        n_clusters_total = 0

        # iterate through all partitions in sequence
        for partition in self.partitions:
            n_clusters_before = n_clusters_total
            # get all cluster indices in partition
            cluster_values = np.unique(partition)
            # iterate through all clusters in partitions
            for i in cluster_values:
                # define cluster as a set
                c = frozenset(np.argwhere(partition == i).flatten())
                # check if cluster is new
                if c in all_clusters:
                    continue
                else:
                    # add cluster if new
                    all_clusters.add(c)
                    ind_to_c[n_clusters_total] = c
                    n_clusters_total += 1
            # store all indices of new clusters per partition
            partitions_c_ind.append(np.arange(n_clusters_before, n_clusters_total))

        # initialise simplices of different dimensions
        nodes = list()
        edges = list()
        triangles = list()

        # iterate through filtration indices
        for i, t in tqdm(
            enumerate(self.filtration_indices),
            total=len(self.filtration_indices),
            disable=tqdm_disable,
        ):
            # get new cluster indices
            c_ind_new = partitions_c_ind[i]
            # iterate through indices
            for c_ind in c_ind_new:
                c = ind_to_c[c_ind]

                # add tetrahedra
                if self.max_dim > 2:
                    for triangle_ind in triangles:
                        # get intersection of clusters corresponding to triangle
                        triangle_intersection = (
                            ind_to_c[triangle_ind[0]]
                            .intersection(ind_to_c[triangle_ind[1]])
                            .intersection(ind_to_c[triangle_ind[2]])
                        )
                        # check if new cluster intersects with triangle
                        if len(c.intersection(triangle_intersection)) > 0:
                            tetrahedron = triangle_ind + [c_ind]
                            # insert tetrahedron into simplex tree
                            self.filtration_gudhi.insert(tetrahedron, filtration=t)

                # add triangles
                if self.max_dim > 1:
                    for edge_ind in edges:
                        # get intersection of clusters corresponding to edge
                        edge_intersection = ind_to_c[edge_ind[0]].intersection(
                            ind_to_c[edge_ind[1]]
                        )
                        # check if new cluster intersects with edge
                        if len(c.intersection(edge_intersection)) > 0:
                            triangle = edge_ind + [c_ind]
                            # insert triangle into simplex tree
                            self.filtration_gudhi.insert(triangle, filtration=t)
                            # add triangle to set
                            triangles.append(triangle)

                # add edges
                for node_ind in nodes:
                    # get cluster corresponding to node
                    node_intersection = ind_to_c[node_ind[0]]
                    # check if new cluster intersects with node
                    if len(c.intersection(node_intersection)) > 0:
                        edge = node_ind + [c_ind]
                        # insert edge into simplex tree
                        self.filtration_gudhi.insert(edge, filtration=t)
                        # add edge to set
                        edges.append(edge)

                # add nodes
                node = [c_ind]
                self.filtration_gudhi.insert(node, filtration=t)
                nodes.append(node)
