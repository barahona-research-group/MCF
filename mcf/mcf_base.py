"""Code for Multiscale Clustering Filtration (MCF)."""

import itertools
import gudhi as gd
import numpy as np

from tqdm import tqdm

from mcf.io import save_results
from mcf.measures import (
    compute_bettis,
    compute_partition_size,
    compute_persistent_hierarchy,
    compute_persistent_conflict,
)
from mcf.utils import node_id_to_dict, _cluster_id_preprocessing
from mcf.plotting import plot_sankey, plot_pd


class MCF:
    """Main class to construct MCF from a sequence of partitions and analyse
    its persistent homology."""

    def __init__(self, method="standard", max_dim=3):
        """Initialise MCF object.

        Parameters:
            max_dim (int): Maximum dimension of simplices considered
                in filtration, between 1 and 3.

            method (str): Method to construct the MCF. Both methods lead to the
                same persistent homology, see our paper.
                - 'standard': Standard method where nodes in MCF correspond
                    to points. Faster when the number of points is smaller
                    than the total number of distinct clusters.
                - 'nerve': Nerve-based method where nodes in MCF correspond to
                    clusters. Faster when the total number of distinct clusters
                    is smaller than the number of points.
        """

        # initialise sequence of partitions
        self.partitions = []
        self.filtration_indices = []

        # set max dimension of filtration
        self.max_dim = min(3, max_dim)

        # set method to construct filtration, either standard or nerve-based
        self.method = method

        # initialise for gudhi
        self.filtration_gudhi = gd.SimplexTree()

        # initialise persistent homology attribute
        self.persistence = []

    @property
    def n_partitions(self):
        """ "Computes number of partitions in sequence."""
        return len(self.partitions)
    
    @property
    def n_simplices(self):
        """Computes number of simplices in MCF."""
        return self.filtration_gudhi.num_simplices()

    def load_data(self, partitions, filtration_indices=[]):
        """Method to load sequence of partitions and
        filtration indices."""
        self.partitions = partitions

        if len(filtration_indices) > 0:
            self.filtration_indices = filtration_indices
        else:
            # if no filtration indices are given use enumeration
            self.filtration_indices = np.arange(1, self.n_partitions + 1)

    def _build_filtration_standard(self, tqdm_disable=False):
        """Construct MCF via standard method."""
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

    def _build_filtration_nerve(self, tqdm_disable=False):
        """Construct MCF via nerve-based method."""

        # initialise simplex tree
        self.filtration_gudhi = gd.SimplexTree()

        # we compute cluster indices of new clusters per partition
        # and a dictionary that maps cluster indices to sets
        partitions_c_ind, ind_to_c = _cluster_id_preprocessing(self.partitions)

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
                        if not c.isdisjoint(triangle_intersection):
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
                        if not c.isdisjoint(edge_intersection):
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
                    if not c.isdisjoint(node_intersection):
                        edge = node_ind + [c_ind]
                        # insert edge into simplex tree
                        self.filtration_gudhi.insert(edge, filtration=t)
                        # add edge to set
                        edges.append(edge)

                # add nodes
                node = [c_ind]
                self.filtration_gudhi.insert(node, filtration=t)
                nodes.append(node)

    def build_filtration(self, tqdm_disable=False):
        """Build MCF filtration."""

        if self.method == "standard":
            self._build_filtration_standard(tqdm_disable=tqdm_disable)

        elif self.method == "nerve":
            self._build_filtration_nerve(tqdm_disable=tqdm_disable)

    def compute_persistence(self):
        """Compute persistent homology of MCF using GUDHI."""

        # compute persistence with GUDHI (over 2 element field)
        self.filtration_gudhi.persistence(homology_coeff_field=2)

        # summarise persistence
        self.persistence = []
        for i in range(self.max_dim):
            PD = self.filtration_gudhi.persistence_intervals_in_dimension(i)
            self.persistence.append(PD)

    def plot_pd(self, alpha=0.5, marker_size=None):
        """Plot MCF persistence diagram."""
        return plot_pd(self, alpha, marker_size)

    def plot_sankey(self, step=1, color=True, alpha=0.5, pad=0.1, thickness=1):
        """Plot Sankey diagram of partitions."""
        return plot_sankey(self, step, color, alpha, pad, thickness)

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

    def compute_all_measures(
        self,
        file_path="mcf_results.pkl",
        tqdm_disable=False,
    ):
        """Construct MCF, compute PH and compute all derived measures."""

        # build filtration
        self.build_filtration(tqdm_disable=tqdm_disable)

        # compute persistent homology
        self.compute_persistence()

        # obtain persistence
        persistence = [
            self.filtration_gudhi.persistence_intervals_in_dimension(dim)
            for dim in range(self.max_dim)
        ]

        # compute Betti numbers
        betti_0, betti_1, betti_2 = self.compute_bettis()

        # compute size of partitions
        s_partitions = self.compute_partition_size()

        # compute persistent hierarchy
        h, h_bar = self.compute_persistent_hierarchy()

        # compute persistent conflict
        c_1, c_2, c = self.compute_persistent_conflict()

        # compile results dictionary
        mcf_results = {}
        mcf_results["n_simplices"] = self.n_simplices
        mcf_results["filtration_indices"] = self.filtration_indices
        mcf_results["max_dim"] = self.max_dim
        mcf_results["method"] = self.method
        mcf_results["persistence"] = persistence
        mcf_results["betti_0"] = betti_0
        mcf_results["betti_1"] = betti_1
        mcf_results["betti_2"] = betti_2
        mcf_results["s_partitions"] = s_partitions
        mcf_results["h"] = h
        mcf_results["h_bar"] = h_bar
        mcf_results["c_1"] = c_1
        mcf_results["c_2"] = c_2
        mcf_results["c"] = c

        if file_path is not None:
            # save results
            save_results(mcf_results, file_path)

        return mcf_results
