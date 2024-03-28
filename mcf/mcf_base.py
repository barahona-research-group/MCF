"""Code for MCF."""

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
from mcf.plotting import plot_sankey, plot_persistence_diagram


class MCF:
    """Main class to construct MCF from a sequence of
    partitions and analyse its persistent homology."""

    def __init__(self):

        # initialise sequence of partitions
        self.partitions = []
        self.n_partitions = 0
        self.filtration_indices = []
        self.max_dim = 3

        # initialise for gudhi
        self.filtration_gudhi = None

        # initialise for eirene
        self.jl = None
        self.filtration_eirene = None

        # initialise persistent homology attributes
        self.persistence = []
        self.class_rep = []

    def load_data(self, partitions, filtration_indices=[]):
        """Method to load sequence of partitions and
        filtration indices."""
        self.partitions = partitions
        self.n_partitions = len(partitions)
        if len(filtration_indices):
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

    def plot_persistence_diagram(self, alpha=0.5):
        """Plot MCF persistence diagram."""
        return plot_persistence_diagram(self, alpha)

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
