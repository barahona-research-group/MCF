import gudhi as gd
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pygenstability import run

# from skimage.feature import peak_local_max
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
        """
        Method to load partitions and filtration indices
        """
        self.partitions = partitions
        self.n_partitions = len(partitions)
        self.filtration_indices = filtration_indices

    def load_ms_results(self, ms_results):
        """Method to directly load results from PyGenStability"""
        # get community assignments
        self.partitions = ms_results["community_id"]
        # get log_scales
        self.filtration_indices = np.log10(ms_results["scales"])
        # get number of scales
        self.n_partitions = len(self.filtration_indices)

    def load_julia(self):
        """
        Method to load Julia code
        """

        if self.jl == None:

            print("Loading Julia ...")

            from julia.api import Julia

            jpath = "/home/tc/julia-1.8.5/bin/julia"
            self.jl = Julia(runtime=jpath, compiled_modules=False)

            # this line takes a very long time and throws warnings:
            # warning: unknown data in line table prologue at offset 0x00010337:
            # parsing ended (at offset 0x0001057d) before reaching the prologue end at offset 0x00010bec
            self.jl.eval(
                'include("/home/tc/OneDrive/Software/MSFiltration/msfiltration/MCF.jl")'
            )

    def build_filtration(self, max_dim=3, method="gudhi"):
        """
        The filtration is built from the community assignments stored in self.community_ids
        """

        # define max_dim of filtration
        self.max_dim = min(3, max_dim)

        if method == "gudhi":
            self.build_filtration_gudhi()

        elif method == "eirene":
            self.build_filtration_eirene()

    def build_filtration_gudhi(self):
        """
        The filtration is built from the community assignments stored in self.community_ids
        """

        print("Constructing filtration with GUDHI ...")

        # initialise simplex tree
        self.filtration_gudhi = gd.SimplexTree()

        # store all communities to later avoid repetitious computations
        all_communities = set()

        for t in tqdm(range(len(self.filtration_indices))):

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

    def build_filtration_eirene(self, max_dim=3):

        # load Julia
        self.load_julia()
        from julia import Main

        print("Constructing filtration with Eirene ...")

        # pass data to Julia
        Main.partitions = self.partitions
        Main.filtration_indices = self.filtration_indices
        Main.max_dim = self.max_dim

        # construct MCF
        rv, cp, dv, fv = self.jl.eval(
            "Construct_MCF(partitions,filtration_indices,max_dim)"
        )
        self.filtration_eirene = (rv, cp, dv, fv)

    def compute_persistence(self, method="gudhi"):

        if method == "gudhi":
            self.compute_persistence_gudhi()
        elif method == "eirene":
            self.compute_persistence_eirene()

    def compute_persistence_gudhi(self):

        print("Computing persistence with GUDHI ... ")

        # compute persistence with GUDHI (over 2 element field)
        self.filtration_gudhi.persistence(homology_coeff_field=2)

        # summarise persistence
        self.persistence = []
        for i in range(self.max_dim):
            PD = self.filtration_gudhi.persistence_intervals_in_dimension(i)
            self.persistence.append(PD)

    def compute_persistence_eirene(self):

        # load Julia
        self.load_julia()
        from julia import Main

        print("Computing persistence with Eirene ... ")

        # pass data to Julia
        Main.rv = self.filtration_eirene[0]
        Main.cp = self.filtration_eirene[1]
        Main.dv = self.filtration_eirene[2]
        Main.fv = self.filtration_eirene[3]
        Main.max_dim = self.max_dim

        # compute persistence with Eirene
        PD0, PD1, PD2, CR0, CR1, CR2 = self.jl.eval("Compute_PH(rv,cp,dv,fv,max_dim)")

        # summarise persistence
        self.persistence = [PD0, PD1, PD2]

        # summarise class rep
        self.class_rep = [CR0, CR1, CR2]

    def plot_persistence_diagram(self, alpha=0.5):
        """plot persistence diagram"""
        return plot_persistence_diagram(self, alpha)

    def plot_sankey(self, step=1, color=True, alpha=0.5):
        """Plot Sankey diagram of partitions"""
        return plot_sankey(self, step, color, alpha)

    def compute_bettis(self):
        """compute Betti numbers"""
        betti_0, betti_1, betti_2 = compute_bettis(self)
        return betti_0, betti_1, betti_2

    def compute_partition_size(self):
        """compute size of partitions"""
        s_partitions = compute_partition_size(self)
        return s_partitions

    def compute_persistent_hierarchy(self):
        """compute persistent hierarchy"""
        h, h_bar = compute_persistent_hierarchy(self)
        print("Average persistent hierarchy:", np.around(h_bar, 3))
        return h, h_bar

    def compute_persistent_conflict(self):
        """compute persistent conflict"""
        c_1, c_2, c = compute_persistent_conflict(self)
        return c_1, c_2, c
