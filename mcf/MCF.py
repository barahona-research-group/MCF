import gudhi as gd
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pygenstability import run

# from skimage.feature import peak_local_max
from tqdm import tqdm

from msfiltration.scale_selection import select_scales_gaps
from msfiltration.utils import node_id_to_dict


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

        # initial optimal scales attributes
        self.optimal_scales = []
        self.gap_width = None

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
        self.partitions = self.ms_results["community_id"]
        # get log_scales
        self.filtration_indices = np.log10(self.ms_results["scales"])
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

        # # summarise persistence for different dimensions
        # self.persistence = []
        # for i, PD in enumerate([PD2, PD1, PD0]):
        #     if len(PD) > 0:
        #         for birth_death in PD:
        #             self.persistence.append((2 - i, (birth_death[0], birth_death[1])))

        # summarise persistence
        self.persistence = [PD0, PD1, PD2]

        # summarise class rep
        self.class_rep = [CR0, CR1, CR2]

    def select_scales(self, threshold_abs=0.2, min_gap_width=0.2, with_plot=False):

        # get set of deaths
        deaths = np.asarray(
            [self.persistence[i][1][1] for i in range(len(self.persistence))]
        )

        if with_plot:

            self.optimal_scales, self.gap_width, ax = select_scales_gaps(
                deaths, self.filtration_indices, threshold_abs, min_gap_width, True
            )

            return ax

        else:

            self.optimal_scales, self.gap_width = select_scales_gaps(
                deaths, self.filtration_indices, threshold_abs, min_gap_width, False
            )

    def plot_persistence_diagram(self, alpha=0.5):
        """
        code is a modified version of the GUDHI's plot_persistence_diagram
        -> move to plotting.py
        """

        # obtain min and max values and define value for infinity
        tmin = self.filtration_indices[0]
        tmax = self.filtration_indices[-1]
        delta = 0.1 * abs(tmax - tmin)
        infinity = tmax + delta

        # font size
        # plt.rcParams.update({"font.size": 20})

        # create axis
        fig, ax = plt.subplots(1, figsize=(8, 7))

        # define colormap
        colormap = plt.cm.Set1.colors

        # infinity line
        ax.plot(
            [tmin - 0.5 * delta, tmax],
            [infinity, infinity],
            linewidth=1.0,
            color="k",
            alpha=0.5,
        )

        # plot persistences
        for dim, PD in enumerate(self.persistence):
            if len(PD) > 0:
                ax.scatter(
                    PD[:, 0],
                    np.nan_to_num(PD[:, 1], posinf=infinity),
                    color=colormap[dim],
                    alpha=alpha,
                    label=r"$H_{}$".format(dim),
                )

        # plot top line
        ax.plot([tmin - 0.5 * delta, tmax], [tmax, tmax], linewidth=1.0, color="k")

        # plot diag
        ax.plot([tmin, tmax], [tmin, tmax], linewidth=1.0, color="k")

        # plot lower diag patch
        ax.add_patch(
            mpatches.Polygon(
                [[tmin, tmin], [tmax, tmin], [tmax, tmax]], fill=True, color="lightgrey"
            )
        )

        # labels and axes limits
        ax.set(
            xlabel="Birth",
            ylabel="Death",
            xlim=(tmin - 0.5 * delta, tmax),
            ylim=(tmin, infinity + 0.5 * delta),
        )

        # Infinity and y-axis label
        yt = ax.get_yticks()
        yt = yt[np.where(yt <= tmax)]  # to avoid ploting ticklabel higher than infinity
        yt = np.append(yt, infinity)
        ytl = ["%.2f" % e for e in yt]  # to avoid float precision error
        ytl[-1] = r"$+\infty$"
        ax.set_yticks(yt)
        ax.set_yticklabels(ytl)

        # x-axis label
        ax.set_xticks(yt[:-1])
        ax.set_xticklabels(ytl[:-1])

        # plot optimal scales
        if len(self.optimal_scales) > 0:
            ax.hlines(
                self.filtration_indices[self.optimal_scales],
                self.filtration_indices[0] - 1,
                self.filtration_indices[-1] + 1,
                color="gold",
                label="Optimal scales",
            )

        ax.legend(loc=4)

        return ax

