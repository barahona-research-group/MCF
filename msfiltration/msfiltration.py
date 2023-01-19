import gudhi as gd
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pygenstability import run
from skimage.feature import peak_local_max
from tqdm import tqdm

from msfiltration.scale_selection import select_scales_gaps
from msfiltration.utils import node_id_to_dict


class MSF:
    def __init__(self):

        # initialise adjacency matrix
        self.graph = None

        # initialise simplex tree
        self.filtration = gd.SimplexTree()

        # initialise Markov Stabiliy attributes
        self.ms_results = None
        self.community_ids = None
        self.log_scales = None
        self.n_scales = None

        # initialise persistent homology attributes
        self.max_dim = None
        self.persistence = None

        # initial optimal scales attributes
        self.optimal_scales = []
        self.gap_width = None

    def markov_stability_analysis(
        self,
        graph,
        min_scale=-1,
        max_scale=1,
        n_scale=50,
        n_workers=4,
        constructor="continuous_normalized",
        with_postprocessing=True,
        with_ttprime=False,
        with_optimal_scales=False,
    ):
        # store graph as attribute
        self.graph = graph

        # apply Markov Stability analysis
        print("Running Markov Stability analysis ... ")
        self.ms_results = run(
            self.graph,
            constructor=constructor,
            min_scale=min_scale,
            max_scale=max_scale,
            n_scale=n_scale,
            n_workers=n_workers,
            with_postprocessing=with_postprocessing,
            with_ttprime=with_ttprime,
            with_optimal_scales=with_optimal_scales,
        )

        # get community assignments
        self.community_ids = self.ms_results["community_id"]
        # get log_scales
        self.log_scales = np.log10(self.ms_results["scales"])
        # get number of scales
        self.n_scales = len(self.log_scales)

    def load_ms_results(self, ms_results):
        # store MS results as attribute
        self.ms_results = ms_results
        # get community assignments
        self.community_ids = self.ms_results["community_id"]
        # get log_scales
        self.log_scales = np.log10(self.ms_results["scales"])
        # get number of scales
        self.n_scales = len(self.log_scales)

    def build_filtration(self, max_dim=4):
        """
        The filtration is built from the community assignments stored in self.community_ids
        """
        # define max_dim of filtration
        self.max_dim = max_dim

        # store all communities to later avoid repetitious computations
        all_communities = set()

        print("Building filtration ...")
        for t in tqdm(range(len(self.log_scales))):

            # continue if partition at scale t has appeared before
            is_repetition = False
            for s in range(t - 1, -1, -1):
                if np.array_equal(self.community_ids[s], self.community_ids[t]):
                    is_repetition = True
                    break
            if is_repetition:
                continue

            # add communities at scale t as simplices to tree
            for community in node_id_to_dict(self.community_ids[t]).values():
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
                    community, min(self.max_dim, s_community)
                ):
                    self.filtration.insert(list(face), filtration=self.log_scales[t])

    def compute_persistence(self):
        print("Computing persistence ... ")
        self.persistence = self.filtration.persistence()

    def fit(
        self,
        graph,
        min_scale=-1,
        max_scale=1,
        n_scale=50,
        n_workers=4,
        constructor="continuous_normalized",
        max_dim=4,
        with_postprocessing=True,
        with_ttprime=False,
        with_optimal_scales=False,
    ):

        # apply Markov Stability analysis
        self.markov_stability_analysis(
            graph,
            min_scale,
            max_scale,
            n_scale,
            n_workers,
            constructor,
            with_postprocessing,
            with_ttprime,
            with_optimal_scales,
        )

        # build filtration
        self.build_filtration(max_dim)

    def transform(self):
        self.compute_persistence()

    def fit_transform(
        self,
        graph,
        min_scale=-1,
        max_scale=1,
        n_scale=50,
        n_workers=4,
        constructor="continuous_normalized",
        max_dim=4,
        with_postprocessing=True,
        with_ttprime=False,
        with_optimal_scales=False,
    ):

        self.fit(
            graph,
            min_scale,
            max_scale,
            n_scale,
            n_workers,
            constructor,
            max_dim,
            with_postprocessing,
            with_ttprime,
            with_optimal_scales,
        )

        self.transform()

    def select_scales(self, threshold_abs=0.2, min_gap_width=0.2, with_plot=False):

        # get set of deaths
        deaths = np.asarray(
            [self.persistence[i][1][1] for i in range(len(self.persistence))]
        )

        if with_plot:

            self.optimal_scales, self.gap_width, ax = select_scales_gaps(
                deaths, self.log_scales, threshold_abs, min_gap_width, True
            )

            return ax

        else:

            self.optimal_scales, self.gap_width = select_scales_gaps(
                deaths, self.log_scales, threshold_abs, min_gap_width, False
            )

    def plot_persistence_diagram(self, alpha=0.5):
        """
        code is a modified version of the GUDHI's plot_persistence_diagram
        """

        # obtain min and max values and define value for infinity
        tmin = self.log_scales[0]
        tmax = self.log_scales[-1]
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
        for dim in range(self.max_dim - 1):
            persistences = self.filtration.persistence_intervals_in_dimension(dim)
            if len(persistences) > 0:
                ax.scatter(
                    persistences[:, 0],
                    np.nan_to_num(persistences[:, 1], posinf=infinity),
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
                self.log_scales[self.optimal_scales],
                self.log_scales[0] - 1,
                self.log_scales[-1] + 1,
                color="gold",
                label="Optimal scales",
            )

        ax.legend(loc=4)

        return ax

