import gudhi as gd
import itertools
import matplotlib.pyplot as plt
import numpy as np

from pygenstability import run
from skimage.feature import peak_local_max
from tqdm import tqdm

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
        self.log_times = None

        # initialise persistent homology attributes
        self.persistence = None
        self.optimal_scales = None

    def markov_stability_analysis(
        self,
        graph,
        min_time=-1,
        max_time=1,
        n_time=50,
        constructor="continuous_normalized",
        with_postprocessing=True,
        with_ttprime=False,
        with_optimal_scales=False,
    ):

        self.graph = graph

        # apply Markov Stability analysis
        print("Running Markov Stability analysis ... ")
        self.ms_results = run(
            self.graph,
            constructor=constructor,
            min_time=min_time,
            max_time=max_time,
            n_time=n_time,
            with_postprocessing=with_postprocessing,
            with_ttprime=with_ttprime,
            with_optimal_scales=with_optimal_scales,
        )

    def load_ms_results(self, ms_results):
        self.ms_results = ms_results

    def build_filtration(self, max_dim=4):
        # get community assignments
        self.community_ids = self.ms_results["community_id"]
        # get log_times
        self.log_times = np.log10(self.ms_results["times"])

        # add communities at scale t as simplices to tree
        print("Building filtration ...")
        for t in tqdm(range(len(self.log_times))):
            for community in node_id_to_dict(self.community_ids[t]).values():
                # only add simplices of dimension smaller than max_dim
                for face in itertools.combinations(community, max_dim):
                    self.filtration.insert(list(face), filtration=self.log_times[t])

    def compute_persistence(self):
        print("Computing persistence ... ")
        self.persistence = self.filtration.persistence()

    def fit(
        self,
        graph,
        min_time=-1,
        max_time=1,
        n_time=50,
        constructor="continuous_normalized",
        max_dim=4,
    ):

        # apply Markov Stability analysis
        self.markov_stability_analysis(
            graph, min_time, max_time, n_time, constructor,
        )

        # build filtration
        self.build_filtration(max_dim)

    def transform(self):
        self.compute_persistence()

    def fit_transform(
        self,
        graph,
        min_time=-1,
        max_time=1,
        n_time=50,
        constructor="continuous_normalized",
        max_dim=4,
    ):

        self.fit(
            graph, min_time, max_time, n_time, constructor, max_dim,
        )

        self.transform()

    def select_scales(self, threshold_abs=0.2, plot=False):

        # get set of deaths
        deaths = np.asarray(
            [self.persistence[i][1][1] for i in range(len(self.persistence))]
        )
        # drop duplicates
        deaths = np.unique(deaths)
        # replace inf with max time
        deaths[deaths == np.inf] = self.log_times[-1]
        # sort
        deaths.sort()

        # Compute differences to next death time
        diff_deaths = deaths[1:] - deaths[:-1]
        diff_deaths = np.append(diff_deaths, 0)

        # Find local maxima
        local_max_ind = peak_local_max(
            diff_deaths, threshold_abs=threshold_abs
        ).flatten()
        local_max_ind.sort()

        # Find indices of local max in log_times and of their succesors
        left_gap = [
            np.argwhere(self.log_times == deaths[local_max_ind[i]]).flatten()[0]
            for i in range(len(local_max_ind))
        ]
        right_gap = [
            np.argwhere(self.log_times == deaths[local_max_ind[i] + 1]).flatten()[0]
            for i in range(len(local_max_ind))
        ]

        # The optimal scales lie in the middle of the gaps
        self.optimal_scales = (np.array(left_gap) + np.array(right_gap)) // 2

        if plot:

            fig, ax = plt.subplots(1, figsize=(10, 5))
            ax.plot(deaths, diff_deaths, label="Difference to successor")
            ax.scatter(
                deaths[local_max_ind],
                diff_deaths[local_max_ind],
                color="green",
                label="Left gap",
            )
            ax.scatter(
                deaths[local_max_ind + 1],
                diff_deaths[local_max_ind + 1],
                color="lightgreen",
                label="Right gap",
            )
            ax.vlines(
                self.log_times[self.optimal_scales],
                0,
                diff_deaths.max(),
                color="gold",
                label="Optimal scales",
            )
            ax.set(xlabel="Deaths", ylabel="Difference")
            ax.legend()
            plt.show()

            return ax

    def plot_persistence_diagram(self):

        ax = gd.plot_persistence_diagram(self.persistence)

        if len(self.optimal_scales) > 0:
            ax.hlines(
                self.log_times[self.optimal_scales],
                self.log_times[0],
                self.log_times[-1],
                color="gold",
                label="Optimal scales",
            )
            ax.legend()

        plt.show()
        
        return ax

