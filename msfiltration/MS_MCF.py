import gudhi as gd
import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pygenstability import run
from skimage.feature import peak_local_max
from tqdm import tqdm

from msfiltration.MCF import MCF
from msfiltration.scale_selection import select_scales_gaps
from msfiltration.utils import node_id_to_dict


class MS_MCF(MCF):
    def __init__(self):

        super().__init__()

        # initialise adjacency matrix
        self.graph = None

        # initialise Markov Stabiliy attributes
        self.ms_results = None

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
        self.partitions = self.ms_results["community_id"]
        # get log_scales
        self.filtration_indices = np.log10(self.ms_results["scales"])
        # get number of scales
        self.n_partitions = len(self.filtration_indices)

    def load_ms_results(self, ms_results):
        # store MS results as attribute
        self.ms_results = ms_results
        # get community assignments
        self.partitions = self.ms_results["community_id"]
        # get log_scales
        self.filtration_indices = np.log10(self.ms_results["scales"])
        # get number of scales
        self.n_partitions = len(self.filtration_indices)

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

