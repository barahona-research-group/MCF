"""Code for Multiscale Clustering Bi-Filtration (MCbiF)"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from tqdm import tqdm

from mcf import MCF
from mcf.io import load_results, save_results
from mcf.utils import _get_partition_clusters, _cluster_id_preprocessing
from mcf.plotting import plot_sankey


class MCbiF:
    """Main class to construct MCbiF from a sequence of partitions and analyse
    its multiparameter persistent homology."""

    def __init__(self, method="standard", max_dim=2):
        """Initialise MCF object.

        Parameters:
            max_dim (int): Maximum dimension of simplices considered
                in filtration, between 1 and 2.

            method (str): Method to construct the self. Both methods lead to the
                same persistent homology, see our paper.
                - 'standard': Standard method where nodes in MCbiF correspond
                    to points. Faster when the number of points is smaller
                    than the total number of distinct clusters.
                - 'nerve': Nerve-based method where nodes in MCbiF correspond to
                    clusters. Faster when the total number of distinct clusters
                    is smaller than the number of points.
        """

        # initialise sequence of partitions
        self.partitions = []
        self.filtration_indices = []

        # set max dimension of filtration
        self.max_dim = min(2, max_dim)

        # set method to construct filtration, either standard or nerve-based
        self.method = method

        # initialise simplices and bigrades for rivet
        self.simplices = []
        self.bigrades = [[]]

        # initialise 0- and 1-dimensional Bettis (Hilbert functions)
        self.betti_0_ = None
        self.betti_0_rank_ = None
        self.betti_1_ = None
        self.betti_1_rank_ = None

        # initialise persistence landscapes
        self.bimod = None
        self.ls_0 = None
        self.ls_1 = None

        # initialse persistent hierarchy and conflict
        self.h_ = None
        self.h_bar_ = None
        self.c_ = None

    @property
    def n_partitions(self):
        """Computes number of partitions in sequence."""
        return len(self.partitions)

    @property
    def n_simplices(self):
        """Computes number of simplices in self."""
        return len(self.simplices)

    @property
    def n_bigrades(self):
        """Computes number of bigrades in self."""
        n_bigrades = 0
        for bigrade in self.bigrades:
            n_bigrades += len(bigrade)
        return n_bigrades

    def load_data(self, partitions, filtration_indices=None):
        """Method to load sequence of partitions and
        filtration indices."""
        self.partitions = partitions

        if filtration_indices is None:
            # if no filtration indices are given use enumeration
            self.filtration_indices = np.arange(1, self.n_partitions + 1)
        else:
            self.filtration_indices = filtration_indices

    def load_data_from_file(self, file_path):
        """Method to load data from precomputed MCbiF file."""

        # load results dictionary
        mcbif_results = load_results(file_path)

        # unpack dictionary
        self.filtration_indices = mcbif_results["filtration_indices"]
        self.max_dim = mcbif_results["max_dim"]
        self.method = mcbif_results["method"]
        self.betti_0_ = mcbif_results["betti_0"]
        self.betti_0_rank_ = mcbif_results["betti_0_rank"]
        self.betti_1_ = mcbif_results["betti_1"]
        self.betti_1_rank_ = mcbif_results["betti_1_rank"]
        self.h_ = mcbif_results["h"]
        self.h_bar_ = mcbif_results["h_bar"]
        self.c_ = mcbif_results["c"]
        self.ls_0 = mcbif_results["landscapes_0"]
        self.ls_1 = mcbif_results["landscapes_1"]

        # add dummy sequence of partitions
        self.partitions = np.zeros(len(self.filtration_indices))

    def _compute_mcbif_bigrades_standard(
        self,
        tqdm_disable=False,
        precomp_mcf=None,
    ):
        if precomp_mcf is not None:
            assert (
                precomp_mcf.method == "standard"
            ), "Precomputed MCF must be of standard type."
            # if MCF was precomputed, use it
            mcf_filtration_gudhi = precomp_mcf.filtration_gudhi.copy()
            # prune MCF to max_dim
            mcf_filtration_gudhi.prune_above_dimension(self.max_dim)
        else:
            if not tqdm_disable:
                print("Construct one-parameter filtration ...")
            # compute one-parameter MCF
            mcf = MCF(max_dim=self.max_dim, method="standard")
            mcf.load_data(
                partitions=self.partitions, filtration_indices=self.filtration_indices
            )
            mcf.build_filtration(tqdm_disable=tqdm_disable)
            mcf_filtration_gudhi = mcf.filtration_gudhi

        # get list of clusters as frozensets
        partitions_clusters = _get_partition_clusters(self.partitions)

        if not tqdm_disable:
            print("Construct two-parameter filtration ...")

        # initialise list of simplices and bigrades
        self.simplices = []
        self.bigrades = []

        # get first scale
        s0 = self.filtration_indices[0]

        # extract bigrades (s_0,t(s_0)) where s_0 is the first critical value
        for simplex, t_s0 in mcf_filtration_gudhi.get_filtration():
            self.simplices.append(simplex)
            self.bigrades.append([(s0, t_s0)])

        # delete MCF object as not needed anymore
        del mcf_filtration_gudhi

        # iterate through all simplices
        for i, simplex in tqdm(
            enumerate(self.simplices),
            total=len(self.simplices),
            disable=tqdm_disable,
        ):
            t_s0 = self.bigrades[i][0][1]
            # iterate through all start scales s
            for j in range(1, len(self.filtration_indices)):
                s = self.filtration_indices[j]
                ts_found = False
                # check bigrades that were already discovered for simplex
                for bigrade in self.bigrades[i]:
                    t_s_previous = bigrade[1]
                    # case 1: s \le t_s, then add (s,t_s)
                    if s <= t_s_previous:
                        self.bigrades[i].append((s, t_s_previous))
                        ts_found = True
                        break
                # case 2: there is no sl such that t_s \ge s
                # hence, we need to look from scratch at which t \ge s simplex appears after s (if at all)
                if not ts_found:
                    simplex_set = set(simplex)
                    for l in range(j, len(self.filtration_indices)):
                        for cluster_l in partitions_clusters[l]:
                            if simplex_set.issubset(cluster_l):
                                t_s = self.filtration_indices[l]
                                self.bigrades[i].append((s, t_s))
                                ts_found = True
                                break
                        if ts_found:
                            break
                # if we could not find t_s for s then we also can't find t_s' for s' > s
                if not ts_found:
                    break

    def _compute_mcbif_bigrades_nerve(
        self,
        tqdm_disable=False,
        precomp_mcf=None,
    ):
        if precomp_mcf is not None:
            assert (
                precomp_mcf.method == "nerve"
            ), "Precomputed MCF must be of nerve type."
            # if MCF was precomputed, use it
            mcf_filtration_gudhi = precomp_mcf.filtration_gudhi.copy()
            # prune MCF to max_dim
            mcf_filtration_gudhi.prune_above_dimension(self.max_dim)

        else:
            if not tqdm_disable:
                print("Construct one-parameter filtration ...")
            # compute one-parameter nerve-based MCF
            mcf = MCF(max_dim=self.max_dim, method="nerve")
            mcf.load_data(
                partitions=self.partitions, filtration_indices=self.filtration_indices
            )
            mcf.build_filtration(tqdm_disable=tqdm_disable)
            mcf_filtration_gudhi = mcf.filtration_gudhi

        # get list of clusters as frozensets
        partitions_clusters = _get_partition_clusters(self.partitions)

        if not tqdm_disable:
            print("Construct two-parameter filtration ...")

        # initialise list of simplices and bigrades
        self.simplices = []
        self.bigrades = []

        # get first scale
        s0 = self.filtration_indices[0]

        # extract bigrades (s_0,t(s_0)) where s_0 is the first critical value
        for simplex, t_s0 in mcf_filtration_gudhi.get_filtration():
            self.simplices.append(simplex)
            self.bigrades.append([(s0, t_s0)])

        # delete MCF object as not needed anymore
        del mcf_filtration_gudhi

        # we compute cluster indices of new clusters per partition
        # and a dictionary that maps cluster indices to sets
        _, ind_to_c = _cluster_id_preprocessing(self.partitions)

        # for the fast bigrade computation we also store all cluster occurences
        ind_to_scales = {}
        # iterate through clusters
        for c_ind, c in ind_to_c.items():
            # find occurences of cluster across sequence
            c_ind_scales = []
            # iterate through all scales
            for i in range(len(self.filtration_indices)):
                # add scale if cluster is part of partition
                if c in partitions_clusters[i]:
                    c_ind_scales.append(i)
            ind_to_scales[c_ind] = np.array(c_ind_scales)

        # iterate through all simplices
        for i, simplex in tqdm(
            enumerate(self.simplices),
            total=len(self.simplices),
            disable=tqdm_disable,
        ):
            # iterate through all start scales s
            for j in range(1, len(self.filtration_indices)):
                s = self.filtration_indices[j]
                ts_found = False
                min_ts_clusters = []
                # iterate through points (cluster indices) of simplex in nerve-based MCF
                for c_ind in simplex:
                    ts_found = True
                    c_scales = ind_to_scales[c_ind]
                    c_scales_after_j = c_scales[c_scales >= j]
                    if len(c_scales_after_j) == 0:
                        ts_found = False
                        break
                    else:
                        l = min(c_scales_after_j)
                        min_ts_clusters.append(self.filtration_indices[l])
                if ts_found:
                    t_s = max(min_ts_clusters)
                    self.bigrades[i].append((s, t_s))
                else:
                    break

    def build_filtration(self, tqdm_disable=False, precomp_mcf=None):
        """Build self."""
        if self.method == "standard":
            self._compute_mcbif_bigrades_standard(
                tqdm_disable=tqdm_disable, precomp_mcf=precomp_mcf
            )

        elif self.method == "nerve":
            self._compute_mcbif_bigrades_nerve(
                tqdm_disable=tqdm_disable, precomp_mcf=precomp_mcf
            )

    def compute_persistence(self, dimensions=None, tqdm_disable=False, threads=1):
        """Compute multiparameter persistent homology of MCbiF using Rivet."""

        try:
            from pyrivet import rivet
        except ImportError:
            raise ImportError("Pyrivet is not installed.")

        if dimensions is None:
            dimensions = np.arange(0, self.max_dim)

        # initialise bifiltration
        bifiltration = rivet.Bifiltration(
            x_label="s",
            y_label="t",
            simplices=self.simplices,
            appearances=self.bigrades,
            xreverse=True,
        )

        # compute persistence for all dimensions
        for d in dimensions:
            assert (
                d <= self.max_dim - 1
            ), f"Max filtartion dimension was set to {self.max_dim}."
            # compute 0 dimension
            if d == 0:
                if not tqdm_disable:
                    print("Compute 0-dim MPH ...")
                self.betti_0_ = rivet.betti(bifiltration, homology=0, threads=threads)
                # we rotate the graded rank matrix by 90 degrees
                self.betti_0_rank_ = np.rot90(self.betti_0_.graded_rank)

            # compute 1 dimension
            if d == 1:
                if not tqdm_disable:
                    print("Compute 1-dim MPH ...")
                self.betti_1_ = rivet.betti(bifiltration, homology=1, threads=threads)
                # we expand the graded rank matrix to have correct shape
                self.betti_1_rank_ = _expand_rivet_rank_output(
                    self.betti_1_, self.n_partitions
                )

    def compute_mma(self, tqdm_disable=False, verbose=False, threads=1):
        """Compute multiparameter module approximation using Multipers."""

        try:
            import multipers as mp
        except ImportError:
            raise ImportError("Multipers is not installed.")

        # initialise GUDHI bifiltration for multicritical MCbiF
        bifiltration_gudhi = mp.SimplexTreeMulti(num_parameters=2, kcritical=True)

        # add simplices to bifiltration
        for i, bigrades_i in enumerate(self.bigrades):
            for bigrade in bigrades_i:
                # we have opposite order on first parameter s, so we set it to -s
                bifiltration_gudhi.insert(self.simplices[i], [-bigrade[0], bigrade[1]])

        # compute persistence moulde approximation
        if not tqdm_disable:
            print("Compute multiparameter module approximation ...")
        self.bimod = mp.module_approximation(
            bifiltration_gudhi, n_jobs=threads, verbose=verbose
        )

    def compute_persistence_landscape(
        self, dimensions=None, k_max=None, resolution=256, tqdm_disable=False, threads=1
    ):
        """Compute multiparameter persistence landscapes using Multipers."""

        try:
            import multipers as mp
        except ImportError:
            raise ImportError("Multipers is not installed.")

        if self.bimod is None:
            self.compute_mma(tqdm_disable=tqdm_disable, threads=threads)

        if dimensions is None:
            dimensions = np.arange(0, self.max_dim)

        if k_max is None:
            k_max_0 = self.betti_0_rank_.max() + 1
            k_max_1 = self.betti_1_rank_.max() + 1
        else:
            k_max_0 = k_max
            k_max_1 = k_max

        # step = abs(self.filtration_indices[1] - self.filtration_indices[0]) # this assumes we have a uniform step size
        box = [
            [-self.filtration_indices[-1], self.filtration_indices[0]],
            [-self.filtration_indices[0], self.filtration_indices[-1]],
        ]

        # compute persistence landscape for all k
        for d in dimensions:
            assert (
                d <= self.max_dim - 1
            ), f"Max filtartion dimension was set to {self.max_dim}."
            # compute 0 dimension
            if d == 0:
                if not tqdm_disable:
                    print("Compute 0-dim persistence landscapes ...")
                ls_0 = []
                for k in tqdm(np.arange(0, k_max_0), disable=tqdm_disable):
                    ls_0.append(
                        np.flip(
                            self.bimod.landscape(
                                degree=0,
                                k=int(k),
                                box=box,
                                resolution=[resolution, resolution],
                            ),
                            axis=0,
                        )
                    )
                self.ls_0 = np.array(ls_0)

            elif d == 1:
                if not tqdm_disable:
                    print("Compute 1-dim persistence landscapes ...")
                ls_1 = []
                for k in tqdm(np.arange(0, k_max_1),disable=tqdm_disable):
                    ls_1.append(
                        np.flip(
                            self.bimod.landscape(
                                degree=1,
                                k=int(k),
                                box=box,
                                resolution=[resolution, resolution],
                            ),
                            axis=0,
                        )
                    )
                self.ls_1 = np.array(ls_1)

    def compute_persistent_hierarchy(self):
        """Compute persistent hierarchy."""
        # compute size of partitions
        s_partitions = np.array(
            [len(np.unique(partition)) for partition in self.partitions]
        )
        # compute persistent hierarchy
        self.h_ = self.betti_0_rank_ / s_partitions
        # compute average persistent hierarchy (exclude 1s on diagonal)
        self.h_bar_ = np.triu(self.h_, 1).sum() / (
            self.n_partitions * (self.n_partitions - 1) / 2
        )

    def compute_persistent_conflict(self):
        """Compute persistent conflict."""
        # self.c = ndimage.sobel(self.betti_1_rank, 1)
        self.c_ = np.gradient(self.betti_1_rank_, axis=1)
        return self.c_

    def compute_all_measures(
        self,
        file_path="mcbif_results.pkl",
        precomp_mcf=None,
        landscape_resolution=128,
        landscape_dimensions=None,
        landscape_k_max=None,
        tqdm_disable=False,
        threads=1,
    ):
        """Construct MCbiF, compute PH and compute all derived measures.

        Parameters:
            file_path (str): Path to save results.
            precomp_mcf (MCF): Precomputed MCF object. If None, a new MCF
                object is created.
            landscape_resolution (int): Resolution of persistence landscapes. Set to -1
                to disable computation of landscapes.
            tqdm_disable (bool): Disable tqdm progress bar.
            threads (int): Number of threads to use for computing RIVET."""

        # build filtration
        self.build_filtration(precomp_mcf=precomp_mcf, tqdm_disable=tqdm_disable)

        # compute multiparameter persistent homology
        try:
            self.compute_persistence(tqdm_disable=tqdm_disable, threads=threads)
        except:
            if tqdm_disable:
                pass
            else:
                print("Warning: Could not compute multiparameter persistent homology. ")

        # compute persistent hierarchy and conflict
        try:
            self.compute_persistent_hierarchy()
        except:
            if tqdm_disable:
                pass
            else:
                print("Warning: Could not compute persistent hierarchy. ")
        try:
            self.compute_persistent_conflict()
        except:
            if tqdm_disable:
                pass
            else:
                print("Warning: Could not compute persistent conflict. ")

        # compute persistence landscapes
        if landscape_resolution > 0:
            if landscape_dimensions is None:
                landscape_dimensions = np.arange(0, self.max_dim)
            if 0 in landscape_dimensions:
                try:
                    k_max_0 = landscape_k_max
                    if landscape_k_max is None:
                        k_max_0 = self.betti_0_rank_.max() + 1
                    self.compute_persistence_landscape(
                        dimensions=[0],
                        k_max=k_max_0,
                        resolution=landscape_resolution,
                        tqdm_disable=tqdm_disable,
                        threads=threads,
                    )
                except:
                    if tqdm_disable:
                        pass
                    else:
                        print(
                            "Warning: Could not compute 0-dim persistence landscapes. "
                        )
            if 1 in landscape_dimensions:
                try:
                    k_max_1 = landscape_k_max
                    if landscape_k_max is None:
                        k_max_1 = self.betti_1_rank_.max() + 1
                    self.compute_persistence_landscape(
                        dimensions=[1],
                        k_max=k_max_1,
                        resolution=landscape_resolution,
                        tqdm_disable=tqdm_disable,
                        threads=threads,
                    )
                except:
                    if tqdm_disable:
                        pass
                    else:
                        print(
                            "Warning: Could not compute 1-dim persistence landscapes. "
                        )

        # compile results dictionary
        mcbif_results = {}
        mcbif_results["filtration_indices"] = self.filtration_indices
        mcbif_results["max_dim"] = self.max_dim
        mcbif_results["method"] = self.method
        mcbif_results["betti_0"] = self.betti_0_
        mcbif_results["betti_0_rank"] = self.betti_0_rank_
        mcbif_results["betti_1"] = self.betti_1_
        mcbif_results["betti_1_rank"] = self.betti_1_rank_
        mcbif_results["h"] = self.h_
        mcbif_results["h_bar"] = self.h_bar_
        mcbif_results["c"] = self.c_
        mcbif_results["n_simplices"] = self.n_simplices
        mcbif_results["n_bigrades"] = self.n_bigrades
        mcbif_results["landscapes_0"] = self.ls_0
        mcbif_results["landscapes_1"] = self.ls_1

        if not file_path is None:
            # save results
            save_results(mcbif_results, file_path)

        return mcbif_results

    def plot_sankey(self, step=1, color=True, alpha=0.5, pad=0.1, thickness=1):
        """Plot Sankey diagram of partitions."""
        return plot_sankey(self, step, color, alpha, pad, thickness)

    # def plot_hilbert_function(self, dimension=0, path=None, title=None, vmax=None, discrete_ticks=None):
    #     """Plot Hilbert function."""
    #     assert (
    #         dimension <= self.max_dim - 1
    #     ), f"Max filtartion dimension was set to {self.max_dim}."

    #     if dimension == 0:
    #         hf = self.betti_0_rank_.copy()
    #     elif dimension == 1:
    #         hf = self.betti_1_rank_.copy()

    #     # shade lower diagonal part
    #     shade_lowtri = np.ones_like(hf)
    #     shade_lowtri[np.tril_indices_from(shade_lowtri, k=0)] = np.nan
    #     shade_lowtri = shade_lowtri.T

    #     fig, ax = plt.subplots(1, figsize=(7, 7))

    #     d_extent = abs(self.filtration_indices[-1] - self.filtration_indices[0]) / (
    #         2 * self.n_partitions
    #     )
    #     extent = [
    #         self.filtration_indices[0] - d_extent,
    #         self.filtration_indices[-1] + d_extent,
    #         self.filtration_indices[-1] + d_extent,
    #         self.filtration_indices[0] - d_extent,
    #     ]

    #     if dimension == 0:
    #         cmap = "Reds"
    #     elif dimension == 1:
    #         cmap = "Blues"

    #     if vmax is None:
    #         vmax = np.nanmax(hf)

    #     im = ax.imshow(
    #         hf,
    #         cmap=cmap,
    #         vmin=0,
    #         vmax=vmax,
    #         extent=extent,
    #         interpolation="none",
    #     )

    #     if dimension == 1:
    #         # plot white
    #         white_0 = np.ones_like(hf)
    #         white_0[hf > 0] = np.nan
    #         ax.imshow(white_0, cmap="binary", extent=extent, interpolation="none")

    #     # plot shading
    #     ax.imshow(
    #         shade_lowtri,
    #         cmap="grey",
    #         vmin=0,
    #         vmax=2,
    #         extent=extent,
    #         interpolation="none",
    #     )

    #     # Set the same tick labels for the x-axis
    #     plt.xticks(plt.yticks()[0], plt.yticks()[1])

    #     ax.set(
    #         xlabel=r"$t$",
    #         ylabel=r"$s$",
    #         xlim=(
    #             self.filtration_indices[0] - d_extent,
    #             self.filtration_indices[-1] + d_extent,
    #         ),
    #         ylim=(
    #             self.filtration_indices[-1] + d_extent,
    #             self.filtration_indices[0] - d_extent,
    #         ),
    #     )

    #     if dimension == 0:
    #         label_cbar = r"$\mathrm{HF}_0(s,t)$"
    #     elif dimension == 1:
    #         label_cbar = r"$\mathrm{HF}_1(s,t)$"
    #     cbar = plt.colorbar(im, shrink=0.4, label=label_cbar, location="top", pad=0.03)

    #     if not discrete_ticks is None:
    #         cbar.set_ticks(discrete_ticks)

    #     if not title is None:
    #         fig.suptitle(title)

    #     if not path is None:
    #         plt.savefig(path, dpi=fig.dpi, bbox_inches="tight")

    #     return ax

    def plot_hilbert_function(self, dimension=0, path=None, title=None, vmax=None, discrete_ticks=None):
        """Plot Hilbert function."""
        assert (
            dimension <= self.max_dim - 1
        ), f"Max filtration dimension was set to {self.max_dim}."

        if dimension == 0:
            hf = self.betti_0_rank_.copy()
        elif dimension == 1:
            hf = self.betti_1_rank_.copy()

        # Shade lower diagonal part
        shade_lowtri = np.ones_like(hf)
        shade_lowtri[np.tril_indices_from(shade_lowtri, k=0)] = np.nan
        shade_lowtri = shade_lowtri.T

        fig, ax = plt.subplots(1, figsize=(7, 7))

        d_extent = abs(self.filtration_indices[-1] - self.filtration_indices[0]) / (
            2 * self.n_partitions
        )
        extent = [
            self.filtration_indices[0] - d_extent,
            self.filtration_indices[-1] + d_extent,
            self.filtration_indices[-1] + d_extent,
            self.filtration_indices[0] - d_extent,
        ]

        # Choose colormap based on dimension
        cmap = plt.get_cmap("Reds") if dimension == 0 else plt.get_cmap("Blues")

        if vmax is None:
            vmax = np.nanmax(hf)

        if discrete_ticks is not None:

            # Create discrete color mapping if discrete_ticks is specified
            num_colors = len(discrete_ticks) if discrete_ticks is not None else 1
            colors = cmap(np.linspace(0, 1, num_colors))
            discrete_cmap = ListedColormap(colors)

            im = ax.imshow(
                hf,
                cmap=discrete_cmap,
                vmin=0,
                vmax=num_colors - 1,
                extent=extent,
                interpolation="none",
            )
        
        else:
            im = ax.imshow(
                hf,
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                extent=extent,
                interpolation="none",
            )

        if dimension == 1:
            # Plot white
            white_0 = np.ones_like(hf)
            white_0[hf > 0] = np.nan
            ax.imshow(white_0, cmap="binary", extent=extent, interpolation="none")

        # Plot shading
        ax.imshow(
            shade_lowtri,
            cmap="grey",
            vmin=0,
            vmax=2,
            extent=extent,
            interpolation="none",
        )

        # Set the same tick labels for the x-axis
        plt.xticks(plt.yticks()[0], plt.yticks()[1])

        ax.set(
            xlabel=r"$t$",
            ylabel=r"$s$",
            xlim=(
                self.filtration_indices[0] - d_extent,
                self.filtration_indices[-1] + d_extent,
            ),
            ylim=(
                self.filtration_indices[-1] + d_extent,
                self.filtration_indices[0] - d_extent,
            ),
        )

        if dimension == 0:
            label_cbar = r"$\mathrm{HF}_0(s,t)$"
        elif dimension == 1:
            label_cbar = r"$\mathrm{HF}_1(s,t)$"
        
        cbar = plt.colorbar(im, shrink=0.4, label=label_cbar, location="top", pad=0.03)

        if discrete_ticks is not None:
            tick_offset = (num_colors - 1) / (2 * num_colors)
            tick_positions = np.linspace(tick_offset, num_colors - 1 - tick_offset, num_colors)  # Centering tick positions
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(discrete_ticks)

        if title is not None:
            fig.suptitle(title)

        if path is not None:
            plt.savefig(path, dpi=fig.dpi, bbox_inches="tight")

        return ax

    def plot_persistent_conflict(self, path=None, title=None):
        """Plot persistent conflict."""
        # shade lower diagonal part
        shade_lowtri = np.ones_like(self.c_)
        shade_lowtri[np.tril_indices_from(shade_lowtri, k=0)] = np.nan
        shade_lowtri = shade_lowtri.T

        c_abs = np.nanmax(abs(self.c_))

        d_extent = abs(self.filtration_indices[-1] - self.filtration_indices[0]) / (
            2 * self.n_partitions
        )
        extent = [
            self.filtration_indices[0] - d_extent,
            self.filtration_indices[-1] + d_extent,
            self.filtration_indices[-1] + d_extent,
            self.filtration_indices[0] - d_extent,
        ]

        # plot c_t
        fig, ax = plt.subplots(1, figsize=(7, 7))
        im = ax.imshow(
            self.c_,
            cmap="seismic",
            vmin=-c_abs,
            vmax=c_abs,
            extent=extent,
            interpolation="none",
        )

        # plot shading
        ax.imshow(
            shade_lowtri,
            cmap="grey",
            vmin=0,
            vmax=2,
            extent=extent,
            interpolation="none",
        )

        # Set the same tick labels for the x-axis
        plt.xticks(plt.yticks()[0], plt.yticks()[1])

        ax.set(
            xlabel=r"$t$",
            ylabel=r"$s$",
            xlim=(
                self.filtration_indices[0] - d_extent,
                self.filtration_indices[-1] + d_extent,
            ),
            ylim=(
                self.filtration_indices[-1] + d_extent,
                self.filtration_indices[0] - d_extent,
            ),
        )

        label_cbar = r"$c(s,t)$"
        plt.colorbar(im, shrink=0.4, label=label_cbar, location="top", pad=0.03)

        if not title is None:
            fig.suptitle(title)

        if not path is None:
            plt.savefig(path, dpi=fig.dpi, bbox_inches="tight")

        return ax

    def plot_persistent_hierarchy(self, path=None, title=None):
        """Plot persistent hierarchy."""
        # shade lower diagonal part
        shade_lowtri = np.ones_like(self.h_)
        shade_lowtri[np.tril_indices_from(shade_lowtri, k=0)] = np.nan
        shade_lowtri = shade_lowtri.T

        d_extent = abs(self.filtration_indices[-1] - self.filtration_indices[0]) / (
            2 * self.n_partitions
        )
        extent = [
            self.filtration_indices[0] - d_extent,
            self.filtration_indices[-1] + d_extent,
            self.filtration_indices[-1] + d_extent,
            self.filtration_indices[0] - d_extent,
        ]

        # plot h
        fig, ax = plt.subplots(1, figsize=(7, 7))
        im = ax.imshow(self.h_, vmin=0, vmax=1, extent=extent, interpolation="none")

        # plot shading
        ax.imshow(
            shade_lowtri,
            cmap="grey",
            vmin=0,
            vmax=2,
            extent=extent,
            interpolation="none",
        )

        # Set the same tick labels for the x-axis
        plt.xticks(plt.yticks()[0], plt.yticks()[1])

        ax.set(
            xlabel=r"$t$",
            ylabel=r"$s$",
            xlim=(
                self.filtration_indices[0] - d_extent,
                self.filtration_indices[-1] + d_extent,
            ),
            ylim=(
                self.filtration_indices[-1] + d_extent,
                self.filtration_indices[0] - d_extent,
            ),
        )

        label_cbar = r"$h(s,t)$"
        plt.colorbar(im, shrink=0.4, label=label_cbar, location="top", pad=0.03)

        if not title is None:
            fig.suptitle(title)

        if not path is None:
            plt.savefig(path, dpi=fig.dpi, bbox_inches="tight")

        return ax

    def plot_landscape(self, dimension=0, ks=np.arange(3), path=None):
        """Plot multiparameter persistence landscapes."""
        assert (
            dimension <= self.max_dim - 1
        ), f"Max filtartion dimension was set to {self.max_dim}."

        # get persistence landscapes
        if dimension == 0:
            landscapes = self.ls_0
        elif dimension == 1:
            landscapes = self.ls_1

        # obtain k_max
        ks = ks[ks < len(landscapes)]

        # shade lower diagonal part
        shade_lowtri = np.ones_like(landscapes[0])
        shade_lowtri[np.tril_indices_from(shade_lowtri, k=0)] = np.nan
        shade_lowtri = shade_lowtri.T

        d_extent = abs(self.filtration_indices[-1] - self.filtration_indices[0]) / (
            2 * self.n_partitions
        )
        extent = [
            self.filtration_indices[0] - d_extent,
            self.filtration_indices[-1] + d_extent,
            self.filtration_indices[-1] + d_extent,
            self.filtration_indices[0] - d_extent,
        ]

        # create one subplot for each k
        fig, axs = plt.subplots(1, len(ks), figsize=((len(ks) + 1) * 3, 4))
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        fig.suptitle(f"{dimension}-dimensional persistence landscapes")

        # plot landscapes for different k
        for k in ks:

            landscape = landscapes[k]

            im = axs[k].imshow(
                landscape,
                vmin=0,
                vmax=np.nanmax(landscape),
                extent=extent,
                interpolation="none",
            )

            # plot shading
            axs[k].imshow(
                shade_lowtri,
                cmap="grey",
                vmin=0,
                vmax=2,
                extent=extent,
                interpolation="none",
            )

            # Set the same tick labels for the x-axis
            plt.xticks(plt.yticks()[0], plt.yticks()[1])
            axs[k].set(
                xlabel=r"$t$",
                ylabel=r"$s$",
                xlim=(
                    self.filtration_indices[0] - d_extent,
                    self.filtration_indices[-1] + d_extent,
                ),
                ylim=(
                    self.filtration_indices[-1] + d_extent,
                    self.filtration_indices[0] - d_extent,
                ),
            )

            label_cbar = f"$\lambda_{k+1}(s,t)$"
            plt.colorbar(im, shrink=0.4, label=label_cbar, location="top", pad=0.03)

        if not path is None:
            plt.savefig(path, dpi=fig.dpi, bbox_inches="tight")

        return axs


def _expand_rivet_rank_output(betti, n_partitions):
    """Add 0 columns to the right of Hilbert function matrix to make
    it square."""

    # get graded rank matrix
    hf = betti.graded_rank

    # add 0s so that hf is square and has the right shape
    hf_expanded = np.zeros((n_partitions, n_partitions))
    hf_expanded[: hf.shape[0], : hf.shape[1]] = hf

    # we rotate the graded rank matrix by 90 degrees
    hf_expanded = np.rot90(hf_expanded)

    return hf_expanded
