import gudhi as gd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import peak_local_max

from msfiltration import MCF
from msfiltration.scale_selection import select_scales_gaps


def msf_bootstrap(community_ids, log_times, n_sample, B=20, seed=0):

    # obtain number of nodes
    n_nodes = len(community_ids[0])

    # define array of nodes
    nodes = np.arange(n_nodes)

    # initialise random number generator
    rng = np.random.RandomState(seed)

    # store persistences for different samples
    persistences = []

    for _ in range(B):

        # sample nodes
        nodes_sample = rng.choice(nodes, n_sample)
        nodes_sample.sort()

        # obtain all community assignments for sampled nodes
        community_ids_sampled = []
        for partition in community_ids:
            community_ids_sampled.append(partition[nodes_sample])

        # initialise new MSF object for sample
        msf_sample = MCF()
        msf_sample.partitions = community_ids_sampled
        msf_sample.log_times = log_times

        # compute PD for sample
        msf_sample.build_filtration()
        msf_sample.compute_persistence()

        # add persistence of different dimensions to list
        persistences.append(
            [
                msf_sample.filtration.persistence_intervals_in_dimension(dim)
                for dim in range(4)
            ]
        )

    return persistences


def select_scales_pds(
    all_persistences, log_times, threshold_abs=0.2, min_gap_width=0.2, with_plot=False
):

    # get number of PDs and max dimension
    n_PDs = len(all_persistences)
    max_dim = len(all_persistences[0])

    # get set of deaths from all persistences
    deaths = set()
    for sample in range(n_PDs):
        for dim in range(max_dim):
            deaths = deaths.union(set(all_persistences[sample][dim][:, 1]))

    # apply scale selection via gaps
    if with_plot:

        optimal_scales, gap_width, ax = select_scales_gaps(
            deaths, log_times, threshold_abs, min_gap_width, True
        )

        return optimal_scales, gap_width, ax

    else:

        optimal_scales, gap_width = select_scales_gaps(
            deaths, log_times, threshold_abs, min_gap_width, False
        )

        return optimal_scales, gap_width


def plot_pds(all_persistences, log_times, optimal_scales=[], alpha=0.1):

    # get number of PDs and max dimension
    n_PDs = len(all_persistences)
    max_dim = len(all_persistences[0])

    # get min and max log_times
    tmin = log_times[0]
    tmax = log_times[-1]

    # compute delta to determine where to plot points at infinity
    delta = 0.1 * abs(tmax - tmin)
    infinity = tmax + delta

    # font size
    # plt.rcParams.update({"font.size": 15})

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
    for i in range(n_PDs):
        for dim in range(max_dim):
            persistences = all_persistences[i][dim]
            if i == 0:
                if len(persistences) > 0:
                    ax.scatter(
                        persistences[:, 0],
                        np.nan_to_num(persistences[:, 1], posinf=infinity),
                        color=colormap[dim],
                        alpha=alpha,
                        label=r"$H_{}$".format(dim),
                    )
            else:
                if len(persistences) > 0:
                    ax.scatter(
                        persistences[:, 0],
                        np.nan_to_num(persistences[:, 1], posinf=infinity),
                        color=colormap[dim],
                        alpha=alpha,
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
    if len(optimal_scales) > 0:
        ax.hlines(
            log_times[optimal_scales],
            log_times[0] - 1,
            log_times[-1] + 1,
            color="gold",
            label="Optimal scales",
        )

    ax.legend(loc=4)

    return fig, ax
