import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def plot_sankey(
    mcf,
    step=1,
    color=True,
    alpha=1.0,
):  # pragma: no cover
    """Plot Sankey diagram of communities accros scale (plotly only). The following code is modified from the PyGenStability package.

    Args:
        mcf: MCF object
        scale_index (bool): plot scale of indices
        color (bool): plot colourful Sankey
        alpha (float): transparency of colours
    """

    import matplotlib.colors
    import plotly.graph_objects as go
    from plotly.offline import plot

    sources = []
    targets = []
    values = []
    colors = []
    shift = 0

    tab_20 = [
        "#4E79A7",
        "#A0CBE8",
        "#F28E2B",
        "#FFBE7D",
        "#59A14F",
        "#8CD17D",
        "#B6992D",
        "#F1CE63",
        "#499894",
        "#86BCB6",
        "#E15759",
        "#FF9D9A",
        "#79706E",
        "#BAB0AC",
        "#D37295",
        "#FABFD2",
        "#B07AA1",
        "#D4A6C8",
        "#9D7660",
        "#D7B5A6",
    ]

    tab_20_rgba = [list(matplotlib.colors.to_rgba(c)) for c in tab_20]

    for i in range(len(tab_20_rgba)):
        tab_20_rgba[i][3] = alpha
        tab_20_rgba[i] = "rgba" + str(tuple(tab_20_rgba[i]))

    # choose partitions according to step
    community_ids = mcf.partitions[::step]

    for i in range(len(community_ids) - 1):
        community_source = np.array(community_ids[i])
        community_target = np.array(community_ids[i + 1])
        source_ids = set(community_source)
        target_ids = set(community_target)
        for j, source in enumerate(source_ids):
            for target in target_ids:
                value = sum(community_target[community_source == source] == target)
                if value > 0:
                    values.append(value)
                    sources.append(source + shift)
                    targets.append(target + len(source_ids) + shift)
                    colors.append(tab_20_rgba[j % 20])
        shift += len(source_ids)

    if color:
        link = {"source": sources, "target": targets, "value": values, "color": colors}
    else:
        link = {"source": sources, "target": targets, "value": values}

    layout = go.Layout(autosize=True)
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 0.1,
                    "thickness": 1,
                    "line": {"color": "black", "width": 0.0},
                },
                link=link,
            )
        ],
        layout=layout,
    )

    fig.show()

    return fig


def plot_persistence_diagram(mcf, alpha=0.5):
    """
    code is a modified version of the GUDHI's plot_persistence_diagram
    """

    # obtain min and max values and define value for infinity
    tmin = mcf.filtration_indices[0]
    tmax = mcf.filtration_indices[-1]
    delta = 0.1 * abs(tmax - tmin)
    infinity = tmax + delta

    # font size
    plt.rcParams.update({"font.size": 20})

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
    for dim, PD in enumerate(mcf.persistence):
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
    if len(mcf.optimal_scales) > 0:
        ax.hlines(
            mcf.filtration_indices[mcf.optimal_scales],
            mcf.filtration_indices[0] - 1,
            mcf.filtration_indices[-1] + 1,
            color="gold",
            label="Optimal scales",
        )

    ax.legend(loc=4)

    return ax
