import numpy as np


def node_id_to_dict(node_id):
    """
    Input: Array of node_id's
    Output: Dictionary that maps community number to node_keys
    """
    node_id = np.asarray(node_id)
    node_keys = np.arange(len(node_id))
    n_communities = np.max(node_id)
    community_dict = {}
    for i in range(n_communities + 1):
        c = np.argwhere(node_id == i).flatten()
        c_set = frozenset(node_keys[j] for j in c)
        if len(c) > 0:
            community_dict[i] = c_set
    return community_dict


def plot_sankey(
    all_results,
    optimal_scales=True,
    live=False,
    filename="communities_sankey.html",
    scale_index=None,
    color=True,
    alpha=1.0,
):  # pragma: no cover
    """Plot Sankey diagram of communities accros scale (plotly only).

    Args:
        all_results (dict): results from run function
        optimal_scales (bool): use optimal scales or not
        live (bool): if True, interactive figure will appear in browser
        filename (str): filename to save the plot
        scale_index (bool): plot scale of indices
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

    if not scale_index:
        all_results["community_id_reduced"] = all_results["community_id"]
    else:
        all_results["community_id_reduced"] = [
            all_results["community_id"][i] for i in scale_index
        ]

    community_ids = all_results["community_id_reduced"][::2]
    if optimal_scales and ("selected_partitions" in all_results.keys()):
        community_ids = [community_ids[u] for u in all_results["selected_partitions"]]

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

    plot(fig, filename=filename)

    if live:
        fig.show()

    return fig
