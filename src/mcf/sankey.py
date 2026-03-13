"""Code for plotting optimised Sankey diagrams using OMICS Sankey."""

import numpy as np

np.seterr(invalid="ignore")

import pickle

from sklearn.metrics.cluster import contingency_matrix


class Sankey:
    """Class for plotting optimised Sankey diagrams using OMICS Sankey."""

    def __init__(self, partitions=np.array):
        """Initialise Sankey class."""
        self.partitions = partitions

        self.sankey_data_ = None
        self.omics_data_ = None

    @property
    def crossings(self):
        """Return crossing number of layout."""
        if self.omics_data_ is None:
            return None
        else:
            return int(self.omics_data_["Stage 2 Crossing"]) # if 0 this can be wrong and Stage 1 correct

    @property
    def weighted_crossings(self):
        """Return weighted crossing number of layout."""
        if self.omics_data_ is None:
            return None
        else:
            return int(self.omics_data_["Stage 2 WeightedCrossing"]) # if 0 this can be wrong and Stage 1 correct


    def compute_sankey_data(self):
        """Compute Sankey data from partitions."""
        self.sankey_data_ = _compute_sankey(self.partitions)

    def compute_omics_sankey(self, stage1_iter=100, stage2_iter=100, file_path=None):
        """Computed optimised Sankey layout using OMICS Sankey."""
        if self.sankey_data_ is None:
            self.compute_sankey_data()
        self.omics_data_ = _compute_omics_sankey(
            self.sankey_data_, stage1_iter, stage2_iter
        )

        # save data
        if file_path is not None:
            self.save_sankey_data(file_path)

    def plot_sankey(
        self,
        pad=0.3,
        thickness=1,
        contraction_factor=3,
        with_labels=False,
        stage1_iter=100,
        stage2_iter=100,
    ):
        """Plot Sankey diagram using optimised OMICS Sankey layout."""
        if self.omics_data_ is None:
            self.compute_sankey_data()
            self.compute_omics_sankey(stage1_iter=100, stage2_iter=100)
        fig = _plot_sankey_omics(
            self.sankey_data_, pad, thickness, contraction_factor, with_labels
        )
        return fig

    def save_sankey_data(self, file_path):
        """Save computed Sankey data to file."""

        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "sankey_data": self.sankey_data_,
                    "omics_data": self.omics_data_,
                },
                f,
            )

    def load_sankey_data(self, file_path):
        """Load computed Sankey data from file."""

        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.sankey_data_ = data["sankey_data"]
            self.omics_data_ = data["omics_data"]


def _compute_sankey(partitions):
    """Compute Sankey data from partitions."""

    M = len(partitions)

    nodes = []
    links = []
    levels = {}

    for m in range(M - 1):
        p_m = partitions[m]
        n_m = np.max(p_m) + 1
        p_m_plus_one = partitions[m + 1]
        n_m_plus_one = np.max(p_m_plus_one) + 1
        cm = contingency_matrix(p_m, p_m_plus_one)

        # add nodes
        if m == 0:
            for i in range(n_m):
                nodes.append({"name": f"P{m}_C{i}"})
        for j in range(n_m_plus_one):
            nodes.append({"name": f"P{m+1}_C{j}"})

        # add edges
        for i in range(n_m):
            for j in range(n_m_plus_one):
                cm_ij = cm[i, j]
                if cm_ij > 0:
                    links.append(
                        {
                            "source": f"P{m}_C{i}",
                            "target": f"P{m+1}_C{j}",
                            "value": int(cm_ij),
                        }
                    )

        # add levels
        if m == 0:
            levels[m] = [f"P{m}_C{i}" for i in range(n_m)]
        levels[m + 1] = [f"P{m+1}_C{j}" for j in range(n_m_plus_one)]

    # compile Sankey data
    data = {"nodes": nodes, "links": links, "level": levels}
    return data


def _compute_omics_sankey(sankey_data, stage1_iter=100, stage2_iter=100):
    """Computed optimised Sankey layout using OMICS Sankey."""

    # use OmicsSankey code from fork: https://github.com/juni-schindler/OmicsSankey
    from omics_sankey.main import run_method as omics_sankey_run

    # Set parameters like stated in OMICS Sankey paper
    alpha1 = 0.01
    alpha2 = 0.2

    # optimise OMICS Sankey layout
    result = omics_sankey_run(
        algo="BC",
        data=sankey_data,
        n=len(sankey_data["level"]),
        alpha1=alpha1,
        alpha2=alpha2,
        N=stage1_iter,
        M=stage2_iter,
        dummy_signal=True,
        cycle_signal=False,
        level=sankey_data["level"],
    )

    return result


def _plot_sankey_omics(
    sankey_data, pad=0.3, thickness=1, contraction_factor=3, with_labels=False
):
    """Plot Sankey diagram using optimised OMICS Sankey layout."""

    import plotly.graph_objects as go

    # map node names to their indices
    nodes = sankey_data["nodes"]
    node_id = {}
    for i, node in enumerate(nodes):
        node_id[node["name"]] = i

    labels = list(node_id.keys())

    # compile link data for plotly
    sources_id = []
    targets_id = []
    values = []
    sourcepos = {}
    targetpos = {}

    # extract link data
    for link in sankey_data["links"]:
        source = node_id[link["source"]]
        target = node_id[link["target"]]
        sources_id.append(source)
        targets_id.append(target)
        values.append(link["value"])
        sourcepos[source] = link["sourcepos"]
        targetpos[target] = link["targetpos"]

    # compile link dict
    link = {"source": sources_id, "target": targets_id, "value": values}

    # assign x node positions according to levels
    x_nodepos_dict = {}
    node_level_n = {}
    xs = np.linspace(0.001, 0.999, len(sankey_data["level"]))
    for level, level_nodes in sankey_data["level"].items():
        for node in level_nodes:
            x_nodepos_dict[node_id[node]] = xs[level]
            node_level_n[node_id[node]] = len(level_nodes)

    # get x node positions in correct order
    x_nodepos = []
    for _, id in node_id.items():
        x_nodepos.append(x_nodepos_dict[id])

    # get contracted y positions in correct order
    y_nodepos_contracted = []
    for node, id in node_id.items():
        n = node_level_n[id]
        if n == 1:
            y_nodepos_contracted.append(0.5)
        elif id in sourcepos:
            y_nodepos_contracted.append(
                _contract_yscale(sourcepos[id], n, contraction_factor)
            )
        else:
            y_nodepos_contracted.append(
                _contract_yscale(targetpos[id], n, contraction_factor)
            )

    # compile node dict
    node_dict = {
        "pad": pad,
        "thickness": thickness,
        "line": {"color": "black", "width": 0.0},
        "x": x_nodepos,
        "y": y_nodepos_contracted,
    }

    if with_labels:
        node_dict["label"] = labels

    # plot Sankey diagram
    layout = go.Layout(autosize=True)
    fig = go.Figure(
        data=[
            go.Sankey(
                node=node_dict,
                link=link,
            )
        ],
        layout=layout,
    )

    fig.show()

    return fig


def _contract_yscale(x, n, contraction_factor=3):
    """
    Maps a float x from the interval [0, 1] to the interval [1/(contraction_factor*n), 1 - 1/(contraction_factor*n)],
    maintaining proportions between points.

    Parameters:
    x (float): A float in the interval [0, 1].
    n (int): A positive integer for scaling the output range.

    Returns:
    float: A float in the range [1/(contraction_factor*n), 1 - 1/(contraction_factor*n)].
    """
    # Calculate the lower and upper bounds of the new interval
    lower_bound = 1 / (contraction_factor * n)
    upper_bound = 1 - 1 / (contraction_factor * n)

    # Apply the linear mapping
    mapped_value = lower_bound + x * (upper_bound - lower_bound)
    return mapped_value
