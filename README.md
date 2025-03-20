[![DOI](https://zenodo.org/badge/486166159.svg)](https://zenodo.org/doi/10.5281/zenodo.12665882)

# MCF: Multiscale Clustering Filtration

This repository provides a Python implementation of the *Multiscale Clustering Filtration* (MCF) to analyse (non-hierarchical) sequences of partitions with persistent homology using `gudhi`. It is based on the paper "Analysing Multiscale Clusterings with Persistent Homology" by Dominik J. Schindler and Mauricio Barahona: https://arxiv.org/abs/2305.04281.


## Installation
Clone the repository and open the folder in your terminal. 

```bash
$ git clone https://github.com/barahona-research-group/MCF.git
$ cd MCF/
```

Then, to install the package with ``pip``, execute the following command:

```bash
$ pip install .
```

## Using the code

Given a (not necessarily hierarchical) sequence of partitions `theta` (a list of cluster indices lists) and a list of scales `t` we can construct the MCF filtration using the `MCF` class.

```Python
from mcf import MCF

# initialise MCF object
mcf = MCF(max_dim=3, method="standard)

# load sequence of partitions
mcf.load_data(theta, t)

# build filtration
mcf.build_filtration()
```

Note that the construction of the MCF (based on `gudhi.SimplexTree`) can require excessive memory when clusters become too large. When the total number of (distinct) clusters is smaller than the number of points, it can be computationally advantageous to construct the MCF using the equivalent nerve-based construction with `method='nerve'`. We can then compute the persistent homology of the MCF and also plot the persistence diagram.

```Python
# compute persistent homology
mcf.compute_persistence()

# plot persistence diagram
ax = mcf.plot_persistence_diagram()
```

From the persistent homology we can then compute the measure of *persistent hierarchy* to quantify the level of hierarchy in the sequence of partitions and the measure of *total persistent conflict* to quantify the presence of multiscale structure. 

```Python
# compute persistent hierarchy
h, h_bar = mcf.compute_persistent_hierarchy()
print("Average persistent hierarchy:",round(h_bar,4))

# compute persistent conflict
c_1, c_2, c = mcf.compute_persistent_conflict()
```

Our heuristic for scale selection is that robust partitions resolve many conflicts and are thus located at plateaus after dips in the total persistent conflict.

To compute all MCF measures and store them in a dictionary one can simply use the `compute_all_measures()` method.

```Python
# initialise MCF object
mcf = MCF()

# load sequence of partitions
mcf.load_data(theta,t)

# compute all MCF measures
mcf.compute_all_measures(file_path="mcf_results.pkl",)
```


## Experiments

We apply the MCF framework to sequences of partitions corresponding to four different stochastic block models with different intrinsic structure using our `sbm` module:

- Erdös-Renyi (Er) model: no scale, non-hierarchical
- single-scale stochastic block model (sSBM): 1 scale, hierarchical
- multiscale stochastic block model (mSBM): 3 scales, hierarchical
- non-hierarchical stochastic block model (nh-mSBM): 3 scales, non-hierarchical

To obtain sequences of partitions from the sampled graphs we use the `PyGenStability` Python package for multiscale clustering with Markov Stability analysis available at: https://github.com/barahona-research-group/PyGenStability

We then analyse the MCF persistence diagrams, persistent hierarchy and persistent conflict to analyse the level of hierarchy and multiscale structures. All scripts and notebooks to reproduce our experiments can be found in the `\experiments` directory.

## Contributors

- Dominik Schindler, GitHub: `d-schindler <https://github.com/d-schindler>`

We always look out for individuals that are interested in contributing to this open-source project. Even if you are just using `LGDE` and made some minor updates, we would be interested in your input.

## Cite

Please cite our paper if you use this code in your own work:

```
@article{schindlerAnalysingMultiscaleClusterings2024,
  author = {Schindler, Dominik J. and Barahona, Mauricio},
  title = {Analysing Multiscale Clusterings with Persistent Homology},
  publisher = {arXiv},
  year = {2024},
  doi = {10.48550/arXiv.2305.04281},
  url = {http://arxiv.org/abs/2305.04281},
}
```

## Licence

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
