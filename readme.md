# Correlational Robust Influence Maximisation
## Overview
This repository contains code relating to the paper titled "Correlation Robust Influence Maximization". The code is provided primarily for reproducibility, and certain code are left as-is to reproduce results from when the code was written. Ideas from this code may be used to create a library.

## Data
### Datasets used
We use two real datasets:
* [`polblogs`](http://www-personal.umich.edu/~mejn/netdata/)
* [`wikivote`](https://snap.stanford.edu/data/wiki-Vote.html)

Please place the downloaded files `polblogs.txt` and `Wiki-Vote.txt` in a folder called `data` after download.

### Use your own dataset
To use your own dataset, please modify the files [`graph_functions.py`](graph_functions.py) and [`config.py`](config.py). `graph_functions.py` is used in both loading graphs and assigning edge weights. `config.py` influences the graphs loaded in `experiment.py`.

## Usage
Assuming all dependencies and data are in place, to reproduce the results, a user simply needs to run

```
python paper.py
```

The script will begin calculating correlation-robust expected influence as well as the expected influence under independence cascade and place such data files into a folder called `out/`. After such data is gathered, the script will run analyses which should produce the graphs and tables as shown in the paper.

If a user wishes to run a specific experiment, users may also use `experiment.py`. For example, to run the correlation greedy experiment on `polblogs` with `k` up to 20 and homogeneous edge weights of 0.37, a user can use

```
python experiment.py polblogs 20 graph_techniques 0 -p 0.37
```

Because `polblogs` and `graph_techniques` are actually `IntEnums` (see [`config.py`](config.py)), the same experiment can be represented by

```
python experiment.py 0 20 0 0 -p 0.37
```

Users are encouraged to run `python experiment.py --help` to look at the options.

## Dependencies
### Python Packages
To run and reproduce the results as shown, users are encouraged to run
```
pip install -r requirements.txt
```

Summarily, the following packages are required:
* [igraph](https://igraph.org/)
* [NetworkX](https://networkx.github.io/)
* [Pyomo](http://www.pyomo.org/)
* [Matplotlib](https://matplotlib.org/)
* [Pillow](https://pillow.readthedocs.io/en/stable/)

To represent graphs, we mainly use [igraph](https://igraph.org/). [NetworkX](https://networkx.github.io/) was used in synthetic graph generation and visualisation. We use [Pyomo](http://www.pyomo.org/) as one of the methods to find the seed with the highest marginal gain, but even with a fast linear program solver [Gurobi](https://www.gurobi.com/), it is not nearly as fast as graph-based methods. We use [Matplotlib](https://matplotlib.org/) to plot graphs and [Pillow](https://pillow.readthedocs.io/en/stable/) for image manipulation.

### Other dependecies
* Pruned Monte Carlo Simulator - because it is used for comparison, we also use a [pruned Monte Carlo simulator](https://github.com/todo314/pruned-monte-carlo), though we have modified it as in the [fork](https://github.com/justanothergithubber/pruned-monte-carlo) for addition of some features mostly relating to data collection. Note that binaries are not provided but the code should compile relatively quickly - it takes an i7-7500U processor less than a minute. The resulting `pmc_greed.exe` and `pmc_est.exe` need to be in PATH for `paper.py` or `experiment.py` to run properly.
* Linear Program Solver - if, for example, a user wants to use a linear program for influence maximization, then a solver is required. The files as provided assume that `gurobipy` is installed. Any solver that Pyomo can interface with can be used, but only [CBC](https://github.com/coin-or/Cbc) has been tested. To use an alternative solver, please modify line 9 of [`linear_program.py`](linear_program.py).

We note that users who only wish to use the algorithm may modify the code and simply use [igraph](https://igraph.org/).

## Files
* [`paper.py`](paper.py) - this file is the main file that generates then analyses all data as shown in the paper. Running `paper.py` or specifically its `get_data()` function is expected to take around 20 hours on an i7-7500U processor. After the objective values and computational times are stored in folder `out/`, the rest of the functions should not take longer than 5 minutes.
* [`experiment.py`](experiment.py) - this file represents a single experiment where the influence diffusion process proceeds to completion on a graph. We collect data while performing the experiments, mostly relating to objective values and computational times. We consider only the computational time from the start of the diffusion process to the end of the diffusion process, and do not include the setup computations.
* [`greed.py`](greed.py) - this file only contains `accelgreedy`, which is the accelerated greedy algorithm which seeks the seed with the highest marginal gain. The algorithm is also alternatively called 'Lazy Greedy.'
* [`cascade.py`](cascade.py) - this file stores the functions used for calculating expected influence. Calculations for comonotone and independent cascade models are provided, but not exactly used, as the Python implementation of independent cascade could not compare to the C++ alternative.
* [`linear_program.py`](linear_program.py) - this file stores the linear program for the correlation robust influence calculation problem. That is, given a seed set, the expected influence under a adversarial correlations given that the marginal probabilities for nodes to activate other nodes are fixed is calculated.
* [`graph_functions.py`](graph_functions.py) - this file stores all functions relating to the graphs themselves.
* [`pmc.py`](pmc.py) - this file stores all functions for interfacing with the Pruned Monte Carlo simulation program. This requires `pmc_greed.exe` and `pmc_est.exe` to be in PATH.
* [`config.py`](config.py) - this file stores defaults and constants.

## Other notes
Note that within `accelgreedy(...)`, there is a part which [builds a distance matrix](https://igraph.org/python/doc/igraph.GraphBase-class.html#shortest_paths) based on the graph, which speeds up the expected influence calculation. This has memory requirements scaling quadratically with the number of nodes of the graph, and so is not expected to scale well above millions. While the script can be changed to not require this, it will slow down the calculations.

# Citation
If you find our work useful, please cite our paper. The preliminary .bib entry is given below, which we will update as soon as the final version is published:
```
@incollection{NIPS2020_4113,
title = {Correlation Robust Influence Maximization},
author = {Chen, Louis and Padmanabhan, Divya and Lim, Chee Chin and Natarajan, Karthik},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2020}
}
```
