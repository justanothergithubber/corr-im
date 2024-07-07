# Correlational Robust Influence Maximization

## Overview

This repository contains code relating to the paper titled ["Correlation Robust
Influence Maximization"][neurips_paper], and its
[expanded version published in Management Science](mnsc_paper). The code is
provided primarily for reproducibility, and certain code are left as-is to
reproduce results from when the code was written. Ideas from this code may be
used to create a library.

We use Python 3 for general scripting and C++ to speed up certain computations.
More details on specific packages are given below. We run our code on existing
data.

## Usage

At a high level there are three steps to reproduce results:

1. Acquire [data](#data) by downloading from sources, and place them in `data`
   directory, same level as the scripts in here.
2. Ensure [dependencies](#dependencies) are in place. This includes compiling
   C++ programs whose source code is also located in this repository.
3. Run main Python script `paper_management_science.py`.

Necessary subdirectories should be created and intermediate outputs should be
placed in them.

### Management Science Results

If one wishes to reproduce the results in the Management Science paper, assuming
all dependencies and data are in place, a user should run

```shell
poetry run python paper_management_science.py
```

The script will gather data, including data we have gotten in NeurIPS. Also, we

- gather data CVaR parameter and the convex combination parameter.
- analyze a very small (|V| = 16) randomly generated graph, and obtain optimal
  solutions through brute force.
- analyze a subset of amazon DVDs and greedy solutions resulting
- summarize a lot of it into a summary folder

### NeurIPS Results

To reproduce results published in NeurIPS paper[neurips_paper], one may run

```shell
poetry run python paper_neurips.py
```

The script will begin calculating correlation-robust expected influence as well
as the expected influence under independence cascade and place such data files
into a folder called `out/` . After such data is gathered, the script will run
analyses which should produce the graphs and tables as shown in the paper.

### Specific Experiments

If a user wishes to run a specific experiment, users may also use
`experiment.py` . For example, to run the correlation greedy experiment on
`polblogs` with `k` up to 20 and homogeneous edge weights of 0.37, a user can
use

```shell
poetry run python experiment.py polblogs 20 correlation_robust 0 -p 0.37
```

Because `polblogs` and `correlation_robust` are actually `IntEnum` s (see
[`config.py`](config.py)), the same experiment can be represented by

```shell
poetry run python experiment.py 0 20 0 0 -p 0.37
```

Users are encouraged to run `python experiment.py --help` to look at the
options.

## Data

### Datasets used

We use real datasets:

- [`polblogs`](http://www-personal.umich.edu/~mejn/netdata/) - It seems that
  access to Mark Newman's page is not guaranteed, but mirrors of this dataset
  should be readily available. We provide two SHA256 hashes of files which
  should have the same content for this dataset, which depends on
  [CRLF line terminators](https://en.wikipedia.org/wiki/Newline):

  - 5391B1CABC2C171426EAE9BBC4196468FF93B731AE98C96B098F342332817BD6
  - BD83E3DD2397A57D2252CC212524C21FEE6708663136898D4ACE523224BDC632

- [`wikivote`](https://snap.stanford.edu/data/wiki-Vote.html), with SHA256 hash
  (hash of the _text file_ not the .gz)

  - D2AFBEDF262126F820C6B3DD9F39A6D68E6F5EA839C0508297032CA77578B28A

- [`amazon`](https://snap.stanford.edu/data/amazon-meta.html), with SHA256 hash

  - 600135116A05B7CE2DCB7E842E892D663C6190A0567D00373E0C5C4F3C908F02

It should be possible to download directly from data sources linked.

Please place the downloaded files `polblogs.gml` , `Wiki-Vote.txt` and
`amazon-meta.txt` in a folder called `data` after download. Note that we modify
all input graphs to simple graphs with [`Graph.simplify()`][igraph_simplify]. In
this case, `polblogs` is changed from a multidigraph with self-loops to a simple
graph.

### Use your own dataset

To use your own dataset, please modify the files
[`graph_functions.py`](graph_functions.py) and [`config.py`](config.py).
`graph_functions.py` is used in both loading graphs and assigning edge weights.
`config.py` influences the graphs loaded in `experiment.py` .

### Processed Outputs

We try to describe here some code functions that map to the Management Science
paper tables and figures.

- `common.get_dataset_graph_summaries` -> Table 2
- `paper_management_science.get_heterogenous_seed_summaries` -> Table 3
- `paper_management_science.tabulate_conv_data` -> Tables 4 and 5
- `paper_management_science.tabulate_alpha_greedy_compare` -> Tables 6 and 7
- `paper_management_science.plot_small_example` -> Figures 8 and 9 - converted
  to TikZ and we have changed how it looks
- `common.plot_compute_times` -> Figure 10
- `common.plot_hists` -> Figure 11
- `common.plot_expected_influence_graphs` -> Figure 12
- `paper_management_science.get_greedy_genres` -> Figures 13 and 14
- `paper_management_science.graph_degree_distribution_cdf` -> Figure 15

### Raw Outputs

Note that outputs are only produced after running our scripts.

Our (raw) outputs are stored as `.csv` files but have different outputs
depending on what we need. For files produced in `out` (but not its
subdirectories), files have four columns and no headers. The four columns are

- Seed Index
- Objective Function Value under Correlation Robust Influence
- Objective Function Value under Independent Cascade Influence
- Computational Time

However, in subdirectory `out/conv_cvar` , files contain our computations at
different alpha and lambda parameters. There should be two types of files
produced in this subdirectory. The typical file contains

- Seed Index
- Objective Function Value
- Computational Time

and the other type of file only for `amazon` (see the `get_amazon_est_data`
function in `paper_management_science.py` ) measures marginal objective function
values and computational times.

We also produce other `.pickle` and `tsv` files. The former are serialized data
to skip repeated computation and the later are graph data stored in the form
`FROM_INDEX[TAB_CHARACTER]TO_INDEX[TAB_CHARACTER]EDGE_WEIGHT` .

All other outputs are higher level outputs such as summary CSVs or PNGs. We use
and present these in our paper.

## Dependencies

### Python Packages

To run and reproduce the results as shown, users are encouraged to run

```{sh}
poetry install
```

and run the scripts with the virtual environment generated.

Summarily, the following packages are required:

- [igraph](https://igraph.org/)
- [NetworkX](https://networkx.github.io/)
- [Pyomo](http://www.pyomo.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow](https://pillow.readthedocs.io/en/stable/)

To represent graphs, we mainly use [igraph](https://igraph.org/).
[NetworkX](https://networkx.github.io/) was used in synthetic graph generation
and visualisation. We use [Pyomo](http://www.pyomo.org/) as one of the methods
to find the seed with the highest marginal gain, but even with a fast linear
program solver [Gurobi](https://www.gurobi.com/), it is not nearly as fast as
graph-based methods. We use [Matplotlib](https://matplotlib.org/) to plot graphs
and [Pillow](https://pillow.readthedocs.io/en/stable/) for image manipulation.

### Other dependecies

- Pruned Monte Carlo Simulator - because it is used for comparison, we also use
  a [pruned Monte Carlo simulator][pmc], though we have modified it as in the
  [fork](https://github.com/justanothergithubber/pruned-monte-carlo) for
  addition of some features mostly relating to data collection. Note that
  binaries are not provided but the code should compile relatively quickly - it
  takes an i7-7500U processor less than a minute. The resulting `pmc_greed.exe`
  and `pmc_est.exe` need to be in PATH for the paper `.py` scripts or
  `experiment.py` to run properly.

  - We have included a small helper
    [`build_conv.sh`](conv_greed_dir/build_conv.sh) that creates the executables
    assuming one has compatible C++ compilers. This also uses
    [`meson`](https://mesonbuild.com/). The resulting executables should not
    take long to compile.

- Linear Program Solver - if, for example, a user wants to use a linear program
  for influence maximization, then a solver is required. The files as provided
  assume that `gurobipy` is installed. Any solver that Pyomo can interface with
  can be used, but only [CBC](https://github.com/coin-or/Cbc) among free solvers
  has been tested. To use an alternative solver, please modify line 22 of
  [`linear_program.py`](linear_program.py).

We note that users who only wish to use the algorithm may modify the code and
simply use [`igraph`](https://igraph.org/).

## Files

Code files:

- [`.gitignore`](.gitignore) - for managing the git repository with ignores. We
  ignore a lot of intermediate outputs and raw data. Outputs should be produced
  from running the code and raw data should be downloaded directly from sources.
- [LICENSE](LICENSE) - License file. As part of abundance of caution we use
  GPL-3.0 as we are using a GPL library (specifically `igraph`)
- [`poetry.lock`](poetry.lock) - For `poetry install`. It is used for
  [reproducible installs](https://python-poetry.org/docs/basic-usage/#installing-with-poetrylock)
  so results are reproducible.
- [`pyproject.toml`](pyproject.toml) - For `poetry`. This is how we specify the
  environment to run our Python scripts in at a high level. It specifies
  depedencies.

Python scripts:

- [`paper_management_science.py`](paper_management_science.py) - this file is
  the main script that generates then analyses all data as shown in the expanded
  paper version. Running `paper_management_science.py` or specifically its is
  expected to take, like below in `paper_neurips.py` about a day or so on modern
  hardware. After the objective values and computational times are stored in
  folder `out/`, re-running the entire script should not take longer than 5
  minutes.
- [`paper_neurips.py`](paper_neurips.py) - this file is the main file that
  generates then analyses all data as shown in the NeurIPS proceedings. Running
  `paper_neurips.py` or specifically its `get_data()` function is expected to
  take around 20 hours on an i7-7500U processor. After the objective values and
  computational times are stored in folder `out/`, re-running the entire script
  should not take longer than 5 minutes.
- [`config.py`](config.py) - this file stores defaults and constants. The
  defaults are used for `paper_management_science.py` and `paper_neurips.py`,
  and is used to standardize things across different scripts.
- [`common.py`](common.py) - this file stores common functions used in both
  `paper_management_science.py` and `paper_neurips.py`.
- [`experiment.py`](experiment.py) - this file represents a single experiment
  where the influence diffusion process proceeds to completion on a graph. We
  collect data while performing the experiments, mostly relating to objective
  values and computational times. We consider only the computational time from
  the start of the diffusion process to the end of the diffusion process, and do
  not include the setup computations. We use this as an easy interface for both
  `paper_neurips.py` and `paper_management_science.py`.
- [`greed.py`](greed.py) - this file only contains `accelgreedy`, which is the
  accelerated greedy algorithm which seeks the seed with the highest marginal
  gain. The algorithm is also alternatively called 'Lazy Greedy.'
- [`dro.py`](dro.py) - this file stores the functions used for calculating
  expected influence under the correlation robust model
- [`independence_cascade.py`](independence_cascade.py) - similar to above, but
  gives results under independence cascade.
- [`linear_program.py`](linear_program.py) - this file stores the linear program
  for the correlation robust influence calculation problem. That is, given a
  seed set, the expected influence under a adversarial correlations given that
  the marginal probabilities for nodes to activate other nodes are fixed is
  calculated.
- [`graph_functions.py`](graph_functions.py) - this file stores all functions
  relating to the graphs themselves. Specifically we standardize graph
  processing, handle TSVs, convert between different libraries, processing
  specific data graphs here. We heavily use features of `igraph` here.
- [`cvar.py`](cvar.py) - this file stores functions for computation of
  Conditional Value at Risk, and is only used in `paper_management_science.py`
  as an extension of prior work.
- [`pmc.py`](pmc.py) - this file stores all functions for interfacing with the
  Pruned Monte Carlo simulation program. This requires `pmc_greed.exe` and
  `pmc_est.exe` to be in PATH.
- [`data.py`](data.py) - this file stores basic `amazon` graph data processing.
  Specifically it creates a genre `.csv` file, used in some analysis in our
  extended paper.
- [`influence.py`](influence.py) - this file stores some other influence
  functions not stored elsewhere (i.e. not DRO or IC). The 'deterministic'
  influence and comonotone influence were used as part of our research and
  comparison but do not feature heavily in the paper(s).

Subdirectories:

- [conv_greed_dir](conv_greed_dir) - This directory stores our C++ fork of the
  pruned Monte Carlo simulation _along with necessary changes to compute
  convex/cvar objectives_.
- [docs](docs) - We have basic documentation skeleton here. One may opt to run
  `sphinx-apidoc` and build the docs from code comments.

## Other notes

Note that within `accelgreedy(...)`, there is a part which [builds a distance
matrix][igraph_shortest] based on the graph, which speeds up the expected
influence calculation. This has memory requirements scaling quadratically with
the number of nodes of the graph, and so is not expected to scale well above
millions. While the script can be changed to not require this, it will slow down
the calculations.

To generate basic documentation, we use
[`sphinx-apidoc`](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html)
and auto-generate documentation from code comments.

## Citation

We provide below the Bibtex entry as provided by Management Science and NeurIPS 2020:

```bib
@article{doi:10.1287/mnsc.2021.03445,
author = {Chen, Louis L. and Lim, Chee Chin and Padmanabhan, Divya and Natarajan, Karthik},
title = {Robustness to Dependency in Influence Maximization},
journal = {Management Science},
volume = {0},
number = {0},
pages = {null},
year = {0},
doi = {10.1287/mnsc.2021.03445},
URL = {
        https://doi.org/10.1287/mnsc.2021.03445
},
eprint = {
        https://doi.org/10.1287/mnsc.2021.03445
},
    abstract = { In this paper, we pursue a correlation-robust study of the influence maximization problem. Departing from the classic independent cascade model, we study a diffusion process adversarially adapted to the choice of seed set. More precisely, rather than the independent coupling of known individual edge probabilities, we now evaluate a seed set’s expected influence under all possible correlations, specifically, the one that presents the worst case. We find that the worst case expected influence can be efficiently computed, its NP-hard optimization done approximately (1−1/e) with greedy construction, and we provide complete, efficient characterizations of the adversarial coupling, the random graph, and the random number of influenced nodes. But, most importantly, upon mixing the independent cascade with the worst case, we attain a tunable and more comprehensive model better suited for real-world diffusion phenomena than the independent cascade alone and without increased computational complexity. Extensions to the correlation-robust study of risk follow along with numerical experiments on network data sets with demonstration of how our models can be tuned.This paper was accepted by George Shanthikumar, data science.Funding: This work was supported by the Air Force Office of Scientific Research (Mathematical Optimization Program) under the grant: “Optimal Decision Making under Tight Performance Requirements in Adversarial and Uncertain Environments: Insight from Rockafellian Functions”.Supplemental Material: The online appendix and data files are available at https://doi.org/10.1287/mnsc.2021.03445. }
}
```

```bib
@inproceedings{NEURIPS2020_4ee78d41,
 author = {Chen, Louis and Padmanabhan, Divya and Lim, Chee Chin and Natarajan, Karthik},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {7078--7089},
 publisher = {Curran Associates, Inc.},
 title = {Correlation Robust Influence Maximization},
 url = {https://proceedings.neurips.cc/paper/2020/file/4ee78d4122ef8503fe01cdad3e9ea4ee-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

[igraph_shortest]: https://python.igraph.org/en/stable/api/igraph.GraphBase.html#distances
[igraph_simplify]: https://python.igraph.org/en/stable/api/igraph.GraphBase.html#simplify
[neurips_paper]: https://proceedings.neurips.cc/paper/2020/hash/4ee78d4122ef8503fe01cdad3e9ea4ee-Abstract.html
[mnsc_paper]: https://pubsonline.informs.org/doi/10.1287/mnsc.2021.03445
[pmc]: https://github.com/todo314/pruned-monte-carlo
