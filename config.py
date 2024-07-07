"""Configuration used in the experiments."""

from enum import IntEnum
from pathlib import Path
from typing import Type

from igraph import Graph
from numpy import float64
from numpy.typing import NDArray

# Defaults
# For data
## Folders
COMMON_PAPER_FOLDER = Path("paper_dir")
PAPER_MS_OUTPUT_FOLDER = COMMON_PAPER_FOLDER / "ms"
"""Folder to contain outputs shown in Management Science paper."""
NEURIPS_FOLDER = COMMON_PAPER_FOLDER / "neurips"
"""
Folder to contain outputs shown in NeurIPS paper.
Some outputs shared with Management Science paper.
"""

## Inputs
DATA_FOLDER = Path("data")

## Intermediate outputs
### Folders
"""Input data folder"""
TSV_FOLDER = Path("tsvs")
"""Folder contain .tsv version for each graph."""

RAW_OUTPUT_FOLDER = Path("out")
"""Folder for intermediate outputs"""
CONV_CVAR_OUT_PATH = RAW_OUTPUT_FOLDER / "conv_cvar"  # includes cvar-only
"""Folder for convex combination outputs"""
AMAZON_FILE = DATA_FOLDER / "amazon-meta.txt"
AMAZON_PAPER_OUTPUTS = PAPER_MS_OUTPUT_FOLDER / "amazon"
"""Folder for amazon processed outputs"""
AMAZON_RAW_OUTPUTS = RAW_OUTPUT_FOLDER / "amazon"
"""Folder for amazon relatively more unprocessed outputs."""

### Files
SUMMARY_CSV = COMMON_PAPER_FOLDER / Path("summary.csv")
AMAZON_ALL_GENRE_FILE = AMAZON_RAW_OUTPUTS / Path("amazon_allgenres.csv")
AMAZON_NODEIDS = AMAZON_RAW_OUTPUTS / Path("amazon_nodeids.csv")

## Small Random graph constants
DEFAULT_RANDOM_GRAPH_SIZE = 1000

DEFAULT_TARGET_SIZE = 40
"""Capital *K* inside paper"""
K_RANGE = range(1, DEFAULT_TARGET_SIZE + 1)
"""Alias for all values that k takes."""
EST_SEEDS = range(0, 100000, 10000)
"""Alias for estimation seeds we use"""

## Executables for gathering data
CONV_GREED_EXE_NAME = "conv_greed"
PMC_GREED_EXE_NAME = "pmc_greed"
CVAR_GREED_EXE_NAME = "cvar_greed"
PMC_EST_EXE_NAME = "pmc_est"
CVAR_EST_EXE_NAME = "cvar_est"

# Type hint aliases
GRT = tuple[list[int], list[float], list[float]]
"""Type hint alias for greedy algorithm results tuple."""
DISTMAT = NDArray[float64]
"""Type hint alias for distance matrix"""

#  More aliases at end of file


class GraphType(IntEnum):
    """IntEnum to denote graph types used."""

    polblogs = 0
    """
    Refers to the `polblogs` data set by Adamic and Glance (2004)
    
    Lada A. Adamic and Natalie Glance. 2005. The political blogosphere
    and the 2004 U.S. election: divided they blog. In Proceedings of
    the 3rd international workshop on Link discovery (LinkKDD '05).
    Association for Computing Machinery, New York, NY, USA, 36-43.
    https://doi.org/10.1145/1134271.1134277
    """
    wikivote = 1
    """
    Wikipedia adminship vote network

    See https://snap.stanford.edu/data/wiki-Vote.html
    """
    random_scale_free = 2
    """
    Randomly generated directed scale free graph

    Algorithm via `<https://networkx.org>`_, which uses
    B. Bollobás, C. Borgs, J. Chayes, and O. Riordan,
    Directed scale-free graphs, Proceedings of the fourteenth
    annual ACM-SIAM Symposium on Discrete Algorithms, 132-139,
    2003."""
    amazon = 3
    """
    Amazon co-purchasing network

    See https://snap.stanford.edu/data/amazon-meta.html
    """

    # This code bit is left in to easily allow for customization
    # other = 4

    def __str__(self) -> str:
        """Redefining __str__ for nicer printing in help."""
        return str(self.name)


class HetEdgeWeightType(IntEnum):
    """IntEnum to denote different heterogeneneous edge weights."""

    uniform = 0
    """Uniform random edge types between 0 and 1"""
    trivalency = 1
    """Equal probability between 1e-1, 1e-2, 1e-3"""
    weightedcascade = 2
    """Edge weight depending on indegree of target of edge"""
    amazonlow = 3
    """For amazon, low p values"""
    amazonhigh = 4
    """For amazon, high p values"""
    custom = 3
    """
    For special graph weight functions fit the other possibilities

    Specially given a definition within
    :py:func:`graph_functions.standardize_graph`.
    """

    def __str__(self) -> str:
        """Redefining __str__ for nicer printing in help."""
        return str(self.name)


class SolveMethod(IntEnum):
    """
    IntEnum to denote solution method used.

    'solution' here denotes getting a seed set.
    """

    correlation_robust = 0
    """
    Solution method corresponding to the distributionally robust model.
    
    Leverages shortest paths in graph.
    """
    independence_cascade = 1
    """Uses independence cascade program."""
    highest_outdegree = 2
    """Uses highest outdegree heuristic."""
    linear_program = 3
    """Solves as a linear program using Pyomo."""

    def __str__(self) -> str:
        """Redefining __str__ for nicer printing in help."""
        return str(self.name)


class EnumParser:
    """This generalises the parser for various IntEnums."""

    def __init__(self, enumcls: Type[IntEnum]) -> None:
        """Initialize the Enum."""
        self.cls = enumcls

    def __call__(self, arg: str) -> IntEnum:
        """Create an IntEnum via arg."""
        try:
            return self.cls(int(arg))
        except ValueError:
            pass
        try:
            return self.cls[arg]
        except KeyError:
            pass
        raise ValueError(f"Invalid {self.cls.__name__} input:'{arg}'")


# More Aliases
GRAPHS_USED = (GraphType.polblogs, GraphType.wikivote)
GraphDict = dict[GraphType | tuple[GraphType, float | GraphType], Graph]
"""Graph Dictionary Type"""
HETEDGEWEIGHT = [
    HetEdgeWeightType.uniform,
    HetEdgeWeightType.trivalency,
    HetEdgeWeightType.weightedcascade,
]

# Final constants
P_LIST = [0.01] + [x / 20 for x in range(1, 20)]
"""0.01, 0.05, 0.1, ... 0.95"""
EDGE_WEIGHTS: list[float | HetEdgeWeightType] = P_LIST + HETEDGEWEIGHT

## NeurIPS constants
## Here to prevent circular imports - config is the "root" of all Python imports
NEURIPS_METHODS_USED = (
    SolveMethod.correlation_robust,
    SolveMethod.independence_cascade,
)
NEURIPS_SOL_NAMES = {
    SolveMethod.correlation_robust: "corr",
    SolveMethod.independence_cascade: "ic",
}
"""Solution used in the NeurIPS paper."""
NEURIPS_SOL_SHORT_NAMES = {
    SolveMethod.correlation_robust.name: "corr",
    SolveMethod.independence_cascade.name: "ic",
}
"""Maps longer names like `correlation_robust` to something shorter, like `corr`"""
NEURIPS_SOL_TUPLE: tuple[str, str] = ("corr", "ic")
