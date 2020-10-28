"""Configuration used in the experiments."""
from enum import IntEnum
from pathlib import Path
from typing import List, Tuple, Type

# Defaults
DEFAULT_RANDOM_GRAPH_SIZE = 1000
DEFAULT_TARGET_SIZE = 40
OUTPUT_FOLDER = Path("out")
TSV_FOLDER = Path("data/tsvs")

# Type hint alias for greedy algorithm results tuple
GRT = Tuple[List[int], List[float], List[float]]
# Type hint alias for distance matrix
DISTMAT = List[List[float]]


class GraphType(IntEnum):
    """IntEnum to denote graph types used."""

    polblogs = 0
    wikivote = 1
    random_scale_free = 2

    # This code bit is left in to easily allow for customization
    # other = 3

    def __str__(self) -> str:
        """Redefining __str__ for nicer printing in help."""
        return str(self.name)


class HetEdgeWeightType(IntEnum):
    """IntEnum to denote different heterogeneneous edge weights."""

    uniform = 0
    trivalency = 1
    weighted_cascade = 2

    # This code bit is left in to easily allow for customization
    # Actual logic should be put in graph_functions.py `process_graph`
    # other = 3

    def __str__(self) -> str:
        """Redefining __str__ for nicer printing in help."""
        return str(self.name)


class SolveMethod(IntEnum):
    """IntEnum to denote solution method used."""

    graph_techniques = 0
    independence_cascade = 1
    linear_program = 2

    def __str__(self) -> str:
        """Redefining __str__ for nicer printing in help."""
        return str(self.name)


class EnumParser:
    """This generalises the parser for various IntEnums."""

    def __init__(self, enumcls: Type[IntEnum]):
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
