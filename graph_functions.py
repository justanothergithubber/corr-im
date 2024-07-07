"""Graph generation, instantantiation and conversion functions."""

# Python Standard library
from collections.abc import Iterable
from itertools import chain, product
from pathlib import Path
from random import choice, random, seed

# packages
from igraph import Graph, read
from networkx import (
    DiGraph,
    all_pairs_shortest_path_length,
    scale_free_graph,
    to_edgelist,
)

# other .py files
from config import (
    AMAZON_FILE,
    AMAZON_NODEIDS,
    DATA_FOLDER,
    TSV_FOLDER,
    GraphType,
    HetEdgeWeightType,
)

_DVD_GROUP_STR = "  group: DVD"


def networkx_to_igraph(networkx_graph: DiGraph) -> Graph:
    """
    Convert a networkx DiGraph to an iGraph Graph.

    Code via https://stackoverflow.com/a/39085829
    by https://stackoverflow.com/users/1628638/ulrich-stern and
    SO contributors

    License: CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/deed.en

    License should be compatible with codebase license.
    """
    return Graph(
        len(networkx_graph),
        list(zip(*list(zip(*to_edgelist(networkx_graph)))[:2])),
        directed=True,  # Results in change from prior results
    )


def _get_similarity_ranks(input_amazon_file: Path) -> dict[str, list[str]]:
    """Get similar ASINS to current ASIN."""
    similarity_ranks: dict[str, list[str]] = {}
    cur_asin: str = ""  # should not exist in similarity ranks
    for line in open(input_amazon_file, encoding="utf-8"):
        ls: list[str] = line.split()
        if line.startswith("ASIN: "):
            cur_asin = ls[1]
        if line.startswith("  similar: "):
            similarity_ranks[cur_asin] = ls[2:]
    if "" in similarity_ranks:
        raise KeyError("Unexpected case")
    return similarity_ranks


def standardize_graph(
    input_graph: Graph,
    input_edge_weight: HetEdgeWeightType | float,
    inp_seed: int = 0,
) -> Graph:
    """
    Process graphs in a common manner.

    Removes self-loops and then adds edge attribute `p` depending on
    `input_edge_weight`.

    Typically run immediately after reading in data as
    :py:class:`~igraph.Graph` object.
    """
    output: Graph = input_graph.simplify(combine_edges="ignore")
    output["edge_type"] = input_edge_weight
    seed(inp_seed)
    if isinstance(input_edge_weight, float):
        output.es["p"] = input_edge_weight
    else:
        if input_edge_weight == HetEdgeWeightType.uniform:
            seed(inp_seed)  # left in for reproducibiility
            for edge in output.es:
                edge["p"] = random()
        elif input_edge_weight == HetEdgeWeightType.trivalency:
            seed(0)  # left in for reproducibiility
            for edge in output.es:
                edge["p"] = choice((0.1, 0.01, 0.001))
        elif input_edge_weight == HetEdgeWeightType.weightedcascade:
            for edge in output.es:
                edge["p"] = 1 / output.degree(edge.target, mode="IN")
        elif input_edge_weight in (
            HetEdgeWeightType.amazonlow,
            HetEdgeWeightType.amazonhigh,
        ):
            if input_graph["type"] == GraphType.amazon:
                sim_rank = _get_similarity_ranks(AMAZON_FILE)
                for dvd_asin in input_graph.vs["name"]:
                    for idx, other_asin in enumerate(sim_rank[dvd_asin]):
                        try:
                            edge_ind = input_graph.get_eid(dvd_asin, other_asin)
                            if input_edge_weight == HetEdgeWeightType.amazonlow:
                                # .10, .075, .05, .025, .01
                                input_graph.es[edge_ind]["p"] = max(
                                    0.01, 0.1 - 0.025 * idx
                                )
                            else:
                                # .95, .85, .75, .65, .55
                                input_graph.es[edge_ind]["p"] = 0.95 - idx * 0.1
                        except ValueError:
                            pass
            else:
                raise ValueError(
                    "Unknown combination used. Only "
                    "certain graphs have specifically implemented "
                    "custom edge weight functions."
                )
        else:
            raise ValueError("Only certain inputs to input_edge_weight allowed")
    for edge in output.es:
        edge["q"] = 1 - edge["p"]
    return output


def txt_to_graph(input_txt_file: Path, reverse: bool) -> Graph:
    """
    Convert a txt file from SNAP to an igraph Graph.

    Refer to SNAP networks at https://snap.stanford.edu/
    """
    re_index: dict[int, int] = {}
    igraph_edges_set: set[tuple[int, int]] = set()
    node_counter: int = 0

    # Read the txt file first
    with open(input_txt_file, encoding="utf-8") as file_obj:
        if reverse:
            node_u_idx = 1
            node_v_idx = 0
        else:
            node_u_idx = 0
            node_v_idx = 1
        for line in file_obj:
            if not line.startswith("#"):
                line_split = line.split()
                node_u = int(line_split[node_u_idx])
                node_v = int(line_split[node_v_idx])
                if node_u not in re_index:
                    re_index[node_u] = node_counter
                    node_counter += 1
                if node_v not in re_index:
                    re_index[node_v] = node_counter
                    node_counter += 1
                igraph_edges_set.add((re_index[node_u], re_index[node_v]))

    # Initialize graph object and add to it
    out_graph = Graph(directed=True)
    out_graph.add_vertices(len(re_index))
    out_graph.add_edges(igraph_edges_set)
    return out_graph


def _create_amazon_graph_nodes(
    input_amazon_file: Path,
) -> tuple[Graph, set[str], dict[str, int]]:
    """
    Create the nodes of the amazon graph.

    Resulting graph *only* contains DVDs. Resulting graph should
    have 19828 nodes and 50908 edges.

    Private helper function.
    """
    amazon_graph = Graph(directed=True)
    # ASINs are IDs of amazon objects
    amazon_dvd_asins: set[str] = set()
    amazon_asin_dict: dict[str, int] = {}
    count: int = 0
    cur_asin: str = ""

    if not AMAZON_NODEIDS.exists():
        with open(AMAZON_NODEIDS, mode="w", encoding="utf-8") as amazon_nodeids_file:
            for line in open(input_amazon_file, encoding="utf-8"):
                if line.startswith("ASIN: "):
                    cur_asin = line.split()[1]
                if line.startswith(_DVD_GROUP_STR):
                    amazon_dvd_asins.add(cur_asin)
                    amazon_graph.add_vertex(name=cur_asin)
                    amazon_asin_dict[cur_asin] = count
                    amazon_nodeids_file.write(f"{count}\t{cur_asin}\n")
                    count += 1
    else:
        for line in open(input_amazon_file, mode="r", encoding="utf-8"):
            if line.startswith("ASIN: "):
                cur_asin = line.split()[1]
            if line.startswith(_DVD_GROUP_STR):
                amazon_dvd_asins.add(cur_asin)
                amazon_graph.add_vertex(name=cur_asin)
                amazon_asin_dict[cur_asin] = count
                count += 1
    return (amazon_graph, amazon_dvd_asins, amazon_asin_dict)


def create_graph_basic(
    graph_type: GraphType,
    num_nodes: int | None = None,
    graph_seed: int | None = None,
) -> Graph:
    """
    Create a basic graph object with graph topology based on type.

    Type here refers to different types of graphs considered.

    Output only has graph topology (i.e. nodes and edges). The only added
    attribute to the graph is its type.
    """
    match graph_type:
        case GraphType.random_scale_free:
            output = networkx_to_igraph(scale_free_graph(num_nodes, seed=graph_seed))

        case GraphType.polblogs:
            polblogs_graph: Graph = read(DATA_FOLDER / "polblogs.gml")
            # reversing edges
            output = Graph(
                n=polblogs_graph.vcount(),
                edges=((e.target, e.source) for e in polblogs_graph.es),
                directed=True,
                vertex_attrs={"value": [v["value"] for v in polblogs_graph.vs]},
            )

        case GraphType.wikivote:
            output = txt_to_graph(DATA_FOLDER / "Wiki-Vote.txt", True)

        case GraphType.amazon:
            (output, dvd_asins, asin_dict) = _create_amazon_graph_nodes(AMAZON_FILE)
            cur_asin: str = ""
            cur_is_dvd: bool = False
            for line in open(AMAZON_FILE, encoding="utf-8"):
                ls = line.split()
                # Check if reached new object
                if line.startswith("ASIN: "):
                    cur_asin = ls[1]
                # Identify if DVD
                if line.startswith("  group:"):
                    cur_is_dvd = bool(line.startswith(_DVD_GROUP_STR))
                # Add edge if DVD
                if all((cur_is_dvd, line.startswith("  similar: "), len(ls) > 2)):
                    for other_asin in ls[2:]:
                        if other_asin in dvd_asins:
                            output.add_edge(
                                asin_dict[cur_asin],
                                asin_dict[other_asin],
                            )
            if "" in asin_dict:  # should not happen
                raise ValueError("Unexpected empty ASIN")

        # For end users who wish to try other graphs
        # Load your own graph here
        # case GraphType.other:
        #     output = --SOME_LOADING_FUNCTION--(ARGS)

        case _:
            raise ValueError("Unexpected graph type given")

    # for ease of access later
    output["type"] = GraphType(graph_type)

    return output


def get_tsv_path(
    inp_graph_type: GraphType,
    inp_edge_type: float | HetEdgeWeightType,
    num_nodes: int | None = None,
) -> Path:
    """Retrieve a Path object relating to a graph's attributes."""
    if num_nodes is not None:
        num_nodes_str: str = f",{num_nodes}"
    else:
        num_nodes_str = ""
    if isinstance(inp_edge_type, float):
        edge_name = inp_edge_type
    elif isinstance(inp_edge_type, HetEdgeWeightType):
        edge_name = inp_edge_type.name
    else:
        raise ValueError("An unexpected type of `inp_edge_type` encountered.")
    return TSV_FOLDER / f"{inp_graph_type.name},{edge_name}{num_nodes_str}.tsv"


def graph_to_tsv(input_graph: Graph) -> None:
    """Convert an input igraph Graph to TSV format."""
    num_nodes: int | None = None
    if input_graph["type"] == GraphType.random_scale_free:
        num_nodes: int | None = input_graph.vcount()
    tsv_filename = get_tsv_path(
        input_graph["type"], input_graph["edge_type"], num_nodes=num_nodes
    )
    with open(tsv_filename, mode="w", encoding="utf-8") as tsv_file:
        for edge in input_graph.es:
            tsv_file.write(
                " ".join((str(edge.source), str(edge.target), str(edge["p"]))) + "\n"
            )


def solution_diameter(input_graph: Graph, input_solution: list[int]) -> float:
    """
    Get the solution diameter.

    Diameter refers to the longest of all pair shortest paths while
    treating the graph as an undirected graph.

    Uses :py:meth:`~igraph.Graph.as_undirected` and
    :py:meth:`~igraph._igraph.GraphBase.distances`.
    """
    undir_graph: Graph = input_graph.as_undirected()
    spl = undir_graph.distances(source=input_solution, target=input_solution)
    return max(chain(*spl))


def average_distance(input_graph: DiGraph, subset_of_nodes: Iterable[int]) -> float:
    """
    Calculate average distances between a subset of nodes in NetworkX.

    Uses
    :py:func:`~networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length`.
    """
    all_shortest = dict(all_pairs_shortest_path_length(input_graph))
    dist: float = 0
    counter: int = 0
    for i, j in product(subset_of_nodes, repeat=2):
        if i != j:
            dist += all_shortest[i][j]
            counter += 1
    return dist / counter


def highest_outdegrees_seeds(input_graph: Graph, number_of_seeds: int) -> list[int]:
    """
    Get the highest outdegree seeds in graph.

    Defined here for standardization.

    Sort the vertices based on their out-degree in decreasing order
    range(...) -> [0, 1, 2, ...]
    this is sent to out_degrees and `sorted`
    gets the highest, 2nd highest, ...
    """
    out_degrees = input_graph.degree(mode="out")
    return sorted(
        range(input_graph.vcount()),
        key=lambda x: out_degrees[x],
        reverse=True,
    )[:number_of_seeds]
