"""Graph generation, instantantiation and conversion functions."""
# Python Standard library
from itertools import chain, product
from pathlib import Path
from random import choice, random, seed
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
# packages
from igraph import Graph, load
from networkx import (DiGraph, all_pairs_shortest_path_length,
                      scale_free_graph, to_edgelist)
# other .py files
from config import GraphType, HetEdgeWeightType


def networkx_to_igraph(networkx_graph: DiGraph) -> Graph:
    """
    Convert a networkx DiGraph to an iGraph Graph.

    Code via https://stackoverflow.com/a/39085829
    By https://stackoverflow.com/users/1628638/ulrich-stern and SO contributors
    """
    return Graph(len(networkx_graph),
                 list(zip(*list(zip(*to_edgelist(networkx_graph)))[:2])))


def process_graph(input_graph: Graph,
                  input_edge_weight: Union[HetEdgeWeightType, float]) -> Graph:
    """Process graphs in a common manner."""
    graph = input_graph.simplify(combine_edges="ignore")
    seed(0)
    if isinstance(input_edge_weight, float):
        graph.es["p"] = input_edge_weight
    elif input_edge_weight == HetEdgeWeightType.uniform:
        seed(0)  # left in for reproducibiility
        for edge in graph.es:
            edge["p"] = random()
    elif input_edge_weight == HetEdgeWeightType.trivalency:
        seed(0)  # left in for reproducibiility
        for edge in graph.es:
            edge["p"] = choice((0.1, 0.01, 0.001))
    elif input_edge_weight == HetEdgeWeightType.weighted_cascade:
        for edge in graph.es:
            edge["p"] = 1/graph.degree(edge.target, mode="IN")

    # elif input_edge_weight == HetEdgeWeightType.other:
    #     for edge in graph.es:
    #         edge["p"] = --SOME_FUNCTION_THAT_ASSIGNS_WEIGHTS--

    else:
        raise ValueError("Only certain inputs to input_edge_weight allowed")
    for edge in graph.es:
        edge["q"] = 1 - edge["p"]
    return graph


def txt_to_graph(input_txt_file: str, reverse: bool) -> Graph:
    """Convert a txt file from SNAP to an igraph Graph."""
    re_index: Dict[int, int] = {}
    igraph_edges_set: Set[Tuple[int, int]] = set()
    node_counter = 0

    # Read the txt file first
    with open(input_txt_file) as file_obj:
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
                igraph_edges_set.add((re_index[node_u],
                                      re_index[node_v]))

    # Initialize graph object and add to it
    out_graph = Graph(directed=True)
    out_graph.add_vertices(len(re_index))
    out_graph.add_edges(igraph_edges_set)
    return out_graph


def get_graph_topo(graph_type: GraphType, num_nodes: Optional[int] = None,
                   graph_seed: Optional[int] = None) -> Graph:
    """Create a basic graph object with graph topology based on type."""
    if graph_type == GraphType.random_scale_free:
        graph = scale_free_graph(num_nodes, seed=graph_seed)
        graph = networkx_to_igraph(graph)
    elif graph_type == GraphType.polblogs:
        polblogs_graph = load("data/polblogs.gml")
        # reversing edges
        graph = Graph(n=polblogs_graph.vcount(),
                      edges=((e.target, e.source) for e in polblogs_graph.es),
                      directed=True)
    elif graph_type == GraphType.wikivote:
        graph = txt_to_graph("data/Wiki-Vote.txt", True)

    # For end users who wish to try other graphs
    # Load your own graph here
    # elif graph_type == GraphType.other:
    #     graph = --SOME_LOADING_FUNCTION--(ARGS)

    else:
        raise ValueError("Unexpected graph type given")
    return graph


def graph_to_tsv(input_graph: Graph, tsv_filename: Path) -> None:
    """Convert an input graph to TSV format."""
    with open(tsv_filename, "w") as tsv_file:
        for edge in input_graph.es:
            tsv_file.write(" ".join((str(edge.source),
                                     str(edge.target),
                                     str(edge["p"]))
                                    )+"\n"
                           )


def get_solution_diameter(input_graph: Graph,
                          input_solution: List[int]) -> Any:
    """Get the solution diameter."""
    undir_graph = input_graph.as_undirected()
    spl = undir_graph.shortest_paths(source=input_solution,
                                     target=input_solution)
    return max(chain(*spl))


def get_average_distance(input_graph: DiGraph,
                         subset_of_nodes: Iterable[int]) -> float:
    """Calculate average distances between a subset of nodes in NetworkX."""
    all_shortest = dict(all_pairs_shortest_path_length(input_graph))
    dist, counter = (0, 0)
    for (i, j) in product(subset_of_nodes, repeat=2):
        if i != j:
            dist += (all_shortest[i][j])
            counter += 1
    return dist / counter
