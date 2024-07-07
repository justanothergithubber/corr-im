"""
Module defining influence functions not belonging elsewhere.

This module defines functions relating to influence which do not
fit either the independence cascade or the correlation robust models.
"""
# Python Standard library
from collections import deque
from typing import Hashable

# packages
from igraph import Graph


def det_inf_func(input_seed_set: list[Hashable], input_graph: Graph) -> set[Hashable]:
    """
    Calculate influence of seed set on input graph.

    Output and inputs are deterministic.
    """
    total_num_nodes: int = input_graph.vcount()
    influenced: set[Hashable] = set()
    to_be_processed = deque(input_seed_set)
    already_processed: set[Hashable] = set()
    while to_be_processed:  # if this set is non-empty, then
        zeroth_index_node = to_be_processed.popleft()
        influenced.add(zeroth_index_node)
        already_processed.add(zeroth_index_node)
        # get immediate out-neighbours of specified node
        # they will be set to be influenced
        for succ_of_proc_node in input_graph.successors(zeroth_index_node):
            influenced.add(succ_of_proc_node)
            if len(influenced) == total_num_nodes:
                return set(v.index for v in input_graph.vs())
            if succ_of_proc_node not in already_processed:
                to_be_processed.append(succ_of_proc_node)
                # position of thing to be processed
    return influenced


def calc_comonotone_inf(input_graph: Graph, input_seed_set: list[Hashable]) -> float:
    """Calculate exact comonotone expected influence."""
    sorted_edges = sorted(input_graph.es, key=lambda x: x["p"])
    cur_graph = input_graph.copy()
    cur_prob: float = 0
    cur_est: float = 0
    for edge in sorted_edges:
        cur_est += (edge["p"] - cur_prob) * len(det_inf_func(input_seed_set, cur_graph))
        cur_graph.delete_edges([(edge.source, edge.target)])
        cur_prob = edge["p"]
    cur_est += (1 - cur_prob) * len(det_inf_func(input_seed_set, cur_graph))
    return cur_est
