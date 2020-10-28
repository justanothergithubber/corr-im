"""
This module defines the cascade models used.

Provided here are the independent cascade model by naive Monte-Carlo simulation
and the correlation robust influence maximisation cascade model
"""
# Python Standard library
from collections import deque
from random import random, seed
from typing import Dict, Hashable, List, Optional, Sequence, Set
# packages
from igraph import Graph
# other .py files
from config import DISTMAT


def marg_dro_inf(input_graph: Graph, pi_: Dict[int, float],
                 seed_to_add: int,
                 dist_mat: Optional[DISTMAT] = None) -> Dict[int, float]:
    """Calculate influence given a prior vector pi and new candidate seed."""
    new_pi = pi_.copy()
    new_pi[seed_to_add] = 1
    if dist_mat is None:
        all_dist = input_graph.shortest_paths(source=seed_to_add,
                                              weights="q")[0]
    else:
        all_dist = dist_mat[seed_to_add]
    for node in range(input_graph.vcount()):
        temp_var = 1-all_dist[node]
        if new_pi[node] >= temp_var:
            continue
        # else
        new_pi[node] = temp_var
    return new_pi


def dro_inf(input_graph: Graph, input_seed_set: List[int]) -> float:
    """Calculate DR influence given a seed set."""
    # pylint: disable=invalid-name
    pi_: Dict[int, float] = {n.index: 0 for n in input_graph.vs()}
    for candidate_seed in input_seed_set:
        pi_ = marg_dro_inf(input_graph, pi_, candidate_seed)
    return sum(pi_.values())


def all_marg_dro_inf(input_graph: Graph,
                     input_seed_set: Sequence[int]) -> List[float]:
    """Calculate marginal influences of each item in the seed set."""
    pi_: Dict[int, float] = {n.index: 0 for n in input_graph.vs()}
    marg_gain = []
    cur_inf: float = 0
    for seed_ in input_seed_set:
        pi_ = marg_dro_inf(input_graph, pi_, seed_)
        temp_sum_pi_var = sum(pi_.values())
        marg_gain.append(temp_sum_pi_var-cur_inf)
        cur_inf = temp_sum_pi_var
    return marg_gain


def det_inf_func(input_seed_set: List[Hashable],
                 input_graph: Graph) -> Set[Hashable]:
    """
    Calculate influence of seed set on input graph.

    Output and inputs are deterministic.
    """
    total_num_nodes: int = input_graph.vcount()
    influenced = set()
    to_be_processed = deque(input_seed_set)
    already_processed = set()
    while to_be_processed:  # if this set is non-empty, then
        zeroth_index_node = to_be_processed.popleft()
        influenced.add(zeroth_index_node)
        already_processed.add(zeroth_index_node)
        # get immediate out-neighbours of specified node
        # they will be set to be influenced
        for succ_of_proc_node in input_graph.successors(zeroth_index_node):
            influenced.add(succ_of_proc_node)
            if len(influenced) == total_num_nodes:
                return set(input_graph.nodes())
            if succ_of_proc_node not in already_processed:
                to_be_processed.append(succ_of_proc_node)
                # position of thing to be processed
    return influenced


def ic_realisation(input_graph: Graph, randomness_seed: int) -> Graph:
    """Create a realisation out of an independence cascade model. Seeded."""
    copy_graph = input_graph.copy()
    seed(randomness_seed)
    to_remove = [(x[0], x[1]) for x in
                 copy_graph.edges.data("weight") if x[2] < random()]
    copy_graph.remove_edges_from(to_remove)
    return copy_graph


def mixic_realisation(input_graph: Graph, prob_mix: float,
                      p_1: float, p_2: float,
                      randomness_seed: int = 1) -> Graph:
    """Return a mixture of two independent cascade models."""
    copy_graph = input_graph.copy()
    seed(randomness_seed)
    to_remove = []
    for edge in copy_graph.edges():
        u_1 = random()
        u_2 = random()
        if u_1 > prob_mix:
            if p_1 < u_2:
                to_remove.append(edge)
        else:
            if p_2 < u_2:
                to_remove.append(edge)
    copy_graph.remove_edges_from(to_remove)
    return copy_graph


def _calc_ic_inf(input_seed: int, input_graph: Graph,
                 seed_set: List[Hashable]) -> int:
    particular_realisation = ic_realisation(input_graph,
                                            randomness_seed=input_seed)
    return len(det_inf_func(seed_set, particular_realisation))


def calc_comonotone_inf(input_graph: Graph,
                        input_seed_set: List[Hashable]) -> float:
    """Calculate exact comonotone expected influence."""
    sorted_edges = sorted(input_graph.es, key=lambda x: x["p"])
    cur_graph = input_graph.copy()
    cur_prob: float = 0
    cur_est: float = 0
    for edge in sorted_edges:
        cur_est += (edge["p"] - cur_prob) * \
            len(det_inf_func(input_seed_set, cur_graph))
        cur_graph.delete_edges([(edge.source, edge.target)])
        cur_prob = edge["p"]
    cur_est += (1 - cur_prob) * len(det_inf_func(input_seed_set, cur_graph))
    return cur_est
