"""
This module defines functions relating to independence cascade.

Provided here are the independent cascade model by naive Monte-Carlo simulation
and the correlation robust influence maximisation cascade model
"""
# Python Standard library
from random import random, seed
from typing import Hashable

# packages
from igraph import Graph

from influence import det_inf_func


def ic_realisation(input_graph: Graph, randomness_seed: int) -> Graph:
    """Create a realisation out of an independence cascade model. Seeded."""
    copy_graph = input_graph.copy()
    seed(randomness_seed)
    to_remove = [
        (x[0], x[1]) for x in copy_graph.edges.data("weight") if x[2] < random()
    ]
    copy_graph.remove_edges_from(to_remove)
    return copy_graph


def mixic_realisation(
    input_graph: Graph,
    prob_mix: float,
    p_1: float,
    p_2: float,
    randomness_seed: int = 1,
) -> Graph:
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


def _calc_ic_inf(input_seed: int, input_graph: Graph, seed_set: list[Hashable]) -> int:
    particular_realisation = ic_realisation(input_graph, randomness_seed=input_seed)
    return len(det_inf_func(seed_set, particular_realisation))
