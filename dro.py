"""
Module for anything relating to the distributionally robust model.

We provide correlation robust influence maximisation-related
functions here, such as those computing the influence under said model.
"""

# Python Standard library
from collections.abc import Sequence

# packages
from igraph import Graph

# other .py files
from config import DISTMAT


def marg_dro_inf(
    input_graph: Graph,
    pi_: dict[int, float],
    seed_to_add: int,
    dist_mat: DISTMAT | None = None,
) -> dict[int, float]:
    """
    Calculate influence given a prior vector pi and new candidate seed.

    Uses the distance matrix. (Section 7.1)
    """
    new_pi = pi_.copy()
    new_pi[seed_to_add] = 1
    if dist_mat is None:
        all_dist = input_graph.distances(source=seed_to_add, weights="q")[0]
    else:
        all_dist = dist_mat[seed_to_add]
    for node in range(input_graph.vcount()):
        temp_var: float = 1 - all_dist[node]
        if new_pi[node] >= temp_var:
            continue
        # else
        new_pi[node] = temp_var
    return new_pi


def dro_inf(input_graph: Graph, input_seed_set: list[int]) -> float:
    """
    Calculate distributionally robust influence given a seed set.

    Sums across every node using :py:func:`marg_dro_inf`.
    """
    # pylint: disable=invalid-name
    pi_: dict[int, float] = {n.index: 0 for n in input_graph.vs()}
    for candidate_seed in input_seed_set:
        pi_ = marg_dro_inf(input_graph, pi_, candidate_seed)
    return sum(pi_.values())


def all_marg_dro_inf(input_graph: Graph, input_seed_set: Sequence[int]) -> list[float]:
    """Calculate marginal influences of each item in the seed set."""
    pi_: dict[int, float] = {n.index: 0 for n in input_graph.vs()}
    marg_gain: list[float] = []
    cur_inf: float = 0
    for seed_ in input_seed_set:
        pi_ = marg_dro_inf(input_graph, pi_, seed_)
        temp_sum_pi_var = sum(pi_.values())
        marg_gain.append(temp_sum_pi_var - cur_inf)
        cur_inf = temp_sum_pi_var
    return marg_gain


def get_pi_star(
    input_graph: Graph, input_seed_set: list[int]
) -> list[tuple[int, float]]:
    """Compute pi star for a specific set of seeds."""
    pi_: dict[int, float] = {n.index: 0 for n in input_graph.vs()}
    marg_gain: list[float] = []
    cur_inf: float = 0
    for seed_ in input_seed_set:
        pi_ = marg_dro_inf(input_graph, pi_, seed_)
        temp_sum_pi_var = sum(pi_.values())
        marg_gain.append(temp_sum_pi_var - cur_inf)
        cur_inf = temp_sum_pi_var
    pi_final = sorted(pi_.items(), key=lambda x: x[1])
    return pi_final


def get_robust_graph(
    input_graph: Graph, pi_star: dict[int, float], q_tilde: float
) -> Graph:
    r"""Compute the graph for a specific pi star and \tilde{q}."""
    new_graph = Graph(
        n=input_graph.vcount(),
        directed=True,
        vertex_attrs={"value": [v["value"] for v in input_graph.vs]},
    )
    for e in input_graph.es:
        if (
            (pi_star[e.source] > pi_star[e.target])
            and not (pi_star[e.source] - 1 + e["p"] <= q_tilde <= pi_star[e.source])
        ) or ((pi_star[e.source] <= pi_star[e.target]) and (0 < q_tilde <= e["p"])):
            new_graph.add_edge(e.source, e.target)
    return new_graph
