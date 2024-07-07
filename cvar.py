"""Module containing all CVaR calculation logic."""

from igraph import Graph
from numpy import asarray, maximum, ndarray

from config import DISTMAT


def get_phi_sorted(
    input_graph: Graph,
    input_seed_set: set[int],
    dist_mat: DISTMAT | None = None,
) -> list[float]:
    """Calculate phi.

    Uses :py:meth:`~igraph  .GraphBase.distances` method
    with :math:`1-p` as the pairwise distances.
    """
    if dist_mat is None:
        dist_mat = asarray(input_graph.distances(weights="q"))

    phi_sorted: list[float] = sorted(dist_mat[list(input_seed_set), :].min(axis=0))
    return phi_sorted


def d_func(input_tau: int, input_seed_set: set[int], phi_sorted: list[float]) -> float:
    """
    Calculate d.

    d is defined in the paper in Lemma 3.
    """
    ls_ = len(input_seed_set)
    if input_tau <= ls_:
        return 0
    # else
    # -1 correction due to 0-indexing
    return min(phi_sorted[input_tau - 1], 1)


def var(
    input_graph: Graph,
    alpha: float,
    input_seed_set: set[int],
    phi_sorted: list[float],
) -> int:
    """
    Calculate Value at Risk corresponding to worst case CVaR.

    Value at Risk (VaR) is :math:`\\tau^{*}\\left( \\mathcal{S}\\right)`.
    """
    z__ = len(input_seed_set)
    cur_val = d_func(z__, input_seed_set, phi_sorted)
    while cur_val <= alpha:
        z__ += 1
        cur_val = d_func(z__, input_seed_set, phi_sorted)
    if alpha < d_func(z__, input_seed_set, phi_sorted):
        z__ = z__ - 1
    if z__ > input_graph.vcount():
        z__ = input_graph.vcount()
    return z__


def cvar(
    input_graph: Graph,
    input_alpha: float,
    input_seed_set: list[int],
    dist_mat: DISTMAT | None = None,
) -> float:
    """
    Calculate worst-case Conditional Value at Risk (CVaR).

    CVaR is calculated with the following formula

    .. math::

        f(S) =
            \\sum_{i\\in V}\\left(\\alpha-\\phi_{i}(S)\\right)^{+}/\\alpha
    """
    len_seed = len(input_seed_set)
    if len_seed == 0:
        return 0
    if dist_mat is None:
        dist_mat = asarray(input_graph.distances(weights="q"))
    phi: ndarray = dist_mat[list(input_seed_set), :].min(axis=0)
    return maximum(input_alpha - phi, 0).sum() / input_alpha


def marg_dro_cvar(
    input_graph: Graph,
    input_alpha: float,
    current_seed_set: list[int],
    prev_cvar: float,
    seed_to_add: int,
    dist_mat: DISTMAT | None = None,
) -> float:
    """Calculate marginal CVaR gained from candidate seed."""
    return (
        cvar(
            input_graph,
            input_alpha,
            current_seed_set + [seed_to_add],
            dist_mat=dist_mat,
        )
        - prev_cvar
    )
