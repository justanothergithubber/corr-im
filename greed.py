"""
Greedy algorithms for influence maximization.

The main function `accelgreedy` is designed to allow for both
graph techniques and using a Linear Program to find the next
seed with the greatest marginal gain.
"""
from pprint import pprint
from time import perf_counter

# Python Standard library
from typing import Callable

from igraph import Graph

# packages
from numpy import asarray

# other .py files
from config import DISTMAT, GRT, SolveMethod
from cvar import cvar, marg_dro_cvar
from dro import marg_dro_inf
from linear_program import inf_abs_mod, inf_conc_mod_solve


def _verbose_output(
    verbosity_setting: int,
    input_k: int,
    current_solution: list[int],
    to_be_added: int,
    current_value: float,
    compute_time: float,
) -> None:
    if verbosity_setting > 1:
        print(
            f"When k={input_k}, seed to be added is {to_be_added}, "
            f"with value {current_value}. Compute time = {compute_time}s"
        )
    if (verbosity_setting > 0) and (not (input_k + 1) % 5):
        print(f"When k={input_k}, solution = ", end="")
        pprint(current_solution, compact=True)


def accelgreedy(
    input_graph: Graph,
    desired_size: int,
    method: SolveMethod,
    verbosity: int = 0,
) -> GRT:
    """
    Run the CELF algorithm for the DRO IM problem.

    The Cost Effective Lazy Forward (CELF) algorithm does not evaluate all
    marginal gains for all possible inputs. It utilises the submodular
    property and checks if the current best known input is indeed still
    the best known input, as marginal gains can only decrease, not increase.
    """
    greedy_solution: list[int] = []
    marg_gain_return: list[float] = []
    compute_times: list[float] = []
    if desired_size == 0:
        return ([], [], [])
    start_time = perf_counter()
    graph_nodes: list[int] = [n.index for n in input_graph.vs()]
    # The variables below should not be unbound
    dist_mat: DISTMAT | None = None
    marg_gain_list: list[float] = []
    pi_: dict[int, float] = {}
    if method == SolveMethod.correlation_robust:
        pi_ = {n: 0 for n in graph_nodes}
        # Build a distance matrix
        # This speeds up the expected influence calculation
        dist_mat = input_graph.distances(weights="q")
        marg_gain_list = [
            sum(marg_dro_inf(input_graph, pi_, node, dist_mat=dist_mat).values())
            for node in graph_nodes
        ]
    elif method == SolveMethod.linear_program:
        inf_fun_args: dict = {"input_graph": input_graph}
        abstract_model = inf_abs_mod(input_graph)
        inf_fun: Callable[..., float] = inf_conc_mod_solve
        inf_fun_args["abs_mod"] = abstract_model
        marg_gain_list = [
            inf_fun(seed_set=[node], **inf_fun_args) for node in graph_nodes
        ]
    sorted_list: list[tuple[int, float]] = sorted(
        zip(graph_nodes, marg_gain_list), key=lambda x: x[1], reverse=True
    )
    # First seed, always optimal
    compute_times.append(perf_counter() - start_time)
    greedy_to_add = sorted_list[0][0]
    if method == SolveMethod.correlation_robust:
        pi_ = marg_dro_inf(input_graph, pi_, greedy_to_add, dist_mat=dist_mat)
    cur_spread: float = sorted_list[0][1]
    greedy_solution.append(greedy_to_add)
    marg_gain_return.append(cur_spread)
    sorted_list.pop(0)
    for k in range(1, desired_size):
        # Finding next seed with highest marginal gain
        need_to_re_eval: bool = True
        while need_to_re_eval:
            cur_node = sorted_list[0][0]
            if method == SolveMethod.linear_program:
                inf_fun_args["seed_set"] = greedy_solution + [cur_node]
                sorted_list[0] = (
                    cur_node,
                    inf_fun(**inf_fun_args) - cur_spread,
                )
            else:
                sorted_list[0] = (
                    cur_node,
                    sum(
                        marg_dro_inf(
                            input_graph, pi_, cur_node, dist_mat=dist_mat
                        ).values()
                    )
                    - cur_spread,
                )
            sorted_list = sorted(sorted_list, key=lambda x: x[1], reverse=True)
            need_to_re_eval = sorted_list[0][0] != cur_node

        # Found highest marginal gain
        comp_time = perf_counter() - start_time
        compute_times.append(comp_time)
        greedy_to_add = sorted_list[0][0]
        if method == SolveMethod.correlation_robust:
            pi_ = marg_dro_inf(input_graph, pi_, greedy_to_add, dist_mat=dist_mat)
        greedy_solution.append(greedy_to_add)
        marg_gain = sorted_list[0][1]
        cur_spread += marg_gain
        marg_gain_return.append(marg_gain)
        sorted_list.pop(0)

        # Verbose output
        _verbose_output(
            verbosity, k, greedy_solution, greedy_to_add, cur_spread, comp_time
        )
    return (greedy_solution, marg_gain_return, compute_times)


def accelgreedy_cvar(
    input_graph: Graph,
    desired_size: int,
    input_alpha: float,
    verbosity: int = 0,
) -> GRT:
    """
    Run the CELF algorithm for the CVaR maximization problem.

    This function is not as necessary as there is a C++ implementation
    of computing CVaR greedily that seems to be much faster.
    """
    start_time = perf_counter()
    greedy_solution: list[int] = []
    marg_gain_return: list[float] = []
    compute_times: list[float] = []
    if desired_size == 0:
        return ([], [], [])
    graph_nodes: list[int] = [n.index for n in input_graph.vs()]
    # Build a distance matrix
    # This speeds up the expected influence calculation
    dist_mat: DISTMAT = asarray(input_graph.distances(weights="q"))
    marg_gain_list: list[float] = [
        cvar(input_graph, input_alpha, [node], dist_mat=dist_mat)
        for node in graph_nodes
    ]
    # Sort all nodes by their marginal gains
    sorted_list: list[tuple[int, float]] = sorted(
        zip(graph_nodes, marg_gain_list), key=lambda x: x[1], reverse=True
    )
    # First seed, always optimal
    compute_times.append(perf_counter() - start_time)
    greedy_to_add = sorted_list[0][0]
    cur_cvar: float = sorted_list[0][1]
    greedy_solution.append(greedy_to_add)
    marg_gain_return.append(cur_cvar)
    sorted_list.pop(0)
    for k in range(1, desired_size):
        # Finding next seed with highest marginal gain
        need_to_re_eval: bool = True
        while need_to_re_eval:
            cur_node = sorted_list[0][0]
            # update marginal spread
            sorted_list[0] = (
                cur_node,
                marg_dro_cvar(
                    input_graph,
                    input_alpha,
                    greedy_solution,
                    cur_cvar,
                    cur_node,
                    dist_mat=dist_mat,
                ),
            )
            sorted_list = sorted(sorted_list, key=lambda x: x[1], reverse=True)
            need_to_re_eval = sorted_list[0][0] != cur_node

        # Found highest marginal gain
        comp_time = perf_counter() - start_time
        compute_times.append(comp_time)
        greedy_to_add = sorted_list[0][0]
        greedy_solution.append(greedy_to_add)
        marg_gain = sorted_list[0][1]
        cur_cvar += marg_gain
        marg_gain_return.append(marg_gain)
        sorted_list.pop(0)

        # Verbose output
        if verbosity:
            _verbose_output(
                verbosity,
                k,
                greedy_solution,
                greedy_to_add,
                cur_cvar,
                comp_time,
            )
    return (greedy_solution, marg_gain_return, compute_times)
