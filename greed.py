"""
Greedy algorithms for influence maximization.

The main function `accelgreedy` is designed to allow for both
graph techniques and using a Linear Program to find the next
seed with the greatest marginal gain.
"""
# Python Standard library
from pprint import pprint
from time import perf_counter
from typing import Any, Callable, Dict, List
# packages
from igraph import Graph
# other .py files
from cascade import marg_dro_inf
from config import GRT, SolveMethod
from linear_program import inf_abs_mod, inf_conc_mod


def accelgreedy(input_graph: Graph, desired_size: int,
                method: SolveMethod, verbosity: int = 0) -> GRT:
    """
    Run the CELF algorithm for the DRO IM problem.

    The Cost Effective Lazy Forward (CELF) algorithm does not evaluate all
    marginal gains for all possible inputs. It utilises the submodular
    property and checks if the current best known input is indeed still
    the best known input, as marginal gains can only decrease, not increase.
    """
    greedy_solution: List[int] = []
    marg_gain_return: List[float] = []
    compute_times: List[float] = []
    if desired_size == 0:
        return ([], [], [])
    graph_nodes = [n.index for n in input_graph.vs()]
    if method == SolveMethod.graph_techniques:
        pi_: Dict[int, float] = {n: 0 for n in graph_nodes}
        # Build a distance matrix
        # This speeds up the expected influence calculation
        dist_mat = input_graph.shortest_paths(weights="q")
        marg_gain_list: List[float] = [sum(marg_dro_inf(input_graph,
                                                        pi_,
                                                        node,
                                                        dist_mat=dist_mat
                                                        ).values())
                                       for node in graph_nodes]
    elif method == SolveMethod.linear_program:
        inf_fun_args = {"input_graph": input_graph}
        abstract_model = inf_abs_mod(input_graph)
        inf_fun: Callable[..., Any] = inf_conc_mod
        inf_fun_args["abs_mod"] = abstract_model
        inf_fun_args["solve"] = True
        marg_gain_list = [inf_fun(seed_set=[node], **inf_fun_args)
                          for node in graph_nodes]
    sorted_list = sorted(zip(graph_nodes,
                             marg_gain_list),
                         key=lambda x: x[1], reverse=True)
    # First seed, always optimal
    compute_times.append(perf_counter())
    greedy_to_add = sorted_list[0][0]
    if method == SolveMethod.graph_techniques:
        pi_ = marg_dro_inf(input_graph, pi_,
                           greedy_to_add, dist_mat=dist_mat)
    cur_spread: float = sorted_list[0][1]
    greedy_solution.append(greedy_to_add)
    marg_gain_return.append(cur_spread)
    sorted_list.pop(0)
    for k in range(1, desired_size):
        # Finding next seed with highest marginal gain
        need_to_re_eval = True
        while need_to_re_eval:
            cur_node = sorted_list[0][0]
            if method == SolveMethod.linear_program:
                inf_fun_args["seed_set"] = greedy_solution + [cur_node]
                sorted_list[0] = (cur_node,
                                  inf_fun(**inf_fun_args) - cur_spread)
            else:
                sorted_list[0] = (cur_node,
                                  sum(marg_dro_inf(input_graph,
                                                   pi_,
                                                   cur_node,
                                                   dist_mat=dist_mat
                                                   ).values()
                                      ) - cur_spread
                                  )
            sorted_list = sorted(sorted_list, key=lambda x: x[1],
                                 reverse=True)
            need_to_re_eval = sorted_list[0][0] != cur_node

        # Found highest marginal gain
        compute_times.append(perf_counter())
        greedy_to_add = sorted_list[0][0]
        if method == SolveMethod.graph_techniques:
            pi_ = marg_dro_inf(input_graph, pi_,
                               greedy_to_add,
                               dist_mat=dist_mat
                               )
        greedy_solution.append(greedy_to_add)
        marg_gain = sorted_list[0][1]
        cur_spread += marg_gain
        marg_gain_return.append(marg_gain)
        if verbosity > 0:
            if not (k+1) % 5:
                print(f"When k={k}, solution = ", end="")
                pprint(greedy_solution, compact=True)
        if verbosity > 1:
            greedy_step_str = (f"When k={k}, "
                               f"seed to be added is {greedy_to_add}, "
                               f"with value {cur_spread}")
            print(greedy_step_str)
        sorted_list.pop(0)
    return (greedy_solution, marg_gain_return, compute_times)
