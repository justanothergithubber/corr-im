"""
Contains interfacing logic with the C++ convex combination greedy program.

Most of the computation is done on the C++ side.
"""

from os import name
from pathlib import Path
from pprint import pprint
from subprocess import check_output
from time import perf_counter
from typing import cast

from igraph import Graph
from numpy import asarray

from config import (
    CONV_GREED_EXE_NAME,
    CVAR_EST_EXE_NAME,
    CVAR_GREED_EXE_NAME,
    DISTMAT,
    GRT,
    PMC_EST_EXE_NAME,
    PMC_GREED_EXE_NAME,
)
from cvar import cvar, marg_dro_cvar
from graph_functions import get_tsv_path
from pmc import all_pmc_inf_est, pmc_inf_est


def conv(
    seed_set: list[int],
    conv_graph: Graph,
    conv_lambda: float,
    conv_alpha: float,
    tsv_path: Path,
    dist_mat: DISTMAT | None = None,
) -> float:
    """
    Compute convex combination objective.

    conv_lambda: Proportion of objective function which comes from
    independent cascade.
    conv_alpha: CVaR parameter
    """
    return conv_lambda * pmc_inf_est(tsv_path, seed_set) + (1 - conv_lambda) * cvar(
        conv_graph, conv_alpha, seed_set, dist_mat=dist_mat
    )


def marg_conv(
    seed_set: list[int],
    conv_graph: Graph,
    conv_lambda: float,
    conv_alpha: float,
    tsv_path: Path,
    prev_obj_val: float,
    dist_mat: DISTMAT | None = None,
):
    """Compute convex combination objective."""
    return (
        conv(
            seed_set,
            conv_graph,
            conv_lambda,
            conv_alpha,
            tsv_path,
            dist_mat=dist_mat,
        )
        - prev_obj_val
    )


def marg_conv_est(
    input_graph: Graph,
    input_seed_set: list[int],
    inp_alpha: float,
    inp_lambda: float,
    tsv_path: Path,
) -> tuple[list[float], list[float]]:
    """
    Python CVaR + C++ Pruned Monte Carlo implementation.

    For testing out values.
    """
    num_seeds = len(input_seed_set)
    if inp_lambda > 0:
        pmc_marg_gain, pmc_compute_times = all_pmc_inf_est(tsv_path, input_seed_set)
    else:
        pmc_marg_gain = [0] * num_seeds
        pmc_compute_times = [0] * num_seeds
    if inp_lambda < 1:
        cvar_start = perf_counter()
        cur_seed_set: list[int] = []
        cumulative_cvar = 0.0
        dist_mat: DISTMAT = asarray(input_graph.distances(weights="q"))
        cvar_marg: list[float] = []
        cvar_compute_times: list[float] = []
        for seed in input_seed_set:
            marg_cvar = marg_dro_cvar(
                input_graph,
                inp_alpha,
                cur_seed_set,
                cumulative_cvar,
                seed,
                dist_mat=dist_mat,
            )
            cur_seed_set.append(seed)
            cumulative_cvar += marg_cvar
            cvar_marg.append(marg_cvar)
            cvar_compute_times.append(perf_counter() - cvar_start)
    else:
        cvar_marg = [0] * num_seeds
        cvar_compute_times = [0] * num_seeds
    total_marg_gain: list[float] = []
    total_compute_time: list[float] = []
    for pmc_gain, cvar_gain in zip(pmc_marg_gain, cvar_marg):
        total_marg_gain.append(inp_lambda * pmc_gain + (1 - inp_lambda) * cvar_gain)
    for pmc_compute_time, cvar_compute_time in zip(
        pmc_compute_times, cvar_compute_times
    ):
        total_compute_time.append(pmc_compute_time + cvar_compute_time)
    return (total_marg_gain, total_compute_time)


def _get_output_from_program(input_program: str, *args) -> GRT:
    """
    Get a tuple of lists from programs in standardised way.

    Private helper function.
    """
    if name == "posix":  # windows vs linux
        input_program = "./" + input_program
    program_output = cast(bytes, check_output(str(x) for x in (input_program, *args)))  # type: ignore
    program_output = filter(
        None,
        program_output.decode("utf-8").replace("\r", "").replace("\t", " ").split("\n"),
    )
    seed_list: list[int] = []
    marg_gain_list: list[float] = []
    compute_time_list: list[float] = []
    for output in program_output:
        seed, marg_gain, compute_time = output.split()
        seed_list.append(int(seed))
        marg_gain_list.append(float(marg_gain))
        compute_time_list.append(float(compute_time))
    return (seed_list, marg_gain_list, compute_time_list)


def conv_est(
    input_seed_file_path: Path,
    input_tsv: Path,
    est_lambda: float,
    input_alpha: float,
) -> tuple[list[float], list[float]]:
    """
    Estimate convex combination of IC and CVaR objective value function.

    Uses the C++ interface.
    """
    num_seeds = 0
    with open(input_seed_file_path, encoding="utf-8") as io_obj:
        num_seeds = len((io_obj.read()).split())
    if est_lambda > 0:
        _, pmc_marg, pmc_compute_time = _get_output_from_program(
            PMC_EST_EXE_NAME, input_tsv, input_seed_file_path, 10000, 0
        )
    else:
        pmc_marg = [0] * num_seeds
        pmc_compute_time = [0] * num_seeds
    if est_lambda < 1:
        _, cvar_marg, cvar_compute_time = _get_output_from_program(
            CVAR_EST_EXE_NAME,
            input_tsv,
            input_seed_file_path,
            input_alpha,
        )
    else:
        cvar_marg = [0] * num_seeds
        cvar_compute_time = [0] * num_seeds
    total_marg = [
        est_lambda * pmc_contribution + (1 - est_lambda) * cvar_contribution
        for (pmc_contribution, cvar_contribution) in zip(pmc_marg, cvar_marg)
    ]
    total_compute_time = [
        pmc_contribution + cvar_contribution
        for (pmc_contribution, cvar_contribution) in zip(
            pmc_compute_time, cvar_compute_time
        )
    ]
    return (total_marg, total_compute_time)


def conv_greed(
    inp_graph: Graph,
    input_k: int,
    inp_alpha: float,
    inp_lambda: float,
    inp_seed_set_size: int = 10000,
) -> GRT:
    """Interface C++ program to run greedy pruned Monte Carlo algorithm."""
    if input_k == 0:
        return ([], [], [])
    if 0 < inp_lambda < 1:
        exe_str = CONV_GREED_EXE_NAME
    elif inp_lambda == 0:
        exe_str = CVAR_GREED_EXE_NAME
    elif inp_lambda == 1:
        exe_str = PMC_GREED_EXE_NAME
    else:
        raise ValueError("Lambda value not appropriate.")
    input_tsv_path = get_tsv_path(inp_graph["type"], inp_graph["edge_type"])
    if 0 < inp_lambda < 1:
        return _get_output_from_program(
            exe_str,
            input_tsv_path,
            input_k,
            inp_seed_set_size,  # 10000 simulations
            0,  # seed delta
            inp_alpha,
            inp_lambda,
        )
    elif inp_lambda == 1:
        return _get_output_from_program(
            exe_str,
            input_tsv_path,
            input_k,
            inp_seed_set_size,  # 10000 simulations
            0,  # seed delta
        )
    elif inp_lambda == 0:
        return _get_output_from_program(
            exe_str,
            input_tsv_path,
            input_k,
            inp_alpha,
        )
    else:
        raise ValueError("Lambda value not appropriate.")


def _verbose_output(
    verbosity_setting: int,
    input_k: int,
    current_solution: list[int],
    to_be_added: int,
    current_value: float,
    compute_time: float,
) -> None:
    """Function for outputting to stdout."""
    if verbosity_setting > 1:
        print(
            f"When k = {input_k}, seed to be added is {to_be_added}, "
            f"with value {current_value}. Compute time = {compute_time}s"
        )
    if (verbosity_setting > 0) and (not (input_k + 1) % 5):
        print(f"When k = {input_k}, solution = ", end="")
        pprint(current_solution, compact=True)


def accelgreedy_conv(
    input_graph: Graph,
    input_tsv: Path,
    desired_size: int,
    input_alpha: float,
    input_lambda: float,
    verbosity: int = 0,
) -> GRT:
    """
    Run the CELF algorithm.

    This runs the algorithm for the convex combination of both
    independent cascade and CVaR maximization problem.

    Not used because of speed.
    """
    start_time = perf_counter()
    greedy_solution: list[int] = []
    marg_gain_output: list[float] = []
    compute_times: list[float] = []
    if desired_size == 0:
        return ([], [], [])
    graph_nodes: list[int] = [n.index for n in input_graph.vs()]
    # Build a distance matrix
    # This speeds up the expected influence calculation
    dist_mat: DISTMAT = asarray(input_graph.distances(weights="q"))
    marg_gain_list: list[float] = [
        input_lambda * pmc_inf_est(input_tsv, [node])
        + (1 - input_lambda) * cvar(input_graph, input_alpha, [node], dist_mat=dist_mat)
        for node in graph_nodes
    ]
    # Sort all nodes by their marginal gains
    sorted_list: list[tuple[int, float]] = sorted(
        zip(graph_nodes, marg_gain_list), key=lambda x: x[1], reverse=True
    )
    # First seed, always optimal
    compute_times.append(perf_counter() - start_time)
    greedy_to_add = sorted_list[0][0]
    cur_val: float = sorted_list[0][1]
    greedy_solution.append(greedy_to_add)
    marg_gain_output.append(cur_val)
    sorted_list.pop(0)
    for k in range(1, desired_size):
        # Finding next seed with highest marginal gain
        need_to_re_eval: bool = True
        cur_cvar = cvar(input_graph, input_alpha, greedy_solution, dist_mat=dist_mat)
        cur_ic = pmc_inf_est(input_tsv, greedy_solution)
        while need_to_re_eval:
            cur_node = sorted_list[0][0]
            # update marginal spread
            sorted_list[0] = (
                cur_node,
                input_lambda
                * (pmc_inf_est(input_tsv, greedy_solution + [cur_node]) - cur_ic)
                + (1 - input_lambda)
                * marg_dro_cvar(
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
        cur_val += marg_gain
        marg_gain_output.append(marg_gain)
        sorted_list.pop(0)

        # Verbose output
        if verbosity:
            _verbose_output(
                verbosity,
                k,
                greedy_solution,
                greedy_to_add,
                cur_val,
                comp_time,
            )
    return (greedy_solution, marg_gain_output, compute_times)
