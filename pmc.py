"""
Algorithms based on Pruned Monte Carlo method by Ohsaka et. al. 2014.

Ohsaka, N., Akiba, T., Yoshida, Y., & Kawarabayashi, K.- ichi. (2014).
Fast and Accurate Influence Maximization on Large Networks with Pruned
Monte-Carlo Simulations. Proceedings of the AAAI Conference on
Artificial Intelligence, 28(1).
https://doi.org/10.1609/aaai.v28i1.8726
"""

# Python Standard library imports
from os import name, remove
from pathlib import Path
from subprocess import check_output
from tempfile import NamedTemporaryFile
from typing import cast

from config import GRT, PMC_EST_EXE_NAME


def pmc_greed(input_tsv_path: Path, input_k: int) -> GRT:
    """Interface C++ program to run greedy pruned Monte Carlo algorithm."""
    if input_k == 0:
        return ([], [], [])
    # not varying the greedy algorithm to ensure same results for each run
    pmc_output = cast(
        bytes,
        check_output(
            str(x)
            for x in (
                "./pmc_greed",
                input_tsv_path,
                input_k,
                10000,  # 10000 simulations
                0,
            )  # type: ignore
        ),
    )
    pmc_output = filter(
        None,
        pmc_output.decode("utf-8").replace("\r", "").replace("\t", " ").split("\n"),
    )
    seed_list: list[int] = []
    marg_gain_list: list[float] = []
    compute_time_list: list[float] = []
    for output in pmc_output:
        seed, marg_gain, compute_time = output.split()
        seed_list.append(int(seed))
        marg_gain_list.append(float(marg_gain))
        compute_time_list.append(float(compute_time))
    return (seed_list, marg_gain_list, compute_time_list)


def all_pmc_inf_est(
    input_tsv_path: Path, input_seed_set: list[int], estimation_seed: int = 0
) -> tuple[list[float], list[float]]:
    """Interface C++ program estimate seed set influence."""
    exe_str = f"./{PMC_EST_EXE_NAME}"
    if name == "nt":  # windows vs linux
        exe_str = exe_str[2:]
    tmp = NamedTemporaryFile(mode="w+", delete=False)
    tmp.write(" ".join(str(s) for s in input_seed_set))
    tmp.close()
    inf_est_output = cast(
        bytes,
        check_output(
            str(x)
            for x in (
                exe_str,
                input_tsv_path,
                tmp.name,
                10000,  # 10000 simulations
                estimation_seed,
            )  # type: ignore
        ),
    )
    inf_est_output = filter(
        None,
        inf_est_output.decode("utf-8").replace("\r", "").replace("\t", " ").split("\n"),
    )
    marg_gain_list: list[float] = []
    compute_time_list: list[float] = []
    for output in inf_est_output:
        _, marg_gain, compute_time = output.split()
        marg_gain_list.append(float(marg_gain))
        compute_time_list.append(float(compute_time))
    remove(tmp.name)
    return (marg_gain_list, compute_time_list)


def pmc_inf_est(
    input_tsv_path: Path, input_seed_set: list[int], estimation_seed: int = 0
) -> float:
    """
    Estimate the Independent cascade objective value.

    Estimation computed given specific seeds.
    """
    return sum(all_pmc_inf_est(input_tsv_path, input_seed_set, estimation_seed)[0])
