"""Algorithms based on Pruned Monte Carlo method by Ohsaka et. al. 2014."""
# Python Standard library
from os import listdir, remove
from pathlib import Path
from shutil import which
from subprocess import check_output
from tempfile import NamedTemporaryFile
from typing import Iterable, List
# other .py files
from config import GRT


def pmc_greed(input_tsv_path: Path, input_k: int) -> GRT:
    """Interface C++ program to run greedy pruned Monte Carlo algorithm."""
    if input_k == 0:
        return ([], [], [])
    if which("pmc_greed") is None:
        raise ValueError("pmc_greed program not found in PATH")
    pmc_output = check_output(str(x) for x in ("./pmc_greed",
                                               input_tsv_path,
                                               input_k,
                                               10000,  # 10000 simulations
                                               0  # not varying the greedy
                                                  # algorithm to ensure same
                                               )  # results for each run
                              )  # type:ignore
    pmc_output = filter(None,
                        pmc_output.decode("utf-8").replace("\r", "")
                        .replace("\t", " ").split("\n")
                        )
    seed_list = []
    marg_gain_list = []
    compute_time_list = []
    for output in pmc_output:
        seed, marg_gain, compute_time = output.split()
        seed_list.append(int(seed))
        marg_gain_list.append(float(marg_gain))
        compute_time_list.append(float(compute_time))
    return (seed_list, marg_gain_list, compute_time_list)


def pmc_inf_est(input_tsv_path: Path, input_seed_set: List[int],
                estimation_seed: int = 0) -> Iterable[float]:
    """Interface C++ program estimate seed set influence."""
    if which("pmc_est") is None:
        raise ValueError("pmc_est program not found in PATH")
    tmp = NamedTemporaryFile(mode="w+", delete=False)
    tmp.write(" ".join(str(s) for s in input_seed_set))
    tmp.close()
    inf_est_output = check_output(str(x) for x in ("./pmc_est",
                                                   input_tsv_path,
                                                   tmp.name,
                                                   10000,  # 10000 simulations
                                                   estimation_seed
                                                   )
                                  )  # type:ignore
    marg_inf = (float(x.split("\t")[1].strip())
                for x in inf_est_output.decode().split("\n")[:-1])
    remove(tmp.name)
    return marg_inf
