"""Script to run for calculating and presenting influence values."""
# Standard Python library
from argparse import ArgumentParser
from csv import writer
from itertools import accumulate
from pprint import pprint
from time import perf_counter
from typing import Any, Dict, Iterator, List, Union
# other .py files
from cascade import marg_dro_inf
from config import (DEFAULT_RANDOM_GRAPH_SIZE, DEFAULT_TARGET_SIZE,
                    OUTPUT_FOLDER, TSV_FOLDER, EnumParser, GraphType,
                    HetEdgeWeightType, SolveMethod)
from graph_functions import get_graph_topo, graph_to_tsv, process_graph
from greed import accelgreedy
from pmc import pmc_greed, pmc_inf_est

# pylint: disable=E1101

METHOR_STR_PRINT_DICT = dict(zip(
    SolveMethod,
    ("Running accelerated correlationally robust greedy algorithm on graph "
     "via graph-techniques (Corr-Greedy)",
     "Running C++ pruned Monte Carlo simulations on graph (IC-Greedy)",
     "Running accelerated correlationally robust greedy algorithm on graph "
     "via LP (Corr-Greedy)")))


class Interval():
    """For checking if a number lies within an interval."""

    def __init__(self, start: float, end: float) -> None:
        """Set the left and right endpoints of interval."""
        self.start = start
        self.end = end

    def __eq__(self, other: float) -> bool:  # type: ignore
        """Use __eq__ to check for membership."""
        return self.start <= other <= self.end

    def __contains__(self, item: float) -> bool:
        """Use __contains__ so no need to put in list."""
        return self.__eq__(item)

    def __iter__(self):  # type: ignore
        """Fit choice keyword arg requiring an iterable."""
        yield self

    def __repr__(self) -> str:
        """Display endpoints of range."""
        return f'[{self.start},{self.end}]'


class Experiment():
    """Experiment class to store all experiment data."""

    def __init__(self, **experiment_args: Any) -> None:
        """Read in all the data from parsing args."""
        for key, value in experiment_args.items():
            setattr(self, key, value)
        if self.p is not None:
            self.edge_weights = self.p
        elif self.het is not None:
            self.edge_weights = self.het
        else:
            raise ValueError("Unexpected edge weights")
        self.graph = get_graph_topo(self.graph_type,
                                    self.num_nodes,
                                    self.graph_seed)
        self.graph = process_graph(self.graph, self.edge_weights)
        if self.graph_type == GraphType.random_scale_free:
            if self.graph_seed is None or self.num_nodes is None:
                raise ValueError("Expected graph_seed and num_node input.")

    def print_summary(self) -> None:
        """Print summary of experiment."""
        print(f"Number of nodes = {self.graph.vcount()}")
        print(f"Number of edges = {self.graph.ecount()}")
        print(f"Estimation seed = {self.estimation_seed}")
        if isinstance(self.edge_weights, float):
            print(f"p = {self.edge_weights}")
        elif isinstance(self.edge_weights, HetEdgeWeightType):
            print(f"Using {self.edge_weights} edge weights")
        print(METHOR_STR_PRINT_DICT[self.solution_method])

    def run(self) -> None:
        """Run influence maximization experiment on a network."""
        output_filepath = OUTPUT_FOLDER / (
            f"{self.graph_type.name},"
            f"{self.edge_weights},"
            f"{self.solution_method.name},"
            f"{self.estimation_seed}"
            ".csv"
            )
        if self.graph_seed and self.num_nodes:
            rand_str = f",{self.graph_seed},{self.num_nodes}"
        else:
            rand_str = ""
        tsv_path = TSV_FOLDER / (f"{self.graph_type.name},"
                                 f"{self.edge_weights}"
                                 f"{rand_str}"
                                 ".tsv")
        if not tsv_path.is_file():
            graph_to_tsv(self.graph, tsv_path)
        corr_marg_inf: Union[Iterator[float], List[float]]

        # Correlation Robust
        if (self.solution_method == SolveMethod.graph_techniques or
                self.solution_method == SolveMethod.linear_program):
            greed_res = accelgreedy(self.graph, self.target_seed_set_size,
                                    self.solution_method, verbosity=1)
            seed_list = greed_res[0]
            corr_marg_inf = accumulate(greed_res[1])
            pmc_marg_inf = list(pmc_inf_est(
                tsv_path,
                seed_list,
                estimation_seed=self.estimation_seed))
            ic_marg_inf = accumulate(pmc_marg_inf)
            timing_info = [x - self.start_time for x in greed_res[2]]

        # Independence Cascade
        elif self.solution_method == SolveMethod.independence_cascade:
            pmc_greed_res = pmc_greed(tsv_path, self.target_seed_set_size)
            py_time = perf_counter() - start_time
            (seed_list, _, timing_info) = pmc_greed_res
            timing_info = [x + py_time for x in timing_info]
            marg_gain_list = pmc_inf_est(
                tsv_path, seed_list,
                estimation_seed=self.estimation_seed
            )
            ic_marg_inf = accumulate(marg_gain_list)
            pprint(seed_list, compact=True)
            pi_: Dict[int, float] = {n.index: 0 for n in self.graph.vs()}
            corr_marg_inf = []
            for seed_to_add in seed_list:
                pi_ = marg_dro_inf(self.graph, pi_, seed_to_add)
                corr_marg_inf.append(sum(pi_.values()))
        else:
            raise ValueError("Unknown solution method indicated.")

        # Write output
        with open(output_filepath, mode="w", newline="") as out_file:
            csv_writer = writer(out_file)
            for seed, inf_corr, inf_ic, t_info in zip(seed_list,
                                                      corr_marg_inf,
                                                      ic_marg_inf,
                                                      timing_info
                                                      ):
                csv_writer.writerow([seed, inf_corr, inf_ic, t_info])


if __name__ == "__main__":
    # Folders are made presumably via paper.py
    # They are not included in the timing process
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    TSV_FOLDER.mkdir(parents=True, exist_ok=True)
    start_time = perf_counter()
    argparser = ArgumentParser()
    argparser.add_argument("graph_type",
                           help="type of graph for experiment to run on",
                           type=EnumParser(GraphType),  # type:ignore
                           choices=GraphType)
    argparser.add_argument("target_seed_set_size",
                           help="final size of seed set",
                           default=DEFAULT_TARGET_SIZE, type=int)
    argparser.add_argument("solution_method",
                           help="method of influence calculation",
                           type=EnumParser(SolveMethod),  # type:ignore
                           choices=SolveMethod)
    argparser.add_argument("estimation_seed",
                           help="seed used in independent cascade simulations",
                           type=int)
    argparser.add_argument("-s", "--graph_seed",
                           help=("set seed for constructing random graph,"
                                 "ignored if using real dataset"),
                           type=int)
    argparser.add_argument("-n", "--num_nodes",
                           help=("number of nodes in the graph for "
                                 "a synthetic graph"),
                           default=DEFAULT_RANDOM_GRAPH_SIZE, type=int)

    edges = argparser.add_mutually_exclusive_group(required=True)
    edges.add_argument("-p",
                       help="homogeneous edge weights for the graph",
                       type=float, choices=Interval(0, 1))
    edges.add_argument("-het",
                       help="heterogeneous edge weights for the graph",
                       type=EnumParser(HetEdgeWeightType),  # type:ignore
                       choices=HetEdgeWeightType)

    args = vars(argparser.parse_args())
    args["start_time"] = start_time
    experiment = Experiment(**args)
    experiment.print_summary()
    experiment.run()
