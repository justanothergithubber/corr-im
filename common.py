"""
Module that contains functions common to both papers.
"""

from csv import reader, writer
from itertools import cycle, product
from pathlib import Path
from pickle import dump, load
from statistics import mean
from subprocess import call
from typing import Any, Literal, cast

from matplotlib.pyplot import (
    close,
    errorbar,
    figure,
    gca,
    hist,
    legend,
    plot,
    savefig,
    title,
    xlabel,
    xlim,
    ylabel,
)
from matplotlib.ticker import MaxNLocator

from config import (
    COMMON_PAPER_FOLDER,
    DEFAULT_TARGET_SIZE,
    EDGE_WEIGHTS,
    EST_SEEDS,
    GRAPHS_USED,
    K_RANGE,
    NEURIPS_METHODS_USED,
    NEURIPS_SOL_NAMES,
    NEURIPS_SOL_SHORT_NAMES,
    NEURIPS_SOL_TUPLE,
    P_LIST,
    RAW_OUTPUT_FOLDER,
    SUMMARY_CSV,
    TSV_FOLDER,
    GraphDict,
    GraphType,
    HetEdgeWeightType,
    SolveMethod,
)
from graph_functions import create_graph_basic, solution_diameter, standardize_graph
from util import trim


def get_dataset_graph_summaries(input_graph_dict: GraphDict) -> None:
    """
    Create a `.csv` file that contains basic dataset summary statistics.

    Table 1 for NeuRIPS and Table 2 for Management Science.
    """
    with open(
        COMMON_PAPER_FOLDER / "datasets_summary.csv",
        mode="w",
        encoding="utf-8",
        newline="",
    ) as dataset_csv:
        csv_writer = writer(dataset_csv)
        csv_writer.writerow(
            ["Dataset", "|V|", "|E|", "Min Deg", "Average Deg", "Max Deg"]
        )
        for graph_type in list(GRAPHS_USED)[::-1]:
            # any edge weight will do
            graph = input_graph_dict[graph_type, 0.01]
            graph_degree = cast(list[int], graph.degree())
            csv_writer.writerow(
                [
                    graph_type.name,
                    graph.vcount(),
                    graph.ecount(),
                    min(graph_degree),
                    mean(graph_degree),
                    max(graph_degree),
                ]
            )
    print("Finished writing dataset graphs summaries.")


def get_data(input_methods: tuple[SolveMethod, ...]) -> None:
    """
    Gather all data at once.

    This function gathers all data with methods specified as input.
    We vary over edge weights, which are a single number to denote a common
    probability for every edge, or over some simple methods of assigning
    heterogenous probabilities over every edge.
    """
    print("Gathering data...")
    RAW_OUTPUT_FOLDER.mkdir(exist_ok=True)
    TSV_FOLDER.mkdir(parents=True, exist_ok=True)
    for graph_type, solution_method, est_seed, edge_weight in product(
        GRAPHS_USED, input_methods, EST_SEEDS, EDGE_WEIGHTS
    ):
        expected_filename = (
            RAW_OUTPUT_FOLDER
            / f"{graph_type},{edge_weight},{solution_method},{est_seed}.csv"
        )
        if not expected_filename.exists():
            if isinstance(edge_weight, HetEdgeWeightType):
                edge_weight_type = "-het"
            else:
                edge_weight_type = "-p"
            call(
                map(
                    str,
                    (
                        "python3",
                        "experiment.py",
                        graph_type,
                        DEFAULT_TARGET_SIZE,
                        solution_method,
                        est_seed,
                        edge_weight_type,
                        edge_weight,
                    ),
                )  # type: ignore
            )
    print("Finished gathering data")


def make_summary_csv(input_graph_dict: GraphDict) -> None:
    """
    Create a CSV summarising all data.

    Common to both papers.
    """
    with open(SUMMARY_CSV, mode="w", encoding="utf-8", newline="") as summary_file:
        csv_writer = writer(summary_file)
        csv_writer.writerow(
            [
                "Graph Type",
                "Edge Weight",
                "Solution Method",
                "Estimation Seed",
                "f_corr",
                "f_ic",
                "Compute time (s)",
                "Min Deg(S)",
                "Avg Deg(S)",
                "Max Deg (S)",
                "Diameter(S)",
            ]
            + [f"S_{x}" for x in K_RANGE]
        )
        # disable unbound errors
        soln_stats: list[float] = []
        deg: list[int] = []
        for idx, filename in enumerate(RAW_OUTPUT_FOLDER.glob("*.csv")):
            # Simple output for end user to read
            if not (idx + 1) % 100:
                print(f"Processed {idx + 1} raw output files")

            # 4th character onwards because no need for 'out'
            fnsplit: list[Any] = str(filename)[4::].split(",")
            fnsplit[3] = fnsplit[3][:-4:]  # remove .csv from string
            fnsplit[0] = GraphType[fnsplit[0]]

            # Try converting to constant p
            try:
                fnsplit[1] = float(fnsplit[1])
            except ValueError:
                pass
            # or recognise as a heterogeneous edge weight
            try:
                fnsplit[1] = HetEdgeWeightType[fnsplit[1]]  # type: ignore
            except KeyError:
                pass
            graph = input_graph_dict[tuple(fnsplit[0:2])]
            with open(filename, mode="r", encoding="utf-8") as data_file:
                data_reader = reader(data_file)
                seeds: list[int] = []
                for counter, line in enumerate(data_reader):
                    seeds.append(int(line[0]))
                    if counter + 1 == DEFAULT_TARGET_SIZE:
                        soln_stats = [float(x) for x in line[1:4]]
                        # default "degree" is "ALL"
                        # so includes in-edges and out-edges
                        deg = graph.degree(seeds)

            # Write in data in the order of
            # expertiment config, k=40 stats, degree stats
            # diameter, then seeds
            csv_writer.writerow(
                fnsplit
                + soln_stats
                + [
                    min(deg),
                    mean(deg),
                    max(deg),
                    solution_diameter(graph, seeds),
                ]
                + seeds
            )
    print(f"Finished writing {SUMMARY_CSV}")


def read_in_graph_summary_data() -> tuple[
    dict[tuple[str, ...], list[tuple[float, ...]]],
    dict[tuple[str, ...], tuple[float, ...]],
]:
    """
    Helper function that reads in the summary file, getting summary results.

    Requires `summary.csv` generated by :py:func:`make_summary_csv`
    """
    out_data_dict: dict[tuple[str, ...], list[tuple[float, ...]]] = {}
    out_table_data: dict[tuple[str, ...], tuple[float, ...]] = {}
    with open(SUMMARY_CSV, mode="r", encoding="utf-8") as summary_file:
        summary_reader = reader(summary_file)
        for line in summary_reader:
            try:
                HetEdgeWeightType[line[1]]
            except KeyError:
                continue
            graph_config = tuple(line[0:3])
            obj_data = tuple(float(x) for x in line[4:6])
            seed_set_graph_data = tuple(float(x) for x in line[7:11])
            if graph_config in out_data_dict:
                out_data_dict[graph_config].append(obj_data)
            else:
                out_table_data[graph_config] = seed_set_graph_data
                out_data_dict[graph_config] = [obj_data]
    return out_data_dict, out_table_data


def get_all_graphs(input_paper_type: Literal["ms"] | Literal["neurips"]) -> GraphDict:
    """
    Initialize the graph dictionary.

    The graphs mapped to by the dictionary will be used in the later functions.
    """
    out_graph_dict: GraphDict = {}
    common_pickle_path = RAW_OUTPUT_FOLDER / Path(f"neurips.pickle")
    if not common_pickle_path.exists():
        for graph_type in GRAPHS_USED:
            for edge_weight in EDGE_WEIGHTS:
                out_graph_dict[graph_type, edge_weight] = standardize_graph(
                    create_graph_basic(graph_type), edge_weight
                )
        with open(common_pickle_path, "wb") as graph_pickle_file:
            dump(out_graph_dict, graph_pickle_file)
    else:
        with open(common_pickle_path, "rb") as graph_pickle_file:
            out_graph_dict = load(graph_pickle_file)
    if input_paper_type == "ms":
        ms_only_graphs: GraphDict = {}
        ms_pickle_path = RAW_OUTPUT_FOLDER / Path(f"{input_paper_type}.pickle")
        if not ms_pickle_path.exists():
            for amazon_edge in (
                HetEdgeWeightType.amazonlow,
                HetEdgeWeightType.amazonhigh,
            ):
                ms_only_graphs[GraphType.amazon, amazon_edge] = standardize_graph(
                    create_graph_basic(GraphType.amazon), amazon_edge
                )
            with open(ms_pickle_path, "wb") as graph_pickle_file:
                dump(ms_only_graphs, graph_pickle_file)
            out_graph_dict = out_graph_dict | ms_only_graphs
        else:
            with open(ms_pickle_path, "rb") as graph_pickle_file:
                out_graph_dict = out_graph_dict | load(graph_pickle_file)
    return out_graph_dict


def plot_compute_times() -> None:
    """
    Plot computational time against various k, up to 40.

    Requires outputs of :py:func:`get_data` over NeuRIPS methods used.
    NeuRIPS methods are a subset of methods in MS paper.
    """
    # read in data
    all_data: dict[tuple[GraphType, SolveMethod, float], list[list[float]]] = {}
    for graph_type in GRAPHS_USED:
        for sol_method in NEURIPS_METHODS_USED:
            for edge_weight in (0.01, 0.95):
                data: list[list[float]] = []
                for data_filename in RAW_OUTPUT_FOLDER.glob(
                    f"{graph_type},{edge_weight},{sol_method}" "*"
                ):
                    file_data: list[float] = []
                    with open(data_filename, mode="r", encoding="utf-8") as data_file:
                        data_reader = reader(data_file)
                        for line in data_reader:
                            file_data.append(float(line[3]))
                    data.append(file_data)
                all_data[(graph_type, sol_method, edge_weight)] = data

    # plot
    linestyles_used = cycle(("dashdot", "dashed", "dotted", "solid"))
    for edge_weight, figlabel in zip((0.01, 0.95), "ab"):
        figure()
        for graph_type in GRAPHS_USED:
            for sol_method in NEURIPS_METHODS_USED:
                data = all_data[graph_type, sol_method, edge_weight]
                method_name = NEURIPS_SOL_NAMES[sol_method]
                mean_data = list(map(mean, zip(*data)))
                min_data = list(map(min, zip(*data)))
                max_data = list(map(max, zip(*data)))
                label = f"{graph_type.name}, {method_name}"
                if sol_method == SolveMethod.correlation_robust:
                    plot(
                        K_RANGE,
                        mean_data,
                        label=label,
                        linestyle=next(linestyles_used),
                    )
                else:
                    errorbar(
                        K_RANGE,
                        mean_data,
                        yerr=(
                            [x - y for x, y in zip(mean_data, min_data)],
                            [y - x for x, y in zip(mean_data, max_data)],
                        ),
                        label=label,
                        linestyle=next(linestyles_used),
                    )
        xlabel("$k$")
        ylabel("Computational time in seconds")
        title(("Computational times for greedy algorithms " f"when p = {edge_weight}"))
        legend()
        figure_filename = COMMON_PAPER_FOLDER / f"compute_vs_k_{figlabel}.png"
        savefig(figure_filename, dpi=300)
        close()
        trim(figure_filename)

    print("Finished plotting computational times against k.")


def plot_hists(input_graph_dict: GraphDict) -> None:
    """
    Plot the histograms of the degrees of the seed sets.

    Requires `summary.csv` generated by :py:func:`make_summary_csv`
    """
    seeds: dict[str, list[int]] = {}
    with open(SUMMARY_CSV, mode="r", encoding="utf-8") as summary_file:
        summary_reader = reader(summary_file)
        for line in summary_reader:
            if line[3] != "0":  # Estimation seed 0
                continue
            if (
                line[0] == GraphType.polblogs.name
                and line[1] == HetEdgeWeightType.uniform.name
            ) and (
                line[2]
                in (
                    SolveMethod.correlation_robust.name,
                    SolveMethod.independence_cascade.name,
                )
            ):
                seeds[NEURIPS_SOL_SHORT_NAMES[line[2]]] = [int(x) for x in line[11::]]

    for sol_type, figlab in zip(NEURIPS_SOL_TUPLE, "ab"):
        figure()
        graph = input_graph_dict[GraphType.polblogs, HetEdgeWeightType.uniform]
        # .degree default mode is "ALL"
        deg: list[int] = sorted(graph.degree(seeds[sol_type]), reverse=True)
        hist(deg, bins="auto")
        gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        title(f"Histogram of $S_{{{sol_type}}}^{{g}}$ for " "polblogs dataset, $k=40$")
        xlabel("Degree of nodes")
        ylabel("Frequency")
        figure_filename = COMMON_PAPER_FOLDER / f"deg_hist_{figlab}.png"
        savefig(figure_filename, dpi=300)
        close()
        trim(figure_filename)

    print("Finished plotting histograms of degrees.")


def plot_expected_influence_graphs() -> None:
    """
    Plot expected influence against marginal probabilities.

    Requires `summary.csv` generated by :py:func:`make_summary_csv`.

    Plot expected influence against marginal probabilities for
    `wikivote` and `polblogs` for both objectives for seed sets
    obtained in both approximation methods.
    """
    # read data
    markers = (x for x in cycle(("o", "^", "s", "p")))
    all_data: dict[tuple[str, ...], list[tuple[float, ...]]] = {}
    with open(SUMMARY_CSV, mode="r", encoding="utf-8") as summary_file:
        summary_reader = reader(summary_file)
        next(summary_reader)
        for line in summary_reader:
            graph_config = tuple(line[0:3])
            obj_data = tuple(float(x) for x in line[4:6])
            if graph_config in all_data:
                all_data[graph_config].append(obj_data)
            else:
                all_data[graph_config] = [obj_data]

    # plot
    for graph_type, figlabel in zip(GRAPHS_USED, "ba"):
        figure()
        for sol_method in NEURIPS_METHODS_USED:
            sol_str = NEURIPS_SOL_NAMES[sol_method]
            for obj_idx, obj_str in zip((0, 1), NEURIPS_SOL_TUPLE):
                e_ix: list[float] = []
                for edge_weight in P_LIST:
                    expt_config = (
                        graph_type.name,
                        str(edge_weight),
                        sol_method.name,
                    )
                    e_ix.append(mean(y[obj_idx] for y in all_data[expt_config]))
                plot(
                    P_LIST,
                    e_ix,
                    label=f"$f_{{{obj_str}}}\\left(S_{{{sol_str}}}^{{g}}\\right)$",
                    linewidth=3,
                    marker=next(markers),
                    markersize=8,
                )
        xlim(0, 1)
        xlabel("Marginal Probabilities $p$")
        ylabel("Expected Influence")
        legend()
        fig_filename = COMMON_PAPER_FOLDER / f"inf_vs_p_{figlabel}.png"
        savefig(fig_filename, dpi=300)
        close()
        trim(fig_filename)

    print("Finished plotting expected influence against p.")
