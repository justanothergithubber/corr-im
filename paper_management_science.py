"""Code to run to get data for extended version of paper."""

from collections import defaultdict
from csv import reader, writer
from itertools import product
from os import remove
from pathlib import Path
from pickle import dump, load
from statistics import mean, median
from tempfile import NamedTemporaryFile
from typing import Callable, Literal, cast

from igraph import Graph
from matplotlib.pyplot import (
    close,
    figure,
    legend,
    plot,
    savefig,
    title,
    xlabel,
    ylabel,
)
from more_itertools import powerset
from networkx import DiGraph, draw, draw_networkx_edge_labels, spring_layout
from numpy import (
    arange,
    array,
    asarray,
    cumsum,
    float64,
    fromiter,
    int64,
    ndarray,
    outer,
    savetxt,
    zeros,
)
from numpy.typing import NDArray
from pandas import DataFrame, Series, read_csv

from common import (
    get_all_graphs,
    get_data,
    get_dataset_graph_summaries,
    make_summary_csv,
    plot_compute_times,
    plot_expected_influence_graphs,
    plot_hists,
    read_in_graph_summary_data,
)
from config import (
    AMAZON_ALL_GENRE_FILE,
    AMAZON_NODEIDS,
    AMAZON_PAPER_OUTPUTS,
    AMAZON_RAW_OUTPUTS,
    CONV_CVAR_OUT_PATH,
    DEFAULT_TARGET_SIZE,
    DISTMAT,
    GRAPHS_USED,
    GRT,
    HETEDGEWEIGHT,
    PAPER_MS_OUTPUT_FOLDER,
    RAW_OUTPUT_FOLDER,
    TSV_FOLDER,
    GraphType,
    HetEdgeWeightType,
    SolveMethod,
)
from conv import conv_est, conv_greed
from cvar import cvar
from data import make_genre_csv
from dro import get_pi_star, get_robust_graph
from graph_functions import (
    create_graph_basic,
    get_tsv_path,
    graph_to_tsv,
    highest_outdegrees_seeds,
    solution_diameter,
    standardize_graph,
)
from greed import accelgreedy
from pmc import pmc_inf_est
from util import WriterProtocol, executable_check

CVAR_CONV_SUMMARY_PATH = Path("polblogs_cvar_conv_summary.csv")
"""Convex combinations summary file."""

MS_METHODS_USED = (
    SolveMethod.correlation_robust,
    SolveMethod.independence_cascade,
    SolveMethod.highest_outdegree,
)

# Graph settings
AMAZON_EDGES = (HetEdgeWeightType.amazonlow, HetEdgeWeightType.amazonhigh)
CVAR_EDGES = (HetEdgeWeightType.uniform, HetEdgeWeightType.trivalency)

REVERSE_EDGETYPE_DICT = {
    "trivalency": HetEdgeWeightType.trivalency,
    "uniform": HetEdgeWeightType.uniform,
}

NUM_ALPHAS_LAMBDAS = 4
ALPHA_LAMB_BASE = [x / NUM_ALPHAS_LAMBDAS for x in range(1, NUM_ALPHAS_LAMBDAS + 1)]
"""Linearly spaced alpha (CVaR parameter) values."""
ALPHAS = sorted([0.01] + ALPHA_LAMB_BASE)
"""Linearly spaced alpha (CVaR parameter) values, and 0.01."""
TRIVALENCY_ALPHAS = sorted(ALPHAS + [0.9, 0.92, 0.94, 0.96, 0.98])
"""
Alpha values to handle trivalency small values. This shows that at low
probability edge weights, a CVaR alpha which is too low will not result
in a meaningful solution, or objective
"""
LAMBDAS: tuple[float, ...] = (0.0,) + tuple(ALPHA_LAMB_BASE)
"""
Lambda refers to the convex combination coefficient of independence
cascade vs. correlation-robust solution. Lambda = 1 => pure IC
"""
OPTIMAL_SMALL_GRAPH_SIZE = 16
OPTIMAL_CSV_FILENAME = (
    PAPER_MS_OUTPUT_FOLDER / f"small_graph_{OPTIMAL_SMALL_GRAPH_SIZE}_all_data.csv"
)
NUM_ALPHAS_LAMBDAS_SMALL_GRAPH = 40
LAMBDAS_ALPHAS_SMALL_GRAPH: tuple[float, ...] = tuple(
    sorted(
        [0.01]
        + [
            x / NUM_ALPHAS_LAMBDAS_SMALL_GRAPH
            for x in range(1, NUM_ALPHAS_LAMBDAS_SMALL_GRAPH + 1)
        ]
    )
)
"""
Values that both CVaR and Convex combination parameters take for small graph.
"""

WORST_CASE_PATH = RAW_OUTPUT_FOLDER / Path("worst_case_pmf.pickle")
ECOUNT_PATH = RAW_OUTPUT_FOLDER / Path("robust_ecounts.pickle")


def get_heterogenous_seed_summaries() -> None:
    """
    Summarises some data regarding the heterogeneous edge weight graphs.

    Table 3 for Management Science version.
    """
    # read data
    data_dict, table_data = read_in_graph_summary_data()

    # Start writing
    with open(
        PAPER_MS_OUTPUT_FOLDER / "heterogeneous_edges.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as table2csv:
        table2writer = writer(table2csv)
        table2writer.writerow(
            [
                "Seed Set S",
                "Dataset",
                "p",
                "f_corr (S)",
                "f_ic (S)",
                "Min Deg",
                "Avg Deg",
                "Max Deg",
                "Diam",
            ]
        )
        for sol_method in MS_METHODS_USED:
            for graph_type in GRAPHS_USED:
                for edge_weight in HETEDGEWEIGHT:
                    expt_config = (
                        graph_type.name,
                        edge_weight.name,
                        sol_method.name,
                    )
                    # Following the table division as made
                    # in the paper
                    conf_as_written = list(
                        (expt_config[0], expt_config[2], expt_config[1])
                    )
                    table2writer.writerow(
                        conf_as_written
                        + [
                            mean(y[idx] for y in data_dict[expt_config])
                            for idx in range(2)
                        ]
                        + list(table_data[expt_config])
                    )
    print("Finished writing graph summary data")


def _data_gather(
    output_file: Path,
    data_function: Callable[..., GRT],
    *data_args,
    verbosity: int = 1,
) -> None:
    """
    Gather data if data file does not exist.

    Helper function used in other functions. If expected output file
    does not exist, a standardised method of writing output is
    utilised.
    """
    if not output_file.exists():
        data = data_function(*data_args)
        if verbosity:
            print(f"Writing {output_file}")
        with open(output_file, mode="w", newline="", encoding="utf-8") as out_io:
            csv_writer = writer(out_io)
            for data_elements in zip(*data):
                csv_writer.writerow(data_elements)


def get_cvar_conv_data(inp_graph: Graph) -> None:
    """Retrieve greedy data for all convex combination cases wanted."""
    output_file_stem: Path = CONV_CVAR_OUT_PATH / inp_graph["type"].name
    edge_type: HetEdgeWeightType = inp_graph["edge_type"]
    if inp_graph["type"] == GraphType.polblogs:
        if inp_graph["edge_type"] == HetEdgeWeightType.uniform:
            alpha_iter = ALPHAS
            lambda_iter = LAMBDAS
        else:  # if trivalency
            alpha_iter = TRIVALENCY_ALPHAS
            lambda_iter = (0,)
    else:
        alpha_iter = (1,)
        lambda_iter = LAMBDAS
    for alpha, lambda_ in product(alpha_iter, lambda_iter):
        _data_gather(
            Path(f"{output_file_stem}_{edge_type}_{alpha:.3f}_{lambda_:.3f}.csv"),
            conv_greed,
            inp_graph,
            DEFAULT_TARGET_SIZE,
            alpha,
            lambda_,
        )


def get_amazon_est_data(input_graph: Graph) -> None:
    """Get model-misspecified data from greedy seeds."""
    output_file_stem = CONV_CVAR_OUT_PATH / "amazon"
    graph_tsv = get_tsv_path(input_graph["type"], input_graph["edge_type"])
    if not graph_tsv.exists():
        graph_to_tsv(input_graph)
    edge_type = input_graph["edge_type"]
    for greedy_lambda, est_lambda in product(LAMBDAS, repeat=2):
        greedy_result_file = Path(
            f"{output_file_stem}_{edge_type}_1.000_{greedy_lambda:.3f}.csv"
        )
        out_file = Path(
            f"{output_file_stem}_{edge_type}_1.000_{greedy_lambda:.3f}_{est_lambda:.3f}.csv"
        )
        if not out_file.exists():
            greedy_seed_set: list[int] = []
            with open(
                greedy_result_file, mode="r", encoding="utf-8"
            ) as greedy_result_io:
                for line in greedy_result_io:
                    line_split = line.split(",")
                    greedy_seed_set.append(int(line_split[0]))
            with NamedTemporaryFile(mode="w+", delete=False) as tmp:
                tmp.write(" ".join(str(s) for s in greedy_seed_set))
            marg_list, compute_list = conv_est(Path(tmp.name), graph_tsv, est_lambda, 1)
            with open(
                out_file, mode="w", newline="", encoding="utf-8"
            ) as out_file_object:
                csv_writer = writer(out_file_object)
                for marg, compute_time in zip(marg_list, compute_list):
                    csv_writer.writerow([marg, compute_time])
            remove(tmp.name)
            print(f"Writing {out_file}")


def _process_csv_file(
    input_summary_writer: WriterProtocol,
    input_csv_file: Path,
    inp_graph: Graph,
) -> None:
    """
    Process csv file for summary.

    Only contains summary for `polblogs`.
    """
    with open(input_csv_file, encoding="utf-8") as csvfile:
        name_spl = input_csv_file.name.split("_")
        alpha = float(name_spl[2])
        lambda_ = float(name_spl[-1][:-4])  # removing ".csv"
        csv_reader = reader(csvfile)
        seeds: list[int] = []
        cvar_val: float = 0.0
        time = 0.0  # prevent 'possibly unbound'
        for idx, line in enumerate(csv_reader):
            cvar_val += float(line[1])
            seeds.append(int(line[0]))
            if idx == (DEFAULT_TARGET_SIZE - 1):
                time = float(line[2])

        deg: list[int] = inp_graph.degree(seeds)  # default degree mode is 'all'
        input_summary_writer.writerow(
            [inp_graph["edge_type"], alpha, lambda_, cvar_val, time]
            + [
                min(deg),
                mean(deg),
                max(deg),
                solution_diameter(inp_graph, seeds),
            ]
            + seeds
        )


def make_cvar_conv_summary(
    input_summary_csv_filename: Path,
    inp_graph_dictionary: dict[HetEdgeWeightType, Graph],
) -> None:
    """
    Create CVaR and conv summary file.

    inp_graph_dictionary is expected to be `polblogs`.
    """
    with open(
        input_summary_csv_filename,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as summary_csv:
        summary_writer = writer(summary_csv)
        summary_writer.writerow(
            [
                "Edge Type",
                "Alpha",
                "Lambda",
                "Objective Function",
                "Compute Time (s)",
            ]
            + ["Min Deg (S)", "Mean Deg (S)", "Max Deg (S)", "Diam (S)"]
            + [f"S_{x}" for x in range(DEFAULT_TARGET_SIZE)]
        )
        for file_count, csv_file in enumerate(
            CONV_CVAR_OUT_PATH.rglob("polblogs_*.csv"), 1
        ):
            filesplit = csv_file.name.split("_")
            edge_type = REVERSE_EDGETYPE_DICT[filesplit[1]]
            csv_graph = inp_graph_dictionary[edge_type]
            if not file_count % 100:
                print(f"Processed {file_count} files")
            _process_csv_file(summary_writer, csv_file, csv_graph)
    print("Finished writing CVaR and convex summary.")


def get_seeds_from_summary() -> dict[tuple[HetEdgeWeightType, float, float], list[int]]:
    """
    Retrieve solution sets for each configuration and edge type.

    First float is alpha, CVaR parameter.
    Second float is lambda, convex combination parameter.
    """
    out_seeds: dict[tuple[HetEdgeWeightType, float, float], list[int]] = {}
    with open(
        PAPER_MS_OUTPUT_FOLDER / CVAR_CONV_SUMMARY_PATH, mode="r", encoding="utf-8"
    ) as csv_file:
        csv_reader = reader(csv_file)
        next(csv_reader)  # skip header
        for line in csv_reader:
            edge_type = REVERSE_EDGETYPE_DICT[line[0]]
            alpha = float(line[1])
            lambda_ = float(line[2])
            seeds = [int(x) for x in line[9:]]
            out_seeds[(edge_type, alpha, lambda_)] = seeds
    return out_seeds


def tabulate_alpha_greedy_compare(
    inp_graph: Graph,
    inp_solns: dict[tuple[HetEdgeWeightType, float, float], list[int]],
    alphas_to_compare: list[float],
) -> None:
    """
    Create alpha greedy comparison data file if it does not exist.

    Should correspond to Tables 6 and 7 in final version.
    """
    output_csv_filename = (
        PAPER_MS_OUTPUT_FOLDER / f"cvar_{inp_graph['edge_type']}_table.csv"
    )
    if not output_csv_filename.exists() or True:
        alpha_df = DataFrame(
            columns=list(f"Eval Alpha {alpha:.2f}" for alpha in alphas_to_compare)
        )
        alpha_df.index.name = "Greedy Alpha"
        for alpha_cmpr in alphas_to_compare:
            alpha_df.loc[alpha_cmpr] = [  # type: ignore
                cvar(
                    inp_graph,
                    alpha,
                    inp_solns[(inp_graph["edge_type"], alpha_cmpr, 0)],
                )
                for alpha in alphas_to_compare
            ]
        alpha_df.loc["deg"] = [
            cvar(
                inp_graph,
                alpha,
                highest_outdegrees_seeds(inp_graph, DEFAULT_TARGET_SIZE),
            )
            for alpha in alphas_to_compare
        ]
        intermediate_df = alpha_df.round(2)
        # should silence new warnings from
        # https://pandas.pydata.org/docs/whatsnew/v2.1.0.html#deprecated-silent-upcasting-in-setitem-like-series-operations
        alpha_df = alpha_df.astype("object")
        alpha_df = intermediate_df.apply(
            lambda ser: ser.apply(lambda elem: f"{elem} ({elem/ser.max():.3f})")
        )
        alpha_df.to_csv(output_csv_filename)
    print(f"Alpha comparison table for {inp_graph['edge_type']} edges created.")


def tabulate_conv_data(input_graph: Graph) -> None:
    """
    Create tabular version of f^Mix greedy lambda vs evaluation lambda.

    Corresponds to Tables 4 and 5 in Management Science paper.
    """
    amazon_file_stem = CONV_CVAR_OUT_PATH / "amazon"
    edge_type = cast(HetEdgeWeightType, input_graph["edge_type"])
    if edge_type not in (HetEdgeWeightType.amazonlow, HetEdgeWeightType.amazonhigh):
        raise ValueError("Unexpected edge type input to function.")
    results: dict[float, list[float]] = {}
    for greedy_lambda, est_lambda in product(LAMBDAS, repeat=2):
        file_to_read = Path(
            f"{amazon_file_stem}_{edge_type}_1.000_{greedy_lambda:.3f}_{est_lambda:.3f}.csv"
        )
        with open(file_to_read, encoding="utf-8") as file_io:
            obj_gain: float = 0.0
            for line in file_io:
                obj_gain += float(line.split(",")[0])
            if greedy_lambda not in results:
                results[greedy_lambda] = [obj_gain]
            else:
                results[greedy_lambda].append(obj_gain)
    results_df = DataFrame(results).T
    results_df.index.name = "greedy lambda"
    results_df.rename(
        columns={x: f"eval lambda = {LAMBDAS[x]}" for x in range(5)}, inplace=True
    )
    intermediate_df = results_df.round(2)
    results_df = results_df.astype("object")
    results_df = intermediate_df.apply(
        lambda ser: ser.apply(lambda elem: f"{elem} ({elem/ser.max():.3f})")
    )
    results_df.to_csv(PAPER_MS_OUTPUT_FOLDER / f"conv_{edge_type}.csv")
    print(f"Finished gathering data for f^Mix on amazon for edge type {edge_type}.")


def check_stability(input_graph: Graph) -> None:
    """
    Check stability of seed sets with respect to sample size.

    Requires C++ programs to use random values rather than
    current implementation of being seeded by specific values.
    """
    stability_dir = CONV_CVAR_OUT_PATH / Path("stab")
    stability_dir.mkdir(parents=True, exist_ok=True)
    for seed_set_size in (2**i for i in range(15)):
        _data_gather(
            stability_dir / f"{seed_set_size}.csv",
            conv_greed,
            input_graph,
            DEFAULT_TARGET_SIZE,
            1,
            0.5,
            seed_set_size,
        )
    _data_gather(
        stability_dir / "stability.csv",
        conv_greed,
        input_graph,
        DEFAULT_TARGET_SIZE,
        1,
        0.5,
        2,
    )


def get_genres() -> tuple[list[str], dict[str, list[str]]]:
    """
    Get the genres of every seed.

    Result is a tuple of genre names and a dictionary with DVDs as keys and a list of essentially flags
    of whether it is that genre. An empty string means it is not that genre, and "1" represents that it
    is that genre.
    """
    out_genres: dict[str, list[str]] = {}
    with open(AMAZON_ALL_GENRE_FILE, mode="r", encoding="utf-8") as genre_io:
        out_genre_names = next(genre_io).split(",")[1:]
        out_genre_names[-1] = out_genre_names[-1].strip()
        for line in genre_io:
            ls_ = line.split(",")
            ls_[-1] = ls_[-1].strip()
            out_genres[ls_[0]] = ls_[1:]
    return out_genre_names, out_genres


def get_genre_similarity(
    input_graph: Graph, input_genre_dictionary: dict[str, list[str]]
) -> None:
    """
    Get the one-hop genres from every node / edge.

    Results were analyzed briefly but not used in paper.
    """
    node_names: list[str] = input_graph.vs()["name"]
    num_genres = len(next(iter(input_genre_dictionary.values())))
    out_matrix = zeros((num_genres, num_genres), dtype=int64)
    for node, genres in input_genre_dictionary.items():
        node_genres = fromiter(map(lambda x: int(x.strip() or 0), genres), dtype=int64)
        for out_neighbor in input_graph.neighbors(node, mode="out"):
            neighbor_genres = fromiter(
                map(
                    lambda x: int(x.strip() or 0),
                    input_genre_dictionary[node_names[out_neighbor]],
                ),
                dtype=int64,
            )
            out_matrix += outer(node_genres, neighbor_genres)
    savetxt(CONV_CVAR_OUT_PATH / "similarity.txt", out_matrix, fmt="%d")


def get_amazon_greedy_seeds(
    input_amazon_weight_type: (
        Literal[HetEdgeWeightType.amazonhigh] | Literal[HetEdgeWeightType.amazonlow]
    ),
) -> tuple[set[int], dict[float, list[int]]]:
    """
    Read the greedy seeds into Python.

    Uses only low p configuration of amazon
    """
    output_file_stem = CONV_CVAR_OUT_PATH / "amazon"
    all_greedy_seeds: set[int] = set()
    greedy_seeds_lambda: dict[float, list[int]] = {}
    for greedy_lambda in LAMBDAS:
        greedy_seeds_lambda[greedy_lambda] = []
        file_to_read = Path(
            f"{output_file_stem}_{input_amazon_weight_type}_1.000_{greedy_lambda:.3f}.csv"
        )
        with open(file_to_read, encoding="utf-8") as file_io:
            for line in file_io:
                seed = int(line.split(",")[0])
                greedy_seeds_lambda[greedy_lambda].append(seed)
                all_greedy_seeds.add(seed)
    return (all_greedy_seeds, greedy_seeds_lambda)


def get_greedy_genres(
    input_amazon_edge_type: (
        Literal[HetEdgeWeightType.amazonlow] | Literal[HetEdgeWeightType.amazonhigh]
    ),
    input_all_greedy: set[int],
    input_greedy_dict: dict[float, list[int]],
    input_headers: list[str],
    input_genre_dict: dict[str, list[str]],
) -> None:
    """
    Get the greedy genres into a summary file.

    Corresponds with Figures 13 and 14 in Management Science Paper.
    """
    nodeids: dict[int, str] = {}
    with open(AMAZON_NODEIDS, mode="r", encoding="utf-8") as nodeid_io:
        for line in nodeid_io:
            ls_ = line.split()
            nodeid, asin = int(ls_[0]), ls_[1]
            nodeids[nodeid] = asin
    with open(
        AMAZON_PAPER_OUTPUTS / f"{input_amazon_edge_type}_greedy.csv",
        mode="w",
        newline="",
        encoding="utf-8",
    ) as greedy_io:
        csv_writer = writer(greedy_io)
        csv_writer.writerow(["ASIN"] + list(LAMBDAS) + input_headers)
        for seed in input_all_greedy:
            asin = nodeids[seed]
            inside_greedy = [""] * len(LAMBDAS)
            for idx, greedy_lambda in enumerate(LAMBDAS):
                if seed in input_greedy_dict[greedy_lambda]:
                    inside_greedy[idx] = input_greedy_dict[greedy_lambda].index(seed)  # type: ignore
            csv_writer.writerow([asin] + inside_greedy + input_genre_dict[asin])

    partition_dict: defaultdict[frozenset[float], set[int]] = defaultdict(set)
    for seed in input_all_greedy:
        partition: set[float] = set()
        for partition_lambda in (0, 0.5, 1):
            if seed in input_greedy_dict[partition_lambda]:
                partition.add(partition_lambda)
        partition_dict[frozenset(partition)].add(seed)
    for combinations, combination_seeds in partition_dict.items():
        if len(combinations) > 0:
            combination_csv_file = AMAZON_PAPER_OUTPUTS / Path(
                "amazon_" + "_".join(map(str, combinations)) + ".csv"
            )
            with open(
                combination_csv_file,
                mode="w",
                newline="",
                encoding="utf-8",
            ) as combination_io:
                combination_writer = writer(combination_io)
                combination_writer.writerow(["ASIN"] + input_headers)
                for seed in combination_seeds:
                    asin = nodeids[seed]
                    combination_writer.writerow([asin] + input_genre_dict[asin])

            pd_read: DataFrame = read_csv(combination_csv_file)
            filtered_data = pd_read.dropna(axis="columns", how="all")
            filtered_data.to_csv(combination_csv_file, index=False)


def get_small_example_data() -> None:
    """Run the small example where an optimal value is found."""
    print("Getting small graph (|V| = 16) data.")
    rand_graph = standardize_graph(
        create_graph_basic(
            GraphType.random_scale_free,
            num_nodes=OPTIMAL_SMALL_GRAPH_SIZE,
            graph_seed=0,
        ),
        HetEdgeWeightType.uniform,
    )
    small_tsv_path = get_tsv_path(
        GraphType.random_scale_free,
        HetEdgeWeightType.uniform,
        OPTIMAL_SMALL_GRAPH_SIZE,
    )

    if not small_tsv_path.exists():
        graph_to_tsv(rand_graph)

    if not OPTIMAL_CSV_FILENAME.exists():
        dist_mat: DISTMAT = asarray(rand_graph.distances(weights="q"))
        with open(
            OPTIMAL_CSV_FILENAME, mode="w", newline="", encoding="utf-8"
        ) as optimal_csv:
            optimal_writer = writer(optimal_csv)
            # Header
            optimal_writer.writerow(
                ["Seed set S", "|S|"]
                + [f"conv({alpha}, 0)" for alpha in LAMBDAS_ALPHAS_SMALL_GRAPH]
                + [f"conv(1, {lambda_})" for lambda_ in LAMBDAS_ALPHAS_SMALL_GRAPH]
            )
            for seed_set in powerset(x.index for x in rand_graph.vs()):
                if len(seed_set) <= 4:
                    seed_set = cast(list[int], list(seed_set))
                    exp_inf: float = pmc_inf_est(small_tsv_path, seed_set)
                    cvar_vals = [
                        cvar(
                            rand_graph,
                            alpha,
                            seed_set,
                            dist_mat=dist_mat,
                        )
                        for alpha in LAMBDAS_ALPHAS_SMALL_GRAPH
                    ]
                    correlation_worst_case = cvar_vals[-1]  # case of alpha = 1
                    conv_vals = [
                        exp_inf * lambda_ + (1 - lambda_) * correlation_worst_case
                        for lambda_ in LAMBDAS_ALPHAS_SMALL_GRAPH
                    ]
                    optimal_writer.writerow(
                        [";".join(map(str, seed_set)), len(seed_set)]
                        + cvar_vals
                        + conv_vals
                    )


def analyse_small_example() -> None:
    """Run some small analysis of the small dataset."""
    df = read_csv(OPTIMAL_CSV_FILENAME, index_col=0)

    # Maximisers
    opti_seeds_by_size: DataFrame = df.groupby("|S|").idxmax()  # type: ignore
    opti_seeds_by_size.to_csv(
        PAPER_MS_OUTPUT_FOLDER / "small_graph_optimal_seeds_by_seed_size.csv"
    )
    opti_seeds: Series = opti_seeds_by_size.iloc[4]
    opti_seeds.to_csv(PAPER_MS_OUTPUT_FOLDER / "opti_seeds.csv")


def plot_small_example() -> None:
    """
    Plot small example for paper, highlighting optimal seeds.

    Corresponds somewhat to Figure 8
    """
    rand_graph = standardize_graph(
        create_graph_basic(GraphType.random_scale_free, OPTIMAL_SMALL_GRAPH_SIZE, 0),
        HetEdgeWeightType.uniform,
    )
    nxg = DiGraph(rand_graph.get_edgelist())
    pos = spring_layout(nxg, seed=0, k=1)  # distance, further than default
    opti_graphs = [
        [6, 7, 13, 14],  # lambda from 0.6 to 1, alpha=1
        [0, 8, 10, 14],  # alpha from 0 to 0.405
        [8, 10, 13, 14],  # alpha from 0.405 to 0.497
        [6, 10, 13, 14],  # alpha from 0.497 to 1
    ]
    node_cols: dict[int, list[str]] = {x: [] for x in range(len(opti_graphs))}
    for opti_graph_idx, opti_graph in enumerate(opti_graphs):
        figure(figsize=(16, 16))
        for node in nxg:
            if node in opti_graph:
                node_cols[opti_graph_idx].append("#FF0000")
            else:
                node_cols[opti_graph_idx].append("#8888FF")
        draw(
            nxg,
            pos=pos,
            node_color=node_cols[opti_graph_idx],
            labels={node: node for node in nxg.nodes()},
        )
        draw_networkx_edge_labels(
            nxg,
            pos,
            {e.tuple: f'{e["p"]:.3f}' for e in rand_graph.es()},
        )
        savefig(PAPER_MS_OUTPUT_FOLDER / f"small{opti_graph_idx}.png", dpi=300)
        close()


def count_like_to_like(input_graph: Graph) -> dict[tuple[int, int], int]:
    """Homophily computation."""
    counter: dict[tuple[int, int], int] = {
        x: 0 for x in ((0, 0), (0, 1), (1, 0), (1, 1))
    }
    for e in input_graph.es:
        counter[
            (
                input_graph.vs[e.source]["value"],
                input_graph.vs[e.target]["value"],
            )
        ] += 1
    return counter


def get_homophily_data(input_graph: Graph) -> None:
    """Compute data required for arguments about graph homophily."""
    if (not WORST_CASE_PATH.exists()) or (not ECOUNT_PATH.exists()):
        total_nodes: int = input_graph.vcount()
        worst_case_pmf = zeros(total_nodes)
        ecount_data: list[dict[tuple[int, int], int]] = []

        greedy_seeds = accelgreedy(input_graph, 40, SolveMethod.correlation_robust)[0]
        pi_star_tuple = get_pi_star(input_graph, greedy_seeds)
        pi_star = {pi_node[0]: pi_node[1] for pi_node in pi_star_tuple}
        pi_breakpoints = sorted(set(x[1] for x in pi_star_tuple))

        # unused variable are the breakpoints separating
        # different edge outcomes
        for breakpoint_idx, _ in enumerate(pi_breakpoints[:-1], start=1):
            q_tilde = median(
                (
                    pi_breakpoints[breakpoint_idx],
                    pi_breakpoints[breakpoint_idx - 1],
                )
            )  # any value not between but not equal to either breakpoint
            q_length = (
                pi_breakpoints[breakpoint_idx] - pi_breakpoints[breakpoint_idx - 1]
            )
            tilde_graph = get_robust_graph(
                input_graph, pi_star=pi_star, q_tilde=q_tilde
            )
            ecount_data.append(count_like_to_like(tilde_graph))
            res = zeros(total_nodes)
            histogram_counts = array(
                [y[2] for y in tilde_graph.degree_distribution().bins()]
            )
            res[0 : histogram_counts.shape[0]] = histogram_counts
            worst_case_pmf += q_length * res / total_nodes
        with open(WORST_CASE_PATH, "wb") as worst_case_file:
            dump(worst_case_pmf, worst_case_file)
        with open(ECOUNT_PATH, "wb") as ecount_file:
            dump(ecount_data, ecount_file)


def _plot_cdf(
    input_data: NDArray[float64], output_label: str, input_data_range: ndarray
) -> None:
    cdf: NDArray[float64] = cumsum(input_data)
    plot(input_data_range, cdf, label=output_label)


def graph_degree_distribution_cdf(input_graph: Graph, data_file: Path) -> None:
    """
    Create a graph of cumulative distribution function of degrees.

    Corresponds to Figure 15 of Management Science paper.
    """
    total_nodes = input_graph.vcount()
    if not isinstance(total_nodes, int):
        raise ValueError("Unexpected type returned from `igraph.Graph.vcount()")
    data_range = arange(total_nodes)
    original_res = zeros(total_nodes)
    original_histogram_counts = [
        y[2] for y in input_graph.degree_distribution().bins()
    ]  # default degree type is 'all'
    original_res[0 : len(original_histogram_counts)] = original_histogram_counts
    original_data: NDArray[float64] = original_res / total_nodes
    with open(data_file, "rb") as input_file:
        worst_case_data: NDArray[float64] = load(input_file)
    _plot_cdf(
        original_data,
        r"$\tilde{c} \sim $base $ \mathtt{polblogs}$",
        data_range,
    )
    _plot_cdf(worst_case_data, r"$\tilde{c} \sim \theta^{*}$", data_range)
    legend()
    xlabel("Degree")
    ylabel("Probability")
    title(r"Cumulative Distribution of $\tilde{D} = deg(\tilde{v}_{\tilde{c}})$")
    savefig(PAPER_MS_OUTPUT_FOLDER / "cdfs.png", dpi=300)


def make_ecount_table(input_graph: Graph, ecount_data_file: Path) -> None:
    """Create a table of edgetypes."""
    original_etypes = count_like_to_like(input_graph)
    with open(ecount_data_file, "rb") as input_file:
        robust_ecount_data = load(input_file)

    greedy_seeds = accelgreedy(input_graph, 40, SolveMethod.correlation_robust)[0]
    pi_star_tuple = get_pi_star(input_graph, greedy_seeds)
    pi_breakpoints = sorted(set(x[1] for x in pi_star_tuple))

    q_lengths: list[float] = []
    for breakpoint_idx, _ in enumerate(pi_breakpoints[:-1], start=1):
        q_length = pi_breakpoints[breakpoint_idx] - pi_breakpoints[breakpoint_idx - 1]
        q_lengths.append(q_length)
    robust_ratio: float = 0.0
    robust_ecount = {x: 0 for x in ((0, 0), (0, 1), (1, 0), (1, 1))}
    for q_len, robust_edata in zip(q_lengths, robust_ecount_data):
        for e_type in robust_ecount:
            robust_ecount[e_type] += q_len * robust_edata[e_type]
        for homophilic_type in [(0, 0), (1, 1)]:
            robust_ratio += q_len * (
                robust_edata[homophilic_type] / sum(robust_edata.values())
            )
    with open(
        PAPER_MS_OUTPUT_FOLDER / "homophily.csv",
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        csv_writer = writer(csv_file)
        csv_writer.writerow(
            [
                "E[R->R]",
                "E[D->D]",
                "E[R->D]",
                "E[D->R]",
                "E[((R->R) + E(D->D)) / # edges]",
            ]
        )
        csv_writer.writerow(
            [
                original_etypes[(0, 0)],
                original_etypes[(1, 1)],
                original_etypes[(0, 1)],
                original_etypes[(1, 0)],
                (original_etypes[(0, 0)] + original_etypes[(1, 1)])
                / input_graph.ecount(),
            ]
        )
        csv_writer.writerow(
            [
                robust_ecount[(0, 0)],
                robust_ecount[(1, 1)],
                robust_ecount[(0, 1)],
                robust_ecount[(1, 0)],
                robust_ratio,
            ]
        )


if __name__ == "__main__":
    # Create folders if they do not exist
    for fold in (
        TSV_FOLDER,
        CONV_CVAR_OUT_PATH,
        PAPER_MS_OUTPUT_FOLDER,
        AMAZON_RAW_OUTPUTS,
        AMAZON_PAPER_OUTPUTS,
    ):
        fold.mkdir(parents=True, exist_ok=True)

    # Check required executables exist
    executable_check()

    # Gather data
    get_data(MS_METHODS_USED)

    # Read in graphs
    neurips_graph_dict = get_all_graphs("neurips")
    ms_graph_dict = get_all_graphs("ms")

    # Summary stuff
    make_summary_csv(neurips_graph_dict)
    get_dataset_graph_summaries(neurips_graph_dict)
    get_heterogenous_seed_summaries()
    plot_compute_times()
    plot_hists(neurips_graph_dict)
    plot_expected_influence_graphs()

    # polblogs graphs
    polblogs = {
        polblogs_edge: ms_graph_dict[GraphType.polblogs, polblogs_edge]
        for polblogs_edge in HETEDGEWEIGHT
    }

    # amazon graphs
    make_genre_csv()
    amazon = {
        amazon_edge: ms_graph_dict[GraphType.amazon, amazon_edge]
        for amazon_edge in AMAZON_EDGES
    }

    # do convex combination greedy
    print("Starting to get amazon convex combination data.")
    for amazon_edge in AMAZON_EDGES:
        amazon_tsv = get_tsv_path(GraphType.amazon, amazon_edge)
        if not amazon_tsv.exists():
            graph_to_tsv(amazon[amazon_edge])
        get_cvar_conv_data(amazon[amazon_edge])
        get_amazon_est_data(amazon[amazon_edge])
        tabulate_conv_data(amazon[amazon_edge])
    print("Finished getting amazon convex combination data.")

    # amazon genre
    genre_names, genre_dict = get_genres()
    for amazon_edge in AMAZON_EDGES:
        all_greedy, greedy_lambda_dict = get_amazon_greedy_seeds(amazon_edge)
        get_greedy_genres(
            amazon_edge, all_greedy, greedy_lambda_dict, genre_names, genre_dict
        )

    # Not output in final paper
    get_genre_similarity(amazon[HetEdgeWeightType.amazonlow], genre_dict)

    # polblogs conv, cvar data
    # Ensure necessary tsvs exist
    polblog_tsvs = {
        HetEdgeWeightType.uniform: get_tsv_path(
            GraphType.polblogs, HetEdgeWeightType.uniform
        ),
        HetEdgeWeightType.trivalency: get_tsv_path(
            GraphType.polblogs, HetEdgeWeightType.trivalency
        ),
    }
    for polblog_edge_type, polblog_tsv in polblog_tsvs.items():
        if not polblog_tsv.exists():
            graph_to_tsv(polblogs[polblog_edge_type])
    for cvar_edge in CVAR_EDGES:
        get_cvar_conv_data(polblogs[cvar_edge])

    # make summary for convex and CVaR experiments
    make_cvar_conv_summary(PAPER_MS_OUTPUT_FOLDER / CVAR_CONV_SUMMARY_PATH, polblogs)

    # get solutions from summary
    conv_cvar_solns = get_seeds_from_summary()

    # tabulate alpha comparison graphs for certain greedy solutions
    for cvar_edge, cvar_alphas in zip(CVAR_EDGES, (ALPHAS, TRIVALENCY_ALPHAS)):
        tabulate_alpha_greedy_compare(polblogs[cvar_edge], conv_cvar_solns, cvar_alphas)

    # homophily investigations
    # Output in appendix
    get_homophily_data(polblogs[HetEdgeWeightType.weightedcascade])
    graph_degree_distribution_cdf(
        polblogs[HetEdgeWeightType.weightedcascade], WORST_CASE_PATH
    )
    make_ecount_table(polblogs[HetEdgeWeightType.weightedcascade], ECOUNT_PATH)

    # find optimal for small example
    get_small_example_data()
    analyse_small_example()
    plot_small_example()

    print("Finished running script.")
