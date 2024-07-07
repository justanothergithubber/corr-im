"""Code to run to get data for NeurIPS paper."""

# Python standard library
from csv import reader, writer
from itertools import cycle
from statistics import mean
from typing import cast

# packages
from matplotlib.pyplot import close, figure, savefig
from networkx import DiGraph, draw, spring_layout, strongly_connected_components
from PIL import Image

# from other files
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
    GRAPHS_USED,
    HETEDGEWEIGHT,
    NEURIPS_FOLDER,
    NEURIPS_METHODS_USED,
    NEURIPS_SOL_TUPLE,
    RAW_OUTPUT_FOLDER,
    SUMMARY_CSV,
    GraphDict,
    GraphType,
    HetEdgeWeightType,
    SolveMethod,
)
from util import executable_check, trim


def draw_viz(input_graph_dict: GraphDict) -> None:
    """
    Create graph visualization.

    Requires some specific data files.
    """
    data = RAW_OUTPUT_FOLDER.glob("polblogs,weightedcascade,*,0.csv")
    seeds: dict[str, set[int]] = {x: set() for x in NEURIPS_SOL_TUPLE}
    for data_filename, sol_type in zip(data, NEURIPS_SOL_TUPLE):
        with open(data_filename, mode="r", encoding="utf-8") as data_file:
            for line in data_file:
                seeds[sol_type].add(int(line.split(",")[0]))
    graph = input_graph_dict[GraphType.polblogs, HetEdgeWeightType.weightedcascade]

    # Converting to NetworkX to draw graphs
    nxg = DiGraph()
    nxg.add_nodes_from([v.index for v in graph.vs()])
    nxg.add_edges_from([e.tuple for e in graph.es()])
    largest_scc: set[int] = max(strongly_connected_components(nxg), key=len)
    largest_scc_graph = cast(DiGraph, nxg.subgraph(largest_scc))

    # For later
    pixel_limit = Image.MAX_IMAGE_PIXELS
    for sol_type, figlabel in zip(NEURIPS_SOL_TUPLE, "ab"):
        node_color: list[str] = []
        edge_col: list[str] = []
        for scc_node in largest_scc_graph:
            if scc_node in seeds[sol_type]:
                # alternative could be #6161FF
                node_color.append("#EB0000")  # red, slightly darker
                edge_col.append("#FFFF00")  # yellow
            else:
                node_color.append("white")
                edge_col.append("black")
        figure(figsize=(32, 32))
        draw(
            largest_scc_graph,
            pos=spring_layout(nxg, seed=0),
            node_size=80,
            width=0.04,
            node_color=node_color,
            edgecolors=edge_col,
            linewidths=0.8,
            arrows=False,
        )
        img_name = NEURIPS_FOLDER / f"viz_{figlabel}_full.png"
        # Suppress DecompressionBombWarning
        Image.MAX_IMAGE_PIXELS = 163840001
        savefig(img_name, dpi=400)
        close()
        trim(img_name)
        # crop dimensions
        # hardcoded for polblogs largest scc
        image = Image.open(img_name)
        crop_dim = (3600, 3000, 9000, 7700)
        cropped_example = image.crop(crop_dim)
        cropped_example.save(NEURIPS_FOLDER / f"viz_{figlabel}_crop.png")
    Image.MAX_IMAGE_PIXELS = pixel_limit  # Reset to default.

    print("Finished drawing graph visualizations.")


def get_graph_summary_data() -> None:
    """
    Summarises some data regarding the heterogeneous edge weight graphs.

    Table 2 for NeurIPS paper.
    """
    # read data
    data_dict, table_data = read_in_graph_summary_data()

    # Start writing
    with open(
        NEURIPS_FOLDER / "table2.csv", mode="w", encoding="utf-8", newline=""
    ) as table2csv:
        table2writer = writer(table2csv)
        for graph_type in list(GRAPHS_USED)[::-1]:
            for sol_method in NEURIPS_METHODS_USED:
                for edge_weight in HETEDGEWEIGHT:
                    if sol_method == SolveMethod.correlation_robust:
                        other_method = SolveMethod.independence_cascade
                        obj_idx = 1  # with reference to above obj_data
                    else:
                        other_method = SolveMethod.correlation_robust
                        obj_idx = 0
                    expt_config = (
                        graph_type.name,
                        edge_weight.name,
                        sol_method.name,
                    )
                    cmpr_config = (
                        graph_type.name,
                        edge_weight.name,
                        other_method.name,
                    )
                    mispec = mean(y[obj_idx] for y in data_dict[expt_config]) / mean(
                        y[obj_idx] for y in data_dict[cmpr_config]
                    )
                    # Following the table division as made
                    # in the paper
                    conf_as_written = list(
                        (expt_config[0], expt_config[2], expt_config[1])
                    )
                    table2writer.writerow(
                        conf_as_written
                        + [round(mispec, 3)]
                        + list(table_data[expt_config])
                    )

    print("Finished writing graph summary")


def make_table2_tex() -> None:
    """Create table2 tex from CSV data."""
    # read data
    table2_data: list[list[str | float | int]] = []
    with open(NEURIPS_FOLDER / "table2.csv", mode="r", encoding="utf-8") as table2_csv:
        csv_reader = reader(table2_csv)
        for line in csv_reader:  # type: ignore
            # Preconverting line numbers
            line: list[str | float]
            line[3] = round(float(line[3]), 3)
            line[4] = int(float(line[4]))
            line[5] = float(line[5])
            line[6] = int(float(line[6]))
            line[7] = int(float(line[7]))
            table2_data.append(line)

    graph = cycle(("wikivote", "polblogs"))
    ic_corr = cycle(NEURIPS_SOL_TUPLE)
    edge_weights = cycle(("Unif(0,1)", "Trivalency", "W.C."))
    col_widths = {
        # change this when column widths change
        # Numbering starts from 1
        1: 34,
        2: 41,
        3: 12,
        4: 14,
        5: 12,
        6: 16,
        7: 12,
        8: 27,
    }
    headers = [
        "Dataset",
        "Seed Set",
        "$\\mathbf{p}$",
        "Mis-spec Ratio",
        "Min Deg($S$)",
        "Average Deg($S$)",
        "Max Deg($S$)",
    ]

    # Headers and beginning the table, tabular environment
    with open(NEURIPS_FOLDER / "table2.tex", mode="w", encoding="utf-8") as table2_tex:
        table2_tex.write("\\begin{table}[h!]\n")
        cur_indent: int = 2
        table2_tex.write(" " * cur_indent)
        table2_tex.write("\\begin{tabularx}{\\textwidth}{|l|X|XXXXXX|}\n")
        cur_indent += 2
        table2_tex.write(" " * cur_indent)
        table2_tex.write("\\hline\n")
        table2_tex.write(" " * cur_indent)
        for idx, header in enumerate(headers):
            table2_tex.write(header.ljust(col_widths[idx + 1]))
            table2_tex.write(" & ")
        table2_tex.write("$\\text{Diam}\\left(S\\right)$".ljust(col_widths[8]))
        table2_tex.write(" \\\\ \\hline")
        table2_tex.write("\n")

        # Table data proper
        for line_idx, line in enumerate(table2_data):
            table2_tex.write(" " * cur_indent)

            # Dataset
            if (line_idx) % 6 == 0:  # every 6 lines
                text = f"\\multirow{{6}}{{*}}{{\\texttt{{{next(graph)}}}}}"
            else:
                text = ""
            table2_tex.write(text.ljust(col_widths[1]))
            table2_tex.write(" & ")

            # Seed set
            if (line_idx + 3) % 3 == 0:  # every 3 lines
                text = (
                    "\\multirow{3}{*}" f"{{$\\mathcal{{S}}^{{g}}_{{{next(ic_corr)}}}$}}"
                )
            else:
                text = ""
            table2_tex.write(text.ljust(col_widths[2]))
            table2_tex.write(" & ")
            # Edge weight type
            table2_tex.write(f"{next(edge_weights)}".ljust(col_widths[3]))

            # Seed set statistics
            for numb_idx, number in enumerate(line[3:]):
                table2_tex.write(" & ")
                table2_tex.write(str(number).ljust(col_widths[numb_idx + 4]))

            # Ending each line of table
            table2_tex.write(" \\\\")
            if (line_idx + 1) % 6 == 0:  # on lines 6, 12 wrt table
                table2_tex.write(" \\hline")
            elif (line_idx + 4) % 6 == 0:  # on lines 3, 9
                table2_tex.write(" \\cline{2-8}")
            table2_tex.write("\n")

        # Captions and closing environment
        cur_indent = 2
        closing_environment_items = (
            " " * cur_indent,
            "\\end{tabularx}\n",
            "\\caption{Properties of $\\mathcal{S}_{ic}^g$ and ",
            "$\\mathcal{S}_{corr}^g$ for non-identical edge ",
            "probabilities. $k=40$.}\n",
            "\\label{tab:summary}\n",
            "\\vspace{-5mm}\n",
            "\\end{table}",
        )
        for closing_item in closing_environment_items:
            table2_tex.write(closing_item)
    print("Finished writing .tex for table 2")


def _get_variation_against_mean(input_list: list[float]) -> float:
    """Return variation against mean."""
    return (max(input_list) - min(input_list)) / mean(input_list)


def get_variation() -> None:
    """
    Print out the variation between independence cascade runs.

    Function requires C++ executable to have random results, rather
    than being seeded as it is now.
    """
    with open(SUMMARY_CSV, mode="r", encoding="utf-8") as summary_file:
        summary_reader = reader(summary_file)
        next(summary_reader)  # skip headers
        all_data: dict[tuple[str, ...], list[float]] = {}
        for line in summary_reader:
            expt_config = tuple(line[0:3])
            if expt_config in all_data:
                all_data[expt_config].append(float(line[5]))
            else:
                all_data[expt_config] = [float(line[5])]
        max_var = max(map(_get_variation_against_mean, all_data.values()))
        print(
            "Maximum variation of independent cascade values against mean "
            f"is {max_var:.3%}"
        )


if __name__ == "__main__":
    NEURIPS_FOLDER.mkdir(exist_ok=True, parents=True)
    executable_check()
    get_data(NEURIPS_METHODS_USED)
    neurips_graph_dict = get_all_graphs("neurips")
    make_summary_csv(neurips_graph_dict)
    get_graph_summary_data()
    plot_compute_times()
    plot_hists(neurips_graph_dict)
    plot_expected_influence_graphs()
    draw_viz(neurips_graph_dict)
    get_dataset_graph_summaries(neurips_graph_dict)
    make_table2_tex()
    get_variation()

    print("Finished running script.")
