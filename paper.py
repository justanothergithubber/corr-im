"""Code to run to get data for paper."""
# Python standard library
from csv import reader, writer
from itertools import cycle
from pathlib import Path
from statistics import mean
from subprocess import call
from typing import Dict, List, Set, Tuple, Union
# packages
from igraph import Graph
from matplotlib.pyplot import (close, errorbar, figure, gca, hist, legend,
                               plot, savefig, title, xlabel, xlim, ylabel)
from matplotlib.ticker import MaxNLocator
from networkx import (DiGraph, draw, spring_layout,
                      strongly_connected_components)
from PIL import Image, ImageOps
# from other files
from config import (DEFAULT_TARGET_SIZE, OUTPUT_FOLDER, TSV_FOLDER,
                    GraphType, HetEdgeWeightType, SolveMethod)
from graph_functions import (get_graph_topo, get_solution_diameter,
                             process_graph)

# These constants are paper-specific
GRAPHS_USED = (GraphType.polblogs, GraphType.wikivote)
SOLUTION_METHODS_USED = (SolveMethod.graph_techniques,
                         SolveMethod.independence_cascade)
# A bit of aliasing
SOL_SHORT_NAMES = {
    SolveMethod.graph_techniques.name: "corr",
    SolveMethod.independence_cascade.name: "ic",
}
SOL_NAMES = {
    SolveMethod.graph_techniques: "corr",
    SolveMethod.independence_cascade: "ic"
}
SOL_TUPLE: Tuple[str, str] = ("corr", "ic")

# Paper outputs, processed data as compared to folder "out" which is raw data
PAPER_DATA_FOLDER = Path("paper")
# Summary CSV, to be written then read for analysis
SUMMARY_CSV = "summary.csv"
# More Aliasing
# Graph Dictionary Type
GraphDict = Dict[Tuple[GraphType, Union[float, GraphType]], Graph]
# Alias for all values that k takes
K_RANGE = range(1, DEFAULT_TARGET_SIZE + 1)
# Aliasing for Edge weights
P_LIST = [0.01] + [x/20 for x in range(1, 20)]
EDGE_WEIGHTS = P_LIST + list(HetEdgeWeightType)
# Alias for estimation seeds we use
EST_SEEDS = range(0, 100000, 10000)


def trim(img_name: Union[Path, str]) -> None:
    """
    Trim and save image.

    Idea to invert from https://stackoverflow.com/a/57552536
    Removes white and black borders.
    """
    image = Image.open(img_name)
    black_crop = image.crop(image.getbbox())
    inverted_image = ImageOps.invert(black_crop.convert('RGB'))
    final_crop = image.crop(inverted_image.getbbox())
    final_crop.save(img_name)


def get_all_graphs() -> GraphDict:
    """
    Initialize the graph dictionary.

    The graph dictionary will be subsequently used in later functions.
    """
    out_graph_dict = {}
    for graph_type in GRAPHS_USED:
        for edge_weight in EDGE_WEIGHTS:
            graph = get_graph_topo(graph_type)
            graph = process_graph(graph, edge_weight)
            out_graph_dict[graph_type, edge_weight] = graph
    return out_graph_dict


def get_data() -> None:
    """Gather all required data at once."""
    print("Gathering data...")
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    TSV_FOLDER.mkdir(parents=True, exist_ok=True)
    for graph_type in GRAPHS_USED:
        for solution_method in SOLUTION_METHODS_USED:
            for est_seed in EST_SEEDS:
                cmd_to_call = ["python", "experiment.py",
                               graph_type, DEFAULT_TARGET_SIZE,
                               solution_method, est_seed,
                               None, None]
                for edge_weight in EDGE_WEIGHTS:
                    if isinstance(edge_weight, HetEdgeWeightType):
                        cmd_to_call[6] = "-het"
                    else:
                        cmd_to_call[6] = "-p"
                    cmd_to_call[7] = edge_weight
                    call(map(str, cmd_to_call))
    print("Finished gathering data")


def make_summary_csv(input_graph_dict: GraphDict) -> None:
    """Create a CSV summarising all data."""
    with open(PAPER_DATA_FOLDER / SUMMARY_CSV,
              mode="w", newline="") as summary_file:
        csv_writer = writer(summary_file)
        csv_writer.writerow(
            ["Graph Type", "Edge Weight", "Solution Method",
             "Estimation Seed",
             "f_corr", "f_ic", "Compute time (s)",
             "Min Deg(S)", "Avg Deg(S)",
             "Max Deg (S)", "Diameter(S)"
             ] +
            [f"S_{x}" for x in K_RANGE]
        )
        for idx, filename in enumerate(OUTPUT_FOLDER.glob("*.csv")):
            # Simple output for end user to read
            if not (idx + 1) % 50:
                print(f"Processed {idx + 1} files")

            # 4th character onwards because no need for 'out'
            fnsplit = str(filename)[4::].split(",")
            fnsplit[3] = fnsplit[3][:-4:]  # remove .csv from string
            fnsplit[0] = GraphType[fnsplit[0]]

            # Try converting to constant p
            try:
                fnsplit[1] = float(fnsplit[1])
            except ValueError:
                pass
            # or recognise as a heterogeneous edge weight
            try:
                fnsplit[1] = HetEdgeWeightType[fnsplit[1]]
            except KeyError:
                pass
            graph = input_graph_dict[tuple(fnsplit[0:2])]
            with open(filename) as data_file:
                data_reader = reader(data_file)
                seeds = []
                for counter, line in enumerate(data_reader):
                    seeds.append(int(line[0]))
                    if counter + 1 == DEFAULT_TARGET_SIZE:
                        soln_stats = [float(x) for x in line[1:4]]
                        deg = graph.degree(seeds)

            # Write in data in the order of
            # expertiment config, k=40 stats, degree stats
            # diameter, then seeds
            csv_writer.writerow(fnsplit +
                                soln_stats +
                                [min(deg),
                                 mean(deg),
                                 max(deg),
                                 get_solution_diameter(graph,
                                                       seeds)
                                 ] +
                                seeds
                                )

    print("Finished writing summary.csv")


def plot_compute_times() -> None:
    """
    Plot computational time against various k, up to 40.

    Requires summary.csv
    """
    # read in data
    all_data = {}
    for graph_type in GRAPHS_USED:
        for sol_method in SOLUTION_METHODS_USED:
            for edge_weight in (0.01, 0.95):
                data = []
                for data_filename in OUTPUT_FOLDER.glob(f"{graph_type},"
                                                        f"{edge_weight},"
                                                        f"{sol_method}"
                                                        "*"):
                    file_data = []
                    with open(data_filename) as data_file:
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
            for sol_method in SOLUTION_METHODS_USED:
                data = all_data[graph_type, sol_method, edge_weight]
                method_name = SOL_NAMES[sol_method]
                mean_data = list(map(mean, zip(*data)))
                min_data = list(map(min, zip(*data)))
                max_data = list(map(max, zip(*data)))
                label = f"{graph_type.name}, {method_name}"
                if sol_method == SolveMethod.graph_techniques:
                    plot(K_RANGE, mean_data,
                         label=label, linestyle=next(linestyles_used))
                else:
                    errorbar(
                        K_RANGE, mean_data,
                        yerr=([x - y for x, y in zip(mean_data, min_data)],
                              [y - x for x, y in zip(mean_data, max_data)]),
                        label=label, linestyle=next(linestyles_used)
                        )
        xlabel("$k$")
        ylabel("Computational time in seconds")
        title(("Computational times for greedy algorithms "
               f"when p = {edge_weight}"))
        legend()
        figure_filename = PAPER_DATA_FOLDER / f"compute_vs_k_{figlabel}.png"
        savefig(figure_filename, dpi=300)
        close()
        trim(figure_filename)

    print("Finished plotting computational times against k.")


def plot_hists(input_graph_dict: GraphDict) -> None:
    """
    Plot the histograms of the degrees of the seed sets.

    Requires summary.csv
    """
    seeds = {}
    with open(PAPER_DATA_FOLDER / SUMMARY_CSV) as summary_file:
        summary_reader = reader(summary_file)
        for line in summary_reader:
            if line[3] != "0":  # Estimation seed 0
                continue
            if (line[0] == GraphType.polblogs.name and
                    line[1] == HetEdgeWeightType.uniform.name):
                if line[2] == SolveMethod.graph_techniques.name:
                    seeds[SOL_SHORT_NAMES[line[2]]] = [
                        int(x) for x in line[11::]]
                elif line[2] == SolveMethod.independence_cascade.name:
                    seeds[SOL_SHORT_NAMES[line[2]]] = [
                        int(x) for x in line[11::]]
                # else pass

    for sol_type, figlab in zip(SOL_TUPLE, "ab"):
        figure()
        graph = input_graph_dict[GraphType.polblogs, HetEdgeWeightType.uniform]
        deg = sorted(list(graph.degree(seeds[sol_type])),
                     reverse=True)
        hist(deg, bins="auto")
        gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        title(f"Histogram of $S_{{{sol_type}}}^{{g}}$ for "
              "polblogs dataset, $k=40$")
        xlabel("Degree of nodes")
        ylabel("Frequency")
        figure_filename = PAPER_DATA_FOLDER / f"deg_hist_{figlab}.png"
        savefig(figure_filename, dpi=300)
        close()
        trim(figure_filename)

    print("Finished plotting histograms of degrees.")


def plot_expected_influence_graphs() -> None:
    """
    Plot expected influence against marginal probabilities.

    Requires summary.csv.

    Plot expected influence against marginal probabilities for wikivote and
    polblogs for both objectives for seed sets obtained in both approximation
    methods.
    """
    # read data
    markers = (x for x in cycle(("o", "^", "s", "p")))
    all_data: Dict[Tuple[str, ...], List[Tuple[float, ...]]] = {}
    with open(PAPER_DATA_FOLDER / SUMMARY_CSV) as summary_file:
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
        for sol_method in SOLUTION_METHODS_USED:
            sol_str = SOL_NAMES[sol_method]
            for obj_idx, obj_str in zip((0, 1), SOL_TUPLE):
                e_ix = []
                for edge_weight in P_LIST:
                    expt_config = (graph_type.name,
                                   str(edge_weight),
                                   sol_method.name
                                   )
                    e_ix.append(mean(y[obj_idx] for y in
                                     all_data[expt_config])
                                )
                plot(P_LIST, e_ix,
                     label=f"$f_{{{obj_str}}}"
                           f"\\left(S_{{{sol_str}}}^{{g}}\\right)$",
                     linewidth=3,
                     marker=next(markers),
                     markersize=8
                     )
        xlim(0, 1)
        xlabel("Marginal Probabilities $p$")
        ylabel("Expected Influence")
        legend()
        fig_filename = PAPER_DATA_FOLDER / f"inf_vs_p_{figlabel}.png"
        savefig(fig_filename, dpi=300)
        close()
        trim(fig_filename)

    print("Finished plotting expected influence against p.")


def draw_viz(input_graph_dict: GraphDict) -> None:
    """
    Create graph visualization.

    Requires some specific data files.
    """
    data = OUTPUT_FOLDER.glob("polblogs,weighted_cascade,*,0.csv")
    seeds: Dict[str, Set[int]] = {x: set() for x in SOL_TUPLE}
    for data_filename, sol_type in zip(data, SOL_TUPLE):
        with open(data_filename) as data_file:
            for line in data_file:
                seeds[sol_type].add(int(line.split(",")[0]))
    graph = input_graph_dict[GraphType.polblogs,
                             HetEdgeWeightType.weighted_cascade]

    # Converting to NetworkX to draw graphs
    nxg = DiGraph()
    nxg.add_nodes_from([v.index for v in graph.vs()])
    nxg.add_edges_from([e.tuple for e in graph.es()])
    largest_scc = max(strongly_connected_components(nxg), key=len)
    largest_scc_graph = nxg.subgraph(largest_scc)
    # alternative could be #6161FF
    for sol_type, figlabel in zip(SOL_TUPLE, "ab"):
        node_color = []
        edge_col = []
        for scc_node in largest_scc_graph:
            if scc_node in seeds[sol_type]:
                node_color.append("#EB0000")  # red, slightly darker
                edge_col.append("#FFFF00")  # yellow
            else:
                node_color.append("white")
                edge_col.append("black")
        figure(figsize=(32, 32))
        draw(largest_scc_graph,
             pos=spring_layout(nxg, seed=0),
             node_size=80,
             arrowsize=4, width=0.04,
             node_color=node_color,
             edgecolors=edge_col,
             linewidths=0.8,
             arrows=False
             )
        img_name = PAPER_DATA_FOLDER / f"viz_{figlabel}_full.png"
        savefig(img_name, dpi=400)
        close()
        trim(img_name)
        # crop dimensions
        # hardcoded for polblogs largest scc
        image = Image.open(img_name)
        crop_dim = (3600, 3000, 9000, 7700)
        cropped_example = image.crop(crop_dim)
        cropped_example.save(PAPER_DATA_FOLDER / f"viz_{figlabel}_crop.png")

    print("Finished drawing graph visualizations.")


def get_table1_data(input_graph_dict: GraphDict) -> None:
    """Create a .csv file that contains basic dataset summary statistics."""
    with open(PAPER_DATA_FOLDER / "table1.csv",
              mode="w", newline="") as dataset_csv:
        csv_writer = writer(dataset_csv)
        csv_writer.writerow(["Dataset", "|V|", "|E|", "Min Deg",
                             "Average Deg", "Max Deg"])
        for graph_type in list(GRAPHS_USED)[::-1]:
            # any edge weight will do
            graph = input_graph_dict[graph_type, 0.01]
            graph_degree = graph.degree()
            csv_writer.writerow([graph_type.name,
                                 graph.vcount(),
                                 graph.ecount(),
                                 min(graph_degree),
                                 mean(graph_degree),
                                 max(graph_degree)]
                                )

    print("Finished writing Table 1 data")


def _read_in_table2_data() -> Tuple[Dict[Tuple[str, ...],
                                         List[Tuple[float, ...]]],
                                    Dict[Tuple[str, ...],
                                         Tuple[float, ...]]]:
    out_data_dict: Dict[Tuple[str, ...], List[Tuple[float, ...]]] = {}
    out_table_data = {}
    with open(PAPER_DATA_FOLDER / SUMMARY_CSV) as summary_file:
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


def get_table2_data() -> None:
    """Summarises some data regarding the heterogeneous edge weight graphs."""
    # read data
    data_dict, table_data = _read_in_table2_data()

    # Start writing
    with open(PAPER_DATA_FOLDER / "table2.csv", "w", newline="") as table2csv:
        table2writer = writer(table2csv)
        for graph_type in list(GRAPHS_USED)[::-1]:
            for sol_method in SOLUTION_METHODS_USED:
                for edge_weight in HetEdgeWeightType:
                    if sol_method == SolveMethod.graph_techniques:
                        other_method = SolveMethod.independence_cascade
                        obj_idx = 1  # with reference to above obj_data
                    else:
                        other_method = SolveMethod.graph_techniques
                        obj_idx = 0
                    expt_config = (graph_type.name,
                                   edge_weight.name,
                                   sol_method.name
                                   )
                    cmpr_config = (graph_type.name,
                                   edge_weight.name,
                                   other_method.name
                                   )
                    mispec = (mean(y[obj_idx] for y in
                                   data_dict[expt_config]) /
                              mean(y[obj_idx] for y in
                                   data_dict[cmpr_config])
                              )
                    # Following the table division as made
                    # in the paper
                    conf_as_written = list((expt_config[0],
                                            expt_config[2],
                                            expt_config[1])
                                           )
                    table2writer.writerow(conf_as_written +
                                          [round(mispec, 3)] +
                                          list(table_data[expt_config])
                                          )

    print("Finished writing Table 2 data")


def make_table2_tex() -> None:
    """Create table2 tex from CSV data."""
    # read data
    all_dat = []
    with open(PAPER_DATA_FOLDER / "table2.csv") as table2_csv:
        csv_reader = reader(table2_csv)
        for line in csv_reader:
            # Preconverting line numbers
            line[3] = round(float(line[3]), 3)
            line[4] = int(float(line[4]))
            line[5] = float(line[5])
            line[6] = int(float(line[6]))
            line[7] = int(float(line[7]))
            all_dat.append(line)

    graph = cycle(("wikivote", "polblogs"))
    ic_corr = cycle(SOL_TUPLE)
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
        8: 27
    }
    headers = ["Dataset", "Seed Set", "$\\mathbf{p}$", "Mis-spec Ratio",
               "Min Deg($S$)", "Average Deg($S$)", "Max Deg($S$)"]

    # Headers and beginning the table, tabular environment
    with open(PAPER_DATA_FOLDER / "table2.tex", "w") as table2_tex:
        table2_tex.write("\\begin{table}[h!]\n")
        cur_indent = 2
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
        for line_idx, line in enumerate(all_dat):
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
                text = ("\\multirow{3}{*}"
                        f"{{$\\mathcal{{S}}^{{g}}_{{{next(ic_corr)}}}$}}")
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
            "\\end{table}"
        )
        for closing_item in closing_environment_items:
            table2_tex.write(closing_item)
    print("Finished writing .tex for table 2")


def _get_variation_against_mean(input_list: List[float]) -> float:
    """Return variation against mean."""
    return (max(input_list) - min(input_list))/mean(input_list)


def print_other_stats() -> None:
    """Print some other stats used in the paper."""
    with open(PAPER_DATA_FOLDER / SUMMARY_CSV) as summary_file:
        summary_reader = reader(summary_file)
        next(summary_reader)  # skip headers
        all_data: Dict[Tuple[str, ...], List[float]] = {}
        for line in summary_reader:
            expt_config = tuple(line[0:3])
            if expt_config in all_data:
                all_data[expt_config].append(float(line[5]))
            else:
                all_data[expt_config] = [float(line[5])]
        max_var = max(map(_get_variation_against_mean, all_data.values()))
        print("Maximum variation of independent cascade values against mean "
              f"is {max_var*100:0.3f}%")


if __name__ == "__main__":
    PAPER_DATA_FOLDER.mkdir(exist_ok=True)
    get_data()
    graph_dict = get_all_graphs()
    make_summary_csv(graph_dict)
    plot_compute_times()
    plot_hists(graph_dict)
    plot_expected_influence_graphs()
    draw_viz(graph_dict)
    get_table1_data(graph_dict)
    get_table2_data()
    make_table2_tex()
    print_other_stats()
