#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "inf_max.hpp"
#include "graph.hpp"

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cerr << "./conv_greed tsv_file target_seed_size "
                     "number_of_random_graphs seed_delta "
                     "alpha lambda"
                  << std::endl;
        exit(1);
    }

    // Start the clock
    const std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();

    // Read in arguments
    const std::string tsv_file = argv[1];
    const int k = std::stoi(argv[2]);
    const int R = std::stoi(argv[3]);
    const int seed_delta = std::stoi(argv[4]);
    const double alpha = std::stod(argv[5]);
    const double lambda = std::stod(argv[6]);

    // Read in graph data
    std::ifstream is(tsv_file.c_str());
    // weighted edge sequence
    std::vector<std::pair<std::pair<int, int>, double>> es;
    int u, v;
    double p;
    for (; is >> u >> v >> p;)
    {
        if (u == v)
        {
            continue;
        }
        es.push_back(std::make_pair(std::make_pair(u, v), p));
    }
    is.close();

    const Graph cvar_graph = read_tsv(tsv_file);

    InfluenceMaximizer im;
    std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>
        inf_max_res =
            im.conv_greed(es, cvar_graph, k, R, seed_delta,
                          alpha, lambda, start_time);
    for (int i = 0; i < k; i++)
    {
        // Seed
        std::cout << std::get<0>(inf_max_res)[i] << "\t"
                  // Marginal gain
                  << std::get<1>(inf_max_res)[i] << "\t"
                  // Computational Time
                  << std::get<2>(inf_max_res)[i] << std::endl;
    }

    return 0;
}
