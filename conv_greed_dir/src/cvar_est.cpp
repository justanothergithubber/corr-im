#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "cvar.hpp"
#include "inf_max.hpp"
#include "graph.hpp"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "./cvar_est tsv_file seed_set_file alpha"
                  << std::endl;
        exit(1);
    }

    // Start the clock
    const std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();

    // Read in arguments
    const std::string tsv_file = argv[1];
    const std::string seed_set_file = argv[2];
    const double alpha = std::stod(argv[3]);

    // Read in seeds
    std::ifstream sf(seed_set_file.c_str());
    std::vector<int> seeds;
    int seed;
    while (sf >> seed)
    {
        seeds.push_back(seed);
    }
    sf.close();

    // Read in graph data
    const Graph cvar_graph = read_tsv(tsv_file);

    // Estimation function
    InfluenceMaximizer im;
    std::tuple<std::vector<double>, std::vector<double>> inf_max_res =
        im.cvar_est(seeds, cvar_graph, alpha, start_time);

    // Start printing results
    long long k = seeds.size();
    for (int i = 0; i < k; i++)
    {
        // Seed
        std::cout << seeds[i] << "\t"
                  // Marginal gain
                  << std::get<0>(inf_max_res)[i] << "\t"
                  //   Computational Time
                  << std::get<1>(inf_max_res)[i] << std::endl;
    }

    return 0;
}
