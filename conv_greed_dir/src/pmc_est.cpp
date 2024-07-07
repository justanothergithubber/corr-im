#include <iostream>
#include <fstream>
#include <vector>

#include "graph.hpp"
#include "inf_max.hpp"

int main(int argc, char **argv)
{
	if (argc < 5)
	{
		std::cerr << "./pmc_est tsv_file seed_set_file R seed_delta" << std::endl;
		exit(1);
	}

	// Start the clock
	const std::chrono::high_resolution_clock::time_point start_time =
		std::chrono::high_resolution_clock::now();

	// Read in arguments
	const std::string tsv_file = argv[1];
	const std::string seed_file = argv[2];
	const int R = atoi(argv[3]);
	const int seed_delta = atoi(argv[4]);

	// Read in graph data
	std::ifstream is(tsv_file.c_str());
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

	// Read in seeds
	std::ifstream sf(seed_file.c_str());
	std::vector<int> seeds;
	int seed;
	while (sf >> seed)
	{
		seeds.push_back(seed);
	}
	sf.close();

	// Estimation higher function
	InfluenceMaximizer im;
	std::tuple<std::vector<double>, std::vector<double>> inf_max_res =
		im.pmc_est(es,
				   seeds,
				   R,
				   seed_delta,
				   start_time);

	// Start printing results
	unsigned long long k = seeds.size();
	for (unsigned int i = 0; i < k; i++)
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
