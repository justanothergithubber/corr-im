#pragma once
#include <vector>
#include "graph.hpp"

std::vector<double> get_phi(const Graph& phi_graph,
                            const std::vector<int>& phi_seeds,
                            const std::vector<std::vector<double>>& phi_dist);


double cvar(const Graph& cvar_graph,
            const std::vector<std::vector<double>>& cvar_dist,
            const std::vector<int>& cvar_seed_set, const double cvar_alpha);

double marg_cvar(const Graph &marg_graph,
                 const std::vector<std::vector<double>> &marg_dist,
                 const std::vector<int> &prev_cvar_set,
                 const double marg_alpha,
                 const int new_seed,
                 const double prev_cvar);
