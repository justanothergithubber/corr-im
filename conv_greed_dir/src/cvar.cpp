#include <algorithm> // max
#include <iostream>
#include <vector>
#include <limits>
#include "graph.hpp"
#include "cvar.hpp"

std::vector<double> get_phi(const Graph &phi_graph,
                            const std::vector<int> &phi_seeds,
                            const std::vector<std::vector<double>> &phi_dist)
{
    const int num_nodes = phi_graph.successors.size();
    std::vector<double> out_phi(num_nodes, std::numeric_limits<double>::infinity());
    int node_idx;
    for (node_idx = 0; node_idx < num_nodes; ++node_idx)
    {
        for (const auto &seed : phi_seeds)
        {
            if (phi_dist[seed][node_idx] < out_phi[node_idx])
            {
                out_phi[node_idx] = phi_dist[seed][node_idx];
            }
        }
    }
    return out_phi;
}

/* Computes CVaR
 */
double cvar(const Graph &cvar_graph,
            const std::vector<std::vector<double>> &cvar_dist,
            const std::vector<int> &cvar_seed_set,
            const double cvar_alpha)
{
    const int num_nodes = cvar_graph.successors.size();
    const std::vector<double> phi = get_phi(cvar_graph, cvar_seed_set, cvar_dist);
    double cvar = 0.0;
    int node_idx;
    for (node_idx = 0; node_idx < num_nodes; ++node_idx)
    {
        cvar += std::max(cvar_alpha - phi[node_idx], 0.0);
    }
    return (cvar / cvar_alpha);
}

/* Computes the marginal CVaR value
 *
 * Uses the previous CVaR value to compute the next CVaR value.
 */
double marg_cvar(const Graph &marg_graph,
                 const std::vector<std::vector<double>> &marg_dist,
                 const std::vector<int> &prev_cvar_set,
                 const double marg_alpha,
                 const int new_seed,
                 const double prev_cvar)
{
    std::vector<int> new_seed_set = prev_cvar_set;
    new_seed_set.push_back(new_seed);
    return cvar(marg_graph, marg_dist, new_seed_set, marg_alpha) - prev_cvar;
}
