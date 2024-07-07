#include <string>
#include <iostream>
#include <limits>
#include <deque>
#include <queue>

#include "graph.hpp"

Graph read_tsv(const std::string &inp_tsv_filename)
{
    std::ifstream fin(inp_tsv_filename);
    std::string line;
    Edge e;
    uint32_t highest_seen = 0;
    uint32_t s, t;
    double q;

    // get number of nodes inside graph
    while (std::getline(fin, line))
    {
        std::stringstream ss(line);
        std::string temp_str;
        std::string temp_str2;
        std::getline(ss, temp_str, ' ');
        std::getline(ss, temp_str2, ' ');
        s = std::stol(temp_str);
        t = std::stol(temp_str2);
        if (highest_seen < s)
        {
            highest_seen = s;
        }
        if (highest_seen < t)
        {
            highest_seen = t;
        }
    }
    Graph g = {
        .successors = std::vector<std::vector<uint32_t>>(highest_seen + 1),
        .successor_weights = std::vector<std::vector<double>>(highest_seen + 1)};

    // Read again
    fin.clear();
    fin.seekg(0, std::ios::beg);
    while (std::getline(fin, line))
    {
        std::stringstream ss(line);
        std::string temp_str;
        std::string temp_str2;
        std::getline(ss, temp_str, ' ');
        std::getline(ss, temp_str2, ' ');
        s = std::stol(temp_str);
        t = std::stol(temp_str2);
        g.successors[s].push_back(t); // Add edge from s to t
        std::getline(ss, temp_str, ' ');
        q = (1.0 - std::stod(temp_str));
        g.successor_weights[s].push_back(q);
    }
    return g;
}

std::vector<double> dijkstra(const Graph &inp_graph, const uint32_t source)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    const uint32_t num_nodes = inp_graph.successors.size();
    std::vector<double> dist(num_nodes);
    std::vector<bool> visited(num_nodes, false);
    std::priority_queue<std::pair<double, uint32_t>,
                        std::deque<std::pair<double, uint32_t>>,
                        std::greater<std::pair<double, uint32_t>>>
        to_visit;

    for (auto &iter_dist : dist)
    {
        iter_dist = inf;
    }
    dist[source] = 0;
    to_visit.push({dist[source], source});
    uint32_t node_i;    // index for node which is being visited, and will get
                        // successors from
    uint32_t node_j;    // index for node which is an out-neighbour
    uint32_t j_idx;     // index for successor nodes within successor vector
    double edge_weight; // edge weight from i to j
    while (!to_visit.empty())
    {
        node_i = to_visit.top().second;
        visited[node_i] = true;
        to_visit.pop();
        for (j_idx = 0; j_idx < inp_graph.successors[node_i].size(); ++j_idx)
        {
            node_j = inp_graph.successors[node_i][j_idx];
            edge_weight = inp_graph.successor_weights[node_i][j_idx];

            // update as necessary
            if (!visited[node_j] &&
                (dist[node_j] > dist[node_i] + edge_weight))
            {
                dist[node_j] = dist[node_i] + edge_weight;
                to_visit.push({dist[node_j], node_j});
            }
        }
    }
    return dist;
}

std::vector<std::vector<double>> distmat(const Graph &inp_graph)
{
    const uint32_t num_nodes = inp_graph.successors.size();
    std::vector<std::vector<double>> out_vec(num_nodes);
    for (uint32_t i = 0; i < num_nodes; ++i)
    {
        out_vec[i] = dijkstra(inp_graph, i);
    }
    return out_vec;
}
