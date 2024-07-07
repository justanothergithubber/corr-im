#pragma once
#include <cstdint>
#include <vector>
#include <utility>  // pair
#include <fstream>
#include <string>
#include <sstream>

struct Edge{
    std::pair<uint32_t, uint32_t> e;
    double weight;
};

struct Graph {
    std::vector<std::vector<uint32_t>> successors;
    std::vector<std::vector<double>> successor_weights;
};

Graph read_tsv(const std::string& inp_tsv_filename);

std::vector<double> dijkstra(const Graph& inp_graph, const uint32_t source);

std::vector<std::vector<double>> distmat(const Graph& inp_graph);
