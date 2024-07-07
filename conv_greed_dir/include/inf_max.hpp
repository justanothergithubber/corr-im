#include <vector>
#include <tuple>
#include <chrono>
#include <cstdlib>
#include <ctime>

#include "graph.hpp"

class PrunedEstimater
{
private:
  int n;
  unsigned long long n1; // vector.size() type
  std::vector<int> weight, comp, sigmas;
  std::vector<int> pmoc;
  std::vector<int> at_p;

  std::vector<int> up;
  std::vector<bool> memo, removed;

  std::vector<int> es, rs;
  std::vector<int> at_e, at_r;

  std::vector<bool> visited;

  int hub;
  std::vector<bool> descendant, ancestor;
  bool flag;

  void first();
  int sigma(const int v);
  inline int unique_child(const int v);

public:
  void init(int n, std::vector<std::pair<int, int>> &es,
            std::vector<int> &comp);
  int sigma1(const int v);
  void add(int v);
  void update(std::vector<long long> &sums);
};

class InfluenceMaximizer
{
private:
  int n;                // |V|
  unsigned long long m; // |E|

  std::vector<int> es1, at_e;
  std::vector<int> rs1, at_r;

  int scc(std::vector<int> &comp);

public:
  std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>
  pmc_greed(std::vector<std::pair<std::pair<int, int>, double>> &es, const int k, const int R,
            const int seed_delta,
            const std::chrono::high_resolution_clock::time_point start_time);
  std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>
  cvar_greed(const Graph &cvar_graph, const int k, const double alpha,
             const std::chrono::high_resolution_clock::time_point start_time);
  std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>
  conv_greed(std::vector<std::pair<std::pair<int, int>, double>> &es,
             const Graph &cvar_graph, const int k, const int R,
             const int seed_delta, const double alpha, const double lambda,
             const std::chrono::high_resolution_clock::time_point start_time);
  std::tuple<std::vector<double>, std::vector<double>>
  pmc_est(std::vector<std::pair<std::pair<int, int>, double>> &es,
          const std::vector<int> &seeds,
          const int R, const int seed_delta,
          const std::chrono::high_resolution_clock::time_point start_time);
  std::tuple<std::vector<double>, std::vector<double>>
  cvar_est(const std::vector<int> &seeds,
           const Graph &cvar_graph, const double alpha,
           const std::chrono::high_resolution_clock::time_point start_time);
};

// Random Number Generator
class Xorshift
{
private:
  unsigned int x, y, z, w;

public:
  Xorshift(int seed)
  {
    // For random results
    // srand(time(NULL));
    x = _(seed, 0);
    y = _(x, 1);
    z = _(y, 2);
    w = _(z, 3);
  }

  int _(int s, int i) { return 1812433253 * (s ^ (s >> 30)) + i + 1; }

  inline int gen_int()
  {
    unsigned int t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return w = w ^ (w >> 19) ^ t ^ (t >> 8);
  }

  inline int gen_int(int n) { return static_cast<int>(n * gen_double()); }

  inline double gen_double()
  {
    unsigned int a = (static_cast<unsigned int>(gen_int())) >> 5,
                 b = (static_cast<unsigned int>(gen_int())) >> 6;
    return (a * 67108864.0 + b) * (1.0 / (1LL << 53));
  }
};
