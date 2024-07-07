#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <tuple>
#include <chrono>

#include "cvar.hpp"
#include "graph.hpp"
#include "inf_max.hpp"

inline int PrunedEstimater::unique_child(const int v)
{
    int outdeg = 0, child = -1;
    for (int i = at_e[v]; i < at_e[v + 1]; i++)
    {
        const int u = es[i];
        if (!removed[u])
        {
            outdeg++;
            child = u;
        }
    }
    if (outdeg == 0)
    {
        return -1;
    }
    else if (outdeg == 1)
    {
        return child;
    }
    else
    {
        return -2;
    }
}

void PrunedEstimater::init(const int _n,
                           std::vector<std::pair<int, int>> &_es, std::vector<int> &_comp)
{
    flag = true;
    n = _n;
    n1 = _comp.size();

    visited.resize(n, false);

    unsigned long long m = _es.size();
    std::vector<int> outdeg(n), indeg(n);

    for (unsigned int i = 0; i < m; i++)
    {
        int a = _es[i].first, b = _es[i].second;
        outdeg[a]++;
        indeg[b]++;
    }
    es.resize(m, -1);
    rs.resize(m, -1);

    at_e.resize(n + 1, 0);
    at_r.resize(n + 1, 0);

    at_e[0] = at_r[0] = 0;
    for (int i = 1; i <= n; i++)
    {
        at_e[i] = at_e[i - 1] + outdeg[i - 1];
        at_r[i] = at_r[i - 1] + indeg[i - 1];
    }

    for (unsigned int i = 0; i < m; i++)
    {
        int a = _es[i].first, b = _es[i].second;
        es[at_e[a]++] = b;
        rs[at_r[b]++] = a;
    }

    at_e[0] = at_r[0] = 0;
    for (int i = 1; i <= n; i++)
    {
        at_e[i] = at_e[i - 1] + outdeg[i - 1];
        at_r[i] = at_r[i - 1] + indeg[i - 1];
    }

    sigmas.resize(n);
    comp = _comp;
    std::vector<std::pair<int, int>> ps;
    for (unsigned int i = 0; i < n1; i++)
    {
        ps.push_back(std::make_pair(comp[i], i));
    }
    sort(ps.begin(), ps.end());
    at_p.resize(n + 1);
    for (unsigned int i = 0; i < n1; i++)
    {
        pmoc.push_back(ps[i].second);
        at_p[ps[i].first + 1]++;
    }
    for (int i = 1; i <= n; i++)
    {
        at_p[i] += at_p[i - 1];
    }

    memo.resize(n);
    removed.resize(n);

    weight.resize(n1, 0);
    for (unsigned int i = 0; i < n1; i++)
    {
        weight[comp[i]]++;
    }

    first();
}

int PrunedEstimater::sigma1(const int v)
{
    return sigma(comp[v]);
}

int PrunedEstimater::sigma(const int v0)
{
    if (memo[v0])
    {
        return sigmas[v0];
    }
    memo[v0] = true;
    if (removed[v0])
    {
        return sigmas[v0] = 0;
    }
    else
    {
        int child = unique_child(v0);
        if (child == -1)
        {
            return sigmas[v0] = weight[v0];
        }
        else if (child >= 0)
        {
            return sigmas[v0] = sigma(child) + weight[v0];
        }
        else
        {
            int delta = 0;
            std::vector<int> vec;
            visited[v0] = true;
            vec.push_back(v0);
            std::queue<int> Q;
            Q.push(v0);
            bool prune = ancestor[v0];

            if (prune)
            {
                delta += sigma(hub);
            }

            for (; !Q.empty();)
            {
                const int v = Q.front();
                Q.pop();
                if (removed[v])
                {
                    continue;
                }
                if (prune && descendant[v])
                {
                    continue;
                }
                delta += weight[v];
                for (int i = at_e[v]; i < at_e[v + 1]; i++)
                {
                    const int u = es[i];
                    if (removed[u])
                    {
                        continue;
                    }
                    if (!visited[u])
                    {
                        visited[u] = true;
                        vec.push_back(u);
                        Q.push(u);
                    }
                }
            }
            const size_t vec_size = vec.size();
            for (unsigned int i = 0; i < vec_size; i++)
            {
                visited[vec[i]] = false;
            }
            return sigmas[v0] = delta;
        }
    }
}

void PrunedEstimater::first()
{
    hub = 0;
    for (int i = 0; i < n; i++)
    {
        if ((at_e[i + 1] - at_e[i]) + (at_r[i + 1] - at_r[i]) > (at_e[hub + 1] - at_e[hub]) + (at_r[hub + 1] - at_r[hub]))
        {
            hub = i;
        }
    }

    descendant.resize(n);
    std::queue<int> Q;
    Q.push(hub);
    for (; !Q.empty();)
    {
        // forall v, !remove[v]
        const int v = Q.front();
        Q.pop();
        descendant[v] = true;
        for (int i = at_e[v]; i < at_e[v + 1]; i++)
        {
            const int u = es[i];
            if (!descendant[u])
            {
                descendant[u] = true;
                Q.push(u);
            }
        }
    }

    ancestor.resize(n);
    Q.push(hub);
    for (; !Q.empty();)
    {
        const int v = Q.front();
        Q.pop();
        ancestor[v] = true;
        for (int i = at_r[v]; i < at_r[v + 1]; i++)
        {
            const int u = rs[i];
            if (!ancestor[u])
            {
                ancestor[u] = true;
                Q.push(u);
            }
        }
    }
    ancestor[hub] = false;

    for (int i = 0; i < n; i++)
    {
        sigma(i);
    }
    ancestor.assign(n, false);
    descendant.assign(n, false);

    for (unsigned int i = 0; i < n1; i++)
    {
        up.push_back(i);
    }
}

void PrunedEstimater::update(std::vector<long long> &sums)
{
    unsigned long long us = up.size();
    for (unsigned int i = 0; i < us; i++)
    {
        int v = up[i];
        if (!flag)
        {
            sums[v] -= sigmas[comp[v]];
        }
    }
    for (unsigned int i = 0; i < us; i++)
    {
        int v = up[i];
        sums[v] += sigma1(v);
    }
    flag = false;
}

void PrunedEstimater::add(int v0)
{
    v0 = comp[v0];
    std::queue<int> Q;
    Q.push(v0);
    removed[v0] = true;
    std::vector<int> rm;
    for (; !Q.empty();)
    {
        const int v = Q.front();
        Q.pop();
        rm.push_back(v);
        for (int i = at_e[v]; i < at_e[v + 1]; i++)
        {
            const int u = es[i];
            if (!removed[u])
            {
                Q.push(u);
                removed[u] = true;
            }
        }
    }

    up.clear();

    std::vector<int> vec;
    for (int i = 0; i < static_cast<int>(rm.size()); i++)
    {
        const int v = rm[i];
        memo[v] = false; // for update()
        for (int j = at_p[v]; j < at_p[v + 1]; j++)
        {
            up.push_back(pmoc[j]);
        }
        for (int j = at_r[v]; j < at_r[v + 1]; j++)
        {
            const int u = rs[j];
            if (!removed[u] && !visited[u])
            {
                visited[u] = true;
                vec.push_back(u);
                Q.push(u);
            }
        }
    }
    // reachable to removed node
    for (; !Q.empty();)
    {
        const int v = Q.front();
        Q.pop();
        memo[v] = false;
        for (int j = at_p[v]; j < at_p[v + 1]; j++)
        {
            up.push_back(pmoc[j]);
        }
        for (int i = at_r[v]; i < at_r[v + 1]; i++)
        {
            const int u = rs[i];
            if (!visited[u])
            {
                visited[u] = true;
                vec.push_back(u);
                Q.push(u);
            }
        }
    }
    const size_t vec_size = vec.size();
    for (unsigned int i = 0; i < vec_size; i++)
    {
        visited[vec[i]] = false;
    }
}

int InfluenceMaximizer::scc(std::vector<int> &comp)
{
    std::vector<bool> vis(n);
    std::stack<std::pair<int, int>> S;
    std::vector<int> lis;
    int k = 0;
    for (int i = 0; i < n; i++)
    {
        S.push(std::make_pair(i, 0));
    }

    for (; !S.empty();)
    {
        int v = S.top().first, state = S.top().second;
        S.pop();
        if (state == 0)
        {
            if (vis[v])
            {
                continue;
            }
            vis[v] = true;
            S.push(std::make_pair(v, 1));
            for (int i = at_e[v]; i < at_e[v + 1]; i++)
            {
                int u = es1[i];
                S.push(std::make_pair(u, 0));
            }
        }
        else
        {
            lis.push_back(v);
        }
    }

    for (int i = 0; i < n; i++)
    {
        S.push(std::make_pair(lis[i], -1));
    }

    vis.assign(n, false);
    for (; !S.empty();)
    {
        int v = S.top().first, arg = S.top().second;
        S.pop();
        if (vis[v])
        {
            continue;
        }
        vis[v] = true;
        comp[v] = arg == -1 ? k++ : arg;
        for (int i = at_r[v]; i < at_r[v + 1]; i++)
        {
            int u = rs1[i];
            S.push(std::make_pair(u, comp[v]));
        }
    }
    return k;
}

std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>
InfluenceMaximizer::conv_greed(
    std::vector<std::pair<std::pair<int, int>, double>> &es,
    const Graph &cvar_graph, const int k, const int R, const int seed_delta,
    const double alpha, const double lambda,
    const std::chrono::high_resolution_clock::time_point start_time)
{
    // Pruned Monte Carlo Portion
    n = 0;
    m = es.size();
    for (unsigned int i = 0; i < m; i++)
    {
        n = std::max(n, std::max(es[i].first.first, es[i].first.second) + 1);
    }
    sort(es.begin(), es.end());

    es1.resize(m);
    rs1.resize(m);
    at_e.resize(n + 1);
    at_r.resize(n + 1);

    std::vector<PrunedEstimater> infs(R);

    for (int t = 0; t < R; t++)
    {
        Xorshift xs = Xorshift(t + seed_delta);

        int mp = 0;
        at_e.assign(n + 1, 0);
        at_r.assign(n + 1, 0);
        std::vector<std::pair<int, int>> ps;
        for (unsigned int i = 0; i < m; i++)
        {
            if (xs.gen_double() < es[i].second)
            {
                es1[mp++] = es[i].first.second;
                at_e[es[i].first.first + 1]++;
                ps.push_back(std::make_pair(es[i].first.second,
                                            es[i].first.first));
            }
        }
        at_e[0] = 0;
        sort(ps.begin(), ps.end());
        for (int i = 0; i < mp; i++)
        {
            rs1[i] = ps[i].second;
            at_r[ps[i].first + 1]++;
        }
        for (int i = 1; i <= n; i++)
        {
            at_e[i] += at_e[i - 1];
            at_r[i] += at_r[i - 1];
        }

        std::vector<int> comp(n);
        int nscc = scc(comp);

        std::vector<std::pair<int, int>> es2;
        for (int u = 0; u < n; u++)
        {
            int a = comp[u];
            for (int i = at_e[u]; i < at_e[u + 1]; i++)
            {
                int b = comp[es1[i]];
                if (a != b)
                {
                    es2.push_back(std::make_pair(a, b));
                }
            }
        }

        sort(es2.begin(), es2.end());
        es2.erase(unique(es2.begin(), es2.end()), es2.end());

        infs[t].init(nscc, es2, comp);
    }

    // initialise gain vector of size n
    std::vector<long long> gain(n);

    // shared
    std::vector<double> marg;
    std::vector<double> compute_times;
    std::vector<int> seeds;

    // CVaR Portion
    const std::vector<std::vector<double>> dist = distmat(cvar_graph);
    double prev_cvar = 0.0;
    double cvar_val = 0.0;

    for (int t = 0; t < k; t++)
    {
        for (int j = 0; j < R; j++)
        {
            infs[j].update(gain);
        }
        int next = 0;
        prev_cvar = cvar_val;
        for (int i = 0; i < n; i++)
        {
            // If the 'i' has higher marginal gain
            if ((lambda * (static_cast<double>(gain[i]) / R)) +
                    ((1 - lambda) *
                     marg_cvar(cvar_graph, dist, seeds, alpha, i, prev_cvar)) >
                (lambda * (static_cast<double>(gain[next]) / R)) +
                    ((1 - lambda) * marg_cvar(cvar_graph, dist, seeds, alpha,
                                              next, prev_cvar)))
            // Then set next = i
            {
                next = i;
            }
        }
        for (int j = 0; j < R; j++)
        {
            infs[j].add(next);
        }
        seeds.push_back(next);
        cvar_val = cvar(cvar_graph, dist, seeds, alpha);
        marg.push_back((lambda * (static_cast<double>(gain[next]) / R)) +
                       ((1 - lambda) * (cvar_val - prev_cvar)));
        double compute_time_len =
            std::chrono::duration_cast<
                std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() -
                start_time)
                .count();
        compute_times.push_back(compute_time_len);
    }
    return std::make_tuple(seeds, marg, compute_times);
}

std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>
InfluenceMaximizer::cvar_greed(
    const Graph &cvar_graph, const int k,
    const double alpha,
    const std::chrono::high_resolution_clock::time_point start_time)
{
    n = cvar_graph.successors.size();
    // shared
    std::vector<double> marg;
    std::vector<double> compute_times;
    std::vector<int> seeds;

    // CVaR Portion
    const std::vector<std::vector<double>> dist = distmat(cvar_graph);
    double prev_cvar = 0.0;
    double cvar_val = 0.0;

    for (int t = 0; t < k; t++)
    {
        int next = 0;
        prev_cvar = cvar_val;
        for (int i = 0; i < n; i++)
        {
            // If the 'i' has higher marginal gain
            if (
                (marg_cvar(cvar_graph, dist, seeds, alpha, i, prev_cvar)) >
                (marg_cvar(cvar_graph, dist, seeds, alpha,
                           next, prev_cvar)))
            // Then set next = i
            {
                next = i;
            }
        }
        seeds.push_back(next);
        cvar_val = cvar(cvar_graph, dist, seeds, alpha);
        marg.push_back((cvar_val - prev_cvar));
        double compute_time_len =
            std::chrono::duration_cast<
                std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() -
                start_time)
                .count();
        compute_times.push_back(compute_time_len);
    }
    return std::make_tuple(seeds, marg, compute_times);
}

std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>
InfluenceMaximizer::pmc_greed(
    std::vector<std::pair<std::pair<int, int>, double>> &es, const int k,
    const int R, const int seed_delta,
    const std::chrono::high_resolution_clock::time_point start_time)
{
    // Pruned Monte Carlo Only
    n = 0;
    m = es.size();
    for (unsigned int i = 0; i < m; i++)
    {
        n = std::max(n, std::max(es[i].first.first, es[i].first.second) + 1);
    }
    sort(es.begin(), es.end());

    es1.resize(m);
    rs1.resize(m);
    at_e.resize(n + 1);
    at_r.resize(n + 1);

    std::vector<PrunedEstimater> infs(R);

    for (int t = 0; t < R; t++)
    {
        Xorshift xs = Xorshift(t + seed_delta);

        int mp = 0;
        at_e.assign(n + 1, 0);
        at_r.assign(n + 1, 0);
        std::vector<std::pair<int, int>> ps;
        for (unsigned int i = 0; i < m; i++)
        {
            if (xs.gen_double() < es[i].second)
            {
                es1[mp++] = es[i].first.second;
                at_e[es[i].first.first + 1]++;
                ps.push_back(std::make_pair(es[i].first.second,
                                            es[i].first.first));
            }
        }
        at_e[0] = 0;
        sort(ps.begin(), ps.end());
        for (int i = 0; i < mp; i++)
        {
            rs1[i] = ps[i].second;
            at_r[ps[i].first + 1]++;
        }
        for (int i = 1; i <= n; i++)
        {
            at_e[i] += at_e[i - 1];
            at_r[i] += at_r[i - 1];
        }

        std::vector<int> comp(n);
        int nscc = scc(comp);

        std::vector<std::pair<int, int>> es2;
        for (int u = 0; u < n; u++)
        {
            int a = comp[u];
            for (int i = at_e[u]; i < at_e[u + 1]; i++)
            {
                int b = comp[es1[i]];
                if (a != b)
                {
                    es2.push_back(std::make_pair(a, b));
                }
            }
        }

        sort(es2.begin(), es2.end());
        es2.erase(unique(es2.begin(), es2.end()), es2.end());

        infs[t].init(nscc, es2, comp);
    }

    // initialise gain vector of size n
    std::vector<long long> gain(n);

    // shared
    std::vector<double> marg;
    std::vector<double> compute_times;
    std::vector<int> seeds;

    for (int t = 0; t < k; t++)
    {
        for (int j = 0; j < R; j++)
        {
            infs[j].update(gain);
        }
        int next = 0;
        for (int i = 0; i < n; i++)
        {
            // If the 'i' has higher marginal gain
            if (gain[i] > gain[next])
            {
                // Then set next = i
                next = i;
            }
        }
        for (int j = 0; j < R; j++)
        {
            infs[j].add(next);
        }
        seeds.push_back(next);
        marg.push_back(static_cast<double>(gain[next]) / R);
        double compute_time_len =
            std::chrono::duration_cast<
                std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() -
                start_time)
                .count();
        compute_times.push_back(compute_time_len);
    }
    return std::make_tuple(seeds, marg, compute_times);
}

std::tuple<std::vector<double>, std::vector<double>>
InfluenceMaximizer::pmc_est(std::vector<std::pair<std::pair<int, int>, double>> &es,
                            const std::vector<int> &seeds, const int R,
                            const int seed_delta,
                            const std::chrono::high_resolution_clock::time_point start_time)
{
    // Pruned Monte Carlo
    const long long k = seeds.size(); // number of seeds
    n = 0;
    m = es.size();
    for (unsigned int i = 0; i < m; i++)
    {
        n = std::max(n, std::max(es[i].first.first, es[i].first.second) + 1);
    }

    sort(es.begin(), es.end());

    es1.resize(m);
    rs1.resize(m);
    at_e.resize(n + 1);
    at_r.resize(n + 1);

    std::vector<PrunedEstimater> infs(R);

    for (int t = 0; t < R; t++)
    {
        Xorshift xs = Xorshift(t + seed_delta);

        int mp = 0;
        at_e.assign(n + 1, 0);
        at_r.assign(n + 1, 0);
        std::vector<std::pair<int, int>> ps;
        for (unsigned int i = 0; i < m; i++)
        {
            if (xs.gen_double() < es[i].second)
            {
                es1[mp++] = es[i].first.second;
                at_e[es[i].first.first + 1]++;
                ps.push_back(std::make_pair(es[i].first.second,
                                            es[i].first.first));
            }
        }
        at_e[0] = 0;
        sort(ps.begin(), ps.end());
        for (int i = 0; i < mp; i++)
        {
            rs1[i] = ps[i].second;
            at_r[ps[i].first + 1]++;
        }
        for (int i = 1; i <= n; i++)
        {
            at_e[i] += at_e[i - 1];
            at_r[i] += at_r[i - 1];
        }

        std::vector<int> comp(n);
        int nscc = scc(comp);

        std::vector<std::pair<int, int>> es2;
        for (int u = 0; u < n; u++)
        {
            int a = comp[u];
            for (int i = at_e[u]; i < at_e[u + 1]; i++)
            {
                int b = comp[es1[i]];
                if (a != b)
                {
                    es2.push_back(std::make_pair(a, b));
                }
            }
        }

        sort(es2.begin(), es2.end());
        es2.erase(unique(es2.begin(), es2.end()), es2.end());

        infs[t].init(nscc, es2, comp);
    }

    std::vector<long long> ic_gain(n);
    std::vector<double> compute_times;
    std::vector<double> marg;

    for (unsigned int t = 0; t < k; t++)
    {
        for (int j = 0; j < R; j++)
        {
            infs[j].update(ic_gain);
        }
        unsigned int next = seeds[t];
        for (int j = 0; j < R; j++)
        {
            infs[j].add(next);
        }
        marg.push_back((static_cast<double>(ic_gain[next]) / R));
        double compute_time_len =
            std::chrono::duration_cast<
                std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() -
                start_time)
                .count();
        compute_times.push_back(compute_time_len);
    }
    return std::make_tuple(marg, compute_times);
}

std::tuple<std::vector<double>, std::vector<double>>
InfluenceMaximizer::cvar_est(const std::vector<int> &seeds,
                             const Graph &cvar_graph, const double alpha,
                             const std::chrono::high_resolution_clock::time_point start_time)
{
    // CVaR Estimation

    const long long k = seeds.size();

    // Initialise outputs
    std::vector<double> compute_times;
    std::vector<double> marg;
    double temp_marg;

    // Initialise computational variables
    const std::vector<std::vector<double>> dist = distmat(cvar_graph);
    std::vector<int> cvar_seeds;
    double cvar_val = 0;
    double prev_cvar = 0;

    for (unsigned int t = 0; t < k; t++)
    {
        temp_marg = 0;
        unsigned int next = seeds[t];
        cvar_seeds.push_back(next);
        // cvar update
        cvar_val = cvar(cvar_graph, dist, cvar_seeds, alpha);
        temp_marg += (cvar_val - prev_cvar);
        marg.push_back(temp_marg);
        prev_cvar = cvar_val;
        double compute_time_len =
            std::chrono::duration_cast<
                std::chrono::duration<double>>(
                std::chrono::high_resolution_clock::now() -
                start_time)
                .count();
        compute_times.push_back(compute_time_len);
    }
    return std::make_tuple(marg, compute_times);
}
