#include <cstdio>
#include <cstdlib>
#include <omp.h>

#include "common.h"
#include "graph.h"
#include "gspan.h"

#include <chrono>

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("usage: %s input_graph min_support k\n", argv[0]);
        return 0;
    }

    using namespace std::chrono;
    auto t1 = steady_clock::now();

    CSRGraph g;
    g.load(argv[1]);
    int min_sup = atoi(argv[2]);
    int k = atoi(argv[3]);

    auto t2 = steady_clock::now();

    // CSRGraph pruned;
    // prune_graph(pruned, g, min_sup);

    auto t3 = steady_clock::now();

    fsm(g, k, min_sup);

    auto t4 = steady_clock::now();

    auto load_time = duration_cast<microseconds>(t2 - t1).count();
    auto prepare_time = duration_cast<microseconds>(t3 - t2).count();
    auto fsm_time = duration_cast<microseconds>(t4 - t3).count();
    printf("load: %lfs prepare: %lfs fsm: %lfs\n", load_time * 1e-6, prepare_time * 1e-6, fsm_time * 1e-6);

    return 0;
}
