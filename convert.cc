#include "graph.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("usage: %s input_name output_name\n", argv[0]);
        return 0;
    }

    Graph g = Graph::from_txt(argv[1]);
    CSRGraph csr(g);
    csr.save(argv[2]);
}
