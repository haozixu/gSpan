#include "graph.h"

#include <cstdio>
#include <string>

#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include <stdexcept>

using std::string;
using std::vector;

static void split(const string& str, vector<string>& tokens, const string& delimiters = " ")
{
    auto last_pos = str.find_last_not_of(delimiters, 0);
    auto pos = str.find_first_of(delimiters, last_pos);

    while (pos != string::npos || last_pos != string::npos) {
        tokens.push_back(str.substr(last_pos, pos - last_pos));
        last_pos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, last_pos);
    }
}

void Graph::add_vertex(int u, int label)
{
    (*this)[u].label = label;
}

void Graph::add_edge(int u, int v, int label)
{
    (*this)[u].edges.push_back(Edge { .from = u, .to = v, .label = label });
    (*this)[v].edges.push_back(Edge { .from = v, .to = u, .label = label });
}

Graph Graph::from_txt(const char *filename)
{
    Graph g;
    char op[16], dummy[4];
    int u, v, w, nr_edges = 0;

    FILE *f = fopen(filename, "r");
    if (!f)
        throw std::runtime_error("failed to open file\n");

    while (std::fscanf(f, "%s", op) == 1) {
        if (op[0] == 't') {
            std::fscanf(f, "%s%d", dummy, &u);
        } else if (op[0] == 'v') {
            std::fscanf(f, "%d%d", &v, &w);
            g.add_vertex(v, w);
        } else if (op[0] == 'e') {
            std::fscanf(f, "%d%d%d", &u, &v, &w);
            g.add_edge(u, v, w);
            ++nr_edges;
        }
    }
    fclose(f);

    std::printf("|V| = %ld |E| = %d\n", g.size(), nr_edges);
    return g;
}

CSRGraph::CSRGraph(Graph& g)
{
    std::unordered_map<int, int> old2new, new2old;

    nr_nodes = g.size();
    vlabels = new int[nr_nodes];
    offsets = new int[nr_nodes + 1];

    int new_v = 0;
    nr_edges = 0;
    for (const auto &pair : g) {
        int old_v = pair.first;
        auto &vertex = pair.second;
        nr_edges += vertex.edges.size();

        old2new[old_v] = new_v;
        new2old[new_v] = old_v;
        ++new_v;
    }

    int offset = 0;
    offsets[0] = 0;
    edges = new Edge[nr_edges];
    for (int v = 0; v < nr_nodes; ++v) {
        int old = new2old[v];
        auto &vertex = g[old];

        std::vector<Edge> neighbors;
        for (const auto &e : vertex.edges)
            neighbors.push_back(Edge { .from = v, .to = old2new[e.to], .label = e.label });
        std::sort(neighbors.begin(), neighbors.end(), [](const Edge& e1, const Edge& e2) -> bool {
            if (e1.to != e2.to)
                return e1.to < e2.to;
            return e1.label < e2.label;            
        });

        int new_offset = offset + vertex.edges.size();
        for (auto it = neighbors.begin(); offset < new_offset; ++offset, ++it) {
            edges[offset].from = v;
            edges[offset].to = it->to;
            edges[offset].label = it->label;
        }

        vlabels[v] = vertex.label;
        offsets[v + 1] = offset;
    }
}

CSRGraph::~CSRGraph()
{
    if (offsets) delete[] offsets;
    if (vlabels) delete[] vlabels;
    if (edges) delete[] edges;
}

// DFSCode -> CSRGraph
CSRGraph::CSRGraph(const DFSCode& dfs_code)
{
    int *degrees = new int[dfs_code.size() + 1]{0};

    int v_max = 0;
    for (auto &e : dfs_code) {
        if (v_max < e.from) v_max = e.from;
        if (v_max < e.to) v_max = e.to;
        ++degrees[e.from];
        ++degrees[e.to];
    }

    nr_nodes = v_max + 1;
    nr_edges = 2 * dfs_code.size();
    offsets = new int[nr_nodes + 1];
    vlabels = new int[nr_nodes];
    edges = new Edge[nr_edges];

    offsets[0] = 0;
    for (int v = 0; v <= v_max; ++v)
        offsets[v + 1] = offsets[v] + degrees[v];
    
    for (auto &e : dfs_code) {
        int u = e.from, v = e.to;
        int from_label = e.from_label;
        int to_label = e.to_label;
        if (from_label != -1) vlabels[u] = from_label;
        if (to_label != -1) vlabels[v] = to_label;

        int i = offsets[u] + (--degrees[u]);
        edges[i] = Edge { .from = u, .to = v, .label = e.edge_label };

        int j = offsets[v] + (--degrees[v]);
        edges[j] = Edge { .from = v, .to = u, .label = e.edge_label };
    }

    delete[] degrees;
}

void CSRGraph::dump() const
{
    printf("vlabels:");
    for (int v = 0; v < nr_nodes; ++v)
        printf(" %d", vlabels[v]);
    printf("\noffsets:");
    for (int v = 0; v <= nr_nodes; ++v)
        printf(" %d", offsets[v]);
    printf("\nfrom: ");
    for (int i = 0; i < nr_edges; ++i)
        printf(" %d", edges[i].from);
    printf("\nto:");
    for (int i = 0; i < nr_edges; ++i)
        printf(" %d", edges[i].to);
    printf("\n");
}

void CSRGraph::load(const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        throw std::runtime_error("failed to open file");
    if (fread(&nr_nodes, sizeof(nr_nodes), 1, f) != 1)
        throw std::runtime_error("nr_nodes required");
    if (fread(&nr_edges, sizeof(nr_edges), 1, f) != 1)
        throw std::runtime_error("nr_edges required");
    
    offsets = new int[nr_nodes + 1];
    vlabels = new int[nr_nodes];
    edges = new Edge[nr_edges];
    
    if (fread(offsets, sizeof(int), nr_nodes + 1, f) != nr_nodes + 1)
        throw std::runtime_error("bad offsets");
    if (fread(vlabels, sizeof(int), nr_nodes, f) != nr_nodes)
        throw std::runtime_error("bad vertex labels");
    if (fread(edges, sizeof(Edge), nr_edges, f) != nr_edges)
        throw std::runtime_error("bad edges");

    fclose(f);
}

void CSRGraph::save(const char* filename)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        throw std::runtime_error("failed to open file");
    
    fwrite(&nr_nodes, sizeof(nr_nodes), 1, f);
    fwrite(&nr_edges, sizeof(nr_edges), 1, f);
    fwrite(offsets, sizeof(int), nr_nodes + 1, f);
    fwrite(vlabels, sizeof(int), nr_nodes, f);
    fwrite(edges, sizeof(Edge), nr_edges, f);
    fclose(f);
}

void prune_graph(CSRGraph &pruned, const CSRGraph &g, int min_sup)
{
    std::unordered_map<int, int> vertex_label_counts;
    std::unordered_map<int, int> vertex_mapping;
    std::vector<Edge> edges;

    for (int v = 0; v < g.nr_nodes; ++v)
        vertex_label_counts[g.vlabels[v]]++;

    // for (auto &pair : vertex_label_counts)
    //     printf("label: %d count: %d\n", pair.first, pair.second);
    
    for (int v = 0; v < g.nr_nodes; ++v) {
        if (vertex_label_counts[g.vlabels[v]] >= min_sup) {
            int new_v = vertex_mapping.size();
            vertex_mapping[v] = new_v;
        }
    }
    pruned.nr_nodes = vertex_mapping.size();
    pruned.vlabels = new int[pruned.nr_nodes];
    pruned.offsets = new int[pruned.nr_nodes + 1];
    pruned.offsets[0] = 0;

    int degree, new_v;
    for (int old_v = 0; old_v < g.nr_nodes; ++old_v) {
        if (!vertex_mapping.count(old_v))
            continue;
        
        new_v = vertex_mapping[old_v];
        degree = 0;
        pruned.vlabels[new_v] = g.vlabels[old_v];
        for (int i = g.offsets[old_v]; i < g.offsets[old_v + 1]; ++i) {
            auto &e = g.edges[i];
            if (!vertex_mapping.count(e.to))
                continue;
            
            int new_to = vertex_mapping[e.to];
            edges.push_back(Edge { .from = new_v, .to = new_to, .label = e.label });
            ++degree;
        }
        int new_offset = pruned.offsets[new_v] + degree;
        pruned.offsets[new_v + 1] = new_offset;
    }

    pruned.nr_edges = edges.size();
    pruned.edges = new Edge[pruned.nr_edges];
    std::copy(edges.begin(), edges.end(), pruned.edges);

    printf("|V|: %d -> %d |E|: %d -> %d\n",
        g.nr_nodes, pruned.nr_nodes, g.nr_edges, pruned.nr_edges);
}
