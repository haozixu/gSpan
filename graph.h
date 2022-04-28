#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <map>
#include <vector>

struct Edge {
    int from, to, label;

    // Edge() {}
    // Edge(int u, int v, int w) : from(u), to(v), label(w) {}
};

struct Vertex {
    int label;
    std::vector<Edge> edges;
};

// undirected graph
struct Graph : std::map<int, Vertex> {
    void add_vertex(int u, int label);
    void add_edge(int u, int v, int label);

    static Graph from_metis(const char* filename);
    static Graph from_txt(const char* filename);
};

struct PatternEdge {
    // int u, v, ulabel, elabel, vlabel;
    int from, to, from_label, edge_label, to_label;

    PatternEdge() {}
    // PatternEdge(int from, int to, int from_label, int edge_label, int to_label) :
    //     u(from), v(to), ulabel(from_label), elabel(edge_label), vlabel(to_label) {}
    PatternEdge(int u, int v, int ulabel, int elabel, int vlabel) :
        from(u), to(v), from_label(ulabel), edge_label(elabel), to_label(vlabel) {}

    bool is_forward() const { return from < to; }
    bool is_backward() const { return from > to; }

    bool operator== (const PatternEdge& rhs) const
    { 
        return (from == rhs.from) && (to == rhs.to) && 
            (from_label == rhs.from_label) && (edge_label == rhs.edge_label) && (to_label == rhs.to_label);
    }
    bool operator!= (const PatternEdge& rhs) const { return !(*this == rhs); }
};

using DFSCode = std::vector<PatternEdge>;

struct CSRGraph {
    int nr_nodes, nr_edges;
    int *offsets, *vlabels;
    Edge *edges;

    CSRGraph() : nr_nodes(0), nr_edges(0), 
        offsets(nullptr), vlabels(nullptr), edges(nullptr) {}
    CSRGraph(Graph&);
    CSRGraph(const DFSCode&);
    ~CSRGraph();

    void load(const char* filename); // load from a binary file
    void save(const char* filename); // save into a binary file
    void dump() const;
};

void prune_graph(CSRGraph& pruned, const CSRGraph& g, int min_sup);

#endif
