#ifndef _GSPAN_H_
#define _GSPAN_H_

#include "graph.h"

// RightMostPath is a vector which stores indices to PatternEdges in a DFSCode
using RightMostPath = std::vector<int>;

// an edge / node in DFS search tree
struct Embedding {
    Embedding *prev;
    int edge_id;

    Embedding() {}
    Embedding(Embedding* p, int i) : prev(p), edge_id(i) {}
};

struct EmbeddingRel {
    int prev_idx;
    int edge_id;

    EmbeddingRel() {}
    EmbeddingRel(int p, int i) : prev_idx(p), edge_id(i) {}
};

using Embeddings = std::vector<Embedding>;
using EmbeddingRels = std::vector<EmbeddingRel>;
using LayeredEmbeddings = std::vector<const EmbeddingRels *>;

// EmbVector is a vector of embedding edges
struct EmbVector : std::vector<Edge> {
    bool has_edge(const Edge& e) const;
    bool has_vertex(int v) const;
    void build(const CSRGraph& g, const Embedding* p);
    void build(const CSRGraph& g, const LayeredEmbeddings& le, int emb_idx);

    EmbVector(const CSRGraph& g, const Embedding& emb) { build(g, &emb); }
    EmbVector(const CSRGraph& g, const LayeredEmbeddings& le, int emb_idx) { build(g, le, emb_idx); }
};

void fsm(const CSRGraph& g, int k, int min_support);

#endif
