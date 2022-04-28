#include "graph.h"
#include "gspan.h"
#include "bitmap.cuh"
#include "hashset.cuh"

#include <cstddef>
#include <cstring>

#include <map>
#include <vector>
#include <unordered_set>

#include <algorithm>

#include <omp.h>

#define PROFILE 0

#if PROFILE
#include <chrono>
#include <string>
#include <cstdint>

using namespace std::chrono;

std::map<std::string, uint64_t> timers;

struct TimerGuard {
    steady_clock::time_point begin;
    std::string name;

    TimerGuard(const std::string& key) : begin(steady_clock::now()), name(key) {}
    ~TimerGuard()
    {
        auto end = steady_clock::now();
        timers[name] += duration_cast<microseconds>(end - begin).count();
    }
};
#endif

struct InitialEdgeCompare_t {
    // u = 0, v = 1 for initial pattern edge
    // compare order: from_label -> edge_label -> to_label
    bool operator() (const PatternEdge& e1, const PatternEdge& e2) const
    {
        if (e1.from_label != e2.from_label) return e1.from_label < e2.from_label;
        if (e1.edge_label != e2.edge_label) return e1.edge_label < e2.edge_label;
        return e1.to_label < e2.to_label;
    }
};

// note that rmpath stores indices to PatternEdges of DFSCode in reversed order
void build_right_most_path(const DFSCode& dfs_code, RightMostPath& rmpath)
{
    int last_v = -1;
    rmpath.clear();
    // find the shortest forward path from v_0 to v_n
    for (int i = dfs_code.size() - 1; i >= 0; --i) {
        auto &e = dfs_code[i];
        if (e.is_forward() && (rmpath.empty() || e.to == last_v)) {
            rmpath.push_back(i);
            last_v = e.from;
        }
    }
}

void print_dfs_code(const DFSCode& dfs_code)
{
    for (auto &d : dfs_code)
        printf("(%d %d %d %d %d)", d.from, d.to, d.from_label, d.edge_label, d.to_label);
}

void report(const DFSCode& dfs_code, int support)
{
    print_dfs_code(dfs_code);
    printf(": %d\n", support);
}

struct GPUContext {
    int *g_offsets, *g_vlabels;
    Edge *g_edges; // Graph on gpu
    uint32_t hashset_capacity, *dev_hashset_keys; // hash set implementation
    uint32_t bitmap_el_count, *dev_bitmaps; // bitmap implementation
    int *dev_set_sizes, *dev_op_array;
    EmbeddingRel **cur_embedding_ptrs, **cur_dev_embeddings, **dev_embedding_ptrs;
    // cur_embedding_ptrs: an array of (host) pointers to each level of embeddings (on cpu)
    // cur_dev_embeddings: an array of (device) pointers to each level of embeddings (on gpu)
    int *dev_rmpath, *max_nr_backward_embeddings, *new_backward_count, *max_nr_forward_embeddings, *new_forward_count;
    int *dev_backward_embeddings, *dev_forward_embeddings;
} ctx;

struct CPUContext {
    Bitmap *bitmaps;
    int bitmap_size;
} cctx;

void init_gpu_context(const CSRGraph& g, int k)
{
    // allocate space for the graph on gpu
    size_t edges_size = g.nr_edges * sizeof(Edge);
    cudaMalloc(&ctx.g_edges, edges_size);
    cudaMalloc(&ctx.g_vlabels, g.nr_nodes * sizeof(int));
    cudaMalloc(&ctx.g_offsets, (g.nr_nodes + 1) * sizeof(int));

    cudaMemcpyAsync(ctx.g_edges, g.edges, edges_size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(ctx.g_vlabels, g.vlabels, g.nr_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(ctx.g_offsets, g.offsets, (g.nr_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

    ctx.hashset_capacity = get_hashset_capacity(g.nr_nodes, 0.5);
    ctx.bitmap_el_count = get_bitmap_size(g.nr_nodes) / sizeof(uint32_t);

    // pre-allocate space for gpu hashset / bitmap, set sizes & op_array
    cudaMalloc(&ctx.dev_hashset_keys, (k + 1) * ctx.hashset_capacity * sizeof(uint32_t));
    cudaMalloc(&ctx.dev_bitmaps, (k + 1) * ctx.bitmap_el_count * sizeof(uint32_t));
    cudaMalloc(&ctx.dev_set_sizes, (k + 1) * sizeof(int));
    cudaMalloc(&ctx.dev_op_array, k * sizeof(int));

    ctx.cur_embedding_ptrs = reinterpret_cast<EmbeddingRel **>(new std::intptr_t[k]{0});
    ctx.cur_dev_embeddings = reinterpret_cast<EmbeddingRel **>(new std::intptr_t[k]{0});
    cudaMalloc(&ctx.dev_embedding_ptrs, k * sizeof(void *));

    // cudaMalloc(&ctx.dev_rmpath, k * sizeof(int));
    // cudaMallocManaged(&ctx.new_backward_count, sizeof(int));
    // cudaMallocManaged(&ctx.new_forward_count, sizeof(int));
    // cudaMallocManaged(&ctx.dev_backward_embeddings, 512 * 1024 * 1024UL);
    // cudaMallocManaged(&ctx.dev_forward_embeddings, 4 * 1024 * 1024 * 1024UL);
}

void init_cpu_context(const CSRGraph& g, int k)
{
    cctx.bitmap_size = get_bitmap_size(g.nr_nodes);
    cctx.bitmaps = new Bitmap[k + 1];
    for (int i = 0; i < k + 1; ++i) {
        cctx.bitmaps[i].bits = (char *) std::malloc(cctx.bitmap_size);
    }
}

__global__ void support_kernel(int nr_pattern_nodes, int nr_pattern_edges, const Edge* g_edges,
    EmbeddingRel** embedding_ptrs, int last_embeddings_count, int* op_array,
    uint32_t* hashset_keys, uint32_t hashset_capacity, int* set_sizes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < last_embeddings_count) {
        int i = tid;
        for (int layer = nr_pattern_edges - 1; layer >= 0; --layer) {
            EmbeddingRel *p = embedding_ptrs[layer] + i;
            int v = op_array[layer];
            if (v > 0) {
                uint32_t *base = hashset_keys + v * hashset_capacity;
                int u = g_edges[p->edge_id].to;
                hashset_insert(base, u, &set_sizes[v], hashset_capacity);
            }

            if (layer == 0) {
                int u = g_edges[p->edge_id].from;
                hashset_insert(hashset_keys, u, &set_sizes[0], hashset_capacity);
            }

            i = p->prev_idx;
        }
    }
}

__global__ void support_kernel_v2(int nr_pattern_nodes, int nr_pattern_edges, const Edge* g_edges,
    EmbeddingRel** embedding_ptrs, int last_embeddings_count, int* op_array,
    uint32_t* bitmaps, uint32_t bitmap_el_count, int* set_sizes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < last_embeddings_count) {
        int i = tid;
        for (int layer = nr_pattern_edges - 1; layer >= 0; --layer) {
            EmbeddingRel *p = embedding_ptrs[layer] + i;
            int v = op_array[layer];
            if (v > 0) {
                uint32_t *base = bitmaps + v * bitmap_el_count;
                int u = g_edges[p->edge_id].to;
                bitmap_insert(base, u, &set_sizes[v]);
            }

            if (layer == 0) {
                int u = g_edges[p->edge_id].from;
                bitmap_insert(bitmaps, u, &set_sizes[0]);
            }

            i = p->prev_idx;
        }
    }
}

__global__ void check(int n, int* set_sizes)
{
    printf("set sizes:");
    for (int i = 0; i < n; ++i)
        printf(" %d", set_sizes[i]);
    printf("\n");
}

int support_gpu(const CSRGraph& g, const DFSCode& dfs_code, const LayeredEmbeddings& layered_embeddings)
{
#if PROFILE
    TimerGuard guard("support_gpu");
#endif
    
    int v_max = 0;
    for (auto &e : dfs_code) {
        if (v_max < e.from) v_max = e.from;
        if (v_max < e.to) v_max = e.to;
    }
    int nr_pattern_nodes = v_max + 1;
    int nr_pattern_edges = dfs_code.size();

    // initialize hashsets keys
    // size_t hashset_size_total = nr_pattern_nodes * ctx.hashset_capacity * sizeof(uint32_t);
    // cudaMemsetAsync(ctx.dev_hashset_keys, KEY_EMPTY, hashset_size_total);
    // or clear bitmap
    size_t bitmap_size_total = nr_pattern_nodes * ctx.bitmap_el_count * sizeof(uint32_t);
    cudaMemsetAsync(ctx.dev_bitmaps, 0, bitmap_size_total);

    // initialize set sizes
    cudaMemset(ctx.dev_set_sizes, 0, nr_pattern_nodes * sizeof(int));

    for (int i = 0; i < layered_embeddings.size(); ++i) {
        auto host_ptr = &(layered_embeddings[i]->front()); // pointer to contiguous embeddings
        if (ctx.cur_embedding_ptrs[i] != host_ptr) { // update
            auto prev_host_ptr = ctx.cur_embedding_ptrs[i];
            ctx.cur_embedding_ptrs[i] = const_cast<EmbeddingRel *>(host_ptr);

            if (prev_host_ptr) {
                // release corresponding device memory
                cudaFree(ctx.cur_dev_embeddings[i]);
            }
            // initialize new device embeddings
            size_t size = layered_embeddings[i]->size() * sizeof(EmbeddingRel);
            cudaMalloc(&ctx.cur_dev_embeddings[i], size);
            cudaMemcpyAsync(ctx.cur_dev_embeddings[i], host_ptr, size, cudaMemcpyHostToDevice);
        }
    }
    cudaMemcpy(ctx.dev_embedding_ptrs, ctx.cur_dev_embeddings, nr_pattern_edges * sizeof(EmbeddingRel *), cudaMemcpyHostToDevice);

    // allocate memory for embeddings
    // EmbeddingRel* embedding_ptrs[nr_pattern_edges];
    // int embedding_counts[nr_pattern_edges];
    // for (int i = 0; i < layered_embeddings.size(); ++i) {
    //     auto p = layered_embeddings[i];
    //     embedding_counts[i] = p->size();
    //     size_t size = p->size() * sizeof(EmbeddingRel);
    //     cudaMalloc(&embedding_ptrs[i], size);
    //     cudaMemcpyAsync(embedding_ptrs[i], &(p->front()), size, cudaMemcpyHostToDevice);
    // }
    // cudaMemcpy(ctx.dev_embedding_ptrs, &embedding_ptrs, sizeof(embedding_ptrs), cudaMemcpyHostToDevice);

    // initialize op_array (simplified dfs_code) 
    int op_array[nr_pattern_edges];
    for (int i = 0; i < nr_pattern_edges; ++i) {
        op_array[i] = 0;
        if (dfs_code[i].is_forward())
            op_array[i] = dfs_code[i].to;
    }
    cudaMemcpy(ctx.dev_op_array, &op_array, sizeof(op_array), cudaMemcpyHostToDevice);

    int block_size = 256;
    int last_embeddings_count = layered_embeddings.back()->size();
    int nr_blocks = (last_embeddings_count + block_size - 1) / block_size;
    // support_kernel<<<nr_blocks, block_size>>>(
    //     nr_pattern_nodes, nr_pattern_edges, ctx.g_edges,
    //     ctx.dev_embedding_ptrs, last_embeddings_count, ctx.dev_op_array,
    //     ctx.dev_hashset_keys, ctx.hashset_capacity, ctx.dev_set_sizes);
    support_kernel_v2<<<nr_blocks, block_size>>>(
        nr_pattern_nodes, nr_pattern_edges, ctx.g_edges,
        ctx.dev_embedding_ptrs, last_embeddings_count, ctx.dev_op_array,
        ctx.dev_bitmaps, ctx.bitmap_el_count, ctx.dev_set_sizes);
    cudaDeviceSynchronize();

    int set_sizes[nr_pattern_nodes];
    cudaMemcpy(&set_sizes, ctx.dev_set_sizes, sizeof(set_sizes), cudaMemcpyDeviceToHost);

    // for (auto p : embedding_ptrs)
    //     cudaFree(p);

    int sup = 0x7fffffff;
    for (int i = 0; i < nr_pattern_nodes; ++i) {
        int s = set_sizes[i];
        if (s > 0 && sup > s)
            sup = s;
    }
    return sup;
}

int support(const CSRGraph& g, const DFSCode& dfs_code, const LayeredEmbeddings& layered_embeddings)
{
#if PROFILE
    TimerGuard guard("support");
#endif

    std::map<int, std::unordered_set<int>> node_ids;

    auto &last_layer_embeddings = *(layered_embeddings.back());
    for (int emb_idx = 0; emb_idx < last_layer_embeddings.size(); ++emb_idx) {
        int i = emb_idx;
        for (int layer = dfs_code.size() - 1; layer >= 0; --layer) {
            auto &emb = (*layered_embeddings[layer])[i];
            auto &e = dfs_code[layer];
            if (e.is_forward()) // forward pattern edge
                node_ids[e.to].insert(g.edges[emb.edge_id].to);
            if (emb.prev_idx != -1)
                node_ids[e.from].insert(g.edges[emb.edge_id].from);
            
            i = emb.prev_idx;
        }
    }

    int sup = 0x7fffffff;
    for (const auto &pair : node_ids)
        if (sup > pair.second.size())
            sup = pair.second.size();
    return sup;
}

int support_omp(const CSRGraph& g, const DFSCode& dfs_code, const LayeredEmbeddings& layered_embeddings)
{
#if PROFILE
    TimerGuard guard("support_omp");
#endif

    int v_max = 0;
    for (auto &e : dfs_code) {
        if (v_max < e.from) v_max = e.from;
        if (v_max < e.to) v_max = e.to;
    }
    int nr_pattern_nodes = v_max + 1;

    auto bitmaps = cctx.bitmaps;
    for (int v = 0; v < nr_pattern_nodes; ++v) {
        bitmaps[v].count = 0;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nr_threads = omp_get_num_threads();

            int local_size = (cctx.bitmap_size + nr_threads - 1) / nr_threads;
            int local_begin = tid * local_size;
            if (local_size > cctx.bitmap_size - local_begin)
                local_size = cctx.bitmap_size - local_begin;

            std::memset(&bitmaps[v].bits[local_begin], 0, local_size);
        }
    }

    int nr_embeddings = layered_embeddings.back()->size();
    #pragma omp parallel for schedule(static)
    for (int emb_idx = 0; emb_idx < nr_embeddings; ++emb_idx) {
        int i = emb_idx;
        for (int layer = dfs_code.size() - 1; layer >= 0; --layer) {
            auto &emb = (*layered_embeddings[layer])[i];
            auto &e = dfs_code[layer];
            if (e.is_forward()) // forward pattern edge
                bitmaps[e.to].insert(g.edges[emb.edge_id].to);
            if (emb.prev_idx != -1)
                bitmaps[e.from].insert(g.edges[emb.edge_id].from);
            
            i = emb.prev_idx;
        }
    }

    int sup = 0x7fffffff;
    for (int i = 0; i < nr_pattern_nodes; ++i) {
        int s = bitmaps[i].count;
        if (s > 0 && sup > s)
            sup = s;
    }
    return sup;
}

void EmbVector::build(const CSRGraph& g, const Embedding* p)
{
    for (; p; p = p->prev) {
        auto i = p->edge_id;
        push_back(g.edges[i]);
    }
    std::reverse(begin(), end());
}

void EmbVector::build(const CSRGraph& g, const LayeredEmbeddings& layered_embeddings, int emb_idx)
{
    for (int layer = layered_embeddings.size() - 1; layer >= 0; --layer) {
        auto &emb = (*layered_embeddings[layer])[emb_idx];
        auto i = emb.edge_id;
        push_back(g.edges[i]);
        emb_idx = emb.prev_idx;
    }
    std::reverse(begin(), end());
}

bool EmbVector::has_edge(const Edge &e) const
{
    return std::find_if(begin(), end(), [&e](const Edge& e1) -> bool {
        return (e1.label == e.label) && ((e1.from == e.from && e1.to == e.to) || (e1.to == e.from && e1.from == e.to));
    }) != end();
}

bool EmbVector::has_vertex(int v) const
{
    return std::find_if(begin(), end(), [v](const Edge& e) -> bool {
        return e.from == v || e.to == v;
    }) != end();
}

// mine all single-edge patterns (DFSCode) and corresponding embeddings
template <typename Fn>
void get_initial_embeddings(const CSRGraph& g, const Fn& fn)
{
    for (int from = 0; from < g.nr_nodes; ++from) { // foreach vertex u in G
        auto from_label = g.vlabels[from];
        for (auto i = g.offsets[from]; i < g.offsets[from + 1]; ++i) { // foreach adjacent edge
            auto to = g.edges[i].to;
            auto edge_label = g.edges[i].label;
            auto to_label = g.vlabels[to];

            if (from_label <= to_label) { // constraints to reduce redundancy
                fn(from_label, edge_label, to_label, i);
            }
        }
    }
}

bool get_backward_edge(const CSRGraph& g, const Edge& e, int u_n, const EmbVector& history, int* out_idx)
{
    for (auto i = g.offsets[u_n]; i < g.offsets[u_n + 1]; ++i) {
        Edge cur = { .from = u_n, .to = g.edges[i].to, .label = g.edges[i].label };

        if (history.has_edge(cur))
            continue;

        if (
            (cur.to == e.from) &&
            ((e.label < cur.label) || (e.label == cur.label && g.vlabels[e.to] <= g.vlabels[u_n]))
        ) {
            *out_idx = i;
            return true;
        }
    }
    return false;
}

template <typename Fn>
bool get_first_forward_edge(const CSRGraph& g, int u_n, int min_label, const EmbVector& history, const Fn& fn)
{
    bool has_edge = false;
    for (auto i = g.offsets[u_n]; i < g.offsets[u_n + 1]; ++i) {
        auto edge_label = g.edges[i].label;
        auto to = g.edges[i].to;
        auto to_label = g.vlabels[to];
        if (to_label < min_label || history.has_vertex(to))
            continue;

        fn(edge_label, to_label, i);
        has_edge = true;
    }
    return has_edge;
}

template <typename Fn>
bool get_other_forward_edge(const CSRGraph& g, const Edge& e, int min_label, const EmbVector& history, const Fn& fn)
{
    bool has_edge = false;
    auto to_label_bound = g.vlabels[e.to];
    auto u_i = e.from;
    for (auto i = g.offsets[u_i]; i < g.offsets[u_i + 1]; ++i) {
        auto edge_label = g.edges[i].label;
        auto to = g.edges[i].to;
        auto to_label = g.vlabels[to];

        if (to_label < min_label || history.has_vertex(to))
            continue;
        if (e.label < edge_label || ((e.label == edge_label) && (to_label_bound <= to_label))) {
            fn(edge_label, to_label, i);
            has_edge = true;
        }
    }
    return has_edge;
}

bool projection_is_canonical(const CSRGraph& g, Embeddings& embeddings, 
    const DFSCode& dfs_code, DFSCode& min_dfs_code)
{
    RightMostPath rmpath;
    build_right_most_path(min_dfs_code, rmpath);
    int min_label = min_dfs_code[0].from_label;
    int v_n = min_dfs_code[rmpath[0]].to; // largest pattern vertex id in current min_dfs_code
    int v_n_label = min_dfs_code[rmpath[0]].to_label;

    // check for backward edge
    // starting from v_0, check along the right most path to see if there's backward edge v_n -> v_i
    bool has_backward_edge = false;
    int edge_idx, v_i, v_i_label;
    std::map<int, Embeddings> mapping; // edge_label -> embeddings

    for (int i = rmpath.size() - 1; !has_backward_edge && i >= 1; --i) {
        int k = rmpath[i];
        for (auto &emb : embeddings) {
            EmbVector history(g, emb);
            auto u_n = history[rmpath[0]].to; // u_n is the graph vertex id, while v_n is the pattern vertex id
            if (get_backward_edge(g, history[k], u_n, history, &edge_idx)) {
                mapping[g.edges[edge_idx].label].emplace_back(&emb, edge_idx);
                v_i = min_dfs_code[k].from;
                v_i_label = min_dfs_code[k].to_label;
                has_backward_edge = true;
            }
        }
    }

    if (has_backward_edge) {
        auto it = mapping.begin();
        auto edge_label = it->first;

        // PatternEdge pe(v_n, v_i, v_n_label, edge_label, v_i_label);
        PatternEdge pe(v_n, v_i, -1, edge_label, -1);
        if (dfs_code[min_dfs_code.size()] != pe)
            return false;
        min_dfs_code.push_back(pe);
        return projection_is_canonical(g, it->second, dfs_code, min_dfs_code);
    }

    // check for forward edges
    bool has_forward_edge = false;
    std::map<int, std::map<int, Embeddings>> mapping2; // edge_label -> to_label -> embeddings

    // 1. first forward edge v_n -> v_{n+1}
    for (auto &emb : embeddings) {
        EmbVector history(g, emb);
        auto u_n = history[rmpath[0]].to;
        auto extend = [&emb, &mapping2](auto edge_label, auto to_label, auto edge_idx) {
            mapping2[edge_label][to_label].emplace_back(&emb, edge_idx);
        };
        if (get_first_forward_edge(g, u_n, min_label, history, extend)) {
            has_forward_edge = true;
        }
    }

    if (has_forward_edge) {
        v_i = v_n;
        v_i_label = v_n_label;
    }

    // 2. secondary forward edges
    // walk along the right most path from v_{n-1} to v_0
    for (int i = 0; !has_forward_edge && i < rmpath.size(); ++i) {
        int k = rmpath[i];
        for (auto &emb : embeddings) {
            EmbVector history(g, emb);
            auto &e = history[k];
            auto extend = [&emb, &mapping2](auto edge_label, auto to_label, auto edge_idx) {
                mapping2[edge_label][to_label].emplace_back(&emb, edge_idx);
            };
            if (get_other_forward_edge(g, e, min_label, history, extend)) {
                has_forward_edge = true;
            }
        }

        if (has_forward_edge) {
            v_i = min_dfs_code[k].from;
            v_i_label = min_dfs_code[k].from_label;
            break;
        }
    }

    if (has_forward_edge) {
        auto it1 = mapping2.begin();
        auto it2 = it1->second.begin();

        auto edge_label = it1->first;
        auto to_label = it2->first;
        // PatternEdge pe(v_i, v_n + 1, v_i_label, edge_label, to_label);
        PatternEdge pe(v_i, v_n + 1, -1, edge_label, to_label);
        if (dfs_code[min_dfs_code.size()] != pe)
            return false;
        min_dfs_code.push_back(pe);
        return projection_is_canonical(g, it2->second, dfs_code, min_dfs_code);
    }

    return true;
}

bool is_canonical(const DFSCode& dfs_code)
{
#if PROFILE
    TimerGuard guard("is_min");
#endif

    if (dfs_code.size() == 1)
        return true;
    
    CSRGraph g(dfs_code); // construct the graph from dfs_code

    std::map<PatternEdge, Embeddings, InitialEdgeCompare_t> mapping;
    get_initial_embeddings(g, [&mapping](auto from_label, auto edge_label, auto to_label, auto edge_idx) {
        PatternEdge pe(0, 1, from_label, edge_label, to_label);
        mapping[pe].emplace_back(nullptr, edge_idx);
    });

    auto it = mapping.begin();
    DFSCode min_dfs_code {it->first};
    return projection_is_canonical(g, it->second, dfs_code, min_dfs_code);
}

using ForwardMapping = std::map<int, std::map<int, std::map<int, EmbeddingRels>>>;
using BackwardMapping = std::map<int, std::map<int, EmbeddingRels>>;

void extend(const CSRGraph& g, const DFSCode& dfs_code, LayeredEmbeddings& layered_embeddings,
    const RightMostPath& rmpath, ForwardMapping& forward_mapping, BackwardMapping& backward_mapping)
{
#if PROFILE
    TimerGuard guard("extend");
#endif

    auto min_label = dfs_code[0].from_label;
    auto v_n = dfs_code[rmpath[0]].to;
    auto &embeddings = *(layered_embeddings.back());

    for (int emb_idx = 0; emb_idx < embeddings.size(); ++emb_idx) {
        EmbVector history(g, layered_embeddings, emb_idx);
        auto u_n = history[rmpath[0]].to;

        // extend backward edges
        for (int i = rmpath.size() - 1; i >= 1; --i) {
            int k = rmpath[i];
            int edge_idx;
            auto v_i = dfs_code[k].from;
            if (get_backward_edge(g, history[k], u_n, history, &edge_idx)) {
                auto edge_label = g.edges[edge_idx].label;
                backward_mapping[v_i][edge_label].emplace_back(emb_idx, edge_idx);
            }
        }

        // extend first forward edges
        auto extend_first = [v_n, emb_idx, &forward_mapping](auto edge_label, auto to_label, auto edge_idx) {
            forward_mapping[v_n][edge_label][to_label].emplace_back(emb_idx, edge_idx);
        };
        get_first_forward_edge(g, u_n, min_label, history, extend_first);

        // extend secondary forward edges
        for (int i = 0; i < rmpath.size(); ++i) {
            int k = rmpath[i];
            auto &e = history[k];
            auto v_i = dfs_code[k].from;
            auto extend_other = [v_i, emb_idx, &forward_mapping](auto edge_label, auto to_label, auto edge_idx) {
                forward_mapping[v_i][edge_label][to_label].emplace_back(emb_idx, edge_idx);
            };
            get_other_forward_edge(g, e, min_label, history, extend_other);
        }
    }
}

#define foreach_backward_begin(root) \
for (auto &pair1 : root) { \
    for (auto &pair2 : pair1.second) { \
        auto v = pair1.first; \
        auto edge_label = pair2.first;

#define foreach_backward_end() }}

#define foreach_forward_begin(root) \
for (auto &pair1 : root) { \
    for (auto &pair2 : pair1.second) { \
        for (auto &pair3 : pair2.second) { \
            auto v = pair1.first; \
            auto edge_label = pair2.first; \
            auto to_label = pair3.first;

#define foreach_forward_end() }}}

void extend_omp(const CSRGraph& g, const DFSCode& dfs_code, LayeredEmbeddings& layered_embeddings,
    const RightMostPath& rmpath, ForwardMapping& forward_mapping, BackwardMapping& backward_mapping)
{
#if PROFILE
    TimerGuard guard("extend_omp");
#endif

    using BackwardCounts = std::map<int, std::map<int, int>>;
    using ForwardCounts = std::map<int, std::map<int, std::map<int, int>>>;

    int nr_threads = omp_get_max_threads();
    auto local_backward_mappings = new BackwardMapping[nr_threads];
    auto local_forward_mappings = new ForwardMapping[nr_threads];
    auto local_backward_counts = new BackwardCounts[nr_threads];
    auto local_forward_counts = new ForwardCounts[nr_threads];

    auto min_label = dfs_code[0].from_label;
    auto v_n = dfs_code[rmpath[0]].to;
    auto &embeddings = *(layered_embeddings.back());

    #pragma omp parallel for schedule(guided)
    for (int emb_idx = 0; emb_idx < embeddings.size(); ++emb_idx) {
        EmbVector history(g, layered_embeddings, emb_idx);
        auto u_n = history[rmpath[0]].to;

        int tid = omp_get_thread_num();

        // extend backward edges
        for (int i = rmpath.size() - 1; i >= 1; --i) {
            int k = rmpath[i];
            int edge_idx;
            auto v_i = dfs_code[k].from;
            if (get_backward_edge(g, history[k], u_n, history, &edge_idx)) {
                auto edge_label = g.edges[edge_idx].label;
                local_backward_mappings[tid][v_i][edge_label].emplace_back(emb_idx, edge_idx);
                local_backward_counts[tid][v_i][edge_label]++;
            }
        }

        // extend first forward edges
        auto extend_first = 
        [tid, v_n, emb_idx, &local_forward_mappings, &local_forward_counts]
        (auto edge_label, auto to_label, auto edge_idx) {
            local_forward_mappings[tid][v_n][edge_label][to_label].emplace_back(emb_idx, edge_idx);
            local_forward_counts[tid][v_n][edge_label][to_label]++;
        };
        get_first_forward_edge(g, u_n, min_label, history, extend_first);

        // extend secondary forward edges
        for (int i = 0; i < rmpath.size(); ++i) {
            int k = rmpath[i];
            auto &e = history[k];
            auto v_i = dfs_code[k].from;
            auto extend_other = 
            [tid, v_i, emb_idx, &local_forward_mappings, &local_forward_counts]
            (auto edge_label, auto to_label, auto edge_idx) {
                local_forward_mappings[tid][v_i][edge_label][to_label].emplace_back(emb_idx, edge_idx);
                local_forward_counts[tid][v_i][edge_label][to_label]++;
            };
            get_other_forward_edge(g, e, min_label, history, extend_other);
        }
    }

    for (int i = 1; i < nr_threads; ++i) {
        // calculate prefix sum
        foreach_backward_begin(local_backward_counts[i - 1])
            local_backward_counts[i][v][edge_label] += pair2.second;
        foreach_backward_end()
        
        foreach_forward_begin(local_forward_counts[i - 1])
            local_forward_counts[i][v][edge_label][to_label] += pair3.second;
        foreach_forward_end()
    }

    // reserve space
    foreach_backward_begin(local_backward_counts[nr_threads - 1])
        backward_mapping[v][edge_label].resize(pair2.second);
    foreach_backward_end()

    foreach_forward_begin(local_forward_counts[nr_threads - 1])
        forward_mapping[v][edge_label][to_label].resize(pair3.second);
    foreach_forward_end()

    // copy
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        foreach_backward_begin(local_backward_mappings[i])
            auto dst = backward_mapping[v][edge_label].begin()
                + ((i > 0) ? local_backward_counts[i - 1][v][edge_label] : 0);
            auto &local_embeddings = pair2.second;
            std::copy(local_embeddings.begin(), local_embeddings.end(), dst);
        foreach_backward_end()
            
        foreach_forward_begin(local_forward_mappings[i])
            auto dst = forward_mapping[v][edge_label][to_label].begin()
                + ((i > 0) ? local_forward_counts[i - 1][v][edge_label][to_label] : 0);
            auto &local_embeddings = pair3.second;
            std::copy(local_embeddings.begin(), local_embeddings.end(), dst);
        foreach_forward_end()
    }

    delete[] local_backward_mappings;
    delete[] local_forward_mappings;
    delete[] local_backward_counts;
    delete[] local_forward_counts;
}

constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

// each thread creates the corresponding History object of an Embedding
__global__ void build_histories(Edge* histories, EmbeddingRel** embedding_ptrs, const Edge* g_edges,
    int nr_embeddings, int nr_pattern_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nr_embeddings) {
        auto *history = histories + (tid * nr_pattern_edges);
        int i = tid;
        for (int layer = nr_pattern_edges - 1; layer >= 0; --layer) {
            EmbeddingRel *emb = embedding_ptrs[layer] + i;
            history[layer] = g_edges[emb->edge_id];

            i = emb->prev_idx;
        }
    }
}

__device__ bool history_has_edge(const Edge* h, const Edge& e, int history_len)
{
    for (int i = 0; i < history_len; ++i) {
        if ((h[i].from == e.from && h[i].to == e.to && h[i].label == e.label) || 
            (h[i].to == e.from && h[i].from == e.to && h[i].label == e.label))
            return true;
    }
    return false;
}

__device__ bool history_has_vertex(const Edge* h, int v, int history_len)
{
    for (int i = 0; i < history_len; ++i)
        if (h[i].from == v || h[i].to == v)
            return true;
    return false;
}

template <typename T>
__device__ void warp_inclusive_scan(T sum[], int lane)
{
    #pragma unroll
    for (int s = 1; s < THREADS_PER_WARP; s *= 2) {
        int v = lane >= s ? sum[lane - s] : 0;
        sum[lane] += v;
        __threadfence_block();
    }
}

// k is often small, so we use a constant here
constexpr int MAX_K = 4;

__global__ void extend_kernel(
    int* backward_out, int* backward_idx, int* forward_out, int* forward_idx, 
    const Edge* histories, const Edge* g_edges, const int* g_vlabels, const int* g_offsets,
    int* rmpath, int rmpath_len, int nr_pattern_edges, int nr_embeddings, int min_label)
{
    extern __shared__ Edge block_sh_history[];
    __shared__ int block_out_base[WARPS_PER_BLOCK];
    __shared__ int block_out_offset[THREADS_PER_BLOCK];
    __shared__ int block_degree_sums[WARPS_PER_BLOCK * MAX_K];

    int gtid = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
    int gwid = gtid / THREADS_PER_WARP; // global warp id
    int emb_idx = gwid;
    if (emb_idx >= nr_embeddings)
        return;
    
    int tid = threadIdx.x;
    int wid = tid / THREADS_PER_WARP; // warp id
    int lid = tid % THREADS_PER_WARP; // lane id

    // load history into shared memory
    auto *sh_history = &block_sh_history[wid * nr_pattern_edges];
    auto *gl_history = &histories[emb_idx * nr_pattern_edges];
    if (lid == 0) { // TODO: collective load
        for (int i = 0; i < nr_pattern_edges; ++i) {
            sh_history[i].from  = gl_history[i].from;
            sh_history[i].to    = gl_history[i].to;
            sh_history[i].label = gl_history[i].label;
        }
    }
    __threadfence_block();
    
    int u = sh_history[rmpath[0]].to;
    int k, ei, ei_base = g_offsets[u];
    int degree = g_offsets[u + 1] - g_offsets[u];
    int nr_tasks = rmpath_len * degree;

    Edge cur;

    int &out_base = block_out_base[wid];
    int *out_offset = block_out_offset + wid * THREADS_PER_WARP;

    // backward
    for (int task_base = 0; task_base < nr_tasks; task_base += THREADS_PER_WARP) {
        int task_id = task_base + lid;
        bool ext = false;

        if (task_id < nr_tasks) {
            int rmpath_idx = task_id % rmpath_len;
            int neighbor_idx = task_id / rmpath_len;

            ei = ei_base + neighbor_idx;
            k = rmpath[rmpath_idx];
            auto &e = sh_history[k];
    
            cur.from = u, cur.to = g_edges[ei].to, cur.label = g_edges[ei].label;
            ext = 
                (!history_has_edge(sh_history, cur, nr_pattern_edges)) &&
                (cur.to == e.from) &&
                ((e.label < cur.label) || (e.label == cur.label && g_vlabels[e.to] <= g_vlabels[u]));
        }
        out_offset[lid] = int(ext);
        __threadfence_block();

        warp_inclusive_scan(out_offset, lid);

        if (lid == 0)
            out_base = atomicAdd(backward_idx, out_offset[THREADS_PER_WARP - 1]);
        __threadfence_block();

        if (ext) {
            int i = out_base + out_offset[lid] - 1;
            int *p = backward_out + (4 * i);
            p[0] = k;
            p[1] = cur.label;
            p[2] = emb_idx;
            p[3] = ei;
        }
    }

    // first forward
    nr_tasks = degree;
    for (int task_base = 0; task_base < nr_tasks; task_base += THREADS_PER_WARP) {
        int task_id = task_base + lid;
        bool ext = false;
        int to, edge_label, to_label;

        if (task_id < nr_tasks) {
            ei = ei_base + task_id;
            edge_label = g_edges[ei].label;
            to = g_edges[ei].to;
            to_label = g_vlabels[to];

            ext = (to_label >= min_label && !history_has_vertex(sh_history, to, nr_pattern_edges));
        }
        out_offset[lid] = int(ext);
        __threadfence_block();

        warp_inclusive_scan(out_offset, lid);

        if (lid == 0)
            out_base = atomicAdd(forward_idx, out_offset[THREADS_PER_WARP - 1]);
        __threadfence_block();

        if (ext) {
            int i = out_base + out_offset[lid] - 1;
            int *p = forward_out + (5 * i);
            p[0] = -1;
            p[1] = edge_label;
            p[2] = to_label;
            p[3] = emb_idx;
            p[4] = ei;
        }
    }

    // other forward
    int *degree_sums = &block_degree_sums[wid * MAX_K];
    if (lid == 0) { // TODO: collective load
        degree_sums[0] = 0;
        for (int i = 0; i < rmpath_len; ++i) {
            u = sh_history[rmpath[i]].from;
            degree_sums[i + 1] = degree_sums[i] + (g_offsets[u + 1] - g_offsets[u]);
        }
    }
    __threadfence_block();

    nr_tasks = degree_sums[rmpath_len];
    for (int task_base = 0; task_base < nr_tasks; task_base += THREADS_PER_WARP) {
        int task_id = task_base + lid;
        bool ext = false;
        int to, edge_label, to_label, to_label_bound;

        if (task_id < nr_tasks) {
            int i = 0;
            while (task_id >= degree_sums[i]) ++i;
            --i;

            k = rmpath[i];
            u = sh_history[k].from;
            to_label_bound = g_vlabels[sh_history[k].to];
            ei = g_offsets[u] + (task_id - degree_sums[i]);

            edge_label = g_edges[ei].label;
            to = g_edges[ei].to;
            to_label = g_vlabels[to];

            ext = (
                (to_label >= min_label) && (!history_has_vertex(sh_history, to, nr_pattern_edges)) &&
                (sh_history[k].label < edge_label || (((sh_history[k].label == edge_label) && (to_label_bound <= to_label))))
            );
        }
        out_offset[lid] = int(ext);
        __threadfence_block();

        warp_inclusive_scan(out_offset, lid);

        if (lid == 0)
            out_base = atomicAdd(forward_idx, out_offset[THREADS_PER_WARP - 1]);
        __threadfence_block();

        if (ext) {
            int i = out_base + out_offset[lid] - 1;
            int *p = forward_out + (5 * i);
            p[0] = k;
            p[1] = edge_label;
            p[2] = to_label;
            p[3] = emb_idx;
            p[4] = ei;
        }
    }
}

void extend_cuda(const CSRGraph& g, const DFSCode& dfs_code, LayeredEmbeddings& layered_embeddings,
    const RightMostPath& rmpath, ForwardMapping& forward_mapping, BackwardMapping& backward_mapping)
{
#if PROFILE
    TimerGuard guard("extend_cuda");
#endif

    auto &embeddings = *(layered_embeddings.back());
    int nr_embeddings = embeddings.size();

    size_t histories_size = nr_embeddings * dfs_code.size() * sizeof(Edge);
    Edge *dev_histories;

    cudaMalloc(&dev_histories, histories_size);
    cudaMemcpy(ctx.dev_rmpath, &rmpath[0], rmpath.size() * sizeof(int), cudaMemcpyHostToDevice);
    *ctx.new_backward_count = 0;
    *ctx.new_forward_count = 0;

    int block_size = THREADS_PER_BLOCK;
    int nr_blocks = (nr_embeddings + block_size - 1) / block_size;
    build_histories<<<nr_blocks, block_size>>>(dev_histories, ctx.dev_embedding_ptrs,
        ctx.g_edges, nr_embeddings, dfs_code.size());
    
    nr_blocks = (nr_embeddings + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem_size = WARPS_PER_BLOCK * dfs_code.size() * sizeof(Edge);
    extend_kernel<<<nr_blocks, block_size, shmem_size>>>(
        ctx.dev_backward_embeddings, ctx.new_backward_count, ctx.dev_forward_embeddings, ctx.new_forward_count,
        dev_histories, ctx.g_edges, ctx.g_vlabels, ctx.g_offsets,
        ctx.dev_rmpath, rmpath.size(), dfs_code.size(), nr_embeddings, dfs_code[0].from_label);
    ///////////
    using BackwardCounts = std::map<int, std::map<int, int>>;
    using ForwardCounts = std::map<int, std::map<int, std::map<int, int>>>;

    int nr_threads = omp_get_max_threads();
    auto local_backward_mappings = new BackwardMapping[nr_threads];
    auto local_forward_mappings = new ForwardMapping[nr_threads];
    auto local_backward_counts = new BackwardCounts[nr_threads];
    auto local_forward_counts = new ForwardCounts[nr_threads];
    ///////////
    cudaDeviceSynchronize();

    int new_backward_count = *ctx.new_backward_count, new_forward_count = *ctx.new_forward_count;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < new_backward_count; ++i) {
        int tid = omp_get_thread_num();
        auto p = ctx.dev_backward_embeddings + (4 * i);
        int k = p[0], edge_label = p[1], emb_idx = p[2], edge_idx = p[3];
        int v_i = dfs_code[k].from;
        local_backward_mappings[tid][v_i][edge_label].emplace_back(emb_idx, edge_idx);
        local_backward_counts[tid][v_i][edge_label]++;
    }

    int v_n = dfs_code[rmpath[0]].to;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < new_forward_count; ++i) {
        int tid = omp_get_thread_num();
        auto p = ctx.dev_forward_embeddings + (5 * i);
        int k = p[0], edge_label = p[1], to_label = p[2], emb_idx = p[3], edge_idx = p[4];
        int v_i = k < 0 ? v_n : dfs_code[k].from;
        local_forward_mappings[tid][v_i][edge_label][to_label].emplace_back(emb_idx, edge_idx);
        local_forward_counts[tid][v_i][edge_label][to_label]++;
    }

    for (int i = 1; i < nr_threads; ++i) {
        // calculate prefix sum
        foreach_backward_begin(local_backward_counts[i - 1])
            local_backward_counts[i][v][edge_label] += pair2.second;
        foreach_backward_end()
        
        foreach_forward_begin(local_forward_counts[i - 1])
            local_forward_counts[i][v][edge_label][to_label] += pair3.second;
        foreach_forward_end()
    }

    // reserve space
    foreach_backward_begin(local_backward_counts[nr_threads - 1])
        backward_mapping[v][edge_label].resize(pair2.second);
    foreach_backward_end()

    foreach_forward_begin(local_forward_counts[nr_threads - 1])
        forward_mapping[v][edge_label][to_label].resize(pair3.second);
    foreach_forward_end()

    // copy
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        foreach_backward_begin(local_backward_mappings[i])
            auto dst = backward_mapping[v][edge_label].begin()
                + ((i > 0) ? local_backward_counts[i - 1][v][edge_label] : 0);
            auto &local_embeddings = pair2.second;
            std::copy(local_embeddings.begin(), local_embeddings.end(), dst);
        foreach_backward_end()
            
        foreach_forward_begin(local_forward_mappings[i])
            auto dst = forward_mapping[v][edge_label][to_label].begin()
                + ((i > 0) ? local_forward_counts[i - 1][v][edge_label][to_label] : 0);
            auto &local_embeddings = pair3.second;
            std::copy(local_embeddings.begin(), local_embeddings.end(), dst);
        foreach_forward_end()
    }

    delete[] local_backward_mappings;
    delete[] local_forward_mappings;
    delete[] local_backward_counts;
    delete[] local_forward_counts;
    cudaFree(dev_histories);
}

void mine(const CSRGraph& g, DFSCode& dfs_code, LayeredEmbeddings& layered_embeddings, int k, int min_support)
{
    if (!is_canonical(dfs_code))
        return;

    auto sup = support_gpu(g, dfs_code, layered_embeddings);
    if (sup < min_support)
        return;
    
    report(dfs_code, sup);
    if (dfs_code.size() >= k)
        return;

    BackwardMapping backward_mapping;
    ForwardMapping forward_mapping;

    RightMostPath rmpath;
    build_right_most_path(dfs_code, rmpath);
    auto v_n = dfs_code[rmpath[0]].to;

    extend_omp(g, dfs_code, layered_embeddings, rmpath, forward_mapping, backward_mapping);
    // extend_cuda(g, dfs_code, layered_embeddings, rmpath, forward_mapping, backward_mapping);

    // backward
    for (auto &pair1 : backward_mapping) {
        auto v_i = pair1.first;
        for (auto &pair2 : pair1.second) {
            auto edge_label = pair2.first;
            dfs_code.emplace_back(v_n, v_i, -1, edge_label, -1);
            layered_embeddings.push_back(&pair2.second);
            mine(g, dfs_code, layered_embeddings, k, min_support);
            layered_embeddings.pop_back();
            dfs_code.pop_back();
        }
    }

    // forward (note: first level reversed)
    for (auto it1 = forward_mapping.rbegin(); it1 != forward_mapping.rend(); ++it1) {
        auto v_i = it1->first;
        for (auto &pair2 : it1->second) {
            auto edge_label = pair2.first;
            for (auto &pair3 : pair2.second) {
                auto to_label = pair3.first;
                dfs_code.emplace_back(v_i, v_n + 1, -1, edge_label, to_label);
                layered_embeddings.push_back(&pair3.second);
                mine(g, dfs_code, layered_embeddings, k, min_support);
                layered_embeddings.pop_back();
                dfs_code.pop_back();
            }
        }
    }
}

void fsm(const CSRGraph& g, int k, int min_support)
{
    init_gpu_context(g, k);
    init_cpu_context(g, k);

    std::map<PatternEdge, EmbeddingRels, InitialEdgeCompare_t> mapping;
    get_initial_embeddings(g, [&mapping](auto from_label, auto edge_label, auto to_label, auto edge_idx) {
        PatternEdge pe(0, 1, from_label, edge_label, to_label);
        mapping[pe].emplace_back(-1, edge_idx);
    });

    LayeredEmbeddings layered_embeddings;

    // printf("%ld single-edge code\n", mapping.size());
    for (auto &pair : mapping) {
        DFSCode dfs_code {pair.first};
        layered_embeddings.push_back(&pair.second);
        mine(g, dfs_code, layered_embeddings, k, min_support);
        layered_embeddings.pop_back();
    }

#if PROFILE
    printf("statistics:\n");
    for (auto &pair : timers) {
        double time = pair.second * 1e-6;
        printf("    %s - %lfs\n", pair.first.c_str(), time);
    }
#endif
}
