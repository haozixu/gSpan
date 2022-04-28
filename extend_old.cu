struct EdgeGPU {
    int from, to, label;
};

// each thread creates the corresponding History object of an Embedding
__global__ void build_histories(EdgeGPU* histories, EmbeddingRel** embedding_ptrs, int* out_degrees,
    const int* g_from, const int* g_to, const int* g_offsets, const int* g_elabels,
    int nr_embeddings, int nr_pattern_edges, const int* rmpath)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nr_embeddings) {
        EdgeGPU *history = histories + (tid * nr_pattern_edges);
        int i = tid;
        for (int layer = nr_pattern_edges - 1; layer >= 0; --layer) {
            EmbeddingRel *emb = embedding_ptrs[layer] + i;
            auto ei = emb->edge_id;
            auto from = g_from[ei];
            auto to = g_to[ei];
            auto edge_label = g_elabels[ei];

            EdgeGPU *e = &history[layer];
            e->from = from, e->to = to, e->label = edge_label;

            i = emb->prev_idx;
        }

        auto u_n = history[rmpath[0]].to;
        out_degrees[tid] = g_offsets[u_n + 1] - g_offsets[u_n];
    }
}

__device__ bool history_has_edge(const EdgeGPU* h, const EdgeGPU& e, int history_len)
{
    for (int i = 0; i < history_len; ++i) {
        if ((h[i].from == e.from && h[i].to == e.to && h[i].label == e.label) || 
            (h[i].to == e.from && h[i].from == e.to && h[i].label == e.label))
            return true;
    }
    return false;
}

constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

__global__ void extend_backward_kernel(int* out, int* out_idx, const EdgeGPU* histories, 
    const int* g_to, const int* g_elabels, const int* g_vlabels, const int* g_offsets,
    int* rmpath, int rmpath_len, int nr_pattern_edges, int nr_embeddings, int max_nr_backward_embeddings)
{
    extern __shared__ EdgeGPU block_sh_history[];
    __shared__ int block_out_base[WARPS_PER_BLOCK];
    __shared__ int block_out_offset[THREADS_PER_BLOCK];

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
            sh_history[i].from = gl_history[i].from;
            sh_history[i].to = gl_history[i].to;
            sh_history[i].label = gl_history[i].label;
        }
    }
    __threadfence_block();
    
    int u_n = sh_history[rmpath[0]].to;
    auto ei_base = g_offsets[u_n];
    int un_degree = g_offsets[u_n + 1] - g_offsets[u_n];
    int nr_tasks = rmpath_len * un_degree;

    int &out_base = block_out_base[wid];
    int *out_offset = block_out_offset + wid * THREADS_PER_WARP;
    for (int task_base = 0; task_base < nr_tasks; task_base += THREADS_PER_WARP) {
        int task_id = task_base + lid;
        int ei, k;
        EdgeGPU cur;
        bool ext = false;

        if (task_id < nr_tasks) {
            int rmpath_idx = task_id % rmpath_len;
            int neighbor_idx = task_id / rmpath_len;

            ei = ei_base + neighbor_idx;
            k = rmpath[rmpath_idx];
            EdgeGPU &e = sh_history[k];
    
            cur.from = u_n, cur.to = g_to[ei], cur.label = g_elabels[ei];
            ext = 
                (!history_has_edge(sh_history, cur, nr_pattern_edges)) &&
                (cur.to == e.from) &&
                ((e.label < cur.label) || (e.label == cur.label && g_vlabels[e.to] <= g_vlabels[u_n]));
        }
        out_offset[lid] = int(ext);
        __threadfence_block();

        #pragma unroll
        for (int s = 1; s < THREADS_PER_WARP; s *= 2) {
            int v = lid >= s ? out_offset[lid - s] : 0;
            out_offset[lid] += v;
            __threadfence_block();
        }

        if (lid == 0) {
            out_base = atomicAdd(out_idx, out_offset[THREADS_PER_WARP - 1]);
        }
        __threadfence_block();

        if (ext) {
            int i = out_base + out_offset[lid] - 1;
            // if (i < max_nr_backward_embeddings) {
                int *p = out + (4 * i);
                p[0] = k;
                p[1] = cur.label;
                p[2] = emb_idx;
                p[3] = ei;
            // } else {
                // printf("index=%d limit=%d\n", i, max_nr_backward_embeddings);
            // }
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

    size_t histories_size = nr_embeddings * dfs_code.size() * sizeof(EdgeGPU);
    EdgeGPU *dev_histories;
    int *dev_un_degrees;

    cudaMalloc(&dev_histories, histories_size);
    cudaMalloc(&dev_un_degrees, nr_embeddings * sizeof(int));
    cudaMemcpy(ctx.dev_rmpath, &rmpath[0], rmpath.size() * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = THREADS_PER_BLOCK;
    int nr_blocks = (nr_embeddings + block_size - 1) / block_size;
    build_histories<<<nr_blocks, block_size>>>(dev_histories, ctx.dev_embedding_ptrs, dev_un_degrees,
        ctx.g_from, ctx.g_to, ctx.g_offsets, ctx.g_elabels, nr_embeddings, dfs_code.size(), ctx.dev_rmpath);
    
    cudaMemset(ctx.new_backward_count, 0, sizeof(int));

    void *dev_temp = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(dev_temp, temp_storage_bytes, dev_un_degrees, ctx.max_nr_backward_embeddings, nr_embeddings);
    cudaMalloc(&dev_temp, temp_storage_bytes);

    cudaDeviceSynchronize(); // build_histories -> dev_un_degrees
    cub::DeviceReduce::Sum(dev_temp, temp_storage_bytes, dev_un_degrees, ctx.max_nr_backward_embeddings, nr_embeddings);
    cudaDeviceSynchronize();

    // 4: v_i, edge_label | prev_idx, edge_id
    size_t max_backward_embeddings_size = (*ctx.max_nr_backward_embeddings) * 4 * sizeof(int);
    int *dev_backward_embeddings;
    cudaMallocManaged(&dev_backward_embeddings, max_backward_embeddings_size);

    nr_blocks = (nr_embeddings + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem_size = WARPS_PER_BLOCK * dfs_code.size() * sizeof(EdgeGPU);
    extend_backward_kernel<<<nr_blocks, block_size, shmem_size>>>(dev_backward_embeddings, ctx.new_backward_count, dev_histories,
        ctx.g_to, ctx.g_elabels, ctx.g_vlabels, ctx.g_offsets,
        ctx.dev_rmpath, rmpath.size(), dfs_code.size(), nr_embeddings, *ctx.max_nr_backward_embeddings);
    cudaDeviceSynchronize();

    // BackwardMapping new_backward_mapping;
    int new_backward_count = *ctx.new_backward_count;
    if (new_backward_count)
        printf("new backward count: %d real usage: %lf\n", new_backward_count, double(new_backward_count) / (*ctx.max_nr_backward_embeddings));
    for (int i = 0; i < new_backward_count; ++i) {
        auto p = dev_backward_embeddings + (4 * i);
        int k = p[0], edge_label = p[1], emb_idx = p[2], edge_id = p[3];
        int v_i = dfs_code[k].from;
        backward_mapping[v_i][edge_label].emplace_back(emb_idx, edge_id);
    }

    extend_forward_only_omp(g, dfs_code, layered_embeddings, rmpath, forward_mapping, backward_mapping);

    cudaFree(dev_backward_embeddings);
    cudaFree(dev_temp);
    cudaFree(dev_un_degrees);
    cudaFree(dev_histories);
}

void extend_backward_only_omp(const CSRGraph& g, const DFSCode& dfs_code, LayeredEmbeddings& layered_embeddings,
    const RightMostPath& rmpath, ForwardMapping& forward_mapping, BackwardMapping& backward_mapping)
{
#if PROFILE
    TimerGuard guard("extend_backward_omp");
#endif

    using BackwardCounts = std::map<int, std::map<int, int>>;

    int nr_threads = omp_get_max_threads();
    auto local_backward_mappings = new BackwardMapping[nr_threads];
    auto local_backward_counts = new BackwardCounts[nr_threads];

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
                auto edge_label = g.elabels[edge_idx];
                local_backward_mappings[tid][v_i][edge_label].emplace_back(emb_idx, edge_idx);
                local_backward_counts[tid][v_i][edge_label]++;
            }
        }
    }

    for (int i = 1; i < nr_threads; ++i) {
        // calculate prefix sum
        foreach_backward_begin(local_backward_counts[i - 1])
            local_backward_counts[i][v][edge_label] += pair2.second;
        foreach_backward_end()
    }

    // reserve space
    foreach_backward_begin(local_backward_counts[nr_threads - 1])
        backward_mapping[v][edge_label].resize(pair2.second);
    foreach_backward_end()

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
    }

    delete[] local_backward_mappings;
    delete[] local_backward_counts;
}

void extend_forward_only_omp(const CSRGraph& g, const DFSCode& dfs_code, LayeredEmbeddings& layered_embeddings,
    const RightMostPath& rmpath, ForwardMapping& forward_mapping, BackwardMapping& backward_mapping)
{
#if PROFILE
    // TimerGuard guard("extend_forward_omp");
#endif

    using ForwardCounts = std::map<int, std::map<int, std::map<int, int>>>;

    int nr_threads = omp_get_max_threads();
    auto local_forward_mappings = new ForwardMapping[nr_threads];
    auto local_forward_counts = new ForwardCounts[nr_threads];

    auto min_label = dfs_code[0].from_label;
    auto v_n = dfs_code[rmpath[0]].to;
    auto &embeddings = *(layered_embeddings.back());

    #pragma omp parallel for schedule(guided)
    for (int emb_idx = 0; emb_idx < embeddings.size(); ++emb_idx) {
        EmbVector history(g, layered_embeddings, emb_idx);
        auto u_n = history[rmpath[0]].to;

        int tid = omp_get_thread_num();

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
        foreach_forward_begin(local_forward_counts[i - 1])
            local_forward_counts[i][v][edge_label][to_label] += pair3.second;
        foreach_forward_end()
    }

    // reserve space
    foreach_forward_begin(local_forward_counts[nr_threads - 1])
        forward_mapping[v][edge_label][to_label].resize(pair3.second);
    foreach_forward_end()

    // copy
    #pragma omp parallel
    {
        int i = omp_get_thread_num();            
        foreach_forward_begin(local_forward_mappings[i])
            auto dst = forward_mapping[v][edge_label][to_label].begin()
                + ((i > 0) ? local_forward_counts[i - 1][v][edge_label][to_label] : 0);
            auto &local_embeddings = pair3.second;
            std::copy(local_embeddings.begin(), local_embeddings.end(), dst);
        foreach_forward_end()
    }

    delete[] local_forward_mappings;
    delete[] local_forward_counts;
}