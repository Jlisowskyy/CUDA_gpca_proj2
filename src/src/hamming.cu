/* internal includes */
#include <hamming.cuh>
#include <cuda_trie.cuh>
#include <data.cuh>
#include <thread_mgr.cuh>

/* external includes */
#include <chrono>
#include <iostream>

// ------------------------------
// Kernels
// ------------------------------

__global__ void BuildTrieKernel(cuda_Trie *out_trie, uint32_t *buckets, uint32_t *prefix_len, cuda_Data *data,
                                cuda_Allocator *allocator) {
    // TODO
}

__global__ void FindAllHamming1Pairs(cuda_Trie *out_trie, const cuda_Data *data, cuda_Solution *solutions) {
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t num_threads = gridDim.x * blockDim.x;
    const uint32_t num_sequences = data->GetNumSequences();

    for (uint32_t seq_idx = thread_idx; seq_idx < num_sequences; seq_idx += num_threads) {
        out_trie->FindPairs(seq_idx, *data, solutions[seq_idx]);
    }
}

// ------------------------------
// GPU interface functions
// ------------------------------

std::tuple<cuda_Trie *, cuda_Data *, cuda_Allocator *> InitHamming(const BinSequencePack &pack) {
    /* convert pack to cuda_Data */
    cuda_Data data{pack};

    /* calculate management data */
    const ThreadMgr mgr{};
    const auto mgr_data = mgr.PrepareTrieBuildData(pack);

    /* prepare allocator */
    cuda_Allocator allocator(mgr_data.max_nodes, mgr_data.max_threads, mgr_data.max_nodes_per_thread);

    /* prepare GPU data */

    const auto t_mem_start = std::chrono::high_resolution_clock::now();

    cuda_Trie *d_trie;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_trie, sizeof(cuda_Trie)));
    cuda_Allocator *d_allocator = allocator.DumpToGPU();
    cuda_Data *d_data = data.DumpToGPU();

    const auto t_mem_end = std::chrono::high_resolution_clock::now();

    std::cout << "Memory allocation and transfer time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms"
            << std::endl;

    const auto t_start = std::chrono::high_resolution_clock::now();

    /* start trie build */
    BuildTrieKernel<<<mgr_data.num_blocks, mgr_data.num_threads_per_block>>>(d_trie, mgr_data.d_buckets,
                                                                             mgr_data.d_bucket_prefix_len, d_data,
                                                                             d_allocator);

    /* cleanup host data */
    data.DeallocHost();
    allocator.DeallocHost();

    /* sync */
    CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());

    const auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Trie build time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms"
            << std::endl;

    /* cleanup management data */
    CUDA_ASSERT_SUCCESS(cudaFree(mgr_data.d_buckets));
    CUDA_ASSERT_SUCCESS(cudaFree(mgr_data.d_bucket_prefix_len));

    return {d_trie, d_data, d_allocator};
}

std::vector<std::tuple<uint32_t, uint32_t> > Hamming1Distance(cuda_Trie *d_trie, cuda_Data *d_data,
                                                              cuda_Allocator *d_allocator) {
    // ------------------------------
    // Mem init
    // ------------------------------

    /* calculate management data */
    const ThreadMgr mgr{};
    const auto mgr_data = mgr.PrepareSearchData();

    const auto t_mem_start = std::chrono::high_resolution_clock::now();

    /* prepare allocator for solutions */
    auto [d_sol, d_mem_block] = cuda_Solution::DumpToGPU(mgr_data.num_solutions);

    const auto t_mem_end = std::chrono::high_resolution_clock::now();

    std::cout << "Memory allocation and transfer time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms"
            << std::endl;

    // ------------------------------
    // Processing
    // ------------------------------

    const auto t_start = std::chrono::high_resolution_clock::now();

    /* find all pairs */
    FindAllHamming1Pairs<<<mgr_data.num_blocks, mgr_data.num_threads_per_block>>>(d_trie, d_data, d_sol);
    CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());

    const auto t_end = std::chrono::high_resolution_clock::now();

    std::cout << "Search time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms"
            << std::endl;

    // ------------------------------
    // Mem back transfer
    // ------------------------------

    const auto t_mem_back_start = std::chrono::high_resolution_clock::now();

    /* Process solutions */
    std::vector<std::tuple<uint32_t, uint32_t> > solutions;
    uint32_t *h_sol;
    CUDA_ASSERT_SUCCESS(cudaMemcpy(&h_sol, d_mem_block, cuda_Solution::GetMemBlockSize(mgr_data.num_solutions),
        cudaMemcpyDeviceToHost));

    for (size_t idx = 0; idx < mgr_data.num_solutions; ++idx) {
        const uint32_t idx1 = h_sol[idx * 2];
        const uint32_t idx2 = h_sol[idx * 2 + 1];

        if (idx1 == INT_MAX || idx2 == INT_MAX) {
            /* Solution are terminated by INT_MAX */
            break;
        }

        solutions.emplace_back(idx1, idx2);
    }

    const auto t_mem_back_end = std::chrono::high_resolution_clock::now();

    std::cout << "Memory back transfer time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_mem_back_end - t_mem_back_start).count() << "ms"
            << std::endl;

    // ------------------------------
    // Cleanup
    // ------------------------------

    const auto t_cleanup_start = std::chrono::high_resolution_clock::now();

    /* cleanup solutions */
    CUDA_ASSERT_SUCCESS(cudaFree(d_mem_block));
    CUDA_ASSERT_SUCCESS(cudaFree(d_sol));

    /* cleanup data */
    cuda_Data::DeallocGPU(d_data);

    /* cleanup allocator */
    cuda_Allocator::DeallocGPU(d_allocator);

    /* cleanup trie */
    CUDA_ASSERT_SUCCESS(cudaFree(d_trie));

    const auto t_cleanup_end = std::chrono::high_resolution_clock::now();

    std::cout << "Cleanup time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_cleanup_end - t_cleanup_start).count() << "ms"
            << std::endl;

    return solutions;
}
