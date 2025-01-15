/* internal includes */
#include <hamming.cuh>
#include <cuda_trie.cuh>
#include <data.cuh>
#include <thread_mgr.cuh>
#include <thread_pool.hpp>
#include <allocators.hpp>

/* external includes */
#include <chrono>
#include <iostream>
#include <barrier>
#include <bit>
#include <errno.h>

static std::tuple<cuda_Trie *, cuda_Data *, cuda_Allocator *> InitHamming(const BinSequencePack &pack);

static std::vector<std::tuple<uint32_t, uint32_t> > Hamming1Distance(cuda_Trie *d_trie, cuda_Data *d_data,
                                                                     cuda_Allocator *d_allocator);

// ------------------------------
// Helpers
// ------------------------------

static uint32_t _getMask(const uint32_t num_bits) {
    uint32_t mask{};

    for (uint32_t idx = 0; idx < num_bits; ++idx) {
        mask |= static_cast<uint32_t>(1) << idx;
    }

    return mask;
}

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
// Static functions
// ------------------------------

static std::tuple<cuda_Trie *, cuda_Data *, cuda_Allocator *> _buildOnDevice(
    const MgrTrieBuildData &mgr_data, const BinSequencePack &pack) {
    /* convert pack to cuda_Data */
    cuda_Data data{pack};

    /* prepare allocator */
    cuda_Allocator allocator(mgr_data.max_nodes, mgr_data.max_threads, mgr_data.max_nodes_per_thread);

    /* prepare GPU data */
    std::cout << "Preparing GPU data..." << std::endl;
    const auto t_mem_start = std::chrono::high_resolution_clock::now();

    cuda_Trie *d_trie;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_trie, sizeof(cuda_Trie)));
    cuda_Allocator *d_allocator = allocator.DumpToGPU();
    cuda_Data *d_data = data.DumpToGPU();

    const auto t_mem_end = std::chrono::high_resolution_clock::now();

    std::cout << "Memory allocation and transfer time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_mem_end - t_mem_start).count() << "ms"
            << std::endl;

    std::cout << "Building trie..." << std::endl;
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

static std::tuple<cuda_Trie *, cuda_Data *, cuda_Allocator *> _buildOnHost(const BinSequencePack &pack) {
    const uint64_t num_threads = std::bit_floor(std::thread::hardware_concurrency());
    const uint64_t power_of_2 = std::countr_zero(num_threads);
    const uint32_t prefix_mask = _getMask(power_of_2);

    std::cout << "Preparing data..." << std::endl;
    const auto t_build_start = std::chrono::high_resolution_clock::now();

    /* split sequences into `num_threads` buckets */
    StabAllocator<Node<uint32_t> > allocator(pack.sequences.size() + num_threads);

    /* Prepare buckets */
    std::vector<ThreadSafeStack<uint32_t> > buckets{};
    buckets.reserve(num_threads);

    for (size_t idx = 0; idx < num_threads; ++idx) {
        buckets.emplace_back(allocator);
    }

    /* Prepare thread pool */
    ThreadPool pool(num_threads);
    std::barrier barrier(static_cast<ptrdiff_t>(num_threads));
    std::atomic<int32_t> thread_counter;
    thread_counter.store(static_cast<int32_t>(num_threads));

    /* Prepare cuda alloca */
    cuda_Allocator cuda_allocator(pack.sequences.size() * pack.max_seq_size_bits, num_threads, pack.max_seq_size_bits);

    /* convert pack to cuda_Data */
    cuda_Data data{pack};

    /* Prepare tries */
    std::vector<cuda_Trie> tries;
    tries.resize(num_threads);

    std::cout << "Sorting sequences..." << std::endl;

    /* Run threads */
    pool.RunThreads([&](const uint32_t thread_idx) {
        /* Bucket sorting */
        for (size_t seq_idx = thread_idx; seq_idx < pack.sequences.size(); seq_idx += num_threads) {
            const size_t key = pack.sequences[seq_idx].GetWord(0) & prefix_mask;
            buckets[key].PushSafe(seq_idx);
        }

        if (thread_idx == 0) {
            std::cout << "Bucket stats: " << std::endl;
            for (size_t b_idx = 0; b_idx < buckets.size(); ++b_idx) {
                std::cout << "Bucket " << b_idx << " size: " << buckets[b_idx].GetSize() << std::endl;
            }
            std::cout << "Started trie build..." << std::endl;
        }

        /* Wait for all threads finish radix sort */
        barrier.arrive_and_wait();

        /* fill the tries */
        auto &bucket = buckets[thread_idx];
        auto &trie = tries[thread_idx];

        while (!bucket.IsEmpty()) {
            const auto seq_idx = bucket.PopNotSafe();
            trie.Insert(thread_idx, cuda_allocator, seq_idx, power_of_2, data);

            cuda_allocator.ConsolidateHost(thread_idx, barrier, bucket.IsEmpty());
        }
    });

    pool.Wait();

    /* merge tries */
    cuda_Trie final_trie{};

    std::cout << "Merging tries..." << std::endl;
    final_trie.MergeByPrefixHost(tries, power_of_2);

    const auto t_build_end = std::chrono::high_resolution_clock::now();

    std::cout << "Trie build time using CPU: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_build_end - t_build_start).count() << "ms"
            << std::endl;

    /* transfer data to GPU */
    const auto t_transfer_start = std::chrono::high_resolution_clock::now();

    cuda_Trie *d_trie = final_trie.DumpToGpu();
    cuda_Data *d_data = data.DumpToGPU();
    cuda_Allocator *d_allocator = cuda_allocator.DumpToGPU();

    data.DeallocHost();
    cuda_allocator.DeallocHost();

    const auto t_transfer_end = std::chrono::high_resolution_clock::now();

    std::cout << "Memory transfer time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_transfer_end - t_transfer_start).count() << "ms"
            << std::endl;

    return {d_trie, d_data, d_allocator};
}

// ------------------------------
// GPU interface functions
// ------------------------------

std::tuple<cuda_Trie *, cuda_Data *, cuda_Allocator *> InitHamming(const BinSequencePack &pack) {
    /* calculate management data */
    const ThreadMgr mgr{};
    const auto mgr_data = mgr.PrepareTrieBuildData(pack);

    if (mgr_data.build_on_device) {
        std::cout << "Building on device..." << std::endl;
        return _buildOnDevice(mgr_data, pack);
    }

    std::cout << "Building on host..." << std::endl;
    return _buildOnHost(pack);
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

std::vector<std::tuple<uint32_t, uint32_t> > FindHamming1PairsCUDA(const BinSequencePack &pack) {
    const auto [d_trie, d_data, d_alloca] = InitHamming(pack);
    return Hamming1Distance(d_trie, d_data, d_alloca);
}
