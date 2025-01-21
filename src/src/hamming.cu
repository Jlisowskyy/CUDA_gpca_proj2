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
#include <cstdio>
#include <defines.hpp>
#include <errno.h>
#include <global_conf.hpp>

static std::tuple<cuda_Trie *, cuda_Data *, cuda_Allocator *> InitHamming(const BinSequencePack &pack);

static std::vector<std::tuple<uint32_t, uint32_t> > Hamming1Distance(cuda_Trie *d_trie, cuda_Data *d_data,
                                                                     cuda_Allocator *d_allocator);

// ------------------------------
// Kernels
// ------------------------------

__global__ void BuildTrieKernel(cuda_Trie *out_trie, uint32_t *buckets, uint32_t *prefix_len, cuda_Data *data,
                                cuda_Allocator *allocator) {
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

}

__global__ void FindAllHamming1Pairs(const cuda_Trie *out_trie, const cuda_Allocator *alloca, const cuda_Data *data,
                                     cuda_Solution *solutions) {
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t num_threads = gridDim.x * blockDim.x;
    const uint32_t num_sequences = data->GetNumSequences();

    for (uint32_t seq_idx = thread_idx; seq_idx < num_sequences; seq_idx += num_threads) {
        out_trie->FindPairs(seq_idx, *alloca, *data, *solutions);
    }
}

__global__ void TestTrieBuild(const cuda_Trie *trie, const cuda_Data *data, const cuda_Allocator *alloca,
                              bool *result) {
    for (size_t idx = 0; idx < data->GetNumSequences(); ++idx) {
        result[idx] = trie->Search(*alloca, idx, *data);
    }
}

// ------------------------------
// Static functions
// ------------------------------

static void _testTrie(const size_t num_seq, cuda_Trie *d_trie, cuda_Data *d_data, cuda_Allocator *d_alloca) {
    bool *d_result;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_result, sizeof(bool) * num_seq));
    TestTrieBuild<<<1, 1>>>(d_trie, d_data, d_alloca, d_result);

    auto h_result = new bool[num_seq];
    CUDA_ASSERT_SUCCESS(cudaMemcpy(h_result, d_result, sizeof(bool) * num_seq, cudaMemcpyDeviceToHost));
    CUDA_ASSERT_SUCCESS(cudaFree(d_result));

    uint64_t num_failed{};
    for (size_t idx = 0; idx < num_seq; ++idx) {
        num_failed += !h_result[idx];
        if (!h_result[idx]) {
            std::cerr << "Failed to find sequence " << idx << std::endl;
        }
    }

    if (num_failed == 0) {
        std::cout << "[SUCCESS] All sequences found" << std::endl;
    } else {
        std::cout << "[FAILED] " << num_failed << " sequences not found" << std::endl;
    }

    CUDA_ASSERT_SUCCESS(cudaFree(d_trie));
    cuda_Data::DeallocGPU(d_data);
    cuda_Allocator::DeallocGPU(d_alloca);
    delete h_result;
}

static std::tuple<cuda_Trie *, cuda_Data *, cuda_Allocator *> _buildOnDevice(
    const MgrTrieBuildData &mgr_data, const BinSequencePack &pack) {
    /* prepare GPU data */
    std::cout << "Preparing GPU data..." << std::endl;
    const auto t_mem_start = std::chrono::high_resolution_clock::now();

    /* convert pack to cuda_Data */
    cuda_Data data{pack};
    cuda_Data *d_data = data.DumpToGPU();

    /* prepare allocator */
    cuda_Allocator allocator(mgr_data.max_nodes, mgr_data.max_threads, mgr_data.max_nodes_per_thread);
    cuda_Allocator *d_allocator = allocator.DumpToGPU();

    cuda_Trie *d_trie;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_trie, sizeof(cuda_Trie)));

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
    const auto [num_threads, prefix_mask, power_of_2] = GetBatchSplitData();

    std::cout << "Preparing data..." << std::endl;
    const auto t_build_start = std::chrono::high_resolution_clock::now();

    /* split sequences into `num_threads` buckets */

    /* prepare allocator */
    BigChunkAllocator allocator(kDefaultAllocChunkSize, num_threads);

    /* prepare buckets */
    Buckets buckets(num_threads, num_threads, allocator);

    /* Prepare thread pool */
    ThreadPool pool(num_threads);
    std::barrier barrier(static_cast<ptrdiff_t>(num_threads));
    const size_t thread_job_size = pack.sequences.size() / num_threads;

    cuda_Data data{pack};
    /* Dump before creating cuda_alloc to know available memory */
    cuda_Data *d_data = data.DumpToGPU();

    /* Prepare cuda alloca */
    cuda_Allocator cuda_allocator(pack.sequences.size() * pack.max_seq_size_bits, num_threads, pack.max_seq_size_bits);

    /* Prepare tries */
    std::vector<cuda_Trie> tries;
    tries.resize(num_threads);

    std::cout << "Sorting sequences..." << std::endl;

    /* Run threads */
    pool.RunThreads([&](const uint32_t thread_idx) {
        const size_t job_start = thread_idx * thread_job_size;
        const size_t job_end = thread_idx == num_threads - 1 ? pack.sequences.size()
            : (thread_idx + 1) * thread_job_size;

        /* Bucket sorting */
        for (size_t seq_idx = job_start; seq_idx < job_end; ++seq_idx) {
            const size_t key = pack.sequences[seq_idx].GetWord(0) & prefix_mask;
            buckets.PushToBucket(thread_idx, key, seq_idx);
        }

        /* Wait for all threads finish radix sort */
        barrier.arrive_and_wait();

        if (thread_idx == 0) {
            buckets.MergeBuckets();

            std::cout << "Bucket stats: " << std::endl;
            for (size_t b_idx = 0; b_idx < num_threads; ++b_idx) {
                std::cout << "Bucket " << b_idx << " size: " << buckets.GetBucketSize(b_idx) << std::endl;
            }
            std::cout << "Started trie build..." << std::endl;
        }

        /* wait for bucket merge */
        barrier.arrive_and_wait();

        /* fill the tries */
        auto &trie = tries[thread_idx];

        while (!buckets.IsEmpty(thread_idx)) {
            const auto seq_idx = buckets.PopBucket(thread_idx);
            trie.Insert(thread_idx, cuda_allocator, seq_idx, power_of_2, data);

            cuda_allocator.ConsolidateHost(thread_idx, barrier, buckets.IsEmpty(thread_idx));
        }
    });

    pool.Wait();

    /* merge tries */
    cuda_Trie final_trie{};

    std::cout << "Merging tries..." << std::endl;
    final_trie.MergeByPrefixHost(cuda_allocator, data, tries, power_of_2);

    const auto t_build_end = std::chrono::high_resolution_clock::now();

    std::cout << "Trie build time using CPU: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_build_end - t_build_start).count() << "ms"
            << std::endl;

    cuda_allocator.DisplayAllocInfo();

    if (GlobalConfig.WriteDotFiles) {
        const bool result = final_trie.DumpToDotFile(cuda_allocator, data, "/tmp/trie.dot", "TRIE");
        assert(result);
    }

    /* transfer data to GPU */
    const auto t_transfer_start = std::chrono::high_resolution_clock::now();

    cuda_Trie *d_trie = final_trie.DumpToGpu();
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
    FindAllHamming1Pairs<<<mgr_data.num_blocks, mgr_data.num_threads_per_block>>>(d_trie, d_allocator, d_data, d_sol);
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
    std::vector<uint32_t> h_sol;
    h_sol.resize(cuda_Solution::GetMemBlockSize(mgr_data.num_solutions));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(h_sol.data(), d_mem_block, cuda_Solution::GetMemBlockSize(mgr_data.num_solutions),
        cudaMemcpyDeviceToHost));

    for (size_t idx = 0; idx < mgr_data.num_solutions; ++idx) {
        const uint32_t idx1 = h_sol[idx * 2];
        const uint32_t idx2 = h_sol[idx * 2 + 1];

        if (idx1 == UINT32_MAX || idx2 == UINT32_MAX) {
            /* Solution are terminated by UINT32_MAX */
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

void TestCUDATrieCpuBuild(const BinSequencePack &pack) {
    const auto [d_trie, d_data, d_alloca] = _buildOnHost(pack);
    _testTrie(pack.sequences.size(), d_trie, d_data, d_alloca);
}

void TestCUDATrieGPUBuild(const BinSequencePack &pack) {
    const ThreadMgr mgr{};
    const auto mgr_data = mgr.PrepareTrieBuildData(pack);

    const auto [d_trie, d_data, d_alloca] = _buildOnDevice(mgr_data, pack);
    _testTrie(pack.sequences.size(), d_trie, d_data, d_alloca);
}
