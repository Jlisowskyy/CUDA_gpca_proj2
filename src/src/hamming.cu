/* internal includes */
#include <hamming.cuh>
#include <cuda_trie.cuh>
#include <data.cuh>
#include <thread_mgr.cuh>
#include <thread_pool.hpp>
#include <allocators.hpp>
#include <allocators.cuh>

/* external includes */
#include <chrono>
#include <iostream>
#include <barrier>
#include <bit>
#include <cstdio>
#include <defines.hpp>
#include <errno.h>
#include <global_conf.hpp>

static std::tuple<cuda_Trie *, cuda_Data *, FastAllocator *> InitHamming(const BinSequencePack &pack);

static std::vector<std::tuple<uint32_t, uint32_t> > Hamming1Distance(cuda_Trie *d_trie, cuda_Data *d_data,
                                                                     FastAllocator *d_allocator);

// ------------------------------
// Kernels
// ------------------------------

// TODO: Does not work due to allocator deadlock
__global__ void BuildTrieKernel(cuda_Trie *out_tries,
                                const uint32_t *buckets,
                                const uint32_t *bucket_sizes,
                                const uint32_t prefix_len,
                                const uint32_t max_occup,
                                const cuda_Data *data,
                                FastAllocator *allocator) {
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t num_threads = gridDim.x * blockDim.x;

    /* load own data */
    cuda_Trie &my_trie = out_tries[thread_idx];
    my_trie.Reset();

    const uint32_t my_bucket_size = bucket_sizes[thread_idx];
    assert(my_bucket_size <= max_occup);

    /* build trie */
    uint32_t bucket_offset = 0;
    for (; bucket_offset < my_bucket_size; ++bucket_offset) {
        const uint32_t seq_idx = buckets[bucket_offset * num_threads + thread_idx];
        assert(seq_idx < data->GetNumSequences());

        my_trie.Insert<true>(thread_idx, *allocator, seq_idx, prefix_len, *data);
    }

    /* wait for all threads to finish */
    __syncthreads();

    /* start merging by logarithmic reduction */
    uint32_t other_thread_idx = thread_idx;
    for (auto bit_pos = static_cast<int32_t>(prefix_len - 1); bit_pos >= 0; --bit_pos) {
        const uint32_t mask = static_cast<uint32_t>(1) << bit_pos;

        if ((thread_idx & mask) != 0) {
            /* only threads with 0 on given position will merge */
            return;
        }

        /* remove next bit */
        other_thread_idx &= ~mask;

        /* take the other trie */
        cuda_Trie &other_trie = out_tries[other_thread_idx | mask];

        my_trie.MergeWithOther<true>(thread_idx, other_trie, *allocator);

        /* wait for other thread to finish */
        __syncthreads();
    }
}

__global__ void FindAllHamming1Pairs(const cuda_Trie *out_trie, const FastAllocator *alloca, const cuda_Data *data,
                                     cuda_Solution *solutions) {
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t num_threads = gridDim.x * blockDim.x;
    const uint32_t num_sequences = data->GetNumSequences();

    for (uint32_t seq_idx = thread_idx; seq_idx < num_sequences; seq_idx += num_threads) {
        out_trie->FindPairs(seq_idx, *alloca, *data, *solutions);
    }
}

__global__ void TestTrieBuild(const cuda_Trie *trie, const cuda_Data *data, const FastAllocator *alloca,
                              bool *result) {
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t num_threads = gridDim.x * blockDim.x;

    for (uint32_t idx = thread_idx; idx < data->GetNumSequences(); idx += num_threads) {
        result[idx] = trie->Search(*alloca, idx, *data);
    }
}

// ------------------------------
// Static functions
// ------------------------------

static void _testTrie(const size_t num_seq, cuda_Trie *d_trie, cuda_Data *d_data, FastAllocator *d_alloca) {
    bool *d_result;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_result, sizeof(bool) * num_seq));
    TestTrieBuild<<<64, 1024>>>(d_trie, d_data, d_alloca, d_result);
    CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());

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
    FastAllocator::DeallocGPU(d_alloca);
    delete h_result;
}

static std::tuple<cuda_Trie *, cuda_Data *, FastAllocator *> _buildOnDevice(
    const MgrTrieBuildData &mgr_data, const BinSequencePack &pack) {
    // ------------------------------
    // Prepare objects
    // ------------------------------

    /* convert pack to cuda_Data */
    cuda_Data data{pack};
    cuda_Data *d_data = data.DumpToGPU();

    /* prepare allocator */
    FastAllocator allocator(kGpuThreadChunkSize, mgr_data.max_threads);
    FastAllocator *d_allocator = allocator.DumpToGPU();

    cuda_Trie *d_tries;
    CUDA_ASSERT_SUCCESS(
        cudaMallocAsync(&d_tries, mgr_data.max_threads * sizeof(cuda_Trie), g_cudaGlobalConf->asyncStream));

    // ------------------------------
    // Build the TRIE on GPU
    // ------------------------------

    /* start trie build */
    BuildTrieKernel<<<mgr_data.num_blocks, mgr_data.num_threads_per_block, 0, g_cudaGlobalConf->asyncStream>>>(
        d_tries,
        mgr_data.d_buckets,
        mgr_data.d_bucket_sizes,
        mgr_data.bucket_prefix_len,
        mgr_data.max_occup,
        d_data,
        d_allocator
    );

    // ------------------------------
    // Cleanup
    // ------------------------------

    /* cleanup management data */
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(mgr_data.d_buckets, g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(mgr_data.d_bucket_sizes, g_cudaGlobalConf->asyncStream));

    /* sync with stream */
    CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(g_cudaGlobalConf->asyncStream));

    /* cleanup host data */
    data.DeallocHost();
    allocator.DeallocHost();

    return {d_tries, d_data, d_allocator};
}

static std::tuple<cuda_Trie *, cuda_Data *, FastAllocator *> _buildOnHost(const BinSequencePack &pack) {
    // ------------------------------
    // Prepare objects
    // ------------------------------

    /* split sequences into `num_threads` buckets */
    const auto [num_threads, prefix_mask, power_of_2] = GetBatchSplitData();

    /* prepare allocator */
    BigChunkAllocator allocator(kDefaultAllocChunkSize, num_threads);

    /* prepare buckets */
    Buckets buckets(num_threads, num_threads, allocator);

    /* Prepare thread pool */
    ThreadPool pool(num_threads);
    std::barrier barrier(static_cast<ptrdiff_t>(num_threads));
    const size_t thread_job_size = pack.sequences.size() / num_threads;

    /* Prepare data */
    cuda_Data data{pack};

    /* initialize data to device transfer */
    cuda_Data *d_data = data.DumpToGPU();

    /* Prepare cuda alloca */
    FastAllocator cuda_allocator(kCpuThreadChunkSize, num_threads);

    /* Prepare tries */
    std::vector<cuda_Trie> tries;
    tries.resize(num_threads);

    // ---------------------------------
    // Process buckets in parallel
    // ---------------------------------

    /* Run threads */
    pool.RunThreads([&](const uint32_t thread_idx) {
        const size_t job_start = thread_idx * thread_job_size;
        const size_t job_end = thread_idx == num_threads - 1
                                   ? pack.sequences.size()
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

            std::cout << "Bucket stats:\n";
            for (size_t b_idx = 0; b_idx < num_threads; ++b_idx) {
                std::cout << "Bucket " << b_idx << " size: " << buckets.GetBucketSize(b_idx) << '\n';
            }
        }

        /* wait for bucket merge */
        barrier.arrive_and_wait();

        /* fill the tries */
        auto &trie = tries[thread_idx];

        while (!buckets.IsEmpty(thread_idx)) {
            const auto seq_idx = buckets.PopBucket(thread_idx);
            trie.Insert<false>(thread_idx, cuda_allocator, seq_idx, power_of_2, data);
        }
    });

    /*  wait for trie build end */
    pool.Wait();

    // ------------------------------
    // Merge tries together
    // ------------------------------

    /* merge tries */
    cuda_Trie final_trie{};
    final_trie.MergeByPrefixHost(cuda_allocator, data, tries, power_of_2);

    if (GlobalConfig.WriteDotFiles) {
        const bool result = final_trie.DumpToDotFile(cuda_allocator, data, "/tmp/trie.dot", "TRIE");
        assert(result);
    }

    // ---------------------------------------
    // Initialize transfer to GPU memory
    // ---------------------------------------

    /* transfer data to GPU */
    cuda_Trie *d_trie = final_trie.DumpToGpu();
    FastAllocator *d_allocator = cuda_allocator.DumpToGPU();

    /* wait to finish transfer */
    CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(g_cudaGlobalConf->asyncStream));

    // ------------------------------
    // Cleanup
    // ------------------------------

    data.DeallocHost();
    cuda_allocator.DeallocHost();

    return {d_trie, d_data, d_allocator};
}

// ------------------------------
// GPU interface functions
// ------------------------------

std::tuple<cuda_Trie *, cuda_Data *, FastAllocator *> InitHamming(const BinSequencePack &pack) {
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
                                                              FastAllocator *d_allocator) {
    // ------------------------------
    // Mem init
    // ------------------------------

    /* calculate management data */
    const ThreadMgr mgr{};
    const auto mgr_data = mgr.PrepareSearchData();

    /* prepare allocator for solutions */
    auto [d_sol, d_mem_block] = cuda_Solution::DumpToGPU(mgr_data.num_solutions);

    // ------------------------------
    // Processing
    // ------------------------------

    /* find all pairs */
    FindAllHamming1Pairs<<<mgr_data.num_blocks, mgr_data.num_threads_per_block, 0, g_cudaGlobalConf->asyncStream>>>(
        d_trie, d_allocator, d_data, d_sol);

    // ------------------------------
    // Mem back transfer
    // ------------------------------

    /* Process solutions */
    std::vector<std::tuple<uint32_t, uint32_t> > solutions;
    std::vector<uint32_t> h_sol;
    h_sol.resize(cuda_Solution::GetMemBlockSize(mgr_data.num_solutions));
    CUDA_ASSERT_SUCCESS(
        cudaMemcpyAsync(h_sol.data(), d_mem_block, cuda_Solution::GetMemBlockSize(mgr_data.num_solutions),
            cudaMemcpyDeviceToHost, g_cudaGlobalConf->asyncStream));

    CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(g_cudaGlobalConf->asyncStream));

    for (size_t idx = 0; idx < mgr_data.num_solutions; ++idx) {
        const uint32_t idx1 = h_sol[idx * 2];
        const uint32_t idx2 = h_sol[idx * 2 + 1];

        if (idx1 == UINT32_MAX || idx2 == UINT32_MAX) {
            /* Solution are terminated by UINT32_MAX */
            break;
        }

        solutions.emplace_back(idx1, idx2);
    }

    // ------------------------------
    // Cleanup
    // ------------------------------

    /* cleanup solutions */
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_mem_block, g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_sol, g_cudaGlobalConf->asyncStream));

    /* cleanup data */
    cuda_Data::DeallocGPU(d_data);

    /* cleanup allocator */
    FastAllocator::DeallocGPU(d_allocator);

    /* cleanup trie */
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_trie, g_cudaGlobalConf->asyncStream));

    return solutions;
}

std::vector<std::tuple<uint32_t, uint32_t> > FindHamming1PairsCUDA(const BinSequencePack &pack) {
    const auto t_init_start = std::chrono::high_resolution_clock::now();
    const auto [d_trie, d_data, d_alloca] = InitHamming(pack);
    const auto t_init_end = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent on TRIE build and memory transfer: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_init_end - t_init_start).count() << "ms" <<
            std::endl;

    const auto t_search_start = std::chrono::high_resolution_clock::now();
    const auto result = Hamming1Distance(d_trie, d_data, d_alloca);
    const auto t_search_end = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent on search: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_search_end - t_search_start).count() << "ms" <<
            std::endl;

    return result;
}

void TestCUDATrieCpuBuild(const BinSequencePack &pack) {
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto [d_trie, d_data, d_alloca] = _buildOnHost(pack);
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent on TRIE build and memory transfer: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    _testTrie(pack.sequences.size(), d_trie, d_data, d_alloca);
}

void TestCUDATrieGPUBuild(const BinSequencePack &pack) {
    const ThreadMgr mgr{};

    const auto mgr_data = mgr.PrepareTrieBuildData(pack, true);

    const auto t_trie_build_start = std::chrono::high_resolution_clock::now();
    const auto [d_trie, d_data, d_alloca] = _buildOnDevice(mgr_data, pack);
    const auto t_trie_build_end = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent on TRIE build and memory transfer: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_trie_build_end - t_trie_build_start).count()
            << "ms" << std::endl;

    _testTrie(pack.sequences.size(), d_trie, d_data, d_alloca);
}
