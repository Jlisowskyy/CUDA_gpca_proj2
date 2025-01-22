/* internal includes */
#include <tester.hpp>
#include <hamming.hpp>
#include <allocators.hpp>
#include <trie.hpp>
#include <thread_pool.hpp>

/* external includes */
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <set>
#include <array>

/* Cuda includes */
#include <barrier>
#include <hamming.cuh>
#include <cuda_tests.cuh>
#include <defines.hpp>

const char *Tester::TestNames[kMaxNumTests]{
    "cpu_single_naive",
    "cpu_naive",
    "cpu_single_trie",
    "cpu_trie",
    "gpu",
    "test_malloc",
    "test_alloc",
    "test_cuda_data",
    "test_trie_build_cpu",
    "test_trie_build_gpu",
    "test_batch_split",
};

Tester::TestFuncT Tester::TestFuncs[kMaxNumTests]{
    &Tester::TestCpuSingleNaive_,
    &Tester::TestCpuNaive_,
    &Tester::TestCpuSingleTrie_,
    &Tester::TestCpuTrie_,
    &Tester::TestGPU_,
    &Tester::TestMalloc_,
    &Tester::TestAlloc_,
    &Tester::TestCudaData_,
    &Tester::TestCpuTrieBuild_,
    &Tester::TestGPUTrieBuild_,
    &Tester::TestBatchSplit_,
};

size_t Tester::NumTests = 11;

void Tester::RunTests(const std::vector<const char *> &test_names, const BinSequencePack &bin_sequence_pack) {
    for (const auto &test_name: test_names) {
        std::cout << std::string(80, '=') << std::endl;
        RunTest_(test_name, bin_sequence_pack);
        std::cout << std::string(80, '=') << std::endl;
    }
}

std::vector<std::string> Tester::GetTestNames() {
    std::vector<std::string> out{};

    for (size_t idx = 0; idx < NumTests; ++idx) {
        out.emplace_back(TestNames[idx]);
    }

    return out;
}

void Tester::TestCpuSingleNaive_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuSingleNaive:" << std::endl;

    ProcessSingleTest_(
        bin_sequence_pack,
        [](const BinSequencePack &bin_sequence_pack, std::vector<std::pair<size_t, size_t> > &out) {
            for (size_t idx = 0; idx < bin_sequence_pack.sequences.size(); ++idx) {
                CalculateHammingDistancesSingleThreadNaive(bin_sequence_pack.sequences, idx, out);
            }
        }
    );
}

void Tester::TestCpuNaive_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuNaive:" << std::endl;

    ProcessSingleTest_(
        bin_sequence_pack,
        [](const BinSequencePack &bin_sequence_pack, std::vector<std::pair<size_t, size_t> > &out) {
            CalculateHammingDistancesNaive(bin_sequence_pack.sequences, out);
        }
    );
}

void Tester::TestCpuSingleTrie_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuSingleTrie:" << std::endl;

    ProcessSingleTest_(
        bin_sequence_pack,
        [](const BinSequencePack &bin_sequence_pack, std::vector<std::pair<size_t, size_t> > &out) {
            CalculateHammingDistancesSingleThreadTrie(bin_sequence_pack.sequences, out);
        }
    );
}

void Tester::TestCpuTrie_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuTrie:" << std::endl;

    ProcessSingleTest_(
        bin_sequence_pack,
        [](const BinSequencePack &bin_sequence_pack, std::vector<std::pair<size_t, size_t> > &out) {
            CalculateHammingDistancesTrie(bin_sequence_pack.sequences, out);
        }
    );
}

void Tester::TestGPU_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestGPU" << std::endl;

    ProcessSingleTest_(
        bin_sequence_pack,
        [](const BinSequencePack &bin_sequence_pack, std::vector<std::pair<size_t, size_t> > &out) {
            auto result = FindHamming1PairsCUDA(bin_sequence_pack);

            for (const auto [i1, i2]: result) {
                out.emplace_back(i1, i2);
            }
        }
    );
}

static void FreeVectors(std::vector<char *> &mems) {
    for (const char *ptr: mems) {
        delete ptr;
    }
}

void Tester::TestMalloc_([[maybe_unused]] const BinSequencePack &bin_sequence_pack) {
    std::cout << "Testing malloc...\n" << std::endl;

    static constexpr size_t kNumAllocs = 1'000'000;
    static constexpr size_t kNumThreads = 20;

    std::vector<char *> mems;
    mems.reserve(kNumAllocs);

    const auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t idx = 0; idx < kNumAllocs; ++idx) {
        mems.emplace_back(new char[64]);
    }
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent using single thread: "
            << std::chrono::duration<double, std::milli>(t2 - t1).count() << "ms" << std::endl;

    FreeVectors(mems);

    std::vector<std::vector<char *> > mems1(kNumThreads);
    for (size_t idx = 0; idx < kNumThreads; ++idx) {
        mems1[idx].reserve(kNumAllocs / kNumThreads);
    }

    const auto t3 = std::chrono::high_resolution_clock::now();

    ThreadPool pool(kNumThreads);
    pool.RunThreads([&](const uint32_t idx) {
        std::vector<char *> &mem = mems1[idx];

        for (size_t i = 0; i < kNumAllocs / kNumThreads; ++i) {
            mem.emplace_back(new char[64]);
        }
    });
    pool.Wait();

    const auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent using " << kNumThreads << " threads: "
            << std::chrono::duration<double, std::milli>(t4 - t3).count() << "ms" << std::endl;

    for (size_t idx = 0; idx < kNumThreads; ++idx) {
        FreeVectors(mems1[idx]);
    }
}

void Tester::TestAlloc_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "Testing allocation using custom allocator...\n" << std::endl;

    static constexpr size_t kNumAllocs = 1'000'000;
    static size_t kNumThreads = std::thread::hardware_concurrency();

    /* prepare allocator */
    BigChunkAllocator allocator(kDefaultAllocChunkSize, 1);
    std::vector<void *> mems{};

    /* run first test */
    const auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t idx = 0; idx < kNumAllocs; ++idx) {
        mems.emplace_back(reinterpret_cast<void *>(allocator.Alloc<std::array<char, 64> >(0)));
    }
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent using single thread: "
            << std::chrono::duration<double, std::milli>(t2 - t1).count() << "ms" << std::endl;

    /* prepare for second test */
    BigChunkAllocator allocator1(kDefaultAllocChunkSize, kNumThreads);
    std::vector<std::vector<void *> > mems1(kNumThreads);
    for (size_t idx = 0; idx < kNumThreads; ++idx) {
        mems1[idx].reserve(kNumAllocs / kNumThreads);
    }

    /* run second test */
    const auto t3 = std::chrono::high_resolution_clock::now();

    ThreadPool pool(kNumThreads);
    pool.RunThreads([&](const uint32_t idx) {
        std::vector<void *> &mem = mems1[idx];

        for (size_t i = 0; i < kNumAllocs / kNumThreads; ++i) {
            mem.emplace_back(reinterpret_cast<void *>(allocator1.Alloc<std::array<char, 64> >(idx)));
        }
    });
    pool.Wait();

    const auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent using " << kNumThreads << " threads: "
            << std::chrono::duration<double, std::milli>(t4 - t3).count() << "ms" << std::endl;
}

void Tester::TestCudaData_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "Testing CUDA data" << std::endl;

    TestCudaData(bin_sequence_pack);
}

static void _verifyTrie(const Trie &trie, const std::vector<BinSequence> &sequences) {
    uint64_t num_errors{};
    for (size_t idx = 0; idx < sequences.size(); ++idx) {
        const bool result = trie.Search(idx);
        num_errors += !result;

        if (!result) {
            std::cout << "[ERROR] Sequence not found: " << idx << std::endl;
        }
    }

    if (num_errors == 0) {
        std::cout << "[SUCCESS] All sequences found" << std::endl;
    } else {
        std::cout << "[ERROR] " << num_errors << " sequences not found" << std::endl;
    }
}

void Tester::TestCpuTrieBuild_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "Testing CPU trie build" << std::endl;

    Trie trie(bin_sequence_pack.sequences);
    BuildTrieParallel(trie, bin_sequence_pack.sequences);
    _verifyTrie(trie, bin_sequence_pack.sequences);

    Trie trie1(bin_sequence_pack.sequences);
    BuildTrieSingleThread(trie1, bin_sequence_pack.sequences);
    _verifyTrie(trie1, bin_sequence_pack.sequences);
}

void Tester::TestGPUTrieBuild_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "Testing GPU trie build" << std::endl;

    TestCUDATrieCpuBuild(bin_sequence_pack);
    // TestCUDATrieGPUBuild(bin_sequence_pack);
}

static void _splitBatchMultithreaded(const BinSequencePack &bin_sequence_pack) {
    const auto [num_threads, prefix_mask, _] = GetBatchSplitData();

    /* prepare allocator */
    BigChunkAllocator allocator(kDefaultAllocChunkSize, num_threads);

    /* prepare buckets */
    Buckets buckets(num_threads, num_threads, allocator);

    /* Prepare thread pool */
    ThreadPool pool(num_threads);

    const size_t thread_job_size = bin_sequence_pack.sequences.size() / num_threads;

    /* Run threads */
    pool.RunThreads([&](const uint32_t thread_idx) {
        const size_t job_start = thread_idx * thread_job_size;
        const size_t job_end = thread_idx == num_threads - 1
                                   ? bin_sequence_pack.sequences.size()
                                   : (thread_idx + 1) * thread_job_size;

        /* Bucket sorting */
        for (size_t seq_idx = job_start; seq_idx < job_end; ++seq_idx) {
            const size_t key = bin_sequence_pack.sequences[seq_idx].GetWord(0) & prefix_mask;
            buckets.PushToBucket(thread_idx, key, seq_idx);
        }
    });
    pool.Wait();

    buckets.MergeBuckets();
    for (size_t bucket_idx = 0; bucket_idx < num_threads; ++bucket_idx) {
        std::cout << "Bucket " << bucket_idx << " size: " << buckets.GetBucketSize(bucket_idx) << '\n';
    }
    std::cout << std::endl;
}

static void _splitBatchSingleThread(const BinSequencePack &bin_sequence_pack) {
    const auto [num_threads, prefix_mask, _] = GetBatchSplitData();

    std::vector<std::vector<uint32_t> > buckets(num_threads);
    const size_t avg_bucket_size = bin_sequence_pack.sequences.size() / num_threads;
    for (auto &bucket: buckets) {
        buckets.reserve(avg_bucket_size);
    }

    for (size_t seq_idx = 0; seq_idx < bin_sequence_pack.sequences.size(); ++seq_idx) {
        const size_t key = bin_sequence_pack.sequences[seq_idx].GetWord(0) & prefix_mask;
        buckets[key].emplace_back(seq_idx);
    }

    std::cout << "Bucket status:\n";
    for (size_t idx = 0; idx < num_threads; ++idx) {
        std::cout << "Bucket " << idx << " size: " << buckets[idx].size() << '\n';
    }
    std::cout << std::endl;
}

void Tester::TestBatchSplit_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "Testing batch split...\n" << std::endl;

    const auto t_multithread_start = std::chrono::high_resolution_clock::now();
    _splitBatchMultithreaded(bin_sequence_pack);
    const auto t_multithread_end = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent when utilizing many threads: "
            << std::chrono::duration<double, std::milli>(t_multithread_end - t_multithread_start).count() << "ms" <<
            std::endl;

    const auto t_single_thread_start = std::chrono::high_resolution_clock::now();
    _splitBatchSingleThread(bin_sequence_pack);
    const auto t_single_thread_end = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent when utilizing single thread: "
            << std::chrono::duration<double, std::milli>(t_single_thread_end - t_single_thread_start).count() << "ms" <<
            std::endl;
}

void Tester::RunTest_(const char *test_name, const BinSequencePack &bin_sequence_pack) {
    for (size_t idx = 0; idx < NumTests; ++idx) {
        if (std::string(TestNames[idx]) == std::string(test_name)) {
            const auto test_func = TestFuncs[idx];
            (this->*test_func)(bin_sequence_pack);

            return;
        }
    }

    throw std::runtime_error("Test not found");
}

void Tester::VerifySolution_(const BinSequencePack &bin_sequence_pack,
                             const std::vector<std::pair<size_t, size_t> > &out) {
    uint64_t num_errors{};

    std::set<std::pair<size_t, size_t> > correct_set{
        bin_sequence_pack.solution.begin(), bin_sequence_pack.solution.end()
    };

    for (const auto &pair: out) {
        if (correct_set.contains(pair)) {
            correct_set.erase(pair);
        } else if (correct_set.contains({pair.second, pair.first})) {
            correct_set.erase({pair.second, pair.first});
        } else {
            ++num_errors;
            std::cout << "[ERROR] Generated additional pair: " << pair.first << " " << pair.second << std::endl;
        }
    }

    if (!correct_set.empty()) {
        for (const auto &[fst, snd]: correct_set) {
            std::cout << "[ERROR] Missed pair: " << fst << " " << snd << std::endl;
            ++num_errors;
        }
    }

    if (num_errors == 0) {
        std::cout << "[SUCCESS] All pairs are correct" << std::endl;
    } else {
        std::cout << "[ERROR] " << num_errors << " errors found" << std::endl;
    }
}
