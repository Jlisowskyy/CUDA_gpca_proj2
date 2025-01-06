/* internal includes */
#include <tester.hpp>
#include <hamming.hpp>

/* external includes */
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <set>
#include <thread_pool.hpp>
#include <trie.hpp>

const char *Tester::TestNames[kMaxNumTests]{
    "cpu_single_naive",
    "cpu_naive",
    "cpu_single_trie",
    "cpu_trie",
    "gpu",
    "trie_build",
    "test_malloc",
};

Tester::TestFuncT Tester::TestFuncs[kMaxNumTests]{
    &Tester::TestCpuSingleNaive_,
    &Tester::TestCpuNaive_,
    &Tester::TestCpuSingleTrie_,
    &Tester::TestCpuTrie_,
    &Tester::TestGPU_,
    &Tester::TestTrieBuild_,
    &Tester::TestMalloc_,
};

size_t Tester::NumTests = 7;

void Tester::RunTests(const std::vector<const char *> &test_names, const BinSequencePack &bin_sequence_pack) {
    for (const auto &test_name: test_names) {
        RunTest_(test_name, bin_sequence_pack);
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
}

void Tester::TestTrieBuild_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestTrieBuild" << std::endl;

    Trie trie1(bin_sequence_pack.sequences);
    BuildTrieSingleThread(trie1, bin_sequence_pack.sequences);

    Trie trie2(bin_sequence_pack.sequences);
    BuildTrieParallel(trie2, bin_sequence_pack.sequences);

    if (trie1 != trie2) {
        std::cout << "[ERROR] Tries are not equal" << std::endl;
    }
}

static void FreeVectors(std::vector<char *> &mems) {
    for (const char *ptr: mems) {
        delete ptr;
    }
}

void Tester::TestMalloc_([[maybe_unused]] const BinSequencePack &bin_sequence_pack) {
    static constexpr size_t kNumAllocs = 1'000'000;

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

    std::vector<std::vector<char *> > mems1(20);
    for (size_t idx = 0; idx < 20; ++idx) {
        mems1[idx].reserve(kNumAllocs / 20);
    }

    const auto t3 = std::chrono::high_resolution_clock::now();

    ThreadPool pool(20);
    pool.RunThreads([&](const uint32_t idx) {
        std::vector<char *> &mem = mems1[idx];

        for (size_t i = 0; i < kNumAllocs / 20; ++i) {
            mem.emplace_back(new char[64]);
        }
    });
    pool.Wait();

    const auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent using 20 threads: "
            << std::chrono::duration<double, std::milli>(t4 - t3).count() << "ms" << std::endl;

    for (size_t idx = 0; idx < 20; ++idx) {
        FreeVectors(mems1[idx]);
    }
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
    std::set<std::pair<size_t, size_t> > correct_set{
        bin_sequence_pack.solution.begin(), bin_sequence_pack.solution.end()
    };

    for (const auto &pair: out) {
        if (correct_set.contains(pair)) {
            correct_set.erase(pair);
        } else if (correct_set.contains({pair.second, pair.first})) {
            correct_set.erase({pair.second, pair.first});
        } else {
            std::cout << "[ERROR] Generated additional pair: " << pair.first << " " << pair.second << std::endl;
        }
    }

    if (!correct_set.empty()) {
        for (const auto &[fst, snd]: correct_set) {
            std::cout << "[ERROR] Missed pair: " << fst << " " << snd << std::endl;
        }
    }
}
