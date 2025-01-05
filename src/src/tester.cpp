/* internal includes */
#include <tester.hpp>
#include <hamming.hpp>

/* external includes */
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <set>

const char *Tester::TestNames[kMaxNumTests]{
    "cpu_single_naive",
    "cpu_naive",
    "cpu_single_trie",
    "cpu_trie",
    "gpu",
};

Tester::TestFuncT Tester::TestFuncs[kMaxNumTests]{
    &Tester::TestCpuSingleNaive_,
    &Tester::TestCpuNaive_,
    &Tester::TestCpuSingleTrie_,
    &Tester::TestCpuTrie_,
    &Tester::TestGPU_,
};

size_t Tester::NumTests = 5;

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

    std::vector<std::pair<size_t, size_t> > out{};

    const auto start = std::chrono::high_resolution_clock::now();

    for (size_t idx = 0; idx < bin_sequence_pack.sequences.size(); ++idx) {
        CalculateHammingDistancesSingleThreadNaive(bin_sequence_pack.sequences, idx, out);
    }

    const auto end = std::chrono::high_resolution_clock::now();

    const double timeMs = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time spent: " << timeMs << "ms" << std::endl;
    std::cout << "Average time spent on a single sequence: " << timeMs / bin_sequence_pack.sequences.size() << "ms" <<
            std::endl;

    VerifySolution_(bin_sequence_pack, out);
}

void Tester::TestCpuNaive_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuNaive:" << std::endl;

    std::vector<std::pair<size_t, size_t> > out{};

    const auto start = std::chrono::high_resolution_clock::now();
    CalculateHammingDistancesNaive(bin_sequence_pack.sequences, out);
    const auto end = std::chrono::high_resolution_clock::now();

    const double timeMs = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time spent: " << timeMs << "ms" << std::endl;
    std::cout << "Average time spent on a single sequence: " << timeMs / bin_sequence_pack.sequences.size() << "ms" <<
            std::endl;

    VerifySolution_(bin_sequence_pack, out);
}

void Tester::TestCpuSingleTrie_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuSingleTrie:" << std::endl;
}

void Tester::TestCpuTrie_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuTrie:" << std::endl;
}

void Tester::TestGPU_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestGPU" << std::endl;
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
        for (const auto &pair: correct_set) {
            std::cout << "[ERROR] Missed pair: " << pair.first << " " << pair.second << std::endl;
        }
    }
}
