/* internal includes */
#include <tester.hpp>

/* external includes */
#include <stdexcept>
#include <iostream>

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
    std::cout << "TestCpuSingleNaive" << std::endl;
}

void Tester::TestCpuNaive_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuNaive" << std::endl;
}

void Tester::TestCpuSingleTrie_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuSingleTrie" << std::endl;
}

void Tester::TestCpuTrie_(const BinSequencePack &bin_sequence_pack) {
    std::cout << "TestCpuTrie" << std::endl;
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
