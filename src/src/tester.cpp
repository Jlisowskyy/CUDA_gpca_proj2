#include <stdexcept>
#include <tester.hpp>

const char *Tester::TestNames[kMaxNumTests]{};
TestFuncT Tester::TestFuncs[kMaxNumTests]{};
size_t Tester::NumTests{};

void Tester::RunTest(const char *test_name, const BinSequencePack &bin_sequence_pack) {
    for (size_t idx = 0; idx < NumTests; ++idx) {
        if (TestNames[idx] == test_name) {
            const auto test_func = TestFuncs[idx];
            test_func(bin_sequence_pack);

            return;
        }
    }

    throw std::runtime_error("Test not found");
}

std::vector<std::string> Tester::GetTestNames() {
    std::vector<std::string> out{};

    for (size_t idx = 0; idx < NumTests; ++idx) {
        out.emplace_back(TestNames[idx]);
    }

    return out;
}
