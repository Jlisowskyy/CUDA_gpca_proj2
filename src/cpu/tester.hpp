#ifndef TESTER_HPP
#define TESTER_HPP

/* internal includes */
#include <data.hpp>

/* external includes */
#include <vector>
#include <string>
#include <iostream>
#include <chrono>

class Tester {
    using TestFuncT = void (Tester::*)(const BinSequencePack &bin_sequence_pack);

    // ------------------------------
    // Class creation
    // ------------------------------
public:
    Tester() = default;

    ~Tester() = default;

    // ------------------------------
    // class methods
    // ------------------------------

    void RunTests(const std::vector<const char *> &test_names, const BinSequencePack &bin_sequence_pack);

    static std::vector<std::string> GetTestNames();

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:
    void TestCpuSingleNaive_(const BinSequencePack &bin_sequence_pack);

    void TestCpuNaive_(const BinSequencePack &bin_sequence_pack);

    void TestCpuSingleTrie_(const BinSequencePack &bin_sequence_pack);

    void TestCpuTrie_(const BinSequencePack &bin_sequence_pack);

    void TestGPU_(const BinSequencePack &bin_sequence_pack);

    void TestTrieBuild_(const BinSequencePack &bin_sequence_pack);

    void RunTest_(const char *test_name, const BinSequencePack &bin_sequence_pack);

    void VerifySolution_(const BinSequencePack &bin_sequence_pack, const std::vector<std::pair<size_t, size_t> > &out);

    template<class FuncT>
    void ProcessSingleTest_(const BinSequencePack &bin_sequence_pack, FuncT test_func) {
        std::vector<std::pair<size_t, size_t> > out{};

        const auto start = std::chrono::high_resolution_clock::now();

       test_func(bin_sequence_pack, out);

        const auto end = std::chrono::high_resolution_clock::now();

        const double timeMs = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Total time spent: " << timeMs << "ms" << std::endl;
        std::cout << "Average time spent on a single sequence: " << timeMs / bin_sequence_pack.sequences.size() << "ms"
                <<
                std::endl;

        VerifySolution_(bin_sequence_pack, out);
    }

    // ------------------------------
    // class fields
    // ------------------------------

    static constexpr size_t kMaxNumTests = 32;

    static const char *TestNames[kMaxNumTests];
    static TestFuncT TestFuncs[kMaxNumTests];
    static size_t NumTests;
};


#endif //TESTER_HPP
