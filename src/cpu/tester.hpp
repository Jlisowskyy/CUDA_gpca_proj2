#ifndef TESTER_HPP
#define TESTER_HPP

/* internal includes */
#include <data.hpp>

/* external includes */
#include <vector>
#include <string>

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

    void RunTest_(const char* test_name, const BinSequencePack &bin_sequence_pack);

    void VerifySolution_(const BinSequencePack &bin_sequence_pack, const std::vector<std::pair<size_t, size_t> > &out);

    // ------------------------------
    // class fields
    // ------------------------------

    static constexpr size_t kMaxNumTests = 32;

    static const char *TestNames[kMaxNumTests];
    static TestFuncT TestFuncs[kMaxNumTests];
    static size_t NumTests;
};


#endif //TESTER_HPP
