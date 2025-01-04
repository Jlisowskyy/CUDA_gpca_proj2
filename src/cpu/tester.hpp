#ifndef TESTER_HPP
#define TESTER_HPP

/* internal includes */
#include <data.hpp>

/* external includes */
#include <vector>
#include <string>

using TestFuncT = void (*)(const BinSequencePack& bin_sequence_pack);

class Tester {
    // ------------------------------
    // Class creation
    // ------------------------------
public:
    Tester() = default;

    ~Tester() = default;

    // ------------------------------
    // class methods
    // ------------------------------

    void RunTest(const char *test_name, const BinSequencePack& bin_sequence_pack);

    static std::vector<std::string> GetTestNames();

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:
    // ------------------------------
    // class fields
    // ------------------------------

    static constexpr size_t kMaxNumTests = 32;

    static const char *TestNames[kMaxNumTests];
    static TestFuncT TestFuncs[kMaxNumTests];
    static size_t NumTests;
};


#endif //TESTER_HPP
