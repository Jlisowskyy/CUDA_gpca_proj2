#ifndef GENERATOR_HPP
#define GENERATOR_HPP

/* internal includes */
#include <data.hpp>

/* external includes */
#include <string>
#include <vector>

class Generator {
    using GeneratorFuncT = BinSequencePack (Generator::*)();

    struct GeneratorParams {
        uint32_t min_length;
        uint32_t max_length;
        uint32_t num_sequences;
    };

    // ------------------------------
    // Class creation
    // ------------------------------
public:
    Generator() = default;

    ~Generator() = default;

    // ------------------------------
    // class methods
    // ------------------------------

    [[nodiscard]] BinSequencePack GenerateData(const char *generator_name);

    [[nodiscard]] static std::vector<std::string> GetGeneratorNames();

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:

    [[nodiscard]] static GeneratorParams GetGeneratorParams();

    [[nodiscard]] BinSequencePack GenerateRandomData();

    // ------------------------------
    // class fields
    // ------------------------------

    static constexpr size_t kMaxNumGenerators = 32;

    static const char *GeneratorNames[kMaxNumGenerators];
    static GeneratorFuncT GeneratorFuncs[kMaxNumGenerators];
    static size_t NumGenerators;
};

#endif //GENERATOR_HPP
