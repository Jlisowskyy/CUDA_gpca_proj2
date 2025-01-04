/* internal includes */
#include <generator.hpp>

/* external includes */
#include <stdexcept>

const char *Generator::GeneratorNames[kMaxNumGenerators]{};
GeneratorFuncT Generator::GeneratorFuncs[kMaxNumGenerators]{};
size_t Generator::NumGenerators{};

BinSequencePack Generator::GenerateData(const char *generator_name) {
    for (size_t idx = 0; idx < NumGenerators; ++idx) {
        if (GeneratorNames[idx] == generator_name) {
            const auto generator_func = GeneratorFuncs[idx];

            return generator_func();
        }
    }

    throw std::runtime_error("Generator not found");
}

std::vector<std::string> Generator::GetGeneratorNames() {
    std::vector<std::string> out{};

    for (size_t idx = 0; idx < NumGenerators; ++idx) {
        out.emplace_back(GeneratorNames[idx]);
    }

    return out;
}
