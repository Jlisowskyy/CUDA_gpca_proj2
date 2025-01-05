/* internal includes */
#include <generator.hpp>
#include <thread_pool.hpp>
#include <defines.hpp>
#include <hamming.hpp>

/* external includes */
#include <stdexcept>
#include <iostream>
#include <mutex>
#include <random>


const char *Generator::GeneratorNames[kMaxNumGenerators]{
    "random_gen"
};

Generator::GeneratorFuncT Generator::GeneratorFuncs[kMaxNumGenerators]{
    &Generator::GenerateRandomData
};

size_t Generator::NumGenerators = 1;

BinSequencePack Generator::GenerateData(const char *generator_name) {
    for (size_t idx = 0; idx < NumGenerators; ++idx) {
        if (std::string(GeneratorNames[idx]) == std::string(generator_name)) {
            const auto generator_func = GeneratorFuncs[idx];

            BinSequencePack data = (this->*generator_func)();
            CalculateHammingDistancesNaive(data);
            return data;
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

Generator::GeneratorParams Generator::GetGeneratorParams() {
    GeneratorParams out{};

    std::cout << "Enter min length: " << std::endl;
    std::cin >> out.min_length;

    if (out.min_length < kMinSequenceLength) {
        throw std::runtime_error("Min length is too small");
    }

    std::cout << "Enter max length: " << std::endl;
    std::cin >> out.max_length;

    if (out.max_length < out.min_length) {
        throw std::runtime_error("Max length is too small");
    }

    std::cout << "Enter number of sequences: " << std::endl;
    std::cin >> out.num_sequences;

    if (out.num_sequences < kMinNumSequences) {
        throw std::runtime_error("Number of sequences is too small");
    }

    return out;
}


BinSequencePack Generator::GenerateRandomData() {
    const auto params = GetGeneratorParams();

    BinSequencePack out{};

    std::mutex m{};
    ThreadPool pool(std::thread::hardware_concurrency());

    size_t seq_to_gen = params.num_sequences;
    pool.RunThreads([&]([[maybe_unused]] uint32_t idx) {
        std::mt19937_64 gen(std::random_device{}());
        const uint64_t len_dist = (params.max_length - params.min_length) + 1;

        while (true) {
            const uint64_t seq_length = gen() % len_dist + params.min_length;

            m.lock();

            if (seq_to_gen == 0) {
                m.unlock();
                break;
            }

            auto sequence = out.sequences.emplace_back(seq_length);
            m.unlock();

            for (size_t bit_idx = 0; bit_idx < seq_length; ++bit_idx) {
                sequence.SetBit(bit_idx, gen() % 2);
            }
        }
    });

    pool.Wait();

    return out;
}
