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
#include <set>
#include <cstring>

const char *Generator::GeneratorNames[kMaxNumGenerators]{
    "random_gen",
    "ensure_sol",
    "fixed",
    "ensure_sol_defaults",
};

Generator::GeneratorFuncT Generator::GeneratorFuncs[kMaxNumGenerators]{
    &Generator::GenerateRandomData,
    &Generator::GenerateEnsureSolution,
    &Generator::_GeneratedFixed,
    &Generator::_GenerateEnsureSolutionDefaults,
};

const char *Generator::GeneratorDesc[kMaxNumGenerators]{
    "Generates random binary sequences of varying lengths",
    "Generates sequences with guaranteed minimum number of solution pairs",
    "Returns a fixed set of predefined binary sequences",
    "Generates sequences with 10000 solution pairs using minimum parameters",
};

size_t Generator::NumGenerators = 4;

BinSequencePack Generator::GenerateData(const char *generator_name) {
    for (size_t idx = 0; idx < NumGenerators; ++idx) {
        if (std::string(GeneratorNames[idx]) == std::string(generator_name)) {
            const auto generator_func = GeneratorFuncs[idx];

            std::cout << "Generating sequences..." << std::endl;
            BinSequencePack data = (this->*generator_func)();

            std::cout << "Solving generated sequences..." << std::endl;
            data.solution.reserve(data.sequences.size());
            CalculateHammingDistancesNaive(data.sequences, data.solution);
            data.solution.shrink_to_fit();

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

std::vector<std::string> Generator::GetGeneratorDesc() {
    std::vector<std::string> out{};

    for (size_t idx = 0; idx < NumGenerators; ++idx) {
        out.emplace_back(GeneratorDesc[idx]);
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
    return GenerateRandomData(params);
}

BinSequencePack Generator::GenerateRandomData(const GeneratorParams &params) {
    BinSequencePack out{};
    out.sequences.reserve(params.num_sequences);

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

            out.sequences.emplace_back(seq_length);
            auto &sequence = out.sequences.back();
            --seq_to_gen;
            m.unlock();

            for (size_t bit_idx = 0; bit_idx < seq_length; ++bit_idx) {
                sequence.SetBit(bit_idx, gen() % 2);
            }
        }
    });

    pool.Wait();

    return out;
}

BinSequencePack Generator::GenerateEnsureSolution() {
    std::cout << "Provide minimal number of solutions to generate: " << std::endl;
    size_t min_solutions;
    std::cin >> min_solutions;

    const auto params = GetGeneratorParams();

    return _GenerateEnsureSolution(min_solutions, params);
}

BinSequencePack Generator::_GenerateEnsureSolution(size_t min_solutions, const GeneratorParams &params) {
    auto data = GenerateRandomData(params);

    std::set<size_t> used{};
    std::mt19937_64 gen(std::random_device{}());

    while (min_solutions > 0) {
        size_t left = gen() % data.sequences.size();
        const size_t right = gen() % data.sequences.size();

        if (left == right) {
            continue;
        }

        if (used.contains(left) && used.contains(right)) {
            continue;
        }

        if (used.contains(left) || used.contains(right)) {
            left = used.contains(left) ? right : left;
        }

        used.insert(left);
        used.insert(right);
        --min_solutions;

        data.sequences[left] = data.sequences[right];

        const size_t random_bit = gen() % data.sequences[left].GetSizeBits();
        data.sequences[left].SetBit(random_bit, !data.sequences[left].GetBit(random_bit));
    }

    return data;
}

BinSequencePack Generator::_GenerateEnsureSolutionDefaults() {
    GeneratorParams params;
    params.min_length = kMinSequenceLength;
    params.max_length = kMinSequenceLength;
    params.num_sequences = kMinNumSequences;

    return _GenerateEnsureSolution(10000, params);
}

BinSequencePack Generator::_GeneratedFixed() {
    static constexpr const char *kFixedSequence[]{
        "1000111110",
        "1100111110",
        "1010111110",
        "1001111110",
        "1000111010",
        "1001101100",
        "1001100110",
    };

    BinSequencePack data{};
    data.sequences.reserve(std::size(kFixedSequence));

    for (const char *seq: kFixedSequence) {
        const size_t len = std::strlen(seq);
        data.sequences.emplace_back(len);
        auto &sequence = data.sequences.back();

        for (size_t idx = 0; idx < len; ++idx) {
            sequence.SetBit(idx, seq[idx] == '1');
        }
    }

    return data;
}
