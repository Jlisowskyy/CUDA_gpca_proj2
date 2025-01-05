/* internal includes */
#include <hamming.hpp>
#include <thread_pool.hpp>

/* external includes */
#include <stdexcept>
#include <thread>
#include <mutex>

void CalculateHammingDistancesSingleThreadNaive(const std::vector<BinSequence> &sequences,
                                                const size_t sequence_idx,
                                                std::vector<std::pair<size_t, size_t> > &out) {
    const auto &sequence = sequences[sequence_idx];

    for (size_t idx = 0; idx < sequences.size(); ++idx) {
        if (idx == sequence_idx) {
            continue;
        }

        const auto &other_sequence = sequences[idx];
        size_t distance = std::abs(
            static_cast<int64_t>(other_sequence.GetSizeBits()) - static_cast<int64_t>(sequence.GetSizeBits()));

        if (distance > 1) {
            continue;
        }

        const size_t range = std::min(other_sequence.GetSizeBits(), sequence.GetSizeBits());
        for (size_t bit_idx = 0; bit_idx < range; ++bit_idx) {
            if (sequence.GetBit(bit_idx) != other_sequence.GetBit(bit_idx)) {
                ++distance;
            }

            if (distance > 1) {
                break;
            }
        }

        if (distance == 1) {
            out.emplace_back(idx, distance);
        }
    }
}

void CalculateHammingDistancesSingleThreadTrie(const std::vector<BinSequence> &sequences,
                                               size_t sequence_idx,
                                               std::vector<std::pair<size_t, size_t> > &out) {
    throw std::runtime_error("Not implemented");
}

void CalculateHammingDistancesNaive(BinSequencePack &data) {
    ThreadPool thread_pool(std::thread::hardware_concurrency());

    std::mutex m{};
    thread_pool.RunThreads([&data, &m](const uint32_t idx) {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::pair<size_t, size_t> > out{};

        for (size_t seq_idx = idx; seq_idx < data.sequences.size(); seq_idx += num_threads) {
            CalculateHammingDistancesSingleThreadNaive(data.sequences, seq_idx, out);
        }

        m.lock();
        for (const auto &pair: out) {
            data.solution.emplace_back(pair);
        }
        m.unlock();
    });
    thread_pool.Wait();
}

void CalculateHammingDistancesTrie(const BinSequencePack &data) {
    throw std::runtime_error("Not implemented");
}
