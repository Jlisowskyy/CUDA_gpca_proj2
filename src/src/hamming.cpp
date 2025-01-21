/* internal includes */
#include <hamming.hpp>
#include <thread_pool.hpp>
#include <trie.hpp>

/* external includes */
#include <stdexcept>
#include <thread>
#include <mutex>
#include <iostream>

// ------------------------------
// Naive solution
// ------------------------------

void CalculateHammingDistancesSingleThreadNaive(const std::vector<BinSequence> &sequences,
                                                const size_t offset,
                                                std::vector<std::pair<size_t, size_t> > &out) {
    const auto &sequence = sequences[offset];

    for (size_t idx = offset + 1; idx < sequences.size(); ++idx) {
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
            out.emplace_back(idx, offset);
        }
    }
}

void CalculateHammingDistancesNaive(const std::vector<BinSequence> &sequences,
                                    std::vector<std::pair<size_t, size_t> > &out) {
    ThreadPool thread_pool(std::thread::hardware_concurrency());

    std::mutex m{};
    thread_pool.RunThreads([&sequences, &out, &m](const uint32_t idx) {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::pair<size_t, size_t> > pairs{};

        for (size_t seq_idx = idx; seq_idx < sequences.size(); seq_idx += num_threads) {
            CalculateHammingDistancesSingleThreadNaive(sequences, seq_idx, pairs);
        }

        m.lock();
        for (const auto &pair: pairs) {
            out.emplace_back(pair);
        }
        m.unlock();
    });
    thread_pool.Wait();
}

// ------------------------------
// Trie solution
// ------------------------------


void CalculateHammingDistancesSingleThreadTrie(const std::vector<BinSequence> &sequences,
                                               std::vector<std::pair<size_t, size_t> > &out) {
    Trie trie(sequences);

    const auto build_start = std::chrono::high_resolution_clock::now();
    BuildTrieSingleThread(trie, sequences);
    const auto build_end = std::chrono::high_resolution_clock::now();

    const auto find_start = std::chrono::high_resolution_clock::now();
    for (size_t idx = 0; idx < sequences.size(); ++idx) {
        trie.FindPairs(idx, out);
    }
    const auto find_end = std::chrono::high_resolution_clock::now();

    std::cout << "Total time spent on building trie: " << std::chrono::duration<double,
        std::milli>(build_end - build_start).count() << "ms" << std::endl;

    std::cout << "Total time spent on finding pairs: " << std::chrono::duration<double,
        std::milli>(find_end - find_start).count() << "ms" << std::endl;
}

void CalculateHammingDistancesTrie(const std::vector<BinSequence> &sequences,
                                   std::vector<std::pair<size_t, size_t> > &out) {
    Trie trie(sequences);
    std::vector<std::vector<std::pair<size_t, size_t> > > thread_out(std::thread::hardware_concurrency());

    const auto build_start = std::chrono::high_resolution_clock::now();
    BuildTrieParallel(trie, sequences);
    const auto build_end = std::chrono::high_resolution_clock::now();

    const auto find_start = std::chrono::high_resolution_clock::now();
    ThreadPool thread_pool(std::thread::hardware_concurrency());
    thread_pool.RunThreads([&trie, &out, &sequences, &thread_out](const uint32_t idx) {
        const size_t num_threads = std::thread::hardware_concurrency();

        for (size_t seq_idx = idx; seq_idx < sequences.size(); seq_idx += num_threads) {
            trie.FindPairs(seq_idx, thread_out[idx]);
        }
    });

    thread_pool.Wait();

    for (auto &thread_pair: thread_out) {
        for (const auto &pair: thread_pair) {
            out.emplace_back(pair);
        }
    }
    const auto find_end = std::chrono::high_resolution_clock::now();

    std::cout << "Total time spent on building trie: " << std::chrono::duration<double,
        std::milli>(build_end - build_start).count() << "ms" << std::endl;
    std::cout << "Total time spent on finding pairs: " << std::chrono::duration<double,
        std::milli>(find_end - find_start).count() << "ms" << std::endl;
}
