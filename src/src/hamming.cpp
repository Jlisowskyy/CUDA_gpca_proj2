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

std::vector<std::pair<size_t, size_t> > CalculateHammingDistancesSingleThreadNaive(
    const std::vector<BinSequence> &sequences,
    const size_t offset,
    std::vector<std::pair<size_t, size_t> > &out) {
    const auto &sequence = sequences[offset];
    std::vector<std::pair<size_t, size_t> > reps{};

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

        if (distance == 0) {
            reps.emplace_back(idx, offset);
        }
    }

    return reps;
}

void CalculateHammingDistancesNaive(const std::vector<BinSequence> &sequences,
                                    std::vector<std::pair<size_t, size_t> > &out) {
    ThreadPool thread_pool(std::thread::hardware_concurrency());

    std::vector<std::vector<size_t> > repeats(sequences.size());

    std::mutex m{};
    thread_pool.RunThreads([&](const uint32_t idx) {
        const size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::pair<size_t, size_t> > pairs{};
        std::vector<std::pair<size_t, size_t> > repeat_pairs{};


        for (size_t seq_idx = idx; seq_idx < sequences.size(); seq_idx += num_threads) {
            const auto reps = CalculateHammingDistancesSingleThreadNaive(sequences, seq_idx, pairs);

            for (const auto &rep: reps) {
                repeat_pairs.push_back(rep);
            }
        }

        m.lock();
        for (const auto &pair: pairs) {
            out.emplace_back(pair);
        }

        for (const auto &[l, r]: repeat_pairs) {
            repeats[l].push_back(r);
            repeats[r].push_back(l);
        }

        m.unlock();
    });
    thread_pool.Wait();

    FilterOutRepeats(repeats, out);
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

void FilterOutRepeats(const std::vector<std::vector<size_t> > &repeats,
                      std::vector<std::pair<size_t, size_t> > &results) {
    std::vector<std::pair<size_t, size_t> > final_results{};
    std::vector<int8_t> should_be_removed(repeats.size(), 0);

    for (size_t idx = 0; idx < repeats.size(); ++idx) {
        const auto &reps = repeats[idx];

        if (reps.empty()) {
            should_be_removed[idx] = -1;
            continue;
        }

        if (should_be_removed[idx] == 1) {
            continue;
        }

        for (const auto &rep: reps) {
            should_be_removed[rep] = 1;
        }
        should_be_removed[idx] = -1;
    }

    while (!results.empty()) {
        const auto [l, r] = results.back();
        results.pop_back();

        if (should_be_removed[l] == 1 || should_be_removed[r] == 1) {
            continue;
        }

        final_results.emplace_back(l, r);
    }

    results = std::move(final_results);
}
