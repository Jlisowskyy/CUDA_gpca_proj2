#ifndef HAMMING_HPP
#define HAMMING_HPP

#include <data.hpp>

/***
 * @returns repeated sequences
 */
std::vector<std::pair<size_t, size_t> > CalculateHammingDistancesSingleThreadNaive(
    const std::vector<BinSequence> &sequences,
    size_t offset,
    std::vector<std::pair<size_t, size_t> > &out);

/**
 * @note Repetitions are naturally filtered out during TRIE building
 */
void CalculateHammingDistancesSingleThreadTrie(const std::vector<BinSequence> &sequences,
                                               std::vector<std::pair<size_t, size_t> > &out);

/***
 * @returns filters out repeated sequences to match main implementation behavior (repeats are not included, only first occurrence)
 */
void CalculateHammingDistancesNaive(const std::vector<BinSequence> &sequences,
                                    std::vector<std::pair<size_t, size_t> > &out);

/**
 * @note Repetitions are naturally filtered out during TRIE building
 */
void CalculateHammingDistancesTrie(const std::vector<BinSequence> &sequences,
                                   std::vector<std::pair<size_t, size_t> > &out);

/**
 * @note Removes all repetitions from results to match main implementation behavior
 */
void FilterOutRepeats(const std::vector<std::vector<size_t>>& repeats,
                      std::vector<std::pair<size_t, size_t> > &results);

#endif //HAMMING_HPP
