#ifndef HAMMING_HPP
#define HAMMING_HPP

#include <data.hpp>

void CalculateHammingDistancesSingleThreadNaive(const std::vector<BinSequence> &sequences,
                                                size_t offset,
                                                std::vector<std::pair<size_t, size_t> > &out);

void CalculateHammingDistancesSingleThreadTrie(const std::vector<BinSequence> &sequences,
                                               std::vector<std::pair<size_t, size_t> > &out);

void CalculateHammingDistancesNaive(const std::vector<BinSequence> &sequences,
                                    std::vector<std::pair<size_t, size_t> > &out);

void CalculateHammingDistancesTrie(const std::vector<BinSequence> &sequences,
                                   std::vector<std::pair<size_t, size_t> > &out);

#endif //HAMMING_HPP
