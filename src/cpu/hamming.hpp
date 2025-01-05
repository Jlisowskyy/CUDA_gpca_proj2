#ifndef HAMMING_HPP
#define HAMMING_HPP

#include <data.hpp>

void CalculateHammingDistancesSingleThreadNaive(const std::vector<BinSequence> &sequences,
                                                size_t offset,
                                                std::vector<std::pair<size_t, size_t> > &out);

void CalculateHammingDistancesSingleThreadTrie(const std::vector<BinSequence> &sequences,
                                               size_t offset,
                                               std::vector<std::pair<size_t, size_t> > &out);

void CalculateHammingDistancesNaive(BinSequencePack &data);

void CalculateHammingDistancesTrie(BinSequencePack &data);

#endif //HAMMING_HPP
