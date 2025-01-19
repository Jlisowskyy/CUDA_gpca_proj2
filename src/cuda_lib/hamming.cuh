#ifndef HAMMING_CUH
#define HAMMING_CUH

/* external includes */
#include <data.hpp>
#include <cstdint>
#include <tuple>
#include <vector>

std::vector<std::tuple<uint32_t, uint32_t> > FindHamming1PairsCUDA(const BinSequencePack &pack);

void TestCUDATrieCpuBuild(const BinSequencePack &pack);

void TestCUDATrieGPUBuild(const BinSequencePack &pack);

#endif //HAMMING_CUH
