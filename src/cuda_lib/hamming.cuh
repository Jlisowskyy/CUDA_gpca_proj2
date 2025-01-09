#ifndef HAMMING_CUH
#define HAMMING_CUH

/* internal includes */
#include <data.cuh>
#include <cuda_trie.cuh>
#include <data.hpp>

/* external includes */
#include <cstdint>
#include <tuple>
#include <vector>

std::tuple<cuda_Trie *, cuda_Data *, cuda_Allocator *> InitHamming(const BinSequencePack &pack);

std::vector<std::tuple<uint32_t, uint32_t> > Hamming1Distance(cuda_Trie *d_trie, cuda_Data *d_data,
                                                              cuda_Allocator *d_allocator);

#endif //HAMMING_CUH
