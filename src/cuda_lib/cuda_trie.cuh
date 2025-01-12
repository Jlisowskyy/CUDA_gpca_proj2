#ifndef CUDA_TRIE_CUH
#define CUDA_TRIE_CUH

/* internal includes */
#include <defines.cuh>
#include <data.cuh>

/* external includes */
#include <cstdint>

class cuda_Trie {
    // ------------------------------
    // Class creation
    // ------------------------------
public:
    cuda_Trie() = default;

    ~cuda_Trie() = default;

    // ------------------------------
    // class interaction
    // ------------------------------

    HYBRID bool Insert(cuda_Allocator &allocator, uint32_t seq_idx, uint32_t bit_idx, const cuda_Data &data);

    __device__ void FindPairs(uint32_t seq_idx, const cuda_Data &data, cuda_Solution &solutions);

    cuda_Trie *DumpToGpu();

    void MergeByPrefixHost(const std::vector<cuda_Trie>& tries, uint32_t prefix_len);

    // ------------------------------
    // private methods
    // ------------------------------
private:
    // ------------------------------
    // Class fields
    // ------------------------------

    uint32_t _root_idx{};
    cuda_Data *_data{};
    Node_ *_nodes{};
};

#endif // CUDA_TRIE_CUH
