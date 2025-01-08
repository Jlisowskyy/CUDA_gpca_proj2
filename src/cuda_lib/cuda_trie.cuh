#ifndef CUDA_TRIE_CUH
#define CUDA_TRIE_CUH

/* internal includes */
#include <defines.cuh>
#include <data.cuh>

/* external includes */
#include <cstdint>

class cuda_Trie {
    // ------------------------------
    // Class inner types
    // ------------------------------

    static constexpr uint32_t kNextCount = 2;

    struct Node_ {
        HYBRID explicit Node_(const uint32_t idx) : seq_idx(idx) {
        }

        uint32_t next[kNextCount]{};
        uint32_t seq_idx;
    };

    // ------------------------------
    // Class creation
    // ------------------------------
public:
    cuda_Trie() = default;

    ~cuda_Trie() = default;

    // ------------------------------
    // class interaction
    // ------------------------------

    __device__ bool Insert(uint32_t seq_idx, uint32_t bit_idx, const cuda_Data& data);
    __device__ void FindPairs(uint32_t seq_idx, const cuda_Data& data, cuda_Solution& solutions);

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
