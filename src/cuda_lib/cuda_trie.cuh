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

    HYBRID bool Insert(uint32_t t_idx, cuda_Allocator &allocator, uint32_t seq_idx, uint32_t start_bit_idx,
                       const cuda_Data &data);

    HYBRID bool Search(const cuda_Allocator &allocator, uint32_t seq_idx, const cuda_Data &data) const {
        const auto sequence = data[seq_idx];
        uint32_t node_idx = _root_idx;
        uint32_t bit_idx = 0;

        /* traverse existing tree or until we reach the end of the sequence */
        while (node_idx && (allocator[node_idx].next[0] || allocator[node_idx].next[1])
               && bit_idx < sequence.GetSequenceLength()) {
            node_idx = allocator[node_idx].next[sequence.GetBit(bit_idx++)];
        }

        return node_idx && allocator[node_idx].seq_idx == seq_idx;
    }

    __device__ void FindPairs(const uint32_t seq_idx, const cuda_Allocator &allocator, const cuda_Data &data,
                              cuda_Solution &solutions) const {
        const auto sequence = data[seq_idx];
        uint32_t bit_idx = 0;

        uint32_t node_idx = _root_idx;

        /* follow the path */
        while (node_idx && bit_idx < sequence.GetSequenceLength()
               && (allocator[node_idx].next[0] || allocator[node_idx].next[1])) {
            const bool bit = sequence.GetBit(bit_idx++);

            /* Try path with flipped bit */
            _tryToFindPair(allocator[node_idx].next[!bit], bit_idx, seq_idx, allocator, data, solutions);

            if (bit_idx == sequence.GetSequenceLength() && allocator[node_idx].seq_idx != UINT32_MAX
                && seq_idx < allocator[node_idx].seq_idx) {
                /* check if there is some shorter sequence that is a valid pair */
                solutions.PushSolution(seq_idx, allocator[node_idx].seq_idx);
            }

            node_idx = allocator[node_idx].next[bit];
        }

        if (!node_idx) {
            /* no path found */
            return;
        }

        if (bit_idx == sequence.GetSequenceLength()) {
            /* we reached the end of the sequence */

            for (const uint32_t idx: allocator[node_idx].next) {
                if (idx && allocator[idx].seq_idx != UINT32_MAX && seq_idx < allocator[idx].seq_idx) {
                    solutions.PushSolution(seq_idx, allocator[idx].seq_idx);
                }
            }
        }

        /* TODO: assert here */
    }

    [[nodiscard]] cuda_Trie *DumpToGpu() const;

    void MergeByPrefixHost(cuda_Allocator &allocator, const cuda_Data &data, std::vector<cuda_Trie> &tries,
                           uint32_t prefix_len);

    [[nodiscard]] std::string DumpToDot(const cuda_Allocator &allocator, const cuda_Data &data,
                                        const std::string &graph_name = "Trie") const;

    [[nodiscard]] bool DumpToDotFile(const cuda_Allocator &allocator, const cuda_Data &data,
                                     const std::string &filename, const std::string &graph_name = "Trie") const;

    // ------------------------------
    // private methods
    // ------------------------------
private:
    __device__ void _tryToFindPair(uint32_t node_idx, uint32_t bit_idx, const uint32_t seq_idx,
                                   const cuda_Allocator &allocator,
                                   const cuda_Data &data, cuda_Solution &solutions) const {
        /* Check if we have a valid node */
        if (!node_idx) {
            return;
        }

        /*  Use original sequence to follow path after the flipped bit */
        const auto sequence = data[seq_idx];

        /* Follow the path */
        while (node_idx && (allocator[node_idx].next[0] || allocator[node_idx].next[1])
               && bit_idx < sequence.GetSequenceLength()) {
            node_idx = allocator[node_idx].next[sequence.GetBit(bit_idx++)];
        }

        /* Check if we found a valid sequence */
        if (!node_idx) {
            return;
        }

        /* Check if remaining bits match after the single flip */
        if (const auto other_sequence = data[allocator[node_idx].seq_idx];
            seq_idx < allocator[node_idx].seq_idx && sequence.Compare(other_sequence, bit_idx)) {
            solutions.PushSolution(seq_idx, allocator[node_idx].seq_idx);
        }
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    uint32_t _root_idx{};
};

#endif // CUDA_TRIE_CUH
