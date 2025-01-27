#ifndef CUDA_TRIE_CUH
#define CUDA_TRIE_CUH

/* internal includes */
#include <defines.cuh>
#include <data.cuh>
#include <allocators.cuh>

/* external includes */
#include <cstdint>
#include <fstream>
#include <sstream>
#include <functional>

// ------------------------------
// static functions
// ------------------------------

template<bool isGpu>
[[nodiscard]] FAST_CALL_ALWAYS static uint32_t AllocateNode(FastAllocator &allocator, const uint32_t t_idx,
                                                            const uint32_t seq_idx = UINT32_MAX) {
    const uint32_t node_idx = allocator.AllocateNode<isGpu>(t_idx);
    assert(node_idx != 0);

    allocator[node_idx].seq_idx = seq_idx;
    allocator[node_idx].next[0] = 0;
    allocator[node_idx].next[1] = 0;

    return node_idx;
}

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

    HYBRID void Reset() {
        _root_idx = 0;
    }

    template<bool isGpu>
    HYBRID bool Insert(const uint32_t t_idx,
                       FastAllocator &allocator,
                       const uint32_t seq_idx,
                       const uint32_t start_bit_idx,
                       const cuda_Data &data) {
        const auto sequence = data[seq_idx];
        uint32_t *node_idx = &_root_idx;

        if (start_bit_idx >= sequence.GetSequenceLength()) {
            return false;
        }

        uint32_t bit_idx = start_bit_idx;
        /* traverse existing tree or until we reach the end of the sequence */
        while (*node_idx && (allocator[*node_idx].next[0] || allocator[*node_idx].next[1])
               && bit_idx < sequence.GetSequenceLength()) {
            node_idx = &allocator[*node_idx].next[sequence.GetBit(bit_idx++)];
        }

        if (!*node_idx) {
            /* we reached the end of the tree */
            *node_idx = AllocateNode<isGpu>(allocator, t_idx, seq_idx);
            return true;
        }

        if (bit_idx == sequence.GetSequenceLength()) {
            /* we reached the end of the sequence */
            /* we are also sure that the p is not null */

            assert(allocator[*node_idx].seq_idx == UINT32_MAX && "DETECTED OVERWRITE");
            /* assign the sequence index to the node */
            allocator[*node_idx].seq_idx = seq_idx;

            return true;
        }

        if (sequence.Compare(data[allocator[*node_idx].seq_idx], bit_idx)) {
            /* we found node with assigned sequence */
            return false;
        }

        /* we found node with assigned sequence */
        const uint32_t old_node_idx = *node_idx;
        const auto old_seq = data[allocator[old_node_idx].seq_idx];
        *node_idx = AllocateNode<isGpu>(allocator, t_idx);
        assert(allocator[old_node_idx].next[0] == 0 && allocator[old_node_idx].next[1] == 0);

        while (bit_idx < sequence.GetSequenceLength() &&
               bit_idx < old_seq.GetSequenceLength() &&
               sequence.GetBit(bit_idx) == old_seq.GetBit(bit_idx)) {
            /* add nodes until we reach the difference or there is no more bits to compare */
            const bool bit = sequence.GetBit(bit_idx++);

            allocator[*node_idx].next[bit] = AllocateNode<isGpu>(allocator, t_idx);
            node_idx = &allocator[*node_idx].next[bit];
        }

        if (bit_idx == sequence.GetSequenceLength() && bit_idx == old_seq.GetSequenceLength()) {
            /* we reached the end of both sequences and no difference was found assign on of them and exit */
            allocator[*node_idx].seq_idx = seq_idx;

            return true;
        }

        if (bit_idx == old_seq.GetSequenceLength()) {
            /* we reached the end of the old sequence */
            assert(allocator[*node_idx].seq_idx == UINT32_MAX);

            allocator[*node_idx].seq_idx = allocator[old_node_idx].seq_idx;
            allocator[old_node_idx].seq_idx = seq_idx;
            allocator[*node_idx].next[sequence.GetBit(bit_idx)] = old_node_idx;

            return true;
        }

        if (bit_idx == sequence.GetSequenceLength()) {
            /* we reached the end of the new sequence */
            assert(allocator[*node_idx].seq_idx == UINT32_MAX);

            allocator[*node_idx].seq_idx = seq_idx;
            allocator[*node_idx].next[old_seq.GetBit(bit_idx)] = old_node_idx;

            return true;
        }

        /* we reached the difference */
        allocator[*node_idx].next[old_seq.GetBit(bit_idx)] = old_node_idx;
        allocator[*node_idx].next[sequence.GetBit(bit_idx)] = AllocateNode<isGpu>(allocator, t_idx, seq_idx);

        return true;
    }

    HYBRID bool Search(const FastAllocator &allocator, const uint32_t seq_idx, const cuda_Data &data) const {
        const auto sequence = data[seq_idx];
        uint32_t node_idx = _root_idx;
        uint32_t bit_idx = 0;

        /* traverse existing tree or until we reach the end of the sequence */
        while (node_idx && (allocator[node_idx].next[0] || allocator[node_idx].next[1])
               && bit_idx < sequence.GetSequenceLength()) {
            node_idx = allocator[node_idx].next[sequence.GetBit(bit_idx++)];
        }

        return node_idx && (allocator[node_idx].seq_idx == seq_idx ||
                            sequence.Compare(data[allocator[node_idx].seq_idx]));
    }

    __device__ void FindPairs(const uint32_t seq_idx, const FastAllocator &allocator, const cuda_Data &data,
                              cuda_Solution &solutions) const {
        const auto sequence = data[seq_idx];
        uint32_t bit_idx = 0;

        uint32_t node_idx = _root_idx;

        /* follow the path */
        while (node_idx && (allocator[node_idx].next[0] || allocator[node_idx].next[1])
               && bit_idx < sequence.GetSequenceLength()) {
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

        if (node_idx == 0) {
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

        assert(
            (node_idx != 0 && allocator[node_idx].seq_idx == seq_idx) || sequence.Compare(data[allocator[node_idx].
                seq_idx]));
    }


    [[nodiscard]] cuda_Trie *DumpToGpu() const;

    /* assume the other always goes to bit 1 */
    template<bool isGpu>
    __device__ void MergeWithOther(const uint32_t thread_idx, cuda_Trie &other, FastAllocator &allocator) {
        const uint32_t root0 = _root_idx;
        const uint32_t root1 = other._root_idx;

        _root_idx = AllocateNode<isGpu>(allocator, thread_idx);
        allocator[_root_idx].next[0] = root0;
        allocator[_root_idx].next[1] = root1;

        other._root_idx = 0;
    }

    void MergeByPrefixHost(FastAllocator &allocator, const cuda_Data &data, std::vector<cuda_Trie> &tries, uint32_t prefix_len);

    [[nodiscard]] std::string DumpToDot(const FastAllocator &allocator, const cuda_Data &data,
                                        const std::string &graph_name = "Trie") const;

    [[nodiscard]] bool DumpToDotFile(const FastAllocator &allocator, const cuda_Data &data,
                                     const std::string &filename, const std::string &graph_name = "Trie") const;

    // ------------------------------
    // private methods
    // ------------------------------
private:
    __device__ void _tryToFindPair(uint32_t node_idx, uint32_t bit_idx, const uint32_t seq_idx,
                                   const FastAllocator &allocator,
                                   const cuda_Data &data, cuda_Solution &solutions) const {
        /*  Use original sequence to follow path after the flipped bit */
        const auto sequence = data[seq_idx];

        /* Follow the path */
        while (node_idx != 0 && (allocator[node_idx].next[0] || allocator[node_idx].next[1])
               && bit_idx < sequence.GetSequenceLength()) {
            node_idx = allocator[node_idx].next[sequence.GetBit(bit_idx++)];
        }

        /* Check if we found a valid sequence */
        if (node_idx == 0) {
            return;
        }

        /* Check if remaining bits match after the single flip */
        const auto other_sequence = data[allocator[node_idx].seq_idx];
        if (seq_idx < allocator[node_idx].seq_idx && sequence.Compare(other_sequence, bit_idx)) {
            solutions.PushSolution(seq_idx, allocator[node_idx].seq_idx);
        }
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    uint32_t _root_idx{};
};

#endif // CUDA_TRIE_CUH
