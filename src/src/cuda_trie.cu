/* internal includes */
#include <cuda_trie.cuh>

bool cuda_Trie::Insert(const uint32_t t_idx, cuda_Allocator &allocator, const uint32_t seq_idx,
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
        *node_idx = allocator.AllocateNode(t_idx);
        allocator[*node_idx].seq_idx = seq_idx;
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

    /* we found node with assigned sequence */
    const uint32_t old_node_idx = *node_idx;
    const auto old_seq = data[old_node_idx];
    *node_idx = allocator.AllocateNode(t_idx);

    while (bit_idx < sequence.GetSequenceLength() &&
           bit_idx < old_seq.GetSequenceLength() &&
           sequence.GetBit(bit_idx) == old_seq.GetBit(bit_idx)) {
        /* add nodes until we reach the difference or there is no more bits to compare */
        const bool bit = sequence.GetBit(bit_idx++);

        allocator[*node_idx].next[bit] = allocator.AllocateNode(t_idx);
        node_idx = &allocator[*node_idx].next[bit];
    }

    if (bit_idx == sequence.GetSequenceLength() && bit_idx == old_seq.GetSequenceLength()) {
        /* we reached the end of both sequences and no difference was found assign on of them and exit */
        allocator[*node_idx].seq_idx = seq_idx;

        return true;
    }

    if (bit_idx == old_seq.GetSequenceLength()) {
        /* we reached the end of the old sequence */

        allocator[*node_idx].seq_idx = allocator[old_node_idx].seq_idx;

        const uint32_t new_node_idx = allocator.AllocateNode(t_idx);
        allocator[new_node_idx].seq_idx = seq_idx;
        allocator[*node_idx].next[sequence.GetBit(bit_idx)] = new_node_idx;

        return true;
    }

    if (bit_idx == sequence.GetSequenceLength()) {
        /* we reached the end of the new sequence */

        allocator[*node_idx].seq_idx = seq_idx;

        const uint32_t new_node_idx = allocator.AllocateNode(t_idx);
        allocator[new_node_idx].seq_idx = allocator[old_node_idx].seq_idx;
        allocator[*node_idx].next[old_seq.GetBit(bit_idx)] = new_node_idx;

        return true;
    }

    /* we reached the difference */
    allocator[*node_idx].next[old_seq.GetBit(bit_idx)] = old_node_idx;

    const uint32_t new_node_idx = allocator.AllocateNode(t_idx);
    allocator[*node_idx].next[sequence.GetBit(bit_idx)] = new_node_idx;
    allocator[new_node_idx].seq_idx = seq_idx;

    return true;
}

cuda_Trie *cuda_Trie::DumpToGpu() const {
    cuda_Trie *d_trie;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_trie, sizeof(cuda_Trie)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(d_trie, this, sizeof(cuda_Trie), cudaMemcpyHostToDevice));

    return d_trie;
}

void cuda_Trie::MergeByPrefixHost(cuda_Allocator &allocator, const cuda_Data &data, std::vector<cuda_Trie> &tries,
                                  const uint32_t prefix_len) {
    static const auto ExtractBit = [](const size_t item, const size_t idx) {
        return (item >> idx) & 1;
    };

    _root_idx = allocator.AllocateNode(0);

    for (uint32_t idx = 0; idx < tries.size(); ++idx) {
        cuda_Trie &trie = tries[idx];
        const uint32_t root_idx = trie._root_idx;
        trie._root_idx = 0;

        if (!root_idx) {
            continue;
        }

        uint32_t *node_idx = &_root_idx;
        uint32_t bit = 0;
        while (bit < prefix_len) {
            const bool value = ExtractBit(idx, bit++);

            if (!allocator[*node_idx].next[value]) {
                allocator[*node_idx].next[value] = allocator.AllocateNode(0);
            }

            node_idx = &allocator[*node_idx].next[value];
        }

        *node_idx = root_idx;
    }
}
