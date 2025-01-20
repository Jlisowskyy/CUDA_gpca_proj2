/* internal includes */
#include <cuda_trie.cuh>

/* external includes */
#include <sstream>
#include <string>
#include <functional>
#include <fstream>

// ------------------------------
// static functions
// ------------------------------

[[nodiscard]] FAST_CALL_ALWAYS static uint32_t AllocateNode(cuda_Allocator& allocator, const uint32_t t_idx) {
    return allocator.AllocateNode(t_idx);
}

[[nodiscard]] FAST_CALL_ALWAYS static uint32_t AllocateNode(cuda_Allocator& allocator, const uint32_t t_idx, const uint32_t seq_idx) {
    const uint32_t node_idx = allocator.AllocateNode(t_idx);

    allocator[node_idx].seq_idx = seq_idx;
    return node_idx;
}

// ------------------------------
// implementations
// ------------------------------


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
        *node_idx = AllocateNode(allocator, t_idx);
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

    if (sequence.Compare(data[allocator[*node_idx].seq_idx], bit_idx)) {
        /* we found node with assigned sequence */
        return false;
    }

    /* we found node with assigned sequence */
    const uint32_t old_node_idx = *node_idx;
    const auto old_seq = data[allocator[old_node_idx].seq_idx];
    *node_idx = allocator.AllocateNode(t_idx);
    assert(allocator[old_node_idx].next[0] == 0 && allocator[old_node_idx].next[1] == 0);

    while (bit_idx < sequence.GetSequenceLength() &&
           bit_idx < old_seq.GetSequenceLength() &&
           sequence.GetBit(bit_idx) == old_seq.GetBit(bit_idx)) {
        /* add nodes until we reach the difference or there is no more bits to compare */
        const bool bit = sequence.GetBit(bit_idx++);

        allocator[*node_idx].next[bit] = AllocateNode(allocator, t_idx);
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
    allocator[*node_idx].next[sequence.GetBit(bit_idx)] = AllocateNode(allocator, t_idx, seq_idx);

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

    _root_idx = allocator.AllocateNode();

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
                allocator[*node_idx].next[value] = allocator.AllocateNode();
            }

            node_idx = &allocator[*node_idx].next[value];
        }

        *node_idx = root_idx;
    }
}

std::string cuda_Trie::DumpToDot(const cuda_Allocator &allocator, const cuda_Data &data,
                                 const std::string &graph_name) const {
    std::stringstream dot;

    dot << "digraph " << graph_name << " {\n";
    dot << "    node [shape=record];\n";

    auto getNodeName = [](const uint32_t idx) {
        return "node_" + std::to_string(idx);
    };

    std::function<void(uint32_t)> dumpNode = [&](uint32_t node_idx) {
        if (!node_idx) return;

        const auto &node = allocator[node_idx];

        std::string seq_label = (node.seq_idx != UINT32_MAX) ? "\\nseq=" + std::to_string(node.seq_idx) : "";

        dot << "    " << getNodeName(node_idx) << " [label=\""
                << node_idx << seq_label << "\"];\n";

        for (int i = 0; i < 2; ++i) {
            if (node.next[i]) {
                dot << "    " << getNodeName(node_idx) << " -> "
                        << getNodeName(node.next[i])
                        << " [label=\"" << i << "\"];\n";

                dumpNode(node.next[i]);
            }
        }
    };

    if (_root_idx) {
        dumpNode(_root_idx);
    }

    dot << "}\n";

    return dot.str();
}

bool cuda_Trie::DumpToDotFile(const cuda_Allocator &allocator, const cuda_Data &data, const std::string &filename,
                              const std::string &graph_name) const {
    std::ofstream file(filename);
    if (!file.is_open()) return false;

    file << DumpToDot(allocator, data, graph_name);
    return true;
}
