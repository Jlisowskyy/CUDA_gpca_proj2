/* internal includes */
#include <cuda_trie.cuh>

// ------------------------------
// implementations
// ------------------------------

cuda_Trie *cuda_Trie::DumpToGpu() const {
    cuda_Trie *d_trie;
    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_trie, sizeof(cuda_Trie), g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(
        cudaMemcpyAsync(d_trie, this, sizeof(cuda_Trie), cudaMemcpyHostToDevice, g_cudaGlobalConf->asyncStream));

    return d_trie;
}

void cuda_Trie::MergeByPrefixHost(FastAllocator &allocator, const cuda_Data &data, std::vector<cuda_Trie> &tries,
    const uint32_t prefix_len) {
    static const auto ExtractBit = [](const size_t item, const size_t idx) {
        return (item >> idx) & 1;
    };

    _root_idx = AllocateNode<false>(allocator, 0);

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
                allocator[*node_idx].next[value] = AllocateNode<false>(allocator, 0);
            }

            node_idx = &allocator[*node_idx].next[value];
        }

        *node_idx = root_idx;
    }
}

std::string cuda_Trie::DumpToDot(const FastAllocator &allocator, const cuda_Data &data,
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

bool cuda_Trie::DumpToDotFile(const FastAllocator &allocator, const cuda_Data &data, const std::string &filename,
    const std::string &graph_name) const {
    std::ofstream file(filename);
    if (!file.is_open()) return false;

    file << DumpToDot(allocator, data, graph_name);
    return true;
}
