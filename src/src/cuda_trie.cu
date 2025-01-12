/* internal includes */
#include <cuda_trie.cuh>

bool cuda_Trie::Insert(cuda_Allocator &allocator, uint32_t seq_idx, uint32_t bit_idx, const cuda_Data &data) {
}

void cuda_Trie::FindPairs(uint32_t seq_idx, const cuda_Data &data, cuda_Solution &solutions) {
}

cuda_Trie * cuda_Trie::DumpToGpu() {
}

void cuda_Trie::MergeByPrefixHost(const std::vector<cuda_Trie> &tries, uint32_t prefix_len) {
}
