/* internal includes */
#include <cuda_trie.cuh>

// ------------------------------
// implementations
// ------------------------------

cuda_Trie *cuda_Trie::DumpToGpu() const {
    cuda_Trie *d_trie;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_trie, sizeof(cuda_Trie)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(d_trie, this, sizeof(cuda_Trie), cudaMemcpyHostToDevice));

    return d_trie;
}
