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
