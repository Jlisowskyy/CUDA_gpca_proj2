/* internal includes */
#include <hamming.cuh>
#include <cuda_trie.cuh>
#include <data.cuh>

std::tuple<cuda_Trie *, cuda_Data *, cuda_Allocator *> InitHamming(const BinSequencePack &pack) {
    /* convert pack to cuda_Data */
    const cuda_Data data{pack};
    cuda_Data *d_data = data.DumpToGPU();

    /* prepare allocator */
    // TODO

    /* build trie */
    // TODO
}

std::vector<std::tuple<uint32_t, uint32_t> > Hamming1Distance(cuda_Trie *d_trie, cuda_Data *d_data,
                                                              cuda_Allocator *allocator) {
    /* TODO: replace with some logic */
    static constexpr uint32_t kMaxSolutions = 1'000'00;

    /* prepare allocator for solutions */
    auto [d_sol, d_mem_block] = cuda_Solution::DumpToGPU(kMaxSolutions);

    /* find all pairs */
    // TODO

    /* Process solutions */
    std::vector<std::tuple<uint32_t, uint32_t> > solutions;
    uint32_t *h_sol;
    CUDA_ASSERT_SUCCESS(cudaMemcpy(&h_sol, d_mem_block, cuda_Solution::GetMemBlockSize(kMaxSolutions),
        cudaMemcpyDeviceToHost));

    for (size_t idx = 0; idx < kMaxSolutions; ++idx) {
        const uint32_t idx1 = h_sol[idx * 2];
        const uint32_t idx2 = h_sol[idx * 2 + 1];

        if (idx1 == INT_MAX || idx2 == INT_MAX) {
            /* Solution are terminated by INT_MAX */
            break;
        }

        solutions.emplace_back(idx1, idx2);
    }

    /* cleanup solutions */
    CUDA_ASSERT_SUCCESS(cudaFree(d_mem_block));
    CUDA_ASSERT_SUCCESS(cudaFree(d_sol));

    /* cleanup data */
    uint32_t *d_data_ptr = cuda_Data::GetDataPtrHost(d_data);
    CUDA_ASSERT_SUCCESS(cudaFree(d_data_ptr));
    CUDA_ASSERT_SUCCESS(cudaFree(d_data));

    /* cleanup trie */
    // TODO

    return solutions;
}
