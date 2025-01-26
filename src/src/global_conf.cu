#include <global_conf.cuh>
#include <defines.cuh>
#include <allocators.cuh>

#include <iostream>

cuda_GlobalConf *g_cudaGlobalConf{};

void cuda_InitGlobalConf() {
    g_cudaGlobalConf = new cuda_GlobalConf{};

    CUDA_ASSERT_SUCCESS(cudaStreamCreate(&g_cudaGlobalConf->asyncStream));

    // /* query free memory */

    // size_t free, total;
    // CUDA_ASSERT_SUCCESS(cudaMemGetInfo(&free, &total));
    //
    // /* set heap size to 80% of free memory */
    // const size_t heap_size = std::max(static_cast<size_t>(0), (8 * free / 10) - kPageSize);
    // CUDA_ASSERT_SUCCESS(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    //
    // std::cout << "Set CUDA heap size to " << heap_size << " bytes, MB: " << heap_size / 1024 / 1024 << std::endl;
}

void cuda_DestroyGlobalConf() {
    CUDA_ASSERT_SUCCESS(cudaStreamDestroy(g_cudaGlobalConf->asyncStream));

    delete g_cudaGlobalConf;
    g_cudaGlobalConf = nullptr;
}
