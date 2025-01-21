#include <global_conf.cuh>
#include <defines.cuh>

cuda_GlobalConf *g_cudaGlobalConf{};

void cuda_InitGlobalConf() {
    g_cudaGlobalConf = new cuda_GlobalConf{};

    CUDA_ASSERT_SUCCESS(cudaStreamCreate(&g_cudaGlobalConf->asyncStream));
}

void cuda_DestroyGlobalConf() {
    CUDA_ASSERT_SUCCESS(cudaStreamDestroy(g_cudaGlobalConf->asyncStream));

    delete g_cudaGlobalConf;
    g_cudaGlobalConf = nullptr;
}
