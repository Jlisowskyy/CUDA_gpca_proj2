#ifndef GLOBAL_CONF_CUH
#define GLOBAL_CONF_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>

struct cuda_GlobalConf {
    cudaStream_t asyncStream{};
};
#else
struct cuda_GlobalConf;
#endif // __CUDACC__

extern cuda_GlobalConf* g_cudaGlobalConf;

void cuda_InitGlobalConf();
void cuda_DestroyGlobalConf();

#endif //GLOBAL_CONF_CUH
