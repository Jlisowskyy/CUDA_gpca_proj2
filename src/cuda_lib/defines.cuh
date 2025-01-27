#ifndef DEFINES_CUH
#define DEFINES_CUH

void AssertSuccess(cudaError_t error, const char *file, int line);

bool TraceError(cudaError_t error, const char *file, int line);

#define CUDA_ASSERT_SUCCESS(err) AssertSuccess(err, __FILE__, __LINE__)
#define CUDA_TRACE_ERROR(err) TraceError(err, __FILE__, __LINE__)

#define HYBRID __host__ __device__

#define FORCE_INLINE __forceinline__

#define FAST_CALL_ALWAYS FORCE_INLINE HYBRID

#define FAST_DCALL_ALWAYS __forceinline__ __device__

template<typename T>
static constexpr bool IsPowerOfTwo(T x) {
    if constexpr (-static_cast<T>(1) < static_cast<T>(1)) {
        if (x < 0) { return false; }
    }

    return x && (!(x & (x - 1)));
}


#endif //DEFINES_CUH
