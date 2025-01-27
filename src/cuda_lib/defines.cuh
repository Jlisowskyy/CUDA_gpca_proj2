#ifndef DEFINES_CUH
#define DEFINES_CUH

#include <cinttypes>

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

static constexpr uint32_t pow2(const uint32_t pow) {
    if (pow == 0) {
        return 1;
    }

    return 2 * pow2(pow - 1);
}

static constexpr uint32_t GenMask(const uint32_t size) {
    uint32_t mask{};

    for (uint32_t i = 0; i < size; ++i) {
        mask |= static_cast<uint32_t>(1) << i;
    }

    return mask;
}

#endif //DEFINES_CUH
