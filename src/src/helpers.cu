/* internal includes */
#include <defines.cuh>

/* external includes */
#include <format>
#include <iostream>

void AssertSuccess(const cudaError_t error, const char *file, const int line) {
    TraceError(error, file, line);

    if (error != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
}

bool TraceError(const cudaError_t error, const char *file, const int line) {
    if (error != cudaSuccess) {
        std::cerr << std::format("CUDA Error at {}:{} - {}\n", file, line, cudaGetErrorString(error)) << std::endl;
    }

    return error != cudaSuccess;
}
