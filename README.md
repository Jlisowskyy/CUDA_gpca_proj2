# CUDA-Hamming-Pairs

A CUDA-accelerated algorithm for finding all pairs of binary sequences with a Hamming distance of 1. This implementation achieves **O(nl²)** efficiency, where:
- **n** = number of sequences
- **l** = length of the longest binary sequence

## Features
- **Efficient Parallelized Algorithm**
  - Custom algorithm optimized for performance.
  - Achieves **O(nl²)** complexity.
- **Optimized TRIE Data Structure**
  - **Parallel CPU TRIE Build**
  - **CUDA-accelerated TRIE Build**
  - **Parallel TRIE Traversal for CPU & GPU**
- **Thread-Safe, Synchronization-Free Allocators**
  - Enables high-speed memory allocation.
  - Boosts execution efficiency for both CPU and GPU.
- **Fully Parallelized Execution**
  - Sequence batching and processing executed in parallel.
  
## Installation
### Prerequisites
- CUDA-enabled GPU
- NVIDIA CUDA Toolkit
- C++ compiler preferably gcc
- CMake (for build automation)

### Build Instructions
```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j$(nproc)
```

## Usage

Application contains built in CLI, simply type --help for more information.

## Example results

``` sh
================================================================================
TestGPU

Time spent on TRIE build and memory transfer: 822ms
Time spent on search: 147ms
Total time spent: 970.968ms
Average time spent on a single sequence: 0.000970968ms

[SUCCESS] All pairs are correct
================================================================================

================================================================================
TestCpuTrie:

Total time spent on building trie: 277.413ms
Total time spent on finding pairs: 445.002ms
Total time spent: 732.225ms
Average time spent on a single sequence: 0.000732225ms

[SUCCESS] All pairs are correct
================================================================================

================================================================================
TestCpuNaive:

Total time spent: 37068.8ms
Average time spent on a single sequence: 0.0370688ms
Tested function returned: 100001 pairs

[SUCCESS] All pairs are correct
================================================================================
```

## Performance Highlights
- **Highly optimized memory allocation** with synchronization-free techniques.
- **Significant speedups** compared to purely CPU-based approaches.
- **Full utilization of parallel processing** across both CPU and GPU.

## License
This project is licensed under the MIT License.
