#ifndef ALLOCATORS_CUH
#define ALLOCATORS_CUH

#include <data.cuh>
#include <atomic>
#include <global_conf.cuh>
#include <cuda_runtime.h>
#include <cuda/atomic>

static constexpr uint64_t kKb = 1024;
static constexpr uint64_t kMb = 1024 * kKb;
static constexpr size_t kCpuThreadChunkSize = 16 * kMb;
static constexpr size_t kGpuThreadChunkSize = 8 * kKb;

static constexpr uint64_t kPageSize = kMb * 512;
static constexpr uint32_t kMaxPages = 64;

template<class T>
__global__ void CleanupHeap(T **pages, const uint32_t num_pages) {
    for (uint32_t page_idx = 0; page_idx < num_pages; ++page_idx) {
        free(pages[page_idx]);
    }
}

template<class T>
class alignas(128) BaseFastAllocator_ {
    // ------------------------------
    // inner constants
    // ------------------------------
    static constexpr uint64_t kPageSizeInTypeSize = kPageSize / sizeof(T);

    static_assert(IsPowerOfTwo(sizeof(T)));
    static_assert(IsPowerOfTwo(kPageSizeInTypeSize));

    static constexpr uint32_t kPageRemainder = kPageSizeInTypeSize - 1;
    static constexpr uint32_t kPageDivider = std::countr_zero(kPageSizeInTypeSize);

public:
    // ------------------------------
    // Type creation
    // ------------------------------

    BaseFastAllocator_(const size_t thread_chunk_size, const size_t num_threads,
                  const bool isCudaAlloc) : _thread_chunk_size_in_type(
                                                thread_chunk_size / sizeof(T)), _num_threads(num_threads),
                                            _is_cuda_alloc(isCudaAlloc) {
        /* verify provided data */
        assert(num_threads * thread_chunk_size < kPageSize);
        assert(thread_chunk_size % sizeof(T) == 0);
        assert(kPageSizeInTypeSize % _thread_chunk_size_in_type == 0);

        /* allocate memory for management data */
        _pages = new T *[kMaxPages]{};
        _thread_tops = new uint32_t[num_threads]{};
        _thread_bottoms = new uint32_t[num_threads];

        /* preallocate first page */
        _pages[0] = static_cast<T *>(malloc(sizeof(T) * kPageSizeInTypeSize));

        /* set thread bottoms */
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            _thread_bottoms[thread_idx] = thread_idx * _thread_chunk_size_in_type;
        }

        /* prepare allocator data */
        _last_page = 0;
        _last_page_offset = num_threads * _thread_chunk_size_in_type;

        /* ensure no null pointer is allocated */
        _thread_tops[0] = 1;

        assert(_last_page_offset < kPageSizeInTypeSize);
        assert(_last_page < kMaxPages);
    }

    // ------------------------------
    // Data path methods
    // ------------------------------

    [[nodiscard]] FAST_CALL_ALWAYS T &operator[](const uint32_t idx) {
        assert(idx < _last_page_offset);
        assert(_pages[idx / kPageSizeInTypeSize] != nullptr);
        assert(idx != 0);
        return _pages[idx >> kPageDivider][idx & kPageRemainder];
    }

    [[nodiscard]] FAST_CALL_ALWAYS const T &operator[](const uint32_t idx) const {
        assert(idx < _last_page_offset);
        assert(_pages[idx / kPageSizeInTypeSize] != nullptr);
        assert(idx != 0);
        return _pages[idx >> kPageDivider][idx & kPageRemainder];
    }

    template<bool isGpu = false>
    [[nodiscard]] FAST_CALL_ALWAYS uint32_t AllocateNode(const uint32_t t_idx) {
        /* get current thread top */
        uint32_t top = _thread_tops[t_idx]++;

        /* check if we need new chunk - lazy allocation */
        if (top == _thread_chunk_size_in_type) {
            /* get new chunk */
            if constexpr (isGpu) {
                _thread_bottoms[t_idx] = _AllocateNewChunkGpu();
            } else {
                _thread_bottoms[t_idx] = _AllocateNewChunkCpu();
            }

            /* already offset for new chunk */
            top = 0;
            _thread_tops[t_idx] = 1;
        }

        assert(_thread_tops[t_idx] <= _thread_chunk_size_in_type);
        assert(_thread_bottoms[t_idx] + top != 0);
        return _thread_bottoms[t_idx] + top;
    }

    // ------------------------------
    // Management methods
    // ------------------------------

    [[nodiscard]] FAST_CALL_ALWAYS size_t GetNumNodesAllocated() const {
        return _last_page_offset;
    }

    void DeallocHost() {
        for (uint32_t page_idx = 0; page_idx <= _last_page; ++page_idx) {
            free(_pages[page_idx]);
        }

        delete[] _pages;
        delete[] _thread_tops;
        delete[] _thread_bottoms;

        _pages = nullptr;
        _thread_tops = nullptr;
        _thread_bottoms = nullptr;
    }

    [[nodiscard]] BaseFastAllocator_ *DumpToGPU() const {
        /* allocate the object itself */
        BaseFastAllocator_ *d_allocator;
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_allocator, sizeof(BaseFastAllocator_), g_cudaGlobalConf->asyncStream));

        /* allocate thread tops */
        uint32_t *d_thread_tops;
        CUDA_ASSERT_SUCCESS(
            cudaMallocAsync(&d_thread_tops, _num_threads * sizeof(uint32_t), g_cudaGlobalConf->asyncStream));

        /* allocate thread bottoms */
        uint32_t *d_thread_bottoms;
        CUDA_ASSERT_SUCCESS(
            cudaMallocAsync(&d_thread_bottoms, _num_threads * sizeof(uint32_t), g_cudaGlobalConf->asyncStream));

        /* allocate pages */
        auto h_pages = new T *[kMaxPages]{};
        for (uint32_t page_idx = 0; page_idx <= _last_page; ++page_idx) {
            T *d_page;
            CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_page, kPageSize, g_cudaGlobalConf->asyncStream));
            h_pages[page_idx] = d_page;
        }

        /* allocate page table */
        T **d_pages;
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_pages, kMaxPages * sizeof(T *), g_cudaGlobalConf->asyncStream));

        /* copy allocator object */
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(d_allocator, this, sizeof(BaseFastAllocator_), cudaMemcpyHostToDevice, g_cudaGlobalConf->
                asyncStream));

        /* copy thread tops */
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(d_thread_tops, _thread_tops, _num_threads * sizeof(uint32_t), cudaMemcpyHostToDevice,
                g_cudaGlobalConf->asyncStream));

        /* copy thread bottoms */
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(d_thread_bottoms, _thread_bottoms, _num_threads * sizeof(uint32_t), cudaMemcpyHostToDevice,
                g_cudaGlobalConf->asyncStream));

        /* copy pages */
        for (uint32_t page_idx = 0; page_idx <= _last_page; ++page_idx) {
            CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(h_pages[page_idx], _pages[page_idx], kPageSize, cudaMemcpyHostToDevice,
                g_cudaGlobalConf->asyncStream));
        }

        /* copy page table */
        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(d_pages, h_pages, kMaxPages * sizeof(T *), cudaMemcpyHostToDevice,
            g_cudaGlobalConf->asyncStream));

        /* update object pointers */
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_allocator->_thread_tops, &d_thread_tops, sizeof(uint32_t *), cudaMemcpyHostToDevice,
                g_cudaGlobalConf->asyncStream));
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_allocator->_thread_bottoms, &d_thread_bottoms, sizeof(uint32_t *), cudaMemcpyHostToDevice
                , g_cudaGlobalConf->asyncStream));
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_allocator->_pages, &d_pages, sizeof(T **), cudaMemcpyHostToDevice, g_cudaGlobalConf->
                asyncStream));

        /* final sync */
        CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(g_cudaGlobalConf->asyncStream));

        /* free temporary data */
        delete[] h_pages;

        return d_allocator;
    }

    static void DeallocGPU(BaseFastAllocator_ *allocator) {
        /* prepare host page table */
        auto h_pages = new T *[kMaxPages]{};

        /* copy pointer to pages */
        T **d_pages;
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_pages, &allocator->_pages, sizeof(T **), cudaMemcpyDeviceToHost,
                g_cudaGlobalConf->asyncStream));

        /* copy page table */
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(h_pages, d_pages, kMaxPages * sizeof(T *), cudaMemcpyDeviceToHost,
                g_cudaGlobalConf->asyncStream));

        /* delete individual pages */
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(h_pages[0], g_cudaGlobalConf->asyncStream));

        bool is_cuda_alloc;
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&is_cuda_alloc, &allocator->_is_cuda_alloc, sizeof(bool), cudaMemcpyDeviceToHost,
                g_cudaGlobalConf->asyncStream));

        if (is_cuda_alloc) {
            /* cleanup heap */
            CleanupHeap<<<1, 1, 0, g_cudaGlobalConf->asyncStream>>>(d_pages + 1, kMaxPages - 1);

        } else {
            for (uint32_t page_idx = 1; page_idx < kMaxPages; ++page_idx) {
                if (h_pages[page_idx] != nullptr) {
                    CUDA_ASSERT_SUCCESS(cudaFreeAsync(h_pages[page_idx], g_cudaGlobalConf->asyncStream));
                }
            }
        }


        /* free page table */
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_pages, g_cudaGlobalConf->asyncStream));

        /* free thread tops */
        uint32_t *d_thread_tops;
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_thread_tops, &allocator->_thread_tops, sizeof(uint32_t *), cudaMemcpyDeviceToHost,
                g_cudaGlobalConf->asyncStream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_thread_tops, g_cudaGlobalConf->asyncStream));

        /* free thread bottoms */
        uint32_t *d_thread_bottoms;
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_thread_bottoms, &allocator->_thread_bottoms, sizeof(uint32_t *), cudaMemcpyDeviceToHost,
                g_cudaGlobalConf->asyncStream));
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_thread_bottoms, g_cudaGlobalConf->asyncStream));

        /* free allocator object */
        CUDA_ASSERT_SUCCESS(cudaFreeAsync(allocator, g_cudaGlobalConf->asyncStream));

        /* final sync */
        CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(g_cudaGlobalConf->asyncStream));

        /* cleanup */
        delete[] h_pages;
    }

    // ------------------------------
    // protected methods
    // ------------------------------
protected:
    // ------------------------------
    // Allocator calls
    // ------------------------------

    [[nodiscard]] uint32_t _AllocateNewChunkCpu() {
        /* reserve address block */
        const uint32_t address = reinterpret_cast<std::atomic<uint32_t> *>(const_cast<uint32_t *>(&_last_page_offset))->
                fetch_add(_thread_chunk_size_in_type);
        const uint32_t page_offset = address & kPageRemainder;
        const uint32_t cur_page = address >> kPageDivider;

        /* we are first on the page we are obliged to allocate it */
        if (page_offset == 0) {
            /* wait for all previous pages to be allocated */
            const uint32_t prev_page = cur_page - 1;
            while (_last_page < prev_page) {
                // spin
            }
            assert(_last_page == prev_page);

            /* allocate new page */
            const auto page = static_cast<T *>(malloc(sizeof(T) * kPageSizeInTypeSize));
            assert(page != nullptr);
            assert(_last_page == prev_page);
            assert(_pages[cur_page] == nullptr);
            _pages[cur_page] = page;

            /* update page counter */
            _last_page = _last_page + 1;
            std::atomic_thread_fence(std::memory_order_release);
        } else {
            /* otherwise we should just wait for our page be allocated if it is not */
            while (_last_page < cur_page) {
                // spin
            }
            assert(_last_page >= cur_page);
        }

        return address;
    }

    [[nodiscard]] __device__ uint32_t _AllocateNewChunkGpu() {
        /* reserve address block */
        const uint32_t address = atomicAdd(const_cast<uint32_t *>(&_last_page_offset), _thread_chunk_size_in_type);
        const uint32_t page_offset = address & kPageRemainder;
        const uint32_t cur_page = address >> kPageDivider;

        /* we are first on the page we are obliged to allocate it */
        if (page_offset == 0) {
            /* wait for all previous pages to be allocated */
            const uint32_t prev_page = cur_page - 1;
            while (_last_page < prev_page) {
                // spin
            }
            assert(_last_page == prev_page);

            /* allocate new page */
            const auto page = static_cast<T *>(malloc(sizeof(T) * kPageSizeInTypeSize));
            assert(page != nullptr);
            assert(_last_page == prev_page);
            assert(_pages[cur_page] == nullptr);
            _pages[cur_page] = page;

            /* update page counter */
            _last_page = _last_page + 1;
            __threadfence();
        } else {
            /* otherwise we should just wait for our page be allocated if it is not */
            while (_last_page < cur_page) {
                // spin
            }
            assert(_last_page >= cur_page);
        }

        return address;
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    /* general data */
    volatile uint32_t _thread_chunk_size_in_type{};
    uint32_t _num_threads{};
    bool _is_cuda_alloc{};

    /* thread data */
    uint32_t *_thread_tops{};
    uint32_t *_thread_bottoms{};

    /* base allocator data */
    T **_pages{};
    volatile uint32_t _last_page{};
    volatile uint32_t _last_page_offset{};
};

using FastAllocator = BaseFastAllocator_<Node_>;

#endif //ALLOCATORS_CUH
