#ifndef ALLOCATORS_CUH
#define ALLOCATORS_CUH

#include <data.cuh>
#include <atomic>
#include <global_conf.cuh>
#include <cuda_runtime.h>

static constexpr uint64_t kKb = 1024;
static constexpr uint64_t kMb = 1024 * kKb;
static constexpr size_t kCpuThreadChunkSize = 16 * kMb;
static constexpr size_t kGpuThreadChunkSize = 8 * kKb;

static constexpr uint32_t kSpinLockFree = 0;
static constexpr uint32_t kSpinLockLocked = 1;


static constexpr uint64_t kPageSize = kMb * 512;
static constexpr uint32_t kMaxPages = 64;
static constexpr uint64_t kPageSizeInTypeSize = kPageSize / sizeof(Node_);

static constexpr uint32_t kPageRemainder = kPageSizeInTypeSize - 1;
static constexpr uint32_t kPageDivider = std::countr_zero(kPageSizeInTypeSize);

class CpuSpinLock {
public:
    explicit CpuSpinLock(volatile uint32_t &lock_base) : _lock_base(
        reinterpret_cast<std::atomic<uint32_t> *>(const_cast<uint32_t *>(&lock_base))) {
    }

    FORCE_INLINE void lock() {
        uint32_t expected = kSpinLockFree;

        while (!_lock_base->compare_exchange_strong(expected, kSpinLockLocked, std::memory_order_acquire)) {
            expected = kSpinLockFree;
        }
    }

    FORCE_INLINE void unlock() {
        _lock_base->store(kSpinLockFree, std::memory_order_release);
    }

protected:
    std::atomic<uint32_t> *_lock_base;
};

class GpuSpinLock {
public:
    FAST_DCALL_ALWAYS explicit GpuSpinLock(volatile uint32_t &lock_base): _lock_base(&lock_base) {
    }

    FAST_DCALL_ALWAYS void lock() {
        while (atomicCAS(const_cast<uint32_t *>(_lock_base), kSpinLockFree, kSpinLockLocked) != kSpinLockFree) {
            __threadfence();
            // __nanosleep(100);
        }

        printf("locked\n");
        __threadfence();
    }

    FAST_DCALL_ALWAYS void unlock() {
        __threadfence();
        const auto result = atomicExch(const_cast<uint32_t *>(_lock_base), kSpinLockFree);
        assert(result == kSpinLockLocked);
        __threadfence();
        printf("unlocked...\n");
    }

protected:
    volatile uint32_t *_lock_base;
};

class FastAllocator {
    // ------------------------------
    // inner constants
    // ------------------------------

    static_assert(sizeof(Node_) == 16);
    static_assert(kPageSizeInTypeSize % 2 == 0);

public:
    // ------------------------------
    // Type creation
    // ------------------------------

    FastAllocator(const size_t thread_chunk_size, const size_t num_threads) : _thread_chunk_size_in_type(
                                                                                  thread_chunk_size / sizeof(Node_)),
                                                                              _num_threads(num_threads) {
        /* verify provided data */
        assert(num_threads * thread_chunk_size < kPageSize);
        assert(thread_chunk_size % sizeof(Node_) == 0);
        assert(kPageSizeInTypeSize % _thread_chunk_size_in_type == 0);

        /* allocate memory for management data */
        _pages = new Node_ *[kMaxPages]{};
        _thread_tops = new uint32_t[num_threads]{};
        _thread_bottoms = new uint32_t[num_threads];

        /* preallocate first page */
        _pages[0] = new Node_[kPageSizeInTypeSize];

        /* set thread bottoms */
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            _thread_bottoms[thread_idx] = thread_idx * _thread_chunk_size_in_type;
        }

        /* prepare allocator data */
        _last_page = 0;
        _last_page_offset = num_threads * _thread_chunk_size_in_type;
        _spin_lock = kSpinLockFree;

        /* ensure no null pointer is allocated */
        _thread_tops[0] = 1;

        assert(_last_page_offset < kPageSizeInTypeSize);
        assert(_last_page < kMaxPages);
    }

    // ------------------------------
    // Data path methods
    // ------------------------------

    [[nodiscard]] FAST_CALL_ALWAYS Node_ &operator[](const uint32_t idx) {
        assert(idx < kPageSizeInTypeSize * _last_page + _last_page_offset);
        assert(_pages[idx / kPageSizeInTypeSize] != nullptr);
        return _pages[idx >> kPageDivider][idx & kPageRemainder];
    }

    [[nodiscard]] FAST_CALL_ALWAYS const Node_ &operator[](const uint32_t idx) const {
        assert(idx < kPageSizeInTypeSize * _last_page + _last_page_offset);
        assert(_pages[idx / kPageSizeInTypeSize] != nullptr);
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
        return _thread_bottoms[t_idx] + top;
    }

    // ------------------------------
    // Management methods
    // ------------------------------

    [[nodiscard]] FAST_CALL_ALWAYS size_t GetNumNodesAllocated() const {
        return _last_page * kPageSizeInTypeSize + _last_page_offset;
    }

    void DeallocHost() {
        for (uint32_t page_idx = 0; page_idx <= _last_page; ++page_idx) {
            delete[] _pages[page_idx];
        }

        delete[] _pages;
        delete[] _thread_tops;
        delete[] _thread_bottoms;

        _pages = nullptr;
        _thread_tops = nullptr;
        _thread_bottoms = nullptr;
    }

    [[nodiscard]] FastAllocator *DumpToGPU() const {
        /* allocate the object itself */
        FastAllocator *d_allocator;
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_allocator, sizeof(FastAllocator), g_cudaGlobalConf->asyncStream));

        /* allocate thread tops */
        uint32_t *d_thread_tops;
        CUDA_ASSERT_SUCCESS(
            cudaMallocAsync(&d_thread_tops, _num_threads * sizeof(uint32_t), g_cudaGlobalConf->asyncStream));

        /* allocate thread bottoms */
        uint32_t *d_thread_bottoms;
        CUDA_ASSERT_SUCCESS(
            cudaMallocAsync(&d_thread_bottoms, _num_threads * sizeof(uint32_t), g_cudaGlobalConf->asyncStream));

        /* allocate pages */
        auto h_pages = new Node_ *[kMaxPages]{};
        for (uint32_t page_idx = 0; page_idx <= _last_page; ++page_idx) {
            Node_ *d_page;
            CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_page, kPageSize, g_cudaGlobalConf->asyncStream));
            h_pages[page_idx] = d_page;
        }

        /* allocate page table */
        Node_ **d_pages;
        CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_pages, kMaxPages * sizeof(Node_ *), g_cudaGlobalConf->asyncStream));

        /* copy allocator object */
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(d_allocator, this, sizeof(FastAllocator), cudaMemcpyHostToDevice, g_cudaGlobalConf->
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
        CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(d_pages, h_pages, kMaxPages * sizeof(Node_ *), cudaMemcpyHostToDevice,
            g_cudaGlobalConf->asyncStream));

        /* update object pointers */
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_allocator->_thread_tops, &d_thread_tops, sizeof(uint32_t *), cudaMemcpyHostToDevice,
                g_cudaGlobalConf->asyncStream));
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_allocator->_thread_bottoms, &d_thread_bottoms, sizeof(uint32_t *), cudaMemcpyHostToDevice
                , g_cudaGlobalConf->asyncStream));
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_allocator->_pages, &d_pages, sizeof(Node_ **), cudaMemcpyHostToDevice, g_cudaGlobalConf->
                asyncStream));

        /* final sync */
        CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(g_cudaGlobalConf->asyncStream));

        /* free temporary data */
        delete[] h_pages;

        return d_allocator;
    }

    static void DeallocGPU(FastAllocator *allocator) {
        /* prepare host page table */
        auto h_pages = new Node_ *[kMaxPages]{};

        /* copy pointer to pages */
        Node_ **d_pages;
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(&d_pages, &allocator->_pages, sizeof(Node_ **), cudaMemcpyDeviceToHost,
                g_cudaGlobalConf->asyncStream));

        /* copy page table */
        CUDA_ASSERT_SUCCESS(
            cudaMemcpyAsync(h_pages, d_pages, kMaxPages * sizeof(Node_ *), cudaMemcpyDeviceToHost,
                g_cudaGlobalConf->asyncStream));

        /* delete individual pages */
        for (uint32_t page_idx = 0; page_idx < kMaxPages; ++page_idx) {
            if (h_pages[page_idx] != nullptr) {
                CUDA_ASSERT_SUCCESS(cudaFreeAsync(h_pages[page_idx], g_cudaGlobalConf->asyncStream));
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
        CpuSpinLock lock(_spin_lock);
        lock.lock();

        uint32_t page_offset = _last_page_offset;
        _last_page_offset += _thread_chunk_size_in_type;
        if (page_offset == kPageSizeInTypeSize) {
            /* we exhausted current page -> we need new one */
            /* performed in lazy manner */

            ++_last_page;
            assert(_last_page < kMaxPages);

            _pages[_last_page] = new Node_[kPageSizeInTypeSize];

            /* already offset for new chunk */
            page_offset = 0;
            _last_page_offset = _thread_chunk_size_in_type;
        }

        assert(
            static_cast<uint64_t>(_last_page) * kPageSizeInTypeSize + static_cast<uint64_t>(page_offset) <= static_cast<
            uint64_t>(UINT32_MAX));
        const uint32_t idx = _last_page * kPageSizeInTypeSize + page_offset;
        lock.unlock();

        assert(_last_page_offset <= kPageSizeInTypeSize);
        return idx;
    }

    [[nodiscard]] __device__ uint32_t _AllocateNewChunkGpu() {
        GpuSpinLock lock(_spin_lock);
        lock.lock();

        uint32_t page_offset = _last_page_offset;
        _last_page_offset += _thread_chunk_size_in_type;
        if (page_offset == kPageSizeInTypeSize) {
            /* we exhausted current page -> we need new one */
            /* performed in lazy manner */

            ++_last_page;
            assert(_last_page < kMaxPages);

            _pages[_last_page] = new Node_[kPageSizeInTypeSize];

            /* already offset for new chunk */
            page_offset = 0;
            _last_page_offset = _thread_chunk_size_in_type;
        }

        assert(
            static_cast<uint64_t>(_last_page) * kPageSizeInTypeSize + static_cast<uint64_t>(page_offset) <= static_cast<
            uint64_t>(UINT32_MAX));
        const uint32_t idx = _last_page * kPageSizeInTypeSize + page_offset;
        lock.unlock();

        assert(_last_page_offset <= kPageSizeInTypeSize);
        return idx;
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    /* general data */
    uint32_t _thread_chunk_size_in_type{};
    uint32_t _num_threads{};

    /* thread data */
    uint32_t *_thread_tops{};
    uint32_t *_thread_bottoms{};

    /* base allocator data */
    Node_ **_pages{};
    uint32_t _last_page{};
    uint32_t _last_page_offset{};
    volatile uint32_t _spin_lock{};
};

#endif //ALLOCATORS_CUH
