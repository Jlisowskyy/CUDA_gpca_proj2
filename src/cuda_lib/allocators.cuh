#ifndef ALLOCATORS_CUH
#define ALLOCATORS_CUH

#include <data.cuh>

static constexpr uint64_t kKb = 1024;
static constexpr uint64_t kMb = 1024 * kKb;
static constexpr size_t kCpuThreadChunkSize = 16 * kMb;
static constexpr size_t kGpuThreadChunkSize = 8 * kKb;

class FastAllocator {
    // ------------------------------
    // inner constants
    // ------------------------------

    static constexpr uint64_t kPageSize = kMb * 512;
    static constexpr uint32_t kMaxPages = 64;
    static constexpr uint64_t kPageSizeInTypeSize = kPageSize / sizeof(Node_);

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

        /* allocate memory for management data */
        _pages = new Node_ *[kMaxPages]{};
        _thread_tops = new uint32_t[num_threads]{};
        _thread_bottoms = new uint32_t[num_threads];

        /* preallocate first page */
        _pages[0] = new Node_[kPageSizeInTypeSize];

        /* set thread bottoms */
        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            _thread_bottoms[thread_idx] = thread_idx * thread_chunk_size;
        }

        /* prepare allocator data */
        _last_page = 0;
        _last_page_offset = num_threads * thread_chunk_size;
    }

    // ------------------------------
    // Data path methods
    // ------------------------------

    [[nodiscard]] FAST_CALL_ALWAYS Node_ &operator[](const uint32_t idx) {
        return _pages[idx / kPageSizeInTypeSize][idx % kPageSizeInTypeSize];
    }

    [[nodiscard]] FAST_CALL_ALWAYS const Node_ &operator[](const uint32_t idx) const {
        return _pages[idx / kPageSizeInTypeSize][idx % kPageSizeInTypeSize];
    }

    template<bool isGpu = false>
    [[nodiscard]] FAST_CALL_ALWAYS uint32_t AllocateNode(const uint32_t t_idx) {
        if constexpr (isGpu) {
            return _allocateNodeGpu(t_idx);
        } else {
            return _allocateNodeCpu(t_idx);
        }
    }

    // ------------------------------
    // Management methods
    // ------------------------------

    void DeallocHost() {
        for (uint32_t page_idx = 0; page_idx < _last_page; ++page_idx) {
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
        return nullptr;
    }

    static void DeallocGPU(FastAllocator *allocator) {
    }

    // ------------------------------
    // protected methods
    // ------------------------------
protected:
    [[nodiscard]] FAST_DCALL_ALWAYS uint32_t _allocateNodeGpu(const uint32_t t_idx) {
        return 0;
    }

    [[nodiscard]] uint32_t _allocateNodeCpu(const uint32_t t_idx) {
        return 0;
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    /* general data */
    uint64_t _thread_chunk_size_in_type{};
    uint64_t _num_threads{};

    /* thread data */
    uint32_t *_thread_tops{};
    uint32_t *_thread_bottoms{};

    /* base allocator data */
    Node_ **_pages{};
    uint32_t _last_page{};
    uint32_t _last_page_offset{};
};


#endif //ALLOCATORS_CUH
