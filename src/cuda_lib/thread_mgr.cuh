#ifndef THREAD_MGR_HPP
#define THREAD_MGR_HPP

/* internal includes */
#include <data.hpp>
#include <allocators.hpp>
#include <defines.cuh>

/* external includes */
#include <cinttypes>

// ------------------------------
// Constants
// ------------------------------

static constexpr uint32_t kPrefixSize = 11;
static constexpr uint32_t kMaxThreadsBuild = pow2(kPrefixSize);
static constexpr uint32_t kNumThreadsPerBlockBuild = 1024;
static constexpr uint32_t kLog2NumThreadsPerBlockBuild = std::countr_zero(kNumThreadsPerBlockBuild);
static constexpr uint32_t kNumBlocksBuild = kMaxThreadsBuild / kNumThreadsPerBlockBuild;
static constexpr uint32_t kLog2NumBlocksBuild = std::countr_zero(kNumBlocksBuild);
static constexpr uint32_t kThreadsPerBlockSearch = 512;
static constexpr uint32_t kMaxBlocksSearch = 128;

static_assert(kNumThreadsPerBlockBuild <= 1024);
static_assert(kNumBlocksBuild <= 1024);
static_assert(kThreadsPerBlockSearch <= 1024);
static_assert(kMaxBlocksSearch <= 1024);

// ------------------------------
// Protocol layout
// ------------------------------

struct MgrTrieBuildData {
    /* allocator management */
    uint32_t max_nodes{};
    uint32_t max_threads{};
    uint32_t max_nodes_per_thread{};

    /* bucket building management */
    uint32_t *d_buckets{};
    uint32_t *d_bucket_sizes{};
    uint32_t bucket_prefix_len{};
    uint32_t max_occup{};
    bool build_on_device{};
};

// ------------------------------
// Thread MGR
// ------------------------------


class ThreadMgr {
public:
    // ------------------------------
    // Constructors
    // ------------------------------

    ThreadMgr() = default;

    ~ThreadMgr() = default;

    // ------------------------------
    // Interactions
    // ------------------------------

    [[nodiscard]] MgrTrieBuildData PrepareTrieBuildData(const BinSequencePack &pack,
                                                        bool enforce_gpu_build = false) const;

    // ------------------------------
    // Private methods
    // ------------------------------
protected:
    void _prepareBuckets(const BinSequencePack &pack, MgrTrieBuildData &data, bool enforce_gpu_build) const;

    void _dumpBucketsToGpu(Buckets &buckets, MgrTrieBuildData &data, uint32_t max_occup) const;

    // ------------------------------
    // Class fields
    // ------------------------------
};

#endif //THREAD_MGR_HPP
