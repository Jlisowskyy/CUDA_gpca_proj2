/* internal includes */
#include <thread_mgr.cuh>
#include <defines.cuh>

/* external includes */
#include <vector>

// ------------------------------
// Helpers
// ------------------------------

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

// ------------------------------
// Constants
// ------------------------------

/* TODO: replace with bucket search logic */
static constexpr uint32_t kPrefixSize = 15;

// ------------------------------
// Implementations
// ------------------------------

MgrTrieBuildData ThreadMgr::PrepareTrieBuildData(const BinSequencePack &pack) const {
    /* TODO: adjust */
    static constexpr uint32_t kMaxThreads = pow2(kPrefixSize);

    /* TODO: adjust */
    static constexpr uint32_t kNumThreadsPerBlock = 512;

    MgrTrieBuildData data{};

    /* fill allocator management */
    data.max_nodes = pack.sequences.size() * pack.max_seq_size_bits;
    data.max_threads = kMaxThreads;
    data.max_nodes_per_thread = pack.max_seq_size_bits;

    /* fill trie kernel management */
    data.num_threads_per_block = kNumThreadsPerBlock;
    data.num_blocks = data.max_threads / data.num_threads_per_block;

    _prepareBuckets(pack, data);

    return data;
}

MgrTrieSearchData ThreadMgr::PrepareSearchData() const {
    /* TODO: replace with some logic */
    static constexpr uint32_t kMaxSolutions = 1'000'00;

    /* TODO: adjust */
    static constexpr uint32_t kThreadsPerBlock = 1024;

    /* TODO: adjust */
    static constexpr uint32_t kMaxBlocks = 1024;

    MgrTrieSearchData data{};

    data.num_solutions = kMaxSolutions;
    data.num_threads_per_block = kThreadsPerBlock;
    data.num_blocks = kMaxBlocks;

    return data;
}

void ThreadMgr::_prepareBuckets(const BinSequencePack &pack, MgrTrieBuildData &data) const {
    /* TODO: replace */
    static constexpr uint32_t kPrefixMask = GenMask(kPrefixSize);

    std::vector<uint32_t> prefixes{};
    prefixes.resize(data.max_threads);

    for (size_t t_idx = 0; t_idx < data.max_threads; ++t_idx) {
        /* TODO: */
        prefixes[t_idx] = kPrefixSize;
    }

    std::vector<uint32_t> buckets{};
    buckets.resize(2 * pack.sequences.size());

    for (uint32_t seq_idx = 0; seq_idx < pack.sequences.size(); ++seq_idx) {
        /* TODO: */
        const auto &seq = pack.sequences[seq_idx];
        const auto idx = seq.GetWord(0) & kPrefixMask;

        uint32_t &counter = buckets[idx];
        buckets[(1 + counter) * data.max_threads + idx] = seq_idx;
        ++counter;
    }

    /* transfer to device */
    CUDA_ASSERT_SUCCESS(cudaMalloc(&data.d_buckets, buckets.size() * sizeof(uint32_t)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(data.d_buckets, buckets.data(), buckets.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    CUDA_ASSERT_SUCCESS(cudaMalloc(&data.d_bucket_prefix_len, prefixes.size() * sizeof(uint32_t)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(data.d_bucket_prefix_len, prefixes.data(), prefixes.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
}
