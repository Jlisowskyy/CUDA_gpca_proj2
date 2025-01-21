/* internal includes */
#include <thread_mgr.cuh>
#include <defines.cuh>
#include <thread_pool.hpp>
#include <allocators.hpp>
#include <defines.hpp>

/* external includes */
#include <vector>
#include <iostream>
#include <chrono>
#include <barrier>
#include <atomic>

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

    const auto t_bucket_start = std::chrono::high_resolution_clock::now();
    _prepareBuckets(pack, data);
    const auto t_bucket_end = std::chrono::high_resolution_clock::now();

    std::cout << "Time spent on gpu bucket preparation: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_bucket_end - t_bucket_start).count()
            << "ms" << std::endl;

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
    static constexpr double kMaxDeviation = 0.5;
    static constexpr uint32_t kPrefixMask = GenMask(kPrefixSize);

    const size_t num_threads = std::thread::hardware_concurrency();

    /* prepare allocator */
    BigChunkAllocator allocator(kDefaultAllocChunkSize, num_threads);

    /* prepare buckets */
    Buckets buckets(num_threads, data.max_threads, allocator);

    /* prepare pool */
    ThreadPool pool(num_threads);
    std::barrier barrier(static_cast<ptrdiff_t>(num_threads));

    /* prepare prefixes */
    std::vector<uint32_t> prefixes{};
    prefixes.resize(data.max_threads);

    /* prepare mean and standard deviation */
    std::atomic<double> dev_sum{};
    std::atomic<uint32_t> mean_sum{};
    double mean{};
    double std_dev{};

    /* prepare max occupancy */
    uint32_t max_occup{};
    std::vector<uint32_t> max_occups(num_threads, 0);

    /* prepare boundaries */
    const size_t prefix_thread_range = data.max_threads / num_threads;
    const size_t buckets_thread_range = prefix_thread_range;
    const size_t seq_thread_range = pack.sequences.size() / num_threads;

    /* Run in parallel */
    pool.RunThreads([&](const uint32_t thread_idx) {
        /* fill prefix array */
        const size_t prefix_start = thread_idx * prefix_thread_range;
        const size_t prefix_end = thread_idx == num_threads - 1 ? data.max_threads : prefix_start + prefix_thread_range;

        for (size_t idx = prefix_start; idx < prefix_end; ++idx) {
            prefixes[idx] = kPrefixSize;
        }

        /* split sequences to buckets */
        const size_t seq_start = thread_idx * seq_thread_range;
        const size_t seq_end = thread_idx == num_threads - 1 ? pack.sequences.size() : seq_start + seq_thread_range;

        for (size_t seq_idx = seq_start; seq_idx < seq_end; ++seq_idx) {
            const size_t key = pack.sequences[seq_idx].GetWord(0) & kPrefixMask;
            buckets.PushToBucket(thread_idx, key, seq_idx);
        }

        /* merge buckets */
        barrier.arrive_and_wait();
        if (thread_idx == 0) {
            buckets.MergeBuckets();
        }
        barrier.arrive_and_wait();

        /* calculate statistics */
        const size_t bucket_start = thread_idx * buckets_thread_range;
        const size_t bucket_end = thread_idx == num_threads - 1 ? data.max_threads : bucket_start + buckets_thread_range;

        /* local accumulators */
        uint32_t mean_sum_local{};
        uint32_t max_occup_local{};

        /* calculate local statistics */
        for (size_t bucket_idx = bucket_start; bucket_idx < bucket_end; ++bucket_idx) {
            const uint32_t bucket_size = buckets.GetBucketSize(bucket_idx);
            max_occup_local = std::max(max_occup_local, bucket_size);
            mean_sum_local += bucket_size;
        }

        /* save results */
        mean_sum.fetch_add(mean_sum_local);
        max_occups[thread_idx] = max_occup_local;

        /* calculate mean */
        barrier.arrive_and_wait();
        if (thread_idx == 0) {
            mean = static_cast<double>(mean_sum.load()) / data.max_threads;
        }
        barrier.arrive_and_wait();

        /* calculate deviation */
        double dev_sum_local{};

        for (size_t bucket_idx = bucket_start; bucket_idx < bucket_end; ++bucket_idx) {
            const uint32_t bucket_size = buckets.GetBucketSize(bucket_idx);
            const double a = static_cast<double>(bucket_size) - mean;
            dev_sum_local += a * a;
        }

        /* save results */
        dev_sum.fetch_add(dev_sum_local);
    });
    pool.Wait();

    /* calculate standard deviation */
    std_dev = std::sqrt(dev_sum.load() / data.max_threads);

    /* verify standard deviation */
    const double dev_coef = std_dev / mean;

    std::cout << "Acquired standard deviation: " << std_dev << " and mean: " << mean << '\n';
    std::cout << "Acquired deviation coef: " << dev_coef << '\n';

    if (dev_coef > kMaxDeviation) {
        std::cout << "Standard deviation is too high!\n";
        std::cout << "Fallback to cpu algorithm\n";
        data.build_on_device = false;
        return;
    }

    std::cout << "Standard deviation is acceptable proceeding to GPU TRIE build...\n";
    data.build_on_device = true;

    /* Note: data for gpu is dumped inside the thread pool */
    _dumpBucketsToGpu(buckets, prefixes, data, *std::max_element(max_occups.begin(), max_occups.end()));
}

void ThreadMgr::_dumpBucketsToGpu(Buckets &buckets, const std::vector<uint32_t> &prefixes, MgrTrieBuildData &data,
    const uint32_t max_occup) const {
    /* Format data for GPU */
    auto h_buckets = std::make_unique<uint32_t[]>(data.max_threads * max_occup);
    auto h_bucket_sizes = std::make_unique<uint32_t[]>(data.max_threads);
    for (size_t bucket_idx = 0; bucket_idx < data.max_threads; ++bucket_idx) {
        h_bucket_sizes[bucket_idx] = buckets.GetBucketSize(bucket_idx);

        uint32_t cur{};
        while (!buckets.IsEmpty(bucket_idx)) {
            h_buckets[bucket_idx * max_occup + cur++] = buckets.PopBucket(bucket_idx);
        }
    }

    /* transfer to device */
    CUDA_ASSERT_SUCCESS(cudaMalloc(&data.d_buckets, data.max_threads * max_occup * sizeof(uint32_t)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(data.d_buckets, h_buckets.get(), data.max_threads * max_occup * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    CUDA_ASSERT_SUCCESS(cudaMalloc(&data.d_bucket_prefix_len, data.max_threads * sizeof(uint32_t)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(data.d_bucket_prefix_len, prefixes.data(), data.max_threads * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    CUDA_ASSERT_SUCCESS(cudaMalloc(&data.d_bucket_sizes, data.max_threads * sizeof(uint32_t)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(data.d_bucket_sizes, h_bucket_sizes.get(), data.max_threads * sizeof(uint32_t),
        cudaMemcpyHostToDevice));
}
