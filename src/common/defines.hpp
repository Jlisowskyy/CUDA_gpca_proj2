#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <thread>

static constexpr size_t kMinSequenceLength = 1000;
static constexpr size_t kMinNumSequences = 100'000;

#ifdef NDEBUG
static constexpr bool kIsDebug = false;
#else
static constexpr bool kIsDebug = true;
#endif

static constexpr size_t kDefaultAllocChunkSize = 4 * 1024 * 1024; // 4MB

static uint32_t GetMask(const uint32_t num_bits) {
    uint32_t mask{};

    for (uint32_t idx = 0; idx < num_bits; ++idx) {
        mask |= static_cast<uint32_t>(1) << idx;
    }

    return mask;
}

static std::tuple<uint64_t, uint32_t, uint32_t> GetBatchSplitData() {
    const uint64_t num_threads = std::bit_floor(std::thread::hardware_concurrency());
    const uint32_t power_of_2 = std::countr_zero(num_threads);
    const uint32_t prefix_mask = GetMask(power_of_2);

    return {num_threads, prefix_mask, power_of_2};
}

#endif //DEFINES_HPP
