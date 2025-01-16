#ifndef DEFINES_HPP
#define DEFINES_HPP

// static constexpr size_t kMinSequenceLength = 1000;
static constexpr size_t kMinSequenceLength = 1;
// static constexpr size_t kMinNumSequences = 100'000;
static constexpr size_t kMinNumSequences = 1;

#ifdef NDEBUG
static constexpr bool kIsDebug = false;
#else
static constexpr bool kIsDebug = true;
#endif

#endif //DEFINES_HPP
