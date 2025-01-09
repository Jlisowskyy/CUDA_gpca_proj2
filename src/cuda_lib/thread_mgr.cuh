#ifndef THREAD_MGR_HPP
#define THREAD_MGR_HPP

/* internal includes */
#include <data.hpp>

/* external includes */
#include <cinttypes>

// ------------------------------
// Protocol layout
// ------------------------------

struct MgrTrieBuildData {
    /* allocator management */
    uint32_t max_nodes{};
    uint32_t max_threads{};
    uint32_t max_nodes_per_thread{};

    /* trie kernel management */
    uint32_t num_blocks{};
    uint32_t num_threads_per_block{};

    /* bucket building management */
    uint32_t *d_buckets{};
    uint32_t *d_bucket_prefix_len{};
};

struct MgrTrieSearchData {
    uint32_t num_solutions{};

    uint32_t num_blocks{};
    uint32_t num_threads_per_block{};
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

    [[nodiscard]] MgrTrieBuildData PrepareTrieBuildData(const BinSequencePack &pack) const;

    [[nodiscard]] MgrTrieSearchData PrepareSearchData() const;

    // ------------------------------
    // Private methods
    // ------------------------------
protected:

    void _prepareBuckets(const BinSequencePack& pack, MgrTrieBuildData& data) const;

    // ------------------------------
    // Class fields
    // ------------------------------
};

#endif //THREAD_MGR_HPP
