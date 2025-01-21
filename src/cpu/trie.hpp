#ifndef TRIE_HPP
#define TRIE_HPP

/* internal includes */
#include <data.hpp>

/* external includes */
#include <iostream>

/* Forward declarations */
class BigChunkAllocator;

class Trie {
    // ------------------------------
    // Class inner types
    // ------------------------------

    static constexpr size_t NextCount = 2;

    struct alignas(1) Node_ {
        Node_() = default;

        explicit Node_(const uint32_t seq_idx) : idx{seq_idx} {
        }

        Node_ *next[NextCount]{};
        uint32_t idx{UINT32_MAX};
    };

    // ------------------------------
    // Class creation
    // ------------------------------

public:
    Trie() = delete;

    explicit Trie(const std::vector<BinSequence> &sequences) : _sequences(&sequences) {
    }

    ~Trie();

    // ------------------------------
    // class interaction
    // ------------------------------

    bool Insert(const uint32_t idx, const size_t thread_idx, const uint32_t bit_idx = 0) { return _insert(idx, thread_idx, bit_idx); }

    [[nodiscard]] bool Search(uint32_t idx) const;

    void FindPairs(uint32_t idx, std::vector<std::pair<size_t, size_t> > &out);

    void MergeTriesByPrefix(std::vector<Trie> &tries, size_t prefix_size);

    void DumpToDot(const std::string &filename) const;

    void SetOwner(const bool is_owner) {
        _is_root_owner = is_owner;
    }

    void SetAllocator(BigChunkAllocator *allocator) {
        _allocator = allocator;
    }

    // ------------------------------
    // private methods
    // ------------------------------

private:
    bool _insert(uint32_t idx, size_t thread_idx, uint32_t start_bit_idx);

    void _tryToFindPair(Node_ *p, uint32_t idx, uint32_t bit_idx, std::vector<std::pair<size_t, size_t> > &out);

    template<typename... Args>
    [[nodiscard]] Node_ *_allocateNode(size_t thread_idx, Args... args) const;

    // ------------------------------
    // Class fields
    // ------------------------------

    BigChunkAllocator *_allocator{};
    Node_ *_root{};
    const std::vector<BinSequence> *_sequences{};
    bool _is_root_owner{false};
};


void BuildTrieSingleThread(Trie &trie, const std::vector<BinSequence> &sequences);

void BuildTrieParallel(Trie &trie, const std::vector<BinSequence> &sequences);

#endif //TRIE_HPP
