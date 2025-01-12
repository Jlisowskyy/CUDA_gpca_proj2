#ifndef TRIE_HPP
#define TRIE_HPP

/* internal includes */
#include <data.hpp>

/* external includes */
#include <iostream>

/* Forward declarations */
class BigMemChunkAllocator;

static constexpr size_t kMbInBytes = 1024 * 1024;
static constexpr size_t kDefaultAllocSize = 4 * 2048 * kMbInBytes;

template<class nodeT, size_t arrSize>
bool CompareTries(const nodeT *rootLeft, const nodeT *rootRight) {
    if (!rootLeft && !rootRight) {
        return true;
    }

    if (!rootLeft || !rootRight) {
        std::cout << "One of the roots is null" << std::endl;
        return false;
    }

    if (rootLeft->idx != rootRight->idx) {
        std::cout << "Roots have different idx" << std::endl;
        return false;
    }

    for (size_t i = 0; i < arrSize; ++i) {
        if (!CompareTries<nodeT, arrSize>(rootLeft->next[i], rootRight->next[i])) {
            return false;
        }
    }

    return true;
}

class Trie {
    // ------------------------------
    // Class inner types
    // ------------------------------

    static constexpr size_t NextCount = 2;

    struct alignas(1) Node_ {
        Node_() = default;

        explicit Node_(const uint32_t seq_idx) : idx{seq_idx} {
        }

        bool operator==(const Node_ &n) const {
            return idx == n.idx;
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

    bool Insert(const uint32_t idx, const uint32_t bit_idx = 0) { return _insert(idx, bit_idx); }

    void FindPairs(uint32_t idx, std::vector<std::pair<size_t, size_t> > &out);

    [[nodiscard]] size_t GetSizeMB() const;

    void MergeTriesByPrefix(std::vector<Trie> &tries, size_t prefix_size);

    bool operator==(const Trie &trie) const {
        return CompareTries<Node_, NextCount>(_root, trie._root);
    }

    void DumpToDot(const std::string &filename) const;

    void SetOwner(const bool is_owner) {
        _is_root_owner = is_owner;
    }

    void SetAllocator(BigMemChunkAllocator *allocator) {
        _allocator = allocator;
    }

    // ------------------------------
    // private methods
    // ------------------------------

private:
    bool _insert(uint32_t idx, uint32_t start_bit_idx);

    void _tryToFindPair(Node_ *p, uint32_t idx, uint32_t bit_idx, std::vector<std::pair<size_t, size_t> > &out);

    template<typename... Args>
    [[nodiscard]] Node_ *_allocateNode(Args... args) const;

    // ------------------------------
    // Class fields
    // ------------------------------

    BigMemChunkAllocator*_allocator{};
    Node_ *_root{};
    const std::vector<BinSequence> *_sequences{};
    bool _is_root_owner{false};
};


void BuildTrieSingleThread(Trie &trie, const std::vector<BinSequence> &sequences);

void BuildTrieParallel(Trie &trie, const std::vector<BinSequence> &sequences);

#endif //TRIE_HPP
