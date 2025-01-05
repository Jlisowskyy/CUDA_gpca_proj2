#ifndef TRIE_HPP
#define TRIE_HPP

#include <data.hpp>

template<class nodeT, size_t arrSize>
void CleanArrBasedTree(const nodeT *n) {
    if (!n) {
        return;
    }

    for (size_t i = 0; i < arrSize; ++i) { CleanArrBasedTree<nodeT, arrSize>(n->next[i]); }

    delete n;
}

template<class nodeT, size_t arrSize>
size_t CalcTreeSize(const nodeT *n) {
    if (!n) {
        return 0;
    }

    size_t size{};
    for (size_t i = 0; i < arrSize; ++i) {
        size += CalcTreeSize<nodeT, arrSize>(n->next[i]);
    }

    return size + sizeof(nodeT);
}

template<class nodeT, size_t arrSize>
bool CompareTries(nodeT* rootLeft, nodeT* rootRight) {
    if (!rootLeft && !rootRight) {
        return true;
    }

    if (!rootLeft || !rootRight) {
        return false;
    }

    if (rootLeft->idx != rootRight->idx) {
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

    struct Node_ {
        Node_() = default;

        explicit Node_(const uint32_t seq_idx) : idx{seq_idx} {
        }

        bool operator==(const Node_ &n) const {
            return idx == n.idx;
        }

        Node_ *next[NextCount]{};
        uint32_t idx{};
    };

    // ------------------------------
    // Class creation
    // ------------------------------

public:
    Trie() = delete;

    explicit Trie(const std::vector<BinSequence> &sequences) : _sequences(&sequences) {
    }

    ~Trie() { CleanArrBasedTree<Node_, NextCount>(_root); }

    // ------------------------------
    // class interaction
    // ------------------------------

    bool Insert(const uint32_t idx, const uint32_t bit_idx = 0) { return _insert(idx, bit_idx); }

    void FindPairs(uint32_t idx, std::vector<std::pair<size_t, size_t> > &out) const;

    [[nodiscard]] size_t GetSizeMB() const {
        static constexpr size_t kBytesInMB = 1024 * 1024;

        return (CalcTreeSize<Node_, NextCount>(_root) + kBytesInMB) / kBytesInMB;
    }

    void MergeTriesByPrefix(std::vector<Trie> &tries, size_t prefix_size);

    // ------------------------------
    // private methods
    // ------------------------------

private:
    bool _insert(uint32_t idx, uint32_t start_bit_idx);

    // ------------------------------
    // Class fields
    // ------------------------------

    Node_ *_root{};
    const std::vector<BinSequence> *_sequences{};
};


void BuildTrieSingleThread(Trie &trie, const std::vector<BinSequence> &sequences);

void BuildTrieParallel(Trie &trie, const std::vector<BinSequence> &sequences);

#endif //TRIE_HPP
