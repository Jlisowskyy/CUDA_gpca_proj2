/* internal includes */
#include <trie.hpp>
#include <thread_pool.hpp>

/* external includes */
#include <iostream>
#include <barrier>

template<typename ItemT>
struct Node {
    ItemT item{};
    uint32_t mem_idx{};
};

template<typename ItemT>
class StabAllocator {
public:
    explicit StabAllocator(const size_t num_items): _num_items(num_items), _items(new ItemT[num_items + 1]) {
    };

    uint32_t AllocNotSafe() {
        return ++_num_allocated;
    }

    uint32_t AllocSafe() {
        size_t current_idx;
        do {
            current_idx = _num_allocated;
            assert(current_idx < _num_items);
        } while (!std::atomic_compare_exchange_weak(
            reinterpret_cast<std::atomic<size_t> *>(&_num_allocated),
            &current_idx,
            current_idx + 1
        ));

        return current_idx + 1;
    }

    ItemT *GetBase() const { return _items; }
    ItemT *GetItem(uint32_t idx) { return _items + idx; }

    ~StabAllocator() {
        delete []_items;
    }

protected:
    size_t _num_items{};
    size_t _num_allocated{};
    ItemT *_items{};
};

template<typename ItemT>
class ThreadSafeStack {
public:
    explicit ThreadSafeStack(StabAllocator<Node<ItemT> > &alloc): _top(alloc.AllocNotSafe()), _alloc(&alloc) {
    }

    bool IsEmpty() {
        return _alloc->GetItem(_top)->mem_idx == 0;
    }

    void PushSafe(const ItemT &item) {
        uint32_t new_idx = _alloc->AllocSafe();
        Node<ItemT> *new_node = _alloc->GetItem(new_idx);
        new_node->item = item;

        uint32_t current_top;
        do {
            current_top = _top;
            new_node->mem_idx = current_top;
        } while (!std::atomic_compare_exchange_strong(
            reinterpret_cast<std::atomic<uint32_t> *>(&_top),
            &current_top,
            new_idx
        ));

        ++_counter;
    }

    ItemT PopNotSafe() {
        uint32_t current_top = _top;
        Node<ItemT> *top_node = _alloc->GetItem(current_top);
        _top = top_node->mem_idx;

        --_counter;
        return top_node->item;
    }

    [[nodiscard]] uint32_t GetSize() const {
        return _counter;
    }

protected:
    uint32_t _top{};
    uint32_t _counter{};
    StabAllocator<Node<ItemT> > *_alloc{};
};

void Trie::FindPairs(uint32_t idx, std::vector<std::pair<size_t, size_t> > &out) const {
}

void Trie::MergeTriesByPrefix(std::vector<Trie> &tries, const size_t prefix_size) {
    static const auto ExtractBit = [](const size_t item, const size_t idx) {
        return (item >> idx) & 1;
    };

    _root = new Node_{};

    for (size_t idx = 0; idx < tries.size(); ++idx) {
        Trie &trie = tries[idx];
        Node_ *p = trie._root;
        trie._root = nullptr;

        if (!p) {
            continue;
        }

        Node_ **n = &_root;
        size_t bit = 0;
        while (bit < prefix_size) {
            const bool value = ExtractBit(idx, bit++);

            if (!(*n)->next[value]) {
                (*n)->next[value] = new Node_{};
            }

            n = &((*n)->next[value]);
        }

        *n = p;
    }
}

bool Trie::_insert(const uint32_t idx, const uint32_t start_bit_idx) {
    const BinSequence &sequence = (*_sequences)[idx];
    Node_ **p = &_root;

    size_t bit_idx = start_bit_idx;
    while (*p && ((*p)->next[0] || (*p)->next[1])) {
        p = &((*p)->next[sequence.GetBit(bit_idx++)]);
    }

    if (*p && sequence.Compare((*_sequences)[(*p)->idx], bit_idx)) {
        return false;
    }

    if (!*p) {
        *p = new Node_(idx);
        return true;
    }

    // if not perform branching on these one
    Node_ *oldNode = *p;
    Node_ *newNode = new Node_(idx);
    const BinSequence &oldSequence = (*_sequences)[oldNode->idx];
    *p = new Node_{};

    while (oldSequence.GetBit(bit_idx) == sequence.GetBit(bit_idx))
    // we need to perform another branching
    {
        Node_ *nNode = new Node_();
        const bool bit = sequence.GetBit(bit_idx++);

        (*p)->next[bit] = nNode;
        p = &((*p)->next[bit]);
    }

    (*p)->next[oldSequence.GetBit(bit_idx)] = oldNode;
    (*p)->next[sequence.GetBit(bit_idx)] = newNode;

    return true;
}

void BuildTrieSingleThread(Trie &trie, const std::vector<BinSequence> &sequences) {
    for (size_t idx = 0; idx < sequences.size(); ++idx) {
        trie.Insert(idx);
    }
}

void BuildTrieParallel(Trie &trie, const std::vector<BinSequence> &sequences) {
    StabAllocator<Node<uint32_t> > allocator(sequences.size() + 16);
    std::vector<ThreadSafeStack<uint32_t> > buckets{};
    buckets.reserve(16);

    for (size_t idx = 0; idx < 16; ++idx) {
        buckets.emplace_back(allocator);
    }

    std::vector<Trie> tries{};
    for (size_t idx = 0; idx < buckets.size(); ++idx) {
        tries.emplace_back(sequences);
    }

    ThreadPool pool(16);
    std::barrier<> barrier(16);
    pool.RunThreads([&](const uint32_t idx) {
        /* Bucket sorting */
        for (size_t seq_idx = idx; seq_idx < sequences.size(); seq_idx += 16) {
            const size_t key = sequences[seq_idx].GetWord(0) & 0xF;
            buckets[key].PushSafe(seq_idx);
        }

        if (idx == 0) {
            std::cout << "Bucket stats: " << std::endl;
            for (size_t b_idx = 0; b_idx < buckets.size(); ++b_idx) {
                std::cout << "Bucket " << b_idx << " size: " << buckets[b_idx].GetSize() << std::endl;
            }
        }

        barrier.arrive_and_wait();

        /* fill the tries */
        auto &bucket = buckets[idx];
        while (!bucket.IsEmpty()) {
            const uint32_t seq_idx = bucket.PopNotSafe();
            tries[idx].Insert(seq_idx, 4);
        }
    });

    pool.Wait();

    trie.MergeTriesByPrefix(tries, 4);
}
