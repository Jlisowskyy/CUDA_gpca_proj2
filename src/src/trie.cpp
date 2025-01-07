/* internal includes */
#include <trie.hpp>
#include <thread_pool.hpp>
#include <global_conf.hpp>

/* external includes */
#include <iostream>
#include <barrier>
#include <fstream>
#include <queue>
#include <map>
#include <string>

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

template<size_t kChunkSize>
class BigMemChunkAllocator {
public:
    BigMemChunkAllocator() = default;

    explicit BigMemChunkAllocator(const size_t bytes) {
        Alloc((bytes + kChunkSize - 1) / kChunkSize);
    }

    ~BigMemChunkAllocator() {
        Clear();
    }

    void Alloc(const size_t num_blocks) {
        assert(num_blocks > 0);

        _chunk = new char[kChunkSize];
        _chunk_top = _chunk;

        for (size_t idx = 0; idx < num_blocks - 1; ++idx) {
            _chunks.push_back(new char[kChunkSize]);
        }
    }

    void Clear() {
        delete[] _chunk;
        _chunk = nullptr;

        for (const char *ptr: _chunks) {
            delete[] ptr;
        }
        _chunks.clear();

        for (const char *ptr: _wasted_chunks) {
            delete[] ptr;
        }
        _wasted_chunks.clear();
    }

    template<class ItemT>
    ItemT *AllocItem() {
        char *current_top;
        char *new_top;

        do {
            do {
                std::atomic_thread_fence(std::memory_order_acquire);
                current_top = _chunk_top;
                new_top = current_top + sizeof(ItemT);

                if (new_top > _chunk + kChunkSize) {
                    SwapChunk_(sizeof(ItemT));
                }
            } while (new_top > _chunk + kChunkSize);
        } while (!std::atomic_compare_exchange_strong(
            reinterpret_cast<std::atomic<char *> *>(&_chunk_top),
            &current_top,
            new_top
        ));

        return reinterpret_cast<ItemT *>(current_top);
    }

protected:
    void SwapChunk_(const size_t size) {
        if (m.try_lock()) {
            if (_chunk_top + size < _chunk + kChunkSize) {
                m.unlock();
                return;
            }

            /* ensure atomicity */
            char *old_chunk = _chunk;
            _chunk = nullptr;
            _chunk_top = nullptr;
            std::atomic_thread_fence(std::memory_order_release);

            _wasted_chunks.push_back(old_chunk);
            assert(!_chunks.empty());

            _chunk_top = _chunks.back();
            _chunk = _chunk_top;

            _chunks.pop_back();
            m.unlock();
        }
    }

    std::mutex m{};
    char *_chunk{};
    char *_chunk_top{};
    std::vector<char *> _chunks{};
    std::vector<char *> _wasted_chunks{};
};

static std::string GetDotNodeLabel(const BinSequence &sequence) {
    std::string out{};
    for (size_t idx = 0; idx < sequence.GetSizeBits(); ++idx) {
        out += sequence.GetBit(idx) ? "1" : "0";
    }

    return out;
}

Trie::~Trie() {
    if (_is_root_owner) {
        delete _allocator;
    }
}

void Trie::FindPairs(const uint32_t idx, std::vector<std::pair<size_t, size_t> > &out) {
    const BinSequence &sequence = (*_sequences)[idx];
    size_t bit_idx = 0;

    Node_ *p = _root;
    /* follow the path */
    while (p && (p->next[0] || p->next[1]) && bit_idx < sequence.GetSizeBits()) {
        const bool value = sequence.GetBit(bit_idx++);

        /* Try path with flipped bit */
        _tryToFindPair(p->next[!value], idx, bit_idx, out);

        if (bit_idx == sequence.GetSizeBits() && p->idx != UINT32_MAX && idx < p->idx) {
            /* check if there is some shorter sequence that is a valid pair */
            out.emplace_back(p->idx, idx);
        }

        /* Continue on main path */
        p = p->next[value];
    }

    if (!p) {
        /* nothing to do */
        return;
    }

    /* we used all bits from the sequence */
    if (bit_idx == sequence.GetSizeBits()) {
        /* add child if are valid pair */

        if (p->next[0] && p->next[0]->idx != UINT32_MAX && idx < p->next[0]->idx) {
            out.emplace_back(p->next[0]->idx, idx);
        }

        if (p->next[1] && p->next[1]->idx != UINT32_MAX && idx < p->next[1]->idx) {
            out.emplace_back(p->next[1]->idx, idx);
        }

        return;
    }

    /* Validate we found our sequence */
    assert((p && p->idx == idx) || sequence.Compare((*_sequences)[p->idx]));
}

void Trie::_tryToFindPair(Node_ *p, const uint32_t idx, uint32_t bit_idx,
                          std::vector<std::pair<size_t, size_t> > &out) {
    /* Check if we have a valid node */
    if (!p) {
        return;
    }

    /*  Use original sequence to follow path after the flipped bit */
    const BinSequence &sequence = (*_sequences)[idx];

    /* Follow the path */
    while (p && (p->next[0] || p->next[1]) && bit_idx < sequence.GetSizeBits()) {
        p = p->next[sequence.GetBit(bit_idx++)];
    }

    /* Check if we found a valid sequence */
    if (!p) {
        return;
    }

    /* Check if remaining bits match after the single flip */
    if (const BinSequence &other_sequence = (*_sequences)[p->idx];
        idx < p->idx && sequence.Compare(other_sequence, bit_idx)) {
        out.emplace_back(p->idx, idx);
    }
}

Trie::Node_ *Trie::_allocateNode() const {
    assert(_allocator != nullptr);
    return _allocator->AllocItem<Node_>();
}

void Trie::MergeTriesByPrefix(std::vector<Trie> &tries, const size_t prefix_size) {
    /* TODO: does not work when prefix is longer than some sequences */
    /* TODO: ExtractBit */

    static const auto ExtractBit = [](const size_t item, const size_t idx) {
        return (item >> idx) & 1;
    };

    _root = _allocateNode();

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
                (*n)->next[value] = _allocateNode();
            }

            n = &((*n)->next[value]);
        }

        *n = p;
    }
}

void Trie::DumpToDot(const std::string &filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    out << "digraph Trie {\n";
    out << "    node [shape=circle];\n";
    out << "    edge [arrowsize=0.5];\n\n";

    std::queue<std::pair<Node_ *, size_t> > queue;
    size_t next_id = 0;
    std::map<Node_ *, size_t> node_ids;

    if (_root) {
        queue.push({_root, next_id});
        node_ids[_root] = next_id++;
    }

    while (!queue.empty()) {
        auto [node, id] = queue.front();
        queue.pop();

        out << "    node" << id << " [";
        if (!(node->next[0] || node->next[1])) {
            out << "shape=box, ";
        }

        if (!(node->next[0] || node->next[1]) || node->idx != 0) {
            out << "label=\"" << GetDotNodeLabel((*_sequences)[node->idx]) << ", " << node->idx << "\"";
        } else {
            out << "label=\"\"";
        }
        out << "];\n";

        for (size_t i = 0; i < NextCount; ++i) {
            if (node->next[i]) {
                if (node_ids.find(node->next[i]) == node_ids.end()) {
                    node_ids[node->next[i]] = next_id;
                    queue.push({node->next[i], next_id++});
                }

                out << "    node" << id << " -> node" << node_ids[node->next[i]]
                        << " [label=\"" << i << "\"];\n";
            }
        }
    }

    out << "}\n";
    out.close();
}

bool Trie::_insert(const uint32_t idx, const uint32_t start_bit_idx) {
    const BinSequence &sequence = (*_sequences)[idx];
    Node_ **p = &_root;

    if (start_bit_idx >= sequence.GetSizeBits()) {
        return false;
    }

    size_t bit_idx = start_bit_idx;
    /* traverse existing tree or until we reach the end of the sequence */
    while (*p && ((*p)->next[0] || (*p)->next[1]) && bit_idx < sequence.GetSizeBits()) {
        p = &((*p)->next[sequence.GetBit(bit_idx++)]);
    }

    if (!*p) {
        /* we reached the end of the tree */
        *p = _allocateNode();
        return true;
    }

    if (bit_idx == sequence.GetSizeBits()) {
        /* we reached the end of the sequence */
        /* we are also sure that the p is not null */

        assert((*p)->idx == UINT32_MAX);
        /* assign the sequence index to the node */
        (*p)->idx = idx;

        return true;
    }

    if (*p && sequence.Compare((*_sequences)[(*p)->idx], bit_idx)) {
        /* we reached leaf node and the sequence is the same -> leave */
        return false;
    }

    /* we found node with assigned sequence */
    Node_ *oldNode = *p;
    const BinSequence &oldSequence = (*_sequences)[oldNode->idx];
    *p = _allocateNode();

    while (bit_idx < oldSequence.GetSizeBits() &&
           bit_idx < sequence.GetSizeBits() &&
           oldSequence.GetBit(bit_idx) == sequence.GetBit(bit_idx)) {
        /* add nodes until we reach the difference or there is no more bits to compare */

        Node_ *nNode = _allocateNode();
        const bool bit = sequence.GetBit(bit_idx++);

        (*p)->next[bit] = nNode;
        p = &((*p)->next[bit]);
    }

    if (bit_idx == oldSequence.GetSizeBits()) {
        /* we reached the end of the old sequence */
        (*p)->idx = oldNode->idx;
        delete oldNode;

        Node_ *nNode = _allocateNode();
        nNode->idx = idx;

        (*p)->next[sequence.GetBit(bit_idx)] = nNode;

        return true;
    }

    if (bit_idx == sequence.GetSizeBits()) {
        /* we reached the end of the new sequence */
        (*p)->idx = idx;
        (*p)->next[oldSequence.GetBit(bit_idx)] = oldNode;

        return true;
    }

    /* we reached the difference */
    assert(!(bit_idx == oldSequence.GetSizeBits() && bit_idx == sequence.GetSizeBits()));
    (*p)->next[oldSequence.GetBit(bit_idx)] = oldNode;

    Node_ *nNode = _allocateNode();
    nNode->idx = idx;
    (*p)->next[sequence.GetBit(bit_idx)] = nNode;

    return true;
}

void BuildTrieSingleThread(Trie &trie, const std::vector<BinSequence> &sequences) {
    for (size_t idx = 0; idx < sequences.size(); ++idx) {
        trie.Insert(idx);
    }

    if (GlobalConfig.WriteDotFiles) {
        trie.DumpToDot("/tmp/trie.dot");
    }
}

void BuildTrieParallel(Trie &trie, const std::vector<BinSequence> &sequences) {
    StabAllocator<Node<uint32_t> > allocator(sequences.size() + 16);
    auto *big_mem_chunk_allocator = new BigMemChunkAllocator<kDefaultAllocSize>();
    big_mem_chunk_allocator->Alloc(1);

    std::vector<ThreadSafeStack<uint32_t> > buckets{};
    buckets.reserve(16);

    trie.SetAllocator(big_mem_chunk_allocator);
    trie.SetOwner(true);

    for (size_t idx = 0; idx < 16; ++idx) {
        buckets.emplace_back(allocator);
    }

    std::vector<Trie> tries{};
    for (size_t idx = 0; idx < buckets.size(); ++idx) {
        tries.emplace_back(sequences);
        tries.back().SetAllocator(big_mem_chunk_allocator);
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

    if (GlobalConfig.WriteDotFiles) {
        trie.DumpToDot("/tmp/trie.dot");
    }
}
