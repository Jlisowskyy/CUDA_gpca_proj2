/* internal includes */
#include <trie.hpp>
#include <thread_pool.hpp>
#include <global_conf.hpp>
#include <allocators.hpp>

/* external includes */
#include <iostream>
#include <barrier>
#include <defines.hpp>
#include <fstream>
#include <queue>
#include <map>
#include <string>

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

        for (const auto &i: p->next) {
            if (i && i->idx != UINT32_MAX && idx < i->idx) {
                out.emplace_back(i->idx, idx);
            }
        }

        return;
    }

    /* Validate we found our sequence */
    assert((p && p->idx == idx) || sequence.Compare((*_sequences)[p->idx]));
}

size_t Trie::GetSizeMB() const {
    static constexpr size_t kBytesInMB = 1024 * 1024;
    return _allocator->GetUsedSize() / kBytesInMB;
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

template<typename... Args>
Trie::Node_ *Trie::_allocateNode(Args... args) const {
    assert(_allocator != nullptr);

    auto n = _allocator->AllocItem<Node_>(args...);
    assert(n != nullptr);

    return n;
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
        *p = _allocateNode(idx);
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

    if (sequence.Compare((*_sequences)[(*p)->idx], bit_idx)) {
        /* we reached leaf node and the sequence is the same -> leave */
        return false;
    }

    /* we found node with assigned sequence */
    Node_ *oldNode = *p;
    const BinSequence &oldSequence = (*_sequences)[oldNode->idx];
    *p = _allocateNode();
    assert(oldNode->next[0] == nullptr && oldNode->next[1] == nullptr);

    while (bit_idx < oldSequence.GetSizeBits() &&
           bit_idx < sequence.GetSizeBits() &&
           oldSequence.GetBit(bit_idx) == sequence.GetBit(bit_idx)) {
        /* add nodes until we reach the difference or there is no more bits to compare */

        const bool bit = sequence.GetBit(bit_idx++);

        (*p)->next[bit] = _allocateNode();
        p = &((*p)->next[bit]);
    }

    if (bit_idx == sequence.GetSizeBits() && bit_idx == oldSequence.GetSizeBits()) {
        /* we reached the end of both sequences and no difference was found assign on of them and exit */
        (*p)->idx = idx;

        return true;
    }

    if (bit_idx == oldSequence.GetSizeBits()) {
        /* we reached the end of the old sequence */
        assert((*p)->idx == UINT32_MAX);

        (*p)->idx = oldNode->idx;
        oldNode->idx = idx;
        (*p)->next[sequence.GetBit(bit_idx)] = oldNode;

        return true;
    }

    if (bit_idx == sequence.GetSizeBits()) {
        /* we reached the end of the new sequence */
        assert((*p)->idx == UINT32_MAX);

        (*p)->idx = idx;
        (*p)->next[oldSequence.GetBit(bit_idx)] = oldNode;

        return true;
    }

    /* we reached the difference */
    (*p)->next[oldSequence.GetBit(bit_idx)] = oldNode;
    (*p)->next[sequence.GetBit(bit_idx)] = _allocateNode(idx);

    return true;
}

void BuildTrieSingleThread(Trie &trie, const std::vector<BinSequence> &sequences) {
    auto *big_mem_chunk_allocator = new BigMemChunkAllocator();
    big_mem_chunk_allocator->Alloc(kDefaultAllocSize);
    trie.SetAllocator(big_mem_chunk_allocator);
    trie.SetOwner(true);

    for (size_t idx = 0; idx < sequences.size(); ++idx) {
        trie.Insert(idx);
    }

    big_mem_chunk_allocator->DisplayAllocaInfo();

    if (GlobalConfig.WriteDotFiles) {
        trie.DumpToDot("/tmp/trie.dot");
    }
}

void BuildTrieParallel(Trie &trie, const std::vector<BinSequence> &sequences) {
    StabAllocator<Node<uint32_t> > allocator(sequences.size() + 16);
    auto *big_mem_chunk_allocator = new BigMemChunkAllocator();
    big_mem_chunk_allocator->Alloc(kDefaultAllocSize);

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
    std::barrier barrier(16);
    pool.RunThreads([&](const uint32_t idx) {
        /* Bucket sorting */
        for (size_t seq_idx = idx; seq_idx < sequences.size(); seq_idx += 16) {
            const size_t key = sequences[seq_idx].GetWord(0) & 0xF;
            buckets[key].PushSafe(seq_idx);
        }

        barrier.arrive_and_wait();

        if (idx == 0) {
            std::cout << "Bucket stats: " << std::endl;
            for (size_t b_idx = 0; b_idx < buckets.size(); ++b_idx) {
                std::cout << "Bucket " << b_idx << " size: " << buckets[b_idx].GetSize() << std::endl;
            }
        }

        if constexpr (kIsDebug) {
            if (idx == 0) {
                size_t sum{};

                for (const auto &bucket: buckets) {
                    sum += bucket.GetSize();
                }

                assert(sum == sequences.size());
            }

            barrier.arrive_and_wait();
        }

        /* fill the tries */
        auto &bucket = buckets[idx];
        while (!bucket.IsEmpty()) {
            const uint32_t seq_idx = bucket.PopNotSafe();
            tries[idx].Insert(seq_idx, 4);
        }
    });

    pool.Wait();

    trie.MergeTriesByPrefix(tries, 4);

    big_mem_chunk_allocator->DisplayAllocaInfo();

    if (GlobalConfig.WriteDotFiles) {
        trie.DumpToDot("/tmp/trie.dot");
    }
}
