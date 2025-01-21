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

bool Trie::Search(const uint32_t idx) const {
    const BinSequence &sequence = (*_sequences)[idx];
    size_t bit_idx = 0;

    const Node_ *p = _root;
    /* follow the path */
    while (p && (p->next[0] || p->next[1]) && bit_idx < sequence.GetSizeBits()) {
        const bool value = sequence.GetBit(bit_idx++);

        /* Continue on main path */
        p = p->next[value];
    }

    return p && (p->idx == idx || sequence.Compare((*_sequences)[p->idx]));
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
Trie::Node_ *Trie::_allocateNode(const size_t thread_idx, Args... args) const {
    assert(_allocator != nullptr);

    auto n = _allocator->Alloc<Node_>(thread_idx, args...);
    assert(n != nullptr);

    return n;
}

void Trie::MergeTriesByPrefix(std::vector<Trie> &tries, const size_t prefix_size) {
    /* TODO: does not work when prefix is longer than some sequences */
    /* TODO: ExtractBit */

    static const auto ExtractBit = [](const size_t item, const size_t idx) {
        return (item >> idx) & 1;
    };

    _root = _allocateNode(0);

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
                (*n)->next[value] = _allocateNode(0);
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

bool Trie::_insert(const uint32_t idx, const size_t thread_idx, const uint32_t start_bit_idx) {
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
    *p = _allocateNode(thread_idx);
    assert(oldNode->next[0] == nullptr && oldNode->next[1] == nullptr);

    while (bit_idx < oldSequence.GetSizeBits() &&
           bit_idx < sequence.GetSizeBits() &&
           oldSequence.GetBit(bit_idx) == sequence.GetBit(bit_idx)) {
        /* add nodes until we reach the difference or there is no more bits to compare */

        const bool bit = sequence.GetBit(bit_idx++);

        (*p)->next[bit] = _allocateNode(thread_idx);
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
    trie.SetAllocator(new BigChunkAllocator(kDefaultAllocChunkSize, 1));
    trie.SetOwner(true);

    for (size_t idx = 0; idx < sequences.size(); ++idx) {
        trie.Insert(idx, 0);
    }

    if (GlobalConfig.WriteDotFiles) {
        trie.DumpToDot("/tmp/trie.dot");
    }
}

void BuildTrieParallel(Trie &trie, const std::vector<BinSequence> &sequences) {
    const auto [num_threads, prefix_mask, power_of_2] = GetBatchSplitData();

    /* prepare allocator */
    auto *allocator = new BigChunkAllocator(kDefaultAllocChunkSize, num_threads);
    trie.SetAllocator(allocator);
    trie.SetOwner(true);

    /* prepare buckets */
    Buckets buckets(num_threads, num_threads, *allocator);

    /* prepare tries */
    std::vector<Trie> tries{};
    for (size_t idx = 0; idx < num_threads; ++idx) {
        tries.emplace_back(sequences);
        tries.back().SetAllocator(allocator);
    }

    ThreadPool pool(num_threads);
    std::barrier barrier(static_cast<ptrdiff_t>(num_threads));
    const size_t thread_job_size = sequences.size() / num_threads;

    pool.RunThreads([&](const uint32_t thread_idx) {
        /* Bucket sorting */
        const size_t job_start = thread_idx * thread_job_size;
        const size_t job_end = thread_idx == num_threads - 1
                                   ? sequences.size()
                                   : (thread_idx + 1) * thread_job_size;

        for (size_t seq_idx = job_start; seq_idx < job_end; ++seq_idx) {
            const size_t key = sequences[seq_idx].GetWord(0) & prefix_mask;
            buckets.PushToBucket(thread_idx, key, seq_idx);
        }

        /* wait for all threads to finish sorting */
        barrier.arrive_and_wait();

        if (thread_idx == 0) {
            /* merge buckets */
            buckets.MergeBuckets();

            /* display bucket stats */
            std::cout << "Bucket stats:\n";
            for (size_t b_idx = 0; b_idx < num_threads; ++b_idx) {
                std::cout << "Bucket " << b_idx << " size: " << buckets.GetBucketSize(b_idx) << '\n';
            }
            std::cout << std::endl;
        }

        /* wait for finish merging */
        barrier.arrive_and_wait();

        /* fill the tries */
        while (!buckets.IsEmpty(thread_idx)) {
            const uint32_t seq_idx = buckets.PopBucket(thread_idx);
            tries[thread_idx].Insert(seq_idx, thread_idx, power_of_2);
        }
    });

    pool.Wait();

    trie.MergeTriesByPrefix(tries, power_of_2);

    if (GlobalConfig.WriteDotFiles) {
        trie.DumpToDot("/tmp/trie.dot");
    }
}
