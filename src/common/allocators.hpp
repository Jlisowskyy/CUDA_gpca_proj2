#ifndef ALLOCATORS_HPP
#define ALLOCATORS_HPP

#include <cinttypes>

class BigChunkAllocator {
    struct MemNode {
        MemNode() = default;

        ~MemNode() = default;

        uint8_t *mem_chunk{};
        MemNode *next{};
    };

public:
    BigChunkAllocator(const size_t chunk_size, const size_t num_threads) : _chunk_size(chunk_size),
                                                                           _num_threads(num_threads) {
        _thread_tops = new size_t[num_threads]{};
        _heads = new MemNode *[num_threads];

        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            _heads[thread_idx] = new MemNode();
            _heads[thread_idx]->mem_chunk = new uint8_t[chunk_size];
        }
    }

    ~BigChunkAllocator() {
        delete [] _thread_tops;

        for (size_t thread_idx = 0; thread_idx < _num_threads; ++thread_idx) {
            const MemNode *node = _heads[thread_idx];

            while (node) {
                const MemNode *next = node->next;
                delete node->mem_chunk;
                delete node;
                node = next;
            }
        }

        delete [] _heads;
    }

    template<typename T, typename... Args>
    [[nodiscard]] T *Alloc(const size_t thread_idx, Args... args) {
        const size_t size_needed = sizeof(T);
        const size_t cur_top = _thread_tops[thread_idx];

        if (cur_top + size_needed > _chunk_size) {
            auto *new_node = new MemNode();
            new_node->mem_chunk = new uint8_t[_chunk_size];
            new_node->next = _heads[thread_idx];

            _heads[thread_idx] = new_node;
            _thread_tops[thread_idx] = size_needed;
            return new(new_node->mem_chunk) T(args...);
        }

        MemNode *cur_node = _heads[thread_idx];
        _thread_tops[thread_idx] += size_needed;

        return new(cur_node->mem_chunk + cur_top) T(args...);
    }

protected:
    size_t _chunk_size{};
    size_t _num_threads{};

    size_t *_thread_tops{};
    MemNode **_heads{};
};

class Buckets {
    struct BucketNode {
        uint32_t seq_idx;
        BucketNode *next;
    };

    class LinkedList {
    public:
        LinkedList() = default;

        /* There is no need to dealloc the nodes as they are managed by external allocator */
        ~LinkedList() = default;

        void Insert(BigChunkAllocator &allocator, const size_t thread_idx, const uint32_t seq_idx) {
            if (!_tail) {
                _head = _tail = allocator.Alloc<BucketNode>(thread_idx);
                _head->seq_idx = seq_idx;
                _size = 1;
                return;
            }

            auto *new_node = allocator.Alloc<BucketNode>(thread_idx);
            new_node->seq_idx = seq_idx;
            new_node->next = _head;
            _head = new_node;
            ++_size;
        }

        [[nodiscard]] bool IsEmpty() const {
            return !_head;
        }

        [[nodiscard]] uint32_t Pop() {
            assert(_head != nullptr);

            const uint32_t seq_idx = _head->seq_idx;
            _head = _head->next;
            --_size;
            return seq_idx;
        }

        [[nodiscard]] size_t GetSize() const {
            return _size;
        }

        void MergeLists(LinkedList &dst) {
            if (&dst == this) {
                return;
            }

            if (!dst._head) {
                dst._head = _head;
                dst._tail = _tail;
                dst._size = _size;
            } else {
                dst._tail->next = _head;
                dst._tail = _tail;
                dst._size += _size;
            }

            _tail = nullptr;
            _head = nullptr;
        }

    protected:
        size_t _size{};
        BucketNode *_head{};
        BucketNode *_tail{};
    };

public:
    Buckets(const size_t num_threads, const size_t num_buckets, BigChunkAllocator &alloca) : _num_threads(num_threads),
        _num_buckets(num_buckets), _allocator(&alloca) {
        _buckets = new LinkedList *[_num_threads];
        for (size_t thread_idx = 0; thread_idx < _num_threads; ++thread_idx) {
            _buckets[thread_idx] = new LinkedList[num_buckets]{};
        }
    }

    ~Buckets() {
        for (size_t thread_idx = 0; thread_idx < _num_threads; ++thread_idx) {
            /* There is no need to dealloc the nodes as they are managed by external allocator */
            delete []_buckets[thread_idx];
        }
        delete []_buckets;
    }

    void PushToBucket(const size_t thread_idx, const size_t bucket_idx, const uint32_t seq_idx) {
        _buckets[thread_idx][bucket_idx].Insert(*_allocator, thread_idx, seq_idx);
    }

    void MergeBuckets() {
        for (size_t bucket_idx = 0; bucket_idx < _num_buckets; ++bucket_idx) {
            LinkedList &dst_list = _buckets[0][bucket_idx];

            for (size_t thread_idx = 1; thread_idx < _num_threads; ++thread_idx) {
                LinkedList &src_list = _buckets[thread_idx][bucket_idx];
                src_list.MergeLists(dst_list);
            }
        }
    }

    [[nodiscard]] uint32_t PopBucket(const size_t bucket_idx) {
        return _buckets[0][bucket_idx].Pop();
    }

    [[nodiscard]] bool IsEmpty(const size_t bucket_idx) const {
        return _buckets[0][bucket_idx].IsEmpty();
    }

    [[nodiscard]] size_t GetBucketSize(const size_t bucket_idx) const {
        return _buckets[0][bucket_idx].GetSize();
    }

protected:
    size_t _num_threads{};
    size_t _num_buckets{};
    LinkedList **_buckets{};
    BigChunkAllocator *_allocator{};
};

#endif //ALLOCATORS_HPP
