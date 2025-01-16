#ifndef ALLOCATORS_HPP
#define ALLOCATORS_HPP

#include <cinttypes>
#include <atomic>
#include <cstdlib>

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
        return reinterpret_cast<std::atomic<size_t> *>(&_num_allocated)->fetch_add(1) + 1;
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
        } while (!std::atomic_compare_exchange_weak(
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

class BigMemChunkAllocator {
public:
    BigMemChunkAllocator() = default;

    ~BigMemChunkAllocator() {
        Clear();
    }

    void Alloc(const size_t num_bytes) {
        assert(num_bytes > 0);

        _chunk.store(new char[num_bytes]);
        _chunk_top.store(_chunk.load());
        _num_bytes = num_bytes;
    }

    void Clear() {
        delete[] _chunk;
        _chunk = nullptr;
        _chunk_top = nullptr;
    }

    template<class ItemT, typename... Args>
    ItemT *AllocItem(Args &&... args) {
        char *old_ptr = _chunk_top.fetch_add(sizeof(ItemT), std::memory_order_relaxed);
        assert(_chunk_top <= _chunk.load() + _num_bytes);

        return new(reinterpret_cast<void *>(old_ptr)) ItemT(std::forward<Args>(args)...);
    }

    [[nodiscard]] size_t GetMaxSize() const {
        return _num_bytes;
    }

    [[nodiscard]] size_t GetUsedSize() const {
        return _chunk_top.load() - _chunk.load();
    }

    [[nodiscard]] size_t GetFreeSize() const {
        return _num_bytes - GetUsedSize();
    }

    void DisplayAllocaInfo() {
        printf("Total nodes allocated: %zu\n", GetUsedSize() / sizeof(Node<uint32_t>));
    }

protected:
    std::atomic<char *> _chunk_top{};
    std::atomic<char *> _chunk{};
    size_t _num_bytes{};
};

#endif //ALLOCATORS_HPP
