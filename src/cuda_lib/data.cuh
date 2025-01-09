#ifndef DATA_CUH
#define DATA_CUH

/* external includes */
#include <cinttypes>
#include <cuda_runtime.h>
#include <tuple>

/* internal includes */
#include <defines.cuh>
#include <data.hpp>

#include "../cpu/allocators.hpp"

// ------------------------------
// GPU data Node
// ------------------------------

static constexpr uint32_t kNextCount = 2;

struct Node_ {
    Node_() = default;

    HYBRID explicit Node_(const uint32_t idx) : seq_idx(idx) {
    }

    uint32_t next[kNextCount]{};
    uint32_t seq_idx{};
};

// ------------------------------
// Solution GPU storage
// ------------------------------

class alignas(128) cuda_Solution {
public:
    // ------------------------------
    // Constructors
    // ------------------------------

    cuda_Solution() = default;

    ~cuda_Solution() = default;

    // ------------------------------
    // Interactions
    // ------------------------------

    void FAST_DCALL_ALWAYS PushSolution(const uint32_t idx1, const uint32_t idx2) {
        const auto address =
                reinterpret_cast<uint32_t *>(atomicAdd(reinterpret_cast<unsigned long long int *>(&_data), 2));

        address[0] = idx1;
        address[1] = idx2;
    }

    [[nodiscard]] static size_t GetMemBlockSize(const size_t num_solutions) {
        return num_solutions * 2 * sizeof(uint32_t);
    }

    static std::tuple<cuda_Solution *, uint32_t *> DumpToGPU(size_t num_solutions);

    // ------------------------------
    // Class fields
    // ------------------------------
protected:
    uint32_t *_data{};
};

// ------------------------------
// Trie Node allocator
// ------------------------------

class cuda_Allocator {
public:
    // ------------------------------
    // Constructors
    // ------------------------------

    cuda_Allocator() = delete;

    cuda_Allocator(uint32_t max_nodes, uint32_t max_threads, uint32_t max_node_per_thread);

    ~cuda_Allocator() = default;

    // ------------------------------
    // Interactions
    // ------------------------------

    void DeallocHost() {
        delete _data;
        delete _node_counters;
        delete _thread_nodes;

        _data = nullptr;
        _node_counters = nullptr;
        _thread_nodes = nullptr;
    }

    [[nodiscard]] cuda_Allocator *DumpToGPU() const;

    void static DeallocGPU(cuda_Allocator *d_allocator);

    [[nodiscard]] FAST_DCALL_ALWAYS uint32_t AllocateNode(const uint32_t t_idx) const {
        --_node_counters[t_idx];

        const uint32_t idx = _thread_nodes[t_idx];
        assert(idx != 0);
        const uint32_t new_node_idx = _data[idx].next[0];

        _thread_nodes[t_idx] = new_node_idx;
        return idx;
    }

    __device__ void Consolidate(uint32_t t_idx);

    // ------------------------------
    // Private methods
    // ------------------------------
protected:
    __device__ void _prepareIdxes();

    __device__ void _cleanUpOwnSpace(uint32_t t_idx);

    // ------------------------------
    // Class fields
    // ------------------------------
    Node_ *_data{};

    uint32_t _max_nodes{};
    uint32_t _last_node{};

    uint32_t *_node_counters{};
    uint32_t *_thread_nodes{};

    uint32_t _max_threads{};
    uint32_t _max_node_per_thread{};

    // ------------------------------
    // GPU only fields
    // ------------------------------

    uint32_t *_d_idxes{};
};

// ------------------------------
// Sequence GPU storage
// ------------------------------

class alignas(128) cuda_Data {
public:
    // ------------------------------
    // Inner types
    // ------------------------------

    class SequenceFetcher {
    public:
        // ------------------------------
        // Constructors
        // ------------------------------

        SequenceFetcher() = default;

        ~SequenceFetcher() = default;

        HYBRID explicit constexpr
        SequenceFetcher(const uint32_t idx, const cuda_Data *__restrict__ data) : _idx(idx), _data(data) {
        }

        // ------------------------------
        // Getters
        // ------------------------------

        [[nodiscard]] FAST_CALL_ALWAYS uint32_t &GetSequenceLength() {
            return _data->_data[_data->_max_sequence_length * _data->_num_sequences_padded32 + _idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS const uint32_t &GetSequenceLength() const {
            return _data->_data[_data->_max_sequence_length * _data->_num_sequences_padded32 + _idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS bool GetBit(const uint32_t bit_idx) const {
            const uint32_t word_idx = bit_idx / 32;
            const uint32_t bit_offset = bit_idx % 32;
            const uint32_t word = _data->_data[word_idx * _data->_num_sequences_padded32 + _idx];

            return (word >> bit_offset) & 1;
        }

        [[nodiscard]] FAST_CALL_ALWAYS uint32_t &GetWord(const uint32_t word_idx) {
            return _data->_data[word_idx * _data->_num_sequences_padded32 + _idx];
        }

        [[nodiscard]] FAST_CALL_ALWAYS const uint32_t &GetWord(const uint32_t word_idx) const {
            return _data->_data[word_idx * _data->_num_sequences_padded32 + _idx];
        }

        FAST_CALL_ALWAYS void SetBit(const uint32_t bit_idx, const bool value) {
            const uint32_t word_idx = bit_idx / 32;
            const uint32_t bit_offset = bit_idx % 32;
            uint32_t &word = _data->_data[word_idx * _data->_num_sequences_padded32 + _idx];

            word &= ~(1 << bit_offset);
            word |= value << bit_offset;
        }

        // ------------------------------
        // Fields
        // ------------------------------
    protected:
        uint32_t _idx{};
        const cuda_Data *__restrict__ _data{};
    };

    // ------------------------------
    // Constructors
    // ------------------------------

    explicit cuda_Data(const uint32_t num_sequences,
                       const uint32_t max_sequence_length)
        : _num_sequences(num_sequences),
          _num_sequences_padded32(
              num_sequences + (32 - num_sequences % 32) % 32),
          _max_sequence_length(max_sequence_length) {
        _data = new uint32_t[_num_sequences_padded32 * (_max_sequence_length + 1)];
    }

    explicit cuda_Data(const BinSequencePack &pack);

    ~cuda_Data() = default;

    // ------------------------------
    // Interaction
    // ------------------------------

    [[nodiscard]] cuda_Data *DumpToGPU() const;

    [[nodiscard]] static uint32_t *GetDataPtrHost(const cuda_Data *d_data);

    [[nodiscard]] uint32_t FAST_CALL_ALWAYS GetNumSequences() const {
        return _num_sequences;
    }

    [[nodiscard]] FAST_CALL_ALWAYS constexpr SequenceFetcher operator[](const uint32_t seq_idx) const {
        assert(seq_idx < _num_sequences && "DETECTED OVERFLOW");

        return SequenceFetcher(seq_idx, this);
    }

    static void DeallocGPU(cuda_Data *d_data);

    void DeallocHost() {
        delete _data;
        _data = nullptr;
    }

    // ------------------------------
    // Fields
    // ------------------------------

protected:
    uint32_t *_data{};
    uint32_t _num_sequences{};
    uint32_t _num_sequences_padded32{};
    uint32_t _max_sequence_length{};
};

#endif //DATA_CUH
