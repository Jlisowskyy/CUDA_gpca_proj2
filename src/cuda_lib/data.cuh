#ifndef DATA_CUH
#define DATA_CUH

/* external includes */
#include <cinttypes>
#include <cuda_runtime.h>
#include <tuple>

/* internal includes */
#include <defines.cuh>
#include <data.hpp>
#include <barrier>
#include <iostream>
#include <cstdio>

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

    uint32_t padding{};
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
                reinterpret_cast<uint32_t *>(atomicAdd(reinterpret_cast<unsigned long long int *>(&_data), 2*sizeof(uint32_t)));

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

        FAST_CALL_ALWAYS bool Compare(const SequenceFetcher &other, const size_t bit_idx = 0) const {
            if (GetSequenceLength() != other.GetSequenceLength()) {
                return false;
            }

            for (uint32_t b_idx = bit_idx; b_idx < GetSequenceLength(); ++b_idx) {
                if (GetBit(b_idx) != other.GetBit(b_idx)) {
                    return false;
                }
            }

            return true;
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

    explicit cuda_Data(uint32_t num_sequences, uint32_t max_sequence_length);

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
