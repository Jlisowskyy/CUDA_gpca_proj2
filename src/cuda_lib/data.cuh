#ifndef DATA_CUH
#define DATA_CUH

/* external includes */
#include <cinttypes>
#include <cuda_runtime.h>

/* internal includes */
#include <defines.cuh>
#include <data.hpp>

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

    explicit cuda_Data(const BinSequencePack &pack) : cuda_Data(pack.sequences.size(),
                                                                (pack.max_seq_size_bits + 31) / 32) {
        for (size_t seq_idx = 0; seq_idx < pack.sequences.size(); ++seq_idx) {
            const auto &sequence = pack.sequences[seq_idx];
            auto fetcher = (*this)[seq_idx];

            for (size_t bit_idx = 0; bit_idx < sequence.GetSizeBits(); ++bit_idx) {
                fetcher.SetBit(bit_idx, sequence.GetBit(bit_idx));
            }
        }
    }

    ~cuda_Data() = default;

    // ------------------------------
    // Interaction
    // ------------------------------

    [[nodiscard]] cuda_Data *DumpToGPU() {
        // cuda_Data *d_data = cudaMalloc()
        return nullptr;
    }

    [[nodiscard]] uint32_t FAST_CALL_ALWAYS GetNumSequences() const {
        return _num_sequences;
    }

    [[nodiscard]] FAST_CALL_ALWAYS constexpr SequenceFetcher operator[](const uint32_t seq_idx) const {
        assert(seq_idx < _num_sequences && "DETECTED OVERFLOW");

        return SequenceFetcher(seq_idx, this);
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
