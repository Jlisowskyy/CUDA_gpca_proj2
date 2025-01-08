#ifndef DATA_CUH
#define DATA_CUH

/* external includes */
#include <cinttypes>
#include <cuda_runtime.h>
#include <tuple>

/* internal includes */
#include <defines.cuh>
#include <data.hpp>

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

    static std::tuple<cuda_Solution *, uint32_t *> DumpToGPU(const size_t num_solutions) {
        uint32_t *d_data{};
        CUDA_ASSERT_SUCCESS(cudaMalloc(&d_data, GetMemBlockSize(num_solutions)));
        CUDA_ASSERT_SUCCESS(cudaMemset(d_data, INT_MAX, GetMemBlockSize(num_solutions)));

        cuda_Solution *d_solutions{};
        CUDA_ASSERT_SUCCESS(cudaMalloc(&d_solutions, sizeof(cuda_Solution)));
        CUDA_ASSERT_SUCCESS(cudaMemcpy(d_solutions->_data, &d_data, sizeof(uint32_t *), cudaMemcpyHostToDevice));

        return {d_solutions, d_data};
    }

    // ------------------------------
    // Class fields
    // ------------------------------
protected:
    uint32_t *_data{};
};

// ------------------------------
// Trie Node allocator
// ------------------------------

// TODO:
class cuda_Allocator {
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

    explicit cuda_Data(const BinSequencePack &pack) : cuda_Data(pack.sequences.size(),
                                                                (pack.max_seq_size_bits + 31) / 32) {
        static constexpr uint64_t kBitMask32 = ~static_cast<uint32_t>(0);

        for (size_t seq_idx = 0; seq_idx < pack.sequences.size(); ++seq_idx) {
            const auto &sequence = pack.sequences[seq_idx];
            auto fetcher = (*this)[seq_idx];

            /* user dwords for better performance */
            for (size_t qword_idx = 0; qword_idx < sequence.GetSizeWords(); ++qword_idx) {
                const uint64_t qword = sequence.GetWord(qword_idx);
                const size_t dword_idx = qword_idx * 2;
                const uint32_t lo = qword & kBitMask32;
                const uint32_t hi = (qword >> 32) & kBitMask32;

                fetcher.GetWord(dword_idx) = lo;
                fetcher.GetWord(dword_idx + 1) = hi;
            }
        }
    }

    ~cuda_Data() = default;

    // ------------------------------
    // Interaction
    // ------------------------------

    [[nodiscard]] cuda_Data *DumpToGPU() const {
        /* allocate manager object */
        cuda_Data *d_data;

        CUDA_ASSERT_SUCCESS(cudaMalloc(&d_data, sizeof(cuda_Data)));
        CUDA_ASSERT_SUCCESS(cudaMemcpy(d_data, this, sizeof(cuda_Data), cudaMemcpyHostToDevice));

        /* allocate data itself */
        uint32_t *d_data_data;

        const size_t data_size = _num_sequences_padded32 * (_max_sequence_length + 1) * sizeof(uint32_t);
        CUDA_ASSERT_SUCCESS(cudaMalloc(&d_data_data, data_size));
        CUDA_ASSERT_SUCCESS(cudaMemcpy(d_data_data, _data, data_size, cudaMemcpyHostToDevice));

        /* update manager object */
        CUDA_ASSERT_SUCCESS(cudaMemcpy(&d_data->_data, &d_data_data, sizeof(uint32_t *), cudaMemcpyHostToDevice));

        return d_data;
    }

    [[nodiscard]] static uint32_t *GetDataPtrHost(const cuda_Data *d_data) {
        uint32_t *ptr;
        CUDA_ASSERT_SUCCESS(cudaMemcpy(&ptr, d_data->_data, sizeof(uint32_t *), cudaMemcpyDeviceToHost));
        return ptr;
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
