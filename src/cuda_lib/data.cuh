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

class cuda_Solution {
    static constexpr uint32_t kMaxPages = 32;
    static constexpr uint32_t kPageSize = 16 * 1024 * 1024;

    struct Solution_ {
        uint32_t idx1{};
        uint32_t idx2{};
    };

    static_assert(sizeof(Solution_) == 8);

    static constexpr uint32_t kPageSizeInTypeSize = kPageSize / sizeof(Solution_);
    static constexpr uint32_t kPageDivider = std::countr_zero(kPageSizeInTypeSize);
    static constexpr uint32_t kPageRemainder = kPageSizeInTypeSize - 1;

    static_assert(IsPowerOfTwo(sizeof(Solution_)));
    static_assert(IsPowerOfTwo(kPageSizeInTypeSize));

public:
    // ------------------------------
    // Constructors
    // ------------------------------

    cuda_Solution();

    ~cuda_Solution() = default;

    // ------------------------------
    // Interactions
    // ------------------------------

    void FAST_DCALL_ALWAYS PushSolution(const uint32_t idx1, const uint32_t idx2) {
        /* increment atomically offset */
        const uint32_t my_address = atomicAdd(const_cast<uint32_t *>(&_last_page_offset), 1);

        /* allocate page if needed */
        const uint32_t page_offset = my_address & kPageRemainder;
        const uint32_t cur_page = my_address >> kPageDivider;

        if (page_offset == 0) {
            /* wait for all previous pages to be allocated */
            const uint32_t prev_page = cur_page - 1;
            while (_last_page < prev_page) {
                // spin
            }
            assert(_last_page == prev_page);

            /* allocate new page */
            const auto page = static_cast<Solution_ *>(malloc(sizeof(Solution_) * kPageSizeInTypeSize));
            assert(page != nullptr);
            assert(_last_page == prev_page);
            assert(_pages[cur_page] == nullptr);
            _pages[cur_page] = page;

            /* update page counter */
            _last_page = _last_page + 1;
            __threadfence();
        } else {
            /* otherwise we should just wait for our page be allocated if it is not */
            while (_last_page < cur_page) {
                // spin
            }
            assert(_last_page >= cur_page);
        }

        /* save sol */
        auto &sol = (*this)[my_address];
        sol.idx1 = idx1;
        sol.idx2 = idx2;
    }

    [[nodiscard]] FAST_CALL_ALWAYS Solution_ &operator[](const uint32_t idx) {
        assert(idx < _last_page_offset);
        assert(_pages[idx / kPageSizeInTypeSize] != nullptr);
        assert(idx != 0);
        return _pages[idx >> kPageDivider][idx & kPageRemainder];
    }

    [[nodiscard]] FAST_CALL_ALWAYS const Solution_ &operator[](const uint32_t idx) const {
        assert(idx < _last_page_offset);
        assert(_pages[idx / kPageSizeInTypeSize] != nullptr);
        assert(idx != 0);
        return _pages[idx >> kPageDivider][idx & kPageRemainder];
    }

    cuda_Solution* DumpToGPU();

    static std::vector<std::tuple<uint32_t, uint32_t> > DeallocGPU(cuda_Solution *d_solution);

    static void DeallocHost(const cuda_Solution *h_solution);

    // ------------------------------
    // Class fields
    // ------------------------------
protected:
    alignas(8) Solution_ **_pages{};
    alignas(4) volatile uint32_t _last_page{};
    alignas(4) volatile uint32_t _last_page_offset{};
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
