#ifndef DATA_HPP
#define DATA_HPP

/* external includes */
#include <cinttypes>
#include <cassert>
#include <vector>
#include <utility>

class BinSequence {
public:
    explicit BinSequence(const size_t size_bits) : size_bits_(size_bits) {
        size_words_ = (size_bits + 63) / 64;
        data_ = new uint64_t[size_words_];
    }

    ~BinSequence() {
        delete [] data_;
    }

    [[nodiscard]] size_t GetSizeBits() const {
        return size_bits_;
    }

    [[nodiscard]] size_t GetSizeWords() const {
        return size_words_;
    }

    [[nodiscard]] const uint64_t *GetData() const {
        return data_;
    }

    [[nodiscard]] bool GetBit(const size_t idx) const {
        assert(idx < size_bits_);
        const size_t word_idx = idx / 64;
        const size_t bit_idx = idx % 64;

        return (data_[word_idx] >> bit_idx) & 1ULL;
    }

    void SetBit(const size_t idx, const bool value) {
        assert(idx < size_bits_);
        const size_t word_idx = idx / 64;
        const size_t bit_idx = idx % 64;

        data_[word_idx] &= ~(1ULL << bit_idx);
        data_[word_idx] |= (static_cast<uint64_t>(value) << bit_idx);
    }

    [[nodiscard]] const uint64_t &GetWord(const size_t idx) const {
        assert(idx < size_words_);
        return data_[idx];
    }

    [[nodiscard]] uint64_t &GetWord(const size_t idx) {
        assert(idx < size_words_);
        return data_[idx];
    }

protected:
    size_t size_bits_;
    size_t size_words_;
    uint64_t *data_;
};

struct BinSequencePack {
    std::vector<BinSequence> sequences;
    std::vector<std::pair<size_t, size_t> > solution;
};

#endif //DATA_HPP
