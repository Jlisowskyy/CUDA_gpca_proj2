#include <cuda_tests.cuh>
#include <data.cuh>

#include <iostream>

void TestCudaData(const BinSequencePack &data) {
    const cuda_Data cuda_data(data);

    if (data.sequences.size() != cuda_data.GetNumSequences()) {
        std::cerr << "Data sizes mismatch" << std::endl;
        std::cout << "Cuda data size: " << cuda_data.GetNumSequences() << std::endl;
        std::cout << "Host data size: " << data.sequences.size() << std::endl;
        return;
    }

    for (size_t idx = 0; idx < data.sequences.size(); ++idx) {
        auto sequence = data.sequences[idx];
        auto cuda_sequence = cuda_data[idx];

        if (sequence.GetSizeBits() != cuda_sequence.GetSequenceLength()) {
            std::cerr << "Sequence sizes mismatch" << std::endl;
            std::cout << "Cuda sequence size: " << cuda_sequence.GetSequenceLength() << std::endl;
            std::cout << "Host sequence size: " << sequence.GetSizeBits() << std::endl;
            return;
        }
        for (size_t bit_idx = 0; bit_idx < sequence.GetSizeBits(); ++bit_idx) {
            if (sequence.GetBit(bit_idx) != cuda_sequence.GetBit(bit_idx)) {
                std::cerr << "Bit mismatch" << std::endl;
                std::cout << "Bit idx: " << bit_idx << std::endl;
                std::cout << "Cuda bit: " << cuda_sequence.GetBit(bit_idx) << std::endl;
                std::cout << "Host bit: " << sequence.GetBit(bit_idx) << std::endl;
                return;
            }
        }
    }

    /* test compare */
    for (size_t idx = 0; idx < data.sequences.size(); ++idx) {
        const size_t next_idx = (idx + 1) % data.sequences.size();

        /* compare with itself */
        if (!cuda_data[idx].Compare(cuda_data[idx])) {
            std::cerr << "Self compare failed" << std::endl;
            return;
        }

        /* compare with next */
        if (cuda_data[idx].Compare(cuda_data[next_idx])) {
            std::cerr << "Next compare failed" << std::endl;
            return;
        }
    }

    std::cout << "Test passed" << std::endl;
}
