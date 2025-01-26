/* internal includes */
#include <data.cuh>

/* external includes */
#include <barrier>
#include <global_conf.cuh>
#include <iostream>
#include <memory>

__device__ volatile int sem;

std::tuple<cuda_Solution *, uint32_t *> cuda_Solution::DumpToGPU(const size_t num_solutions) {
    uint32_t *d_data{};
    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_data, GetMemBlockSize(num_solutions), g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(
        cudaMemsetAsync(d_data, UINT32_MAX, GetMemBlockSize(num_solutions), g_cudaGlobalConf->asyncStream));

    cuda_Solution *d_solutions{};
    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_solutions, sizeof(cuda_Solution), g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(
        cudaMemcpyAsync(&d_solutions->_data, &d_data, sizeof(uint32_t *), cudaMemcpyHostToDevice, g_cudaGlobalConf->
            asyncStream));

    return {d_solutions, d_data};
}

// ------------------------------
// Cuda data functions
// ------------------------------

cuda_Data::cuda_Data(const uint32_t num_sequences, const uint32_t max_sequence_length): _num_sequences(num_sequences),
    _num_sequences_padded32(
        num_sequences + (32 - (num_sequences % 32)) % 32),
    _max_sequence_length(max_sequence_length) {
    _data = new uint32_t[_num_sequences_padded32 * (_max_sequence_length + 1)];

    std::cout << "Num sequences: " << _num_sequences << std::endl;
    std::cout << "Num sequences padded: " << _num_sequences_padded32 << std::endl;

    size_t total_mem_used = _num_sequences_padded32 * (_max_sequence_length + 1) * sizeof(uint32_t);
    std::cout << "Allocated " << total_mem_used / (1024 * 1024) << " mega bytes for sequences" << std::endl;
}

cuda_Data::cuda_Data(const BinSequencePack &pack): cuda_Data(pack.sequences.size(),
                                                             (pack.max_seq_size_bits + 63) / 32) {
    std::cout << "Max sequence size: " << pack.max_seq_size_bits << std::endl;

    static constexpr uint64_t kBitMask32 = ~static_cast<uint32_t>(0);

    for (size_t seq_idx = 0; seq_idx < pack.sequences.size(); ++seq_idx) {
        const auto &sequence = pack.sequences[seq_idx];
        auto fetcher = (*this)[seq_idx];
        fetcher.GetSequenceLength() = sequence.GetSizeBits();

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

cuda_Data *cuda_Data::DumpToGPU() const {
    /* allocate manager object */
    cuda_Data *d_data;

    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_data, sizeof(cuda_Data), g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(
        cudaMemcpyAsync(d_data, this, sizeof(cuda_Data), cudaMemcpyHostToDevice, g_cudaGlobalConf->asyncStream));

    /* allocate data itself */
    uint32_t *d_data_data;

    const size_t data_size = _num_sequences_padded32 * (_max_sequence_length + 1) * sizeof(uint32_t);
    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_data_data, data_size, g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(
        cudaMemcpyAsync(d_data_data, _data, data_size, cudaMemcpyHostToDevice, g_cudaGlobalConf->asyncStream));

    /* update manager object */
    CUDA_ASSERT_SUCCESS(
        cudaMemcpyAsync(&d_data->_data, &d_data_data, sizeof(uint32_t *), cudaMemcpyHostToDevice, g_cudaGlobalConf->
            asyncStream));

    return d_data;
}

uint32_t *cuda_Data::GetDataPtrHost(const cuda_Data *d_data) {
    uint32_t *ptr;
    CUDA_ASSERT_SUCCESS(
        cudaMemcpyAsync(&ptr, &d_data->_data, sizeof(uint32_t *), cudaMemcpyDeviceToHost, g_cudaGlobalConf->asyncStream
        ));
    return ptr;
}

void cuda_Data::DeallocGPU(cuda_Data *d_data) {
    uint32_t *d_data_ptr = GetDataPtrHost(d_data);
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_data_ptr, g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_data, g_cudaGlobalConf->asyncStream));
}
