/* internal includes */
#include <data.cuh>

/* external includes */
#include <barrier>
#include <global_conf.cuh>
#include <iostream>
#include <memory>

__device__ volatile int sem;

cuda_Solution::cuda_Solution() {
    _pages = new Solution_ *[kMaxPages]{};
    _pages[0] = static_cast<Solution_ *>(malloc(kPageSize));

    _last_page = 0;
    _last_page_offset = 1;
}

cuda_Solution *cuda_Solution::DumpToGPU() {
    Solution_ **d_pages;
    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_pages, kMaxPages * sizeof(Solution_ *), g_cudaGlobalConf->asyncStream));

    auto **h_pages = new Solution_ *[kMaxPages]{};
    for (uint32_t i = 0; i < kMaxPages; ++i) {
        if (_pages[i] != nullptr) {
            CUDA_ASSERT_SUCCESS(cudaMallocAsync(&h_pages[i], kPageSize, g_cudaGlobalConf->asyncStream));
            CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(h_pages[i], _pages[i], kPageSize, cudaMemcpyHostToDevice,
                g_cudaGlobalConf->asyncStream));
        }
    }

    // Copy the pages array to the GPU
    CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(d_pages, h_pages, kMaxPages * sizeof(Solution_ *), cudaMemcpyHostToDevice,
        g_cudaGlobalConf->asyncStream));

    // Allocate the solution object
    cuda_Solution *d_solution;
    CUDA_ASSERT_SUCCESS(cudaMallocAsync(&d_solution, sizeof(cuda_Solution), g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(d_solution, this, sizeof(cuda_Solution), cudaMemcpyHostToDevice,
        g_cudaGlobalConf->asyncStream));

    // update the pages array
    CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(&d_solution->_pages, &d_pages, sizeof(Solution_ **), cudaMemcpyHostToDevice,
        g_cudaGlobalConf->asyncStream));

    delete[] h_pages;
    return d_solution;
}

std::vector<std::tuple<uint32_t, uint32_t> > cuda_Solution::DeallocGPU(cuda_Solution *d_solution) {
    // Move solutions back to cpu
    auto h_solution = static_cast<cuda_Solution *>(malloc(sizeof(cuda_Solution)));

    CUDA_ASSERT_SUCCESS(
        cudaMemcpyAsync(h_solution, d_solution, sizeof(cuda_Solution), cudaMemcpyDeviceToHost, g_cudaGlobalConf->
            asyncStream));

    // Move pages back to cpu
    auto h_pages = new Solution_ *[kMaxPages]{};

    CUDA_ASSERT_SUCCESS(cudaMemcpyAsync(h_pages, h_solution->_pages, kMaxPages * sizeof(Solution_ *),
        cudaMemcpyDeviceToHost, g_cudaGlobalConf->asyncStream));

    CUDA_ASSERT_SUCCESS(cudaStreamSynchronize(g_cudaGlobalConf->asyncStream));
    auto d_pages = h_solution->_pages;

    for (uint32_t i = 0; i < kMaxPages; ++i) {
        if (h_pages[i] != nullptr) {
            auto p = static_cast<Solution_ *>(malloc(kPageSize));

            CUDA_ASSERT_SUCCESS(
                cudaMemcpyAsync(p, h_pages[i], kPageSize, cudaMemcpyDeviceToHost, g_cudaGlobalConf->
                    asyncStream));
            CUDA_ASSERT_SUCCESS(cudaFreeAsync(h_pages[i], g_cudaGlobalConf->asyncStream));

            h_pages[i] = p;
        }
    }
    h_solution->_pages = h_pages;

    // Move solutions back to cpu
    std::vector<std::tuple<uint32_t, uint32_t> > results{};

    for (uint32_t idx = 1; idx < h_solution->_last_page_offset; ++idx) {
        auto &sol = (*h_solution)[idx];
        results.emplace_back(sol.idx1, sol.idx2);
    }

    // Free the pages on CPU
    DeallocHost(h_solution);

    // Free the solution on CPU
    free(h_solution);

    CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_solution, g_cudaGlobalConf->asyncStream));
    CUDA_ASSERT_SUCCESS(cudaFreeAsync(d_pages, g_cudaGlobalConf->asyncStream));

    return results;
}

void cuda_Solution::DeallocHost(const cuda_Solution *h_solution) {
    for (uint32_t i = 0; i <= h_solution->_last_page; ++i) {
        if (h_solution->_pages[i] != nullptr) {
            free(h_solution->_pages[i]);
        }
    }

    // Free the pages array
    delete[] h_solution->_pages;
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
