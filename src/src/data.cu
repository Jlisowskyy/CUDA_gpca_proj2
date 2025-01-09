#include <data.cuh>

std::tuple<cuda_Solution *, uint32_t *> cuda_Solution::DumpToGPU(const size_t num_solutions) {
    uint32_t *d_data{};
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_data, GetMemBlockSize(num_solutions)));
    CUDA_ASSERT_SUCCESS(cudaMemset(d_data, INT_MAX, GetMemBlockSize(num_solutions)));

    cuda_Solution *d_solutions{};
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_solutions, sizeof(cuda_Solution)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(d_solutions->_data, &d_data, sizeof(uint32_t *), cudaMemcpyHostToDevice));

    return {d_solutions, d_data};
}

cuda_Allocator::cuda_Allocator(const uint32_t max_nodes, const uint32_t max_threads,
                               const uint32_t max_node_per_thread): _max_nodes(max_nodes),
                                                                    _max_threads(max_threads),
                                                                    _max_node_per_thread(max_node_per_thread) {
    _data = new Node_[max_nodes + 1];
    _node_counters = new uint32_t[max_threads];
    _thread_nodes = new uint32_t[max_threads];

    uint32_t last_node = 1;
    /* prepare data for nodes */
    for (uint32_t t_idx = 0; t_idx < max_threads; ++t_idx) {
        uint32_t t_node_idx = _thread_nodes[t_idx] = last_node++;
        _node_counters[t_idx] = max_node_per_thread;

        /* prepare nodes */
        assert(last_node <= max_nodes && "DETECTED OVERFLOW");
        for (uint32_t node_idx = 0; node_idx < max_node_per_thread; ++node_idx) {
            _data[t_node_idx].seq_idx = UINT32_MAX;
            _data[t_node_idx].next[0] = last_node;
            t_node_idx = last_node++;

            assert(last_node <= max_nodes && "DETECTED OVERFLOW");
        }
    }

    _last_node = last_node;
}

cuda_Allocator *cuda_Allocator::DumpToGPU() const {
    cuda_Allocator *d_allocator;

    /* copy object */
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_allocator, sizeof(cuda_Allocator)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(d_allocator, this, sizeof(cuda_Allocator), cudaMemcpyHostToDevice));

    /* copy data */
    Node_ *d_data;

    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_data, (_max_nodes + 1) * sizeof(Node_)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(d_data, _data, (_max_nodes + 1) * sizeof(Node_), cudaMemcpyHostToDevice));

    /* copy node_counters */
    uint32_t *d_node_counters;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_node_counters, _max_threads * sizeof(uint32_t)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(d_node_counters, _node_counters, _max_threads * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    /* copy thread_nodes */
    uint32_t *d_thread_nodes;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_thread_nodes, _max_threads * sizeof(uint32_t)));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(d_thread_nodes, _thread_nodes, _max_threads * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    uint32_t *d_idxes;
    CUDA_ASSERT_SUCCESS(cudaMalloc(&d_idxes, _max_threads * sizeof(uint32_t)));

    /* update object */
    CUDA_ASSERT_SUCCESS(cudaMemcpy(&d_allocator->_data, &d_data, sizeof(Node_ *), cudaMemcpyHostToDevice));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(&d_allocator->_node_counters, &d_node_counters, sizeof(uint32_t *),
        cudaMemcpyHostToDevice));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(&d_allocator->_thread_nodes, &d_thread_nodes, sizeof(uint32_t *),
        cudaMemcpyHostToDevice));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(&d_allocator->_d_idxes, &d_idxes, sizeof(uint32_t *), cudaMemcpyHostToDevice));

    return d_allocator;
}

void cuda_Allocator::DeallocGPU(cuda_Allocator *d_allocator) {
    Node_ *d_data;
    uint32_t *d_node_counters;
    uint32_t *d_thread_nodes;

    CUDA_ASSERT_SUCCESS(cudaMemcpy(&d_data, &d_allocator->_data, sizeof(Node_ *), cudaMemcpyDeviceToHost));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(&d_node_counters, &d_allocator->_node_counters, sizeof(uint32_t *),
        cudaMemcpyDeviceToHost));
    CUDA_ASSERT_SUCCESS(cudaMemcpy(&d_thread_nodes, &d_allocator->_thread_nodes, sizeof(uint32_t *),
        cudaMemcpyDeviceToHost));

    CUDA_ASSERT_SUCCESS(cudaFree(d_data));
    CUDA_ASSERT_SUCCESS(cudaFree(d_node_counters));
    CUDA_ASSERT_SUCCESS(cudaFree(d_thread_nodes));
    CUDA_ASSERT_SUCCESS(cudaFree(d_allocator));
}

// ------------------------------
// GPU functions
// ------------------------------

void cuda_Allocator::Consolidate(const uint32_t t_idx) {
    /* wait for all threads to finish using allocator */
    __syncthreads();

    /* first thread will update global data */
    if (t_idx == 0) {
        _prepareIdxes();
    }

    /* wait for first thread to update global data */
    __syncthreads();

    /* each of threads will clean up its allocator space */
    _cleanUpOwnSpace(t_idx);

    /* No need to wait as each thread is working on its own space */
}

void cuda_Allocator::_prepareIdxes() {
    uint32_t last_node = _last_node;

    for (uint32_t t_idx = 0; t_idx < _max_threads; ++t_idx) {
        _d_idxes[t_idx] = last_node;

        last_node += (_max_node_per_thread - _node_counters[t_idx]);
    }

    _last_node = last_node;
}

void cuda_Allocator::_cleanUpOwnSpace(const uint32_t t_idx) {
}

// ------------------------------
// Cuda data functions
// ------------------------------

cuda_Data::cuda_Data(const BinSequencePack &pack): cuda_Data(pack.sequences.size(),
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

cuda_Data * cuda_Data::DumpToGPU() const {
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

uint32_t * cuda_Data::GetDataPtrHost(const cuda_Data *d_data) {
    uint32_t *ptr;
    CUDA_ASSERT_SUCCESS(cudaMemcpy(&ptr, d_data->_data, sizeof(uint32_t *), cudaMemcpyDeviceToHost));
    return ptr;
}

void cuda_Data::DeallocGPU(cuda_Data *d_data) {
    uint32_t *d_data_ptr = GetDataPtrHost(d_data);
    CUDA_ASSERT_SUCCESS(cudaFree(d_data_ptr));
    CUDA_ASSERT_SUCCESS(cudaFree(d_data));
}
