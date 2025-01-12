#ifndef SRC_THREADPOOL_CUH
#define SRC_THREADPOOL_CUH

#include <thread>
#include <vector>
#include <cassert>

/**
 * @brief Manages a dynamic pool of worker threads
 *
 * Provides thread creation, execution, and synchronization utilities
 *
 * @note Not thread safe itself should be run only within single thread
 */
class ThreadPool {
public:
    // ------------------------------
    // Internal types
    // ------------------------------

    /**
     * @brief Constant indicating invalid thread number
     */
    static constexpr uint32_t INVALID_THREAD_NUM = 0;

    // ------------------------------
    // Class creation
    // ------------------------------

    /**
     * @brief Construct a thread pool with specified number of threads
     *
     * @param numThreads Number of threads to manage
     */
    explicit ThreadPool(const uint32_t numThreads) : m_numThreadsToSpawn(numThreads) {
        assert(m_numThreadsToSpawn != INVALID_THREAD_NUM && "ThreadPool: numThreads cannot be 0");
    }

    ~ThreadPool() {
        for (std::thread *pThread: m_threads) {
            if (pThread) {
                std::abort();
            }
        }
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    /**
    * @brief Run threads with a given function and arguments
    *
    * @tparam FuncT Function type
    * @tparam Args Argument types
    * @param func Function to execute
    * @param args Function arguments
    *
    * @note it can be run only once after creation or reset, its safety measure
    */
    template<class FuncT, class... Args>
    void RunThreads(FuncT&& func, Args&&... args) {
        assert(m_numThreadsToSpawn != INVALID_THREAD_NUM && "Detected second run usage on the thread pool");

        m_threads.reserve(m_numThreadsToSpawn);
        for (uint32_t idx = 0; idx < m_numThreadsToSpawn; ++idx) {
            m_threads.push_back(new std::thread(func, idx, std::forward<Args>(args)...));
        }

        m_numThreadsToSpawn = INVALID_THREAD_NUM;
    }

    /**
     * @brief Wait for all threads to complete
     */
    void Wait() {
        assert(m_numThreadsToSpawn == INVALID_THREAD_NUM && "Detected early wait on thread pool");

        while (!m_threads.empty()) {
            std::thread *pThread = m_threads.back();
            m_threads.pop_back();

            pThread->join();
            delete pThread;
        }
    }

    /**
     * @brief Reset thread pool with new thread count,
     *
     * @param numThreads New number of threads
     */
    void Reset(const uint32_t numThreads) {
        assert(m_threads.empty() && "Detected early reset on thread pool");

        m_numThreadsToSpawn = numThreads;
    }

    // ------------------------------
    // Protected methods
    // ------------------------------
protected:

    // ------------------------------
    // Class fields
    // ------------------------------

    uint32_t m_numThreadsToSpawn{};
    std::vector<std::thread *> m_threads{};
};

#endif //SRC_THREADPOOL_CUH
