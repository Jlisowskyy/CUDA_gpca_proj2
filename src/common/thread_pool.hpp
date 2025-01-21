#ifndef SRC_THREADPOOL_CUH
#define SRC_THREADPOOL_CUH

#include <thread>
#include <vector>
#include <cassert>
#include <memory>
#include <semaphore>
#include <mutex>

class ThreadPool {
    // ------------------------------
    // Internal types
    // ------------------------------

    using sem_t = std::counting_semaphore<UINT16_MAX>;

    struct _taskBase {
        explicit _taskBase(const uint32_t thread_idx, sem_t *sem): m_threadIdx(thread_idx), m_sem(sem) {
        }

        virtual ~_taskBase() = default;

        void Run() {
            _run(m_threadIdx);
            m_sem->release();
        }

    protected:
        virtual void _run(uint32_t thread_idx) = 0;

        uint32_t m_threadIdx;
        sem_t *m_sem;
    };

    template<class FuncT, class... Args>
    struct _task final : _taskBase {
        FuncT m_func;
        std::tuple<Args...> m_args;

        _task(sem_t *sem, const uint32_t thread_idx, FuncT &&func, Args &&... args)
            : _taskBase(thread_idx, sem), m_func(std::forward<FuncT>(func)), m_args(std::forward<Args>(args)...) {
        }

        void _run(uint32_t thread_idx) override {
            std::apply(m_func, std::tuple_cat(std::make_tuple(thread_idx), m_args));
        }
    };

public:
    // ------------------------------
    // Class creation
    // ------------------------------

    explicit ThreadPool(const uint32_t numThreads) : m_localJobsSemaphore(0), m_numThreads(numThreads) {
        /* extend global thread pool if needed */
        _tryExtendingGlobalWorkers(numThreads);
    }

    ~ThreadPool() {
        assert(m_wasWaited && "Detected early destruction of thread pool");
    }

    // ------------------------------
    // Class interaction
    // ------------------------------

    template<class FuncT, class... Args>
    void RunThreads(FuncT &&func, Args &&... args) {
        assert(!m_wasRun && "Detected double run on thread pool");

        for (uint32_t thread_idx = 0; thread_idx < m_numThreads; ++thread_idx) {
            std::lock_guard lock(m_globalMutex);

            m_globalTasks.emplace_back(
                std::make_unique<_task<FuncT, Args...> >(&m_localJobsSemaphore,
                                                         thread_idx,
                                                         std::forward<FuncT>(func),
                                                         std::forward<Args>(args)...)
            );
        }

        m_globalJobsSemaphore.release(m_numThreads);
        m_wasRun = true;
    }

    void Wait() {
        assert(!m_wasWaited && "Detected double wait on thread pool");
        assert(m_wasRun && "Detected wait without run on thread pool");

        m_wasWaited = true;

        for (uint32_t retry = 0; retry < m_numThreads; ++retry) {
            m_localJobsSemaphore.acquire();
        }
    }

    void Reset() {
        Reset(m_numThreads);
    }

    void Reset(const uint32_t numThreads) {
        assert(m_wasRun && m_wasWaited && "Detected reset without wait on thread pool");

        if (numThreads > m_numThreads) {
            _tryExtendingGlobalWorkers(numThreads);
        }

        m_numThreads = numThreads;
        m_wasRun = false;
        m_wasWaited = false;
    }

    // ------------------------------------------
    // static global thread pool management
    // ------------------------------------------

    static void InitGlobalWorkers() {
        _tryExtendingGlobalWorkers(std::thread::hardware_concurrency());
    }

protected:
    static void _tryExtendingGlobalWorkers(const size_t num_workers) {
        std::lock_guard lock(m_globalMutex);

        if (m_globalThreads.size() < num_workers) {
            _addGlobalWorkers(num_workers - m_globalThreads.size());
        }
    }

    static void _addGlobalWorkers(size_t num_workers) {
        while (num_workers-- > 0) {
            m_globalThreads.emplace_back(
                new std::thread(_globalWorkerThread),
                [](std::thread *pThread) {
                    /* allows automatic cleanup of threads on program stop */
                    m_shouldStop.store(true);
                    m_globalJobsSemaphore.release(static_cast<ptrdiff_t>(m_globalThreads.size()));
                    pThread->join();
                    delete pThread;
                }
            );
        }
    }

    static void _globalWorkerThread() {
        while (!m_shouldStop) {
            m_globalJobsSemaphore.acquire();

            /* perform sanity check */
            if (m_shouldStop) {
                break;
            }

            /* load task */
            std::unique_ptr<_taskBase> task; {
                std::lock_guard lock(m_globalMutex);
                assert(!m_globalTasks.empty());

                task = std::move(m_globalTasks.back());
                m_globalTasks.pop_back();
            }

            /* run task */
            assert(task);
            task->Run();
        }
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    static std::vector<std::shared_ptr<std::thread> > m_globalThreads;
    static volatile std::atomic<bool> m_shouldStop;
    static sem_t m_globalJobsSemaphore;
    static std::mutex m_globalMutex;
    static std::vector<std::unique_ptr<_taskBase> > m_globalTasks;

    sem_t m_localJobsSemaphore;
    uint32_t m_numThreads;
    bool m_wasWaited = false;
    bool m_wasRun = false;
};

#endif //SRC_THREADPOOL_CUH
