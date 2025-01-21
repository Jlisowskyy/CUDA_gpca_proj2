#include <thread_pool.hpp>

std::vector<std::shared_ptr<std::thread> > ThreadPool::m_globalThreads{};
volatile std::atomic<bool> ThreadPool::m_shouldStop{false};
ThreadPool::sem_t ThreadPool::m_globalJobsSemaphore{0};
std::mutex ThreadPool::m_globalMutex{};
std::vector<std::unique_ptr<ThreadPool::_taskBase>> ThreadPool::m_globalTasks{};
