#include "core/job_system.h"

#include <utility>

namespace odai::core {

JobSystem::JobSystem(unsigned threadCount) {
    m_workers.reserve(threadCount);
    for (unsigned i = 0; i < threadCount; ++i) {
        m_workers.emplace_back([this]() { workerLoop(); });
    }
}

JobSystem::~JobSystem() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stopRequested = true;
    }
    m_wakeCv.notify_all();
    for (std::thread& worker : m_workers) {
        worker.join();
    }
}

void JobSystem::enqueue(std::function<void()> job) {
    if (m_workers.empty()) {
        job();
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push_back(std::move(job));
    }
    m_wakeCv.notify_one();
}

void JobSystem::waitIdle() {
    if (m_workers.empty()) {
        return;
    }
    std::unique_lock<std::mutex> lock(m_mutex);
    m_idleCv.wait(lock, [this]() { return m_queue.empty() && m_activeJobCount == 0; });
}

void JobSystem::workerLoop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    for (;;) {
        m_wakeCv.wait(lock, [this]() { return m_stopRequested || !m_queue.empty(); });
        if (m_stopRequested && m_queue.empty()) {
            return;
        }
        std::function<void()> job = std::move(m_queue.front());
        m_queue.pop_front();
        ++m_activeJobCount;
        lock.unlock();
        job();
        lock.lock();
        --m_activeJobCount;
        if (m_queue.empty() && m_activeJobCount == 0) {
            m_idleCv.notify_all();
        }
    }
}

} // namespace odai::core
