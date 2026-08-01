#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Minimal fixed-size worker pool: enqueue fire-and-forget jobs, wait for the
// queue to drain. Deliberately has no job dependencies, priorities, or work
// stealing — higher-level systems (e.g. world::ChunkMeshScheduler) own all
// scheduling policy and use this purely as an executor, keeping lifecycle
// state explicit instead of encoding it in a job graph.
//
// threadCount == 0 selects synchronous mode: enqueue() runs the job inline on
// the calling thread. Tests use this for deterministic, single-threaded runs
// of the exact same scheduling code.
namespace odai::core {

class JobSystem {
public:
    explicit JobSystem(unsigned threadCount);
    ~JobSystem();

    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    void enqueue(std::function<void()> job);
    // Blocks until the queue is empty and no worker is executing a job.
    void waitIdle();
    [[nodiscard]] std::size_t workerCount() const { return m_workers.size(); }

private:
    void workerLoop();

    std::vector<std::thread> m_workers;
    std::deque<std::function<void()>> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_wakeCv;
    std::condition_variable m_idleCv;
    std::size_t m_activeJobCount = 0;
    bool m_stopRequested = false;
};

} // namespace odai::core
