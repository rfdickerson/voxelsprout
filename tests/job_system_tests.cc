#include "core/job_system.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        ++g_failures;
        std::cerr << "[job system test] FAILED: " << message << "\n";
    }
}

void testSynchronousModeRunsInlineInOrder() {
    odai::core::JobSystem jobs(0);
    expectTrue(jobs.workerCount() == 0, "synchronous mode has no workers");

    std::vector<int> order;
    for (int i = 0; i < 8; ++i) {
        jobs.enqueue([&order, i]() { order.push_back(i); });
    }
    jobs.waitIdle();
    expectTrue(order.size() == 8u, "synchronous mode ran every job");
    bool inOrder = true;
    for (int i = 0; i < 8; ++i) {
        inOrder = inOrder && order[static_cast<std::size_t>(i)] == i;
    }
    expectTrue(inOrder, "synchronous mode runs jobs inline in submission order");
}

void testThreadedJobsAllRun() {
    odai::core::JobSystem jobs(4);
    expectTrue(jobs.workerCount() == 4, "threaded mode spawned requested workers");

    std::atomic<int> counter{0};
    constexpr int kJobCount = 200;
    for (int i = 0; i < kJobCount; ++i) {
        jobs.enqueue([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    jobs.waitIdle();
    expectTrue(counter.load() == kJobCount, "waitIdle observes every enqueued job completed");

    // The pool stays usable after an idle wait.
    jobs.enqueue([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
    jobs.waitIdle();
    expectTrue(counter.load() == kJobCount + 1, "pool accepts work after waitIdle");
}

void testDestructorDrainsQueuedJobs() {
    std::atomic<int> counter{0};
    constexpr int kJobCount = 64;
    {
        odai::core::JobSystem jobs(2);
        for (int i = 0; i < kJobCount; ++i) {
            jobs.enqueue([&counter]() {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                counter.fetch_add(1, std::memory_order_relaxed);
            });
        }
        // No waitIdle: the destructor must drain the queue before joining.
    }
    expectTrue(counter.load() == kJobCount, "destructor drains queued jobs before joining");
}

void testWaitIdleWithNoWork() {
    odai::core::JobSystem jobs(2);
    jobs.waitIdle();
    expectTrue(true, "waitIdle with an empty queue returns immediately");
}

void testSlowCaptureDestructionDoesNotBlockEnqueue() {
    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        bool destroying = false;
        bool release = false;
    } state;
    struct Capture {
        State& state;
        ~Capture() {
            std::unique_lock lock(state.mutex);
            state.destroying = true;
            state.cv.notify_all();
            state.cv.wait(lock, [&] { return state.release; });
        }
    };

    odai::core::JobSystem jobs(1);
    jobs.enqueue([capture = std::make_shared<Capture>(state)] {});
    {
        std::unique_lock lock(state.mutex);
        expectTrue(state.cv.wait_for(lock, std::chrono::seconds(2),
            [&] { return state.destroying; }), "worker entered capture destruction");
    }

    std::promise<void> submitted;
    auto finished = submitted.get_future();
    std::thread producer([&] {
        jobs.enqueue([] {});
        submitted.set_value();
    });
    const bool queuedDuringCleanup =
        finished.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    {
        std::lock_guard lock(state.mutex);
        state.release = true;
    }
    state.cv.notify_all();
    producer.join();
    jobs.waitIdle();
    expectTrue(queuedDuringCleanup, "slow capture destruction does not hold queue mutex");
}

} // namespace

int main() {
    testSynchronousModeRunsInlineInOrder();
    testThreadedJobsAllRun();
    testDestructorDrainsQueuedJobs();
    testWaitIdleWithNoWork();
    testSlowCaptureDestructionDoesNotBlockEnqueue();

    if (g_failures != 0) {
        std::cerr << "[job system test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[job system test] all checks passed\n";
    return 0;
}
