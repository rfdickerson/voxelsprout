#include "core/job_system.h"

#include <atomic>
#include <chrono>
#include <iostream>
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

} // namespace

int main() {
    testSynchronousModeRunsInlineInOrder();
    testThreadedJobsAllRun();
    testDestructorDrainsQueuedJobs();
    testWaitIdleWithNoWork();

    if (g_failures != 0) {
        std::cerr << "[job system test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[job system test] all checks passed\n";
    return 0;
}
