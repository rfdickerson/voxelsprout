#pragma once

#include "core/ring_buffer.h"

#include <chrono>
#include <cstddef>

// Core FrameProfiler subsystem
// Responsible for: wall-clock measurement primitives and bounded per-zone
// timing windows for CPU hot paths.
// Should NOT do: own a global zone registry, allocate per sample, sleep, or
// know anything about the GPU (render/ already owns timestamp queries).
namespace odai::core {

// steady_clock wrapper. Deliberately NOT the simulation clock: this measures
// wall time for telemetry only, so nothing here may feed gameplay state or
// determinism.
class Stopwatch {
public:
    Stopwatch() : m_start(Clock::now()) {}

    void restart() { m_start = Clock::now(); }

    [[nodiscard]] float elapsedMs() const {
        const std::chrono::duration<float, std::milli> delta = Clock::now() - m_start;
        return delta.count();
    }

    // elapsedMs() and restart() against a single clock read, so back-to-back
    // zones cannot lose (or double-count) the gap between two separate reads.
    float lapMs() {
        const Clock::time_point now = Clock::now();
        const std::chrono::duration<float, std::milli> delta = now - m_start;
        m_start = now;
        return delta.count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point m_start;
};

// RAII zone timer: folds its own lifetime into a caller-owned float on scope
// exit. Accumulates (+=) rather than assigns so a zone entered several times
// in one frame sums instead of reporting only its last visit.
class ScopedTimerMs {
public:
    explicit ScopedTimerMs(float& sinkMs) : m_sink(&sinkMs) {}
    ~ScopedTimerMs() { *m_sink += m_watch.elapsedMs(); }

    ScopedTimerMs(const ScopedTimerMs&) = delete;
    ScopedTimerMs& operator=(const ScopedTimerMs&) = delete;
    ScopedTimerMs(ScopedTimerMs&&) = delete;
    ScopedTimerMs& operator=(ScopedTimerMs&&) = delete;

private:
    Stopwatch m_watch;
    float* m_sink;
};

// One zone's rolling sample window: newest value, EWMA, and nearest-rank
// percentiles over the retained history. Fixed capacity, never allocates.
//
// Percentiles are computed on demand rather than cached per push. Each call
// copies and partially sorts the window, so it is the caller's job to only ask
// when something is actually going to read the number -- a hidden per-frame
// sort across every zone is exactly the "hidden work in hot paths" this
// codebase's performance contract rules out.
template <std::size_t Capacity>
class TimingChannel {
public:
    // Matches the renderer's existing CPU frame-time smoothing so overlay
    // numbers read consistently across the two lineages.
    static constexpr float kEwmaAlpha = 0.1f;

    void push(float sampleMs) {
        // First sample seeds the average outright; otherwise a zone starts at
        // 0 and spends ~20 frames climbing to its true cost.
        m_ewma = m_history.empty() ? sampleMs : (kEwmaAlpha * sampleMs + (1.0f - kEwmaAlpha) * m_ewma);
        m_last = sampleMs;
        m_history.push(sampleMs);
    }

    void clear() {
        m_history.clear();
        m_last = 0.0f;
        m_ewma = 0.0f;
    }

    [[nodiscard]] float lastMs() const noexcept { return m_last; }
    [[nodiscard]] float ewmaMs() const noexcept { return m_ewma; }
    [[nodiscard]] std::size_t sampleCount() const noexcept { return m_history.size(); }
    [[nodiscard]] const RingBuffer<float, Capacity>& history() const noexcept { return m_history; }

    [[nodiscard]] float percentileMs(float percentile01) const {
        return percentile(m_history, percentile01);
    }
    [[nodiscard]] float p50Ms() const { return percentileMs(0.50f); }
    [[nodiscard]] float p95Ms() const { return percentileMs(0.95f); }
    [[nodiscard]] float p99Ms() const { return percentileMs(0.99f); }

    // Worst retained sample. Cheap linear scan -- the spike is usually what a
    // frame-pacing investigation is actually hunting for.
    [[nodiscard]] float maxMs() const {
        float worst = 0.0f;
        for (std::size_t i = 0; i < m_history.size(); ++i) {
            if (m_history[i] > worst) worst = m_history[i];
        }
        return worst;
    }

private:
    RingBuffer<float, Capacity> m_history{};
    float m_last = 0.0f;
    float m_ewma = 0.0f;
};

}  // namespace odai::core
