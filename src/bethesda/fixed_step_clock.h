#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace odai::bethesda {

struct FixedStepConfig {
    double stepSeconds = 1.0 / 60.0;
    double maxFrameDeltaSeconds = 0.25;
    std::uint32_t maxCatchUpSteps = 8u;
};

struct FixedStepResult {
    std::uint32_t steps = 0u;
    std::uint32_t droppedSteps = 0u;
    double interpolationAlpha = 0.0;
};

class FixedStepClock {
public:
    explicit FixedStepClock(FixedStepConfig config = {}) : m_config(config) {}

    template <typename Step>
    FixedStepResult advance(double frameDeltaSeconds, Step&& step) {
        FixedStepResult result;
        if (!std::isfinite(frameDeltaSeconds) || frameDeltaSeconds < 0.0) frameDeltaSeconds = 0.0;
        m_accumulator += std::min(frameDeltaSeconds, m_config.maxFrameDeltaSeconds);
        while (m_accumulator + 1e-12 >= m_config.stepSeconds &&
               result.steps < m_config.maxCatchUpSteps) {
            std::forward<Step>(step)(m_tick, m_config.stepSeconds);
            ++m_tick;
            ++result.steps;
            m_accumulator -= m_config.stepSeconds;
        }
        if (m_accumulator >= m_config.stepSeconds) {
            result.droppedSteps = static_cast<std::uint32_t>(m_accumulator / m_config.stepSeconds);
            m_droppedSteps += result.droppedSteps;
            m_accumulator = std::fmod(m_accumulator, m_config.stepSeconds);
        }
        result.interpolationAlpha = std::clamp(m_accumulator / m_config.stepSeconds, 0.0, 1.0);
        return result;
    }

    void reset(std::uint64_t tick = 0u, double accumulatorSeconds = 0.0);
    [[nodiscard]] std::uint64_t tick() const { return m_tick; }
    [[nodiscard]] std::uint64_t droppedStepCount() const { return m_droppedSteps; }
    [[nodiscard]] double accumulatorSeconds() const { return m_accumulator; }
    [[nodiscard]] const FixedStepConfig& config() const { return m_config; }

private:
    FixedStepConfig m_config;
    std::uint64_t m_tick = 0u;
    std::uint64_t m_droppedSteps = 0u;
    double m_accumulator = 0.0;
};

}  // namespace odai::bethesda
