#include "bethesda/fixed_step_clock.h"

namespace odai::bethesda {

void FixedStepClock::reset(std::uint64_t tick, double accumulatorSeconds) {
    m_tick = tick;
    m_droppedSteps = 0u;
    m_accumulator = std::clamp(accumulatorSeconds, 0.0, m_config.stepSeconds);
}

}  // namespace odai::bethesda
