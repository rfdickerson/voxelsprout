#pragma once

// Core Time subsystem
// Responsible for: providing lightweight time-related data containers.
// Should NOT do: own the main loop, sleep the thread, or manage frame pacing yet.
namespace core {

struct TimeStep {
    float deltaSeconds = 0.0f;
};

} // namespace core
