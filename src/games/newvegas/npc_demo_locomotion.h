#pragma once
#include "math/math.h"
#include <algorithm>
#include <cmath>

namespace odai::newvegas {
// Input and route policy shared by the interactive studio and its Jolt tests.
struct NpcDemoLocomotion {
    enum class Mode { Idle, Manual, Circle };
    Mode mode = Mode::Idle;
    math::Vector3 manual{};
    float preparationRemaining = 0.0f;
    bool preparing = false;
    math::Vector3 airborneVelocity{};
    bool advanceJump(bool requested, bool grounded, float, math::Vector3 velocity) {
        preparing = false;
        preparationRemaining = 0.0f;
        if (!requested || !grounded) return false;
        airborneVelocity = {velocity.x, 0, velocity.z};
        return true;
    }
    static constexpr float radius = 140.0f;
    static constexpr float speed = 230.0f;

    void input(bool stop, bool run, math::Vector3 arrows) {
        if (stop) mode = Mode::Idle;
        else if (run) mode = Mode::Circle;
        else if (math::length(arrows) > 0.0f) mode = Mode::Manual;
        manual = arrows;
    }
    math::Vector3 target(math::Vector3 position) const {
        if (mode == Mode::Idle) return {};
        if (mode == Mode::Manual) return manual * speed;
        const float r = std::hypot(position.x, position.z);
        const float nx = r > 0.01f ? position.x / r : 1.0f;
        const float nz = r > 0.01f ? position.z / r : 0.0f;
        // Positive rotation about world +Y (counterclockwise viewed from above).
        // Suppress tangent at the centre for a smooth outward entry.
        const float tangent = speed * std::clamp(r / radius, 0.0f, 1.0f);
        // Anticipate the inward acceleration needed by the velocity smoothing.
        const float radial = std::clamp((radius - r) * 4.0f -
            0.12f * tangent * tangent / std::max(r, radius), -speed, speed);
        math::Vector3 velocity{nx * radial + nz * tangent, 0, nz * radial - nx * tangent};
        const float length = math::length(velocity);
        return length > speed ? velocity * (speed / length) : velocity;
    }
    math::Vector3 step(math::Vector3 position, math::Vector3 velocity, float dt) const {
        const auto desired = target(position);
        if (math::length(desired) == 0.0f) {
            // Reach a real stop within 100 ms; exponential decay otherwise
            // keeps the physics-derived walk state alive for almost a second.
            const float currentSpeed = math::length(velocity);
            const float braking = speed * std::max(dt, 0.0f) / 0.10f;
            return currentSpeed <= braking ? math::Vector3{} :
                velocity * ((currentSpeed - braking) / currentSpeed);
        }
        velocity += (desired - velocity) * (1.0f - std::exp(-dt / 0.12f));
        return math::length(velocity) < 0.5f ? math::Vector3{} : velocity;
    }
};
}
