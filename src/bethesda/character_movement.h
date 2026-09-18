#pragma once
#include "math/math.h"
#include <algorithm>
#include <cmath>

namespace odai::bethesda {
enum class JumpPhase { Grounded, Takeoff, Ascending, Apex, Falling, Landing };
enum class LandingSeverity { Light, Hard, Stagger, Severe };
struct CharacterMovementSettings {
    bool enabled = false;
    float coyoteSeconds = .10f;
    float bufferSeconds = .12f;
    float jumpSpeedMetres = 4.572f;
    float airAccelerationMetres = 8.f;
    float hardLandingMetres = 2.f, staggerLandingMetres = 5.f, severeLandingMetres = 8.f;
};
struct CharacterMovementState {
    float coyoteRemaining = 0, bufferRemaining = 0;
    float airVelocityX = 0, airVelocityZ = 0; // metres/second, excludes external impulses
    bool jumpHeld = false, jumpConsumed = false;
    friend bool operator==(const CharacterMovementState&, const CharacterMovementState&) = default;
};
inline bool validCharacterMovementState(const CharacterMovementState& state) {
    return std::isfinite(state.coyoteRemaining) && state.coyoteRemaining >= 0 && state.coyoteRemaining <= 1 &&
        std::isfinite(state.bufferRemaining) && state.bufferRemaining >= 0 && state.bufferRemaining <= 1 &&
        std::isfinite(state.airVelocityX) && std::isfinite(state.airVelocityZ);
}
inline bool validCharacterMovementSettings(const CharacterMovementSettings& settings) {
    return std::isfinite(settings.coyoteSeconds) && settings.coyoteSeconds >= 0 && settings.coyoteSeconds <= 1 &&
        std::isfinite(settings.bufferSeconds) && settings.bufferSeconds >= 0 && settings.bufferSeconds <= 1 &&
        std::isfinite(settings.jumpSpeedMetres) && settings.jumpSpeedMetres > 0 && settings.jumpSpeedMetres <= 30 &&
        std::isfinite(settings.airAccelerationMetres) && settings.airAccelerationMetres >= 0 &&
        std::isfinite(settings.hardLandingMetres) && settings.hardLandingMetres >= 0 &&
        std::isfinite(settings.staggerLandingMetres) && settings.staggerLandingMetres >= settings.hardLandingMetres &&
        std::isfinite(settings.severeLandingMetres) && settings.severeLandingMetres >= settings.staggerLandingMetres;
}
// Called exactly once per physics tick. Input is a held button; only a rising
// edge buffers a jump, so holding it across a landing cannot cause auto-jumps.
inline bool advanceCharacterMovement(CharacterMovementState& state, const CharacterMovementSettings& settings,
    bool supported, bool landed, bool jumpRequested, float delta, float& velocityX, float& velocityZ) {
    if (landed) state.jumpConsumed = false;
    if (supported && !state.jumpConsumed) state.coyoteRemaining = settings.coyoteSeconds;
    if (jumpRequested && !state.jumpHeld) state.bufferRemaining = settings.bufferSeconds;
    const bool immediate = jumpRequested && !state.jumpHeld;
    state.jumpHeld = jumpRequested;
    const bool jump = !state.jumpConsumed && (supported || state.coyoteRemaining > 0) &&
        (immediate || state.bufferRemaining > 0);
    if (jump) { state.jumpConsumed = true; state.coyoteRemaining = state.bufferRemaining = 0; }
    if (supported) {
        state.airVelocityX = velocityX; state.airVelocityZ = velocityZ;
    } else {
        const float dx = velocityX - state.airVelocityX, dz = velocityZ - state.airVelocityZ;
        const float distance = std::sqrt(dx * dx + dz * dz);
        const float alpha = distance > 0 ? std::min(1.f, settings.airAccelerationMetres * delta / distance) : 0;
        state.airVelocityX += dx * alpha; state.airVelocityZ += dz * alpha;
        velocityX = state.airVelocityX; velocityZ = state.airVelocityZ;
    }
    state.coyoteRemaining = std::max(0.f, state.coyoteRemaining - delta);
    state.bufferRemaining = std::max(0.f, state.bufferRemaining - delta);
    return jump;
}
inline LandingSeverity classifyLanding(float impactMetres, const CharacterMovementSettings& settings) {
    return impactMetres >= settings.severeLandingMetres ? LandingSeverity::Severe :
        impactMetres >= settings.staggerLandingMetres ? LandingSeverity::Stagger :
        impactMetres >= settings.hardLandingMetres ? LandingSeverity::Hard : LandingSeverity::Light;
}
} // namespace odai::bethesda
