#pragma once

#include "math/math.h"

#include <cstddef>
#include <span>
#include <string>

namespace odai::app {

enum class MorrowindActorScheduleState {
    None,
    Idle,
    Wander,
    Travel,
    Wait
};

[[nodiscard]] MorrowindActorScheduleState parseMorrowindActorScheduleState(const std::string& value);
[[nodiscard]] const char* morrowindActorScheduleStateName(MorrowindActorScheduleState state);
[[nodiscard]] bool morrowindActorScheduleUsesPath(MorrowindActorScheduleState state);
[[nodiscard]] bool morrowindSeydaNeenIsNightHour(float gameHour);

struct MorrowindActorAvoidanceInput {
    odai::math::Vector3 position{};
    odai::math::Vector3 desiredDirection{};
    std::size_t selfIndex = 0;
    float separationRadius = 72.0f;
    float strength = 0.55f;
};

[[nodiscard]] odai::math::Vector3 computeMorrowindActorAvoidanceDirection(
    const MorrowindActorAvoidanceInput& input,
    std::span<const odai::math::Vector3> actorPositions);

[[nodiscard]] std::string makeMorrowindActorPathCacheKey(
    const odai::math::Vector3& start,
    const std::string& anchorId,
    float bucketSize);
[[nodiscard]] std::string makeMorrowindActorEndpointPathCacheKey(
    const std::string& startLabel,
    const odai::math::Vector3& start,
    const std::string& targetLabel,
    const odai::math::Vector3& target,
    float bucketSize);
[[nodiscard]] bool isMorrowindActorPathCacheExpired(float ageSeconds, float maxAgeSeconds);

} // namespace odai::app
