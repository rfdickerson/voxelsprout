#include "app/morrowind_actor_ai.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace odai::app {

MorrowindActorScheduleState parseMorrowindActorScheduleState(const std::string& value) {
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered == "idle") {
        return MorrowindActorScheduleState::Idle;
    }
    if (lowered == "wander") {
        return MorrowindActorScheduleState::Wander;
    }
    if (lowered == "travel") {
        return MorrowindActorScheduleState::Travel;
    }
    if (lowered == "wait") {
        return MorrowindActorScheduleState::Wait;
    }
    return MorrowindActorScheduleState::None;
}

const char* morrowindActorScheduleStateName(MorrowindActorScheduleState state) {
    switch (state) {
    case MorrowindActorScheduleState::Idle: return "idle";
    case MorrowindActorScheduleState::Wander: return "wander";
    case MorrowindActorScheduleState::Travel: return "travel";
    case MorrowindActorScheduleState::Wait: return "wait";
    case MorrowindActorScheduleState::None:
    default:
        return "";
    }
}

bool morrowindActorScheduleUsesPath(MorrowindActorScheduleState state) {
    return state == MorrowindActorScheduleState::Wander ||
           state == MorrowindActorScheduleState::Travel;
}

bool morrowindSeydaNeenIsNightHour(float gameHour) {
    const float wrappedHour = std::fmod(gameHour + 24.0f, 24.0f);
    return wrappedHour >= 21.0f || wrappedHour < 6.0f;
}

odai::math::Vector3 computeMorrowindActorAvoidanceDirection(
    const MorrowindActorAvoidanceInput& input,
    std::span<const odai::math::Vector3> actorPositions
) {
    odai::math::Vector3 separation{};
    const float radiusSq = input.separationRadius * input.separationRadius;
    if (input.separationRadius <= 0.0f || radiusSq <= 0.0f) {
        return odai::math::normalize(input.desiredDirection);
    }

    for (std::size_t actorIndex = 0; actorIndex < actorPositions.size(); ++actorIndex) {
        if (actorIndex == input.selfIndex) {
            continue;
        }
        const odai::math::Vector3 delta{
            input.position.x - actorPositions[actorIndex].x,
            0.0f,
            input.position.z - actorPositions[actorIndex].z
        };
        const float distanceSq = (delta.x * delta.x) + (delta.z * delta.z);
        if (distanceSq <= 0.001f || distanceSq >= radiusSq) {
            continue;
        }
        const float distance = std::sqrt(distanceSq);
        const float weight = (input.separationRadius - distance) / input.separationRadius;
        separation += (delta / distance) * weight;
    }

    const odai::math::Vector3 desired = odai::math::normalize(input.desiredDirection);
    const odai::math::Vector3 blended = desired + (separation * std::max(input.strength, 0.0f));
    const odai::math::Vector3 normalized = odai::math::normalize(blended);
    return odai::math::lengthSquared(normalized) > 0.0f ? normalized : desired;
}

std::string makeMorrowindActorPathCacheKey(
    const odai::math::Vector3& start,
    const std::string& anchorId,
    float bucketSize
) {
    const float clampedBucket = std::max(bucketSize, 1.0f);
    const int bucketX = static_cast<int>(std::floor(start.x / clampedBucket));
    const int bucketZ = static_cast<int>(std::floor(start.z / clampedBucket));
    std::ostringstream out;
    out << anchorId << ':' << bucketX << ':' << bucketZ;
    return out.str();
}

std::string makeMorrowindActorEndpointPathCacheKey(
    const std::string& startLabel,
    const odai::math::Vector3& start,
    const std::string& targetLabel,
    const odai::math::Vector3& target,
    float bucketSize
) {
    const float clampedBucket = std::max(bucketSize, 1.0f);
    const int startBucketX = static_cast<int>(std::floor(start.x / clampedBucket));
    const int startBucketZ = static_cast<int>(std::floor(start.z / clampedBucket));
    const int targetBucketX = static_cast<int>(std::floor(target.x / clampedBucket));
    const int targetBucketZ = static_cast<int>(std::floor(target.z / clampedBucket));
    std::ostringstream out;
    out << startLabel << ':' << startBucketX << ':' << startBucketZ
        << "->"
        << targetLabel << ':' << targetBucketX << ':' << targetBucketZ;
    return out.str();
}

bool isMorrowindActorPathCacheExpired(float ageSeconds, float maxAgeSeconds) {
    return ageSeconds > std::max(maxAgeSeconds, 0.0f);
}

} // namespace odai::app
