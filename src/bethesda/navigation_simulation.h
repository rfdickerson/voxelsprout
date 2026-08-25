#pragma once

// Headless navigation probe for imported Bethesda scenes.  It drives the same
// Jolt CharacterVirtual used by the runtime along ActorNavigationWorld routes;
// tests therefore exercise path planning and actual capsule collision rather
// than moving a point directly between waypoints.

#include "bethesda/bethesda_physics_world.h"
#include "bethesda/navigation_world.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace odai::games::newvegas {

enum class NavigationSimulationStatus : std::uint8_t {
    Arrived,
    NoPath,
    Blocked,
    Fell,
    TimedOut,
    MissingCharacter,
};

struct NavigationSimulationConfig {
    float fixedDeltaSeconds = 1.0f / 60.0f;
    float speed = 120.0f;
    float waypointRadius = 14.0f;
    float maximumFallBelowRoute = 160.0f;
    std::uint32_t blockedStepLimit = 90u;
    std::uint32_t maximumSteps = 60u * 90u;
};

struct NavigationSimulationResult {
    NavigationSimulationStatus status = NavigationSimulationStatus::NoPath;
    std::uint32_t physicsSteps = 0u;
    std::vector<std::uint32_t> activatedDoorReferences;
    std::vector<odai::math::Vector3> trace;
};

struct SimulatedQuestObjective {
    std::string id;
    odai::math::Vector3 destination{};
};

struct SimulatedQuestResult {
    NavigationSimulationStatus status = NavigationSimulationStatus::NoPath;
    std::vector<std::string> completedObjectives;
    std::vector<std::uint32_t> activatedDoorReferences;
    std::vector<odai::math::Vector3> trace;
};

class NavMeshPlayerSimulator {
public:
    NavMeshPlayerSimulator(
        const ActorNavigationWorld& navigation,
        bethesda::BethesdaPhysicsWorld& physics,
        bethesda::ObjectId player,
        NavigationSimulationConfig config = {});

    [[nodiscard]] NavigationSimulationResult runTo(
        const odai::math::Vector3& destination);
    [[nodiscard]] SimulatedQuestResult runQuest(
        std::span<const SimulatedQuestObjective> objectives);

private:
    const ActorNavigationWorld& m_navigation;
    bethesda::BethesdaPhysicsWorld& m_physics;
    bethesda::ObjectId m_player;
    NavigationSimulationConfig m_config;
};

}  // namespace odai::games::newvegas
