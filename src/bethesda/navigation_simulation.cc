#include "bethesda/navigation_simulation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace odai::games::newvegas {

namespace {

float horizontalDistanceSquared(
    const odai::math::Vector3& left, const odai::math::Vector3& right) {
    const float dx = left.x - right.x;
    const float dz = left.z - right.z;
    return (dx * dx) + (dz * dz);
}

}  // namespace

NavMeshPlayerSimulator::NavMeshPlayerSimulator(
    const ActorNavigationWorld& navigation,
    bethesda::BethesdaPhysicsWorld& physics,
    bethesda::ObjectId player,
    NavigationSimulationConfig config)
    : m_navigation(navigation), m_physics(physics), m_player(std::move(player)),
      m_config(config) {
    m_config.fixedDeltaSeconds =
        std::clamp(m_config.fixedDeltaSeconds, 1.0e-4f, 0.1f);
    m_config.speed = std::max(m_config.speed, 1.0f);
    m_config.waypointRadius = std::max(m_config.waypointRadius, 1.0f);
    m_config.maximumFallBelowRoute = std::max(m_config.maximumFallBelowRoute, 1.0f);
    m_config.blockedStepLimit = std::max(m_config.blockedStepLimit, 1u);
    m_config.maximumSteps = std::max(m_config.maximumSteps, 1u);
}

NavigationSimulationResult NavMeshPlayerSimulator::runTo(
    const odai::math::Vector3& destination) {
    NavigationSimulationResult result;
    const auto initial = m_physics.characterState(m_player);
    if (!initial.has_value()) {
        result.status = NavigationSimulationStatus::MissingCharacter;
        return result;
    }
    std::vector<ActorNavigationStep> route;
    if (!m_navigation.buildPath(initial->position, destination, route) || route.empty()) {
        result.status = NavigationSimulationStatus::NoPath;
        return result;
    }
    float routeFloor = initial->position.y;
    for (const ActorNavigationStep& step : route) {
        routeFloor = std::min(routeFloor, step.position.y);
        if (step.kind == ActorNavigationStepKind::ActivateDoor) {
            routeFloor = std::min(routeFloor, step.arrivalPosition.y);
        }
    }
    const float fallLimit = routeFloor - m_config.maximumFallBelowRoute;
    const float waypointRadiusSquared = m_config.waypointRadius * m_config.waypointRadius;
    std::size_t waypoint = 0u;
    std::uint32_t blockedSteps = 0u;
    result.trace.push_back(initial->position);

    for (std::uint32_t tick = 0u; tick < m_config.maximumSteps; ++tick) {
        auto state = m_physics.characterState(m_player);
        if (!state.has_value()) {
            result.status = NavigationSimulationStatus::MissingCharacter;
            return result;
        }
        while (waypoint < route.size() &&
            horizontalDistanceSquared(state->position, route[waypoint].position) <=
                waypointRadiusSquared &&
            std::abs(state->position.y - route[waypoint].position.y) <=
                std::max(48.0f, m_config.waypointRadius * 2.0f)) {
            if (route[waypoint].kind == ActorNavigationStepKind::ActivateDoor) {
                bethesda::PhysicsCharacterSnapshot relocated;
                relocated.object = m_player;
                relocated.position = route[waypoint].arrivalPosition;
                relocated.rotation = state->rotation;
                relocated.groundNormal = {0.0f, 1.0f, 0.0f};
                std::string error;
                if (!m_physics.restoreCharacter(relocated, error)) {
                    result.status = NavigationSimulationStatus::MissingCharacter;
                    return result;
                }
                result.activatedDoorReferences.push_back(
                    route[waypoint].doorReferenceFormId);
                state = m_physics.characterState(m_player);
            }
            ++waypoint;
        }
        if (waypoint >= route.size()) {
            (void)m_physics.setCharacterInput(m_player, bethesda::PhysicsCharacterInput{});
            result.status = NavigationSimulationStatus::Arrived;
            return result;
        }

        const odai::math::Vector3 delta = route[waypoint].position - state->position;
        const float horizontalLength = std::sqrt((delta.x * delta.x) + (delta.z * delta.z));
        bethesda::PhysicsCharacterInput input;
        if (horizontalLength > 1.0e-5f) {
            input.desiredVelocity = {delta.x * (m_config.speed / horizontalLength),
                0.0f, delta.z * (m_config.speed / horizontalLength)};
        }
        (void)m_physics.setCharacterInput(m_player, input);
        const auto stepped = m_physics.step(m_config.fixedDeltaSeconds);
        (void)stepped;
        ++result.physicsSteps;
        state = m_physics.characterState(m_player);
        if (!state.has_value()) {
            result.status = NavigationSimulationStatus::MissingCharacter;
            return result;
        }
        result.trace.push_back(state->position);
        if (state->position.y < fallLimit) {
            result.status = NavigationSimulationStatus::Fell;
            return result;
        }
        blockedSteps = state->blocked ? blockedSteps + 1u : 0u;
        if (blockedSteps >= m_config.blockedStepLimit) {
            result.status = NavigationSimulationStatus::Blocked;
            return result;
        }
    }
    (void)m_physics.setCharacterInput(m_player, bethesda::PhysicsCharacterInput{});
    result.status = NavigationSimulationStatus::TimedOut;
    return result;
}

SimulatedQuestResult NavMeshPlayerSimulator::runQuest(
    std::span<const SimulatedQuestObjective> objectives) {
    SimulatedQuestResult result;
    result.status = NavigationSimulationStatus::Arrived;
    for (const SimulatedQuestObjective& objective : objectives) {
        NavigationSimulationResult leg = runTo(objective.destination);
        result.trace.insert(result.trace.end(), leg.trace.begin(), leg.trace.end());
        result.activatedDoorReferences.insert(result.activatedDoorReferences.end(),
            leg.activatedDoorReferences.begin(), leg.activatedDoorReferences.end());
        if (leg.status != NavigationSimulationStatus::Arrived) {
            result.status = leg.status;
            return result;
        }
        result.completedObjectives.push_back(objective.id);
    }
    return result;
}

}  // namespace odai::games::newvegas
