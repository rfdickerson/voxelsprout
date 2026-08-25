#include "bethesda/tes3_runtime.h"
#include "bethesda/navigation_simulation.h"
#include "games/newvegas/bethesda_collision.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using odai::bethesda::BethesdaPhysicsWorld;
using odai::bethesda::ObjectId;
using odai::bethesda::PhysicsCharacterConfig;
using odai::games::newvegas::ActorNavigationWorld;
using odai::games::newvegas::NavMeshPlayerSimulator;
using odai::games::newvegas::NavigationSimulationStatus;
using odai::games::newvegas::SimulatedQuestObjective;
using odai::importer::ImportedScene;
using odai::math::Vector3;

void addTriangle(ImportedScene& scene,
    const Vector3& a, const Vector3& b, const Vector3& c) {
    odai::importer::ImportedSceneCollisionTriangle triangle;
    const Vector3 vertices[3] = {a, b, c};
    for (std::size_t corner = 0u; corner < 3u; ++corner) {
        triangle.vertices[corner * 3u] = vertices[corner].x;
        triangle.vertices[(corner * 3u) + 1u] = vertices[corner].y;
        triangle.vertices[(corner * 3u) + 2u] = vertices[corner].z;
    }
    scene.collisionTriangles.push_back(triangle);
}

void addFloor(ImportedScene& scene,
    float minX, float maxX, float minZ, float maxZ, float y) {
    addTriangle(scene, {minX, y, minZ}, {minX, y, maxZ}, {maxX, y, minZ});
    addTriangle(scene, {maxX, y, minZ}, {minX, y, maxZ}, {maxX, y, maxZ});
}

void addWall(ImportedScene& scene,
    float x, float minZ, float maxZ, float minY, float maxY) {
    addTriangle(scene, {x, minY, minZ}, {x, maxY, minZ}, {x, minY, maxZ});
    addTriangle(scene, {x, minY, maxZ}, {x, maxY, minZ}, {x, maxY, maxZ});
}

void registerCollision(
    BethesdaPhysicsWorld& physics, const ImportedScene& scene, std::uint64_t token) {
    std::vector<Vector3> vertices;
    std::vector<std::uint32_t> indices;
    for (const auto& triangle : scene.collisionTriangles) {
        const std::uint32_t first = static_cast<std::uint32_t>(vertices.size());
        // BethesdaPhysicsWorld reverses triangle winding for Jolt.  Reverse it
        // here so the upward-facing navigation fixture remains upward-facing
        // to the character controller after that adapter conversion.
        for (const std::size_t corner : {0u, 2u, 1u}) {
            vertices.push_back({triangle.vertices[corner * 3u],
                triangle.vertices[(corner * 3u) + 1u],
                triangle.vertices[(corner * 3u) + 2u]});
        }
        indices.insert(indices.end(), {first, first + 1u, first + 2u});
    }
    std::string error;
    assert(physics.addStreamedStaticCollision(token, vertices, indices, error));
}

// Mirrors BethesdaApp::cacheBethesdaCollisionCell for an installed-data probe:
// authored collision plus the exterior LAND mesh, in the exact winding passed
// to BethesdaPhysicsWorld by the live streamer.
void registerRuntimeCollision(
    BethesdaPhysicsWorld& physics, const ImportedScene& scene, std::uint64_t token) {
    std::vector<Vector3> vertices;
    std::vector<std::uint32_t> indices;
    const auto append = [&](const float* triangle) {
        const std::uint32_t first = static_cast<std::uint32_t>(vertices.size());
        for (std::size_t corner = 0u; corner < 3u; ++corner) {
            vertices.push_back({triangle[corner * 3u], triangle[(corner * 3u) + 1u],
                triangle[(corner * 3u) + 2u]});
        }
        indices.insert(indices.end(), {first, first + 1u, first + 2u});
    };
    for (const auto& triangle : scene.collisionTriangles) append(triangle.vertices);
    if (!scene.meshes.empty() && scene.meshes.front().name == "terrain") {
        const auto& terrain = scene.meshes.front();
        for (std::size_t offset = 0u; offset + 2u < terrain.indices.size(); offset += 3u) {
            const std::uint32_t a = terrain.indices[offset];
            const std::uint32_t b = terrain.indices[offset + 1u];
            const std::uint32_t c = terrain.indices[offset + 2u];
            if (a >= terrain.vertices.size() || b >= terrain.vertices.size() ||
                c >= terrain.vertices.size()) continue;
            float triangle[9] = {};
            std::copy_n(terrain.vertices[a].position, 3u, triangle);
            std::copy_n(terrain.vertices[b].position, 3u, triangle + 3u);
            std::copy_n(terrain.vertices[c].position, 3u, triangle + 6u);
            append(triangle);
        }
    }
    std::string error;
    assert(!indices.empty());
    assert(physics.addStreamedStaticCollision(token, vertices, indices, error));
}

ImportedScene makeStairQuestScene() {
    ImportedScene scene;
    constexpr float kTread = 64.0f;
    constexpr float kRise = 12.0f;
    for (int step = 0; step < 8; ++step) {
        addFloor(scene, step * kTread, (step + 1) * kTread,
            0.0f, 192.0f, step * kRise);
        if (step > 0) {
            addWall(scene, step * kTread, 0.0f, 192.0f,
                (step - 1) * kRise, step * kRise);
        }
    }
    // A disconnected quest room reached only through the authored door link.
    addFloor(scene, 1024.0f, 1280.0f, 0.0f, 192.0f, 84.0f);
    return scene;
}

odai::bethesda::Tes3DialogueDefinition makeQuest() {
    using namespace odai::bethesda;
    Tes3DialogueDefinition quest;
    quest.record = makeTes3RecordKey("DIAL", "TR_NavmeshProbe");
    quest.id = "TR_NavmeshProbe";
    quest.type = Tes3DialogueType::Journal;
    for (const auto [id, index, status] : {
             std::tuple{"start", 10, Tes3QuestStatus::Name},
             std::tuple{"stairs", 20, Tes3QuestStatus::None},
             std::tuple{"door", 100, Tes3QuestStatus::Finished}}) {
        Tes3DialogueInfo info;
        info.record = makeTes3RecordKey("INFO", std::string("TR_NavmeshProbe_") + id);
        info.id = id;
        info.dispositionOrJournalIndex = index;
        info.questStatus = status;
        info.response = id;
        quest.infos.push_back(std::move(info));
    }
    return quest;
}

void testJoltPlayerCompletesTes3QuestRoute() {
    ImportedScene scene = makeStairQuestScene();
    ActorNavigationWorld navigation;
    navigation.addGeneratedCell({0, 0}, scene);
    odai::importer::ImportedSceneDoor door;
    door.position[0] = 480.0f;
    door.position[1] = 84.0f;
    door.position[2] = 96.0f;
    door.arrivalPosition[0] = 1056.0f;
    door.arrivalPosition[1] = 84.0f;
    door.arrivalPosition[2] = 96.0f;
    door.sourceReferenceFormId = 0x00c0ffeeu;
    navigation.setResidentDoors({door});

    BethesdaPhysicsWorld physics;
    std::string error;
    assert(physics.initialize(error));
    registerCollision(physics, scene, 1u);
    const ObjectId player = ObjectId::persistent(
        odai::bethesda::makeTes3RecordKey("NPC_", "player"));
    PhysicsCharacterConfig character;
    character.position = {32.0f, 1.0f, 96.0f};
    character.boundsHalfExtents = {16.0f, 56.0f, 16.0f};
    character.stepHeight = 18.0f;
    assert(physics.addCharacter(player, character, error));

    NavMeshPlayerSimulator simulator(navigation, physics, player);
    const SimulatedQuestObjective objectives[] = {
        {"climb_the_stairs", {416.0f, 72.0f, 96.0f}},
        {"enter_the_quest_room", {1184.0f, 84.0f, 96.0f}},
    };
    const auto result = simulator.runQuest(objectives);
    assert(result.status == NavigationSimulationStatus::Arrived);
    assert(result.completedObjectives.size() == 2u);
    assert(result.activatedDoorReferences == std::vector<std::uint32_t>{0x00c0ffeeu});
    assert(!result.trace.empty());
    const auto [minimumY, maximumY] = std::minmax_element(result.trace.begin(), result.trace.end(),
        [](const Vector3& left, const Vector3& right) { return left.y < right.y; });
    assert(minimumY->y > -2.0f);  // never passes through the floor
    assert(maximumY->y > 65.0f);  // physically climbed the staircase

    odai::bethesda::Tes3Journal journal;
    const odai::bethesda::Tes3DialogueDefinition quest = makeQuest();
    assert(journal.addEntry(quest, 10, 0u, error));
    assert(journal.addEntry(quest, 20, 1u, error));
    assert(journal.addEntry(quest, 100, 2u, error));
    const auto* state = journal.find("TR_NavmeshProbe");
    assert(state != nullptr && state->currentIndex == 100);
    assert(state->classification ==
        odai::bethesda::Tes3JournalQuestClassification::Completed);
}

void testJoltPlayerAndNavmeshRejectSolidWall() {
    ImportedScene scene;
    addFloor(scene, 0.0f, 640.0f, 0.0f, 192.0f, 0.0f);
    addWall(scene, 320.0f, 0.0f, 192.0f, 0.0f, 180.0f);
    ActorNavigationWorld navigation;
    navigation.addGeneratedCell({0, 0}, scene);
    BethesdaPhysicsWorld physics;
    std::string error;
    assert(physics.initialize(error));
    registerCollision(physics, scene, 2u);
    const ObjectId player = ObjectId::runtime(77u);
    PhysicsCharacterConfig character;
    character.position = {64.0f, 1.0f, 96.0f};
    character.boundsHalfExtents = {16.0f, 56.0f, 16.0f};
    assert(physics.addCharacter(player, character, error));
    NavMeshPlayerSimulator simulator(navigation, physics, player);
    const auto result = simulator.runTo({576.0f, 0.0f, 96.0f});
    assert(result.status == NavigationSimulationStatus::NoPath);

    odai::bethesda::PhysicsCharacterInput input;
    input.desiredVelocity = {120.0f, 0.0f, 0.0f};
    assert(physics.setCharacterInput(player, input));
    for (int tick = 0; tick < 300; ++tick) physics.step(1.0f / 60.0f);
    const auto state = physics.characterState(player);
    assert(state.has_value());
    assert(state->position.x < 310.0f);  // Jolt capsule cannot cross the wall
    assert(state->position.y > -2.0f);   // nor tunnel through the floor
}

void testIntersectingBridgeDeckRecovery() {
    ImportedScene scene;
    addFloor(scene, -256.0f, 256.0f, -256.0f, 256.0f, 0.0f);
    addFloor(scene, -128.0f, 128.0f, -128.0f, 128.0f, 80.0f);
    odai::games::newvegas::CollisionWorld collision;
    collision.addCell({0, 0}, scene);

    float recoveredFeetY = 0.0f;
    assert(collision.recoverFeetAboveIntersectingFloor(
        0.0f, 0.0f, 0.0f, 120.0f, 18.0f, recoveredFeetY));
    assert(std::fabs(recoveredFeetY - 80.0f) < 1.0e-4f);
    assert(!collision.recoverFeetAboveIntersectingFloor(
        0.0f, 0.0f, 80.0f, 200.0f, 18.0f, recoveredFeetY));
}

void runOptionalRetailSceneProbe(const char* environmentName, const char* label) {
    const char* path = std::getenv(environmentName);
    if (path == nullptr || *path == '\0') return;
    ImportedScene scene;
    assert(odai::importer::loadImportedScene(path, scene));
    ActorNavigationWorld navigation;
    navigation.addGeneratedCell({0, 0}, scene);
    assert(navigation.generatedNodeCount() > 0u);

    const std::string startEnvironment = std::string(environmentName) + "_START";
    const char* startText = std::getenv(startEnvironment.c_str());
    if (startText == nullptr || *startText == '\0') {
        std::cout << label << " optional scene probe: "
                  << navigation.generatedNodeCount() << " generated nodes from "
                  << scene.collisionTriangles.size() << " collision triangles\n";
        return;
    }
    Vector3 requestedStart;
    assert(std::sscanf(startText, "%f,%f,%f", &requestedStart.x,
        &requestedStart.y, &requestedStart.z) == 3);
    Vector3 start;
    assert(navigation.projectPoint(requestedStart.x, requestedStart.y, requestedStart.z,
        256.0f, 500.0f, start));

    // Sample deterministic local routes and select the one with the strongest
    // stair signature: several legal sub-step-height rises and a meaningfully
    // higher destination. This remains useful with modded Balmora geometry;
    // it does not hardcode one mesh's exact top landing.
    std::vector<odai::games::newvegas::ActorNavigationStep> bestRoute;
    float bestScore = -1.0f;
    float bestRise = 0.0f;
    std::uint32_t bestRisingEdges = 0u;
    const auto considerRoute = [&](std::vector<odai::games::newvegas::ActorNavigationStep> route) {
        if (route.empty()) return;
        float peak = start.y;
        float previousY = start.y;
        std::uint32_t risingEdges = 0u;
        for (const auto& waypoint : route) {
            peak = std::max(peak, waypoint.position.y);
            const float rise = waypoint.position.y - previousY;
            if (rise >= 2.0f && rise <= 18.5f) ++risingEdges;
            previousY = waypoint.position.y;
        }
        const float verticalGain = peak - start.y;
        const float score = static_cast<float>(risingEdges) * 10000.0f + verticalGain;
        if (score > bestScore) {
            bestScore = score;
            bestRise = verticalGain;
            bestRisingEdges = risingEdges;
            bestRoute = std::move(route);
        }
    };
    for (std::uint32_t seed = 0u; seed < 4096u; ++seed) {
        std::vector<odai::games::newvegas::ActorNavigationStep> route;
        if (!navigation.buildWanderPath(start, start, 950.0f, seed, route) ||
            route.empty()) continue;
        considerRoute(std::move(route));
    }

    // A local wander enumerates only the start component. If a bad generated
    // link cuts the city stairs off, it cannot reveal the higher landing at
    // all. Raster-probe nearby layers and explicitly ask A* for those landings;
    // success proves both that the landing exists and that it is connected.
    if (bestRise < 24.0f || bestRisingEdges < 2u) {
        std::vector<Vector3> elevated;
        for (int dz = -15; dz <= 15; ++dz) {
            for (int dx = -15; dx <= 15; ++dx) {
                for (int layer = 1; layer <= 20; ++layer) {
                    Vector3 point;
                    if (!navigation.projectPoint(start.x + static_cast<float>(dx * 64),
                            start.y + static_cast<float>(layer * 32),
                            start.z + static_cast<float>(dz * 64), 8.0f, 18.0f, point) ||
                        point.y < start.y + 24.0f) continue;
                    if (std::none_of(elevated.begin(), elevated.end(), [&](const Vector3& value) {
                            return std::fabs(value.x - point.x) < 0.5f &&
                                std::fabs(value.y - point.y) < 0.5f &&
                                std::fabs(value.z - point.z) < 0.5f;
                        })) elevated.push_back(point);
                }
            }
        }
        std::sort(elevated.begin(), elevated.end(), [&](const Vector3& left, const Vector3& right) {
            const float ldx = left.x - start.x;
            const float ldz = left.z - start.z;
            const float rdx = right.x - start.x;
            const float rdz = right.z - start.z;
            return (ldx * ldx) + (ldz * ldz) < (rdx * rdx) + (rdz * rdz);
        });
        for (const Vector3& destination : elevated) {
            std::vector<odai::games::newvegas::ActorNavigationStep> route;
            if (navigation.buildPath(start, destination, route)) {
                considerRoute(std::move(route));
                if (bestRise >= 24.0f && bestRisingEdges >= 2u) break;
            }
        }
    }
    assert(!bestRoute.empty());
    std::cout << label << " retail stair candidate: start=(" << start.x << ","
              << start.y << "," << start.z << ") rise=" << bestRise
              << " risingEdges=" << bestRisingEdges << " waypoints="
              << bestRoute.size() << " goal=(" << bestRoute.back().position.x << ","
              << bestRoute.back().position.y << ","
              << bestRoute.back().position.z << ")\n";
    assert(bestRise >= 24.0f);
    assert(bestRisingEdges >= 2u);

    BethesdaPhysicsWorld physics;
    std::string error;
    assert(physics.initialize(error));
    registerRuntimeCollision(physics, scene, 0xbabau);
    const ObjectId player = ObjectId::runtime(0xbabau);
    PhysicsCharacterConfig character;
    character.position = start + Vector3{0.0f, 1.0f, 0.0f};
    character.boundsHalfExtents = {16.0f, 56.0f, 16.0f};
    character.stepHeight = 18.0f;
    assert(physics.addCharacter(player, character, error));
    odai::games::newvegas::NavigationSimulationConfig config;
    config.speed = 90.0f;
    config.waypointRadius = 18.0f;
    config.blockedStepLimit = 180u;
    config.maximumSteps = 60u * 120u;
    NavMeshPlayerSimulator simulator(navigation, physics, player, config);
    const auto result = simulator.runTo(bestRoute.back().position);
    assert(result.status == NavigationSimulationStatus::Arrived);
    assert(!result.trace.empty());
    const auto [minimumY, maximumY] = std::minmax_element(result.trace.begin(), result.trace.end(),
        [](const Vector3& left, const Vector3& right) { return left.y < right.y; });
    assert(maximumY->y - result.trace.front().y >= 24.0f);
    assert(minimumY->y >= start.y - 4.0f);
    const auto finalState = physics.characterState(player);
    assert(finalState.has_value() && finalState->grounded);
    std::cout << label << " Jolt city stair probe: arrived in "
              << result.physicsSteps << " steps, climbed "
              << (maximumY->y - result.trace.front().y) << " units across "
              << bestRisingEdges << " stair edges; minimum feet y="
              << minimumY->y << "\n";
}

void runOptionalRetailBridgeProbe(const char* environmentName, const char* label) {
    const char* path = std::getenv(environmentName);
    if (path == nullptr || *path == '\0') return;
    const std::string startEnvironment = std::string(environmentName) + "_START";
    const char* startText = std::getenv(startEnvironment.c_str());
    if (startText == nullptr || *startText == '\0') return;

    ImportedScene scene;
    assert(odai::importer::loadImportedScene(path, scene));
    Vector3 start;
    assert(std::sscanf(startText, "%f,%f,%f", &start.x, &start.y, &start.z) == 3);
    odai::games::newvegas::CollisionWorld collision;
    collision.addCell({0, 0}, scene);
    float recoveredFeetY = 0.0f;
    assert(collision.recoverFeetAboveIntersectingFloor(
        start.x, start.z, start.y, start.y + 120.0f, 18.0f, recoveredFeetY));
    std::cout << label << " bridge overlap recovery: feet y=" << start.y
              << " -> " << recoveredFeetY << "\n";
    start.y = recoveredFeetY + 0.1f;

    BethesdaPhysicsWorld physics;
    std::string error;
    assert(physics.initialize(error));
    registerRuntimeCollision(physics, scene, 0xb11d6eu);
    const ObjectId player = ObjectId::runtime(0xb11d6eu);
    PhysicsCharacterConfig character;
    character.position = start;
    assert(physics.addCharacter(player, character, error));

    odai::bethesda::PhysicsCharacterInput idle;
    assert(physics.setCharacterInput(player, idle));
    for (int tick = 0; tick < 30; ++tick) physics.step(1.0f / 60.0f);
    const auto settled = physics.characterState(player);
    assert(settled.has_value());
    std::cout << label << " bridge controller settled at (" << settled->position.x << ","
              << settled->position.y << "," << settled->position.z << ") grounded="
              << settled->grounded << "\n";

    float bestDistance = 0.0f;
    for (int direction = 0; direction < 16; ++direction) {
        odai::bethesda::PhysicsCharacterSnapshot reset;
        reset.object = player;
        reset.position = start;
        reset.groundNormal = {0.0f, 1.0f, 0.0f};
        reset.grounded = true;
        assert(physics.restoreCharacter(reset, error));
        const float angle = static_cast<float>(direction) * (2.0f * 3.14159265f / 16.0f);
        odai::bethesda::PhysicsCharacterInput input;
        input.desiredVelocity = {std::cos(angle) * 180.0f, 0.0f,
            std::sin(angle) * 180.0f};
        assert(physics.setCharacterInput(player, input));
        for (int tick = 0; tick < 120; ++tick) physics.step(1.0f / 60.0f);
        const auto state = physics.characterState(player);
        assert(state.has_value());
        const float dx = state->position.x - start.x;
        const float dz = state->position.z - start.z;
        const float distance = std::sqrt((dx * dx) + (dz * dz));
        bestDistance = std::max(bestDistance, distance);
        std::cout << "  direction " << direction << " distance=" << distance
                  << " final=(" << state->position.x << "," << state->position.y << ","
                  << state->position.z << ") blocked=" << state->blocked << "\n";
    }
    std::cout << label << " bridge best escape distance=" << bestDistance << "\n";
    assert(bestDistance > 250.0f);
}

}  // namespace

int main() {
    testJoltPlayerCompletesTes3QuestRoute();
    testJoltPlayerAndNavmeshRejectSolidWall();
    testIntersectingBridgeDeckRecovery();
    // Game data remains optional and is never redistributed. Point either
    // variable at a locally cooked exterior/interior .bin to smoke-test the
    // exact Morrowind or Tamriel Rebuilt collision authored on this machine.
    runOptionalRetailSceneProbe("ODAI_MORROWIND_NAV_SCENE", "Morrowind");
    runOptionalRetailSceneProbe("ODAI_TAMRIEL_REBUILT_NAV_SCENE", "Tamriel Rebuilt");
    runOptionalRetailBridgeProbe("ODAI_MORROWIND_BRIDGE_SCENE", "Morrowind");
    std::cout << "TES3 generated-navmesh/Jolt quest probes passed\n";
    return 0;
}
