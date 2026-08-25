#include "bethesda/navigation_world.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using odai::games::newvegas::ActorNavigationWorld;
using odai::games::newvegas::ActorNavigationStep;
using odai::games::newvegas::ActorNavigationStepKind;
using odai::games::newvegas::GeneratedNavigationConfig;
using odai::importer::CellCoord;
using odai::importer::fnv::FalloutNavMeshRecord;
using odai::importer::fnv::FalloutNavMeshTriangle;
using odai::importer::fnv::kNavMeshNoNeighbour;
using odai::math::Vector3;

void addCollisionTriangle(odai::importer::ImportedScene& scene,
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

void addFloor(odai::importer::ImportedScene& scene,
    float minX, float maxX, float minZ, float maxZ, float y) {
    addCollisionTriangle(scene, {minX, y, minZ}, {minX, y, maxZ}, {maxX, y, minZ});
    addCollisionTriangle(scene, {maxX, y, minZ}, {minX, y, maxZ}, {maxX, y, maxZ});
}

void addWall(odai::importer::ImportedScene& scene,
    float x, float minZ, float maxZ, float minY, float maxY) {
    addCollisionTriangle(scene, {x, minY, minZ}, {x, maxY, minZ}, {x, minY, maxZ});
    addCollisionTriangle(scene, {x, minY, maxZ}, {x, maxY, minZ}, {x, maxY, maxZ});
}

FalloutNavMeshRecord triangleMesh(
    std::uint32_t formId, const std::vector<float>& vertices) {
    FalloutNavMeshRecord result;
    result.formId = formId;
    result.vertices = vertices;
    FalloutNavMeshTriangle triangle;
    triangle.vertex[0] = 0u;
    triangle.vertex[1] = 1u;
    triangle.vertex[2] = 2u;
    triangle.neighbour[0] = kNavMeshNoNeighbour;
    triangle.neighbour[1] = kNavMeshNoNeighbour;
    triangle.neighbour[2] = kNavMeshNoNeighbour;
    result.triangles.push_back(triangle);
    return result;
}

bool samePoint(const Vector3& left, const Vector3& right) {
    return std::fabs(left.x - right.x) < 1.0e-4f &&
        std::fabs(left.y - right.y) < 1.0e-4f &&
        std::fabs(left.z - right.z) < 1.0e-4f;
}

void testResidentMeshesStitchAtSharedBorder() {
    const FalloutNavMeshRecord lower = triangleMesh(0x100u,
        {0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 100.0f, 100.0f, 0.0f});
    const FalloutNavMeshRecord upper = triangleMesh(0x200u,
        {0.0f, 0.0f, 0.0f, 100.0f, 100.0f, 0.0f, 0.0f, 100.0f, 0.0f});

    ActorNavigationWorld first;
    first.addCell(CellCoord{0, 0}, {lower});
    first.addCell(CellCoord{1, 0}, {upper});
    std::vector<ActorNavigationStep> route;
    assert(first.buildPath({80.0f, 0.0f, -10.0f}, {10.0f, 0.0f, -80.0f}, route));
    assert(route.size() == 2u);
    assert(samePoint(route[0].position, {50.0f, 0.0f, -50.0f}));
    assert(samePoint(route[1].position, {10.0f, 0.0f, -80.0f}));

    // Completion order cannot change the chosen crossings.
    ActorNavigationWorld reversed;
    reversed.addCell(CellCoord{1, 0}, {upper});
    reversed.addCell(CellCoord{0, 0}, {lower});
    std::vector<ActorNavigationStep> replay;
    assert(reversed.buildPath(
        {80.0f, 0.0f, -10.0f}, {10.0f, 0.0f, -80.0f}, replay));
    assert(replay.size() == route.size());
    for (std::size_t index = 0u; index < route.size(); ++index) {
        assert(route[index] == replay[index]);
    }

    first.removeCell(CellCoord{1, 0});
    assert(first.meshCount() == 1u);
}

void testVisibleGapDoesNotStitch() {
    const FalloutNavMeshRecord lower = triangleMesh(0x100u,
        {0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 100.0f, 100.0f, 0.0f});
    const FalloutNavMeshRecord distant = triangleMesh(0x300u,
        {200.0f, 0.0f, 0.0f, 300.0f, 100.0f, 0.0f, 200.0f, 100.0f, 0.0f});
    ActorNavigationWorld world;
    world.addCell(CellCoord{0, 0}, {lower});
    world.addCell(CellCoord{1, 0}, {distant});
    std::vector<ActorNavigationStep> route;
    assert(!world.buildPath(
        {80.0f, 0.0f, -10.0f}, {210.0f, 0.0f, -80.0f}, route));
    assert(route.empty());
}

void testAuthoredDoorPortalIsTypedOffMeshLink() {
    FalloutNavMeshRecord source = triangleMesh(0x100u,
        {0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 0.0f, 100.0f, 0.0f});
    source.doorPortals.push_back({0x1234u, 0u});
    const FalloutNavMeshRecord destination = triangleMesh(0x200u,
        {1000.0f, 0.0f, 0.0f, 1100.0f, 0.0f, 0.0f, 1000.0f, 100.0f, 0.0f});
    ActorNavigationWorld world;
    world.addCell(CellCoord{0, 0}, {source});
    world.addCell(CellCoord{1, 0}, {destination});

    odai::importer::ImportedSceneDoor door;
    door.sourceReferenceFormId = 0x1234u;
    door.arrivalPosition[0] = 1010.0f;
    door.arrivalPosition[1] = 0.0f;
    door.arrivalPosition[2] = -10.0f;
    world.setResidentDoors({door});

    std::vector<ActorNavigationStep> route;
    assert(world.buildPath({10.0f, 0.0f, -10.0f},
        {1020.0f, 0.0f, -20.0f}, route));
    assert(route.size() == 2u);
    assert(route[0].kind == ActorNavigationStepKind::ActivateDoor);
    assert(route[0].doorReferenceFormId == 0x1234u);
    assert(samePoint(route[0].arrivalPosition, {1010.0f, 0.0f, -10.0f}));
    assert(route[1].kind == ActorNavigationStepKind::Walk);

    world.setResidentDoors({});
    route.clear();
    assert(!world.buildPath({10.0f, 0.0f, -10.0f},
        {1020.0f, 0.0f, -20.0f}, route));
}

void testTes3GeneratedMeshUsesDoorwayAndRejectsWalls() {
    odai::importer::ImportedScene scene;
    addFloor(scene, 0.0f, 640.0f, 0.0f, 320.0f, 0.0f);
    addWall(scene, 320.0f, 0.0f, 128.0f, 0.0f, 180.0f);
    addWall(scene, 320.0f, 192.0f, 320.0f, 0.0f, 180.0f);
    ActorNavigationWorld world;
    GeneratedNavigationConfig config;
    config.cellSize = 64.0f;
    config.agentRadius = 22.0f;
    world.addGeneratedCell({0, 0}, scene, config);
    assert(world.generatedNodeCount() > 0u);
    std::vector<ActorNavigationStep> route;
    assert(world.buildPath({32.0f, 0.0f, 32.0f}, {608.0f, 0.0f, 32.0f}, route));
    assert(std::any_of(route.begin(), route.end(), [](const ActorNavigationStep& step) {
        return step.position.z >= 150.0f && step.position.z <= 170.0f;
    }));

    odai::importer::ImportedScene sealed = scene;
    addWall(sealed, 320.0f, 128.0f, 192.0f, 0.0f, 180.0f);
    world.addGeneratedCell({0, 0}, sealed, config);
    route.clear();
    assert(!world.buildPath(
        {32.0f, 0.0f, 32.0f}, {608.0f, 0.0f, 32.0f}, route));
    assert(route.empty());
}

void testTes3GeneratedMeshClimbsStairsButNotFloorGap() {
    odai::importer::ImportedScene stairs;
    constexpr float kTread = 64.0f;
    constexpr float kRise = 16.0f;
    for (int step = 0; step < 8; ++step) {
        addFloor(stairs, step * kTread, (step + 1) * kTread,
            0.0f, 128.0f, step * kRise);
        if (step > 0) {
            addWall(stairs, step * kTread, 0.0f, 128.0f,
                (step - 1) * kRise, step * kRise);
        }
    }
    ActorNavigationWorld world;
    world.addGeneratedCell({0, 0}, stairs);
    std::vector<ActorNavigationStep> route;
    assert(world.buildPath({32.0f, 0.0f, 32.0f},
        {480.0f, 112.0f, 32.0f}, route));
    assert(!route.empty());
    assert(std::fabs(route.back().position.y - 112.0f) < 1.0e-4f);

    odai::importer::ImportedScene gap;
    for (int step = 0; step < 8; ++step) {
        if (step == 3) continue;
        addFloor(gap, step * kTread, (step + 1) * kTread,
            0.0f, 128.0f, step * kRise);
    }
    world.addGeneratedCell({0, 0}, gap);
    route.clear();
    assert(!world.buildPath({32.0f, 0.0f, 32.0f},
        {480.0f, 112.0f, 32.0f}, route));
}

void testTes3GeneratedMeshUsesTypedTeleportDoor() {
    odai::importer::ImportedScene scene;
    addFloor(scene, 0.0f, 192.0f, 0.0f, 192.0f, 0.0f);
    addFloor(scene, 1024.0f, 1216.0f, 0.0f, 192.0f, 48.0f);
    ActorNavigationWorld world;
    world.addGeneratedCell({0, 0}, scene);
    odai::importer::ImportedSceneDoor door;
    door.position[0] = 160.0f;
    door.position[1] = 0.0f;
    door.position[2] = 96.0f;
    door.arrivalPosition[0] = 1056.0f;
    door.arrivalPosition[1] = 48.0f;
    door.arrivalPosition[2] = 96.0f;
    door.sourceReferenceFormId = 0x55aau;
    world.setResidentDoors({door});
    std::vector<ActorNavigationStep> route;
    assert(world.buildPath({32.0f, 0.0f, 96.0f},
        {1184.0f, 48.0f, 96.0f}, route));
    const auto activation = std::find_if(route.begin(), route.end(),
        [](const ActorNavigationStep& step) {
            return step.kind == ActorNavigationStepKind::ActivateDoor;
        });
    assert(activation != route.end());
    assert(activation->doorReferenceFormId == 0x55aau);
    assert(samePoint(activation->arrivalPosition, {1056.0f, 48.0f, 96.0f}));
}

}  // namespace

int main() {
    testResidentMeshesStitchAtSharedBorder();
    testVisibleGapDoesNotStitch();
    testAuthoredDoorPortalIsTypedOffMeshLink();
    testTes3GeneratedMeshUsesDoorwayAndRejectsWalls();
    testTes3GeneratedMeshClimbsStairsButNotFloorGap();
    testTes3GeneratedMeshUsesTypedTeleportDoor();
    std::cout << "Resident authored/generated navigation tests passed\n";
    return 0;
}
