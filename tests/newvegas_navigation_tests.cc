#include "games/newvegas/newvegas_navigation.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using odai::games::newvegas::ActorNavigationWorld;
using odai::games::newvegas::ActorNavigationStep;
using odai::games::newvegas::ActorNavigationStepKind;
using odai::importer::CellCoord;
using odai::importer::fnv::FalloutNavMeshRecord;
using odai::importer::fnv::FalloutNavMeshTriangle;
using odai::importer::fnv::kNavMeshNoNeighbour;
using odai::math::Vector3;

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

}  // namespace

int main() {
    testResidentMeshesStitchAtSharedBorder();
    testVisibleGapDoesNotStitch();
    testAuthoredDoorPortalIsTypedOffMeshLink();
    std::cout << "Resident NAVM stitching tests passed\n";
    return 0;
}
