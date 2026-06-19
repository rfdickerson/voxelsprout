#include "world/navmesh.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[navmesh test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectNear(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) > epsilon) {
        std::cerr << "[navmesh test] FAIL: " << message
                  << " (expected " << expected
                  << ", got " << actual << ")\n";
        ++g_failures;
    }
}

void appendQuad(
    std::vector<odai::world::NavmeshBuildTriangle>& triangles,
    float minX,
    float maxX,
    float y,
    float minZ,
    float maxZ,
    odai::world::NavmeshArea area = odai::world::NavmeshArea::Ground
) {
    triangles.push_back({
        {minX, y, minZ},
        {maxX, y, maxZ},
        {maxX, y, minZ},
        area
    });
    triangles.push_back({
        {minX, y, minZ},
        {minX, y, maxZ},
        {maxX, y, maxZ},
        area
    });
}

odai::world::NavmeshSettings testSettings() {
    odai::world::NavmeshSettings settings{};
    settings.agentRadius = 0.4f;
    settings.agentHeight = 2.0f;
    settings.maxClimb = 1.0f;
    settings.maxSlopeDegrees = 50.0f;
    settings.nearestPointMaxDistance = 20.0f;
    settings.edgeMergeEpsilon = 0.01f;
    return settings;
}

void testFlatPlanePath() {
    std::vector<odai::world::NavmeshBuildTriangle> triangles;
    appendQuad(triangles, 0.0f, 10.0f, 0.0f, 0.0f, 10.0f);

    odai::world::Navmesh navmesh;
    expectTrue(navmesh.build(triangles, testSettings()), "Flat navmesh builds");
    expectTrue(navmesh.stats().walkableTriangleCount == 2u, "Flat navmesh has two walkable triangles");
    expectTrue(navmesh.stats().linkCount == 1u, "Flat navmesh links shared triangle edge");

    std::vector<odai::world::NavmeshPathPoint> path;
    expectTrue(
        navmesh.findPath({1.0f, 3.0f, 1.0f}, {9.0f, 3.0f, 9.0f}, path),
        "Flat navmesh finds path across shared edge");
    expectTrue(path.size() >= 2u, "Flat navmesh path has start and end points");
    expectNear(path.front().position.y, 0.0f, 1e-5f, "Flat navmesh start snaps to ground");
    expectNear(path.back().position.y, 0.0f, 1e-5f, "Flat navmesh end snaps to ground");
}

void testStepClimbLimit() {
    std::vector<odai::world::NavmeshBuildTriangle> climbable;
    appendQuad(climbable, 0.0f, 5.0f, 0.0f, 0.0f, 5.0f);
    appendQuad(climbable, 5.0f, 10.0f, 0.75f, 0.0f, 5.0f);

    odai::world::Navmesh navmesh;
    expectTrue(navmesh.build(climbable, testSettings()), "Climbable step navmesh builds");
    expectTrue(navmesh.stats().linkCount >= 1u, "Climbable step links across shared edge");

    std::vector<odai::world::NavmeshPathPoint> path;
    expectTrue(
        navmesh.findPath({1.0f, 1.0f, 2.5f}, {9.0f, 2.0f, 2.5f}, path),
        "Climbable step path succeeds");

    std::vector<odai::world::NavmeshBuildTriangle> tooHigh;
    appendQuad(tooHigh, 0.0f, 5.0f, 0.0f, 0.0f, 5.0f);
    appendQuad(tooHigh, 5.0f, 10.0f, 2.0f, 0.0f, 5.0f);
    expectTrue(navmesh.build(tooHigh, testSettings()), "Too-high step navmesh builds");
    expectTrue(
        !navmesh.findPath({1.0f, 1.0f, 2.5f}, {9.0f, 3.0f, 2.5f}, path),
        "Too-high step path fails");
}

void testNearestPoint() {
    std::vector<odai::world::NavmeshBuildTriangle> triangles;
    appendQuad(triangles, -5.0f, 5.0f, 3.0f, -5.0f, 5.0f);

    odai::world::Navmesh navmesh;
    expectTrue(navmesh.build(triangles, testSettings()), "Nearest-point navmesh builds");

    odai::math::Vector3 point{};
    expectTrue(navmesh.findNearestPoint({0.0f, 20.0f, 0.0f}, point), "Nearest point query succeeds");
    expectNear(point.x, 0.0f, 1e-5f, "Nearest point x");
    expectNear(point.y, 3.0f, 1e-5f, "Nearest point y");
    expectNear(point.z, 0.0f, 1e-5f, "Nearest point z");
}

void testBuildFromGpuSceneAsset() {
    odai::importer::GpuSceneAsset scene{};

    scene.vertices = {
        odai::importer::ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        odai::importer::ImportedSceneVertex{{4.0f, 0.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
        odai::importer::ImportedSceneVertex{{4.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        odai::importer::ImportedSceneVertex{{0.0f, 0.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    scene.indices = {0u, 1u, 2u, 0u, 3u, 1u};

    odai::importer::GpuSceneMeshAsset mesh{};
    mesh.firstVertex = 0u;
    mesh.vertexCount = 4u;
    mesh.firstIndex = 0u;
    mesh.indexCount = 6u;
    scene.meshAssets.push_back(mesh);

    scene.objects.appliedTransforms.push_back({
        1.0f, 0.0f, 0.0f, 10.0f,
        0.0f, 1.0f, 0.0f, 2.0f,
        0.0f, 0.0f, 1.0f, 20.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    });
    scene.instances.objectIndices.push_back(0u);
    scene.instances.meshAssetIndices.push_back(0u);

    odai::world::Navmesh navmesh;
    expectTrue(
        navmesh.buildFromGpuSceneAsset(scene, testSettings()),
        "Navmesh builds from GPU scene asset");
    expectTrue(navmesh.stats().walkableTriangleCount == 2u, "GPU scene navmesh keeps walkable triangles");

    odai::math::Vector3 point{};
    expectTrue(navmesh.findNearestPoint({12.0f, 10.0f, 22.0f}, point), "GPU scene navmesh nearest query succeeds");
    expectNear(point.x, 12.0f, 1e-5f, "GPU scene navmesh applies transform x");
    expectNear(point.y, 2.0f, 1e-5f, "GPU scene navmesh applies transform y");
    expectNear(point.z, 22.0f, 1e-5f, "GPU scene navmesh applies transform z");
}

} // namespace

int main() {
    testFlatPlanePath();
    testStepClimbLimit();
    testNearestPoint();
    testBuildFromGpuSceneAsset();

    if (g_failures != 0) {
        std::cerr << "[navmesh test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[navmesh test] all checks passed\n";
    return 0;
}
