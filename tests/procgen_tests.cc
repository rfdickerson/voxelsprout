#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "import/imported_scene.h"
#include "procgen/building_generator.h"
#include "procgen/csg.h"
#include "procgen/mesh_emit.h"
#include "procgen/primitives.h"
#include "procgen/props.h"

namespace {

using odai::math::Vector3;
using odai::procgen::Color3;
using odai::procgen::CsgMesh;
using odai::procgen::Polygon;

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[procgen test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectEqualU32(std::uint32_t actual, std::uint32_t expected, const char* message) {
    if (actual != expected) {
        std::cerr << "[procgen test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

void expectNear(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) > epsilon) {
        std::cerr << "[procgen test] FAIL: " << message
                  << " (expected " << expected << ", got " << actual << ")\n";
        ++g_failures;
    }
}

// Divergence theorem over the fan-triangulated boundary. Positive and equal to
// the enclosed volume only for closed, outward-wound meshes — so it doubles as
// a winding check.
float signedVolume(const CsgMesh& mesh) {
    float volume = 0.0f;
    for (const Polygon& polygon : mesh.polygons) {
        for (std::size_t i = 1; i + 1 < polygon.vertices.size(); ++i) {
            const Vector3& a = polygon.vertices[0];
            const Vector3& b = polygon.vertices[i];
            const Vector3& c = polygon.vertices[i + 1];
            volume += odai::math::dot(a, odai::math::cross(b, c));
        }
    }
    return volume / 6.0f;
}

// Sum of area-weighted face normals; ~0 for any geometrically closed surface.
Vector3 normalAreaSum(const CsgMesh& mesh) {
    Vector3 sum{};
    for (const Polygon& polygon : mesh.polygons) {
        for (std::size_t i = 1; i + 1 < polygon.vertices.size(); ++i) {
            sum += odai::math::cross(polygon.vertices[i] - polygon.vertices[0],
                                     polygon.vertices[i + 1] - polygon.vertices[0]) *
                   0.5f;
        }
    }
    return sum;
}

// V - E + F with vertices welded on a 1e-3 grid. Valid for pristine primitives
// (no T-junctions); expect 2 for genus-0 solids.
int eulerCharacteristic(const CsgMesh& mesh) {
    using Key = std::tuple<long, long, long>;
    const auto keyOf = [](const Vector3& v) {
        return Key{std::lround(v.x * 1000.0f), std::lround(v.y * 1000.0f),
                   std::lround(v.z * 1000.0f)};
    };
    std::map<Key, int> vertexIds;
    std::set<std::pair<int, int>> edges;
    const auto idOf = [&](const Vector3& v) {
        const auto [it, inserted] = vertexIds.emplace(keyOf(v), static_cast<int>(vertexIds.size()));
        (void)inserted;
        return it->second;
    };
    for (const Polygon& polygon : mesh.polygons) {
        for (std::size_t i = 0; i < polygon.vertices.size(); ++i) {
            const int a = idOf(polygon.vertices[i]);
            const int b = idOf(polygon.vertices[(i + 1) % polygon.vertices.size()]);
            edges.emplace(std::min(a, b), std::max(a, b));
        }
    }
    return static_cast<int>(vertexIds.size()) - static_cast<int>(edges.size()) +
           static_cast<int>(mesh.polygons.size());
}

void expectClosed(const CsgMesh& mesh, const char* label) {
    const Vector3 sum = normalAreaSum(mesh);
    expectNear(odai::math::length(sum), 0.0f, 1e-3f, label);
}

const Color3 kGrey{0.5f, 0.5f, 0.5f};

void testBoxPrimitive() {
    const CsgMesh box = odai::procgen::makeBox({0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}, kGrey);
    expectEqualU32(static_cast<std::uint32_t>(box.polygons.size()), 6u, "box has 6 faces");
    expectNear(signedVolume(box), 6.0f, 1e-3f, "box volume 1x2x3");
    expectClosed(box, "box normal sum ~ 0");
    expectTrue(eulerCharacteristic(box) == 2, "box Euler characteristic 2");
}

void testPrimitivesClosed() {
    const CsgMesh cylinder = odai::procgen::makeCylinder({0.0f, 0.0f, 0.0f}, 0.5f, 1.0f, 8, kGrey);
    // Regular octagon with circumradius r: area = n/2 * r^2 * sin(2pi/n).
    const float octagonArea = 4.0f * 0.25f * std::sin(2.0f * odai::math::kPi / 8.0f);
    expectNear(signedVolume(cylinder), octagonArea, 1e-3f, "cylinder volume");
    expectClosed(cylinder, "cylinder normal sum ~ 0");
    expectTrue(eulerCharacteristic(cylinder) == 2, "cylinder Euler characteristic 2");

    const CsgMesh gable =
        odai::procgen::makeGablePrism(0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.5f, 0.8f, kGrey);
    expectNear(signedVolume(gable), 0.5f + 0.15f, 1e-3f, "gable prism volume");
    expectClosed(gable, "gable normal sum ~ 0");
    expectTrue(eulerCharacteristic(gable) == 2, "gable Euler characteristic 2");

    const std::vector<std::array<float, 2>> square = {
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
    const CsgMesh frustum = odai::procgen::makeConvexPrism(square, 0.0f, 1.0f, kGrey, 0.5f);
    // Square frustum: h/3 * (A0 + A1 + sqrt(A0*A1)).
    expectNear(signedVolume(frustum), (1.0f + 0.25f + 0.5f) / 3.0f, 1e-3f, "tapered prism volume");
    expectClosed(frustum, "tapered prism normal sum ~ 0");
    expectTrue(eulerCharacteristic(frustum) == 2, "tapered prism Euler characteristic 2");
}

void testUnionOverlappingBoxes() {
    const CsgMesh a = odai::procgen::makeBox({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, kGrey);
    const CsgMesh b = odai::procgen::makeBox({0.5f, 0.0f, 0.0f}, {1.5f, 1.0f, 1.0f}, kGrey);
    const CsgMesh result = odai::procgen::csgUnion(a, b);
    expectNear(signedVolume(result), 1.5f, 1e-3f, "union volume = A + B - overlap");
    expectClosed(result, "union normal sum ~ 0");
}

void testSubtractCornerNotch() {
    const CsgMesh a = odai::procgen::makeBox({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, kGrey);
    const CsgMesh b = odai::procgen::makeBox({0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f}, kGrey);
    const CsgMesh result = odai::procgen::csgSubtract(a, b);
    expectNear(signedVolume(result), 1.0f - 0.125f, 1e-3f, "subtract corner notch volume");
    expectTrue(signedVolume(result) > 0.0f, "subtract result wound outward");
    expectClosed(result, "subtract normal sum ~ 0");
}

void testSubtractThroughHole() {
    const CsgMesh a = odai::procgen::makeBox({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, kGrey);
    const CsgMesh peg = odai::procgen::makeBox({0.4f, -0.1f, 0.4f}, {0.6f, 1.1f, 0.6f}, kGrey);
    const CsgMesh result = odai::procgen::csgSubtract(a, peg);
    expectNear(signedVolume(result), 1.0f - 0.04f, 1e-3f, "through-hole volume (genus 1)");
    expectClosed(result, "through-hole normal sum ~ 0");
}

void testIntersectBoxes() {
    const CsgMesh a = odai::procgen::makeBox({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, kGrey);
    const CsgMesh b = odai::procgen::makeBox({0.5f, 0.5f, 0.5f}, {1.5f, 1.5f, 1.5f}, kGrey);
    const CsgMesh result = odai::procgen::csgIntersect(a, b);
    expectNear(signedVolume(result), 0.125f, 1e-3f, "intersect volume = overlap");
    expectClosed(result, "intersect normal sum ~ 0");
    for (const Polygon& polygon : result.polygons) {
        for (const Vector3& v : polygon.vertices) {
            expectTrue(v.x > 0.5f - 1e-3f && v.x < 1.0f + 1e-3f && v.y > 0.5f - 1e-3f &&
                           v.y < 1.0f + 1e-3f && v.z > 0.5f - 1e-3f && v.z < 1.0f + 1e-3f,
                       "intersect vertices inside overlap box");
        }
    }
}

void testDisjointOps() {
    const CsgMesh a = odai::procgen::makeBox({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, kGrey);
    const CsgMesh b = odai::procgen::makeBox({2.0f, 0.0f, 0.0f}, {3.0f, 1.0f, 1.0f}, kGrey);
    expectNear(signedVolume(odai::procgen::csgUnion(a, b)), 2.0f, 1e-3f, "disjoint union volume");
    expectNear(signedVolume(odai::procgen::csgSubtract(a, b)), 1.0f, 1e-3f,
               "disjoint subtract leaves A intact");
}

void testCoplanarStackedUnion() {
    // Two boxes sharing an exact face — the classic epsilon killer.
    const CsgMesh a = odai::procgen::makeBox({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, kGrey);
    const CsgMesh b = odai::procgen::makeBox({0.0f, 1.0f, 0.0f}, {1.0f, 2.0f, 1.0f}, kGrey);
    const CsgMesh result = odai::procgen::csgUnion(a, b);
    expectNear(signedVolume(result), 2.0f, 1e-3f, "coplanar stacked union volume");
    expectClosed(result, "coplanar stacked union normal sum ~ 0");
}

void testTriangulateAndAppend() {
    const CsgMesh box = odai::procgen::makeBox({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, kGrey);
    const odai::procgen::TriMesh tri = odai::procgen::triangulate(box);
    expectEqualU32(static_cast<std::uint32_t>(tri.vertices.size()), 24u, "box TriMesh 24 vertices");
    expectEqualU32(static_cast<std::uint32_t>(tri.indices.size()), 36u, "box TriMesh 36 indices");
    for (const auto& v : tri.vertices) {
        expectEqualU32(v.textureIndex, 0xffffffffu, "vertex uses vertex-color path");
        const float normalLength = std::sqrt(v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] +
                                             v.normal[2] * v.normal[2]);
        expectNear(normalLength, 1.0f, 1e-4f, "vertex normal unit length");
    }
    expectNear(tri.boundsMin.x, 0.0f, 1e-6f, "TriMesh bounds min");
    expectNear(tri.boundsMax.y, 1.0f, 1e-6f, "TriMesh bounds max");

    odai::importer::ImportedScene scene;
    odai::procgen::appendTriMesh(tri, {10.0f, 0.0f, 5.0f}, {1.0f, 1.0f, 1.0f}, scene);
    odai::procgen::appendTriMesh(tri, {12.0f, 0.0f, 5.0f}, {0.5f, 0.5f, 0.5f}, scene);
    expectEqualU32(static_cast<std::uint32_t>(scene.packedVertices.size()), 48u,
                   "scene has both appended meshes");
    expectEqualU32(static_cast<std::uint32_t>(scene.packedIndices.size()), 72u,
                   "scene has both index ranges");
    std::uint32_t maxFirstBatch = 0;
    std::uint32_t minSecondBatch = 0xffffffffu;
    for (std::size_t i = 0; i < 36; ++i) {
        maxFirstBatch = std::max(maxFirstBatch, scene.packedIndices[i]);
        minSecondBatch = std::min(minSecondBatch, scene.packedIndices[i + 36]);
    }
    expectTrue(maxFirstBatch < 24u, "first batch indices unrebased");
    expectEqualU32(minSecondBatch, 24u, "second batch indices rebased by 24");
    expectNear(scene.packedVertices[0].position[0], 10.0f, 1e-6f, "append applies offset");
    expectNear(scene.packedVertices[24].color[0], 0.25f, 1e-6f, "append applies color multiplier");
    expectTrue(scene.boundsMax[0] >= 13.0f - 1e-3f, "scene bounds expanded by appends");
}

void testGeneratorDeterminism() {
    odai::procgen::BuildingDesc desc;
    desc.era = odai::procgen::Era::E1930s;
    desc.kind = odai::procgen::BuildingKind::Commercial;
    desc.level = 2;
    desc.seed = 0xC0FFEEu;
    const odai::procgen::TriMesh a = odai::procgen::generateBuilding(desc);
    const odai::procgen::TriMesh b = odai::procgen::generateBuilding(desc);
    expectEqualU32(static_cast<std::uint32_t>(a.vertices.size()),
                   static_cast<std::uint32_t>(b.vertices.size()), "determinism: vertex count");
    expectEqualU32(static_cast<std::uint32_t>(a.indices.size()),
                   static_cast<std::uint32_t>(b.indices.size()), "determinism: index count");
    bool identical = a.indices == b.indices;
    for (std::size_t i = 0; identical && i < a.vertices.size(); ++i) {
        identical = std::memcmp(&a.vertices[i], &b.vertices[i],
                                sizeof(odai::importer::ImportedScenePackedVertex)) == 0;
    }
    expectTrue(identical, "determinism: bit-identical streams");
}

void testAllEraGeneratorsValid() {
    const odai::procgen::Era eras[] = {odai::procgen::Era::E1890s, odai::procgen::Era::E1930s,
                                       odai::procgen::Era::E1960s};
    const odai::procgen::BuildingKind kinds[] = {odai::procgen::BuildingKind::Residential,
                                                 odai::procgen::BuildingKind::Commercial,
                                                 odai::procgen::BuildingKind::Industrial};
    for (const auto era : eras) {
        for (const auto kind : kinds) {
            for (int level = 1; level <= 3; ++level) {
                for (int tier = 0; tier <= 2; ++tier) {
                    for (std::uint32_t variant = 0; variant < 8; ++variant) {
                        odai::procgen::BuildingDesc desc;
                        desc.era = era;
                        desc.kind = kind;
                        desc.level = level;
                        desc.wealthTier = tier;
                        desc.lotWidth = 0.8f;
                        desc.lotDepth = 0.8f;
                        desc.seed = variant * 0x9E3779B9u + 17u;
                        const odai::procgen::TriMesh mesh = odai::procgen::generateBuilding(desc);
                        expectTrue(!mesh.vertices.empty(), "generator produces geometry");
                        expectTrue(mesh.indices.size() % 3 == 0, "index count is triangles");
                        expectTrue(mesh.indices.size() / 3 < 600, "triangle budget");
                        for (const std::uint32_t index : mesh.indices) {
                            expectTrue(index < mesh.vertices.size(), "indices in range");
                        }
                        for (const auto& v : mesh.vertices) {
                            expectTrue(v.position[0] > -0.06f && v.position[0] < 0.86f &&
                                           v.position[2] > -0.06f && v.position[2] < 0.86f,
                                       "positions within lot bounds");
                            expectTrue(v.position[1] > -0.06f && v.position[1] < 6.0f,
                                       "heights within sane range");
                            const float len =
                                std::sqrt(v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] +
                                          v.normal[2] * v.normal[2]);
                            expectTrue(std::isfinite(len) && std::fabs(len - 1.0f) < 1e-3f,
                                       "normals finite and unit");
                        }
                    }
                }
            }
        }
    }
}

void testVariantDiversity() {
    // The feature-draw grammar should produce visibly distinct buildings from
    // the same style bucket. Geometry counts are a cheap proxy: across 8
    // variants we expect at least 3 distinct (vertex, index) signatures.
    const odai::procgen::Era eras[] = {odai::procgen::Era::E1890s, odai::procgen::Era::E1930s,
                                       odai::procgen::Era::E1960s};
    for (const auto era : eras) {
        std::set<std::pair<std::size_t, std::size_t>> signatures;
        for (std::uint32_t variant = 0; variant < 8; ++variant) {
            odai::procgen::BuildingDesc desc;
            desc.era = era;
            desc.kind = odai::procgen::BuildingKind::Residential;
            desc.level = 2;
            desc.wealthTier = 1;
            desc.seed = variant * 0x9E3779B9u + 17u;
            const odai::procgen::TriMesh mesh = odai::procgen::generateBuilding(desc);
            signatures.emplace(mesh.vertices.size(), mesh.indices.size());
        }
        expectTrue(signatures.size() >= 3, "8 same-style variants span >= 3 distinct shapes");
    }
}

void testAppendRotated() {
    // A box rotated 4x90 degrees about the lot centre must land where the
    // unrotated append does; 1 turn must swap the footprint's long axis.
    const CsgMesh box = odai::procgen::makeBox({0.1f, 0.0f, 0.2f}, {0.7f, 0.5f, 0.4f}, kGrey);
    const odai::procgen::TriMesh tri = odai::procgen::triangulate(box);
    const Vector3 pivot{0.4f, 0.0f, 0.4f};

    odai::importer::ImportedScene plain;
    odai::procgen::appendTriMesh(tri, {2.0f, 0.0f, 3.0f}, {1.0f, 1.0f, 1.0f}, plain);
    odai::importer::ImportedScene full;
    odai::procgen::appendTriMeshRotated(tri, {2.0f, 0.0f, 3.0f}, 4, pivot, {1.0f, 1.0f, 1.0f}, full);
    expectEqualU32(static_cast<std::uint32_t>(full.packedVertices.size()),
                   static_cast<std::uint32_t>(plain.packedVertices.size()),
                   "4 quarter turns = plain append (count)");
    for (std::size_t i = 0; i < plain.packedVertices.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            expectNear(full.packedVertices[i].position[axis], plain.packedVertices[i].position[axis],
                       1e-5f, "4 quarter turns = plain append (positions)");
        }
    }

    odai::importer::ImportedScene turned;
    for (int axis = 0; axis < 3; ++axis) {
        turned.boundsMin[axis] = std::numeric_limits<float>::max();
        turned.boundsMax[axis] = std::numeric_limits<float>::lowest();
    }
    odai::procgen::appendTriMeshRotated(tri, {0.0f, 0.0f, 0.0f}, 1, pivot, {1.0f, 1.0f, 1.0f}, turned);
    // Original footprint: x in [0.1,0.7] (span 0.6), z in [0.2,0.4] (span 0.2).
    // After one turn about (0.4, 0.4) the spans swap axes.
    expectNear(turned.boundsMax[0] - turned.boundsMin[0], 0.2f, 1e-5f, "rotated span x");
    expectNear(turned.boundsMax[2] - turned.boundsMin[2], 0.6f, 1e-5f, "rotated span z");
}

void testNonSquareLots() {
    // Multi-tile plots feed the generators rectangular lots (2x1, 3x2, ...).
    // Every era/kind must keep its geometry inside the lot on both axes
    // independently — a recipe that assumes a square lot fails here.
    const odai::procgen::Era eras[] = {odai::procgen::Era::E1890s, odai::procgen::Era::E1930s,
                                       odai::procgen::Era::E1960s};
    const odai::procgen::BuildingKind kinds[] = {odai::procgen::BuildingKind::Residential,
                                                 odai::procgen::BuildingKind::Commercial,
                                                 odai::procgen::BuildingKind::Industrial};
    const struct {
        float w, d;
    } lots[] = {{1.8f, 0.8f}, {0.8f, 1.8f}, {2.8f, 1.8f}, {1.8f, 2.8f}};
    for (const auto era : eras) {
        for (const auto kind : kinds) {
            for (const auto& lot : lots) {
                for (std::uint32_t seed = 0; seed < 4; ++seed) {
                    odai::procgen::BuildingDesc desc;
                    desc.era = era;
                    desc.kind = kind;
                    desc.level = 2;
                    desc.wealthTier = 1;
                    desc.lotWidth = lot.w;
                    desc.lotDepth = lot.d;
                    desc.seed = seed * 0x9E3779B9u + 3u;
                    const odai::procgen::TriMesh mesh = odai::procgen::generateBuilding(desc);
                    expectTrue(!mesh.vertices.empty(), "non-square lot produces geometry");
                    expectTrue(mesh.indices.size() / 3 < 600, "non-square lot triangle budget");
                    for (const auto& v : mesh.vertices) {
                        expectTrue(v.position[0] > -0.06f && v.position[0] < lot.w + 0.06f,
                                   "non-square lot: x within lot");
                        expectTrue(v.position[2] > -0.06f && v.position[2] < lot.d + 0.06f,
                                   "non-square lot: z within lot");
                        expectTrue(v.position[1] > -0.06f && v.position[1] < 6.0f,
                                   "non-square lot: height sane");
                    }
                }
            }
        }
    }
}

void testProps() {
    for (std::uint32_t variant = 0; variant < 6; ++variant) {
        const odai::procgen::TriMesh tree = odai::procgen::generateTree(variant, 42u + variant);
        expectTrue(!tree.vertices.empty(), "tree produces geometry");
        expectTrue(tree.indices.size() / 3 < 200, "tree triangle budget");
        expectTrue(tree.boundsMin.y > -1e-4f, "tree sits on the ground");
        expectTrue(tree.boundsMax.y < 0.6f && tree.boundsMax.x < 0.25f,
                   "tree within prop bounds");
        const odai::procgen::TriMesh again = odai::procgen::generateTree(variant, 42u + variant);
        expectEqualU32(static_cast<std::uint32_t>(again.vertices.size()),
                       static_cast<std::uint32_t>(tree.vertices.size()), "tree deterministic");
    }
    for (std::uint32_t seed = 0; seed < 8; ++seed) {
        const odai::procgen::TriMesh car = odai::procgen::generateVehicle(0xCA5133Du + seed * 7919u);
        expectTrue(!car.vertices.empty(), "car produces geometry");
        expectTrue(car.indices.size() / 3 < 100, "car triangle budget");
        expectTrue(car.boundsMin.y > -1e-4f && car.boundsMax.y < 0.12f, "car height sane");
        expectTrue(car.boundsMax.x < 0.10f && car.boundsMin.x > -0.10f, "car length sane");
    }
    // Seasons: every variant stays valid in every season, and winter strips
    // the broadleaf canopy (fewer triangles than summer's foliage tiers).
    const odai::procgen::Season seasons[] = {
        odai::procgen::Season::Spring, odai::procgen::Season::Summer,
        odai::procgen::Season::Autumn, odai::procgen::Season::Winter};
    for (const auto season : seasons) {
        for (std::uint32_t variant = 0; variant < 6; ++variant) {
            const odai::procgen::TriMesh tree = odai::procgen::generateTree(variant, 42u + variant, season);
            expectTrue(!tree.vertices.empty(), "seasonal tree produces geometry");
            expectTrue(tree.indices.size() / 3 < 200, "seasonal tree triangle budget");
        }
    }
    const odai::procgen::TriMesh summerTree =
        odai::procgen::generateTree(0, 42u, odai::procgen::Season::Summer);
    const odai::procgen::TriMesh winterTree =
        odai::procgen::generateTree(0, 42u, odai::procgen::Season::Winter);
    expectTrue(winterTree.indices.size() < summerTree.indices.size(),
               "winter broadleaf is bare (lighter than summer)");
    for (std::uint32_t seed = 0; seed < 4; ++seed) {
        const odai::procgen::TriMesh pumpkin = odai::procgen::generatePumpkin(0xF00Du + seed * 131u);
        expectTrue(!pumpkin.vertices.empty(), "pumpkin produces geometry");
        expectTrue(pumpkin.indices.size() / 3 < 80, "pumpkin triangle budget");
        expectTrue(pumpkin.boundsMin.y > -1e-4f && pumpkin.boundsMax.y < 0.09f, "pumpkin squat");
    }
    for (std::uint32_t seed = 0; seed < 4; ++seed) {
        const odai::procgen::TriMesh pole = odai::procgen::generatePowerPole(0xB01Eu + seed * 197u);
        expectTrue(!pole.vertices.empty(), "power pole produces geometry");
        expectTrue(pole.indices.size() / 3 < 120, "power pole triangle budget");
        expectTrue(pole.boundsMin.y > -1e-4f && pole.boundsMax.y > 0.25f && pole.boundsMax.y < 0.40f,
                   "power pole height sane");
        expectTrue(pole.boundsMax.x > 0.04f, "power pole cross-arm present");
        const odai::procgen::TriMesh again = odai::procgen::generatePowerPole(0xB01Eu + seed * 197u);
        expectEqualU32(static_cast<std::uint32_t>(again.vertices.size()),
                       static_cast<std::uint32_t>(pole.vertices.size()), "power pole deterministic");
    }
    for (std::uint32_t seed = 0; seed < 4; ++seed) {
        const odai::procgen::TriMesh lamp = odai::procgen::generateStreetlamp(0x7A4Fu + seed * 211u);
        expectTrue(!lamp.vertices.empty(), "streetlamp produces geometry");
        expectTrue(lamp.indices.size() / 3 < 80, "streetlamp triangle budget");
        expectTrue(lamp.boundsMin.y > -1e-4f && lamp.boundsMax.y > 0.15f && lamp.boundsMax.y < 0.30f,
                   "streetlamp height sane");
    }
}

}  // namespace

int main() {
    testBoxPrimitive();
    testPrimitivesClosed();
    testUnionOverlappingBoxes();
    testSubtractCornerNotch();
    testSubtractThroughHole();
    testIntersectBoxes();
    testDisjointOps();
    testCoplanarStackedUnion();
    testTriangulateAndAppend();
    testGeneratorDeterminism();
    testAllEraGeneratorsValid();
    testVariantDiversity();
    testAppendRotated();
    testNonSquareLots();
    testProps();
    if (g_failures != 0) {
        std::cerr << "[procgen test] " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[procgen test] all tests passed\n";
    return 0;
}
