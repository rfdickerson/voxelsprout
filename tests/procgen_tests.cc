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
#include "procgen/city_terrain.h"
#include "procgen/civic_generator.h"
#include "procgen/props.h"
#include "procgen/rng.h"

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

void testRng() {
    odai::procgen::Rng a(0xC0FFEEu);
    odai::procgen::Rng b(0xC0FFEEu);
    for (int i = 0; i < 64; ++i) {
        expectEqualU32(a.next(), b.next(), "rng deterministic per seed");
    }
    odai::procgen::Rng r(7u);
    for (int i = 0; i < 256; ++i) {
        const int v = r.range(3, 9);
        expectTrue(v >= 3 && v <= 9, "rng range bounds inclusive");
        const float f = r.uniform(-2.0f, 5.0f);
        expectTrue(f >= -2.0f && f <= 5.0f, "rng uniform bounds");
    }
    expectTrue(odai::procgen::Rng(0u).next() == odai::procgen::Rng(1u).next(),
               "zero seed coerced to non-degenerate state");

    // hash2d: distinct inputs and salts decorrelate. (All-zero input maps to
    // zero by construction — callers salt position hashes, so that's fine.)
    expectTrue(odai::procgen::hash2d(0, 0, 0xABCDu) != 0u, "hash2d salted origin not zero");
    expectTrue(odai::procgen::hash2d(1, 0) != odai::procgen::hash2d(0, 1),
               "hash2d asymmetric in x/z");
    expectTrue(odai::procgen::hash2d(5, 9, 1u) != odai::procgen::hash2d(5, 9, 2u),
               "hash2d salt changes output");
    std::set<std::uint32_t> seen;
    for (int x = 0; x < 16; ++x) {
        for (int z = 0; z < 16; ++z) {
            seen.insert(odai::procgen::hash2d(x, z, 0xABCDu));
        }
    }
    expectTrue(seen.size() == 256u, "hash2d collision-free on a 16x16 patch");
}

void testCivicGenerators() {
    using odai::procgen::CivicDesc;
    using odai::procgen::CivicKind;
    const CivicKind kinds[] = {CivicKind::Police, CivicKind::Fire, CivicKind::Clinic,
                               CivicKind::School, CivicKind::Park, CivicKind::Library,
                               CivicKind::Amphitheater, CivicKind::PowerPlant};
    for (const CivicKind kind : kinds) {
        std::set<std::pair<std::size_t, std::size_t>> signatures;
        for (std::uint32_t variant = 0; variant < 4; ++variant) {
            CivicDesc desc;
            desc.kind = kind;
            desc.lotWidth = kind == CivicKind::Park ? 0.8f : 1.8f;
            desc.lotDepth = desc.lotWidth;
            desc.seed = variant * 0x9E3779B9u + 41u;
            const odai::procgen::TriMesh mesh = odai::procgen::generateCivicBuilding(desc);
            expectTrue(!mesh.vertices.empty(), "civic generator produces geometry");
            expectTrue(mesh.indices.size() % 3 == 0, "civic index count is triangles");
            expectTrue(mesh.indices.size() / 3 < 600, "civic triangle budget");
            for (const std::uint32_t index : mesh.indices) {
                expectTrue(index < mesh.vertices.size(), "civic indices in range");
            }
            for (const auto& v : mesh.vertices) {
                expectTrue(v.position[0] > -0.08f && v.position[0] < desc.lotWidth + 0.08f &&
                               v.position[2] > -0.12f && v.position[2] < desc.lotDepth + 0.08f,
                           "civic positions within lot bounds");
                expectTrue(v.position[1] > -0.06f && v.position[1] < 4.0f,
                           "civic heights within sane range");
                const float len = std::sqrt(v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] +
                                            v.normal[2] * v.normal[2]);
                expectTrue(std::isfinite(len) && std::fabs(len - 1.0f) < 1e-3f,
                           "civic normals finite and unit");
            }
            // Low kinds stay low; landmarks stay tall.
            if (kind == CivicKind::Park || kind == CivicKind::Amphitheater) {
                expectTrue(mesh.boundsMax.y < 0.9f, "park/amphitheater stays low");
            } else if (kind == CivicKind::PowerPlant) {
                expectTrue(mesh.boundsMax.y > 1.6f && mesh.boundsMax.y < 3.2f,
                           "power plant reads as a landmark");
            } else {
                expectTrue(mesh.boundsMax.y > 0.8f && mesh.boundsMax.y < 2.4f,
                           "civic mass height in band");
            }
            const odai::procgen::TriMesh again = odai::procgen::generateCivicBuilding(desc);
            expectTrue(again.vertices.size() == mesh.vertices.size() &&
                           again.indices == mesh.indices,
                       "civic generator deterministic");
            signatures.emplace(mesh.vertices.size(), mesh.indices.size());
        }
        expectTrue(signatures.size() >= 2, "4 civic variants span >= 2 distinct shapes");
    }
}

void testCityTerrain() {
    using odai::procgen::CityTerrain;
    using odai::procgen::CityTerrainDesc;

    std::set<std::uint32_t> mapHashes;
    for (std::uint32_t seed = 1; seed <= 20; ++seed) {
        CityTerrainDesc desc;
        desc.seed = seed * 0x1F123BB5u + 7u;
        const CityTerrain t = odai::procgen::generateCityTerrain(desc);
        expectTrue(t.width == 56 && t.height == 56, "terrain dimensions");
        expectTrue(t.water.size() == 56u * 56u && t.forest.size() == 56u * 56u,
                   "terrain grids sized");
        expectTrue(t.valid, "terrain invariants hold (or a retry found a valid map)");
        expectTrue(!t.riverPath.empty(), "river path recorded");

        // Determinism: bit-identical water grid and site on a second run.
        const CityTerrain u = odai::procgen::generateCityTerrain(desc);
        expectTrue(t.water == u.water, "terrain deterministic per seed");
        expectTrue(t.siteC == u.siteC && t.siteR == u.siteR, "site deterministic per seed");

        // Land fraction.
        int land = 0;
        for (const std::uint8_t w : t.water) land += (w == 0u) ? 1 : 0;
        expectTrue(static_cast<float>(land) / static_cast<float>(t.water.size()) >= 0.55f,
                   "terrain land fraction >= 55%");

        // Forest mask in range.
        for (const float f : t.forest) {
            expectTrue(f >= 0.0f && f <= 1.0f, "forest mask within [0,1]");
        }

        // River connectivity: BFS over water from one map edge must reach the
        // opposite edge (boats depend on this).
        {
            std::vector<int> dist(t.water.size(), -1);
            std::vector<std::pair<int, int>> queue;
            for (int r = 0; r < t.height; ++r) {
                for (int c = 0; c < t.width; ++c) {
                    const bool edge = c == 0 || c == t.width - 1 || r == 0 || r == t.height - 1;
                    if (edge && t.water[static_cast<std::size_t>(r) * t.width + c] != 0u) {
                        queue.push_back({c, r});
                        dist[static_cast<std::size_t>(r) * t.width + c] = 0;
                    }
                }
            }
            for (std::size_t head = 0; head < queue.size(); ++head) {
                const auto [c, r] = queue[head];
                const int nc[4] = {c - 1, c + 1, c, c};
                const int nr[4] = {r, r, r - 1, r + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nc[k] < 0 || nc[k] >= t.width || nr[k] < 0 || nr[k] >= t.height) continue;
                    const std::size_t idx = static_cast<std::size_t>(nr[k]) * t.width + nc[k];
                    if (dist[idx] != -1 || t.water[idx] == 0u) continue;
                    dist[idx] = 1;
                    queue.push_back({nc[k], nr[k]});
                }
            }
            bool allRiverReached = true;
            for (const auto& [pc, pr] : t.riverPath) {
                if (t.water[static_cast<std::size_t>(pr) * t.width + pc] == 0u ||
                    dist[static_cast<std::size_t>(pr) * t.width + pc] == -1) {
                    allRiverReached = false;
                }
            }
            expectTrue(allRiverReached, "river path is wet and edge-connected");
        }

        // City site: a mostly-grass window around the anchor.
        {
            int grass = 0, total = 0;
            for (int r = t.siteR - 8; r < t.siteR + 8; ++r) {
                for (int c = t.siteC - 12; c < t.siteC + 12; ++c) {
                    if (c < 0 || c >= t.width || r < 0 || r >= t.height) continue;
                    ++total;
                    if (t.water[static_cast<std::size_t>(r) * t.width + c] == 0u) ++grass;
                }
            }
            expectTrue(total > 0 && grass * 10 >= total * 7, "site window is >= 70% buildable");
        }

        std::uint32_t hash = 2166136261u;
        for (const std::uint8_t w : t.water) hash = (hash ^ w) * 16777619u;
        mapHashes.insert(hash);
    }
    expectTrue(mapHashes.size() >= 18u, "different seeds produce different maps");
}

void testProps() {
    // 8 species: broadleaf, conifer, birch, poplar, willow, blossom, oak, shrub.
    std::set<std::pair<std::size_t, float>> treeShapes;
    for (std::uint32_t variant = 0; variant < 8; ++variant) {
        const odai::procgen::TriMesh tree = odai::procgen::generateTree(variant, 42u + variant);
        expectTrue(!tree.vertices.empty(), "tree produces geometry");
        expectTrue(tree.indices.size() / 3 < 200, "tree triangle budget");
        expectTrue(tree.boundsMin.y > -1e-4f, "tree sits on the ground");
        expectTrue(tree.boundsMax.y < 0.6f && tree.boundsMax.x < 0.25f,
                   "tree within prop bounds");
        const odai::procgen::TriMesh again = odai::procgen::generateTree(variant, 42u + variant);
        expectEqualU32(static_cast<std::uint32_t>(again.vertices.size()),
                       static_cast<std::uint32_t>(tree.vertices.size()), "tree deterministic");
        treeShapes.emplace(tree.vertices.size(), tree.boundsMax.y);
    }
    expectTrue(treeShapes.size() >= 6, "species have distinct silhouettes");
    // Species character: the poplar is the tallest narrow tree, the willow is
    // wider than it is tall, the shrub stays knee-high.
    const odai::procgen::TriMesh poplar = odai::procgen::generateTree(3, 7u);
    expectTrue(poplar.boundsMax.y > 0.30f && poplar.boundsMax.x < 0.08f, "poplar tall and narrow");
    const odai::procgen::TriMesh willow = odai::procgen::generateTree(4, 7u);
    expectTrue(willow.boundsMax.x > willow.boundsMax.y * 0.6f, "willow reads wide");
    const odai::procgen::TriMesh shrub = odai::procgen::generateTree(7, 7u);
    expectTrue(shrub.boundsMax.y < 0.13f, "shrub stays knee-high");
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
        for (std::uint32_t variant = 0; variant < 8; ++variant) {
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
    for (std::uint32_t seed = 0; seed < 4; ++seed) {
        const odai::procgen::TriMesh bench = odai::procgen::generateBench(0xBE7C4u + seed * 401u);
        expectTrue(!bench.vertices.empty(), "bench produces geometry");
        expectTrue(bench.indices.size() / 3 < 60, "bench triangle budget");
        expectTrue(bench.boundsMin.y > -1e-4f && bench.boundsMax.y < 0.08f, "bench squat");

        const odai::procgen::TriMesh hydrant = odai::procgen::generateHydrant(0x94D64u + seed * 613u);
        expectTrue(!hydrant.vertices.empty(), "hydrant produces geometry");
        expectTrue(hydrant.indices.size() / 3 < 60, "hydrant triangle budget");
        expectTrue(hydrant.boundsMin.y > -1e-4f && hydrant.boundsMax.y < 0.06f, "hydrant tiny");

        const odai::procgen::TriMesh billboard =
            odai::procgen::generateBillboard(0xB111Bu + seed * 761u);
        expectTrue(!billboard.vertices.empty(), "billboard produces geometry");
        expectTrue(billboard.indices.size() / 3 < 70, "billboard triangle budget");
        expectTrue(billboard.boundsMax.y > 0.20f && billboard.boundsMax.y < 0.36f,
                   "billboard height sane");

        const odai::procgen::TriMesh busStop = odai::procgen::generateBusStop(0xB0557u + seed * 883u);
        expectTrue(!busStop.vertices.empty(), "bus stop produces geometry");
        expectTrue(busStop.indices.size() / 3 < 110, "bus stop triangle budget");
        expectTrue(busStop.boundsMax.y > 0.10f && busStop.boundsMax.y < 0.20f,
                   "bus stop height sane");

        const odai::procgen::TriMesh rowboat = odai::procgen::generateBoat(0, 0xB0A7u + seed * 4409u);
        const odai::procgen::TriMesh barge = odai::procgen::generateBoat(1, 0xB0A7u + seed * 4409u);
        expectTrue(!rowboat.vertices.empty() && !barge.vertices.empty(), "boats produce geometry");
        expectTrue(rowboat.indices.size() / 3 < 90 && barge.indices.size() / 3 < 90,
                   "boat triangle budgets");
        expectTrue(barge.boundsMax.x - barge.boundsMin.x >
                       rowboat.boundsMax.x - rowboat.boundsMin.x,
                   "barge longer than rowboat");
        expectTrue(rowboat.boundsMin.y < 0.0f, "boat hull sits below the waterline");

        const odai::procgen::TriMesh ped = odai::procgen::generatePedestrian(0x9ED0u + seed * 331u);
        expectTrue(!ped.vertices.empty(), "pedestrian produces geometry");
        expectTrue(ped.indices.size() / 3 < 40, "pedestrian triangle budget");
        expectTrue(ped.boundsMin.y > -1e-4f && ped.boundsMax.y > 0.05f && ped.boundsMax.y < 0.10f,
                   "pedestrian height sane");
        const odai::procgen::TriMesh pedAgain = odai::procgen::generatePedestrian(0x9ED0u + seed * 331u);
        expectTrue(pedAgain.vertices.size() == ped.vertices.size(), "pedestrian deterministic");

        const odai::procgen::TriMesh bus = odai::procgen::generateSchoolBus(0x5CB005u + seed * 577u);
        expectTrue(!bus.vertices.empty(), "school bus produces geometry");
        expectTrue(bus.indices.size() / 3 < 110, "school bus triangle budget");
        expectTrue(bus.boundsMax.x - bus.boundsMin.x > 0.15f, "school bus reads long");
        expectTrue(bus.boundsMin.y > -1e-4f && bus.boundsMax.y < 0.11f, "school bus height sane");

        const odai::procgen::TriMesh truck =
            odai::procgen::generateGarbageTruck(0x6A3BA6Eu + seed * 733u);
        expectTrue(!truck.vertices.empty(), "garbage truck produces geometry");
        expectTrue(truck.indices.size() / 3 < 110, "garbage truck triangle budget");
        expectTrue(truck.boundsMin.y > -1e-4f && truck.boundsMax.y < 0.11f,
                   "garbage truck height sane");

        const odai::procgen::TriMesh can = odai::procgen::generateTrashCan(0x7245Cu + seed * 449u);
        expectTrue(!can.vertices.empty(), "trash can produces geometry");
        expectTrue(can.indices.size() / 3 < 50, "trash can triangle budget");
        expectTrue(can.boundsMin.y > -1e-4f && can.boundsMax.y < 0.05f, "trash can tiny");
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
    testRng();
    testCivicGenerators();
    testCityTerrain();
    testProps();
    if (g_failures != 0) {
        std::cerr << "[procgen test] " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[procgen test] all tests passed\n";
    return 0;
}
