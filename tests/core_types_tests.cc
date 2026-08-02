#include "core/grid3.h"
#include "core/hash.h"
#include "core/lcg.h"
#include "core/ring_buffer.h"
#include "math/geometry.h"
#include "math/math.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        ++g_failures;
        std::cerr << "[core types test] FAILED: " << message << "\n";
    }
}

// The coordinate hashes below feed saved worlds and .smap files, so these are
// golden vectors captured from the implementations that lived in
// procgen/rng.h, world/chunk_grid_worldgen.cc, game/game_sim.cc, and
// tools/strategy_map_gen_main.cc before they were merged into core/hash.h.
// A change here means previously generated content no longer reproduces.
void testCoordinateHashGoldenVectors() {
    using odai::core::hashCoords2d;
    expectTrue(hashCoords2d(17, -5, 3) == 0xFF9C24B0u, "hashCoords2d(17,-5,3) golden");
    expectTrue(hashCoords2d(0, 0, 0) == 0x00000000u, "hashCoords2d(0,0,0) golden (zero maps to zero)");
    expectTrue(hashCoords2d(-1, -1, 7) == 0x95B3ED94u, "hashCoords2d(-1,-1,7) golden");
    expectTrue(hashCoords2d(128, 64) == 0x6191C014u, "hashCoords2d default salt golden");

    using odai::core::hashCoordsSeeded;
    expectTrue(hashCoordsSeeded(17, -5, 3u) == 0xAEC705A8u, "hashCoordsSeeded(17,-5,3) golden");
    expectTrue(hashCoordsSeeded(0, 0, 0u) == 0x0E7A4677u, "hashCoordsSeeded(0,0,0) golden");
    expectTrue(hashCoordsSeeded(-1, -1, 7u) == 0x5356AF71u, "hashCoordsSeeded(-1,-1,7) golden");

    // The two are deliberately different algorithms; merging them would change
    // one of the two content pipelines.
    expectTrue(hashCoords2d(17, -5, 3) != hashCoordsSeeded(17, -5, 3u),
               "the two coordinate hashes stay distinct");
}

void testPackCell21GoldenVectors() {
    using odai::core::Cell3i;
    using odai::core::packCell21;
    expectTrue(packCell21(Cell3i{1, -2, 3}) == 0x00000FFFFFC00001ull, "packCell21(1,-2,3) golden");
    expectTrue(packCell21(Cell3i{0, 0, 0}) == 0x0000000000000000ull, "packCell21(0,0,0) golden");
    expectTrue(packCell21(Cell3i{-1, -1, -1}) == 0x7FFFFFFFFFFFFFFFull, "packCell21(-1,-1,-1) golden");

    // Distinct cells inside the representable range must not collide.
    expectTrue(packCell21(Cell3i{1, 0, 0}) != packCell21(Cell3i{0, 1, 0}), "packCell21 separates x from y");
    expectTrue(packCell21(Cell3i{0, 1, 0}) != packCell21(Cell3i{0, 0, 1}), "packCell21 separates y from z");
}

// hashCell3 replaced `hx ^ (hy << 1) ^ (hz << 2)`, which piled whole
// axis-aligned rows of chunks into a handful of buckets. This pins the
// improvement so nobody "simplifies" it back.
void testCell3HashSpreadsAxisAlignedCells() {
    constexpr int kSizeX = 32;
    constexpr int kSizeY = 8;
    constexpr int kSizeZ = 32;
    constexpr std::size_t kTotal = static_cast<std::size_t>(kSizeX) * kSizeY * kSizeZ;

    std::unordered_set<std::size_t> newHashes;
    std::unordered_set<std::size_t> oldHashes;
    for (int x = 0; x < kSizeX; ++x) {
        for (int y = 0; y < kSizeY; ++y) {
            for (int z = 0; z < kSizeZ; ++z) {
                newHashes.insert(odai::core::hashCell3(x, y, z));
                const std::size_t hx = static_cast<std::size_t>(x);
                const std::size_t hy = static_cast<std::size_t>(y);
                const std::size_t hz = static_cast<std::size_t>(z);
                oldHashes.insert(hx ^ (hy << 1u) ^ (hz << 2u));
            }
        }
    }

    expectTrue(newHashes.size() == kTotal, "hashCell3 is collision-free over a 32x8x32 block");
    expectTrue(oldHashes.size() < kTotal / 2u,
               "the replaced XOR-of-shifts really did collide (guards against reverting)");

    expectTrue(odai::core::hashCell3(odai::core::Cell3i{5, -3, 11}) == odai::core::hashCell3(5, -3, 11),
               "Cell3i overload matches the scalar overload");
    expectTrue(odai::core::Cell3Hash{}(odai::core::Cell3i{5, -3, 11}) == odai::core::hashCell3(5, -3, 11),
               "Cell3Hash functor matches hashCell3");
}

void testCell3HashHandlesNegativeCoordinates() {
    std::unordered_set<std::size_t> hashes;
    std::size_t count = 0;
    for (int x = -8; x < 8; ++x) {
        for (int y = -8; y < 8; ++y) {
            for (int z = -8; z < 8; ++z) {
                hashes.insert(odai::core::hashCell3(x, y, z));
                ++count;
            }
        }
    }
    expectTrue(hashes.size() == count, "hashCell3 is collision-free across the sign boundary");
}

void testMixersAvalanche() {
    expectTrue(odai::core::mix32(0u) == 0u, "mix32(0) is 0 (multiplicative finalizer)");
    expectTrue(odai::core::mix64(0u) == 0u, "mix64(0) is 0 (multiplicative finalizer)");
    expectTrue(odai::core::mix32(1u) != odai::core::mix32(2u), "mix32 separates adjacent inputs");
    expectTrue(odai::core::mix64(1u) != odai::core::mix64(2u), "mix64 separates adjacent inputs");

    // Adjacent inputs should differ in a large number of output bits.
    const std::uint64_t a = odai::core::mix64(1000u);
    const std::uint64_t b = odai::core::mix64(1001u);
    int differingBits = 0;
    for (int bit = 0; bit < 64; ++bit) {
        if (((a >> bit) & 1ull) != ((b >> bit) & 1ull)) {
            ++differingBits;
        }
    }
    expectTrue(differingBits > 16, "mix64 avalanches adjacent inputs across many bits");
}

void testScalarHelpersMatchTheImplementationsTheyReplaced() {
    using odai::math::lerp;
    using odai::math::saturate;
    using odai::math::smoothstepUnit;

    expectTrue(saturate(-0.5f) == 0.0f, "saturate clamps below zero");
    expectTrue(saturate(1.5f) == 1.0f, "saturate clamps above one");
    expectTrue(saturate(0.25f) == 0.25f, "saturate passes through the unit range");

    // The four call sites this replaced used three spellings of the same
    // thing; assert they still agree bit-for-bit across the range.
    bool allAgree = true;
    for (int i = -3000; i <= 3000; ++i) {
        const float v = static_cast<float>(i) * 0.001f;
        const float ternary = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        const float branch = (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
        allAgree = allAgree && saturate(v) == ternary && saturate(v) == branch;
        allAgree = allAgree && smoothstepUnit(v) == v * v * (3.0f - (2.0f * v));
    }
    expectTrue(allAgree, "saturate/smoothstepUnit match the replaced bodies bit-for-bit");

    expectTrue(lerp(2.0f, 4.0f, 0.0f) == 2.0f, "lerp at t=0");
    expectTrue(lerp(2.0f, 4.0f, 1.0f) == 4.0f, "lerp at t=1");
    expectTrue(lerp(2.0f, 4.0f, 0.5f) == 3.0f, "lerp at midpoint");
    expectTrue(lerp(2.0f, 4.0f, 2.0f) == 6.0f, "lerp extrapolates (not clamped)");

    const odai::math::Vector3 a{0.0f, 1.0f, 2.0f};
    const odai::math::Vector3 b{4.0f, 5.0f, 6.0f};
    const odai::math::Vector3 mid = lerp(a, b, 0.5f);
    expectTrue(mid.x == 2.0f && mid.y == 3.0f && mid.z == 4.0f, "Vector3 lerp is component-wise");
}

void testVector2() {
    using odai::math::Vector2;
    constexpr Vector2 a{3.0f, 4.0f};
    expectTrue(odai::math::lengthSquared(a) == 25.0f, "Vector2 lengthSquared");
    expectTrue(odai::math::length(a) == 5.0f, "Vector2 length");
    expectTrue(odai::math::dot(a, Vector2{1.0f, 0.0f}) == 3.0f, "Vector2 dot");

    const Vector2 unit = odai::math::normalize(a);
    expectTrue(std::abs(odai::math::length(unit) - 1.0f) < 1e-6f, "Vector2 normalize yields unit length");

    const Vector2 zero = odai::math::normalize(Vector2{0.0f, 0.0f});
    expectTrue(zero == Vector2{0.0f, 0.0f}, "Vector2 normalize of zero stays zero (no NaN)");

    expectTrue((a + Vector2{1.0f, 1.0f}) == Vector2{4.0f, 5.0f}, "Vector2 addition");
    expectTrue((a - Vector2{1.0f, 1.0f}) == Vector2{2.0f, 3.0f}, "Vector2 subtraction");
    expectTrue((a * 2.0f) == Vector2{6.0f, 8.0f}, "Vector2 scalar multiply");
    expectTrue((2.0f * a) == Vector2{6.0f, 8.0f}, "Vector2 scalar multiply is commutative");
    expectTrue((-a) == Vector2{-3.0f, -4.0f}, "Vector2 negation");
}

void testRingBuffer() {
    odai::core::RingBuffer<float, 4> ring;
    expectTrue(ring.empty(), "a fresh ring is empty");
    expectTrue(ring.size() == 0u, "a fresh ring has size 0");
    expectTrue(!ring.full(), "a fresh ring is not full");
    expectTrue(ring.orderedStartIndex() == 0u, "an unwrapped ring starts at 0");

    ring.push(1.0f);
    ring.push(2.0f);
    expectTrue(ring.size() == 2u, "partial fill tracks size");
    expectTrue(ring[0] == 1.0f && ring[1] == 2.0f, "partial fill reads oldest-first");
    expectTrue(ring.orderedStartIndex() == 0u, "a partially filled ring still starts at 0");

    ring.push(3.0f);
    ring.push(4.0f);
    expectTrue(ring.full() && ring.size() == 4u, "the ring reports full at capacity");
    expectTrue(ring[0] == 1.0f && ring[3] == 4.0f, "a just-filled ring is still in order");

    // Overwrite: 1 falls off, 5 becomes newest.
    ring.push(5.0f);
    expectTrue(ring.size() == 4u, "size saturates at capacity");
    expectTrue(ring[0] == 2.0f, "the oldest sample is dropped on overflow");
    expectTrue(ring[3] == 5.0f, "the newest sample lands last");
    expectTrue(ring.orderedStartIndex() == ring.writeIndex(), "a wrapped ring starts at the write cursor");

    ring.clear();
    expectTrue(ring.empty() && ring.writeIndex() == 0u, "clear resets cursors");
    expectTrue(ring.data()[0] == 0.0f, "clear zero-fills the storage");
}

void testRingBufferPercentile() {
    odai::core::RingBuffer<float, 240> ring;
    expectTrue(odai::core::percentile(ring, 0.5f) == 0.0f, "percentile of an empty ring is 0");

    // A known ramp 1..100; nearest-rank with a ceil-based index.
    for (int i = 1; i <= 100; ++i) {
        ring.push(static_cast<float>(i));
    }
    expectTrue(odai::core::percentile(ring, 0.0f) == 1.0f, "p0 is the minimum");
    expectTrue(odai::core::percentile(ring, 1.0f) == 100.0f, "p100 is the maximum");
    expectTrue(odai::core::percentile(ring, 0.5f) == 51.0f, "p50 uses the ceil-based rank");

    // Out-of-range percentiles clamp rather than reading out of bounds.
    expectTrue(odai::core::percentile(ring, -1.0f) == 1.0f, "percentile clamps below 0");
    expectTrue(odai::core::percentile(ring, 2.0f) == 100.0f, "percentile clamps above 1");

    // Percentiles must be computed over the live window only, not stale slots.
    odai::core::RingBuffer<float, 4> small;
    for (float v : {100.0f, 100.0f, 100.0f, 100.0f, 1.0f, 1.0f, 1.0f, 1.0f}) {
        small.push(v);
    }
    expectTrue(odai::core::percentile(small, 1.0f) == 1.0f, "overwritten samples leave the window");
}

void testPushBounded() {
    std::vector<int> values;
    for (int i = 0; i < 6; ++i) {
        odai::core::pushBounded(values, i, 3u);
    }
    expectTrue(values.size() == 3u, "pushBounded caps the size");
    expectTrue(values[0] == 3 && values[1] == 4 && values[2] == 5, "pushBounded drops from the front");

    std::vector<int> capOne;
    odai::core::pushBounded(capOne, 7, 1u);
    odai::core::pushBounded(capOne, 8, 1u);
    expectTrue(capOne.size() == 1u && capOne[0] == 8, "a cap of 1 keeps only the newest");

    std::vector<int> capZero;
    odai::core::pushBounded(capZero, 9, 0u);
    expectTrue(capZero.empty(), "a cap of 0 keeps nothing");

    // Matches the citybuilder/swtor idiom it replaced: push, then trim.
    std::vector<int> reference;
    std::vector<int> migrated;
    for (int i = 0; i < 40; ++i) {
        reference.push_back(i);
        if (reference.size() > 14u) {
            reference.erase(reference.begin());
        }
        odai::core::pushBounded(migrated, i, 14u);
    }
    expectTrue(reference == migrated, "pushBounded matches the hand-rolled drop-oldest loop");
}

// Every generated world, .smap, citybuilder layout, and swtor inventory roll
// depends on these exact outputs. The reference expression below is the
// pre-migration step, copied verbatim from the 29 sites that had it inline.
void testLcgIsBitExactWithTheReplacedInlineCopies() {
    bool allMatch = true;
    for (std::uint32_t seed : {1u, 0xC0FFEE17u, 0xDAEDBEEFu, 0xBAD5EEDu, 0xFFFFFFFFu}) {
        std::uint32_t reference = seed;
        std::uint32_t viaFreeFunction = seed;
        odai::core::Lcg32 viaObject{seed};
        for (int i = 0; i < 1000; ++i) {
            reference = reference * 1664525u + 1013904223u;
            allMatch = allMatch && odai::core::lcgNext(viaFreeFunction) == reference;
            allMatch = allMatch && viaFreeFunction == reference;
            allMatch = allMatch && viaObject.nextState() == reference;
            allMatch = allMatch && viaObject.state() == reference;
        }
    }
    expectTrue(allMatch, "lcgNext and Lcg32 reproduce the inline LCG step exactly");

    std::uint32_t state = 12345u;
    std::uint32_t shifted = 12345u;
    bool topBitsMatch = true;
    for (int i = 0; i < 1000; ++i) {
        shifted = shifted * 1664525u + 1013904223u;
        topBitsMatch = topBitsMatch && odai::core::lcgNext24(state) == (shifted >> 8);
    }
    expectTrue(topBitsMatch, "lcgNext24 returns the top 24 bits of the new state");

    // A zero seed is a fixed point for this LCG only if the increment were 0;
    // assert it still advances, since several call sites can seed from 0.
    std::uint32_t zero = 0u;
    expectTrue(odai::core::lcgNext(zero) == 1013904223u, "a zero seed still advances");
}

void testAabb() {
    using odai::math::Aabb3f;
    using odai::math::Vector3;

    const Aabb3f unit = odai::math::makeAabbFromCenterExtents(Vector3{0.0f, 0.0f, 0.0f},
                                                              Vector3{1.0f, 2.0f, 3.0f});
    expectTrue(unit.minX == -1.0f && unit.maxX == 1.0f, "center/extents sets x");
    expectTrue(unit.minY == -2.0f && unit.maxY == 2.0f, "center/extents sets y");
    expectTrue(unit.minZ == -3.0f && unit.maxZ == 3.0f, "center/extents sets z");

    // Corners in either order must produce the same box.
    const Aabb3f a = odai::math::makeAabbFromCorners(Vector3{1.0f, 2.0f, 3.0f}, Vector3{-1.0f, 0.0f, 5.0f});
    const Aabb3f b = odai::math::makeAabbFromCorners(Vector3{-1.0f, 0.0f, 5.0f}, Vector3{1.0f, 2.0f, 3.0f});
    expectTrue(a == b, "makeAabbFromCorners is order-independent");
    expectTrue(a.minZ == 3.0f && a.maxZ == 5.0f, "makeAabbFromCorners sorts each axis");

    const Vector3 center = odai::math::aabbCenter(a);
    expectTrue(center.x == 0.0f && center.y == 1.0f && center.z == 4.0f, "aabbCenter");

    expectTrue(odai::math::aabbContainsPoint(unit, Vector3{0.0f, 0.0f, 0.0f}), "contains interior point");
    expectTrue(odai::math::aabbContainsPoint(unit, Vector3{1.0f, 2.0f, 3.0f}), "contains corner point");
    expectTrue(!odai::math::aabbContainsPoint(unit, Vector3{1.1f, 0.0f, 0.0f}), "excludes exterior point");

    Aabb3f growing{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    odai::math::expandAabb(growing, Vector3{2.0f, -3.0f, 0.5f});
    expectTrue(growing.maxX == 2.0f && growing.minY == -3.0f && growing.maxZ == 0.5f, "expandAabb grows both ways");

    // Overlap, including the epsilon behavior both collision call sites rely on.
    const Aabb3f left{0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    const Aabb3f right{0.5f, 1.5f, 0.0f, 1.0f, 0.0f, 1.0f};
    const Aabb3f touching{1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    expectTrue(odai::math::aabbOverlaps(left, right), "overlapping boxes overlap");
    expectTrue(!odai::math::aabbOverlaps(left, touching), "exactly touching boxes do not overlap");
    expectTrue(!odai::math::aabbOverlaps(left, touching, 0.001f), "touching boxes still miss with epsilon");
    const Aabb3f barely{0.9995f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    expectTrue(odai::math::aabbOverlaps(left, barely), "a sliver of overlap counts at epsilon 0");
    expectTrue(!odai::math::aabbOverlaps(left, barely, 0.001f), "epsilon suppresses a sub-epsilon overlap");
}

void testRayIntersection() {
    using odai::math::Aabb3f;
    using odai::math::Ray;
    using odai::math::Vector3;

    const Vector3 v0{0.0f, 0.0f, 0.0f};
    const Vector3 v1{1.0f, 0.0f, 0.0f};
    const Vector3 v2{0.0f, 1.0f, 0.0f};

    const auto straightOn = odai::math::intersectRayTriangle(
        Ray{Vector3{0.25f, 0.25f, -2.0f}, Vector3{0.0f, 0.0f, 1.0f}}, v0, v1, v2);
    expectTrue(straightOn.hit, "ray through the triangle hits");
    expectTrue(std::abs(straightOn.distance - 2.0f) < 1e-5f, "hit distance is correct");
    expectTrue(std::abs(straightOn.u - 0.25f) < 1e-5f && std::abs(straightOn.v - 0.25f) < 1e-5f,
               "barycentric weights are correct");

    // Backfaces still hit: the original inline test was double-sided.
    const auto fromBehind = odai::math::intersectRayTriangle(
        Ray{Vector3{0.25f, 0.25f, 2.0f}, Vector3{0.0f, 0.0f, -1.0f}}, v0, v1, v2);
    expectTrue(fromBehind.hit, "the test is double-sided, matching the code it replaced");

    const auto outside = odai::math::intersectRayTriangle(
        Ray{Vector3{0.9f, 0.9f, -2.0f}, Vector3{0.0f, 0.0f, 1.0f}}, v0, v1, v2);
    expectTrue(!outside.hit, "a ray past the hypotenuse misses");

    const auto parallel = odai::math::intersectRayTriangle(
        Ray{Vector3{0.25f, 0.25f, 0.0f}, Vector3{1.0f, 0.0f, 0.0f}}, v0, v1, v2);
    expectTrue(!parallel.hit, "a ray in the triangle's plane misses");

    const auto behindOrigin = odai::math::intersectRayTriangle(
        Ray{Vector3{0.25f, 0.25f, 2.0f}, Vector3{0.0f, 0.0f, 1.0f}}, v0, v1, v2);
    expectTrue(!behindOrigin.hit, "a triangle behind the ray origin misses");

    const Aabb3f box{0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    float distance = 0.0f;
    expectTrue(odai::math::intersectRayAabb(
                   Ray{Vector3{0.5f, 0.5f, -3.0f}, Vector3{0.0f, 0.0f, 1.0f}}, box, 0.0f, 100.0f, distance),
               "ray into the box hits");
    expectTrue(std::abs(distance - 3.0f) < 1e-5f, "box entry distance is correct");
    expectTrue(!odai::math::intersectRayAabb(
                   Ray{Vector3{5.0f, 5.0f, -3.0f}, Vector3{0.0f, 0.0f, 1.0f}}, box, 0.0f, 100.0f, distance),
               "ray beside the box misses");
    expectTrue(!odai::math::intersectRayAabb(
                   Ray{Vector3{5.0f, 0.5f, 0.5f}, Vector3{0.0f, 0.0f, 1.0f}}, box, 0.0f, 100.0f, distance),
               "ray parallel to a slab and outside it misses");
    expectTrue(!odai::math::intersectRayAabb(
                   Ray{Vector3{0.5f, 0.5f, -3.0f}, Vector3{0.0f, 0.0f, 1.0f}}, box, 0.0f, 1.0f, distance),
               "tMax bounds the search");
}

}  // namespace

int main() {
    testAabb();
    testRayIntersection();
    testLcgIsBitExactWithTheReplacedInlineCopies();
    testRingBuffer();
    testRingBufferPercentile();
    testPushBounded();
    testScalarHelpersMatchTheImplementationsTheyReplaced();
    testVector2();
    testCoordinateHashGoldenVectors();
    testPackCell21GoldenVectors();
    testCell3HashSpreadsAxisAlignedCells();
    testCell3HashHandlesNegativeCoordinates();
    testMixersAvalanche();

    if (g_failures != 0) {
        std::cerr << "[core types test] " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[core types test] all checks passed\n";
    return 0;
}
