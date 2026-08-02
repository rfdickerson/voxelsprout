#include "core/grid3.h"
#include "core/hash.h"

#include <cstdint>
#include <iostream>
#include <unordered_set>

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

}  // namespace

int main() {
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
