// Tests for the geometry arena suballocator behind imported-scene streaming.
//
// The invariant that actually matters is coalescing: a streaming session adds
// and removes cells for hours, and an arena that fails to merge freed
// neighbours looks perfectly healthy right up until it cannot satisfy an
// allocation despite being mostly empty. Most of what follows is aimed at that.

#include "render/gpu_arena_allocator.h"

#include <cstdint>
#include <iostream>
#include <vector>

using odai::render::GpuArenaAllocator;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " << #cond \
                      << "\n";                                                   \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

constexpr std::uint64_t kInvalid = GpuArenaAllocator::kInvalidOffset;

void testBasicAllocateAndFree() {
    GpuArenaAllocator arena;
    arena.reset(1024);
    CHECK(arena.capacity() == 1024u);
    CHECK(arena.used() == 0u);
    CHECK(arena.freeBlockCount() == 1u);

    const std::uint64_t a = arena.allocate(256, 1);
    CHECK(a == 0u);
    CHECK(arena.used() == 256u);

    const std::uint64_t b = arena.allocate(256, 1);
    CHECK(b == 256u);
    CHECK(arena.used() == 512u);

    arena.free(a, 256);
    arena.free(b, 256);
    CHECK(arena.used() == 0u);
    // Both halves returned and adjacent: must be back to one whole block.
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.largestFreeBlock() == 1024u);
}

// Zero-size and over-capacity requests must fail cleanly.
void testExhaustionAndDegenerateRequests() {
    GpuArenaAllocator arena;
    arena.reset(512);

    CHECK(arena.allocate(0, 1) == kInvalid);
    CHECK(arena.allocate(513, 1) == kInvalid);
    CHECK(arena.used() == 0u);

    const std::uint64_t whole = arena.allocate(512, 1);
    CHECK(whole == 0u);
    CHECK(arena.freeBlockCount() == 0u);
    CHECK(arena.allocate(1, 1) == kInvalid);

    arena.free(whole, 512);
    CHECK(arena.allocate(512, 1) == 0u);
}

void testAlignmentIsHonoured() {
    GpuArenaAllocator arena;
    arena.reset(4096);

    CHECK(arena.allocate(1, 1) == 0u);       // occupies byte 0
    const std::uint64_t aligned = arena.allocate(64, 256);
    CHECK(aligned == 256u);                   // skipped to the next 256 boundary
    CHECK(aligned % 256u == 0u);

    const std::uint64_t aligned2 = arena.allocate(64, 256);
    CHECK(aligned2 % 256u == 0u);
    CHECK(aligned2 != aligned);

    // The gap left by alignment padding must stay usable by a request that does
    // not need the alignment -- otherwise padding leaks permanently.
    const std::uint64_t small = arena.allocate(8, 1);
    CHECK(small != kInvalid);
    CHECK(small >= 1u && small < 256u);
}

// Free three adjacent blocks in an order that requires merging in both
// directions, including the middle-block case that joins two existing runs.
void testCoalescingInEveryDirection() {
    GpuArenaAllocator arena;
    arena.reset(3000);
    const std::uint64_t a = arena.allocate(1000, 1);
    const std::uint64_t b = arena.allocate(1000, 1);
    const std::uint64_t c = arena.allocate(1000, 1);
    CHECK(arena.freeBlockCount() == 0u);

    // Free the outer two: they are not adjacent to each other, so two blocks.
    arena.free(a, 1000);
    arena.free(c, 1000);
    CHECK(arena.freeBlockCount() == 2u);

    // Freeing the middle must merge with BOTH neighbours, collapsing to one.
    arena.free(b, 1000);
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.largestFreeBlock() == 3000u);
    CHECK(arena.used() == 0u);
}

void testCoalescingBackwardOnly() {
    GpuArenaAllocator arena;
    arena.reset(3000);
    const std::uint64_t a = arena.allocate(1000, 1);
    const std::uint64_t b = arena.allocate(1000, 1);
    (void)arena.allocate(1000, 1);  // held, blocks the forward merge

    arena.free(a, 1000);
    arena.free(b, 1000);  // merges backward into a
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.largestFreeBlock() == 2000u);
}

void testCoalescingForwardOnly() {
    GpuArenaAllocator arena;
    arena.reset(3000);
    (void)arena.allocate(1000, 1);  // held, blocks the backward merge
    const std::uint64_t b = arena.allocate(1000, 1);
    const std::uint64_t c = arena.allocate(1000, 1);

    arena.free(c, 1000);
    arena.free(b, 1000);  // merges forward into c
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.largestFreeBlock() == 2000u);
}

// The headline case from the plan: churn the arena the way streaming does and
// assert it neither grows nor fragments without bound.
void testStreamingChurnDoesNotFragment() {
    GpuArenaAllocator arena;
    const std::uint64_t cellBytes = 2u * 1024u * 1024u;  // ~one cell's geometry
    arena.reset(cellBytes * 64u);

    std::vector<std::uint64_t> resident;
    for (int i = 0; i < 64; ++i) {
        const std::uint64_t offset = arena.allocate(cellBytes, 16);
        CHECK(offset != kInvalid);
        resident.push_back(offset);
    }
    CHECK(arena.used() == cellBytes * 64u);
    CHECK(arena.freeBlockCount() == 0u);

    // Evict every other cell, as walking away from a row of cells would.
    for (std::size_t i = 0; i < resident.size(); i += 2) {
        arena.free(resident[i], cellBytes);
    }
    CHECK(arena.freeBlockCount() == 32u);  // holes are genuinely separated

    // Refill them. Every allocation must succeed by reusing a hole.
    for (std::size_t i = 0; i < resident.size(); i += 2) {
        const std::uint64_t offset = arena.allocate(cellBytes, 16);
        CHECK(offset != kInvalid);
        resident[i] = offset;
    }
    CHECK(arena.used() == cellBytes * 64u);
    CHECK(arena.freeBlockCount() == 0u);

    // Now drain the whole arena. If coalescing works this collapses back to a
    // single free block covering everything -- the actual regression guard.
    for (const std::uint64_t offset : resident) {
        arena.free(offset, cellBytes);
    }
    CHECK(arena.used() == 0u);
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.largestFreeBlock() == cellBytes * 64u);
}

// Cells are not all the same size; a varied workload must still drain cleanly.
void testVariableSizedChurnDrainsClean() {
    GpuArenaAllocator arena;
    arena.reset(1u << 20);

    struct Live {
        std::uint64_t offset;
        std::uint64_t size;
    };
    std::vector<Live> live;

    // Deterministic pseudo-random sizes; no RNG dependency.
    std::uint64_t seed = 12345u;
    const auto nextSize = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return 1024u + ((seed >> 33) % 8192u);
    };

    for (int round = 0; round < 200; ++round) {
        // Free roughly half of what is live, oldest first.
        if (live.size() > 8) {
            for (int i = 0; i < 4; ++i) {
                arena.free(live.front().offset, live.front().size);
                live.erase(live.begin());
            }
        }
        for (int i = 0; i < 5; ++i) {
            const std::uint64_t size = nextSize();
            const std::uint64_t offset = arena.allocate(size, 16);
            if (offset != kInvalid) {
                live.push_back({offset, size});
            }
        }
    }

    CHECK(!live.empty());
    for (const Live& entry : live) {
        arena.free(entry.offset, entry.size);
    }
    CHECK(arena.used() == 0u);
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.largestFreeBlock() == (1u << 20));
}

// Growth must keep live allocations exactly where they are -- the backend
// copies the old buffer contents across at the same offsets and does not
// re-upload anything.
void testGrowPreservesLiveAllocations() {
    GpuArenaAllocator arena;
    arena.reset(1000);
    const std::uint64_t a = arena.allocate(400, 1);
    const std::uint64_t b = arena.allocate(400, 1);
    CHECK(a == 0u);
    CHECK(b == 400u);
    CHECK(arena.allocate(400, 1) == kInvalid);  // only 200 left

    arena.grow(2000);
    CHECK(arena.capacity() == 2000u);
    CHECK(arena.used() == 800u);

    // Now it fits, and the live allocations are untouched.
    const std::uint64_t c = arena.allocate(400, 1);
    CHECK(c != kInvalid);
    CHECK(c >= 800u);  // did not overlap a or b

    arena.free(a, 400);
    arena.free(b, 400);
    arena.free(c, 400);
    CHECK(arena.used() == 0u);
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.largestFreeBlock() == 2000u);
}

// Repeated growth must not leave an unusable seam at each old boundary.
void testRepeatedGrowthCoalescesAtTheSeam() {
    GpuArenaAllocator arena;
    arena.reset(100);
    arena.grow(200);
    arena.grow(400);
    arena.grow(800);
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.largestFreeBlock() == 800u);
    // A single allocation spanning every old boundary must succeed.
    CHECK(arena.allocate(800, 1) == 0u);

    // Growing an arena whose tail is fully allocated appends a fresh block.
    arena.grow(1200);
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.allocate(400, 1) == 800u);

    // Shrinking is ignored rather than corrupting the arena.
    arena.grow(10);
    CHECK(arena.capacity() == 1200u);
}

void testResetDropsEverything() {
    GpuArenaAllocator arena;
    arena.reset(1024);
    (void)arena.allocate(512, 1);
    CHECK(arena.used() == 512u);

    arena.reset(2048);
    CHECK(arena.capacity() == 2048u);
    CHECK(arena.used() == 0u);
    CHECK(arena.freeBlockCount() == 1u);
    CHECK(arena.allocate(2048, 1) == 0u);

    arena.reset(0);
    CHECK(arena.freeBlockCount() == 0u);
    CHECK(arena.allocate(1, 1) == kInvalid);
}

}  // namespace

int main() {
    testBasicAllocateAndFree();
    testExhaustionAndDegenerateRequests();
    testAlignmentIsHonoured();
    testCoalescingInEveryDirection();
    testCoalescingBackwardOnly();
    testCoalescingForwardOnly();
    testStreamingChurnDoesNotFragment();
    testVariableSizedChurnDrainsClean();
    testGrowPreservesLiveAllocations();
    testRepeatedGrowthCoalescesAtTheSeam();
    testResetDropsEverything();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "gpu arena allocator tests passed\n";
    return 0;
}
