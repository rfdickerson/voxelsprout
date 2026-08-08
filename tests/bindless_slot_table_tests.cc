// Tests for the refcounted bindless slot bookkeeping that imported-texture
// streaming rests on. Everything here is about invariants that would otherwise
// fail silently: a shared texture freed too early, a recycled slot handed out
// while still referenced, or a failed upload leaving an addressable slot behind.

#include "render/bindless_slot_table.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using odai::render::BindlessSlotTable;
using odai::render::kInvalidSlotIndex;

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

// The reason the table exists: one texture named by many cells uploads once and
// survives until the last of them lets go.
void testSharedKeyUploadsOnce() {
    BindlessSlotTable table;
    table.setCapacity(64);

    const auto first = table.acquire("textures/landscape/rock01.dds");
    CHECK(first.slotIndex == 0u);
    CHECK(first.needsUpload);

    // Forty more cells place the same rock.
    for (int i = 0; i < 40; ++i) {
        const auto again = table.acquire("textures/landscape/rock01.dds");
        CHECK(again.slotIndex == first.slotIndex);
        CHECK(!again.needsUpload);  // no second upload, ever
    }
    CHECK(table.refCount(first.slotIndex) == 41u);
    CHECK(table.slotCount() == 1u);
    CHECK(table.residentCount() == 1u);

    // Releasing 40 of the 41 must NOT free the image.
    for (int i = 0; i < 40; ++i) {
        CHECK(!table.release(first.slotIndex));
    }
    CHECK(table.refCount(first.slotIndex) == 1u);
    CHECK(table.residentCount() == 1u);

    // The last one does.
    CHECK(table.release(first.slotIndex));
    CHECK(table.refCount(first.slotIndex) == 0u);
    CHECK(table.residentCount() == 0u);
    CHECK(table.freeCount() == 1u);
}

// Slots must be recycled rather than the table growing without bound -- this is
// what keeps a long walk across the Mojave from exhausting the descriptor array.
void testSlotsAreRecycledNotGrown() {
    BindlessSlotTable table;
    table.setCapacity(64);

    std::vector<std::uint32_t> slots;
    for (int i = 0; i < 50; ++i) {
        const auto acquired = table.acquire("tex" + std::to_string(i));
        CHECK(acquired.needsUpload);
        slots.push_back(acquired.slotIndex);
    }
    CHECK(table.slotCount() == 50u);
    CHECK(table.residentCount() == 50u);

    // Evict every other one, as streaming would when half the cells unload.
    for (std::size_t i = 0; i < slots.size(); i += 2) {
        CHECK(table.release(slots[i]));
    }
    CHECK(table.residentCount() == 25u);
    CHECK(table.freeCount() == 25u);

    // The next 25 acquires must reuse those holes, not extend the table.
    for (int i = 0; i < 25; ++i) {
        const auto acquired = table.acquire("fresh" + std::to_string(i));
        CHECK(acquired.needsUpload);
        CHECK(acquired.slotIndex != kInvalidSlotIndex);
    }
    CHECK(table.slotCount() == 50u);  // high-water mark unchanged
    CHECK(table.freeCount() == 0u);
    CHECK(table.residentCount() == 50u);
}

// A recycled slot must not still answer to the key that used to live there.
void testRecycledSlotDropsOldKey() {
    BindlessSlotTable table;
    table.setCapacity(4);

    const auto original = table.acquire("old.dds");
    CHECK(table.release(original.slotIndex));

    const auto reused = table.acquire("new.dds");
    CHECK(reused.slotIndex == original.slotIndex);  // same slot recycled
    CHECK(reused.needsUpload);

    // Asking for the evicted key again must be a genuine miss, not a hit on the
    // slot its name used to occupy.
    const auto stale = table.acquire("old.dds");
    CHECK(stale.slotIndex != reused.slotIndex);
    CHECK(stale.needsUpload);
}

// Capacity is the descriptor array bound; overshooting must fail cleanly rather
// than hand back a slot the shader cannot address.
void testCapacityIsRespected() {
    BindlessSlotTable table;
    table.setCapacity(3);

    CHECK(table.acquire("a").slotIndex == 0u);
    CHECK(table.acquire("b").slotIndex == 1u);
    CHECK(table.acquire("c").slotIndex == 2u);

    const auto overflow = table.acquire("d");
    CHECK(overflow.slotIndex == kInvalidSlotIndex);
    CHECK(!overflow.needsUpload);
    CHECK(table.slotCount() == 3u);

    // An already-resident key still resolves even at capacity.
    const auto resident = table.acquire("b");
    CHECK(resident.slotIndex == 1u);
    CHECK(!resident.needsUpload);

    // Freeing one makes room again.
    CHECK(table.release(0u));
    CHECK(table.acquire("d").slotIndex == 0u);
}

// A failed image upload must leave no trace: the slot goes back on the free
// list and the key does not resolve.
void testAbandonRollsBackAFailedUpload() {
    BindlessSlotTable table;
    table.setCapacity(8);

    const auto good = table.acquire("good.dds");
    const auto doomed = table.acquire("doomed.dds");
    CHECK(doomed.needsUpload);
    CHECK(table.residentCount() == 2u);

    table.abandon(doomed.slotIndex);
    CHECK(table.residentCount() == 1u);
    CHECK(table.refCount(doomed.slotIndex) == 0u);

    // The abandoned key must be a clean miss next time, and its slot reusable.
    const auto retry = table.acquire("doomed.dds");
    CHECK(retry.needsUpload);
    CHECK(retry.slotIndex == doomed.slotIndex);

    // The unrelated neighbour is untouched.
    CHECK(table.refCount(good.slotIndex) == 1u);
}

// Releasing something already free, or out of range, must be a no-op rather
// than corrupting the free list with duplicates.
void testOverReleaseIsIgnored() {
    BindlessSlotTable table;
    table.setCapacity(8);

    const auto acquired = table.acquire("x.dds");
    CHECK(table.release(acquired.slotIndex));
    CHECK(table.freeCount() == 1u);

    // Second release of the same slot: must not push a duplicate onto the free
    // list, which would later hand the same slot to two different textures.
    CHECK(!table.release(acquired.slotIndex));
    CHECK(table.freeCount() == 1u);

    CHECK(!table.release(9999u));
    CHECK(table.freeCount() == 1u);
}

// Keyless textures are never shared, even with each other.
void testEmptyKeyIsNeverDeduplicated() {
    BindlessSlotTable table;
    table.setCapacity(8);

    const auto a = table.acquire("");
    const auto b = table.acquire("");
    CHECK(a.slotIndex != b.slotIndex);
    CHECK(a.needsUpload);
    CHECK(b.needsUpload);
    CHECK(table.residentCount() == 0u);  // keyless entries are not in the key map
    CHECK(table.slotCount() == 2u);

    // They are still individually releasable by index.
    CHECK(table.release(a.slotIndex));
    CHECK(table.release(b.slotIndex));
    CHECK(table.freeCount() == 2u);
}

}  // namespace

int main() {
    testSharedKeyUploadsOnce();
    testSlotsAreRecycledNotGrown();
    testRecycledSlotDropsOldKey();
    testCapacityIsRespected();
    testAbandonRollsBackAFailedUpload();
    testOverReleaseIsIgnored();
    testEmptyKeyIsNeverDeduplicated();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "bindless slot table tests passed\n";
    return 0;
}
