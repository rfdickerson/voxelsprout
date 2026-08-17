// Tests for the cell streaming residency policy.
//
// Every case here targets a failure that is invisible at runtime: a boundary
// that silently thrashes, prediction pointed the wrong way, an in-flight set
// that grows without bound, or a worldspace edge re-requested forever. None of
// these produce an error -- they produce stutter, or a world that fills in
// behind you.

#include "import/cell_residency_planner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using odai::importer::CellCoord;
using odai::importer::CellResidencyConfig;
using odai::importer::CellResidencyPlanner;

namespace {

int g_failures = 0;

// Variadic so braced initializers containing commas -- CellCoord{-1, -1} -- do
// not get split into two macro arguments.
#define CHECK(...)                                                                  \
    do {                                                                            \
        if (!(__VA_ARGS__)) {                                                       \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": "             \
                      << #__VA_ARGS__ << "\n";                                      \
            ++g_failures;                                                           \
        }                                                                           \
    } while (false)

constexpr float kCellSize = 4096.0f;

// World position at the centre of a cell. Fallout is Z-up, so the grid's second
// axis is world Y (index 1), not index 2.
std::vector<float> atCell(std::int32_t x, std::int32_t z) {
    return {(static_cast<float>(x) + 0.5f) * kCellSize,
            (static_cast<float>(z) + 0.5f) * kCellSize,
            0.0f};
}

const std::vector<float> kStill{0.0f, 0.0f, 0.0f};

CellResidencyConfig testConfig() {
    CellResidencyConfig config;
    config.loadRadius = 2;
    config.unloadRadius = 4;
    config.cellSize = kCellSize;
    config.leadTimeSeconds = 2.0f;
    config.maxLeadCells = 3.0f;
    config.maxLoadsInFlight = 64;  // effectively unlimited unless a test says otherwise
    return config;
}

// Drives the planner to a steady state: keep loading whatever it asks for until
// it stops asking. Returns the number of update rounds taken.
int settle(CellResidencyPlanner& planner, const std::vector<float>& position,
           const std::vector<float>& velocity, int maxRounds = 200) {
    for (int round = 0; round < maxRounds; ++round) {
        planner.update(position.data(), velocity.data());
        for (const CellCoord& cell : planner.cellsToEvict()) {
            planner.markEvicted(cell);
        }
        const std::vector<CellCoord> toLoad = planner.cellsToLoad();
        if (toLoad.empty()) {
            return round;
        }
        for (const CellCoord& cell : toLoad) {
            planner.markLoadStarted(cell);
            planner.markLoadFinished(cell);
        }
    }
    return maxRounds;
}

void testLoadsExactlyTheRadiusAtRest() {
    CellResidencyPlanner planner;
    planner.setConfig(testConfig());

    const auto position = atCell(0, 0);
    settle(planner, position, kStill);

    // Radius 2 Chebyshev is a 5x5 block.
    CHECK(planner.stats().residentCount == 25u);
    for (std::int32_t z = -2; z <= 2; ++z) {
        for (std::int32_t x = -2; x <= 2; ++x) {
            CHECK(planner.isResident(CellCoord{x, z}));
        }
    }
    // Nothing outside it.
    CHECK(!planner.isResident(CellCoord{3, 0}));
    CHECK(!planner.isResident(CellCoord{0, -3}));
    CHECK(planner.cellsToLoad().empty());
    CHECK(planner.cellsToEvict().empty());
}

// The whole reason leadTime exists: cells ahead must be requested before cells
// behind, or the world fills in where the player has already been.
void testPredictionPrioritisesTheDirectionOfTravel() {
    CellResidencyPlanner planner;
    CellResidencyConfig config = testConfig();
    config.maxLoadsInFlight = 4;  // small budget makes the ordering observable
    planner.setConfig(config);

    const auto position = atCell(0, 0);
    // Moving +X fast enough that 2 seconds carries well past one cell.
    const std::vector<float> eastward{kCellSize, 0.0f, 0.0f};
    planner.update(position.data(), eastward.data());

    const std::vector<CellCoord>& first = planner.cellsToLoad();
    CHECK(first.size() == 4u);
    // Every cell in the first batch must be ahead of, or level with, the player
    // -- never behind.
    for (const CellCoord& cell : first) {
        CHECK(cell.x >= 0);
    }
    // And the very first must be strictly ahead.
    CHECK(!first.empty() && first[0].x > 0);

    // The mirror case must behave symmetrically, which catches a sign error that
    // a single-direction test would not.
    CellResidencyPlanner westPlanner;
    westPlanner.setConfig(config);
    const std::vector<float> westward{-kCellSize, 0.0f, 0.0f};
    westPlanner.update(position.data(), westward.data());
    for (const CellCoord& cell : westPlanner.cellsToLoad()) {
        CHECK(cell.x <= 0);
    }
    CHECK(!westPlanner.cellsToLoad().empty() && westPlanner.cellsToLoad()[0].x < 0);

    // Fallout is Z-up: travelling along world Y must move along the grid's z
    // axis. If the planner read world index 2 instead, this would rank as if
    // standing still.
    CellResidencyPlanner northPlanner;
    northPlanner.setConfig(config);
    const std::vector<float> northward{0.0f, kCellSize, 0.0f};
    northPlanner.update(position.data(), northward.data());
    CHECK(!northPlanner.cellsToLoad().empty() && northPlanner.cellsToLoad()[0].z > 0);
}

// The headline invariant. Stepping back and forth across a cell boundary must
// not load and evict the same cell over and over.
void testHysteresisPreventsBoundaryThrash() {
    CellResidencyPlanner planner;
    planner.setConfig(testConfig());

    // Settle at the origin cell.
    settle(planner, atCell(0, 0), kStill);
    const std::uint64_t loadsAfterSettle = planner.stats().loadsStarted;
    const std::uint64_t evictionsAfterSettle = planner.stats().evictions;

    // Now oscillate across the boundary between cell 0 and cell 1, twenty times.
    // With loadRadius 2 and unloadRadius 4 the two-cell band absorbs this
    // entirely: cells entering the load radius stay well inside the unload
    // radius when the player steps back.
    for (int i = 0; i < 20; ++i) {
        settle(planner, atCell(1, 0), kStill);
        settle(planner, atCell(0, 0), kStill);
    }

    const std::uint64_t newLoads = planner.stats().loadsStarted - loadsAfterSettle;
    const std::uint64_t newEvictions = planner.stats().evictions - evictionsAfterSettle;

    // Moving one cell east legitimately brings a new column into range, and
    // stepping back brings another. What must NOT happen is eviction, because
    // nothing ever leaves the 4-cell unload radius.
    CHECK(newEvictions == 0u);
    // And the loads must converge: after the first excursion every cell either
    // side is already resident, so 20 further round trips add nothing.
    const std::uint64_t loadsAfterFirstTrip = newLoads;
    for (int i = 0; i < 20; ++i) {
        settle(planner, atCell(1, 0), kStill);
        settle(planner, atCell(0, 0), kStill);
    }
    CHECK(planner.stats().loadsStarted - loadsAfterSettle == loadsAfterFirstTrip);
    CHECK(planner.stats().evictions == evictionsAfterSettle);
}

// Without the hysteresis band, the same walk thrashes. This proves the band is
// doing the work rather than the test being trivially satisfiable.
void testWithoutHysteresisTheSameWalkThrashes() {
    CellResidencyPlanner planner;
    CellResidencyConfig config = testConfig();
    config.unloadRadius = config.loadRadius;  // no band at all
    planner.setConfig(config);

    settle(planner, atCell(0, 0), kStill);
    const std::uint64_t evictionsAfterSettle = planner.stats().evictions;

    for (int i = 0; i < 5; ++i) {
        settle(planner, atCell(1, 0), kStill);
        settle(planner, atCell(0, 0), kStill);
    }
    // Every step now pushes a whole column outside the unload radius.
    CHECK(planner.stats().evictions > evictionsAfterSettle);
}

void testInFlightBudgetIsRespected() {
    CellResidencyPlanner planner;
    CellResidencyConfig config = testConfig();
    config.maxLoadsInFlight = 3;
    planner.setConfig(config);

    const auto position = atCell(0, 0);
    planner.update(position.data(), kStill.data());
    CHECK(planner.cellsToLoad().size() == 3u);

    // Start them but do not finish: the next update must propose nothing.
    for (const CellCoord& cell : planner.cellsToLoad()) {
        planner.markLoadStarted(cell);
    }
    planner.update(position.data(), kStill.data());
    CHECK(planner.cellsToLoad().empty());
    CHECK(planner.stats().loadingCount == 3u);

    // Finishing one frees exactly one slot.
    const CellCoord finished = CellCoord{planner.cellsToEvict().empty() ? 0 : 0, 0};
    (void)finished;
    CellCoord firstLoading{};
    for (std::int32_t z = -2; z <= 2 && firstLoading.x == 0 && firstLoading.z == 0; ++z) {
        for (std::int32_t x = -2; x <= 2; ++x) {
            const CellCoord candidate{x, z};
            if (planner.isTracked(candidate) && !planner.isResident(candidate)) {
                firstLoading = candidate;
                break;
            }
        }
    }
    CHECK(planner.markLoadFinished(firstLoading));
    planner.update(position.data(), kStill.data());
    CHECK(planner.cellsToLoad().size() == 1u);
}

// A load that completes after the player has left must be reported as wasted,
// not silently uploaded.
void testStaleLoadIsRejectedAndCounted() {
    CellResidencyPlanner planner;
    planner.setConfig(testConfig());

    const auto origin = atCell(0, 0);
    planner.update(origin.data(), kStill.data());
    const CellCoord inFlight = planner.cellsToLoad().front();
    planner.markLoadStarted(inFlight);

    // Teleport far away, well outside the unload radius, then let the load land.
    const auto faraway = atCell(100, 100);
    planner.update(faraway.data(), kStill.data());
    CHECK(!planner.markLoadFinished(inFlight));
    CHECK(planner.stats().wastedLoads == 1u);
    CHECK(!planner.isResident(inFlight));
    // And it is no longer tracked, so it can be requested again if the player
    // returns.
    CHECK(!planner.isTracked(inFlight));
}

// A load that finishes while the player is still nearby must be accepted, so the
// staleness check is not simply rejecting everything.
void testTimelyLoadIsAccepted() {
    CellResidencyPlanner planner;
    planner.setConfig(testConfig());

    const auto origin = atCell(0, 0);
    planner.update(origin.data(), kStill.data());
    const CellCoord inFlight = planner.cellsToLoad().front();
    planner.markLoadStarted(inFlight);

    planner.update(origin.data(), kStill.data());
    CHECK(planner.markLoadFinished(inFlight));
    CHECK(planner.isResident(inFlight));
    CHECK(planner.stats().wastedLoads == 0u);
    CHECK(planner.stats().loadsCompleted == 1u);
}

// A capture pins its complete route before it starts. Future cells must not be
// rejected as stale or evicted simply because the camera has not reached them.
void testPinnedCellsSurviveTourTravel() {
    CellResidencyPlanner planner;
    planner.setConfig(testConfig());
    const CellCoord future{40, -12};
    planner.setPinnedCells({future});

    const auto origin = atCell(0, 0);
    planner.update(origin.data(), kStill.data());
    CHECK(!planner.cellsToLoad().empty());
    CHECK(planner.cellsToLoad().front() == future);
    planner.markLoadStarted(future);

    // The camera can be far away before the worker completes. A normal cell
    // would be reported as wasted here; the pinned route cell must be kept.
    const auto faraway = atCell(-30, 30);
    planner.update(faraway.data(), kStill.data());
    CHECK(planner.markLoadFinished(future));
    CHECK(planner.isResident(future));

    planner.update(faraway.data(), kStill.data());
    CHECK(std::find(planner.cellsToEvict().begin(), planner.cellsToEvict().end(), future) ==
          planner.cellsToEvict().end());
    planner.reset();
    CHECK(!planner.isTracked(future));
}

// The worldspace is not a rectangle. Cells that do not exist must be asked for
// once, not every frame forever.
void testUnavailableCellsAreNeverReRequested() {
    CellResidencyPlanner planner;
    planner.setConfig(testConfig());

    const auto position = atCell(0, 0);
    planner.update(position.data(), kStill.data());
    const CellCoord missing = planner.cellsToLoad().front();
    planner.markLoadStarted(missing);
    planner.markUnavailable(missing);

    for (int round = 0; round < 50; ++round) {
        planner.update(position.data(), kStill.data());
        for (const CellCoord& cell : planner.cellsToLoad()) {
            CHECK(!(cell == missing));
            planner.markLoadStarted(cell);
            planner.markLoadFinished(cell);
        }
        if (planner.cellsToLoad().empty()) {
            break;
        }
    }
    CHECK(planner.stats().unavailableCells == 1u);
    CHECK(!planner.isResident(missing));
    // 25 cells in the block, one of which does not exist.
    CHECK(planner.stats().residentCount == 24u);
}

// Walking a long straight line: the resident set must stay bounded and the
// planner must not accumulate wasted work.
void testLongWalkKeepsResidencyBoundedAndWasteAtZero() {
    CellResidencyPlanner planner;
    CellResidencyConfig config = testConfig();
    config.maxLoadsInFlight = 8;
    planner.setConfig(config);

    const std::vector<float> eastward{kCellSize * 0.5f, 0.0f, 0.0f};
    std::size_t peakResident = 0;
    for (std::int32_t step = 0; step < 60; ++step) {
        const auto position = atCell(step, 0);
        // A few updates per step, as frames would do.
        for (int frame = 0; frame < 12; ++frame) {
            planner.update(position.data(), eastward.data());
            for (const CellCoord& cell : planner.cellsToEvict()) {
                planner.markEvicted(cell);
            }
            for (const CellCoord& cell : planner.cellsToLoad()) {
                planner.markLoadStarted(cell);
                planner.markLoadFinished(cell);
            }
        }
        peakResident = std::max(peakResident, planner.stats().residentCount);
    }

    // Bounded by the unload radius: a 9x9 block for unloadRadius 4.
    CHECK(peakResident <= 81u);
    CHECK(planner.stats().residentCount <= 81u);
    // A straight-line walk with immediate completion should waste nothing.
    CHECK(planner.stats().wastedLoads == 0u);
    // And it must actually have streamed: 60 cells of travel moves a lot of ring.
    CHECK(planner.stats().evictions > 0u);
    CHECK(planner.stats().loadsCompleted > 100u);
}

void testEvictionIsFarthestFirst() {
    CellResidencyPlanner planner;
    planner.setConfig(testConfig());

    settle(planner, atCell(0, 0), kStill);
    // Jump far enough that the whole resident set falls outside the unload
    // radius at differing distances.
    const auto position = atCell(9, 0);
    planner.update(position.data(), kStill.data());

    const std::vector<CellCoord>& evictions = planner.cellsToEvict();
    CHECK(!evictions.empty());
    for (std::size_t i = 1; i < evictions.size(); ++i) {
        const std::int32_t previous =
            std::max(std::abs(evictions[i - 1].x - 9), std::abs(evictions[i - 1].z));
        const std::int32_t current =
            std::max(std::abs(evictions[i].x - 9), std::abs(evictions[i].z));
        CHECK(previous >= current);
    }
}

void testResetForgetsEverything() {
    CellResidencyPlanner planner;
    planner.setConfig(testConfig());
    settle(planner, atCell(0, 0), kStill);
    CHECK(planner.stats().residentCount > 0u);

    planner.reset();
    CHECK(planner.stats().residentCount == 0u);
    CHECK(planner.stats().loadsStarted == 0u);
    CHECK(!planner.isTracked(CellCoord{0, 0}));
}

void testCellAtHandlesNegativeCoordinates() {
    CellResidencyPlanner planner;
    planner.setConfig(testConfig());

    // Floor, not truncation: -1.0 world units is in cell -1, not cell 0. Getting
    // this wrong makes the origin cell twice as wide as every other.
    CHECK(planner.cellAt(-1.0f, -1.0f) == CellCoord{-1, -1});
    CHECK(planner.cellAt(0.0f, 0.0f) == CellCoord{0, 0});
    CHECK(planner.cellAt(kCellSize - 1.0f, 0.0f) == CellCoord{0, 0});
    CHECK(planner.cellAt(kCellSize, 0.0f) == CellCoord{1, 0});
    CHECK(planner.cellAt(-kCellSize, 0.0f) == CellCoord{-1, 0});
    CHECK(planner.cellAt(-kCellSize - 1.0f, 0.0f) == CellCoord{-2, 0});
}

// A wild velocity spike must not aim the budget at cells the player will never
// reach.
void testLeadDistanceIsClamped() {
    CellResidencyPlanner planner;
    CellResidencyConfig config = testConfig();
    config.maxLoadsInFlight = 4;
    config.maxLeadCells = 1.0f;
    planner.setConfig(config);

    const auto position = atCell(0, 0);
    // 1000 cells per second: without clamping the aim point lands far outside
    // the load radius and the ranking becomes meaningless.
    const std::vector<float> absurd{kCellSize * 1000.0f, 0.0f, 0.0f};
    planner.update(position.data(), absurd.data());

    CHECK(!planner.cellsToLoad().empty());
    // Still proposing cells inside the load radius, ranked sensibly.
    for (const CellCoord& cell : planner.cellsToLoad()) {
        CHECK(std::abs(cell.x) <= config.loadRadius);
        CHECK(std::abs(cell.z) <= config.loadRadius);
    }
}

}  // namespace

int main() {
    testLoadsExactlyTheRadiusAtRest();
    testPredictionPrioritisesTheDirectionOfTravel();
    testHysteresisPreventsBoundaryThrash();
    testWithoutHysteresisTheSameWalkThrashes();
    testInFlightBudgetIsRespected();
    testStaleLoadIsRejectedAndCounted();
    testTimelyLoadIsAccepted();
    testPinnedCellsSurviveTourTravel();
    testUnavailableCellsAreNeverReRequested();
    testLongWalkKeepsResidencyBoundedAndWasteAtZero();
    testEvictionIsFarthestFirst();
    testResetForgetsEverything();
    testCellAtHandlesNegativeCoordinates();
    testLeadDistanceIsClamped();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "cell residency planner tests passed\n";
    return 0;
}
