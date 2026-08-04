#include "engine/game_frame_stats.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        ++g_failures;
        std::cerr << "[engine frame stats test] FAILED: " << message << "\n";
    }
}

using odai::engine::GameFrameProfiler;
using odai::engine::GameZone;
using odai::engine::gameZoneIsNested;
using odai::engine::gameZoneName;
using odai::engine::kGameZoneCount;

// Every zone the loop in GameApp::run() measures needs a name for the overlay,
// and none may collide -- a duplicate would silently mislabel a row.
void testZoneNamesAreUniqueAndPresent() {
    expectTrue(kGameZoneCount == 9u, "the zone set is the nine GameApp loop phases");

    for (std::size_t i = 0; i < kGameZoneCount; ++i) {
        const char* name = gameZoneName(static_cast<GameZone>(i));
        expectTrue(name != nullptr && name[0] != '\0', "every zone has a name");
        expectTrue(std::strcmp(name, "?") != 0, "no zone falls through to the unknown label");
    }

    for (std::size_t i = 0; i < kGameZoneCount; ++i) {
        for (std::size_t j = i + 1; j < kGameZoneCount; ++j) {
            const char* a = gameZoneName(static_cast<GameZone>(i));
            const char* b = gameZoneName(static_cast<GameZone>(j));
            expectTrue(std::strcmp(a, b) != 0, "zone names are unique");
        }
    }
}

// UiBuild and Submit are timed inside Render. Anything summing "where did the
// frame go" has to skip them or Render gets counted up to three times.
void testNestedZonesAreFlagged() {
    expectTrue(gameZoneIsNested(GameZone::UiBuild), "ui build is nested inside render");
    expectTrue(gameZoneIsNested(GameZone::Submit), "submit is nested inside render");
    expectTrue(!gameZoneIsNested(GameZone::Render), "render itself is top level");
    expectTrue(!gameZoneIsNested(GameZone::Tick), "tick is top level");
    expectTrue(!gameZoneIsNested(GameZone::Frame), "the frame total is top level");
}

void testAccumulateThenCommit() {
    GameFrameProfiler prof;
    expectTrue(prof.frameIndex() == 0u, "a fresh profiler has committed no frames");

    prof.beginFrame();
    prof.zoneMs(GameZone::Tick) += 3.0f;
    prof.zoneMs(GameZone::Render) += 5.0f;
    // A zone visited twice in one frame sums rather than overwrites.
    prof.zoneMs(GameZone::Tick) += 1.0f;
    prof.endFrame(10.0f);

    expectTrue(prof.channel(GameZone::Tick).lastMs() == 4.0f, "repeat zone visits accumulate");
    expectTrue(prof.channel(GameZone::Render).lastMs() == 5.0f, "render commits its accumulator");
    expectTrue(prof.channel(GameZone::Frame).lastMs() == 10.0f, "endFrame commits the frame total");
    expectTrue(prof.frameIndex() == 1u, "endFrame advances the frame index");

    // beginFrame must zero the accumulators, or a zone that does no work in a
    // frame would keep reporting the previous frame's cost forever.
    prof.beginFrame();
    prof.endFrame(2.0f);
    expectTrue(prof.channel(GameZone::Tick).lastMs() == 0.0f, "beginFrame zeroes the accumulators");
    expectTrue(prof.channel(GameZone::Frame).lastMs() == 2.0f, "the new frame total commits");
    expectTrue(prof.frameIndex() == 2u, "the frame index keeps advancing");
}

// The overlay's "other" row must equal frame minus the top-level zones, with
// the nested ones excluded so they cannot drive it negative.
void testUnattributedExcludesNestedZones() {
    GameFrameProfiler prof;
    prof.beginFrame();
    prof.zoneMs(GameZone::Poll) += 1.0f;
    prof.zoneMs(GameZone::UiUpdate) += 1.0f;
    prof.zoneMs(GameZone::Tick) += 2.0f;
    prof.zoneMs(GameZone::Render) += 4.0f;
    // Both nested zones live inside the 4 ms Render above.
    prof.zoneMs(GameZone::UiBuild) += 1.5f;
    prof.zoneMs(GameZone::Submit) += 2.0f;
    prof.endFrame(10.0f);

    // 10 - (1 + 1 + 2 + 4) == 2, and the nested 3.5 ms must not be subtracted.
    expectTrue(std::fabs(prof.unattributedMs() - 2.0f) < 1e-4f,
               "unattributed skips nested zones");

    // Over-attribution clamps at zero rather than reporting a negative row.
    GameFrameProfiler tight;
    tight.beginFrame();
    tight.zoneMs(GameZone::Tick) += 9.0f;
    tight.endFrame(4.0f);
    expectTrue(tight.unattributedMs() == 0.0f, "unattributed never goes negative");
}

void testFpsDerivesFromTheFrameChannel() {
    GameFrameProfiler prof;
    expectTrue(prof.fps() == 0.0f, "fps is 0 before any frame is committed");

    // A steady 10 ms frame is 100 fps; the first sample seeds the EWMA so this
    // is exact after one frame rather than ramping.
    prof.beginFrame();
    prof.endFrame(10.0f);
    expectTrue(std::fabs(prof.fps() - 100.0f) < 1e-3f, "fps derives from the frame EWMA");

    // A zero-length frame must not divide by zero.
    GameFrameProfiler zero;
    zero.beginFrame();
    zero.endFrame(0.0f);
    expectTrue(zero.fps() == 0.0f, "a zero-length frame reports 0 fps, not infinity");
}

}  // namespace

int main() {
    testZoneNamesAreUniqueAndPresent();
    testNestedZonesAreFlagged();
    testAccumulateThenCommit();
    testUnattributedExcludesNestedZones();
    testFpsDerivesFromTheFrameChannel();

    if (g_failures != 0) {
        std::cerr << "[engine frame stats test] " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[engine frame stats test] all checks passed\n";
    return 0;
}
