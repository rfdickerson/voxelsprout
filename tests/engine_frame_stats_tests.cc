#include "engine/game_frame_stats.h"
#include "engine/frame_timing_csv.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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
    expectTrue(!prof.fpsReady(), "headline fps waits for a representative window");

    // A steady 10 ms frame is 100 fps.
    prof.beginFrame();
    prof.endFrame(10.0f);
    expectTrue(std::fabs(prof.fps() - 100.0f) < 1e-3f, "fps derives from frame p50");

    // A startup upload hitch must remain in max/p99 without poisoning the
    // typical headline rate after smooth presentation begins.
    GameFrameProfiler startup;
    startup.beginFrame();
    startup.endFrame(250.0f);
    for (std::size_t i = 1; i < GameFrameProfiler::kDisplayWarmupSamples; ++i) {
        startup.beginFrame();
        startup.endFrame(10.0f);
    }
    expectTrue(startup.fpsReady(), "headline fps becomes ready after warm-up");
    expectTrue(std::fabs(startup.fps() - 100.0f) < 1e-3f,
               "startup hitch does not masquerade as sustained fps");
    expectTrue(startup.channel(GameZone::Frame).maxMs() == 250.0f,
               "startup hitch remains visible in the diagnostic window");

    // A zero-length frame must not divide by zero.
    GameFrameProfiler zero;
    zero.beginFrame();
    zero.endFrame(0.0f);
    expectTrue(zero.fps() == 0.0f, "a zero-length frame reports 0 fps, not infinity");
}

void testCsvAttribution() {
    using namespace odai::engine;
    std::ostringstream output;
    FrameTimingCsv csv(output, true);
    csv.beginFrame(1.0);
    FrameTimingSample tickStall;
    tickStall.frame = 239;
    tickStall.startSeconds = 1.0;
    tickStall.cpuMs = 27;
    tickStall.tickMs = 25;
    tickStall.renderMs = 2;
    tickStall.tick.streamingMs = 25;
    tickStall.tick.streamerUploadMs = 10;
    tickStall.tick.gameplayAnchorMs = 5;
    tickStall.tick.streamerRendererRemoveMs = 3;
    tickStall.tick.evictionCollisionMs = 2;
    tickStall.renderAttempted = tickStall.presentAccepted = true;
    tickStall.submissionId = 41;
    csv.endFrame(tickStall);
    csv.beginFrame(1.030); // includes 3 ms of work outside CPU timer
    FrameTimingSample waitStall;
    waitStall.frame = 240;
    waitStall.startSeconds = 1.030;
    waitStall.cpuMs = 21;
    waitStall.tickMs = 1;
    waitStall.renderMs = 20;
    waitStall.acquireWaitMs = 18;
    waitStall.renderAttempted = true; // timed-out acquire: no submission
    waitStall.gpuSubmissionId = 41;
    waitStall.gpuMs = 4;
    csv.endFrame(waitStall);
    csv.beginFrame(1.052);
    FrameTimingSample last;
    last.frame = 241;
    last.startSeconds = 1.052;
    last.cpuMs = 2;
    csv.endFrame(last);
    csv.finish();
    csv.finish(); // no duplicate terminal record

    std::istringstream lines(output.str());
    std::string line;
    std::vector<std::string> columns;
    std::getline(lines, line);
    std::istringstream header(line);
    std::string cell;
    while (std::getline(header, cell, ',')) columns.push_back(cell);
    std::vector<std::unordered_map<std::string, std::string>> rows;
    while (std::getline(lines, line)) {
        std::istringstream cells(line);
        auto& row = rows.emplace_back();
        for (const auto& column : columns) {
            std::getline(cells, cell, ',');
            row[column] = cell;
        }
    }
    expectTrue(rows.size() == 3, "one CSV row per completed frame including terminal sample");
    if (rows.size() != 3) return;
    expectTrue(rows[0]["frame"] == "239" && rows[0]["interval_ms"] == "30" &&
               rows[0]["tick_ms"] == "25", "tick stall interval belongs to originating frame");
    expectTrue(rows[0]["tick_streaming_ms"] == "25" &&
               rows[0]["streamer_upload_ms"] == "10" &&
               rows[0]["gameplay_anchor_ms"] == "5" &&
               rows[0]["streamer_renderer_remove_ms"] == "3" &&
               rows[0]["eviction_collision_ms"] == "2",
               "nested game-update stages stay on the same frame");
    expectTrue(rows[1]["interval_ms"] == "22" && rows[1]["wait_acquire_ms"] == "18",
               "wait stall and interval belong to the same frame");
    expectTrue(rows[1]["cpu_work_ms"] == "3" && rows[1]["render_work_ms"] == "2",
               "known waits are excluded from work times");
    expectTrue(rows[1]["submission_id"] == "0" && rows[1]["present_accepted"] == "0",
               "skipped submission cannot inherit previous frame outcome");
    expectTrue(rows[1]["gpu_submission_id"] == "41" && rows[1]["gpu_ms"] == "4",
               "delayed GPU sample retains originating submission ID");
    expectTrue(rows[2]["interval_ms"].empty() && rows[2]["gpu_ms"].empty(),
               "terminal interval and unavailable GPU timing remain missing");
    expectTrue(rows[0]["schema_version"] == "3", "CSV attribution is versioned");
}

}  // namespace

int main() {
    testZoneNamesAreUniqueAndPresent();
    testNestedZonesAreFlagged();
    testAccumulateThenCommit();
    testUnattributedExcludesNestedZones();
    testFpsDerivesFromTheFrameChannel();
    testCsvAttribution();

    if (g_failures != 0) {
        std::cerr << "[engine frame stats test] " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[engine frame stats test] all checks passed\n";
    return 0;
}
