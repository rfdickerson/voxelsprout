#pragma once
#include "engine/tick_timing.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ostream>

namespace odai::engine {

struct FrameTimingSample {
    std::uint64_t frame = 0;
    double startSeconds = 0;
    float cpuMs = 0, tickMs = 0, renderMs = 0;
    float slotWaitMs = 0, acquireWaitMs = 0, presentWaitMs = 0, transferWaitMs = 0;
    bool renderAttempted = false, presentAccepted = false;
    std::uint64_t submissionId = 0, gpuSubmissionId = 0;
    float gpuMs = 0;
    std::uint32_t queuedFrames = 0, drawCalls = 0;
    std::uint64_t triangles = 0;
    TickTiming tick{};
};

// Delay one row until the next frame starts: its interval includes all work
// and loop overhead belonging to that row, including CSV output overhead.
// GPU values remain readback events tagged with the originating submission.
class FrameTimingCsv {
public:
    FrameTimingCsv(std::ostream& stream, bool benchmark)
        : m_stream(stream), m_benchmark(benchmark) {
        m_stream << "frame,interval_ms,cpu_ms,tick_ms,render_ms,schema_version";
        if (benchmark) m_stream << ",gpu_ms,draw_calls,triangles,gpu_submission_id,submission_id"
            ",render_attempted,present_accepted,queued_frames,wait_frame_slot_ms"
            ",wait_acquire_ms,wait_present_ms,wait_transfer_ms,cpu_work_ms,render_work_ms"
            ",tick_session_ms,tick_actors_ms,tick_streaming_ms,streamer_update_ms"
            ",streamer_apply_ms,streamer_upload_ms,streamer_callbacks_ms"
            ",streamer_eviction_ms,streamer_renderer_remove_ms,streamer_eviction_callbacks_ms"
            ",eviction_collision_ms,eviction_navigation_ms,eviction_doors_ms"
            ",cell_registration_ms,runtime_objects_ms,gameplay_cell_ms"
            ",gameplay_io_ms,gameplay_compile_ms,gameplay_publish_ms"
            ",gameplay_anchor_ms,gameplay_upsert_ms"
            ",physics_install_ms,collision_world_ms,navigation_ms,broad_phase_ms"
            ",resident_cells";
        m_stream << '\n';
    }

    void beginFrame(double startSeconds) {
        if (m_pending) write((startSeconds - m_pending->startSeconds) * 1000.0);
    }
    void endFrame(const FrameTimingSample& sample) { m_pending = sample; }
    // There is no next frame start at shutdown. Do not invent an interval.
    void finish() { if (m_pending) write(std::nullopt); }

private:
    void write(std::optional<double> intervalMs) {
        const auto& s = *m_pending;
        m_stream << s.frame << ',';
        if (intervalMs) m_stream << *intervalMs;
        m_stream << ',' << s.cpuMs << ',' << s.tickMs << ',' << s.renderMs << ",3";
        if (m_benchmark) {
            m_stream << ',';
            if (s.gpuSubmissionId) m_stream << s.gpuMs;
            m_stream << ',' << s.drawCalls << ',' << s.triangles << ',';
            if (s.gpuSubmissionId) m_stream << s.gpuSubmissionId;
            const float waits = s.slotWaitMs + s.acquireWaitMs + s.presentWaitMs + s.transferWaitMs;
            m_stream << ',' << s.submissionId << ',' << s.renderAttempted << ',' << s.presentAccepted
                << ',' << s.queuedFrames << ',' << s.slotWaitMs << ',' << s.acquireWaitMs
                << ',' << s.presentWaitMs << ',' << s.transferWaitMs
                << ',' << std::max(0.0f, s.cpuMs - waits)
                << ',' << std::max(0.0f, s.renderMs - waits)
                << ',' << s.tick.sessionMs << ',' << s.tick.actorsMs
                << ',' << s.tick.streamingMs << ',' << s.tick.streamerUpdateMs
                << ',' << s.tick.streamerApplyMs << ',' << s.tick.streamerUploadMs
                << ',' << s.tick.streamerCallbacksMs << ',' << s.tick.streamerEvictionMs
                << ',' << s.tick.streamerRendererRemoveMs
                << ',' << s.tick.streamerEvictionCallbacksMs
                << ',' << s.tick.evictionCollisionMs << ',' << s.tick.evictionNavigationMs
                << ',' << s.tick.evictionDoorsMs
                << ',' << s.tick.cellRegistrationMs << ',' << s.tick.runtimeObjectsMs
                << ',' << s.tick.gameplayCellMs << ',' << s.tick.gameplayIoMs
                << ',' << s.tick.gameplayCompileMs << ',' << s.tick.gameplayPublishMs
                << ',' << s.tick.gameplayAnchorMs << ',' << s.tick.gameplayUpsertMs
                << ',' << s.tick.physicsInstallMs << ',' << s.tick.collisionWorldMs
                << ',' << s.tick.navigationMs << ',' << s.tick.broadPhaseMs
                << ',' << s.tick.residentCells;
        }
        m_stream << '\n';
        m_pending.reset();
    }
    std::ostream& m_stream;
    bool m_benchmark;
    std::optional<FrameTimingSample> m_pending;
};

} // namespace odai::engine
