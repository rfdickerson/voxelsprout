#pragma once

#include "core/frame_profiler.h"

#include <array>
#include <cstddef>
#include <cstdint>

// Engine GameFrameStats subsystem
// Responsible for: the fixed set of CPU timing zones every GameApp-based game
// reports, and the per-frame accumulate-then-commit bookkeeping around them.
// Should NOT do: draw anything, touch Vulkan, or let games register new zones
// at runtime (the zone set is deliberately closed and hand-written).
//
// The renderer already reports GPU pass timings and one aggregate CPU number
// (see render/backend/vulkan/frame_run.cc). What was missing for src/games/*
// is the CPU side of the main loop: GameApp::run() drives all seven games and
// had no timing at all, so "the frame got slower" could not be attributed to
// simulation, UI, audio, or submission without a native profiler.
namespace odai::engine {

// Zone set matches the call order in GameApp::run() / GameApp::submitFrame().
// UiBuild and Submit are measured *inside* Render, so they are reported as
// nested rows and must not be added to the flat total.
enum class GameZone : std::size_t {
    Poll = 0,   // glfwPollEvents + per-frame input sampling
    UiUpdate,   // UiContext::setViewport/update/tick
    Tick,       // GameApp::onTick -- the game's own simulation step
    Plugins,    // PluginRegistry::tickAll
    Audio,      // Audio::update
    Render,     // GameApp::onRender in full (contains UiBuild + Submit)
    UiBuild,    // UiContext::buildAppend + cursor      [nested in Render]
    Submit,     // Renderer::renderFrame                [nested in Render]
    Frame,      // the whole loop iteration
    Count
};

inline constexpr std::size_t kGameZoneCount = static_cast<std::size_t>(GameZone::Count);

constexpr std::size_t zoneIndex(GameZone zone) { return static_cast<std::size_t>(zone); }

[[nodiscard]] constexpr const char* gameZoneName(GameZone zone) {
    switch (zone) {
        case GameZone::Poll:     return "poll";
        case GameZone::UiUpdate: return "ui update";
        case GameZone::Tick:     return "tick";
        case GameZone::Plugins:  return "plugins";
        case GameZone::Audio:    return "audio";
        case GameZone::Render:   return "render";
        case GameZone::UiBuild:  return "  ui build";
        case GameZone::Submit:   return "  submit";
        case GameZone::Frame:    return "FRAME";
        case GameZone::Count:    break;
    }
    return "?";
}

// True for zones already counted inside another zone. Callers summing "where
// did the frame go" must skip these or they will double-count Render.
[[nodiscard]] constexpr bool gameZoneIsNested(GameZone zone) {
    return zone == GameZone::UiBuild || zone == GameZone::Submit;
}

// Per-frame accumulators plus a rolling window per zone.
//
// Usage is accumulate-then-commit: beginFrame() zeroes the accumulators,
// scoped timers add into zoneMs(...) through the frame, endFrame() pushes the
// accumulators into their channels. Nothing here allocates after construction.
class GameFrameProfiler {
public:
    // ~4 seconds at 60 Hz -- long enough for a p99 to mean something, short
    // enough that the overlay reacts to a change while you are watching it.
    static constexpr std::size_t kHistorySamples = 240;
    using Channel = core::TimingChannel<kHistorySamples>;

    void beginFrame() { m_current.fill(0.0f); }

    // Accumulator for the frame in flight. Hand this to a core::ScopedTimerMs.
    [[nodiscard]] float& zoneMs(GameZone zone) { return m_current[zoneIndex(zone)]; }

    void endFrame(float frameMs) {
        m_current[zoneIndex(GameZone::Frame)] = frameMs;
        for (std::size_t i = 0; i < kGameZoneCount; ++i) {
            m_channels[i].push(m_current[i]);
        }
        ++m_frameIndex;
    }

    [[nodiscard]] const Channel& channel(GameZone zone) const { return m_channels[zoneIndex(zone)]; }
    [[nodiscard]] std::uint64_t frameIndex() const noexcept { return m_frameIndex; }

    // Smoothed rate derived from the frame channel, so it does not jitter the
    // way a raw 1/dt readout does.
    [[nodiscard]] float fps() const {
        const float ms = m_channels[zoneIndex(GameZone::Frame)].ewmaMs();
        return ms > 0.0001f ? (1000.0f / ms) : 0.0f;
    }

    // Frame time not attributed to any top-level zone: loop overhead, the
    // profiler itself, and anything a game does outside the measured calls.
    [[nodiscard]] float unattributedMs() const {
        float summed = 0.0f;
        for (std::size_t i = 0; i < kGameZoneCount; ++i) {
            const auto zone = static_cast<GameZone>(i);
            if (zone == GameZone::Frame || gameZoneIsNested(zone)) continue;
            summed += m_channels[i].lastMs();
        }
        const float total = m_channels[zoneIndex(GameZone::Frame)].lastMs();
        return total > summed ? (total - summed) : 0.0f;
    }

private:
    std::array<float, kGameZoneCount> m_current{};
    std::array<Channel, kGameZoneCount> m_channels{};
    std::uint64_t m_frameIndex = 0;
};

}  // namespace odai::engine
