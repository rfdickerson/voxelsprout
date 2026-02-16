#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>
#include "core/grid3.h"
#include "core/log.h"
#include "math/math.h"
#include "sim/network_procedural.h"
#include "world/chunk_mesher.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace voxelsprout::render {

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "render/renderer_shared.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

void RendererBackend::setDebugUiVisible(bool visible) {
    if (m_debugUiVisible == visible) {
        return;
    }
    m_debugUiVisible = visible;
    m_showMeshingPanel = visible;
    m_showShadowPanel = visible;
    m_showSunPanel = visible;
}


bool RendererBackend::isDebugUiVisible() const {
    return m_debugUiVisible;
}


void RendererBackend::setFrameStatsVisible(bool visible) {
    m_showFrameStatsPanel = visible;
}


bool RendererBackend::isFrameStatsVisible() const {
    return m_showFrameStatsPanel;
}


void RendererBackend::setSunAngles(float yawDegrees, float pitchDegrees) {
    m_skyDebugSettings.sunYawDegrees = yawDegrees;
    m_skyDebugSettings.sunPitchDegrees = std::clamp(pitchDegrees, -89.0f, 5.0f);
}


float RendererBackend::cameraFovDegrees() const {
    return m_debugCameraFovDegrees;
}


void RendererBackend::buildFrameStatsUi() {
    if (!m_showFrameStatsPanel) {
        return;
    }

    constexpr ImGuiWindowFlags kPanelFlags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings;
    if (!ImGui::Begin("Frame Stats", &m_showFrameStatsPanel, kPanelFlags)) {
        ImGui::End();
        return;
    }

    const float autoScale = std::numeric_limits<float>::max();
    if (m_debugCpuFrameTimingMsHistoryCount > 0) {
        const int cpuHistoryCount = static_cast<int>(m_debugCpuFrameTimingMsHistoryCount);
        const int cpuHistoryOffset =
            (m_debugCpuFrameTimingMsHistoryCount == kTimingHistorySampleCount)
                ? static_cast<int>(m_debugCpuFrameTimingMsHistoryWrite)
                : 0;
        ImGui::PlotLines(
            "CPU Work (ms)",
            m_debugCpuFrameWorkMsHistory.data(),
            cpuHistoryCount,
            cpuHistoryOffset,
            nullptr,
            0.0f,
            autoScale,
            ImVec2(0.0f, 64.0f)
        );
    } else {
        ImGui::Text("CPU Timing (ms): collecting...");
    }

    if (m_gpuTimestampsSupported) {
        if (m_debugGpuFrameTimingMsHistoryCount > 0) {
            const int gpuHistoryCount = static_cast<int>(m_debugGpuFrameTimingMsHistoryCount);
            const int gpuHistoryOffset =
                (m_debugGpuFrameTimingMsHistoryCount == kTimingHistorySampleCount)
                    ? static_cast<int>(m_debugGpuFrameTimingMsHistoryWrite)
                    : 0;
            ImGui::PlotLines(
                "GPU Frame (ms)",
                m_debugGpuFrameTimingMsHistory.data(),
                gpuHistoryCount,
                gpuHistoryOffset,
                nullptr,
                0.0f,
                autoScale,
                ImVec2(0.0f, 64.0f)
            );
        } else {
            ImGui::Text("GPU Frame (ms): collecting...");
        }
    } else {
        ImGui::Text("GPU Frame (ms): unavailable");
    }
    if (m_debugPresentedFrameTimingMsHistoryCount > 0) {
        const int presentHistoryCount = static_cast<int>(m_debugPresentedFrameTimingMsHistoryCount);
        const int presentHistoryOffset =
            (m_debugPresentedFrameTimingMsHistoryCount == kTimingHistorySampleCount)
                ? static_cast<int>(m_debugPresentedFrameTimingMsHistoryWrite)
                : 0;
        ImGui::PlotLines(
            "Presented Frame (ms)",
            m_debugPresentedFrameTimingMsHistory.data(),
            presentHistoryCount,
            presentHistoryOffset,
            nullptr,
            0.0f,
            autoScale,
            ImVec2(0.0f, 64.0f)
        );
    }

    ImGui::Text("FPS (submit/presented): %.1f / %.1f", m_debugFps, m_debugPresentedFps);
    ImGui::Text("Chunks (visible/total): %u / %u", m_debugSpatialVisibleChunkCount, m_debugChunkCount);
    if (m_gpuTimestampsSupported) {
        ImGui::Text(
            "Frame CPU (total/work/ewma): %.2f / %.2f / %.2f ms",
            m_debugFrameTimeMs,
            m_debugCpuFrameWorkMs,
            m_debugCpuFrameEwmaMs
        );
        ImGui::Text("Frame CPU P50/P95/P99: %.2f / %.2f / %.2f ms", m_debugCpuFrameP50Ms, m_debugCpuFrameP95Ms, m_debugCpuFrameP99Ms);
        ImGui::Text("GI Occupancy CPU (chunk pack): %.2f ms", m_debugCpuGiOccupancyBuildMs);
        ImGui::Text("Frame GPU: %.2f ms", m_debugGpuFrameTimeMs);
        ImGui::Text("Frame GPU P50/P95/P99: %.2f / %.2f / %.2f ms", m_debugGpuFrameP50Ms, m_debugGpuFrameP95Ms, m_debugGpuFrameP99Ms);
        if (m_debugPresentedFrameTimingMsHistoryCount > 0) {
            ImGui::Text(
                "Presented Frame (last/P50/P95/P99): %.2f / %.2f / %.2f / %.2f ms",
                m_debugPresentedFrameTimeMs,
                m_debugPresentedFrameP50Ms,
                m_debugPresentedFrameP95Ms,
                m_debugPresentedFrameP99Ms
            );
        }
        if (ImGui::TreeNodeEx("GPU Stages (ms)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Shadow: %.2f", m_debugGpuShadowTimeMs);
            ImGui::Text("GI Inject (compute): %.2f", m_debugGpuGiInjectTimeMs);
            ImGui::Text("GI Propagate (compute): %.2f", m_debugGpuGiPropagateTimeMs);
            ImGui::Text("Auto Exposure (compute): %.2f", m_debugGpuAutoExposureTimeMs);
            ImGui::Text("Sun Shafts (compute): %.2f", m_debugGpuSunShaftTimeMs);
            ImGui::Text("Prepass: %.2f", m_debugGpuPrepassTimeMs);
            ImGui::Text("SSAO: %.2f", m_debugGpuSsaoTimeMs);
            ImGui::Text("SSAO Blur: %.2f", m_debugGpuSsaoBlurTimeMs);
            ImGui::Text("Main: %.2f", m_debugGpuMainTimeMs);
            ImGui::Text("Post: %.2f", m_debugGpuPostTimeMs);
            ImGui::TreePop();
        }
    } else {
        ImGui::Text(
            "Frame CPU (total/work/ewma): %.2f / %.2f / %.2f ms",
            m_debugFrameTimeMs,
            m_debugCpuFrameWorkMs,
            m_debugCpuFrameEwmaMs
        );
        ImGui::Text("Frame CPU P50/P95/P99: %.2f / %.2f / %.2f ms", m_debugCpuFrameP50Ms, m_debugCpuFrameP95Ms, m_debugCpuFrameP99Ms);
        ImGui::Text("GI Occupancy CPU (chunk pack): %.2f ms", m_debugCpuGiOccupancyBuildMs);
        ImGui::Text("Frame GPU: n/a");
        if (m_debugPresentedFrameTimingMsHistoryCount > 0) {
            ImGui::Text(
                "Presented Frame (last/P50/P95/P99): %.2f / %.2f / %.2f / %.2f ms",
                m_debugPresentedFrameTimeMs,
                m_debugPresentedFrameP50Ms,
                m_debugPresentedFrameP95Ms,
                m_debugPresentedFrameP99Ms
            );
        }
    }
    if (m_supportsDisplayTiming) {
        ImGui::Text(
            "Display Timing Present ID submit/presented: %u / %u",
            m_lastSubmittedDisplayTimingPresentId,
            m_lastPresentedDisplayTimingPresentId
        );
        ImGui::Text("Display Refresh: %.3f ms", m_debugDisplayRefreshMs);
        ImGui::Text("Display Present Margin: %.3f ms", m_debugDisplayPresentMarginMs);
        ImGui::Text("Display Actual-Earliest: %.3f ms", m_debugDisplayActualEarliestDeltaMs);
        ImGui::Text("Display Timing Samples: %u", m_debugDisplayTimingSampleCount);
    }
    if (ImGui::TreeNodeEx("Draw Calls", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Total: %u", m_debugDrawCallsTotal);
        ImGui::Text("Shadow: %u", m_debugDrawCallsShadow);
        ImGui::Text("Prepass: %u", m_debugDrawCallsPrepass);
        ImGui::Text("Main: %u", m_debugDrawCallsMain);
        ImGui::Text("Post: %u", m_debugDrawCallsPost);
        ImGui::TreePop();
    }
    ImGui::Text("Chunk Indirect Commands: %u", m_debugChunkIndirectCommandCount);
    ImGui::Text(
        "Spatial Query N/C/V: %u / %u / %u",
        m_debugSpatialQueryStats.visitedNodeCount,
        m_debugSpatialQueryStats.candidateChunkCount,
        m_debugSpatialQueryStats.visibleChunkCount
    );
    if (m_debugSpatialQueryStats.clipmapActiveLevelCount > 0) {
        ImGui::Text(
            "Clipmap L/U/S/B: %u / %u / %u / %u",
            m_debugSpatialQueryStats.clipmapActiveLevelCount,
            m_debugSpatialQueryStats.clipmapUpdatedLevelCount,
            m_debugSpatialQueryStats.clipmapUpdatedSlabCount,
            m_debugSpatialQueryStats.clipmapUpdatedBrickCount
        );
    }
    ImGui::Text("Chunk Mesh Vert/Idx: %u / %u", m_debugChunkMeshVertexCount, m_debugChunkMeshIndexCount);
    ImGui::Text("Last Chunk Remesh: %.2f ms (%u)", m_debugChunkLastRemeshMs, m_debugChunkLastRemeshedChunkCount);
    ImGui::Text("Greedy Reduction vs Naive: %.1f%%", m_debugChunkLastRemeshReductionPercent);
    const bool hasFrameArenaMetrics =
        m_debugFrameArenaUploadBytes > 0 ||
        m_debugFrameArenaUploadAllocs > 0 ||
        m_debugFrameArenaTransientBufferBytes > 0 ||
        m_debugFrameArenaTransientBufferCount > 0 ||
        m_debugFrameArenaTransientImageBytes > 0 ||
        m_debugFrameArenaTransientImageCount > 0 ||
        m_debugFrameArenaAliasReuses > 0 ||
        m_debugFrameArenaResidentBufferBytes > 0 ||
        m_debugFrameArenaResidentBufferCount > 0 ||
        m_debugFrameArenaResidentImageBytes > 0 ||
        m_debugFrameArenaResidentImageCount > 0 ||
        m_debugFrameArenaResidentAliasReuses > 0 ||
        !m_debugAliasedImages.empty();
    if (hasFrameArenaMetrics) {
        ImGui::Separator();
        ImGui::Text("FrameArena");
        if (m_debugFrameArenaUploadBytes > 0 || m_debugFrameArenaUploadAllocs > 0) {
            ImGui::Text(
                "Upload this frame: %llu B (%u allocs)",
                static_cast<unsigned long long>(m_debugFrameArenaUploadBytes),
                m_debugFrameArenaUploadAllocs
            );
        }
        ImGui::Text("Image alias reuses (frame/live): %u / %u", m_debugFrameArenaAliasReuses, m_debugFrameArenaResidentAliasReuses);
        ImGui::Text("Resident images (live): %u", m_debugFrameArenaResidentImageCount);
    }
    ImGui::End();
}


void RendererBackend::buildMeshingDebugUi() {
    if (!m_debugUiVisible || !m_showMeshingPanel) {
        return;
    }

    if (!ImGui::Begin("Meshing", &m_showMeshingPanel)) {
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Use Spatial Queries", &m_debugEnableSpatialQueries);
    int clipmapLevels = static_cast<int>(m_debugClipmapConfig.levelCount);
    int clipmapGridResolution = m_debugClipmapConfig.gridResolution;
    int clipmapBaseVoxelSize = m_debugClipmapConfig.baseVoxelSize;
    int clipmapBrickResolution = m_debugClipmapConfig.brickResolution;
    if (ImGui::SliderInt("Clipmap Levels", &clipmapLevels, 1, 8)) {
        m_debugClipmapConfig.levelCount = static_cast<std::uint32_t>(clipmapLevels);
    }
    if (ImGui::SliderInt("Clipmap Grid Res", &clipmapGridResolution, 32, 256)) {
        m_debugClipmapConfig.gridResolution = clipmapGridResolution;
    }
    if (ImGui::SliderInt("Clipmap Base Voxel", &clipmapBaseVoxelSize, 1, 8)) {
        m_debugClipmapConfig.baseVoxelSize = clipmapBaseVoxelSize;
    }
    if (ImGui::SliderInt("Clipmap Brick Res", &clipmapBrickResolution, 2, 32)) {
        m_debugClipmapConfig.brickResolution = clipmapBrickResolution;
    }

    int meshingModeSelection = (m_chunkMeshingOptions.mode == voxelsprout::world::MeshingMode::Greedy) ? 1 : 0;
    if (ImGui::Combo("Chunk Meshing", &meshingModeSelection, "Naive\0Greedy\0")) {
        const voxelsprout::world::MeshingMode nextMode =
            (meshingModeSelection == 1) ? voxelsprout::world::MeshingMode::Greedy : voxelsprout::world::MeshingMode::Naive;
        if (nextMode != m_chunkMeshingOptions.mode) {
            m_chunkMeshingOptions.mode = nextMode;
            m_chunkLodMeshCacheValid = false;
            m_chunkMeshRebuildRequested = true;
            m_pendingChunkRemeshIndices.clear();
            VOX_LOGI("render") << "chunk meshing mode changed to "
                               << (nextMode == voxelsprout::world::MeshingMode::Greedy ? "Greedy" : "Naive")
                               << ", scheduling full remesh";
        }
    }

    ImGui::Text(
        "Query N/C/V: %u / %u / %u",
        m_debugSpatialQueryStats.visitedNodeCount,
        m_debugSpatialQueryStats.candidateChunkCount,
        m_debugSpatialQueryStats.visibleChunkCount
    );
    if (m_debugSpatialQueryStats.clipmapActiveLevelCount > 0) {
        ImGui::Text(
            "Clipmap L/U/S/B: %u / %u / %u / %u",
            m_debugSpatialQueryStats.clipmapActiveLevelCount,
            m_debugSpatialQueryStats.clipmapUpdatedLevelCount,
            m_debugSpatialQueryStats.clipmapUpdatedSlabCount,
            m_debugSpatialQueryStats.clipmapUpdatedBrickCount
        );
    }

    ImGui::Text("Chunk Mesh Vert/Idx: %u / %u", m_debugChunkMeshVertexCount, m_debugChunkMeshIndexCount);
    ImGui::Text("Last Chunk Remesh: %.2f ms (%u)", m_debugChunkLastRemeshMs, m_debugChunkLastRemeshedChunkCount);
    ImGui::Text("Greedy Reduction vs Naive: %.1f%%", m_debugChunkLastRemeshReductionPercent);
    ImGui::End();
}


void RendererBackend::buildAimReticleUi() {
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    if (drawList == nullptr) {
        return;
    }

    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const ImVec2 center{displaySize.x * 0.5f, displaySize.y * 0.5f};
    constexpr float kOuter = 9.0f;
    constexpr float kInner = 3.0f;
    constexpr float kThickness = 1.6f;
    const ImU32 color = IM_COL32(235, 245, 255, 220);

    drawList->AddLine(ImVec2(center.x - kOuter, center.y), ImVec2(center.x - kInner, center.y), color, kThickness);
    drawList->AddLine(ImVec2(center.x + kInner, center.y), ImVec2(center.x + kOuter, center.y), color, kThickness);
    drawList->AddLine(ImVec2(center.x, center.y - kOuter), ImVec2(center.x, center.y - kInner), color, kThickness);
    drawList->AddLine(ImVec2(center.x, center.y + kInner), ImVec2(center.x, center.y + kOuter), color, kThickness);
}


} // namespace voxelsprout::render
