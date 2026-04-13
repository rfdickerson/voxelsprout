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

#include "render/renderer_shared.h"

namespace {

const char* shadowModeName(ShadowMode mode) {
    switch (mode) {
    case ShadowMode::ShadowMaps: return "shadow_maps";
    case ShadowMode::RayTraced: return "ray_traced";
    case ShadowMode::Auto: return "auto";
    }
    return "shadow_maps";
}

const char* shadowFallbackReasonName(ShadowFallbackReason reason) {
    switch (reason) {
    case ShadowFallbackReason::None: return "none";
    case ShadowFallbackReason::RayTracingUnsupported: return "rt_unsupported";
    case ShadowFallbackReason::RayTracingDisabled: return "rt_runtime_disabled";
    case ShadowFallbackReason::MainPassNotImplemented: return "rt_main_not_implemented";
    case ShadowFallbackReason::RayTracingSceneUnavailable: return "rt_scene_unavailable";
    }
    return "none";
}

const char* voxelGiSurfaceModeName(VoxelGiSurfaceMode mode) {
    switch (mode) {
    case VoxelGiSurfaceMode::Legacy: return "legacy";
    case VoxelGiSurfaceMode::RtSurface: return "rt_surface";
    case VoxelGiSurfaceMode::RestirSurface: return "restir_surface";
    }
    return "legacy";
}

} // namespace

void RendererBackend::setDebugUiVisible(bool visible) {
    if (m_debugUiVisible == visible) {
        return;
    }
    m_debugUiVisible = visible;
    m_showMeshingPanel = visible;
    m_showShadowPanel = visible;
    m_showSunPanel = visible;
}

void RendererBackend::setGameplayUiState(const GameplayUiState& state) {
    m_gameplayUiState = state;
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

void RendererBackend::setFramePacingSettings(const FramePacingSettings& settings) {
    m_framePacingSettings.mode = settings.mode;
    m_framePacingSettings.cadenceDivisor = std::max(1u, settings.cadenceDivisor);
    m_framePacingSettings.maxQueuedFrames = std::clamp(settings.maxQueuedFrames, 1u, kMaxFramesInFlight);
}

FramePacingSettings RendererBackend::framePacingSettings() const {
    return m_framePacingSettings;
}

FramePacingStats RendererBackend::framePacingStats() const {
    return m_framePacingStats;
}

void RendererBackend::setVertexAoEnabled(bool enabled) {
    m_debugEnableVertexAo = enabled;
}

bool RendererBackend::isVertexAoEnabled() const {
    return m_debugEnableVertexAo;
}

void RendererBackend::setSsaoEnabled(bool enabled) {
    m_debugEnableSsao = enabled;
    if (!enabled) {
        m_debugVisualizeSsao = false;
        m_debugVisualizeAoNormals = false;
    }
}

bool RendererBackend::isSsaoEnabled() const {
    return m_debugEnableSsao;
}

bool RendererBackend::rayTracingRuntimeReady() const {
    return m_rayTracingRuntimeEnabled &&
        m_rayTracingCapabilityProbe.accelerationStructureFeature &&
        m_rayTracingCapabilityProbe.rayQueryFeature &&
        m_createAccelerationStructureKhr != nullptr &&
        m_destroyAccelerationStructureKhr != nullptr &&
        m_getAccelerationStructureBuildSizesKhr != nullptr &&
        m_cmdBuildAccelerationStructuresKhr != nullptr &&
        m_getAccelerationStructureDeviceAddressKhr != nullptr;
}

const char* RendererBackend::rayTracingReleaseStatusName() const {
    const bool rtRequested =
        m_shadowSettings.mode == ShadowMode::RayTraced ||
        m_shadowSettings.mode == ShadowMode::Auto;
    if (!m_shadowStats.rayTracingSupported) {
        return "unsupported";
    }
    if (!m_shadowStats.rayTracingRuntimeEnabled || !rtRequested) {
        return "supported_but_disabled";
    }
    if (m_shadowStats.activeMode == ShadowMode::RayTraced && m_shadowStats.mainPassRayTracingReady) {
        return "beta_ready";
    }
    if (m_shadowStats.fallbackActive) {
        return "beta_requested_but_fell_back";
    }
    return "supported_but_disabled";
}

void RendererBackend::refreshShadowStats() {
    m_shadowStats.requestedMode = m_shadowSettings.mode;
    m_shadowStats.rayTracingSupported = m_rayTracingCapabilityProbe.rayTracingCoreReady;
    m_shadowStats.rayQuerySupported = m_rayTracingCapabilityProbe.rayQueryExtension;
    m_shadowStats.accelerationStructureSupported = m_rayTracingCapabilityProbe.accelerationStructureExtension;
    m_shadowStats.rayTracingRuntimeEnabled = rayTracingRuntimeReady();
    const bool mainPassPipelinesReady =
        m_rtMainPassImplemented &&
        m_pipelineRt != VK_NULL_HANDLE &&
        m_magicaPipelineRt != VK_NULL_HANDLE;
    const bool mainPassSceneReady = m_rtTlas.handle != VK_NULL_HANDLE;
    m_shadowStats.mainPassRayTracingReady =
        rayTracingRuntimeReady() &&
        mainPassPipelinesReady &&
        mainPassSceneReady;
    m_shadowStats.mainPassRayTracingActive = false;
    m_shadowStats.activeMode = ShadowMode::ShadowMaps;
    m_shadowStats.fallbackActive = false;
    m_shadowStats.fallbackReason = ShadowFallbackReason::None;

    const bool wantsRayTracing =
        m_shadowSettings.mode == ShadowMode::RayTraced ||
        m_shadowSettings.mode == ShadowMode::Auto;
    if (!wantsRayTracing) {
        return;
    }
    if (!m_shadowStats.rayTracingSupported) {
        m_shadowStats.fallbackActive = true;
        m_shadowStats.fallbackReason = ShadowFallbackReason::RayTracingUnsupported;
        return;
    }
    if (!m_shadowStats.rayTracingRuntimeEnabled) {
        m_shadowStats.fallbackActive = true;
        m_shadowStats.fallbackReason = ShadowFallbackReason::RayTracingDisabled;
        return;
    }
    if (m_shadowStats.mainPassRayTracingReady) {
        m_shadowStats.activeMode = ShadowMode::RayTraced;
        m_shadowStats.mainPassRayTracingActive = true;
        return;
    }
    if (!mainPassPipelinesReady) {
        m_shadowStats.fallbackActive = true;
        m_shadowStats.fallbackReason = ShadowFallbackReason::MainPassNotImplemented;
        return;
    }
    m_shadowStats.fallbackActive = true;
    m_shadowStats.fallbackReason = ShadowFallbackReason::RayTracingSceneUnavailable;
}

void RendererBackend::setShadowSettings(const ShadowSettings& settings) {
    m_shadowSettings = settings;
    refreshShadowStats();
    if (m_device == VK_NULL_HANDLE) {
        return;
    }
    VOX_LOGI("render") << "shadow mode: requested=" << shadowModeName(m_shadowStats.requestedMode)
                       << ", active=" << shadowModeName(m_shadowStats.activeMode)
                       << ", fallback=" << (m_shadowStats.fallbackActive ? "yes" : "no")
                       << ", reason=" << shadowFallbackReasonName(m_shadowStats.fallbackReason)
                       << ", rtCoreReady=" << (m_rayTracingCapabilityProbe.rayTracingCoreReady ? "yes" : "no")
                       << ", rtRuntime=" << (m_shadowStats.rayTracingRuntimeEnabled ? "yes" : "no")
                       << ", rtReleaseStatus=" << rayTracingReleaseStatusName();
}

ShadowSettings RendererBackend::shadowSettings() const {
    return m_shadowSettings;
}

ShadowStats RendererBackend::shadowStats() const {
    return m_shadowStats;
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
    if (ImGui::TreeNodeEx("Frame Pacing", ImGuiTreeNodeFlags_DefaultOpen)) {
        int pacingMode = static_cast<int>(m_framePacingSettings.mode);
        if (ImGui::Combo("Mode", &pacingMode, "Off\0Passive\0Scheduled\0")) {
            m_framePacingSettings.mode = static_cast<FramePacingMode>(pacingMode);
        }
        int cadenceDivisor = static_cast<int>(m_framePacingSettings.cadenceDivisor);
        if (ImGui::SliderInt("Cadence Divisor", &cadenceDivisor, 1, 4)) {
            m_framePacingSettings.cadenceDivisor = static_cast<std::uint32_t>(cadenceDivisor);
        }
        int maxQueuedFrames = static_cast<int>(m_framePacingSettings.maxQueuedFrames);
        if (ImGui::SliderInt("Max Queued Frames", &maxQueuedFrames, 1, static_cast<int>(kMaxFramesInFlight))) {
            m_framePacingSettings.maxQueuedFrames = static_cast<std::uint32_t>(maxQueuedFrames);
        }
        ImGui::Text(
            "Active: %s",
            m_framePacingStats.schedulingActive
                ? "Scheduled"
                : (m_framePacingStats.displayTimingEnabled ? "Passive" : "Off")
        );
        ImGui::Text(
            "Queued Frames / Limit: %u / %u",
            m_framePacingStats.queuedFrames,
            m_framePacingStats.maxQueuedFrames
        );
        ImGui::Text(
            "CPU Waits slot/acquire/present/transfer: %.3f / %.3f / %.3f / %.3f ms",
            m_framePacingStats.cpuWaitFrameSlotMs,
            m_framePacingStats.cpuWaitAcquireMs,
            m_framePacingStats.cpuWaitPresentMs,
            m_framePacingStats.cpuWaitTransferMs
        );
        ImGui::Text(
            "GPU Timestamp Readback: %s (%u skipped)",
            m_framePacingStats.gpuTimestampsPending ? "pending" : "ready",
            m_framePacingStats.gpuTimestampSkippedFrames
        );
        ImGui::Text("Target Present Interval: %.3f ms", m_framePacingStats.targetPresentIntervalMs);
        ImGui::Text("Desired Lead Time: %.3f ms", m_framePacingStats.desiredLeadTimeMs);
        ImGui::Text("Schedule Error: %.3f ms", m_framePacingStats.presentScheduleErrorMs);
        ImGui::Text("Late Presents: %u", m_framePacingStats.latePresentCount);
        ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Shadow Mode", ImGuiTreeNodeFlags_DefaultOpen)) {
        int shadowMode = static_cast<int>(m_shadowSettings.mode);
        if (ImGui::Combo("Shadow Backend", &shadowMode, "Shadow Maps\0Ray Traced (Beta)\0Auto (Beta)\0")) {
            setShadowSettings(ShadowSettings{static_cast<ShadowMode>(shadowMode)});
        }
        const char* activeModeLabel = "Shadow Maps";
        if (m_shadowStats.activeMode == ShadowMode::RayTraced) {
            activeModeLabel = "Ray Traced";
        } else if (m_shadowStats.activeMode == ShadowMode::Auto) {
            activeModeLabel = "Auto";
        }
        ImGui::Text("Active: %s", activeModeLabel);
        ImGui::Text("Fallback Active: %s", m_shadowStats.fallbackActive ? "yes" : "no");
        ImGui::Text("Fallback Reason: %s", shadowFallbackReasonName(m_shadowStats.fallbackReason));
        ImGui::Text("RT Release Status: %s", rayTracingReleaseStatusName());
        ImGui::Text("RT Core Ready: %s", m_shadowStats.rayTracingSupported ? "yes" : "no");
        ImGui::Text("RT Runtime Enabled: %s", m_shadowStats.rayTracingRuntimeEnabled ? "yes" : "no");
        ImGui::Text("Main Pass RT Ready: %s", m_shadowStats.mainPassRayTracingReady ? "yes" : "no");
        ImGui::Text("Main Pass RT Active: %s", m_shadowStats.mainPassRayTracingActive ? "yes" : "no");
        ImGui::Text(
            "RT Samples / Sun Radius: %d / %.2f deg",
            std::clamp(m_shadowDebugSettings.rtShadowSampleCount, 1, 8),
            m_shadowDebugSettings.rtSunAngularRadiusDegrees
        );
        ImGui::Text("RT Beta Scope: main-pass voxels + Magica only");
        ImGui::Text(
            "RT Scene Builds / BLAS / TLAS: %u / %u / %u",
            m_rtSceneBuildCount,
            m_rtBlasBuildCount,
            m_rtTlasBuildCount
        );
        ImGui::Text(
            "GI Surface Mode: requested=%s active=%s",
            voxelGiSurfaceModeName(m_voxelGiDebugSettings.surfaceMode),
            m_voxelGiRestirActiveThisFrame
                ? "restir_surface"
                : (m_voxelGiRtSurfaceActiveThisFrame ? "rt_surface" : "legacy")
        );
        ImGui::Text(
            "GI RT Surface: %s / active=%s",
            m_voxelGiRtSurfaceReady ? "ready" : "fallback",
            m_voxelGiRtSurfaceActiveThisFrame ? "yes" : "no"
        );
        ImGui::Text(
            "GI ReSTIR: %s / active=%s (%d cand, temporal=%s, spatial=%s, radius=%d)",
            m_voxelGiRestirReady ? "ready" : "fallback",
            m_voxelGiRestirActiveThisFrame ? "yes" : "no",
            std::clamp(m_voxelGiDebugSettings.restirCandidateCount, 1, 8),
            m_voxelGiDebugSettings.restirEnableTemporalReuse ? "yes" : "no",
            m_voxelGiDebugSettings.restirEnableSpatialReuse ? "yes" : "no",
            std::clamp(m_voxelGiDebugSettings.restirSpatialRadius, 1, 2)
        );
        if (m_voxelGiDebugSettings.surfaceMode == VoxelGiSurfaceMode::RestirSurface && !m_voxelGiRestirActiveThisFrame) {
            ImGui::Text(
                "GI ReSTIR Fallback: %s",
                m_voxelGiRtSurfaceActiveThisFrame ? "rt_surface" : "legacy"
            );
        }
        ImGui::Text(
            "GI RT Samples / Bias: %d / %.2f",
            std::clamp(m_voxelGiDebugSettings.rtSurfaceSampleCount, 1, 2),
            m_voxelGiDebugSettings.rtSurfaceBiasScale
        );
        ImGui::Text("GI ReSTIR History: %s (%s)", m_voxelGiRestirHistoryValid ? "valid" : "reset", m_voxelGiRestirHistoryResetReason.c_str());
        ImGui::Text("Ray Query: %s", m_shadowStats.rayQuerySupported ? "yes" : "no");
        ImGui::Text(
            "Acceleration Structure: %s",
            m_shadowStats.accelerationStructureSupported ? "yes" : "no"
        );
        ImGui::TreePop();
    }
    if (m_gpuTimestampsSupported) {
        ImGui::Text(
            "Frame CPU (total/work/ewma): %.2f / %.2f / %.2f ms",
            m_debugFrameTimeMs,
            m_debugCpuFrameWorkMs,
            m_debugCpuFrameEwmaMs
        );
        ImGui::Text("Frame CPU P50/P95/P99: %.2f / %.2f / %.2f ms", m_debugCpuFrameP50Ms, m_debugCpuFrameP95Ms, m_debugCpuFrameP99Ms);
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
            ImGui::Text("GI Occupancy (compute): %.2f", m_debugGpuGiOccupancyTimeMs);
            ImGui::Text("GI Surface (compute): %.2f", m_debugGpuGiSurfaceTimeMs);
            ImGui::Text("GI ReSTIR Candidate (compute): %.2f", m_debugGpuGiSurfaceCandidateTimeMs);
            ImGui::Text("GI ReSTIR Temporal (compute): %.2f", m_debugGpuGiSurfaceTemporalTimeMs);
            ImGui::Text("GI ReSTIR Spatial (compute): %.2f", m_debugGpuGiSurfaceSpatialTimeMs);
            ImGui::Text("GI ReSTIR Resolve (compute): %.2f", m_debugGpuGiSurfaceResolveTimeMs);
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
        ImGui::Text("Display Schedule Error: %.3f ms", m_debugDisplayScheduleErrorMs);
        ImGui::Text("Display Timing Samples: %u", m_debugDisplayTimingSampleCount);
    }
    if (ImGui::TreeNodeEx("Desktop Capabilities", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Roadmap 2026 Core Ready: %s", m_desktopCapabilityProbe.roadmap2026CoreReady ? "yes" : "no");
        ImGui::Text("Descriptor Heap: %s", m_desktopCapabilityProbe.descriptorHeapExtension ? "yes" : "no");
        ImGui::Text(
            "Unified Image Layouts: %s",
            m_desktopCapabilityProbe.unifiedImageLayoutsExtension ? "yes" : "no"
        );
        ImGui::Text("Host Image Copy: %s", m_desktopCapabilityProbe.hostImageCopyExtension ? "yes" : "no");
        ImGui::Text("Shader Clock: %s", m_desktopCapabilityProbe.shaderClockExtension ? "yes" : "no");
        ImGui::Text(
            "Compute Shader Derivatives: %s",
            m_desktopCapabilityProbe.computeShaderDerivativesExtension ? "yes" : "no"
        );
        ImGui::Text(
            "Fragment Shading Rate: %s",
            m_desktopCapabilityProbe.fragmentShadingRateExtension ? "yes" : "no"
        );
        ImGui::Text(
            "Swapchain Maintenance1: %s",
            m_desktopCapabilityProbe.swapchainMaintenance1Extension ? "yes" : "no"
        );
        ImGui::Text("Present ID: %s", m_desktopCapabilityProbe.presentIdExtension ? "yes" : "no");
        ImGui::Text("Present Wait: %s", m_desktopCapabilityProbe.presentWaitExtension ? "yes" : "no");
        ImGui::Text(
            "Descriptor Limits S/SI/DS/Frag: %u / %u / %u / %u",
            m_desktopCapabilityProbe.maxPerStageDescriptorSamplers,
            m_desktopCapabilityProbe.maxPerStageDescriptorSampledImages,
            m_desktopCapabilityProbe.maxDescriptorSetSampledImages,
            m_desktopCapabilityProbe.maxFragmentCombinedOutputResources
        );
        ImGui::Text(
            "Storage Image Limit: %u",
            m_desktopCapabilityProbe.maxDescriptorSetStorageImages
        );
        ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Ray Tracing Capabilities", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text(
            "Acceleration Structure: %s",
            m_rayTracingCapabilityProbe.accelerationStructureExtension ? "yes" : "no"
        );
        ImGui::Text("Ray Query: %s", m_rayTracingCapabilityProbe.rayQueryExtension ? "yes" : "no");
        ImGui::Text(
            "Deferred Host Operations: %s",
            m_rayTracingCapabilityProbe.deferredHostOperationsExtension ? "yes" : "no"
        );
        ImGui::Text(
            "RT Pipeline: %s",
            m_rayTracingCapabilityProbe.rayTracingPipelineExtension ? "yes" : "no"
        );
        ImGui::Text(
            "RT Maintenance1: %s",
            m_rayTracingCapabilityProbe.rayTracingMaintenance1Extension ? "yes" : "no"
        );
        ImGui::Text(
            "RT Position Fetch: %s",
            m_rayTracingCapabilityProbe.rayTracingPositionFetchExtension ? "yes" : "no"
        );
        ImGui::Text(
            "Acceleration Structure Feature: %s",
            m_rayTracingCapabilityProbe.accelerationStructureFeature ? "yes" : "no"
        );
        ImGui::Text("Ray Query Feature: %s", m_rayTracingCapabilityProbe.rayQueryFeature ? "yes" : "no");
        ImGui::Text("RT Core Ready: %s", m_rayTracingCapabilityProbe.rayTracingCoreReady ? "yes" : "no");
        ImGui::Text(
            "Scratch Alignment: %llu",
            static_cast<unsigned long long>(m_rayTracingCapabilityProbe.scratchAlignment)
        );
        std::uint32_t chunkRtVertexCount = 0;
        std::uint32_t chunkRtIndexCount = 0;
        for (const RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
            chunkRtVertexCount += chunkRecord.vertexCount;
            chunkRtIndexCount += chunkRecord.indexCount;
        }
        ImGui::Text("Chunk RT Vertices: %u", chunkRtVertexCount);
        ImGui::Text("Chunk RT Indices: %u", chunkRtIndexCount);
        ImGui::Text("Chunk RT Records: %u", static_cast<unsigned>(m_rtChunkSceneRecords.size()));
        ImGui::Text("Magica RT Geometries: %u", static_cast<unsigned>(m_rtMagicaGeometries.size()));
        ImGui::TreePop();
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
        "Spatial Query N/C/V/R/New/Evict: %u / %u / %u / %u / %u / %u",
        m_debugSpatialQueryStats.visitedNodeCount,
        m_debugSpatialQueryStats.candidateChunkCount,
        m_debugSpatialQueryStats.visibleChunkCount,
        m_debugSpatialQueryStats.retainedChunkCount,
        m_debugSpatialQueryStats.newlyVisibleChunkCount,
        m_debugSpatialQueryStats.evictedChunkCount
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
        "Query N/C/V/R/New/Evict: %u / %u / %u / %u / %u / %u",
        m_debugSpatialQueryStats.visitedNodeCount,
        m_debugSpatialQueryStats.candidateChunkCount,
        m_debugSpatialQueryStats.visibleChunkCount,
        m_debugSpatialQueryStats.retainedChunkCount,
        m_debugSpatialQueryStats.newlyVisibleChunkCount,
        m_debugSpatialQueryStats.evictedChunkCount
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
    if (m_gameplayUiState.inventoryVisible || m_debugUiVisible) {
        return;
    }
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

void RendererBackend::buildGameplayHudUi() {
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    if (drawList == nullptr) {
        return;
    }

    auto itemLabel = [](InventoryItemId itemId) -> const char* {
        switch (itemId) {
        case InventoryItemId::Stone: return "Stone";
        case InventoryItemId::Dirt: return "Dirt";
        case InventoryItemId::Grass: return "Grass";
        case InventoryItemId::Wood: return "Wood";
        case InventoryItemId::Red: return "Red";
        case InventoryItemId::Empty:
        default:
            return "";
        }
    };
    auto itemTint = [](InventoryItemId itemId) -> ImU32 {
        switch (itemId) {
        case InventoryItemId::Stone: return IM_COL32(170, 176, 184, 255);
        case InventoryItemId::Dirt: return IM_COL32(145, 100, 62, 255);
        case InventoryItemId::Grass: return IM_COL32(95, 167, 82, 255);
        case InventoryItemId::Wood: return IM_COL32(162, 127, 88, 255);
        case InventoryItemId::Red: return IM_COL32(220, 86, 74, 255);
        case InventoryItemId::Empty:
        default:
            return IM_COL32(80, 86, 96, 255);
        }
    };

    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const GameplayUiLayout layout = buildGameplayUiLayout(displaySize.x, displaySize.y);
    const ImU32 panelColor = IM_COL32(12, 14, 18, 210);
    const ImU32 slotColor = IM_COL32(28, 32, 40, 230);
    const ImU32 borderColor = IM_COL32(110, 120, 136, 255);
    const ImU32 selectedBorderColor = IM_COL32(240, 245, 255, 255);
    const ImU32 textColor = IM_COL32(236, 241, 248, 255);
    const ImU32 emptyTextColor = IM_COL32(152, 160, 172, 255);

    drawList->AddRectFilled(
        ImVec2(layout.hotbarPanel.minX, layout.hotbarPanel.minY),
        ImVec2(layout.hotbarPanel.maxX, layout.hotbarPanel.maxY),
        panelColor,
        12.0f
    );
    for (std::size_t slotIndex = 0; slotIndex < kGameplayHotbarSlotCount; ++slotIndex) {
        const GameplayUiRect& slot = layout.hotbarSlots[slotIndex];
        const InventoryItemId itemId = m_gameplayUiState.hotbarItems[slotIndex];
        const bool selected = slotIndex == m_gameplayUiState.selectedHotbarSlot;
        drawList->AddRectFilled(
            ImVec2(slot.minX, slot.minY),
            ImVec2(slot.maxX, slot.maxY),
            slotColor,
            8.0f
        );
        drawList->AddRect(
            ImVec2(slot.minX, slot.minY),
            ImVec2(slot.maxX, slot.maxY),
            selected ? selectedBorderColor : borderColor,
            8.0f,
            0,
            selected ? 2.5f : 1.5f
        );
        const float centerX = (slot.minX + slot.maxX) * 0.5f;
        const float centerY = (slot.minY + slot.maxY) * 0.5f;
        drawList->AddCircleFilled(ImVec2(centerX, centerY - 6.0f), 11.0f, itemTint(itemId), 18);
        const std::string slotNumber = std::to_string(slotIndex + 1);
        drawList->AddText(ImVec2(slot.minX + 6.0f, slot.minY + 4.0f), emptyTextColor, slotNumber.c_str());
        const char* label = itemLabel(itemId);
        if (label[0] != '\0') {
            const ImVec2 textSize = ImGui::CalcTextSize(label);
            drawList->AddText(
                ImVec2(centerX - (textSize.x * 0.5f), slot.maxY - 18.0f),
                textColor,
                label
            );
        }
    }

    if (!m_gameplayUiState.inventoryVisible) {
        return;
    }

    drawList->AddRectFilled(
        ImVec2(0.0f, 0.0f),
        ImVec2(displaySize.x, displaySize.y),
        IM_COL32(4, 6, 10, 130)
    );
    drawList->AddRectFilled(
        ImVec2(layout.inventoryPanel.minX, layout.inventoryPanel.minY),
        ImVec2(layout.inventoryPanel.maxX, layout.inventoryPanel.maxY),
        IM_COL32(18, 22, 30, 235),
        14.0f
    );
    drawList->AddRect(
        ImVec2(layout.inventoryPanel.minX, layout.inventoryPanel.minY),
        ImVec2(layout.inventoryPanel.maxX, layout.inventoryPanel.maxY),
        borderColor,
        14.0f,
        0,
        2.0f
    );
    drawList->AddText(
        ImVec2(layout.inventoryPanel.minX + 24.0f, layout.inventoryPanel.minY + 20.0f),
        textColor,
        "Creative Inventory"
    );
    drawList->AddText(
        ImVec2(layout.inventoryPanel.minX + 24.0f, layout.inventoryPanel.minY + 44.0f),
        emptyTextColor,
        "Click an item to assign it to the selected hotbar slot."
    );
    for (std::size_t itemIndex = 0; itemIndex < kCreativeInventoryItemCount; ++itemIndex) {
        const GameplayUiRect& slot = layout.inventorySlots[itemIndex];
        const InventoryItemId itemId = m_gameplayUiState.creativeInventoryItems[itemIndex];
        const char* label = itemLabel(itemId);
        drawList->AddRectFilled(
            ImVec2(slot.minX, slot.minY),
            ImVec2(slot.maxX, slot.maxY),
            slotColor,
            10.0f
        );
        drawList->AddRect(
            ImVec2(slot.minX, slot.minY),
            ImVec2(slot.maxX, slot.maxY),
            borderColor,
            10.0f,
            0,
            1.5f
        );
        const float centerX = (slot.minX + slot.maxX) * 0.5f;
        const float centerY = (slot.minY + slot.maxY) * 0.5f;
        drawList->AddCircleFilled(ImVec2(centerX, centerY - 8.0f), 16.0f, itemTint(itemId), 20);
        const ImVec2 textSize = ImGui::CalcTextSize(label);
        drawList->AddText(
            ImVec2(centerX - (textSize.x * 0.5f), slot.maxY - 22.0f),
            textColor,
            label
        );
    }
}


} // namespace voxelsprout::render
