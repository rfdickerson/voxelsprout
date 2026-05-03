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

namespace odai::render {

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

constexpr int kPostLookPresetNeutral = 0;
constexpr int kPostLookPresetPunchy = 1;
constexpr int kPostLookPresetStylizedVivid = 2;

void applyPostLookPreset(RendererBackend::SkyDebugSettings& settings, int preset) {
    settings.postColorLookPreset = preset;
    switch (preset) {
    case kPostLookPresetNeutral:
        settings.autoExposureKeyValue = 0.18f;
        settings.autoExposureMin = 0.75f;
        settings.autoExposureMax = 1.60f;
        settings.autoExposureAdaptUp = 1.80f;
        settings.autoExposureAdaptDown = 0.80f;
        settings.autoExposureLowPercentile = 0.50f;
        settings.autoExposureHighPercentile = 0.92f;
        settings.bloomThreshold = 1.25f;
        settings.bloomSoftKnee = 0.20f;
        settings.bloomBaseIntensity = 0.030f;
        settings.bloomSunFacingBoost = 0.12f;
        settings.colorGradingWhiteBalanceR = 1.01f;
        settings.colorGradingWhiteBalanceG = 1.00f;
        settings.colorGradingWhiteBalanceB = 0.99f;
        settings.colorGradingContrast = 1.06f;
        settings.colorGradingSaturation = 1.02f;
        settings.colorGradingVibrance = 0.06f;
        settings.colorGradingMidtoneContrast = 1.04f;
        settings.colorGradingShadowDensity = 1.02f;
        settings.colorGradingHighlightRolloff = 0.96f;
        settings.colorGradingShadowTintR = 0.00f;
        settings.colorGradingShadowTintG = 0.00f;
        settings.colorGradingShadowTintB = 0.02f;
        settings.colorGradingHighlightTintR = 0.02f;
        settings.colorGradingHighlightTintG = 0.01f;
        settings.colorGradingHighlightTintB = 0.00f;
        break;
    case kPostLookPresetPunchy:
        settings.autoExposureKeyValue = 0.17f;
        settings.autoExposureMin = 0.72f;
        settings.autoExposureMax = 1.70f;
        settings.autoExposureAdaptUp = 1.70f;
        settings.autoExposureAdaptDown = 0.72f;
        settings.autoExposureLowPercentile = 0.52f;
        settings.autoExposureHighPercentile = 0.91f;
        settings.bloomThreshold = 1.20f;
        settings.bloomSoftKnee = 0.22f;
        settings.bloomBaseIntensity = 0.040f;
        settings.bloomSunFacingBoost = 0.15f;
        settings.colorGradingWhiteBalanceR = 1.02f;
        settings.colorGradingWhiteBalanceG = 1.00f;
        settings.colorGradingWhiteBalanceB = 0.98f;
        settings.colorGradingContrast = 1.11f;
        settings.colorGradingSaturation = 1.10f;
        settings.colorGradingVibrance = 0.16f;
        settings.colorGradingMidtoneContrast = 1.08f;
        settings.colorGradingShadowDensity = 1.05f;
        settings.colorGradingHighlightRolloff = 0.93f;
        settings.colorGradingShadowTintR = -0.01f;
        settings.colorGradingShadowTintG = 0.00f;
        settings.colorGradingShadowTintB = 0.04f;
        settings.colorGradingHighlightTintR = 0.04f;
        settings.colorGradingHighlightTintG = 0.02f;
        settings.colorGradingHighlightTintB = -0.01f;
        break;
    case kPostLookPresetStylizedVivid:
    default:
        settings.autoExposureKeyValue = 0.16f;
        settings.autoExposureMin = 0.70f;
        settings.autoExposureMax = 1.75f;
        settings.autoExposureAdaptUp = 1.60f;
        settings.autoExposureAdaptDown = 0.65f;
        settings.autoExposureLowPercentile = 0.55f;
        settings.autoExposureHighPercentile = 0.90f;
        settings.bloomThreshold = 1.15f;
        settings.bloomSoftKnee = 0.25f;
        settings.bloomBaseIntensity = 0.045f;
        settings.bloomSunFacingBoost = 0.18f;
        settings.colorGradingWhiteBalanceR = 1.03f;
        settings.colorGradingWhiteBalanceG = 1.00f;
        settings.colorGradingWhiteBalanceB = 0.97f;
        settings.colorGradingContrast = 1.14f;
        settings.colorGradingSaturation = 1.16f;
        settings.colorGradingVibrance = 0.24f;
        settings.colorGradingMidtoneContrast = 1.12f;
        settings.colorGradingShadowDensity = 1.08f;
        settings.colorGradingHighlightRolloff = 0.90f;
        settings.colorGradingShadowTintR = -0.01f;
        settings.colorGradingShadowTintG = 0.01f;
        settings.colorGradingShadowTintB = 0.06f;
        settings.colorGradingHighlightTintR = 0.06f;
        settings.colorGradingHighlightTintG = 0.03f;
        settings.colorGradingHighlightTintB = -0.01f;
        break;
    }
}

} // namespace

void RendererBackend::setDebugUiVisible(bool visible) {
    if (m_debugUiVisible == visible && m_showFrameStatsPanel == visible) {
        return;
    }
    m_debugUiVisible = visible;
    m_showFrameStatsPanel = visible;
}

void RendererBackend::setGameplayUiState(const GameplayUiState& state) {
    m_gameplayUiState = state;
}

GameplayUiCommand RendererBackend::consumeGameplayUiCommand() {
    GameplayUiCommand command = std::move(m_pendingGameplayUiCommand);
    m_pendingGameplayUiCommand = {};
    return command;
}


bool RendererBackend::isDebugUiVisible() const {
    return m_debugUiVisible && m_showFrameStatsPanel;
}


void RendererBackend::setFrameStatsVisible(bool visible) {
    setDebugUiVisible(visible);
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
    const bool hasVoxelRtMainGeometry =
        m_chunkVertexBufferHandle != kInvalidBufferHandle &&
        m_chunkIndexBufferHandle != kInvalidBufferHandle &&
        !m_chunkDrawRanges.empty();
    const bool hasMagicaRtMainGeometry = !m_magicaMeshDraws.empty();
    const bool hasImportedRtMainGeometry = !m_importedMeshDraws.empty();
    const bool mainPassPipelinesReady =
        m_rtMainPassImplemented &&
        ((m_pipelineRt != VK_NULL_HANDLE && hasVoxelRtMainGeometry) ||
         (m_magicaPipelineRt != VK_NULL_HANDLE && hasMagicaRtMainGeometry) ||
         (m_importedStaticPipelineRt != VK_NULL_HANDLE && hasImportedRtMainGeometry));
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
    m_skyDebugSettings.sunPitchDegrees = std::clamp(pitchDegrees, -89.0f, 89.0f);
}

void RendererBackend::setImportedSceneDebugState(bool showTerrain, bool showStatics, bool showTextures, bool flatShading, bool waterDebug) {
    m_debugShowImportedTerrain = showTerrain;
    m_debugShowImportedStatics = showStatics;
    m_debugShowImportedTextures = showTextures;
    m_debugImportedFlatShading = flatShading;
    m_debugImportedWaterSolid = waterDebug;
}

void RendererBackend::setImportedSceneInteriorMode(bool enabled) {
    m_importedSceneInteriorMode = enabled;
}

void RendererBackend::importedSceneDebugState(
    bool& outShowTerrain,
    bool& outShowStatics,
    bool& outShowTextures,
    bool& outFlatShading,
    bool& outWaterDebug
) const {
    outShowTerrain = m_debugShowImportedTerrain;
    outShowStatics = m_debugShowImportedStatics;
    outShowTextures = m_debugShowImportedTextures;
    outFlatShading = m_debugImportedFlatShading;
    outWaterDebug = m_debugImportedWaterSolid;
}


float RendererBackend::cameraFovDegrees() const {
    return m_debugCameraFovDegrees;
}


void RendererBackend::buildFrameStatsUi() {
    if (!m_debugUiVisible || !m_showFrameStatsPanel) {
        return;
    }

    constexpr ImGuiWindowFlags kPanelFlags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings;
    if (!ImGui::Begin("Morrowind Renderer", &m_showFrameStatsPanel, kPanelFlags)) {
        ImGui::End();
        return;
    }

    const bool importedSceneLoaded =
        !m_importedMeshDraws.empty() ||
        m_importedWaterIndexCount > 0 ||
        m_debugImportedActorInstanceCount > 0u;
    const float autoScale = std::numeric_limits<float>::max();
    if (ImGui::BeginTabBar("MorrowindRendererTabs")) {
    if (ImGui::BeginTabItem("Scene")) {
        ImGui::Text("Viewer: Morrowind scene");
        ImGui::SliderFloat("Camera FOV", &m_debugCameraFovDegrees, 55.0f, 120.0f, "%.1f deg");
        if (importedSceneLoaded) {
            ImGui::Checkbox("Show Terrain", &m_debugShowImportedTerrain);
            ImGui::Checkbox("Show Statics", &m_debugShowImportedStatics);
            ImGui::Checkbox("Show Textures", &m_debugShowImportedTextures);
            ImGui::Checkbox("Flat Static Shading", &m_debugImportedFlatShading);
            ImGui::Checkbox("Solid Water Debug", &m_debugImportedWaterSolid);
            ImGui::TextDisabled("Hotkeys: F5 terrain, F6 statics, K textures, F7 flat shading, F8 water debug");
            ImGui::Text(
                "Imported Draws / Water Indices: %u / %u",
                static_cast<unsigned>(m_importedMeshDraws.size()),
                m_importedWaterIndexCount
            );
            if (m_debugImportedPageRangeCount > 0u) {
                ImGui::Text(
                    "Imported Pages Visible: %u / %u",
                    m_debugImportedMainVisiblePageCount,
                    m_debugImportedPageRangeCount
                );
                ImGui::Text(
                    "Imported Draws Visible: %u / %u",
                    m_debugImportedMainVisibleDrawCount,
                    static_cast<unsigned>(m_importedMeshDraws.size())
                );
                ImGui::Text(
                    "Shadow Pages Visible: %u / %u / %u / %u",
                    m_debugImportedShadowVisiblePageCounts[0],
                    m_debugImportedShadowVisiblePageCounts[1],
                    m_debugImportedShadowVisiblePageCounts[2],
                    m_debugImportedShadowVisiblePageCounts[3]
                );
                ImGui::Text(
                    "Shadow Draws Visible: %u / %u / %u / %u",
                    m_debugImportedShadowVisibleDrawCounts[0],
                    m_debugImportedShadowVisibleDrawCounts[1],
                    m_debugImportedShadowVisibleDrawCounts[2],
                    m_debugImportedShadowVisibleDrawCounts[3]
                );
            }
        }
        if (m_debugImportedActorInstanceCount > 0u) {
            ImGui::Separator();
            ImGui::Text("Actor Debug");
            ImGui::Checkbox("GPU Skinning (palette)", &m_debugActorSkinningEnabled);
            ImGui::RadioButton("Bind/T-Pose", &m_debugActorPoseMode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Walk", &m_debugActorPoseMode, 1);
            ImGui::Text(
                "Actors / Bone Line Vertices: %u / %u",
                m_debugImportedActorInstanceCount,
                m_debugImportedActorBoneLineVertexCount);
        } else if (!importedSceneLoaded) {
            ImGui::TextDisabled("Imported-scene geometry not loaded.");
        }
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Pacing")) {
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
        if (m_supportsDisplayTiming) {
            ImGui::Checkbox("Use Display Timing", &m_enableDisplayTiming);
        } else {
            ImGui::TextDisabled("Display timing unavailable");
            m_enableDisplayTiming = false;
        }
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Lighting")) {
        if (ImGui::BeginTabBar("LightingTabs")) {
        if (ImGui::BeginTabItem("Shadows")) {
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
        ImGui::Text("RT Beta Scope: main-pass geometry + imported scene");
        ImGui::Text(
            "RT Scene Builds / BLAS / TLAS: %u / %u / %u",
            m_rtSceneBuildCount,
            m_rtBlasBuildCount,
            m_rtTlasBuildCount
        );
        ImGui::Text("Imported RT Geometries: %u", static_cast<unsigned>(m_rtImportedSceneRecords.size()));
        ImGui::Text(
            "Cascade Splits: %.1f / %.1f / %.1f / %.1f",
            m_shadowCascadeSplits[0],
            m_shadowCascadeSplits[1],
            m_shadowCascadeSplits[2],
            m_shadowCascadeSplits[3]
        );
        ImGui::Checkbox("Shadow Occluder Culling", &m_shadowDebugSettings.enableOccluderCulling);
        ImGui::SliderFloat("PCF Radius", &m_shadowDebugSettings.pcfRadius, 1.0f, 3.0f, "%.2f");
        ImGui::SliderInt("RT Samples", &m_shadowDebugSettings.rtShadowSampleCount, 1, 8);
        ImGui::SliderFloat(
            "RT Sun Radius (deg)",
            &m_shadowDebugSettings.rtSunAngularRadiusDegrees,
            0.0f,
            1.0f,
            "%.2f"
        );
        ImGui::SliderFloat("Cascade Blend Min", &m_shadowDebugSettings.cascadeBlendMin, 1.0f, 20.0f, "%.2f");
        ImGui::SliderFloat("Cascade Blend Factor", &m_shadowDebugSettings.cascadeBlendFactor, 0.05f, 0.60f, "%.2f");
        ImGui::SliderInt(
            "Grass Shadow Cascades",
            &m_shadowDebugSettings.grassShadowCascadeCount,
            0,
            static_cast<int>(kShadowCascadeCount)
        );
        if (ImGui::TreeNodeEx("Advanced Shadow Bias", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Receiver Bias");
            ImGui::SliderFloat(
                "Normal Offset Near",
                &m_shadowDebugSettings.receiverNormalOffsetNear,
                0.0f,
                0.20f,
                "%.3f"
            );
            ImGui::SliderFloat(
                "Normal Offset Far",
                &m_shadowDebugSettings.receiverNormalOffsetFar,
                0.0f,
                0.35f,
                "%.3f"
            );
            ImGui::SliderFloat(
                "Base Bias Near (texel)",
                &m_shadowDebugSettings.receiverBaseBiasNearTexel,
                0.0f,
                12.0f,
                "%.2f"
            );
            ImGui::SliderFloat(
                "Base Bias Far (texel)",
                &m_shadowDebugSettings.receiverBaseBiasFarTexel,
                0.0f,
                16.0f,
                "%.2f"
            );
            ImGui::SliderFloat(
                "Slope Bias Near (texel)",
                &m_shadowDebugSettings.receiverSlopeBiasNearTexel,
                0.0f,
                14.0f,
                "%.2f"
            );
            ImGui::SliderFloat(
                "Slope Bias Far (texel)",
                &m_shadowDebugSettings.receiverSlopeBiasFarTexel,
                0.0f,
                18.0f,
                "%.2f"
            );
            ImGui::Separator();
            ImGui::Text("Caster Bias");
            ImGui::SliderFloat(
                "Const Bias Base",
                &m_shadowDebugSettings.casterConstantBiasBase,
                0.0f,
                6.0f,
                "%.2f"
            );
            ImGui::SliderFloat(
                "Const Bias Cascade Scale",
                &m_shadowDebugSettings.casterConstantBiasCascadeScale,
                0.0f,
                3.0f,
                "%.2f"
            );
            ImGui::SliderFloat(
                "Slope Bias Base",
                &m_shadowDebugSettings.casterSlopeBiasBase,
                0.0f,
                8.0f,
                "%.2f"
            );
            ImGui::SliderFloat(
                "Slope Bias Cascade Scale",
                &m_shadowDebugSettings.casterSlopeBiasCascadeScale,
                0.0f,
                4.0f,
                "%.2f"
            );
            ImGui::TreePop();
        }
        if (ImGui::Button("Reset Shadow Defaults")) {
            m_shadowDebugSettings = ShadowDebugSettings{};
        }
        ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("AO + GI")) {
        ImGui::Checkbox("Enable SSAO", &m_debugEnableSsao);
        ImGui::SliderFloat("SSAO Radius", &m_shadowDebugSettings.ssaoRadius, 1.0f, 96.0f, "%.1f");
        ImGui::SliderFloat("SSAO Bias", &m_shadowDebugSettings.ssaoBias, 0.0f, 6.0f, "%.2f");
        ImGui::SliderFloat("SSAO Intensity", &m_shadowDebugSettings.ssaoIntensity, 0.0f, 2.0f, "%.2f");
        if (ImGui::TreeNodeEx("Advanced AO Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Visualize SSAO", &m_debugVisualizeSsao);
            ImGui::Checkbox("Visualize AO Normals", &m_debugVisualizeAoNormals);
            ImGui::TreePop();
        }

        ImGui::Separator();
        ImGui::Text("Scene GI");
        ImGui::Text("Compute: %s", m_voxelGiComputeAvailable ? "on" : "fallback");
        int giSurfaceMode = static_cast<int>(m_voxelGiDebugSettings.surfaceMode);
        ImGui::Combo("GI Surface Mode", &giSurfaceMode, "Legacy\0RT Surface\0ReSTIR Surface\0");
        m_voxelGiDebugSettings.surfaceMode = static_cast<VoxelGiSurfaceMode>(giSurfaceMode);
        ImGui::Text(
            "GI Active: %s",
            m_voxelGiRestirActiveThisFrame
                ? "ReSTIR Surface"
                : (m_voxelGiRtSurfaceActiveThisFrame ? "RT Surface" : "Legacy")
        );
        ImGui::Text(
            "GI Readiness: RT=%s ReSTIR=%s TLAS=%s",
            m_voxelGiRtSurfaceReady ? "yes" : "no",
            m_voxelGiRestirReady ? "yes" : "no",
            m_rtTlas.handle != VK_NULL_HANDLE ? "yes" : "no"
        );
        if (m_importedSceneInteriorMode) {
            ImGui::Text(
                "Imported GI: %u tris / %u voxels / %u lights",
                static_cast<unsigned>(m_debugImportedGiTriangleCount),
                static_cast<unsigned>(m_debugImportedGiVoxelizedCellCount),
                static_cast<unsigned>(m_debugImportedLightSelectedCount)
            );
        }
        ImGui::SliderFloat("Bounce Strength", &m_voxelGiDebugSettings.bounceStrength, 0.0f, 2.50f, "%.2f");
        ImGui::SliderFloat("Diffusion Softness", &m_voxelGiDebugSettings.diffusionSoftness, 0.0f, 1.0f, "%.2f");
        ImGui::SliderInt("RT Surface Samples", &m_voxelGiDebugSettings.rtSurfaceSampleCount, 1, 2);
        ImGui::SliderFloat("RT Surface Bias", &m_voxelGiDebugSettings.rtSurfaceBiasScale, 0.25f, 4.0f, "%.2f");
        if (importedSceneLoaded) {
            ImGui::SeparatorText("Morrowind GI");
            ImGui::SliderFloat("MW GI Strength", &m_voxelGiDebugSettings.morrowindGiStrength, 0.0f, 0.60f, "%.2f");
            ImGui::SliderFloat("MW GI Radius", &m_voxelGiDebugSettings.morrowindGiRadiusScale, 0.50f, 4.0f, "%.2fx");
            ImGui::SliderFloat("MW GI Occlusion Floor", &m_voxelGiDebugSettings.morrowindGiOcclusionFloor, 0.0f, 0.75f, "%.2f");
            ImGui::SliderFloat("MW GI Color Bleed", &m_voxelGiDebugSettings.morrowindGiColorBleed, 0.0f, 1.0f, "%.2f");
        }
        ImGui::SeparatorText("ReSTIR GI");
        ImGui::SliderInt("ReSTIR Candidates", &m_voxelGiDebugSettings.restirCandidateCount, 1, 8);
        ImGui::Checkbox("ReSTIR Temporal", &m_voxelGiDebugSettings.restirEnableTemporalReuse);
        ImGui::Checkbox("ReSTIR Spatial", &m_voxelGiDebugSettings.restirEnableSpatialReuse);
        ImGui::SliderInt("ReSTIR Radius", &m_voxelGiDebugSettings.restirSpatialRadius, 1, 2);
        ImGui::Text(
            "History: %s (%s)",
            m_voxelGiRestirHistoryValid ? "valid" : "reset",
            m_voxelGiRestirHistoryResetReason.c_str()
        );
        if (ImGui::Button("Reset ReSTIR History")) {
            m_voxelGiDebugSettings.restirHistoryResetRequested = true;
        }
        if (ImGui::TreeNodeEx("Advanced GI Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* giVisualizationModes =
                "Off\0Radiance\0False Color Luma\0Radiance (Gray)\0Occupancy Albedo\0Imported Lights\0";
            ImGui::Combo("GI Visualize", &m_voxelGiDebugSettings.visualizationMode, giVisualizationModes);
            if (m_voxelGiDebugSettings.visualizationMode > 0) {
                m_debugVisualizeSsao = false;
                m_debugVisualizeAoNormals = false;
            }
            ImGui::TreePop();
        }
        if (ImGui::Button("Reset GI Defaults")) {
            m_voxelGiDebugSettings = VoxelGiDebugSettings{};
        }
        ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Imported Lights")) {
            ImGui::Checkbox("Enable Imported Lights", &m_debugImportedLightsEnabled);
            ImGui::SliderFloat("Imported Light Intensity", &m_debugImportedLightIntensity, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Outdoor Lamp Strength", &m_debugImportedOutdoorLightStrength, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Imported Light Radius", &m_debugImportedLightRadiusScale, 0.25f, 8.0f, "%.2fx");
            ImGui::Text(
                "Imported Lights: %u total / %u selected",
                static_cast<unsigned>(m_importedLocalLights.size()),
                static_cast<unsigned>(m_debugImportedLightSelectedCount)
            );
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
        }
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Art Direction")) {
        if (ImGui::BeginTabBar("ArtDirectionTabs")) {
        if (ImGui::BeginTabItem("Sun/Sky")) {
        ImGui::SliderFloat("Sun Yaw", &m_skyDebugSettings.sunYawDegrees, -180.0f, 180.0f, "%.1f deg");
        ImGui::SliderFloat("Sun Pitch", &m_skyDebugSettings.sunPitchDegrees, -89.0f, 89.0f, "%.1f deg");
        ImGui::SliderFloat("Sky Exposure", &m_skyDebugSettings.skyExposure, 0.25f, 3.0f, "%.2f");
        ImGui::SliderFloat("Sun Disk Intensity", &m_skyDebugSettings.sunDiskIntensity, 300.0f, 2200.0f, "%.0f");
        ImGui::SliderFloat("Sun Halo Intensity", &m_skyDebugSettings.sunHaloIntensity, 4.0f, 64.0f, "%.1f");
        if (ImGui::TreeNodeEx("Advanced Atmosphere", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Auto Sunrise Tuning", &m_skyDebugSettings.autoSunriseTuning);
            ImGui::SliderFloat("Auto Sunrise Blend", &m_skyDebugSettings.autoSunriseBlend, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Auto Adapt Speed", &m_skyDebugSettings.autoSunriseAdaptSpeed, 0.5f, 12.0f, "%.2f");
            ImGui::Separator();
            ImGui::SliderFloat("Rayleigh Strength", &m_skyDebugSettings.rayleighStrength, 0.1f, 4.0f, "%.2f");
            ImGui::SliderFloat("Mie Strength", &m_skyDebugSettings.mieStrength, 0.05f, 4.0f, "%.2f");
            ImGui::SliderFloat("Mie Anisotropy", &m_skyDebugSettings.mieAnisotropy, 0.0f, 0.95f, "%.2f");
            ImGui::SliderFloat("Sun Disk Size", &m_skyDebugSettings.sunDiskSize, 0.5f, 6.0f, "%.2f");
            ImGui::SliderFloat("Sun Haze Falloff", &m_skyDebugSettings.sunHazeFalloff, 0.10f, 1.20f, "%.2f");
            ImGui::TreePop();
        }
        ImGui::Text(
            "Runtime: Rayleigh %.2f, Mie %.2f, Exposure %.2f, Disk %.2f",
            m_skyTuningRuntime.rayleighStrength,
            m_skyTuningRuntime.mieStrength,
            m_skyTuningRuntime.skyExposure,
            m_skyTuningRuntime.sunDiskSize
        );
        ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Water/Fog")) {
        ImGui::Checkbox("Enable Fog", &m_skyDebugSettings.volumetricFogEnabled);
        ImGui::SliderFloat("Fog Density", &m_skyDebugSettings.volumetricFogDensity, 0.0f, 0.03f, "%.4f");
        ImGui::SliderFloat("Fog Sun Scatter", &m_skyDebugSettings.volumetricSunScattering, 0.0f, 3.0f, "%.2f");
        if (ImGui::TreeNodeEx("Advanced Fog", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat(
                "Fog Height Falloff",
                &m_skyDebugSettings.volumetricFogHeightFalloff,
                0.0f,
                0.30f,
                "%.4f"
            );
            ImGui::SliderFloat(
                "Fog Base Height",
                &m_skyDebugSettings.volumetricFogBaseHeight,
                -32.0f,
                320.0f,
                "%.1f"
            );
            ImGui::TreePop();
        }
        ImGui::SeparatorText("Water");
        ImGui::SliderFloat("Water Speed", &m_skyDebugSettings.waterAnimationSpeed, 0.25f, 4.0f, "%.2f");
        ImGui::SliderFloat("Water Normal Strength", &m_skyDebugSettings.waterNormalStrength, 0.25f, 2.5f, "%.2f");
        ImGui::SliderFloat("Water Reflection", &m_skyDebugSettings.waterReflectionStrength, 0.25f, 4.0f, "%.2f");
        ImGui::SliderFloat("Water Absorption", &m_skyDebugSettings.waterRefractionDecay, 0.25f, 5.0f, "%.2f");
        if (ImGui::TreeNodeEx("Advanced Water", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Water Refraction Strength", &m_skyDebugSettings.waterRefractionStrength, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat(
                "Water Refraction Distortion",
                &m_skyDebugSettings.waterRefractionDistortionPixels,
                0.0f,
                160.0f,
                "%.0f px"
            );
            ImGui::TreePop();
        }
        ImGui::SliderFloat(
            "Plant Quad Directionality",
            &m_skyDebugSettings.plantQuadDirectionality,
            0.0f,
            1.0f,
            "%.2f"
        );
        ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Post")) {
        ImGui::Text("Depth of Field");
        ImGui::Checkbox("Enable DoF", &m_skyDebugSettings.depthOfFieldEnabled);
        ImGui::SliderFloat("DoF Intensity", &m_skyDebugSettings.depthOfFieldMaxRadiusPixels, 0.0f, 14.0f, "%.1f");
        ImGui::SliderFloat("DoF Target Distance", &m_skyDebugSettings.depthOfFieldFocusDistance, 1.0f, 240.0f, "%.1f");
        ImGui::SliderFloat("Focus Range", &m_skyDebugSettings.depthOfFieldFocusRange, 1.0f, 120.0f, "%.1f");
        ImGui::Separator();
        ImGui::Text("Eye Adaptation");
        ImGui::Checkbox("Auto Exposure", &m_skyDebugSettings.autoExposureEnabled);
        ImGui::SliderFloat("Manual Exposure", &m_skyDebugSettings.manualExposure, 0.05f, 4.0f, "%.3f");
        ImGui::Text(
            "Resolved Exposure: %.3f (target %.3f, avg luma %.3f)",
            m_debugResolvedExposure,
            m_debugTargetExposure,
            m_debugAverageSceneLuminance
        );
        if (m_skyDebugSettings.autoExposureEnabled && ImGui::TreeNodeEx("Advanced Exposure", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderInt("AE Update Interval", &m_skyDebugSettings.autoExposureUpdateIntervalFrames, 1, 16);
            ImGui::SliderFloat("AE Key Value", &m_skyDebugSettings.autoExposureKeyValue, 0.05f, 0.50f, "%.3f");
            ImGui::SliderFloat("AE Min Exposure", &m_skyDebugSettings.autoExposureMin, 0.05f, 2.50f, "%.3f");
            ImGui::SliderFloat("AE Max Exposure", &m_skyDebugSettings.autoExposureMax, 0.20f, 12.00f, "%.3f");
            ImGui::SliderFloat("AE Adapt Up", &m_skyDebugSettings.autoExposureAdaptUp, 0.10f, 12.00f, "%.2f");
            ImGui::SliderFloat("AE Adapt Down", &m_skyDebugSettings.autoExposureAdaptDown, 0.10f, 12.00f, "%.2f");
            ImGui::SliderFloat("AE Low Percentile", &m_skyDebugSettings.autoExposureLowPercentile, 0.00f, 0.95f, "%.2f");
            ImGui::SliderFloat("AE High Percentile", &m_skyDebugSettings.autoExposureHighPercentile, 0.05f, 1.00f, "%.2f");
            ImGui::TreePop();
        }
        if (!m_autoExposureComputeAvailable) {
            ImGui::TextDisabled("Auto exposure compute unavailable; manual exposure is active.");
        }

        ImGui::SeparatorText("Bloom");
        ImGui::SliderFloat("Bloom Global Intensity", &m_skyDebugSettings.bloomBaseIntensity, 0.0f, 0.35f, "%.3f");
        if (ImGui::TreeNodeEx("Advanced Bloom", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Bloom Threshold", &m_skyDebugSettings.bloomThreshold, 0.25f, 4.0f, "%.2f");
            ImGui::SliderFloat("Bloom Soft Knee", &m_skyDebugSettings.bloomSoftKnee, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Bloom Sun Boost", &m_skyDebugSettings.bloomSunFacingBoost, 0.0f, 0.40f, "%.3f");
            ImGui::TreePop();
        }

        ImGui::SeparatorText("Color Grading");
        const char* postLookPresets = "Neutral\0Punchy\0Stylized Vivid\0";
        if (ImGui::Combo("Post Look", &m_skyDebugSettings.postColorLookPreset, postLookPresets)) {
            applyPostLookPreset(m_skyDebugSettings, m_skyDebugSettings.postColorLookPreset);
        }
        ImGui::SliderFloat("Contrast", &m_skyDebugSettings.colorGradingContrast, 0.70f, 1.40f, "%.2f");
        ImGui::SliderFloat("Saturation", &m_skyDebugSettings.colorGradingSaturation, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Vibrance", &m_skyDebugSettings.colorGradingVibrance, -1.0f, 1.0f, "%.2f");
        if (ImGui::TreeNodeEx("Advanced Color Grading", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Midtone Contrast", &m_skyDebugSettings.colorGradingMidtoneContrast, 0.80f, 1.40f, "%.2f");
            ImGui::SliderFloat("Shadow Density", &m_skyDebugSettings.colorGradingShadowDensity, 0.70f, 1.40f, "%.2f");
            ImGui::SliderFloat("Highlight Rolloff", &m_skyDebugSettings.colorGradingHighlightRolloff, 0.70f, 1.10f, "%.2f");
            ImGui::SliderFloat("White Balance R", &m_skyDebugSettings.colorGradingWhiteBalanceR, 0.80f, 1.20f, "%.2f");
            ImGui::SliderFloat("White Balance G", &m_skyDebugSettings.colorGradingWhiteBalanceG, 0.80f, 1.20f, "%.2f");
            ImGui::SliderFloat("White Balance B", &m_skyDebugSettings.colorGradingWhiteBalanceB, 0.80f, 1.20f, "%.2f");
            ImGui::SliderFloat("Shadow Tint R", &m_skyDebugSettings.colorGradingShadowTintR, -0.20f, 0.20f, "%.2f");
            ImGui::SliderFloat("Shadow Tint G", &m_skyDebugSettings.colorGradingShadowTintG, -0.20f, 0.20f, "%.2f");
            ImGui::SliderFloat("Shadow Tint B", &m_skyDebugSettings.colorGradingShadowTintB, -0.20f, 0.20f, "%.2f");
            ImGui::SliderFloat("Highlight Tint R", &m_skyDebugSettings.colorGradingHighlightTintR, -0.20f, 0.20f, "%.2f");
            ImGui::SliderFloat("Highlight Tint G", &m_skyDebugSettings.colorGradingHighlightTintG, -0.20f, 0.20f, "%.2f");
            ImGui::SliderFloat("Highlight Tint B", &m_skyDebugSettings.colorGradingHighlightTintB, -0.20f, 0.20f, "%.2f");
            ImGui::TreePop();
        }
        ImGui::EndTabItem();
        }

        if (ImGui::Button("Reset Sun/Sky Defaults")) {
            m_skyDebugSettings = SkyDebugSettings{};
            m_skyTuningRuntime = RendererBackend::SkyTuningRuntimeState{};
        }
        ImGui::EndTabBar();
        }
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Performance")) {
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
        ImGui::Text(
            "Frame CPU (total/work/ewma): %.2f / %.2f / %.2f ms",
            m_debugFrameTimeMs,
            m_debugCpuFrameWorkMs,
            m_debugCpuFrameEwmaMs
        );
        ImGui::Text(
            "Frame CPU P50/P95/P99: %.2f / %.2f / %.2f ms",
            m_debugCpuFrameP50Ms,
            m_debugCpuFrameP95Ms,
            m_debugCpuFrameP99Ms
        );
        if (m_gpuTimestampsSupported) {
            ImGui::Text("Frame GPU: %.2f ms", m_debugGpuFrameTimeMs);
            ImGui::Text(
                "Frame GPU P50/P95/P99: %.2f / %.2f / %.2f ms",
                m_debugGpuFrameP50Ms,
                m_debugGpuFrameP95Ms,
                m_debugGpuFrameP99Ms
            );
        } else {
            ImGui::Text("Frame GPU: n/a");
        }
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
        if (ImGui::TreeNodeEx("Draw Calls", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Total: %u", m_debugDrawCallsTotal);
            ImGui::Text("Shadow: %u", m_debugDrawCallsShadow);
            ImGui::Text("Prepass: %u", m_debugDrawCallsPrepass);
            ImGui::Text("Main: %u", m_debugDrawCallsMain);
            ImGui::Text("Post: %u", m_debugDrawCallsPost);
            ImGui::TreePop();
        }
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
        if (hasFrameArenaMetrics && ImGui::TreeNodeEx("FrameArena", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (m_debugFrameArenaUploadBytes > 0 || m_debugFrameArenaUploadAllocs > 0) {
                ImGui::Text(
                    "Upload this frame: %llu B (%u allocs)",
                    static_cast<unsigned long long>(m_debugFrameArenaUploadBytes),
                    m_debugFrameArenaUploadAllocs
                );
            }
            ImGui::Text(
                "Image alias reuses (frame/live): %u / %u",
                m_debugFrameArenaAliasReuses,
                m_debugFrameArenaResidentAliasReuses
            );
            ImGui::Text("Resident images (live): %u", m_debugFrameArenaResidentImageCount);
            ImGui::TreePop();
        }
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Diagnostics")) {
        const char* waterDebugViews =
            "Final\0Normals\0Depth/Thickness\0Scene Refraction\0Refraction UV Offset\0Reflection\0Fresnel\0";
        ImGui::Combo("Water Debug View", &m_skyDebugSettings.waterDebugMode, waterDebugViews);
        if (ImGui::TreeNodeEx("Meshing", ImGuiTreeNodeFlags_DefaultOpen)) {
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
            int meshingModeSelection = (m_chunkMeshingOptions.mode == odai::world::MeshingMode::Greedy) ? 1 : 0;
            if (ImGui::Combo("Chunk Meshing", &meshingModeSelection, "Naive\0Greedy\0")) {
                const odai::world::MeshingMode nextMode =
                    (meshingModeSelection == 1) ? odai::world::MeshingMode::Greedy : odai::world::MeshingMode::Naive;
                if (nextMode != m_chunkMeshingOptions.mode) {
                    m_chunkMeshingOptions.mode = nextMode;
                    m_chunkLodMeshCacheValid = false;
                    m_chunkMeshRebuildRequested = true;
                    m_pendingChunkRemeshKeys.clear();
                    VOX_LOGI("render") << "chunk meshing mode changed to "
                                       << (nextMode == odai::world::MeshingMode::Greedy ? "Greedy" : "Naive")
                                       << ", scheduling full remesh";
                }
            }
            ImGui::Text("Chunk Mesh Vert/Idx: %u / %u", m_debugChunkMeshVertexCount, m_debugChunkMeshIndexCount);
            ImGui::Text("Last Chunk Remesh: %.2f ms (%u)", m_debugChunkLastRemeshMs, m_debugChunkLastRemeshedChunkCount);
            ImGui::Text("Chunk Remesh Pending/Batch: %u / %u", m_debugChunkPendingRemeshCount, m_debugChunkRemeshBatchCount);
            ImGui::Text("RT Active Chunks: %u", m_debugRtActiveChunkCount);
            ImGui::TreePop();
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
            ImGui::Text("Storage Image Limit: %u", m_desktopCapabilityProbe.maxDescriptorSetStorageImages);
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
            ImGui::Text("RT Pipeline: %s", m_rayTracingCapabilityProbe.rayTracingPipelineExtension ? "yes" : "no");
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
            ImGui::Text("Imported RT Geometries: %u", static_cast<unsigned>(m_rtImportedSceneRecords.size()));
            ImGui::Text("Magica RT Geometries: %u", static_cast<unsigned>(m_rtMagicaGeometries.size()));
            ImGui::TreePop();
        }
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
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

    int meshingModeSelection = (m_chunkMeshingOptions.mode == odai::world::MeshingMode::Greedy) ? 1 : 0;
    if (ImGui::Combo("Chunk Meshing", &meshingModeSelection, "Naive\0Greedy\0")) {
        const odai::world::MeshingMode nextMode =
            (meshingModeSelection == 1) ? odai::world::MeshingMode::Greedy : odai::world::MeshingMode::Naive;
        if (nextMode != m_chunkMeshingOptions.mode) {
            m_chunkMeshingOptions.mode = nextMode;
            m_chunkLodMeshCacheValid = false;
            m_chunkMeshRebuildRequested = true;
            m_pendingChunkRemeshKeys.clear();
            VOX_LOGI("render") << "chunk meshing mode changed to "
                               << (nextMode == odai::world::MeshingMode::Greedy ? "Greedy" : "Naive")
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
    ImGui::Text("Chunk Remesh Pending/Batch: %u / %u", m_debugChunkPendingRemeshCount, m_debugChunkRemeshBatchCount);
    ImGui::Text("RT Active Chunks: %u", m_debugRtActiveChunkCount);
    ImGui::Text("Greedy Reduction vs Naive: %.1f%%", m_debugChunkLastRemeshReductionPercent);
    ImGui::End();
}


void RendererBackend::buildAimReticleUi() {
    if (m_gameplayUiState.inventoryVisible || m_gameplayUiState.dialogueVisible || m_debugUiVisible) {
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

void RendererBackend::buildDialogueUi() {
    if (!m_gameplayUiState.dialogueVisible) {
        return;
    }

    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    const float panelWidth = std::clamp(displaySize.x * 0.72f, 720.0f, 1040.0f);
    const float panelHeight = std::clamp(displaySize.y * 0.58f, 420.0f, 620.0f);
    const ImVec2 panelPos{
        (displaySize.x - panelWidth) * 0.5f,
        displaySize.y - panelHeight - 42.0f
    };

    ImGui::SetNextWindowPos(panelPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    constexpr ImGuiWindowFlags kFlags =
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoSavedSettings;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.13f, 0.10f, 0.07f, 0.96f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.72f, 0.58f, 0.34f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.92f, 0.84f, 0.66f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.18f, 0.10f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.42f, 0.31f, 0.16f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.56f, 0.42f, 0.22f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 4.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    if (!ImGui::Begin("Dialogue", nullptr, kFlags)) {
        ImGui::End();
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(6);
        return;
    }

    ImGui::TextUnformatted(m_gameplayUiState.dialogueActorName.c_str());
    ImGui::SameLine();
    ImGui::SetCursorPosX(panelWidth - 88.0f);
    if (ImGui::Button("Close", ImVec2(72.0f, 0.0f))) {
        m_pendingGameplayUiCommand = {GameplayUiCommandType::CloseDialogue, {}};
    }
    ImGui::Separator();

    const float footerHeight = 116.0f;
    const float topicWidth = 240.0f;
    const float contentHeight = panelHeight - footerHeight - 72.0f;
    ImGui::BeginChild("DialogueText", ImVec2(panelWidth - topicWidth - 34.0f, contentHeight), true);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(m_gameplayUiState.dialogueText.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("DialogueTopics", ImVec2(topicWidth, contentHeight), true);
    ImGui::TextUnformatted("Topics");
    ImGui::Separator();
    for (const auto& topic : m_gameplayUiState.dialogueTopics) {
        const bool selected = topic.first == m_gameplayUiState.dialogueSelectedTopicId;
        if (selected) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.88f, 0.50f, 1.0f));
        }
        if (ImGui::Selectable(topic.second.c_str(), selected)) {
            m_pendingGameplayUiCommand = {GameplayUiCommandType::SelectDialogueTopic, topic.first};
        }
        if (selected) {
            ImGui::PopStyleColor();
        }
    }
    ImGui::EndChild();

    ImGui::Separator();
    if (!m_gameplayUiState.dialogueLastMessage.empty()) {
        ImGui::TextWrapped("%s", m_gameplayUiState.dialogueLastMessage.c_str());
    }
    ImGui::TextDisabled("%s", m_gameplayUiState.dialogueJournalSummary.c_str());

    if (!m_gameplayUiState.dialogueChoices.empty()) {
        ImGui::Separator();
        for (const auto& choice : m_gameplayUiState.dialogueChoices) {
            if (ImGui::Button(choice.second.c_str(), ImVec2(0.0f, 0.0f))) {
                m_pendingGameplayUiCommand = {GameplayUiCommandType::SelectDialogueChoice, choice.first};
            }
            ImGui::SameLine();
        }
        ImGui::NewLine();
    }

    ImGui::End();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(6);
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
    const ImU32 slotColor = IM_COL32(28, 32, 40, 230);
    const ImU32 borderColor = IM_COL32(110, 120, 136, 255);
    const ImU32 textColor = IM_COL32(236, 241, 248, 255);
    const ImU32 emptyTextColor = IM_COL32(152, 160, 172, 255);

    auto drawBar = [&](float x, float y, float width, const char* label, int value, int maxValue, ImU32 fillColor) {
        const float clampedMax = static_cast<float>(std::max(maxValue, 1));
        const float fraction = std::clamp(static_cast<float>(value) / clampedMax, 0.0f, 1.0f);
        drawList->AddRectFilled(ImVec2(x, y), ImVec2(x + width, y + 14.0f), IM_COL32(18, 20, 24, 230), 3.0f);
        drawList->AddRectFilled(ImVec2(x, y), ImVec2(x + (width * fraction), y + 14.0f), fillColor, 3.0f);
        drawList->AddRect(ImVec2(x, y), ImVec2(x + width, y + 14.0f), IM_COL32(20, 20, 20, 220), 3.0f);
        const std::string text = std::string(label) + " " + std::to_string(std::max(value, 0)) + "/" + std::to_string(std::max(maxValue, 0));
        drawList->AddText(ImVec2(x + 6.0f, y - 1.0f), IM_COL32(245, 245, 236, 245), text.c_str());
    };

    const float hudX = 22.0f;
    const float hudY = displaySize.y - 172.0f;
    const float hudWidth = 282.0f;
    drawList->AddRectFilled(ImVec2(hudX - 12.0f, hudY - 14.0f), ImVec2(hudX + hudWidth + 12.0f, hudY + 94.0f), IM_COL32(12, 14, 16, 190), 5.0f);
    drawBar(hudX, hudY, hudWidth, "Health", m_gameplayUiState.health, m_gameplayUiState.maxHealth, IM_COL32(172, 48, 42, 245));
    drawBar(hudX, hudY + 24.0f, hudWidth, "Magicka", m_gameplayUiState.magicka, m_gameplayUiState.maxMagicka, IM_COL32(60, 92, 190, 245));
    drawBar(hudX, hudY + 48.0f, hudWidth, "Fatigue", m_gameplayUiState.fatigue, m_gameplayUiState.maxFatigue, IM_COL32(58, 142, 72, 245));
    const std::string goldText = "Gold " + std::to_string(m_gameplayUiState.gold);
    drawList->AddText(ImVec2(hudX, hudY + 72.0f), IM_COL32(236, 205, 116, 245), goldText.c_str());

    if (!m_gameplayUiState.trackedQuestText.empty() && !m_gameplayUiState.dialogueVisible) {
        const float questWidth = std::clamp(displaySize.x * 0.42f, 360.0f, 640.0f);
        const ImVec2 questPos{displaySize.x - questWidth - 22.0f, 24.0f};
        drawList->AddRectFilled(questPos, ImVec2(questPos.x + questWidth, questPos.y + 52.0f), IM_COL32(14, 16, 18, 180), 5.0f);
        drawList->AddText(ImVec2(questPos.x + 14.0f, questPos.y + 8.0f), IM_COL32(238, 221, 172, 245), "Tracked Quest");
        drawList->AddText(ImVec2(questPos.x + 14.0f, questPos.y + 28.0f), IM_COL32(232, 236, 230, 245), m_gameplayUiState.trackedQuestText.c_str());
    }

    if (m_gameplayUiState.playerDead) {
        const char* message = "You are down. Press R to recover.";
        const ImVec2 textSize = ImGui::CalcTextSize(message);
        const ImVec2 boxMin{(displaySize.x - textSize.x) * 0.5f - 24.0f, displaySize.y * 0.28f};
        const ImVec2 boxMax{boxMin.x + textSize.x + 48.0f, boxMin.y + 52.0f};
        drawList->AddRectFilled(boxMin, boxMax, IM_COL32(40, 8, 8, 220), 5.0f);
        drawList->AddText(ImVec2(boxMin.x + 24.0f, boxMin.y + 18.0f), IM_COL32(255, 230, 220, 255), message);
    }

    for (std::size_t slotIndex = 0; slotIndex < layout.hotbarSlots.size(); ++slotIndex) {
        const GameplayUiRect& slot = layout.hotbarSlots[slotIndex];
        const InventoryItemId itemId = m_gameplayUiState.hotbarItems[slotIndex];
        const bool selected = slotIndex == m_gameplayUiState.selectedHotbarSlot;
        drawList->AddRectFilled(
            ImVec2(slot.minX, slot.minY),
            ImVec2(slot.maxX, slot.maxY),
            slotColor,
            6.0f
        );
        drawList->AddRect(
            ImVec2(slot.minX, slot.minY),
            ImVec2(slot.maxX, slot.maxY),
            selected ? IM_COL32(240, 206, 106, 255) : borderColor,
            6.0f,
            0,
            selected ? 2.4f : 1.4f
        );
        const float centerX = (slot.minX + slot.maxX) * 0.5f;
        const float centerY = (slot.minY + slot.maxY) * 0.5f;
        drawList->AddCircleFilled(ImVec2(centerX, centerY - 7.0f), 13.0f, itemTint(itemId), 20);
        const char* label = itemLabel(itemId);
        const ImVec2 textSize = ImGui::CalcTextSize(label);
        drawList->AddText(
            ImVec2(centerX - (textSize.x * 0.5f), slot.maxY - 18.0f),
            textColor,
            label
        );
    }

    if (!m_gameplayUiState.inventoryVisible) {
        return;
    }

    drawList->AddRectFilled(ImVec2(0.0f, 0.0f), ImVec2(displaySize.x, displaySize.y), IM_COL32(4, 6, 10, 130));
    drawList->AddRectFilled(
        ImVec2(layout.inventoryPanel.minX, layout.inventoryPanel.minY),
        ImVec2(layout.inventoryPanel.maxX + 360.0f, layout.inventoryPanel.maxY + 180.0f),
        IM_COL32(18, 22, 30, 235),
        8.0f
    );
    drawList->AddRect(
        ImVec2(layout.inventoryPanel.minX, layout.inventoryPanel.minY),
        ImVec2(layout.inventoryPanel.maxX + 360.0f, layout.inventoryPanel.maxY + 180.0f),
        borderColor,
        8.0f,
        0,
        2.0f
    );
    drawList->AddText(ImVec2(layout.inventoryPanel.minX + 24.0f, layout.inventoryPanel.minY + 20.0f), textColor, "Inventory");
    drawList->AddText(ImVec2(layout.inventoryPanel.minX + 24.0f, layout.inventoryPanel.minY + 44.0f), emptyTextColor, "Click a block item to assign it to the selected hotbar slot.");
    for (std::size_t itemIndex = 0; itemIndex < kCreativeInventoryItemCount; ++itemIndex) {
        const GameplayUiRect& slot = layout.inventorySlots[itemIndex];
        const InventoryItemId itemId = m_gameplayUiState.creativeInventoryItems[itemIndex];
        const char* label = itemLabel(itemId);
        drawList->AddRectFilled(ImVec2(slot.minX, slot.minY), ImVec2(slot.maxX, slot.maxY), slotColor, 6.0f);
        drawList->AddRect(ImVec2(slot.minX, slot.minY), ImVec2(slot.maxX, slot.maxY), borderColor, 6.0f, 0, 1.5f);
        const float centerX = (slot.minX + slot.maxX) * 0.5f;
        const float centerY = (slot.minY + slot.maxY) * 0.5f;
        drawList->AddCircleFilled(ImVec2(centerX, centerY - 8.0f), 16.0f, itemTint(itemId), 20);
        const ImVec2 textSize = ImGui::CalcTextSize(label);
        drawList->AddText(ImVec2(centerX - (textSize.x * 0.5f), slot.maxY - 22.0f), textColor, label);
    }

    float listX = layout.inventoryPanel.minX + 24.0f;
    float listY = layout.inventoryPanel.maxY + 18.0f;
    drawList->AddText(ImVec2(listX, listY), textColor, "Items");
    listY += 24.0f;
    if (m_gameplayUiState.inventoryEntries.empty()) {
        drawList->AddText(ImVec2(listX, listY), emptyTextColor, "No quest items yet.");
    } else {
        for (const auto& entry : m_gameplayUiState.inventoryEntries) {
            drawList->AddText(ImVec2(listX, listY), textColor, entry.second.c_str());
            listY += 20.0f;
        }
    }

    float questX = layout.inventoryPanel.maxX + 34.0f;
    float questY = layout.inventoryPanel.minY + 20.0f;
    drawList->AddText(ImVec2(questX, questY), textColor, "Journal");
    questY += 26.0f;
    if (m_gameplayUiState.questEntries.empty()) {
        drawList->AddText(ImVec2(questX, questY), emptyTextColor, "No quests started.");
    } else {
        for (const auto& quest : m_gameplayUiState.questEntries) {
            drawList->AddText(ImVec2(questX, questY), textColor, quest.second.c_str());
            questY += 22.0f;
        }
    }
}


} // namespace odai::render
