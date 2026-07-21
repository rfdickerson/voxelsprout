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
#include <cstdio>
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

void RendererBackend::setUiDrawData(const odai::ui::UiDrawData& drawData) {
    m_uiDrawData = drawData;
}

bool RendererBackend::setUiFontAtlas(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height) {
    return m_uiRenderer.setFontAtlasR8(pixels, width, height);
}

odai::ui::UiTextureId RendererBackend::registerUiFontAtlas(const std::uint8_t* pixels, std::uint32_t width,
                                                           std::uint32_t height) {
    return m_uiRenderer.registerFontAtlasR8(pixels, width, height);
}

odai::ui::UiTextureId RendererBackend::registerUiTextureRgba8(const std::uint8_t* pixels, std::uint32_t width,
                                                              std::uint32_t height) {
    return m_uiRenderer.registerTextureRgba8(pixels, width, height);
}

odai::ui::UiTextureId RendererBackend::registerUiTextureRgba8Mipmapped(const std::uint8_t* pixels,
                                                                         std::uint32_t width,
                                                                         std::uint32_t height) {
    return m_uiRenderer.registerTextureRgba8Mipmapped(pixels, width, height);
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

UiRenderStats RendererBackend::uiRenderStats() const {
    const UiRenderer::Stats& stats = m_uiRenderer.stats();
    return {
        stats.textureSlots,
        stats.commandCount,
        stats.drawCallCount,
        stats.dynamicUploadBytes,
        stats.skippedDrawCalls,
    };
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
    if (!ImGui::Begin("Strategy Map", &m_showFrameStatsPanel, kPanelFlags)) {
        ImGui::End();
        return;
    }

    const float autoScale = std::numeric_limits<float>::max();
    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Camera FOV", &m_debugCameraFovDegrees, 55.0f, 120.0f, "%.1f deg");
        ImGui::Text("Map Draws: %u", static_cast<unsigned>(m_importedMeshDraws.size()));
    }

    if (ImGui::CollapsingHeader("Frame Pacing", ImGuiTreeNodeFlags_DefaultOpen)) {
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
    }

    if (ImGui::CollapsingHeader("Lighting & Shadows", ImGuiTreeNodeFlags_DefaultOpen)) {
        int shadowMode = static_cast<int>(m_shadowSettings.mode);
        if (ImGui::Combo("Shadow Backend", &shadowMode, "Shadow Maps\0Ray Traced (Beta)\0Auto (Beta)\0")) {
            setShadowSettings(ShadowSettings{static_cast<ShadowMode>(shadowMode)});
        }
        ImGui::Text(
            "Cascade Splits: %.1f / %.1f / %.1f / %.1f",
            m_shadowCascadeSplits[0],
            m_shadowCascadeSplits[1],
            m_shadowCascadeSplits[2],
            m_shadowCascadeSplits[3]
        );
        ImGui::SliderFloat("Sun Yaw", &m_skyDebugSettings.sunYawDegrees, -180.0f, 180.0f, "%.1f deg");
        ImGui::SliderFloat("Sun Pitch", &m_skyDebugSettings.sunPitchDegrees, -89.0f, 89.0f, "%.1f deg");
        ImGui::SliderFloat("Sky Exposure", &m_skyDebugSettings.skyExposure, 0.25f, 3.0f, "%.2f");
        ImGui::SliderFloat("Sun Disk Intensity", &m_skyDebugSettings.sunDiskIntensity, 300.0f, 2200.0f, "%.0f");
        ImGui::SliderFloat("Sun Halo Intensity", &m_skyDebugSettings.sunHaloIntensity, 4.0f, 64.0f, "%.1f");
        ImGui::Checkbox("Shadow Occluder Culling", &m_shadowDebugSettings.enableOccluderCulling);
        ImGui::SliderFloat("PCF Radius", &m_shadowDebugSettings.pcfRadius, 1.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("Cascade Blend Min", &m_shadowDebugSettings.cascadeBlendMin, 1.0f, 20.0f, "%.2f");
        ImGui::SliderFloat("Cascade Blend Factor", &m_shadowDebugSettings.cascadeBlendFactor, 0.05f, 0.60f, "%.2f");
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
    }

    if (ImGui::CollapsingHeader("Sky & Atmosphere", ImGuiTreeNodeFlags_DefaultOpen)) {
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
        ImGui::Text(
            "Runtime: Rayleigh %.2f, Mie %.2f, Exposure %.2f, Disk %.2f",
            m_skyTuningRuntime.rayleighStrength,
            m_skyTuningRuntime.mieStrength,
            m_skyTuningRuntime.skyExposure,
            m_skyTuningRuntime.sunDiskSize
        );
        if (ImGui::Button("Reset Sun/Sky Defaults")) {
            m_skyDebugSettings = SkyDebugSettings{};
            m_skyTuningRuntime = RendererBackend::SkyTuningRuntimeState{};
        }
    }

    if (ImGui::CollapsingHeader("Post", ImGuiTreeNodeFlags_DefaultOpen)) {
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
            ImGui::SliderFloat(
                "AE Low Percentile",
                &m_skyDebugSettings.autoExposureLowPercentile,
                0.00f,
                0.95f,
                "%.2f"
            );
            ImGui::SliderFloat(
                "AE High Percentile",
                &m_skyDebugSettings.autoExposureHighPercentile,
                0.05f,
                1.00f,
                "%.2f"
            );
            ImGui::TreePop();
        }
        if (!m_autoExposureComputeAvailable) {
            ImGui::TextDisabled("Auto exposure compute unavailable; manual exposure is active.");
        }

        ImGui::Separator();
        ImGui::Text("Bloom");
        ImGui::SliderFloat("Bloom Global Intensity", &m_skyDebugSettings.bloomBaseIntensity, 0.0f, 0.35f, "%.3f");
        if (ImGui::TreeNodeEx("Advanced Bloom", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Bloom Threshold", &m_skyDebugSettings.bloomThreshold, 0.25f, 4.0f, "%.2f");
            ImGui::SliderFloat("Bloom Soft Knee", &m_skyDebugSettings.bloomSoftKnee, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Bloom Sun Boost", &m_skyDebugSettings.bloomSunFacingBoost, 0.0f, 0.40f, "%.3f");
            ImGui::TreePop();
        }

        ImGui::Separator();
        ImGui::Text("Color Grading");
        const char* postLookPresets = "Neutral\0Punchy\0Stylized Vivid\0";
        ImGui::Combo("Post Look", &m_skyDebugSettings.postColorLookPreset, postLookPresets);
        ImGui::SliderFloat("Contrast", &m_skyDebugSettings.colorGradingContrast, 0.70f, 1.40f, "%.2f");
        ImGui::SliderFloat("Saturation", &m_skyDebugSettings.colorGradingSaturation, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Vibrance", &m_skyDebugSettings.colorGradingVibrance, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat(
            "Midtone Contrast",
            &m_skyDebugSettings.colorGradingMidtoneContrast,
            0.80f,
            1.40f,
            "%.2f"
        );
        ImGui::SliderFloat(
            "Shadow Density",
            &m_skyDebugSettings.colorGradingShadowDensity,
            0.70f,
            1.40f,
            "%.2f"
        );
        ImGui::SliderFloat(
            "Highlight Rolloff",
            &m_skyDebugSettings.colorGradingHighlightRolloff,
            0.70f,
            1.10f,
            "%.2f"
        );
        if (ImGui::TreeNodeEx("Advanced Color Grading", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("White Balance R", &m_skyDebugSettings.colorGradingWhiteBalanceR, 0.80f, 1.20f, "%.2f");
            ImGui::SliderFloat("White Balance G", &m_skyDebugSettings.colorGradingWhiteBalanceG, 0.80f, 1.20f, "%.2f");
            ImGui::SliderFloat("White Balance B", &m_skyDebugSettings.colorGradingWhiteBalanceB, 0.80f, 1.20f, "%.2f");
            ImGui::SliderFloat("Shadow Tint R", &m_skyDebugSettings.colorGradingShadowTintR, -0.20f, 0.20f, "%.2f");
            ImGui::SliderFloat("Shadow Tint G", &m_skyDebugSettings.colorGradingShadowTintG, -0.20f, 0.20f, "%.2f");
            ImGui::SliderFloat("Shadow Tint B", &m_skyDebugSettings.colorGradingShadowTintB, -0.20f, 0.20f, "%.2f");
            ImGui::SliderFloat(
                "Highlight Tint R",
                &m_skyDebugSettings.colorGradingHighlightTintR,
                -0.20f,
                0.20f,
                "%.2f"
            );
            ImGui::SliderFloat(
                "Highlight Tint G",
                &m_skyDebugSettings.colorGradingHighlightTintG,
                -0.20f,
                0.20f,
                "%.2f"
            );
            ImGui::SliderFloat(
                "Highlight Tint B",
                &m_skyDebugSettings.colorGradingHighlightTintB,
                -0.20f,
                0.20f,
                "%.2f"
            );
            ImGui::TreePop();
        }
    }

    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
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
            // Rough bottleneck read: whichever of CPU-work / GPU-frame dominates. A
            // wide margin is a confident bound; near-parity means it's balanced.
            const float cpuMs = m_debugCpuFrameWorkMs;
            const float gpuMs = m_debugGpuFrameTimeMs;
            const float maxMs = std::max(cpuMs, gpuMs);
            const char* bound = "balanced";
            ImVec4 boundColor(0.85f, 0.85f, 0.35f, 1.0f);
            if (maxMs > 0.0001f) {
                const float ratio = std::fabs(cpuMs - gpuMs) / maxMs;
                if (ratio > 0.15f) {
                    const bool gpuBound = gpuMs > cpuMs;
                    bound = gpuBound ? "GPU-bound" : "CPU-bound";
                    boundColor = gpuBound ? ImVec4(0.95f, 0.45f, 0.30f, 1.0f)
                                          : ImVec4(0.40f, 0.70f, 0.95f, 1.0f);
                }
            }
            ImGui::TextColored(boundColor, "Bottleneck: %s (CPU %.2f vs GPU %.2f ms)",
                               bound, cpuMs, gpuMs);
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
        if (m_gpuTimestampsSupported &&
            ImGui::TreeNodeEx("GPU Stages", ImGuiTreeNodeFlags_DefaultOpen)) {
            const float frameGpu = m_debugGpuFrameTimeMs;
            // Voxel GI's per-frame cost is occupancy + surface + inject + propagate;
            // the ReSTIR candidate/temporal/spatial/resolve timers are a breakdown of
            // the surface pass, not separate additive stages.
            const float giTotalMs =
                m_debugGpuGiOccupancyTimeMs + m_debugGpuGiSurfaceTimeMs +
                m_debugGpuGiInjectTimeMs + m_debugGpuGiPropagateTimeMs;

            // Green (cheap) -> red (a large share of the frame), saturating near 40%.
            auto stageColor = [](float frac) -> ImVec4 {
                const float t = std::clamp(frac / 0.4f, 0.0f, 1.0f);
                return ImVec4(0.20f + 0.70f * t, 0.20f + 0.70f * (1.0f - t), 0.16f, 1.0f);
            };
            auto stageRow = [&](const char* label, float ms, int indent) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                if (indent > 0) ImGui::Indent(static_cast<float>(indent) * 12.0f);
                ImGui::TextUnformatted(label);
                if (indent > 0) ImGui::Unindent(static_cast<float>(indent) * 12.0f);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%6.3f", ms);
                ImGui::TableSetColumnIndex(2);
                const float frac = (frameGpu > 0.0001f) ? std::clamp(ms / frameGpu, 0.0f, 1.0f) : 0.0f;
                char pct[16];
                std::snprintf(pct, sizeof(pct), "%.1f%%", frac * 100.0f);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, stageColor(frac));
                ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0.0f), pct);
                ImGui::PopStyleColor();
            };

            constexpr ImGuiTableFlags kStageTableFlags =
                ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
            if (ImGui::BeginTable("gpuStages", 3, kStageTableFlags)) {
                ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthFixed, 158.0f);
                ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthFixed, 52.0f);
                ImGui::TableSetupColumn("share of GPU frame", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                // Compute passes (run before main lighting).
                stageRow("Voxel GI", giTotalMs, 0);
                stageRow("occupancy", m_debugGpuGiOccupancyTimeMs, 1);
                stageRow("surface", m_debugGpuGiSurfaceTimeMs, 1);
                if (m_voxelGiRestirActiveThisFrame) {
                    stageRow("candidate", m_debugGpuGiSurfaceCandidateTimeMs, 2);
                    stageRow("temporal", m_debugGpuGiSurfaceTemporalTimeMs, 2);
                    stageRow("spatial", m_debugGpuGiSurfaceSpatialTimeMs, 2);
                    stageRow("resolve", m_debugGpuGiSurfaceResolveTimeMs, 2);
                }
                stageRow("inject", m_debugGpuGiInjectTimeMs, 1);
                stageRow("propagate", m_debugGpuGiPropagateTimeMs, 1);
                stageRow("Auto Exposure", m_debugGpuAutoExposureTimeMs, 0);
                stageRow("Sun Shafts", m_debugGpuSunShaftTimeMs, 0);
                // Graphics passes.
                stageRow("Shadow", m_debugGpuShadowTimeMs, 0);
                stageRow("Prepass (nrm/depth)", m_debugGpuPrepassTimeMs, 0);
                stageRow("SSAO", m_debugGpuSsaoTimeMs, 0);
                stageRow("SSAO Blur", m_debugGpuSsaoBlurTimeMs, 0);
                stageRow("Main", m_debugGpuMainTimeMs, 0);
                stageRow("Post", m_debugGpuPostTimeMs, 0);
                stageRow("UI", m_debugGpuUiTimeMs, 0);
                ImGui::EndTable();
            }

            // Sum of top-level stages vs the measured frame; the remainder is barrier/idle
            // gaps and any untimed work, useful for spotting pipeline bubbles.
            const float accountedMs =
                giTotalMs + m_debugGpuAutoExposureTimeMs + m_debugGpuSunShaftTimeMs +
                m_debugGpuShadowTimeMs + m_debugGpuPrepassTimeMs + m_debugGpuSsaoTimeMs +
                m_debugGpuSsaoBlurTimeMs + m_debugGpuMainTimeMs + m_debugGpuPostTimeMs +
                m_debugGpuUiTimeMs;
            const float otherMs = std::max(0.0f, frameGpu - accountedMs);
            ImGui::Text(
                "Accounted %.3f  |  Frame GPU %.3f  |  Other/idle %.3f ms",
                accountedMs, frameGpu, otherMs);

            const UiRenderer::Stats& uiStats = m_uiRenderer.stats();
            ImGui::Text(
                "UI batches/draws/textures: %u / %u / %u",
                uiStats.commandCount,
                uiStats.drawCallCount,
                uiStats.textureSlots
            );
            ImGui::Text(
                "UI dynamic upload: %.2f KiB (skipped draws: %llu)",
                static_cast<double>(uiStats.dynamicUploadBytes) / 1024.0,
                static_cast<unsigned long long>(uiStats.skippedDrawCalls)
            );
            ImGui::TreePop();
        }

        if (m_vmaAllocator != VK_NULL_HANDLE &&
            ImGui::TreeNodeEx("GPU Memory", ImGuiTreeNodeFlags_DefaultOpen)) {
            VkPhysicalDeviceMemoryProperties memProps{};
            vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
            std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
            vmaGetHeapBudgets(m_vmaAllocator, budgets.data());
            constexpr double kMiB = 1024.0 * 1024.0;
            for (uint32_t heap = 0; heap < memProps.memoryHeapCount; ++heap) {
                const bool deviceLocal =
                    (memProps.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
                const VmaBudget& b = budgets[heap];
                const float frac = (b.budget > 0)
                    ? std::clamp(static_cast<float>(static_cast<double>(b.usage) /
                                                    static_cast<double>(b.budget)), 0.0f, 1.0f)
                    : 0.0f;
                char overlay[64];
                std::snprintf(overlay, sizeof(overlay), "%.0f / %.0f MiB",
                              static_cast<double>(b.usage) / kMiB,
                              static_cast<double>(b.budget) / kMiB);
                ImGui::Text("Heap %u %s", heap, deviceLocal ? "(device-local)" : "(host)");
                ImGui::SameLine();
                ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0.0f), overlay);
                ImGui::Text(
                    "   VMA blocks %u (%.1f MiB), allocations %u (%.1f MiB)",
                    b.statistics.blockCount,
                    static_cast<double>(b.statistics.blockBytes) / kMiB,
                    b.statistics.allocationCount,
                    static_cast<double>(b.statistics.allocationBytes) / kMiB);
            }
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
    }

    if (ImGui::CollapsingHeader("Renderer Diagnostics", ImGuiTreeNodeFlags_DefaultOpen)) {
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
            ImGui::Text("Descriptor Buffer: %s", m_desktopCapabilityProbe.descriptorBufferExtension ? "yes" : "no");
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


} // namespace odai::render
