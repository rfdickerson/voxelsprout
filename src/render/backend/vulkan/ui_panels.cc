#include "render/backend/vulkan/renderer_backend.h"

#include <imgui.h>

namespace odai::render {

namespace {

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

void RendererBackend::buildDofDebugUi() {
    if (!m_debugUiVisible || !m_showFrameStatsPanel) {
        return;
    }

    constexpr ImGuiWindowFlags kPanelFlags =
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoSavedSettings;
    ImGui::SetNextWindowPos(ImVec2(16.0f, 420.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Depth of Field", nullptr, kPanelFlags)) {
        ImGui::End();
        return;
    }

    ImGui::Checkbox("Enable DoF", &m_skyDebugSettings.depthOfFieldEnabled);
    ImGui::SliderFloat("DoF Intensity", &m_skyDebugSettings.depthOfFieldMaxRadiusPixels, 0.0f, 14.0f, "%.1f");
    ImGui::SliderFloat("DoF Target Distance", &m_skyDebugSettings.depthOfFieldFocusDistance, 1.0f, 240.0f, "%.1f");

    ImGui::End();
}

void RendererBackend::buildShadowDebugUi() {
    if (!m_debugUiVisible || !m_showShadowPanel) {
        return;
    }

    if (!ImGui::Begin("Shadows", &m_showShadowPanel)) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("ShadowsTabs")) {
        if (ImGui::BeginTabItem("Shadows")) {
            ImGui::Text(
                "Macro Cells U/R4/R1: %u / %u / %u",
                m_debugMacroCellUniformCount,
                m_debugMacroCellRefined4Count,
                m_debugMacroCellRefined1Count
            );
            ImGui::Text(
                "Drawn LOD ranges 0/1/2: %u / %u / %u",
                m_debugDrawnLod0Ranges,
                m_debugDrawnLod1Ranges,
                m_debugDrawnLod2Ranges
            );
            ImGui::Text("Cascade Splits: %.1f / %.1f / %.1f / %.1f",
                m_shadowCascadeSplits[0],
                m_shadowCascadeSplits[1],
                m_shadowCascadeSplits[2],
                m_shadowCascadeSplits[3]
            );
            ImGui::Separator();
            ImGui::Checkbox("Shadow Occluder Culling", &m_shadowDebugSettings.enableOccluderCulling);
            ImGui::SliderFloat("PCF Radius", &m_shadowDebugSettings.pcfRadius, 1.0f, 3.0f, "%.2f");
            ImGui::SeparatorText("RT Main Pass");
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
            ImGui::SliderInt("Grass Shadow Cascades", &m_shadowDebugSettings.grassShadowCascadeCount, 0, static_cast<int>(kShadowCascadeCount));
            if (ImGui::CollapsingHeader("Advanced Bias Controls")) {
                ImGui::Text("Receiver Bias");
                ImGui::SliderFloat("Normal Offset Near", &m_shadowDebugSettings.receiverNormalOffsetNear, 0.0f, 0.20f, "%.3f");
                ImGui::SliderFloat("Normal Offset Far", &m_shadowDebugSettings.receiverNormalOffsetFar, 0.0f, 0.35f, "%.3f");
                ImGui::SliderFloat("Base Bias Near (texel)", &m_shadowDebugSettings.receiverBaseBiasNearTexel, 0.0f, 12.0f, "%.2f");
                ImGui::SliderFloat("Base Bias Far (texel)", &m_shadowDebugSettings.receiverBaseBiasFarTexel, 0.0f, 16.0f, "%.2f");
                ImGui::SliderFloat("Slope Bias Near (texel)", &m_shadowDebugSettings.receiverSlopeBiasNearTexel, 0.0f, 14.0f, "%.2f");
                ImGui::SliderFloat("Slope Bias Far (texel)", &m_shadowDebugSettings.receiverSlopeBiasFarTexel, 0.0f, 18.0f, "%.2f");
                ImGui::Separator();
                ImGui::Text("Caster Bias");
                ImGui::SliderFloat("Const Bias Base", &m_shadowDebugSettings.casterConstantBiasBase, 0.0f, 6.0f, "%.2f");
                ImGui::SliderFloat("Const Bias Cascade Scale", &m_shadowDebugSettings.casterConstantBiasCascadeScale, 0.0f, 3.0f, "%.2f");
                ImGui::SliderFloat("Slope Bias Base", &m_shadowDebugSettings.casterSlopeBiasBase, 0.0f, 8.0f, "%.2f");
                ImGui::SliderFloat("Slope Bias Cascade Scale", &m_shadowDebugSettings.casterSlopeBiasCascadeScale, 0.0f, 4.0f, "%.2f");
            }
            if (ImGui::Button("Reset Shadow Defaults")) {
                m_shadowDebugSettings = ShadowDebugSettings{};
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("AO + GI")) {
            ImGui::Checkbox("Enable Vertex AO", &m_debugEnableVertexAo);
            ImGui::Checkbox("Enable SSAO", &m_debugEnableSsao);
            ImGui::SliderFloat("SSAO Radius", &m_shadowDebugSettings.ssaoRadius, 1.0f, 96.0f, "%.1f");
            ImGui::SliderFloat("SSAO Bias", &m_shadowDebugSettings.ssaoBias, 0.0f, 6.0f, "%.2f");
            ImGui::SliderFloat("SSAO Intensity", &m_shadowDebugSettings.ssaoIntensity, 0.0f, 2.0f, "%.2f");
            if (ImGui::CollapsingHeader("Advanced AO Debug")) {
                ImGui::Checkbox("Visualize SSAO", &m_debugVisualizeSsao);
                ImGui::Checkbox("Visualize AO Normals", &m_debugVisualizeAoNormals);
            }

            ImGui::Separator();
            ImGui::Text("Voxel GI");
            ImGui::Text("Compute: %s", m_voxelGiComputeAvailable ? "on" : "fallback");
            ImGui::SliderFloat("Bounce Strength", &m_voxelGiDebugSettings.bounceStrength, 0.0f, 2.50f, "%.2f");
            ImGui::SliderFloat("Diffusion Softness", &m_voxelGiDebugSettings.diffusionSoftness, 0.0f, 1.0f, "%.2f");
            int giSurfaceMode = static_cast<int>(m_voxelGiDebugSettings.surfaceMode);
            ImGui::Combo("GI Surface Mode", &giSurfaceMode, "Legacy\0RT Surface\0ReSTIR Surface\0");
            m_voxelGiDebugSettings.surfaceMode = static_cast<VoxelGiSurfaceMode>(giSurfaceMode);
            ImGui::SliderInt("RT Surface Samples", &m_voxelGiDebugSettings.rtSurfaceSampleCount, 1, 2);
            ImGui::SliderFloat("RT Surface Bias", &m_voxelGiDebugSettings.rtSurfaceBiasScale, 0.25f, 4.0f, "%.2f");
            ImGui::SeparatorText("ReSTIR GI");
            ImGui::SliderInt("ReSTIR Candidates", &m_voxelGiDebugSettings.restirCandidateCount, 1, 8);
            ImGui::Checkbox("ReSTIR Temporal", &m_voxelGiDebugSettings.restirEnableTemporalReuse);
            ImGui::Checkbox("ReSTIR Spatial", &m_voxelGiDebugSettings.restirEnableSpatialReuse);
            ImGui::SliderInt("ReSTIR Radius", &m_voxelGiDebugSettings.restirSpatialRadius, 1, 2);
            if (ImGui::Button("Reset ReSTIR History")) {
                m_voxelGiDebugSettings.restirHistoryResetRequested = true;
            }
            if (ImGui::CollapsingHeader("Advanced GI Debug")) {
                const char* giVisualizationModes = "Off\0Radiance\0False Color Luma\0Radiance (Gray)\0Occupancy Albedo\0";
                ImGui::Combo("GI Visualize", &m_voxelGiDebugSettings.visualizationMode, giVisualizationModes);
                if (m_voxelGiDebugSettings.visualizationMode > 0) {
                    m_debugVisualizeSsao = false;
                    m_debugVisualizeAoNormals = false;
                }
            }
            if (ImGui::Button("Reset GI Defaults")) {
                m_voxelGiDebugSettings = VoxelGiDebugSettings{};
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Display")) {
            if (m_supportsDisplayTiming) {
                ImGui::Checkbox("Use Display Timing", &m_enableDisplayTiming);
            } else {
                ImGui::TextDisabled("Display Timing: unsupported");
                m_enableDisplayTiming = false;
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}

void RendererBackend::buildSunDebugUi() {
    if (!m_debugUiVisible || !m_showSunPanel) {
        return;
    }

    if (!ImGui::Begin("Sun/Sky", &m_showSunPanel)) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("SunSkyTabs")) {
        if (ImGui::BeginTabItem("Sun & Atmosphere")) {
            ImGui::Checkbox("Enable DoF", &m_skyDebugSettings.depthOfFieldEnabled);
            ImGui::SliderFloat("DoF Intensity", &m_skyDebugSettings.depthOfFieldMaxRadiusPixels, 0.0f, 14.0f, "%.1f");
            ImGui::SliderFloat("DoF Target Distance", &m_skyDebugSettings.depthOfFieldFocusDistance, 1.0f, 240.0f, "%.1f");
            ImGui::Separator();
            ImGui::SliderFloat("Sun Yaw", &m_skyDebugSettings.sunYawDegrees, -180.0f, 180.0f, "%.1f deg");
            ImGui::SliderFloat("Sun Pitch", &m_skyDebugSettings.sunPitchDegrees, -89.0f, 5.0f, "%.1f deg");
            ImGui::SliderFloat("Camera FOV", &m_debugCameraFovDegrees, 55.0f, 120.0f, "%.1f deg");
            ImGui::SliderFloat("Sky Exposure", &m_skyDebugSettings.skyExposure, 0.25f, 3.0f, "%.2f");
            ImGui::SliderFloat("Sun Disk Intensity", &m_skyDebugSettings.sunDiskIntensity, 300.0f, 2200.0f, "%.0f");
            ImGui::SliderFloat("Sun Halo Intensity", &m_skyDebugSettings.sunHaloIntensity, 4.0f, 64.0f, "%.1f");
            if (ImGui::CollapsingHeader("Advanced Atmosphere")) {
                ImGui::Checkbox("Auto Sunrise Tuning", &m_skyDebugSettings.autoSunriseTuning);
                ImGui::SliderFloat("Auto Sunrise Blend", &m_skyDebugSettings.autoSunriseBlend, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Auto Adapt Speed", &m_skyDebugSettings.autoSunriseAdaptSpeed, 0.5f, 12.0f, "%.2f");
                ImGui::Separator();
                ImGui::SliderFloat("Rayleigh Strength", &m_skyDebugSettings.rayleighStrength, 0.1f, 4.0f, "%.2f");
                ImGui::SliderFloat("Mie Strength", &m_skyDebugSettings.mieStrength, 0.05f, 4.0f, "%.2f");
                ImGui::SliderFloat("Mie Anisotropy", &m_skyDebugSettings.mieAnisotropy, 0.0f, 0.95f, "%.2f");
                ImGui::SliderFloat("Sun Disk Size", &m_skyDebugSettings.sunDiskSize, 0.5f, 6.0f, "%.2f");
                ImGui::SliderFloat("Sun Haze Falloff", &m_skyDebugSettings.sunHazeFalloff, 0.10f, 1.20f, "%.2f");
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Post")) {
            ImGui::Text("Depth of Field");
            ImGui::Checkbox("Enable DoF", &m_skyDebugSettings.depthOfFieldEnabled);
            ImGui::SliderFloat("DoF Intensity", &m_skyDebugSettings.depthOfFieldMaxRadiusPixels, 0.0f, 14.0f, "%.1f");
            ImGui::SliderFloat("DoF Target Distance", &m_skyDebugSettings.depthOfFieldFocusDistance, 1.0f, 240.0f, "%.1f");
            ImGui::SliderFloat("Focus Range", &m_skyDebugSettings.depthOfFieldFocusRange, 1.0f, 120.0f, "%.1f");
            if (ImGui::CollapsingHeader("Advanced Depth of Field")) {
                ImGui::SliderFloat("Near Blur Scale", &m_skyDebugSettings.depthOfFieldNearBlurScale, 0.25f, 3.0f, "%.2f");
            }

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
            if (m_skyDebugSettings.autoExposureEnabled && ImGui::CollapsingHeader("Advanced Exposure")) {
                ImGui::SliderInt("AE Update Interval", &m_skyDebugSettings.autoExposureUpdateIntervalFrames, 1, 16);
                ImGui::SliderFloat("AE Key Value", &m_skyDebugSettings.autoExposureKeyValue, 0.05f, 0.50f, "%.3f");
                ImGui::SliderFloat("AE Min Exposure", &m_skyDebugSettings.autoExposureMin, 0.05f, 2.50f, "%.3f");
                ImGui::SliderFloat("AE Max Exposure", &m_skyDebugSettings.autoExposureMax, 0.20f, 12.00f, "%.3f");
                ImGui::SliderFloat("AE Adapt Up", &m_skyDebugSettings.autoExposureAdaptUp, 0.10f, 12.00f, "%.2f");
                ImGui::SliderFloat("AE Adapt Down", &m_skyDebugSettings.autoExposureAdaptDown, 0.10f, 12.00f, "%.2f");
                ImGui::SliderFloat("AE Low Percentile", &m_skyDebugSettings.autoExposureLowPercentile, 0.00f, 0.95f, "%.2f");
                ImGui::SliderFloat("AE High Percentile", &m_skyDebugSettings.autoExposureHighPercentile, 0.05f, 1.00f, "%.2f");
            }
            if (!m_autoExposureComputeAvailable) {
                ImGui::TextDisabled("Auto exposure compute unavailable; manual exposure is active.");
            }

            ImGui::Separator();
            ImGui::Text("Bloom");
            ImGui::SliderFloat("Bloom Global Intensity", &m_skyDebugSettings.bloomBaseIntensity, 0.0f, 0.35f, "%.3f");
            if (ImGui::CollapsingHeader("Advanced Bloom")) {
                ImGui::SliderFloat("Bloom Threshold", &m_skyDebugSettings.bloomThreshold, 0.25f, 4.0f, "%.2f");
                ImGui::SliderFloat("Bloom Soft Knee", &m_skyDebugSettings.bloomSoftKnee, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("Bloom Sun Boost", &m_skyDebugSettings.bloomSunFacingBoost, 0.0f, 0.40f, "%.3f");
                ImGui::TextDisabled("Bloom is hidden in GI/SSAO debug visualization modes.");
            }

            ImGui::Separator();
            ImGui::Text("Color Grading");
            const char* postLookPresets = "Neutral\0Punchy\0Stylized Vivid\0";
            if (ImGui::Combo("Post Look", &m_skyDebugSettings.postColorLookPreset, postLookPresets)) {
                applyPostLookPreset(m_skyDebugSettings, m_skyDebugSettings.postColorLookPreset);
            }
            ImGui::SliderFloat("Contrast", &m_skyDebugSettings.colorGradingContrast, 0.70f, 1.40f, "%.2f");
            ImGui::SliderFloat("Saturation", &m_skyDebugSettings.colorGradingSaturation, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Vibrance", &m_skyDebugSettings.colorGradingVibrance, -1.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Midtone Contrast", &m_skyDebugSettings.colorGradingMidtoneContrast, 0.80f, 1.40f, "%.2f");
            ImGui::SliderFloat("Shadow Density", &m_skyDebugSettings.colorGradingShadowDensity, 0.70f, 1.40f, "%.2f");
            ImGui::SliderFloat("Highlight Rolloff", &m_skyDebugSettings.colorGradingHighlightRolloff, 0.70f, 1.10f, "%.2f");
            if (ImGui::CollapsingHeader("Advanced Color Grading")) {
                ImGui::SliderFloat("White Balance R", &m_skyDebugSettings.colorGradingWhiteBalanceR, 0.80f, 1.20f, "%.2f");
                ImGui::SliderFloat("White Balance G", &m_skyDebugSettings.colorGradingWhiteBalanceG, 0.80f, 1.20f, "%.2f");
                ImGui::SliderFloat("White Balance B", &m_skyDebugSettings.colorGradingWhiteBalanceB, 0.80f, 1.20f, "%.2f");
                ImGui::SliderFloat("Shadow Tint R", &m_skyDebugSettings.colorGradingShadowTintR, -0.20f, 0.20f, "%.2f");
                ImGui::SliderFloat("Shadow Tint G", &m_skyDebugSettings.colorGradingShadowTintG, -0.20f, 0.20f, "%.2f");
                ImGui::SliderFloat("Shadow Tint B", &m_skyDebugSettings.colorGradingShadowTintB, -0.20f, 0.20f, "%.2f");
                ImGui::SliderFloat("Highlight Tint R", &m_skyDebugSettings.colorGradingHighlightTintR, -0.20f, 0.20f, "%.2f");
                ImGui::SliderFloat("Highlight Tint G", &m_skyDebugSettings.colorGradingHighlightTintG, -0.20f, 0.20f, "%.2f");
                ImGui::SliderFloat("Highlight Tint B", &m_skyDebugSettings.colorGradingHighlightTintB, -0.20f, 0.20f, "%.2f");
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Fog & Foliage")) {
            ImGui::SliderFloat("Fog Density", &m_skyDebugSettings.volumetricFogDensity, 0.0f, 0.03f, "%.4f");
            ImGui::SliderFloat("Fog Sun Scatter", &m_skyDebugSettings.volumetricSunScattering, 0.0f, 3.0f, "%.2f");
            if (ImGui::CollapsingHeader("Advanced Fog")) {
                ImGui::SliderFloat("Fog Height Falloff", &m_skyDebugSettings.volumetricFogHeightFalloff, 0.0f, 0.30f, "%.4f");
                ImGui::SliderFloat("Fog Base Height", &m_skyDebugSettings.volumetricFogBaseHeight, -32.0f, 320.0f, "%.1f");
            }
            ImGui::Separator();
            ImGui::Text("Water");
            ImGui::SliderFloat("Water Speed", &m_skyDebugSettings.waterAnimationSpeed, 0.25f, 4.0f, "%.2f");
            ImGui::SliderFloat("Water Normal Strength", &m_skyDebugSettings.waterNormalStrength, 0.25f, 2.5f, "%.2f");
            ImGui::SliderFloat("Water Reflection", &m_skyDebugSettings.waterReflectionStrength, 0.25f, 4.0f, "%.2f");
            ImGui::SliderFloat("Water Absorption", &m_skyDebugSettings.waterRefractionDecay, 0.25f, 5.0f, "%.2f");
            ImGui::Separator();
            ImGui::SliderFloat("Plant Quad Directionality", &m_skyDebugSettings.plantQuadDirectionality, 0.0f, 1.0f, "%.2f");
            ImGui::Text("Active: Rayleigh %.2f, Mie %.2f, Exposure %.2f, Disk %.2f",
                m_skyTuningRuntime.rayleighStrength,
                m_skyTuningRuntime.mieStrength,
                m_skyTuningRuntime.skyExposure,
                m_skyTuningRuntime.sunDiskSize
            );
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    if (ImGui::Button("Reset Sun/Sky Defaults")) {
        m_skyDebugSettings = SkyDebugSettings{};
        m_skyTuningRuntime = RendererBackend::SkyTuningRuntimeState{};
    }
    ImGui::End();
}

} // namespace odai::render
