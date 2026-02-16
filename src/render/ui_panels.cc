#include "render/renderer_backend.h"

#include <imgui.h>

namespace render {

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
            ImGui::SliderFloat("SSAO Radius", &m_shadowDebugSettings.ssaoRadius, 0.10f, 2.00f, "%.2f");
            ImGui::SliderFloat("SSAO Bias", &m_shadowDebugSettings.ssaoBias, 0.0f, 0.20f, "%.3f");
            ImGui::SliderFloat("SSAO Intensity", &m_shadowDebugSettings.ssaoIntensity, 0.0f, 1.50f, "%.2f");
            if (ImGui::CollapsingHeader("Advanced AO Debug")) {
                ImGui::Checkbox("Visualize SSAO", &m_debugVisualizeSsao);
                ImGui::Checkbox("Visualize AO Normals", &m_debugVisualizeAoNormals);
            }

            ImGui::Separator();
            ImGui::Text("Voxel GI");
            ImGui::Text("Compute: %s", m_voxelGiComputeAvailable ? "on" : "fallback");
            ImGui::SliderFloat("Bounce Strength", &m_voxelGiDebugSettings.bounceStrength, 0.0f, 2.50f, "%.2f");
            ImGui::SliderFloat("Diffusion Softness", &m_voxelGiDebugSettings.diffusionSoftness, 0.0f, 1.0f, "%.2f");
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
            ImGui::Text("Eye Adaptation");
            ImGui::Checkbox("Auto Exposure", &m_skyDebugSettings.autoExposureEnabled);
            ImGui::SliderFloat("Manual Exposure", &m_skyDebugSettings.manualExposure, 0.05f, 4.0f, "%.3f");
            if (m_skyDebugSettings.autoExposureEnabled && ImGui::CollapsingHeader("Advanced Exposure")) {
                ImGui::TextDisabled("AE Update Interval: fixed to every frame");
                ImGui::SliderFloat("AE Key Value", &m_skyDebugSettings.autoExposureKeyValue, 0.05f, 0.50f, "%.3f");
                ImGui::SliderFloat("AE Min Exposure", &m_skyDebugSettings.autoExposureMin, 0.05f, 2.50f, "%.3f");
                ImGui::SliderFloat("AE Max Exposure", &m_skyDebugSettings.autoExposureMax, 0.20f, 8.00f, "%.3f");
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
            ImGui::SliderFloat("Contrast", &m_skyDebugSettings.colorGradingContrast, 0.70f, 1.40f, "%.2f");
            ImGui::SliderFloat("Saturation", &m_skyDebugSettings.colorGradingSaturation, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("Vibrance", &m_skyDebugSettings.colorGradingVibrance, -1.0f, 1.0f, "%.2f");
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
                ImGui::SliderFloat("Fog Height Falloff", &m_skyDebugSettings.volumetricFogHeightFalloff, 0.0f, 0.30f, "%.3f");
                ImGui::SliderFloat("Fog Base Height", &m_skyDebugSettings.volumetricFogBaseHeight, -32.0f, 64.0f, "%.1f");
            }
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

} // namespace render

