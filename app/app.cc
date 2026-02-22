#include "app/app.h"

#include "core/log.h"
#include "core/noise_field.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace voxelsprout::app {
namespace {

constexpr int kInteractivePreviewMaxBounces = 1;
constexpr int kInteractivePreviewCooldownFrames = 8;

voxelsprout::core::Vec3 sunDirectionFromAngles(float azimuthDegrees, float elevationDegrees) {
    const float azimuthRadians = azimuthDegrees * (3.1415926535f / 180.0f);
    const float elevationRadians = elevationDegrees * (3.1415926535f / 180.0f);
    const float cosElevation = std::cos(elevationRadians);
    return voxelsprout::core::normalize(voxelsprout::core::Vec3{
        cosElevation * std::cos(azimuthRadians),
        std::sin(elevationRadians),
        cosElevation * std::sin(azimuthRadians)});
}

void sunAnglesFromDirection(const voxelsprout::core::Vec3& direction, float& outAzimuthDegrees, float& outElevationDegrees) {
    const float len = std::sqrt((direction.x * direction.x) + (direction.y * direction.y) + (direction.z * direction.z));
    if (len <= 1e-6f) {
        outAzimuthDegrees = 0.0f;
        outElevationDegrees = 15.0f;
        return;
    }

    const float nx = direction.x / len;
    const float ny = direction.y / len;
    const float nz = direction.z / len;
    outAzimuthDegrees = std::atan2(nz, nx) * (180.0f / 3.1415926535f);
    outElevationDegrees = std::asin(std::clamp(ny, -1.0f, 1.0f)) * (180.0f / 3.1415926535f);
}

} // namespace

bool App::init() {
    if (!glfwInit()) {
        VOX_LOGE("app") << "failed to initialize GLFW";
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(1600, 900, "Voxelsprout Compute Lab", nullptr, nullptr);
    if (m_window == nullptr) {
        VOX_LOGE("app") << "failed to create window";
        return false;
    }

    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    if (!m_renderer.init(m_window)) {
        VOX_LOGE("app") << "renderer init failed";
        return false;
    }

    const voxelsprout::core::NoiseSamples noiseProbe = voxelsprout::core::sampleLibNoise(0.0, 0.0, 0.0);
    VOX_LOGI("app") << "noise probe perlin=" << noiseProbe.perlin << " worley=" << noiseProbe.worley;

    m_renderParams.camera.position = {0.0f, 2.0f, 10.0f};
    m_renderParams.camera.yawDegrees = -90.0f;
    m_renderParams.camera.pitchDegrees = 12.0f;
    m_renderParams.camera.fovDegrees = 75.0f;
    // Start with a high frontal sun for bright front-lit cumulus.
    m_renderParams.scene.sun.direction = sunDirectionFromAngles(-45.0f, 70.0f);
    m_renderParams.scene.sun.intensity = 28.0f;
    m_renderParams.scene.volume.densityScale = 3.5f;
    m_renderParams.scene.volume.anisotropyG = 0.82f;
    m_renderParams.scene.volume.albedo = 0.97f;
    m_renderParams.scene.volume.macroScale = 1.0f;
    m_renderParams.scene.volume.detailScale = 1.2f;
    m_renderParams.scene.volume.coverage = 0.72f;
    m_renderParams.scene.volume.weatherScale = 1.0f;
    m_renderParams.scene.volume.cloudBase = 2.5f;
    m_renderParams.scene.volume.cloudTop = 7.5f;
    m_renderParams.scene.volume.erosionStrength = 0.75f;
    m_renderParams.scene.volume.brightnessBoost = 1.0f;
    m_renderParams.scene.volume.ambientLift = 0.40f;
    m_renderParams.scene.volume.maxBounces = 1;
    m_renderParams.exposure = 0.14f;
    m_renderParams.toneMapOperator = 2;
    m_renderParams.toneMapWhitePoint = 1.0f;
    m_renderParams.toneMapShoulder = 2.4f;
    m_renderParams.toneMapContrast = 1.0f;
    m_renderParams.toneMapSaturation = 1.0f;
    m_renderParams.toneMapGamma = 2.2f;
    m_renderParams.cloudUpdateInterval = 2;
    m_renderParams.maxAccumulationSamples = 256;
    m_maxAccumulationSamplesUi = 256;
    sunAnglesFromDirection(m_renderParams.scene.sun.direction, m_sunAzimuthDegreesUi, m_sunElevationDegreesUi);
    return true;
}

void App::pollInput() {
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }

    const bool tabDown = glfwGetKey(m_window, GLFW_KEY_TAB) == GLFW_PRESS;
    if (tabDown && !m_tabWasDown) {
        m_showUi = !m_showUi;
    }
    m_tabWasDown = tabDown;
}

void App::buildUi() {
    m_uiInteracting = false;
    if (!m_showUi) {
        return;
    }

    ImGui::Begin("Compute Renderer");
    ImGui::Text("CloudPathTracePass + ToneMapPass");
    ImGui::Separator();

    ImGui::Checkbox("Progressive accumulation", &m_renderParams.enableAccumulation);
    ImGui::Checkbox("Debug: Sun Tr grayscale", &m_renderParams.debugSunTransmittance);
    ImGui::SeparatorText("Tone Mapping");
    ImGui::SliderFloat("Exposure", &m_renderParams.exposure, 0.05f, 8.0f, "%.2f");
    const char* toneMapLabels[] = {"Linear", "Reinhard", "ACES"};
    int toneMapOperatorUi = static_cast<int>(m_renderParams.toneMapOperator);
    ImGui::Combo("Tone map curve", &toneMapOperatorUi, toneMapLabels, IM_ARRAYSIZE(toneMapLabels));
    m_renderParams.toneMapOperator = static_cast<std::uint32_t>(std::clamp(toneMapOperatorUi, 0, 2));
    ImGui::SliderFloat("White point", &m_renderParams.toneMapWhitePoint, 0.1f, 16.0f, "%.2f");
    ImGui::SliderFloat("Shoulder", &m_renderParams.toneMapShoulder, 0.2f, 8.0f, "%.2f");
    ImGui::SliderFloat("Contrast", &m_renderParams.toneMapContrast, 0.5f, 1.8f, "%.2f");
    ImGui::SliderFloat("Saturation", &m_renderParams.toneMapSaturation, 0.0f, 1.8f, "%.2f");
    ImGui::Text("Gamma: 2.20 (fixed)");
    m_renderParams.toneMapWhitePoint = std::max(m_renderParams.toneMapWhitePoint, 0.1f);
    m_renderParams.toneMapShoulder = std::max(m_renderParams.toneMapShoulder, 0.2f);
    m_renderParams.toneMapGamma = 2.2f;

    ImGui::SeparatorText("Cloud");
    ImGui::SliderFloat("Density scale", &m_renderParams.scene.volume.densityScale, 0.1f, 8.0f, "%.2f");
    ImGui::SliderFloat("g parameter", &m_renderParams.scene.volume.anisotropyG, 0.0f, 0.90f, "%.2f");
    m_renderParams.scene.volume.anisotropyG = std::clamp(m_renderParams.scene.volume.anisotropyG, 0.0f, 0.90f);
    ImGui::SliderFloat("Cloud albedo", &m_renderParams.scene.volume.albedo, 0.9f, 1.0f, "%.3f");
    ImGui::SliderFloat("Macro scale", &m_renderParams.scene.volume.macroScale, 0.20f, 3.00f, "%.2f");
    ImGui::SliderFloat("Detail scale", &m_renderParams.scene.volume.detailScale, 0.20f, 3.00f, "%.2f");
    ImGui::SliderFloat("Coverage", &m_renderParams.scene.volume.coverage, 0.1f, 0.98f, "%.2f");
    ImGui::SliderFloat("Weather scale", &m_renderParams.scene.volume.weatherScale, 0.05f, 2.0f, "%.2f");
    ImGui::SliderFloat("Cloud base", &m_renderParams.scene.volume.cloudBase, -1.0f, 6.0f, "%.2f");
    ImGui::SliderFloat("Cloud top", &m_renderParams.scene.volume.cloudTop, 4.0f, 14.0f, "%.2f");
    ImGui::SliderFloat("Erosion strength", &m_renderParams.scene.volume.erosionStrength, 0.0f, 1.5f, "%.2f");
    ImGui::SliderFloat("Ambient lift", &m_renderParams.scene.volume.ambientLift, 0.0f, 1.5f, "%.2f");
    ImGui::SliderInt("Max bounces", &m_renderParams.scene.volume.maxBounces, 1, 12);
    ImGui::Text("1 = real-time approx, >1 = full path traced");
    m_renderParams.scene.volume.albedo = std::clamp(m_renderParams.scene.volume.albedo, 0.9f, 1.0f);
    m_renderParams.scene.volume.cloudTop =
        std::max(m_renderParams.scene.volume.cloudTop, m_renderParams.scene.volume.cloudBase + 0.25f);
    m_renderParams.scene.volume.maxBounces = std::clamp(m_renderParams.scene.volume.maxBounces, 1, 12);
    ImGui::SliderFloat("Sun intensity", &m_renderParams.scene.sun.intensity, 1.0f, 80.0f, "%.2f");
    ImGui::SliderFloat("Sun azimuth", &m_sunAzimuthDegreesUi, -180.0f, 180.0f, "%.1f deg");
    ImGui::SliderFloat("Sun elevation", &m_sunElevationDegreesUi, -10.0f, 89.0f, "%.1f deg");
    m_renderParams.scene.sun.direction = sunDirectionFromAngles(m_sunAzimuthDegreesUi, m_sunElevationDegreesUi);
    ImGui::Text(
        "Sun dir: (%.2f, %.2f, %.2f)",
        m_renderParams.scene.sun.direction.x,
        m_renderParams.scene.sun.direction.y,
        m_renderParams.scene.sun.direction.z);
    if (ImGui::Button("Reset accumulation")) {
        m_renderParams.forceReset = true;
    }
    ImGui::SliderInt("Cloud update interval (frames)", &m_cloudUpdateIntervalUi, 1, 8);
    m_cloudUpdateIntervalUi = std::clamp(m_cloudUpdateIntervalUi, 1, 8);
    m_renderParams.cloudUpdateInterval = static_cast<std::uint32_t>(m_cloudUpdateIntervalUi);
    ImGui::SliderInt("Max accumulation samples", &m_maxAccumulationSamplesUi, 1, 4096);
    m_maxAccumulationSamplesUi = std::clamp(m_maxAccumulationSamplesUi, 1, 4096);
    m_renderParams.maxAccumulationSamples = static_cast<std::uint32_t>(m_maxAccumulationSamplesUi);
    ImGui::Text("Frame index: %u", m_renderer.frameIndex());
    if (m_renderer.frameIndex() >= m_renderParams.maxAccumulationSamples) {
        ImGui::Text("Accumulation paused (sample cap reached)");
    }

    const voxelsprout::render::GpuTimingInfo& timings = m_renderer.gpuTimings();
    ImGui::Text("GPU cloud: %.3f ms", timings.cloudPathTraceMs);
    ImGui::Text("GPU tone map: %.3f ms", timings.toneMapMs);
    ImGui::Text("GPU total: %.3f ms", timings.totalMs);

    ImGui::Text("Camera locked for parameter tuning. ESC quit.");
    m_uiInteracting = ImGui::IsAnyItemActive();
    if (m_uiInteracting) {
        ImGui::Text("Interactive preview: ON");
    }
    ImGui::End();
}

void App::run() {
    using clock = std::chrono::steady_clock;
    auto prevTime = clock::now();

    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();

        const auto now = clock::now();
        const float dt = std::chrono::duration<float>(now - prevTime).count();
        (void)dt;
        prevTime = now;

        pollInput();

        m_renderer.beginUiFrame();
        buildUi();

        if (m_uiInteracting) {
            m_interactionCooldownFrames = kInteractivePreviewCooldownFrames;
        } else if (m_interactionCooldownFrames > 0) {
            m_interactionCooldownFrames -= 1;
        }

        const bool interactivePreview = m_uiInteracting || (m_interactionCooldownFrames > 0);
        voxelsprout::render::RenderParameters frameParams = m_renderParams;
        if (interactivePreview) {
            frameParams.scene.volume.maxBounces = std::min(frameParams.scene.volume.maxBounces, kInteractivePreviewMaxBounces);
            frameParams.cloudUpdateInterval = 1;
            if (m_uiInteracting) {
                frameParams.enableAccumulation = false;
                frameParams.forceReset = true;
            }
        }

        if (!m_renderer.renderFrame(frameParams)) {
            VOX_LOGE("app") << "render frame failed";
            break;
        }

        m_renderParams.forceReset = false;
    }
}

void App::shutdown() {
    m_renderer.shutdown();
    if (m_window != nullptr) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

} // namespace voxelsprout::app
