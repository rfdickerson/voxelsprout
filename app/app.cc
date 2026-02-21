#include "app/app.h"

#include "core/log.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace voxelsprout::app {
namespace {

void applyCloudPreset(int presetIndex, voxelsprout::render::RenderParameters& params) {
    auto& volume = params.scene.volume;
    switch (presetIndex) {
    case 0: // Cumulus Soft
        volume.densityScale = 0.85f;
        volume.anisotropyG = 0.78f;
        volume.albedo = 0.995f;
        volume.macroScale = 0.28f;
        volume.detailScale = 0.62f;
        volume.densityCutoff = 0.09f;
        volume.chunkiness = 0.65f;
        volume.coverage = 0.55f;
        volume.weatherScale = 0.07f;
        volume.cloudBase = 1.8f;
        volume.cloudTop = 8.9f;
        volume.warpStrength = 0.6f;
        volume.erosionStrength = 0.42f;
        volume.brightnessBoost = 2.15f;
        volume.ambientLift = 0.70f;
        volume.maxBounces = 2;
        params.scene.sun.intensity = 18.0f;
        break;
    case 1: // Cumulus Chunky
        volume.densityScale = 1.00f;
        volume.anisotropyG = 0.85f;
        volume.albedo = 0.995f;
        volume.macroScale = 0.24f;
        volume.detailScale = 0.42f;
        volume.densityCutoff = 0.04f;
        volume.chunkiness = 1.20f;
        volume.coverage = 0.68f;
        volume.weatherScale = 0.08f;
        volume.cloudBase = 1.4f;
        volume.cloudTop = 9.2f;
        volume.warpStrength = 1.05f;
        volume.erosionStrength = 0.58f;
        volume.brightnessBoost = 2.60f;
        volume.ambientLift = 0.75f;
        volume.maxBounces = 3;
        params.scene.sun.intensity = 20.0f;
        break;
    case 2: // Storm
    default:
        volume.densityScale = 1.45f;
        volume.anisotropyG = 0.88f;
        volume.albedo = 0.93f;
        volume.macroScale = 0.22f;
        volume.detailScale = 0.34f;
        volume.densityCutoff = 0.01f;
        volume.chunkiness = 1.55f;
        volume.coverage = 0.85f;
        volume.weatherScale = 0.10f;
        volume.cloudBase = 0.8f;
        volume.cloudTop = 9.8f;
        volume.warpStrength = 1.25f;
        volume.erosionStrength = 0.25f;
        volume.brightnessBoost = 1.40f;
        volume.ambientLift = 0.28f;
        volume.maxBounces = 4;
        params.scene.sun.intensity = 14.0f;
        break;
    }
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

    m_renderParams.camera.position = {0.0f, 2.0f, 10.0f};
    m_renderParams.camera.yawDegrees = -90.0f;
    m_renderParams.camera.pitchDegrees = 12.0f;
    m_renderParams.camera.fovDegrees = 75.0f;
    m_renderParams.scene.sun.direction = {1.0f, 1.0f, 0.5f};
    m_renderParams.scene.sun.intensity = 20.0f;
    m_renderParams.scene.volume.densityScale = 1.0f;
    m_renderParams.scene.volume.anisotropyG = 0.85f;
    m_renderParams.scene.volume.albedo = 0.995f;
    m_renderParams.scene.volume.macroScale = 0.24f;
    m_renderParams.scene.volume.detailScale = 0.42f;
    m_renderParams.scene.volume.densityCutoff = 0.04f;
    m_renderParams.scene.volume.chunkiness = 1.2f;
    m_renderParams.scene.volume.coverage = 0.68f;
    m_renderParams.scene.volume.weatherScale = 0.08f;
    m_renderParams.scene.volume.cloudBase = 1.4f;
    m_renderParams.scene.volume.cloudTop = 9.2f;
    m_renderParams.scene.volume.warpStrength = 1.05f;
    m_renderParams.scene.volume.erosionStrength = 0.58f;
    m_renderParams.scene.volume.brightnessBoost = 2.6f;
    m_renderParams.scene.volume.ambientLift = 0.75f;
    m_renderParams.scene.volume.maxBounces = 3;
    m_renderParams.cloudUpdateInterval = 2;
    m_renderParams.maxAccumulationSamples = 256;
    m_maxAccumulationSamplesUi = 256;
    m_cloudPresetUi = 1;
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
    if (!m_showUi) {
        return;
    }

    ImGui::Begin("Compute Renderer");
    ImGui::Text("CloudPathTracePass + ToneMapPass");
    ImGui::Separator();

    ImGui::Checkbox("Progressive accumulation", &m_renderParams.enableAccumulation);
    ImGui::SliderFloat("Exposure", &m_renderParams.exposure, 0.1f, 4.0f, "%.2f");
    const char* presetLabels[] = {"Cumulus Soft", "Cumulus Chunky", "Storm"};
    const int prevPreset = m_cloudPresetUi;
    ImGui::Combo("Cloud preset", &m_cloudPresetUi, presetLabels, IM_ARRAYSIZE(presetLabels));
    if (m_cloudPresetUi != prevPreset) {
        applyCloudPreset(m_cloudPresetUi, m_renderParams);
        m_renderParams.forceReset = true;
    }
    ImGui::SliderFloat("Density scale", &m_renderParams.scene.volume.densityScale, 0.1f, 3.0f, "%.2f");
    ImGui::SliderFloat("g parameter", &m_renderParams.scene.volume.anisotropyG, 0.0f, 0.95f, "%.2f");
    m_renderParams.scene.volume.anisotropyG = std::clamp(m_renderParams.scene.volume.anisotropyG, 0.0f, 0.95f);
    ImGui::SliderFloat("Cloud albedo", &m_renderParams.scene.volume.albedo, 0.9f, 1.0f, "%.3f");
    ImGui::SliderFloat("Chunkiness", &m_renderParams.scene.volume.chunkiness, 0.2f, 2.5f, "%.2f");
    ImGui::SliderFloat("Macro scale", &m_renderParams.scene.volume.macroScale, 0.08f, 0.60f, "%.2f");
    ImGui::SliderFloat("Detail scale", &m_renderParams.scene.volume.detailScale, 0.16f, 1.20f, "%.2f");
    ImGui::SliderFloat("Coverage", &m_renderParams.scene.volume.coverage, 0.1f, 0.98f, "%.2f");
    ImGui::SliderFloat("Weather scale", &m_renderParams.scene.volume.weatherScale, 0.02f, 0.20f, "%.2f");
    ImGui::SliderFloat("Cloud base", &m_renderParams.scene.volume.cloudBase, -1.0f, 6.0f, "%.2f");
    ImGui::SliderFloat("Cloud top", &m_renderParams.scene.volume.cloudTop, 4.0f, 14.0f, "%.2f");
    ImGui::SliderFloat("Warp strength", &m_renderParams.scene.volume.warpStrength, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Erosion strength", &m_renderParams.scene.volume.erosionStrength, 0.0f, 1.5f, "%.2f");
    ImGui::SliderFloat("Density cutoff", &m_renderParams.scene.volume.densityCutoff, 0.0f, 0.3f, "%.2f");
    ImGui::SliderFloat("Cloud brightness", &m_renderParams.scene.volume.brightnessBoost, 0.5f, 5.0f, "%.2f");
    ImGui::SliderFloat("Ambient lift", &m_renderParams.scene.volume.ambientLift, 0.0f, 1.5f, "%.2f");
    ImGui::SliderInt("Max bounces", &m_renderParams.scene.volume.maxBounces, 1, 6);
    m_renderParams.scene.volume.albedo = std::clamp(m_renderParams.scene.volume.albedo, 0.9f, 1.0f);
    m_renderParams.scene.volume.cloudTop =
        std::max(m_renderParams.scene.volume.cloudTop, m_renderParams.scene.volume.cloudBase + 0.25f);
    m_renderParams.scene.volume.maxBounces = std::clamp(m_renderParams.scene.volume.maxBounces, 1, 6);
    ImGui::SliderFloat("Sun intensity", &m_renderParams.scene.sun.intensity, 1.0f, 40.0f, "%.2f");
    ImGui::SliderFloat3(
        "Sun direction",
        &m_renderParams.scene.sun.direction.x,
        -1.0f,
        1.0f,
        "%.2f");
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

        if (!m_renderer.renderFrame(m_renderParams)) {
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
