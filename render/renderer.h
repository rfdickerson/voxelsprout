#pragma once

#include "core/camera.h"
#include "scene/scene.h"

#include <cstdint>

struct GLFWwindow;

namespace voxelsprout::render {

struct RenderParameters {
    voxelsprout::core::Camera camera{};
    voxelsprout::scene::SceneState scene{};
    float exposure = 1.0f;
    bool enableAccumulation = true;
    bool forceReset = false;
    std::uint32_t cloudUpdateInterval = 2;
    std::uint32_t maxAccumulationSamples = 256;
};

struct GpuTimingInfo {
    float cloudPathTraceMs = 0.0f;
    float toneMapMs = 0.0f;
    float totalMs = 0.0f;
};

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool init(GLFWwindow* window);
    void beginUiFrame();
    bool renderFrame(const RenderParameters& parameters);
    void shutdown();

    [[nodiscard]] std::uint32_t frameIndex() const;
    [[nodiscard]] const GpuTimingInfo& gpuTimings() const;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};

} // namespace voxelsprout::render
