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

#include "render/backend/vulkan/frame_math.h"

namespace voxelsprout::render {

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "render/renderer_shared.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

void RendererBackend::renderFrame(
    const voxelsprout::world::ChunkGrid& chunkGrid,
    const voxelsprout::sim::Simulation& simulation,
    const CameraPose& camera,
    const VoxelPreview& preview,
    float simulationAlpha,
    std::span<const std::size_t> visibleChunkIndices
) {
    const auto cpuFrameStartTime = std::chrono::steady_clock::now();
    float cpuWaitMs = 0.0f;

    if (m_device == VK_NULL_HANDLE || m_swapchain == VK_NULL_HANDLE) {
        return;
    }
    if (m_window != nullptr && glfwWindowShouldClose(m_window) == GLFW_TRUE) {
        return;
    }

    const double frameNowSeconds = glfwGetTime();
    float frameDeltaSeconds = 1.0f / 60.0f;
    if (m_lastFrameTimestampSeconds > 0.0) {
        const double deltaSeconds = std::max(0.0, frameNowSeconds - m_lastFrameTimestampSeconds);
        frameDeltaSeconds = static_cast<float>(deltaSeconds);
        m_debugFps = (deltaSeconds > 0.0) ? static_cast<float>(1.0 / deltaSeconds) : 0.0f;
    }
    m_lastFrameTimestampSeconds = frameNowSeconds;
    if (!m_debugCameraFovInitialized) {
        m_debugCameraFovDegrees = camera.fovDegrees;
        m_debugCameraFovInitialized = true;
    }
    m_debugCameraFovDegrees = std::clamp(m_debugCameraFovDegrees, 20.0f, 120.0f);
    const float activeFovDegrees = m_debugCameraFovDegrees;

    m_debugChunkCount = static_cast<std::uint32_t>(chunkGrid.chunks().size());
    m_debugMacroCellUniformCount = 0;
    m_debugMacroCellRefined4Count = 0;
    m_debugMacroCellRefined1Count = 0;
    for (const voxelsprout::world::Chunk& chunk : chunkGrid.chunks()) {
        for (int my = 0; my < voxelsprout::world::Chunk::kMacroSizeY; ++my) {
            for (int mz = 0; mz < voxelsprout::world::Chunk::kMacroSizeZ; ++mz) {
                for (int mx = 0; mx < voxelsprout::world::Chunk::kMacroSizeX; ++mx) {
                    const voxelsprout::world::Chunk::MacroCell cell = chunk.macroCellAt(mx, my, mz);
                    switch (cell.resolution) {
                    case voxelsprout::world::Chunk::CellResolution::Uniform:
                        ++m_debugMacroCellUniformCount;
                        break;
                    case voxelsprout::world::Chunk::CellResolution::Refined4:
                        ++m_debugMacroCellRefined4Count;
                        break;
                    case voxelsprout::world::Chunk::CellResolution::Refined1:
                        ++m_debugMacroCellRefined1Count;
                        break;
                    }
                }
            }
        }
    }
    // FrameGraph scaffold: record the pass DAG now, then replace direct sequencing incrementally.
    m_frameGraph.reset();
    const FrameGraph::PassId fgShadow = m_frameGraph.addPass({"shadow", FrameGraphQueue::Graphics});
    const FrameGraph::PassId fgGiSurface = m_frameGraph.addPass({"gi_surface", FrameGraphQueue::Compute});
    const FrameGraph::PassId fgGiInject = m_frameGraph.addPass({"gi_inject", FrameGraphQueue::Compute});
    const FrameGraph::PassId fgGiPropagate = m_frameGraph.addPass({"gi_propagate", FrameGraphQueue::Compute});
    const FrameGraph::PassId fgAutoExposure = m_frameGraph.addPass({"auto_exposure", FrameGraphQueue::Compute});
    const FrameGraph::PassId fgSunShafts = m_frameGraph.addPass({"sun_shafts", FrameGraphQueue::Compute});
    const FrameGraph::PassId fgPrepass = m_frameGraph.addPass({"prepass", FrameGraphQueue::Graphics});
    const FrameGraph::PassId fgSsao = m_frameGraph.addPass({"ssao", FrameGraphQueue::Graphics});
    const FrameGraph::PassId fgSsaoBlur = m_frameGraph.addPass({"ssao_blur", FrameGraphQueue::Graphics});
    const FrameGraph::PassId fgMain = m_frameGraph.addPass({"main", FrameGraphQueue::Graphics});
    const FrameGraph::PassId fgPost = m_frameGraph.addPass({"post", FrameGraphQueue::Graphics});
    const FrameGraph::PassId fgUi = m_frameGraph.addPass({"imgui", FrameGraphQueue::Graphics});
    const FrameGraph::PassId fgPresent = m_frameGraph.addPass({"present", FrameGraphQueue::Graphics});

    m_frameGraph.addDependency(fgShadow, fgPrepass);
    m_frameGraph.addDependency(fgGiSurface, fgGiInject);
    m_frameGraph.addDependency(fgGiInject, fgGiPropagate);
    m_frameGraph.addDependency(fgGiPropagate, fgMain);
    m_frameGraph.addDependency(fgAutoExposure, fgPost);
    m_frameGraph.addDependency(fgSunShafts, fgPost);
    m_frameGraph.addDependency(fgPrepass, fgSsao);
    m_frameGraph.addDependency(fgSsao, fgSsaoBlur);
    m_frameGraph.addDependency(fgSsaoBlur, fgMain);
    m_frameGraph.addDependency(fgMain, fgPost);
    m_frameGraph.addDependency(fgPost, fgUi);
    m_frameGraph.addDependency(fgUi, fgPresent);

    collectCompletedBufferReleases();

    FrameResources& frame = m_frames[m_currentFrame];
    if (!isTimelineValueReached(m_frameTimelineValues[m_currentFrame])) {
        static double lastStallLogTimeSeconds = 0.0;
        uint64_t completedValue = 0;
        const VkResult counterResult =
            vkGetSemaphoreCounterValue(m_device, m_renderTimelineSemaphore, &completedValue);
        if (counterResult == VK_SUCCESS) {
            const uint64_t targetValue = m_frameTimelineValues[m_currentFrame];
            const uint64_t lag = (targetValue > completedValue) ? (targetValue - completedValue) : 0u;
            const double nowSeconds = glfwGetTime();
            if (lag >= kFrameTimelineWarnLagThreshold &&
                (nowSeconds - lastStallLogTimeSeconds) >= kFrameTimelineWarnCooldownSeconds) {
                VOX_LOGW("render")
                    << "frame slot stalled on timeline value "
                    << targetValue
                    << ", completed=" << completedValue
                    << ", lag=" << lag
                    << ", frameIndex=" << m_currentFrame;
                lastStallLogTimeSeconds = nowSeconds;
            }
        } else {
            logVkFailure("vkGetSemaphoreCounterValue(stuckFrame)", counterResult);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return;
    }
    if (m_frameTimelineValues[m_currentFrame] > 0) {
        readGpuTimestampResults(m_currentFrame);
    }
    if (m_transferCommandBufferInFlightValue > 0) {
        if (isTimelineValueReached(m_transferCommandBufferInFlightValue)) {
            m_transferCommandBufferInFlightValue = 0;
            m_pendingTransferTimelineValue = 0;
            collectCompletedBufferReleases();
        }
    }
    m_frameArena.beginFrame(m_currentFrame);

    if (m_chunkMeshRebuildRequested || !m_pendingChunkRemeshIndices.empty()) {
        // Avoid CPU stalls when async transfer is still in flight.
        if (m_transferCommandBufferInFlightValue == 0 ||
            isTimelineValueReached(m_transferCommandBufferInFlightValue)) {
            const std::span<const std::size_t> pendingRemeshIndices =
                m_chunkMeshRebuildRequested
                    ? std::span<const std::size_t>{}
                    : std::span<const std::size_t>(m_pendingChunkRemeshIndices.data(), m_pendingChunkRemeshIndices.size());
            if (createChunkBuffers(chunkGrid, pendingRemeshIndices)) {
                m_chunkMeshRebuildRequested = false;
                m_pendingChunkRemeshIndices.clear();
            } else {
                VOX_LOGE("render") << "failed deferred chunk remesh";
            }
        }
    }

    uint32_t imageIndex = 0;
    const auto acquireStartTime = std::chrono::steady_clock::now();
    const VkResult acquireResult = vkAcquireNextImageKHR(
        m_device,
        m_swapchain,
        kAcquireNextImageTimeoutNs,
        frame.imageAvailable,
        VK_NULL_HANDLE,
        &imageIndex
    );
    cpuWaitMs += static_cast<float>(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - acquireStartTime).count()
    );

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        VOX_LOGI("render") << "swapchain out of date during acquire, recreating\n";
        recreateSwapchain();
        return;
    }
    if (acquireResult == VK_TIMEOUT) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        logVkFailure("vkAcquireNextImageKHR", acquireResult);
        return;
    }

    const VkSemaphore renderFinishedSemaphore = m_renderFinishedSemaphores[imageIndex];
    const uint32_t aoFrameIndex = m_currentFrame % kMaxFramesInFlight;

    vkResetCommandPool(m_device, frame.commandPool, 0);

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = frame.commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &allocateInfo, &commandBuffer) != VK_SUCCESS) {
        VOX_LOGE("render") << "vkAllocateCommandBuffers failed\n";
        return;
    }
    {
        const std::string commandBufferName = "frame." + std::to_string(m_currentFrame) + ".graphics.commandBuffer";
        setObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER, vkHandleToUint64(commandBuffer), commandBufferName.c_str());
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        VOX_LOGE("render") << "vkBeginCommandBuffer failed\n";
        return;
    }
    const VkQueryPool gpuTimestampQueryPool =
        m_gpuTimestampsSupported ? m_gpuTimestampQueryPools[m_currentFrame] : VK_NULL_HANDLE;
    auto writeGpuTimestampTop = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            gpuTimestampQueryPool,
            queryIndex
        );
    };
    auto writeGpuTimestampBottom = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            gpuTimestampQueryPool,
            queryIndex
        );
    };
    if (gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(commandBuffer, gpuTimestampQueryPool, 0, kGpuTimestampQueryCount);
        writeGpuTimestampTop(kGpuTimestampQueryFrameStart);
    }
    beginDebugLabel(commandBuffer, "Frame", 0.22f, 0.22f, 0.26f, 1.0f);
    if (m_imguiInitialized) {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        buildFrameStatsUi();
        buildMeshingDebugUi();
        buildShadowDebugUi();
        buildSunDebugUi();
        m_debugUiVisible = m_showMeshingPanel || m_showShadowPanel || m_showSunPanel;
        buildAimReticleUi();
        ImGui::Render();
    }
    // Keep previous frame counters visible in UI, then reset for this frame's capture.
    m_debugDrawnLod0Ranges = 0;
    m_debugDrawnLod1Ranges = 0;
    m_debugDrawnLod2Ranges = 0;
    m_debugChunkIndirectCommandCount = 0;
    m_debugDrawCallsTotal = 0;
    m_debugDrawCallsShadow = 0;
    m_debugDrawCallsPrepass = 0;
    m_debugDrawCallsMain = 0;
    m_debugDrawCallsPost = 0;

    const float aspectRatio = static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);
    const float nearPlane = 0.1f;
    const float farPlane = 500.0f;
    const float halfFovRadians = voxelsprout::math::radians(activeFovDegrees) * 0.5f;
    const float tanHalfFov = std::tan(halfFovRadians);
    const voxelsprout::math::Vector3 eye{camera.x, camera.y, camera.z};
    const CameraFrameDerived cameraFrame = ComputeCameraFrame(camera);
    const int cameraChunkX = cameraFrame.chunkX;
    const int cameraChunkY = cameraFrame.chunkY;
    const int cameraChunkZ = cameraFrame.chunkZ;
    const voxelsprout::math::Vector3 forward = cameraFrame.forward;

    const voxelsprout::math::Matrix4 view = lookAt(eye, eye + forward, voxelsprout::math::Vector3{0.0f, 1.0f, 0.0f});
    const voxelsprout::math::Matrix4 projection = perspectiveVulkan(voxelsprout::math::radians(activeFovDegrees), aspectRatio, nearPlane, farPlane);
    const voxelsprout::math::Matrix4 mvp = projection * view;
    const voxelsprout::math::Matrix4 mvpColumnMajor = transpose(mvp);
    const voxelsprout::math::Matrix4 viewColumnMajor = transpose(view);
    const voxelsprout::math::Matrix4 projectionColumnMajor = transpose(projection);

    const bool projectionParamsChanged =
        std::abs(m_shadowStableAspectRatio - aspectRatio) > 0.0001f ||
        std::abs(m_shadowStableFovDegrees - activeFovDegrees) > 0.0001f;
    if (projectionParamsChanged) {
        m_shadowStableAspectRatio = aspectRatio;
        m_shadowStableFovDegrees = activeFovDegrees;
        m_shadowStableCascadeRadii.fill(0.0f);
    }

    voxelsprout::math::Vector3 sunDirection = voxelsprout::math::normalize(ComputeSunDirection(
        m_skyDebugSettings.sunYawDegrees,
        m_skyDebugSettings.sunPitchDegrees
    ));
    const voxelsprout::math::Vector3 toSun = -voxelsprout::math::normalize(sunDirection);
    const float sunElevationDegrees = voxelsprout::math::degrees(std::asin(std::clamp(toSun.y, -1.0f, 1.0f)));

    SkyTuningSample manualTuning{};
    manualTuning.rayleighStrength = m_skyDebugSettings.rayleighStrength;
    manualTuning.mieStrength = m_skyDebugSettings.mieStrength;
    manualTuning.mieAnisotropy = m_skyDebugSettings.mieAnisotropy;
    manualTuning.skyExposure = m_skyDebugSettings.skyExposure;
    manualTuning.sunDiskIntensity = m_skyDebugSettings.sunDiskIntensity;
    manualTuning.sunHaloIntensity = m_skyDebugSettings.sunHaloIntensity;
    manualTuning.sunDiskSize = m_skyDebugSettings.sunDiskSize;
    manualTuning.sunHazeFalloff = m_skyDebugSettings.sunHazeFalloff;

    SkyTuningSample targetTuning = manualTuning;
    if (m_skyDebugSettings.autoSunriseTuning) {
        const SkyTuningSample autoTuning = evaluateSunriseSkyTuning(sunElevationDegrees);
        targetTuning = blendSkyTuningSample(manualTuning, autoTuning, m_skyDebugSettings.autoSunriseBlend);
    }

    if (!m_skyDebugSettings.autoSunriseTuning || m_skyDebugSettings.autoSunriseBlend <= 0.0f) {
        m_skyTuningRuntime.initialized = true;
        m_skyTuningRuntime.rayleighStrength = targetTuning.rayleighStrength;
        m_skyTuningRuntime.mieStrength = targetTuning.mieStrength;
        m_skyTuningRuntime.mieAnisotropy = targetTuning.mieAnisotropy;
        m_skyTuningRuntime.skyExposure = targetTuning.skyExposure;
        m_skyTuningRuntime.sunDiskIntensity = targetTuning.sunDiskIntensity;
        m_skyTuningRuntime.sunHaloIntensity = targetTuning.sunHaloIntensity;
        m_skyTuningRuntime.sunDiskSize = targetTuning.sunDiskSize;
        m_skyTuningRuntime.sunHazeFalloff = targetTuning.sunHazeFalloff;
    } else if (!m_skyTuningRuntime.initialized) {
        m_skyTuningRuntime.initialized = true;
        m_skyTuningRuntime.rayleighStrength = targetTuning.rayleighStrength;
        m_skyTuningRuntime.mieStrength = targetTuning.mieStrength;
        m_skyTuningRuntime.mieAnisotropy = targetTuning.mieAnisotropy;
        m_skyTuningRuntime.skyExposure = targetTuning.skyExposure;
        m_skyTuningRuntime.sunDiskIntensity = targetTuning.sunDiskIntensity;
        m_skyTuningRuntime.sunHaloIntensity = targetTuning.sunHaloIntensity;
        m_skyTuningRuntime.sunDiskSize = targetTuning.sunDiskSize;
        m_skyTuningRuntime.sunHazeFalloff = targetTuning.sunHazeFalloff;
    } else {
        const float adaptSpeed = std::max(m_skyDebugSettings.autoSunriseAdaptSpeed, 0.01f);
        const float alpha = 1.0f - std::exp(-std::max(frameDeltaSeconds, 0.0f) * adaptSpeed);
        m_skyTuningRuntime.rayleighStrength =
            std::lerp(m_skyTuningRuntime.rayleighStrength, targetTuning.rayleighStrength, alpha);
        m_skyTuningRuntime.mieStrength = std::lerp(m_skyTuningRuntime.mieStrength, targetTuning.mieStrength, alpha);
        m_skyTuningRuntime.mieAnisotropy =
            std::lerp(m_skyTuningRuntime.mieAnisotropy, targetTuning.mieAnisotropy, alpha);
        m_skyTuningRuntime.skyExposure = std::lerp(m_skyTuningRuntime.skyExposure, targetTuning.skyExposure, alpha);
        m_skyTuningRuntime.sunDiskIntensity =
            std::lerp(m_skyTuningRuntime.sunDiskIntensity, targetTuning.sunDiskIntensity, alpha);
        m_skyTuningRuntime.sunHaloIntensity =
            std::lerp(m_skyTuningRuntime.sunHaloIntensity, targetTuning.sunHaloIntensity, alpha);
        m_skyTuningRuntime.sunDiskSize = std::lerp(m_skyTuningRuntime.sunDiskSize, targetTuning.sunDiskSize, alpha);
        m_skyTuningRuntime.sunHazeFalloff =
            std::lerp(m_skyTuningRuntime.sunHazeFalloff, targetTuning.sunHazeFalloff, alpha);
    }

    SkyDebugSettings effectiveSkySettings = m_skyDebugSettings;
    effectiveSkySettings.rayleighStrength = m_skyTuningRuntime.rayleighStrength;
    effectiveSkySettings.mieStrength = m_skyTuningRuntime.mieStrength;
    effectiveSkySettings.mieAnisotropy = m_skyTuningRuntime.mieAnisotropy;
    effectiveSkySettings.skyExposure = m_skyTuningRuntime.skyExposure;
    effectiveSkySettings.sunDiskIntensity = m_skyTuningRuntime.sunDiskIntensity;
    effectiveSkySettings.sunHaloIntensity = m_skyTuningRuntime.sunHaloIntensity;
    effectiveSkySettings.sunDiskSize = m_skyTuningRuntime.sunDiskSize;
    effectiveSkySettings.sunHazeFalloff = m_skyTuningRuntime.sunHazeFalloff;
    const bool isNight = sunElevationDegrees <= 0.0f;
    if (isNight) {
        // Hard night mode: low, cool ambient sky and no direct sun disk/halo.
        effectiveSkySettings.rayleighStrength = 0.12f;
        effectiveSkySettings.mieStrength = 0.015f;
        effectiveSkySettings.skyExposure = 0.14f;
        effectiveSkySettings.sunDiskIntensity = 0.0f;
        effectiveSkySettings.sunHaloIntensity = 0.0f;
    }

    const voxelsprout::math::Vector3 sunColor = isNight
        ? voxelsprout::math::Vector3{0.0f, 0.0f, 0.0f}
        : computeSunColor(effectiveSkySettings, sunDirection);

    constexpr float kCascadeLambda = 0.70f;
    constexpr float kCascadeSplitQuantization = 0.5f;
    constexpr float kCascadeSplitUpdateThreshold = 0.5f;
    std::array<float, kShadowCascadeCount> cascadeDistances{};
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        const float p = static_cast<float>(cascadeIndex + 1) / static_cast<float>(kShadowCascadeCount);
        const float logarithmicSplit = nearPlane * std::pow(farPlane / nearPlane, p);
        const float uniformSplit = nearPlane + ((farPlane - nearPlane) * p);
        const float desiredSplit =
            (kCascadeLambda * logarithmicSplit) + ((1.0f - kCascadeLambda) * uniformSplit);
        const float quantizedSplit =
            std::round(desiredSplit / kCascadeSplitQuantization) * kCascadeSplitQuantization;

        float split = m_shadowCascadeSplits[cascadeIndex];
        if (projectionParamsChanged || std::abs(quantizedSplit - split) > kCascadeSplitUpdateThreshold) {
            split = quantizedSplit;
        }

        const float previousSplit = (cascadeIndex == 0) ? nearPlane : m_shadowCascadeSplits[cascadeIndex - 1];
        split = std::max(split, previousSplit + kCascadeSplitQuantization);
        split = std::min(split, farPlane);
        m_shadowCascadeSplits[cascadeIndex] = split;
        cascadeDistances[cascadeIndex] = split;
    }

    std::array<voxelsprout::math::Matrix4, kShadowCascadeCount> lightViewProjMatrices{};
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        const float cascadeFar = cascadeDistances[cascadeIndex];
        const float farHalfHeight = cascadeFar * tanHalfFov;
        const float farHalfWidth = farHalfHeight * aspectRatio;

        // Camera-position-only cascades: only translation moves cascade centers; rotation does not.
        const voxelsprout::math::Vector3 frustumCenter = eye;
        float boundingRadius =
            std::sqrt((cascadeFar * cascadeFar) + (farHalfWidth * farHalfWidth) + (farHalfHeight * farHalfHeight));
        boundingRadius = std::max(boundingRadius * 1.04f, 24.0f);
        boundingRadius = std::ceil(boundingRadius * 16.0f) / 16.0f;
        if (m_shadowStableCascadeRadii[cascadeIndex] <= 0.0f) {
            m_shadowStableCascadeRadii[cascadeIndex] = boundingRadius;
        }
        const float cascadeRadius = m_shadowStableCascadeRadii[cascadeIndex];
        const float orthoWidth = 2.0f * cascadeRadius;
        const float texelSize = orthoWidth / static_cast<float>(kShadowCascadeResolution[cascadeIndex]);

        // Keep the light farther than the cascade sphere but avoid overly large depth spans.
        const float lightDistance = (cascadeRadius * 1.9f) + 48.0f;
        const float sunUpDot = std::abs(voxelsprout::math::dot(sunDirection, voxelsprout::math::Vector3{0.0f, 1.0f, 0.0f}));
        const voxelsprout::math::Vector3 lightUpHint =
            (sunUpDot > 0.95f) ? voxelsprout::math::Vector3{0.0f, 0.0f, 1.0f} : voxelsprout::math::Vector3{0.0f, 1.0f, 0.0f};
        const voxelsprout::math::Vector3 lightForward = voxelsprout::math::normalize(sunDirection);
        const voxelsprout::math::Vector3 lightRight = voxelsprout::math::normalize(voxelsprout::math::cross(lightForward, lightUpHint));
        const voxelsprout::math::Vector3 lightUp = voxelsprout::math::cross(lightRight, lightForward);

        // Stabilize translation by snapping the cascade center along light-view right/up texel units
        // before constructing the view matrix.
        const float centerRight = voxelsprout::math::dot(frustumCenter, lightRight);
        const float centerUp = voxelsprout::math::dot(frustumCenter, lightUp);
        const float snappedCenterRight = std::floor((centerRight / texelSize) + 0.5f) * texelSize;
        const float snappedCenterUp = std::floor((centerUp / texelSize) + 0.5f) * texelSize;
        const voxelsprout::math::Vector3 snappedFrustumCenter =
            frustumCenter +
            (lightRight * (snappedCenterRight - centerRight)) +
            (lightUp * (snappedCenterUp - centerUp));

        const voxelsprout::math::Vector3 lightPosition = snappedFrustumCenter - (lightForward * lightDistance);
        const voxelsprout::math::Matrix4 lightView = lookAt(lightPosition, snappedFrustumCenter, lightUp);

        const float left = -cascadeRadius;
        const float right = cascadeRadius;
        const float bottom = -cascadeRadius;
        const float top = cascadeRadius;
        // Keep a stable but tighter depth range per cascade to improve depth precision.
        const float casterPadding = std::max(24.0f, cascadeRadius * 0.35f);
        const float lightNear = std::max(0.1f, lightDistance - cascadeRadius - casterPadding);
        const float lightFar = lightDistance + cascadeRadius + casterPadding;
        const voxelsprout::math::Matrix4 lightProjection = orthographicVulkan(
            left,
            right,
            bottom,
            top,
            lightNear,
            lightFar
        );
        lightViewProjMatrices[cascadeIndex] = lightProjection * lightView;
    }

    std::array<voxelsprout::math::Vector3, 9> shIrradiance{};
    if (!isNight) {
        shIrradiance = computeIrradianceShCoefficients(sunDirection, sunColor, effectiveSkySettings);
    } else {
        for (voxelsprout::math::Vector3& coefficient : shIrradiance) {
            coefficient = voxelsprout::math::Vector3{0.0f, 0.0f, 0.0f};
        }
        // Constant dark-blue ambient irradiance for night.
        constexpr float kShY00 = 0.282095f;
        const voxelsprout::math::Vector3 nightAmbientIrradiance{0.050f, 0.078f, 0.155f};
        shIrradiance[0] = nightAmbientIrradiance * (1.0f / kShY00);
    }

    const std::optional<FrameArenaSlice> mvpSliceOpt =
        m_frameArena.allocateUpload(
            sizeof(CameraUniform),
            m_uniformBufferAlignment,
            FrameArenaUploadKind::CameraUniform
        );
    if (!mvpSliceOpt.has_value() || mvpSliceOpt->mapped == nullptr) {
        VOX_LOGE("render") << "failed to allocate MVP uniform slice\n";
        return;
    }

    CameraUniform mvpUniform{};
    std::memcpy(mvpUniform.mvp, mvpColumnMajor.m, sizeof(mvpUniform.mvp));
    std::memcpy(mvpUniform.view, viewColumnMajor.m, sizeof(mvpUniform.view));
    std::memcpy(mvpUniform.proj, projectionColumnMajor.m, sizeof(mvpUniform.proj));
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        const voxelsprout::math::Matrix4 lightViewProjColumnMajor = transpose(lightViewProjMatrices[cascadeIndex]);
        std::memcpy(
            mvpUniform.lightViewProj[cascadeIndex],
            lightViewProjColumnMajor.m,
            sizeof(mvpUniform.lightViewProj[cascadeIndex])
        );
        mvpUniform.shadowCascadeSplits[cascadeIndex] = cascadeDistances[cascadeIndex];
        const ShadowAtlasRect atlasRect = kShadowAtlasRects[cascadeIndex];
        mvpUniform.shadowAtlasUvRects[cascadeIndex][0] = static_cast<float>(atlasRect.x) / static_cast<float>(kShadowAtlasSize);
        mvpUniform.shadowAtlasUvRects[cascadeIndex][1] = static_cast<float>(atlasRect.y) / static_cast<float>(kShadowAtlasSize);
        mvpUniform.shadowAtlasUvRects[cascadeIndex][2] = static_cast<float>(atlasRect.size) / static_cast<float>(kShadowAtlasSize);
        mvpUniform.shadowAtlasUvRects[cascadeIndex][3] = static_cast<float>(atlasRect.size) / static_cast<float>(kShadowAtlasSize);
    }
    mvpUniform.sunDirectionIntensity[0] = sunDirection.x;
    mvpUniform.sunDirectionIntensity[1] = sunDirection.y;
    mvpUniform.sunDirectionIntensity[2] = sunDirection.z;
    mvpUniform.sunDirectionIntensity[3] = isNight ? 0.0f : 2.2f;
    mvpUniform.sunColorShadow[0] = sunColor.x;
    mvpUniform.sunColorShadow[1] = sunColor.y;
    mvpUniform.sunColorShadow[2] = sunColor.z;
    mvpUniform.sunColorShadow[3] = 1.0f;
    for (uint32_t i = 0; i < shIrradiance.size(); ++i) {
        mvpUniform.shIrradiance[i][0] = shIrradiance[i].x;
        mvpUniform.shIrradiance[i][1] = shIrradiance[i].y;
        mvpUniform.shIrradiance[i][2] = shIrradiance[i].z;
        mvpUniform.shIrradiance[i][3] = 0.0f;
    }
    mvpUniform.shadowConfig0[0] = m_shadowDebugSettings.receiverNormalOffsetNear;
    mvpUniform.shadowConfig0[1] = m_shadowDebugSettings.receiverNormalOffsetFar;
    mvpUniform.shadowConfig0[2] = m_shadowDebugSettings.receiverBaseBiasNearTexel;
    mvpUniform.shadowConfig0[3] = m_shadowDebugSettings.receiverBaseBiasFarTexel;

    mvpUniform.shadowConfig1[0] = m_shadowDebugSettings.receiverSlopeBiasNearTexel;
    mvpUniform.shadowConfig1[1] = m_shadowDebugSettings.receiverSlopeBiasFarTexel;
    mvpUniform.shadowConfig1[2] = m_shadowDebugSettings.cascadeBlendMin;
    mvpUniform.shadowConfig1[3] = m_shadowDebugSettings.cascadeBlendFactor;

    mvpUniform.shadowConfig2[0] = m_shadowDebugSettings.ssaoRadius;
    mvpUniform.shadowConfig2[1] = m_shadowDebugSettings.ssaoBias;
    mvpUniform.shadowConfig2[2] = m_shadowDebugSettings.ssaoIntensity;
    constexpr float kVoxelGiInjectSunScale = 0.70f;
    constexpr float kVoxelGiInjectShScale = 0.95f;
    constexpr float kVoxelGiPropagateFrameDecay = 0.93f;
    constexpr float kVoxelGiAmbientRebalanceStrength = 0.95f;
    constexpr float kVoxelGiAmbientFloor = 0.55f;
    constexpr float kVoxelGiStrength = 0.70f;
    const float kVoxelGiPropagateDecay = std::pow(
        std::clamp(kVoxelGiPropagateFrameDecay, 0.0f, 1.0f),
        1.0f / static_cast<float>(kVoxelGiPropagationIterations)
    );
    mvpUniform.shadowConfig2[3] = kVoxelGiInjectSunScale;

    mvpUniform.shadowConfig3[0] = kVoxelGiInjectShScale;
    mvpUniform.shadowConfig3[1] = std::clamp(m_voxelGiDebugSettings.bounceStrength, 0.0f, 4.0f);
    mvpUniform.shadowConfig3[2] = std::clamp(m_voxelGiDebugSettings.diffusionSoftness, 0.0f, 1.0f);
    mvpUniform.shadowConfig3[3] = m_shadowDebugSettings.pcfRadius;

    // Reuse origin XYZ for fixed GI rebalance + debug mode to avoid enlarging camera UBO.
    mvpUniform.shadowVoxelGridOrigin[0] = kVoxelGiAmbientRebalanceStrength;
    mvpUniform.shadowVoxelGridOrigin[1] = kVoxelGiAmbientFloor;
    mvpUniform.shadowVoxelGridOrigin[2] =
        static_cast<float>(std::clamp(m_voxelGiDebugSettings.visualizationMode, 0, 4));
    // W channel remains AO enable: 1.0 enables vertex AO, 0.0 disables.
    mvpUniform.shadowVoxelGridOrigin[3] = m_debugEnableVertexAo ? 1.0f : 0.0f;

    // Reuse currently-unused XYZ channels to provide camera world position to shaders.
    mvpUniform.shadowVoxelGridSize[0] = camera.x;
    mvpUniform.shadowVoxelGridSize[1] = camera.y;
    mvpUniform.shadowVoxelGridSize[2] = camera.z;
    // Reuse unused W channel for AO debug mode:
    // 0.0 = SSAO off, 1.0 = SSAO on, 2.0 = visualize SSAO, 3.0 = visualize AO normals.
    if (m_debugVisualizeAoNormals) {
        mvpUniform.shadowVoxelGridSize[3] = 3.0f;
    } else if (m_debugVisualizeSsao) {
        mvpUniform.shadowVoxelGridSize[3] = 2.0f;
    } else {
        mvpUniform.shadowVoxelGridSize[3] = m_debugEnableSsao ? 1.0f : 0.0f;
    }

    mvpUniform.skyConfig0[0] = effectiveSkySettings.rayleighStrength;
    mvpUniform.skyConfig0[1] = effectiveSkySettings.mieStrength;
    mvpUniform.skyConfig0[2] = effectiveSkySettings.mieAnisotropy;
    mvpUniform.skyConfig0[3] = effectiveSkySettings.skyExposure;

    const float flowTimeSeconds = static_cast<float>(std::fmod(frameNowSeconds, 4096.0));
    mvpUniform.skyConfig1[0] = effectiveSkySettings.sunDiskIntensity;
    mvpUniform.skyConfig1[1] = effectiveSkySettings.sunHaloIntensity;
    mvpUniform.skyConfig1[2] = flowTimeSeconds;
    mvpUniform.skyConfig1[3] = 1.85f;
    mvpUniform.skyConfig2[0] = effectiveSkySettings.sunDiskSize;
    mvpUniform.skyConfig2[1] = effectiveSkySettings.sunHazeFalloff;
    mvpUniform.skyConfig2[2] = effectiveSkySettings.plantQuadDirectionality;
    mvpUniform.skyConfig2[3] = kVoxelGiPropagateDecay;
    mvpUniform.skyConfig3[0] = std::clamp(m_skyDebugSettings.bloomThreshold, 0.0f, 16.0f);
    mvpUniform.skyConfig3[1] = std::clamp(m_skyDebugSettings.bloomSoftKnee, 0.0f, 1.0f);
    mvpUniform.skyConfig3[2] = std::clamp(m_skyDebugSettings.bloomBaseIntensity, 0.0f, 2.0f);
    mvpUniform.skyConfig3[3] = std::clamp(m_skyDebugSettings.bloomSunFacingBoost, 0.0f, 2.0f);
    mvpUniform.skyConfig4[0] = std::clamp(m_skyDebugSettings.volumetricFogDensity, 0.0f, 1.0f);
    mvpUniform.skyConfig4[1] = std::clamp(m_skyDebugSettings.volumetricFogHeightFalloff, 0.0f, 1.0f);
    mvpUniform.skyConfig4[2] = m_skyDebugSettings.volumetricFogBaseHeight;
    mvpUniform.skyConfig4[3] = std::clamp(m_skyDebugSettings.volumetricSunScattering, 0.0f, 8.0f);
    const bool autoExposureEnabled = m_skyDebugSettings.autoExposureEnabled && m_autoExposureComputeAvailable;
    mvpUniform.skyConfig5[0] = autoExposureEnabled ? 1.0f : 0.0f;
    mvpUniform.skyConfig5[1] = std::clamp(m_skyDebugSettings.manualExposure, 0.05f, 8.0f);
    mvpUniform.skyConfig5[2] = 0.0f;
    mvpUniform.skyConfig5[3] = 0.0f;
    mvpUniform.colorGrading0[0] = std::clamp(m_skyDebugSettings.colorGradingWhiteBalanceR, 0.0f, 4.0f);
    mvpUniform.colorGrading0[1] = std::clamp(m_skyDebugSettings.colorGradingWhiteBalanceG, 0.0f, 4.0f);
    mvpUniform.colorGrading0[2] = std::clamp(m_skyDebugSettings.colorGradingWhiteBalanceB, 0.0f, 4.0f);
    mvpUniform.colorGrading0[3] = std::clamp(m_skyDebugSettings.colorGradingContrast, 0.70f, 1.40f);
    mvpUniform.colorGrading1[0] = std::clamp(m_skyDebugSettings.colorGradingSaturation, 0.0f, 2.0f);
    mvpUniform.colorGrading1[1] = std::clamp(m_skyDebugSettings.colorGradingVibrance, -1.0f, 1.0f);
    mvpUniform.colorGrading1[2] = 0.0f;
    mvpUniform.colorGrading1[3] = 0.0f;
    mvpUniform.colorGrading2[0] = std::clamp(m_skyDebugSettings.colorGradingShadowTintR, -1.0f, 1.0f);
    mvpUniform.colorGrading2[1] = std::clamp(m_skyDebugSettings.colorGradingShadowTintG, -1.0f, 1.0f);
    mvpUniform.colorGrading2[2] = std::clamp(m_skyDebugSettings.colorGradingShadowTintB, -1.0f, 1.0f);
    mvpUniform.colorGrading2[3] = 0.0f;
    mvpUniform.colorGrading3[0] = std::clamp(m_skyDebugSettings.colorGradingHighlightTintR, -1.0f, 1.0f);
    mvpUniform.colorGrading3[1] = std::clamp(m_skyDebugSettings.colorGradingHighlightTintG, -1.0f, 1.0f);
    mvpUniform.colorGrading3[2] = std::clamp(m_skyDebugSettings.colorGradingHighlightTintB, -1.0f, 1.0f);
    mvpUniform.colorGrading3[3] = 0.0f;
    const float voxelGiGridSpan = static_cast<float>(kVoxelGiGridResolution) * kVoxelGiCellSize;
    const float voxelGiHalfSpan = voxelGiGridSpan * 0.5f;
    const float voxelGiOriginX = ComputeVoxelGiAxisOrigin(camera.x, voxelGiHalfSpan, kVoxelGiCellSize);
    const float voxelGiDesiredOriginY = ComputeVoxelGiAxisOrigin(camera.y, voxelGiHalfSpan, kVoxelGiCellSize);
    const float kVoxelGiVerticalFollowThreshold = kVoxelGiCellSize * 4.0f;
    const float voxelGiOriginY = ComputeVoxelGiStableOriginY(
        voxelGiDesiredOriginY,
        m_voxelGiPreviousGridOrigin[1],
        m_voxelGiHasPreviousFrameState,
        kVoxelGiVerticalFollowThreshold
    );
    const float voxelGiOriginZ = ComputeVoxelGiAxisOrigin(camera.z, voxelGiHalfSpan, kVoxelGiCellSize);
    constexpr float kVoxelGiGridMoveThreshold = 0.001f;
    constexpr float kVoxelGiLightingChangeThreshold = 0.001f;
    constexpr float kVoxelGiTuningChangeThreshold = 0.001f;
    const bool voxelGiGridMoved =
        !m_voxelGiHasPreviousFrameState ||
        std::abs(voxelGiOriginX - m_voxelGiPreviousGridOrigin[0]) > kVoxelGiGridMoveThreshold ||
        std::abs(voxelGiOriginY - m_voxelGiPreviousGridOrigin[1]) > kVoxelGiGridMoveThreshold ||
        std::abs(voxelGiOriginZ - m_voxelGiPreviousGridOrigin[2]) > kVoxelGiGridMoveThreshold;
    const bool voxelGiSunDirectionChanged =
        !m_voxelGiHasPreviousFrameState ||
        std::abs(sunDirection.x - m_voxelGiPreviousSunDirection[0]) > kVoxelGiLightingChangeThreshold ||
        std::abs(sunDirection.y - m_voxelGiPreviousSunDirection[1]) > kVoxelGiLightingChangeThreshold ||
        std::abs(sunDirection.z - m_voxelGiPreviousSunDirection[2]) > kVoxelGiLightingChangeThreshold;
    const bool voxelGiSunColorChanged =
        !m_voxelGiHasPreviousFrameState ||
        std::abs(sunColor.x - m_voxelGiPreviousSunColor[0]) > kVoxelGiLightingChangeThreshold ||
        std::abs(sunColor.y - m_voxelGiPreviousSunColor[1]) > kVoxelGiLightingChangeThreshold ||
        std::abs(sunColor.z - m_voxelGiPreviousSunColor[2]) > kVoxelGiLightingChangeThreshold;
    bool voxelGiShChanged = !m_voxelGiHasPreviousFrameState;
    if (!voxelGiShChanged) {
        for (std::size_t coeffIndex = 0; coeffIndex < shIrradiance.size(); ++coeffIndex) {
            const std::array<float, 3>& previousCoeff = m_voxelGiPreviousShIrradiance[coeffIndex];
            const voxelsprout::math::Vector3& currentCoeff = shIrradiance[coeffIndex];
            if (std::abs(currentCoeff.x - previousCoeff[0]) > kVoxelGiLightingChangeThreshold ||
                std::abs(currentCoeff.y - previousCoeff[1]) > kVoxelGiLightingChangeThreshold ||
                std::abs(currentCoeff.z - previousCoeff[2]) > kVoxelGiLightingChangeThreshold) {
                voxelGiShChanged = true;
                break;
            }
        }
    }
    const bool voxelGiComputeSettingsChanged =
        !m_voxelGiHasPreviousFrameState ||
        std::abs(m_voxelGiDebugSettings.bounceStrength - m_voxelGiPreviousBounceStrength) >
            kVoxelGiTuningChangeThreshold ||
        std::abs(m_voxelGiDebugSettings.diffusionSoftness - m_voxelGiPreviousDiffusionSoftness) >
            kVoxelGiTuningChangeThreshold;
    const bool voxelGiLightingChanged = voxelGiSunDirectionChanged || voxelGiSunColorChanged || voxelGiShChanged;
    const bool voxelGiNeedsOccupancyUpload =
        m_voxelGiWorldDirty ||
        voxelGiGridMoved ||
        !m_voxelGiOccupancyInitialized;
    const bool voxelGiNeedsComputeUpdate =
        voxelGiNeedsOccupancyUpload ||
        voxelGiLightingChanged ||
        voxelGiComputeSettingsChanged ||
        !m_voxelGiInitialized;
    m_voxelGiHasPreviousFrameState = true;
    m_voxelGiPreviousGridOrigin = {voxelGiOriginX, voxelGiOriginY, voxelGiOriginZ};
    m_voxelGiPreviousSunDirection = {sunDirection.x, sunDirection.y, sunDirection.z};
    m_voxelGiPreviousSunColor = {sunColor.x, sunColor.y, sunColor.z};
    for (std::size_t coeffIndex = 0; coeffIndex < shIrradiance.size(); ++coeffIndex) {
        const voxelsprout::math::Vector3& coeff = shIrradiance[coeffIndex];
        m_voxelGiPreviousShIrradiance[coeffIndex] = {coeff.x, coeff.y, coeff.z};
    }
    m_voxelGiPreviousBounceStrength = m_voxelGiDebugSettings.bounceStrength;
    m_voxelGiPreviousDiffusionSoftness = m_voxelGiDebugSettings.diffusionSoftness;
    mvpUniform.voxelGiGridOriginCellSize[0] = voxelGiOriginX;
    mvpUniform.voxelGiGridOriginCellSize[1] = voxelGiOriginY;
    mvpUniform.voxelGiGridOriginCellSize[2] = voxelGiOriginZ;
    mvpUniform.voxelGiGridOriginCellSize[3] = kVoxelGiCellSize;
    mvpUniform.voxelGiGridExtentStrength[0] = voxelGiGridSpan;
    mvpUniform.voxelGiGridExtentStrength[1] = voxelGiGridSpan;
    mvpUniform.voxelGiGridExtentStrength[2] = voxelGiGridSpan;
    mvpUniform.voxelGiGridExtentStrength[3] = kVoxelGiStrength;
    for (std::size_t colorIndex = 0; colorIndex < m_voxelBaseColorPaletteRgba.size(); ++colorIndex) {
        const std::uint32_t rgba = m_voxelBaseColorPaletteRgba[colorIndex];
        mvpUniform.voxelBaseColorPalette[colorIndex][0] = static_cast<float>(rgba & 0xFFu) / 255.0f;
        mvpUniform.voxelBaseColorPalette[colorIndex][1] = static_cast<float>((rgba >> 8u) & 0xFFu) / 255.0f;
        mvpUniform.voxelBaseColorPalette[colorIndex][2] = static_cast<float>((rgba >> 16u) & 0xFFu) / 255.0f;
        mvpUniform.voxelBaseColorPalette[colorIndex][3] = static_cast<float>((rgba >> 24u) & 0xFFu) / 255.0f;
    }
    std::memcpy(mvpSliceOpt->mapped, &mvpUniform, sizeof(mvpUniform));

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_bufferAllocator.getBuffer(mvpSliceOpt->buffer);
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(CameraUniform);
    if (mvpSliceOpt->offset > static_cast<VkDeviceSize>(std::numeric_limits<uint32_t>::max())) {
        VOX_LOGI("render") << "dynamic UBO offset exceeds uint32 range\n";
        return;
    }
    const uint32_t mvpDynamicOffset = static_cast<uint32_t>(mvpSliceOpt->offset);
    const VkBuffer autoExposureStateBuffer = m_bufferAllocator.getBuffer(m_autoExposureStateBufferHandle);
    const VkBuffer autoExposureHistogramBuffer = m_bufferAllocator.getBuffer(m_autoExposureHistogramBufferHandle);
    if (autoExposureStateBuffer == VK_NULL_HANDLE) {
        VOX_LOGE("render") << "auto exposure state buffer unavailable";
        return;
    }

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

    VkDescriptorImageInfo hdrSceneImageInfo{};
    hdrSceneImageInfo.sampler = m_hdrResolveSampler;
    hdrSceneImageInfo.imageView = m_hdrResolveSampleImageViews[aoFrameIndex];
    hdrSceneImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo diffuseTextureImageInfo{};
    diffuseTextureImageInfo.sampler = m_diffuseTextureSampler;
    diffuseTextureImageInfo.imageView = m_diffuseTextureImageView;
    diffuseTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo plantDiffuseTextureImageInfo{};
    plantDiffuseTextureImageInfo.sampler = m_diffuseTexturePlantSampler;
    plantDiffuseTextureImageInfo.imageView = m_diffuseTextureImageView;
    plantDiffuseTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo shadowMapImageInfo{};
    shadowMapImageInfo.sampler = m_shadowDepthSampler;
    shadowMapImageInfo.imageView = m_shadowDepthImageView;
    shadowMapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo normalDepthImageInfo{};
    normalDepthImageInfo.sampler = m_normalDepthSampler;
    normalDepthImageInfo.imageView = m_normalDepthImageViews[aoFrameIndex];
    normalDepthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo ssaoBlurImageInfo{};
    ssaoBlurImageInfo.sampler = m_ssaoSampler;
    ssaoBlurImageInfo.imageView = m_ssaoBlurImageViews[aoFrameIndex];
    ssaoBlurImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo ssaoRawImageInfo{};
    ssaoRawImageInfo.sampler = m_ssaoSampler;
    ssaoRawImageInfo.imageView = m_ssaoRawImageViews[aoFrameIndex];
    ssaoRawImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo voxelGiVolumeImageInfo{};
    voxelGiVolumeImageInfo.sampler = m_voxelGiSampler;
    voxelGiVolumeImageInfo.imageView = m_voxelGiImageViews[1];
    voxelGiVolumeImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo voxelGiOccupancyDebugImageInfo{};
    voxelGiOccupancyDebugImageInfo.sampler = m_voxelGiOccupancySampler;
    voxelGiOccupancyDebugImageInfo.imageView = m_voxelGiOccupancyImageView;
    voxelGiOccupancyDebugImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo sunShaftImageInfo{};
    sunShaftImageInfo.sampler = m_sunShaftSampler;
    sunShaftImageInfo.imageView =
        (aoFrameIndex < m_sunShaftImageViews.size()) ? m_sunShaftImageViews[aoFrameIndex] : VK_NULL_HANDLE;
    sunShaftImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo autoExposureStateBufferInfo{};
    autoExposureStateBufferInfo.buffer = autoExposureStateBuffer;
    autoExposureStateBufferInfo.offset = 0;
    autoExposureStateBufferInfo.range = sizeof(float) * 4u;

    std::array<VkWriteDescriptorSet, 11> writes{};
    writes[0] = write;
    writes[0].dstSet = m_descriptorSets[m_currentFrame];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    writes[0].pBufferInfo = &bufferInfo;

    writes[1] = write;
    writes[1].dstSet = m_descriptorSets[m_currentFrame];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &diffuseTextureImageInfo;

    writes[2] = write;
    writes[2].dstSet = m_descriptorSets[m_currentFrame];
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &autoExposureStateBufferInfo;

    writes[3] = write;
    writes[3].dstSet = m_descriptorSets[m_currentFrame];
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &hdrSceneImageInfo;

    writes[4] = write;
    writes[4].dstSet = m_descriptorSets[m_currentFrame];
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &shadowMapImageInfo;

    writes[5] = write;
    writes[5].dstSet = m_descriptorSets[m_currentFrame];
    writes[5].dstBinding = 6;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[5].pImageInfo = &normalDepthImageInfo;

    writes[6] = write;
    writes[6].dstSet = m_descriptorSets[m_currentFrame];
    writes[6].dstBinding = 7;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[6].pImageInfo = &ssaoBlurImageInfo;

    writes[7] = write;
    writes[7].dstSet = m_descriptorSets[m_currentFrame];
    writes[7].dstBinding = 8;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].pImageInfo = &ssaoRawImageInfo;

    writes[8] = write;
    writes[8].dstSet = m_descriptorSets[m_currentFrame];
    writes[8].dstBinding = 9;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[8].pImageInfo = &voxelGiVolumeImageInfo;

    writes[9] = write;
    writes[9].dstSet = m_descriptorSets[m_currentFrame];
    writes[9].dstBinding = 10;
    writes[9].descriptorCount = 1;
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[9].pImageInfo = &sunShaftImageInfo;

    writes[10] = write;
    writes[10].dstSet = m_descriptorSets[m_currentFrame];
    writes[10].dstBinding = 11;
    writes[10].descriptorCount = 1;
    writes[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[10].pImageInfo = &voxelGiOccupancyDebugImageInfo;

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    if (m_voxelGiComputeAvailable && m_voxelGiDescriptorSets[m_currentFrame] != VK_NULL_HANDLE) {
        VkDescriptorImageInfo voxelGiStorageAInfo{};
        voxelGiStorageAInfo.sampler = VK_NULL_HANDLE;
        voxelGiStorageAInfo.imageView = m_voxelGiImageViews[0];
        voxelGiStorageAInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo voxelGiStorageReadInfo{};
        voxelGiStorageReadInfo.sampler = VK_NULL_HANDLE;
        voxelGiStorageReadInfo.imageView = m_voxelGiImageViews[0];
        voxelGiStorageReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo voxelGiStorageBInfo{};
        voxelGiStorageBInfo.sampler = VK_NULL_HANDLE;
        voxelGiStorageBInfo.imageView = m_voxelGiImageViews[1];
        voxelGiStorageBInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo voxelGiOccupancyInfo{};
        voxelGiOccupancyInfo.sampler = VK_NULL_HANDLE;
        voxelGiOccupancyInfo.imageView = m_voxelGiOccupancyImageView;
        voxelGiOccupancyInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        std::array<VkDescriptorImageInfo, 6> voxelGiSurfaceFaceInfos{};
        for (std::size_t faceIndex = 0; faceIndex < voxelGiSurfaceFaceInfos.size(); ++faceIndex) {
            voxelGiSurfaceFaceInfos[faceIndex].sampler = VK_NULL_HANDLE;
            voxelGiSurfaceFaceInfos[faceIndex].imageView = m_voxelGiSurfaceFaceImageViews[faceIndex];
            voxelGiSurfaceFaceInfos[faceIndex].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        VkDescriptorImageInfo voxelGiSkyExposureInfo{};
        voxelGiSkyExposureInfo.sampler = VK_NULL_HANDLE;
        voxelGiSkyExposureInfo.imageView = m_voxelGiSkyExposureImageView;
        voxelGiSkyExposureInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 13> voxelGiWrites{};
        voxelGiWrites[0] = write;
        voxelGiWrites[0].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[0].dstBinding = 0;
        voxelGiWrites[0].descriptorCount = 1;
        voxelGiWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        voxelGiWrites[0].pBufferInfo = &bufferInfo;

        voxelGiWrites[1] = write;
        voxelGiWrites[1].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[1].dstBinding = 1;
        voxelGiWrites[1].descriptorCount = 1;
        voxelGiWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        voxelGiWrites[1].pImageInfo = &shadowMapImageInfo;

        voxelGiWrites[2] = write;
        voxelGiWrites[2].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[2].dstBinding = 2;
        voxelGiWrites[2].descriptorCount = 1;
        voxelGiWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[2].pImageInfo = &voxelGiStorageAInfo;

        voxelGiWrites[3] = write;
        voxelGiWrites[3].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[3].dstBinding = 3;
        voxelGiWrites[3].descriptorCount = 1;
        voxelGiWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        voxelGiWrites[3].pImageInfo = &voxelGiStorageReadInfo;

        voxelGiWrites[4] = write;
        voxelGiWrites[4].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[4].dstBinding = 4;
        voxelGiWrites[4].descriptorCount = 1;
        voxelGiWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[4].pImageInfo = &voxelGiStorageBInfo;

        voxelGiWrites[5] = write;
        voxelGiWrites[5].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[5].dstBinding = 5;
        voxelGiWrites[5].descriptorCount = 1;
        voxelGiWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        voxelGiWrites[5].pImageInfo = &voxelGiOccupancyInfo;

        voxelGiWrites[6] = write;
        voxelGiWrites[6].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[6].dstBinding = 6;
        voxelGiWrites[6].descriptorCount = 1;
        voxelGiWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[6].pImageInfo = &voxelGiSurfaceFaceInfos[0];

        voxelGiWrites[7] = write;
        voxelGiWrites[7].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[7].dstBinding = 7;
        voxelGiWrites[7].descriptorCount = 1;
        voxelGiWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[7].pImageInfo = &voxelGiSurfaceFaceInfos[1];

        voxelGiWrites[8] = write;
        voxelGiWrites[8].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[8].dstBinding = 8;
        voxelGiWrites[8].descriptorCount = 1;
        voxelGiWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[8].pImageInfo = &voxelGiSurfaceFaceInfos[2];

        voxelGiWrites[9] = write;
        voxelGiWrites[9].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[9].dstBinding = 9;
        voxelGiWrites[9].descriptorCount = 1;
        voxelGiWrites[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[9].pImageInfo = &voxelGiSurfaceFaceInfos[3];

        voxelGiWrites[10] = write;
        voxelGiWrites[10].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[10].dstBinding = 10;
        voxelGiWrites[10].descriptorCount = 1;
        voxelGiWrites[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[10].pImageInfo = &voxelGiSurfaceFaceInfos[4];

        voxelGiWrites[11] = write;
        voxelGiWrites[11].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[11].dstBinding = 11;
        voxelGiWrites[11].descriptorCount = 1;
        voxelGiWrites[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[11].pImageInfo = &voxelGiSurfaceFaceInfos[5];

        voxelGiWrites[12] = write;
        voxelGiWrites[12].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[12].dstBinding = 12;
        voxelGiWrites[12].descriptorCount = 1;
        voxelGiWrites[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[12].pImageInfo = &voxelGiSkyExposureInfo;

        vkUpdateDescriptorSets(
            m_device,
            static_cast<uint32_t>(voxelGiWrites.size()),
            voxelGiWrites.data(),
            0,
            nullptr
        );
    }

    if (m_autoExposureComputeAvailable &&
        m_autoExposureDescriptorSets[m_currentFrame] != VK_NULL_HANDLE &&
        autoExposureHistogramBuffer != VK_NULL_HANDLE &&
        autoExposureStateBuffer != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo autoExposureHistogramBufferInfo{};
        autoExposureHistogramBufferInfo.buffer = autoExposureHistogramBuffer;
        autoExposureHistogramBufferInfo.offset = 0;
        autoExposureHistogramBufferInfo.range = static_cast<VkDeviceSize>(kAutoExposureHistogramBins * sizeof(uint32_t));

        VkDescriptorBufferInfo autoExposureStateComputeBufferInfo{};
        autoExposureStateComputeBufferInfo.buffer = autoExposureStateBuffer;
        autoExposureStateComputeBufferInfo.offset = 0;
        autoExposureStateComputeBufferInfo.range = sizeof(float) * 4u;

        std::array<VkWriteDescriptorSet, 3> autoExposureWrites{};
        autoExposureWrites[0] = write;
        autoExposureWrites[0].dstSet = m_autoExposureDescriptorSets[m_currentFrame];
        autoExposureWrites[0].dstBinding = 0;
        autoExposureWrites[0].descriptorCount = 1;
        autoExposureWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        autoExposureWrites[0].pImageInfo = &hdrSceneImageInfo;

        autoExposureWrites[1] = write;
        autoExposureWrites[1].dstSet = m_autoExposureDescriptorSets[m_currentFrame];
        autoExposureWrites[1].dstBinding = 1;
        autoExposureWrites[1].descriptorCount = 1;
        autoExposureWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        autoExposureWrites[1].pBufferInfo = &autoExposureHistogramBufferInfo;

        autoExposureWrites[2] = write;
        autoExposureWrites[2].dstSet = m_autoExposureDescriptorSets[m_currentFrame];
        autoExposureWrites[2].dstBinding = 2;
        autoExposureWrites[2].descriptorCount = 1;
        autoExposureWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        autoExposureWrites[2].pBufferInfo = &autoExposureStateComputeBufferInfo;

        vkUpdateDescriptorSets(
            m_device,
            static_cast<uint32_t>(autoExposureWrites.size()),
            autoExposureWrites.data(),
            0,
            nullptr
        );
    }

    if (m_sunShaftComputeAvailable &&
        m_sunShaftDescriptorSets[m_currentFrame] != VK_NULL_HANDLE &&
        aoFrameIndex < m_sunShaftImageViews.size() &&
        m_sunShaftImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        VkDescriptorImageInfo sunShaftOutputImageInfo{};
        sunShaftOutputImageInfo.sampler = VK_NULL_HANDLE;
        sunShaftOutputImageInfo.imageView = m_sunShaftImageViews[aoFrameIndex];
        sunShaftOutputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 4> sunShaftWrites{};
        sunShaftWrites[0] = write;
        sunShaftWrites[0].dstSet = m_sunShaftDescriptorSets[m_currentFrame];
        sunShaftWrites[0].dstBinding = 0;
        sunShaftWrites[0].descriptorCount = 1;
        sunShaftWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        sunShaftWrites[0].pBufferInfo = &bufferInfo;

        sunShaftWrites[1] = write;
        sunShaftWrites[1].dstSet = m_sunShaftDescriptorSets[m_currentFrame];
        sunShaftWrites[1].dstBinding = 1;
        sunShaftWrites[1].descriptorCount = 1;
        sunShaftWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sunShaftWrites[1].pImageInfo = &normalDepthImageInfo;

        sunShaftWrites[2] = write;
        sunShaftWrites[2].dstSet = m_sunShaftDescriptorSets[m_currentFrame];
        sunShaftWrites[2].dstBinding = 2;
        sunShaftWrites[2].descriptorCount = 1;
        sunShaftWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sunShaftWrites[2].pImageInfo = &shadowMapImageInfo;

        sunShaftWrites[3] = write;
        sunShaftWrites[3].dstSet = m_sunShaftDescriptorSets[m_currentFrame];
        sunShaftWrites[3].dstBinding = 3;
        sunShaftWrites[3].descriptorCount = 1;
        sunShaftWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sunShaftWrites[3].pImageInfo = &sunShaftOutputImageInfo;

        vkUpdateDescriptorSets(
            m_device,
            static_cast<uint32_t>(sunShaftWrites.size()),
            sunShaftWrites.data(),
            0,
            nullptr
        );
    }

    if (m_bindlessDescriptorSet != VK_NULL_HANDLE && m_bindlessTextureCapacity >= kBindlessTextureStaticCount) {
        std::array<VkDescriptorImageInfo, kBindlessTextureStaticCount> bindlessImageInfos{};
        bindlessImageInfos[kBindlessTextureIndexDiffuse] = diffuseTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexHdrResolved] = hdrSceneImageInfo;
        bindlessImageInfos[kBindlessTextureIndexShadowAtlas] = shadowMapImageInfo;
        bindlessImageInfos[kBindlessTextureIndexNormalDepth] = normalDepthImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoBlur] = ssaoBlurImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoRaw] = ssaoRawImageInfo;
        bindlessImageInfos[kBindlessTextureIndexPlantDiffuse] = plantDiffuseTextureImageInfo;

        VkWriteDescriptorSet bindlessWrite{};
        bindlessWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bindlessWrite.dstSet = m_bindlessDescriptorSet;
        bindlessWrite.dstBinding = 0;
        bindlessWrite.dstArrayElement = 0;
        bindlessWrite.descriptorCount = kBindlessTextureStaticCount;
        bindlessWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindlessWrite.pImageInfo = bindlessImageInfos.data();
        vkUpdateDescriptorSets(m_device, 1, &bindlessWrite, 0, nullptr);
    }
    std::array<VkDescriptorSet, 2> boundDescriptorSets = {
        m_descriptorSets[m_currentFrame],
        m_bindlessDescriptorSet
    };
    const uint32_t boundDescriptorSetCount =
        (m_bindlessDescriptorSet != VK_NULL_HANDLE) ? 2u : 1u;

    std::optional<FrameArenaSlice> voxelGiOccupancySliceOpt = std::nullopt;
    VkBuffer voxelGiOccupancyUploadBuffer = VK_NULL_HANDLE;
    if (m_voxelGiComputeAvailable &&
        m_voxelGiOccupancyImage != VK_NULL_HANDLE &&
        m_voxelGiOccupancyImageView != VK_NULL_HANDLE &&
        voxelGiNeedsOccupancyUpload) {
        const std::size_t voxelGiCellCount =
            static_cast<std::size_t>(kVoxelGiGridResolution) *
            static_cast<std::size_t>(kVoxelGiGridResolution) *
            static_cast<std::size_t>(kVoxelGiGridResolution);
        // RGBA8 payload for GI occupancy:
        // R = occupancy, GBA = albedo RGB for solid cells.
        std::vector<std::uint8_t> occupancyData(voxelGiCellCount * 4u, 0u);

        std::array<int, kVoxelGiGridResolution> worldXCoords{};
        std::array<int, kVoxelGiGridResolution> worldYCoords{};
        std::array<int, kVoxelGiGridResolution> worldZCoords{};
        for (uint32_t i = 0; i < kVoxelGiGridResolution; ++i) {
            const float offset = (static_cast<float>(i) + 0.5f) * kVoxelGiCellSize;
            worldXCoords[i] = static_cast<int>(std::floor(voxelGiOriginX + offset));
            worldYCoords[i] = static_cast<int>(std::floor(voxelGiOriginY + offset));
            worldZCoords[i] = static_cast<int>(std::floor(voxelGiOriginZ + offset));
        }

        std::unordered_map<ChunkCoordKey, const voxelsprout::world::Chunk*, ChunkCoordKeyHash> chunkByCoord;
        chunkByCoord.reserve(chunkGrid.chunkCount() * 2u);
        for (const voxelsprout::world::Chunk& chunk : chunkGrid.chunks()) {
            chunkByCoord[ChunkCoordKey{chunk.chunkX(), chunk.chunkY(), chunk.chunkZ()}] = &chunk;
        }

        for (uint32_t z = 0; z < kVoxelGiGridResolution; ++z) {
            const int worldZ = worldZCoords[z];
            const int chunkZ = floorDiv(worldZ, voxelsprout::world::Chunk::kSizeZ);
            const int localZ = worldZ - (chunkZ * voxelsprout::world::Chunk::kSizeZ);
            for (uint32_t y = 0; y < kVoxelGiGridResolution; ++y) {
                const int worldY = worldYCoords[y];
                const int chunkY = floorDiv(worldY, voxelsprout::world::Chunk::kSizeY);
                const int localY = worldY - (chunkY * voxelsprout::world::Chunk::kSizeY);
                for (uint32_t x = 0; x < kVoxelGiGridResolution; ++x) {
                    const int worldX = worldXCoords[x];
                    const int chunkX = floorDiv(worldX, voxelsprout::world::Chunk::kSizeX);
                    const int localX = worldX - (chunkX * voxelsprout::world::Chunk::kSizeX);
                    const ChunkCoordKey key{chunkX, chunkY, chunkZ};
                    const auto chunkIt = chunkByCoord.find(key);
                    if (chunkIt == chunkByCoord.end()) {
                        continue;
                    }
                    const voxelsprout::world::Chunk* chunk = chunkIt->second;
                    if (chunk == nullptr || !chunk->isSolid(localX, localY, localZ)) {
                        continue;
                    }
                    const voxelsprout::world::Voxel voxel = chunk->voxelAt(localX, localY, localZ);
                    const std::array<std::uint8_t, 3> albedoRgb =
                        voxelGiAlbedoRgb(voxel, m_voxelBaseColorPaletteRgba);
                    const std::size_t index =
                        static_cast<std::size_t>(x) +
                        (static_cast<std::size_t>(kVoxelGiGridResolution) *
                         (static_cast<std::size_t>(y) +
                          (static_cast<std::size_t>(kVoxelGiGridResolution) * static_cast<std::size_t>(z))));
                    const std::size_t rgbaIndex = index * 4u;
                    occupancyData[rgbaIndex + 0u] = 255u;
                    occupancyData[rgbaIndex + 1u] = albedoRgb[0];
                    occupancyData[rgbaIndex + 2u] = albedoRgb[1];
                    occupancyData[rgbaIndex + 3u] = albedoRgb[2];
                }
            }
        }

        const VkDeviceSize occupancyBytes = static_cast<VkDeviceSize>(occupancyData.size());
        voxelGiOccupancySliceOpt = m_frameArena.allocateUpload(
            occupancyBytes,
            static_cast<VkDeviceSize>(4u),
            FrameArenaUploadKind::Unknown
        );
        if (voxelGiOccupancySliceOpt.has_value() && voxelGiOccupancySliceOpt->mapped != nullptr) {
            std::memcpy(
                voxelGiOccupancySliceOpt->mapped,
                occupancyData.data(),
                static_cast<std::size_t>(occupancyBytes)
            );
            voxelGiOccupancyUploadBuffer = m_bufferAllocator.getBuffer(voxelGiOccupancySliceOpt->buffer);
        } else {
            voxelGiOccupancySliceOpt.reset();
            VOX_LOGW("render") << "voxel GI occupancy upload allocation failed";
        }
    }
    const bool voxelGiHasOccupancyUpload =
        voxelGiOccupancySliceOpt.has_value() &&
        voxelGiOccupancyUploadBuffer != VK_NULL_HANDLE;

    uint32_t pipeInstanceCount = 0;
    std::optional<FrameArenaSlice> pipeInstanceSliceOpt = std::nullopt;
    uint32_t transportInstanceCount = 0;
    std::optional<FrameArenaSlice> transportInstanceSliceOpt = std::nullopt;
    uint32_t beltCargoInstanceCount = 0;
    std::optional<FrameArenaSlice> beltCargoInstanceSliceOpt = std::nullopt;
    if (m_pipeIndexCount > 0 || m_transportIndexCount > 0) {
        const std::vector<voxelsprout::sim::Pipe>& pipes = simulation.pipes();
        const std::vector<voxelsprout::sim::Belt>& belts = simulation.belts();
        const std::vector<voxelsprout::sim::Track>& tracks = simulation.tracks();
        const std::vector<voxelsprout::sim::BeltCargo>& beltCargoes = simulation.beltCargoes();
        const float clampedSimulationAlpha = std::clamp(simulationAlpha, 0.0f, 1.0f);
        const std::vector<PipeEndpointState> endpointStates =
            pipes.empty() ? std::vector<PipeEndpointState>{} : buildPipeEndpointStates(pipes);
        std::vector<PipeInstance> pipeInstances;
        pipeInstances.reserve(pipes.size());
        for (std::size_t pipeIndex = 0; pipeIndex < pipes.size(); ++pipeIndex) {
            const voxelsprout::sim::Pipe& pipe = pipes[pipeIndex];
            const PipeEndpointState& endpointState = endpointStates[pipeIndex];
            PipeInstance instance{};
            instance.originLength[0] = static_cast<float>(pipe.x);
            instance.originLength[1] = static_cast<float>(pipe.y);
            instance.originLength[2] = static_cast<float>(pipe.z);
            instance.originLength[3] = std::max(pipe.length, 0.05f);
            instance.axisRadius[0] = endpointState.axis.x;
            instance.axisRadius[1] = endpointState.axis.y;
            instance.axisRadius[2] = endpointState.axis.z;
            instance.axisRadius[3] = endpointState.renderedRadius;
            instance.tint[0] = std::clamp(pipe.tint.x, 0.0f, 1.0f);
            instance.tint[1] = std::clamp(pipe.tint.y, 0.0f, 1.0f);
            instance.tint[2] = std::clamp(pipe.tint.z, 0.0f, 1.0f);
            instance.tint[3] = 0.0f; // style 0 = pipe
            instance.extensions[0] = endpointState.startExtension;
            instance.extensions[1] = endpointState.endExtension;
            instance.extensions[2] = 1.0f;
            instance.extensions[3] = 1.0f;
            pipeInstances.push_back(instance);
        }

        std::vector<PipeInstance> transportInstances;
        transportInstances.reserve(belts.size() + tracks.size());
        for (const voxelsprout::sim::Belt& belt : belts) {
            PipeInstance instance{};
            instance.originLength[0] = static_cast<float>(belt.x);
            instance.originLength[1] = static_cast<float>(belt.y);
            instance.originLength[2] = static_cast<float>(belt.z);
            instance.originLength[3] = 1.0f;
            const voxelsprout::math::Vector3 axis = beltDirectionAxis(belt.direction);
            instance.axisRadius[0] = axis.x;
            instance.axisRadius[1] = axis.y;
            instance.axisRadius[2] = axis.z;
            instance.axisRadius[3] = kBeltRadius;
            instance.tint[0] = kBeltTint.x;
            instance.tint[1] = kBeltTint.y;
            instance.tint[2] = kBeltTint.z;
            instance.tint[3] = 1.0f; // style 1 = conveyor
            instance.extensions[0] = 0.0f;
            instance.extensions[1] = 0.0f;
            // Conveyors: 2x wider cross-span, 0.25x height.
            instance.extensions[2] = 2.0f;
            instance.extensions[3] = 0.25f;
            transportInstances.push_back(instance);
        }

        for (const voxelsprout::sim::Track& track : tracks) {
            PipeInstance instance{};
            instance.originLength[0] = static_cast<float>(track.x);
            instance.originLength[1] = static_cast<float>(track.y);
            instance.originLength[2] = static_cast<float>(track.z);
            instance.originLength[3] = 1.0f;
            const voxelsprout::math::Vector3 axis = trackDirectionAxis(track.direction);
            instance.axisRadius[0] = axis.x;
            instance.axisRadius[1] = axis.y;
            instance.axisRadius[2] = axis.z;
            instance.axisRadius[3] = kTrackRadius;
            instance.tint[0] = kTrackTint.x;
            instance.tint[1] = kTrackTint.y;
            instance.tint[2] = kTrackTint.z;
            instance.tint[3] = 2.0f; // style 2 = track
            instance.extensions[0] = 0.0f;
            instance.extensions[1] = 0.0f;
            // Tracks: 2x wider cross-span, 0.25x height.
            instance.extensions[2] = 2.0f;
            instance.extensions[3] = 0.25f;
            transportInstances.push_back(instance);
        }

        std::vector<PipeInstance> beltCargoInstances;
        beltCargoInstances.reserve(beltCargoes.size());
        for (const voxelsprout::sim::BeltCargo& cargo : beltCargoes) {
            if (cargo.beltIndex < 0 || static_cast<std::size_t>(cargo.beltIndex) >= belts.size()) {
                continue;
            }
            const float worldX = std::lerp(cargo.prevWorldPos[0], cargo.currWorldPos[0], clampedSimulationAlpha);
            const float worldY = std::lerp(cargo.prevWorldPos[1], cargo.currWorldPos[1], clampedSimulationAlpha);
            const float worldZ = std::lerp(cargo.prevWorldPos[2], cargo.currWorldPos[2], clampedSimulationAlpha);
            const voxelsprout::sim::Belt& belt = belts[static_cast<std::size_t>(cargo.beltIndex)];
            const voxelsprout::math::Vector3 axis = beltDirectionAxis(belt.direction);
            const voxelsprout::math::Vector3 tint = kBeltCargoTints[static_cast<std::size_t>(cargo.typeId % kBeltCargoTints.size())];

            PipeInstance instance{};
            instance.originLength[0] = worldX - 0.5f;
            instance.originLength[1] = worldY - 0.5f;
            instance.originLength[2] = worldZ - 0.5f;
            instance.originLength[3] = kBeltCargoLength;
            instance.axisRadius[0] = axis.x;
            instance.axisRadius[1] = axis.y;
            instance.axisRadius[2] = axis.z;
            instance.axisRadius[3] = kBeltCargoRadius;
            instance.tint[0] = tint.x;
            instance.tint[1] = tint.y;
            instance.tint[2] = tint.z;
            instance.tint[3] = 2.0f; // style 2 = neutral solid transport block
            instance.extensions[0] = 0.0f;
            instance.extensions[1] = 0.0f;
            instance.extensions[2] = 1.0f;
            instance.extensions[3] = 1.0f;
            beltCargoInstances.push_back(instance);
        }

        if (!pipeInstances.empty() && m_pipeIndexCount > 0) {
            pipeInstanceSliceOpt = m_frameArena.allocateUpload(
                static_cast<VkDeviceSize>(pipeInstances.size() * sizeof(PipeInstance)),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::InstanceData
            );
            if (pipeInstanceSliceOpt.has_value() && pipeInstanceSliceOpt->mapped != nullptr) {
                std::memcpy(
                    pipeInstanceSliceOpt->mapped,
                    pipeInstances.data(),
                    static_cast<size_t>(pipeInstanceSliceOpt->size)
                );
                pipeInstanceCount = static_cast<uint32_t>(pipeInstances.size());
            }
        }

        if (!transportInstances.empty() && m_transportIndexCount > 0) {
            transportInstanceSliceOpt = m_frameArena.allocateUpload(
                static_cast<VkDeviceSize>(transportInstances.size() * sizeof(PipeInstance)),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::InstanceData
            );
            if (transportInstanceSliceOpt.has_value() && transportInstanceSliceOpt->mapped != nullptr) {
                std::memcpy(
                    transportInstanceSliceOpt->mapped,
                    transportInstances.data(),
                    static_cast<size_t>(transportInstanceSliceOpt->size)
                );
                transportInstanceCount = static_cast<uint32_t>(transportInstances.size());
            }
        }

        if (!beltCargoInstances.empty() && m_transportIndexCount > 0) {
            beltCargoInstanceSliceOpt = m_frameArena.allocateUpload(
                static_cast<VkDeviceSize>(beltCargoInstances.size() * sizeof(PipeInstance)),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::InstanceData
            );
            if (beltCargoInstanceSliceOpt.has_value() && beltCargoInstanceSliceOpt->mapped != nullptr) {
                std::memcpy(
                    beltCargoInstanceSliceOpt->mapped,
                    beltCargoInstances.data(),
                    static_cast<size_t>(beltCargoInstanceSliceOpt->size)
                );
                beltCargoInstanceCount = static_cast<uint32_t>(beltCargoInstances.size());
            }
        }
    }

    const VkBuffer chunkVertexBuffer = m_bufferAllocator.getBuffer(m_chunkVertexBufferHandle);
    const VkBuffer chunkIndexBuffer = m_bufferAllocator.getBuffer(m_chunkIndexBufferHandle);
    const bool chunkDrawBuffersReady = chunkVertexBuffer != VK_NULL_HANDLE && chunkIndexBuffer != VK_NULL_HANDLE;

    std::vector<ChunkInstanceData> chunkInstanceData;
    chunkInstanceData.reserve(m_chunkDrawRanges.size() + 1);
    chunkInstanceData.push_back(ChunkInstanceData{});
    std::vector<VkDrawIndexedIndirectCommand> chunkIndirectCommands;
    chunkIndirectCommands.reserve(m_chunkDrawRanges.size());
    std::vector<ChunkInstanceData> shadowChunkInstanceData;
    shadowChunkInstanceData.reserve(m_chunkDrawRanges.size() + 1);
    shadowChunkInstanceData.push_back(ChunkInstanceData{});
    std::vector<VkDrawIndexedIndirectCommand> shadowChunkIndirectCommands;
    shadowChunkIndirectCommands.reserve(m_chunkDrawRanges.size());
    std::array<std::vector<VkDrawIndexedIndirectCommand>, kShadowCascadeCount> shadowCascadeIndirectCommands{};
    for (auto& cascadeCommands : shadowCascadeIndirectCommands) {
        cascadeCommands.reserve((m_chunkDrawRanges.size() / kShadowCascadeCount) + 1u);
    }
    const std::vector<voxelsprout::world::Chunk>& chunks = chunkGrid.chunks();
    auto appendChunkLods = [&](
                               std::size_t chunkArrayIndex,
                               std::vector<ChunkInstanceData>& outInstanceData,
                               std::vector<VkDrawIndexedIndirectCommand>& outIndirectCommands,
                               bool countVisibleLodStats
                           ) {
        if (chunkArrayIndex >= chunkGrid.chunks().size()) {
            return;
        }
        const voxelsprout::world::Chunk& drawChunk = chunks[chunkArrayIndex];
        const bool allowDetailLods =
            drawChunk.chunkX() == cameraChunkX &&
            drawChunk.chunkY() == cameraChunkY &&
            drawChunk.chunkZ() == cameraChunkZ;
        for (std::size_t lodIndex = 0; lodIndex < voxelsprout::world::kChunkMeshLodCount; ++lodIndex) {
            if (lodIndex > 0 && !allowDetailLods) {
                continue;
            }
            const std::size_t drawRangeIndex = (chunkArrayIndex * voxelsprout::world::kChunkMeshLodCount) + lodIndex;
            if (drawRangeIndex >= m_chunkDrawRanges.size()) {
                continue;
            }
            const ChunkDrawRange& drawRange = m_chunkDrawRanges[drawRangeIndex];
            if (drawRange.indexCount == 0 || !chunkDrawBuffersReady) {
                continue;
            }

            const uint32_t instanceIndex = static_cast<uint32_t>(outInstanceData.size());
            ChunkInstanceData instance{};
            instance.chunkOffset[0] = drawRange.offsetX;
            instance.chunkOffset[1] = drawRange.offsetY;
            instance.chunkOffset[2] = drawRange.offsetZ;
            instance.chunkOffset[3] = 0.0f;
            outInstanceData.push_back(instance);

            VkDrawIndexedIndirectCommand indirectCommand{};
            indirectCommand.indexCount = drawRange.indexCount;
            indirectCommand.instanceCount = 1;
            indirectCommand.firstIndex = drawRange.firstIndex;
            indirectCommand.vertexOffset = drawRange.vertexOffset;
            indirectCommand.firstInstance = instanceIndex;
            outIndirectCommands.push_back(indirectCommand);

            if (countVisibleLodStats) {
                if (lodIndex == 0) {
                    ++m_debugDrawnLod0Ranges;
                } else if (lodIndex == 1) {
                    ++m_debugDrawnLod1Ranges;
                } else {
                    ++m_debugDrawnLod2Ranges;
                }
            }
        }
    };
    if (!visibleChunkIndices.empty()) {
        for (const std::size_t chunkArrayIndex : visibleChunkIndices) {
            appendChunkLods(chunkArrayIndex, chunkInstanceData, chunkIndirectCommands, true);
        }
    } else {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            appendChunkLods(chunkArrayIndex, chunkInstanceData, chunkIndirectCommands, true);
        }
    }

    const auto appendShadowChunkLods = [&](std::size_t chunkArrayIndex, uint32_t cascadeMask) {
        if (chunkArrayIndex >= chunkGrid.chunks().size()) {
            return;
        }
        const voxelsprout::world::Chunk& drawChunk = chunks[chunkArrayIndex];
        const bool allowDetailLods =
            drawChunk.chunkX() == cameraChunkX &&
            drawChunk.chunkY() == cameraChunkY &&
            drawChunk.chunkZ() == cameraChunkZ;
        for (std::size_t lodIndex = 0; lodIndex < voxelsprout::world::kChunkMeshLodCount; ++lodIndex) {
            if (lodIndex > 0 && !allowDetailLods) {
                continue;
            }
            const std::size_t drawRangeIndex = (chunkArrayIndex * voxelsprout::world::kChunkMeshLodCount) + lodIndex;
            if (drawRangeIndex >= m_chunkDrawRanges.size()) {
                continue;
            }
            const ChunkDrawRange& drawRange = m_chunkDrawRanges[drawRangeIndex];
            if (drawRange.indexCount == 0 || !chunkDrawBuffersReady) {
                continue;
            }

            const uint32_t instanceIndex = static_cast<uint32_t>(shadowChunkInstanceData.size());
            ChunkInstanceData instance{};
            instance.chunkOffset[0] = drawRange.offsetX;
            instance.chunkOffset[1] = drawRange.offsetY;
            instance.chunkOffset[2] = drawRange.offsetZ;
            instance.chunkOffset[3] = 0.0f;
            shadowChunkInstanceData.push_back(instance);

            VkDrawIndexedIndirectCommand indirectCommand{};
            indirectCommand.indexCount = drawRange.indexCount;
            indirectCommand.instanceCount = 1;
            indirectCommand.firstIndex = drawRange.firstIndex;
            indirectCommand.vertexOffset = drawRange.vertexOffset;
            indirectCommand.firstInstance = instanceIndex;
            shadowChunkIndirectCommands.push_back(indirectCommand);
            for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
                if ((cascadeMask & (1u << cascadeIndex)) != 0u) {
                    shadowCascadeIndirectCommands[cascadeIndex].push_back(indirectCommand);
                }
            }
        }
    };

    const std::vector<std::uint8_t> shadowCandidateMask = buildShadowCandidateMask(chunks, visibleChunkIndices);

    constexpr float kShadowCasterClipMargin = 0.08f;
    if (!m_shadowDebugSettings.enableOccluderCulling) {
        const uint32_t allCascadeMask = (1u << kShadowCascadeCount) - 1u;
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            appendShadowChunkLods(chunkArrayIndex, allCascadeMask);
        }
    } else {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            if (!shadowCandidateMask.empty() && shadowCandidateMask[chunkArrayIndex] == 0u) {
                continue;
            }
            const voxelsprout::world::Chunk& chunk = chunks[chunkArrayIndex];
            uint32_t cascadeMask = 0u;
            for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
                if (chunkIntersectsShadowCascadeClip(chunk, lightViewProjMatrices[cascadeIndex], kShadowCasterClipMargin)) {
                    cascadeMask |= (1u << cascadeIndex);
                }
            }
            if (cascadeMask != 0u) {
                appendShadowChunkLods(chunkArrayIndex, cascadeMask);
            }
        }
    }

    const VkDeviceSize chunkInstanceBytes =
        static_cast<VkDeviceSize>(chunkInstanceData.size() * sizeof(ChunkInstanceData));
    std::optional<FrameArenaSlice> chunkInstanceSliceOpt = std::nullopt;
    if (chunkInstanceBytes > 0) {
        chunkInstanceSliceOpt = m_frameArena.allocateUpload(
            chunkInstanceBytes,
            static_cast<VkDeviceSize>(alignof(ChunkInstanceData)),
            FrameArenaUploadKind::InstanceData
        );
        if (chunkInstanceSliceOpt.has_value() && chunkInstanceSliceOpt->mapped != nullptr) {
            std::memcpy(chunkInstanceSliceOpt->mapped, chunkInstanceData.data(), static_cast<size_t>(chunkInstanceBytes));
        } else {
            chunkInstanceSliceOpt.reset();
        }
    }

    const VkDeviceSize chunkIndirectBytes =
        static_cast<VkDeviceSize>(chunkIndirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand));
    std::optional<FrameArenaSlice> chunkIndirectSliceOpt = std::nullopt;
    if (chunkIndirectBytes > 0) {
        chunkIndirectSliceOpt = m_frameArena.allocateUpload(
            chunkIndirectBytes,
            static_cast<VkDeviceSize>(alignof(VkDrawIndexedIndirectCommand)),
            FrameArenaUploadKind::Unknown
        );
        if (chunkIndirectSliceOpt.has_value() && chunkIndirectSliceOpt->mapped != nullptr) {
            std::memcpy(
                chunkIndirectSliceOpt->mapped,
                chunkIndirectCommands.data(),
                static_cast<size_t>(chunkIndirectBytes)
            );
        } else {
            chunkIndirectSliceOpt.reset();
        }
    }

    const VkDeviceSize shadowChunkInstanceBytes =
        static_cast<VkDeviceSize>(shadowChunkInstanceData.size() * sizeof(ChunkInstanceData));
    std::optional<FrameArenaSlice> shadowChunkInstanceSliceOpt = std::nullopt;
    if (shadowChunkInstanceBytes > 0) {
        shadowChunkInstanceSliceOpt = m_frameArena.allocateUpload(
            shadowChunkInstanceBytes,
            static_cast<VkDeviceSize>(alignof(ChunkInstanceData)),
            FrameArenaUploadKind::InstanceData
        );
        if (shadowChunkInstanceSliceOpt.has_value() && shadowChunkInstanceSliceOpt->mapped != nullptr) {
            std::memcpy(
                shadowChunkInstanceSliceOpt->mapped,
                shadowChunkInstanceData.data(),
                static_cast<size_t>(shadowChunkInstanceBytes)
            );
        } else {
            shadowChunkInstanceSliceOpt.reset();
        }
    }

    std::array<std::optional<FrameArenaSlice>, kShadowCascadeCount> shadowCascadeIndirectSliceOpts{};
    std::array<VkBuffer, kShadowCascadeCount> shadowCascadeIndirectBuffers{};
    shadowCascadeIndirectBuffers.fill(VK_NULL_HANDLE);
    std::array<uint32_t, kShadowCascadeCount> shadowCascadeIndirectDrawCounts{};
    shadowCascadeIndirectDrawCounts.fill(0u);
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        const VkDeviceSize shadowCascadeIndirectBytes = static_cast<VkDeviceSize>(
            shadowCascadeIndirectCommands[cascadeIndex].size() * sizeof(VkDrawIndexedIndirectCommand)
        );
        if (shadowCascadeIndirectBytes == 0) {
            continue;
        }
        shadowCascadeIndirectSliceOpts[cascadeIndex] = m_frameArena.allocateUpload(
            shadowCascadeIndirectBytes,
            static_cast<VkDeviceSize>(alignof(VkDrawIndexedIndirectCommand)),
            FrameArenaUploadKind::Unknown
        );
        if (!shadowCascadeIndirectSliceOpts[cascadeIndex].has_value() ||
            shadowCascadeIndirectSliceOpts[cascadeIndex]->mapped == nullptr) {
            shadowCascadeIndirectSliceOpts[cascadeIndex].reset();
            continue;
        }
        std::memcpy(
            shadowCascadeIndirectSliceOpts[cascadeIndex]->mapped,
            shadowCascadeIndirectCommands[cascadeIndex].data(),
            static_cast<size_t>(shadowCascadeIndirectBytes)
        );
        shadowCascadeIndirectBuffers[cascadeIndex] =
            m_bufferAllocator.getBuffer(shadowCascadeIndirectSliceOpts[cascadeIndex]->buffer);
        shadowCascadeIndirectDrawCounts[cascadeIndex] =
            static_cast<uint32_t>(shadowCascadeIndirectCommands[cascadeIndex].size());
    }

    const VkBuffer chunkInstanceBuffer =
        chunkInstanceSliceOpt.has_value() ? m_bufferAllocator.getBuffer(chunkInstanceSliceOpt->buffer) : VK_NULL_HANDLE;
    const VkBuffer chunkIndirectBuffer =
        chunkIndirectSliceOpt.has_value() ? m_bufferAllocator.getBuffer(chunkIndirectSliceOpt->buffer) : VK_NULL_HANDLE;
    const VkBuffer shadowChunkInstanceBuffer =
        shadowChunkInstanceSliceOpt.has_value() ? m_bufferAllocator.getBuffer(shadowChunkInstanceSliceOpt->buffer)
                                                : VK_NULL_HANDLE;
    struct ReadyMagicaDraw {
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        std::uint32_t indexCount = 0;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        float offsetZ = 0.0f;
    };
    std::vector<ReadyMagicaDraw> readyMagicaDraws;
    if (chunkInstanceSliceOpt.has_value() && chunkInstanceBuffer != VK_NULL_HANDLE) {
        readyMagicaDraws.reserve(m_magicaMeshDraws.size());
        for (const MagicaMeshDraw& draw : m_magicaMeshDraws) {
            if (draw.indexCount == 0 ||
                draw.vertexBufferHandle == kInvalidBufferHandle ||
                draw.indexBufferHandle == kInvalidBufferHandle) {
                continue;
            }
            const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(draw.vertexBufferHandle);
            const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(draw.indexBufferHandle);
            if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE) {
                continue;
            }
            readyMagicaDraws.push_back(ReadyMagicaDraw{
                vertexBuffer,
                indexBuffer,
                draw.indexCount,
                draw.offsetX,
                draw.offsetY,
                draw.offsetZ
            });
        }
    }
    const bool canDrawMagica = !readyMagicaDraws.empty() && m_magicaPipeline != VK_NULL_HANDLE;
    const uint32_t chunkIndirectDrawCount = static_cast<uint32_t>(chunkIndirectCommands.size());
    m_debugChunkIndirectCommandCount = chunkIndirectDrawCount;
    const bool canDrawChunksIndirect =
        chunkIndirectDrawCount > 0 &&
        chunkInstanceSliceOpt.has_value() &&
        chunkIndirectSliceOpt.has_value() &&
        chunkInstanceBuffer != VK_NULL_HANDLE &&
        chunkIndirectBuffer != VK_NULL_HANDLE &&
        chunkDrawBuffersReady;
    std::array<bool, kShadowCascadeCount> canDrawShadowChunksIndirectByCascade{};
    canDrawShadowChunksIndirectByCascade.fill(false);
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        canDrawShadowChunksIndirectByCascade[cascadeIndex] =
            shadowCascadeIndirectDrawCounts[cascadeIndex] > 0 &&
            shadowChunkInstanceSliceOpt.has_value() &&
            shadowCascadeIndirectSliceOpts[cascadeIndex].has_value() &&
            shadowChunkInstanceBuffer != VK_NULL_HANDLE &&
            shadowCascadeIndirectBuffers[cascadeIndex] != VK_NULL_HANDLE &&
            chunkDrawBuffersReady;
    }
    auto countDrawCalls = [&](std::uint32_t& passCounter, std::uint32_t drawCount) {
        passCounter += drawCount;
        m_debugDrawCallsTotal += drawCount;
    };
    const auto drawChunkIndirect = [&](std::uint32_t& passCounter) {
        if (!canDrawChunksIndirect) {
            return;
        }
        if (m_supportsMultiDrawIndirect) {
            countDrawCalls(passCounter, chunkIndirectDrawCount);
            vkCmdDrawIndexedIndirect(
                commandBuffer,
                chunkIndirectBuffer,
                chunkIndirectSliceOpt->offset,
                chunkIndirectDrawCount,
                sizeof(VkDrawIndexedIndirectCommand)
            );
            return;
        }
        const VkDeviceSize stride = static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand));
        VkDeviceSize drawOffset = chunkIndirectSliceOpt->offset;
        for (uint32_t drawIndex = 0; drawIndex < chunkIndirectDrawCount; ++drawIndex) {
            countDrawCalls(passCounter, 1);
            vkCmdDrawIndexedIndirect(
                commandBuffer,
                chunkIndirectBuffer,
                drawOffset,
                1,
                static_cast<uint32_t>(stride)
            );
            drawOffset += stride;
        }
    };
    const auto drawShadowChunkIndirect = [&](std::uint32_t& passCounter, uint32_t cascadeIndex) {
        if (cascadeIndex >= kShadowCascadeCount || !canDrawShadowChunksIndirectByCascade[cascadeIndex]) {
            return;
        }
        const uint32_t cascadeDrawCount = shadowCascadeIndirectDrawCounts[cascadeIndex];
        const VkBuffer cascadeIndirectBuffer = shadowCascadeIndirectBuffers[cascadeIndex];
        const std::optional<FrameArenaSlice>& cascadeIndirectSlice = shadowCascadeIndirectSliceOpts[cascadeIndex];
        if (!cascadeIndirectSlice.has_value()) {
            return;
        }
        if (m_supportsMultiDrawIndirect) {
            countDrawCalls(passCounter, cascadeDrawCount);
            vkCmdDrawIndexedIndirect(
                commandBuffer,
                cascadeIndirectBuffer,
                cascadeIndirectSlice->offset,
                cascadeDrawCount,
                sizeof(VkDrawIndexedIndirectCommand)
            );
            return;
        }
        const VkDeviceSize stride = static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand));
        VkDeviceSize drawOffset = cascadeIndirectSlice->offset;
        for (uint32_t drawIndex = 0; drawIndex < cascadeDrawCount; ++drawIndex) {
            countDrawCalls(passCounter, 1);
            vkCmdDrawIndexedIndirect(
                commandBuffer,
                cascadeIndirectBuffer,
                drawOffset,
                1,
                static_cast<uint32_t>(stride)
            );
            drawOffset += stride;
        }
    };

    writeGpuTimestampTop(kGpuTimestampQueryShadowStart);
    beginDebugLabel(commandBuffer, "Pass: Shadow Atlas", 0.28f, 0.22f, 0.22f, 1.0f);
    const bool shadowInitialized = m_shadowDepthInitialized;
    transitionImageLayout(
        commandBuffer,
        m_shadowDepthImage,
        shadowInitialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        shadowInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        shadowInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        0,
        1
    );

    VkClearValue shadowDepthClearValue{};
    shadowDepthClearValue.depthStencil.depth = 0.0f;
    shadowDepthClearValue.depthStencil.stencil = 0;

    if (m_shadowPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );

        for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
            if (m_cmdInsertDebugUtilsLabel != nullptr) {
                const std::string cascadeLabel = "Shadow Cascade " + std::to_string(cascadeIndex);
                insertDebugLabel(commandBuffer, cascadeLabel.c_str(), 0.48f, 0.32f, 0.32f, 1.0f);
            }
            const ShadowAtlasRect atlasRect = kShadowAtlasRects[cascadeIndex];
            VkViewport shadowViewport{};
            shadowViewport.x = static_cast<float>(atlasRect.x);
            shadowViewport.y = static_cast<float>(atlasRect.y);
            shadowViewport.width = static_cast<float>(atlasRect.size);
            shadowViewport.height = static_cast<float>(atlasRect.size);
            shadowViewport.minDepth = 0.0f;
            shadowViewport.maxDepth = 1.0f;

            VkRect2D shadowScissor{};
            shadowScissor.offset = {
                static_cast<int32_t>(atlasRect.x),
                static_cast<int32_t>(atlasRect.y)
            };
            shadowScissor.extent = {atlasRect.size, atlasRect.size};

            VkRenderingAttachmentInfo shadowDepthAttachment{};
            shadowDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            shadowDepthAttachment.imageView = m_shadowDepthImageView;
            shadowDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            shadowDepthAttachment.clearValue = shadowDepthClearValue;

            VkRenderingInfo shadowRenderingInfo{};
            shadowRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            shadowRenderingInfo.renderArea.offset = shadowScissor.offset;
            shadowRenderingInfo.renderArea.extent = shadowScissor.extent;
            shadowRenderingInfo.layerCount = 1;
            shadowRenderingInfo.colorAttachmentCount = 0;
            shadowRenderingInfo.pDepthAttachment = &shadowDepthAttachment;

            vkCmdBeginRendering(commandBuffer, &shadowRenderingInfo);
            vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);
            vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);
            const float cascadeF = static_cast<float>(cascadeIndex);
            const float constantBias =
                m_shadowDebugSettings.casterConstantBiasBase +
                (m_shadowDebugSettings.casterConstantBiasCascadeScale * cascadeF);
            const float slopeBias =
                m_shadowDebugSettings.casterSlopeBiasBase +
                (m_shadowDebugSettings.casterSlopeBiasCascadeScale * cascadeF);
            // Reverse-Z uses GREATER depth tests, so flip bias sign.
            vkCmdSetDepthBias(commandBuffer, -constantBias, 0.0f, -slopeBias);

            if (cascadeIndex < kShadowCascadeCount && canDrawShadowChunksIndirectByCascade[cascadeIndex]) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
                vkCmdBindDescriptorSets(
                    commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    boundDescriptorSetCount,
                    boundDescriptorSets.data(),
                    1,
                    &mvpDynamicOffset
                );
                const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, shadowChunkInstanceBuffer};
                const VkDeviceSize voxelVertexOffsets[2] = {0, shadowChunkInstanceSliceOpt->offset};
                vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

                ChunkPushConstants chunkPushConstants{};
                chunkPushConstants.chunkOffset[0] = 0.0f;
                chunkPushConstants.chunkOffset[1] = 0.0f;
                chunkPushConstants.chunkOffset[2] = 0.0f;
                chunkPushConstants.chunkOffset[3] = 0.0f;
                chunkPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                chunkPushConstants.cascadeData[1] = 0.0f;
                chunkPushConstants.cascadeData[2] = 0.0f;
                chunkPushConstants.cascadeData[3] = 0.0f;
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &chunkPushConstants
                );
                drawShadowChunkIndirect(m_debugDrawCallsShadow, cascadeIndex);
            }
            if (canDrawMagica) {
                for (const ReadyMagicaDraw& magicaDraw : readyMagicaDraws) {
                    const VkBuffer magicaVertexBuffers[2] = {magicaDraw.vertexBuffer, chunkInstanceBuffer};
                    const VkDeviceSize magicaVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 2, magicaVertexBuffers, magicaVertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, magicaDraw.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

                    ChunkPushConstants magicaPushConstants{};
                    magicaPushConstants.chunkOffset[0] = magicaDraw.offsetX;
                    magicaPushConstants.chunkOffset[1] = magicaDraw.offsetY;
                    magicaPushConstants.chunkOffset[2] = magicaDraw.offsetZ;
                    magicaPushConstants.chunkOffset[3] = 0.0f;
                    magicaPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                    magicaPushConstants.cascadeData[1] = 0.0f;
                    magicaPushConstants.cascadeData[2] = 0.0f;
                    magicaPushConstants.cascadeData[3] = 0.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &magicaPushConstants
                    );
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(commandBuffer, magicaDraw.indexCount, 1, 0, 0, 0);
                }
            }

            if (m_pipeShadowPipeline != VK_NULL_HANDLE) {
                auto drawShadowInstances = [&](
                                               BufferHandle vertexHandle,
                                               BufferHandle indexHandle,
                                               uint32_t indexCount,
                                               uint32_t instanceCount,
                                           const std::optional<FrameArenaSlice>& instanceSlice
                                           ) {
                    if (instanceCount == 0 || !instanceSlice.has_value() || indexCount == 0) {
                        return;
                    }
                    const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(vertexHandle);
                    const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(indexHandle);
                    const VkBuffer instanceBuffer = m_bufferAllocator.getBuffer(instanceSlice->buffer);
                    if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE || instanceBuffer == VK_NULL_HANDLE) {
                        return;
                    }
                    const VkBuffer vertexBuffers[2] = {vertexBuffer, instanceBuffer};
                    const VkDeviceSize vertexOffsets[2] = {0, instanceSlice->offset};
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeShadowPipeline);
                    vkCmdBindDescriptorSets(
                        commandBuffer,
                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                        m_pipelineLayout,
                        0,
                        boundDescriptorSetCount,
                        boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
                    vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

                    ChunkPushConstants pipeShadowPushConstants{};
                    pipeShadowPushConstants.chunkOffset[0] = 0.0f;
                    pipeShadowPushConstants.chunkOffset[1] = 0.0f;
                    pipeShadowPushConstants.chunkOffset[2] = 0.0f;
                    pipeShadowPushConstants.chunkOffset[3] = 0.0f;
                    pipeShadowPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                    pipeShadowPushConstants.cascadeData[1] = 0.0f;
                    pipeShadowPushConstants.cascadeData[2] = 0.0f;
                    pipeShadowPushConstants.cascadeData[3] = 0.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &pipeShadowPushConstants
                    );
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, 0, 0, 0);
                };
                drawShadowInstances(
                    m_pipeVertexBufferHandle,
                    m_pipeIndexBufferHandle,
                    m_pipeIndexCount,
                    pipeInstanceCount,
                    pipeInstanceSliceOpt
                );
                drawShadowInstances(
                    m_transportVertexBufferHandle,
                    m_transportIndexBufferHandle,
                    m_transportIndexCount,
                    transportInstanceCount,
                    transportInstanceSliceOpt
                );
                drawShadowInstances(
                    m_transportVertexBufferHandle,
                    m_transportIndexBufferHandle,
                    m_transportIndexCount,
                    beltCargoInstanceCount,
                    beltCargoInstanceSliceOpt
                );
            }

            const uint32_t grassShadowCascadeCount = static_cast<uint32_t>(std::clamp(
                m_shadowDebugSettings.grassShadowCascadeCount,
                0,
                static_cast<int>(kShadowCascadeCount)
            ));
            if (cascadeIndex < grassShadowCascadeCount &&
                m_grassBillboardShadowPipeline != VK_NULL_HANDLE &&
                m_grassBillboardIndexCount > 0 &&
                m_grassBillboardInstanceCount > 0 &&
                m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
                const VkBuffer grassVertexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardVertexBufferHandle);
                const VkBuffer grassIndexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardIndexBufferHandle);
                const VkBuffer grassInstanceBuffer = m_bufferAllocator.getBuffer(m_grassBillboardInstanceBufferHandle);
                if (grassVertexBuffer != VK_NULL_HANDLE &&
                    grassIndexBuffer != VK_NULL_HANDLE &&
                    grassInstanceBuffer != VK_NULL_HANDLE) {
                    const VkBuffer vertexBuffers[2] = {grassVertexBuffer, grassInstanceBuffer};
                    const VkDeviceSize vertexOffsets[2] = {0, 0};
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_grassBillboardShadowPipeline);
                    vkCmdBindDescriptorSets(
                        commandBuffer,
                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                        m_pipelineLayout,
                        0,
                        boundDescriptorSetCount,
                        boundDescriptorSets.data(),
                        1,
                        &mvpDynamicOffset
                    );
                    vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
                    vkCmdBindIndexBuffer(commandBuffer, grassIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

                    ChunkPushConstants grassShadowPushConstants{};
                    grassShadowPushConstants.chunkOffset[0] = 0.0f;
                    grassShadowPushConstants.chunkOffset[1] = 0.0f;
                    grassShadowPushConstants.chunkOffset[2] = 0.0f;
                    grassShadowPushConstants.chunkOffset[3] = 0.0f;
                    grassShadowPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                    grassShadowPushConstants.cascadeData[1] = 0.0f;
                    grassShadowPushConstants.cascadeData[2] = 0.0f;
                    grassShadowPushConstants.cascadeData[3] = 0.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &grassShadowPushConstants
                    );
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(commandBuffer, m_grassBillboardIndexCount, m_grassBillboardInstanceCount, 0, 0, 0);
                }
            }

            vkCmdEndRendering(commandBuffer);
        }
    }

    transitionImageLayout(
        commandBuffer,
        m_shadowDepthImage,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        0,
        1
    );
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryShadowEnd);

    bool wroteVoxelGiTimestamps = false;
    bool wroteAutoExposureTimestamps = false;
    bool wroteSunShaftTimestamps = false;
    const bool voxelGiSurfaceFacesReady = std::all_of(
        m_voxelGiSurfaceFaceImages.begin(),
        m_voxelGiSurfaceFaceImages.end(),
        [](VkImage image) { return image != VK_NULL_HANDLE; }
    );
    if (m_voxelGiComputeAvailable &&
        m_voxelGiSkyExposurePipeline != VK_NULL_HANDLE &&
        m_voxelGiSurfacePipeline != VK_NULL_HANDLE &&
        m_voxelGiInjectPipeline != VK_NULL_HANDLE &&
        m_voxelGiPropagatePipeline != VK_NULL_HANDLE &&
        m_voxelGiPipelineLayout != VK_NULL_HANDLE &&
        m_voxelGiDescriptorSets[m_currentFrame] != VK_NULL_HANDLE &&
        voxelGiSurfaceFacesReady &&
        m_voxelGiSkyExposureImage != VK_NULL_HANDLE &&
        m_voxelGiOccupancyImage != VK_NULL_HANDLE &&
        voxelGiNeedsComputeUpdate &&
        (!voxelGiNeedsOccupancyUpload || voxelGiHasOccupancyUpload)) {
        wroteVoxelGiTimestamps = true;
        beginDebugLabel(commandBuffer, "Pass: Voxel GI", 0.38f, 0.28f, 0.12f, 1.0f);

        if (voxelGiNeedsOccupancyUpload) {
            transitionImageLayout(
                commandBuffer,
                m_voxelGiOccupancyImage,
                m_voxelGiOccupancyInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                m_voxelGiOccupancyInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
                m_voxelGiOccupancyInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
            VkBufferImageCopy occupancyCopyRegion{};
            occupancyCopyRegion.bufferOffset = voxelGiOccupancySliceOpt->offset;
            occupancyCopyRegion.bufferRowLength = 0;
            occupancyCopyRegion.bufferImageHeight = 0;
            occupancyCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            occupancyCopyRegion.imageSubresource.mipLevel = 0;
            occupancyCopyRegion.imageSubresource.baseArrayLayer = 0;
            occupancyCopyRegion.imageSubresource.layerCount = 1;
            occupancyCopyRegion.imageOffset = {0, 0, 0};
            occupancyCopyRegion.imageExtent = {
                kVoxelGiGridResolution,
                kVoxelGiGridResolution,
                kVoxelGiGridResolution
            };
            vkCmdCopyBufferToImage(
                commandBuffer,
                voxelGiOccupancyUploadBuffer,
                m_voxelGiOccupancyImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &occupancyCopyRegion
            );
            transitionImageLayout(
                commandBuffer,
                m_voxelGiOccupancyImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
            m_voxelGiOccupancyInitialized = true;
        }

        for (std::size_t faceIndex = 0; faceIndex < m_voxelGiSurfaceFaceImages.size(); ++faceIndex) {
            transitionImageLayout(
                commandBuffer,
                m_voxelGiSurfaceFaceImages[faceIndex],
                m_voxelGiInitialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                m_voxelGiInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
                m_voxelGiInitialized
                    ? (VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)
                    : VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
        }
        transitionImageLayout(
            commandBuffer,
            m_voxelGiSkyExposureImage,
            m_voxelGiSkyExposureInitialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            m_voxelGiSkyExposureInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
            m_voxelGiSkyExposureInitialized ? VK_ACCESS_2_SHADER_STORAGE_READ_BIT : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        transitionImageLayout(
            commandBuffer,
            m_voxelGiImages[0],
            m_voxelGiInitialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            m_voxelGiInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
            m_voxelGiInitialized ? (VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        transitionImageLayout(
            commandBuffer,
            m_voxelGiImages[1],
            m_voxelGiInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            m_voxelGiInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
            m_voxelGiInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );

        recordVoxelGiDispatchSequence(commandBuffer, mvpDynamicOffset, gpuTimestampQueryPool);
        endDebugLabel(commandBuffer);
    } else if (!m_voxelGiInitialized &&
               m_voxelGiImages[0] != VK_NULL_HANDLE &&
               m_voxelGiImages[1] != VK_NULL_HANDLE) {
        transitionImageLayout(
            commandBuffer,
            m_voxelGiImages[0],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        transitionImageLayout(
            commandBuffer,
            m_voxelGiImages[1],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );

        VkClearColorValue clearColor{};
        clearColor.float32[0] = 0.0f;
        clearColor.float32[1] = 0.0f;
        clearColor.float32[2] = 0.0f;
        clearColor.float32[3] = 1.0f;
        VkImageSubresourceRange clearRange{};
        clearRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        clearRange.baseMipLevel = 0;
        clearRange.levelCount = 1;
        clearRange.baseArrayLayer = 0;
        clearRange.layerCount = 1;
        vkCmdClearColorImage(
            commandBuffer,
            m_voxelGiImages[0],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &clearColor,
            1,
            &clearRange
        );
        vkCmdClearColorImage(
            commandBuffer,
            m_voxelGiImages[1],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &clearColor,
            1,
            &clearRange
        );

        transitionImageLayout(
            commandBuffer,
            m_voxelGiImages[0],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        transitionImageLayout(
            commandBuffer,
            m_voxelGiImages[1],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        m_voxelGiInitialized = true;
    }
    if (!wroteVoxelGiTimestamps) {
        writeGpuTimestampTop(kGpuTimestampQueryGiInjectStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiInjectEnd);
        writeGpuTimestampTop(kGpuTimestampQueryGiPropagateStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiPropagateEnd);
    }

    const VkExtent2D aoExtent = {
        std::max(1u, m_aoExtent.width),
        std::max(1u, m_aoExtent.height)
    };

    const bool normalDepthInitialized = m_normalDepthImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer,
        m_normalDepthImages[aoFrameIndex],
        normalDepthInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        normalDepthInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        normalDepthInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    const bool aoDepthInitialized = m_aoDepthImageInitialized[imageIndex];
    transitionImageLayout(
        commandBuffer,
        m_aoDepthImages[imageIndex],
        aoDepthInitialized ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        aoDepthInitialized
            ? (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
            : VK_PIPELINE_STAGE_2_NONE,
        aoDepthInitialized ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT
    );

    const bool ssaoRawInitialized = m_ssaoRawImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer,
        m_ssaoRawImages[aoFrameIndex],
        ssaoRawInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        ssaoRawInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        ssaoRawInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    const bool ssaoBlurInitialized = m_ssaoBlurImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer,
        m_ssaoBlurImages[aoFrameIndex],
        ssaoBlurInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        ssaoBlurInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        ssaoBlurInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkViewport aoViewport{};
    aoViewport.x = 0.0f;
    aoViewport.y = 0.0f;
    aoViewport.width = static_cast<float>(aoExtent.width);
    aoViewport.height = static_cast<float>(aoExtent.height);
    aoViewport.minDepth = 0.0f;
    aoViewport.maxDepth = 1.0f;

    VkRect2D aoScissor{};
    aoScissor.offset = {0, 0};
    aoScissor.extent = aoExtent;

    VkClearValue normalDepthClearValue{};
    normalDepthClearValue.color.float32[0] = 0.5f;
    normalDepthClearValue.color.float32[1] = 0.5f;
    normalDepthClearValue.color.float32[2] = 0.5f;
    normalDepthClearValue.color.float32[3] = 0.0f;

    VkClearValue aoDepthClearValue{};
    aoDepthClearValue.depthStencil.depth = 0.0f;
    aoDepthClearValue.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo normalDepthColorAttachment{};
    normalDepthColorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    normalDepthColorAttachment.imageView = m_normalDepthImageViews[aoFrameIndex];
    normalDepthColorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    normalDepthColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    normalDepthColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    normalDepthColorAttachment.clearValue = normalDepthClearValue;

    VkRenderingAttachmentInfo aoDepthAttachment{};
    aoDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    aoDepthAttachment.imageView = m_aoDepthImageViews[imageIndex];
    aoDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    aoDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    aoDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    aoDepthAttachment.clearValue = aoDepthClearValue;

    VkRenderingInfo normalDepthRenderingInfo{};
    normalDepthRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    normalDepthRenderingInfo.renderArea.offset = {0, 0};
    normalDepthRenderingInfo.renderArea.extent = aoExtent;
    normalDepthRenderingInfo.layerCount = 1;
    normalDepthRenderingInfo.colorAttachmentCount = 1;
    normalDepthRenderingInfo.pColorAttachments = &normalDepthColorAttachment;
    normalDepthRenderingInfo.pDepthAttachment = &aoDepthAttachment;

    writeGpuTimestampTop(kGpuTimestampQueryPrepassStart);
    beginDebugLabel(commandBuffer, "Pass: Normal+Depth Prepass", 0.20f, 0.30f, 0.40f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &normalDepthRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &aoViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &aoScissor);

    if (m_voxelNormalDepthPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_voxelNormalDepthPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        if (canDrawChunksIndirect) {
            const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, chunkInstanceBuffer};
            const VkDeviceSize voxelVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

            ChunkPushConstants chunkPushConstants{};
            chunkPushConstants.chunkOffset[0] = 0.0f;
            chunkPushConstants.chunkOffset[1] = 0.0f;
            chunkPushConstants.chunkOffset[2] = 0.0f;
            chunkPushConstants.chunkOffset[3] = 0.0f;
            chunkPushConstants.cascadeData[0] = 0.0f;
            chunkPushConstants.cascadeData[1] = 0.0f;
            chunkPushConstants.cascadeData[2] = 0.0f;
            chunkPushConstants.cascadeData[3] = 0.0f;
            vkCmdPushConstants(
                commandBuffer,
                m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(ChunkPushConstants),
                &chunkPushConstants
            );
            drawChunkIndirect(m_debugDrawCallsPrepass);
        }
        if (canDrawMagica) {
            for (const ReadyMagicaDraw& magicaDraw : readyMagicaDraws) {
                const VkBuffer magicaVertexBuffers[2] = {magicaDraw.vertexBuffer, chunkInstanceBuffer};
                const VkDeviceSize magicaVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
                vkCmdBindVertexBuffers(commandBuffer, 0, 2, magicaVertexBuffers, magicaVertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, magicaDraw.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

                ChunkPushConstants magicaPushConstants{};
                magicaPushConstants.chunkOffset[0] = magicaDraw.offsetX;
                magicaPushConstants.chunkOffset[1] = magicaDraw.offsetY;
                magicaPushConstants.chunkOffset[2] = magicaDraw.offsetZ;
                magicaPushConstants.chunkOffset[3] = 0.0f;
                magicaPushConstants.cascadeData[0] = 0.0f;
                magicaPushConstants.cascadeData[1] = 0.0f;
                magicaPushConstants.cascadeData[2] = 0.0f;
                magicaPushConstants.cascadeData[3] = 0.0f;
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &magicaPushConstants
                );
                countDrawCalls(m_debugDrawCallsPrepass, 1);
                vkCmdDrawIndexed(commandBuffer, magicaDraw.indexCount, 1, 0, 0, 0);
            }
        }
    }

    if (m_pipeNormalDepthPipeline != VK_NULL_HANDLE) {
        auto drawNormalDepthInstances = [&](
                                            BufferHandle vertexHandle,
                                            BufferHandle indexHandle,
                                            uint32_t indexCount,
                                            uint32_t instanceCount,
                                        const std::optional<FrameArenaSlice>& instanceSlice
                                        ) {
            if (instanceCount == 0 || !instanceSlice.has_value() || indexCount == 0) {
                return;
            }
            const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(vertexHandle);
            const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(indexHandle);
            const VkBuffer instanceBuffer = m_bufferAllocator.getBuffer(instanceSlice->buffer);
            if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE || instanceBuffer == VK_NULL_HANDLE) {
                return;
            }
            const VkBuffer vertexBuffers[2] = {vertexBuffer, instanceBuffer};
            const VkDeviceSize vertexOffsets[2] = {0, instanceSlice->offset};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeNormalDepthPipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                boundDescriptorSetCount,
                boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsPrepass, 1);
            vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, 0, 0, 0);
        };
        drawNormalDepthInstances(
            m_pipeVertexBufferHandle,
            m_pipeIndexBufferHandle,
            m_pipeIndexCount,
            pipeInstanceCount,
            pipeInstanceSliceOpt
        );
        drawNormalDepthInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            transportInstanceCount,
            transportInstanceSliceOpt
        );
        drawNormalDepthInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            beltCargoInstanceCount,
            beltCargoInstanceSliceOpt
        );
    }
    if (m_grassBillboardNormalDepthPipeline != VK_NULL_HANDLE &&
        m_grassBillboardIndexCount > 0 &&
        m_grassBillboardInstanceCount > 0 &&
        m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
        const VkBuffer grassVertexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardVertexBufferHandle);
        const VkBuffer grassIndexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardIndexBufferHandle);
        const VkBuffer grassInstanceBuffer = m_bufferAllocator.getBuffer(m_grassBillboardInstanceBufferHandle);
        if (grassVertexBuffer != VK_NULL_HANDLE &&
            grassIndexBuffer != VK_NULL_HANDLE &&
            grassInstanceBuffer != VK_NULL_HANDLE) {
            const VkBuffer vertexBuffers[2] = {grassVertexBuffer, grassInstanceBuffer};
            const VkDeviceSize vertexOffsets[2] = {0, 0};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_grassBillboardNormalDepthPipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                boundDescriptorSetCount,
                boundDescriptorSets.data(),
                1,
                &mvpDynamicOffset
            );
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, grassIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsPrepass, 1);
            vkCmdDrawIndexed(commandBuffer, m_grassBillboardIndexCount, m_grassBillboardInstanceCount, 0, 0, 0);
        }
    }
    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryPrepassEnd);

    transitionImageLayout(
        commandBuffer,
        m_normalDepthImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkClearValue ssaoClearValue{};
    ssaoClearValue.color.float32[0] = 1.0f;
    ssaoClearValue.color.float32[1] = 1.0f;
    ssaoClearValue.color.float32[2] = 1.0f;
    ssaoClearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo ssaoRawAttachment{};
    ssaoRawAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ssaoRawAttachment.imageView = m_ssaoRawImageViews[aoFrameIndex];
    ssaoRawAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ssaoRawAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ssaoRawAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ssaoRawAttachment.clearValue = ssaoClearValue;

    VkRenderingInfo ssaoRenderingInfo{};
    ssaoRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ssaoRenderingInfo.renderArea.offset = {0, 0};
    ssaoRenderingInfo.renderArea.extent = aoExtent;
    ssaoRenderingInfo.layerCount = 1;
    ssaoRenderingInfo.colorAttachmentCount = 1;
    ssaoRenderingInfo.pColorAttachments = &ssaoRawAttachment;

    writeGpuTimestampTop(kGpuTimestampQuerySsaoStart);
    beginDebugLabel(commandBuffer, "Pass: SSAO", 0.20f, 0.36f, 0.26f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &ssaoRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &aoViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &aoScissor);
    if (m_ssaoPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ssaoPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        countDrawCalls(m_debugDrawCallsPrepass, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQuerySsaoEnd);

    transitionImageLayout(
        commandBuffer,
        m_ssaoRawImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkRenderingAttachmentInfo ssaoBlurAttachment{};
    ssaoBlurAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ssaoBlurAttachment.imageView = m_ssaoBlurImageViews[aoFrameIndex];
    ssaoBlurAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ssaoBlurAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ssaoBlurAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ssaoBlurAttachment.clearValue = ssaoClearValue;

    VkRenderingInfo ssaoBlurRenderingInfo{};
    ssaoBlurRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ssaoBlurRenderingInfo.renderArea.offset = {0, 0};
    ssaoBlurRenderingInfo.renderArea.extent = aoExtent;
    ssaoBlurRenderingInfo.layerCount = 1;
    ssaoBlurRenderingInfo.colorAttachmentCount = 1;
    ssaoBlurRenderingInfo.pColorAttachments = &ssaoBlurAttachment;

    writeGpuTimestampTop(kGpuTimestampQuerySsaoBlurStart);
    beginDebugLabel(commandBuffer, "Pass: SSAO Blur", 0.22f, 0.40f, 0.30f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &ssaoBlurRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &aoViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &aoScissor);
    if (m_ssaoBlurPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ssaoBlurPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        countDrawCalls(m_debugDrawCallsPrepass, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQuerySsaoBlurEnd);

    transitionImageLayout(
        commandBuffer,
        m_ssaoBlurImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    m_normalDepthImageInitialized[aoFrameIndex] = true;
    m_aoDepthImageInitialized[imageIndex] = true;
    m_ssaoRawImageInitialized[aoFrameIndex] = true;
    m_ssaoBlurImageInitialized[aoFrameIndex] = true;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchainExtent.width);
    viewport.height = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchainExtent;

    if (!m_msaaColorImageInitialized[imageIndex]) {
        transitionImageLayout(
            commandBuffer,
            m_msaaColorImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }
    const bool hdrResolveInitialized = m_hdrResolveImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer,
        m_hdrResolveImages[aoFrameIndex],
        hdrResolveInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        hdrResolveInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        hdrResolveInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    transitionImageLayout(
        commandBuffer,
        m_depthImages[imageIndex],
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT
    );

    VkClearValue clearValue{};
    clearValue.color.float32[0] = 0.06f;
    clearValue.color.float32[1] = 0.08f;
    clearValue.color.float32[2] = 0.12f;
    clearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = m_msaaColorImageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.clearValue = clearValue;
    colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    colorAttachment.resolveImageView = m_hdrResolveImageViews[aoFrameIndex];
    colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkClearValue depthClearValue{};
    depthClearValue.depthStencil.depth = 0.0f;
    depthClearValue.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = m_depthImageViews[imageIndex];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue = depthClearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = m_swapchainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    writeGpuTimestampTop(kGpuTimestampQueryMainStart);
    beginDebugLabel(commandBuffer, "Pass: Main Scene", 0.20f, 0.20f, 0.45f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout,
        0,
        boundDescriptorSetCount,
        boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
    if (canDrawChunksIndirect) {
        const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, chunkInstanceBuffer};
        const VkDeviceSize voxelVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

        ChunkPushConstants chunkPushConstants{};
        chunkPushConstants.chunkOffset[0] = 0.0f;
        chunkPushConstants.chunkOffset[1] = 0.0f;
        chunkPushConstants.chunkOffset[2] = 0.0f;
        chunkPushConstants.chunkOffset[3] = 0.0f;
        chunkPushConstants.cascadeData[0] = 0.0f;
        chunkPushConstants.cascadeData[1] = 0.0f;
        chunkPushConstants.cascadeData[2] = 0.0f;
        chunkPushConstants.cascadeData[3] = 0.0f;
        vkCmdPushConstants(
            commandBuffer,
            m_pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(ChunkPushConstants),
            &chunkPushConstants
        );
        drawChunkIndirect(m_debugDrawCallsMain);
    }
    if (canDrawMagica) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_magicaPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        for (const ReadyMagicaDraw& magicaDraw : readyMagicaDraws) {
            const VkBuffer magicaVertexBuffers[2] = {magicaDraw.vertexBuffer, chunkInstanceBuffer};
            const VkDeviceSize magicaVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, magicaVertexBuffers, magicaVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, magicaDraw.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            ChunkPushConstants magicaPushConstants{};
            magicaPushConstants.chunkOffset[0] = magicaDraw.offsetX;
            magicaPushConstants.chunkOffset[1] = magicaDraw.offsetY;
            magicaPushConstants.chunkOffset[2] = magicaDraw.offsetZ;
            magicaPushConstants.chunkOffset[3] = 0.0f;
            magicaPushConstants.cascadeData[0] = 0.0f;
            magicaPushConstants.cascadeData[1] = 0.0f;
            magicaPushConstants.cascadeData[2] = 0.0f;
            magicaPushConstants.cascadeData[3] = 0.0f;
            vkCmdPushConstants(
                commandBuffer,
                m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(ChunkPushConstants),
                &magicaPushConstants
            );
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, magicaDraw.indexCount, 1, 0, 0, 0);
        }
    }

    if (m_pipePipeline != VK_NULL_HANDLE) {
        auto drawLitInstances = [&](
                                    BufferHandle vertexHandle,
                                    BufferHandle indexHandle,
                                    uint32_t indexCount,
                                    uint32_t instanceCount,
                                const std::optional<FrameArenaSlice>& instanceSlice
                                ) {
            if (instanceCount == 0 || !instanceSlice.has_value() || indexCount == 0) {
                return;
            }
            const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(vertexHandle);
            const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(indexHandle);
            const VkBuffer instanceBuffer = m_bufferAllocator.getBuffer(instanceSlice->buffer);
            if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE || instanceBuffer == VK_NULL_HANDLE) {
                return;
            }
            const VkBuffer vertexBuffers[2] = {vertexBuffer, instanceBuffer};
            const VkDeviceSize vertexOffsets[2] = {0, instanceSlice->offset};

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipePipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                boundDescriptorSetCount,
                boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, 0, 0, 0);
        };
        drawLitInstances(
            m_pipeVertexBufferHandle,
            m_pipeIndexBufferHandle,
            m_pipeIndexCount,
            pipeInstanceCount,
            pipeInstanceSliceOpt
        );
        drawLitInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            transportInstanceCount,
            transportInstanceSliceOpt
        );
        drawLitInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            beltCargoInstanceCount,
            beltCargoInstanceSliceOpt
        );
    }

    if (m_grassBillboardPipeline != VK_NULL_HANDLE &&
        m_grassBillboardIndexCount > 0 &&
        m_grassBillboardInstanceCount > 0 &&
        m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
        const VkBuffer grassVertexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardVertexBufferHandle);
        const VkBuffer grassIndexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardIndexBufferHandle);
        const VkBuffer grassInstanceBuffer = m_bufferAllocator.getBuffer(m_grassBillboardInstanceBufferHandle);
        if (grassVertexBuffer != VK_NULL_HANDLE &&
            grassIndexBuffer != VK_NULL_HANDLE &&
            grassInstanceBuffer != VK_NULL_HANDLE) {
            const VkBuffer vertexBuffers[2] = {grassVertexBuffer, grassInstanceBuffer};
            const VkDeviceSize vertexOffsets[2] = {0, 0};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_grassBillboardPipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                boundDescriptorSetCount,
                boundDescriptorSets.data(),
                1,
                &mvpDynamicOffset
            );
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, grassIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, m_grassBillboardIndexCount, m_grassBillboardInstanceCount, 0, 0, 0);
        }
    }

    const VkPipeline activePreviewPipeline =
        (preview.mode == VoxelPreview::Mode::Remove) ? m_previewRemovePipeline : m_previewAddPipeline;
    const bool drawCubePreview = !preview.pipeStyle && preview.visible && activePreviewPipeline != VK_NULL_HANDLE;
    const bool drawFacePreview =
        !preview.pipeStyle && preview.faceVisible && preview.brushSize == 1 && m_previewRemovePipeline != VK_NULL_HANDLE;

    if (preview.pipeStyle && preview.visible && m_pipePipeline != VK_NULL_HANDLE) {
        PipeInstance previewInstance{};
        previewInstance.originLength[0] = static_cast<float>(preview.x);
        previewInstance.originLength[1] = static_cast<float>(preview.y);
        previewInstance.originLength[2] = static_cast<float>(preview.z);
        previewInstance.originLength[3] = 1.0f;
        voxelsprout::math::Vector3 previewAxis = voxelsprout::math::normalize(voxelsprout::math::Vector3{preview.pipeAxisX, preview.pipeAxisY, preview.pipeAxisZ});
        if (voxelsprout::math::lengthSquared(previewAxis) <= 0.0001f) {
            previewAxis = voxelsprout::math::Vector3{0.0f, 1.0f, 0.0f};
        }
        previewInstance.axisRadius[0] = previewAxis.x;
        previewInstance.axisRadius[1] = previewAxis.y;
        previewInstance.axisRadius[2] = previewAxis.z;
        previewInstance.axisRadius[3] = std::clamp(preview.pipeRadius, 0.02f, 0.5f);
        if (preview.mode == VoxelPreview::Mode::Remove) {
            previewInstance.tint[0] = 1.0f;
            previewInstance.tint[1] = 0.32f;
            previewInstance.tint[2] = 0.26f;
        } else {
            previewInstance.tint[0] = 0.30f;
            previewInstance.tint[1] = 0.95f;
            previewInstance.tint[2] = 1.0f;
        }
        previewInstance.tint[3] = std::clamp(preview.pipeStyleId, 0.0f, 2.0f);
        previewInstance.extensions[0] = 0.0f;
        previewInstance.extensions[1] = 0.0f;
        previewInstance.extensions[2] = 1.0f;
        previewInstance.extensions[3] = 1.0f;
        if (preview.pipeStyleId > 0.5f && preview.pipeStyleId < 1.5f) {
            previewInstance.extensions[2] = 2.0f;
            previewInstance.extensions[3] = 0.25f;
        }
        if (preview.pipeStyleId > 1.5f) {
            previewInstance.extensions[2] = 2.0f;
            previewInstance.extensions[3] = 0.25f;
        }

        const std::optional<FrameArenaSlice> previewInstanceSlice =
            m_frameArena.allocateUpload(
                sizeof(PipeInstance),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::PreviewData
            );
        if (previewInstanceSlice.has_value() && previewInstanceSlice->mapped != nullptr) {
            std::memcpy(previewInstanceSlice->mapped, &previewInstance, sizeof(PipeInstance));
            const bool previewUsesPipeMesh = preview.pipeStyleId < 0.5f;
            const BufferHandle previewVertexHandle =
                previewUsesPipeMesh ? m_pipeVertexBufferHandle : m_transportVertexBufferHandle;
            const BufferHandle previewIndexHandle =
                previewUsesPipeMesh ? m_pipeIndexBufferHandle : m_transportIndexBufferHandle;
            const uint32_t previewIndexCount = previewUsesPipeMesh ? m_pipeIndexCount : m_transportIndexCount;
            if (previewIndexCount == 0) {
                // No mesh data allocated for this preview style.
            }
            const VkBuffer pipeVertexBuffer = m_bufferAllocator.getBuffer(previewVertexHandle);
            const VkBuffer pipeIndexBuffer = m_bufferAllocator.getBuffer(previewIndexHandle);
            const VkBuffer pipeInstanceBuffer = m_bufferAllocator.getBuffer(previewInstanceSlice->buffer);
            if (pipeVertexBuffer != VK_NULL_HANDLE &&
                pipeIndexBuffer != VK_NULL_HANDLE &&
                pipeInstanceBuffer != VK_NULL_HANDLE &&
                previewIndexCount > 0) {
                const VkBuffer vertexBuffers[2] = {pipeVertexBuffer, pipeInstanceBuffer};
                const VkDeviceSize vertexOffsets[2] = {0, previewInstanceSlice->offset};
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipePipeline);
                vkCmdBindDescriptorSets(
                    commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    boundDescriptorSetCount,
                    boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
                vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, pipeIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(commandBuffer, previewIndexCount, 1, 0, 0, 0);
            }
        }
    }

    if (drawCubePreview || drawFacePreview) {
        constexpr uint32_t kPreviewCubeIndexCount = 36u;
        constexpr uint32_t kPreviewFaceIndexCount = 6u;
        constexpr uint32_t kAddCubeFirstIndex = 0u;
        constexpr uint32_t kRemoveCubeFirstIndex = 36u;
        constexpr uint32_t kFaceFirstIndexBase = kRemoveCubeFirstIndex;
        constexpr float kChunkCoordinateScale = 1.0f;

        const VkBuffer previewVertexBuffer = m_bufferAllocator.getBuffer(m_previewVertexBufferHandle);
        const VkBuffer previewIndexBuffer = m_bufferAllocator.getBuffer(m_previewIndexBufferHandle);
        if (previewVertexBuffer != VK_NULL_HANDLE &&
            previewIndexBuffer != VK_NULL_HANDLE &&
            chunkInstanceSliceOpt.has_value() &&
            chunkInstanceBuffer != VK_NULL_HANDLE) {
            const VkBuffer previewVertexBuffers[2] = {previewVertexBuffer, chunkInstanceBuffer};
            const VkDeviceSize previewVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, previewVertexBuffers, previewVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, previewIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

            auto drawPreviewRange = [&](VkPipeline pipeline, uint32_t indexCount, uint32_t firstIndex, int x, int y, int z) {
                if (pipeline == VK_NULL_HANDLE || indexCount == 0) {
                    return;
                }
                ChunkPushConstants previewChunkPushConstants{};
                previewChunkPushConstants.chunkOffset[0] = static_cast<float>(x) * kChunkCoordinateScale;
                previewChunkPushConstants.chunkOffset[1] = static_cast<float>(y) * kChunkCoordinateScale;
                previewChunkPushConstants.chunkOffset[2] = static_cast<float>(z) * kChunkCoordinateScale;
                previewChunkPushConstants.chunkOffset[3] = 0.0f;
                previewChunkPushConstants.cascadeData[0] = 0.0f;
                previewChunkPushConstants.cascadeData[1] = 0.0f;
                previewChunkPushConstants.cascadeData[2] = 0.0f;
                previewChunkPushConstants.cascadeData[3] = 0.0f;

                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                vkCmdBindDescriptorSets(
                    commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    boundDescriptorSetCount,
                    boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &previewChunkPushConstants
                );
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(commandBuffer, indexCount, 1, firstIndex, 0, 0);
            };

            if (drawCubePreview) {
                const uint32_t cubeFirstIndex =
                    (preview.mode == VoxelPreview::Mode::Add) ? kAddCubeFirstIndex : kRemoveCubeFirstIndex;
                const int brushSize = std::max(preview.brushSize, 1);
                for (int localY = 0; localY < brushSize; ++localY) {
                    for (int localZ = 0; localZ < brushSize; ++localZ) {
                        for (int localX = 0; localX < brushSize; ++localX) {
                            drawPreviewRange(
                                activePreviewPipeline,
                                kPreviewCubeIndexCount,
                                cubeFirstIndex,
                                preview.x + localX,
                                preview.y + localY,
                                preview.z + localZ
                            );
                        }
                    }
                }
            }

            if (drawFacePreview) {
                const uint32_t faceFirstIndex = kFaceFirstIndexBase + (std::min(preview.faceId, 5u) * kPreviewFaceIndexCount);
                drawPreviewRange(m_previewRemovePipeline, kPreviewFaceIndexCount, faceFirstIndex, preview.faceX, preview.faceY, preview.faceZ);
            }
        }
    }

    // Draw skybox last with depth-test so sun/sky only appears where no geometry wrote depth.
    if (m_skyboxPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyboxPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        countDrawCalls(m_debugDrawCallsMain, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }

    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryMainEnd);

    if (m_hdrResolveMipLevels > 1u) {
        transitionImageLayout(
            commandBuffer,
            m_hdrResolveImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0u,
            1u,
            0u,
            1u
        );

        const uint32_t bloomMipCount = std::max(1u, m_hdrResolveMipLevels);
        const bool hdrResolveInitialized = m_hdrResolveImageInitialized[aoFrameIndex];
        for (uint32_t mipLevel = 1u; mipLevel < bloomMipCount; ++mipLevel) {
            transitionImageLayout(
                commandBuffer,
                m_hdrResolveImages[aoFrameIndex],
                hdrResolveInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                hdrResolveInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
                hdrResolveInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                0u,
                1u,
                mipLevel,
                1u
            );

            const uint32_t srcWidth = std::max(1u, m_swapchainExtent.width >> (mipLevel - 1u));
            const uint32_t srcHeight = std::max(1u, m_swapchainExtent.height >> (mipLevel - 1u));
            const uint32_t dstWidth = std::max(1u, m_swapchainExtent.width >> mipLevel);
            const uint32_t dstHeight = std::max(1u, m_swapchainExtent.height >> mipLevel);

            VkImageBlit mipBlit{};
            mipBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            mipBlit.srcSubresource.mipLevel = mipLevel - 1u;
            mipBlit.srcSubresource.baseArrayLayer = 0;
            mipBlit.srcSubresource.layerCount = 1;
            mipBlit.srcOffsets[0] = {0, 0, 0};
            mipBlit.srcOffsets[1] = {
                static_cast<int32_t>(srcWidth),
                static_cast<int32_t>(srcHeight),
                1
            };
            mipBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            mipBlit.dstSubresource.mipLevel = mipLevel;
            mipBlit.dstSubresource.baseArrayLayer = 0;
            mipBlit.dstSubresource.layerCount = 1;
            mipBlit.dstOffsets[0] = {0, 0, 0};
            mipBlit.dstOffsets[1] = {
                static_cast<int32_t>(dstWidth),
                static_cast<int32_t>(dstHeight),
                1
            };
            vkCmdBlitImage(
                commandBuffer,
                m_hdrResolveImages[aoFrameIndex],
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_hdrResolveImages[aoFrameIndex],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &mipBlit,
                VK_FILTER_LINEAR
            );

            const bool hasNextMip = (mipLevel + 1u) < bloomMipCount;
            transitionImageLayout(
                commandBuffer,
                m_hdrResolveImages[aoFrameIndex],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                hasNextMip ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                hasNextMip ? VK_PIPELINE_STAGE_2_TRANSFER_BIT : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                hasNextMip ? VK_ACCESS_2_TRANSFER_READ_BIT : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                0u,
                1u,
                mipLevel,
                1u
            );
        }

        transitionImageLayout(
            commandBuffer,
            m_hdrResolveImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0u,
            1u,
            0u,
            bloomMipCount - 1u
        );
    } else {
        transitionImageLayout(
            commandBuffer,
            m_hdrResolveImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0u,
            1u,
            0u,
            1u
        );
    }

    const bool autoExposurePassResourcesReady =
        m_autoExposureComputeAvailable &&
        m_autoExposurePipelineLayout != VK_NULL_HANDLE &&
        m_autoExposureHistogramPipeline != VK_NULL_HANDLE &&
        m_autoExposureUpdatePipeline != VK_NULL_HANDLE &&
        m_autoExposureDescriptorSets[m_currentFrame] != VK_NULL_HANDLE &&
        autoExposureHistogramBuffer != VK_NULL_HANDLE &&
        autoExposureStateBuffer != VK_NULL_HANDLE;
    if (autoExposureEnabled && autoExposurePassResourcesReady) {
        wroteAutoExposureTimestamps = true;
        writeGpuTimestampTop(kGpuTimestampQueryAutoExposureStart);
        beginDebugLabel(commandBuffer, "Pass: Auto Exposure", 0.30f, 0.30f, 0.20f, 1.0f);
        const VkPipelineStageFlags2 exposureSrcStage =
            m_autoExposureHistoryValid ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE;
        const VkAccessFlags2 exposureSrcAccess =
            m_autoExposureHistoryValid ? VK_ACCESS_2_SHADER_STORAGE_READ_BIT : VK_ACCESS_2_NONE;
        transitionBufferAccess(
            commandBuffer,
            autoExposureStateBuffer,
            0,
            sizeof(float) * 4u,
            exposureSrcStage,
            exposureSrcAccess,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );
        vkCmdFillBuffer(
            commandBuffer,
            autoExposureHistogramBuffer,
            0,
            static_cast<VkDeviceSize>(kAutoExposureHistogramBins * sizeof(uint32_t)),
            0u
        );
        transitionBufferAccess(
            commandBuffer,
            autoExposureHistogramBuffer,
            0,
            static_cast<VkDeviceSize>(kAutoExposureHistogramBins * sizeof(uint32_t)),
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );

        // Use a smaller source mip for histogram construction to keep auto-exposure cheaper than heavy fullscreen passes.
        constexpr uint32_t kAutoExposureTargetDownsampleMip = 3u;
        const uint32_t availableHdrMipLevels = std::max(1u, m_hdrResolveMipLevels);
        const uint32_t histogramSourceMip = std::min(
            kAutoExposureTargetDownsampleMip,
            availableHdrMipLevels - 1u
        );
        const uint32_t hdrWidth = std::max(1u, m_swapchainExtent.width >> histogramSourceMip);
        const uint32_t hdrHeight = std::max(1u, m_swapchainExtent.height >> histogramSourceMip);
        AutoExposureHistogramPushConstants histogramPushConstants{};
        histogramPushConstants.width = hdrWidth;
        histogramPushConstants.height = hdrHeight;
        histogramPushConstants.totalPixels = hdrWidth * hdrHeight;
        histogramPushConstants.binCount = kAutoExposureHistogramBins;
        histogramPushConstants.minLogLuminance = -10.0f;
        histogramPushConstants.maxLogLuminance = 4.0f;
        histogramPushConstants.sourceMipLevel = static_cast<float>(histogramSourceMip);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_autoExposureHistogramPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            m_autoExposurePipelineLayout,
            0,
            1,
            &m_autoExposureDescriptorSets[m_currentFrame],
            0,
            nullptr
        );
        vkCmdPushConstants(
            commandBuffer,
            m_autoExposurePipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(AutoExposureHistogramPushConstants),
            &histogramPushConstants
        );
        const uint32_t histogramDispatchX = (hdrWidth + (kAutoExposureWorkgroupSize - 1u)) / kAutoExposureWorkgroupSize;
        const uint32_t histogramDispatchY = (hdrHeight + (kAutoExposureWorkgroupSize - 1u)) / kAutoExposureWorkgroupSize;
        vkCmdDispatch(commandBuffer, histogramDispatchX, histogramDispatchY, 1u);

        transitionBufferAccess(
            commandBuffer,
            autoExposureHistogramBuffer,
            0,
            static_cast<VkDeviceSize>(kAutoExposureHistogramBins * sizeof(uint32_t)),
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        AutoExposureUpdatePushConstants updatePushConstants{};
        updatePushConstants.totalPixels = histogramPushConstants.totalPixels;
        updatePushConstants.binCount = kAutoExposureHistogramBins;
        updatePushConstants.resetHistory = m_autoExposureHistoryValid ? 0u : 1u;
        updatePushConstants.minLogLuminance = histogramPushConstants.minLogLuminance;
        updatePushConstants.maxLogLuminance = histogramPushConstants.maxLogLuminance;
        const float clampedLowPercentile = std::clamp(m_skyDebugSettings.autoExposureLowPercentile, 0.0f, 0.98f);
        const float clampedHighPercentile = std::clamp(
            m_skyDebugSettings.autoExposureHighPercentile,
            clampedLowPercentile + 0.01f,
            1.0f
        );
        updatePushConstants.lowPercentile = clampedLowPercentile;
        updatePushConstants.highPercentile = clampedHighPercentile;
        updatePushConstants.keyValue = std::clamp(m_skyDebugSettings.autoExposureKeyValue, 0.01f, 1.0f);
        const float minExposure = std::clamp(m_skyDebugSettings.autoExposureMin, 0.05f, 32.0f);
        const float maxExposure = std::clamp(m_skyDebugSettings.autoExposureMax, minExposure, 32.0f);
        updatePushConstants.minExposure = minExposure;
        updatePushConstants.maxExposure = maxExposure;
        updatePushConstants.adaptUpRate = std::clamp(m_skyDebugSettings.autoExposureAdaptUp, 0.05f, 20.0f);
        updatePushConstants.adaptDownRate = std::clamp(m_skyDebugSettings.autoExposureAdaptDown, 0.05f, 20.0f);
        updatePushConstants.deltaTimeSeconds = std::clamp(frameDeltaSeconds, 0.0f, 0.25f);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_autoExposureUpdatePipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            m_autoExposurePipelineLayout,
            0,
            1,
            &m_autoExposureDescriptorSets[m_currentFrame],
            0,
            nullptr
        );
        vkCmdPushConstants(
            commandBuffer,
            m_autoExposurePipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(AutoExposureUpdatePushConstants),
            &updatePushConstants
        );
        vkCmdDispatch(commandBuffer, 1u, 1u, 1u);

        transitionBufferAccess(
            commandBuffer,
            autoExposureStateBuffer,
            0,
            sizeof(float) * 4u,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        m_autoExposureHistoryValid = true;
        endDebugLabel(commandBuffer);
        writeGpuTimestampBottom(kGpuTimestampQueryAutoExposureEnd);
    } else {
        if (!autoExposureEnabled || !autoExposurePassResourcesReady) {
            m_autoExposureHistoryValid = false;
        }
    }
    if (!wroteAutoExposureTimestamps) {
        writeGpuTimestampTop(kGpuTimestampQueryAutoExposureStart);
        writeGpuTimestampBottom(kGpuTimestampQueryAutoExposureEnd);
    }

    if (aoFrameIndex < m_sunShaftImages.size() &&
        m_sunShaftImages[aoFrameIndex] != VK_NULL_HANDLE &&
        m_sunShaftImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        wroteSunShaftTimestamps = true;
        writeGpuTimestampTop(kGpuTimestampQuerySunShaftStart);
        const bool sunShaftInitialized = m_sunShaftImageInitialized[aoFrameIndex];
        if (m_sunShaftComputeAvailable &&
            m_sunShaftPipelineLayout != VK_NULL_HANDLE &&
            m_sunShaftPipeline != VK_NULL_HANDLE &&
            m_sunShaftDescriptorSets[m_currentFrame] != VK_NULL_HANDLE) {
            beginDebugLabel(commandBuffer, "Pass: Sun Shafts", 0.26f, 0.24f, 0.16f, 1.0f);
            transitionImageLayout(
                commandBuffer,
                m_normalDepthImages[aoFrameIndex],
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
            transitionImageLayout(
                commandBuffer,
                m_sunShaftImages[aoFrameIndex],
                sunShaftInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                sunShaftInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
                sunShaftInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );

            SunShaftPushConstants sunShaftPushConstants{};
            sunShaftPushConstants.width = std::max(1u, m_aoExtent.width);
            sunShaftPushConstants.height = std::max(1u, m_aoExtent.height);
            sunShaftPushConstants.sampleCount = 20u;

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_sunShaftPipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                m_sunShaftPipelineLayout,
                0,
                1,
                &m_sunShaftDescriptorSets[m_currentFrame],
                1,
                &mvpDynamicOffset
            );
            vkCmdPushConstants(
                commandBuffer,
                m_sunShaftPipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                sizeof(SunShaftPushConstants),
                &sunShaftPushConstants
            );
            const uint32_t dispatchX =
                (sunShaftPushConstants.width + (kSunShaftWorkgroupSize - 1u)) / kSunShaftWorkgroupSize;
            const uint32_t dispatchY =
                (sunShaftPushConstants.height + (kSunShaftWorkgroupSize - 1u)) / kSunShaftWorkgroupSize;
            vkCmdDispatch(commandBuffer, dispatchX, dispatchY, 1u);

            transitionImageLayout(
                commandBuffer,
                m_sunShaftImages[aoFrameIndex],
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
            m_sunShaftImageInitialized[aoFrameIndex] = true;
            endDebugLabel(commandBuffer);
        } else {
            transitionImageLayout(
                commandBuffer,
                m_sunShaftImages[aoFrameIndex],
                sunShaftInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                sunShaftInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
                sunShaftInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
            const VkClearColorValue clearValue = {{0.0f, 0.0f, 0.0f, 1.0f}};
            VkImageSubresourceRange clearRange{};
            clearRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            clearRange.baseMipLevel = 0u;
            clearRange.levelCount = 1u;
            clearRange.baseArrayLayer = 0u;
            clearRange.layerCount = 1u;
            vkCmdClearColorImage(
                commandBuffer,
                m_sunShaftImages[aoFrameIndex],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                &clearValue,
                1,
                &clearRange
            );
            transitionImageLayout(
                commandBuffer,
                m_sunShaftImages[aoFrameIndex],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
            m_sunShaftImageInitialized[aoFrameIndex] = true;
        }
        writeGpuTimestampBottom(kGpuTimestampQuerySunShaftEnd);
    }
    if (!wroteSunShaftTimestamps) {
        writeGpuTimestampTop(kGpuTimestampQuerySunShaftStart);
        writeGpuTimestampBottom(kGpuTimestampQuerySunShaftEnd);
    }

    transitionImageLayout(
        commandBuffer,
        m_swapchainImages[imageIndex],
        m_swapchainImageInitialized[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkRenderingAttachmentInfo toneMapColorAttachment{};
    toneMapColorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    toneMapColorAttachment.imageView = m_swapchainImageViews[imageIndex];
    toneMapColorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toneMapColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    toneMapColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo toneMapRenderingInfo{};
    toneMapRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    toneMapRenderingInfo.renderArea.offset = {0, 0};
    toneMapRenderingInfo.renderArea.extent = m_swapchainExtent;
    toneMapRenderingInfo.layerCount = 1;
    toneMapRenderingInfo.colorAttachmentCount = 1;
    toneMapRenderingInfo.pColorAttachments = &toneMapColorAttachment;

    writeGpuTimestampTop(kGpuTimestampQueryPostStart);
    beginDebugLabel(commandBuffer, "Pass: Tonemap + UI", 0.24f, 0.24f, 0.24f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &toneMapRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    if (m_tonemapPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.data(),
            1,
            &mvpDynamicOffset
        );
        countDrawCalls(m_debugDrawCallsPost, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    if (m_imguiInitialized) {
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    }

    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryPostEnd);

    transitionImageLayout(
        commandBuffer,
        m_swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    writeGpuTimestampBottom(kGpuTimestampQueryFrameEnd);

    endDebugLabel(commandBuffer);
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        VOX_LOGE("render") << "vkEndCommandBuffer failed\n";
        return;
    }

    std::array<VkSemaphore, 2> waitSemaphores{};
    std::array<VkPipelineStageFlags, 2> waitStages{};
    std::array<uint64_t, 2> waitSemaphoreValues{};
    uint32_t waitSemaphoreCount = 0;

    waitSemaphores[waitSemaphoreCount] = frame.imageAvailable;
    waitStages[waitSemaphoreCount] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    waitSemaphoreValues[waitSemaphoreCount] = 0;
    ++waitSemaphoreCount;

    if (m_pendingTransferTimelineValue > 0) {
        waitSemaphores[waitSemaphoreCount] = m_renderTimelineSemaphore;
        waitStages[waitSemaphoreCount] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        waitSemaphoreValues[waitSemaphoreCount] = m_pendingTransferTimelineValue;
        ++waitSemaphoreCount;
    }

    const uint64_t signalTimelineValue = m_nextTimelineValue++;
    std::array<VkSemaphore, 2> signalSemaphores = {
        renderFinishedSemaphore,
        m_renderTimelineSemaphore
    };
    std::array<uint64_t, 2> signalSemaphoreValues = {
        0,
        signalTimelineValue
    };
    VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
    timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineSubmitInfo.waitSemaphoreValueCount = waitSemaphoreCount;
    timelineSubmitInfo.pWaitSemaphoreValues = waitSemaphoreValues.data();
    timelineSubmitInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalSemaphoreValues.size());
    timelineSubmitInfo.pSignalSemaphoreValues = signalSemaphoreValues.data();

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineSubmitInfo;
    submitInfo.waitSemaphoreCount = waitSemaphoreCount;
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        VOX_LOGE("render") << "vkQueueSubmit failed\n";
        return;
    }
    m_frameTimelineValues[m_currentFrame] = signalTimelineValue;
    m_swapchainImageTimelineValues[imageIndex] = signalTimelineValue;
    m_lastGraphicsTimelineValue = signalTimelineValue;

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchain;
    presentInfo.pImageIndices = &imageIndex;
    VkPresentTimesInfoGOOGLE presentTimesInfo{};
    VkPresentTimeGOOGLE presentTime{};
    const bool useDisplayTiming =
        m_supportsDisplayTiming &&
        m_enableDisplayTiming &&
        m_getPastPresentationTimingGoogle != nullptr;
    if (useDisplayTiming) {
        const uint32_t submittedPresentId = m_nextDisplayTimingPresentId++;
        presentTime.presentID = submittedPresentId;
        presentTime.desiredPresentTime = 0;
        presentTimesInfo.sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE;
        presentTimesInfo.swapchainCount = 1;
        presentTimesInfo.pTimes = &presentTime;
        presentInfo.pNext = &presentTimesInfo;
        m_lastSubmittedDisplayTimingPresentId = submittedPresentId;
    } else {
        m_lastSubmittedDisplayTimingPresentId = 0;
    }

    const auto presentStartTime = std::chrono::steady_clock::now();
    const VkResult presentResult = vkQueuePresentKHR(m_graphicsQueue, &presentInfo);
    cpuWaitMs += static_cast<float>(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - presentStartTime).count()
    );
    if (useDisplayTiming && (presentResult == VK_SUCCESS || presentResult == VK_SUBOPTIMAL_KHR)) {
        updateDisplayTimingStats();
    }
    m_shadowDepthInitialized = true;
    m_swapchainImageInitialized[imageIndex] = true;
    m_msaaColorImageInitialized[imageIndex] = true;
    m_hdrResolveImageInitialized[aoFrameIndex] = true;

    if (
        acquireResult == VK_SUBOPTIMAL_KHR ||
        presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR
    ) {
        VOX_LOGI("render") << "swapchain needs recreate after present\n";
        recreateSwapchain();
    } else if (presentResult != VK_SUCCESS) {
        logVkFailure("vkQueuePresentKHR", presentResult);
    }

    const float cpuTotalMs = static_cast<float>(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuFrameStartTime).count()
    );
    m_debugFrameTimeMs = cpuTotalMs;
    m_debugCpuFrameWorkMs = std::max(0.0f, cpuTotalMs - cpuWaitMs);
    if (!m_debugCpuFrameEwmaInitialized) {
        m_debugCpuFrameEwmaMs = m_debugFrameTimeMs;
        m_debugCpuFrameEwmaInitialized = true;
    } else {
        m_debugCpuFrameEwmaMs += kCpuFrameEwmaAlpha * (m_debugFrameTimeMs - m_debugCpuFrameEwmaMs);
    }
    m_debugCpuFrameTotalMsHistory[m_debugCpuFrameTimingMsHistoryWrite] = m_debugFrameTimeMs;
    m_debugCpuFrameWorkMsHistory[m_debugCpuFrameTimingMsHistoryWrite] = m_debugCpuFrameWorkMs;
    m_debugCpuFrameEwmaMsHistory[m_debugCpuFrameTimingMsHistoryWrite] = m_debugCpuFrameEwmaMs;
    m_debugCpuFrameTimingMsHistoryWrite =
        (m_debugCpuFrameTimingMsHistoryWrite + 1u) % kTimingHistorySampleCount;
    m_debugCpuFrameTimingMsHistoryCount =
        std::min(m_debugCpuFrameTimingMsHistoryCount + 1u, kTimingHistorySampleCount);

    const FrameArenaStats& frameArenaStats = m_frameArena.activeStats();
    m_debugFrameArenaUploadBytes = static_cast<std::uint64_t>(frameArenaStats.uploadBytesAllocated);
    m_debugFrameArenaUploadAllocs = frameArenaStats.uploadAllocationCount;
    m_debugFrameArenaTransientBufferBytes = static_cast<std::uint64_t>(frameArenaStats.transientBufferBytes);
    m_debugFrameArenaTransientBufferCount = frameArenaStats.transientBufferCount;
    m_debugFrameArenaTransientImageBytes = frameArenaStats.transientImageBytes;
    m_debugFrameArenaTransientImageCount = frameArenaStats.transientImageCount;
    m_debugFrameArenaAliasReuses = frameArenaStats.transientImageAliasReuses;
    const FrameArenaResidentStats& frameArenaResidentStats = m_frameArena.residentStats();
    m_debugFrameArenaResidentBufferBytes = frameArenaResidentStats.bufferBytes;
    m_debugFrameArenaResidentBufferCount = frameArenaResidentStats.bufferCount;
    m_debugFrameArenaResidentImageBytes = frameArenaResidentStats.imageBytes;
    m_debugFrameArenaResidentImageCount = frameArenaResidentStats.imageCount;
    m_debugFrameArenaResidentAliasReuses = frameArenaResidentStats.imageAliasReuses;
    m_frameArena.collectAliasedImageDebugInfo(m_debugAliasedImages);

    m_currentFrame = (m_currentFrame + 1) % kMaxFramesInFlight;
}


} // namespace voxelsprout::render
