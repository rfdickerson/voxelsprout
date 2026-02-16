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

#include "render/backend/vulkan/frame_graph_core.h"
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
    const std::optional<CoreFrameGraphPlan> coreFrameGraphPlan = buildCoreFrameGraphPlan(&m_frameGraph);
    if (!coreFrameGraphPlan.has_value()) {
        VOX_LOGE("render") << "frame graph has a cycle; refusing to render frame";
        return;
    }
    std::optional<std::uint32_t> lastCorePassOrderIndex = std::nullopt;
    const auto markCorePassEntered = [&](FrameGraph::PassId passId, const char* passName) {
        if (passId >= coreFrameGraphPlan->passOrderById.size()) {
            return;
        }
        const std::uint32_t passOrderIndex = coreFrameGraphPlan->passOrderById[passId];
        if (lastCorePassOrderIndex.has_value() && passOrderIndex < *lastCorePassOrderIndex) {
            VOX_LOGE("render")
                << "core frame pass executed out of graph order: " << passName
                << ", orderIndex=" << passOrderIndex
                << ", previousOrderIndex=" << *lastCorePassOrderIndex;
        }
        lastCorePassOrderIndex = passOrderIndex;
    };

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
    const CameraFrameDerived cameraFrame = computeCameraFrame(camera);
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

    voxelsprout::math::Vector3 sunDirection = voxelsprout::math::normalize(computeSunDirection(
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
    const float voxelGiOriginX = computeVoxelGiAxisOrigin(camera.x, voxelGiHalfSpan, kVoxelGiCellSize);
    const float voxelGiDesiredOriginY = computeVoxelGiAxisOrigin(camera.y, voxelGiHalfSpan, kVoxelGiCellSize);
    const float kVoxelGiVerticalFollowThreshold = kVoxelGiCellSize * 4.0f;
    const float voxelGiOriginY = computeVoxelGiStableOriginY(
        voxelGiDesiredOriginY,
        m_voxelGiPreviousGridOrigin[1],
        m_voxelGiHasPreviousFrameState,
        kVoxelGiVerticalFollowThreshold
    );
    const float voxelGiOriginZ = computeVoxelGiAxisOrigin(camera.z, voxelGiHalfSpan, kVoxelGiCellSize);
    constexpr float kVoxelGiGridMoveThreshold = 0.001f;
    constexpr float kVoxelGiLightingChangeThreshold = 0.001f;
    constexpr float kVoxelGiTuningChangeThreshold = 0.001f;
    const VoxelGiComputeFlags voxelGiFlags = computeVoxelGiFlags(
        shIrradiance,
        m_voxelGiPreviousShIrradiance,
        {voxelGiOriginX, voxelGiOriginY, voxelGiOriginZ},
        m_voxelGiPreviousGridOrigin,
        m_voxelGiHasPreviousFrameState,
        m_voxelGiWorldDirty,
        m_voxelGiOccupancyInitialized,
        sunDirection,
        {m_voxelGiPreviousSunDirection[0], m_voxelGiPreviousSunDirection[1], m_voxelGiPreviousSunDirection[2]},
        sunColor,
        {m_voxelGiPreviousSunColor[0], m_voxelGiPreviousSunColor[1], m_voxelGiPreviousSunColor[2]},
        m_voxelGiDebugSettings.bounceStrength,
        m_voxelGiPreviousBounceStrength,
        m_voxelGiDebugSettings.diffusionSoftness,
        m_voxelGiPreviousDiffusionSoftness,
        kVoxelGiGridMoveThreshold,
        kVoxelGiLightingChangeThreshold,
        kVoxelGiTuningChangeThreshold
    );
    const bool voxelGiNeedsOccupancyUpload = voxelGiFlags.needsOccupancyUpload;
    const bool voxelGiNeedsComputeUpdate =
        voxelGiFlags.needsComputeUpdate || !m_voxelGiInitialized;
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

    const BoundDescriptorSets boundDescriptorSets = updateFrameDescriptorSets(
        aoFrameIndex,
        bufferInfo,
        autoExposureHistogramBuffer,
        autoExposureStateBuffer
    );
    const uint32_t boundDescriptorSetCount = boundDescriptorSets.count;

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

    const FrameInstanceDrawData frameInstanceDrawData = prepareFrameInstanceDrawData(simulation, simulationAlpha);
    const uint32_t pipeInstanceCount = frameInstanceDrawData.pipeInstanceCount;
    const auto& pipeInstanceSliceOpt = frameInstanceDrawData.pipeInstanceSliceOpt;
    const uint32_t transportInstanceCount = frameInstanceDrawData.transportInstanceCount;
    const auto& transportInstanceSliceOpt = frameInstanceDrawData.transportInstanceSliceOpt;
    const uint32_t beltCargoInstanceCount = frameInstanceDrawData.beltCargoInstanceCount;
    const auto& beltCargoInstanceSliceOpt = frameInstanceDrawData.beltCargoInstanceSliceOpt;
    const std::vector<ReadyMagicaDraw>& readyMagicaDraws = frameInstanceDrawData.readyMagicaDraws;

    const FrameChunkDrawData frameChunkDrawData = prepareFrameChunkDrawData(
        chunkGrid.chunks(),
        visibleChunkIndices,
        lightViewProjMatrices,
        cameraChunkX,
        cameraChunkY,
        cameraChunkZ
    );
    const auto& chunkInstanceSliceOpt = frameChunkDrawData.chunkInstanceSliceOpt;
    const auto& shadowChunkInstanceSliceOpt = frameChunkDrawData.shadowChunkInstanceSliceOpt;
    const VkBuffer chunkInstanceBuffer = frameChunkDrawData.chunkInstanceBuffer;
    const VkBuffer shadowChunkInstanceBuffer = frameChunkDrawData.shadowChunkInstanceBuffer;
    const VkBuffer chunkVertexBuffer = m_bufferAllocator.getBuffer(m_chunkVertexBufferHandle);
    const VkBuffer chunkIndexBuffer = m_bufferAllocator.getBuffer(m_chunkIndexBufferHandle);
    const bool canDrawMagica = !readyMagicaDraws.empty() && m_magicaPipeline != VK_NULL_HANDLE;
    auto countDrawCalls = [&](std::uint32_t& passCounter, std::uint32_t drawCount) {
        passCounter += drawCount;
        m_debugDrawCallsTotal += drawCount;
    };

    writeGpuTimestampTop(kGpuTimestampQueryShadowStart);
    markCorePassEntered(coreFrameGraphPlan->shadow, "shadow");
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
            boundDescriptorSets.sets.data(),
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

            if (cascadeIndex < kShadowCascadeCount) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
                vkCmdBindDescriptorSets(
                    commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    boundDescriptorSetCount,
                    boundDescriptorSets.sets.data(),
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
                drawIndirectShadowChunkRanges(commandBuffer, m_debugDrawCallsShadow, cascadeIndex, frameChunkDrawData);
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
                        boundDescriptorSets.sets.data(),
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
                        boundDescriptorSets.sets.data(),
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
    markCorePassEntered(coreFrameGraphPlan->prepass, "prepass");
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
            boundDescriptorSets.sets.data(),
            1,
            &mvpDynamicOffset
        );
        if (frameChunkDrawData.canDrawChunksIndirect) {
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
            drawIndirectChunkRanges(commandBuffer, m_debugDrawCallsPrepass, frameChunkDrawData);
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
                boundDescriptorSets.sets.data(),
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
                boundDescriptorSets.sets.data(),
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
            boundDescriptorSets.sets.data(),
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
            boundDescriptorSets.sets.data(),
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
    markCorePassEntered(coreFrameGraphPlan->main, "main");
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
        boundDescriptorSets.sets.data(),
            1,
            &mvpDynamicOffset
        );
    if (frameChunkDrawData.canDrawChunksIndirect) {
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
        drawIndirectChunkRanges(commandBuffer, m_debugDrawCallsMain, frameChunkDrawData);
    }
    if (canDrawMagica) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_magicaPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.sets.data(),
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
                boundDescriptorSets.sets.data(),
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
                boundDescriptorSets.sets.data(),
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
                    boundDescriptorSets.sets.data(),
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
                    boundDescriptorSets.sets.data(),
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
            boundDescriptorSets.sets.data(),
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
    markCorePassEntered(coreFrameGraphPlan->post, "post");
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
            boundDescriptorSets.sets.data(),
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
