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
#include <utility>
#include <vector>

#include "render/backend/vulkan/frame_graph_core.h"
#include "render/backend/vulkan/frame_graph_runtime.h"
#include "render/backend/vulkan/frame_math.h"

namespace odai::render {

#include "render/renderer_shared.h"

namespace {

const char* voxelGiSurfaceModeName(VoxelGiSurfaceMode mode) {
    switch (mode) {
    case VoxelGiSurfaceMode::Legacy: return "legacy";
    case VoxelGiSurfaceMode::RtSurface: return "rt_surface";
    case VoxelGiSurfaceMode::RestirSurface: return "restir_surface";
    }
    return "legacy";
}

const char* voxelGiSurfaceFallbackReasonName(
    VoxelGiSurfaceMode requestedMode,
    bool computeAvailable,
    bool rtReady,
    bool restirReady,
    bool rtTlasReady
) {
    if (requestedMode == VoxelGiSurfaceMode::Legacy) {
        return "none";
    }
    if (!computeAvailable) {
        return "compute_unavailable";
    }
    if (!rtTlasReady) {
        return "scene_unavailable";
    }
    if (requestedMode == VoxelGiSurfaceMode::RestirSurface && !restirReady) {
        return rtReady ? "restir_unavailable" : "rt_surface_unavailable";
    }
    if (!rtReady) {
        return "rt_surface_unavailable";
    }
    return "none";
}

} // namespace

void RendererBackend::renderFrame(
    const odai::world::ChunkGrid& chunkGrid,
    const odai::sim::Simulation& simulation,
    const CameraPose& camera,
    const VoxelPreview& preview,
    float simulationAlpha,
    std::span<const std::size_t> visibleChunkIndices,
    const ImportedActorFrameData* importedActors
) {
    const auto cpuFrameStartTime = std::chrono::steady_clock::now();
    float cpuWaitMs = 0.0f;
    float cpuWaitFrameSlotMs = 0.0f;
    float cpuWaitAcquireMs = 0.0f;
    float cpuWaitPresentMs = 0.0f;
    float cpuWaitTransferMs = 0.0f;

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
    m_framePacingStats = {};
    m_framePacingStats.displayTimingSupported = m_supportsDisplayTiming;
    m_framePacingStats.displayTimingEnabled = m_supportsDisplayTiming && m_enableDisplayTiming;
    m_framePacingStats.schedulingActive =
        m_framePacingStats.displayTimingEnabled && m_framePacingSettings.mode == FramePacingMode::Scheduled;
    m_framePacingStats.cadenceDivisor = std::max(1u, m_framePacingSettings.cadenceDivisor);
    m_framePacingStats.maxQueuedFrames = std::clamp(m_framePacingSettings.maxQueuedFrames, 1u, kMaxFramesInFlight);
    m_framePacingStats.refreshMs = m_debugDisplayRefreshMs;
    m_framePacingStats.presentMarginMs = m_debugDisplayPresentMarginMs;
    m_framePacingStats.actualPresentDeltaMs = m_debugDisplayActualEarliestDeltaMs;
    m_framePacingStats.presentScheduleErrorMs = m_debugDisplayScheduleErrorMs;
    m_framePacingStats.latePresentCount = m_debugLatePresentCount;
    m_framePacingStats.cpuWaitFrameSlotMs = 0.0f;
    m_framePacingStats.cpuWaitAcquireMs = 0.0f;
    m_framePacingStats.cpuWaitPresentMs = 0.0f;
    m_framePacingStats.cpuWaitTransferMs = 0.0f;
    m_framePacingStats.gpuTimestampsPending = false;
    m_framePacingStats.gpuTimestampSkippedFrames = 0;
    if (m_displayRefreshDurationNs > 0) {
        m_framePacingStats.targetPresentIntervalMs = static_cast<float>(
            (m_displayRefreshDurationNs * m_framePacingStats.cadenceDivisor) * 1.0e-6
        );
    }
    auto shortWait = [&](float& bucketMs) {
        const auto waitStartTime = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(250));
        std::this_thread::yield();
        const float waitMs = static_cast<float>(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waitStartTime).count()
        );
        bucketMs += waitMs;
        cpuWaitMs += waitMs;
    };
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
    for (const odai::world::Chunk& chunk : chunkGrid.chunks()) {
        for (int my = 0; my < odai::world::Chunk::kMacroSizeY; ++my) {
            for (int mz = 0; mz < odai::world::Chunk::kMacroSizeZ; ++mz) {
                for (int mx = 0; mx < odai::world::Chunk::kMacroSizeX; ++mx) {
                    const odai::world::Chunk::MacroCell cell = chunk.macroCellAt(mx, my, mz);
                    switch (cell.resolution) {
                    case odai::world::Chunk::CellResolution::Uniform:
                        ++m_debugMacroCellUniformCount;
                        break;
                    case odai::world::Chunk::CellResolution::Refined4:
                        ++m_debugMacroCellRefined4Count;
                        break;
                    case odai::world::Chunk::CellResolution::Refined1:
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
    CoreFrameGraphOrderValidator coreFramePassOrderValidator(*coreFrameGraphPlan);

    collectCompletedBufferReleases();
    const uint64_t completedTimelineValueBeforeFrame = completedTimelineValue();
    m_framePacingStats.queuedFrames = countQueuedFrames(completedTimelineValueBeforeFrame);
    if (shouldThrottleFrameStart(completedTimelineValueBeforeFrame, &cpuWaitFrameSlotMs)) {
        cpuWaitMs += cpuWaitFrameSlotMs;
        m_framePacingStats.cpuWaitFrameSlotMs = cpuWaitFrameSlotMs;
        return;
    }

    FrameResources& frame = m_frames[m_currentFrame];
    if (!isTimelineValueReached(m_frameTimelineValues[m_currentFrame])) {
        static double lastStallLogTimeSeconds = 0.0;
        const uint64_t completedValue = completedTimelineValue();
        if (completedValue > 0 || m_frameTimelineValues[m_currentFrame] > 0) {
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
        }
        shortWait(cpuWaitFrameSlotMs);
        m_framePacingStats.cpuWaitFrameSlotMs = cpuWaitFrameSlotMs;
        return;
    }
    if (m_frameTimelineValues[m_currentFrame] > 0) {
        if (!readGpuTimestampResults(m_currentFrame)) {
            m_framePacingStats.gpuTimestampsPending = true;
            ++m_framePacingStats.gpuTimestampSkippedFrames;
        }
    }
    if (m_transferCommandBufferInFlightValue > 0) {
        if (isTimelineValueReached(m_transferCommandBufferInFlightValue)) {
            m_transferCommandBufferInFlightValue = 0;
            m_pendingTransferTimelineValue = 0;
            collectCompletedBufferReleases();
        } else {
            shortWait(cpuWaitTransferMs);
            m_framePacingStats.cpuWaitTransferMs = cpuWaitTransferMs;
            return;
        }
    }
    m_frameArena.beginFrame(m_currentFrame);

    if (!m_pendingChunkRemeshKeys.empty()) {
        std::erase_if(m_pendingChunkRemeshKeys, [&](const ChunkResidentKey& pendingKey) {
            return std::find_if(
                       chunkGrid.chunks().begin(),
                       chunkGrid.chunks().end(),
                       [&](const odai::world::Chunk& chunk) {
                           return chunk.chunkX() == pendingKey.chunkX &&
                                  chunk.chunkY() == pendingKey.chunkY &&
                                  chunk.chunkZ() == pendingKey.chunkZ;
                       }) == chunkGrid.chunks().end();
        });
    }
    m_debugChunkPendingRemeshCount = static_cast<std::uint32_t>(m_pendingChunkRemeshKeys.size());
    m_debugChunkRemeshBatchCount = 0;
    if (m_chunkMeshRebuildRequested || !m_pendingChunkRemeshKeys.empty()) {
        // Avoid CPU stalls when async transfer is still in flight.
        if (m_transferCommandBufferInFlightValue == 0 ||
            isTimelineValueReached(m_transferCommandBufferInFlightValue)) {
            std::vector<ChunkResidentKey> remeshBatchKeys;
            std::vector<std::size_t> resolvedRemeshIndices;
            if (!m_chunkMeshRebuildRequested) {
                remeshBatchKeys = m_pendingChunkRemeshKeys;
                const int remeshCameraChunkX = static_cast<int>(std::floor(
                    camera.x / static_cast<float>(odai::world::Chunk::kSizeX)));
                const int remeshCameraChunkZ = static_cast<int>(std::floor(
                    camera.z / static_cast<float>(odai::world::Chunk::kSizeZ)));
                std::sort(
                    remeshBatchKeys.begin(),
                    remeshBatchKeys.end(),
                    [&](const ChunkResidentKey& a, const ChunkResidentKey& b) {
                        const auto chunkDistance = [&](const ChunkResidentKey& key) {
                            const int dx = std::abs(key.chunkX - remeshCameraChunkX);
                            const int dz = std::abs(key.chunkZ - remeshCameraChunkZ);
                            return std::max(dx, dz);
                        };
                        const int distanceA = chunkDistance(a);
                        const int distanceB = chunkDistance(b);
                        if (distanceA != distanceB) {
                            return distanceA < distanceB;
                        }
                        if (a.chunkX != b.chunkX) {
                            return a.chunkX < b.chunkX;
                        }
                        if (a.chunkY != b.chunkY) {
                            return a.chunkY < b.chunkY;
                        }
                        return a.chunkZ < b.chunkZ;
                    });
                if (remeshBatchKeys.size() > kChunkRemeshBudgetPerFrame) {
                    remeshBatchKeys.resize(kChunkRemeshBudgetPerFrame);
                }
                resolvedRemeshIndices.reserve(remeshBatchKeys.size());
                for (const ChunkResidentKey& key : remeshBatchKeys) {
                    const auto residentIt = std::find_if(
                        chunkGrid.chunks().begin(),
                        chunkGrid.chunks().end(),
                        [&](const odai::world::Chunk& chunk) {
                            return chunk.chunkX() == key.chunkX &&
                                   chunk.chunkY() == key.chunkY &&
                                   chunk.chunkZ() == key.chunkZ;
                        });
                    if (residentIt != chunkGrid.chunks().end()) {
                        resolvedRemeshIndices.push_back(
                            static_cast<std::size_t>(std::distance(chunkGrid.chunks().begin(), residentIt)));
                    }
                }
            }
            const std::span<const std::size_t> pendingRemeshIndices =
                m_chunkMeshRebuildRequested
                    ? std::span<const std::size_t>{}
                    : std::span<const std::size_t>(resolvedRemeshIndices.data(), resolvedRemeshIndices.size());
            m_debugChunkRemeshBatchCount = static_cast<std::uint32_t>(pendingRemeshIndices.size());
            if (!m_chunkMeshRebuildRequested && remeshBatchKeys.empty()) {
                m_pendingChunkRemeshKeys.clear();
                m_debugChunkPendingRemeshCount = 0;
            } else if (!m_chunkMeshRebuildRequested && pendingRemeshIndices.empty()) {
                for (const ChunkResidentKey& processedKey : remeshBatchKeys) {
                    const auto pendingIt =
                        std::find(m_pendingChunkRemeshKeys.begin(), m_pendingChunkRemeshKeys.end(), processedKey);
                    if (pendingIt != m_pendingChunkRemeshKeys.end()) {
                        m_pendingChunkRemeshKeys.erase(pendingIt);
                    }
                }
                m_debugChunkPendingRemeshCount = static_cast<std::uint32_t>(m_pendingChunkRemeshKeys.size());
            } else if (createChunkBuffers(chunkGrid, pendingRemeshIndices)) {
                if (m_chunkMeshRebuildRequested) {
                    m_chunkMeshRebuildRequested = false;
                    m_pendingChunkRemeshKeys.clear();
                } else {
                    for (const ChunkResidentKey& processedKey : remeshBatchKeys) {
                        const auto pendingIt =
                            std::find(m_pendingChunkRemeshKeys.begin(), m_pendingChunkRemeshKeys.end(), processedKey);
                        if (pendingIt != m_pendingChunkRemeshKeys.end()) {
                            m_pendingChunkRemeshKeys.erase(pendingIt);
                        }
                    }
                }
                m_debugChunkPendingRemeshCount = static_cast<std::uint32_t>(m_pendingChunkRemeshKeys.size());
            } else {
                VOX_LOGE("render") << "failed deferred chunk remesh";
            }
        }
    }
    if (m_rtSceneDirty &&
        !m_chunkMeshRebuildRequested &&
        m_pendingChunkRemeshKeys.empty() &&
        m_transferCommandBufferInFlightValue == 0) {
        if (rayTracingRuntimeReady() && !rebuildRayTracingScene()) {
            VOX_LOGE("render") << "deferred chunk RT scene rebuild failed";
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
    const float acquireWaitMs = static_cast<float>(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - acquireStartTime).count()
    );
    cpuWaitMs += acquireWaitMs;
    cpuWaitAcquireMs += acquireWaitMs;

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        VOX_LOGI("render") << "swapchain out of date during acquire, recreating\n";
        recreateSwapchain();
        return;
    }
    if (acquireResult == VK_TIMEOUT) {
        m_framePacingStats.cpuWaitAcquireMs = cpuWaitAcquireMs;
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
        buildDofDebugUi();
        m_debugUiVisible = m_showFrameStatsPanel;
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
    const bool renderingImportedActors =
        importedActors != nullptr &&
        !importedActors->vertices.empty() &&
        !importedActors->indices.empty() &&
        !importedActors->draws.empty();
    const bool renderingImportedScene = !m_importedMeshDraws.empty() || renderingImportedActors;
    const bool legacyVoxelRenderingEnabled = !renderingImportedScene;
    const bool importedInteriorGiEnabled =
        m_importedSceneInteriorMode &&
        !m_importedGiTriangles.empty();
    const bool voxelGiSceneEnabled = legacyVoxelRenderingEnabled || importedInteriorGiEnabled;
    const float farPlane = renderingImportedScene ? 50000.0f : 500.0f;
    const float halfFovRadians = odai::math::radians(activeFovDegrees) * 0.5f;
    const float tanHalfFov = std::tan(halfFovRadians);
    const odai::math::Vector3 eye{camera.x, camera.y, camera.z};
    const CameraFrameDerived cameraFrame = computeCameraFrame(camera);
    const int cameraChunkX = cameraFrame.chunkX;
    const int cameraChunkY = cameraFrame.chunkY;
    const int cameraChunkZ = cameraFrame.chunkZ;
    const odai::math::Vector3 forward = cameraFrame.forward;

    const odai::math::Matrix4 view = lookAt(eye, eye + forward, odai::math::Vector3{0.0f, 1.0f, 0.0f});
    const odai::math::Matrix4 projection = perspectiveVulkan(odai::math::radians(activeFovDegrees), aspectRatio, nearPlane, farPlane);
    const odai::math::Matrix4 mvp = projection * view;
    const odai::math::Matrix4 mvpColumnMajor = transpose(mvp);
    const odai::math::Matrix4 viewColumnMajor = transpose(view);
    const odai::math::Matrix4 projectionColumnMajor = transpose(projection);

    const bool projectionParamsChanged =
        std::abs(m_shadowStableAspectRatio - aspectRatio) > 0.0001f ||
        std::abs(m_shadowStableFovDegrees - activeFovDegrees) > 0.0001f;
    if (projectionParamsChanged) {
        m_shadowStableAspectRatio = aspectRatio;
        m_shadowStableFovDegrees = activeFovDegrees;
        m_shadowStableCascadeRadii.fill(0.0f);
    }

    odai::math::Vector3 sunDirection = odai::math::normalize(computeSunDirection(
        m_skyDebugSettings.sunYawDegrees,
        m_skyDebugSettings.sunPitchDegrees
    ));
    const odai::math::Vector3 toSun = -odai::math::normalize(sunDirection);
    const float sunElevationDegrees = odai::math::degrees(std::asin(std::clamp(toSun.y, -1.0f, 1.0f)));

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

    const odai::math::Vector3 sunColor = isNight
        ? odai::math::Vector3{0.0f, 0.0f, 0.0f}
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

    std::array<odai::math::Matrix4, kShadowCascadeCount> lightViewProjMatrices{};
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        const float cascadeFar = cascadeDistances[cascadeIndex];
        const float farHalfHeight = cascadeFar * tanHalfFov;
        const float farHalfWidth = farHalfHeight * aspectRatio;

        // Camera-position-only cascades: only translation moves cascade centers; rotation does not.
        const odai::math::Vector3 frustumCenter = eye;
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
        const float sunUpDot = std::abs(odai::math::dot(sunDirection, odai::math::Vector3{0.0f, 1.0f, 0.0f}));
        const odai::math::Vector3 lightUpHint =
            (sunUpDot > 0.95f) ? odai::math::Vector3{0.0f, 0.0f, 1.0f} : odai::math::Vector3{0.0f, 1.0f, 0.0f};
        const odai::math::Vector3 lightForward = odai::math::normalize(sunDirection);
        const odai::math::Vector3 lightRight = odai::math::normalize(odai::math::cross(lightForward, lightUpHint));
        const odai::math::Vector3 lightUp = odai::math::cross(lightRight, lightForward);

        // Stabilize translation by snapping the cascade center along light-view right/up texel units
        // before constructing the view matrix.
        const float centerRight = odai::math::dot(frustumCenter, lightRight);
        const float centerUp = odai::math::dot(frustumCenter, lightUp);
        const float snappedCenterRight = std::floor((centerRight / texelSize) + 0.5f) * texelSize;
        const float snappedCenterUp = std::floor((centerUp / texelSize) + 0.5f) * texelSize;
        const odai::math::Vector3 snappedFrustumCenter =
            frustumCenter +
            (lightRight * (snappedCenterRight - centerRight)) +
            (lightUp * (snappedCenterUp - centerUp));

        const odai::math::Vector3 lightPosition = snappedFrustumCenter - (lightForward * lightDistance);
        const odai::math::Matrix4 lightView = lookAt(lightPosition, snappedFrustumCenter, lightUp);

        const float left = -cascadeRadius;
        const float right = cascadeRadius;
        const float bottom = -cascadeRadius;
        const float top = cascadeRadius;
        // Keep a stable but tighter depth range per cascade to improve depth precision.
        const float casterPadding = std::max(24.0f, cascadeRadius * 0.35f);
        const float lightNear = std::max(0.1f, lightDistance - cascadeRadius - casterPadding);
        const float lightFar = lightDistance + cascadeRadius + casterPadding;
        const odai::math::Matrix4 lightProjection = orthographicVulkan(
            left,
            right,
            bottom,
            top,
            lightNear,
            lightFar
        );
        lightViewProjMatrices[cascadeIndex] = lightProjection * lightView;
    }

    std::array<odai::math::Vector3, 9> shIrradiance{};
    if (!isNight) {
        shIrradiance = computeIrradianceShCoefficients(sunDirection, sunColor, effectiveSkySettings);
    } else {
        for (odai::math::Vector3& coefficient : shIrradiance) {
            coefficient = odai::math::Vector3{0.0f, 0.0f, 0.0f};
        }
        // Constant dark-blue ambient irradiance for night.
        constexpr float kShY00 = 0.282095f;
        const odai::math::Vector3 nightAmbientIrradiance{0.050f, 0.078f, 0.155f};
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
        const odai::math::Matrix4 lightViewProjColumnMajor = transpose(lightViewProjMatrices[cascadeIndex]);
        const odai::math::Matrix4 inverseLightViewProjColumnMajor =
            transpose(odai::math::inverse(lightViewProjMatrices[cascadeIndex]));
        std::memcpy(
            mvpUniform.lightViewProj[cascadeIndex],
            lightViewProjColumnMajor.m,
            sizeof(mvpUniform.lightViewProj[cascadeIndex])
        );
        std::memcpy(
            mvpUniform.invLightViewProj[cascadeIndex],
            inverseLightViewProjColumnMajor.m,
            sizeof(mvpUniform.invLightViewProj[cascadeIndex])
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
    mvpUniform.shadowConfig4[0] = static_cast<float>(std::clamp(m_shadowDebugSettings.rtShadowSampleCount, 1, 8));
    mvpUniform.shadowConfig4[1] = std::clamp(m_shadowDebugSettings.rtSunAngularRadiusDegrees, 0.0f, 1.0f);
    mvpUniform.shadowConfig4[2] = static_cast<float>(std::clamp(m_voxelGiDebugSettings.rtSurfaceSampleCount, 1, 2));
    mvpUniform.shadowConfig4[3] = std::clamp(m_voxelGiDebugSettings.rtSurfaceBiasScale, 0.25f, 4.0f);
    mvpUniform.voxelGiRestirConfig0[0] = static_cast<float>(static_cast<int>(m_voxelGiDebugSettings.surfaceMode));
    mvpUniform.voxelGiRestirConfig0[1] =
        static_cast<float>(std::clamp(m_voxelGiDebugSettings.restirCandidateCount, 1, 8));
    mvpUniform.voxelGiRestirConfig0[2] = m_voxelGiDebugSettings.restirEnableTemporalReuse ? 1.0f : 0.0f;
    mvpUniform.voxelGiRestirConfig0[3] = m_voxelGiRestirHistoryValid ? 1.0f : 0.0f;
    mvpUniform.voxelGiRestirConfig1[0] = m_voxelGiDebugSettings.restirEnableSpatialReuse ? 1.0f : 0.0f;
    mvpUniform.voxelGiRestirConfig1[1] =
        static_cast<float>(std::clamp(m_voxelGiDebugSettings.restirSpatialRadius, 1, 2));
    mvpUniform.voxelGiRestirConfig1[2] = 4.0f;
    mvpUniform.voxelGiRestirConfig1[3] = 0.0f;

    // Reuse origin XYZ for fixed GI rebalance + debug mode to avoid enlarging camera UBO.
    mvpUniform.shadowVoxelGridOrigin[0] = kVoxelGiAmbientRebalanceStrength;
    mvpUniform.shadowVoxelGridOrigin[1] = kVoxelGiAmbientFloor;
    mvpUniform.shadowVoxelGridOrigin[2] =
        static_cast<float>(std::clamp(m_voxelGiDebugSettings.visualizationMode, 0, 5));
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
    mvpUniform.skyConfig4[0] = m_importedSceneInteriorMode
        ? 0.0f
        : std::clamp(m_skyDebugSettings.volumetricFogDensity, 0.0f, 1.0f);
    mvpUniform.skyConfig4[1] = std::clamp(m_skyDebugSettings.volumetricFogHeightFalloff, 0.0f, 1.0f);
    mvpUniform.skyConfig4[2] = m_skyDebugSettings.volumetricFogBaseHeight;
    mvpUniform.skyConfig4[3] = std::clamp(m_skyDebugSettings.volumetricSunScattering, 0.0f, 8.0f);
    const uint32_t autoExposureUpdateIntervalFrames = std::max(
        1u,
        static_cast<uint32_t>(std::max(1, m_skyDebugSettings.autoExposureUpdateIntervalFrames))
    );
    const bool autoExposureEnabled = m_skyDebugSettings.autoExposureEnabled && m_autoExposureComputeAvailable;
    mvpUniform.skyConfig5[0] = autoExposureEnabled ? 1.0f : 0.0f;
    mvpUniform.skyConfig5[1] = std::clamp(m_skyDebugSettings.manualExposure, 0.05f, 8.0f);
    mvpUniform.skyConfig5[2] = (m_morrowindSkyTextureImageView != VK_NULL_HANDLE) ? 1.0f : 0.0f;
    mvpUniform.skyConfig5[3] = std::clamp(m_skyDebugSettings.waterRefractionDecay, 0.25f, 3.0f);
    mvpUniform.colorGrading0[0] = std::clamp(m_skyDebugSettings.colorGradingWhiteBalanceR, 0.0f, 4.0f);
    mvpUniform.colorGrading0[1] = std::clamp(m_skyDebugSettings.colorGradingWhiteBalanceG, 0.0f, 4.0f);
    mvpUniform.colorGrading0[2] = std::clamp(m_skyDebugSettings.colorGradingWhiteBalanceB, 0.0f, 4.0f);
    mvpUniform.colorGrading0[3] = std::clamp(m_skyDebugSettings.colorGradingContrast, 0.70f, 1.40f);
    mvpUniform.colorGrading1[0] = std::clamp(m_skyDebugSettings.colorGradingSaturation, 0.0f, 2.0f);
    mvpUniform.colorGrading1[1] = std::clamp(m_skyDebugSettings.colorGradingVibrance, -1.0f, 1.0f);
    mvpUniform.colorGrading1[2] = std::clamp(m_skyDebugSettings.colorGradingMidtoneContrast, 0.80f, 1.40f);
    mvpUniform.colorGrading1[3] = std::clamp(m_skyDebugSettings.colorGradingShadowDensity, 0.70f, 1.40f);
    mvpUniform.colorGrading2[0] = std::clamp(m_skyDebugSettings.colorGradingShadowTintR, -1.0f, 1.0f);
    mvpUniform.colorGrading2[1] = std::clamp(m_skyDebugSettings.colorGradingShadowTintG, -1.0f, 1.0f);
    mvpUniform.colorGrading2[2] = std::clamp(m_skyDebugSettings.colorGradingShadowTintB, -1.0f, 1.0f);
    mvpUniform.colorGrading2[3] = std::clamp(m_skyDebugSettings.colorGradingHighlightRolloff, 0.70f, 1.10f);
    mvpUniform.colorGrading3[0] = std::clamp(m_skyDebugSettings.colorGradingHighlightTintR, -1.0f, 1.0f);
    mvpUniform.colorGrading3[1] = std::clamp(m_skyDebugSettings.colorGradingHighlightTintG, -1.0f, 1.0f);
    mvpUniform.colorGrading3[2] = std::clamp(m_skyDebugSettings.colorGradingHighlightTintB, -1.0f, 1.0f);
    mvpUniform.colorGrading3[3] = 0.0f;
    mvpUniform.dofConfig[0] = m_skyDebugSettings.depthOfFieldEnabled ? 1.0f : 0.0f;
    mvpUniform.dofConfig[1] = std::clamp(m_skyDebugSettings.depthOfFieldFocusDistance, 0.5f, 5000.0f);
    mvpUniform.dofConfig[2] = std::clamp(m_skyDebugSettings.depthOfFieldFocusRange, 0.5f, 1000.0f);
    mvpUniform.dofConfig[3] = std::clamp(m_skyDebugSettings.depthOfFieldMaxRadiusPixels, 0.0f, 20.0f);
    mvpUniform.dofConfig2[0] = std::clamp(m_skyDebugSettings.depthOfFieldNearBlurScale, 0.25f, 3.0f);
    mvpUniform.dofConfig2[1] = std::clamp(m_skyDebugSettings.waterRefractionStrength, 0.0f, 3.0f);
    mvpUniform.dofConfig2[2] =
        std::clamp(m_skyDebugSettings.waterRefractionDistortionPixels, 0.0f, 160.0f);
    mvpUniform.dofConfig2[3] = static_cast<float>(std::clamp(m_skyDebugSettings.waterDebugMode, 0, 6));
    mvpUniform.waterConfig[0] = std::clamp(m_skyDebugSettings.waterAnimationSpeed, 0.25f, 4.0f);
    mvpUniform.waterConfig[1] = std::clamp(m_skyDebugSettings.waterNormalStrength, 0.25f, 2.5f);
    mvpUniform.waterConfig[2] = std::clamp(m_skyDebugSettings.waterReflectionStrength, 0.25f, 4.0f);
    mvpUniform.waterConfig[3] = std::clamp(m_skyDebugSettings.waterRefractionDecay, 0.25f, 5.0f);

    struct SelectedImportedLight {
        const ImportedLocalLight* light = nullptr;
        float score = 0.0f;
    };
    std::array<SelectedImportedLight, kImportedLocalLightCapacity> selectedImportedLights{};
    std::size_t selectedImportedLightCount = 0;
    const float importedLightRadiusScale = std::clamp(m_debugImportedLightRadiusScale, 0.25f, 8.0f);
    if (m_debugImportedLightsEnabled && !m_importedLocalLights.empty()) {
        for (const ImportedLocalLight& light : m_importedLocalLights) {
            const odai::math::Vector3 lightPosition{light.position[0], light.position[1], light.position[2]};
            const odai::math::Vector3 cameraToLight = lightPosition - eye;
            const float influenceRadius = std::max(light.radius * importedLightRadiusScale, 1.0f);
            const float alongView = odai::math::dot(cameraToLight, forward);
            const odai::math::Vector3 closestViewPoint = eye + (forward * std::max(alongView, 0.0f));
            const odai::math::Vector3 lightToViewRay = lightPosition - closestViewPoint;
            const float viewRayDistanceSquared = odai::math::lengthSquared(lightToViewRay);
            const float distanceSquared = odai::math::lengthSquared(cameraToLight);
            const float radiusSquared = influenceRadius * influenceRadius;
            const float viewInfluenceScore =
                std::max(viewRayDistanceSquared - radiusSquared, 0.0f) / radiusSquared;
            const float distanceScore = distanceSquared / std::max(radiusSquared, 1.0f);
            const float behindCameraPenalty = alongView < -influenceRadius ? 16.0f : 0.0f;
            const SelectedImportedLight selected{
                &light,
                viewInfluenceScore + (distanceScore * 0.08f) + behindCameraPenalty
            };
            if (selectedImportedLightCount < selectedImportedLights.size()) {
                selectedImportedLights[selectedImportedLightCount++] = selected;
                continue;
            }
            std::size_t worstIndex = 0;
            float worstScore = selectedImportedLights[0].score;
            for (std::size_t lightIndex = 1; lightIndex < selectedImportedLights.size(); ++lightIndex) {
                if (selectedImportedLights[lightIndex].score > worstScore) {
                    worstScore = selectedImportedLights[lightIndex].score;
                    worstIndex = lightIndex;
                }
            }
            if (selected.score < worstScore) {
                selectedImportedLights[worstIndex] = selected;
            }
        }
        std::sort(
            selectedImportedLights.begin(),
            selectedImportedLights.begin() + static_cast<std::ptrdiff_t>(selectedImportedLightCount),
            [](const SelectedImportedLight& a, const SelectedImportedLight& b) {
                return a.score < b.score;
            });
    }
    m_debugImportedLightSelectedCount = static_cast<std::uint32_t>(selectedImportedLightCount);
    const float importedLightGlobalIntensity =
        std::clamp(m_debugImportedLightIntensity, 0.0f, 8.0f);
    auto mixImportedLightSignature = [](std::uint64_t hash, std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
        return hash;
    };
    auto mixImportedLightFloat = [&](std::uint64_t hash, float value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return mixImportedLightSignature(hash, static_cast<std::uint64_t>(bits));
    };
    std::uint64_t importedLightSignature = 1469598103934665603ull;
    importedLightSignature = mixImportedLightSignature(
        importedLightSignature,
        m_debugImportedLightsEnabled ? 1ull : 0ull);
    importedLightSignature = mixImportedLightSignature(
        importedLightSignature,
        static_cast<std::uint64_t>(selectedImportedLightCount));
    importedLightSignature = mixImportedLightFloat(importedLightSignature, importedLightGlobalIntensity);
    importedLightSignature = mixImportedLightFloat(importedLightSignature, importedLightRadiusScale);
    for (std::size_t lightIndex = 0; lightIndex < selectedImportedLightCount; ++lightIndex) {
        const ImportedLocalLight& light = *selectedImportedLights[lightIndex].light;
        const float lightRadius = std::max(light.radius * importedLightRadiusScale, 1.0f);
        mvpUniform.importedLightPositionRadius[lightIndex][0] = light.position[0];
        mvpUniform.importedLightPositionRadius[lightIndex][1] = light.position[1];
        mvpUniform.importedLightPositionRadius[lightIndex][2] = light.position[2];
        mvpUniform.importedLightPositionRadius[lightIndex][3] = lightRadius;
        mvpUniform.importedLightColorIntensity[lightIndex][0] = light.color[0];
        mvpUniform.importedLightColorIntensity[lightIndex][1] = light.color[1];
        mvpUniform.importedLightColorIntensity[lightIndex][2] = light.color[2];
        mvpUniform.importedLightColorIntensity[lightIndex][3] = light.intensity;
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.position[0]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.position[1]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.position[2]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, lightRadius);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.color[0]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.color[1]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.color[2]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.intensity);
    }
    mvpUniform.importedLightConfig[0] = static_cast<float>(selectedImportedLightCount);
    mvpUniform.importedLightConfig[1] = importedLightGlobalIntensity;
    mvpUniform.importedLightConfig[2] = m_debugImportedLightsEnabled ? 1.0f : 0.0f;
    mvpUniform.importedLightConfig[3] = static_cast<float>(m_importedLocalLights.size());

    const float voxelGiGridSpan = static_cast<float>(kVoxelGiGridResolution) * kVoxelGiCellSize;
    const float voxelGiHalfSpan = voxelGiGridSpan * 0.5f;
    const float voxelGiDesiredOriginX = computeVoxelGiAxisOrigin(camera.x, voxelGiHalfSpan, kVoxelGiCellSize);
    const float voxelGiDesiredOriginY = computeVoxelGiAxisOrigin(camera.y, voxelGiHalfSpan, kVoxelGiCellSize);
    const float voxelGiDesiredOriginZ = computeVoxelGiAxisOrigin(camera.z, voxelGiHalfSpan, kVoxelGiCellSize);
    const float kVoxelGiHorizontalFollowThreshold = kVoxelGiCellSize * 8.0f;
    const float kVoxelGiVerticalFollowThreshold = kVoxelGiCellSize * 4.0f;
    float voxelGiOriginX = voxelGiDesiredOriginX;
    float voxelGiOriginY = voxelGiDesiredOriginY;
    float voxelGiOriginZ = voxelGiDesiredOriginZ;
    const bool keepVoxelGiBuildAnchor =
        m_voxelGiOccupancyFullRebuildInProgress || m_voxelGiOccupancyFullRebuildNeedsClear;
    const bool keepVoxelGiGridAnchored =
        m_voxelGiHasPreviousFrameState &&
        m_voxelGiOccupancyInitialized &&
        !m_voxelGiWorldDirty;
    if (keepVoxelGiBuildAnchor) {
        voxelGiOriginX = m_voxelGiOccupancyBuildOrigin[0];
        voxelGiOriginY = m_voxelGiOccupancyBuildOrigin[1];
        voxelGiOriginZ = m_voxelGiOccupancyBuildOrigin[2];
    } else if (keepVoxelGiGridAnchored) {
        voxelGiOriginX = m_voxelGiPreviousGridOrigin[0];
        voxelGiOriginY = m_voxelGiPreviousGridOrigin[1];
        voxelGiOriginZ = m_voxelGiPreviousGridOrigin[2];
    } else {
        voxelGiOriginX = computeVoxelGiStableOriginY(
            voxelGiDesiredOriginX,
            m_voxelGiPreviousGridOrigin[0],
            m_voxelGiHasPreviousFrameState,
            kVoxelGiHorizontalFollowThreshold
        );
        voxelGiOriginY = computeVoxelGiStableOriginY(
            voxelGiDesiredOriginY,
            m_voxelGiPreviousGridOrigin[1],
            m_voxelGiHasPreviousFrameState,
            kVoxelGiVerticalFollowThreshold
        );
        voxelGiOriginZ = computeVoxelGiStableOriginY(
            voxelGiDesiredOriginZ,
            m_voxelGiPreviousGridOrigin[2],
            m_voxelGiHasPreviousFrameState,
            kVoxelGiHorizontalFollowThreshold
        );
    }
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
    if (!voxelGiSceneEnabled) {
        m_voxelGiWorldDirty = false;
        m_voxelGiOccupancyFullRebuildInProgress = false;
        m_voxelGiOccupancyFullRebuildNeedsClear = false;
        m_voxelGiOccupancyFullRebuildCursor = 0;
        m_voxelGiDirtyChunkIndices.clear();
    }
    const bool voxelGiNeedsOccupancyUpload = voxelGiSceneEnabled && voxelGiFlags.needsOccupancyUpload;
    const bool voxelGiRtSurfaceSettingsChanged =
        !m_voxelGiHasPreviousFrameState ||
        (m_voxelGiDebugSettings.surfaceMode != VoxelGiSurfaceMode::Legacy) != m_voxelGiPreviousRtSurfaceTracingEnabled ||
        std::abs(static_cast<float>(m_voxelGiDebugSettings.rtSurfaceSampleCount) - m_voxelGiPreviousRtSurfaceSampleCount) >
            kVoxelGiTuningChangeThreshold ||
        std::abs(m_voxelGiDebugSettings.rtSurfaceBiasScale - m_voxelGiPreviousRtSurfaceBiasScale) >
            kVoxelGiTuningChangeThreshold ||
        std::abs(m_shadowDebugSettings.rtSunAngularRadiusDegrees - m_voxelGiPreviousRtSunAngularRadiusDegrees) >
            kVoxelGiTuningChangeThreshold;
    const bool voxelGiRestirSettingsChanged =
        !m_voxelGiHasPreviousFrameState ||
        m_voxelGiDebugSettings.surfaceMode != m_voxelGiPreviousSurfaceMode ||
        std::abs(static_cast<float>(m_voxelGiDebugSettings.restirCandidateCount) - m_voxelGiPreviousRestirCandidateCount) >
            kVoxelGiTuningChangeThreshold ||
        m_voxelGiDebugSettings.restirEnableTemporalReuse != m_voxelGiPreviousRestirTemporalReuseEnabled ||
        m_voxelGiDebugSettings.restirEnableSpatialReuse != m_voxelGiPreviousRestirSpatialReuseEnabled ||
        std::abs(static_cast<float>(m_voxelGiDebugSettings.restirSpatialRadius) - m_voxelGiPreviousRestirSpatialRadius) >
            kVoxelGiTuningChangeThreshold;
    const bool importedGiLightStateChanged =
        importedInteriorGiEnabled &&
        (!m_voxelGiPreviousImportedLightSignatureValid ||
         importedLightSignature != m_voxelGiPreviousImportedLightSignature);
    if (m_voxelGiDebugSettings.restirHistoryResetRequested) {
        m_voxelGiRestirHistoryValid = false;
        m_voxelGiRestirHistoryResetReason = "manual_reset";
        m_voxelGiDebugSettings.restirHistoryResetRequested = false;
    }
    if (m_voxelGiWorldDirty) {
        m_voxelGiRestirHistoryValid = false;
        m_voxelGiRestirHistoryResetReason = "world_dirty";
    } else if (!m_voxelGiHasPreviousFrameState) {
        m_voxelGiRestirHistoryValid = false;
        m_voxelGiRestirHistoryResetReason = "startup";
    } else if (voxelGiRestirSettingsChanged) {
        m_voxelGiRestirHistoryValid = false;
        m_voxelGiRestirHistoryResetReason = "restir_settings";
    } else if (importedGiLightStateChanged) {
        m_voxelGiRestirHistoryValid = false;
        m_voxelGiRestirHistoryResetReason = "imported_lights";
    } else if (voxelGiFlags.needsComputeUpdate && voxelGiFlags.needsOccupancyUpload) {
        m_voxelGiRestirHistoryValid = false;
        m_voxelGiRestirHistoryResetReason = "occupancy_rebuild";
    }
    const bool voxelGiNeedsComputeUpdate =
        voxelGiSceneEnabled &&
        (voxelGiFlags.needsComputeUpdate ||
         voxelGiRtSurfaceSettingsChanged ||
         voxelGiRestirSettingsChanged ||
         importedGiLightStateChanged ||
         !m_voxelGiInitialized);
    m_voxelGiHasPreviousFrameState = true;
    m_voxelGiPreviousGridOrigin = {voxelGiOriginX, voxelGiOriginY, voxelGiOriginZ};
    m_voxelGiPreviousSunDirection = {sunDirection.x, sunDirection.y, sunDirection.z};
    m_voxelGiPreviousSunColor = {sunColor.x, sunColor.y, sunColor.z};
    for (std::size_t coeffIndex = 0; coeffIndex < shIrradiance.size(); ++coeffIndex) {
        const odai::math::Vector3& coeff = shIrradiance[coeffIndex];
        m_voxelGiPreviousShIrradiance[coeffIndex] = {coeff.x, coeff.y, coeff.z};
    }
    m_voxelGiPreviousBounceStrength = m_voxelGiDebugSettings.bounceStrength;
    m_voxelGiPreviousDiffusionSoftness = m_voxelGiDebugSettings.diffusionSoftness;
    m_voxelGiPreviousRtSurfaceTracingEnabled = m_voxelGiDebugSettings.surfaceMode != VoxelGiSurfaceMode::Legacy;
    m_voxelGiPreviousRtSurfaceSampleCount = static_cast<float>(m_voxelGiDebugSettings.rtSurfaceSampleCount);
    m_voxelGiPreviousRtSurfaceBiasScale = m_voxelGiDebugSettings.rtSurfaceBiasScale;
    m_voxelGiPreviousRtSunAngularRadiusDegrees = m_shadowDebugSettings.rtSunAngularRadiusDegrees;
    m_voxelGiPreviousSurfaceMode = m_voxelGiDebugSettings.surfaceMode;
    m_voxelGiPreviousRestirCandidateCount = static_cast<float>(m_voxelGiDebugSettings.restirCandidateCount);
    m_voxelGiPreviousRestirTemporalReuseEnabled = m_voxelGiDebugSettings.restirEnableTemporalReuse;
    m_voxelGiPreviousRestirSpatialReuseEnabled = m_voxelGiDebugSettings.restirEnableSpatialReuse;
    m_voxelGiPreviousRestirSpatialRadius = static_cast<float>(m_voxelGiDebugSettings.restirSpatialRadius);
    m_voxelGiPreviousImportedLightSignature = importedLightSignature;
    m_voxelGiPreviousImportedLightSignatureValid = importedInteriorGiEnabled;
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
    mvpUniform.voxelGiRestirConfig0[3] = m_voxelGiRestirHistoryValid ? 1.0f : 0.0f;
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
    if (const void* exposureStateMapped = m_bufferAllocator.mapBuffer(
            m_autoExposureStateBufferHandle,
            0,
            sizeof(float) * 4u
        )) {
        const auto* exposureState = static_cast<const float*>(exposureStateMapped);
        m_debugResolvedExposure = std::max(exposureState[0], 0.001f);
        m_debugTargetExposure = std::max(exposureState[1], 0.001f);
        m_debugAverageSceneLuminance = std::max(exposureState[2], 0.0f);
        m_bufferAllocator.unmapBuffer(m_autoExposureStateBufferHandle);
    } else {
        m_debugResolvedExposure = std::max(m_skyDebugSettings.manualExposure, 0.001f);
        m_debugTargetExposure = m_debugResolvedExposure;
        m_debugAverageSceneLuminance = 0.0f;
    }

    struct VoxelGiChunkMetaUpload {
        int32_t worldMinX = 0;
        int32_t worldMinY = 0;
        int32_t worldMinZ = 0;
        uint32_t voxelOffset = 0;
    };
    struct ImportedGiOccupancyChunk {
        int32_t worldMinX = 0;
        int32_t worldMinY = 0;
        int32_t worldMinZ = 0;
        std::vector<uint32_t> voxels;
    };
    constexpr uint32_t kVoxelGiChunkVoxelCount =
        static_cast<uint32_t>(odai::world::Chunk::kSizeX) *
        static_cast<uint32_t>(odai::world::Chunk::kSizeY) *
        static_cast<uint32_t>(odai::world::Chunk::kSizeZ);
    constexpr uint32_t kVoxelGiOccupancyChunkBudgetPerFrame = 8u;
    constexpr float kVoxelGiOccupancyOriginRebuildThreshold = 0.001f;
    constexpr uint32_t kImportedGiVoxelType = 250u;

    std::optional<VkDescriptorBufferInfo> voxelGiChunkMetaDescriptorInfo = std::nullopt;
    std::optional<VkDescriptorBufferInfo> voxelGiChunkVoxelDescriptorInfo = std::nullopt;
    uint32_t voxelGiOccupancyDispatchZ = 0;
    bool voxelGiOccupancyClearThisFrame = false;
    float voxelGiOccupancyCpuMs = 0.0f;
    uint32_t importedGiVoxelizedCellCount = 0u;

    auto buildImportedGiOccupancyChunks = [&]() {
        std::vector<ImportedGiOccupancyChunk> chunks;
        if (m_importedGiTriangles.empty()) {
            return chunks;
        }
        chunks.reserve(8u);
        const int32_t originX = static_cast<int32_t>(std::floor(voxelGiOriginX));
        const int32_t originY = static_cast<int32_t>(std::floor(voxelGiOriginY));
        const int32_t originZ = static_cast<int32_t>(std::floor(voxelGiOriginZ));
        for (int cz = 0; cz < 2; ++cz) {
            for (int cy = 0; cy < 2; ++cy) {
                for (int cx = 0; cx < 2; ++cx) {
                    ImportedGiOccupancyChunk chunk{};
                    chunk.worldMinX = originX + (cx * odai::world::Chunk::kSizeX);
                    chunk.worldMinY = originY + (cy * odai::world::Chunk::kSizeY);
                    chunk.worldMinZ = originZ + (cz * odai::world::Chunk::kSizeZ);
                    chunk.voxels.assign(kVoxelGiChunkVoxelCount, 0u);
                    chunks.push_back(std::move(chunk));
                }
            }
        }
        const auto packVoxel = [&](const float albedo[3]) {
            const uint32_t r = static_cast<uint32_t>(std::clamp(albedo[0], 0.0f, 1.0f) * 255.0f + 0.5f);
            const uint32_t g = static_cast<uint32_t>(std::clamp(albedo[1], 0.0f, 1.0f) * 255.0f + 0.5f);
            const uint32_t b = static_cast<uint32_t>(std::clamp(albedo[2], 0.0f, 1.0f) * 255.0f + 0.5f);
            return kImportedGiVoxelType | (r << 8u) | (g << 16u) | (b << 24u);
        };
        const auto markCell = [&](int gx, int gy, int gz, const float albedo[3]) {
            if (gx < 0 || gy < 0 || gz < 0 ||
                gx >= static_cast<int>(kVoxelGiGridResolution) ||
                gy >= static_cast<int>(kVoxelGiGridResolution) ||
                gz >= static_cast<int>(kVoxelGiGridResolution)) {
                return;
            }
            const int chunkX = std::clamp(gx / odai::world::Chunk::kSizeX, 0, 1);
            const int chunkY = std::clamp(gy / odai::world::Chunk::kSizeY, 0, 1);
            const int chunkZ = std::clamp(gz / odai::world::Chunk::kSizeZ, 0, 1);
            const int localX = gx - (chunkX * odai::world::Chunk::kSizeX);
            const int localY = gy - (chunkY * odai::world::Chunk::kSizeY);
            const int localZ = gz - (chunkZ * odai::world::Chunk::kSizeZ);
            const std::size_t chunkIndex = static_cast<std::size_t>((chunkZ * 4) + (chunkY * 2) + chunkX);
            const std::size_t voxelIndex =
                static_cast<std::size_t>(localX) +
                (static_cast<std::size_t>(odai::world::Chunk::kSizeX) *
                    (static_cast<std::size_t>(localZ) +
                     (static_cast<std::size_t>(odai::world::Chunk::kSizeZ) * static_cast<std::size_t>(localY))));
            if (chunkIndex >= chunks.size() || voxelIndex >= chunks[chunkIndex].voxels.size()) {
                return;
            }
            if (chunks[chunkIndex].voxels[voxelIndex] == 0u) {
                ++importedGiVoxelizedCellCount;
            }
            chunks[chunkIndex].voxels[voxelIndex] = packVoxel(albedo);
        };
        const auto markPoint = [&](const float p[3], const float albedo[3]) {
            markCell(
                static_cast<int>(std::floor((p[0] - voxelGiOriginX) / kVoxelGiCellSize)),
                static_cast<int>(std::floor((p[1] - voxelGiOriginY) / kVoxelGiCellSize)),
                static_cast<int>(std::floor((p[2] - voxelGiOriginZ) / kVoxelGiCellSize)),
                albedo);
        };
        constexpr int kMaxFilledCellsPerTriangle = 512;
        for (const ImportedGiTriangle& triangle : m_importedGiTriangles) {
            const float minX = std::min({triangle.p0[0], triangle.p1[0], triangle.p2[0]});
            const float minY = std::min({triangle.p0[1], triangle.p1[1], triangle.p2[1]});
            const float minZ = std::min({triangle.p0[2], triangle.p1[2], triangle.p2[2]});
            const float maxX = std::max({triangle.p0[0], triangle.p1[0], triangle.p2[0]});
            const float maxY = std::max({triangle.p0[1], triangle.p1[1], triangle.p2[1]});
            const float maxZ = std::max({triangle.p0[2], triangle.p1[2], triangle.p2[2]});
            const float gridMaxX = voxelGiOriginX + static_cast<float>(kVoxelGiGridResolution) * kVoxelGiCellSize;
            const float gridMaxY = voxelGiOriginY + static_cast<float>(kVoxelGiGridResolution) * kVoxelGiCellSize;
            const float gridMaxZ = voxelGiOriginZ + static_cast<float>(kVoxelGiGridResolution) * kVoxelGiCellSize;
            if (maxX < voxelGiOriginX || maxY < voxelGiOriginY || maxZ < voxelGiOriginZ ||
                minX > gridMaxX || minY > gridMaxY || minZ > gridMaxZ) {
                continue;
            }
            const int gx0 = std::clamp(static_cast<int>(std::floor((minX - voxelGiOriginX) / kVoxelGiCellSize)) - 1, 0, static_cast<int>(kVoxelGiGridResolution) - 1);
            const int gy0 = std::clamp(static_cast<int>(std::floor((minY - voxelGiOriginY) / kVoxelGiCellSize)) - 1, 0, static_cast<int>(kVoxelGiGridResolution) - 1);
            const int gz0 = std::clamp(static_cast<int>(std::floor((minZ - voxelGiOriginZ) / kVoxelGiCellSize)) - 1, 0, static_cast<int>(kVoxelGiGridResolution) - 1);
            const int gx1 = std::clamp(static_cast<int>(std::floor((maxX - voxelGiOriginX) / kVoxelGiCellSize)) + 1, 0, static_cast<int>(kVoxelGiGridResolution) - 1);
            const int gy1 = std::clamp(static_cast<int>(std::floor((maxY - voxelGiOriginY) / kVoxelGiCellSize)) + 1, 0, static_cast<int>(kVoxelGiGridResolution) - 1);
            const int gz1 = std::clamp(static_cast<int>(std::floor((maxZ - voxelGiOriginZ) / kVoxelGiCellSize)) + 1, 0, static_cast<int>(kVoxelGiGridResolution) - 1);
            const int cellCount = (gx1 - gx0 + 1) * (gy1 - gy0 + 1) * (gz1 - gz0 + 1);
            if (cellCount > kMaxFilledCellsPerTriangle) {
                markPoint(triangle.p0, triangle.albedo);
                markPoint(triangle.p1, triangle.albedo);
                markPoint(triangle.p2, triangle.albedo);
                const float center[3] = {
                    (triangle.p0[0] + triangle.p1[0] + triangle.p2[0]) * (1.0f / 3.0f),
                    (triangle.p0[1] + triangle.p1[1] + triangle.p2[1]) * (1.0f / 3.0f),
                    (triangle.p0[2] + triangle.p1[2] + triangle.p2[2]) * (1.0f / 3.0f)
                };
                markPoint(center, triangle.albedo);
                continue;
            }
            for (int gz = gz0; gz <= gz1; ++gz) {
                for (int gy = gy0; gy <= gy1; ++gy) {
                    for (int gx = gx0; gx <= gx1; ++gx) {
                        markCell(gx, gy, gz, triangle.albedo);
                    }
                }
            }
        }
        return chunks;
    };

    if (voxelGiNeedsOccupancyUpload &&
        m_voxelGiComputeAvailable &&
        m_voxelGiOccupancyImage != VK_NULL_HANDLE &&
        m_voxelGiOccupancyImageView != VK_NULL_HANDLE) {
        const auto occupancyCpuStartTime = std::chrono::steady_clock::now();
        const std::array<float, 3> voxelGiBuildOrigin = {voxelGiOriginX, voxelGiOriginY, voxelGiOriginZ};
        const bool occupancyBuildOriginChanged =
            std::abs(m_voxelGiOccupancyBuildOrigin[0] - voxelGiBuildOrigin[0]) > kVoxelGiOccupancyOriginRebuildThreshold ||
            std::abs(m_voxelGiOccupancyBuildOrigin[1] - voxelGiBuildOrigin[1]) > kVoxelGiOccupancyOriginRebuildThreshold ||
            std::abs(m_voxelGiOccupancyBuildOrigin[2] - voxelGiBuildOrigin[2]) > kVoxelGiOccupancyOriginRebuildThreshold;
        if (occupancyBuildOriginChanged || !m_voxelGiOccupancyInitialized) {
            m_voxelGiOccupancyBuildOrigin = voxelGiBuildOrigin;
            m_voxelGiOccupancyFullRebuildInProgress = true;
            m_voxelGiOccupancyFullRebuildNeedsClear = true;
            m_voxelGiOccupancyFullRebuildCursor = 0;
            m_voxelGiDirtyChunkIndices.clear();
        } else if (!m_voxelGiOccupancyFullRebuildInProgress &&
                   m_voxelGiDirtyChunkIndices.empty() &&
                   m_voxelGiWorldDirty) {
            m_voxelGiOccupancyFullRebuildInProgress = true;
            m_voxelGiOccupancyFullRebuildNeedsClear = true;
            m_voxelGiOccupancyFullRebuildCursor = 0;
        }

        const std::size_t chunkCount = legacyVoxelRenderingEnabled ? chunkGrid.chunkCount() : 0u;
        if (m_voxelGiOccupancyFullRebuildInProgress && chunkCount == 0u && m_importedGiTriangles.empty()) {
            m_voxelGiOccupancyFullRebuildInProgress = false;
            m_voxelGiOccupancyFullRebuildCursor = 0u;
        }

        std::vector<std::size_t> occupancyChunkBatch;
        occupancyChunkBatch.reserve(kVoxelGiOccupancyChunkBudgetPerFrame);
        const bool buildFromFullRebuild = m_voxelGiOccupancyFullRebuildInProgress;
        const std::size_t fullRebuildBatchBegin = m_voxelGiOccupancyFullRebuildCursor;
        std::size_t dirtyBatchCount = 0;
        if (buildFromFullRebuild) {
            const std::size_t remainingChunks =
                (chunkCount > fullRebuildBatchBegin) ? (chunkCount - fullRebuildBatchBegin) : 0u;
            const std::size_t batchCount =
                std::min<std::size_t>(kVoxelGiOccupancyChunkBudgetPerFrame, remainingChunks);
            for (std::size_t i = 0; i < batchCount; ++i) {
                occupancyChunkBatch.push_back(fullRebuildBatchBegin + i);
            }
        } else {
            dirtyBatchCount = std::min<std::size_t>(
                kVoxelGiOccupancyChunkBudgetPerFrame,
                m_voxelGiDirtyChunkIndices.size()
            );
            const std::size_t dirtyStart = m_voxelGiDirtyChunkIndices.size() - dirtyBatchCount;
            for (std::size_t i = 0; i < dirtyBatchCount; ++i) {
                occupancyChunkBatch.push_back(m_voxelGiDirtyChunkIndices[dirtyStart + i]);
            }
        }
        std::vector<ImportedGiOccupancyChunk> importedGiChunks;
        if (buildFromFullRebuild || occupancyBuildOriginChanged || !m_voxelGiOccupancyInitialized) {
            importedGiChunks = buildImportedGiOccupancyChunks();
        }
        const std::size_t occupancySourceCount = occupancyChunkBatch.size() + importedGiChunks.size();

        if (occupancySourceCount != 0u) {
            const VkDeviceSize chunkMetaBytes =
                static_cast<VkDeviceSize>(occupancySourceCount * sizeof(VoxelGiChunkMetaUpload));
            const VkDeviceSize chunkVoxelsBytes =
                static_cast<VkDeviceSize>(occupancySourceCount) *
                static_cast<VkDeviceSize>(kVoxelGiChunkVoxelCount) *
                static_cast<VkDeviceSize>(sizeof(uint32_t));
            const std::optional<FrameArenaSlice> chunkMetaSliceOpt = m_frameArena.allocateUpload(
                chunkMetaBytes,
                static_cast<VkDeviceSize>(alignof(VoxelGiChunkMetaUpload)),
                FrameArenaUploadKind::Unknown
            );
            const std::optional<FrameArenaSlice> chunkVoxelsSliceOpt = m_frameArena.allocateUpload(
                chunkVoxelsBytes,
                static_cast<VkDeviceSize>(alignof(uint32_t)),
                FrameArenaUploadKind::Unknown
            );
            if (chunkMetaSliceOpt.has_value() &&
                chunkVoxelsSliceOpt.has_value() &&
                chunkMetaSliceOpt->mapped != nullptr &&
                chunkVoxelsSliceOpt->mapped != nullptr) {
                auto* chunkMeta = static_cast<VoxelGiChunkMetaUpload*>(chunkMetaSliceOpt->mapped);
                auto* chunkVoxels = static_cast<uint32_t*>(chunkVoxelsSliceOpt->mapped);
                const std::vector<odai::world::Chunk>& chunks = chunkGrid.chunks();
                for (std::size_t batchIndex = 0; batchIndex < occupancyChunkBatch.size(); ++batchIndex) {
                    const std::size_t chunkIndex = occupancyChunkBatch[batchIndex];
                    if (chunkIndex >= chunks.size()) {
                        continue;
                    }
                    const odai::world::Chunk& chunk = chunks[chunkIndex];
                    chunkMeta[batchIndex].worldMinX = chunk.chunkX() * odai::world::Chunk::kSizeX;
                    chunkMeta[batchIndex].worldMinY = chunk.chunkY() * odai::world::Chunk::kSizeY;
                    chunkMeta[batchIndex].worldMinZ = chunk.chunkZ() * odai::world::Chunk::kSizeZ;
                    chunkMeta[batchIndex].voxelOffset =
                        static_cast<uint32_t>(batchIndex * static_cast<std::size_t>(kVoxelGiChunkVoxelCount));

                    const std::vector<odai::world::Voxel>& voxels = chunk.voxels();
                    const std::size_t voxelWriteOffset = batchIndex * static_cast<std::size_t>(kVoxelGiChunkVoxelCount);
                    for (std::size_t voxelIndex = 0; voxelIndex < voxels.size(); ++voxelIndex) {
                        const odai::world::Voxel& voxel = voxels[voxelIndex];
                        const uint32_t packedVoxel = static_cast<uint32_t>(
                            static_cast<uint32_t>(static_cast<uint8_t>(voxel.type)) |
                            static_cast<uint32_t>(static_cast<uint32_t>(voxel.baseColorIndex) << 8u)
                        );
                        chunkVoxels[voxelWriteOffset + voxelIndex] = packedVoxel;
                    }
                }
                const std::size_t importedChunkBase = occupancyChunkBatch.size();
                for (std::size_t importedIndex = 0; importedIndex < importedGiChunks.size(); ++importedIndex) {
                    const std::size_t batchIndex = importedChunkBase + importedIndex;
                    const ImportedGiOccupancyChunk& importedChunk = importedGiChunks[importedIndex];
                    chunkMeta[batchIndex].worldMinX = importedChunk.worldMinX;
                    chunkMeta[batchIndex].worldMinY = importedChunk.worldMinY;
                    chunkMeta[batchIndex].worldMinZ = importedChunk.worldMinZ;
                    chunkMeta[batchIndex].voxelOffset =
                        static_cast<uint32_t>(batchIndex * static_cast<std::size_t>(kVoxelGiChunkVoxelCount)) |
                        0x80000000u;
                    const std::size_t voxelWriteOffset = batchIndex * static_cast<std::size_t>(kVoxelGiChunkVoxelCount);
                    std::copy(
                        importedChunk.voxels.begin(),
                        importedChunk.voxels.end(),
                        chunkVoxels + voxelWriteOffset);
                }

                const VkBuffer chunkMetaUploadBuffer = m_bufferAllocator.getBuffer(chunkMetaSliceOpt->buffer);
                const VkBuffer chunkVoxelsUploadBuffer = m_bufferAllocator.getBuffer(chunkVoxelsSliceOpt->buffer);
                if (chunkMetaUploadBuffer != VK_NULL_HANDLE && chunkVoxelsUploadBuffer != VK_NULL_HANDLE) {
                    voxelGiChunkMetaDescriptorInfo = VkDescriptorBufferInfo{
                        chunkMetaUploadBuffer,
                        chunkMetaSliceOpt->offset,
                        chunkMetaSliceOpt->size
                    };
                    voxelGiChunkVoxelDescriptorInfo = VkDescriptorBufferInfo{
                        chunkVoxelsUploadBuffer,
                        chunkVoxelsSliceOpt->offset,
                        chunkVoxelsSliceOpt->size
                    };
                    voxelGiOccupancyDispatchZ = static_cast<uint32_t>(
                        occupancySourceCount * static_cast<std::size_t>(odai::world::Chunk::kSizeZ)
                    );
                    if (buildFromFullRebuild) {
                        m_voxelGiOccupancyFullRebuildCursor = fullRebuildBatchBegin + occupancyChunkBatch.size();
                        if (m_voxelGiOccupancyFullRebuildCursor >= chunkCount) {
                            m_voxelGiOccupancyFullRebuildCursor = 0;
                            m_voxelGiOccupancyFullRebuildInProgress = false;
                        }
                    } else if (dirtyBatchCount > 0u) {
                        m_voxelGiDirtyChunkIndices.resize(m_voxelGiDirtyChunkIndices.size() - dirtyBatchCount);
                    }
                }
            } else {
                VOX_LOGW("render") << "voxel GI chunk occupancy upload allocation failed";
            }
        }

        voxelGiOccupancyCpuMs = static_cast<float>(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - occupancyCpuStartTime).count()
        );
        m_debugImportedGiVoxelizedCellCount = importedGiVoxelizedCellCount;
        if (importedGiVoxelizedCellCount > 0u) {
            VOX_LOGI("render") << "imported GI occupancy voxelized cells="
                               << importedGiVoxelizedCellCount;
        }
    } else if (!voxelGiNeedsOccupancyUpload) {
        m_voxelGiOccupancyFullRebuildInProgress = false;
        m_voxelGiOccupancyFullRebuildNeedsClear = false;
        m_voxelGiOccupancyFullRebuildCursor = 0;
        m_voxelGiDirtyChunkIndices.clear();
    }
    m_debugCpuGiOccupancyBuildMs = voxelGiOccupancyCpuMs;

    const BoundDescriptorSets boundDescriptorSets = updateFrameDescriptorSets(
        aoFrameIndex,
        bufferInfo,
        autoExposureHistogramBuffer,
        autoExposureStateBuffer,
        voxelGiChunkMetaDescriptorInfo.has_value() ? &(*voxelGiChunkMetaDescriptorInfo) : nullptr,
        voxelGiChunkVoxelDescriptorInfo.has_value() ? &(*voxelGiChunkVoxelDescriptorInfo) : nullptr
    );
    const uint32_t boundDescriptorSetCount = boundDescriptorSets.count;

    if (m_voxelGiOccupancyImage != VK_NULL_HANDLE &&
        m_voxelGiOccupancyFullRebuildNeedsClear &&
        (voxelGiNeedsOccupancyUpload || !m_voxelGiOccupancyInitialized)) {
        const VkImageLayout oldLayout = m_voxelGiOccupancyInitialized
            ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            : VK_IMAGE_LAYOUT_UNDEFINED;
        const VkPipelineStageFlags2 srcStageMask = m_voxelGiOccupancyInitialized
            ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
            : VK_PIPELINE_STAGE_2_NONE;
        const VkAccessFlags2 srcAccessMask = m_voxelGiOccupancyInitialized
            ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
            : VK_ACCESS_2_NONE;
        transitionImageLayout(
            commandBuffer,
            m_voxelGiOccupancyImage,
            oldLayout,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            srcStageMask,
            srcAccessMask,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        VkClearColorValue occupancyClearColor{};
        occupancyClearColor.float32[0] = 0.0f;
        occupancyClearColor.float32[1] = 0.0f;
        occupancyClearColor.float32[2] = 0.0f;
        occupancyClearColor.float32[3] = 0.0f;
        VkImageSubresourceRange occupancyClearRange{};
        occupancyClearRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        occupancyClearRange.baseMipLevel = 0;
        occupancyClearRange.levelCount = 1;
        occupancyClearRange.baseArrayLayer = 0;
        occupancyClearRange.layerCount = 1;
        vkCmdClearColorImage(
            commandBuffer,
            m_voxelGiOccupancyImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &occupancyClearColor,
            1,
            &occupancyClearRange
        );
        if (voxelGiOccupancyDispatchZ > 0u) {
            transitionImageLayout(
                commandBuffer,
                m_voxelGiOccupancyImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
        } else {
            transitionImageLayout(
                commandBuffer,
                m_voxelGiOccupancyImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
        }
        m_voxelGiOccupancyFullRebuildNeedsClear = false;
        m_voxelGiOccupancyInitialized = true;
        voxelGiOccupancyClearThisFrame = true;
    } else if (m_voxelGiOccupancyImage != VK_NULL_HANDLE &&
               !m_voxelGiOccupancyInitialized &&
               voxelGiOccupancyDispatchZ == 0u) {
        transitionImageLayout(
            commandBuffer,
            m_voxelGiOccupancyImage,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        VkClearColorValue occupancyClearColor{};
        occupancyClearColor.float32[0] = 0.0f;
        occupancyClearColor.float32[1] = 0.0f;
        occupancyClearColor.float32[2] = 0.0f;
        occupancyClearColor.float32[3] = 0.0f;
        VkImageSubresourceRange occupancyClearRange{};
        occupancyClearRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        occupancyClearRange.baseMipLevel = 0;
        occupancyClearRange.levelCount = 1;
        occupancyClearRange.baseArrayLayer = 0;
        occupancyClearRange.layerCount = 1;
        vkCmdClearColorImage(
            commandBuffer,
            m_voxelGiOccupancyImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &occupancyClearColor,
            1,
            &occupancyClearRange
        );
        transitionImageLayout(
            commandBuffer,
            m_voxelGiOccupancyImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        m_voxelGiOccupancyInitialized = true;
        voxelGiOccupancyClearThisFrame = true;
    } else if (m_voxelGiOccupancyImage != VK_NULL_HANDLE &&
               voxelGiOccupancyDispatchZ > 0u &&
               !voxelGiOccupancyClearThisFrame) {
        transitionImageLayout(
            commandBuffer,
            m_voxelGiOccupancyImage,
            m_voxelGiOccupancyInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            m_voxelGiOccupancyInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
            m_voxelGiOccupancyInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }

    const bool legacySceneRenderingEnabled = legacyVoxelRenderingEnabled;
    const FrameInstanceDrawData frameInstanceDrawData = legacySceneRenderingEnabled
        ? prepareFrameInstanceDrawData(simulation, simulationAlpha)
        : FrameInstanceDrawData{};
    const uint32_t pipeInstanceCount = frameInstanceDrawData.pipeInstanceCount;
    const auto& pipeInstanceSliceOpt = frameInstanceDrawData.pipeInstanceSliceOpt;
    const uint32_t transportInstanceCount = frameInstanceDrawData.transportInstanceCount;
    const auto& transportInstanceSliceOpt = frameInstanceDrawData.transportInstanceSliceOpt;
    const uint32_t beltCargoInstanceCount = frameInstanceDrawData.beltCargoInstanceCount;
    const auto& beltCargoInstanceSliceOpt = frameInstanceDrawData.beltCargoInstanceSliceOpt;
    const std::vector<ReadyMagicaDraw>& readyMagicaDraws = frameInstanceDrawData.readyMagicaDraws;

    const FrameChunkDrawData frameChunkDrawData = legacySceneRenderingEnabled
        ? prepareFrameChunkDrawData(
            chunkGrid.chunks(),
            visibleChunkIndices,
            lightViewProjMatrices,
            cameraChunkX,
            cameraChunkY,
            cameraChunkZ)
        : FrameChunkDrawData{};
    const auto& chunkInstanceSliceOpt = frameChunkDrawData.chunkInstanceSliceOpt;
    const auto& shadowChunkInstanceSliceOpt = frameChunkDrawData.shadowChunkInstanceSliceOpt;
    const VkBuffer chunkInstanceBuffer = frameChunkDrawData.chunkInstanceBuffer;
    const VkBuffer shadowChunkInstanceBuffer = frameChunkDrawData.shadowChunkInstanceBuffer;
    const VkBuffer chunkVertexBuffer = legacySceneRenderingEnabled
        ? m_bufferAllocator.getBuffer(m_chunkVertexBufferHandle)
        : VK_NULL_HANDLE;
    const VkBuffer chunkIndexBuffer = legacySceneRenderingEnabled
        ? m_bufferAllocator.getBuffer(m_chunkIndexBufferHandle)
        : VK_NULL_HANDLE;
    const VkBuffer importedVertexBuffer = m_bufferAllocator.getBuffer(m_importedVertexBufferHandle);
    const VkBuffer importedIndexBuffer = m_bufferAllocator.getBuffer(m_importedIndexBufferHandle);
    std::vector<ImportedMeshDraw> importedActorMeshDraws;
    std::optional<FrameArenaSlice> importedActorVertexSliceOpt = std::nullopt;
    std::optional<FrameArenaSlice> importedActorIndexSliceOpt = std::nullopt;
    VkBuffer importedActorVertexBuffer = VK_NULL_HANDLE;
    VkBuffer importedActorIndexBuffer = VK_NULL_HANDLE;
    if (renderingImportedActors) {
        std::vector<ImportedMeshVertex> actorVertices;
        actorVertices.reserve(importedActors->vertices.size());
        for (const odai::importer::ImportedScenePackedVertex& srcVertex : importedActors->vertices) {
            ImportedMeshVertex dstVertex{};
            std::memcpy(dstVertex.position, srcVertex.position, sizeof(dstVertex.position));
            std::memcpy(dstVertex.normal, srcVertex.normal, sizeof(dstVertex.normal));
            std::memcpy(dstVertex.color, srcVertex.color, sizeof(dstVertex.color));
            std::memcpy(dstVertex.uv, srcVertex.uv, sizeof(dstVertex.uv));
            dstVertex.flags = srcVertex.flags;
            if (srcVertex.textureIndex < m_importedTextureSlots.size()) {
                dstVertex.textureIndex = m_importedTextureSlots[srcVertex.textureIndex];
            } else {
                dstVertex.textureIndex = std::numeric_limits<std::uint32_t>::max();
            }
            actorVertices.push_back(dstVertex);
        }

        const VkDeviceSize actorVertexBytes =
            static_cast<VkDeviceSize>(actorVertices.size() * sizeof(ImportedMeshVertex));
        const VkDeviceSize actorIndexBytes =
            static_cast<VkDeviceSize>(importedActors->indices.size() * sizeof(std::uint32_t));
        importedActorVertexSliceOpt = m_frameArena.allocateUpload(
            actorVertexBytes,
            static_cast<VkDeviceSize>(alignof(ImportedMeshVertex)),
            FrameArenaUploadKind::Unknown);
        importedActorIndexSliceOpt = m_frameArena.allocateUpload(
            actorIndexBytes,
            static_cast<VkDeviceSize>(alignof(std::uint32_t)),
            FrameArenaUploadKind::Unknown);
        if (importedActorVertexSliceOpt.has_value() &&
            importedActorIndexSliceOpt.has_value() &&
            importedActorVertexSliceOpt->mapped != nullptr &&
            importedActorIndexSliceOpt->mapped != nullptr) {
            std::memcpy(importedActorVertexSliceOpt->mapped, actorVertices.data(), actorVertexBytes);
            std::memcpy(importedActorIndexSliceOpt->mapped, importedActors->indices.data(), actorIndexBytes);
            importedActorVertexBuffer = m_bufferAllocator.getBuffer(importedActorVertexSliceOpt->buffer);
            importedActorIndexBuffer = m_bufferAllocator.getBuffer(importedActorIndexSliceOpt->buffer);
            importedActorMeshDraws.reserve(importedActors->draws.size());
            for (const odai::importer::ImportedScenePackedDraw& srcDraw : importedActors->draws) {
                if (srcDraw.indexCount == 0u ||
                    srcDraw.firstIndex >= importedActors->indices.size()) {
                    continue;
                }
                ImportedMeshDraw draw{};
                draw.firstIndex = srcDraw.firstIndex;
                draw.indexCount = std::min<std::uint32_t>(
                    srcDraw.indexCount,
                    static_cast<std::uint32_t>(importedActors->indices.size() - srcDraw.firstIndex));
                importedActorMeshDraws.push_back(draw);
            }
        }
    }
    std::span<const ImportedMeshDraw> importedMeshDrawsForFrame(
        m_importedMeshDraws.data(),
        m_importedMeshDraws.size());
    std::uint32_t importedTerrainDrawCountForFrame = m_importedTerrainDrawCount;
    const bool importedPageCullingEnabled = !m_importedPageDrawRanges.empty();
    auto importedPageIntersectsClip = [](
                                          const ImportedScenePageDrawRange& pageRange,
                                          const odai::math::Matrix4& clipMatrix,
                                          float clipMargin
                                      ) -> bool {
        if (pageRange.drawCount == 0u) {
            return false;
        }
        if (pageRange.boundsMin[0] > pageRange.boundsMax[0] ||
            pageRange.boundsMin[1] > pageRange.boundsMax[1] ||
            pageRange.boundsMin[2] > pageRange.boundsMax[2]) {
            return true;
        }

        std::array<odai::math::Vector3, 8> corners = {
            odai::math::Vector3{pageRange.boundsMin[0], pageRange.boundsMin[1], pageRange.boundsMin[2]},
            odai::math::Vector3{pageRange.boundsMax[0], pageRange.boundsMin[1], pageRange.boundsMin[2]},
            odai::math::Vector3{pageRange.boundsMin[0], pageRange.boundsMax[1], pageRange.boundsMin[2]},
            odai::math::Vector3{pageRange.boundsMax[0], pageRange.boundsMax[1], pageRange.boundsMin[2]},
            odai::math::Vector3{pageRange.boundsMin[0], pageRange.boundsMin[1], pageRange.boundsMax[2]},
            odai::math::Vector3{pageRange.boundsMax[0], pageRange.boundsMin[1], pageRange.boundsMax[2]},
            odai::math::Vector3{pageRange.boundsMin[0], pageRange.boundsMax[1], pageRange.boundsMax[2]},
            odai::math::Vector3{pageRange.boundsMax[0], pageRange.boundsMax[1], pageRange.boundsMax[2]},
        };

        float ndcMinX = std::numeric_limits<float>::max();
        float ndcMinY = std::numeric_limits<float>::max();
        float ndcMinZ = std::numeric_limits<float>::max();
        float ndcMaxX = std::numeric_limits<float>::lowest();
        float ndcMaxY = std::numeric_limits<float>::lowest();
        float ndcMaxZ = std::numeric_limits<float>::lowest();
        for (const odai::math::Vector3& corner : corners) {
            const odai::math::Vector3 clip = odai::math::transformPoint(clipMatrix, corner);
            ndcMinX = std::min(ndcMinX, clip.x);
            ndcMinY = std::min(ndcMinY, clip.y);
            ndcMinZ = std::min(ndcMinZ, clip.z);
            ndcMaxX = std::max(ndcMaxX, clip.x);
            ndcMaxY = std::max(ndcMaxY, clip.y);
            ndcMaxZ = std::max(ndcMaxZ, clip.z);
        }

        return !(ndcMaxX < (-1.0f - clipMargin) ||
                 ndcMinX > (1.0f + clipMargin) ||
                 ndcMaxY < (-1.0f - clipMargin) ||
                 ndcMinY > (1.0f + clipMargin) ||
                 ndcMaxZ < (0.0f - clipMargin) ||
                 ndcMinZ > (1.0f + clipMargin));
    };
    auto buildVisibleImportedDraws = [&](
                                      const odai::math::Matrix4& clipMatrix,
                                      float clipMargin,
                                      std::vector<ImportedMeshDraw>& outDraws
                                  ) -> std::uint32_t {
        outDraws.clear();
        if (outDraws.capacity() < m_importedMeshDraws.size()) {
            outDraws.reserve(m_importedMeshDraws.size());
        }
        m_visibleImportedPageScratch.assign(m_importedPageDrawRanges.size(), 0u);
        for (std::size_t pageIndex = 0; pageIndex < m_importedPageDrawRanges.size(); ++pageIndex) {
            if (importedPageIntersectsClip(m_importedPageDrawRanges[pageIndex], clipMatrix, clipMargin)) {
                m_visibleImportedPageScratch[pageIndex] = 1u;
            }
        }

        auto appendDrawRange = [&](std::uint32_t firstDraw, std::uint32_t drawCount) -> std::uint32_t {
            if (drawCount == 0u || firstDraw >= m_importedMeshDraws.size()) {
                return 0u;
            }
            const std::size_t availableDrawCount = m_importedMeshDraws.size() - firstDraw;
            const std::uint32_t clampedDrawCount =
                std::min<std::uint32_t>(drawCount, static_cast<std::uint32_t>(availableDrawCount));
            outDraws.insert(
                outDraws.end(),
                m_importedMeshDraws.begin() + static_cast<std::ptrdiff_t>(firstDraw),
                m_importedMeshDraws.begin() + static_cast<std::ptrdiff_t>(firstDraw + clampedDrawCount));
            return clampedDrawCount;
        };

        std::uint32_t visibleTerrainDrawCount = 0;
        for (std::size_t pageIndex = 0; pageIndex < m_importedPageDrawRanges.size(); ++pageIndex) {
            if (m_visibleImportedPageScratch[pageIndex] == 0u) {
                continue;
            }
            const ImportedScenePageDrawRange& pageRange = m_importedPageDrawRanges[pageIndex];
            const std::uint32_t terrainDrawCount = std::min(pageRange.terrainDrawCount, pageRange.drawCount);
            visibleTerrainDrawCount += appendDrawRange(pageRange.firstDraw, terrainDrawCount);
        }
        for (std::size_t pageIndex = 0; pageIndex < m_importedPageDrawRanges.size(); ++pageIndex) {
            if (m_visibleImportedPageScratch[pageIndex] == 0u) {
                continue;
            }
            const ImportedScenePageDrawRange& pageRange = m_importedPageDrawRanges[pageIndex];
            const std::uint32_t terrainDrawCount = std::min(pageRange.terrainDrawCount, pageRange.drawCount);
            appendDrawRange(pageRange.firstDraw + terrainDrawCount, pageRange.drawCount - terrainDrawCount);
        }
        return visibleTerrainDrawCount;
    };
    if (importedPageCullingEnabled) {
        constexpr float kImportedMainClipMargin = 0.04f;
        constexpr float kImportedShadowClipMargin = 0.08f;
        m_visibleImportedTerrainDrawCount =
            buildVisibleImportedDraws(mvp, kImportedMainClipMargin, m_visibleImportedMeshDraws);
        importedMeshDrawsForFrame = std::span<const ImportedMeshDraw>(
            m_visibleImportedMeshDraws.data(),
            m_visibleImportedMeshDraws.size());
        importedTerrainDrawCountForFrame = m_visibleImportedTerrainDrawCount;
        for (std::uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
            m_visibleImportedShadowTerrainDrawCounts[cascadeIndex] = buildVisibleImportedDraws(
                lightViewProjMatrices[cascadeIndex],
                kImportedShadowClipMargin,
                m_visibleImportedShadowMeshDraws[cascadeIndex]);
        }
    }
    const bool canDrawMagica =
        legacySceneRenderingEnabled && !readyMagicaDraws.empty() && m_magicaPipeline != VK_NULL_HANDLE;
    auto countDrawCalls = [&](std::uint32_t& passCounter, std::uint32_t drawCount) {
        passCounter += drawCount;
        m_debugDrawCallsTotal += drawCount;
    };
    FrameExecutionContext frameExecutionContext{};
    frameExecutionContext.commandBuffer = commandBuffer;
    frameExecutionContext.gpuTimestampQueryPool = gpuTimestampQueryPool;
    frameExecutionContext.frameOrderValidator = &coreFramePassOrderValidator;
    frameExecutionContext.frameGraphPlan = &(*coreFrameGraphPlan);
    frameExecutionContext.boundDescriptorSets = &boundDescriptorSets;
    frameExecutionContext.mvpDynamicOffset = mvpDynamicOffset;

    ShadowPassInputs shadowPassInputs{};
    shadowPassInputs.frameChunkDrawData = &frameChunkDrawData;
    shadowPassInputs.chunkInstanceSliceOpt = &chunkInstanceSliceOpt;
    shadowPassInputs.shadowChunkInstanceSliceOpt = &shadowChunkInstanceSliceOpt;
    shadowPassInputs.chunkInstanceBuffer = chunkInstanceBuffer;
    shadowPassInputs.shadowChunkInstanceBuffer = shadowChunkInstanceBuffer;
    shadowPassInputs.chunkVertexBuffer = chunkVertexBuffer;
    shadowPassInputs.chunkIndexBuffer = chunkIndexBuffer;
    shadowPassInputs.canDrawMagica = canDrawMagica;
    shadowPassInputs.readyMagicaDraws = readyMagicaDraws;
    shadowPassInputs.importedVertexBuffer = importedVertexBuffer;
    shadowPassInputs.importedIndexBuffer = importedIndexBuffer;
    shadowPassInputs.importedMeshDraws = m_importedMeshDraws;
    shadowPassInputs.importedTerrainDrawCount = m_importedTerrainDrawCount;
    shadowPassInputs.importedActorVertexBuffer = importedActorVertexBuffer;
    shadowPassInputs.importedActorVertexOffset =
        importedActorVertexSliceOpt.has_value() ? importedActorVertexSliceOpt->offset : 0u;
    shadowPassInputs.importedActorIndexBuffer = importedActorIndexBuffer;
    shadowPassInputs.importedActorIndexOffset =
        importedActorIndexSliceOpt.has_value() ? importedActorIndexSliceOpt->offset : 0u;
    shadowPassInputs.importedActorMeshDraws = importedActorMeshDraws;
    shadowPassInputs.importedPageCullingEnabled = importedPageCullingEnabled;
    if (importedPageCullingEnabled) {
        for (std::uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
            shadowPassInputs.importedMeshDrawsByCascade[cascadeIndex] = std::span<const ImportedMeshDraw>(
                m_visibleImportedShadowMeshDraws[cascadeIndex].data(),
                m_visibleImportedShadowMeshDraws[cascadeIndex].size());
            shadowPassInputs.importedTerrainDrawCountsByCascade[cascadeIndex] =
                m_visibleImportedShadowTerrainDrawCounts[cascadeIndex];
        }
    }
    shadowPassInputs.pipeInstanceCount = pipeInstanceCount;
    shadowPassInputs.pipeInstanceSliceOpt = &pipeInstanceSliceOpt;
    shadowPassInputs.transportInstanceCount = transportInstanceCount;
    shadowPassInputs.transportInstanceSliceOpt = &transportInstanceSliceOpt;
    shadowPassInputs.beltCargoInstanceCount = beltCargoInstanceCount;
    shadowPassInputs.beltCargoInstanceSliceOpt = &beltCargoInstanceSliceOpt;
    recordShadowAtlasPass(frameExecutionContext, shadowPassInputs);

    bool wroteVoxelGiTimestamps = false;
    m_voxelGiRtSurfaceActiveThisFrame = false;
    m_voxelGiRestirActiveThisFrame = false;
    bool wroteAutoExposureTimestamps = false;
    bool wroteSunShaftTimestamps = false;
    const bool voxelGiSurfaceFacesReady = std::all_of(
        m_voxelGiSurfaceFaceImages.begin(),
        m_voxelGiSurfaceFaceImages.end(),
        [](VkImage image) { return image != VK_NULL_HANDLE; }
    );
    const bool voxelGiCanRunCompute =
        !voxelGiNeedsOccupancyUpload ||
        voxelGiOccupancyDispatchZ > 0u ||
        voxelGiOccupancyClearThisFrame;
    if (m_voxelGiComputeAvailable &&
        m_voxelGiOccupancyPipeline != VK_NULL_HANDLE &&
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
        voxelGiCanRunCompute) {
        wroteVoxelGiTimestamps = true;
        beginDebugLabel(commandBuffer, "Pass: Voxel GI", 0.38f, 0.28f, 0.12f, 1.0f);

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

        recordVoxelGiDispatchSequence(
            commandBuffer,
            mvpDynamicOffset,
            gpuTimestampQueryPool,
            voxelGiOccupancyDispatchZ
        );
        if (voxelGiOccupancyDispatchZ > 0u) {
            m_voxelGiOccupancyInitialized = true;
        }
        if (m_voxelGiOccupancyFullRebuildInProgress || !m_voxelGiDirtyChunkIndices.empty()) {
            m_voxelGiWorldDirty = true;
        }
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
        writeGpuTimestampTop(kGpuTimestampQueryGiOccupancyStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiOccupancyEnd);
        writeGpuTimestampTop(kGpuTimestampQueryGiSurfaceStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiSurfaceEnd);
        writeGpuTimestampTop(kGpuTimestampQueryGiSurfaceCandidateStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiSurfaceCandidateEnd);
        writeGpuTimestampTop(kGpuTimestampQueryGiSurfaceTemporalStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiSurfaceTemporalEnd);
        writeGpuTimestampTop(kGpuTimestampQueryGiSurfaceSpatialStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiSurfaceSpatialEnd);
        writeGpuTimestampTop(kGpuTimestampQueryGiSurfaceResolveStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiSurfaceResolveEnd);
        writeGpuTimestampTop(kGpuTimestampQueryGiInjectStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiInjectEnd);
        writeGpuTimestampTop(kGpuTimestampQueryGiPropagateStart);
        writeGpuTimestampBottom(kGpuTimestampQueryGiPropagateEnd);
    }
    const bool voxelGiRtSurfaceRequested = m_voxelGiDebugSettings.surfaceMode != VoxelGiSurfaceMode::Legacy;
    const bool voxelGiRtSurfaceCanRun =
        m_voxelGiComputeAvailable &&
        voxelGiRtSurfaceRequested &&
        m_rayTracingRuntimeEnabled &&
        m_voxelGiSurfacePipelineRt != VK_NULL_HANDLE &&
        m_rtTlas.handle != VK_NULL_HANDLE;
    const bool voxelGiRestirRequested = m_voxelGiDebugSettings.surfaceMode == VoxelGiSurfaceMode::RestirSurface;
    const bool voxelGiRestirCanRun =
        m_voxelGiComputeAvailable &&
        voxelGiRestirRequested &&
        m_rayTracingRuntimeEnabled &&
        m_voxelGiRestirReady &&
        m_rtTlas.handle != VK_NULL_HANDLE;
    const VoxelGiSurfaceMode activeVoxelGiSurfaceMode =
        voxelGiRestirCanRun ? VoxelGiSurfaceMode::RestirSurface
        : (voxelGiRtSurfaceCanRun ? VoxelGiSurfaceMode::RtSurface : VoxelGiSurfaceMode::Legacy);
    if (!wroteVoxelGiTimestamps) {
        m_voxelGiRtSurfaceActiveThisFrame = false;
        m_voxelGiRestirActiveThisFrame = false;
    }
    const char* voxelGiSurfaceFallbackReason = voxelGiSurfaceFallbackReasonName(
        m_voxelGiDebugSettings.surfaceMode,
        m_voxelGiComputeAvailable,
        voxelGiRtSurfaceCanRun,
        voxelGiRestirCanRun,
        m_rtTlas.handle != VK_NULL_HANDLE
    );
    if (!m_voxelGiSurfaceLastLoggedValid ||
        m_voxelGiSurfaceLastLoggedRequestedRt != voxelGiRtSurfaceRequested ||
        m_voxelGiSurfaceLastLoggedRtReady != voxelGiRtSurfaceCanRun ||
        m_voxelGiSurfaceLastLoggedRequestedRestir != voxelGiRestirRequested ||
        m_voxelGiSurfaceLastLoggedRestirReady != voxelGiRestirCanRun) {
        VOX_LOGI("render") << "voxel GI surface mode: requested="
                           << voxelGiSurfaceModeName(m_voxelGiDebugSettings.surfaceMode)
                           << ", active=" << voxelGiSurfaceModeName(activeVoxelGiSurfaceMode)
                           << ", fallback=" << (activeVoxelGiSurfaceMode != m_voxelGiDebugSettings.surfaceMode ? "yes" : "no")
                           << ", reason=" << voxelGiSurfaceFallbackReason
                           << ", compute=" << (m_voxelGiComputeAvailable ? "yes" : "no")
                           << ", rtReady=" << (voxelGiRtSurfaceCanRun ? "yes" : "no")
                           << ", restirReady=" << (voxelGiRestirCanRun ? "yes" : "no")
                           << ", tlas=" << (m_rtTlas.handle != VK_NULL_HANDLE ? "yes" : "no")
                           << ", rtPipeline=" << (m_voxelGiSurfacePipelineRt != VK_NULL_HANDLE ? "yes" : "no")
                           << ", restirPipelines="
                           << ((m_voxelGiRestirCandidatePipeline != VK_NULL_HANDLE &&
                                m_voxelGiRestirTemporalPipeline != VK_NULL_HANDLE &&
                                m_voxelGiRestirSpatialPipeline != VK_NULL_HANDLE &&
                                m_voxelGiRestirResolvePipeline != VK_NULL_HANDLE) ? "yes" : "no");
        m_voxelGiSurfaceLastLoggedRequestedRt = voxelGiRtSurfaceRequested;
        m_voxelGiSurfaceLastLoggedRequestedRestir = voxelGiRestirRequested;
        m_voxelGiSurfaceLastLoggedRtReady = voxelGiRtSurfaceCanRun;
        m_voxelGiSurfaceLastLoggedRestirReady = voxelGiRestirCanRun;
        m_voxelGiSurfaceLastLoggedValid = true;
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
    frameExecutionContext.aoFrameIndex = aoFrameIndex;
    frameExecutionContext.imageIndex = imageIndex;
    frameExecutionContext.aoExtent = aoExtent;
    frameExecutionContext.aoViewport = aoViewport;
    frameExecutionContext.aoScissor = aoScissor;

    PrepassInputs prepassInputs{};
    prepassInputs.frameChunkDrawData = &frameChunkDrawData;
    prepassInputs.chunkInstanceSliceOpt = &chunkInstanceSliceOpt;
    prepassInputs.chunkInstanceBuffer = chunkInstanceBuffer;
    prepassInputs.chunkVertexBuffer = chunkVertexBuffer;
    prepassInputs.chunkIndexBuffer = chunkIndexBuffer;
    prepassInputs.canDrawMagica = canDrawMagica;
    prepassInputs.readyMagicaDraws = readyMagicaDraws;
    prepassInputs.importedVertexBuffer = importedVertexBuffer;
    prepassInputs.importedIndexBuffer = importedIndexBuffer;
    prepassInputs.importedMeshDraws = importedMeshDrawsForFrame;
    prepassInputs.importedTerrainDrawCount = importedTerrainDrawCountForFrame;
    prepassInputs.importedActorVertexBuffer = importedActorVertexBuffer;
    prepassInputs.importedActorVertexOffset =
        importedActorVertexSliceOpt.has_value() ? importedActorVertexSliceOpt->offset : 0u;
    prepassInputs.importedActorIndexBuffer = importedActorIndexBuffer;
    prepassInputs.importedActorIndexOffset =
        importedActorIndexSliceOpt.has_value() ? importedActorIndexSliceOpt->offset : 0u;
    prepassInputs.importedActorMeshDraws = importedActorMeshDraws;
    prepassInputs.pipeInstanceCount = pipeInstanceCount;
    prepassInputs.pipeInstanceSliceOpt = &pipeInstanceSliceOpt;
    prepassInputs.transportInstanceCount = transportInstanceCount;
    prepassInputs.transportInstanceSliceOpt = &transportInstanceSliceOpt;
    prepassInputs.beltCargoInstanceCount = beltCargoInstanceCount;
    prepassInputs.beltCargoInstanceSliceOpt = &beltCargoInstanceSliceOpt;
    recordNormalDepthPrepass(frameExecutionContext, prepassInputs);

    recordSsaoPasses(frameExecutionContext);

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
    frameExecutionContext.viewport = viewport;
    frameExecutionContext.scissor = scissor;

    MainPassInputs mainPassInputs{};
    mainPassInputs.frameChunkDrawData = &frameChunkDrawData;
    mainPassInputs.chunkInstanceSliceOpt = &chunkInstanceSliceOpt;
    mainPassInputs.chunkInstanceBuffer = chunkInstanceBuffer;
    mainPassInputs.chunkVertexBuffer = chunkVertexBuffer;
    mainPassInputs.chunkIndexBuffer = chunkIndexBuffer;
    mainPassInputs.canDrawMagica = canDrawMagica;
    mainPassInputs.readyMagicaDraws = readyMagicaDraws;
    mainPassInputs.importedVertexBuffer = importedVertexBuffer;
    mainPassInputs.importedIndexBuffer = importedIndexBuffer;
    mainPassInputs.importedMeshDraws = importedMeshDrawsForFrame;
    mainPassInputs.importedTerrainDrawCount = importedTerrainDrawCountForFrame;
    mainPassInputs.importedActorVertexBuffer = importedActorVertexBuffer;
    mainPassInputs.importedActorVertexOffset =
        importedActorVertexSliceOpt.has_value() ? importedActorVertexSliceOpt->offset : 0u;
    mainPassInputs.importedActorIndexBuffer = importedActorIndexBuffer;
    mainPassInputs.importedActorIndexOffset =
        importedActorIndexSliceOpt.has_value() ? importedActorIndexSliceOpt->offset : 0u;
    mainPassInputs.importedActorMeshDraws = importedActorMeshDraws;
    mainPassInputs.pipeInstanceCount = pipeInstanceCount;
    mainPassInputs.pipeInstanceSliceOpt = &pipeInstanceSliceOpt;
    mainPassInputs.transportInstanceCount = transportInstanceCount;
    mainPassInputs.transportInstanceSliceOpt = &transportInstanceSliceOpt;
    mainPassInputs.beltCargoInstanceCount = beltCargoInstanceCount;
    mainPassInputs.beltCargoInstanceSliceOpt = &beltCargoInstanceSliceOpt;
    mainPassInputs.preview = &preview;
    recordMainScenePass(frameExecutionContext, mainPassInputs);

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
    const bool shouldRunAutoExposureThisFrame =
        autoExposureEnabled &&
        autoExposurePassResourcesReady &&
        (m_autoExposureUpdateFrameIndex % autoExposureUpdateIntervalFrames) == 0u;
    if (shouldRunAutoExposureThisFrame) {
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

        // Use a smaller source mip for histogram construction to keep auto-exposure cheaper than SSAO.
        constexpr uint32_t kAutoExposureTargetDownsampleMip = 4u;
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
            m_autoExposureUpdateFrameIndex = 0u;
            m_autoExposureHistoryValid = false;
        }
    }
    if (autoExposureEnabled && autoExposurePassResourcesReady) {
        ++m_autoExposureUpdateFrameIndex;
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
    coreFramePassOrderValidator.markPassEntered(coreFrameGraphPlan->post, "post");
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
    if (m_uiRenderer.ready() && !m_uiDrawData.commands.empty()) {
        beginDebugLabel(commandBuffer, "Pass: UI", 0.85f, 0.72f, 0.44f);
        m_uiRenderer.record(commandBuffer, 0, m_frameArena, m_uiDrawData, m_swapchainExtent);
        endDebugLabel(commandBuffer);
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
    std::array<VkPipelineStageFlags2, 2> waitStages{};
    std::array<uint64_t, 2> waitSemaphoreValues{};
    uint32_t waitSemaphoreCount = 0;

    waitSemaphores[waitSemaphoreCount] = frame.imageAvailable;
    waitStages[waitSemaphoreCount] = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    waitSemaphoreValues[waitSemaphoreCount] = 0;
    ++waitSemaphoreCount;

    if (m_pendingTransferTimelineValue > 0) {
        waitSemaphores[waitSemaphoreCount] = m_renderTimelineSemaphore;
        waitStages[waitSemaphoreCount] = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
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
    std::array<VkSemaphoreSubmitInfo, 2> waitSemaphoreInfos{};
    for (uint32_t i = 0; i < waitSemaphoreCount; ++i) {
        waitSemaphoreInfos[i].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        waitSemaphoreInfos[i].semaphore = waitSemaphores[i];
        waitSemaphoreInfos[i].value = waitSemaphoreValues[i];
        waitSemaphoreInfos[i].stageMask = waitStages[i];
        waitSemaphoreInfos[i].deviceIndex = 0;
    }
    std::array<VkSemaphoreSubmitInfo, 2> signalSemaphoreInfos{};
    for (uint32_t i = 0; i < signalSemaphores.size(); ++i) {
        signalSemaphoreInfos[i].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signalSemaphoreInfos[i].semaphore = signalSemaphores[i];
        signalSemaphoreInfos[i].value = signalSemaphoreValues[i];
        signalSemaphoreInfos[i].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        signalSemaphoreInfos[i].deviceIndex = 0;
    }
    VkCommandBufferSubmitInfo commandBufferSubmitInfo{};
    commandBufferSubmitInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferSubmitInfo.commandBuffer = commandBuffer;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = waitSemaphoreCount;
    submitInfo.pWaitSemaphoreInfos = waitSemaphoreInfos.data();
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
    submitInfo.signalSemaphoreInfoCount = static_cast<uint32_t>(signalSemaphoreInfos.size());
    submitInfo.pSignalSemaphoreInfos = signalSemaphoreInfos.data();

    if (vkQueueSubmit2(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        VOX_LOGE("render") << "vkQueueSubmit2 failed\n";
        return;
    }
    if (gpuTimestampQueryPool != VK_NULL_HANDLE) {
        m_gpuTimestampQuerySubmitted[m_currentFrame] = true;
    }
    m_frameTimelineValues[m_currentFrame] = signalTimelineValue;
    m_swapchainImageTimelineValues[imageIndex] = signalTimelineValue;
    m_lastGraphicsTimelineValue = signalTimelineValue;
    m_framePacingStats.queuedFrames = countQueuedFrames(completedTimelineValue());

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
        const auto nowNs = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()
        );
        std::uint64_t desiredPresentTimeNs = 0;
        if (m_framePacingSettings.mode == FramePacingMode::Scheduled) {
            desiredPresentTimeNs = computeDesiredPresentTimeNs(nowNs);
        }
        presentTime.desiredPresentTime = desiredPresentTimeNs;
        presentTimesInfo.sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE;
        presentTimesInfo.swapchainCount = 1;
        presentTimesInfo.pTimes = &presentTime;
        presentInfo.pNext = &presentTimesInfo;
        m_lastSubmittedDisplayTimingPresentId = submittedPresentId;
        m_lastScheduledDesiredPresentTimeNs = desiredPresentTimeNs;
        if (desiredPresentTimeNs > 0) {
            m_displayTimingDesiredPresentTimesNs[submittedPresentId] = desiredPresentTimeNs;
            m_framePacingStats.desiredLeadTimeMs = static_cast<float>((desiredPresentTimeNs - nowNs) * 1.0e-6);
        } else {
            m_framePacingStats.desiredLeadTimeMs = 0.0f;
        }
        m_framePacingStats.desiredPresentTimeNs = desiredPresentTimeNs;
    } else {
        m_lastSubmittedDisplayTimingPresentId = 0;
        m_lastScheduledDesiredPresentTimeNs = 0;
    }

    const auto presentStartTime = std::chrono::steady_clock::now();
    const VkResult presentResult = vkQueuePresentKHR(m_graphicsQueue, &presentInfo);
    const float presentWaitMs = static_cast<float>(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - presentStartTime).count()
    );
    cpuWaitMs += presentWaitMs;
    cpuWaitPresentMs += presentWaitMs;
    if (useDisplayTiming && (presentResult == VK_SUCCESS || presentResult == VK_SUBOPTIMAL_KHR)) {
        updateDisplayTimingStats();
    }
    m_framePacingStats.refreshMs = m_debugDisplayRefreshMs;
    if (m_displayRefreshDurationNs > 0) {
        m_framePacingStats.targetPresentIntervalMs = static_cast<float>(
            (m_displayRefreshDurationNs * m_framePacingStats.cadenceDivisor) * 1.0e-6
        );
    }
    m_framePacingStats.presentMarginMs = m_debugDisplayPresentMarginMs;
    m_framePacingStats.actualPresentDeltaMs = m_debugDisplayActualEarliestDeltaMs;
    m_framePacingStats.presentScheduleErrorMs = m_debugDisplayScheduleErrorMs;
    m_framePacingStats.latePresentCount = m_debugLatePresentCount;
    m_framePacingStats.cpuWaitFrameSlotMs = cpuWaitFrameSlotMs;
    m_framePacingStats.cpuWaitAcquireMs = cpuWaitAcquireMs;
    m_framePacingStats.cpuWaitPresentMs = cpuWaitPresentMs;
    m_framePacingStats.cpuWaitTransferMs = cpuWaitTransferMs;
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
    updateFrameTimingPercentiles();

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


} // namespace odai::render
