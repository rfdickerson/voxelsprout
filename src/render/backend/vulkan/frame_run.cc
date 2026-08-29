#include "render/backend/vulkan/renderer_backend.h"
#include "render/upscale/upscale_contract.h"

#include <GLFW/glfw3.h>
#include "core/grid3.h"
#include "core/log.h"
#include "math/math.h"
#include "world/chunk_mesher.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
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

// A per-channel multiplier for the renderer's own sun or ambient colour, from
// a WTHR light channel. Returns (1,1,1) at weight 0, so a game that publishes
// no weather lighting is unaffected down to the bit.
//
// The record's colour is display-referred sRGB. Its DIRECTION in colour space
// is authored intent and is taken whole; its MAGNITUDE is not radiance and is
// only used to derive a bounded gain, because this renderer's exposure is
// calibrated against the intensity computeSunColor derives from the sun's
// altitude and a record must not be able to overrule that.
//
// kReferenceLuminance is a bright overcast-to-clear daytime value (sRGB ~178,
// which is what Skyrim's clear-day Sunlight and Ambient both sit near), so a
// typical day is close to a no-op and the gain reads as a DEPARTURE from it.
// The square root halves the departure in stops and the clamp bounds it: a
// weather can take the world down to about a third and up by a third, which
// covers overcast-to-clear without letting one black out the frame.
odai::math::Vector3 weatherLightTint(const float linearColor[3], float weight) {
    const float clampedWeight = std::clamp(weight, 0.0f, 1.0f);
    if (clampedWeight <= 0.0f) {
        return odai::math::Vector3{1.0f, 1.0f, 1.0f};
    }
    const float luminance = (0.2126f * linearColor[0]) + (0.7152f * linearColor[1]) +
        (0.0722f * linearColor[2]);
    if (luminance <= 1e-5f) {
        return odai::math::Vector3{1.0f, 1.0f, 1.0f};
    }
    constexpr float kReferenceLuminance = 0.44f;  // linear(sRGB 178)
    const float gain =
        std::clamp(std::sqrt(luminance / kReferenceLuminance), 0.35f, 1.35f);
    // Hue as a unit-luminance ratio, so the gain above is the only thing that
    // moves brightness.
    const float hue[3] = {linearColor[0] / luminance, linearColor[1] / luminance,
                          linearColor[2] / luminance};
    return odai::math::Vector3{
        std::lerp(1.0f, hue[0] * gain, clampedWeight),
        std::lerp(1.0f, hue[1] * gain, clampedWeight),
        std::lerp(1.0f, hue[2] * gain, clampedWeight)};
}

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

void RendererBackend::renderFrame(const CameraPose& camera) {
    static const odai::world::ChunkGrid chunkGrid;
    static const VoxelPreview preview;
    constexpr std::span<const std::size_t> visibleChunkIndices{};
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
    // The app's FOV is followed every frame unless the debug slider has been
    // dragged. Tracking the app's value into m_debugCameraFovDegrees while it
    // is NOT overriding is what makes the slider start from whatever the game
    // is currently showing rather than snapping to a stale number the moment
    // it is touched.
    if (!m_debugCameraFovOverride) {
        m_debugCameraFovDegrees = camera.fovDegrees;
    }
    m_debugCameraFovDegrees = std::clamp(m_debugCameraFovDegrees, 20.0f, 120.0f);
    const float activeFovDegrees = m_debugCameraFovDegrees;

    const std::uint32_t currentChunkCount = static_cast<std::uint32_t>(chunkGrid.chunks().size());
    if (currentChunkCount != m_debugChunkCount) {
        m_debugMacroCellStatsDirty = true;
    }
    m_debugChunkCount = currentChunkCount;
    if (m_debugMacroCellStatsDirty) {
        m_debugMacroCellStatsDirty = false;
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
    }
    const std::optional<CoreFrameGraphPlan> coreFrameGraphPlan = buildCoreFrameGraphPlan(&m_frameGraph);
    if (!coreFrameGraphPlan.has_value()) {
        VOX_LOGE("render") << "frame graph has a cycle; refusing to render frame";
        return;
    }
    CoreFrameGraphOrderValidator coreFramePassOrderValidator(*coreFrameGraphPlan);

    collectCompletedBufferReleases();
    uint64_t completedTimelineValueBeforeFrame = completedTimelineValue();
    m_framePacingStats.queuedFrames = countQueuedFrames(completedTimelineValueBeforeFrame);
    if (shouldThrottleFrameStart(completedTimelineValueBeforeFrame)) {
        // Too many frames queued: block on the oldest one instead of dropping
        // this frame and re-spinning the main loop at sleep granularity.
        const uint64_t oldestQueuedValue = oldestQueuedFrameTimelineValue(completedTimelineValueBeforeFrame);
        (void)waitTimelineValue(oldestQueuedValue, frameWaitBudgetNs(), &cpuWaitFrameSlotMs);
        cpuWaitMs += cpuWaitFrameSlotMs;
        m_framePacingStats.cpuWaitFrameSlotMs = cpuWaitFrameSlotMs;
        completedTimelineValueBeforeFrame = completedTimelineValue();
        m_framePacingStats.queuedFrames = countQueuedFrames(completedTimelineValueBeforeFrame);
        if (shouldThrottleFrameStart(completedTimelineValueBeforeFrame)) {
            return;
        }
    }

    FrameResources& frame = m_frames[m_currentFrame];
    if (!isTimelineValueReached(m_frameTimelineValues[m_currentFrame])) {
        // Frame slot not retired yet: block on its timeline value. Only give up
        // (and drop the frame) when the bounded wait times out, which keeps the
        // device-loss/stall diagnostics reachable.
        const float waitStartMs = cpuWaitFrameSlotMs;
        const bool frameSlotReady =
            waitTimelineValue(m_frameTimelineValues[m_currentFrame], frameWaitBudgetNs(), &cpuWaitFrameSlotMs);
        cpuWaitMs += cpuWaitFrameSlotMs - waitStartMs;
        m_framePacingStats.cpuWaitFrameSlotMs = cpuWaitFrameSlotMs;
        if (!frameSlotReady) {
            static double lastStallLogTimeSeconds = 0.0;
            const uint64_t completedValue = completedTimelineValue();
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
            return;
        }
    }
    if (m_frameTimelineValues[m_currentFrame] > 0) {
        if (!readGpuTimestampResults(m_currentFrame)) {
            m_framePacingStats.gpuTimestampsPending = true;
            ++m_framePacingStats.gpuTimestampSkippedFrames;
        }
    }
    // The arena for this frame slot is about to be reset; any transfer still
    // reading staging data from it must finish first. Transfers staged from the
    // other frame slot may keep flying.
    for (TransferCommandSlot& transferSlot : m_transferCommandSlots) {
        if (transferSlot.inFlightTimelineValue == 0 || transferSlot.stagingFrameIndex != m_currentFrame) {
            continue;
        }
        if (!waitTimelineValue(transferSlot.inFlightTimelineValue, frameWaitBudgetNs(), &cpuWaitTransferMs)) {
            cpuWaitMs += cpuWaitTransferMs;
            m_framePacingStats.cpuWaitTransferMs = cpuWaitTransferMs;
            return;
        }
        transferSlot.inFlightTimelineValue = 0;
    }
    cpuWaitMs += cpuWaitTransferMs;
    m_framePacingStats.cpuWaitTransferMs = cpuWaitTransferMs;
    collectCompletedBufferReleases();
    m_frameArena.beginFrame(m_currentFrame);
    // ODAI_ARENA_POISON=<bytes> takes a slice here, before any pass allocates,
    // and writes to it. It carries no data and nothing reads it: the point is
    // purely to consume ring space at this exact moment, so that nothing else
    // lands at ring offset 0.
    //
    // This is the regression probe for a bug that made the renderer look
    // broken by adding a character to it. The camera UBO was published to the
    // descriptor-buffer path at ring offset 0 rather than at its actual slice
    // offset (see updateFrameDescriptorSets), which was survivable only while
    // the camera happened to be the frame's FIRST allocation. The
    // skinned-actor pose upload below is one earlier allocation, and it was
    // enough: every pass then read a garbage view-projection and the whole 3-D
    // frame collapsed to one flat colour with the UI still correct on top.
    //
    // With the fix in place, ANY value here must render identically to an
    // unset one. It is worth keeping precisely because nothing else in the
    // renderer notices the difference -- the failure is invisible until some
    // unrelated feature happens to allocate first.
    static const char* const poisonBytes = std::getenv("ODAI_ARENA_POISON");
    if (poisonBytes != nullptr) {
        const auto requested = static_cast<VkDeviceSize>(std::strtoull(poisonBytes, nullptr, 10));
        if (requested > 0) {
            const std::optional<FrameArenaSlice> poisonSlice =
                m_frameArena.allocateUpload(requested, 256u, FrameArenaUploadKind::Unknown);
            if (poisonSlice.has_value() && poisonSlice->mapped != nullptr) {
                std::memset(poisonSlice->mapped, 0, static_cast<std::size_t>(requested));
            }
        }
    }
    // Must run after beginFrame so this frame's bone-matrix FrameArena slice
    // belongs to the frame index recordSkinningPass will actually record for
    // -- see setSkinnedActorPose/uploadSkinnedActorPoseForFrame's comments.
    uploadSkinnedActorPoseForFrame();

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
    if (!m_externalChunkMeshResults.empty()) {
        std::erase_if(m_externalChunkMeshResults, [&](const odai::world::ChunkMeshResult& result) {
            return std::find_if(
                       chunkGrid.chunks().begin(),
                       chunkGrid.chunks().end(),
                       [&](const odai::world::Chunk& chunk) {
                           return chunk.chunkX() == result.key.x &&
                                  chunk.chunkY() == result.key.y &&
                                  chunk.chunkZ() == result.key.z;
                       }) == chunkGrid.chunks().end();
        });
    }
    m_debugChunkPendingRemeshCount = static_cast<std::uint32_t>(m_pendingChunkRemeshKeys.size());
    m_debugChunkRemeshBatchCount = 0;
    if (m_chunkMeshRebuildRequested || !m_pendingChunkRemeshKeys.empty()) {
        // Avoid CPU stalls when every transfer command slot is still in flight.
        if (hasFreeTransferSlot()) {
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
        !anyTransferSlotInFlight()) {
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
        vkCmdWriteTimestamp2(
            commandBuffer,
            VK_PIPELINE_STAGE_2_NONE,
            gpuTimestampQueryPool,
            queryIndex
        );
    };
    auto writeGpuTimestampBottom = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp2(
            commandBuffer,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
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
        // These two were written but never called from anywhere in the frame
        // path, which left the whole shadow/AO and sun/sky/post tuning surface
        // built and unreachable. Both early-out on m_debugUiVisible exactly as
        // the two above do, so this costs nothing while the debug UI is closed.
        buildShadowDebugUi();
        buildSunDebugUi();
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

    // Projection follows the displayed image, not the internal shading grid.
    // Exact render extents may have a different aspect ratio (for example 1800x1200 into a
    // 3200x1800 display); using the internal ratio here and then stretching it
    // at presentation makes circles oval and changes the horizontal FOV.
    const float aspectRatio = static_cast<float>(m_swapchainExtent.width) /
                              static_cast<float>(m_swapchainExtent.height);
    // ODAI_RENDER_NEAR / ODAI_RENDER_FAR open up the view distance.
    //
    // The whole renderer is reverse-Z (perspectiveVulkanReverseZ, and every
    // pipeline tests GREATER_OR_EQUAL), which is what makes a far plane this
    // aggressive workable at all: depth precision under reverse-Z with a float
    // depth buffer is governed almost entirely by the NEAR plane, so pushing far
    // out costs very little and pulling near in costs a great deal. 0.1 Fallout
    // units is 1.4 mm, far tighter than anything the player can stand next to,
    // and it is the number to raise first if distant geometry z-fights.
    //
    // Shadows do NOT follow this out: the cascades are clamped separately by
    // shadowDistanceLimit (ODAI_SHADOW_DISTANCE) below, so a large far plane
    // does not spread the atlas over terrain nothing can see it on.
    static const float s_nearPlaneOverride = []() {
        const char* env = std::getenv("ODAI_RENDER_NEAR");
        const float value = (env != nullptr) ? static_cast<float>(std::atof(env)) : 0.0f;
        return (value > 0.0f) ? value : 0.0f;
    }();
    static const float s_farPlaneOverride = []() {
        const char* env = std::getenv("ODAI_RENDER_FAR");
        const float value = (env != nullptr) ? static_cast<float>(std::atof(env)) : 0.0f;
        return (value > 0.0f) ? value : 0.0f;
    }();
    const bool renderingImportedScene = !m_importedMeshDraws.empty();
    const bool directionalShadowsForFrame =
        shouldRenderImportedDirectionalShadows(m_importedInteriorLighting);
    bool renderInteriorPointShadowsThisFrame = false;
    std::uint32_t interiorPointShadowLightCount = 0;
    const bool sunShaftsForFrame =
        m_sunShaftsRequested && shouldRenderImportedSky(m_importedInteriorLighting);
    const bool legacyVoxelRenderingEnabled = !renderingImportedScene;
    const bool importedInteriorGiEnabled =
        m_importedSceneInteriorMode &&
        !useAuthoredImportedInteriorLighting(m_importedInteriorLighting) &&
        !m_importedGiTriangles.empty();
    const bool voxelGiSceneEnabled = legacyVoxelRenderingEnabled || importedInteriorGiEnabled;
    const float farPlane = (s_farPlaneOverride > 0.0f)
        ? s_farPlaneOverride
        : (renderingImportedScene ? 50000.0f : 500.0f);
    // NEAR is what buys depth precision under reverse-Z, and it has to be paired
    // with the far plane rather than picked once. 0.1 against a 50000 far plane
    // is a ratio of 500,000:1; the voxel worlds it was chosen for run a 500 far
    // plane, where the same 0.1 is only 5,000:1 and perfectly reasonable.
    //
    // A Fallout unit is about 1.42 cm, so 5 units is 7 cm -- far closer than the
    // player's collision radius ever lets the camera get to a wall, and a 50x
    // precision improvement over 0.1 across the whole depth range. This is the
    // number to raise (not the far plane to lower) when distant coplanar
    // geometry z-fights; ODAI_RENDER_NEAR overrides it.
    const float nearPlane = (s_nearPlaneOverride > 0.0f)
        ? s_nearPlaneOverride
        : (renderingImportedScene ? 5.0f : 0.1f);
    // tanHalfFov feeds shadow cascade sphere sizing. In ortho mode approximate
    // from the view half-height so cascade coverage matches the visible area.
    const float tanHalfFov = camera.orthographic
        ? (camera.orthoHalfHeight / farPlane)
        : std::tan(odai::math::radians(activeFovDegrees) * 0.5f);
    const odai::math::Vector3 eye{camera.x, camera.y, camera.z};
    const CameraFrameDerived cameraFrame = computeCameraFrame(camera);
    const int cameraChunkX = cameraFrame.chunkX;
    const int cameraChunkY = cameraFrame.chunkY;
    const int cameraChunkZ = cameraFrame.chunkZ;
    const odai::math::Vector3 forward = cameraFrame.forward;

    const odai::math::Matrix4 view = lookAt(eye, eye + forward, odai::math::Vector3{0.0f, 1.0f, 0.0f});
    odai::math::Matrix4 projection;
    if (camera.orthographic) {
        const float halfH = camera.orthoHalfHeight;
        const float halfW = halfH * aspectRatio;
        projection = orthographicVulkan(-halfW, halfW, -halfH, halfH, nearPlane, farPlane);
    } else {
        projection = perspectiveVulkan(odai::math::radians(activeFovDegrees), aspectRatio, nearPlane, farPlane);
    }
    // SUB-PIXEL JITTER. Each frame the projection is nudged by a fraction of a
    // pixel, so consecutive frames sample the scene on different sub-pixel
    // grids and TAA's history accumulation becomes supersampling rather than
    // just a temporal average. Without it TAA suppresses texture shimmer (what
    // it was written for) but cannot refine an EDGE: every frame samples the
    // same point inside the same pixel, so the edge is the same staircase each
    // time and averaging identical staircases changes nothing.
    //
    // Halton(2,3) rather than random: it is low-discrepancy, so a short window
    // of frames covers the pixel evenly instead of clumping the way white noise
    // does. 8 phases, which is what the history weight below converges over.
    //
    // Gated on TAA being ON. Jitter with no temporal filter to resolve it is
    // pure per-frame wobble -- strictly worse than not jittering.
    // ODAI_TAA_JITTER=0 disables it while keeping TAA, which is the control for
    // "is this artifact the jitter or the accumulation", and is also what a
    // screenshot diff wants: jitter makes consecutive frames differ BY DESIGN.
    // Snapshot last frame's jitter BEFORE this frame's overwrites it, exactly
    // as prevViewProj is snapshotted below and for the same reason: the two
    // describe the same past frame, and reading either one late silently makes
    // it this frame's, which is not an error anything reports -- it just
    // reprojects half a pixel wrong forever.
    const std::array<float, 2> previousFrameJitterNdc = m_taaJitterNdc;
    float jitterNdcX = 0.0f;
    float jitterNdcY = 0.0f;
    if (m_taaEnabled && m_taaJitterEnabled && !camera.orthographic) {
        // THE JITTER SEQUENCE IS THE UPSCALER'S CONTRACT, NOT THIS PASS'S
        // DETAIL, so it lives in render/upscale/upscale_contract.h alongside
        // the quality->scale table and the mip-bias rule. Halton(2,3), a phase
        // count that scales with the upscale ratio, centred on the pixel --
        // stated once, where a vendor backend swapped in for the built-in one
        // can rely on the host having honoured it.
        const upscale::Extent2D renderExtentForJitter{
            m_renderExtent.width, m_renderExtent.height};
        const upscale::Extent2D displayExtentForJitter{
            m_swapchainExtent.width, m_swapchainExtent.height};
        const std::uint32_t jitterPhaseCount =
            upscale::jitterPhaseCount(renderExtentForJitter, displayExtentForJitter);
        // 1-based: Halton(0) is 0 in every base, so a 0-based sequence spends
        // its first frame not jittering at all.
        const std::uint32_t phase = (m_taaJitterPhase % jitterPhaseCount) + 1u;
        const upscale::JitterOffset jitterNdc =
            upscale::jitterOffsetNdc(phase, renderExtentForJitter);
        jitterNdcX = jitterNdc.x;
        jitterNdcY = jitterNdc.y;
        // clip.w is -view.z here (perspectiveVulkan sets (3,2) = -1), so adding
        // -jitter to the column that multiplies view.z shifts clip.xy by
        // jitter*clip.w -- i.e. a constant offset in NDC at every depth, which
        // is what a sub-pixel sample offset is. Doing it in the matrix rather
        // than by offsetting UVs keeps depth, normals and colour all rasterized
        // on the SAME jittered grid, which is what makes them reprojectable
        // together.
        projection(0, 2) += -jitterNdcX;
        projection(1, 2) += -jitterNdcY;
        ++m_taaJitterPhase;
    }
    m_taaJitterNdc[0] = jitterNdcX;
    m_taaJitterNdc[1] = jitterNdcY;

    const odai::math::Matrix4 mvp = projection * view;
    const odai::math::Matrix4 mvpColumnMajor = transpose(mvp);
    // The planar temporal resolve reconstructs the reflected world position
    // from the raw hardware depth and reprojects it into the preceding camera.
    // Capture the old matrices before rolling them forward, exactly as TAA does.
    m_waterReflectionInvViewProjColumnMajor =
        transpose(odai::math::inverse(mvp));
    m_waterReflectionInvViewColumnMajor =
        transpose(odai::math::inverse(view));
    m_waterReflectionViewColumnMajor = transpose(view);
    m_waterReflectionPrevViewColumnMajor =
        transpose(m_waterReflectionPrevView);
    m_waterReflectionPrevViewProjColumnMajor =
        transpose(m_waterReflectionPrevViewProj);
    const float currentReflectionProjectionX = projection(0, 0);
    const float currentReflectionProjectionY = projection(1, 1);
    if (m_waterReflectionPrevMatricesValid) {
        const odai::math::Vector3 cameraDelta = eye - m_waterReflectionPrevEye;
        float viewRotationDeltaSquared = 0.0f;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                const float delta = view(row, column) -
                    m_waterReflectionPrevView(row, column);
                viewRotationDeltaSquared += delta * delta;
            }
        }
        const bool projectionCut =
            std::abs(currentReflectionProjectionX - m_waterReflectionProjection[0]) > 0.01f ||
            std::abs(currentReflectionProjectionY - m_waterReflectionProjection[1]) > 0.01f;
        if (dot(cameraDelta, cameraDelta) > (256.0f * 256.0f) ||
            viewRotationDeltaSquared > 0.25f || projectionCut) {
            m_waterReflectionHistoryValid = false;
        }
    } else {
        m_waterReflectionHistoryValid = false;
    }
    m_waterReflectionPrevView = view;
    m_waterReflectionPrevViewProj = mvp;
    m_waterReflectionPrevEye = eye;
    m_waterReflectionPrevMatricesValid = true;
    m_waterReflectionProjection[0] = currentReflectionProjectionX;
    m_waterReflectionProjection[1] = currentReflectionProjectionY;
    // TAA reprojection inputs. The column-major copies go to the shader (same
    // transpose convention as the camera UBO); prevViewProj must be read
    // BEFORE it is overwritten with this frame's matrix, or reprojection
    // becomes an identity and TAA silently stops doing anything.
    if (m_taaEnabled) {
        m_taaInvViewColumnMajor = transpose(odai::math::inverse(view));
        m_taaPrevViewProjColumnMajor = transpose(m_taaPrevViewProj);
    }
    const bool taaPrevWasValid = m_taaPrevViewProjValid;
    (void)taaPrevWasValid;
    // The velocity pass keeps its own copies because it must run whether or not
    // TAA is enabled -- the motion vectors feed the upscaler too, and
    // m_taaPrevViewProjColumnMajor above is only maintained under m_taaEnabled.
    // Captured HERE, before m_taaPrevViewProj is overwritten below, for the same
    // reason that one is: read it after and prevViewProj becomes this frame's.
    m_velocityPrevViewProj = m_taaPrevViewProj;
    m_velocityCurrentViewProj = mvp;
    m_velocityPrevJitter[0] = previousFrameJitterNdc[0];
    m_velocityPrevJitter[1] = previousFrameJitterNdc[1];
    m_velocityCurrentJitter[0] = jitterNdcX;
    m_velocityCurrentJitter[1] = jitterNdcY;
    m_velocityPrevValid = m_taaPrevViewProjValid;

    // Rolled with prevViewProj: the two describe the same past frame.
    m_taaPrevJitterNdc = previousFrameJitterNdc;
    m_taaPrevViewProj = mvp;
    m_taaPrevViewProjValid = true;
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

    odai::math::Vector3 sunColor = isNight
        ? odai::math::Vector3{0.0f, 0.0f, 0.0f}
        : computeSunColor(effectiveSkySettings, sunDirection);
    // A WEATHER RECORD LIGHTS THE GROUND, NOT ONLY THE SKY. WTHR's Sunlight and
    // Ambient channels are what make an overcast read as overcast on the
    // terrain rather than only overhead; before this they were read and thrown
    // away, so a storm rendered as a dark sky over a sunlit desert.
    //
    // Only the HUE is taken from the record. Its colours are display-referred
    // sRGB authored for a renderer that used them literally, and this one is
    // HDR with auto-exposure -- decoding one and using it as radiance changes
    // the frame's whole calibration. So the record supplies direction in colour
    // space and a BOUNDED gain from its own luminance, and the renderer keeps
    // the intensity it derived from the sun's altitude.
    const odai::math::Vector3 weatherSunlight = weatherLightTint(
        m_weatherSky.sunlightColor, m_weatherSky.lightingWeight);
    sunColor = odai::math::Vector3{
        sunColor.x * weatherSunlight.x, sunColor.y * weatherSunlight.y,
        sunColor.z * weatherSunlight.z};

    // Shadows stop well before the camera's far plane.
    //
    // farPlane is 50000 for an imported scene, and blending the logarithmic and
    // uniform splits with nearPlane = 0.1 makes the log term vanish (0.1 *
    // (500000)^0.25 is under 2 units against a uniform 12500), so the splits came
    // out effectively uniform: 3752 / 7550 / 12566 / 50000. The last cascade then
    // covered 50000 units in a 1024 map -- about 117 units per texel, or 1.7
    // metres, which resolves nothing while re-rendering the whole region. The
    // shadow pass measured 10.5 ms of an 18.5 ms frame.
    //
    // Two changes: cap the shadow distance, and compute the distribution from a
    // practical near distance instead of the camera's 0.1. Geometry closer than
    // kShadowSplitNear is inside cascade 0 regardless, so using it as the
    // distribution's near end costs nothing and restores the logarithmic term
    // that makes cascade 0 tight.
    // 6000 was tuned for a camera standing ON the ground in Goodsprings. It is
    // far too short for a camera that LOOKS ACROSS a landscape: the flythroughs
    // fly 2000-3600 units up, so nothing on screen is ever within cascade 0's
    // few hundred units and the whole frame sits in the last cascade -- or past
    // it. Rendered as a shadow-visibility view with a low sun, that is a hard
    // horizontal line across the middle of the frame with cast shadows below it
    // and nothing at all above.
    //
    // Note this is invisible at midday. A 37-degree sun over rounded terrain
    // casts almost nothing to begin with, so the A/B that matters is a LOW sun
    // (ODAI_FNV_HOUR=17.5); at hour 16 the same change moves the shadowed
    // fraction of the frame by 0.8 points and looks like it did nothing.
    // 50000 is the imported-scene far plane, so this cap is now effectively
    // "shadow everything the frustum can see" rather than a budget.
    //
    // It was 24000, which still left the far third of a high-camera shot
    // unshadowed: from 7000 units up the distant peaks and the whole horizon
    // ridge came through pure white while the near and mid ground had full
    // shadow detail. Going to the far plane is cheaper than it sounds because
    // the near cascades are almost fully logarithmic now -- cascade 0 widens
    // from 517 units to 886, which is 0.66 -> 1.13 world units per texel and
    // measured 1.06/255 mean difference on a ground-level frame, i.e. not
    // visible. Cascade 3 lands at 64 units per texel, coarse but present, and
    // terrain 30km out subtends little enough that a soft blob is the right
    // answer anyway.
    //
    // Cost, interleaved A/B over the same moving camera: shadow pass
    // 1.03 -> 1.77 ms, with frame time unchanged inside run-to-run noise.
    float shadowDistanceLimit = renderingImportedScene ? 50000.0f : farPlane;
    if (const char* shadowDistanceEnv = std::getenv("ODAI_SHADOW_DISTANCE")) {
        const float requested = static_cast<float>(std::atof(shadowDistanceEnv));
        if (requested > 1.0f) {
            shadowDistanceLimit = requested;
        }
    }
    const float shadowFarPlane = std::min(farPlane, shadowDistanceLimit);
    // The distribution's near end, and it is an ABSOLUTE distance rather than a
    // fraction of the far one.
    //
    // It used to be shadowFarPlane * 0.008, which quietly made the two
    // inseparable: pushing shadows further out dragged the near end out with
    // them and coarsened cascade 0 in proportion. Measured on Vvardenfell,
    // raising ODAI_SHADOW_DISTANCE from 6000 to 45000 took cascade 0 from a
    // 573-unit range at 0.73 units per texel to a 4298-unit range at 5.50 --
    // 7.5x blurrier up close, which is the half of the picture that already
    // looked right. "More shadow distance" should not be a trade against "the
    // shadows near me".
    //
    // The number answers a different question anyway: how close geometry gets
    // to the camera. Anything nearer is inside cascade 0 regardless.
    //
    // 48 is what 6000 * 0.008 evaluated to, so this is EXACTLY the old
    // behaviour at the old default and changes nothing until the distance moves.
    constexpr float kShadowDistributionNear = 48.0f;
    const float shadowSplitNear =
        std::min(std::max(nearPlane, kShadowDistributionNear), shadowFarPlane * 0.25f);
    // How far the splits lean logarithmic (1.0) versus uniform (0.0). The
    // logarithmic term is the one that keeps cascade 0 tight over a long range,
    // so a range this is not tuned for shows up as a fat first cascade: the
    // uniform term at p=0.25 is a QUARTER OF THE WHOLE DISTANCE, and at 0.30
    // weight that alone is 1811 units of a 24000-unit range.
    // 0.70 is right for a 6000-unit range and wrong for a 24000-unit one: the
    // uniform term at p=0.25 is a quarter of the WHOLE distance, so at 0.30
    // weight it alone puts 1811 units into cascade 0 and blurs the near
    // shadows that already looked right. Leaning almost fully logarithmic keeps
    // cascade 0 tight no matter how far the far end goes -- measured, cascade 0
    // at 24000/0.95 is 517 units at 0.66 per texel, against 573 at 0.73 for the
    // old 6000/0.70. Four times the range AND a slightly sharper near cascade.
    //
    // Only the imported-scene path moves; the voxel games keep 0.70 with their
    // own far plane, because nothing here measured them.
    float cascadeLambda = renderingImportedScene ? 0.95f : 0.70f;
    if (const char* lambdaEnv = std::getenv("ODAI_SHADOW_LAMBDA")) {
        cascadeLambda = std::clamp(static_cast<float>(std::atof(lambdaEnv)), 0.0f, 1.0f);
    }
    const float kCascadeLambda = cascadeLambda;
    constexpr float kCascadeSplitQuantization = 0.5f;
    constexpr float kCascadeSplitUpdateThreshold = 0.5f;
    std::array<float, kShadowCascadeCount> cascadeDistances{};
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        if (!directionalShadowsForFrame) {
            break;
        }
        const float p = static_cast<float>(cascadeIndex + 1) / static_cast<float>(kShadowCascadeCount);
        const float logarithmicSplit = shadowSplitNear * std::pow(shadowFarPlane / shadowSplitNear, p);
        const float uniformSplit = shadowSplitNear + ((shadowFarPlane - shadowSplitNear) * p);
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
        split = std::min(split, shadowFarPlane);
        m_shadowCascadeSplits[cascadeIndex] = split;
        cascadeDistances[cascadeIndex] = split;
    }

    // Logged whenever the set CHANGES, not once. Once meant frame 0 -- before a
    // streaming game has uploaded any geometry, so renderingImportedScene is
    // still false and the far plane is the 500-unit fallback. The numbers that
    // printed described a frame nobody was looking at, which is worse than not
    // printing them.
    const bool cascadeSplitsChanged = directionalShadowsForFrame &&
        std::abs(cascadeDistances[kShadowCascadeCount - 1] - m_loggedShadowCascadeFar) >
        std::max(1.0f, m_loggedShadowCascadeFar * 0.02f);
    if (std::getenv("ODAI_GPU_TIMINGS") != nullptr && cascadeSplitsChanged) {
        m_loggedShadowCascadeFar = cascadeDistances[kShadowCascadeCount - 1];
        m_shadowCascadeSplitsLogged = true;
        for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
            const float cascadeFar = cascadeDistances[cascadeIndex];
            const float halfHeight = cascadeFar * tanHalfFov;
            const float halfWidth = halfHeight * aspectRatio;
            const float radius = std::sqrt((cascadeFar * cascadeFar) + (halfWidth * halfWidth) +
                                           (halfHeight * halfHeight));
            VOX_LOGI("render") << "shadow cascade " << cascadeIndex << ": far=" << cascadeFar
                               << " radius=" << radius
                               << " texels=" << kShadowCascadeResolution[cascadeIndex]
                               << " texelSize=" << ((2.0f * radius) / static_cast<float>(kShadowCascadeResolution[cascadeIndex]));
        }
    }

    std::array<odai::math::Matrix4, kShadowCascadeCount> lightViewProjMatrices{};
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        if (!directionalShadowsForFrame) {
            break;
        }
        const float cascadeFar = cascadeDistances[cascadeIndex];
        const float farHalfHeight = cascadeFar * tanHalfFov;
        const float farHalfWidth = farHalfHeight * aspectRatio;

        // Camera-position-only cascades: only translation moves cascade centers; rotation does not.
        odai::math::Vector3 frustumCenter = eye;
        float boundingRadius =
            std::sqrt((cascadeFar * cascadeFar) + (farHalfWidth * farHalfWidth) + (farHalfHeight * farHalfHeight));
        // Orthographic cameras have no perspective frustum to slice: the fov-derived
        // radius above balloons toward farPlane, making shadow texels larger than
        // entire scene objects (a 56-unit city under a 50000-unit farPlane gets
        // ~2-world-unit texels — every building sub-texel, shadows unresolvable).
        // Fit every cascade to the uploaded scene's bounding sphere instead; all
        // cascades match, so chooseShadowCascade() may pick any of them safely.
        // The same ballooning hits narrow-FOV perspective cameras over small
        // scenes (CityBuilder's diorama camera), so scene-fit those too; large
        // imported worlds keep the sliced perspective cascades.
        constexpr float kSceneFitMaxRadius = 200.0f;
        const bool orthoSceneFit =
            m_importedSceneBoundsValid &&
            (camera.orthographic || m_importedSceneBoundsRadius < kSceneFitMaxRadius);
        if (orthoSceneFit) {
            frustumCenter = odai::math::Vector3{
                m_importedSceneBoundsCenter[0],
                m_importedSceneBoundsCenter[1],
                m_importedSceneBoundsCenter[2]};
            boundingRadius = m_importedSceneBoundsRadius * 1.05f;
        }
        boundingRadius = std::max(boundingRadius * 1.04f, orthoSceneFit ? 8.0f : 24.0f);
        boundingRadius = std::ceil(boundingRadius * 16.0f) / 16.0f;
        // Re-latch when the fit the cascade actually wants has moved away from
        // the cached one.
        //
        // The cache exists to stop the ortho box resizing every frame, which
        // makes shadow edges crawl. It used to be invalidated only by
        // projectionParamsChanged, i.e. aspect ratio and FOV -- but farPlane
        // also flips 500 -> 50000 the moment the first imported chunk becomes
        // resident, which moves every cascade split and therefore every radius.
        // A streaming game latched its radii on frame 0 with no geometry loaded
        // and kept them forever: cascade 0 wanted 1048 units and was pinned at
        // 87.9, so the shader routed every fragment inside 573 units of view
        // depth to a box only 88 units across, and the rest of that range
        // sampled outside the map and came back fully lit. Fallout had no
        // shadows past roughly one fence post.
        //
        // Relative tolerance rather than exact equality: splits are already
        // quantized to 0.5 with a 0.5 update threshold, so in a steady state
        // this compares equal and the cache still does its job.
        constexpr float kCascadeRadiusRelatchTolerance = 0.01f;
        const float cachedRadius = m_shadowStableCascadeRadii[cascadeIndex];
        if (cachedRadius <= 0.0f ||
            std::abs(boundingRadius - cachedRadius) > cachedRadius * kCascadeRadiusRelatchTolerance) {
            m_shadowStableCascadeRadii[cascadeIndex] = boundingRadius;
        }
        static const bool s_logShadowFit = std::getenv("ODAI_DEBUG_SHADOW_FIT") != nullptr;
        if (s_logShadowFit) {
            static int s_fitFrame = 0;
            if (cascadeIndex == 0) {
                ++s_fitFrame;
            }
            if (s_fitFrame % 240 == 0) {
                VOX_LOGI("render") << "shadow fit cascade " << cascadeIndex
                                   << " far=" << cascadeFar
                                   << " wanted=" << boundingRadius
                                   << " cached=" << m_shadowStableCascadeRadii[cascadeIndex]
                                   << " sceneFit=" << (orthoSceneFit ? "yes" : "no");
            }
        }
        // Scene-fitted ortho cascades bypass the stable-radius cache: the cache
        // only invalidates on projection changes, not on scene re-uploads.
        const float cascadeRadius =
            orthoSceneFit ? boundingRadius : m_shadowStableCascadeRadii[cascadeIndex];
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
        // The FAR side is padded around the cascade sphere; the NEAR side is
        // opened all the way to the light. They are not symmetric on purpose.
        //
        // A caster between the light and the cascade box is not "outside the
        // cascade" -- it is precisely the thing casting into it. Clipping it at
        // a near plane just above the sphere loses its shadow, and because the
        // SAME matrix is what buildVisibleImportedDraws culls pages against, it
        // loses the draw as well. Both happen together, and the geometry that
        // hits it is the tall stuff: with the old padding, cascade 0 had about
        // 390 units (5.6 m) of headroom above a sphere centred on the camera,
        // so Goodsprings' radio mast and water tower dropped in and out of the
        // shadow set as the camera moved. That is the flicker.
        //
        // Opening the near plane costs range, not precision here: an
        // orthographic depth range is linear and the shadow format is
        // D32_SFLOAT, so going from ~2.9r to ~3.3r of span is free in practice.
        const float casterPadding = std::max(24.0f, cascadeRadius * 0.35f);
        constexpr float kLightNearPlane = 1.0f;
        // ODAI_SHADOW_LEGACY_NEAR=1 restores the symmetric near plane, for
        // A/B-ing the flicker this change fixes.
        static const bool s_legacyNear = std::getenv("ODAI_SHADOW_LEGACY_NEAR") != nullptr;
        const float lightNear = s_legacyNear
            ? std::max(0.1f, lightDistance - cascadeRadius - casterPadding)
            : kLightNearPlane;
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

    // Cascade interleaving: skip re-rendering a cascade whose atlas tile is
    // still exactly right, and alternate the two far cascades under motion.
    //
    // Two skip conditions, one exact and one approximate:
    //   * The computed matrix is BITWISE the one the tile was rendered with.
    //     Texel snapping and radius quantization make this the common case for
    //     a slow or stationary camera, and the skip is then free -- the tile
    //     would have been re-rendered identical.
    //   * Far cascades (2, 3) alternate by frame parity while moving. Their
    //     texels are tens of world units, so serving a tile whose snap origin
    //     is one frame stale moves distant shadows by less than a screen pixel
    //     -- and it halves the largest single block of repeated geometry work
    //     in the frame.
    //
    // A skipped cascade samples with the matrix its tile was RENDERED with
    // (cached), never this frame's -- content and matrix must agree or far
    // shadows swim. Animated actors and rigid imported machinery break the
    // exact-skip (they move without moving the matrix); see the note below for
    // why that does not also rule out deferring the far cascades.
    std::uint32_t shadowSkipCascadeMask = 0;
    static const bool s_shadowInterleaveDisabled =
        std::getenv("ODAI_SHADOW_INTERLEAVE") != nullptr &&
        std::getenv("ODAI_SHADOW_INTERLEAVE")[0] == '0';
    m_shadowInterleaveParity ^= 1u;
    const bool anyAnimatedShadowCasters =
        !m_skinningMeshDraws.empty() || !m_importedRigidAnimations.empty();
    if (!s_shadowInterleaveDisabled) {
        for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
            if (!directionalShadowsForFrame) {
                break;
            }
            if (!m_shadowRenderedValid[cascadeIndex]) {
                continue;
            }
            const bool matrixUnchanged =
                std::memcmp(
                    m_shadowRenderedMatrices[cascadeIndex].m,
                    lightViewProjMatrices[cascadeIndex].m,
                    sizeof(lightViewProjMatrices[cascadeIndex].m)) == 0;
            // THE EXACT-SKIP AND THE PARITY DEFERRAL ARE DIFFERENT CLAIMS, and
            // conflating them is what made a single skinned actor cost the
            // whole atlas.
            //
            // The exact-skip says "nothing in this cascade moved", which a
            // animated caster falsifies: it animates without moving the light
            // matrix, so its shadow would freeze while the caster moved.
            //
            // The parity deferral says only "this cascade may be one frame
            // stale", which is a bound on ERROR rather than a claim of
            // stillness -- and it is only applied to cascades 2 and 3, which
            // start 107 world units out and have texels 0.75 and 3.4 units
            // across. An actor covers a fraction of one of those texels per
            // frame. Refusing it whenever any skinned actor exists anywhere was
            // free when the Fallout viewer had none; with a populated town it
            // means every cascade re-renders every frame forever.
            const bool canExactSkip = matrixUnchanged && !anyAnimatedShadowCasters;
            const bool parityDefersFarCascade =
                cascadeIndex >= 2u && ((cascadeIndex & 1u) == m_shadowInterleaveParity);
            if (canExactSkip || parityDefersFarCascade) {
                shadowSkipCascadeMask |= (1u << cascadeIndex);
                lightViewProjMatrices[cascadeIndex] = m_shadowRenderedMatrices[cascadeIndex];
            }
        }
    }
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        if (!directionalShadowsForFrame) {
            break;
        }
        if ((shadowSkipCascadeMask & (1u << cascadeIndex)) == 0u) {
            m_shadowRenderedMatrices[cascadeIndex] = lightViewProjMatrices[cascadeIndex];
            m_shadowRenderedValid[cascadeIndex] = true;
        }
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
    // The same treatment for the sky's fill light, and applied to the WHOLE SH
    // set rather than only its DC term: tinting the constant term alone leaves
    // the directional terms carrying the old hue, so a surface facing up and a
    // surface facing sideways end up lit by two different weathers.
    {
        const odai::math::Vector3 weatherAmbient = weatherLightTint(
            m_weatherSky.ambientColor, m_weatherSky.lightingWeight);
        for (odai::math::Vector3& coefficient : shIrradiance) {
            coefficient = odai::math::Vector3{coefficient.x * weatherAmbient.x,
                                              coefficient.y * weatherAmbient.y,
                                              coefficient.z * weatherAmbient.z};
        }
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
    const odai::math::Matrix4 inverseViewColumnMajor =
        transpose(odai::math::inverse(view));
    std::memcpy(mvpUniform.contactShadowInvView,
                inverseViewColumnMajor.m,
                sizeof(mvpUniform.contactShadowInvView));
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
    mvpUniform.shadowConfig2[1] = m_shadowDebugSettings.activeAoBias();
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

    const double visualTimeSeconds = m_visualTimeSeconds >= 0.0f
        ? static_cast<double>(m_visualTimeSeconds)
        : frameNowSeconds;
    const float flowTimeSeconds = static_cast<float>(std::fmod(visualTimeSeconds, 4096.0));
    m_importedRigidAnimationTimeSeconds = flowTimeSeconds;
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
    // VOLUMETRIC FOG BASE HEIGHT IS AN ABSOLUTE WORLD Y, which cannot be one
    // constant across games at different scales. The 42.0 default suits a
    // voxel world whose ground is near the origin; Fallout's terrain sits at
    // y 8700-13000, so `worldPos.y - 42` is ~9000 and the marcher's
    // exp(-relativeHeight * falloff) underflows to zero at every step. The
    // pass ran, cost its 1.9 ms, and produced nothing.
    //
    // For an imported scene, anchor it to the camera instead: fog is a local
    // effect and the interesting band is the few hundred units around the
    // viewer, not an altitude fixed at world build time.
    float fogBaseHeight = m_skyDebugSettings.volumetricFogBaseHeight;
    if (renderingImportedScene &&
        !useAuthoredImportedInteriorLighting(m_importedInteriorLighting)) {
        fogBaseHeight = eye.y - 240.0f;
    }
    // An authored weather says how far it can be seen through, and that beats
    // any constant density here. WTHR's fog-far is in world units, so deriving
    // the density from it makes the haze correct at Bethesda's ~70 units/metre
    // without a scene-specific tuning pass.
    //
    // Two things about the height term have to change with it, and both were
    // visible whitewash before:
    //
    //  - `exp(-(h - base) * falloff)` GROWS below the base, up to the shader's
    //    e^7 clamp -- a thousandfold. With the base pinned 240 units under the
    //    eye, standing on Goodsprings' hill put the whole valley floor below it
    //    and rendered a solid white lake over the town.
    //  - A desert vista is aerial perspective, not ground fog: near-uniform
    //    density with a very slow falloff over thousands of units.
    //
    // An imported scene with NO weather record takes the same treatment against
    // a default distance. It has the same problem for the same reason -- the
    // constant density and the growing height term are both authored for a
    // world at the origin -- and leaving it out meant an unmodded run, the one
    // with no weather plugin loaded, was the only configuration still rendering
    // the white lake.
    constexpr float kDefaultFogFarDistance = 60000.0f;  // ~850 m, a clear day
    if (renderingImportedScene &&
        !useAuthoredImportedInteriorLighting(m_importedInteriorLighting)) {
        // Distance only, NOT gated on weight -- matching applyAerialPerspective
        // in imported_static.frag.slang, which the comment above says these two
        // atmospheres are calibrated against each other by. A caller publishing
        // a fog-far with weight 0 is saying "leave my sky colours procedural,
        // but this is how far you can see"; gating the density on weight here
        // and not there made the two disagree by 2.7x on an Oblivion worldspace.
        const bool authored = m_weatherSky.fogFarDistance > 1.0f;
        const float fogFarDistance =
            authored ? m_weatherSky.fogFarDistance : kDefaultFogFarDistance;
        // Mean of fogExtinctionCoefficient() in tone_map.frag.slang, which is
        // what the density is multiplied by before it reaches transmittance.
        constexpr float kMeanExtinction = 0.20f;
        // Leave ~6% of the far surface visible at the authored distance rather
        // than driving it to zero: the record's fog-far is where the game hid
        // its draw distance, and a wall of opaque haze there is worse than the
        // horizon reading through faintly.
        //
        // Half of that 2.8, because this is not the only atmosphere in the
        // frame: imported_static.frag.slang's aerial perspective calibrates
        // itself against the same fog-far and the two compose multiplicatively.
        constexpr float kOpticalDepthAtFogFar = 1.4f;
        const float density = kOpticalDepthAtFogFar / (fogFarDistance * kMeanExtinction);
        mvpUniform.skyConfig4[0] = std::clamp(density, 1.0e-5f, 0.02f);
        mvpUniform.skyConfig4[1] = 0.00012f;  // ~20% thinner per 2000 units up
        fogBaseHeight = eye.y;
    }
    // Tuning knobs -- these are look values and want sweeping per scene.
    if (const char* env = std::getenv("ODAI_FOG_BASE")) {
        fogBaseHeight = static_cast<float>(std::atof(env));
    }
    if (const char* env = std::getenv("ODAI_FOG_DENSITY")) {
        mvpUniform.skyConfig4[0] = std::clamp(static_cast<float>(std::atof(env)), 0.0f, 1.0f);
    }
    if (const char* env = std::getenv("ODAI_FOG_FALLOFF")) {
        mvpUniform.skyConfig4[1] = std::clamp(static_cast<float>(std::atof(env)), 0.0f, 1.0f);
    }
    mvpUniform.skyConfig4[2] = fogBaseHeight;
    mvpUniform.skyConfig4[3] = std::clamp(m_skyDebugSettings.volumetricSunScattering, 0.0f, 8.0f);
    if (const char* env = std::getenv("ODAI_FOG_SCATTER")) {
        mvpUniform.skyConfig4[3] = std::clamp(static_cast<float>(std::atof(env)), 0.0f, 8.0f);
    }
    const uint32_t autoExposureUpdateIntervalFrames = std::max(
        1u,
        static_cast<uint32_t>(std::max(1, m_skyDebugSettings.autoExposureUpdateIntervalFrames))
    );
    const bool autoExposureEnabled = m_skyDebugSettings.autoExposureEnabled && m_autoExposureComputeAvailable;
    mvpUniform.skyConfig5[0] = autoExposureEnabled ? 1.0f : 0.0f;
    mvpUniform.skyConfig5[1] = std::clamp(m_skyDebugSettings.manualExposure, 0.05f, 8.0f);
    mvpUniform.skyConfig5[2] = (m_morrowindSkyTextureImageView != VK_NULL_HANDLE) ? 1.0f : 0.0f;
    mvpUniform.skyConfig5[3] = std::clamp(m_skyDebugSettings.waterRefractionDecay, 0.25f, 3.0f);

    // Authored sky, if a WTHR record pushed one in. Weight 0 makes every term
    // below inert, which is the state every game other than New Vegas is in.
    mvpUniform.weatherSkyUpper[3] = std::clamp(m_weatherSky.weight, 0.0f, 1.0f);
    for (int channel = 0; channel < 3; ++channel) {
        mvpUniform.weatherSkyUpper[channel] = m_weatherSky.skyUpper[channel];
        mvpUniform.weatherSkyLower[channel] = m_weatherSky.skyLower[channel];
        mvpUniform.weatherHorizon[channel] = m_weatherSky.horizon[channel];
        mvpUniform.weatherFog[channel] = m_weatherSky.fogColor[channel];
    }
    // Spare channel: WTHR-driven rain strength. Keeping this in the existing
    // camera block avoids another descriptor, image, or synchronization edge
    // for a purely procedural full-screen effect.
    mvpUniform.weatherSkyLower[3] =
        std::clamp(m_weatherSky.precipitationIntensity, 0.0f, 1.0f);
    // Spare channel: the weather's sun-glare scale. 1 when no weather is
    // published, which is the look every other game has.
    mvpUniform.weatherHorizon[3] =
        (m_weatherSky.weight > 0.0f) ? std::clamp(m_weatherSky.sunGlare, 0.0f, 1.0f) : 1.0f;
    mvpUniform.weatherFog[3] = m_weatherSky.fogFarDistance;

    for (int layer = 0; layer < kWeatherCloudLayerCount; ++layer) {
        const bool hasSlot = m_weatherCloudSlots[layer] != kInvalidImportedTextureSlot;
        for (int channel = 0; channel < 3; ++channel) {
            mvpUniform.weatherCloudTint[layer][channel] = m_weatherSky.cloudTint[layer][channel];
        }
        // Opacity is what gates the layer in the shader, so a layer whose
        // texture failed to upload is switched off here rather than sampling
        // whatever happens to live in slot 0.
        mvpUniform.weatherCloudTint[layer][3] =
            hasSlot ? std::clamp(m_weatherSky.cloudOpacity[layer], 0.0f, 1.0f) : 0.0f;
        mvpUniform.weatherCloudParams[layer][0] =
            hasSlot ? static_cast<float>(m_weatherCloudSlots[layer]) : -1.0f;
        mvpUniform.weatherCloudParams[layer][1] = m_weatherCloudLayers[layer].scrollU;
        mvpUniform.weatherCloudParams[layer][2] = m_weatherCloudLayers[layer].scale;
        mvpUniform.weatherCloudParams[layer][3] = m_weatherCloudLayers[layer].scrollV;
        mvpUniform.weatherCloudBand[layer][0] = m_weatherCloudLayers[layer].bandLow;
        mvpUniform.weatherCloudBand[layer][1] = m_weatherCloudLayers[layer].bandHigh;
        mvpUniform.weatherCloudBand[layer][2] =
            static_cast<float>(m_weatherCloudLayers[layer].mapping);
        // An authored cloud mesh consumes the same active WTHR textures. The
        // fullscreen sky pass sees this flag and does not draw a second,
        // projection-guessed copy underneath it.
        mvpUniform.weatherCloudBand[layer][3] =
            (m_skyCloudIndexCount > 0u) ? 1.0f : 0.0f;
    }

    mvpUniform.tonemapConfig[0] =
        (m_tonemapSettings.mode == TonemapMode::Enb) ? 1.0f : 0.0f;
    mvpUniform.tonemapConfig[1] = m_tonemapSettings.contrast;
    mvpUniform.tonemapConfig[2] = m_tonemapSettings.saturation;
    mvpUniform.tonemapConfig[3] = m_tonemapSettings.curve;
    mvpUniform.tonemapConfig2[0] = m_tonemapSettings.overbrightDampening;
    mvpUniform.tonemapConfig2[1] = static_cast<float>(static_cast<std::uint32_t>(m_debugView));
    mvpUniform.tonemapConfig2[2] = m_taaJitterNdc[0];
    mvpUniform.tonemapConfig2[3] = m_taaJitterNdc[1];
    mvpUniform.hdrHighlightConfig[0] = m_tonemapSettings.whitePoint;
    mvpUniform.hdrHighlightConfig[1] = m_tonemapSettings.highlightShoulder;
    mvpUniform.hdrHighlightConfig[2] = 0.0f;
    mvpUniform.hdrHighlightConfig[3] = 0.0f;
    // See renderer_shared.h: this is what the PCF sites use instead of asking
    // the sampler for the atlas size once (twice, in the cascade blend) per
    // fragment. kShadowAtlasSize is one of the mirrored atlas constants -- if
    // it moves, this follows it for free.
    mvpUniform.shadowAtlasConfig[0] = 1.0f / static_cast<float>(kShadowAtlasSize);
    mvpUniform.shadowAtlasConfig[1] = 0.0f;
    mvpUniform.shadowAtlasConfig[2] = 0.0f;
    mvpUniform.shadowAtlasConfig[3] = 0.0f;
    for (int channel = 0; channel < 3; ++channel) {
        mvpUniform.interiorAmbient[channel] = m_importedInteriorLighting.ambientColor[channel];
        mvpUniform.interiorDirectional[channel] = m_importedInteriorLighting.directionalColor[channel];
        mvpUniform.interiorFog[channel] = m_importedInteriorLighting.fogColor[channel];
    }
    mvpUniform.interiorAmbient[3] = m_importedInteriorLighting.hasAuthoredLighting ? 1.0f : 0.0f;
    mvpUniform.interiorDirectional[3] = m_importedInteriorLighting.useSkyLighting ? 1.0f : 0.0f;
    mvpUniform.interiorFog[3] = m_importedInteriorLighting.showSky ? 1.0f : 0.0f;
    mvpUniform.interiorFogRange[0] = m_importedInteriorLighting.fogNear;
    mvpUniform.interiorFogRange[1] = m_importedInteriorLighting.fogFar;
    mvpUniform.interiorFogRange[2] =
        shouldRenderImportedDirectionalShadows(m_importedInteriorLighting) ? 1.0f : 0.0f;
    mvpUniform.interiorFogRange[3] = m_importedInteriorLighting.enabled ? 1.0f : 0.0f;
    // Terrain layer-blend shaping, exposed so it can be turned OFF.
    //
    // VTXT is now reconstructed on a filtered, 2x-denser grid in cell_builder.
    // The old full-strength smoothstep would simply sharpen that improved field
    // back into a narrow, visibly faceted band. Keep a little shaping and
    // world-space breakup, but let most of the authored/reconstructed ramp
    // survive. Whiterun's DirtPath01 approach is the pinned real-data case.
    //
    // ODAI_FNV_TERRAIN_BLEND=<sharpness>,<coarseUnits>,<fineUnits>,<amount>.
    // Sharpness 0 is the control any future attempt at this should start from.
    static const std::array<float, 4> s_terrainBlend = []() {
        std::array<float, 4> values{0.2f, 520.0f, 170.0f, 0.2f};
        if (const char* env = std::getenv("ODAI_FNV_TERRAIN_BLEND")) {
            std::array<float, 4> parsed{};
            const int count = std::sscanf(
                env, "%f,%f,%f,%f", &parsed[0], &parsed[1], &parsed[2], &parsed[3]);
            for (int i = 0; i < count && i < 4; ++i) {
                values[static_cast<std::size_t>(i)] = parsed[static_cast<std::size_t>(i)];
            }
        }
        return values;
    }();
    mvpUniform.terrainBlendConfig[0] = s_terrainBlend[0];
    mvpUniform.terrainBlendConfig[1] = s_terrainBlend[1];
    mvpUniform.terrainBlendConfig[2] = s_terrainBlend[2];
    mvpUniform.terrainBlendConfig[3] = s_terrainBlend[3];
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
    int waterDebugMode = std::clamp(m_skyDebugSettings.waterDebugMode, 0, 9);
    if (const char* waterDebugOverride = std::getenv("ODAI_WATER_DEBUG")) {
        char* end = nullptr;
        const long parsed = std::strtol(waterDebugOverride, &end, 10);
        if (end != waterDebugOverride && *end == '\0') {
            waterDebugMode = std::clamp(static_cast<int>(parsed), 0, 9);
        }
    }
    mvpUniform.dofConfig2[3] = static_cast<float>(waterDebugMode);
    mvpUniform.waterConfig[0] = std::clamp(m_skyDebugSettings.waterAnimationSpeed, 0.25f, 4.0f);
    mvpUniform.waterConfig[1] = std::clamp(m_skyDebugSettings.waterNormalStrength, 0.25f, 2.5f);
    mvpUniform.waterConfig[2] = std::clamp(m_skyDebugSettings.waterReflectionStrength, 0.25f, 4.0f);
    mvpUniform.waterConfig[3] = std::clamp(m_skyDebugSettings.waterRefractionDecay, 0.25f, 5.0f);

    // One planar reflection target represents one horizontal plane. Select the
    // water patch nearest the camera in XZ (distance to the patch rectangle,
    // not its centre, so a 4096-unit cell directly under the camera wins).
    // This is stable along a river whose cells all share a level and prevents a
    // distant lake at another height from replacing the river reflection.
    static const bool s_planarWaterReflections = []() {
        const char* env = std::getenv("ODAI_WATER_PLANAR_REFLECTIONS");
        return env == nullptr || env[0] != '0';
    }();
    m_waterReflectionPlaneValid = false;
    float nearestWaterDistanceSquared = std::numeric_limits<float>::max();
    if (s_planarWaterReflections &&
        !m_importedSceneInteriorMode &&
        m_colorSampleCount == VK_SAMPLE_COUNT_1_BIT) {
        for (const ImportedSceneChunk& chunk : m_importedSceneChunks) {
            if (!chunk.alive) {
                continue;
            }
            for (const odai::importer::ImportedSceneWaterPatch& patch : chunk.waterPatches) {
                // A camera below the plane is in the underwater case; the
                // mirrored above-water pass is not the right optical source.
                if (eye.y < patch.waterLevel - 0.5f) {
                    continue;
                }
                const float minX = std::min(patch.originX, patch.originX + patch.sizeX);
                const float maxX = std::max(patch.originX, patch.originX + patch.sizeX);
                const float minZ = std::min(patch.originZ, patch.originZ + patch.sizeZ);
                const float maxZ = std::max(patch.originZ, patch.originZ + patch.sizeZ);
                const float nearestX = std::clamp(eye.x, minX, maxX);
                const float nearestZ = std::clamp(eye.z, minZ, maxZ);
                const float dx = eye.x - nearestX;
                const float dz = eye.z - nearestZ;
                const float distanceSquared = (dx * dx) + (dz * dz);
                if (distanceSquared < nearestWaterDistanceSquared) {
                    nearestWaterDistanceSquared = distanceSquared;
                    m_waterReflectionPlaneHeight = patch.waterLevel;
                    m_waterReflectionPlaneValid = true;
                }
            }
        }
    }
    mvpUniform.waterReflectionConfig[0] = m_waterReflectionPlaneHeight;
    mvpUniform.waterReflectionConfig[1] = m_waterReflectionPlaneValid ? 1.0f : 0.0f;
    mvpUniform.waterReflectionConfig[2] = 0.0f;
    mvpUniform.waterReflectionConfig[3] = 0.0f;
    const bool reflectionPlaneChanged =
        m_waterReflectionPlaneValid != m_waterReflectionPreviousPlaneValid ||
        (m_waterReflectionPlaneValid &&
         std::abs(m_waterReflectionPlaneHeight -
                  m_waterReflectionPreviousPlaneHeight) > 0.25f);
    if (reflectionPlaneChanged) {
        m_waterReflectionHistoryValid = false;
    }
    m_waterReflectionPreviousPlaneHeight = m_waterReflectionPlaneHeight;
    m_waterReflectionPreviousPlaneValid = m_waterReflectionPlaneValid;
    const bool reflectionResourcesAvailable =
        m_waterReflectionPlaneValid &&
        m_colorSampleCount == VK_SAMPLE_COUNT_1_BIT &&
        m_importedWaterPipeline != VK_NULL_HANDLE &&
        m_importedWaterVertexBufferHandle != kInvalidBufferHandle &&
        m_importedWaterIndexBufferHandle != kInvalidBufferHandle &&
        m_importedWaterIndexCount > 0u &&
        !m_waterReflectionImages.empty() &&
        !m_waterReflectionImageViews.empty() &&
        !m_waterReflectionImageInitialized.empty() &&
        !m_waterReflectionDepthImages.empty() &&
        !m_waterReflectionDepthImageViews.empty() &&
        !m_waterReflectionDepthImageInitialized.empty() &&
        aoFrameIndex < m_waterReflectionImages.size() &&
        aoFrameIndex < m_waterReflectionImageViews.size() &&
        aoFrameIndex < m_waterReflectionImageInitialized.size() &&
        aoFrameIndex < m_waterReflectionDepthImages.size() &&
        aoFrameIndex < m_waterReflectionDepthImageViews.size() &&
        aoFrameIndex < m_waterReflectionDepthImageInitialized.size() &&
        m_waterReflectionImages[aoFrameIndex] != VK_NULL_HANDLE &&
        m_waterReflectionImageViews[aoFrameIndex] != VK_NULL_HANDLE &&
        m_waterReflectionDepthImages[aoFrameIndex] != VK_NULL_HANDLE &&
        m_waterReflectionDepthImageViews[aoFrameIndex] != VK_NULL_HANDLE &&
        m_importedStaticPipelineTwoSided != VK_NULL_HANDLE &&
        m_importedStaticDepthPrewritePipelineTwoSided != VK_NULL_HANDLE &&
        m_bufferAllocator.getBuffer(m_importedVertexBufferHandle) != VK_NULL_HANDLE &&
        m_bufferAllocator.getBuffer(m_importedIndexBufferHandle) != VK_NULL_HANDLE &&
        !m_importedMeshDraws.empty();

    struct SelectedImportedLight {
        const ImportedLocalLight* light = nullptr;
        float score = 0.0f;
    };
    std::array<SelectedImportedLight, kImportedLocalLightCapacity> selectedImportedLights{};
    std::size_t selectedImportedLightCount = 0;
    // XCLL interiors use the LIGH records at authored radius/intensity. The
    // viewer's historical 3x radius / 1.65x intensity are outdoor readability
    // knobs; applying them inside made overlapping chandeliers bleach the hall
    // even after the exterior sun had correctly been removed.
    const bool authoredInteriorLighting =
        useAuthoredImportedInteriorLighting(m_importedInteriorLighting);
    const float importedLightRadiusScale = authoredInteriorLighting
        ? 1.0f
        : std::clamp(m_debugImportedLightRadiusScale, 0.25f, 8.0f);
    // THE SHADER LOOP IS BOUNDED BY HOW MANY LIGHTS WE UPLOAD, and a light that
    // cannot reach a visible pixel still costs every fragment a full iteration.
    // The scoring below keeps the best 64 by distance from the VIEW AXIS, which
    // is not the same as being in the frustum: a torch inside a building behind
    // the camera scores ~0 and takes a slot. Measured on Whiterun at 4K,
    // capping the loop at 16 instead of 64 moved the main pass 17.15 -> 14.03,
    // i.e. the surplus lights cost ~0.065 ms each in pure rejection.
    //
    // The four SIDE planes come straight out of the row-major view-projection
    // (Gribb-Hartmann: plane = row3 +/- rowN). Near/far are done separately
    // against the view axis rather than from rows 2/3, because those two depend
    // on the depth convention and this renderer is reverse-Z -- getting them
    // backwards would cull lights that are plainly visible, which is a far
    // worse failure than keeping a few extra.
    struct FrustumPlane {
        odai::math::Vector3 normal;
        float distance = 0.0f;
    };
    std::array<FrustumPlane, 4> sidePlanes{};
    {
        const float* m = mvp.m;  // row-major: m[(row * 4) + col]
        const auto makePlane = [](float a, float b, float c, float d) {
            const float length = std::sqrt((a * a) + (b * b) + (c * c));
            const float inverseLength = (length > 1e-6f) ? (1.0f / length) : 0.0f;
            return FrustumPlane{
                odai::math::Vector3{a * inverseLength, b * inverseLength, c * inverseLength},
                d * inverseLength};
        };
        for (int axis = 0; axis < 2; ++axis) {
            const int row = axis * 4;
            sidePlanes[static_cast<std::size_t>(axis * 2)] = makePlane(
                m[12] + m[row + 0], m[13] + m[row + 1], m[14] + m[row + 2], m[15] + m[row + 3]);
            sidePlanes[static_cast<std::size_t>((axis * 2) + 1)] = makePlane(
                m[12] - m[row + 0], m[13] - m[row + 1], m[14] - m[row + 2], m[15] - m[row + 3]);
        }
    }
    std::size_t importedLightsFrustumCulled = 0;
    if (m_debugImportedLightsEnabled && !m_importedLocalLights.empty()) {
        for (const ImportedLocalLight& light : m_importedLocalLights) {
            const odai::math::Vector3 lightPosition{light.position[0], light.position[1], light.position[2]};
            const odai::math::Vector3 cameraToLight = lightPosition - eye;
            const float influenceRadius = std::max(light.radius * importedLightRadiusScale, 1.0f);
            const float alongView = odai::math::dot(cameraToLight, forward);
            // Conservative on purpose: a sphere is only rejected when it is
            // wholly outside a plane, so a light grazing the screen edge is
            // kept. Over-inclusion costs a loop iteration; under-inclusion
            // makes a lamp go out when the camera turns, which is worse.
            bool outsideFrustum = alongView > (farPlane + influenceRadius);
            for (const FrustumPlane& plane : sidePlanes) {
                if (outsideFrustum) {
                    break;
                }
                outsideFrustum =
                    (odai::math::dot(plane.normal, lightPosition) + plane.distance) < -influenceRadius;
            }
            if (outsideFrustum) {
                ++importedLightsFrustumCulled;
                continue;
            }
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
    if (std::getenv("ODAI_DRAW_COUNTS") != nullptr) {
        static std::uint64_t s_lightLogFrame = 0;
        if ((s_lightLogFrame++ % 60u) == 0u) {
            VOX_LOGI("render") << "local lights: uploaded=" << selectedImportedLightCount
                               << " of " << m_importedLocalLights.size()
                               << " (frustum culled " << importedLightsFrustumCulled << ")";
        }
    }
    const float importedLightGlobalIntensity = authoredInteriorLighting
        ? 1.0f
        : std::clamp(m_debugImportedLightIntensity, 0.0f, 8.0f);
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
    const auto animatedImportedLightIntensity = [&](const ImportedLocalLight& light) {
        const float phase =
            (light.position[0] * 0.0131f) + (light.position[1] * 0.0173f) +
            (light.position[2] * 0.0117f);
        float modulation = 1.0f;
        if ((light.flags & 0x08u) != 0u) {  // flicker
            const float a = std::sin((flowTimeSeconds * 12.7f) + phase);
            const float b = std::sin((flowTimeSeconds * 19.1f) + (phase * 1.73f));
            modulation = 0.82f + (0.18f * (0.5f + (0.5f * a * b)));
        } else if ((light.flags & 0x40u) != 0u) {  // flicker slow
            const float a = std::sin((flowTimeSeconds * 4.1f) + phase);
            const float b = std::sin((flowTimeSeconds * 6.7f) + (phase * 1.37f));
            modulation = 0.84f + (0.16f * (0.5f + (0.5f * a * b)));
        } else if ((light.flags & 0x80u) != 0u) {  // pulse
            modulation = 0.78f + (0.22f * (0.5f + 0.5f * std::sin(
                (flowTimeSeconds * 5.5f) + phase)));
        } else if ((light.flags & 0x100u) != 0u) {  // pulse slow
            modulation = 0.80f + (0.20f * (0.5f + 0.5f * std::sin(
                (flowTimeSeconds * 2.2f) + phase)));
        }
        return light.intensity * modulation;
    };
    for (std::size_t lightIndex = 0; lightIndex < selectedImportedLightCount; ++lightIndex) {
        const ImportedLocalLight& light = *selectedImportedLights[lightIndex].light;
        const float lightRadius = std::max(light.radius * importedLightRadiusScale, 1.0f);
        const float animatedIntensity = animatedImportedLightIntensity(light);
        mvpUniform.importedLightPositionRadius[lightIndex][0] = light.position[0];
        mvpUniform.importedLightPositionRadius[lightIndex][1] = light.position[1];
        mvpUniform.importedLightPositionRadius[lightIndex][2] = light.position[2];
        mvpUniform.importedLightPositionRadius[lightIndex][3] = lightRadius;
        mvpUniform.importedLightColorIntensity[lightIndex][0] = light.color[0];
        mvpUniform.importedLightColorIntensity[lightIndex][1] = light.color[1];
        mvpUniform.importedLightColorIntensity[lightIndex][2] = light.color[2];
        mvpUniform.importedLightColorIntensity[lightIndex][3] = animatedIntensity;
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.position[0]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.position[1]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.position[2]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, lightRadius);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.color[0]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.color[1]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.color[2]);
        importedLightSignature = mixImportedLightFloat(importedLightSignature, light.intensity);
    }
    for (std::uint32_t lightIndex = 0; lightIndex < kImportedLocalLightCapacity; ++lightIndex) {
        mvpUniform.interiorPointShadowLightIndices[lightIndex / 4u][lightIndex % 4u] = -1.0f;
    }
    const bool useInteriorPointShadowMaps =
        shouldUseImportedPointShadowMaps(m_importedInteriorLighting);
    if (useInteriorPointShadowMaps) {
        if (!m_interiorPointShadowAtlasValid ||
            m_interiorPointShadowLightSourceCount == 0) {
            m_interiorPointShadowLightSourceCount = 0;
            const auto appendShadowSource = [&](std::uint32_t sourceIndex) {
                if (sourceIndex >= m_importedLocalLights.size() ||
                    m_interiorPointShadowLightSourceCount >= kInteriorPointShadowLightCount) {
                    return;
                }
                const auto first = m_interiorPointShadowLightSourceIndices.begin();
                const auto last = first + static_cast<std::ptrdiff_t>(
                    m_interiorPointShadowLightSourceCount);
                if (std::find(first, last, sourceIndex) != last) {
                    return;
                }
                m_interiorPointShadowLightSourceIndices[
                    m_interiorPointShadowLightSourceCount++] = sourceIndex;
            };
            // Seed with the entry-view relevance order, then use the remaining
            // capacity for off-camera lights. The old code stopped after the
            // visible seed (16 in the fixed Dragonsreach view), so those
            // unshadowed lights still filled table and contact shadows even
            // though the atlas had room for the complete 34-light interior.
            for (std::size_t selectedIndex = 0;
                 selectedIndex < selectedImportedLightCount;
                 ++selectedIndex) {
                appendShadowSource(static_cast<std::uint32_t>(
                    selectedImportedLights[selectedIndex].light -
                    m_importedLocalLights.data()));
            }
            for (std::uint32_t sourceIndex = 0;
                 sourceIndex < static_cast<std::uint32_t>(m_importedLocalLights.size());
                 ++sourceIndex) {
                appendShadowSource(sourceIndex);
            }
        }
        interiorPointShadowLightCount = m_interiorPointShadowLightSourceCount;
        std::array<const ImportedLocalLight*, kInteriorPointShadowLightCount> shadowLights{};
        for (std::uint32_t slot = 0; slot < interiorPointShadowLightCount; ++slot) {
            const std::uint32_t sourceIndex =
                m_interiorPointShadowLightSourceIndices[slot];
            if (sourceIndex < m_importedLocalLights.size()) {
                shadowLights[slot] = &m_importedLocalLights[sourceIndex];
            }
        }
        // The lighting list remains camera-selected. Publish an atlas slot for
        // any of those lights that belonged to the entry-view selection; newly
        // visible lights still illuminate, but do not force a depth rebuild.
        for (std::uint32_t lightIndex = 0;
             lightIndex < static_cast<std::uint32_t>(selectedImportedLightCount);
             ++lightIndex) {
            for (std::uint32_t slot = 0; slot < interiorPointShadowLightCount; ++slot) {
                if (selectedImportedLights[lightIndex].light == shadowLights[slot]) {
                    mvpUniform.interiorPointShadowLightIndices[lightIndex / 4u][lightIndex % 4u] =
                        static_cast<float>(slot);
                    break;
                }
            }
        }
        constexpr std::array<odai::math::Vector3, kInteriorPointShadowFaceCount> kFaceDirections = {{
            {1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f},
        }};
        constexpr std::array<odai::math::Vector3, kInteriorPointShadowFaceCount> kFaceUps = {{
            {0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f},
            {0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
        }};
        std::uint64_t pointShadowSignature = 1469598103934665603ull;
        pointShadowSignature = mixImportedLightSignature(
            pointShadowSignature, interiorPointShadowLightCount);
        for (std::uint32_t slot = 0; slot < interiorPointShadowLightCount; ++slot) {
            if (shadowLights[slot] == nullptr) {
                continue;
            }
            const ImportedLocalLight& light = *shadowLights[slot];
            const odai::math::Vector3 lightPosition{
                light.position[0], light.position[1], light.position[2]};
            const float lightRadius = std::max(light.radius * importedLightRadiusScale, 1.0f);
            pointShadowSignature = mixImportedLightFloat(pointShadowSignature, light.position[0]);
            pointShadowSignature = mixImportedLightFloat(pointShadowSignature, light.position[1]);
            pointShadowSignature = mixImportedLightFloat(pointShadowSignature, light.position[2]);
            pointShadowSignature = mixImportedLightFloat(pointShadowSignature, lightRadius);
            const float shadowNear = std::max(2.0f, lightRadius * 0.003f);
            const odai::math::Matrix4 pointProjection = perspectiveVulkan(
                odai::math::radians(91.0f), 1.0f, shadowNear, lightRadius);
            for (std::uint32_t face = 0; face < kInteriorPointShadowFaceCount; ++face) {
                const odai::math::Matrix4 pointView = lookAt(
                    lightPosition,
                    lightPosition + kFaceDirections[face],
                    kFaceUps[face]);
                const odai::math::Matrix4 pointViewProj = pointProjection * pointView;
                const odai::math::Matrix4 pointViewProjColumnMajor = transpose(pointViewProj);
                std::memcpy(
                    mvpUniform.interiorPointShadowViewProj[
                        (slot * kInteriorPointShadowFaceCount) + face],
                    pointViewProjColumnMajor.m,
                    sizeof(pointViewProjColumnMajor.m));
            }
        }
        mvpUniform.interiorPointShadowParams[0] =
            static_cast<float>(interiorPointShadowLightCount);
        mvpUniform.interiorPointShadowParams[1] =
            static_cast<float>(kInteriorPointShadowFaceSize) /
            static_cast<float>(kShadowAtlasSize);
        mvpUniform.interiorPointShadowParams[2] =
            1.0f / static_cast<float>(kShadowAtlasSize);
        mvpUniform.interiorPointShadowParams[3] =
            interiorPointShadowLightCount > 0 ? 1.0f : 0.0f;
        renderInteriorPointShadowsThisFrame =
            interiorPointShadowLightCount > 0 &&
            (!m_interiorPointShadowAtlasValid ||
             m_interiorPointShadowSignature != pointShadowSignature);
        m_interiorPointShadowSignature = pointShadowSignature;
    } else {
        m_interiorPointShadowAtlasValid = false;
        m_interiorPointShadowSignature = 0;
        m_interiorPointShadowLightSourceCount = 0;
    }
    mvpUniform.importedLightConfig[0] = static_cast<float>(selectedImportedLightCount);
    mvpUniform.importedLightConfig[1] = importedLightGlobalIntensity;
    mvpUniform.importedLightConfig[2] = m_debugImportedLightsEnabled ? 1.0f : 0.0f;
    mvpUniform.importedLightConfig[3] = static_cast<float>(m_importedLocalLights.size());

    // Clustered (Forward+) light culling. The grid is published to the shader
    // ONLY when the pass will actually run this frame; a zero grid is how the
    // fragment shader knows to walk the full light array instead of reading a
    // mask nothing wrote. That direction is the safe one -- the fallback costs
    // performance, a stale mask would cost light.
    //
    // ODAI_LIGHT_CLUSTERS=0 forces the fallback. The two paths must render
    // identically -- the cull only decides which lights are ITERATED, never how
    // they shade -- so this is the control that says whether a lighting
    // difference came from the culling or from something else.
    static const bool s_lightClustersEnabled = []() {
        const char* env = std::getenv("ODAI_LIGHT_CLUSTERS");
        return env == nullptr || (env[0] != '0');
    }();
    m_lightClusterCullActive =
        s_lightClustersEnabled && m_lightClusterAvailable &&
        m_lightClusterBufferHandle != kInvalidBufferHandle &&
        selectedImportedLightCount > 0 && importedLightGlobalIntensity > 0.0f;
    computeLightClusterSliceParams(
        nearPlane, farPlane, m_lightClusterSliceScale, m_lightClusterSliceBias);
    if (m_lightClusterCullActive) {
        mvpUniform.lightClusterConfig0[0] = static_cast<float>(m_lightClusterGridX);
        mvpUniform.lightClusterConfig0[1] = static_cast<float>(m_lightClusterGridY);
        mvpUniform.lightClusterConfig0[2] = static_cast<float>(kLightClusterSliceCount);
        mvpUniform.lightClusterConfig0[3] = static_cast<float>(kLightClusterTileSize);
    } else {
        mvpUniform.lightClusterConfig0[0] = 0.0f;
        mvpUniform.lightClusterConfig0[1] = 0.0f;
        mvpUniform.lightClusterConfig0[2] = 0.0f;
        mvpUniform.lightClusterConfig0[3] = 0.0f;
    }
    mvpUniform.lightClusterConfig1[0] = m_lightClusterSliceScale;
    mvpUniform.lightClusterConfig1[1] = m_lightClusterSliceBias;
    mvpUniform.lightClusterConfig1[2] = 0.0f;
    mvpUniform.lightClusterConfig1[3] = 0.0f;

    m_contactShadowActive =
        shouldUseImportedContactShadows(m_importedInteriorLighting) &&
        m_contactShadowAvailable && m_lightClusterCullActive &&
        m_contactShadowDepthBufferHandle != kInvalidBufferHandle &&
        m_contactShadowHalfBufferHandle != kInvalidBufferHandle &&
        m_contactShadowFullMaskBufferHandle != kInvalidBufferHandle;
    m_screenSpaceGiActive =
        shouldUseImportedScreenSpaceGi(m_importedInteriorLighting) &&
        m_screenSpaceGiAvailable && m_taaEnabled && useMergedDepthPrepass() &&
        m_contactShadowDepthBufferHandle != kInvalidBufferHandle &&
        m_screenSpaceGiRecordBufferHandles[0] != kInvalidBufferHandle &&
        m_screenSpaceGiRecordBufferHandles[1] != kInvalidBufferHandle;
    mvpUniform.contactShadowConfig[0] = static_cast<float>(m_renderExtent.width);
    mvpUniform.contactShadowConfig[1] = static_cast<float>(m_renderExtent.height);
    mvpUniform.contactShadowConfig[2] = m_contactShadowActive ? 1.0f : 0.0f;
    mvpUniform.contactShadowConfig[3] = static_cast<float>(m_taaJitterPhase & 3u);
    mvpUniform.screenSpaceGiConfig[0] = static_cast<float>(m_screenSpaceGiExtent.width);
    mvpUniform.screenSpaceGiConfig[1] = static_cast<float>(m_screenSpaceGiExtent.height);
    mvpUniform.screenSpaceGiConfig[2] = m_screenSpaceGiActive ? 1.0f : 0.0f;
    mvpUniform.screenSpaceGiConfig[3] = 0.18f;
    mvpUniform.importedPbrConfig[0] = m_importedPbrDefaults.objectRoughness;
    mvpUniform.importedPbrConfig[1] = m_importedPbrDefaults.terrainRoughness;
    mvpUniform.importedPbrConfig[2] = m_importedPbrDefaults.metallic;
    mvpUniform.importedPbrConfig[3] = m_importedPbrDefaults.enabled ? 1.0f : 0.0f;

    mvpUniform.fogMapConfig[0] = m_fogMapInvExtentX;
    mvpUniform.fogMapConfig[1] = m_fogMapInvExtentZ;
    mvpUniform.fogMapConfig[2] = 0.0f;
    mvpUniform.fogMapConfig[3] = m_fogMapEnabled ? 1.0f : 0.0f;

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
        m_voxelGiRequested &&
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

    updateFrameDescriptorSets(
        aoFrameIndex,
        bufferInfo,
        mvpSliceOpt->offset,
        autoExposureHistogramBuffer,
        autoExposureStateBuffer,
        voxelGiChunkMetaDescriptorInfo.has_value() ? &(*voxelGiChunkMetaDescriptorInfo) : nullptr,
        voxelGiChunkVoxelDescriptorInfo.has_value() ? &(*voxelGiChunkVoxelDescriptorInfo) : nullptr
    );

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
    const FrameInstanceDrawData frameInstanceDrawData{};
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
    const VkBuffer importedActorVertexBuffer = VK_NULL_HANDLE;
    const VkBuffer importedActorIndexBuffer = VK_NULL_HANDLE;
    constexpr VkDeviceSize importedActorVertexOffset = 0;
    constexpr VkDeviceSize importedActorIndexOffset = 0;
    std::span<const ImportedMeshDraw> importedMeshDrawsForFrame(
        m_importedMeshDraws.data(),
        m_importedMeshDraws.size());
    std::uint32_t importedTerrainDrawCountForFrame = m_importedTerrainDrawCount;
    // Without page culling there is no near/far split to compute, so the whole
    // terrain prefix counts as near -- a cooked scene is a bounded region.
    m_visibleImportedNearTerrainDrawCount = m_importedTerrainDrawCount;
    static const bool s_disableImportedPageCulling =
        std::getenv("ODAI_DEBUG_NO_IMPORTED_PAGE_CULL") != nullptr;
    const bool importedPageCullingEnabled =
        !s_disableImportedPageCulling && !m_importedPageDrawRanges.empty();
    if (s_disableImportedPageCulling) {
        static bool s_loggedDisabledImportedPageCulling = false;
        if (!s_loggedDisabledImportedPageCulling) {
            VOX_LOGW("render")
                << "ODAI_DEBUG_NO_IMPORTED_PAGE_CULL active: all imported draws are submitted";
            s_loggedDisabledImportedPageCulling = true;
        }
    }
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
            const odai::math::Vector4 clip =
                odai::math::multiply(clipMatrix, odai::math::Vector4{corner, 1.0f});
            // Any corner at/behind the near plane makes the perspective divide
            // (sign-flipped w) produce garbage NDC, which would wrongly cull pages
            // straddling the camera — common for foreground chunks and the whole-map
            // overlay under the tilted 3D camera. Treat such pages as visible.
            if (clip.w <= 1e-4f) {
                return true;
            }
            const float invW = 1.0f / clip.w;
            const float ndcX = clip.x * invW;
            const float ndcY = clip.y * invW;
            const float ndcZ = clip.z * invW;
            ndcMinX = std::min(ndcMinX, ndcX);
            ndcMinY = std::min(ndcMinY, ndcY);
            ndcMinZ = std::min(ndcMinZ, ndcZ);
            ndcMaxX = std::max(ndcMaxX, ndcX);
            ndcMaxY = std::max(ndcMaxY, ndcY);
            ndcMaxZ = std::max(ndcMaxZ, ndcZ);
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
        // Page order is arena order, i.e. the order cells happened to stream in.
        //
        // Sorting these front-to-back was tried and REMOVED. It was worth ~2 ms
        // while the main pass rejected occluded fragments using only its own
        // progressive depth writes, but the depth prewrite (frame_pass_main.cc)
        // lays all opaque depth before any shading draw, so submission order no
        // longer decides what gets rejected. Measured with the prewrite in
        // place: 11.4 ms unsorted against 11.9 ms sorted, and 0.0019% of pixels
        // different -- below the 0.048% run-to-run noise floor. It bought
        // nothing and cost a sort per cull pass per frame.
        m_visibleImportedPageScratch.assign(m_importedPageDrawRanges.size(), 0u);
        m_visibleImportedPageOrder.clear();
        for (std::size_t pageIndex = 0; pageIndex < m_importedPageDrawRanges.size(); ++pageIndex) {
            if (!importedPageIntersectsClip(m_importedPageDrawRanges[pageIndex], clipMatrix, clipMargin)) {
                continue;
            }
            m_visibleImportedPageScratch[pageIndex] = 1u;
            m_visibleImportedPageOrder.push_back(static_cast<std::uint32_t>(pageIndex));
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

        // Terrain stays a prefix of the visible list -- callers read
        // m_visibleImportedTerrainDrawCount as "the first N draws are terrain" --
        // so the sort applies WITHIN each group rather than across the two.
        //
        // The terrain prefix is itself partitioned NEAR-FIRST. The tessellated
        // terrain pipeline pays hull/domain invocations for every patch it
        // touches even at factor 1, and routing ALL terrain through it measured
        // ~3.5 ms on the LNL iGPU while everything past the tessellation ramp
        // subdivides to nothing anyway. Only pages whose bounds come within the
        // ramp go in the near prefix; the passes draw [0, near) tessellated and
        // [near, terrainCount) through the flat pipeline.
        std::uint32_t visibleTerrainDrawCount = 0;
        std::uint32_t visibleNearTerrainDrawCount = 0;
        const auto pageWithinTessRange = [&](const ImportedScenePageDrawRange& pageRange) {
            // Conservative point-to-AABB distance against the tessellation
            // ramp's far end (imported_terrain.tesc stops at 10000).
            constexpr float kTessRangeUnits = 10500.0f;
            float distanceSq = 0.0f;
            const float eyePosition[3] = {eye.x, eye.y, eye.z};
            for (int axis = 0; axis < 3; ++axis) {
                const float clamped = std::clamp(
                    eyePosition[axis], pageRange.boundsMin[axis], pageRange.boundsMax[axis]);
                const float delta = eyePosition[axis] - clamped;
                distanceSq += delta * delta;
            }
            return distanceSq < (kTessRangeUnits * kTessRangeUnits);
        };
        for (const std::uint32_t pageIndex : m_visibleImportedPageOrder) {
            const ImportedScenePageDrawRange& pageRange = m_importedPageDrawRanges[pageIndex];
            if (!pageWithinTessRange(pageRange)) {
                continue;
            }
            const std::uint32_t terrainDrawCount = std::min(pageRange.terrainDrawCount, pageRange.drawCount);
            visibleNearTerrainDrawCount += appendDrawRange(pageRange.firstDraw, terrainDrawCount);
        }
        visibleTerrainDrawCount = visibleNearTerrainDrawCount;
        for (const std::uint32_t pageIndex : m_visibleImportedPageOrder) {
            const ImportedScenePageDrawRange& pageRange = m_importedPageDrawRanges[pageIndex];
            if (pageWithinTessRange(pageRange)) {
                continue;
            }
            const std::uint32_t terrainDrawCount = std::min(pageRange.terrainDrawCount, pageRange.drawCount);
            visibleTerrainDrawCount += appendDrawRange(pageRange.firstDraw, terrainDrawCount);
        }
        m_visibleImportedNearTerrainDrawCount = visibleNearTerrainDrawCount;
        for (const std::uint32_t pageIndex : m_visibleImportedPageOrder) {
            const ImportedScenePageDrawRange& pageRange = m_importedPageDrawRanges[pageIndex];
            const std::uint32_t terrainDrawCount = std::min(pageRange.terrainDrawCount, pageRange.drawCount);
            appendDrawRange(pageRange.firstDraw + terrainDrawCount, pageRange.drawCount - terrainDrawCount);
        }
        return visibleTerrainDrawCount;
    };
    if (importedPageCullingEnabled) {
        constexpr float kImportedMainClipMargin = 0.04f;
        // Margin on the page-vs-cascade test, in NDC. Generous because the
        // cost of being wrong is asymmetric: an extra page costs one merged
        // indirect draw of geometry that is already resident, while a missing
        // one costs a shadow that visibly pops as the camera moves.
        static const bool s_legacyShadowMargin = std::getenv("ODAI_SHADOW_LEGACY_NEAR") != nullptr;
        const float kImportedShadowClipMargin = s_legacyShadowMargin ? 0.08f : 0.25f;
        m_visibleImportedTerrainDrawCount =
            buildVisibleImportedDraws(mvp, kImportedMainClipMargin, m_visibleImportedMeshDraws);
        if (std::getenv("ODAI_DEBUG_IMPORTED_VIS") != nullptr) {
            static int s_visFrame = 0;
            ++s_visFrame;
            if (s_visFrame % 240 == 0) {
                std::size_t chunk0Visible = 0;
                std::size_t chunk0Total = 0;
                if (!m_importedSceneChunks.empty()) {
                    const std::int32_t chunk0VertexOffset =
                        static_cast<std::int32_t>(m_importedSceneChunks.front().firstVertex);
                    for (const ImportedMeshDraw& visibleDraw : m_visibleImportedMeshDraws) {
                        if (visibleDraw.vertexOffset == chunk0VertexOffset) {
                            ++chunk0Visible;
                        }
                    }
                    for (const ImportedMeshDraw& tableDraw : m_importedMeshDraws) {
                        if (tableDraw.vertexOffset == chunk0VertexOffset) {
                            ++chunk0Total;
                        }
                    }
                }
                VOX_LOGI("render") << "imported visibility: chunk0 draws " << chunk0Visible
                                   << "/" << chunk0Total << " visible";
                VOX_LOGI("render") << "imported visibility: totalDraws=" << m_importedMeshDraws.size()
                                   << " pages=" << m_importedPageDrawRanges.size()
                                   << " visibleDraws=" << m_visibleImportedMeshDraws.size()
                                   << " terrainDrawCount=" << m_importedTerrainDrawCount
                                   << " indexCount=" << m_importedIndexCount
                                   << " chunkSlots=" << m_importedSceneChunks.size()
                                   << " liveChunks=" << liveImportedSceneChunkCount()
                                   << " splits=[" << m_shadowCascadeSplits[0] << ","
                                   << m_shadowCascadeSplits[1] << "," << m_shadowCascadeSplits[2]
                                   << "," << m_shadowCascadeSplits[3] << "]"
                                   << " blendedDraws=" << m_importedBlendedDrawOrder.size()
                                   << " shadowDraws=[" << m_visibleImportedShadowMeshDraws[0].size()
                                   << "," << m_visibleImportedShadowMeshDraws[1].size()
                                   << "," << m_visibleImportedShadowMeshDraws[2].size()
                                   << "," << m_visibleImportedShadowMeshDraws[3].size() << "]";
            }
        }
        // ODAI_SHADOW_STABILITY=1 reports how often a cascade's caster set
        // CHANGES SIZE between frames. That is the number that matters for
        // flicker: a caster entering or leaving the set is a shadow appearing
        // or vanishing, and no amount of texel snapping hides it.
        if (std::getenv("ODAI_SHADOW_STABILITY") != nullptr) {
            static std::array<std::size_t, kShadowCascadeCount> s_previousCounts{};
            static std::array<std::uint64_t, kShadowCascadeCount> s_changeCount{};
            static std::uint64_t s_frames = 0;
            ++s_frames;
            for (std::uint32_t i = 0; i < kShadowCascadeCount; ++i) {
                const std::size_t count = m_visibleImportedShadowMeshDraws[i].size();
                if (s_frames > 1 && count != s_previousCounts[i]) {
                    ++s_changeCount[i];
                }
                s_previousCounts[i] = count;
            }
            if ((s_frames % 300u) == 0u) {
                VOX_LOGI("render") << "shadow caster-set changes per cascade over " << s_frames
                                   << " frames: " << s_changeCount[0] << ", " << s_changeCount[1]
                                   << ", " << s_changeCount[2] << ", " << s_changeCount[3]
                                   << "  (counts now " << s_previousCounts[0] << ", "
                                   << s_previousCounts[1] << ", " << s_previousCounts[2] << ", "
                                   << s_previousCounts[3] << ")";
            }
        }
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
    // Back-to-front order for the blended replay in the main pass.
    //
    // The blended pipeline does not write depth, so overlapping blended
    // surfaces composite in submission order, and submission order is upload
    // order -- which is whatever order the cells happened to stream in. Sorting
    // by distance from the camera is what makes two panes of glass in front of
    // each other look right from both sides. This is per-frame because the
    // answer changes as the camera moves; it stays cheap because it only ever
    // touches the blended subset, not the whole draw list.
    //
    // Draw granularity, not triangle granularity: a single merged draw whose
    // own triangles overlap still composites in index order. That is the
    // standard limitation of a sorted-draw transparency pass and is not worth
    // an OIT scheme for the amount of glass Fallout places.
    const odai::math::Vector3 blendSortEye = eye;
    auto buildBlendedDrawOrder = [&](std::span<const ImportedMeshDraw> draws,
                                     std::vector<std::uint32_t>& outOrder) {
        outOrder.clear();
        for (std::size_t drawIndex = 0; drawIndex < draws.size(); ++drawIndex) {
            if (draws[drawIndex].blended) {
                outOrder.push_back(static_cast<std::uint32_t>(drawIndex));
            }
        }
        if (outOrder.size() <= 1u) {
            return;
        }
        std::sort(
            outOrder.begin(),
            outOrder.end(),
            [&](std::uint32_t lhs, std::uint32_t rhs) {
                auto distanceSquared = [&](std::uint32_t index) {
                    const float* center = draws[index].center;
                    const float dx = center[0] - blendSortEye.x;
                    const float dy = center[1] - blendSortEye.y;
                    const float dz = center[2] - blendSortEye.z;
                    return (dx * dx) + (dy * dy) + (dz * dz);
                };
                return distanceSquared(lhs) > distanceSquared(rhs);
            });
    };
    buildBlendedDrawOrder(importedMeshDrawsForFrame, m_importedBlendedDrawOrder);
    // Actors get the same treatment. They are a separate vertex/index buffer and
    // a separate draw list, so they need their own order -- one list cannot
    // index into two buffers.
    buildBlendedDrawOrder(
        std::span<const ImportedMeshDraw>(importedActorMeshDraws.data(), importedActorMeshDraws.size()),
        m_importedActorBlendedDrawOrder);

    // Page culling happens after descriptor setup, but before the first GPU
    // pass. Keep the UBO flag aligned with the exact main-pass gate so water
    // never samples a temporal target that this frame cannot produce.
    const bool reflectionAvailable =
        reflectionResourcesAvailable && !importedMeshDrawsForFrame.empty();
    if (reflectionAvailable != m_waterReflectionPreviousAvailable) {
        m_waterReflectionHistoryValid = false;
    }
    m_waterReflectionPreviousAvailable = reflectionAvailable;
    mvpUniform.waterReflectionConfig[1] = reflectionAvailable ? 1.0f : 0.0f;
    std::memcpy(mvpSliceOpt->mapped, &mvpUniform, sizeof(mvpUniform));

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
    frameExecutionContext.mvpDynamicOffset = mvpDynamicOffset;

    // Runs before shadow/prepass/main so its output is ready for all three
    // consumers' skinned-actor draw blocks.
    recordSkinningPass(frameExecutionContext);

    ShadowPassInputs shadowPassInputs{};
    shadowPassInputs.skipDirectionalShadows =
        !shouldRenderImportedDirectionalShadows(m_importedInteriorLighting);
    shadowPassInputs.renderInteriorPointShadows = renderInteriorPointShadowsThisFrame;
    shadowPassInputs.interiorPointShadowLightCount = interiorPointShadowLightCount;
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
    // ODAI_FAT_SHADOW_STREAM=1 puts the shadow pass back on the 72-byte main
    // vertex stream instead of the 28-byte compact one.
    //
    // This is the measuring stick for "how much does vertex WIDTH cost a
    // vertex-bound pass here", and it is the cheapest one in the tree because
    // both streams and both pipelines already exist over identical geometry.
    // Measured on Goodsprings, interleaved A/B: cutting 44 of 72 bytes -- 61%,
    // and from a tightly packed dedicated buffer rather than a strided read, so
    // an upper bound -- moves the shadow pass 2.12 -> 1.76 ms and 2.00 -> 1.87
    // ms. That is 7-17% of the pass for the most aggressive cut available on
    // the pass most sensitive to it.
    //
    // Worth knowing before slimming ImportedMeshVertex on the strength of an
    // estimate: these passes are bound by geometry submission and primitive
    // throughput far more than by attribute fetch.
    static const bool s_fatShadowStream = std::getenv("ODAI_FAT_SHADOW_STREAM") != nullptr;
    shadowPassInputs.importedShadowVertexBuffer =
        (!s_fatShadowStream && m_importedShadowVertexBufferHandle != kInvalidBufferHandle)
            ? m_bufferAllocator.getBuffer(m_importedShadowVertexBufferHandle)
            : VK_NULL_HANDLE;
    shadowPassInputs.importedIndexBuffer = importedIndexBuffer;
    shadowPassInputs.importedMeshDraws = m_importedMeshDraws;
    shadowPassInputs.importedTerrainDrawCount = m_importedTerrainDrawCount;
    shadowPassInputs.importedActorVertexBuffer = importedActorVertexBuffer;
    shadowPassInputs.importedActorVertexOffset =
        importedActorVertexOffset;
    shadowPassInputs.importedActorIndexBuffer = importedActorIndexBuffer;
    shadowPassInputs.importedActorIndexOffset =
        importedActorIndexOffset;
    shadowPassInputs.importedActorMeshDraws = importedActorMeshDraws;
    shadowPassInputs.skinnedActorMeshDraws = m_skinningMeshDraws;
    shadowPassInputs.skipCascadeMask = shadowSkipCascadeMask;
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
    if (m_voxelGiRequested &&
        m_voxelGiComputeAvailable &&
        m_voxelGiOccupancyPipeline != VK_NULL_HANDLE &&
        m_voxelGiSkyExposurePipeline != VK_NULL_HANDLE &&
        m_voxelGiSurfacePipeline != VK_NULL_HANDLE &&
        m_voxelGiInjectPipeline != VK_NULL_HANDLE &&
        m_voxelGiPropagatePipeline != VK_NULL_HANDLE &&
        m_voxelGiPipelineLayout != VK_NULL_HANDLE &&
        m_voxelGiBufferSet.valid() &&
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
    // The app-level opt-out (setVoxelGiEnabled) folds in here as well as at the
    // dispatch gates above, so the state reported below is what actually ran
    // rather than what the hardware could have run.
    const bool voxelGiComputeActive = m_voxelGiRequested && m_voxelGiComputeAvailable;
    const bool voxelGiRtSurfaceRequested = m_voxelGiDebugSettings.surfaceMode != VoxelGiSurfaceMode::Legacy;
    const bool voxelGiRtSurfaceCanRun =
        voxelGiComputeActive &&
        voxelGiRtSurfaceRequested &&
        m_rayTracingRuntimeEnabled &&
        m_voxelGiSurfacePipelineRt != VK_NULL_HANDLE &&
        m_rtTlas.handle != VK_NULL_HANDLE;
    const bool voxelGiRestirRequested = m_voxelGiDebugSettings.surfaceMode == VoxelGiSurfaceMode::RestirSurface;
    const bool voxelGiRestirCanRun =
        voxelGiComputeActive &&
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
        voxelGiComputeActive,
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
                           << ", compute=" << (voxelGiComputeActive ? "yes" : "no")
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

    // The merged prepass has no ao.depth image to transition -- it depth-tests
    // against m_depthImages, which the main pass owns and the prepass hands
    // over with its own barrier. But it is the prepass that now touches that
    // image FIRST, so the transition the main pass used to do has to move here
    // with it. UNDEFINED as the old layout is correct and cheap: the prepass
    // clears depth, so last frame's contents are expendable by definition.
    if (useMergedDepthPrepass()) {
        transitionImageLayout(
            commandBuffer,
            m_depthImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT
        );
    } else {
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
    }

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
    // Built here rather than after the prepass: the merged prepass renders at
    // the main pass's resolution and needs these.
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_renderExtent.width);
    viewport.height = static_cast<float>(m_renderExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    frameExecutionContext.aoFrameIndex = aoFrameIndex;
    frameExecutionContext.imageIndex = imageIndex;
    frameExecutionContext.aoExtent = aoExtent;
    frameExecutionContext.aoViewport = aoViewport;
    frameExecutionContext.aoScissor = aoScissor;
    frameExecutionContext.viewport = viewport;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_renderExtent;
    frameExecutionContext.scissor = scissor;


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
        importedActorVertexOffset;
    prepassInputs.importedActorIndexBuffer = importedActorIndexBuffer;
    prepassInputs.importedActorIndexOffset =
        importedActorIndexOffset;
    prepassInputs.importedActorMeshDraws = importedActorMeshDraws;
    prepassInputs.skinnedActorMeshDraws = m_skinningMeshDraws;
    prepassInputs.pipeInstanceCount = pipeInstanceCount;
    prepassInputs.pipeInstanceSliceOpt = &pipeInstanceSliceOpt;
    prepassInputs.transportInstanceCount = transportInstanceCount;
    prepassInputs.transportInstanceSliceOpt = &transportInstanceSliceOpt;
    prepassInputs.beltCargoInstanceCount = beltCargoInstanceCount;
    prepassInputs.beltCargoInstanceSliceOpt = &beltCargoInstanceSliceOpt;
    // Exactly the three consumers of the normal-depth buffer. Sun shafts and
    // water are checked against what will actually run, not what is merely
    // compiled in, so a scene with AO off, shafts off and no water skips a full
    // re-render of its geometry every frame.
    prepassInputs.normalDepthNeeded =
        m_debugEnableSsao ||
        m_contactShadowActive ||
        m_screenSpaceGiActive ||
        (sunShaftsForFrame && m_sunShaftComputeAvailable) ||
        m_importedWaterIndexCount > 0u;
    recordNormalDepthPrepass(frameExecutionContext, prepassInputs);

    if (m_debugEnableSsao) {
        recordSsaoPasses(frameExecutionContext);
    }

    // Before main, after the prepass. It depends on neither -- the clusters are
    // pure geometry, not scene depth -- so it sits here only to keep the
    // compute dispatch off the critical path between prepass and main.
    recordLightClusterPass(frameExecutionContext);
    recordScreenSpaceDepthHierarchyPass(frameExecutionContext);
    recordContactShadowPass(frameExecutionContext);
    recordScreenSpaceGiPass(frameExecutionContext);

    m_normalDepthImageInitialized[aoFrameIndex] = true;
    m_aoDepthImageInitialized[imageIndex] = true;





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
    mainPassInputs.importedBlendedDrawOrder = std::span<const std::uint32_t>(
        m_importedBlendedDrawOrder.data(), m_importedBlendedDrawOrder.size());
    mainPassInputs.importedActorVertexBuffer = importedActorVertexBuffer;
    mainPassInputs.importedActorVertexOffset =
        importedActorVertexOffset;
    mainPassInputs.importedActorIndexBuffer = importedActorIndexBuffer;
    mainPassInputs.importedActorIndexOffset =
        importedActorIndexOffset;
    mainPassInputs.importedActorMeshDraws = importedActorMeshDraws;
    mainPassInputs.importedActorBlendedDrawOrder = std::span<const std::uint32_t>(
        m_importedActorBlendedDrawOrder.data(), m_importedActorBlendedDrawOrder.size());
    mainPassInputs.skinnedActorMeshDraws = m_skinningMeshDraws;
    mainPassInputs.pipeInstanceCount = pipeInstanceCount;
    mainPassInputs.pipeInstanceSliceOpt = &pipeInstanceSliceOpt;
    mainPassInputs.transportInstanceCount = transportInstanceCount;
    mainPassInputs.transportInstanceSliceOpt = &transportInstanceSliceOpt;
    mainPassInputs.beltCargoInstanceCount = beltCargoInstanceCount;
    mainPassInputs.beltCargoInstanceSliceOpt = &beltCargoInstanceSliceOpt;
    mainPassInputs.preview = &preview;
    recordMainScenePass(frameExecutionContext, mainPassInputs);
    // After the main pass so it can depth-test against what actually ended up
    // visible, and before anything that consumes motion vectors.
    recordSkinnedVelocityPass(frameExecutionContext);

    // TAA runs on the resolved HDR image before bloom mips are cut from it, so
    // bloom blooms the stabilized frame rather than the shimmering one. When it
    // ran, mip0 is left in TRANSFER_DST (the copy-back) instead of
    // COLOR_ATTACHMENT -- the transitions below take the matching source.
    const TaaPassOutcome taaOutcome = recordTaaPass(
        commandBuffer, aoFrameIndex, frameExecutionContext.gpuTimestampQueryPool);

    if (m_hdrResolveMipLevels > 1u) {
        transitionImageLayout(
            commandBuffer,
            m_hdrResolveImages[aoFrameIndex],
            taaOutcome.hdrResolveLayout,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            taaOutcome.hdrResolveStage,
            taaOutcome.hdrResolveAccess,
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

            const uint32_t srcWidth = std::max(1u, m_renderExtent.width >> (mipLevel - 1u));
            const uint32_t srcHeight = std::max(1u, m_renderExtent.height >> (mipLevel - 1u));
            const uint32_t dstWidth = std::max(1u, m_renderExtent.width >> mipLevel);
            const uint32_t dstHeight = std::max(1u, m_renderExtent.height >> mipLevel);

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
            taaOutcome.hdrResolveLayout,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            taaOutcome.hdrResolveStage,
            taaOutcome.hdrResolveAccess,
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
        m_autoExposureBufferSet.valid() &&
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
        const uint32_t hdrWidth = std::max(1u, m_renderExtent.width >> histogramSourceMip);
        const uint32_t hdrHeight = std::max(1u, m_renderExtent.height >> histogramSourceMip);
        AutoExposureHistogramPushConstants histogramPushConstants{};
        histogramPushConstants.width = hdrWidth;
        histogramPushConstants.height = hdrHeight;
        histogramPushConstants.totalPixels = hdrWidth * hdrHeight;
        histogramPushConstants.binCount = kAutoExposureHistogramBins;
        histogramPushConstants.minLogLuminance = -10.0f;
        histogramPushConstants.maxLogLuminance = 4.0f;
        histogramPushConstants.sourceMipLevel = static_cast<float>(histogramSourceMip);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_autoExposureHistogramPipeline);
        bindDescriptorBuffer(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_autoExposurePipelineLayout,
            0, m_autoExposureBufferSet, m_currentFrame);
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
        bindDescriptorBuffer(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_autoExposurePipelineLayout,
            0, m_autoExposureBufferSet, m_currentFrame);
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
        if (sunShaftsForFrame &&
            m_sunShaftComputeAvailable &&
            m_sunShaftPipelineLayout != VK_NULL_HANDLE &&
            m_sunShaftPipeline != VK_NULL_HANDLE &&
            m_sunShaftBufferSet.valid()) {
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
            bindDescriptorBuffer(
                commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_sunShaftPipelineLayout,
                0, m_sunShaftBufferSet, m_currentFrame);
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

            transitionImageLayout(commandBuffer, m_sunShaftImages[aoFrameIndex],
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT);
            m_sunShaftImageInitialized[aoFrameIndex] = true;
            m_sunShaftImageHasContent[aoFrameIndex] = true;
            endDebugLabel(commandBuffer);
        } else {
          const bool clearDisabledShafts =
              !sunShaftInitialized || m_sunShaftImageHasContent[aoFrameIndex];
          if (clearDisabledShafts) {
            transitionImageLayout(
                commandBuffer, m_sunShaftImages[aoFrameIndex],
                sunShaftInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                    : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                sunShaftInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                    : VK_PIPELINE_STAGE_2_NONE,
                sunShaftInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                    : VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            const VkClearColorValue clearValue = {{0.0f, 0.0f, 0.0f, 1.0f}};
            VkImageSubresourceRange clearRange{};
            clearRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            clearRange.baseMipLevel = 0u;
            clearRange.levelCount = 1u;
            clearRange.baseArrayLayer = 0u;
            clearRange.layerCount = 1u;
            vkCmdClearColorImage(commandBuffer, m_sunShaftImages[aoFrameIndex],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &clearValue, 1, &clearRange);
            transitionImageLayout(commandBuffer, m_sunShaftImages[aoFrameIndex],
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT);
            m_sunShaftImageInitialized[aoFrameIndex] = true;
            m_sunShaftImageHasContent[aoFrameIndex] = false;
          }
        }
        writeGpuTimestampBottom(kGpuTimestampQuerySunShaftEnd);
    }
    if (!wroteSunShaftTimestamps) {
      writeGpuTimestampTop(kGpuTimestampQuerySunShaftStart);
      writeGpuTimestampBottom(kGpuTimestampQuerySunShaftEnd);
    }

    transitionImageLayout(
        commandBuffer, m_swapchainImages[imageIndex],
        m_swapchainImageInitialized[imageIndex]
            ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

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

    // Upload UI geometry and emit HOST→VERTEX barrier before entering the
    // rendering pass (VERTEX_INPUT_BIT is not a framebuffer-space stage and
    // cannot be used in a barrier inside vkCmdBeginRendering).
    if (m_uiRenderer.ready()) {
        m_uiRenderer.uploadGeometry(commandBuffer, m_frameArena, m_uiDrawData);
    }
    writeGpuTimestampTop(kGpuTimestampQueryPostStart);
    coreFramePassOrderValidator.markPassEntered(coreFrameGraphPlan->post, "post");
    beginDebugLabel(commandBuffer, "Pass: Tonemap + UI", 0.24f, 0.24f, 0.24f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &toneMapRenderingInfo);
    // The tonemap/UI pass writes the SWAPCHAIN image and must cover all of it.
    // `viewport`/`scissor` above are the scene pair, sized to m_renderExtent --
    // reusing them here is what squeezed the upscaled frame into the top-left
    // corner of the window when the render scale first went below 1.0, with
    // black filling the rest.
    VkViewport presentViewport{};
    presentViewport.width = static_cast<float>(m_swapchainExtent.width);
    presentViewport.height = static_cast<float>(m_swapchainExtent.height);
    presentViewport.minDepth = 0.0f;
    presentViewport.maxDepth = 1.0f;
    VkRect2D presentScissor{};
    presentScissor.extent = m_swapchainExtent;
    vkCmdSetViewport(commandBuffer, 0, 1, &presentViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &presentScissor);

    if (m_tonemapPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);
        bindGraphicsDescriptorBuffers(commandBuffer);
        countDrawCalls(m_debugDrawCallsPost, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    if (m_imguiInitialized) {
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    }
    // Bracket the custom-UI pass with its own GPU timestamps (unconditional so the
    // duration reads ~0 on frames with no UI, rather than stale values).
    writeGpuTimestampTop(kGpuTimestampQueryUiStart);
    if (m_uiRenderer.ready()) {
        beginDebugLabel(commandBuffer, "Pass: UI", 0.85f, 0.72f, 0.44f);
        m_uiRenderer.record(commandBuffer, 0, m_uiDrawData, m_swapchainExtent);
        endDebugLabel(commandBuffer);
    }
    writeGpuTimestampBottom(kGpuTimestampQueryUiEnd);

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
    if (presentResult == VK_SUCCESS || presentResult == VK_SUBOPTIMAL_KHR) {
        m_lastPresentedImageIndex = imageIndex;
    }
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
    if (imageIndex < m_msaaColorImageInitialized.size()) {
        m_msaaColorImageInitialized[imageIndex] = true;
    }
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
    m_debugCpuFrameTotalMsHistory.push(m_debugFrameTimeMs);
    m_debugCpuFrameWorkMsHistory.push(m_debugCpuFrameWorkMs);
    m_debugCpuFrameEwmaMsHistory.push(m_debugCpuFrameEwmaMs);
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
