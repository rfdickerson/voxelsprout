#include "render/backend/vulkan/renderer_backend.h"

#include "core/log.h"

#include <cstdlib>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace odai::render {

namespace {

template <typename VkHandleT>
uint64_t vkHandleToUint64(VkHandleT handle) {
    if constexpr (std::is_pointer_v<VkHandleT>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

const char* vkResultName(VkResult result) {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    default: return "VK_RESULT_UNKNOWN";
    }
}

void logVkFailure(const char* context, VkResult result) {
    VOX_LOGE("render") << context << " failed: "
                       << vkResultName(result) << " (" << static_cast<int>(result) << ")";
}


} // namespace

uint64_t RendererBackend::completedTimelineValue() const {
    if (m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return 0;
    }
    uint64_t completedValue = 0;
    const VkResult result = vkGetSemaphoreCounterValue(m_device, m_renderTimelineSemaphore, &completedValue);
    if (result != VK_SUCCESS) {
        logVkFailure("vkGetSemaphoreCounterValue(timeline)", result);
        return 0;
    }
    return completedValue;
}

std::uint32_t RendererBackend::countQueuedFrames(uint64_t completedValue) const {
    std::uint32_t queuedFrames = 0;
    for (uint64_t frameTimelineValue : m_frameTimelineValues) {
        if (frameTimelineValue > completedValue) {
            ++queuedFrames;
        }
    }
    return queuedFrames;
}

bool RendererBackend::shouldThrottleFrameStart(uint64_t completedValue) const {
    if (m_framePacingSettings.maxQueuedFrames >= kMaxFramesInFlight) {
        return false;
    }
    return countQueuedFrames(completedValue) >= m_framePacingSettings.maxQueuedFrames;
}

bool RendererBackend::waitTimelineValue(uint64_t value, uint64_t timeoutNs, float* outWaitMs) {
    if (value == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return true;
    }
    const auto waitStart = std::chrono::steady_clock::now();
    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &m_renderTimelineSemaphore;
    waitInfo.pValues = &value;
    const VkResult result = vkWaitSemaphores(m_device, &waitInfo, timeoutNs);
    if (outWaitMs != nullptr) {
        *outWaitMs += static_cast<float>(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waitStart).count()
        );
    }
    if (result == VK_TIMEOUT) {
        return false;
    }
    if (result != VK_SUCCESS) {
        logVkFailure("vkWaitSemaphores(timeline)", result);
        return false;
    }
    return true;
}

uint64_t RendererBackend::oldestQueuedFrameTimelineValue(uint64_t completedValue) const {
    uint64_t oldestValue = 0;
    for (uint64_t frameTimelineValue : m_frameTimelineValues) {
        if (frameTimelineValue > completedValue &&
            (oldestValue == 0 || frameTimelineValue < oldestValue)) {
            oldestValue = frameTimelineValue;
        }
    }
    return oldestValue;
}

uint64_t RendererBackend::frameWaitBudgetNs() const {
    constexpr uint64_t kFallbackBudgetNs = 50'000'000;
    if (m_displayRefreshDurationNs == 0) {
        return kFallbackBudgetNs;
    }
    const uint64_t cadenceDivisor = std::max<uint64_t>(1, m_framePacingSettings.cadenceDivisor);
    return 2 * m_displayRefreshDurationNs * cadenceDivisor;
}

void RendererBackend::resetDisplayTimingTracking() {
    m_debugDisplayRefreshMs = 0.0f;
    m_debugDisplayPresentMarginMs = 0.0f;
    m_debugDisplayActualEarliestDeltaMs = 0.0f;
    m_debugDisplayScheduleErrorMs = 0.0f;
    m_debugDisplayTimingSampleCount = 0;
    m_debugLatePresentCount = 0;
    m_lastSubmittedDisplayTimingPresentId = 0;
    m_lastPresentedDisplayTimingPresentId = 0;
    m_lastProcessedDisplayTimingPresentId = 0;
    m_lastDisplayTimingActualPresentTimeNs = 0;
    m_displayRefreshDurationNs = 0;
    m_lastScheduledDesiredPresentTimeNs = 0;
    m_displayTimingDesiredPresentTimesNs.clear();
    m_pastPresentationTimings.clear();
    m_framePacingStats.presentMarginMs = 0.0f;
    m_framePacingStats.actualPresentDeltaMs = 0.0f;
    m_framePacingStats.presentScheduleErrorMs = 0.0f;
    m_framePacingStats.desiredLeadTimeMs = 0.0f;
    m_framePacingStats.desiredPresentTimeNs = 0;
    m_framePacingStats.latePresentCount = 0;
}

uint64_t RendererBackend::computeDesiredPresentTimeNs(std::uint64_t nowNs) const {
    if (m_displayRefreshDurationNs == 0) {
        return 0;
    }
    const std::uint64_t cadenceDivisor = std::max<std::uint32_t>(1u, m_framePacingSettings.cadenceDivisor);
    const std::uint64_t presentIntervalNs = m_displayRefreshDurationNs * cadenceDivisor;
    const std::uint64_t minimumLeadNs = std::max<std::uint64_t>(m_displayRefreshDurationNs / 4u, 500000u);
    const std::uint64_t targetFloorNs = nowNs + minimumLeadNs;

    std::uint64_t desiredPresentTimeNs = 0;
    if (m_lastDisplayTimingActualPresentTimeNs > 0) {
        desiredPresentTimeNs = m_lastDisplayTimingActualPresentTimeNs + presentIntervalNs;
    } else if (m_lastScheduledDesiredPresentTimeNs > 0) {
        desiredPresentTimeNs = m_lastScheduledDesiredPresentTimeNs + presentIntervalNs;
    } else {
        desiredPresentTimeNs = targetFloorNs + presentIntervalNs;
    }

    while (desiredPresentTimeNs < targetFloorNs) {
        desiredPresentTimeNs += presentIntervalNs;
    }
    return desiredPresentTimeNs;
}

bool RendererBackend::createFrameResources() {
    for (size_t frameIndex = 0; frameIndex < m_frames.size(); ++frameIndex) {
        FrameResources& frame = m_frames[frameIndex];
        VkCommandPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;

        if (vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &frame.commandPool) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed creating command pool for frame resource\n";
            return false;
        }
        {
            const std::string poolName = "frame." + std::to_string(frameIndex) + ".graphics.commandPool";
            setObjectName(VK_OBJECT_TYPE_COMMAND_POOL, vkHandleToUint64(frame.commandPool), poolName.c_str());
        }

        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &frame.imageAvailable) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed creating imageAvailable semaphore\n";
            return false;
        }
        {
            const std::string semaphoreName = "frame." + std::to_string(frameIndex) + ".imageAvailable";
            setObjectName(VK_OBJECT_TYPE_SEMAPHORE, vkHandleToUint64(frame.imageAvailable), semaphoreName.c_str());
        }
    }

    VOX_LOGI("render") << "frame resources ready (" << kMaxFramesInFlight
                       << " frames in flight, timestampReadback=deferred)\n";
    return true;
}

bool RendererBackend::createGpuTimestampResources() {
    if (!m_gpuTimestampsSupported) {
        return true;
    }
    for (size_t frameIndex = 0; frameIndex < m_gpuTimestampQueryPools.size(); ++frameIndex) {
        if (m_gpuTimestampQueryPools[frameIndex] != VK_NULL_HANDLE) {
            continue;
        }
        VkQueryPoolCreateInfo queryPoolCreateInfo{};
        queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolCreateInfo.queryCount = kGpuTimestampQueryCount;
        const VkResult result = vkCreateQueryPool(
            m_device,
            &queryPoolCreateInfo,
            nullptr,
            &m_gpuTimestampQueryPools[frameIndex]
        );
        if (result != VK_SUCCESS) {
            logVkFailure("vkCreateQueryPool(gpuTimestamps)", result);
            return false;
        }
        const std::string queryPoolName = "frame." + std::to_string(frameIndex) + ".gpuTimestampQueryPool";
        setObjectName(
            VK_OBJECT_TYPE_QUERY_POOL,
            vkHandleToUint64(m_gpuTimestampQueryPools[frameIndex]),
            queryPoolName.c_str()
        );
    }
    VOX_LOGI("render") << "GPU timestamp query pools ready (" << m_gpuTimestampQueryPools.size()
        << " pools, " << kGpuTimestampQueryCount << " queries each)\n";
    return true;
}

bool RendererBackend::isTimelineValueReached(uint64_t value) const {
    if (value == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return true;
    }
    return completedTimelineValue() >= value;
}

bool RendererBackend::readGpuTimestampResults(uint32_t frameIndex) {
    if (!m_gpuTimestampsSupported || m_device == VK_NULL_HANDLE || frameIndex >= m_gpuTimestampQueryPools.size()) {
        return false;
    }
    const VkQueryPool queryPool = m_gpuTimestampQueryPools[frameIndex];
    if (queryPool == VK_NULL_HANDLE) {
        return false;
    }
    if (!m_gpuTimestampQuerySubmitted[frameIndex]) {
        return false;
    }

    // Per-query availability, NOT a plain bulk read.
    //
    // Without WITH_AVAILABILITY, vkGetQueryPoolResults reports VK_NOT_READY for
    // the WHOLE range if any single query in it is unwritten -- and most of
    // these are conditional. A frame with voxel GI, SSAO and sun shafts turned
    // off never writes their queries, so the entire readback failed every frame
    // and every pass timing read zero. Availability makes a skipped pass mean
    // "this pass did not run", which is the truth, instead of poisoning the
    // other 30 measurements.
    struct TimestampResult {
        std::uint64_t ticks = 0;
        std::uint64_t available = 0;
    };
    std::array<TimestampResult, kGpuTimestampQueryCount> timestamps{};
    const VkResult result = vkGetQueryPoolResults(
        m_device,
        queryPool,
        0,
        kGpuTimestampQueryCount,
        sizeof(timestamps),
        timestamps.data(),
        sizeof(TimestampResult),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
    );
    if (result != VK_SUCCESS && result != VK_NOT_READY) {
        logVkFailure("vkGetQueryPoolResults(gpuTimestamps)", result);
        return false;
    }
    // The frame's own start/end bracket every pass, so if those two are not
    // ready the submission is genuinely still in flight and nothing is usable.
    if (timestamps[kGpuTimestampQueryFrameStart].available == 0u ||
        timestamps[kGpuTimestampQueryFrameEnd].available == 0u) {
        return false;
    }

    const auto durationMs = [&](uint32_t startIndex, uint32_t endIndex) -> float {
        if (startIndex >= kGpuTimestampQueryCount || endIndex >= kGpuTimestampQueryCount) {
            return 0.0f;
        }
        if (timestamps[startIndex].available == 0u || timestamps[endIndex].available == 0u) {
            return 0.0f;  // pass did not run this frame
        }
        const std::uint64_t startTicks = timestamps[startIndex].ticks;
        const std::uint64_t endTicks = timestamps[endIndex].ticks;
        if (endTicks <= startTicks) {
            return 0.0f;
        }
        const double deltaNs = static_cast<double>(endTicks - startTicks) * static_cast<double>(m_gpuTimestampPeriodNs);
        return static_cast<float>(deltaNs * 1.0e-6);
    };

    m_debugGpuFrameTimeMs = durationMs(kGpuTimestampQueryFrameStart, kGpuTimestampQueryFrameEnd);
    m_debugGpuShadowTimeMs = durationMs(kGpuTimestampQueryShadowStart, kGpuTimestampQueryShadowEnd);
    m_debugGpuGiOccupancyTimeMs = durationMs(kGpuTimestampQueryGiOccupancyStart, kGpuTimestampQueryGiOccupancyEnd);
    m_debugGpuGiSurfaceTimeMs = durationMs(kGpuTimestampQueryGiSurfaceStart, kGpuTimestampQueryGiSurfaceEnd);
    m_debugGpuGiSurfaceCandidateTimeMs = durationMs(kGpuTimestampQueryGiSurfaceCandidateStart, kGpuTimestampQueryGiSurfaceCandidateEnd);
    m_debugGpuGiSurfaceTemporalTimeMs = durationMs(kGpuTimestampQueryGiSurfaceTemporalStart, kGpuTimestampQueryGiSurfaceTemporalEnd);
    m_debugGpuGiSurfaceSpatialTimeMs = durationMs(kGpuTimestampQueryGiSurfaceSpatialStart, kGpuTimestampQueryGiSurfaceSpatialEnd);
    m_debugGpuGiSurfaceResolveTimeMs = durationMs(kGpuTimestampQueryGiSurfaceResolveStart, kGpuTimestampQueryGiSurfaceResolveEnd);
    m_debugGpuGiInjectTimeMs = durationMs(kGpuTimestampQueryGiInjectStart, kGpuTimestampQueryGiInjectEnd);
    m_debugGpuGiPropagateTimeMs = durationMs(kGpuTimestampQueryGiPropagateStart, kGpuTimestampQueryGiPropagateEnd);
    m_debugGpuAutoExposureTimeMs = durationMs(kGpuTimestampQueryAutoExposureStart, kGpuTimestampQueryAutoExposureEnd);
    m_debugGpuSunShaftTimeMs = durationMs(kGpuTimestampQuerySunShaftStart, kGpuTimestampQuerySunShaftEnd);
    m_debugGpuPrepassTimeMs = durationMs(kGpuTimestampQueryPrepassStart, kGpuTimestampQueryPrepassEnd);
    m_debugGpuSsaoTimeMs = durationMs(kGpuTimestampQuerySsaoStart, kGpuTimestampQuerySsaoEnd);
    m_debugGpuSsaoBlurTimeMs = durationMs(kGpuTimestampQuerySsaoBlurStart, kGpuTimestampQuerySsaoBlurEnd);
    m_debugGpuMainTimeMs = durationMs(kGpuTimestampQueryMainStart, kGpuTimestampQueryMainEnd);
    m_debugGpuPrewriteTimeMs =
        durationMs(kGpuTimestampQueryPrewriteStart, kGpuTimestampQueryPrewriteEnd);
    m_debugGpuVelocityTimeMs =
        durationMs(kGpuTimestampQueryVelocityStart, kGpuTimestampQueryVelocityEnd);
    m_debugGpuTaaTimeMs = durationMs(kGpuTimestampQueryTaaStart, kGpuTimestampQueryTaaEnd);
    m_debugGpuPostTimeMs = durationMs(kGpuTimestampQueryPostStart, kGpuTimestampQueryPostEnd);
    m_debugGpuUiTimeMs = durationMs(kGpuTimestampQueryUiStart, kGpuTimestampQueryUiEnd);
    // Per-pass GPU breakdown. Everything above was already measured and then
    // kept private, so "why is the frame 17 ms" could only be answered by
    // disabling passes one at a time and re-measuring. ODAI_GPU_TIMINGS prints
    // it directly.
    // Draw-call census alongside the pass timings. "The frame is 30 ms of CPU
    // in submit" does not say whether that is a few slow calls or a great many
    // cheap ones, and those have completely different fixes.
    if (std::getenv("ODAI_DRAW_COUNTS") != nullptr) {
        static std::uint64_t s_drawLogFrame = 0;
        if ((s_drawLogFrame++ % 60u) == 0u) {
            VOX_LOGI("render") << "draw calls: total=" << m_debugDrawCallsTotal
                               << " shadow=" << m_debugDrawCallsShadow
                               << " prepass=" << m_debugDrawCallsPrepass
                               << " main=" << m_debugDrawCallsMain
                               << "  importedDraws=" << m_importedMeshDraws.size()
                               << " mergedAway=" << m_debugImportedDrawsMerged;
        }
    }
    if (std::getenv("ODAI_GPU_TIMINGS") != nullptr) {
        static std::uint64_t s_timingLogFrame = 0;
        if ((s_timingLogFrame++ % 60u) == 0u) {
            VOX_LOGI("render")
                << "GPU ms: frame=" << m_debugGpuFrameTimeMs
                << " shadow=" << m_debugGpuShadowTimeMs
                << " prepass=" << m_debugGpuPrepassTimeMs
                << " ssao=" << m_debugGpuSsaoTimeMs
                << " ssaoBlur=" << m_debugGpuSsaoBlurTimeMs
                << " main=" << m_debugGpuMainTimeMs
                << " (prewrite=" << m_debugGpuPrewriteTimeMs << ")"
                << " velocity=" << m_debugGpuVelocityTimeMs
                << " taa=" << m_debugGpuTaaTimeMs
                << " post=" << m_debugGpuPostTimeMs
                << " ui=" << m_debugGpuUiTimeMs
                << " autoExposure=" << m_debugGpuAutoExposureTimeMs
                << " sunShaft=" << m_debugGpuSunShaftTimeMs
                << " giOccupancy=" << m_debugGpuGiOccupancyTimeMs;
        }
    }
    m_debugGpuFrameTimingMsHistory.push(m_debugGpuFrameTimeMs);
    updateFrameTimingPercentiles();
    m_gpuTimestampQuerySubmitted[frameIndex] = false;
    return true;
}

void RendererBackend::updateDisplayTimingStats() {
    if (!m_supportsDisplayTiming || !m_enableDisplayTiming || m_swapchain == VK_NULL_HANDLE) {
        return;
    }
    if (m_getRefreshCycleDurationGoogle != nullptr) {
        VkRefreshCycleDurationGOOGLE refreshCycle{};
        const VkResult refreshResult = m_getRefreshCycleDurationGoogle(m_device, m_swapchain, &refreshCycle);
        if (refreshResult == VK_SUCCESS) {
            m_displayRefreshDurationNs = refreshCycle.refreshDuration;
            m_debugDisplayRefreshMs = static_cast<float>(refreshCycle.refreshDuration * 1.0e-6);
        }
    }
    if (m_getPastPresentationTimingGoogle == nullptr) {
        return;
    }

    uint32_t timingCount = 0;
    VkResult timingResult = m_getPastPresentationTimingGoogle(m_device, m_swapchain, &timingCount, nullptr);
    if (timingResult != VK_SUCCESS || timingCount == 0) {
        return;
    }
    m_pastPresentationTimings.resize(timingCount);
    timingResult = m_getPastPresentationTimingGoogle(
        m_device,
        m_swapchain,
        &timingCount,
        m_pastPresentationTimings.data());
    if (timingResult != VK_SUCCESS || timingCount == 0) {
        return;
    }
    m_pastPresentationTimings.resize(timingCount);
    m_debugDisplayTimingSampleCount = timingCount;

    std::sort(
        m_pastPresentationTimings.begin(),
        m_pastPresentationTimings.end(),
        [](const VkPastPresentationTimingGOOGLE& a, const VkPastPresentationTimingGOOGLE& b) {
            return a.presentID < b.presentID;
        }
    );
    const VkPastPresentationTimingGOOGLE& latest = m_pastPresentationTimings.back();
    m_lastPresentedDisplayTimingPresentId = latest.presentID;
    m_debugDisplayPresentMarginMs = static_cast<float>(latest.presentMargin * 1.0e-6);
    m_framePacingStats.presentMarginMs = m_debugDisplayPresentMarginMs;
    if (latest.actualPresentTime >= latest.earliestPresentTime) {
        m_debugDisplayActualEarliestDeltaMs =
            static_cast<float>((latest.actualPresentTime - latest.earliestPresentTime) * 1.0e-6);
    } else {
        m_debugDisplayActualEarliestDeltaMs = 0.0f;
    }

    for (const VkPastPresentationTimingGOOGLE& timing : m_pastPresentationTimings) {
        if (timing.presentID <= m_lastProcessedDisplayTimingPresentId) {
            continue;
        }
        if (m_lastDisplayTimingActualPresentTimeNs > 0 && timing.actualPresentTime > m_lastDisplayTimingActualPresentTimeNs) {
            const float presentFrameMs = static_cast<float>(
                (timing.actualPresentTime - m_lastDisplayTimingActualPresentTimeNs) * 1.0e-6
            );
            if (presentFrameMs > 0.0f) {
                m_debugPresentedFrameTimingMsHistory.push(presentFrameMs);
                m_debugPresentedFrameTimeMs = presentFrameMs;
                m_debugPresentedFps = 1000.0f / presentFrameMs;
            }
        }
        auto desiredPresentTimeIt = m_displayTimingDesiredPresentTimesNs.find(timing.presentID);
        if (desiredPresentTimeIt != m_displayTimingDesiredPresentTimesNs.end()) {
            const std::uint64_t desiredPresentTimeNs = desiredPresentTimeIt->second;
            if (timing.actualPresentTime >= desiredPresentTimeNs) {
                m_debugDisplayScheduleErrorMs =
                    static_cast<float>((timing.actualPresentTime - desiredPresentTimeNs) * 1.0e-6);
            } else {
                m_debugDisplayScheduleErrorMs =
                    -static_cast<float>((desiredPresentTimeNs - timing.actualPresentTime) * 1.0e-6);
            }
            m_framePacingStats.presentScheduleErrorMs = m_debugDisplayScheduleErrorMs;
            if (timing.actualPresentTime > desiredPresentTimeNs + 500000u) {
                ++m_debugLatePresentCount;
            }
            m_displayTimingDesiredPresentTimesNs.erase(desiredPresentTimeIt);
        }
        m_lastDisplayTimingActualPresentTimeNs = timing.actualPresentTime;
        m_lastProcessedDisplayTimingPresentId = timing.presentID;
    }
    m_framePacingStats.actualPresentDeltaMs = m_debugDisplayActualEarliestDeltaMs;
    m_framePacingStats.latePresentCount = m_debugLatePresentCount;
    updateFrameTimingPercentiles();
}

void RendererBackend::updateFrameTimingPercentiles() {
    m_debugCpuFrameP50Ms = odai::core::percentile(m_debugCpuFrameTotalMsHistory, 0.50f);
    m_debugCpuFrameP95Ms = odai::core::percentile(m_debugCpuFrameTotalMsHistory, 0.95f);
    m_debugCpuFrameP99Ms = odai::core::percentile(m_debugCpuFrameTotalMsHistory, 0.99f);

    m_debugGpuFrameP50Ms = odai::core::percentile(m_debugGpuFrameTimingMsHistory, 0.50f);
    m_debugGpuFrameP95Ms = odai::core::percentile(m_debugGpuFrameTimingMsHistory, 0.95f);
    m_debugGpuFrameP99Ms = odai::core::percentile(m_debugGpuFrameTimingMsHistory, 0.99f);

    m_debugPresentedFrameP50Ms = odai::core::percentile(m_debugPresentedFrameTimingMsHistory, 0.50f);
    m_debugPresentedFrameP95Ms = odai::core::percentile(m_debugPresentedFrameTimingMsHistory, 0.95f);
    m_debugPresentedFrameP99Ms = odai::core::percentile(m_debugPresentedFrameTimingMsHistory, 0.99f);
}

void RendererBackend::scheduleBufferRelease(BufferHandle handle, uint64_t timelineValue) {
    if (handle == kInvalidBufferHandle) {
        return;
    }
    if (timelineValue == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        m_bufferAllocator.destroyBuffer(handle);
        return;
    }
    m_deferredBufferReleases.push_back({handle, timelineValue});
}

void RendererBackend::destroyImageResourceNow(
    VkImage image, VmaAllocation allocation, VkImageView imageView) {
    if (imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, imageView, nullptr);
    }
    if (image == VK_NULL_HANDLE) {
        return;
    }
    if (m_vmaAllocator != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(m_vmaAllocator, image, allocation);
    } else {
        vkDestroyImage(m_device, image, nullptr);
    }
}

void RendererBackend::scheduleImageRelease(
    VkImage image, VmaAllocation allocation, VkImageView imageView, uint64_t timelineValue) {
    if (image == VK_NULL_HANDLE && imageView == VK_NULL_HANDLE) {
        return;
    }
    // Nothing has been submitted yet (or there is no timeline to wait on), so
    // no frame can be sampling this image: destroy it immediately rather than
    // parking it on a queue that would never drain.
    if (timelineValue == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        destroyImageResourceNow(image, allocation, imageView);
        return;
    }
    m_deferredImageReleases.push_back({image, allocation, imageView, timelineValue});
}

void RendererBackend::scheduleCommandPoolRelease(VkCommandPool pool, uint64_t timelineValue) {
    if (pool == VK_NULL_HANDLE) {
        return;
    }
    if (timelineValue == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, pool, nullptr);
        return;
    }
    m_deferredCommandPoolReleases.push_back({pool, timelineValue});
}

void RendererBackend::collectCompletedBufferReleases() {
    if (m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return;
    }

    const uint64_t completedValue = completedTimelineValue();

    for (const DeferredBufferRelease& release : m_deferredBufferReleases) {
        if (release.timelineValue <= completedValue) {
            m_bufferAllocator.destroyBuffer(release.handle);
        }
    }
    std::erase_if(
        m_deferredBufferReleases,
        [completedValue](const DeferredBufferRelease& release) {
            return release.timelineValue <= completedValue;
        }
    );

    for (const DeferredImageRelease& release : m_deferredImageReleases) {
        if (release.timelineValue <= completedValue) {
            destroyImageResourceNow(release.image, release.allocation, release.imageView);
        }
    }
    std::erase_if(
        m_deferredImageReleases,
        [completedValue](const DeferredImageRelease& release) {
            return release.timelineValue <= completedValue;
        }
    );

    for (const DeferredCommandPoolRelease& release : m_deferredCommandPoolReleases) {
        if (release.timelineValue <= completedValue) {
            vkDestroyCommandPool(m_device, release.pool, nullptr);
        }
    }
    std::erase_if(
        m_deferredCommandPoolReleases,
        [completedValue](const DeferredCommandPoolRelease& release) {
            return release.timelineValue <= completedValue;
        }
    );

    if (m_pendingTransferTimelineValue > 0 && m_pendingTransferTimelineValue <= completedValue) {
        m_pendingTransferTimelineValue = 0;
    }
    for (TransferCommandSlot& slot : m_transferCommandSlots) {
        if (slot.inFlightTimelineValue > 0 && slot.inFlightTimelineValue <= completedValue) {
            slot.inFlightTimelineValue = 0;
        }
    }
}

bool RendererBackend::anyTransferSlotInFlight() const {
    const uint64_t completedValue = completedTimelineValue();
    for (const TransferCommandSlot& slot : m_transferCommandSlots) {
        if (slot.inFlightTimelineValue > completedValue) {
            return true;
        }
    }
    return false;
}

bool RendererBackend::hasFreeTransferSlot() const {
    const uint64_t completedValue = completedTimelineValue();
    for (const TransferCommandSlot& slot : m_transferCommandSlots) {
        if (slot.inFlightTimelineValue <= completedValue) {
            return true;
        }
    }
    return false;
}

RendererBackend::TransferCommandSlot* RendererBackend::acquireTransferCommandSlot(float* outWaitMs) {
    const uint64_t completedValue = completedTimelineValue();
    TransferCommandSlot* oldestBusySlot = nullptr;
    for (TransferCommandSlot& slot : m_transferCommandSlots) {
        if (slot.commandBuffer == VK_NULL_HANDLE) {
            continue;
        }
        if (slot.inFlightTimelineValue <= completedValue) {
            slot.inFlightTimelineValue = 0;
            return &slot;
        }
        if (oldestBusySlot == nullptr ||
            slot.inFlightTimelineValue < oldestBusySlot->inFlightTimelineValue) {
            oldestBusySlot = &slot;
        }
    }
    if (oldestBusySlot == nullptr) {
        return nullptr;
    }
    // All slots busy: block on the oldest submit rather than dropping the work.
    if (!waitTimelineValue(oldestBusySlot->inFlightTimelineValue, frameWaitBudgetNs(), outWaitMs)) {
        return nullptr;
    }
    oldestBusySlot->inFlightTimelineValue = 0;
    return oldestBusySlot;
}

void RendererBackend::destroyFrameResources() {
    for (FrameResources& frame : m_frames) {
        if (frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, frame.imageAvailable, nullptr);
            frame.imageAvailable = VK_NULL_HANDLE;
        }
        if (frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, frame.commandPool, nullptr);
            frame.commandPool = VK_NULL_HANDLE;
        }
    }
}

void RendererBackend::destroyGpuTimestampResources() {
    for (VkQueryPool& queryPool : m_gpuTimestampQueryPools) {
        if (queryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(m_device, queryPool, nullptr);
            queryPool = VK_NULL_HANDLE;
        }
    }
    m_gpuTimestampQuerySubmitted.fill(false);
}

} // namespace odai::render
