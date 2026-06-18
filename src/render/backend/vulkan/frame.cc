#include "render/backend/vulkan/renderer_backend.h"

#include "core/log.h"

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

float performShortPacingWait() {
    const auto waitStart = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::microseconds(250));
    std::this_thread::yield();
    return static_cast<float>(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waitStart).count()
    );
}

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

template <std::size_t N>
float percentileFromRingBuffer(
    const std::array<float, N>& history,
    std::uint32_t sampleCount,
    std::uint32_t writeIndex,
    float percentile
) {
    if (sampleCount == 0) {
        return 0.0f;
    }

    const std::uint32_t clampedSampleCount = std::min(sampleCount, static_cast<std::uint32_t>(N));
    std::array<float, N> scratch{};
    for (std::uint32_t i = 0; i < clampedSampleCount; ++i) {
        const std::uint32_t historyIndex =
            (clampedSampleCount == static_cast<std::uint32_t>(N))
                ? ((writeIndex + i) % static_cast<std::uint32_t>(N))
                : i;
        scratch[i] = history[historyIndex];
    }

    const float clampedPercentile = std::clamp(percentile, 0.0f, 1.0f);
    const std::uint32_t targetIndex = static_cast<std::uint32_t>(
        std::ceil(clampedPercentile * static_cast<float>(clampedSampleCount - 1u))
    );
    auto first = scratch.begin();
    auto nth = first + targetIndex;
    auto last = first + clampedSampleCount;
    std::nth_element(first, nth, last);
    return *nth;
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

bool RendererBackend::shouldThrottleFrameStart(uint64_t completedValue, float* outCpuWaitMs) const {
    if (m_framePacingSettings.maxQueuedFrames >= kMaxFramesInFlight) {
        return false;
    }
    const std::uint32_t queuedFrames = countQueuedFrames(completedValue);
    if (queuedFrames < m_framePacingSettings.maxQueuedFrames) {
        return false;
    }

    if (outCpuWaitMs != nullptr) {
        *outCpuWaitMs += performShortPacingWait();
    } else {
        (void)performShortPacingWait();
    }
    return true;
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

    std::array<std::uint64_t, kGpuTimestampQueryCount> timestamps{};
    const VkResult result = vkGetQueryPoolResults(
        m_device,
        queryPool,
        0,
        kGpuTimestampQueryCount,
        sizeof(timestamps),
        timestamps.data(),
        sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT
    );
    if (result == VK_NOT_READY) {
        return false;
    }
    if (result != VK_SUCCESS) {
        logVkFailure("vkGetQueryPoolResults(gpuTimestamps)", result);
        return false;
    }

    const auto durationMs = [&](uint32_t startIndex, uint32_t endIndex) -> float {
        if (startIndex >= kGpuTimestampQueryCount || endIndex >= kGpuTimestampQueryCount) {
            return 0.0f;
        }
        const std::uint64_t startTicks = timestamps[startIndex];
        const std::uint64_t endTicks = timestamps[endIndex];
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
    m_debugGpuPostTimeMs = durationMs(kGpuTimestampQueryPostStart, kGpuTimestampQueryPostEnd);
    m_debugGpuUiTimeMs = durationMs(kGpuTimestampQueryUiStart, kGpuTimestampQueryUiEnd);
    m_debugGpuFrameTimingMsHistory[m_debugGpuFrameTimingMsHistoryWrite] = m_debugGpuFrameTimeMs;
    m_debugGpuFrameTimingMsHistoryWrite =
        (m_debugGpuFrameTimingMsHistoryWrite + 1u) % kTimingHistorySampleCount;
    m_debugGpuFrameTimingMsHistoryCount =
        std::min(m_debugGpuFrameTimingMsHistoryCount + 1u, kTimingHistorySampleCount);
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
                m_debugPresentedFrameTimingMsHistory[m_debugPresentedFrameTimingMsHistoryWrite] = presentFrameMs;
                m_debugPresentedFrameTimingMsHistoryWrite =
                    (m_debugPresentedFrameTimingMsHistoryWrite + 1u) % kTimingHistorySampleCount;
                m_debugPresentedFrameTimingMsHistoryCount =
                    std::min(m_debugPresentedFrameTimingMsHistoryCount + 1u, kTimingHistorySampleCount);
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
    m_debugCpuFrameP50Ms = percentileFromRingBuffer(
        m_debugCpuFrameTotalMsHistory,
        m_debugCpuFrameTimingMsHistoryCount,
        m_debugCpuFrameTimingMsHistoryWrite,
        0.50f
    );
    m_debugCpuFrameP95Ms = percentileFromRingBuffer(
        m_debugCpuFrameTotalMsHistory,
        m_debugCpuFrameTimingMsHistoryCount,
        m_debugCpuFrameTimingMsHistoryWrite,
        0.95f
    );
    m_debugCpuFrameP99Ms = percentileFromRingBuffer(
        m_debugCpuFrameTotalMsHistory,
        m_debugCpuFrameTimingMsHistoryCount,
        m_debugCpuFrameTimingMsHistoryWrite,
        0.99f
    );

    m_debugGpuFrameP50Ms = percentileFromRingBuffer(
        m_debugGpuFrameTimingMsHistory,
        m_debugGpuFrameTimingMsHistoryCount,
        m_debugGpuFrameTimingMsHistoryWrite,
        0.50f
    );
    m_debugGpuFrameP95Ms = percentileFromRingBuffer(
        m_debugGpuFrameTimingMsHistory,
        m_debugGpuFrameTimingMsHistoryCount,
        m_debugGpuFrameTimingMsHistoryWrite,
        0.95f
    );
    m_debugGpuFrameP99Ms = percentileFromRingBuffer(
        m_debugGpuFrameTimingMsHistory,
        m_debugGpuFrameTimingMsHistoryCount,
        m_debugGpuFrameTimingMsHistoryWrite,
        0.99f
    );

    m_debugPresentedFrameP50Ms = percentileFromRingBuffer(
        m_debugPresentedFrameTimingMsHistory,
        m_debugPresentedFrameTimingMsHistoryCount,
        m_debugPresentedFrameTimingMsHistoryWrite,
        0.50f
    );
    m_debugPresentedFrameP95Ms = percentileFromRingBuffer(
        m_debugPresentedFrameTimingMsHistory,
        m_debugPresentedFrameTimingMsHistoryCount,
        m_debugPresentedFrameTimingMsHistoryWrite,
        0.95f
    );
    m_debugPresentedFrameP99Ms = percentileFromRingBuffer(
        m_debugPresentedFrameTimingMsHistory,
        m_debugPresentedFrameTimingMsHistoryCount,
        m_debugPresentedFrameTimingMsHistoryWrite,
        0.99f
    );
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

    if (m_pendingTransferTimelineValue > 0 && m_pendingTransferTimelineValue <= completedValue) {
        m_pendingTransferTimelineValue = 0;
    }
    if (m_transferCommandBufferInFlightValue > 0 && m_transferCommandBufferInFlightValue <= completedValue) {
        m_transferCommandBufferInFlightValue = 0;
    }
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
