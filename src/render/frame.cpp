#include "render/Renderer.hpp"

#include "core/Log.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace render {

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

bool Renderer::createFrameResources() {
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

    VOX_LOGI("render") << "frame resources ready (" << kMaxFramesInFlight << " frames in flight)\n";
    return true;
}

bool Renderer::createGpuTimestampResources() {
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

bool Renderer::isTimelineValueReached(uint64_t value) const {
    if (value == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return true;
    }
    uint64_t completedValue = 0;
    const VkResult result = vkGetSemaphoreCounterValue(m_device, m_renderTimelineSemaphore, &completedValue);
    if (result != VK_SUCCESS) {
        logVkFailure("vkGetSemaphoreCounterValue(timeline)", result);
        return false;
    }
    return completedValue >= value;
}

void Renderer::readGpuTimestampResults(uint32_t frameIndex) {
    if (!m_gpuTimestampsSupported || m_device == VK_NULL_HANDLE || frameIndex >= m_gpuTimestampQueryPools.size()) {
        return;
    }
    const VkQueryPool queryPool = m_gpuTimestampQueryPools[frameIndex];
    if (queryPool == VK_NULL_HANDLE) {
        return;
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
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
    );
    if (result != VK_SUCCESS) {
        logVkFailure("vkGetQueryPoolResults(gpuTimestamps)", result);
        return;
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
    m_debugGpuGiInjectTimeMs = durationMs(kGpuTimestampQueryGiInjectStart, kGpuTimestampQueryGiInjectEnd);
    m_debugGpuGiPropagateTimeMs = durationMs(kGpuTimestampQueryGiPropagateStart, kGpuTimestampQueryGiPropagateEnd);
    m_debugGpuAutoExposureTimeMs = durationMs(kGpuTimestampQueryAutoExposureStart, kGpuTimestampQueryAutoExposureEnd);
    m_debugGpuSunShaftTimeMs = durationMs(kGpuTimestampQuerySunShaftStart, kGpuTimestampQuerySunShaftEnd);
    m_debugGpuPrepassTimeMs = durationMs(kGpuTimestampQueryPrepassStart, kGpuTimestampQueryPrepassEnd);
    m_debugGpuSsaoTimeMs = durationMs(kGpuTimestampQuerySsaoStart, kGpuTimestampQuerySsaoEnd);
    m_debugGpuSsaoBlurTimeMs = durationMs(kGpuTimestampQuerySsaoBlurStart, kGpuTimestampQuerySsaoBlurEnd);
    m_debugGpuMainTimeMs = durationMs(kGpuTimestampQueryMainStart, kGpuTimestampQueryMainEnd);
    m_debugGpuPostTimeMs = durationMs(kGpuTimestampQueryPostStart, kGpuTimestampQueryPostEnd);
    m_debugGpuFrameTimingMsHistory[m_debugGpuFrameTimingMsHistoryWrite] = m_debugGpuFrameTimeMs;
    m_debugGpuFrameTimingMsHistoryWrite =
        (m_debugGpuFrameTimingMsHistoryWrite + 1u) % kTimingHistorySampleCount;
    m_debugGpuFrameTimingMsHistoryCount =
        std::min(m_debugGpuFrameTimingMsHistoryCount + 1u, kTimingHistorySampleCount);
}

void Renderer::updateDisplayTimingStats() {
    if (!m_supportsDisplayTiming || !m_enableDisplayTiming || m_swapchain == VK_NULL_HANDLE) {
        return;
    }
    if (m_getRefreshCycleDurationGoogle != nullptr) {
        VkRefreshCycleDurationGOOGLE refreshCycle{};
        const VkResult refreshResult = m_getRefreshCycleDurationGoogle(m_device, m_swapchain, &refreshCycle);
        if (refreshResult == VK_SUCCESS) {
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
    std::vector<VkPastPresentationTimingGOOGLE> timings(timingCount);
    timingResult = m_getPastPresentationTimingGoogle(m_device, m_swapchain, &timingCount, timings.data());
    if (timingResult != VK_SUCCESS || timingCount == 0) {
        return;
    }
    m_debugDisplayTimingSampleCount = timingCount;

    const VkPastPresentationTimingGOOGLE* latest = nullptr;
    for (const VkPastPresentationTimingGOOGLE& timing : timings) {
        if (latest == nullptr || timing.presentID > latest->presentID) {
            latest = &timing;
        }
    }
    if (latest == nullptr) {
        return;
    }
    m_lastPresentedDisplayTimingPresentId = latest->presentID;
    m_debugDisplayPresentMarginMs = static_cast<float>(latest->presentMargin * 1.0e-6);
    if (latest->actualPresentTime >= latest->earliestPresentTime) {
        m_debugDisplayActualEarliestDeltaMs =
            static_cast<float>((latest->actualPresentTime - latest->earliestPresentTime) * 1.0e-6);
    } else {
        m_debugDisplayActualEarliestDeltaMs = 0.0f;
    }
}

void Renderer::scheduleBufferRelease(BufferHandle handle, uint64_t timelineValue) {
    if (handle == kInvalidBufferHandle) {
        return;
    }
    if (timelineValue == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        m_bufferAllocator.destroyBuffer(handle);
        return;
    }
    m_deferredBufferReleases.push_back({handle, timelineValue});
}

void Renderer::collectCompletedBufferReleases() {
    if (m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return;
    }

    uint64_t completedValue = 0;
    const VkResult counterResult = vkGetSemaphoreCounterValue(m_device, m_renderTimelineSemaphore, &completedValue);
    if (counterResult != VK_SUCCESS) {
        logVkFailure("vkGetSemaphoreCounterValue", counterResult);
        return;
    }

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

void Renderer::destroyFrameResources() {
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

void Renderer::destroyGpuTimestampResources() {
    for (VkQueryPool& queryPool : m_gpuTimestampQueryPools) {
        if (queryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(m_device, queryPool, nullptr);
            queryPool = VK_NULL_HANDLE;
        }
    }
}

} // namespace render
