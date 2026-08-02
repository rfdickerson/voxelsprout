#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include "sim/network_procedural.h"

// GPU skeletal skinning compute pre-pass (Dragon Age: Origins touchstone; see
// docs/ROADMAP.md and skinning_resources.cc for the buffer/pipeline setup
// this dispatches). Called from frame_run.cc before recordShadowAtlasPass, so
// its output is ready for all three consuming passes (shadow, prepass, main).
namespace odai::render {

#include "render/renderer_shared.h"

namespace {
struct SkinningPushConstants {
    std::uint32_t vertexCount;
    std::uint32_t boneCount;
    float pad0;
    float pad1;
};
}  // namespace

void RendererBackend::recordSkinningPass(const FrameExecutionContext& context) {
    VkCommandBuffer commandBuffer = context.commandBuffer;
    VkQueryPool gpuTimestampQueryPool = context.gpuTimestampQueryPool;

    if (m_skinningPipeline == VK_NULL_HANDLE || m_skinningActiveInstanceCount == 0 || m_skinningDebugBypass) {
        return;
    }

    const auto writeGpuTimestampTop = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_NONE, gpuTimestampQueryPool, queryIndex);
    };
    const auto writeGpuTimestampBottom = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, gpuTimestampQueryPool, queryIndex);
    };

    writeGpuTimestampTop(kGpuTimestampQuerySkinningStart);
    beginDebugLabel(commandBuffer, "Pass: Skinning", 0.30f, 0.22f, 0.36f, 1.0f);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_skinningPipeline);

    // One dispatch per active instance slot (a small party -- up to
    // kMaxSkinnedInstances -- not a mass-battle crowd, see
    // docs/ROADMAP.md's out-of-scope note). Each slot has its own
    // descriptor-buffer-set region for this frame, bound in turn.
    bool dispatchedAny = false;
    for (std::uint32_t i = 0; i < m_skinningActiveInstanceCount; ++i) {
        const SkinnedInstanceSlot& slot = m_skinningInstances[i];
        if (slot.vertexCount == 0 || !slot.bufferSet.valid()) {
            continue;
        }

        SkinningPushConstants pushConstants{};
        pushConstants.vertexCount = slot.vertexCount;
        pushConstants.boneCount = slot.boneCount;

        bindDescriptorBuffer(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_skinningPipelineLayout,
            0, slot.bufferSet, m_currentFrame);
        vkCmdPushConstants(
            commandBuffer, m_skinningPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(SkinningPushConstants), &pushConstants);

        const std::uint32_t dispatchX = (slot.vertexCount + 63u) / 64u;
        vkCmdDispatch(commandBuffer, dispatchX, 1u, 1u);
        dispatchedAny = true;
    }

    if (dispatchedAny) {
        // Explicit compute-write -> vertex-input-read barrier before any pass
        // binds an instance's output vertex buffer as a vertex buffer. A
        // single global memory barrier covers every instance's write from the
        // loop above (VkMemoryBarrier2 with no buffer handle is a full
        // memory-domain barrier, not per-resource). Same shape already used
        // for acceleration-structure sync in frame_pass_main.cc.
        VkMemoryBarrier2 skinningWriteBarrier{};
        skinningWriteBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        skinningWriteBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        skinningWriteBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        skinningWriteBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        skinningWriteBarrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;

        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &skinningWriteBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQuerySkinningEnd);
}

}  // namespace odai::render
