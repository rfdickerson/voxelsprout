#include "render/backend/vulkan/renderer_backend.h"

// GPU skeletal skinning compute pre-pass (Dragon Age: Origins touchstone; see
// docs/ROADMAP.md and skinning_resources.cc for the buffer/pipeline setup
// this dispatches). NOT YET called from the frame's pass recording sequence
// -- see skinning_resources.cc's integration checklist.
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

    if (m_skinningPipeline == VK_NULL_HANDLE || !m_skinningBufferSet.valid() ||
        m_skinningVertexCount == 0 || m_skinningDebugBypass) {
        return;
    }

    // TODO(skinning): write GPU timestamp query scope here once dedicated
    // kGpuTimestampQuerySkinningStart/End indices are registered alongside
    // the other passes' query slots (see frame_pass_ssao.cc for the
    // writeGpuTimestampTop/Bottom pattern to copy) -- left unmeasured for now
    // rather than guessing at query indices another pass may already own.

    beginDebugLabel(commandBuffer, "Pass: Skinning", 0.30f, 0.22f, 0.36f, 1.0f);

    SkinningPushConstants pushConstants{};
    pushConstants.vertexCount = m_skinningVertexCount;
    pushConstants.boneCount = m_skinningBoneCount;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_skinningPipeline);
    bindDescriptorBuffer(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_skinningPipelineLayout,
        0, m_skinningBufferSet, m_currentFrame);
    vkCmdPushConstants(
        commandBuffer, m_skinningPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(SkinningPushConstants), &pushConstants);

    const std::uint32_t dispatchX = (m_skinningVertexCount + 63u) / 64u;
    vkCmdDispatch(commandBuffer, dispatchX, 1u, 1u);

    // Explicit compute-write -> vertex-input-read barrier before any pass
    // binds m_skinningOutputVertexBufferHandle as a vertex buffer. Same
    // VkMemoryBarrier2/VkDependencyInfo shape already used for
    // acceleration-structure sync in frame_pass_main.cc.
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

    endDebugLabel(commandBuffer);
}

}  // namespace odai::render
