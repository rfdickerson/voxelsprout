#include "render/backend/vulkan/renderer_backend.h"
#include <GLFW/glfw3.h>
#include <cstring>

namespace odai::render {
#include "render/renderer_shared.h"
namespace {
struct PosePush {
  std::uint32_t bones, nodes, output, phase, depth, pad0, pad1, pad2;
};
static_assert(sizeof(PosePush) == 32);
} // namespace
bool RendererBackend::createPoseComputeResources() {
  if (m_posePipeline != VK_NULL_HANDLE)
    return true;
  std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
  for (std::uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorCount = 1;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  if (!createDescriptorSetLayout(
          bindings, m_poseDescriptorLayout, "pose descriptor layout",
          "renderer.pose.layout", nullptr,
          VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT))
    return false;
  VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PosePush)};
  const std::array<VkPushConstantRange, 1> ranges{push};
  if (!createComputePipelineLayout(m_poseDescriptorLayout, ranges,
                                   m_posePipelineLayout, "pose pipeline layout",
                                   "renderer.pose.pipelineLayout")) {
    destroyPoseComputeResources();
    return false;
  }
  VkShaderModule shader = VK_NULL_HANDLE;
  if (!createShaderModuleFromFile(
          m_device, "../src/render/shaders/pose_evaluate.comp.slang.spv",
          "pose_evaluate.comp", shader)) {
    destroyPoseComputeResources();
    return false;
  }
  const bool ok = createComputePipeline(
      m_posePipelineLayout, shader, m_posePipeline, "pose compute pipeline",
      "renderer.pose.pipeline", VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
  vkDestroyShaderModule(m_device, shader, nullptr);
  if (!ok)
    destroyPoseComputeResources();
  return ok;
}
void RendererBackend::destroyPoseComputeResources() {
  for (auto &slot : m_skinningInstances) {
    destroyDescriptorBufferSet(slot.poseBufferSet);
    m_bufferAllocator.destroyBuffer(slot.poseResourceBuffer);
    slot.poseResourceBuffer = kInvalidBufferHandle;
    m_bufferAllocator.destroyBuffer(slot.poseHistoryBuffer);
    slot.poseHistoryBuffer = kInvalidBufferHandle;
    slot.poseHistoryBones = 0;
    slot.poseHistoryValid = false;
    slot.poseHistoryReady = false;
    slot.poseResource = {};
    slot.poseView.reset();
    slot.pendingEvaluation.reset();
    slot.poseComputeReady = false;
  }
  if (m_posePipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(m_device, m_posePipeline, nullptr);
  if (m_posePipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(m_device, m_posePipelineLayout, nullptr);
  if (m_poseDescriptorLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(m_device, m_poseDescriptorLayout, nullptr);
  m_posePipeline = VK_NULL_HANDLE;
  m_posePipelineLayout = VK_NULL_HANDLE;
  m_poseDescriptorLayout = VK_NULL_HANDLE;
}
void RendererBackend::preparePoseCompute(std::uint32_t instanceIndex,
                                         VkDeviceAddress paletteAddress,
                                         VkDeviceSize paletteBytes) {
  auto &slot = m_skinningInstances[instanceIndex];
  if (m_posePipeline == VK_NULL_HANDLE ||
      (!slot.pendingEvaluation &&
       slot.poseHistoryBuffer == kInvalidBufferHandle))
    return;
  if (!slot.poseBufferSet.valid() &&
      !createDescriptorBufferSet(
          m_poseDescriptorLayout, kMaxFramesInFlight,
          VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
          "renderer.pose.descriptors", slot.poseBufferSet))
    return;
  const auto write = [&](std::uint32_t binding, VkDeviceAddress address,
                         VkDeviceSize size) {
    writeDescriptorBufferStorage(
        slot.poseBufferSet, m_currentFrame,
        descriptorBufferBindingOffset(m_poseDescriptorLayout, binding), address,
        size);
  };
  // Valid sentinels for history-only dispatches after CPU pose fallback.
  for (std::uint32_t binding = 0; binding < 4; ++binding)
    write(binding, paletteAddress, paletteBytes);
  const auto prepare = [&]() {
    if (!slot.pendingEvaluation || !slot.poseView || !slot.poseView->skeleton)
      return false;
    std::string error;
    if (slot.poseResourceBuffer == kInvalidBufferHandle) {
      odai::anim::GpuPoseResource resource;
      if (!odai::anim::packGpuPoseResource(
              *slot.poseView->skeleton, slot.poseView->inverseBindMatrices,
              slot.poseView->clips, resource, error))
        return false;
      BufferCreateDesc desc{};
      desc.size = resource.words.size() * sizeof(std::uint32_t);
      desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      desc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      desc.initialData = resource.words.data();
      slot.poseResourceBuffer = m_bufferAllocator.createBuffer(desc);
      if (slot.poseResourceBuffer == kInvalidBufferHandle)
        return false;
      slot.poseResource = std::move(resource);
    }
    if (slot.poseResource.boneCount != slot.boneCount)
      return false;
    std::vector<std::uint32_t> words;
    if (!odai::anim::packGpuPoseFrame(*slot.pendingEvaluation,
                                      slot.poseResource, slot.poseActorWorld,
                                      words, error))
      return false;
    const auto frame =
        m_frameArena.allocateUpload(words.size() * sizeof(std::uint32_t), 256,
                                    FrameArenaUploadKind::Unknown);
    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(slot.boneCount) *
        (slot.pendingEvaluation->instructions.size() * 3 + 4) * 16;
    const auto scratch =
        m_frameArena.allocateUpload(bytes, 256, FrameArenaUploadKind::Unknown);
    if (!frame || !frame->mapped || !scratch)
      return false;
    std::memcpy(frame->mapped, words.data(),
                words.size() * sizeof(std::uint32_t));
    write(0, m_bufferAllocator.getDeviceAddress(slot.poseResourceBuffer),
          slot.poseResource.words.size() * sizeof(std::uint32_t));
    write(1, m_bufferAllocator.getDeviceAddress(frame->buffer) + frame->offset,
          words.size() * sizeof(std::uint32_t));
    write(2,
          m_bufferAllocator.getDeviceAddress(scratch->buffer) + scratch->offset,
          bytes);
    return true;
  };
  slot.poseComputeReady = prepare();
  if (!slot.poseComputeReady && slot.poseHistoryBuffer == kInvalidBufferHandle)
    return;
  if (slot.poseHistoryBones != slot.boneCount) {
    scheduleBufferRelease(slot.poseHistoryBuffer, m_nextTimelineValue - 1);
    slot.poseHistoryBuffer = kInvalidBufferHandle;
    slot.poseHistoryValid = false;
    slot.poseHistoryBones = 0;
  }
  if (slot.poseHistoryBuffer == kInvalidBufferHandle) {
    BufferCreateDesc desc{};
    desc.size = paletteBytes;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    desc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    slot.poseHistoryBuffer = m_bufferAllocator.createBuffer(desc);
    if (slot.poseHistoryBuffer == kInvalidBufferHandle) {
      slot.poseComputeReady = false;
      return;
    }
    slot.poseHistoryBones = slot.boneCount;
  }
  write(4, slot.previousBoneAddress, paletteBytes);
  write(5, m_bufferAllocator.getDeviceAddress(slot.poseHistoryBuffer),
        paletteBytes);
  slot.poseHistoryReady = true;
}
void RendererBackend::recordPoseCompute(VkCommandBuffer commandBuffer) {
  if (m_posePipeline == VK_NULL_HANDLE)
    return;
  const auto barrier = [&](VkPipelineStageFlags2 destination,
                           VkAccessFlags2 access) {
    VkMemoryBarrier2 memory{};
    memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    memory.dstStageMask = destination;
    memory.dstAccessMask = access;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
  };
  std::uint32_t maxNodes = 0, maxDepth = 0;
  bool haveHistory = false;
  for (const auto &slot : m_skinningInstances)
    if (slot.visible && slot.poseHistoryReady) {
      haveHistory = true;
      if (slot.poseComputeReady) {
        maxNodes = std::max(maxNodes,
                            static_cast<std::uint32_t>(
                                slot.pendingEvaluation->instructions.size()));
        maxDepth = std::max(maxDepth, slot.poseResource.depthCount);
      }
    }
  if (!haveHistory)
    return;
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    m_posePipeline);
  const auto dispatch = [&](SkinnedInstanceSlot &slot, std::uint32_t phase,
                            std::uint32_t node, std::uint32_t depth) {
    bindDescriptorBuffer(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                         m_posePipelineLayout, 0, slot.poseBufferSet,
                         m_currentFrame);
    PosePush push{slot.boneCount,
                  slot.poseComputeReady
                      ? static_cast<std::uint32_t>(
                            slot.pendingEvaluation->instructions.size())
                      : 0u,
                  slot.poseComputeReady ? slot.pendingEvaluation->output : 0u,
                  phase,
                  depth,
                  slot.poseHistoryValid ? 1u : 0u,
                  0,
                  node};
    vkCmdPushConstants(commandBuffer, m_posePipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer, phase == 2 ? 1u : (slot.boneCount + 63) / 64,
                  1, 1);
  };
  // Independent actors share each scheduling barrier. Their disjoint frame
  // slices allow the GPU to overlap actor work within a node/depth batch.
  for (std::uint32_t node = 0; node < maxNodes; ++node) {
    barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    for (auto &slot : m_skinningInstances)
      if (slot.visible && slot.poseComputeReady && slot.poseHistoryReady &&
          node < slot.pendingEvaluation->instructions.size())
        dispatch(slot, 0, node, 0);
    bool modifiers = false;
    for (const auto &slot : m_skinningInstances)
      if (slot.visible && slot.poseComputeReady && slot.poseHistoryReady &&
          node < slot.pendingEvaluation->instructions.size()) {
        const auto &instruction = slot.pendingEvaluation->instructions[node];
        modifiers |= instruction.postProcess &&
                     !instruction.postProcess->modifiers.empty();
      }
    if (modifiers) {
      barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
      for (auto &slot : m_skinningInstances)
        if (slot.visible && slot.poseComputeReady && slot.poseHistoryReady &&
            node < slot.pendingEvaluation->instructions.size()) {
          const auto &instruction = slot.pendingEvaluation->instructions[node];
          if (instruction.postProcess &&
              !instruction.postProcess->modifiers.empty())
            dispatch(slot, 2, node, 0);
        }
    }
  }
  for (std::uint32_t depth = 0; depth < maxDepth; ++depth) {
    barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    for (auto &slot : m_skinningInstances)
      if (slot.visible && slot.poseComputeReady && slot.poseHistoryReady &&
          depth < slot.poseResource.depthCount)
        dispatch(slot, 1, 0, depth);
  }
  barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  for (auto &slot : m_skinningInstances)
    if (slot.visible && slot.poseHistoryReady) {
      dispatch(slot, 3, 0, 0);
      slot.poseHistoryValid = true;
    }
  barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
              VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
          VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
}
} // namespace odai::render
