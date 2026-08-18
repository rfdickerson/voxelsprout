#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>


namespace odai::render {

#include "render/renderer_shared.h"

namespace {

struct SkinnedVelocityPushConstants {
    float prevViewProj[16] = {};
    float viewProj[16] = {};
    float jitterCurrPrev[4] = {};
    std::uint32_t boneCount = 0;
    std::uint32_t pad0 = 0;
    std::uint32_t pad1 = 0;
    std::uint32_t pad2 = 0;
};

}  // namespace

// Pipeline for the skinned-actor motion vector pass.
//
// Not fatal if it fails to build: without it the velocity target simply stays
// cleared, every pixel reads as "no dynamic motion here", and consumers fall
// back to reprojecting depth through prevViewProj -- which is exactly the
// behaviour that existed before this pass, so the failure mode is "no better",
// not "broken".
bool RendererBackend::createSkinnedVelocityResources() {
    constexpr const char* kVertexShaderPath =
        "../src/render/shaders/skinned_velocity.vert.slang.spv";
    constexpr const char* kFragmentShaderPath =
        "../src/render/shaders/skinned_velocity.frag.slang.spv";
    if (!std::filesystem::exists(kVertexShaderPath) || !std::filesystem::exists(kFragmentShaderPath)) {
        VOX_LOGW("render") << "skinned velocity shaders missing; motion vectors disabled";
        return false;
    }

    if (m_skinnedVelocityDescriptorSetLayout == VK_NULL_HANDLE) {
        const auto storageBinding = [](uint32_t binding) {
            VkDescriptorSetLayoutBinding layoutBinding{};
            layoutBinding.binding = binding;
            layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            layoutBinding.descriptorCount = 1;
            layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            return layoutBinding;
        };
        const std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
            storageBinding(0),  // this frame's bone matrices
            storageBinding(1),  // last frame's
        };
        if (!createDescriptorSetLayout(
                bindings, m_skinnedVelocityDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(skinnedVelocity)",
                "renderer.descriptorSetLayout.skinnedVelocity", nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
            destroySkinnedVelocityResources();
            return false;
        }
    }

    std::array<VkShaderModule, 2> shaderModules = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    if (!createShaderModuleFromFile(
            m_device, kVertexShaderPath, "skinned_velocity.vert", shaderModules[0]) ||
        !createShaderModuleFromFile(
            m_device, kFragmentShaderPath, "skinned_velocity.frag", shaderModules[1])) {
        destroyShaderModules(m_device, shaderModules);
        destroySkinnedVelocityResources();
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(SkinnedVelocityPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_skinnedVelocityDescriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(
            m_device, &layoutInfo, nullptr, &m_skinnedVelocityPipelineLayout) != VK_SUCCESS) {
        VOX_LOGW("render") << "skinned velocity pipeline layout creation failed";
        destroyShaderModules(m_device, shaderModules);
        destroySkinnedVelocityResources();
        return false;
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = shaderModules[0];
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = shaderModules[1];
    stages[1].pName = "main";

    // The rest pose bound as a vertex buffer. Offsets are GpuSkinnedVertexIn's,
    // which is 84 tightly packed bytes -- see the shader header for why this is
    // a vertex binding rather than a StructuredBuffer.
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = 84u;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::array<VkVertexInputAttributeDescription, 3> attributes{};
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = 0u;   // position
    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32A32_UINT;
    attributes[1].offset = 52u;  // boneIndices
    attributes[2].location = 2;
    attributes[2].binding = 0;
    attributes[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributes[2].offset = 68u;  // boneWeights

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
    vertexInput.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Tests against the depth the main pass already wrote, and writes none of
    // its own: an actor pixel that lost the depth test is hidden, and writing a
    // motion vector for it would hand the consumer the velocity of a surface
    // that is not on screen.
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    const std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &m_velocityFormat;
    renderingInfo.depthAttachmentFormat = m_depthFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_skinnedVelocityPipelineLayout;

    const VkResult result = vkCreateGraphicsPipelines(
        m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_skinnedVelocityPipeline);
    destroyShaderModules(m_device, shaderModules);
    if (result != VK_SUCCESS) {
        VOX_LOGW("render") << "skinned velocity pipeline creation failed; motion vectors disabled";
        destroySkinnedVelocityResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(m_skinnedVelocityPipeline),
        "pipeline.skinnedVelocity");
    VOX_LOGI("render") << "skinned velocity (motion vector) pipeline ready";
    return true;
}

void RendererBackend::destroySkinnedVelocityResources() {
    if (m_skinnedVelocityPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_skinnedVelocityPipeline, nullptr);
        m_skinnedVelocityPipeline = VK_NULL_HANDLE;
    }
    if (m_skinnedVelocityPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_skinnedVelocityPipelineLayout, nullptr);
        m_skinnedVelocityPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_skinnedVelocityDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_skinnedVelocityDescriptorSetLayout, nullptr);
        m_skinnedVelocityDescriptorSetLayout = VK_NULL_HANDLE;
    }
}

// Draws every posed skinned actor into the velocity target.
//
// Runs AFTER the main pass so it can depth-test against what actually ended up
// visible, and clears the target itself -- a cleared texel means "no dynamic
// motion here", which is the signal consumers use to fall back to reprojecting
// depth. The world is static, so that fallback covers everything this pass does
// not draw.
void RendererBackend::recordSkinnedVelocityPass(const FrameExecutionContext& context) {
    VkCommandBuffer commandBuffer = context.commandBuffer;
    const uint32_t aoFrameIndex = context.aoFrameIndex;
    if (m_skinnedVelocityPipeline == VK_NULL_HANDLE ||
        aoFrameIndex >= m_velocityImages.size() ||
        m_velocityImages[aoFrameIndex] == VK_NULL_HANDLE) {
        return;
    }

    if (context.gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(
            commandBuffer, VK_PIPELINE_STAGE_2_NONE, context.gpuTimestampQueryPool,
            kGpuTimestampQueryVelocityStart);
    }
    beginDebugLabel(commandBuffer, "Pass: Skinned Velocity", 0.36f, 0.28f, 0.44f, 1.0f);

    const bool velocityInitialized = m_velocityImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer, m_velocityImages[aoFrameIndex],
        velocityInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        velocityInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        velocityInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    VkClearValue clearValue{};
    clearValue.color.float32[0] = 0.0f;
    clearValue.color.float32[1] = 0.0f;
    clearValue.color.float32[2] = 0.0f;  // validity flag: 0 = nothing drew here
    clearValue.color.float32[3] = 0.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = m_velocityImageViews[aoFrameIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearValue;

    // LOAD, never clear: this is the depth the main pass just wrote, and the
    // whole point is to reject actor pixels that lost to it.
    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = m_depthImageViews[context.imageIndex];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = m_renderExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &context.viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &context.scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skinnedVelocityPipeline);

    SkinnedVelocityPushConstants pushConstants{};
    const odai::math::Matrix4 viewProjT = transpose(m_velocityCurrentViewProj);
    const odai::math::Matrix4 prevViewProjT = transpose(
        m_velocityPrevValid ? m_velocityPrevViewProj : m_velocityCurrentViewProj);
    std::memcpy(pushConstants.viewProj, &viewProjT, sizeof(pushConstants.viewProj));
    std::memcpy(pushConstants.prevViewProj, &prevViewProjT, sizeof(pushConstants.prevViewProj));
    pushConstants.jitterCurrPrev[0] = m_velocityCurrentJitter[0];
    pushConstants.jitterCurrPrev[1] = m_velocityCurrentJitter[1];
    pushConstants.jitterCurrPrev[2] = m_velocityPrevValid ? m_velocityPrevJitter[0] : m_velocityCurrentJitter[0];
    pushConstants.jitterCurrPrev[3] = m_velocityPrevValid ? m_velocityPrevJitter[1] : m_velocityCurrentJitter[1];

    for (std::uint32_t i = 0; i < m_skinningActiveInstanceCount; ++i) {
        const SkinnedInstanceSlot& slot = m_skinningInstances[i];
        if (!slot.visible || slot.vertexCount == 0 || slot.boneCount == 0 ||
            slot.currentBoneAddress == 0 ||
            !slot.velocityBufferSet.valid() ||
            slot.restPoseVertexBufferHandle == kInvalidBufferHandle ||
            slot.indexBufferHandle == kInvalidBufferHandle) {
            continue;
        }
        pushConstants.boneCount = slot.boneCount;
        vkCmdPushConstants(
            commandBuffer, m_skinnedVelocityPipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
            sizeof(pushConstants), &pushConstants);
        bindDescriptorBuffer(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skinnedVelocityPipelineLayout,
            0, slot.velocityBufferSet, m_currentFrame);

        const VkBuffer vertexBuffers[1] = {
            m_bufferAllocator.getBuffer(slot.restPoseVertexBufferHandle)};
        const VkDeviceSize vertexOffsets[1] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);
        vkCmdBindIndexBuffer(
            commandBuffer, m_bufferAllocator.getBuffer(slot.indexBufferHandle), 0,
            VK_INDEX_TYPE_UINT32);
        for (const ImportedMeshDraw& draw : slot.meshDraws) {
            if (draw.indexCount == 0) {
                continue;
            }
            vkCmdDrawIndexed(commandBuffer, draw.indexCount, 1, draw.firstIndex, 0, 0);
        }
    }

    vkCmdEndRendering(commandBuffer);

    transitionImageLayout(
        commandBuffer, m_velocityImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    m_velocityImageInitialized[aoFrameIndex] = true;

    endDebugLabel(commandBuffer);
    if (context.gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(
            commandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, context.gpuTimestampQueryPool,
            kGpuTimestampQueryVelocityEnd);
    }
}

}  // namespace odai::render
