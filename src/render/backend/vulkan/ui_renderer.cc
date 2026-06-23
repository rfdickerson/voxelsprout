#include "render/backend/vulkan/ui_renderer.h"

#include "core/log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <vector>

namespace odai::render {

namespace {

struct UiPushConstants {
    float invScreenSize[2];
    float pad[2];
};

// Edge-extend ("alpha bleed") an RGBA8 image: copy color outward into fully
// transparent texels from their nearest opaque neighbours, leaving alpha
// untouched. GPU blit mip generation filters RGB per-channel without
// premultiplying, so transparent-*black* border texels would otherwise bleed
// dark halos into minified icons. This is a one-time preprocess of mip 0, not a
// resample. A bounded pass count keeps it cheap (the filter footprint of the
// sizes we draw at only reaches a few texels past each edge).
void bleedTransparentRgb(std::vector<std::uint8_t>& px, std::uint32_t w, std::uint32_t h) {
    constexpr int kPasses = 16;
    const std::size_t n = static_cast<std::size_t>(w) * h;
    std::vector<std::uint8_t> known(n);
    for (std::size_t i = 0; i < n; ++i) {
        known[i] = px[i * 4u + 3u] > 8u ? 1u : 0u;
    }
    for (int pass = 0; pass < kPasses; ++pass) {
        std::vector<std::uint8_t> filled = known;
        bool any = false;
        for (std::uint32_t y = 0; y < h; ++y) {
            for (std::uint32_t x = 0; x < w; ++x) {
                const std::size_t idx = static_cast<std::size_t>(y) * w + x;
                if (known[idx]) continue;
                std::uint32_t r = 0, g = 0, b = 0, cnt = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int nx = static_cast<int>(x) + dx;
                        const int ny = static_cast<int>(y) + dy;
                        if (nx < 0 || ny < 0 || nx >= static_cast<int>(w) || ny >= static_cast<int>(h)) continue;
                        const std::size_t nidx = static_cast<std::size_t>(ny) * w + nx;
                        if (!known[nidx]) continue;
                        r += px[nidx * 4u + 0u];
                        g += px[nidx * 4u + 1u];
                        b += px[nidx * 4u + 2u];
                        ++cnt;
                    }
                }
                if (cnt != 0) {
                    px[idx * 4u + 0u] = static_cast<std::uint8_t>(r / cnt);
                    px[idx * 4u + 1u] = static_cast<std::uint8_t>(g / cnt);
                    px[idx * 4u + 2u] = static_cast<std::uint8_t>(b / cnt);
                    filled[idx] = 1u;
                    any = true;
                }
            }
        }
        known.swap(filled);
        if (!any) break;
    }
}

}  // namespace

bool UiRenderer::init(const InitInfo& info) {
    m_info = info;
    if (m_info.device == VK_NULL_HANDLE || m_info.bufferAllocator == nullptr ||
        m_info.vmaAllocator == VK_NULL_HANDLE || m_info.maxTextureCount < 2) {
        return false;
    }
    m_maxTextureCount = m_info.maxTextureCount;

    // Sampler shared by all UI textures.
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (vkCreateSampler(m_info.device, &samplerInfo, nullptr, &m_sampler) != VK_SUCCESS) {
        VOX_LOGE("ui") << "failed to create UI sampler";
        return false;
    }

    // One update-after-bind descriptor array for every UI texture. Texture slots
    // are stored in vertex mode bits, so changing icons/fonts does not split a
    // clip batch or bind a descriptor set per draw.
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = m_maxTextureCount;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkDescriptorBindingFlags bindingFlags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = 1;
    bindingFlagsInfo.pBindingFlags = &bindingFlags;
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    layoutInfo.pNext = &bindingFlagsInfo;
    if (vkCreateDescriptorSetLayout(m_info.device, &layoutInfo, nullptr, &m_setLayout) != VK_SUCCESS) {
        VOX_LOGE("ui") << "failed to create UI descriptor set layout";
        return false;
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = m_maxTextureCount;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(m_info.device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        VOX_LOGE("ui") << "failed to create UI descriptor pool";
        return false;
    }

    VkDescriptorSetAllocateInfo setAlloc{};
    setAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAlloc.descriptorPool = m_descriptorPool;
    setAlloc.descriptorSetCount = 1;
    setAlloc.pSetLayouts = &m_setLayout;
    if (vkAllocateDescriptorSets(m_info.device, &setAlloc, &m_descriptorSet) != VK_SUCCESS) {
        VOX_LOGE("ui") << "failed to allocate bindless UI descriptor set";
        return false;
    }

    // Transient command pool for one-time texture uploads.
    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cmdPoolInfo.queueFamilyIndex = m_info.uploadQueueFamily;
    if (vkCreateCommandPool(m_info.device, &cmdPoolInfo, nullptr, &m_uploadPool) != VK_SUCCESS) {
        VOX_LOGE("ui") << "failed to create UI upload command pool";
        return false;
    }

    if (!createPipeline()) {
        return false;
    }

    // Built-in opaque white 1x1 so solid fills share the textured descriptor path.
    const std::array<std::uint8_t, 4> whitePixel{255u, 255u, 255u, 255u};
    Texture white{};
    if (!uploadTexturePixels(whitePixel.data(), 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 4, white)) {
        VOX_LOGE("ui") << "failed to create UI white texture";
        return false;
    }
    if (!writeTextureDescriptor(odai::ui::kUiNoTexture, white.view)) {
        destroyTexture(white);
        VOX_LOGE("ui") << "failed to write UI white texture descriptor";
        return false;
    }
    m_textures[odai::ui::kUiNoTexture] = white;
    return true;
}

bool UiRenderer::createPipeline() {
    VkShaderModule vert = loadShaderModule("ui.vert.slang.spv");
    VkShaderModule frag = loadShaderModule("ui.frag.slang.spv");
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
        if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(m_info.device, vert, nullptr);
        if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(m_info.device, frag, nullptr);
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(UiPushConstants);
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(m_info.device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(m_info.device, vert, nullptr);
        vkDestroyShaderModule(m_info.device, frag, nullptr);
        return false;
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(odai::ui::UiVertex);  // 40 bytes
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::array<VkVertexInputAttributeDescription, 5> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};                              // posPx
    attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT, 8};                              // uv
    attrs[2] = {2, 0, VK_FORMAT_R8G8B8A8_UNORM, 16};                            // rgba8 -> float4
    attrs[3] = {3, 0, VK_FORMAT_R32_UINT, 20};                                  // mode
    attrs[4] = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 24};                       // sdf params
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attrs.size());
    vertexInput.pVertexAttributeDescriptions = attrs.data();

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
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;

    // Straight-alpha blending.
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blend;

    const std::array<VkDynamicState, 2> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &m_info.colorFormat;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = m_pipelineLayout;

    const VkResult result =
        vkCreateGraphicsPipelines(m_info.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline);
    vkDestroyShaderModule(m_info.device, vert, nullptr);
    vkDestroyShaderModule(m_info.device, frag, nullptr);
    if (result != VK_SUCCESS) {
        VOX_LOGE("ui") << "failed to create UI graphics pipeline";
        return false;
    }
    return true;
}

VkShaderModule UiRenderer::loadShaderModule(const std::string& fileName) const {
    const std::string path = m_info.shaderDir + "/" + fileName;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        VOX_LOGE("ui") << "failed to open UI shader " << path;
        return VK_NULL_HANDLE;
    }
    const std::streamoff size = file.tellg();
    if (size <= 0 || (size % 4) != 0) {
        return VK_NULL_HANDLE;
    }
    std::vector<std::uint32_t> code(static_cast<std::size_t>(size) / 4u);
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(code.data()), size)) {
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = static_cast<std::size_t>(size);
    moduleInfo.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_info.device, &moduleInfo, nullptr, &module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return module;
}

bool UiRenderer::uploadTexturePixels(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height,
                                     VkFormat format, std::uint32_t bytesPerPixel, Texture& outTexture) {
    const VkDeviceSize byteCount = static_cast<VkDeviceSize>(width) * height * bytesPerPixel;

    BufferCreateDesc stagingDesc{};
    stagingDesc.size = byteCount;
    stagingDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    stagingDesc.initialData = pixels;
    const BufferHandle staging = m_info.bufferAllocator->createBuffer(stagingDesc);
    if (staging == kInvalidBufferHandle) {
        return false;
    }

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(m_info.vmaAllocator, &imageInfo, &allocInfo, &outTexture.image, &outTexture.allocation,
                       nullptr) != VK_SUCCESS) {
        m_info.bufferAllocator->destroyBuffer(staging);
        return false;
    }

    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = m_uploadPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_info.device, &cmdAlloc, &cmd);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier toCopy{};
    toCopy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toCopy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toCopy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toCopy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toCopy.image = outTexture.image;
    toCopy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toCopy.srcAccessMask = 0;
    toCopy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &toCopy);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, m_info.bufferAllocator->getBuffer(staging), outTexture.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier toShader = toCopy;
    toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &toShader);

    vkEndCommandBuffer(cmd);
    if (!submitUpload(cmd, staging)) {
        return false;
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = outTexture.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // Identity swizzle: for R8_UNORM the shader's `.r` read returns the stored
    // glyph coverage directly. (An earlier swizzle mapped R->ONE, which made every
    // glyph texel sample as fully opaque -- text rendered as solid blocks.)
    if (vkCreateImageView(m_info.device, &viewInfo, nullptr, &outTexture.view) != VK_SUCCESS) {
        return false;
    }
    return true;
}

bool UiRenderer::submitUpload(VkCommandBuffer commandBuffer, BufferHandle stagingBuffer) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(m_info.device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(m_info.device, m_uploadPool, 1, &commandBuffer);
        m_info.bufferAllocator->destroyBuffer(stagingBuffer);
        return false;
    }
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(m_info.uploadQueue, 1, &submit, fence) != VK_SUCCESS) {
        vkDestroyFence(m_info.device, fence, nullptr);
        vkFreeCommandBuffers(m_info.device, m_uploadPool, 1, &commandBuffer);
        m_info.bufferAllocator->destroyBuffer(stagingBuffer);
        return false;
    }
    m_pendingUploads.push_back({stagingBuffer, commandBuffer, fence});
    return true;
}

void UiRenderer::collectCompletedUploads() {
    auto out = m_pendingUploads.begin();
    for (auto it = m_pendingUploads.begin(); it != m_pendingUploads.end(); ++it) {
        const VkResult status = vkGetFenceStatus(m_info.device, it->fence);
        if (status == VK_NOT_READY) {
            *out++ = *it;
            continue;
        }
        if (status != VK_SUCCESS) {
            VOX_LOGW("ui") << "UI texture upload fence returned " << status;
        }
        vkDestroyFence(m_info.device, it->fence, nullptr);
        vkFreeCommandBuffers(m_info.device, m_uploadPool, 1, &it->commandBuffer);
        m_info.bufferAllocator->destroyBuffer(it->stagingBuffer);
    }
    m_pendingUploads.erase(out, m_pendingUploads.end());
}

bool UiRenderer::writeTextureDescriptor(odai::ui::UiTextureId slot, VkImageView view) {
    if (slot >= m_maxTextureCount || m_descriptorSet == VK_NULL_HANDLE || view == VK_NULL_HANDLE) {
        return false;
    }
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = m_sampler;
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(m_info.device, 1, &write, 0, nullptr);
    return true;
}

odai::ui::UiTextureId UiRenderer::registerTextureRgba8(const std::uint8_t* pixels, std::uint32_t width,
                                                       std::uint32_t height) {
    Texture texture{};
    // sRGB so the sampler returns linear values, matching the UI shader (which no
    // longer linearizes textured samples). Color PNGs are authored in sRGB.
    if (!uploadTexturePixels(pixels, width, height, VK_FORMAT_R8G8B8A8_SRGB, 4, texture)) {
        return odai::ui::kUiNoTexture;
    }
    if (m_nextTextureId >= m_maxTextureCount) {
        destroyTexture(texture);
        VOX_LOGW("ui") << "bindless UI texture table exhausted";
        return odai::ui::kUiNoTexture;
    }
    const odai::ui::UiTextureId id = m_nextTextureId++;
    if (!writeTextureDescriptor(id, texture.view)) {
        destroyTexture(texture);
        return odai::ui::kUiNoTexture;
    }
    m_textures[id] = texture;
    return id;
}

bool UiRenderer::uploadTexturePixelsMipmapped(const std::uint8_t* pixels, std::uint32_t width,
                                               std::uint32_t height, Texture& outTexture) {
    if (width == 0 || height == 0 || pixels == nullptr) return false;

    const std::uint32_t mipLevels = std::bit_width(std::max(width, height));

    // sRGB image format: vkCmdBlitImage's linear filter decodes to linear,
    // filters, and re-encodes to sRGB, so the GPU-generated mip chain is
    // gamma-correct. The sampler reads it back through an sRGB view (returns
    // linear) -- the fragment shader does not re-linearize textured samples.
    constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_SRGB;

    // Stage only mip 0; the GPU derives the rest by successive half-res blits.
    // Alpha-bleed the base level first so blit filtering never averages in the
    // transparent-black border (kept alive until createBuffer copies it).
    const VkDeviceSize level0Bytes = static_cast<VkDeviceSize>(width) * height * 4u;
    std::vector<std::uint8_t> base(pixels, pixels + static_cast<std::size_t>(level0Bytes));
    bleedTransparentRgb(base, width, height);

    BufferCreateDesc stagingDesc{};
    stagingDesc.size = level0Bytes;
    stagingDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    stagingDesc.initialData = base.data();
    const BufferHandle staging = m_info.bufferAllocator->createBuffer(stagingDesc);
    if (staging == kInvalidBufferHandle) return false;

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = kFormat;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(m_info.vmaAllocator, &imageInfo, &allocInfo, &outTexture.image, &outTexture.allocation,
                       nullptr) != VK_SUCCESS) {
        m_info.bufferAllocator->destroyBuffer(staging);
        return false;
    }

    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = m_uploadPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_info.device, &cmdAlloc, &cmd);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Single-mip layout-transition helper.
    const auto barrierMip = [&](std::uint32_t baseMip, VkImageLayout oldL, VkImageLayout newL,
                                VkAccessFlags srcA, VkAccessFlags dstA, VkPipelineStageFlags srcS,
                                VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = oldL;
        b.newLayout = newL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = outTexture.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, 1, 0, 1};
        b.srcAccessMask = srcA;
        b.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // Upload mip 0.
    barrierMip(0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, m_info.bufferAllocator->getBuffer(staging), outTexture.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    // Blit chain: each level i is filtered down from level i-1. After serving as
    // a blit source, level i-1 is moved to its final SHADER_READ layout.
    std::int32_t mw = static_cast<std::int32_t>(width);
    std::int32_t mh = static_cast<std::int32_t>(height);
    for (std::uint32_t i = 1; i < mipLevels; ++i) {
        barrierMip(i - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        barrierMip(i, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);

        const std::int32_t nw = std::max(1, mw / 2);
        const std::int32_t nh = std::max(1, mh / 2);
        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mw, mh, 1};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {nw, nh, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        vkCmdBlitImage(cmd, outTexture.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, outTexture.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        barrierMip(i - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        mw = nw;
        mh = nh;
    }
    // The final (smallest) level is still TRANSFER_DST -> move it to SHADER_READ.
    barrierMip(mipLevels - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    vkEndCommandBuffer(cmd);
    if (!submitUpload(cmd, staging)) {
        return false;
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = outTexture.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = kFormat;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};
    if (vkCreateImageView(m_info.device, &viewInfo, nullptr, &outTexture.view) != VK_SUCCESS) {
        return false;
    }
    return true;
}

odai::ui::UiTextureId UiRenderer::registerTextureRgba8Mipmapped(const std::uint8_t* pixels,
                                                                  std::uint32_t width,
                                                                  std::uint32_t height) {
    Texture texture{};
    if (!uploadTexturePixelsMipmapped(pixels, width, height, texture)) {
        return odai::ui::kUiNoTexture;
    }
    if (m_nextTextureId >= m_maxTextureCount) {
        destroyTexture(texture);
        VOX_LOGW("ui") << "bindless UI texture table exhausted";
        return odai::ui::kUiNoTexture;
    }
    const odai::ui::UiTextureId id = m_nextTextureId++;
    if (!writeTextureDescriptor(id, texture.view)) {
        destroyTexture(texture);
        return odai::ui::kUiNoTexture;
    }
    m_textures[id] = texture;
    return id;
}

bool UiRenderer::setFontAtlasR8(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height) {
    const auto existing = m_textures.find(odai::ui::kUiFontAtlas);
    if (existing != m_textures.end()) {
        destroyTexture(existing->second);
        m_textures.erase(existing);
    }
    Texture texture{};
    if (!uploadTexturePixels(pixels, width, height, VK_FORMAT_R8_UNORM, 1, texture)) {
        return false;
    }
    if (!writeTextureDescriptor(odai::ui::kUiFontAtlas, texture.view)) {
        destroyTexture(texture);
        return false;
    }
    m_textures[odai::ui::kUiFontAtlas] = texture;
    return true;
}

odai::ui::UiTextureId UiRenderer::registerFontAtlasR8(const std::uint8_t* pixels, std::uint32_t width,
                                                      std::uint32_t height) {
    Texture texture{};
    if (!uploadTexturePixels(pixels, width, height, VK_FORMAT_R8_UNORM, 1, texture)) {
        return odai::ui::kUiNoTexture;
    }
    if (m_nextTextureId >= m_maxTextureCount) {
        destroyTexture(texture);
        VOX_LOGW("ui") << "bindless UI texture table exhausted";
        return odai::ui::kUiNoTexture;
    }
    const odai::ui::UiTextureId id = m_nextTextureId++;
    // Bind the atlas into the bindless table. Without this the descriptor slot
    // stays unwritten, and the first glyph drawn from this font samples an
    // unbound partially-bound descriptor -> GPU fault / VK_ERROR_DEVICE_LOST.
    if (!writeTextureDescriptor(id, texture.view)) {
        destroyTexture(texture);
        return odai::ui::kUiNoTexture;
    }
    m_textures[id] = texture;
    return id;
}

void UiRenderer::uploadGeometry(VkCommandBuffer cmd, FrameArena& frameArena,
                                 const odai::ui::UiDrawData& drawData) {
    m_geometryReady = false;
    collectCompletedUploads();
    if (m_pipeline == VK_NULL_HANDLE || drawData.commands.empty()) return;

    const VkDeviceSize vertexBytes = drawData.vertices.size() * sizeof(odai::ui::UiVertex);
    const VkDeviceSize indexBytes  = drawData.indices.size()  * sizeof(std::uint32_t);
    const auto vertexSlice = frameArena.allocateUpload(vertexBytes, 16, FrameArenaUploadKind::UiGeometry);
    const auto indexSlice  = frameArena.allocateUpload(indexBytes,  4,  FrameArenaUploadKind::UiGeometry);
    if (!vertexSlice || !indexSlice) {
        VOX_LOGW("ui") << "frame arena exhausted; dropping UI geometry this frame";
        return;
    }
    std::memcpy(vertexSlice->mapped, drawData.vertices.data(), static_cast<std::size_t>(vertexBytes));
    std::memcpy(indexSlice->mapped,  drawData.indices.data(),  static_cast<std::size_t>(indexBytes));

    m_uploadedVertexBuffer = m_info.bufferAllocator->getBuffer(vertexSlice->buffer);
    m_uploadedIndexBuffer  = m_info.bufferAllocator->getBuffer(indexSlice->buffer);
    m_uploadedVertexOffset = vertexSlice->offset;
    m_uploadedIndexOffset  = indexSlice->offset;

    // HOST→VERTEX barrier: must happen outside the dynamic rendering pass
    // (VERTEX_INPUT_BIT is not a framebuffer-space stage).
    VkMemoryBarrier2 hostToVertex{};
    hostToVertex.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    hostToVertex.srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
    hostToVertex.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    hostToVertex.dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    hostToVertex.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &hostToVertex;
    vkCmdPipelineBarrier2(cmd, &dependency);
    m_geometryReady = true;
}

void UiRenderer::record(VkCommandBuffer cmd, std::uint32_t /*frameIndex*/,
                         const odai::ui::UiDrawData& drawData, VkExtent2D extent) {
    m_stats = {};
    m_stats.textureSlots = static_cast<std::uint32_t>(m_textures.size());
    m_stats.commandCount = static_cast<std::uint32_t>(drawData.commands.size());
    if (!m_geometryReady || m_pipeline == VK_NULL_HANDLE ||
        drawData.commands.empty() || extent.width == 0 || extent.height == 0) {
        return;
    }
    m_stats.dynamicUploadBytes =
        drawData.vertices.size() * sizeof(odai::ui::UiVertex) +
        drawData.indices.size()  * sizeof(std::uint32_t);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_uploadedVertexBuffer, &m_uploadedVertexOffset);
    vkCmdBindIndexBuffer(cmd, m_uploadedIndexBuffer, m_uploadedIndexOffset, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1,
                            &m_descriptorSet, 0, nullptr);

    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    UiPushConstants push{};
    push.invScreenSize[0] = 2.0f / static_cast<float>(extent.width);
    push.invScreenSize[1] = 2.0f / static_cast<float>(extent.height);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);

    for (const odai::ui::UiDrawCmd& command : drawData.commands) {
        if (command.indexCount == 0) {
            ++m_stats.skippedDrawCalls;
            continue;
        }
        const float clipMinX = std::max(command.clipRect.minX, 0.0f);
        const float clipMinY = std::max(command.clipRect.minY, 0.0f);
        const float clipMaxX = std::min(command.clipRect.maxX, static_cast<float>(extent.width));
        const float clipMaxY = std::min(command.clipRect.maxY, static_cast<float>(extent.height));
        if (clipMaxX <= clipMinX || clipMaxY <= clipMinY) {
            ++m_stats.skippedDrawCalls;
            continue;
        }
        VkRect2D scissor{};
        scissor.offset = {static_cast<std::int32_t>(clipMinX), static_cast<std::int32_t>(clipMinY)};
        scissor.extent = {static_cast<std::uint32_t>(clipMaxX - clipMinX),
                          static_cast<std::uint32_t>(clipMaxY - clipMinY)};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDrawIndexed(cmd, command.indexCount, 1, command.indexOffset, 0, 0);
        ++m_stats.drawCallCount;
    }
}

void UiRenderer::destroyTexture(Texture& texture) {
    if (texture.view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_info.device, texture.view, nullptr);
        texture.view = VK_NULL_HANDLE;
    }
    if (texture.image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_info.vmaAllocator, texture.image, texture.allocation);
        texture.image = VK_NULL_HANDLE;
        texture.allocation = VK_NULL_HANDLE;
    }
}

void UiRenderer::shutdown() {
    if (m_info.device == VK_NULL_HANDLE) {
        return;
    }
    if (!m_pendingUploads.empty()) {
        vkQueueWaitIdle(m_info.uploadQueue);
        collectCompletedUploads();
    }
    for (auto& [id, texture] : m_textures) {
        destroyTexture(texture);
    }
    m_textures.clear();
    if (m_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_info.device, m_pipeline, nullptr);
    if (m_pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_info.device, m_pipelineLayout, nullptr);
    if (m_descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_info.device, m_descriptorPool, nullptr);
    if (m_setLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_info.device, m_setLayout, nullptr);
    if (m_uploadPool != VK_NULL_HANDLE) vkDestroyCommandPool(m_info.device, m_uploadPool, nullptr);
    if (m_sampler != VK_NULL_HANDLE) vkDestroySampler(m_info.device, m_sampler, nullptr);
    m_pipeline = VK_NULL_HANDLE;
    m_pipelineLayout = VK_NULL_HANDLE;
    m_descriptorPool = VK_NULL_HANDLE;
    m_descriptorSet = VK_NULL_HANDLE;
    m_setLayout = VK_NULL_HANDLE;
    m_uploadPool = VK_NULL_HANDLE;
    m_sampler = VK_NULL_HANDLE;
    m_maxTextureCount = 0;
}

}  // namespace odai::render
