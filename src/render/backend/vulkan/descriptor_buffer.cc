#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include "core/log.h"
#include "sim/network_procedural.h"

#include <array>
#include <cstring>

namespace odai::render {

#include "render/renderer_shared.h"

namespace {

// Round `value` up to the next multiple of `alignment` (alignment is a power of two
// per the Vulkan spec for descriptorBufferOffsetAlignment).
VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0) {
        return value;
    }
    return (value + (alignment - 1)) & ~(alignment - 1);
}

} // namespace

VkDeviceSize RendererBackend::descriptorBufferBindingOffset(
    VkDescriptorSetLayout layout, uint32_t binding) const {
    if (m_getDescriptorSetLayoutBindingOffset == nullptr || layout == VK_NULL_HANDLE) {
        return 0;
    }
    VkDeviceSize offset = 0;
    m_getDescriptorSetLayoutBindingOffset(m_device, layout, binding, &offset);
    return offset;
}

bool RendererBackend::createDescriptorBufferSet(
    VkDescriptorSetLayout layout,
    uint32_t regionCount,
    VkBufferUsageFlags usageFlags,
    const char* debugName,
    DescriptorBufferSet& outSet
) {
    outSet = {};
    if (!m_descriptorBufferReady || layout == VK_NULL_HANDLE || regionCount == 0) {
        VOX_LOGE("render") << "createDescriptorBufferSet: descriptor buffers unavailable or bad args ("
                           << (debugName != nullptr ? debugName : "<null>") << ")\n";
        return false;
    }

    VkDeviceSize layoutSize = 0;
    m_getDescriptorSetLayoutSize(m_device, layout, &layoutSize);
    if (layoutSize == 0) {
        VOX_LOGE("render") << "createDescriptorBufferSet: zero layout size (" << debugName << ")\n";
        return false;
    }

    // Each per-region copy must start on descriptorBufferOffsetAlignment so
    // vkCmdSetDescriptorBufferOffsetsEXT can address it.
    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        m_descriptorBufferProperties.descriptorBufferOffsetAlignment, 1u);
    const VkDeviceSize regionStride = alignUp(layoutSize, alignment);

    BufferCreateDesc bufferDesc{};
    bufferDesc.size = regionStride * static_cast<VkDeviceSize>(regionCount);
    bufferDesc.usage = usageFlags | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferDesc.memoryProperties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const BufferHandle handle = m_bufferAllocator.createBuffer(bufferDesc);
    if (handle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "createDescriptorBufferSet: buffer alloc failed (" << debugName << ")\n";
        return false;
    }

    void* mapped = m_bufferAllocator.mapBuffer(handle, 0, bufferDesc.size);
    if (mapped == nullptr) {
        VOX_LOGE("render") << "createDescriptorBufferSet: map failed (" << debugName << ")\n";
        m_bufferAllocator.destroyBuffer(handle);
        return false;
    }
    std::memset(mapped, 0, static_cast<size_t>(bufferDesc.size));

    outSet.buffer = handle;
    outSet.baseAddress = m_bufferAllocator.getDeviceAddress(handle);
    outSet.mapped = static_cast<std::uint8_t*>(mapped);
    outSet.layoutSize = layoutSize;
    outSet.regionStride = regionStride;
    outSet.regionCount = regionCount;
    outSet.usageFlags = usageFlags;

    if (debugName != nullptr) {
        setObjectName(VK_OBJECT_TYPE_BUFFER,
                      vkHandleToUint64(m_bufferAllocator.getBuffer(handle)), debugName);
    }
    return true;
}

void RendererBackend::destroyDescriptorBufferSet(DescriptorBufferSet& set) {
    if (set.buffer != kInvalidBufferHandle) {
        m_bufferAllocator.unmapBuffer(set.buffer);
        m_bufferAllocator.destroyBuffer(set.buffer);
    }
    set = {};
}

void RendererBackend::writeDescriptorBufferUniform(
    const DescriptorBufferSet& set, uint32_t region, VkDeviceSize bindingOffset,
    VkDeviceAddress address, VkDeviceSize range) {
    if (!set.valid() || m_getDescriptor == nullptr || region >= set.regionCount) {
        return;
    }
    VkDescriptorAddressInfoEXT addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT;
    addressInfo.address = address;
    addressInfo.range = range;

    VkDescriptorGetInfoEXT getInfo{};
    getInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    getInfo.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    getInfo.data.pUniformBuffer = &addressInfo;

    std::uint8_t* dst = set.mapped + set.regionOffset(region) + bindingOffset;
    m_getDescriptor(m_device, &getInfo,
                    static_cast<size_t>(m_descriptorBufferProperties.uniformBufferDescriptorSize), dst);
}

void RendererBackend::writeDescriptorBufferStorage(
    const DescriptorBufferSet& set, uint32_t region, VkDeviceSize bindingOffset,
    VkDeviceAddress address, VkDeviceSize range) {
    if (!set.valid() || m_getDescriptor == nullptr || region >= set.regionCount) {
        return;
    }
    VkDescriptorAddressInfoEXT addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT;
    addressInfo.address = address;
    addressInfo.range = range;

    VkDescriptorGetInfoEXT getInfo{};
    getInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    getInfo.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    getInfo.data.pStorageBuffer = &addressInfo;

    std::uint8_t* dst = set.mapped + set.regionOffset(region) + bindingOffset;
    m_getDescriptor(m_device, &getInfo,
                    static_cast<size_t>(m_descriptorBufferProperties.storageBufferDescriptorSize), dst);
}

void RendererBackend::writeDescriptorBufferCombinedImageSampler(
    const DescriptorBufferSet& set, uint32_t region, VkDeviceSize bindingOffset,
    uint32_t arrayElement, VkImageView view, VkSampler sampler, VkImageLayout imageLayout) {
    if (!set.valid() || m_getDescriptor == nullptr || region >= set.regionCount) {
        return;
    }
    const size_t descriptorSize =
        static_cast<size_t>(m_descriptorBufferProperties.combinedImageSamplerDescriptorSize);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = sampler;
    imageInfo.imageView = view;
    imageInfo.imageLayout = imageLayout;

    VkDescriptorGetInfoEXT getInfo{};
    getInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    getInfo.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    getInfo.data.pCombinedImageSampler = &imageInfo;

    std::uint8_t* dst = set.mapped + set.regionOffset(region) + bindingOffset +
                        static_cast<VkDeviceSize>(arrayElement) * descriptorSize;
    m_getDescriptor(m_device, &getInfo, descriptorSize, dst);
}

void RendererBackend::writeDescriptorBufferSampledImage(
    const DescriptorBufferSet& set, uint32_t region, VkDeviceSize bindingOffset,
    VkImageView view, VkImageLayout imageLayout) {
    if (!set.valid() || m_getDescriptor == nullptr || region >= set.regionCount) {
        return;
    }
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = VK_NULL_HANDLE;
    imageInfo.imageView = view;
    imageInfo.imageLayout = imageLayout;

    VkDescriptorGetInfoEXT getInfo{};
    getInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    getInfo.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    getInfo.data.pSampledImage = &imageInfo;

    std::uint8_t* dst = set.mapped + set.regionOffset(region) + bindingOffset;
    m_getDescriptor(m_device, &getInfo,
                    static_cast<size_t>(m_descriptorBufferProperties.sampledImageDescriptorSize), dst);
}

void RendererBackend::writeDescriptorBufferAccelerationStructure(
    const DescriptorBufferSet& set, uint32_t region, VkDeviceSize bindingOffset,
    VkDeviceAddress accelerationStructureAddress) {
    if (!set.valid() || m_getDescriptor == nullptr || region >= set.regionCount) {
        return;
    }
    VkDescriptorGetInfoEXT getInfo{};
    getInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    getInfo.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    getInfo.data.accelerationStructure = accelerationStructureAddress;

    std::uint8_t* dst = set.mapped + set.regionOffset(region) + bindingOffset;
    m_getDescriptor(m_device, &getInfo,
                    static_cast<size_t>(m_descriptorBufferProperties.accelerationStructureDescriptorSize), dst);
}

void RendererBackend::bindDescriptorBuffer(
    VkCommandBuffer commandBuffer, VkPipelineBindPoint bindPoint, VkPipelineLayout layout,
    uint32_t firstSet, const DescriptorBufferSet& set, uint32_t region) {
    if (!set.valid() || m_cmdBindDescriptorBuffers == nullptr) {
        return;
    }
    VkDescriptorBufferBindingInfoEXT bindingInfo{};
    bindingInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    bindingInfo.address = set.baseAddress;
    bindingInfo.usage = set.usageFlags;
    m_cmdBindDescriptorBuffers(commandBuffer, 1, &bindingInfo);

    const uint32_t bufferIndex = 0;
    const VkDeviceSize offset = set.regionOffset(region);
    m_cmdSetDescriptorBufferOffsets(commandBuffer, bindPoint, layout, firstSet, 1, &bufferIndex, &offset);
}

void RendererBackend::writeDescriptorBufferCombinedImageSamplerArray(
    const DescriptorBufferSet& set, uint32_t region, VkDeviceSize bindingOffset,
    uint32_t arrayElement, uint32_t arrayCapacity,
    VkImageView view, VkSampler sampler, VkImageLayout imageLayout) {
    if (!set.valid() || m_getDescriptor == nullptr || region >= set.regionCount) {
        return;
    }
    const size_t combinedSize =
        static_cast<size_t>(m_descriptorBufferProperties.combinedImageSamplerDescriptorSize);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = sampler;
    imageInfo.imageView = view;
    imageInfo.imageLayout = imageLayout;

    VkDescriptorGetInfoEXT getInfo{};
    getInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    getInfo.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    getInfo.data.pCombinedImageSampler = &imageInfo;

    std::uint8_t* regionBase = set.mapped + set.regionOffset(region) + bindingOffset;

    if (m_descriptorBufferProperties.combinedImageSamplerDescriptorSingleArray) {
        // Single contiguous array: element i at i * combinedImageSamplerDescriptorSize.
        m_getDescriptor(m_device, &getInfo, combinedSize,
                        regionBase + static_cast<VkDeviceSize>(arrayElement) * combinedSize);
        return;
    }

    // Split layout (spec): [arrayCapacity image descriptors][arrayCapacity sampler
    // descriptors]. The combined descriptor's first sampledImageDescriptorSize bytes
    // are the image part; the remaining samplerDescriptorSize bytes are the sampler.
    const size_t imageSize = static_cast<size_t>(m_descriptorBufferProperties.sampledImageDescriptorSize);
    const size_t samplerSize = static_cast<size_t>(m_descriptorBufferProperties.samplerDescriptorSize);
    std::array<std::uint8_t, 256> temp{};  // combinedImageSamplerDescriptorSize fits well within 256.
    m_getDescriptor(m_device, &getInfo, combinedSize, temp.data());

    std::uint8_t* imageDst = regionBase + static_cast<VkDeviceSize>(arrayElement) * imageSize;
    std::uint8_t* samplerDst = regionBase +
        static_cast<VkDeviceSize>(arrayCapacity) * imageSize +
        static_cast<VkDeviceSize>(arrayElement) * samplerSize;
    std::memcpy(imageDst, temp.data(), imageSize);
    std::memcpy(samplerDst, temp.data() + imageSize, samplerSize);
}

void RendererBackend::bindGraphicsDescriptorBuffers(VkCommandBuffer commandBuffer) {
    if (!m_mainBufferSet.valid() || m_cmdBindDescriptorBuffers == nullptr) {
        return;
    }
    const bool hasBindless = m_bindlessBufferSet.valid();

    std::array<VkDescriptorBufferBindingInfoEXT, 2> bindingInfos{};
    bindingInfos[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    bindingInfos[0].address = m_mainBufferSet.baseAddress;
    bindingInfos[0].usage = m_mainBufferSet.usageFlags;
    uint32_t bufferCount = 1;
    if (hasBindless) {
        bindingInfos[1].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
        bindingInfos[1].address = m_bindlessBufferSet.baseAddress;
        bindingInfos[1].usage = m_bindlessBufferSet.usageFlags;
        bufferCount = 2;
    }
    m_cmdBindDescriptorBuffers(commandBuffer, bufferCount, bindingInfos.data());

    // set 0 -> buffer 0 at this frame's region; set 1 -> buffer 1 (single region).
    const std::array<uint32_t, 2> bufferIndices = {0u, 1u};
    const std::array<VkDeviceSize, 2> offsets = {
        m_mainBufferSet.regionOffset(m_currentFrame),
        hasBindless ? m_bindlessBufferSet.regionOffset(0u) : 0u
    };
    m_cmdSetDescriptorBufferOffsets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
        0, bufferCount, bufferIndices.data(), offsets.data());
}

void RendererBackend::writeDescriptorBufferStorageImage(
    const DescriptorBufferSet& set, uint32_t region, VkDeviceSize bindingOffset,
    VkImageView view, VkImageLayout imageLayout) {
    if (!set.valid() || m_getDescriptor == nullptr || region >= set.regionCount) {
        return;
    }
    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = VK_NULL_HANDLE;
    imageInfo.imageView = view;
    imageInfo.imageLayout = imageLayout;

    VkDescriptorGetInfoEXT getInfo{};
    getInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    getInfo.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    getInfo.data.pStorageImage = &imageInfo;

    std::uint8_t* dst = set.mapped + set.regionOffset(region) + bindingOffset;
    m_getDescriptor(m_device, &getInfo,
                    static_cast<size_t>(m_descriptorBufferProperties.storageImageDescriptorSize), dst);
}

} // namespace odai::render
