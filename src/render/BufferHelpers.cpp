#include "render/BufferHelpers.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

namespace render {

bool BufferAllocator::init(VkPhysicalDevice physicalDevice, VkDevice device
#if defined(VOXEL_HAS_VMA)
    , VmaAllocator vmaAllocator
#endif
) {
    m_physicalDevice = physicalDevice;
    m_device = device;
#if defined(VOXEL_HAS_VMA)
    m_vmaAllocator = vmaAllocator;
#endif
    m_slots.clear();
    m_freeSlots.clear();

    // Reserve slot 0 as an always-invalid sentinel.
    m_slots.push_back({});
    return m_physicalDevice != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE;
}

void BufferAllocator::shutdown() {
    if (m_device != VK_NULL_HANDLE) {
        for (size_t i = 1; i < m_slots.size(); ++i) {
            if (m_slots[i].inUse) {
#if defined(VOXEL_HAS_VMA)
                if (m_vmaAllocator != VK_NULL_HANDLE && m_slots[i].allocation != VK_NULL_HANDLE) {
                    vmaDestroyBuffer(m_vmaAllocator, m_slots[i].buffer, m_slots[i].allocation);
                } else {
                    vkDestroyBuffer(m_device, m_slots[i].buffer, nullptr);
                    vkFreeMemory(m_device, m_slots[i].memory, nullptr);
                }
#else
                vkDestroyBuffer(m_device, m_slots[i].buffer, nullptr);
                vkFreeMemory(m_device, m_slots[i].memory, nullptr);
#endif
                m_slots[i] = {};
            }
        }
    }

    m_slots.clear();
    m_freeSlots.clear();
    m_physicalDevice = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
#if defined(VOXEL_HAS_VMA)
    m_vmaAllocator = VK_NULL_HANDLE;
#endif
}

BufferHandle BufferAllocator::createBuffer(const BufferCreateDesc& desc) {
    if (m_device == VK_NULL_HANDLE || m_physicalDevice == VK_NULL_HANDLE || desc.size == 0) {
        return kInvalidBufferHandle;
    }

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = desc.size;
    bufferCreateInfo.usage = desc.usage;
    if (desc.queueFamilyIndices != nullptr && desc.queueFamilyIndexCount > 1) {
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferCreateInfo.queueFamilyIndexCount = desc.queueFamilyIndexCount;
        bufferCreateInfo.pQueueFamilyIndices = desc.queueFamilyIndices;
    } else {
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
#if defined(VOXEL_HAS_VMA)
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo{};
    bool persistentMapped = false;
    void* persistentMappedData = nullptr;
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo allocationCreateInfo{};
        allocationCreateInfo.flags = 0;
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        const bool wantsHostAccess =
            (desc.memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 ||
            (desc.usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0;
        if (wantsHostAccess) {
            allocationCreateInfo.flags |=
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
            persistentMapped = true;
        }
        allocationCreateInfo.requiredFlags = desc.memoryProperties;
        if (vmaCreateBuffer(
                m_vmaAllocator,
                &bufferCreateInfo,
                &allocationCreateInfo,
                &buffer,
                &allocation,
                &allocationInfo
            ) != VK_SUCCESS) {
            return kInvalidBufferHandle;
        }
        persistentMappedData = allocationInfo.pMappedData;
        std::cerr
            << "[render] alloc buffer (VMA): size=" << static_cast<unsigned long long>(desc.size)
            << ", usage=0x" << std::hex << static_cast<unsigned int>(desc.usage)
            << ", memProps=0x" << static_cast<unsigned int>(desc.memoryProperties) << std::dec
            << ", persistentMapped=" << (persistentMappedData != nullptr ? "yes" : "no")
            << "\n";
    } else
#endif
    {
        if (vkCreateBuffer(m_device, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS) {
            return kInvalidBufferHandle;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(m_device, buffer, &memoryRequirements);

        const uint32_t memoryTypeIndex = findMemoryTypeIndex(memoryRequirements.memoryTypeBits, desc.memoryProperties);
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            vkDestroyBuffer(m_device, buffer, nullptr);
            return kInvalidBufferHandle;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;

        if (vkAllocateMemory(m_device, &allocateInfo, nullptr, &memory) != VK_SUCCESS) {
            vkDestroyBuffer(m_device, buffer, nullptr);
            return kInvalidBufferHandle;
        }

        if (vkBindBufferMemory(m_device, buffer, memory, 0) != VK_SUCCESS) {
            vkDestroyBuffer(m_device, buffer, nullptr);
            vkFreeMemory(m_device, memory, nullptr);
            return kInvalidBufferHandle;
        }
        std::cerr
            << "[render] alloc buffer (vk): size=" << static_cast<unsigned long long>(desc.size)
            << ", usage=0x" << std::hex << static_cast<unsigned int>(desc.usage)
            << ", memProps=0x" << static_cast<unsigned int>(desc.memoryProperties) << std::dec
            << "\n";
    }

    if (desc.initialData != nullptr) {
        void* mappedData = nullptr;
#if defined(VOXEL_HAS_VMA)
        bool mappedNeedsUnmap = true;
#endif
        const bool mapped =
#if defined(VOXEL_HAS_VMA)
            (m_vmaAllocator != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
                ? (
                    (persistentMappedData != nullptr)
                        ? ((mappedData = persistentMappedData), mappedNeedsUnmap = false, true)
                        : (vmaMapMemory(m_vmaAllocator, allocation, &mappedData) == VK_SUCCESS)
                )
                : (vkMapMemory(m_device, memory, 0, desc.size, 0, &mappedData) == VK_SUCCESS);
#else
            (vkMapMemory(m_device, memory, 0, desc.size, 0, &mappedData) == VK_SUCCESS);
#endif
        if (!mapped) {
#if defined(VOXEL_HAS_VMA)
            if (m_vmaAllocator != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE) {
                vmaDestroyBuffer(m_vmaAllocator, buffer, allocation);
            } else {
                vkDestroyBuffer(m_device, buffer, nullptr);
                vkFreeMemory(m_device, memory, nullptr);
            }
#else
            vkDestroyBuffer(m_device, buffer, nullptr);
            vkFreeMemory(m_device, memory, nullptr);
#endif
            return kInvalidBufferHandle;
        }
        std::memcpy(mappedData, desc.initialData, static_cast<size_t>(desc.size));
        std::cerr << "[render] upload initial buffer data: bytes="
                  << static_cast<unsigned long long>(desc.size) << "\n";
#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE) {
            if (mappedNeedsUnmap) {
                vmaUnmapMemory(m_vmaAllocator, allocation);
            }
        } else {
            vkUnmapMemory(m_device, memory);
        }
#else
        vkUnmapMemory(m_device, memory);
#endif
    }

    uint32_t slotIndex = 0;
    if (!m_freeSlots.empty()) {
        slotIndex = m_freeSlots.back();
        m_freeSlots.pop_back();
        m_slots[slotIndex] = {
            buffer,
#if defined(VOXEL_HAS_VMA)
            allocation,
            persistentMappedData,
            persistentMappedData != nullptr,
#endif
            memory,
            desc.size,
            true
        };
    } else {
        slotIndex = static_cast<uint32_t>(m_slots.size());
        m_slots.push_back({
            buffer,
#if defined(VOXEL_HAS_VMA)
            allocation,
            persistentMappedData,
            persistentMappedData != nullptr,
#endif
            memory,
            desc.size,
            true
        });
    }

    return slotIndex;
}

void BufferAllocator::destroyBuffer(BufferHandle handle) {
    BufferSlot* slot = getSlot(handle);
    if (slot == nullptr) {
        return;
    }

#if defined(VOXEL_HAS_VMA)
    if (m_vmaAllocator != VK_NULL_HANDLE && slot->allocation != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_vmaAllocator, slot->buffer, slot->allocation);
    } else {
        vkDestroyBuffer(m_device, slot->buffer, nullptr);
        vkFreeMemory(m_device, slot->memory, nullptr);
    }
#else
    vkDestroyBuffer(m_device, slot->buffer, nullptr);
    vkFreeMemory(m_device, slot->memory, nullptr);
#endif

    *slot = {};
    m_freeSlots.push_back(handle);
}

VkBuffer BufferAllocator::getBuffer(BufferHandle handle) const {
    const BufferSlot* slot = getSlot(handle);
    return (slot != nullptr) ? slot->buffer : VK_NULL_HANDLE;
}

VkDeviceSize BufferAllocator::getSize(BufferHandle handle) const {
    const BufferSlot* slot = getSlot(handle);
    return (slot != nullptr) ? slot->size : 0;
}

void* BufferAllocator::mapBuffer(BufferHandle handle, VkDeviceSize offset, VkDeviceSize size) {
    BufferSlot* slot = getSlot(handle);
    if (slot == nullptr) {
        return nullptr;
    }

    void* mapped = nullptr;
    const bool mappedOk =
#if defined(VOXEL_HAS_VMA)
        (m_vmaAllocator != VK_NULL_HANDLE && slot->allocation != VK_NULL_HANDLE)
            ? (
                slot->persistentMapped
                    ? ((mapped = slot->mappedData), true)
                    : (vmaMapMemory(m_vmaAllocator, slot->allocation, &mapped) == VK_SUCCESS)
            )
            : (vkMapMemory(m_device, slot->memory, offset, size, 0, &mapped) == VK_SUCCESS);
#else
        (vkMapMemory(m_device, slot->memory, offset, size, 0, &mapped) == VK_SUCCESS);
#endif
    if (!mappedOk) {
        return nullptr;
    }
#if defined(VOXEL_HAS_VMA)
    if (m_vmaAllocator != VK_NULL_HANDLE && slot->allocation != VK_NULL_HANDLE) {
        return static_cast<uint8_t*>(mapped) + offset;
    }
#endif
    return mapped;
}

void BufferAllocator::unmapBuffer(BufferHandle handle) {
    BufferSlot* slot = getSlot(handle);
    if (slot == nullptr) {
        return;
    }
#if defined(VOXEL_HAS_VMA)
    if (m_vmaAllocator != VK_NULL_HANDLE && slot->allocation != VK_NULL_HANDLE) {
        if (slot->persistentMapped) {
            return;
        }
        vmaUnmapMemory(m_vmaAllocator, slot->allocation);
    } else {
        vkUnmapMemory(m_device, slot->memory);
    }
#else
    vkUnmapMemory(m_device, slot->memory);
#endif
}

uint32_t BufferAllocator::findMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags requiredProperties) const {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool typeMatches = (typeBits & (1u << i)) != 0;
        const bool propertiesMatch =
            (memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties;
        if (typeMatches && propertiesMatch) {
            return i;
        }
    }

    return std::numeric_limits<uint32_t>::max();
}

BufferAllocator::BufferSlot* BufferAllocator::getSlot(BufferHandle handle) {
    if (handle == kInvalidBufferHandle || handle >= m_slots.size()) {
        return nullptr;
    }
    BufferSlot& slot = m_slots[handle];
    return slot.inUse ? &slot : nullptr;
}

const BufferAllocator::BufferSlot* BufferAllocator::getSlot(BufferHandle handle) const {
    if (handle == kInvalidBufferHandle || handle >= m_slots.size()) {
        return nullptr;
    }
    const BufferSlot& slot = m_slots[handle];
    return slot.inUse ? &slot : nullptr;
}

bool FrameRingBuffer::init(
    BufferAllocator* allocator,
    VkDeviceSize bytesPerFrame,
    uint32_t frameCount,
    VkBufferUsageFlags usage
) {
    if (allocator == nullptr || bytesPerFrame == 0 || frameCount == 0) {
        return false;
    }

    const VkDeviceSize totalBytes = bytesPerFrame * frameCount;
    BufferCreateDesc createDesc{};
    createDesc.size = totalBytes;
    createDesc.usage = usage;
    createDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    m_handle = allocator->createBuffer(createDesc);
    if (m_handle == kInvalidBufferHandle) {
        return false;
    }

    m_bytesPerFrame = bytesPerFrame;
    m_frameCount = frameCount;
    m_activeFrame = 0;
    m_writeOffset = 0;

    // We map once and keep the pointer alive for cheap per-frame uploads.
    m_mappedBase = static_cast<uint8_t*>(allocator->mapBuffer(m_handle, 0, totalBytes));
    if (m_mappedBase == nullptr) {
        allocator->destroyBuffer(m_handle);
        m_handle = kInvalidBufferHandle;
        return false;
    }
    return true;
}

void FrameRingBuffer::shutdown(BufferAllocator* allocator) {
    if (allocator != nullptr && m_handle != kInvalidBufferHandle) {
        allocator->unmapBuffer(m_handle);
        allocator->destroyBuffer(m_handle);
    }

    m_handle = kInvalidBufferHandle;
    m_mappedBase = nullptr;
    m_bytesPerFrame = 0;
    m_frameCount = 0;
    m_activeFrame = 0;
    m_writeOffset = 0;
}

void FrameRingBuffer::beginFrame(uint32_t frameIndex) {
    if (m_frameCount == 0) {
        return;
    }

    m_activeFrame = frameIndex % m_frameCount;
    m_writeOffset = 0;
}

std::optional<RingBufferSlice> FrameRingBuffer::allocate(VkDeviceSize size, VkDeviceSize alignment) {
    if (m_handle == kInvalidBufferHandle || m_bytesPerFrame == 0 || size == 0) {
        return std::nullopt;
    }

    const VkDeviceSize alignedOffset = alignUp(m_writeOffset, std::max<VkDeviceSize>(alignment, 1));
    if (alignedOffset + size > m_bytesPerFrame) {
        return std::nullopt;
    }

    m_writeOffset = alignedOffset + size;

    RingBufferSlice slice{};
    slice.buffer = m_handle;
    slice.offset = (static_cast<VkDeviceSize>(m_activeFrame) * m_bytesPerFrame) + alignedOffset;
    slice.size = size;
    if (m_mappedBase != nullptr) {
        slice.mapped = m_mappedBase + slice.offset;
    }
    return slice;
}

BufferHandle FrameRingBuffer::handle() const {
    return m_handle;
}

VkDeviceSize FrameRingBuffer::alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

bool FrameArena::init(
    BufferAllocator* allocator,
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    const FrameArenaConfig& config
#if defined(VOXEL_HAS_VMA)
    , VmaAllocator vmaAllocator
#endif
) {
    if (allocator == nullptr ||
        physicalDevice == VK_NULL_HANDLE ||
        device == VK_NULL_HANDLE ||
        config.uploadBytesPerFrame == 0 ||
        config.frameCount == 0) {
        return false;
    }

    m_allocator = allocator;
    m_physicalDevice = physicalDevice;
    m_device = device;
#if defined(VOXEL_HAS_VMA)
    m_vmaAllocator = vmaAllocator;
#endif
    m_frameCount = config.frameCount;
    m_activeFrame = 0;
    m_frameTransientBuffers.assign(m_frameCount, {});
    m_frameTransientImages.assign(m_frameCount, {});
    m_imageSlots.clear();
    m_freeImageSlots.clear();
    m_imageSlots.push_back({});
    m_frameStats.assign(m_frameCount, {});

    if (!m_uploadRing.init(allocator, config.uploadBytesPerFrame, config.frameCount, config.uploadUsage)) {
        m_frameTransientBuffers.clear();
        m_frameTransientImages.clear();
        m_imageSlots.clear();
        m_freeImageSlots.clear();
        m_frameStats.clear();
        m_frameCount = 0;
        m_allocator = nullptr;
        m_physicalDevice = VK_NULL_HANDLE;
        m_device = VK_NULL_HANDLE;
#if defined(VOXEL_HAS_VMA)
        m_vmaAllocator = VK_NULL_HANDLE;
#endif
        return false;
    }

    return true;
}

void FrameArena::shutdown(BufferAllocator* allocator) {
    BufferAllocator* activeAllocator = allocator;
    if (activeAllocator == nullptr) {
        activeAllocator = m_allocator;
    }

    if (activeAllocator != nullptr) {
        for (size_t frameIndex = 0; frameIndex < m_frameTransientBuffers.size(); ++frameIndex) {
            for (const BufferHandle handle : m_frameTransientBuffers[frameIndex]) {
                activeAllocator->destroyBuffer(handle);
            }
            m_frameTransientBuffers[frameIndex].clear();
        }
    }

    for (size_t i = 1; i < m_imageSlots.size(); ++i) {
        if (m_imageSlots[i].inUse) {
            destroyImageSlot(static_cast<TransientImageHandle>(i));
        }
    }

    m_frameTransientBuffers.clear();
    m_frameTransientImages.clear();
    m_imageSlots.clear();
    m_freeImageSlots.clear();
    m_frameStats.clear();
    m_uploadRing.shutdown(activeAllocator);
    m_allocator = nullptr;
    m_physicalDevice = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
#if defined(VOXEL_HAS_VMA)
    m_vmaAllocator = VK_NULL_HANDLE;
#endif
    m_frameCount = 0;
    m_activeFrame = 0;
}

void FrameArena::beginFrame(uint32_t frameIndex) {
    if (m_frameCount == 0) {
        return;
    }

    m_activeFrame = frameIndex % m_frameCount;
    clearFrameTransientBuffers(m_activeFrame);
    clearFrameTransientImages(m_activeFrame);
    if (m_activeFrame < m_frameStats.size()) {
        m_frameStats[m_activeFrame] = {};
    }
    m_uploadRing.beginFrame(frameIndex);
}

std::optional<FrameArenaSlice> FrameArena::allocateUpload(
    VkDeviceSize size,
    VkDeviceSize alignment,
    FrameArenaUploadKind kind
) {
    (void)kind;
    const std::optional<RingBufferSlice> ringSlice = m_uploadRing.allocate(size, alignment);
    if (!ringSlice.has_value()) {
        return std::nullopt;
    }

    if (m_activeFrame < m_frameStats.size()) {
        FrameArenaStats& stats = m_frameStats[m_activeFrame];
        stats.uploadBytesAllocated += ringSlice->size;
        ++stats.uploadAllocationCount;
    }

    FrameArenaSlice slice{};
    slice.buffer = ringSlice->buffer;
    slice.offset = ringSlice->offset;
    slice.size = ringSlice->size;
    slice.mapped = ringSlice->mapped;
    return slice;
}

BufferHandle FrameArena::createTransientBuffer(const BufferCreateDesc& desc) {
    if (m_allocator == nullptr || m_frameCount == 0) {
        return kInvalidBufferHandle;
    }

    const BufferHandle handle = m_allocator->createBuffer(desc);
    if (handle == kInvalidBufferHandle) {
        return kInvalidBufferHandle;
    }

    if (m_activeFrame < m_frameTransientBuffers.size()) {
        m_frameTransientBuffers[m_activeFrame].push_back(handle);
    }
    if (m_activeFrame < m_frameStats.size()) {
        FrameArenaStats& stats = m_frameStats[m_activeFrame];
        stats.transientBufferBytes += desc.size;
        ++stats.transientBufferCount;
    }
    return handle;
}

TransientImageHandle FrameArena::createTransientImage(
    const TransientImageDesc& desc,
    FrameArenaImageLifetime lifetime
) {
    if (m_device == VK_NULL_HANDLE || m_physicalDevice == VK_NULL_HANDLE) {
        return kInvalidTransientImageHandle;
    }
    if (desc.format == VK_FORMAT_UNDEFINED || desc.extent.width == 0 || desc.extent.height == 0 || desc.usage == 0) {
        return kInvalidTransientImageHandle;
    }

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.flags = desc.flags;
    imageCreateInfo.imageType = desc.imageType;
    imageCreateInfo.format = desc.format;
    imageCreateInfo.extent = desc.extent;
    imageCreateInfo.mipLevels = std::max(1u, desc.mipLevels);
    imageCreateInfo.arrayLayers = std::max(1u, desc.arrayLayers);
    imageCreateInfo.samples = desc.samples;
    imageCreateInfo.tiling = desc.tiling;
    imageCreateInfo.usage = desc.usage;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = desc.initialLayout;

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
#if defined(VOXEL_HAS_VMA)
    VmaAllocation allocation = VK_NULL_HANDLE;
#endif

#if defined(VOXEL_HAS_VMA)
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo allocationCreateInfo{};
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (vmaCreateImage(
                m_vmaAllocator,
                &imageCreateInfo,
                &allocationCreateInfo,
                &image,
                &allocation,
                nullptr
            ) != VK_SUCCESS) {
            return kInvalidTransientImageHandle;
        }
    } else
#endif
    {
        if (vkCreateImage(m_device, &imageCreateInfo, nullptr, &image) != VK_SUCCESS) {
            return kInvalidTransientImageHandle;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(m_device, image, &memoryRequirements);
        const uint32_t memoryTypeIndex = findMemoryTypeIndex(
            memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            vkDestroyImage(m_device, image, nullptr);
            return kInvalidTransientImageHandle;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        if (vkAllocateMemory(m_device, &allocateInfo, nullptr, &memory) != VK_SUCCESS) {
            vkDestroyImage(m_device, image, nullptr);
            return kInvalidTransientImageHandle;
        }
        if (vkBindImageMemory(m_device, image, memory, 0) != VK_SUCCESS) {
            vkDestroyImage(m_device, image, nullptr);
            vkFreeMemory(m_device, memory, nullptr);
            return kInvalidTransientImageHandle;
        }
    }

    VkImageViewCreateInfo viewCreateInfo{};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = image;
    viewCreateInfo.viewType = desc.viewType;
    viewCreateInfo.format = desc.format;
    viewCreateInfo.subresourceRange.aspectMask = desc.aspectMask;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = std::max(1u, desc.mipLevels);
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = std::max(1u, desc.arrayLayers);

    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(m_device, &viewCreateInfo, nullptr, &view) != VK_SUCCESS) {
#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, image, allocation);
        } else {
            vkDestroyImage(m_device, image, nullptr);
            vkFreeMemory(m_device, memory, nullptr);
        }
#else
        vkDestroyImage(m_device, image, nullptr);
        vkFreeMemory(m_device, memory, nullptr);
#endif
        return kInvalidTransientImageHandle;
    }

    uint32_t slotIndex = 0;
    if (!m_freeImageSlots.empty()) {
        slotIndex = m_freeImageSlots.back();
        m_freeImageSlots.pop_back();
    } else {
        slotIndex = static_cast<uint32_t>(m_imageSlots.size());
        m_imageSlots.push_back({});
    }

    ImageSlot& slot = m_imageSlots[slotIndex];
    slot.info.image = image;
    slot.info.view = view;
    slot.info.format = desc.format;
    slot.info.extent = desc.extent;
#if defined(VOXEL_HAS_VMA)
    slot.allocation = allocation;
#endif
    slot.memory = memory;
    slot.inUse = true;

    if (lifetime == FrameArenaImageLifetime::FrameTransient && m_activeFrame < m_frameTransientImages.size()) {
        m_frameTransientImages[m_activeFrame].push_back(slotIndex);
    }
    if (m_activeFrame < m_frameStats.size()) {
        FrameArenaStats& stats = m_frameStats[m_activeFrame];
        stats.transientImageBytes +=
            static_cast<uint64_t>(desc.extent.width) *
            static_cast<uint64_t>(desc.extent.height) *
            static_cast<uint64_t>(std::max(1u, desc.extent.depth));
        ++stats.transientImageCount;
    }
    std::cerr
        << "[render] alloc image (FrameArena): extent="
        << desc.extent.width << "x" << desc.extent.height << "x" << desc.extent.depth
        << ", format=" << static_cast<int>(desc.format)
        << ", usage=0x" << std::hex << static_cast<unsigned int>(desc.usage) << std::dec
        << ", lifetime=" << (lifetime == FrameArenaImageLifetime::FrameTransient ? "frame" : "persistent")
        << "\n";
    return slotIndex;
}

void FrameArena::destroyTransientImage(TransientImageHandle handle) {
    destroyImageSlot(handle);
}

const TransientImageInfo* FrameArena::getTransientImage(TransientImageHandle handle) const {
    if (handle == kInvalidTransientImageHandle || handle >= m_imageSlots.size()) {
        return nullptr;
    }
    const ImageSlot& slot = m_imageSlots[handle];
    if (!slot.inUse) {
        return nullptr;
    }
    return &slot.info;
}

BufferHandle FrameArena::uploadBufferHandle() const {
    return m_uploadRing.handle();
}

const FrameArenaStats& FrameArena::activeStats() const {
    if (m_activeFrame < m_frameStats.size()) {
        return m_frameStats[m_activeFrame];
    }
    return m_emptyStats;
}

void FrameArena::clearFrameTransientBuffers(uint32_t frameIndex) {
    if (m_allocator == nullptr || frameIndex >= m_frameTransientBuffers.size()) {
        return;
    }
    for (const BufferHandle handle : m_frameTransientBuffers[frameIndex]) {
        m_allocator->destroyBuffer(handle);
    }
    m_frameTransientBuffers[frameIndex].clear();
}

void FrameArena::clearFrameTransientImages(uint32_t frameIndex) {
    if (frameIndex >= m_frameTransientImages.size()) {
        return;
    }
    for (const TransientImageHandle handle : m_frameTransientImages[frameIndex]) {
        destroyImageSlot(handle);
    }
    m_frameTransientImages[frameIndex].clear();
}

void FrameArena::destroyImageSlot(TransientImageHandle handle) {
    if (handle == kInvalidTransientImageHandle || handle >= m_imageSlots.size()) {
        return;
    }
    ImageSlot& slot = m_imageSlots[handle];
    if (!slot.inUse) {
        return;
    }

    if (slot.info.view != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, slot.info.view, nullptr);
    }
#if defined(VOXEL_HAS_VMA)
    if (m_vmaAllocator != VK_NULL_HANDLE && slot.allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(m_vmaAllocator, slot.info.image, slot.allocation);
    } else
#endif
    if (m_device != VK_NULL_HANDLE) {
        if (slot.info.image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, slot.info.image, nullptr);
        }
        if (slot.memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, slot.memory, nullptr);
        }
    }

    slot = {};
    m_freeImageSlots.push_back(handle);
}

uint32_t FrameArena::findMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags requiredProperties) const {
    if (m_physicalDevice == VK_NULL_HANDLE) {
        return std::numeric_limits<uint32_t>::max();
    }

    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool typeMatches = (typeBits & (1u << i)) != 0;
        const bool propertiesMatch =
            (memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties;
        if (typeMatches && propertiesMatch) {
            return i;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

} // namespace render
