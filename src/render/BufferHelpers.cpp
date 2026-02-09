#include "render/BufferHelpers.hpp"

#include <algorithm>
#include <cstring>
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
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo allocationCreateInfo{};
        allocationCreateInfo.flags = 0;
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        if ((desc.memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            allocationCreateInfo.flags |=
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        allocationCreateInfo.requiredFlags = desc.memoryProperties;
        if (vmaCreateBuffer(
                m_vmaAllocator,
                &bufferCreateInfo,
                &allocationCreateInfo,
                &buffer,
                &allocation,
                nullptr
            ) != VK_SUCCESS) {
            return kInvalidBufferHandle;
        }
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
    }

    if (desc.initialData != nullptr) {
        void* mappedData = nullptr;
        const bool mapped =
#if defined(VOXEL_HAS_VMA)
            (m_vmaAllocator != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
                ? (vmaMapMemory(m_vmaAllocator, allocation, &mappedData) == VK_SUCCESS)
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
#if defined(VOXEL_HAS_VMA)
        if (m_vmaAllocator != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE) {
            vmaUnmapMemory(m_vmaAllocator, allocation);
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
            ? (vmaMapMemory(m_vmaAllocator, slot->allocation, &mapped) == VK_SUCCESS)
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

} // namespace render
