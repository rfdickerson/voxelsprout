#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>

namespace render {

// Opaque buffer handle so the renderer can refer to buffers without exposing Vulkan objects.
// Future resource systems can replace this with generation-based handles or IDs from an asset DB.
using BufferHandle = uint32_t;
constexpr BufferHandle kInvalidBufferHandle = 0;

struct BufferCreateDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VkMemoryPropertyFlags memoryProperties = 0;
    const void* initialData = nullptr;
};

// Owns Vulkan VkBuffer + VkDeviceMemory allocations behind uint32 handles.
// Future memory allocators (VMA or custom arenas) can replace this implementation.
class BufferAllocator {
public:
    bool init(VkPhysicalDevice physicalDevice, VkDevice device);
    void shutdown();

    [[nodiscard]] BufferHandle createBuffer(const BufferCreateDesc& desc);
    void destroyBuffer(BufferHandle handle);

    [[nodiscard]] VkBuffer getBuffer(BufferHandle handle) const;
    [[nodiscard]] VkDeviceSize getSize(BufferHandle handle) const;
    [[nodiscard]] void* mapBuffer(BufferHandle handle, VkDeviceSize offset, VkDeviceSize size);
    void unmapBuffer(BufferHandle handle);

private:
    struct BufferSlot {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        bool inUse = false;
    };

    [[nodiscard]] uint32_t findMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags requiredProperties) const;
    [[nodiscard]] BufferSlot* getSlot(BufferHandle handle);
    [[nodiscard]] const BufferSlot* getSlot(BufferHandle handle) const;

    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    std::vector<BufferSlot> m_slots;
    std::vector<uint32_t> m_freeSlots;
};

struct RingBufferSlice {
    BufferHandle buffer = kInvalidBufferHandle;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

// Per-frame rotating upload region in one host-visible buffer.
// Future upload/streaming systems can replace this with transient staging allocators.
class FrameRingBuffer {
public:
    bool init(
        BufferAllocator* allocator,
        VkDeviceSize bytesPerFrame,
        uint32_t frameCount,
        VkBufferUsageFlags usage
    );
    void shutdown(BufferAllocator* allocator);

    void beginFrame(uint32_t frameIndex);
    [[nodiscard]] std::optional<RingBufferSlice> allocate(VkDeviceSize size, VkDeviceSize alignment);

    [[nodiscard]] BufferHandle handle() const;

private:
    static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment);

    BufferHandle m_handle = kInvalidBufferHandle;
    uint8_t* m_mappedBase = nullptr;
    VkDeviceSize m_bytesPerFrame = 0;
    uint32_t m_frameCount = 0;
    uint32_t m_activeFrame = 0;
    VkDeviceSize m_writeOffset = 0;
};

} // namespace render
