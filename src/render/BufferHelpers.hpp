#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>
#if defined(VOXEL_HAS_VMA)
#include <vk_mem_alloc.h>
#endif

namespace render {

// Opaque buffer handle so the renderer can refer to buffers without exposing Vulkan objects.
// Future resource systems can replace this with generation-based handles or IDs from an asset DB.
using BufferHandle = uint32_t;
constexpr BufferHandle kInvalidBufferHandle = 0;

struct BufferCreateDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VkMemoryPropertyFlags memoryProperties = 0;
    const uint32_t* queueFamilyIndices = nullptr;
    uint32_t queueFamilyIndexCount = 0;
    const void* initialData = nullptr;
};

// Owns Vulkan VkBuffer + VkDeviceMemory allocations behind uint32 handles.
// Future memory allocators (VMA or custom arenas) can replace this implementation.
class BufferAllocator {
public:
    bool init(VkPhysicalDevice physicalDevice, VkDevice device
#if defined(VOXEL_HAS_VMA)
        , VmaAllocator vmaAllocator
#endif
    );
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
#if defined(VOXEL_HAS_VMA)
        VmaAllocation allocation = VK_NULL_HANDLE;
        void* mappedData = nullptr;
        bool persistentMapped = false;
#endif
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        bool inUse = false;
    };

    [[nodiscard]] uint32_t findMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags requiredProperties) const;
    [[nodiscard]] BufferSlot* getSlot(BufferHandle handle);
    [[nodiscard]] const BufferSlot* getSlot(BufferHandle handle) const;

    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
#if defined(VOXEL_HAS_VMA)
    VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;
#endif
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

enum class FrameArenaUploadKind : uint8_t {
    Unknown = 0,
    CameraUniform = 1,
    InstanceData = 2,
    PreviewData = 3
};

struct FrameArenaSlice {
    BufferHandle buffer = kInvalidBufferHandle;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

struct FrameArenaStats {
    VkDeviceSize uploadBytesAllocated = 0;
    uint32_t uploadAllocationCount = 0;
    VkDeviceSize transientBufferBytes = 0;
    uint32_t transientBufferCount = 0;
    uint64_t transientImageBytes = 0;
    uint32_t transientImageCount = 0;
};

struct FrameArenaConfig {
    VkDeviceSize uploadBytesPerFrame = 0;
    uint32_t frameCount = 0;
    VkBufferUsageFlags uploadUsage = 0;
};

using TransientImageHandle = uint32_t;
constexpr TransientImageHandle kInvalidTransientImageHandle = 0;

enum class FrameArenaImageLifetime : uint8_t {
    Persistent = 0,
    FrameTransient = 1
};

struct TransientImageDesc {
    VkImageType imageType = VK_IMAGE_TYPE_2D;
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{};
    VkImageUsageFlags usage = 0;
    VkImageAspectFlags aspectMask = 0;
    VkImageCreateFlags flags = 0;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct TransientImageInfo {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{};
};

// Per-frame transient allocator foundation.
// Layer A: one persistently mapped upload ring per frame.
// Layer B: per-frame transient buffer ownership that is reclaimed when the frame slot is reused.
class FrameArena {
public:
    bool init(
        BufferAllocator* allocator,
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        const FrameArenaConfig& config
#if defined(VOXEL_HAS_VMA)
        , VmaAllocator vmaAllocator
#endif
    );
    void shutdown(BufferAllocator* allocator);

    void beginFrame(uint32_t frameIndex);
    [[nodiscard]] std::optional<FrameArenaSlice> allocateUpload(
        VkDeviceSize size,
        VkDeviceSize alignment,
        FrameArenaUploadKind kind = FrameArenaUploadKind::Unknown
    );
    [[nodiscard]] BufferHandle createTransientBuffer(const BufferCreateDesc& desc);
    [[nodiscard]] TransientImageHandle createTransientImage(
        const TransientImageDesc& desc,
        FrameArenaImageLifetime lifetime = FrameArenaImageLifetime::Persistent
    );
    void destroyTransientImage(TransientImageHandle handle);
    [[nodiscard]] const TransientImageInfo* getTransientImage(TransientImageHandle handle) const;

    [[nodiscard]] BufferHandle uploadBufferHandle() const;
    [[nodiscard]] const FrameArenaStats& activeStats() const;

private:
    void clearFrameTransientBuffers(uint32_t frameIndex);
    void clearFrameTransientImages(uint32_t frameIndex);
    void destroyImageSlot(TransientImageHandle handle);
    [[nodiscard]] uint32_t findMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags requiredProperties) const;

    BufferAllocator* m_allocator = nullptr;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
#if defined(VOXEL_HAS_VMA)
    VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;
#endif
    FrameRingBuffer m_uploadRing;
    uint32_t m_frameCount = 0;
    uint32_t m_activeFrame = 0;
    std::vector<std::vector<BufferHandle>> m_frameTransientBuffers;
    std::vector<std::vector<TransientImageHandle>> m_frameTransientImages;

    struct ImageSlot {
        TransientImageInfo info{};
#if defined(VOXEL_HAS_VMA)
        VmaAllocation allocation = VK_NULL_HANDLE;
#endif
        VkDeviceMemory memory = VK_NULL_HANDLE;
        bool inUse = false;
    };
    std::vector<ImageSlot> m_imageSlots;
    std::vector<uint32_t> m_freeImageSlots;
    std::vector<FrameArenaStats> m_frameStats;
    FrameArenaStats m_emptyStats{};
};

} // namespace render
