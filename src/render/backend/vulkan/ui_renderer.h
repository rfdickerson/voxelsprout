#pragma once

#include "render/backend/vulkan/buffer_helpers.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

// Self-contained Vulkan renderer for the UI draw list. Owns its own pipeline,
// sampler, per-texture descriptor sets, and one-time upload command pool, so its
// integration into RendererBackend is just init()/record()/shutdown() plus
// texture registration. Draws inside an already-open dynamic-rendering pass
// targeting the swapchain image (the post pass), with alpha blending and no
// depth. This is the only UI-stack file that touches Vulkan.
namespace odai::render {

class UiRenderer {
public:
    struct Stats {
        std::uint32_t textureSlots = 0;
        std::uint32_t commandCount = 0;
        std::uint32_t drawCallCount = 0;
        std::uint64_t dynamicUploadBytes = 0;
        std::uint64_t skippedDrawCalls = 0;
    };
    // VK_EXT_descriptor_buffer function pointers + sizes, supplied by the renderer
    // (which owns the extension). When funcs is populated the UI backs its texture
    // set with a descriptor buffer instead of a pool-allocated descriptor set.
    struct DescriptorBufferSupport {
        PFN_vkGetDescriptorSetLayoutSizeEXT getLayoutSize = nullptr;
        PFN_vkGetDescriptorSetLayoutBindingOffsetEXT getBindingOffset = nullptr;
        PFN_vkGetDescriptorEXT getDescriptor = nullptr;
        PFN_vkCmdBindDescriptorBuffersEXT cmdBindDescriptorBuffers = nullptr;
        PFN_vkCmdSetDescriptorBufferOffsetsEXT cmdSetDescriptorBufferOffsets = nullptr;
        VkDeviceSize offsetAlignment = 1;
        VkDeviceSize combinedImageSamplerDescriptorSize = 0;
        VkDeviceSize sampledImageDescriptorSize = 0;
        VkDeviceSize samplerDescriptorSize = 0;
        bool combinedImageSamplerSingleArray = true;
        [[nodiscard]] bool enabled() const {
            return getLayoutSize != nullptr && getBindingOffset != nullptr &&
                   getDescriptor != nullptr && cmdBindDescriptorBuffers != nullptr &&
                   cmdSetDescriptorBufferOffsets != nullptr;
        }
    };
    struct InitInfo {
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkPipelineCache pipelineCache = VK_NULL_HANDLE;  // Optional; shared renderer cache.
        VmaAllocator vmaAllocator = VK_NULL_HANDLE;
        BufferAllocator* bufferAllocator = nullptr;
        VkQueue uploadQueue = VK_NULL_HANDLE;        // Graphics queue, for one-time image uploads.
        std::uint32_t uploadQueueFamily = 0;
        VkFormat colorFormat = VK_FORMAT_UNDEFINED;  // Swapchain color format.
        std::uint32_t maxTextureCount = 0;           // Required bindless descriptor capacity.
        std::string shaderDir;                        // Directory holding ui.*.slang.spv.
        DescriptorBufferSupport descriptorBuffer{};
    };

    bool init(const InitInfo& info);
    void shutdown();
    [[nodiscard]] bool ready() const { return m_pipeline != VK_NULL_HANDLE; }

    // Register textures in the bindless UI table. White (kUiNoTexture) is created
    // automatically in init(). Returns kUiNoTexture on failure.
    odai::ui::UiTextureId registerTextureRgba8(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);
    // Same as registerTextureRgba8 but generates a full mip chain via CPU box-filter.
    // Use for large icons rendered much smaller than their source resolution.
    odai::ui::UiTextureId registerTextureRgba8Mipmapped(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);
    // Define/replace the font atlas (kUiFontAtlas), an R8 coverage atlas.
    bool setFontAtlasR8(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);
    // Register an additional R8 coverage atlas (e.g. bold/italic faces) under a
    // fresh texture id. Returns kUiNoTexture on failure.
    odai::ui::UiTextureId registerFontAtlasR8(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);

    // Phase 1: allocate FrameArena slices, memcpy vertex/index data, and emit
    // the HOST→VERTEX_INPUT barrier. Must be called OUTSIDE (before) the
    // vkCmdBeginRendering block that will call record(). Safe to call even
    // when the draw list is empty.
    void uploadGeometry(VkCommandBuffer cmd, FrameArena& frameArena,
                        const odai::ui::UiDrawData& drawData);

    // Phase 2: record draw commands into an already-begun rendering pass.
    // uploadGeometry() must have been called earlier in the same frame,
    // outside the rendering pass.
    void record(VkCommandBuffer cmd, std::uint32_t frameIndex,
                const odai::ui::UiDrawData& drawData, VkExtent2D extent);
    [[nodiscard]] const Stats& stats() const { return m_stats; }

private:
    struct Texture {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };

    bool createPipeline();
    VkShaderModule loadShaderModule(const std::string& fileName) const;
    bool uploadTexturePixels(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height,
                             VkFormat format, std::uint32_t bytesPerPixel, Texture& outTexture);
    bool uploadTexturePixelsMipmapped(const std::uint8_t* pixels, std::uint32_t width,
                                      std::uint32_t height, Texture& outTexture);
    bool writeTextureDescriptor(odai::ui::UiTextureId slot, VkImageView view);
    bool submitUpload(VkCommandBuffer commandBuffer, BufferHandle stagingBuffer);
    void collectCompletedUploads();
    void destroyTexture(Texture& texture);

    InitInfo m_info{};
    // State written by uploadGeometry(), consumed by record().
    VkBuffer     m_uploadedVertexBuffer = VK_NULL_HANDLE;
    VkBuffer     m_uploadedIndexBuffer  = VK_NULL_HANDLE;
    VkDeviceSize m_uploadedVertexOffset = 0;
    VkDeviceSize m_uploadedIndexOffset  = 0;
    bool         m_geometryReady = false;
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    // Descriptor-buffer backing for the UI texture array (single region).
    BufferHandle m_descriptorBufferHandle = kInvalidBufferHandle;
    VkDeviceAddress m_descriptorBufferAddress = 0;
    std::uint8_t* m_descriptorBufferMapped = nullptr;
    VkBufferUsageFlags m_descriptorBufferUsage = 0;
    VkDeviceSize m_descriptorBindingOffset = 0;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkCommandPool m_uploadPool = VK_NULL_HANDLE;
    std::unordered_map<odai::ui::UiTextureId, Texture> m_textures;
    odai::ui::UiTextureId m_nextTextureId = 2;  // 0 = white, 1 = font atlas.
    std::uint32_t m_maxTextureCount = 0;
    Stats m_stats{};
    struct PendingUpload {
        BufferHandle stagingBuffer = kInvalidBufferHandle;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
    };
    std::vector<PendingUpload> m_pendingUploads;
};

}  // namespace odai::render
