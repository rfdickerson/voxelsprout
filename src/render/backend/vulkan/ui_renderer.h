#pragma once

#include "render/backend/vulkan/buffer_helpers.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"

#include <cstdint>
#include <string>
#include <unordered_map>

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
    struct InitInfo {
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VmaAllocator vmaAllocator = VK_NULL_HANDLE;
        BufferAllocator* bufferAllocator = nullptr;
        VkQueue uploadQueue = VK_NULL_HANDLE;        // Graphics queue, for one-time image uploads.
        std::uint32_t uploadQueueFamily = 0;
        VkFormat colorFormat = VK_FORMAT_UNDEFINED;  // Swapchain color format.
        std::string shaderDir;                        // Directory holding ui.*.slang.spv.
    };

    bool init(const InitInfo& info);
    void shutdown();
    [[nodiscard]] bool ready() const { return m_pipeline != VK_NULL_HANDLE; }

    // Register textures for use by UiDrawCmd::textureId. White (kUiNoTexture) is
    // created automatically in init(). Returns kUiNoTexture on failure.
    odai::ui::UiTextureId registerTextureRgba8(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);
    // Define/replace the font atlas (kUiFontAtlas), an R8 coverage atlas.
    bool setFontAtlasR8(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height);

    // Record the UI geometry into an already-begun rendering pass on `cmd`.
    void record(VkCommandBuffer cmd, std::uint32_t frameIndex, FrameArena& frameArena,
                const odai::ui::UiDrawData& drawData, VkExtent2D extent);

private:
    struct Texture {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };

    bool createPipeline();
    VkShaderModule loadShaderModule(const std::string& fileName) const;
    bool uploadTexturePixels(const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height,
                             VkFormat format, std::uint32_t bytesPerPixel, Texture& outTexture);
    VkDescriptorSet allocateTextureDescriptor(VkImageView view);
    void destroyTexture(Texture& texture);

    InitInfo m_info{};
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkCommandPool m_uploadPool = VK_NULL_HANDLE;
    std::unordered_map<odai::ui::UiTextureId, Texture> m_textures;
    odai::ui::UiTextureId m_nextTextureId = 2;  // 0 = white, 1 = font atlas.
};

}  // namespace odai::render
