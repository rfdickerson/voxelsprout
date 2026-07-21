#pragma once

#include <cstddef>

#include <vulkan/vulkan.h>

namespace odai::render {

// Owns the descriptor *set layouts* shared by the renderer. The descriptors
// themselves live in mapped descriptor buffers (VK_EXT_descriptor_buffer), so
// there are no pools or allocated VkDescriptorSets here anymore — only the
// layouts, which pipeline layouts and descriptor buffers are built against.
template <std::size_t FrameCount>
class DescriptorManager {
public:
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout bindlessDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout voxelGiDescriptorSetLayout = VK_NULL_HANDLE;

    void destroyMain(VkDevice device) {
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            descriptorSetLayout = VK_NULL_HANDLE;
        }
        if (bindlessDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, bindlessDescriptorSetLayout, nullptr);
            bindlessDescriptorSetLayout = VK_NULL_HANDLE;
        }
    }

    void destroyVoxelGi(VkDevice device) {
        if (voxelGiDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, voxelGiDescriptorSetLayout, nullptr);
            voxelGiDescriptorSetLayout = VK_NULL_HANDLE;
        }
    }
};

} // namespace odai::render
