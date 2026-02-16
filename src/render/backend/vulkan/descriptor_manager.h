#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan.h>

namespace voxelsprout::render {

template <std::size_t FrameCount>
class DescriptorManager {
public:
    struct BoundDescriptorSets {
        std::array<VkDescriptorSet, 2> sets{};
        uint32_t count = 1u;
    };

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout bindlessDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout voxelGiDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorPool bindlessDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorPool voxelGiDescriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, FrameCount> descriptorSets{};
    std::array<VkDescriptorSet, FrameCount> voxelGiDescriptorSets{};
    VkDescriptorSet bindlessDescriptorSet = VK_NULL_HANDLE;

    void destroyMain(VkDevice device) {
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
        }
        if (bindlessDescriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, bindlessDescriptorPool, nullptr);
            bindlessDescriptorPool = VK_NULL_HANDLE;
        }
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            descriptorSetLayout = VK_NULL_HANDLE;
        }
        if (bindlessDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, bindlessDescriptorSetLayout, nullptr);
            bindlessDescriptorSetLayout = VK_NULL_HANDLE;
        }
        descriptorSets.fill(VK_NULL_HANDLE);
        bindlessDescriptorSet = VK_NULL_HANDLE;
    }

    void destroyVoxelGi(VkDevice device) {
        if (voxelGiDescriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, voxelGiDescriptorPool, nullptr);
            voxelGiDescriptorPool = VK_NULL_HANDLE;
        }
        if (voxelGiDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, voxelGiDescriptorSetLayout, nullptr);
            voxelGiDescriptorSetLayout = VK_NULL_HANDLE;
        }
        voxelGiDescriptorSets.fill(VK_NULL_HANDLE);
    }

    BoundDescriptorSets buildBoundDescriptorSets(uint32_t frameIndex) const {
        BoundDescriptorSets result{};
        if (frameIndex < FrameCount) {
            result.sets[0] = descriptorSets[frameIndex];
        }
        result.sets[1] = bindlessDescriptorSet;
        result.count = (bindlessDescriptorSet != VK_NULL_HANDLE) ? 2u : 1u;
        return result;
    }
};

} // namespace voxelsprout::render
