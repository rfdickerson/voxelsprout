#include "render/Renderer.hpp"

#include <GLFW/glfw3.h>
#include "math/Math.hpp"
#include "world/ChunkMesher.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace render {

namespace {

constexpr std::array<const char*, 1> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};
constexpr std::array<const char*, 5> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
};

struct alignas(16) CameraUniform {
    float mvp[16];
    float view[16];
    float proj[16];
    float lightViewProj[16];
    float sunDirectionIntensity[4];
    float sunColorShadow[4];
};

struct alignas(16) ChunkPushConstants {
    float chunkOffset[4];
};

world::ChunkMeshData buildSingleVoxelPreviewMesh(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t ao,
    std::uint32_t material
) {
    world::ChunkMeshData mesh{};
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    for (std::uint32_t faceId = 0; faceId < 6; ++faceId) {
        const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
        for (std::uint32_t corner = 0; corner < 4; ++corner) {
            world::PackedVoxelVertex vertex{};
            vertex.bits = world::PackedVoxelVertex::pack(x, y, z, faceId, corner, ao, material);
            mesh.vertices.push_back(vertex);
        }

        mesh.indices.push_back(baseVertex + 0);
        mesh.indices.push_back(baseVertex + 1);
        mesh.indices.push_back(baseVertex + 2);
        mesh.indices.push_back(baseVertex + 0);
        mesh.indices.push_back(baseVertex + 2);
        mesh.indices.push_back(baseVertex + 3);
    }

    return mesh;
}

math::Matrix4 transpose(const math::Matrix4& matrix) {
    math::Matrix4 result{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            result(row, col) = matrix(col, row);
        }
    }
    return result;
}

math::Matrix4 perspectiveVulkan(float fovYRadians, float aspectRatio, float nearPlane, float farPlane) {
    math::Matrix4 result{};
    for (int i = 0; i < 16; ++i) {
        result.m[i] = 0.0f;
    }

    const float f = 1.0f / std::tan(fovYRadians * 0.5f);
    result(0, 0) = f / aspectRatio;
    // Vulkan viewport space differs from OpenGL; flip clip-space Y here.
    result(1, 1) = -f;
    result(2, 2) = farPlane / (nearPlane - farPlane);
    result(2, 3) = (farPlane * nearPlane) / (nearPlane - farPlane);
    result(3, 2) = -1.0f;
    return result;
}

math::Matrix4 orthographicVulkan(
    float left,
    float right,
    float bottom,
    float top,
    float nearPlane,
    float farPlane
) {
    math::Matrix4 result{};
    for (int i = 0; i < 16; ++i) {
        result.m[i] = 0.0f;
    }

    result(0, 0) = 2.0f / (right - left);
    // Match the Vulkan clip-space Y handling used by perspectiveVulkan.
    result(1, 1) = -2.0f / (top - bottom);
    result(2, 2) = 1.0f / (nearPlane - farPlane);
    result(3, 3) = 1.0f;
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 3) = nearPlane / (nearPlane - farPlane);
    return result;
}

math::Matrix4 lookAt(const math::Vector3& eye, const math::Vector3& target, const math::Vector3& up) {
    const math::Vector3 forward = math::normalize(target - eye);
    const math::Vector3 right = math::normalize(math::cross(forward, up));
    const math::Vector3 cameraUp = math::cross(right, forward);

    math::Matrix4 view = math::Matrix4::identity();
    view(0, 0) = right.x;
    view(0, 1) = right.y;
    view(0, 2) = right.z;
    view(0, 3) = -math::dot(right, eye);

    view(1, 0) = cameraUp.x;
    view(1, 1) = cameraUp.y;
    view(1, 2) = cameraUp.z;
    view(1, 3) = -math::dot(cameraUp, eye);

    view(2, 0) = -forward.x;
    view(2, 1) = -forward.y;
    view(2, 2) = -forward.z;
    view(2, 3) = math::dot(forward, eye);
    return view;
}

uint32_t findMemoryTypeIndex(
    VkPhysicalDevice physicalDevice,
    uint32_t typeBits,
    VkMemoryPropertyFlags requiredProperties
) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

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

void transitionImageLayout(
    VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask,
    VkImageAspectFlags aspectMask
) {
    VkImageMemoryBarrier2 imageBarrier{};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imageBarrier.srcStageMask = srcStageMask;
    imageBarrier.srcAccessMask = srcAccessMask;
    imageBarrier.dstStageMask = dstStageMask;
    imageBarrier.dstAccessMask = dstAccessMask;
    imageBarrier.oldLayout = oldLayout;
    imageBarrier.newLayout = newLayout;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image;
    imageBarrier.subresourceRange.aspectMask = aspectMask;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

VkFormat findSupportedDepthFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 3> kDepthCandidates = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT
    };

    for (VkFormat format : kDepthCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedShadowDepthFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kShadowDepthCandidates = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D16_UNORM
    };

    for (VkFormat format : kShadowDepthCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedHdrColorFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kHdrCandidates = {
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_B10G11R11_UFLOAT_PACK32
    };

    for (VkFormat format : kHdrCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

// Embedded shaders keep this bootstrap renderer self-contained.
// Source references:
// - src/render/shaders/voxel_packed.vert.glsl
// - src/render/shaders/voxel_packed.frag.glsl
// Future asset/shader systems can replace this with a shader pipeline.
static const uint32_t kVertShaderSpirv[] = {
0x07230203, 0x00010000, 0x000d000b, 0x000000ed, 0x00000000, 0x00020011,
0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x000a000f, 0x00000000,
0x00000004, 0x6e69616d, 0x00000000, 0x00000096, 0x000000cd, 0x000000d0,
0x000000d5, 0x000000db, 0x00030003, 0x00000002, 0x000001c2, 0x000a0004,
0x475f4c47, 0x4c474f4f, 0x70635f45, 0x74735f70, 0x5f656c79, 0x656e696c,
0x7269645f, 0x69746365, 0x00006576, 0x00080004, 0x475f4c47, 0x4c474f4f,
0x6e695f45, 0x64756c63, 0x69645f65, 0x74636572, 0x00657669, 0x00040005,
0x00000004, 0x6e69616d, 0x00000000, 0x00070005, 0x0000000d, 0x6e726f63,
0x664f7265, 0x74657366, 0x3b317528, 0x003b3175, 0x00040005, 0x0000000b,
0x65636166, 0x00000000, 0x00040005, 0x0000000c, 0x6e726f63, 0x00007265,
0x00030005, 0x00000094, 0x00000078, 0x00050005, 0x00000096, 0x61506e69,
0x64656b63, 0x00000000, 0x00030005, 0x0000009b, 0x00000079, 0x00030005,
0x000000a0, 0x0000007a, 0x00040005, 0x000000a5, 0x65636166, 0x00000000,
0x00040005, 0x000000ab, 0x6e726f63, 0x00007265, 0x00030005, 0x000000b0,
0x00006f61, 0x00050005, 0x000000b5, 0x6574616d, 0x6c616972, 0x00000000,
0x00060005, 0x000000bc, 0x65736162, 0x69736f50, 0x6e6f6974, 0x00000000,
0x00060005, 0x000000c4, 0x6c726f77, 0x736f5064, 0x6f697469, 0x0000006e,
0x00040005, 0x000000c6, 0x61726170, 0x0000006d, 0x00040005, 0x000000c8,
0x61726170, 0x0000006d, 0x00040005, 0x000000cd, 0x4674756f, 0x00656361,
0x00040005, 0x000000d0, 0x4174756f, 0x0000006f, 0x00050005, 0x000000d5,
0x4d74756f, 0x72657461, 0x006c6169, 0x00060005, 0x000000d9, 0x505f6c67,
0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x000000d9, 0x00000000,
0x505f6c67, 0x7469736f, 0x006e6f69, 0x00070006, 0x000000d9, 0x00000001,
0x505f6c67, 0x746e696f, 0x657a6953, 0x00000000, 0x00070006, 0x000000d9,
0x00000002, 0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e, 0x00070006,
0x000000d9, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e,
0x00030005, 0x000000db, 0x00000000, 0x00060005, 0x000000df, 0x656d6143,
0x6e556172, 0x726f6669, 0x0000006d, 0x00040006, 0x000000df, 0x00000000,
0x0070766d, 0x00040005, 0x000000e1, 0x656d6163, 0x00006172, 0x00040047,
0x00000096, 0x0000001e, 0x00000000, 0x00030047, 0x000000cd, 0x0000000e,
0x00040047, 0x000000cd, 0x0000001e, 0x00000000, 0x00040047, 0x000000d0,
0x0000001e, 0x00000001, 0x00030047, 0x000000d5, 0x0000000e, 0x00040047,
0x000000d5, 0x0000001e, 0x00000002, 0x00050048, 0x000000d9, 0x00000000,
0x0000000b, 0x00000000, 0x00050048, 0x000000d9, 0x00000001, 0x0000000b,
0x00000001, 0x00050048, 0x000000d9, 0x00000002, 0x0000000b, 0x00000003,
0x00050048, 0x000000d9, 0x00000003, 0x0000000b, 0x00000004, 0x00030047,
0x000000d9, 0x00000002, 0x00040048, 0x000000df, 0x00000000, 0x00000005,
0x00050048, 0x000000df, 0x00000000, 0x00000023, 0x00000000, 0x00050048,
0x000000df, 0x00000000, 0x00000007, 0x00000010, 0x00030047, 0x000000df,
0x00000002, 0x00040047, 0x000000e1, 0x00000022, 0x00000000, 0x00040047,
0x000000e1, 0x00000021, 0x00000000, 0x00020013, 0x00000002, 0x00030021,
0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000,
0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00030016, 0x00000008,
0x00000020, 0x00040017, 0x00000009, 0x00000008, 0x00000003, 0x00050021,
0x0000000a, 0x00000009, 0x00000007, 0x00000007, 0x0004002b, 0x00000006,
0x00000010, 0x00000000, 0x00020014, 0x00000011, 0x0004002b, 0x00000008,
0x00000019, 0x3f800000, 0x0004002b, 0x00000008, 0x0000001a, 0x00000000,
0x0006002c, 0x00000009, 0x0000001b, 0x00000019, 0x0000001a, 0x0000001a,
0x0004002b, 0x00000006, 0x0000001e, 0x00000001, 0x0006002c, 0x00000009,
0x00000022, 0x00000019, 0x00000019, 0x0000001a, 0x0004002b, 0x00000006,
0x00000025, 0x00000002, 0x0006002c, 0x00000009, 0x00000029, 0x00000019,
0x00000019, 0x00000019, 0x0006002c, 0x00000009, 0x0000002b, 0x00000019,
0x0000001a, 0x00000019, 0x0006002c, 0x00000009, 0x00000035, 0x0000001a,
0x0000001a, 0x00000019, 0x0006002c, 0x00000009, 0x0000003b, 0x0000001a,
0x00000019, 0x00000019, 0x0006002c, 0x00000009, 0x00000041, 0x0000001a,
0x00000019, 0x0000001a, 0x0006002c, 0x00000009, 0x00000043, 0x0000001a,
0x0000001a, 0x0000001a, 0x0004002b, 0x00000006, 0x0000005a, 0x00000003,
0x0004002b, 0x00000006, 0x0000006f, 0x00000004, 0x00040020, 0x00000095,
0x00000001, 0x00000006, 0x0004003b, 0x00000095, 0x00000096, 0x00000001,
0x0004002b, 0x00000006, 0x00000099, 0x0000001f, 0x0004002b, 0x00000006,
0x0000009d, 0x00000005, 0x0004002b, 0x00000006, 0x000000a2, 0x0000000a,
0x0004002b, 0x00000006, 0x000000a7, 0x0000000f, 0x0004002b, 0x00000006,
0x000000a9, 0x00000007, 0x0004002b, 0x00000006, 0x000000ad, 0x00000012,
0x0004002b, 0x00000006, 0x000000b2, 0x00000014, 0x0004002b, 0x00000006,
0x000000b7, 0x00000016, 0x0004002b, 0x00000006, 0x000000b9, 0x000000ff,
0x00040020, 0x000000bb, 0x00000007, 0x00000009, 0x00040020, 0x000000cc,
0x00000003, 0x00000006, 0x0004003b, 0x000000cc, 0x000000cd, 0x00000003,
0x00040020, 0x000000cf, 0x00000003, 0x00000008, 0x0004003b, 0x000000cf,
0x000000d0, 0x00000003, 0x0004002b, 0x00000008, 0x000000d3, 0x40400000,
0x0004003b, 0x000000cc, 0x000000d5, 0x00000003, 0x00040017, 0x000000d7,
0x00000008, 0x00000004, 0x0004001c, 0x000000d8, 0x00000008, 0x0000001e,
0x0006001e, 0x000000d9, 0x000000d7, 0x00000008, 0x000000d8, 0x000000d8,
0x00040020, 0x000000da, 0x00000003, 0x000000d9, 0x0004003b, 0x000000da,
0x000000db, 0x00000003, 0x00040015, 0x000000dc, 0x00000020, 0x00000001,
0x0004002b, 0x000000dc, 0x000000dd, 0x00000000, 0x00040018, 0x000000de,
0x000000d7, 0x00000004, 0x0003001e, 0x000000df, 0x000000de, 0x00040020,
0x000000e0, 0x00000002, 0x000000df, 0x0004003b, 0x000000e0, 0x000000e1,
0x00000002, 0x00040020, 0x000000e2, 0x00000002, 0x000000de, 0x00040020,
0x000000eb, 0x00000003, 0x000000d7, 0x00050036, 0x00000002, 0x00000004,
0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007,
0x00000094, 0x00000007, 0x0004003b, 0x00000007, 0x0000009b, 0x00000007,
0x0004003b, 0x00000007, 0x000000a0, 0x00000007, 0x0004003b, 0x00000007,
0x000000a5, 0x00000007, 0x0004003b, 0x00000007, 0x000000ab, 0x00000007,
0x0004003b, 0x00000007, 0x000000b0, 0x00000007, 0x0004003b, 0x00000007,
0x000000b5, 0x00000007, 0x0004003b, 0x000000bb, 0x000000bc, 0x00000007,
0x0004003b, 0x000000bb, 0x000000c4, 0x00000007, 0x0004003b, 0x00000007,
0x000000c6, 0x00000007, 0x0004003b, 0x00000007, 0x000000c8, 0x00000007,
0x0004003d, 0x00000006, 0x00000097, 0x00000096, 0x000500c2, 0x00000006,
0x00000098, 0x00000097, 0x00000010, 0x000500c7, 0x00000006, 0x0000009a,
0x00000098, 0x00000099, 0x0003003e, 0x00000094, 0x0000009a, 0x0004003d,
0x00000006, 0x0000009c, 0x00000096, 0x000500c2, 0x00000006, 0x0000009e,
0x0000009c, 0x0000009d, 0x000500c7, 0x00000006, 0x0000009f, 0x0000009e,
0x00000099, 0x0003003e, 0x0000009b, 0x0000009f, 0x0004003d, 0x00000006,
0x000000a1, 0x00000096, 0x000500c2, 0x00000006, 0x000000a3, 0x000000a1,
0x000000a2, 0x000500c7, 0x00000006, 0x000000a4, 0x000000a3, 0x00000099,
0x0003003e, 0x000000a0, 0x000000a4, 0x0004003d, 0x00000006, 0x000000a6,
0x00000096, 0x000500c2, 0x00000006, 0x000000a8, 0x000000a6, 0x000000a7,
0x000500c7, 0x00000006, 0x000000aa, 0x000000a8, 0x000000a9, 0x0003003e,
0x000000a5, 0x000000aa, 0x0004003d, 0x00000006, 0x000000ac, 0x00000096,
0x000500c2, 0x00000006, 0x000000ae, 0x000000ac, 0x000000ad, 0x000500c7,
0x00000006, 0x000000af, 0x000000ae, 0x0000005a, 0x0003003e, 0x000000ab,
0x000000af, 0x0004003d, 0x00000006, 0x000000b1, 0x00000096, 0x000500c2,
0x00000006, 0x000000b3, 0x000000b1, 0x000000b2, 0x000500c7, 0x00000006,
0x000000b4, 0x000000b3, 0x0000005a, 0x0003003e, 0x000000b0, 0x000000b4,
0x0004003d, 0x00000006, 0x000000b6, 0x00000096, 0x000500c2, 0x00000006,
0x000000b8, 0x000000b6, 0x000000b7, 0x000500c7, 0x00000006, 0x000000ba,
0x000000b8, 0x000000b9, 0x0003003e, 0x000000b5, 0x000000ba, 0x0004003d,
0x00000006, 0x000000bd, 0x00000094, 0x00040070, 0x00000008, 0x000000be,
0x000000bd, 0x0004003d, 0x00000006, 0x000000bf, 0x0000009b, 0x00040070,
0x00000008, 0x000000c0, 0x000000bf, 0x0004003d, 0x00000006, 0x000000c1,
0x000000a0, 0x00040070, 0x00000008, 0x000000c2, 0x000000c1, 0x00060050,
0x00000009, 0x000000c3, 0x000000be, 0x000000c0, 0x000000c2, 0x0003003e,
0x000000bc, 0x000000c3, 0x0004003d, 0x00000009, 0x000000c5, 0x000000bc,
0x0004003d, 0x00000006, 0x000000c7, 0x000000a5, 0x0003003e, 0x000000c6,
0x000000c7, 0x0004003d, 0x00000006, 0x000000c9, 0x000000ab, 0x0003003e,
0x000000c8, 0x000000c9, 0x00060039, 0x00000009, 0x000000ca, 0x0000000d,
0x000000c6, 0x000000c8, 0x00050081, 0x00000009, 0x000000cb, 0x000000c5,
0x000000ca, 0x0003003e, 0x000000c4, 0x000000cb, 0x0004003d, 0x00000006,
0x000000ce, 0x000000a5, 0x0003003e, 0x000000cd, 0x000000ce, 0x0004003d,
0x00000006, 0x000000d1, 0x000000b0, 0x00040070, 0x00000008, 0x000000d2,
0x000000d1, 0x00050088, 0x00000008, 0x000000d4, 0x000000d2, 0x000000d3,
0x0003003e, 0x000000d0, 0x000000d4, 0x0004003d, 0x00000006, 0x000000d6,
0x000000b5, 0x0003003e, 0x000000d5, 0x000000d6, 0x00050041, 0x000000e2,
0x000000e3, 0x000000e1, 0x000000dd, 0x0004003d, 0x000000de, 0x000000e4,
0x000000e3, 0x0004003d, 0x00000009, 0x000000e5, 0x000000c4, 0x00050051,
0x00000008, 0x000000e6, 0x000000e5, 0x00000000, 0x00050051, 0x00000008,
0x000000e7, 0x000000e5, 0x00000001, 0x00050051, 0x00000008, 0x000000e8,
0x000000e5, 0x00000002, 0x00070050, 0x000000d7, 0x000000e9, 0x000000e6,
0x000000e7, 0x000000e8, 0x00000019, 0x00050091, 0x000000d7, 0x000000ea,
0x000000e4, 0x000000e9, 0x00050041, 0x000000eb, 0x000000ec, 0x000000db,
0x000000dd, 0x0003003e, 0x000000ec, 0x000000ea, 0x000100fd, 0x00010038,
0x00050036, 0x00000009, 0x0000000d, 0x00000000, 0x0000000a, 0x00030037,
0x00000007, 0x0000000b, 0x00030037, 0x00000007, 0x0000000c, 0x000200f8,
0x0000000e, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000b, 0x000500aa,
0x00000011, 0x00000012, 0x0000000f, 0x00000010, 0x000300f7, 0x00000014,
0x00000000, 0x000400fa, 0x00000012, 0x00000013, 0x00000014, 0x000200f8,
0x00000013, 0x0004003d, 0x00000006, 0x00000015, 0x0000000c, 0x000500aa,
0x00000011, 0x00000016, 0x00000015, 0x00000010, 0x000300f7, 0x00000018,
0x00000000, 0x000400fa, 0x00000016, 0x00000017, 0x00000018, 0x000200f8,
0x00000017, 0x000200fe, 0x0000001b, 0x000200f8, 0x00000018, 0x0004003d,
0x00000006, 0x0000001d, 0x0000000c, 0x000500aa, 0x00000011, 0x0000001f,
0x0000001d, 0x0000001e, 0x000300f7, 0x00000021, 0x00000000, 0x000400fa,
0x0000001f, 0x00000020, 0x00000021, 0x000200f8, 0x00000020, 0x000200fe,
0x00000022, 0x000200f8, 0x00000021, 0x0004003d, 0x00000006, 0x00000024,
0x0000000c, 0x000500aa, 0x00000011, 0x00000026, 0x00000024, 0x00000025,
0x000300f7, 0x00000028, 0x00000000, 0x000400fa, 0x00000026, 0x00000027,
0x00000028, 0x000200f8, 0x00000027, 0x000200fe, 0x00000029, 0x000200f8,
0x00000028, 0x000200fe, 0x0000002b, 0x000200f8, 0x00000014, 0x0004003d,
0x00000006, 0x0000002d, 0x0000000b, 0x000500aa, 0x00000011, 0x0000002e,
0x0000002d, 0x0000001e, 0x000300f7, 0x00000030, 0x00000000, 0x000400fa,
0x0000002e, 0x0000002f, 0x00000030, 0x000200f8, 0x0000002f, 0x0004003d,
0x00000006, 0x00000031, 0x0000000c, 0x000500aa, 0x00000011, 0x00000032,
0x00000031, 0x00000010, 0x000300f7, 0x00000034, 0x00000000, 0x000400fa,
0x00000032, 0x00000033, 0x00000034, 0x000200f8, 0x00000033, 0x000200fe,
0x00000035, 0x000200f8, 0x00000034, 0x0004003d, 0x00000006, 0x00000037,
0x0000000c, 0x000500aa, 0x00000011, 0x00000038, 0x00000037, 0x0000001e,
0x000300f7, 0x0000003a, 0x00000000, 0x000400fa, 0x00000038, 0x00000039,
0x0000003a, 0x000200f8, 0x00000039, 0x000200fe, 0x0000003b, 0x000200f8,
0x0000003a, 0x0004003d, 0x00000006, 0x0000003d, 0x0000000c, 0x000500aa,
0x00000011, 0x0000003e, 0x0000003d, 0x00000025, 0x000300f7, 0x00000040,
0x00000000, 0x000400fa, 0x0000003e, 0x0000003f, 0x00000040, 0x000200f8,
0x0000003f, 0x000200fe, 0x00000041, 0x000200f8, 0x00000040, 0x000200fe,
0x00000043, 0x000200f8, 0x00000030, 0x0004003d, 0x00000006, 0x00000045,
0x0000000b, 0x000500aa, 0x00000011, 0x00000046, 0x00000045, 0x00000025,
0x000300f7, 0x00000048, 0x00000000, 0x000400fa, 0x00000046, 0x00000047,
0x00000048, 0x000200f8, 0x00000047, 0x0004003d, 0x00000006, 0x00000049,
0x0000000c, 0x000500aa, 0x00000011, 0x0000004a, 0x00000049, 0x00000010,
0x000300f7, 0x0000004c, 0x00000000, 0x000400fa, 0x0000004a, 0x0000004b,
0x0000004c, 0x000200f8, 0x0000004b, 0x000200fe, 0x00000041, 0x000200f8,
0x0000004c, 0x0004003d, 0x00000006, 0x0000004e, 0x0000000c, 0x000500aa,
0x00000011, 0x0000004f, 0x0000004e, 0x0000001e, 0x000300f7, 0x00000051,
0x00000000, 0x000400fa, 0x0000004f, 0x00000050, 0x00000051, 0x000200f8,
0x00000050, 0x000200fe, 0x0000003b, 0x000200f8, 0x00000051, 0x0004003d,
0x00000006, 0x00000053, 0x0000000c, 0x000500aa, 0x00000011, 0x00000054,
0x00000053, 0x00000025, 0x000300f7, 0x00000056, 0x00000000, 0x000400fa,
0x00000054, 0x00000055, 0x00000056, 0x000200f8, 0x00000055, 0x000200fe,
0x00000029, 0x000200f8, 0x00000056, 0x000200fe, 0x00000022, 0x000200f8,
0x00000048, 0x0004003d, 0x00000006, 0x00000059, 0x0000000b, 0x000500aa,
0x00000011, 0x0000005b, 0x00000059, 0x0000005a, 0x000300f7, 0x0000005d,
0x00000000, 0x000400fa, 0x0000005b, 0x0000005c, 0x0000005d, 0x000200f8,
0x0000005c, 0x0004003d, 0x00000006, 0x0000005e, 0x0000000c, 0x000500aa,
0x00000011, 0x0000005f, 0x0000005e, 0x00000010, 0x000300f7, 0x00000061,
0x00000000, 0x000400fa, 0x0000005f, 0x00000060, 0x00000061, 0x000200f8,
0x00000060, 0x000200fe, 0x00000035, 0x000200f8, 0x00000061, 0x0004003d,
0x00000006, 0x00000063, 0x0000000c, 0x000500aa, 0x00000011, 0x00000064,
0x00000063, 0x0000001e, 0x000300f7, 0x00000066, 0x00000000, 0x000400fa,
0x00000064, 0x00000065, 0x00000066, 0x000200f8, 0x00000065, 0x000200fe,
0x00000043, 0x000200f8, 0x00000066, 0x0004003d, 0x00000006, 0x00000068,
0x0000000c, 0x000500aa, 0x00000011, 0x00000069, 0x00000068, 0x00000025,
0x000300f7, 0x0000006b, 0x00000000, 0x000400fa, 0x00000069, 0x0000006a,
0x0000006b, 0x000200f8, 0x0000006a, 0x000200fe, 0x0000001b, 0x000200f8,
0x0000006b, 0x000200fe, 0x0000002b, 0x000200f8, 0x0000005d, 0x0004003d,
0x00000006, 0x0000006e, 0x0000000b, 0x000500aa, 0x00000011, 0x00000070,
0x0000006e, 0x0000006f, 0x000300f7, 0x00000072, 0x00000000, 0x000400fa,
0x00000070, 0x00000071, 0x00000072, 0x000200f8, 0x00000071, 0x0004003d,
0x00000006, 0x00000073, 0x0000000c, 0x000500aa, 0x00000011, 0x00000074,
0x00000073, 0x00000010, 0x000300f7, 0x00000076, 0x00000000, 0x000400fa,
0x00000074, 0x00000075, 0x00000076, 0x000200f8, 0x00000075, 0x000200fe,
0x0000002b, 0x000200f8, 0x00000076, 0x0004003d, 0x00000006, 0x00000078,
0x0000000c, 0x000500aa, 0x00000011, 0x00000079, 0x00000078, 0x0000001e,
0x000300f7, 0x0000007b, 0x00000000, 0x000400fa, 0x00000079, 0x0000007a,
0x0000007b, 0x000200f8, 0x0000007a, 0x000200fe, 0x00000029, 0x000200f8,
0x0000007b, 0x0004003d, 0x00000006, 0x0000007d, 0x0000000c, 0x000500aa,
0x00000011, 0x0000007e, 0x0000007d, 0x00000025, 0x000300f7, 0x00000080,
0x00000000, 0x000400fa, 0x0000007e, 0x0000007f, 0x00000080, 0x000200f8,
0x0000007f, 0x000200fe, 0x0000003b, 0x000200f8, 0x00000080, 0x000200fe,
0x00000035, 0x000200f8, 0x00000072, 0x0004003d, 0x00000006, 0x00000083,
0x0000000c, 0x000500aa, 0x00000011, 0x00000084, 0x00000083, 0x00000010,
0x000300f7, 0x00000086, 0x00000000, 0x000400fa, 0x00000084, 0x00000085,
0x00000086, 0x000200f8, 0x00000085, 0x000200fe, 0x00000043, 0x000200f8,
0x00000086, 0x0004003d, 0x00000006, 0x00000088, 0x0000000c, 0x000500aa,
0x00000011, 0x00000089, 0x00000088, 0x0000001e, 0x000300f7, 0x0000008b,
0x00000000, 0x000400fa, 0x00000089, 0x0000008a, 0x0000008b, 0x000200f8,
0x0000008a, 0x000200fe, 0x00000041, 0x000200f8, 0x0000008b, 0x0004003d,
0x00000006, 0x0000008d, 0x0000000c, 0x000500aa, 0x00000011, 0x0000008e,
0x0000008d, 0x00000025, 0x000300f7, 0x00000090, 0x00000000, 0x000400fa,
0x0000008e, 0x0000008f, 0x00000090, 0x000200f8, 0x0000008f, 0x000200fe,
0x00000022, 0x000200f8, 0x00000090, 0x000200fe, 0x0000001b, 0x00010038,
};

static const uint32_t kFragShaderSpirv[] = {
0x07230203, 0x00010000, 0x000d000b, 0x0000006c, 0x00000000, 0x00020011,
0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0009000f, 0x00000004,
0x00000004, 0x6e69616d, 0x00000000, 0x00000050, 0x00000059, 0x0000005d,
0x00000064, 0x00030010, 0x00000004, 0x00000007, 0x00030003, 0x00000002,
0x000001c2, 0x000a0004, 0x475f4c47, 0x4c474f4f, 0x70635f45, 0x74735f70,
0x5f656c79, 0x656e696c, 0x7269645f, 0x69746365, 0x00006576, 0x00080004,
0x475f4c47, 0x4c474f4f, 0x6e695f45, 0x64756c63, 0x69645f65, 0x74636572,
0x00657669, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060005,
0x0000000c, 0x65636166, 0x6f6c6f43, 0x31752872, 0x0000003b, 0x00040005,
0x0000000b, 0x65636166, 0x00000000, 0x00070005, 0x0000000f, 0x6574616d,
0x6c616972, 0x746e6954, 0x3b317528, 0x00000000, 0x00050005, 0x0000000e,
0x6574616d, 0x6c616972, 0x00000000, 0x00060005, 0x0000004d, 0x72426f61,
0x74686769, 0x7373656e, 0x00000000, 0x00040005, 0x00000050, 0x6f416e69,
0x00000000, 0x00050005, 0x00000057, 0x65736162, 0x6f6c6f43, 0x00000072,
0x00040005, 0x00000059, 0x61466e69, 0x00006563, 0x00040005, 0x0000005a,
0x61726170, 0x0000006d, 0x00050005, 0x0000005d, 0x614d6e69, 0x69726574,
0x00006c61, 0x00040005, 0x0000005e, 0x61726170, 0x0000006d, 0x00050005,
0x00000064, 0x4374756f, 0x726f6c6f, 0x00000000, 0x00040047, 0x00000050,
0x0000001e, 0x00000001, 0x00030047, 0x00000059, 0x0000000e, 0x00040047,
0x00000059, 0x0000001e, 0x00000000, 0x00030047, 0x0000005d, 0x0000000e,
0x00040047, 0x0000005d, 0x0000001e, 0x00000002, 0x00040047, 0x00000064,
0x0000001e, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020,
0x00000007, 0x00000007, 0x00000006, 0x00030016, 0x00000008, 0x00000020,
0x00040017, 0x00000009, 0x00000008, 0x00000003, 0x00040021, 0x0000000a,
0x00000009, 0x00000007, 0x0004002b, 0x00000006, 0x00000012, 0x00000000,
0x00020014, 0x00000013, 0x0004002b, 0x00000008, 0x00000017, 0x3f666666,
0x0004002b, 0x00000008, 0x00000018, 0x3ee147ae, 0x0004002b, 0x00000008,
0x00000019, 0x3eb33333, 0x0006002c, 0x00000009, 0x0000001a, 0x00000017,
0x00000018, 0x00000019, 0x0004002b, 0x00000006, 0x0000001d, 0x00000001,
0x0004002b, 0x00000008, 0x00000021, 0x3f333333, 0x0004002b, 0x00000008,
0x00000022, 0x3eae147b, 0x0004002b, 0x00000008, 0x00000023, 0x3e8f5c29,
0x0006002c, 0x00000009, 0x00000024, 0x00000021, 0x00000022, 0x00000023,
0x0004002b, 0x00000006, 0x00000027, 0x00000002, 0x0004002b, 0x00000008,
0x0000002b, 0x3ecccccd, 0x0004002b, 0x00000008, 0x0000002c, 0x3f59999a,
0x0006002c, 0x00000009, 0x0000002d, 0x0000002b, 0x0000002c, 0x0000002b,
0x0004002b, 0x00000006, 0x00000030, 0x00000003, 0x0004002b, 0x00000008,
0x00000034, 0x3f0ccccd, 0x0006002c, 0x00000009, 0x00000035, 0x00000023,
0x00000034, 0x00000023, 0x0004002b, 0x00000006, 0x00000038, 0x00000004,
0x0006002c, 0x00000009, 0x0000003c, 0x00000019, 0x00000034, 0x00000017,
0x0004002b, 0x00000008, 0x0000003e, 0x3ed70a3d, 0x0006002c, 0x00000009,
0x0000003f, 0x00000023, 0x0000003e, 0x00000021, 0x0004002b, 0x00000008,
0x00000046, 0x3f800000, 0x0006002c, 0x00000009, 0x00000047, 0x00000046,
0x00000046, 0x00000046, 0x0006002c, 0x00000009, 0x00000049, 0x00000017,
0x00000017, 0x00000017, 0x00040020, 0x0000004c, 0x00000007, 0x00000008,
0x0004002b, 0x00000008, 0x0000004e, 0x3ee66666, 0x00040020, 0x0000004f,
0x00000001, 0x00000008, 0x0004003b, 0x0000004f, 0x00000050, 0x00000001,
0x0004002b, 0x00000008, 0x00000052, 0x00000000, 0x00040020, 0x00000056,
0x00000007, 0x00000009, 0x00040020, 0x00000058, 0x00000001, 0x00000006,
0x0004003b, 0x00000058, 0x00000059, 0x00000001, 0x0004003b, 0x00000058,
0x0000005d, 0x00000001, 0x00040017, 0x00000062, 0x00000008, 0x00000004,
0x00040020, 0x00000063, 0x00000003, 0x00000062, 0x0004003b, 0x00000063,
0x00000064, 0x00000003, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x0000004c, 0x0000004d,
0x00000007, 0x0004003b, 0x00000056, 0x00000057, 0x00000007, 0x0004003b,
0x00000007, 0x0000005a, 0x00000007, 0x0004003b, 0x00000007, 0x0000005e,
0x00000007, 0x0004003d, 0x00000008, 0x00000051, 0x00000050, 0x0008000c,
0x00000008, 0x00000053, 0x00000001, 0x0000002b, 0x00000051, 0x00000052,
0x00000046, 0x00050085, 0x00000008, 0x00000054, 0x00000053, 0x00000034,
0x00050081, 0x00000008, 0x00000055, 0x0000004e, 0x00000054, 0x0003003e,
0x0000004d, 0x00000055, 0x0004003d, 0x00000006, 0x0000005b, 0x00000059,
0x0003003e, 0x0000005a, 0x0000005b, 0x00050039, 0x00000009, 0x0000005c,
0x0000000c, 0x0000005a, 0x0004003d, 0x00000006, 0x0000005f, 0x0000005d,
0x0003003e, 0x0000005e, 0x0000005f, 0x00050039, 0x00000009, 0x00000060,
0x0000000f, 0x0000005e, 0x00050085, 0x00000009, 0x00000061, 0x0000005c,
0x00000060, 0x0003003e, 0x00000057, 0x00000061, 0x0004003d, 0x00000009,
0x00000065, 0x00000057, 0x0004003d, 0x00000008, 0x00000066, 0x0000004d,
0x0005008e, 0x00000009, 0x00000067, 0x00000065, 0x00000066, 0x00050051,
0x00000008, 0x00000068, 0x00000067, 0x00000000, 0x00050051, 0x00000008,
0x00000069, 0x00000067, 0x00000001, 0x00050051, 0x00000008, 0x0000006a,
0x00000067, 0x00000002, 0x00070050, 0x00000062, 0x0000006b, 0x00000068,
0x00000069, 0x0000006a, 0x00000046, 0x0003003e, 0x00000064, 0x0000006b,
0x000100fd, 0x00010038, 0x00050036, 0x00000009, 0x0000000c, 0x00000000,
0x0000000a, 0x00030037, 0x00000007, 0x0000000b, 0x000200f8, 0x0000000d,
0x0004003d, 0x00000006, 0x00000011, 0x0000000b, 0x000500aa, 0x00000013,
0x00000014, 0x00000011, 0x00000012, 0x000300f7, 0x00000016, 0x00000000,
0x000400fa, 0x00000014, 0x00000015, 0x00000016, 0x000200f8, 0x00000015,
0x000200fe, 0x0000001a, 0x000200f8, 0x00000016, 0x0004003d, 0x00000006,
0x0000001c, 0x0000000b, 0x000500aa, 0x00000013, 0x0000001e, 0x0000001c,
0x0000001d, 0x000300f7, 0x00000020, 0x00000000, 0x000400fa, 0x0000001e,
0x0000001f, 0x00000020, 0x000200f8, 0x0000001f, 0x000200fe, 0x00000024,
0x000200f8, 0x00000020, 0x0004003d, 0x00000006, 0x00000026, 0x0000000b,
0x000500aa, 0x00000013, 0x00000028, 0x00000026, 0x00000027, 0x000300f7,
0x0000002a, 0x00000000, 0x000400fa, 0x00000028, 0x00000029, 0x0000002a,
0x000200f8, 0x00000029, 0x000200fe, 0x0000002d, 0x000200f8, 0x0000002a,
0x0004003d, 0x00000006, 0x0000002f, 0x0000000b, 0x000500aa, 0x00000013,
0x00000031, 0x0000002f, 0x00000030, 0x000300f7, 0x00000033, 0x00000000,
0x000400fa, 0x00000031, 0x00000032, 0x00000033, 0x000200f8, 0x00000032,
0x000200fe, 0x00000035, 0x000200f8, 0x00000033, 0x0004003d, 0x00000006,
0x00000037, 0x0000000b, 0x000500aa, 0x00000013, 0x00000039, 0x00000037,
0x00000038, 0x000300f7, 0x0000003b, 0x00000000, 0x000400fa, 0x00000039,
0x0000003a, 0x0000003b, 0x000200f8, 0x0000003a, 0x000200fe, 0x0000003c,
0x000200f8, 0x0000003b, 0x000200fe, 0x0000003f, 0x00010038, 0x00050036,
0x00000009, 0x0000000f, 0x00000000, 0x0000000a, 0x00030037, 0x00000007,
0x0000000e, 0x000200f8, 0x00000010, 0x0004003d, 0x00000006, 0x00000042,
0x0000000e, 0x000500aa, 0x00000013, 0x00000043, 0x00000042, 0x0000001d,
0x000300f7, 0x00000045, 0x00000000, 0x000400fa, 0x00000043, 0x00000044,
0x00000045, 0x000200f8, 0x00000044, 0x000200fe, 0x00000047, 0x000200f8,
0x00000045, 0x000200fe, 0x00000049, 0x00010038, 
};

struct QueueFamilyChoice {
    std::optional<uint32_t> graphicsAndPresent;
    std::optional<uint32_t> transfer;
    uint32_t graphicsQueueIndex = 0;
    uint32_t transferQueueIndex = 0;

    [[nodiscard]] bool valid() const {
        return graphicsAndPresent.has_value() && transfer.has_value();
    }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

const char* vkResultName(VkResult result) {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    default: return "VK_RESULT_UNKNOWN";
    }
}

void logVkFailure(const char* context, VkResult result) {
    std::cerr << "[render] " << context << " failed: "
              << vkResultName(result) << " (" << static_cast<int>(result) << ")\n";
}

bool isLayerAvailable(const char* layerName) {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

    for (const VkLayerProperties& layer : layers) {
        if (std::strcmp(layer.layerName, layerName) == 0) {
            return true;
        }
    }
    return false;
}

QueueFamilyChoice findQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    QueueFamilyChoice choice;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());

    std::optional<uint32_t> dedicatedTransferFamily;
    std::optional<uint32_t> anyTransferFamily;

    for (uint32_t familyIndex = 0; familyIndex < familyCount; ++familyIndex) {
        const VkQueueFlags queueFlags = families[familyIndex].queueFlags;
        const bool hasGraphics = (queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool hasTransfer = (queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;

        if (hasGraphics && !choice.graphicsAndPresent.has_value()) {
            VkBool32 hasPresent = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex, surface, &hasPresent);
            if (hasPresent == VK_TRUE) {
                choice.graphicsAndPresent = familyIndex;
            }
        }

        if (hasTransfer) {
            if (!anyTransferFamily.has_value()) {
                anyTransferFamily = familyIndex;
            }
            if (!dedicatedTransferFamily.has_value() && !hasGraphics) {
                dedicatedTransferFamily = familyIndex;
            }
        }
    }

    if (!choice.graphicsAndPresent.has_value()) {
        return choice;
    }

    if (dedicatedTransferFamily.has_value()) {
        choice.transfer = dedicatedTransferFamily.value();
    } else if (anyTransferFamily.has_value()) {
        choice.transfer = anyTransferFamily.value();
    } else {
        choice.transfer = choice.graphicsAndPresent.value();
    }

    if (choice.transfer.value() == choice.graphicsAndPresent.value()) {
        const uint32_t queueCount = families[choice.graphicsAndPresent.value()].queueCount;
        if (queueCount > 1) {
            choice.transferQueueIndex = 1;
        }
    }

    return choice;
}

bool hasRequiredDeviceExtensions(VkPhysicalDevice physicalDevice) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data());

    for (const char* required : kDeviceExtensions) {
        bool found = false;
        for (const VkExtensionProperties& available : extensions) {
            if (std::strcmp(required, available.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    return true;
}

SwapchainSupport querySwapchainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    SwapchainSupport support;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &support.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    support.formats.resize(formatCount);
    if (formatCount > 0) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, support.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    support.presentModes.resize(presentModeCount);
    if (presentModeCount > 0) {
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, support.presentModes.data());
    }

    return support;
}

VkSurfaceFormatKHR chooseSwapchainFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return formats.front();
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes) {
    for (const VkPresentModeKHR presentMode : presentModes) {
        if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return presentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D extent{};
    extent.width = std::clamp(
        static_cast<uint32_t>(std::max(width, 1)),
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width
    );
    extent.height = std::clamp(
        static_cast<uint32_t>(std::max(height, 1)),
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height
    );
    return extent;
}

std::optional<std::filesystem::path> findFirstExistingPath(std::span<const char* const> candidates) {
    for (const char* candidate : candidates) {
        const std::filesystem::path path(candidate);
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return std::nullopt;
}

std::optional<std::vector<std::uint8_t>> readBinaryFile(std::span<const char* const> candidates) {
    const std::optional<std::filesystem::path> path = findFirstExistingPath(candidates);
    if (!path.has_value()) {
        return std::nullopt;
    }

    std::ifstream file(path.value(), std::ios::binary | std::ios::ate);
    if (!file) {
        return std::nullopt;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return std::nullopt;
    }
    file.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> data(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        return std::nullopt;
    }
    return data;
}

bool createShaderModuleFromFileOrFallback(
    VkDevice device,
    std::span<const char* const> fileCandidates,
    const std::uint32_t* fallbackCode,
    size_t fallbackCodeSize,
    const char* debugName,
    VkShaderModule& outShaderModule
) {
    outShaderModule = VK_NULL_HANDLE;

    const std::optional<std::vector<std::uint8_t>> shaderFileData = readBinaryFile(fileCandidates);
    const std::uint32_t* code = fallbackCode;
    size_t codeSize = fallbackCodeSize;
    if (shaderFileData.has_value()) {
        if ((shaderFileData->size() % sizeof(std::uint32_t)) != 0) {
            std::cerr << "[render] invalid SPIR-V byte size for " << debugName << "\n";
            return false;
        }
        code = reinterpret_cast<const std::uint32_t*>(shaderFileData->data());
        codeSize = shaderFileData->size();
    } else {
        if (fallbackCode == nullptr || fallbackCodeSize == 0) {
            std::cerr << "[render] shader file not found and no fallback available for " << debugName << "\n";
            return false;
        }
        std::cerr << "[render] using embedded fallback shader for " << debugName << "\n";
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = code;
    const VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &outShaderModule);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateShaderModule(fileOrFallback)", result);
        return false;
    }
    return true;
}

uint32_t readLeU32(const std::uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8u) |
        (static_cast<uint32_t>(bytes[2]) << 16u) |
        (static_cast<uint32_t>(bytes[3]) << 24u);
}

struct DdsCubemapData {
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 0;
    std::vector<std::uint8_t> pixelData;
    std::vector<VkBufferImageCopy> copyRegions;
};

std::optional<DdsCubemapData> loadDdsCubemap(std::span<const char* const> fileCandidates) {
    const std::optional<std::vector<std::uint8_t>> fileDataOpt = readBinaryFile(fileCandidates);
    if (!fileDataOpt.has_value()) {
        return std::nullopt;
    }
    const std::vector<std::uint8_t>& fileData = fileDataOpt.value();
    if (fileData.size() < 128) {
        return std::nullopt;
    }
    if (std::memcmp(fileData.data(), "DDS ", 4) != 0) {
        return std::nullopt;
    }

    const std::uint8_t* header = fileData.data() + 4;
    const uint32_t height = readLeU32(header + 8);
    const uint32_t width = readLeU32(header + 12);
    const uint32_t mipMapCount = std::max(1u, readLeU32(header + 24));
    const uint32_t ddspfFlags = readLeU32(header + 76);
    const uint32_t ddspfFourCC = readLeU32(header + 80);
    const uint32_t caps2 = readLeU32(header + 108);

    constexpr uint32_t kDdsFourCcDxt1 = 0x31545844u;
    constexpr uint32_t kDdsFourCcDx10 = 0x30315844u;
    constexpr uint32_t kDdsCaps2Cubemap = 0x00000200u;
    constexpr uint32_t kDdsCaps2AllFaces = 0x0000FC00u;
    constexpr uint32_t kDdsPixelFormatFourCc = 0x00000004u;
    constexpr uint32_t kDxgiFormatBc6hUfloat = 95u;
    constexpr uint32_t kDxgiResourceDimensionTexture2D = 3u;
    constexpr uint32_t kD3d11ResourceMiscTextureCube = 0x4u;

    if ((ddspfFlags & kDdsPixelFormatFourCc) == 0) {
        return std::nullopt;
    }
    if ((caps2 & kDdsCaps2Cubemap) == 0 || (caps2 & kDdsCaps2AllFaces) != kDdsCaps2AllFaces) {
        return std::nullopt;
    }

    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t blockSizeBytes = 0;
    size_t payloadOffset = 128;

    if (ddspfFourCC == kDdsFourCcDxt1) {
        format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        blockSizeBytes = 8;
    } else if (ddspfFourCC == kDdsFourCcDx10) {
        if (fileData.size() < 148) {
            return std::nullopt;
        }
        const std::uint8_t* dx10Header = fileData.data() + 128;
        const uint32_t dxgiFormat = readLeU32(dx10Header + 0);
        const uint32_t resourceDimension = readLeU32(dx10Header + 4);
        const uint32_t miscFlag = readLeU32(dx10Header + 8);
        if (
            dxgiFormat != kDxgiFormatBc6hUfloat ||
            resourceDimension != kDxgiResourceDimensionTexture2D ||
            (miscFlag & kD3d11ResourceMiscTextureCube) == 0
        ) {
            return std::nullopt;
        }
        format = VK_FORMAT_BC6H_UFLOAT_BLOCK;
        blockSizeBytes = 16;
        payloadOffset = 148;
    } else {
        return std::nullopt;
    }

    DdsCubemapData ddsData{};
    ddsData.format = format;
    ddsData.width = width;
    ddsData.height = height;
    ddsData.mipLevels = mipMapCount;

    size_t dataCursor = payloadOffset;
    for (uint32_t face = 0; face < 6; ++face) {
        for (uint32_t mip = 0; mip < mipMapCount; ++mip) {
            const uint32_t mipWidth = std::max(1u, width >> mip);
            const uint32_t mipHeight = std::max(1u, height >> mip);
            const uint32_t blocksWide = std::max(1u, (mipWidth + 3u) / 4u);
            const uint32_t blocksHigh = std::max(1u, (mipHeight + 3u) / 4u);
            const size_t mipSize = static_cast<size_t>(blocksWide) * blocksHigh * blockSizeBytes;
            if (dataCursor + mipSize > fileData.size()) {
                return std::nullopt;
            }

            VkBufferImageCopy copyRegion{};
            copyRegion.bufferOffset = static_cast<VkDeviceSize>(dataCursor - payloadOffset);
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.mipLevel = mip;
            copyRegion.imageSubresource.baseArrayLayer = face;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageOffset = {0, 0, 0};
            copyRegion.imageExtent = {mipWidth, mipHeight, 1};
            ddsData.copyRegions.push_back(copyRegion);

            dataCursor += mipSize;
        }
    }

    const size_t pixelByteCount = dataCursor - payloadOffset;
    ddsData.pixelData.resize(pixelByteCount);
    std::memcpy(ddsData.pixelData.data(), fileData.data() + payloadOffset, pixelByteCount);
    return ddsData;
}

} // namespace

bool Renderer::init(GLFWwindow* window, const world::ChunkGrid& chunkGrid) {
    std::cerr << "[render] init begin\n";
    m_window = window;
    if (m_window == nullptr) {
        std::cerr << "[render] init failed: window is null\n";
        return false;
    }

    if (glfwVulkanSupported() == GLFW_FALSE) {
        std::cerr << "[render] init failed: glfwVulkanSupported returned false\n";
        return false;
    }

    if (!createInstance()) {
        std::cerr << "[render] init failed at createInstance\n";
        shutdown();
        return false;
    }
    if (!createSurface()) {
        std::cerr << "[render] init failed at createSurface\n";
        shutdown();
        return false;
    }
    if (!pickPhysicalDevice()) {
        std::cerr << "[render] init failed at pickPhysicalDevice\n";
        shutdown();
        return false;
    }
    if (!createLogicalDevice()) {
        std::cerr << "[render] init failed at createLogicalDevice\n";
        shutdown();
        return false;
    }
    if (!createTimelineSemaphore()) {
        std::cerr << "[render] init failed at createTimelineSemaphore\n";
        shutdown();
        return false;
    }
    if (!m_bufferAllocator.init(m_physicalDevice, m_device)) {
        std::cerr << "[render] init failed at buffer allocator init\n";
        shutdown();
        return false;
    }
    if (!createUploadRingBuffer()) {
        std::cerr << "[render] init failed at createUploadRingBuffer\n";
        shutdown();
        return false;
    }
    if (!createTransferResources()) {
        std::cerr << "[render] init failed at createTransferResources\n";
        shutdown();
        return false;
    }
    if (!createEnvironmentResources()) {
        std::cerr << "[render] init failed at createEnvironmentResources\n";
        shutdown();
        return false;
    }
    if (!createShadowResources()) {
        std::cerr << "[render] init failed at createShadowResources\n";
        shutdown();
        return false;
    }
    if (!createSwapchain()) {
        std::cerr << "[render] init failed at createSwapchain\n";
        shutdown();
        return false;
    }
    if (!createDescriptorResources()) {
        std::cerr << "[render] init failed at createDescriptorResources\n";
        shutdown();
        return false;
    }
    if (!createGraphicsPipeline()) {
        std::cerr << "[render] init failed at createGraphicsPipeline\n";
        shutdown();
        return false;
    }
    if (!createChunkBuffers(chunkGrid)) {
        std::cerr << "[render] init failed at createChunkBuffers\n";
        shutdown();
        return false;
    }
    if (!createPreviewBuffers()) {
        std::cerr << "[render] init failed at createPreviewBuffers\n";
        shutdown();
        return false;
    }
    if (!createFrameResources()) {
        std::cerr << "[render] init failed at createFrameResources\n";
        shutdown();
        return false;
    }

    std::cerr << "[render] init complete\n";
    return true;
}

bool Renderer::updateChunkMesh(const world::ChunkGrid& chunkGrid) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    return createChunkBuffers(chunkGrid);
}

bool Renderer::createInstance() {
#ifndef NDEBUG
    const bool enableValidationLayers = isLayerAvailable(kValidationLayers[0]);
#else
    const bool enableValidationLayers = false;
#endif
    std::cerr << "[render] createInstance (validation="
              << (enableValidationLayers ? "on" : "off") << ")\n";

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr || glfwExtensionCount == 0) {
        std::cerr << "[render] no GLFW Vulkan instance extensions available\n";
        return false;
    }

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "voxel_factory_toy";
    applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.pEngineName = "none";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &applicationInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
    }

    const VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateInstance", result);
        return false;
    }
    return true;
}

bool Renderer::createSurface() {
    const VkResult result = glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface);
    if (result != VK_SUCCESS) {
        logVkFailure("glfwCreateWindowSurface", result);
        return false;
    }
    return true;
}

bool Renderer::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "[render] no Vulkan physical devices found\n";
        return false;
    }
    std::cerr << "[render] physical devices found: " << deviceCount << "\n";

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        std::cerr << "[render] evaluating GPU: " << properties.deviceName
                  << ", apiVersion=" << VK_VERSION_MAJOR(properties.apiVersion) << "."
                  << VK_VERSION_MINOR(properties.apiVersion) << "."
                  << VK_VERSION_PATCH(properties.apiVersion) << "\n";
        if (properties.apiVersion < VK_API_VERSION_1_3) {
            std::cerr << "[render] skip GPU: Vulkan 1.3 required\n";
            continue;
        }
        if ((properties.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0) {
            std::cerr << "[render] skip GPU: 4x MSAA color attachments not supported\n";
            continue;
        }

        const QueueFamilyChoice queueFamily = findQueueFamily(candidate, m_surface);
        if (!queueFamily.valid()) {
            std::cerr << "[render] skip GPU: missing graphics/present/transfer queue support\n";
            continue;
        }
        if (!hasRequiredDeviceExtensions(candidate)) {
            std::cerr << "[render] skip GPU: missing required device extensions\n";
            continue;
        }

        const SwapchainSupport swapchainSupport = querySwapchainSupport(candidate, m_surface);
        if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty()) {
            std::cerr << "[render] skip GPU: swapchain support incomplete\n";
            continue;
        }
        const VkFormat depthFormat = findSupportedDepthFormat(candidate);
        if (depthFormat == VK_FORMAT_UNDEFINED) {
            std::cerr << "[render] skip GPU: no supported depth format\n";
            continue;
        }
        const VkFormat shadowDepthFormat = findSupportedShadowDepthFormat(candidate);
        if (shadowDepthFormat == VK_FORMAT_UNDEFINED) {
            std::cerr << "[render] skip GPU: no supported shadow depth format\n";
            continue;
        }
        const VkFormat hdrColorFormat = findSupportedHdrColorFormat(candidate);
        if (hdrColorFormat == VK_FORMAT_UNDEFINED) {
            std::cerr << "[render] skip GPU: no supported HDR color format\n";
            continue;
        }

        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13Features.pNext = &vulkan12Features;
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &vulkan13Features;
        vkGetPhysicalDeviceFeatures2(candidate, &features2);
        if (vulkan13Features.dynamicRendering != VK_TRUE) {
            std::cerr << "[render] skip GPU: dynamicRendering not supported\n";
            continue;
        }
        if (vulkan12Features.timelineSemaphore != VK_TRUE) {
            std::cerr << "[render] skip GPU: timelineSemaphore not supported\n";
            continue;
        }
        if (vulkan13Features.synchronization2 != VK_TRUE) {
            std::cerr << "[render] skip GPU: synchronization2 not supported\n";
            continue;
        }
        if (vulkan13Features.maintenance4 != VK_TRUE) {
            std::cerr << "[render] skip GPU: maintenance4 not supported\n";
            continue;
        }

        const bool supportsWireframe = features2.features.fillModeNonSolid == VK_TRUE;
        m_physicalDevice = candidate;
        m_graphicsQueueFamilyIndex = queueFamily.graphicsAndPresent.value();
        m_graphicsQueueIndex = queueFamily.graphicsQueueIndex;
        m_transferQueueFamilyIndex = queueFamily.transfer.value();
        m_transferQueueIndex = queueFamily.transferQueueIndex;
        m_supportsWireframePreview = supportsWireframe;
        m_depthFormat = depthFormat;
        m_shadowDepthFormat = shadowDepthFormat;
        m_hdrColorFormat = hdrColorFormat;
        std::cerr << "[render] selected GPU: " << properties.deviceName
                  << ", graphicsQueueFamily=" << m_graphicsQueueFamilyIndex
                  << ", graphicsQueueIndex=" << m_graphicsQueueIndex
                  << ", transferQueueFamily=" << m_transferQueueFamilyIndex
                  << ", transferQueueIndex=" << m_transferQueueIndex
                  << ", wireframePreview=" << (m_supportsWireframePreview ? "yes" : "no")
                  << ", shadowDepthFormat=" << static_cast<int>(m_shadowDepthFormat)
                  << ", hdrColorFormat=" << static_cast<int>(m_hdrColorFormat)
                  << "\n";
        return true;
    }

    std::cerr << "[render] no suitable GPU found\n";
    return false;
}

bool Renderer::createLogicalDevice() {
    const bool sameFamily = (m_graphicsQueueFamilyIndex == m_transferQueueFamilyIndex);
    std::array<VkDeviceQueueCreateInfo, 2> queueCreateInfos{};
    uint32_t queueCreateInfoCount = 0;
    std::array<float, 2> sharedFamilyPriorities = {1.0f, 1.0f};
    float graphicsQueuePriority = 1.0f;
    float transferQueuePriority = 1.0f;

    if (sameFamily) {
        const uint32_t queueCount = std::max(m_graphicsQueueIndex, m_transferQueueIndex) + 1;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
        queueCreateInfo.queueCount = queueCount;
        queueCreateInfo.pQueuePriorities = sharedFamilyPriorities.data();
        queueCreateInfos[queueCreateInfoCount++] = queueCreateInfo;
    } else {
        VkDeviceQueueCreateInfo graphicsQueueCreateInfo{};
        graphicsQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        graphicsQueueCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
        graphicsQueueCreateInfo.queueCount = m_graphicsQueueIndex + 1;
        graphicsQueueCreateInfo.pQueuePriorities = &graphicsQueuePriority;
        queueCreateInfos[queueCreateInfoCount++] = graphicsQueueCreateInfo;

        VkDeviceQueueCreateInfo transferQueueCreateInfo{};
        transferQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        transferQueueCreateInfo.queueFamilyIndex = m_transferQueueFamilyIndex;
        transferQueueCreateInfo.queueCount = m_transferQueueIndex + 1;
        transferQueueCreateInfo.pQueuePriorities = &transferQueuePriority;
        queueCreateInfos[queueCreateInfoCount++] = transferQueueCreateInfo;
    }

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &vulkan12Features;
    vulkan13Features.dynamicRendering = VK_TRUE;
    vulkan13Features.synchronization2 = VK_TRUE;
    vulkan13Features.maintenance4 = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &vulkan13Features;
    createInfo.queueCreateInfoCount = queueCreateInfoCount;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.fillModeNonSolid = m_supportsWireframePreview ? VK_TRUE : VK_FALSE;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();

    const VkResult result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateDevice", result);
        return false;
    }

    vkGetDeviceQueue(m_device, m_graphicsQueueFamilyIndex, m_graphicsQueueIndex, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_transferQueueFamilyIndex, m_transferQueueIndex, &m_transferQueue);

    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &deviceProperties);
    m_uniformBufferAlignment = std::max<VkDeviceSize>(
        deviceProperties.limits.minUniformBufferOffsetAlignment,
        static_cast<VkDeviceSize>(16)
    );
    return true;
}

bool Renderer::createTimelineSemaphore() {
    if (m_renderTimelineSemaphore != VK_NULL_HANDLE) {
        return true;
    }

    VkSemaphoreTypeCreateInfo timelineCreateInfo{};
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateInfo.initialValue = 0;

    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = &timelineCreateInfo;

    const VkResult result = vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_renderTimelineSemaphore);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateSemaphore(timeline)", result);
        return false;
    }

    m_frameTimelineValues.fill(0);
    m_pendingTransferTimelineValue = 0;
    m_currentChunkReadyTimelineValue = 0;
    m_transferCommandBufferInFlightValue = 0;
    m_lastGraphicsTimelineValue = 0;
    m_nextTimelineValue = 1;
    return true;
}

bool Renderer::createUploadRingBuffer() {
    // Minimal per-frame ring buffer used for small CPU uploads.
    // Future streaming code can replace this with dedicated staging allocators.
    const bool ok = m_uploadRing.init(
        &m_bufferAllocator,
        1024 * 64,
        kMaxFramesInFlight,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    );
    if (!ok) {
        std::cerr << "[render] upload ring buffer init failed\n";
    }
    return ok;
}

bool Renderer::createTransferResources() {
    if (m_transferCommandPool != VK_NULL_HANDLE && m_transferCommandBuffer != VK_NULL_HANDLE) {
        return true;
    }

    VkCommandPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCreateInfo.queueFamilyIndex = m_transferQueueFamilyIndex;

    const VkResult poolResult = vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &m_transferCommandPool);
    if (poolResult != VK_SUCCESS) {
        logVkFailure("vkCreateCommandPool(transfer)", poolResult);
        return false;
    }

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = m_transferCommandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    const VkResult commandBufferResult = vkAllocateCommandBuffers(m_device, &allocateInfo, &m_transferCommandBuffer);
    if (commandBufferResult != VK_SUCCESS) {
        logVkFailure("vkAllocateCommandBuffers(transfer)", commandBufferResult);
        vkDestroyCommandPool(m_device, m_transferCommandPool, nullptr);
        m_transferCommandPool = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

bool Renderer::createPreviewBuffers() {
    if (m_previewVertexBufferHandle != kInvalidBufferHandle && m_previewIndexBufferHandle != kInvalidBufferHandle) {
        return true;
    }

    const world::ChunkMeshData mesh = buildSingleVoxelPreviewMesh(0, 0, 0, 3, 1);
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        std::cerr << "[render] preview mesh build failed\n";
        return false;
    }

    BufferCreateDesc vertexCreateDesc{};
    vertexCreateDesc.size = static_cast<VkDeviceSize>(mesh.vertices.size() * sizeof(world::PackedVoxelVertex));
    vertexCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vertexCreateDesc.initialData = mesh.vertices.data();
    m_previewVertexBufferHandle = m_bufferAllocator.createBuffer(vertexCreateDesc);
    if (m_previewVertexBufferHandle == kInvalidBufferHandle) {
        std::cerr << "[render] preview vertex buffer allocation failed\n";
        return false;
    }

    BufferCreateDesc indexCreateDesc{};
    indexCreateDesc.size = static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(std::uint32_t));
    indexCreateDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    indexCreateDesc.initialData = mesh.indices.data();
    m_previewIndexBufferHandle = m_bufferAllocator.createBuffer(indexCreateDesc);
    if (m_previewIndexBufferHandle == kInvalidBufferHandle) {
        std::cerr << "[render] preview index buffer allocation failed\n";
        m_bufferAllocator.destroyBuffer(m_previewVertexBufferHandle);
        m_previewVertexBufferHandle = kInvalidBufferHandle;
        return false;
    }

    m_previewIndexCount = static_cast<uint32_t>(mesh.indices.size());
    return true;
}

bool Renderer::createEnvironmentResources() {
    if (
        m_skyboxTexture.image != VK_NULL_HANDLE &&
        m_skyboxTexture.view != VK_NULL_HANDLE &&
        m_skyboxTexture.sampler != VK_NULL_HANDLE &&
        m_irradianceTexture.image != VK_NULL_HANDLE &&
        m_irradianceTexture.view != VK_NULL_HANDLE &&
        m_irradianceTexture.sampler != VK_NULL_HANDLE
    ) {
        return true;
    }

    auto createCubemapTexture = [&](std::span<const char* const> fileCandidates, CubemapTexture& outTexture) -> bool {
        const std::optional<DdsCubemapData> ddsDataOpt = loadDdsCubemap(fileCandidates);
        if (!ddsDataOpt.has_value()) {
            std::cerr << "[render] failed to load cubemap DDS\n";
            return false;
        }
        const DdsCubemapData& ddsData = ddsDataOpt.value();

        VkFormatProperties formatProperties{};
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, ddsData.format, &formatProperties);
        const bool supportsSampling =
            (formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
        const bool supportsTransferDst =
            (formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) != 0;
        if (!supportsSampling || !supportsTransferDst) {
            std::cerr << "[render] cubemap format not supported for sampled+transfer dst\n";
            return false;
        }

        BufferCreateDesc stagingCreateDesc{};
        stagingCreateDesc.size = static_cast<VkDeviceSize>(ddsData.pixelData.size());
        stagingCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        stagingCreateDesc.initialData = ddsData.pixelData.data();
        const BufferHandle stagingHandle = m_bufferAllocator.createBuffer(stagingCreateDesc);
        if (stagingHandle == kInvalidBufferHandle) {
            std::cerr << "[render] failed to allocate cubemap staging buffer\n";
            return false;
        }

        std::array<uint32_t, 2> queueFamilies = {
            m_graphicsQueueFamilyIndex,
            m_transferQueueFamilyIndex
        };
        const bool useConcurrentSharing = queueFamilies[0] != queueFamilies[1];

        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = ddsData.format;
        imageCreateInfo.extent.width = ddsData.width;
        imageCreateInfo.extent.height = ddsData.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = ddsData.mipLevels;
        imageCreateInfo.arrayLayers = 6;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCreateInfo.sharingMode = useConcurrentSharing ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.queueFamilyIndexCount = useConcurrentSharing ? 2u : 0u;
        imageCreateInfo.pQueueFamilyIndices = useConcurrentSharing ? queueFamilies.data() : nullptr;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage image = VK_NULL_HANDLE;
        const VkResult imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &image);
        if (imageResult != VK_SUCCESS) {
            logVkFailure("vkCreateImage(cubemap)", imageResult);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(m_device, image, &memoryRequirements);
        const uint32_t memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice,
            memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            std::cerr << "[render] no memory type for cubemap image\n";
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &memory);
        if (allocResult != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(cubemap)", allocResult);
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        const VkResult bindResult = vkBindImageMemory(m_device, image, memory, 0);
        if (bindResult != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(cubemap)", bindResult);
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        if (m_transferCommandBufferInFlightValue > 0 && !waitForTimelineValue(m_transferCommandBufferInFlightValue)) {
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }
        m_transferCommandBufferInFlightValue = 0;

        if (vkResetCommandPool(m_device, m_transferCommandPool, 0) != VK_SUCCESS) {
            std::cerr << "[render] vkResetCommandPool failed for cubemap upload\n";
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_transferCommandBuffer, &beginInfo) != VK_SUCCESS) {
            std::cerr << "[render] vkBeginCommandBuffer failed for cubemap upload\n";
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        VkImageMemoryBarrier2 toTransferBarrier{};
        toTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toTransferBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        toTransferBarrier.srcAccessMask = VK_ACCESS_2_NONE;
        toTransferBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        toTransferBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferBarrier.image = image;
        toTransferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toTransferBarrier.subresourceRange.baseMipLevel = 0;
        toTransferBarrier.subresourceRange.levelCount = ddsData.mipLevels;
        toTransferBarrier.subresourceRange.baseArrayLayer = 0;
        toTransferBarrier.subresourceRange.layerCount = 6;

        VkDependencyInfo toTransferDependency{};
        toTransferDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        toTransferDependency.imageMemoryBarrierCount = 1;
        toTransferDependency.pImageMemoryBarriers = &toTransferBarrier;
        vkCmdPipelineBarrier2(m_transferCommandBuffer, &toTransferDependency);

        const VkBuffer stagingBuffer = m_bufferAllocator.getBuffer(stagingHandle);
        vkCmdCopyBufferToImage(
            m_transferCommandBuffer,
            stagingBuffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<uint32_t>(ddsData.copyRegions.size()),
            ddsData.copyRegions.data()
        );

        VkImageMemoryBarrier2 toShaderReadBarrier{};
        toShaderReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toShaderReadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        toShaderReadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toShaderReadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        toShaderReadBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        toShaderReadBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toShaderReadBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toShaderReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toShaderReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toShaderReadBarrier.image = image;
        toShaderReadBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toShaderReadBarrier.subresourceRange.baseMipLevel = 0;
        toShaderReadBarrier.subresourceRange.levelCount = ddsData.mipLevels;
        toShaderReadBarrier.subresourceRange.baseArrayLayer = 0;
        toShaderReadBarrier.subresourceRange.layerCount = 6;

        VkDependencyInfo toShaderReadDependency{};
        toShaderReadDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        toShaderReadDependency.imageMemoryBarrierCount = 1;
        toShaderReadDependency.pImageMemoryBarriers = &toShaderReadBarrier;
        vkCmdPipelineBarrier2(m_transferCommandBuffer, &toShaderReadDependency);

        if (vkEndCommandBuffer(m_transferCommandBuffer) != VK_SUCCESS) {
            std::cerr << "[render] vkEndCommandBuffer failed for cubemap upload\n";
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence uploadFence = VK_NULL_HANDLE;
        if (vkCreateFence(m_device, &fenceCreateInfo, nullptr, &uploadFence) != VK_SUCCESS) {
            std::cerr << "[render] failed to create fence for cubemap upload\n";
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_transferCommandBuffer;
        const VkResult submitResult = vkQueueSubmit(m_transferQueue, 1, &submitInfo, uploadFence);
        if (submitResult != VK_SUCCESS) {
            logVkFailure("vkQueueSubmit(cubemapUpload)", submitResult);
            vkDestroyFence(m_device, uploadFence, nullptr);
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }
        const VkResult waitFenceResult = vkWaitForFences(m_device, 1, &uploadFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
        vkDestroyFence(m_device, uploadFence, nullptr);
        if (waitFenceResult != VK_SUCCESS) {
            logVkFailure("vkWaitForFences(cubemapUpload)", waitFenceResult);
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        m_bufferAllocator.destroyBuffer(stagingHandle);

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = image;
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        viewCreateInfo.format = ddsData.format;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = ddsData.mipLevels;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 6;

        VkImageView view = VK_NULL_HANDLE;
        const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &view);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(cubemap)", viewResult);
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            return false;
        }

        VkSamplerCreateInfo samplerCreateInfo{};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = static_cast<float>(ddsData.mipLevels - 1);
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

        VkSampler sampler = VK_NULL_HANDLE;
        const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &sampler);
        if (samplerResult != VK_SUCCESS) {
            logVkFailure("vkCreateSampler(cubemap)", samplerResult);
            vkDestroyImageView(m_device, view, nullptr);
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyImage(m_device, image, nullptr);
            return false;
        }

        outTexture.image = image;
        outTexture.memory = memory;
        outTexture.view = view;
        outTexture.sampler = sampler;
        outTexture.mipLevels = ddsData.mipLevels;
        return true;
    };

    constexpr std::array<const char*, 3> kSkyboxFileCandidates = {
        "assets/skybox/skybox.dds",
        "../assets/skybox/skybox.dds",
        "../../assets/skybox/skybox.dds",
    };
    constexpr std::array<const char*, 3> kIrradianceFileCandidates = {
        "assets/skybox/irradiance.dds",
        "../assets/skybox/irradiance.dds",
        "../../assets/skybox/irradiance.dds",
    };

    if (!createCubemapTexture(kSkyboxFileCandidates, m_skyboxTexture)) {
        destroyEnvironmentResources();
        return false;
    }
    if (!createCubemapTexture(kIrradianceFileCandidates, m_irradianceTexture)) {
        destroyEnvironmentResources();
        return false;
    }

    std::cerr << "[render] environment cubemaps ready\n";
    return true;
}

bool Renderer::createShadowResources() {
    if (
        m_shadowDepthImage != VK_NULL_HANDLE &&
        m_shadowDepthMemory != VK_NULL_HANDLE &&
        m_shadowDepthImageView != VK_NULL_HANDLE &&
        m_shadowDepthSampler != VK_NULL_HANDLE
    ) {
        return true;
    }

    if (m_shadowDepthFormat == VK_FORMAT_UNDEFINED) {
        std::cerr << "[render] shadow depth format is undefined\n";
        return false;
    }

    constexpr uint32_t kShadowMapSize = 2048;

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = m_shadowDepthFormat;
    imageCreateInfo.extent.width = kShadowMapSize;
    imageCreateInfo.extent.height = kShadowMapSize;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    const VkResult imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_shadowDepthImage);
    if (imageResult != VK_SUCCESS) {
        logVkFailure("vkCreateImage(shadowDepth)", imageResult);
        return false;
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetImageMemoryRequirements(m_device, m_shadowDepthImage, &memoryRequirements);
    const uint32_t memoryTypeIndex = findMemoryTypeIndex(
        m_physicalDevice,
        memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
        std::cerr << "[render] no memory type for shadow depth image\n";
        destroyShadowResources();
        return false;
    }

    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;
    const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_shadowDepthMemory);
    if (allocResult != VK_SUCCESS) {
        logVkFailure("vkAllocateMemory(shadowDepth)", allocResult);
        destroyShadowResources();
        return false;
    }

    const VkResult bindResult = vkBindImageMemory(m_device, m_shadowDepthImage, m_shadowDepthMemory, 0);
    if (bindResult != VK_SUCCESS) {
        logVkFailure("vkBindImageMemory(shadowDepth)", bindResult);
        destroyShadowResources();
        return false;
    }

    VkImageViewCreateInfo viewCreateInfo{};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = m_shadowDepthImage;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = m_shadowDepthFormat;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;
    const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_shadowDepthImageView);
    if (viewResult != VK_SUCCESS) {
        logVkFailure("vkCreateImageView(shadowDepth)", viewResult);
        destroyShadowResources();
        return false;
    }

    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareEnable = VK_TRUE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_shadowDepthSampler);
    if (samplerResult != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(shadowDepth)", samplerResult);
        destroyShadowResources();
        return false;
    }

    m_shadowDepthInitialized = false;
    std::cerr << "[render] shadow resources ready (" << kShadowMapSize << "x" << kShadowMapSize << ")\n";
    return true;
}

bool Renderer::createSwapchain() {
    const SwapchainSupport support = querySwapchainSupport(m_physicalDevice, m_surface);
    if (support.formats.empty() || support.presentModes.empty()) {
        std::cerr << "[render] swapchain support query returned no formats or present modes\n";
        return false;
    }

    const VkSurfaceFormatKHR surfaceFormat = chooseSwapchainFormat(support.formats);
    const VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
    const VkExtent2D extent = chooseExtent(m_window, support.capabilities);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    const VkResult swapchainResult = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain);
    if (swapchainResult != VK_SUCCESS) {
        logVkFailure("vkCreateSwapchainKHR", swapchainResult);
        return false;
    }

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_swapchainImages.data());

    m_swapchainFormat = surfaceFormat.format;
    m_swapchainExtent = extent;

    m_swapchainImageViews.resize(imageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_swapchainImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_swapchainFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_swapchainImageViews[i]) != VK_SUCCESS) {
            std::cerr << "[render] failed to create swapchain image view " << i << "\n";
            return false;
        }
    }

    std::cerr << "[render] swapchain ready: images=" << imageCount
              << ", extent=" << m_swapchainExtent.width << "x" << m_swapchainExtent.height << "\n";
    m_swapchainImageInitialized.assign(imageCount, false);
    m_swapchainImageTimelineValues.assign(imageCount, 0);
    if (!createHdrResolveTargets()) {
        std::cerr << "[render] HDR resolve target creation failed\n";
        return false;
    }
    if (!createMsaaColorTargets()) {
        std::cerr << "[render] MSAA color target creation failed\n";
        return false;
    }
    if (!createDepthTargets()) {
        std::cerr << "[render] depth target creation failed\n";
        return false;
    }
    m_renderFinishedSemaphores.resize(imageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        const VkResult semaphoreResult =
            vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_renderFinishedSemaphores[i]);
        if (semaphoreResult != VK_SUCCESS) {
            logVkFailure("vkCreateSemaphore(renderFinishedPerImage)", semaphoreResult);
            return false;
        }
    }

    return true;
}

bool Renderer::createDepthTargets() {
    if (m_depthFormat == VK_FORMAT_UNDEFINED) {
        std::cerr << "[render] depth format is undefined\n";
        return false;
    }

    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_depthImages.assign(imageCount, VK_NULL_HANDLE);
    m_depthImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_depthImageViews.assign(imageCount, VK_NULL_HANDLE);

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_depthFormat;
        imageCreateInfo.extent.width = m_swapchainExtent.width;
        imageCreateInfo.extent.height = m_swapchainExtent.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = m_colorSampleCount;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        const VkResult imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_depthImages[i]);
        if (imageResult != VK_SUCCESS) {
            logVkFailure("vkCreateImage(depth)", imageResult);
            return false;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(m_device, m_depthImages[i], &memoryRequirements);

        const uint32_t memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice,
            memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            std::cerr << "[render] no memory type for depth image\n";
            return false;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;

        const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_depthImageMemories[i]);
        if (allocResult != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(depth)", allocResult);
            return false;
        }

        const VkResult bindResult = vkBindImageMemory(m_device, m_depthImages[i], m_depthImageMemories[i], 0);
        if (bindResult != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(depth)", bindResult);
            return false;
        }

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_depthImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_depthFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_depthImageViews[i]);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(depth)", viewResult);
            return false;
        }
    }

    return true;
}

bool Renderer::createHdrResolveTargets() {
    if (m_hdrColorFormat == VK_FORMAT_UNDEFINED) {
        std::cerr << "[render] HDR color format is undefined\n";
        return false;
    }

    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_hdrResolveImages.assign(imageCount, VK_NULL_HANDLE);
    m_hdrResolveImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_hdrResolveImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_hdrResolveImageInitialized.assign(imageCount, false);

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_hdrColorFormat;
        imageCreateInfo.extent.width = m_swapchainExtent.width;
        imageCreateInfo.extent.height = m_swapchainExtent.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        const VkResult imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_hdrResolveImages[i]);
        if (imageResult != VK_SUCCESS) {
            logVkFailure("vkCreateImage(hdrResolve)", imageResult);
            return false;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(m_device, m_hdrResolveImages[i], &memoryRequirements);

        const uint32_t memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice,
            memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            std::cerr << "[render] no memory type for HDR resolve image\n";
            return false;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;

        const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_hdrResolveImageMemories[i]);
        if (allocResult != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(hdrResolve)", allocResult);
            return false;
        }

        const VkResult bindResult = vkBindImageMemory(m_device, m_hdrResolveImages[i], m_hdrResolveImageMemories[i], 0);
        if (bindResult != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(hdrResolve)", bindResult);
            return false;
        }

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_hdrResolveImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_hdrColorFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_hdrResolveImageViews[i]);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(hdrResolve)", viewResult);
            return false;
        }
    }

    if (m_hdrResolveSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerCreateInfo{};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

        const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_hdrResolveSampler);
        if (samplerResult != VK_SUCCESS) {
            logVkFailure("vkCreateSampler(hdrResolve)", samplerResult);
            return false;
        }
    }

    return true;
}

bool Renderer::createMsaaColorTargets() {
    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_msaaColorImages.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageInitialized.assign(imageCount, false);

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_hdrColorFormat;
        imageCreateInfo.extent.width = m_swapchainExtent.width;
        imageCreateInfo.extent.height = m_swapchainExtent.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = m_colorSampleCount;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        const VkResult imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_msaaColorImages[i]);
        if (imageResult != VK_SUCCESS) {
            logVkFailure("vkCreateImage(msaaColor)", imageResult);
            return false;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(m_device, m_msaaColorImages[i], &memoryRequirements);

        const uint32_t memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice,
            memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            std::cerr << "[render] no memory type for MSAA color image\n";
            return false;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;

        const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_msaaColorImageMemories[i]);
        if (allocResult != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(msaaColor)", allocResult);
            return false;
        }

        const VkResult bindResult = vkBindImageMemory(m_device, m_msaaColorImages[i], m_msaaColorImageMemories[i], 0);
        if (bindResult != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(msaaColor)", bindResult);
            return false;
        }

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_msaaColorImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_hdrColorFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_msaaColorImageViews[i]);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(msaaColor)", viewResult);
            return false;
        }
    }

    return true;
}

bool Renderer::createDescriptorResources() {
    if (m_descriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding mvpBinding{};
        mvpBinding.binding = 0;
        mvpBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        mvpBinding.descriptorCount = 1;
        mvpBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding irradianceBinding{};
        irradianceBinding.binding = 1;
        irradianceBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        irradianceBinding.descriptorCount = 1;
        irradianceBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding skyboxBinding{};
        skyboxBinding.binding = 2;
        skyboxBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        skyboxBinding.descriptorCount = 1;
        skyboxBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding hdrSceneBinding{};
        hdrSceneBinding.binding = 3;
        hdrSceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        hdrSceneBinding.descriptorCount = 1;
        hdrSceneBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding shadowMapBinding{};
        shadowMapBinding.binding = 4;
        shadowMapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowMapBinding.descriptorCount = 1;
        shadowMapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 5> bindings = {
            mvpBinding,
            irradianceBinding,
            skyboxBinding,
            hdrSceneBinding,
            shadowMapBinding
        };

        VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCreateInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutCreateInfo.pBindings = bindings.data();

        const VkResult layoutResult =
            vkCreateDescriptorSetLayout(m_device, &layoutCreateInfo, nullptr, &m_descriptorSetLayout);
        if (layoutResult != VK_SUCCESS) {
            logVkFailure("vkCreateDescriptorSetLayout", layoutResult);
            return false;
        }
    }

    if (m_descriptorPool == VK_NULL_HANDLE) {
        const std::array<VkDescriptorPoolSize, 2> poolSizes = {
            VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                kMaxFramesInFlight
            },
            VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                4 * kMaxFramesInFlight
            }
        };

        VkDescriptorPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCreateInfo.maxSets = kMaxFramesInFlight;
        poolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolCreateInfo.pPoolSizes = poolSizes.data();

        const VkResult poolResult = vkCreateDescriptorPool(m_device, &poolCreateInfo, nullptr, &m_descriptorPool);
        if (poolResult != VK_SUCCESS) {
            logVkFailure("vkCreateDescriptorPool", poolResult);
            return false;
        }
    }

    std::array<VkDescriptorSetLayout, kMaxFramesInFlight> setLayouts{};
    setLayouts.fill(m_descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = m_descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
    allocateInfo.pSetLayouts = setLayouts.data();

    const VkResult allocateResult = vkAllocateDescriptorSets(m_device, &allocateInfo, m_descriptorSets.data());
    if (allocateResult != VK_SUCCESS) {
        logVkFailure("vkAllocateDescriptorSets", allocateResult);
        return false;
    }

    return true;
}

bool Renderer::createGraphicsPipeline() {
    if (m_depthFormat == VK_FORMAT_UNDEFINED) {
        std::cerr << "[render] cannot create pipeline: depth format undefined\n";
        return false;
    }
    if (m_hdrColorFormat == VK_FORMAT_UNDEFINED) {
        std::cerr << "[render] cannot create pipeline: HDR color format undefined\n";
        return false;
    }
    if (m_shadowDepthFormat == VK_FORMAT_UNDEFINED) {
        std::cerr << "[render] cannot create pipeline: shadow depth format undefined\n";
        return false;
    }

    if (m_pipelineLayout == VK_NULL_HANDLE) {
        VkPushConstantRange chunkPushConstantRange{};
        chunkPushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        chunkPushConstantRange.offset = 0;
        chunkPushConstantRange.size = sizeof(ChunkPushConstants);

        VkPipelineLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCreateInfo.setLayoutCount = 1;
        layoutCreateInfo.pSetLayouts = &m_descriptorSetLayout;
        layoutCreateInfo.pushConstantRangeCount = 1;
        layoutCreateInfo.pPushConstantRanges = &chunkPushConstantRange;
        const VkResult layoutResult = vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_pipelineLayout);
        if (layoutResult != VK_SUCCESS) {
            logVkFailure("vkCreatePipelineLayout", layoutResult);
            return false;
        }
    }

    constexpr std::array<const char*, 4> kWorldVertexShaderPathCandidates = {
        "src/render/shaders/voxel_packed.vert.spv",
        "../src/render/shaders/voxel_packed.vert.spv",
        "../../src/render/shaders/voxel_packed.vert.spv",
        "../../../src/render/shaders/voxel_packed.vert.spv",
    };
    constexpr std::array<const char*, 4> kWorldFragmentShaderPathCandidates = {
        "src/render/shaders/voxel_packed.frag.spv",
        "../src/render/shaders/voxel_packed.frag.spv",
        "../../src/render/shaders/voxel_packed.frag.spv",
        "../../../src/render/shaders/voxel_packed.frag.spv",
    };
    constexpr std::array<const char*, 4> kSkyboxVertexShaderPathCandidates = {
        "src/render/shaders/skybox.vert.spv",
        "../src/render/shaders/skybox.vert.spv",
        "../../src/render/shaders/skybox.vert.spv",
        "../../../src/render/shaders/skybox.vert.spv",
    };
    constexpr std::array<const char*, 4> kSkyboxFragmentShaderPathCandidates = {
        "src/render/shaders/skybox.frag.spv",
        "../src/render/shaders/skybox.frag.spv",
        "../../src/render/shaders/skybox.frag.spv",
        "../../../src/render/shaders/skybox.frag.spv",
    };
    constexpr std::array<const char*, 4> kToneMapVertexShaderPathCandidates = {
        "src/render/shaders/tone_map.vert.spv",
        "../src/render/shaders/tone_map.vert.spv",
        "../../src/render/shaders/tone_map.vert.spv",
        "../../../src/render/shaders/tone_map.vert.spv",
    };
    constexpr std::array<const char*, 4> kToneMapFragmentShaderPathCandidates = {
        "src/render/shaders/tone_map.frag.spv",
        "../src/render/shaders/tone_map.frag.spv",
        "../../src/render/shaders/tone_map.frag.spv",
        "../../../src/render/shaders/tone_map.frag.spv",
    };
    constexpr std::array<const char*, 4> kShadowVertexShaderPathCandidates = {
        "src/render/shaders/shadow_depth.vert.spv",
        "../src/render/shaders/shadow_depth.vert.spv",
        "../../src/render/shaders/shadow_depth.vert.spv",
        "../../../src/render/shaders/shadow_depth.vert.spv",
    };
    constexpr std::array<const char*, 4> kShadowFragmentShaderPathCandidates = {
        "src/render/shaders/shadow_depth.frag.spv",
        "../src/render/shaders/shadow_depth.frag.spv",
        "../../src/render/shaders/shadow_depth.frag.spv",
        "../../../src/render/shaders/shadow_depth.frag.spv",
    };

    VkShaderModule worldVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule worldFragShaderModule = VK_NULL_HANDLE;
    VkShaderModule skyboxVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule skyboxFragShaderModule = VK_NULL_HANDLE;
    VkShaderModule toneMapVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule toneMapFragShaderModule = VK_NULL_HANDLE;
    VkShaderModule shadowVertShaderModule = VK_NULL_HANDLE;
    VkShaderModule shadowFragShaderModule = VK_NULL_HANDLE;

    if (!createShaderModuleFromFileOrFallback(
            m_device,
            kWorldVertexShaderPathCandidates,
            kVertShaderSpirv,
            sizeof(kVertShaderSpirv),
            "voxel_packed.vert",
            worldVertShaderModule
        )) {
        return false;
    }
    if (!createShaderModuleFromFileOrFallback(
            m_device,
            kWorldFragmentShaderPathCandidates,
            kFragShaderSpirv,
            sizeof(kFragShaderSpirv),
            "voxel_packed.frag",
            worldFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }
    if (!createShaderModuleFromFileOrFallback(
            m_device,
            kSkyboxVertexShaderPathCandidates,
            nullptr,
            0,
            "skybox.vert",
            skyboxVertShaderModule
        )) {
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }
    if (!createShaderModuleFromFileOrFallback(
            m_device,
            kSkyboxFragmentShaderPathCandidates,
            nullptr,
            0,
            "skybox.frag",
            skyboxFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }
    if (!createShaderModuleFromFileOrFallback(
            m_device,
            kToneMapVertexShaderPathCandidates,
            nullptr,
            0,
            "tone_map.vert",
            toneMapVertShaderModule
        )) {
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }
    if (!createShaderModuleFromFileOrFallback(
            m_device,
            kToneMapFragmentShaderPathCandidates,
            nullptr,
            0,
            "tone_map.frag",
            toneMapFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo worldVertexShaderStage{};
    worldVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    worldVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    worldVertexShaderStage.module = worldVertShaderModule;
    worldVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo worldFragmentShaderStage{};
    worldFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    worldFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    worldFragmentShaderStage.module = worldFragShaderModule;
    worldFragmentShaderStage.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 2> worldShaderStages = {worldVertexShaderStage, worldFragmentShaderStage};

    // Vertex fetch reads one packed uint per vertex.
    // The vertex shader unpacks position/face/corner/AO procedurally.
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(world::PackedVoxelVertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescription{};
    attributeDescription.location = 0;
    attributeDescription.binding = 0;
    attributeDescription.format = VK_FORMAT_R32_UINT;
    attributeDescription.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = m_colorSampleCount;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingCreateInfo{};
    renderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingCreateInfo.colorAttachmentCount = 1;
    renderingCreateInfo.pColorAttachmentFormats = &m_hdrColorFormat;
    renderingCreateInfo.depthAttachmentFormat = m_depthFormat;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.pNext = &renderingCreateInfo;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(worldShaderStages.size());
    pipelineCreateInfo.pStages = worldShaderStages.data();
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = m_pipelineLayout;
    pipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    pipelineCreateInfo.subpass = 0;

    VkPipeline worldPipeline = VK_NULL_HANDLE;
    const VkResult worldPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipelineCreateInfo,
        nullptr,
        &worldPipeline
    );
    if (worldPipelineResult != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(world)", worldPipelineResult);
        return false;
    }

    VkPipelineRasterizationStateCreateInfo previewAddRasterizer = rasterizer;
    previewAddRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    previewAddRasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    previewAddRasterizer.depthBiasEnable = VK_TRUE;

    VkPipelineRasterizationStateCreateInfo previewRemoveRasterizer = rasterizer;
    previewRemoveRasterizer.polygonMode = m_supportsWireframePreview ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    previewRemoveRasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    previewRemoveRasterizer.depthBiasEnable = VK_TRUE;

    VkPipelineDepthStencilStateCreateInfo previewDepthStencil = depthStencil;
    previewDepthStencil.depthWriteEnable = VK_FALSE;
    previewDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    std::array<VkDynamicState, 3> previewDynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    VkPipelineDynamicStateCreateInfo previewDynamicState{};
    previewDynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    previewDynamicState.dynamicStateCount = static_cast<uint32_t>(previewDynamicStates.size());
    previewDynamicState.pDynamicStates = previewDynamicStates.data();

    VkGraphicsPipelineCreateInfo previewAddPipelineCreateInfo = pipelineCreateInfo;
    previewAddPipelineCreateInfo.pRasterizationState = &previewAddRasterizer;
    previewAddPipelineCreateInfo.pDepthStencilState = &previewDepthStencil;
    previewAddPipelineCreateInfo.pDynamicState = &previewDynamicState;

    VkPipeline previewAddPipeline = VK_NULL_HANDLE;
    const VkResult previewAddPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &previewAddPipelineCreateInfo,
        nullptr,
        &previewAddPipeline
    );
    if (previewAddPipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(previewAdd)", previewAddPipelineResult);
        return false;
    }

    VkGraphicsPipelineCreateInfo previewRemovePipelineCreateInfo = pipelineCreateInfo;
    previewRemovePipelineCreateInfo.pRasterizationState = &previewRemoveRasterizer;
    previewRemovePipelineCreateInfo.pDepthStencilState = &previewDepthStencil;
    previewRemovePipelineCreateInfo.pDynamicState = &previewDynamicState;

    VkPipeline previewRemovePipeline = VK_NULL_HANDLE;
    const VkResult previewRemovePipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &previewRemovePipelineCreateInfo,
        nullptr,
        &previewRemovePipeline
    );

    if (previewRemovePipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(previewRemove)", previewRemovePipelineResult);
        return false;
    }

    VkPipelineShaderStageCreateInfo skyboxVertexShaderStage{};
    skyboxVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    skyboxVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    skyboxVertexShaderStage.module = skyboxVertShaderModule;
    skyboxVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo skyboxFragmentShaderStage{};
    skyboxFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    skyboxFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    skyboxFragmentShaderStage.module = skyboxFragShaderModule;
    skyboxFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> skyboxShaderStages = {
        skyboxVertexShaderStage,
        skyboxFragmentShaderStage
    };

    VkPipelineVertexInputStateCreateInfo skyboxVertexInputInfo{};
    skyboxVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo skyboxInputAssembly = inputAssembly;

    VkPipelineRasterizationStateCreateInfo skyboxRasterizer = rasterizer;
    skyboxRasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineDepthStencilStateCreateInfo skyboxDepthStencil = depthStencil;
    skyboxDepthStencil.depthTestEnable = VK_FALSE;
    skyboxDepthStencil.depthWriteEnable = VK_FALSE;
    skyboxDepthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    VkGraphicsPipelineCreateInfo skyboxPipelineCreateInfo = pipelineCreateInfo;
    skyboxPipelineCreateInfo.stageCount = static_cast<uint32_t>(skyboxShaderStages.size());
    skyboxPipelineCreateInfo.pStages = skyboxShaderStages.data();
    skyboxPipelineCreateInfo.pVertexInputState = &skyboxVertexInputInfo;
    skyboxPipelineCreateInfo.pInputAssemblyState = &skyboxInputAssembly;
    skyboxPipelineCreateInfo.pDepthStencilState = &skyboxDepthStencil;
    skyboxPipelineCreateInfo.pRasterizationState = &skyboxRasterizer;

    VkPipeline skyboxPipeline = VK_NULL_HANDLE;
    const VkResult skyboxPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &skyboxPipelineCreateInfo,
        nullptr,
        &skyboxPipeline
    );

    if (skyboxPipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(skybox)", skyboxPipelineResult);
        return false;
    }

    VkPipelineShaderStageCreateInfo toneMapVertexShaderStage{};
    toneMapVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    toneMapVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    toneMapVertexShaderStage.module = toneMapVertShaderModule;
    toneMapVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo toneMapFragmentShaderStage{};
    toneMapFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    toneMapFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    toneMapFragmentShaderStage.module = toneMapFragShaderModule;
    toneMapFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> toneMapShaderStages = {
        toneMapVertexShaderStage,
        toneMapFragmentShaderStage
    };

    VkPipelineVertexInputStateCreateInfo toneMapVertexInputInfo{};
    toneMapVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo toneMapInputAssembly = inputAssembly;

    VkPipelineRasterizationStateCreateInfo toneMapRasterizer = rasterizer;
    toneMapRasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo toneMapMultisampling{};
    toneMapMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    toneMapMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo toneMapDepthStencil{};
    toneMapDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    toneMapDepthStencil.depthTestEnable = VK_FALSE;
    toneMapDepthStencil.depthWriteEnable = VK_FALSE;
    toneMapDepthStencil.depthBoundsTestEnable = VK_FALSE;
    toneMapDepthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineRenderingCreateInfo toneMapRenderingCreateInfo{};
    toneMapRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    toneMapRenderingCreateInfo.colorAttachmentCount = 1;
    toneMapRenderingCreateInfo.pColorAttachmentFormats = &m_swapchainFormat;
    toneMapRenderingCreateInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo toneMapPipelineCreateInfo{};
    toneMapPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    toneMapPipelineCreateInfo.pNext = &toneMapRenderingCreateInfo;
    toneMapPipelineCreateInfo.stageCount = static_cast<uint32_t>(toneMapShaderStages.size());
    toneMapPipelineCreateInfo.pStages = toneMapShaderStages.data();
    toneMapPipelineCreateInfo.pVertexInputState = &toneMapVertexInputInfo;
    toneMapPipelineCreateInfo.pInputAssemblyState = &toneMapInputAssembly;
    toneMapPipelineCreateInfo.pViewportState = &viewportState;
    toneMapPipelineCreateInfo.pRasterizationState = &toneMapRasterizer;
    toneMapPipelineCreateInfo.pMultisampleState = &toneMapMultisampling;
    toneMapPipelineCreateInfo.pDepthStencilState = &toneMapDepthStencil;
    toneMapPipelineCreateInfo.pColorBlendState = &colorBlending;
    toneMapPipelineCreateInfo.pDynamicState = &dynamicState;
    toneMapPipelineCreateInfo.layout = m_pipelineLayout;
    toneMapPipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    toneMapPipelineCreateInfo.subpass = 0;

    VkPipeline toneMapPipeline = VK_NULL_HANDLE;
    const VkResult toneMapPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &toneMapPipelineCreateInfo,
        nullptr,
        &toneMapPipeline
    );

    if (toneMapPipelineResult != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(toneMap)", toneMapPipelineResult);
        return false;
    }

    if (!createShaderModuleFromFileOrFallback(
            m_device,
            kShadowVertexShaderPathCandidates,
            nullptr,
            0,
            "shadow_depth.vert",
            shadowVertShaderModule
        )) {
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        return false;
    }
    if (!createShaderModuleFromFileOrFallback(
            m_device,
            kShadowFragmentShaderPathCandidates,
            nullptr,
            0,
            "shadow_depth.frag",
            shadowFragShaderModule
        )) {
        vkDestroyShaderModule(m_device, shadowVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
        vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo shadowVertexShaderStage{};
    shadowVertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowVertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    shadowVertexShaderStage.module = shadowVertShaderModule;
    shadowVertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo shadowFragmentShaderStage{};
    shadowFragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shadowFragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shadowFragmentShaderStage.module = shadowFragShaderModule;
    shadowFragmentShaderStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> shadowShaderStages = {
        shadowVertexShaderStage,
        shadowFragmentShaderStage
    };

    VkPipelineMultisampleStateCreateInfo shadowMultisampling{};
    shadowMultisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    shadowMultisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineRasterizationStateCreateInfo shadowRasterizer = rasterizer;
    shadowRasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    shadowRasterizer.depthBiasEnable = VK_TRUE;

    VkPipelineDepthStencilStateCreateInfo shadowDepthStencil = depthStencil;
    shadowDepthStencil.depthTestEnable = VK_TRUE;
    shadowDepthStencil.depthWriteEnable = VK_TRUE;
    shadowDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    std::array<VkDynamicState, 3> shadowDynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS
    };
    VkPipelineDynamicStateCreateInfo shadowDynamicState{};
    shadowDynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    shadowDynamicState.dynamicStateCount = static_cast<uint32_t>(shadowDynamicStates.size());
    shadowDynamicState.pDynamicStates = shadowDynamicStates.data();

    VkPipelineColorBlendStateCreateInfo shadowColorBlending{};
    shadowColorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    shadowColorBlending.attachmentCount = 0;
    shadowColorBlending.pAttachments = nullptr;

    VkPipelineRenderingCreateInfo shadowRenderingCreateInfo{};
    shadowRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    shadowRenderingCreateInfo.colorAttachmentCount = 0;
    shadowRenderingCreateInfo.pColorAttachmentFormats = nullptr;
    shadowRenderingCreateInfo.depthAttachmentFormat = m_shadowDepthFormat;

    VkGraphicsPipelineCreateInfo shadowPipelineCreateInfo{};
    shadowPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    shadowPipelineCreateInfo.pNext = &shadowRenderingCreateInfo;
    shadowPipelineCreateInfo.stageCount = static_cast<uint32_t>(shadowShaderStages.size());
    shadowPipelineCreateInfo.pStages = shadowShaderStages.data();
    shadowPipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    shadowPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    shadowPipelineCreateInfo.pViewportState = &viewportState;
    shadowPipelineCreateInfo.pRasterizationState = &shadowRasterizer;
    shadowPipelineCreateInfo.pMultisampleState = &shadowMultisampling;
    shadowPipelineCreateInfo.pDepthStencilState = &shadowDepthStencil;
    shadowPipelineCreateInfo.pColorBlendState = &shadowColorBlending;
    shadowPipelineCreateInfo.pDynamicState = &shadowDynamicState;
    shadowPipelineCreateInfo.layout = m_pipelineLayout;
    shadowPipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    shadowPipelineCreateInfo.subpass = 0;

    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    const VkResult shadowPipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &shadowPipelineCreateInfo,
        nullptr,
        &shadowPipeline
    );

    vkDestroyShaderModule(m_device, shadowFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, shadowVertShaderModule, nullptr);
    vkDestroyShaderModule(m_device, toneMapFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, toneMapVertShaderModule, nullptr);
    vkDestroyShaderModule(m_device, skyboxFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, skyboxVertShaderModule, nullptr);
    vkDestroyShaderModule(m_device, worldFragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, worldVertShaderModule, nullptr);

    if (shadowPipelineResult != VK_SUCCESS) {
        vkDestroyPipeline(m_device, worldPipeline, nullptr);
        vkDestroyPipeline(m_device, previewAddPipeline, nullptr);
        vkDestroyPipeline(m_device, previewRemovePipeline, nullptr);
        vkDestroyPipeline(m_device, skyboxPipeline, nullptr);
        vkDestroyPipeline(m_device, toneMapPipeline, nullptr);
        logVkFailure("vkCreateGraphicsPipelines(shadow)", shadowPipelineResult);
        return false;
    }

    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
    }
    if (m_skyboxPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_skyboxPipeline, nullptr);
    }
    if (m_shadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_shadowPipeline, nullptr);
    }
    if (m_tonemapPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr);
    }
    if (m_previewAddPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_previewAddPipeline, nullptr);
    }
    if (m_previewRemovePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_previewRemovePipeline, nullptr);
    }
    m_pipeline = worldPipeline;
    m_skyboxPipeline = skyboxPipeline;
    m_shadowPipeline = shadowPipeline;
    m_tonemapPipeline = toneMapPipeline;
    m_previewAddPipeline = previewAddPipeline;
    m_previewRemovePipeline = previewRemovePipeline;
    std::cerr << "[render] graphics pipelines ready (shadow + hdr scene + tonemap + preview="
              << (m_supportsWireframePreview ? "wireframe" : "ghost")
              << ")\n";
    return true;
}

bool Renderer::createChunkBuffers(const world::ChunkGrid& chunkGrid) {
    // Build one chunk mesh and upload through the transfer queue.
    // Future chunk streaming can replace this with region-level mesh patches.
    const world::ChunkMeshData mesh = world::buildSingleChunkMesh(chunkGrid);
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        std::cerr << "[render] chunk mesher produced no geometry\n";
        return false;
    }

    collectCompletedBufferReleases();

    if (m_transferCommandBufferInFlightValue > 0 && !waitForTimelineValue(m_transferCommandBufferInFlightValue)) {
        std::cerr << "[render] failed waiting for prior transfer upload\n";
        return false;
    }
    m_transferCommandBufferInFlightValue = 0;
    collectCompletedBufferReleases();

    const BufferHandle oldVertexHandle = m_vertexBufferHandle;
    const BufferHandle oldIndexHandle = m_indexBufferHandle;
    const uint64_t previousChunkReadyTimelineValue = m_currentChunkReadyTimelineValue;

    const VkDeviceSize vertexBufferSize = static_cast<VkDeviceSize>(mesh.vertices.size() * sizeof(world::PackedVoxelVertex));
    const VkDeviceSize indexBufferSize = static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(std::uint32_t));
    std::array<uint32_t, 2> meshQueueFamilies = {
        m_graphicsQueueFamilyIndex,
        m_transferQueueFamilyIndex
    };
    if (meshQueueFamilies[0] == meshQueueFamilies[1]) {
        meshQueueFamilies[1] = UINT32_MAX;
    }

    BufferCreateDesc vertexCreateDesc{};
    vertexCreateDesc.size = vertexBufferSize;
    vertexCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (meshQueueFamilies[1] != UINT32_MAX) {
        vertexCreateDesc.queueFamilyIndices = meshQueueFamilies.data();
        vertexCreateDesc.queueFamilyIndexCount = 2;
    }
    const BufferHandle newVertexHandle = m_bufferAllocator.createBuffer(vertexCreateDesc);
    if (newVertexHandle == kInvalidBufferHandle) {
        std::cerr << "[render] chunk vertex buffer allocation failed\n";
        return false;
    }

    BufferCreateDesc indexCreateDesc{};
    indexCreateDesc.size = indexBufferSize;
    indexCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (meshQueueFamilies[1] != UINT32_MAX) {
        indexCreateDesc.queueFamilyIndices = meshQueueFamilies.data();
        indexCreateDesc.queueFamilyIndexCount = 2;
    }
    const BufferHandle newIndexHandle = m_bufferAllocator.createBuffer(indexCreateDesc);
    if (newIndexHandle == kInvalidBufferHandle) {
        std::cerr << "[render] chunk index buffer allocation failed\n";
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    BufferCreateDesc vertexStagingCreateDesc{};
    vertexStagingCreateDesc.size = vertexBufferSize;
    vertexStagingCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vertexStagingCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vertexStagingCreateDesc.initialData = mesh.vertices.data();
    const BufferHandle vertexStagingHandle = m_bufferAllocator.createBuffer(vertexStagingCreateDesc);
    if (vertexStagingHandle == kInvalidBufferHandle) {
        std::cerr << "[render] chunk vertex staging buffer allocation failed\n";
        m_bufferAllocator.destroyBuffer(newIndexHandle);
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    BufferCreateDesc indexStagingCreateDesc{};
    indexStagingCreateDesc.size = indexBufferSize;
    indexStagingCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    indexStagingCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    indexStagingCreateDesc.initialData = mesh.indices.data();
    const BufferHandle indexStagingHandle = m_bufferAllocator.createBuffer(indexStagingCreateDesc);
    if (indexStagingHandle == kInvalidBufferHandle) {
        std::cerr << "[render] chunk index staging buffer allocation failed\n";
        m_bufferAllocator.destroyBuffer(vertexStagingHandle);
        m_bufferAllocator.destroyBuffer(newIndexHandle);
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    const VkResult resetResult = vkResetCommandPool(m_device, m_transferCommandPool, 0);
    if (resetResult != VK_SUCCESS) {
        logVkFailure("vkResetCommandPool(transfer)", resetResult);
        m_bufferAllocator.destroyBuffer(indexStagingHandle);
        m_bufferAllocator.destroyBuffer(vertexStagingHandle);
        m_bufferAllocator.destroyBuffer(newIndexHandle);
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(m_transferCommandBuffer, &beginInfo) != VK_SUCCESS) {
        std::cerr << "[render] vkBeginCommandBuffer (transfer) failed\n";
        m_bufferAllocator.destroyBuffer(indexStagingHandle);
        m_bufferAllocator.destroyBuffer(vertexStagingHandle);
        m_bufferAllocator.destroyBuffer(newIndexHandle);
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    VkBufferCopy vertexCopy{};
    vertexCopy.size = vertexBufferSize;
    vkCmdCopyBuffer(
        m_transferCommandBuffer,
        m_bufferAllocator.getBuffer(vertexStagingHandle),
        m_bufferAllocator.getBuffer(newVertexHandle),
        1,
        &vertexCopy
    );

    VkBufferCopy indexCopy{};
    indexCopy.size = indexBufferSize;
    vkCmdCopyBuffer(
        m_transferCommandBuffer,
        m_bufferAllocator.getBuffer(indexStagingHandle),
        m_bufferAllocator.getBuffer(newIndexHandle),
        1,
        &indexCopy
    );

    if (vkEndCommandBuffer(m_transferCommandBuffer) != VK_SUCCESS) {
        std::cerr << "[render] vkEndCommandBuffer (transfer) failed\n";
        m_bufferAllocator.destroyBuffer(indexStagingHandle);
        m_bufferAllocator.destroyBuffer(vertexStagingHandle);
        m_bufferAllocator.destroyBuffer(newIndexHandle);
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    const uint64_t transferSignalValue = m_nextTimelineValue++;
    VkSemaphore timelineSemaphore = m_renderTimelineSemaphore;
    VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
    timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineSubmitInfo.signalSemaphoreValueCount = 1;
    timelineSubmitInfo.pSignalSemaphoreValues = &transferSignalValue;

    VkSubmitInfo transferSubmitInfo{};
    transferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    transferSubmitInfo.pNext = &timelineSubmitInfo;
    transferSubmitInfo.commandBufferCount = 1;
    transferSubmitInfo.pCommandBuffers = &m_transferCommandBuffer;
    transferSubmitInfo.signalSemaphoreCount = 1;
    transferSubmitInfo.pSignalSemaphores = &timelineSemaphore;

    const VkResult submitResult = vkQueueSubmit(m_transferQueue, 1, &transferSubmitInfo, VK_NULL_HANDLE);
    if (submitResult != VK_SUCCESS) {
        logVkFailure("vkQueueSubmit(transfer)", submitResult);
        m_bufferAllocator.destroyBuffer(indexStagingHandle);
        m_bufferAllocator.destroyBuffer(vertexStagingHandle);
        m_bufferAllocator.destroyBuffer(newIndexHandle);
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    m_vertexBufferHandle = newVertexHandle;
    m_indexBufferHandle = newIndexHandle;
    m_indexCount = static_cast<uint32_t>(mesh.indices.size());
    m_currentChunkReadyTimelineValue = transferSignalValue;
    m_pendingTransferTimelineValue = transferSignalValue;
    m_transferCommandBufferInFlightValue = transferSignalValue;

    scheduleBufferRelease(vertexStagingHandle, transferSignalValue);
    scheduleBufferRelease(indexStagingHandle, transferSignalValue);

    const uint64_t oldChunkReleaseValue = std::max(m_lastGraphicsTimelineValue, previousChunkReadyTimelineValue);
    scheduleBufferRelease(oldVertexHandle, oldChunkReleaseValue);
    scheduleBufferRelease(oldIndexHandle, oldChunkReleaseValue);

    std::cerr << "[render] chunk upload queued (vertices=" << mesh.vertices.size()
              << ", indices=" << mesh.indices.size()
              << ", timelineValue=" << transferSignalValue << ")\n";
    return true;
}

bool Renderer::createFrameResources() {
    for (FrameResources& frame : m_frames) {
        VkCommandPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;

        if (vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &frame.commandPool) != VK_SUCCESS) {
            std::cerr << "[render] failed creating command pool for frame resource\n";
            return false;
        }

        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &frame.imageAvailable) != VK_SUCCESS) {
            std::cerr << "[render] failed creating imageAvailable semaphore\n";
            return false;
        }
    }

    std::cerr << "[render] frame resources ready (" << kMaxFramesInFlight << " frames in flight)\n";
    return true;
}

bool Renderer::waitForTimelineValue(uint64_t value) const {
    if (value == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return true;
    }

    VkSemaphore waitSemaphore = m_renderTimelineSemaphore;
    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &waitSemaphore;
    waitInfo.pValues = &value;
    const VkResult waitResult = vkWaitSemaphores(m_device, &waitInfo, std::numeric_limits<uint64_t>::max());
    if (waitResult != VK_SUCCESS) {
        logVkFailure("vkWaitSemaphores(timeline)", waitResult);
        return false;
    }
    return true;
}

void Renderer::scheduleBufferRelease(BufferHandle handle, uint64_t timelineValue) {
    if (handle == kInvalidBufferHandle) {
        return;
    }
    if (timelineValue == 0 || m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        m_bufferAllocator.destroyBuffer(handle);
        return;
    }
    m_deferredBufferReleases.push_back({handle, timelineValue});
}

void Renderer::collectCompletedBufferReleases() {
    if (m_renderTimelineSemaphore == VK_NULL_HANDLE) {
        return;
    }

    uint64_t completedValue = 0;
    const VkResult counterResult = vkGetSemaphoreCounterValue(m_device, m_renderTimelineSemaphore, &completedValue);
    if (counterResult != VK_SUCCESS) {
        logVkFailure("vkGetSemaphoreCounterValue", counterResult);
        return;
    }

    for (const DeferredBufferRelease& release : m_deferredBufferReleases) {
        if (release.timelineValue <= completedValue) {
            m_bufferAllocator.destroyBuffer(release.handle);
        }
    }
    std::erase_if(
        m_deferredBufferReleases,
        [completedValue](const DeferredBufferRelease& release) {
            return release.timelineValue <= completedValue;
        }
    );

    if (m_pendingTransferTimelineValue > 0 && m_pendingTransferTimelineValue <= completedValue) {
        m_pendingTransferTimelineValue = 0;
    }
    if (m_transferCommandBufferInFlightValue > 0 && m_transferCommandBufferInFlightValue <= completedValue) {
        m_transferCommandBufferInFlightValue = 0;
    }
}

void Renderer::renderFrame(
    const world::ChunkGrid& chunkGrid,
    const sim::Simulation& simulation,
    const CameraPose& camera,
    const VoxelPreview& preview
) {
    (void)chunkGrid;
    (void)simulation;

    if (m_device == VK_NULL_HANDLE || m_swapchain == VK_NULL_HANDLE) {
        return;
    }

    collectCompletedBufferReleases();
    m_uploadRing.beginFrame(m_currentFrame);

    FrameResources& frame = m_frames[m_currentFrame];
    if (!waitForTimelineValue(m_frameTimelineValues[m_currentFrame])) {
        return;
    }

    uint32_t imageIndex = 0;
    const VkResult acquireResult = vkAcquireNextImageKHR(
        m_device,
        m_swapchain,
        std::numeric_limits<uint64_t>::max(),
        frame.imageAvailable,
        VK_NULL_HANDLE,
        &imageIndex
    );

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        std::cerr << "[render] swapchain out of date during acquire, recreating\n";
        recreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        logVkFailure("vkAcquireNextImageKHR", acquireResult);
        return;
    }

    if (!waitForTimelineValue(m_swapchainImageTimelineValues[imageIndex])) {
        return;
    }
    const VkSemaphore renderFinishedSemaphore = m_renderFinishedSemaphores[imageIndex];

    vkResetCommandPool(m_device, frame.commandPool, 0);

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = frame.commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &allocateInfo, &commandBuffer) != VK_SUCCESS) {
        std::cerr << "[render] vkAllocateCommandBuffers failed\n";
        return;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        std::cerr << "[render] vkBeginCommandBuffer failed\n";
        return;
    }

    const float aspectRatio = static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);
    const float yawRadians = math::radians(camera.yawDegrees);
    const float pitchRadians = math::radians(camera.pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    const math::Vector3 eye{camera.x, camera.y, camera.z};
    const math::Vector3 forward{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    };
    const math::Matrix4 view = lookAt(eye, eye + forward, math::Vector3{0.0f, 1.0f, 0.0f});
    const math::Matrix4 projection = perspectiveVulkan(math::radians(camera.fovDegrees), aspectRatio, 0.1f, 500.0f);
    const math::Matrix4 mvp = projection * view;
    const math::Matrix4 mvpColumnMajor = transpose(mvp);
    const math::Matrix4 viewColumnMajor = transpose(view);
    const math::Matrix4 projectionColumnMajor = transpose(projection);

    const math::Vector3 sunDirection = math::normalize(math::Vector3{-0.55f, -1.0f, -0.35f});
    const math::Vector3 sceneCenter{16.0f, 6.0f, 16.0f};
    const math::Vector3 lightPosition = sceneCenter - (sunDirection * 110.0f);
    const float sunUpDot = std::abs(math::dot(sunDirection, math::Vector3{0.0f, 1.0f, 0.0f}));
    const math::Vector3 lightUp = (sunUpDot > 0.95f) ? math::Vector3{0.0f, 0.0f, 1.0f} : math::Vector3{0.0f, 1.0f, 0.0f};
    const math::Matrix4 lightView = lookAt(lightPosition, sceneCenter, lightUp);
    const math::Matrix4 lightProjection = orthographicVulkan(-96.0f, 96.0f, -96.0f, 96.0f, 1.0f, 260.0f);
    const math::Matrix4 lightViewProj = lightProjection * lightView;
    const math::Matrix4 lightViewProjColumnMajor = transpose(lightViewProj);

    const std::optional<RingBufferSlice> mvpSliceOpt =
        m_uploadRing.allocate(sizeof(CameraUniform), m_uniformBufferAlignment);
    if (!mvpSliceOpt.has_value() || mvpSliceOpt->mapped == nullptr) {
        std::cerr << "[render] failed to allocate MVP uniform slice\n";
        return;
    }

    CameraUniform mvpUniform{};
    std::memcpy(mvpUniform.mvp, mvpColumnMajor.m, sizeof(mvpUniform.mvp));
    std::memcpy(mvpUniform.view, viewColumnMajor.m, sizeof(mvpUniform.view));
    std::memcpy(mvpUniform.proj, projectionColumnMajor.m, sizeof(mvpUniform.proj));
    std::memcpy(mvpUniform.lightViewProj, lightViewProjColumnMajor.m, sizeof(mvpUniform.lightViewProj));
    mvpUniform.sunDirectionIntensity[0] = sunDirection.x;
    mvpUniform.sunDirectionIntensity[1] = sunDirection.y;
    mvpUniform.sunDirectionIntensity[2] = sunDirection.z;
    mvpUniform.sunDirectionIntensity[3] = 3.0f;
    mvpUniform.sunColorShadow[0] = 1.0f;
    mvpUniform.sunColorShadow[1] = 0.95f;
    mvpUniform.sunColorShadow[2] = 0.86f;
    mvpUniform.sunColorShadow[3] = 1.0f;
    std::memcpy(mvpSliceOpt->mapped, &mvpUniform, sizeof(mvpUniform));

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_bufferAllocator.getBuffer(mvpSliceOpt->buffer);
    bufferInfo.offset = mvpSliceOpt->offset;
    bufferInfo.range = mvpSliceOpt->size;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

    VkDescriptorImageInfo irradianceImageInfo{};
    irradianceImageInfo.sampler = m_irradianceTexture.sampler;
    irradianceImageInfo.imageView = m_irradianceTexture.view;
    irradianceImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo skyboxImageInfo{};
    skyboxImageInfo.sampler = m_skyboxTexture.sampler;
    skyboxImageInfo.imageView = m_skyboxTexture.view;
    skyboxImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo hdrSceneImageInfo{};
    hdrSceneImageInfo.sampler = m_hdrResolveSampler;
    hdrSceneImageInfo.imageView = m_hdrResolveImageViews[imageIndex];
    hdrSceneImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo shadowMapImageInfo{};
    shadowMapImageInfo.sampler = m_shadowDepthSampler;
    shadowMapImageInfo.imageView = m_shadowDepthImageView;
    shadowMapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    std::array<VkWriteDescriptorSet, 5> writes{};
    writes[0] = write;
    writes[0].dstSet = m_descriptorSets[m_currentFrame];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &bufferInfo;

    writes[1] = write;
    writes[1].dstSet = m_descriptorSets[m_currentFrame];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &irradianceImageInfo;

    writes[2] = write;
    writes[2].dstSet = m_descriptorSets[m_currentFrame];
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &skyboxImageInfo;

    writes[3] = write;
    writes[3].dstSet = m_descriptorSets[m_currentFrame];
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &hdrSceneImageInfo;

    writes[4] = write;
    writes[4].dstSet = m_descriptorSets[m_currentFrame];
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &shadowMapImageInfo;

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(m_vertexBufferHandle);
    if (vertexBuffer == VK_NULL_HANDLE) {
        std::cerr << "[render] missing vertex buffer for draw\n";
        return;
    }
    const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(m_indexBufferHandle);
    if (indexBuffer == VK_NULL_HANDLE) {
        std::cerr << "[render] missing index buffer for draw\n";
        return;
    }
    const VkDeviceSize vertexBufferOffset = 0;

    const bool shadowInitialized = m_shadowDepthInitialized;
    transitionImageLayout(
        commandBuffer,
        m_shadowDepthImage,
        shadowInitialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        shadowInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        shadowInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT
    );

    constexpr uint32_t kShadowMapSize = 2048;
    VkClearValue shadowDepthClearValue{};
    shadowDepthClearValue.depthStencil.depth = 1.0f;
    shadowDepthClearValue.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo shadowDepthAttachment{};
    shadowDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    shadowDepthAttachment.imageView = m_shadowDepthImageView;
    shadowDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    shadowDepthAttachment.clearValue = shadowDepthClearValue;

    VkRenderingInfo shadowRenderingInfo{};
    shadowRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    shadowRenderingInfo.renderArea.offset = {0, 0};
    shadowRenderingInfo.renderArea.extent = {kShadowMapSize, kShadowMapSize};
    shadowRenderingInfo.layerCount = 1;
    shadowRenderingInfo.colorAttachmentCount = 0;
    shadowRenderingInfo.pDepthAttachment = &shadowDepthAttachment;

    vkCmdBeginRendering(commandBuffer, &shadowRenderingInfo);

    VkViewport shadowViewport{};
    shadowViewport.x = 0.0f;
    shadowViewport.y = 0.0f;
    shadowViewport.width = static_cast<float>(kShadowMapSize);
    shadowViewport.height = static_cast<float>(kShadowMapSize);
    shadowViewport.minDepth = 0.0f;
    shadowViewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);

    VkRect2D shadowScissor{};
    shadowScissor.offset = {0, 0};
    shadowScissor.extent = {kShadowMapSize, kShadowMapSize};
    vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);

    if (m_shadowPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            1,
            &m_descriptorSets[m_currentFrame],
            0,
            nullptr
        );
        vkCmdSetDepthBias(commandBuffer, 0.75f, 0.0f, 1.0f);
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexBufferOffset);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        constexpr float kChunkSize = static_cast<float>(world::Chunk::kSizeX);
        for (int chunkZ = -1; chunkZ <= 1; ++chunkZ) {
            for (int chunkX = -1; chunkX <= 1; ++chunkX) {
                ChunkPushConstants chunkPushConstants{};
                chunkPushConstants.chunkOffset[0] = static_cast<float>(chunkX) * kChunkSize;
                chunkPushConstants.chunkOffset[1] = 0.0f;
                chunkPushConstants.chunkOffset[2] = static_cast<float>(chunkZ) * kChunkSize;
                chunkPushConstants.chunkOffset[3] = 0.0f;
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &chunkPushConstants
                );
                vkCmdDrawIndexed(commandBuffer, m_indexCount, 1, 0, 0, 0);
            }
        }
    }

    vkCmdEndRendering(commandBuffer);

    transitionImageLayout(
        commandBuffer,
        m_shadowDepthImage,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT
    );

    if (!m_msaaColorImageInitialized[imageIndex]) {
        transitionImageLayout(
            commandBuffer,
            m_msaaColorImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }
    const bool hdrResolveInitialized = m_hdrResolveImageInitialized[imageIndex];
    transitionImageLayout(
        commandBuffer,
        m_hdrResolveImages[imageIndex],
        hdrResolveInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        hdrResolveInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        hdrResolveInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    transitionImageLayout(
        commandBuffer,
        m_depthImages[imageIndex],
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT
    );

    VkClearValue clearValue{};
    clearValue.color.float32[0] = 0.06f;
    clearValue.color.float32[1] = 0.08f;
    clearValue.color.float32[2] = 0.12f;
    clearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = m_msaaColorImageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.clearValue = clearValue;
    colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    colorAttachment.resolveImageView = m_hdrResolveImageViews[imageIndex];
    colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkClearValue depthClearValue{};
    depthClearValue.depthStencil.depth = 1.0f;
    depthClearValue.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = m_depthImageViews[imageIndex];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue = depthClearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = m_swapchainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchainExtent.width);
    viewport.height = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    if (m_skyboxPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyboxPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            1,
            &m_descriptorSets[m_currentFrame],
            0,
            nullptr
        );
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout,
        0,
        1,
        &m_descriptorSets[m_currentFrame],
        0,
        nullptr
    );
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexBufferOffset);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    constexpr float kChunkSize = static_cast<float>(world::Chunk::kSizeX);
    for (int chunkZ = -1; chunkZ <= 1; ++chunkZ) {
        for (int chunkX = -1; chunkX <= 1; ++chunkX) {
            ChunkPushConstants chunkPushConstants{};
            chunkPushConstants.chunkOffset[0] = static_cast<float>(chunkX) * kChunkSize;
            chunkPushConstants.chunkOffset[1] = 0.0f;
            chunkPushConstants.chunkOffset[2] = static_cast<float>(chunkZ) * kChunkSize;
            chunkPushConstants.chunkOffset[3] = 0.0f;
            vkCmdPushConstants(
                commandBuffer,
                m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(ChunkPushConstants),
                &chunkPushConstants
            );
            vkCmdDrawIndexed(commandBuffer, m_indexCount, 1, 0, 0, 0);
        }
    }

    const VkPipeline activePreviewPipeline =
        (preview.mode == VoxelPreview::Mode::Remove) ? m_previewRemovePipeline : m_previewAddPipeline;
    if (preview.visible && activePreviewPipeline != VK_NULL_HANDLE) {
        const uint32_t previewX = static_cast<uint32_t>(std::clamp(preview.x, 0, 31));
        const uint32_t previewY = static_cast<uint32_t>(std::clamp(preview.y, 0, 31));
        const uint32_t previewZ = static_cast<uint32_t>(std::clamp(preview.z, 0, 31));
        const uint32_t previewMaterial = (preview.mode == VoxelPreview::Mode::Add) ? 1u : 0u;
        const uint32_t previewAo = (preview.mode == VoxelPreview::Mode::Add) ? 3u : 0u;
        const world::ChunkMeshData previewMesh = buildSingleVoxelPreviewMesh(
            previewX,
            previewY,
            previewZ,
            previewAo,
            previewMaterial
        );

        const VkDeviceSize previewVertexBytes =
            static_cast<VkDeviceSize>(previewMesh.vertices.size() * sizeof(world::PackedVoxelVertex));
        void* mappedPreviewVertices = m_bufferAllocator.mapBuffer(m_previewVertexBufferHandle, 0, previewVertexBytes);
        if (mappedPreviewVertices != nullptr) {
            std::memcpy(mappedPreviewVertices, previewMesh.vertices.data(), static_cast<size_t>(previewVertexBytes));
            m_bufferAllocator.unmapBuffer(m_previewVertexBufferHandle);

            const VkBuffer previewVertexBuffer = m_bufferAllocator.getBuffer(m_previewVertexBufferHandle);
            const VkBuffer previewIndexBuffer = m_bufferAllocator.getBuffer(m_previewIndexBufferHandle);
            if (previewVertexBuffer != VK_NULL_HANDLE && previewIndexBuffer != VK_NULL_HANDLE) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, activePreviewPipeline);
                vkCmdBindDescriptorSets(
                    commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipelineLayout,
                    0,
                    1,
                    &m_descriptorSets[m_currentFrame],
                    0,
                    nullptr
                );
                ChunkPushConstants previewChunkPushConstants{};
                previewChunkPushConstants.chunkOffset[0] = 0.0f;
                previewChunkPushConstants.chunkOffset[1] = 0.0f;
                previewChunkPushConstants.chunkOffset[2] = 0.0f;
                previewChunkPushConstants.chunkOffset[3] = 0.0f;
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &previewChunkPushConstants
                );
                vkCmdSetDepthBias(commandBuffer, -1.0f, 0.0f, -1.0f);
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, &previewVertexBuffer, &vertexBufferOffset);
                vkCmdBindIndexBuffer(commandBuffer, previewIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(commandBuffer, m_previewIndexCount, 1, 0, 0, 0);
            }
        }
    }

    vkCmdEndRendering(commandBuffer);

    transitionImageLayout(
        commandBuffer,
        m_hdrResolveImages[imageIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    transitionImageLayout(
        commandBuffer,
        m_swapchainImages[imageIndex],
        m_swapchainImageInitialized[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkRenderingAttachmentInfo toneMapColorAttachment{};
    toneMapColorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    toneMapColorAttachment.imageView = m_swapchainImageViews[imageIndex];
    toneMapColorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toneMapColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    toneMapColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo toneMapRenderingInfo{};
    toneMapRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    toneMapRenderingInfo.renderArea.offset = {0, 0};
    toneMapRenderingInfo.renderArea.extent = m_swapchainExtent;
    toneMapRenderingInfo.layerCount = 1;
    toneMapRenderingInfo.colorAttachmentCount = 1;
    toneMapRenderingInfo.pColorAttachments = &toneMapColorAttachment;

    vkCmdBeginRendering(commandBuffer, &toneMapRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    if (m_tonemapPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            1,
            &m_descriptorSets[m_currentFrame],
            0,
            nullptr
        );
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }

    vkCmdEndRendering(commandBuffer);

    transitionImageLayout(
        commandBuffer,
        m_swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        std::cerr << "[render] vkEndCommandBuffer failed\n";
        return;
    }

    std::array<VkSemaphore, 2> waitSemaphores{};
    std::array<VkPipelineStageFlags, 2> waitStages{};
    std::array<uint64_t, 2> waitSemaphoreValues{};
    uint32_t waitSemaphoreCount = 0;

    waitSemaphores[waitSemaphoreCount] = frame.imageAvailable;
    waitStages[waitSemaphoreCount] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    waitSemaphoreValues[waitSemaphoreCount] = 0;
    ++waitSemaphoreCount;

    if (m_pendingTransferTimelineValue > 0) {
        waitSemaphores[waitSemaphoreCount] = m_renderTimelineSemaphore;
        waitStages[waitSemaphoreCount] = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
        waitSemaphoreValues[waitSemaphoreCount] = m_pendingTransferTimelineValue;
        ++waitSemaphoreCount;
    }

    const uint64_t signalTimelineValue = m_nextTimelineValue++;
    std::array<VkSemaphore, 2> signalSemaphores = {
        renderFinishedSemaphore,
        m_renderTimelineSemaphore
    };
    std::array<uint64_t, 2> signalSemaphoreValues = {
        0,
        signalTimelineValue
    };
    VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
    timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineSubmitInfo.waitSemaphoreValueCount = waitSemaphoreCount;
    timelineSubmitInfo.pWaitSemaphoreValues = waitSemaphoreValues.data();
    timelineSubmitInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalSemaphoreValues.size());
    timelineSubmitInfo.pSignalSemaphoreValues = signalSemaphoreValues.data();

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineSubmitInfo;
    submitInfo.waitSemaphoreCount = waitSemaphoreCount;
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        std::cerr << "[render] vkQueueSubmit failed\n";
        return;
    }
    m_frameTimelineValues[m_currentFrame] = signalTimelineValue;
    m_swapchainImageTimelineValues[imageIndex] = signalTimelineValue;
    m_lastGraphicsTimelineValue = signalTimelineValue;

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchain;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(m_graphicsQueue, &presentInfo);
    m_shadowDepthInitialized = true;
    m_swapchainImageInitialized[imageIndex] = true;
    m_msaaColorImageInitialized[imageIndex] = true;
    m_hdrResolveImageInitialized[imageIndex] = true;

    if (
        acquireResult == VK_SUBOPTIMAL_KHR ||
        presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR
    ) {
        std::cerr << "[render] swapchain needs recreate after present\n";
        recreateSwapchain();
    } else if (presentResult != VK_SUCCESS) {
        logVkFailure("vkQueuePresentKHR", presentResult);
    }

    m_currentFrame = (m_currentFrame + 1) % kMaxFramesInFlight;
}

bool Renderer::recreateSwapchain() {
    std::cerr << "[render] recreateSwapchain begin\n";
    int width = 0;
    int height = 0;
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(m_window, &width, &height);
        if (glfwWindowShouldClose(m_window) == GLFW_TRUE) {
            return false;
        }
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(m_device);

    destroyPipeline();
    destroySwapchain();

    if (!createSwapchain()) {
        std::cerr << "[render] recreateSwapchain failed: createSwapchain\n";
        return false;
    }
    if (!createGraphicsPipeline()) {
        std::cerr << "[render] recreateSwapchain failed: createGraphicsPipeline\n";
        return false;
    }
    std::cerr << "[render] recreateSwapchain complete\n";
    return true;
}

void Renderer::destroyHdrResolveTargets() {
    if (m_hdrResolveSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_hdrResolveSampler, nullptr);
        m_hdrResolveSampler = VK_NULL_HANDLE;
    }

    for (VkImageView imageView : m_hdrResolveImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_hdrResolveImageViews.clear();

    for (VkImage image : m_hdrResolveImages) {
        if (image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, image, nullptr);
        }
    }
    m_hdrResolveImages.clear();

    for (VkDeviceMemory memory : m_hdrResolveImageMemories) {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, memory, nullptr);
        }
    }
    m_hdrResolveImageMemories.clear();
    m_hdrResolveImageInitialized.clear();
}

void Renderer::destroyMsaaColorTargets() {
    for (VkImageView imageView : m_msaaColorImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_msaaColorImageViews.clear();

    for (VkImage image : m_msaaColorImages) {
        if (image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, image, nullptr);
        }
    }
    m_msaaColorImages.clear();

    for (VkDeviceMemory memory : m_msaaColorImageMemories) {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, memory, nullptr);
        }
    }
    m_msaaColorImageMemories.clear();
    m_msaaColorImageInitialized.clear();
}

void Renderer::destroyDepthTargets() {
    for (VkImageView imageView : m_depthImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_depthImageViews.clear();

    for (VkImage image : m_depthImages) {
        if (image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, image, nullptr);
        }
    }
    m_depthImages.clear();

    for (VkDeviceMemory memory : m_depthImageMemories) {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, memory, nullptr);
        }
    }
    m_depthImageMemories.clear();
}

void Renderer::destroySwapchain() {
    destroyHdrResolveTargets();
    destroyMsaaColorTargets();
    destroyDepthTargets();

    for (VkSemaphore semaphore : m_renderFinishedSemaphores) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, semaphore, nullptr);
        }
    }
    m_renderFinishedSemaphores.clear();

    for (VkImageView imageView : m_swapchainImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_swapchainImageViews.clear();
    m_swapchainImages.clear();
    m_swapchainImageInitialized.clear();
    m_swapchainImageTimelineValues.clear();

    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

void Renderer::destroyFrameResources() {
    for (FrameResources& frame : m_frames) {
        if (frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, frame.imageAvailable, nullptr);
            frame.imageAvailable = VK_NULL_HANDLE;
        }
        if (frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, frame.commandPool, nullptr);
            frame.commandPool = VK_NULL_HANDLE;
        }
    }
}

void Renderer::destroyTransferResources() {
    m_transferCommandBuffer = VK_NULL_HANDLE;
    if (m_transferCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_transferCommandPool, nullptr);
        m_transferCommandPool = VK_NULL_HANDLE;
    }
}

void Renderer::destroyPreviewBuffers() {
    if (m_previewIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_previewIndexBufferHandle);
        m_previewIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_previewVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_previewVertexBufferHandle);
        m_previewVertexBufferHandle = kInvalidBufferHandle;
    }
    m_previewIndexCount = 0;
}

void Renderer::destroyEnvironmentResources() {
    auto destroyCubemap = [&](CubemapTexture& texture) {
        if (texture.sampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, texture.sampler, nullptr);
            texture.sampler = VK_NULL_HANDLE;
        }
        if (texture.view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, texture.view, nullptr);
            texture.view = VK_NULL_HANDLE;
        }
        if (texture.image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, texture.image, nullptr);
            texture.image = VK_NULL_HANDLE;
        }
        if (texture.memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, texture.memory, nullptr);
            texture.memory = VK_NULL_HANDLE;
        }
        texture.mipLevels = 1;
    };

    destroyCubemap(m_skyboxTexture);
    destroyCubemap(m_irradianceTexture);
}

void Renderer::destroyShadowResources() {
    if (m_shadowDepthSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_shadowDepthSampler, nullptr);
        m_shadowDepthSampler = VK_NULL_HANDLE;
    }
    if (m_shadowDepthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_shadowDepthImageView, nullptr);
        m_shadowDepthImageView = VK_NULL_HANDLE;
    }
    if (m_shadowDepthImage != VK_NULL_HANDLE) {
        vkDestroyImage(m_device, m_shadowDepthImage, nullptr);
        m_shadowDepthImage = VK_NULL_HANDLE;
    }
    if (m_shadowDepthMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_shadowDepthMemory, nullptr);
        m_shadowDepthMemory = VK_NULL_HANDLE;
    }
    m_shadowDepthInitialized = false;
}

void Renderer::destroyChunkBuffers() {
    if (m_indexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_indexBufferHandle);
        m_indexBufferHandle = kInvalidBufferHandle;
    }
    if (m_vertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_vertexBufferHandle);
        m_vertexBufferHandle = kInvalidBufferHandle;
    }

    for (const DeferredBufferRelease& release : m_deferredBufferReleases) {
        if (release.handle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(release.handle);
        }
    }
    m_deferredBufferReleases.clear();

    m_indexCount = 0;
    m_pendingTransferTimelineValue = 0;
    m_currentChunkReadyTimelineValue = 0;
    m_transferCommandBufferInFlightValue = 0;
}

void Renderer::destroyPipeline() {
    if (m_tonemapPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr);
        m_tonemapPipeline = VK_NULL_HANDLE;
    }
    if (m_skyboxPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_skyboxPipeline, nullptr);
        m_skyboxPipeline = VK_NULL_HANDLE;
    }
    if (m_shadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_shadowPipeline, nullptr);
        m_shadowPipeline = VK_NULL_HANDLE;
    }
    if (m_previewRemovePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_previewRemovePipeline, nullptr);
        m_previewRemovePipeline = VK_NULL_HANDLE;
    }
    if (m_previewAddPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_previewAddPipeline, nullptr);
        m_previewAddPipeline = VK_NULL_HANDLE;
    }
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
}

void Renderer::shutdown() {
    std::cerr << "[render] shutdown begin\n";
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    if (m_device != VK_NULL_HANDLE) {
        destroyFrameResources();
        destroyTransferResources();
        if (m_renderTimelineSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, m_renderTimelineSemaphore, nullptr);
            m_renderTimelineSemaphore = VK_NULL_HANDLE;
        }
        m_uploadRing.shutdown(&m_bufferAllocator);
        destroyPreviewBuffers();
        destroyEnvironmentResources();
        destroyShadowResources();
        destroyChunkBuffers();
        destroyPipeline();
        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }
        if (m_descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
            m_descriptorSetLayout = VK_NULL_HANDLE;
        }
        m_descriptorSets.fill(VK_NULL_HANDLE);
        destroySwapchain();
        m_bufferAllocator.shutdown();

        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }

    if (m_surface != VK_NULL_HANDLE && m_instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }

    m_physicalDevice = VK_NULL_HANDLE;
    m_graphicsQueue = VK_NULL_HANDLE;
    m_transferQueue = VK_NULL_HANDLE;
    m_graphicsQueueFamilyIndex = 0;
    m_graphicsQueueIndex = 0;
    m_transferQueueFamilyIndex = 0;
    m_transferQueueIndex = 0;
    m_depthFormat = VK_FORMAT_UNDEFINED;
    m_shadowDepthFormat = VK_FORMAT_UNDEFINED;
    m_hdrColorFormat = VK_FORMAT_UNDEFINED;
    m_supportsWireframePreview = false;
    m_frameTimelineValues.fill(0);
    m_pendingTransferTimelineValue = 0;
    m_currentChunkReadyTimelineValue = 0;
    m_transferCommandBufferInFlightValue = 0;
    m_lastGraphicsTimelineValue = 0;
    m_nextTimelineValue = 1;
    m_currentFrame = 0;
    m_window = nullptr;
    std::cerr << "[render] shutdown complete\n";
}

} // namespace render
