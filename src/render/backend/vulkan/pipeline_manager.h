#pragma once

#include <vulkan/vulkan.h>

namespace voxelsprout::render {

class PipelineManager {
public:
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkPipeline pipeShadowPipeline = VK_NULL_HANDLE;
    VkPipeline grassBillboardShadowPipeline = VK_NULL_HANDLE;
    VkPipeline sdfShadowPipeline = VK_NULL_HANDLE;
    VkPipeline skyboxPipeline = VK_NULL_HANDLE;
    VkPipeline tonemapPipeline = VK_NULL_HANDLE;
    VkPipeline pipePipeline = VK_NULL_HANDLE;
    VkPipeline grassBillboardPipeline = VK_NULL_HANDLE;
    VkPipeline voxelNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline pipeNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline grassBillboardNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline sdfPrepassPipeline = VK_NULL_HANDLE;
    VkPipeline sdfMainPipeline = VK_NULL_HANDLE;
    VkPipeline magicaPipeline = VK_NULL_HANDLE;
    VkPipeline ssaoPipeline = VK_NULL_HANDLE;
    VkPipeline ssaoBlurPipeline = VK_NULL_HANDLE;
    VkPipeline previewAddPipeline = VK_NULL_HANDLE;
    VkPipeline previewRemovePipeline = VK_NULL_HANDLE;
    VkPipelineLayout voxelGiPipelineLayout = VK_NULL_HANDLE;
    VkPipeline voxelGiSurfacePipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiOccupancyPipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiSkyExposurePipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiInjectPipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiPropagatePipeline = VK_NULL_HANDLE;

    void destroyMainPipelines(VkDevice device) {
        if (ssaoBlurPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, ssaoBlurPipeline, nullptr);
            ssaoBlurPipeline = VK_NULL_HANDLE;
        }
        if (sdfMainPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, sdfMainPipeline, nullptr);
            sdfMainPipeline = VK_NULL_HANDLE;
        }
        if (sdfPrepassPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, sdfPrepassPipeline, nullptr);
            sdfPrepassPipeline = VK_NULL_HANDLE;
        }
        if (sdfShadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, sdfShadowPipeline, nullptr);
            sdfShadowPipeline = VK_NULL_HANDLE;
        }
        if (ssaoPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, ssaoPipeline, nullptr);
            ssaoPipeline = VK_NULL_HANDLE;
        }
        if (pipeNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeNormalDepthPipeline, nullptr);
            pipeNormalDepthPipeline = VK_NULL_HANDLE;
        }
        if (grassBillboardNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, grassBillboardNormalDepthPipeline, nullptr);
            grassBillboardNormalDepthPipeline = VK_NULL_HANDLE;
        }
        if (voxelNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelNormalDepthPipeline, nullptr);
            voxelNormalDepthPipeline = VK_NULL_HANDLE;
        }
        if (tonemapPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, tonemapPipeline, nullptr);
            tonemapPipeline = VK_NULL_HANDLE;
        }
        if (skyboxPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, skyboxPipeline, nullptr);
            skyboxPipeline = VK_NULL_HANDLE;
        }
        if (shadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, shadowPipeline, nullptr);
            shadowPipeline = VK_NULL_HANDLE;
        }
        if (pipeShadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeShadowPipeline, nullptr);
            pipeShadowPipeline = VK_NULL_HANDLE;
        }
        if (grassBillboardShadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, grassBillboardShadowPipeline, nullptr);
            grassBillboardShadowPipeline = VK_NULL_HANDLE;
        }
        if (previewRemovePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, previewRemovePipeline, nullptr);
            previewRemovePipeline = VK_NULL_HANDLE;
        }
        if (previewAddPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, previewAddPipeline, nullptr);
            previewAddPipeline = VK_NULL_HANDLE;
        }
        if (pipePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipePipeline, nullptr);
            pipePipeline = VK_NULL_HANDLE;
        }
        if (grassBillboardPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, grassBillboardPipeline, nullptr);
            grassBillboardPipeline = VK_NULL_HANDLE;
        }
        if (magicaPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, magicaPipeline, nullptr);
            magicaPipeline = VK_NULL_HANDLE;
        }
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }
    }

    void destroyVoxelGiPipelines(VkDevice device) {
        if (voxelGiSkyExposurePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiSkyExposurePipeline, nullptr);
            voxelGiSkyExposurePipeline = VK_NULL_HANDLE;
        }
        if (voxelGiOccupancyPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiOccupancyPipeline, nullptr);
            voxelGiOccupancyPipeline = VK_NULL_HANDLE;
        }
        if (voxelGiSurfacePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiSurfacePipeline, nullptr);
            voxelGiSurfacePipeline = VK_NULL_HANDLE;
        }
        if (voxelGiInjectPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiInjectPipeline, nullptr);
            voxelGiInjectPipeline = VK_NULL_HANDLE;
        }
        if (voxelGiPropagatePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiPropagatePipeline, nullptr);
            voxelGiPropagatePipeline = VK_NULL_HANDLE;
        }
        if (voxelGiPipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, voxelGiPipelineLayout, nullptr);
            voxelGiPipelineLayout = VK_NULL_HANDLE;
        }
    }
};

} // namespace voxelsprout::render
