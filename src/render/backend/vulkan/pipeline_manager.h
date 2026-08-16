#pragma once

#include <vulkan/vulkan.h>

namespace odai::render {

class PipelineManager {
public:
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipeline pipelineRt = VK_NULL_HANDLE;
    VkPipeline terrainTessPipeline = VK_NULL_HANDLE;
    VkPipeline hexTerrainPipeline = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkPipeline pipeShadowPipeline = VK_NULL_HANDLE;
    VkPipeline skyboxPipeline = VK_NULL_HANDLE;
    VkPipeline skyCloudPipeline = VK_NULL_HANDLE;
    VkPipeline tonemapPipeline = VK_NULL_HANDLE;
    VkPipeline pipePipeline = VK_NULL_HANDLE;
    VkPipeline importedStaticPipeline = VK_NULL_HANDLE;
    // Same shaders as importedStaticPipeline, with alpha blending on and depth
    // writes off — used for the blended tail of each imported chunk's draws.
    VkPipeline importedStaticPipelineBlended = VK_NULL_HANDLE;
    // ...and the same again with back-face culling off, for draws whose source
    // marked them two-sided.
    VkPipeline importedStaticPipelineBlendedTwoSided = VK_NULL_HANDLE;
    // Opaque two-sided. A shape whose NiStencilProperty says DRAW_BOTH needs
    // culling off whether or not it is blended; only the blended half had a
    // variant, so opaque DRAW_BOTH geometry was back-face culled and went
    // see-through from one side.
    VkPipeline importedStaticPipelineTwoSided = VK_NULL_HANDLE;
    // Depth-only prewrite for the opaque imported pass. Same vertex shader and
    // vertex layout as importedStaticPipeline -- that is what makes the depth it
    // writes bit-identical to what the shading pass computes -- paired with a
    // fragment shader that does nothing but the alpha test, and a zero colour
    // write mask.
    VkPipeline importedStaticDepthPrewritePipeline = VK_NULL_HANDLE;
    VkPipeline importedStaticDepthPrewritePipelineTwoSided = VK_NULL_HANDLE;
    VkPipeline importedStaticPipelineRt = VK_NULL_HANDLE;
    VkPipeline importedWaterPipeline = VK_NULL_HANDLE;
    VkPipeline importedWaterPipelineRt = VK_NULL_HANDLE;
    VkPipeline voxelNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline pipeNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline importedStaticNormalDepthPipeline = VK_NULL_HANDLE;
    // Cull-NONE variant, for the same DRAW_BOTH geometry the main pass draws
    // two-sided. Without it the depth/normal prepass and the lit pass disagree
    // about the silhouette of every thin surface, and SSAO occludes against the
    // wrong one.
    VkPipeline importedStaticNormalDepthPipelineTwoSided = VK_NULL_HANDLE;
    // Tessellated terrain variants of the lit pass and the merged prepass.
    // They exist only under the merged prepass, because they must lay and test
    // IDENTICAL depth -- see pass_pipelines.cc where they are built.
    VkPipeline importedTerrainTessPipeline = VK_NULL_HANDLE;
    VkPipeline importedTerrainTessNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline importedWaterNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline magicaPipeline = VK_NULL_HANDLE;
    VkPipeline magicaPipelineRt = VK_NULL_HANDLE;
    VkPipeline importedStaticShadowPipeline = VK_NULL_HANDLE;
    // Same shaders, 28-byte compact vertex stream instead of the full vertex.
    VkPipeline importedStaticShadowCompactPipeline = VK_NULL_HANDLE;
    // Cull-NONE variants of both. A thin DRAW_BOTH caster has one wound face,
    // so back-face culling makes its shadow appear and disappear depending on
    // which way the sun happens to face it.
    VkPipeline importedStaticShadowPipelineTwoSided = VK_NULL_HANDLE;
    VkPipeline importedStaticShadowCompactPipelineTwoSided = VK_NULL_HANDLE;
    // One pipeline per AO estimator (see ODAI_AO_MODE in ssao.comp.slang). All three
    // share the ssao pipeline layout and descriptor set -- only the shader differs.
    VkPipeline ssaoPipeline = VK_NULL_HANDLE;
    VkPipeline ssaoHbaoPipeline = VK_NULL_HANDLE;
    VkPipeline ssaoGtaoPipeline = VK_NULL_HANDLE;
    VkPipeline ssaoBlurPipeline = VK_NULL_HANDLE;
    VkPipeline previewAddPipeline = VK_NULL_HANDLE;
    VkPipeline previewRemovePipeline = VK_NULL_HANDLE;
    VkPipeline previewFaceOutlinePipeline = VK_NULL_HANDLE;
    VkPipelineLayout voxelGiPipelineLayout = VK_NULL_HANDLE;
    VkPipeline voxelGiSurfacePipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiSurfacePipelineRt = VK_NULL_HANDLE;
    VkPipeline voxelGiRestirCandidatePipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiRestirTemporalPipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiRestirSpatialPipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiRestirResolvePipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiOccupancyPipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiSkyExposurePipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiInjectPipeline = VK_NULL_HANDLE;
    VkPipeline voxelGiPropagatePipeline = VK_NULL_HANDLE;

    void destroyMainPipelines(VkDevice device) {
        if (ssaoBlurPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, ssaoBlurPipeline, nullptr);
            ssaoBlurPipeline = VK_NULL_HANDLE;
        }
        if (ssaoGtaoPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, ssaoGtaoPipeline, nullptr);
            ssaoGtaoPipeline = VK_NULL_HANDLE;
        }
        if (ssaoHbaoPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, ssaoHbaoPipeline, nullptr);
            ssaoHbaoPipeline = VK_NULL_HANDLE;
        }
        if (ssaoPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, ssaoPipeline, nullptr);
            ssaoPipeline = VK_NULL_HANDLE;
        }
        if (pipeNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeNormalDepthPipeline, nullptr);
            pipeNormalDepthPipeline = VK_NULL_HANDLE;
        }
        if (importedStaticNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticNormalDepthPipeline, nullptr);
            importedStaticNormalDepthPipeline = VK_NULL_HANDLE;
        }
        if (importedTerrainTessPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedTerrainTessPipeline, nullptr);
            importedTerrainTessPipeline = VK_NULL_HANDLE;
        }
        if (importedTerrainTessNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedTerrainTessNormalDepthPipeline, nullptr);
            importedTerrainTessNormalDepthPipeline = VK_NULL_HANDLE;
        }
        if (importedWaterNormalDepthPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedWaterNormalDepthPipeline, nullptr);
            importedWaterNormalDepthPipeline = VK_NULL_HANDLE;
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
        if (skyCloudPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, skyCloudPipeline, nullptr);
            skyCloudPipeline = VK_NULL_HANDLE;
        }
        if (shadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, shadowPipeline, nullptr);
            shadowPipeline = VK_NULL_HANDLE;
        }
        if (pipeShadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeShadowPipeline, nullptr);
            pipeShadowPipeline = VK_NULL_HANDLE;
        }
        if (importedStaticShadowCompactPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticShadowCompactPipeline, nullptr);
            importedStaticShadowCompactPipeline = VK_NULL_HANDLE;
        }
        if (importedStaticShadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticShadowPipeline, nullptr);
            importedStaticShadowPipeline = VK_NULL_HANDLE;
        }
        if (importedStaticShadowCompactPipelineTwoSided != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticShadowCompactPipelineTwoSided, nullptr);
            importedStaticShadowCompactPipelineTwoSided = VK_NULL_HANDLE;
        }
        if (importedStaticShadowPipelineTwoSided != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticShadowPipelineTwoSided, nullptr);
            importedStaticShadowPipelineTwoSided = VK_NULL_HANDLE;
        }
        if (importedStaticNormalDepthPipelineTwoSided != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticNormalDepthPipelineTwoSided, nullptr);
            importedStaticNormalDepthPipelineTwoSided = VK_NULL_HANDLE;
        }
        if (previewRemovePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, previewRemovePipeline, nullptr);
            previewRemovePipeline = VK_NULL_HANDLE;
        }
        if (previewFaceOutlinePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, previewFaceOutlinePipeline, nullptr);
            previewFaceOutlinePipeline = VK_NULL_HANDLE;
        }
        if (previewAddPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, previewAddPipeline, nullptr);
            previewAddPipeline = VK_NULL_HANDLE;
        }
        if (pipePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipePipeline, nullptr);
            pipePipeline = VK_NULL_HANDLE;
        }
        if (importedStaticPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticPipeline, nullptr);
            importedStaticPipeline = VK_NULL_HANDLE;
        }
        if (importedStaticPipelineBlended != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticPipelineBlended, nullptr);
            importedStaticPipelineBlended = VK_NULL_HANDLE;
        }
        if (importedStaticPipelineTwoSided != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticPipelineTwoSided, nullptr);
            importedStaticPipelineTwoSided = VK_NULL_HANDLE;
        }
        if (importedStaticDepthPrewritePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticDepthPrewritePipeline, nullptr);
            importedStaticDepthPrewritePipeline = VK_NULL_HANDLE;
        }
        if (importedStaticDepthPrewritePipelineTwoSided != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticDepthPrewritePipelineTwoSided, nullptr);
            importedStaticDepthPrewritePipelineTwoSided = VK_NULL_HANDLE;
        }
        if (importedStaticPipelineBlendedTwoSided != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticPipelineBlendedTwoSided, nullptr);
            importedStaticPipelineBlendedTwoSided = VK_NULL_HANDLE;
        }
        if (importedStaticPipelineRt != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedStaticPipelineRt, nullptr);
            importedStaticPipelineRt = VK_NULL_HANDLE;
        }
        if (importedWaterPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedWaterPipeline, nullptr);
            importedWaterPipeline = VK_NULL_HANDLE;
        }
        if (importedWaterPipelineRt != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, importedWaterPipelineRt, nullptr);
            importedWaterPipelineRt = VK_NULL_HANDLE;
        }
        if (magicaPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, magicaPipeline, nullptr);
            magicaPipeline = VK_NULL_HANDLE;
        }
        if (magicaPipelineRt != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, magicaPipelineRt, nullptr);
            magicaPipelineRt = VK_NULL_HANDLE;
        }
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        if (pipelineRt != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipelineRt, nullptr);
            pipelineRt = VK_NULL_HANDLE;
        }
        if (terrainTessPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, terrainTessPipeline, nullptr);
            terrainTessPipeline = VK_NULL_HANDLE;
        }
        if (hexTerrainPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, hexTerrainPipeline, nullptr);
            hexTerrainPipeline = VK_NULL_HANDLE;
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
        if (voxelGiSurfacePipelineRt != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiSurfacePipelineRt, nullptr);
            voxelGiSurfacePipelineRt = VK_NULL_HANDLE;
        }
        if (voxelGiRestirCandidatePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiRestirCandidatePipeline, nullptr);
            voxelGiRestirCandidatePipeline = VK_NULL_HANDLE;
        }
        if (voxelGiRestirTemporalPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiRestirTemporalPipeline, nullptr);
            voxelGiRestirTemporalPipeline = VK_NULL_HANDLE;
        }
        if (voxelGiRestirSpatialPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiRestirSpatialPipeline, nullptr);
            voxelGiRestirSpatialPipeline = VK_NULL_HANDLE;
        }
        if (voxelGiRestirResolvePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, voxelGiRestirResolvePipeline, nullptr);
            voxelGiRestirResolvePipeline = VK_NULL_HANDLE;
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

} // namespace odai::render
