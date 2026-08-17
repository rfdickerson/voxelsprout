#include "render/backend/vulkan/renderer_backend.h"

#include "core/log.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace odai::render {

namespace {

template <typename VkHandleT>
uint64_t vkHandleToUint64(VkHandleT handle) {
    if constexpr (std::is_pointer_v<VkHandleT>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

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
    VOX_LOGE("render") << context << " failed: "
                       << vkResultName(result) << " (" << static_cast<int>(result) << ")";
}

// Fixed singleton slots at the head of the bindless table. MIRRORED in
// render/renderer_shared.h (which chunk_upload.cc uses to assign imported
// texture slots) and in shaders/imported_static.frag.slang — see the longer
// note in renderer_shared.h. Nothing enforces agreement at compile time; this
// TU does not include that header, so a mismatch is silent.
constexpr uint32_t kBindlessTextureIndexDiffuse = 0u;
constexpr uint32_t kBindlessTextureIndexHdrResolved = 1u;
constexpr uint32_t kBindlessTextureIndexShadowAtlas = 2u;
constexpr uint32_t kBindlessTextureIndexNormalDepth = 3u;
constexpr uint32_t kBindlessTextureIndexSsaoBlur = 4u;
constexpr uint32_t kBindlessTextureIndexSsaoRaw = 5u;
constexpr uint32_t kBindlessTextureIndexPlantDiffuse = 6u;
constexpr uint32_t kBindlessTextureIndexSkyDaylight = 7u;
constexpr uint32_t kBindlessTextureIndexWaterNormal = 8u;
constexpr uint32_t kBindlessTextureIndexTerrainDetail = 9u;
constexpr uint32_t kBindlessTextureIndexFogMap = 10u;
constexpr uint32_t kBindlessTextureStaticCount = 11u;
constexpr uint32_t kAutoExposureHistogramBins = 64u;

} // namespace

bool RendererBackend::createDescriptorResources() {
    if (m_descriptorSetLayout == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(15);

        VkDescriptorSetLayoutBinding mvpBinding{};
        mvpBinding.binding = 0;
        mvpBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        mvpBinding.descriptorCount = 1;
        mvpBinding.stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(mvpBinding);

        VkDescriptorSetLayoutBinding diffuseTextureBinding{};
        diffuseTextureBinding.binding = 1;
        diffuseTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        diffuseTextureBinding.descriptorCount = 1;
        diffuseTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(diffuseTextureBinding);

        VkDescriptorSetLayoutBinding exposureStateBinding{};
        exposureStateBinding.binding = 2;
        exposureStateBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        exposureStateBinding.descriptorCount = 1;
        exposureStateBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(exposureStateBinding);

        VkDescriptorSetLayoutBinding hdrSceneBinding{};
        hdrSceneBinding.binding = 3;
        hdrSceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        hdrSceneBinding.descriptorCount = 1;
        hdrSceneBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(hdrSceneBinding);

        VkDescriptorSetLayoutBinding shadowMapBinding{};
        shadowMapBinding.binding = 4;
        shadowMapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowMapBinding.descriptorCount = 1;
        shadowMapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(shadowMapBinding);

        VkDescriptorSetLayoutBinding waterRefractionBinding{};
        waterRefractionBinding.binding = 5;
        waterRefractionBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        waterRefractionBinding.descriptorCount = 1;
        waterRefractionBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(waterRefractionBinding);

        VkDescriptorSetLayoutBinding waterReflectionBinding{};
        waterReflectionBinding.binding = 18;
        waterReflectionBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        waterReflectionBinding.descriptorCount = 1;
        waterReflectionBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(waterReflectionBinding);

        VkDescriptorSetLayoutBinding normalDepthBinding{};
        normalDepthBinding.binding = 6;
        normalDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalDepthBinding.descriptorCount = 1;
        normalDepthBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(normalDepthBinding);

        VkDescriptorSetLayoutBinding ssaoBlurBinding{};
        ssaoBlurBinding.binding = 7;
        ssaoBlurBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssaoBlurBinding.descriptorCount = 1;
        ssaoBlurBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(ssaoBlurBinding);

        VkDescriptorSetLayoutBinding ssaoRawBinding{};
        ssaoRawBinding.binding = 8;
        ssaoRawBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssaoRawBinding.descriptorCount = 1;
        ssaoRawBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(ssaoRawBinding);

        VkDescriptorSetLayoutBinding voxelGiBinding{};
        voxelGiBinding.binding = 9;
        voxelGiBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        voxelGiBinding.descriptorCount = 1;
        voxelGiBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(voxelGiBinding);

        VkDescriptorSetLayoutBinding sunShaftBinding{};
        sunShaftBinding.binding = 10;
        sunShaftBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sunShaftBinding.descriptorCount = 1;
        sunShaftBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(sunShaftBinding);

        // Bloom source: always the mip-chained render-resolution HDR target. See
        // the note in tone_map.frag.slang for why this cannot share the scene
        // binding once an upscaler is in the chain.
        VkDescriptorSetLayoutBinding bloomSourceBinding{};
        bloomSourceBinding.binding = 14;
        bloomSourceBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bloomSourceBinding.descriptorCount = 1;
        bloomSourceBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(bloomSourceBinding);

        VkDescriptorSetLayoutBinding voxelGiOccupancyDebugBinding{};
        voxelGiOccupancyDebugBinding.binding = 11;
        voxelGiOccupancyDebugBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        voxelGiOccupancyDebugBinding.descriptorCount = 1;
        voxelGiOccupancyDebugBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(voxelGiOccupancyDebugBinding);

        if (m_rayTracingRuntimeEnabled) {
            VkDescriptorSetLayoutBinding shadowSceneBinding{};
            shadowSceneBinding.binding = 12;
            shadowSceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            shadowSceneBinding.descriptorCount = 1;
            shadowSceneBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings.push_back(shadowSceneBinding);
        }

        // Named material library, indexed by vertex flag bits 24-31 (see
        // import/imported_material.h). A storage buffer rather than the camera
        // UBO or a push constant: inFlags is `nointerpolation`, so material
        // identity varies per triangle *within* a draw, which rules push
        // constants out entirely; and this data changes only when someone edits
        // a coefficient, so it has no business being re-uploaded every frame
        // alongside the camera.
        //
        // Pushed unconditionally, after the conditional binding 12 above. The
        // hole when ray tracing is off is fine — descriptorBufferBindingOffset()
        // resolves offsets per binding rather than by position.
        // Clustered light culling: the per-cluster 64-bit mask the cull compute
        // pass writes. See src/render/shaders/light_clusters.slang.
        VkDescriptorSetLayoutBinding lightClusterBinding{};
        lightClusterBinding.binding = 15;
        lightClusterBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lightClusterBinding.descriptorCount = 1;
        lightClusterBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(lightClusterBinding);

        // Full-resolution 64-bit per-pixel mask produced by the contact-shadow
        // resolve compute pass. It is a buffer so the main shader pays one
        // coherent load and no sampler state per fragment.
        VkDescriptorSetLayoutBinding contactShadowBinding{};
        contactShadowBinding.binding = 16;
        contactShadowBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        contactShadowBinding.descriptorCount = 1;
        contactShadowBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(contactShadowBinding);

        VkDescriptorSetLayoutBinding screenSpaceGiBinding{};
        screenSpaceGiBinding.binding = 17;
        screenSpaceGiBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        screenSpaceGiBinding.descriptorCount = 1;
        screenSpaceGiBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(screenSpaceGiBinding);

        VkDescriptorSetLayoutBinding materialTableBinding{};
        materialTableBinding.binding = 13;
        materialTableBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        materialTableBinding.descriptorCount = 1;
        materialTableBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(materialTableBinding);

        if (!createDescriptorSetLayout(
                bindings,
                m_descriptorSetLayout,
                "vkCreateDescriptorSetLayout",
                "renderer.descriptorSetLayout.main",
                nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
            )) {
            return false;
        }
    }

    // Descriptor-buffer backing for the main per-frame set (set 0): camera UBO +
    // storage buffer + combined image samplers (+ optional accel structure).
    if (!m_mainBufferSet.valid()) {
        if (!createDescriptorBufferSet(
                m_descriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                    VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.main",
                m_mainBufferSet
            )) {
            return false;
        }
    }

    if (m_supportsBindlessDescriptors && m_bindlessTextureCapacity > 0) {
        if (m_bindlessDescriptorSetLayout == VK_NULL_HANDLE) {
            VkDescriptorSetLayoutBinding bindlessTexturesBinding{};
            bindlessTexturesBinding.binding = 0;
            bindlessTexturesBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindlessTexturesBinding.descriptorCount = m_bindlessTextureCapacity;
            bindlessTexturesBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            const VkDescriptorBindingFlags bindlessBindingFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCreateInfo{};
            bindingFlagsCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            bindingFlagsCreateInfo.bindingCount = 1;
            bindingFlagsCreateInfo.pBindingFlags = &bindlessBindingFlags;

            const std::array<VkDescriptorSetLayoutBinding, 1> bindlessBindings = {bindlessTexturesBinding};
            if (!createDescriptorSetLayout(
                    bindlessBindings,
                    m_bindlessDescriptorSetLayout,
                    "vkCreateDescriptorSetLayout(bindless)",
                    "renderer.descriptorSetLayout.bindless",
                    &bindingFlagsCreateInfo,
                    VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
                )) {
                return false;
            }
        }

        // Descriptor-buffer backing for the bindless texture array (set 1). One
        // region (shared across frames); partially-bound slots simply stay unwritten.
        if (!m_bindlessBufferSet.valid()) {
            if (!createDescriptorBufferSet(
                    m_bindlessDescriptorSetLayout,
                    1u,
                    VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                        VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                    "renderer.descriptorBuffer.bindless",
                    m_bindlessBufferSet
                )) {
                return false;
            }
        }
    }

    return true;
}

void RendererBackend::updateFrameDescriptorSets(
    uint32_t aoFrameIndex,
    const VkDescriptorBufferInfo& cameraBufferInfo,
    VkDeviceSize cameraSliceOffset,
    VkBuffer autoExposureHistogramBuffer,
    VkBuffer autoExposureStateBuffer,
    const VkDescriptorBufferInfo* voxelGiChunkMetaBufferInfo,
    const VkDescriptorBufferInfo* voxelGiChunkVoxelBufferInfo
) {
    // Camera UBO device address (frame-arena slice) for descriptor-buffer writes.
    //
    // cameraSliceOffset, NOT cameraBufferInfo.offset -- the latter is always 0.
    // (It is left at 0 for a classic descriptor-set path that would carry the
    // slice as a dynamic offset; no 3-D pass takes that path any more, which is
    // why FrameExecutionContext::mvpDynamicOffset is now unread.)
    // Adding 0 here pointed every descriptor-buffer consumer of the camera --
    // main, voxel GI, sun shafts, SSAO -- at ring offset 0 rather than at this
    // frame's camera slice. That was survivable only for as long as the camera
    // UBO happened to be the first allocation of every frame: the first
    // allocation taken ahead of it (the skinned-actor bone matrices) shifted
    // the camera slice, left ring offset 0 holding those matrices, and every
    // pass then read a garbage view-projection -- which renders as a flat
    // single-colour frame with the UI still correct on top of it, because the
    // UI does not go through this camera.
    VkBufferDeviceAddressInfo cameraAddressInfo{};
    cameraAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    cameraAddressInfo.buffer = cameraBufferInfo.buffer;
    const VkDeviceAddress cameraDeviceAddress =
        (cameraBufferInfo.buffer != VK_NULL_HANDLE)
            ? vkGetBufferDeviceAddress(m_device, &cameraAddressInfo) + cameraSliceOffset
            : 0;

    VkDescriptorImageInfo hdrSceneImageInfo{};
    hdrSceneImageInfo.sampler = m_hdrResolveSampler;
    // The tonemap pass's scene input. When the temporal upscaler ran, that is
    // its output -- already at swapchain resolution -- rather than hdrResolve,
    // which is still at render resolution and would be stretched by a bilinear
    // fetch. Sampling the upscaled image is what makes the reconstruction
    // actually reach the screen instead of being resolved and then thrown away.
    // m_taaHistoryIndex ^ 1, not m_taaHistoryIndex.
    //
    // These descriptors are written BEFORE recordTaaPass runs, and at that point
    // m_taaHistoryIndex still names the image the pass is about to read as
    // HISTORY -- it only advances to the freshly written image at the end of the
    // pass. Sampling it here displays last frame's accumulation instead of this
    // frame's, which is a frame of latency plus one accumulation step of
    // staleness on every pixel.
    hdrSceneImageInfo.imageView = temporalUpscaleActive()
        ? m_taaImageViews[m_taaHistoryIndex ^ 1u]
        : m_hdrResolveSampleImageViews[aoFrameIndex];
    hdrSceneImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo diffuseTextureImageInfo{};
    diffuseTextureImageInfo.sampler = m_diffuseTextureSampler;
    diffuseTextureImageInfo.imageView = m_diffuseTextureImageView;
    diffuseTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo plantDiffuseTextureImageInfo{};
    plantDiffuseTextureImageInfo.sampler = m_diffuseTexturePlantSampler;
    plantDiffuseTextureImageInfo.imageView =
        (m_plantDiffuseTextureImageView != VK_NULL_HANDLE) ? m_plantDiffuseTextureImageView : m_diffuseTextureImageView;
    plantDiffuseTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo morrowindSkyTextureImageInfo{};
    morrowindSkyTextureImageInfo.sampler =
        (m_morrowindSkyTextureSampler != VK_NULL_HANDLE) ? m_morrowindSkyTextureSampler : m_diffuseTextureSampler;
    morrowindSkyTextureImageInfo.imageView =
        (m_morrowindSkyTextureImageView != VK_NULL_HANDLE) ? m_morrowindSkyTextureImageView : m_diffuseTextureImageView;
    morrowindSkyTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo waterNormalTextureImageInfo{};
    waterNormalTextureImageInfo.sampler =
        (m_waterNormalTextureSampler != VK_NULL_HANDLE) ? m_waterNormalTextureSampler : m_diffuseTextureSampler;
    waterNormalTextureImageInfo.imageView =
        (m_waterNormalTextureImageView != VK_NULL_HANDLE) ? m_waterNormalTextureImageView : m_diffuseTextureImageView;
    waterNormalTextureImageInfo.imageLayout = m_hostCopyFinalLayout;

    VkDescriptorImageInfo terrainDetailTextureImageInfo{};
    terrainDetailTextureImageInfo.sampler =
        (m_terrainDetailTextureSampler != VK_NULL_HANDLE) ? m_terrainDetailTextureSampler : m_diffuseTextureSampler;
    terrainDetailTextureImageInfo.imageView =
        (m_terrainDetailTextureImageView != VK_NULL_HANDLE) ? m_terrainDetailTextureImageView : m_diffuseTextureImageView;
    terrainDetailTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo fogMapTextureImageInfo{};
    fogMapTextureImageInfo.sampler =
        (m_fogMapSampler != VK_NULL_HANDLE) ? m_fogMapSampler : m_diffuseTextureSampler;
    fogMapTextureImageInfo.imageView =
        (m_fogMapTextureResource.imageView != VK_NULL_HANDLE) ? m_fogMapTextureResource.imageView : m_diffuseTextureImageView;
    fogMapTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo shadowMapImageInfo{};
    shadowMapImageInfo.sampler = m_shadowDepthSampler;
    shadowMapImageInfo.imageView = m_shadowDepthImageView;
    shadowMapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo waterRefractionImageInfo{};
    waterRefractionImageInfo.sampler = m_hdrResolveSampler;
    waterRefractionImageInfo.imageView =
        (aoFrameIndex < m_waterRefractionImageViews.size()) ? m_waterRefractionImageViews[aoFrameIndex] : VK_NULL_HANDLE;
    waterRefractionImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo waterReflectionImageInfo{};
    waterReflectionImageInfo.sampler = m_hdrResolveSampler;
    const std::uint32_t waterReflectionResolveOutput =
        m_waterReflectionHistoryIndex ^ 1u;
    const bool useResolvedWaterReflection =
        m_waterReflectionTemporalEnabled &&
        m_waterReflectionResolvePipeline != VK_NULL_HANDLE &&
        m_waterReflectionHistoryImageViews[waterReflectionResolveOutput] != VK_NULL_HANDLE;
    waterReflectionImageInfo.imageView = useResolvedWaterReflection
        ? m_waterReflectionHistoryImageViews[waterReflectionResolveOutput]
        : ((aoFrameIndex < m_waterReflectionImageViews.size())
            ? m_waterReflectionImageViews[aoFrameIndex]
            : VK_NULL_HANDLE);
    waterReflectionImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo normalDepthImageInfo{};
    normalDepthImageInfo.sampler = m_normalDepthSampler;
    normalDepthImageInfo.imageView = m_normalDepthImageViews[aoFrameIndex];
    normalDepthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo ssaoBlurImageInfo{};
    ssaoBlurImageInfo.sampler = m_ssaoSampler;
    ssaoBlurImageInfo.imageView = m_ssaoBlurImageViews[aoFrameIndex];
    ssaoBlurImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo ssaoRawImageInfo{};
    ssaoRawImageInfo.sampler = m_ssaoSampler;
    ssaoRawImageInfo.imageView = m_ssaoRawImageViews[aoFrameIndex];
    ssaoRawImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo voxelGiVolumeImageInfo{};
    voxelGiVolumeImageInfo.sampler = m_voxelGiSampler;
    voxelGiVolumeImageInfo.imageView = m_voxelGiImageViews[1];
    voxelGiVolumeImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo voxelGiOccupancyDebugImageInfo{};
    voxelGiOccupancyDebugImageInfo.sampler = m_voxelGiOccupancySampler;
    voxelGiOccupancyDebugImageInfo.imageView = m_voxelGiOccupancyImageView;
    voxelGiOccupancyDebugImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo sunShaftImageInfo{};
    sunShaftImageInfo.sampler = m_sunShaftSampler;
    sunShaftImageInfo.imageView =
        (aoFrameIndex < m_sunShaftImageViews.size()) ? m_sunShaftImageViews[aoFrameIndex] : VK_NULL_HANDLE;
    sunShaftImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo autoExposureStateBufferInfo{};
    autoExposureStateBufferInfo.buffer = autoExposureStateBuffer;
    autoExposureStateBufferInfo.offset = 0;
    autoExposureStateBufferInfo.range = sizeof(float) * 4u;
    const bool hasRayTracingSceneDescriptor = m_rayTracingRuntimeEnabled && m_rtTlas.handle != VK_NULL_HANDLE;
    VkWriteDescriptorSetAccelerationStructureKHR rayTracingSceneWriteInfo{};
    rayTracingSceneWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    rayTracingSceneWriteInfo.accelerationStructureCount = hasRayTracingSceneDescriptor ? 1u : 0u;
    rayTracingSceneWriteInfo.pAccelerationStructures = hasRayTracingSceneDescriptor ? &m_rtTlas.handle : nullptr;

    if (m_mainBufferSet.valid()) {
        const uint32_t region = m_currentFrame;
        const VkDescriptorSetLayout layout = m_descriptorSetLayout;
        auto mainOffset = [&](uint32_t binding) { return descriptorBufferBindingOffset(layout, binding); };
        auto sampler = [&](uint32_t binding, const VkDescriptorImageInfo& info) {
            writeDescriptorBufferCombinedImageSampler(
                m_mainBufferSet, region, mainOffset(binding), 0, info.imageView, info.sampler, info.imageLayout);
        };
        // Camera UBO (binding 0) and exposure-state storage buffer (binding 2).
        writeDescriptorBufferUniform(m_mainBufferSet, region, mainOffset(0), cameraDeviceAddress, cameraBufferInfo.range);
        VkBufferDeviceAddressInfo exposureAddrInfo{};
        exposureAddrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        exposureAddrInfo.buffer = autoExposureStateBufferInfo.buffer;
        const VkDeviceAddress exposureAddress = (autoExposureStateBufferInfo.buffer != VK_NULL_HANDLE)
            ? vkGetBufferDeviceAddress(m_device, &exposureAddrInfo) : 0;
        writeDescriptorBufferStorage(m_mainBufferSet, region, mainOffset(2), exposureAddress, autoExposureStateBufferInfo.range);

        // Material table (binding 13). Each frame in flight gets its own region
        // of the buffer, so an edit landing between frames cannot mutate memory
        // the GPU is still reading. The CPU mirror is copied in only when it
        // actually changed -- a slider drag dirties it, an idle frame does not.
        const VkBuffer materialBuffer = m_bufferAllocator.getBuffer(m_importedMaterialBufferHandle);
        if (materialBuffer != VK_NULL_HANDLE) {
            constexpr VkDeviceSize kMaterialRegionSize =
                static_cast<VkDeviceSize>(sizeof(importer::GpuImportedMaterial)) *
                importer::kImportedSceneMaterialTableCapacity;
            const VkDeviceSize materialRegionOffset = kMaterialRegionSize * region;
            // Dirty is a countdown, not a flag: each frame in flight owns a
            // separate region, so one edit has to be copied into every one of
            // them before it is fully applied.
            if (m_importedMaterialTableDirtyFrames > 0u) {
                if (void* mapped = m_bufferAllocator.mapBuffer(
                        m_importedMaterialBufferHandle, materialRegionOffset, kMaterialRegionSize)) {
                    std::memcpy(mapped, m_importedMaterialTable.data(),
                                static_cast<std::size_t>(kMaterialRegionSize));
                    m_bufferAllocator.unmapBuffer(m_importedMaterialBufferHandle);
                    --m_importedMaterialTableDirtyFrames;
                }
            }
            VkBufferDeviceAddressInfo materialAddrInfo{};
            materialAddrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            materialAddrInfo.buffer = materialBuffer;
            const VkDeviceAddress materialAddress =
                vkGetBufferDeviceAddress(m_device, &materialAddrInfo) + materialRegionOffset;
            writeDescriptorBufferStorage(m_mainBufferSet, region, mainOffset(13), materialAddress,
                                         kMaterialRegionSize);
        }
        // Combined image samplers (bindings 1, 3-11).
        sampler(1, diffuseTextureImageInfo);
        sampler(3, hdrSceneImageInfo);
        sampler(4, shadowMapImageInfo);
        sampler(5, waterRefractionImageInfo);
        sampler(18, waterReflectionImageInfo);
        sampler(6, normalDepthImageInfo);
        sampler(7, ssaoBlurImageInfo);
        sampler(8, ssaoRawImageInfo);
        sampler(9, voxelGiVolumeImageInfo);
        sampler(10, sunShaftImageInfo);
        // Bloom always reads the mip-chained hdrResolve, even when the scene
        // above came from the upscaler's output.
        VkDescriptorImageInfo bloomSourceImageInfo{};
        bloomSourceImageInfo.sampler = m_hdrResolveSampler;
        bloomSourceImageInfo.imageView = m_hdrResolveSampleImageViews[aoFrameIndex];
        bloomSourceImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sampler(14, bloomSourceImageInfo);
        sampler(11, voxelGiOccupancyDebugImageInfo);
        // Ray-traced scene acceleration structure (binding 12) when RT is live.
        if (hasRayTracingSceneDescriptor) {
            writeDescriptorBufferAccelerationStructure(m_mainBufferSet, region, mainOffset(12), m_rtTlas.deviceAddress);
        }

        // Cluster light mask (binding 15). Written unconditionally when the
        // buffer exists: the descriptor must be valid even on a frame where the
        // cull pass is skipped, because the fragment shader's fallback branch
        // is decided by a uniform, not by the descriptor, and a null descriptor
        // in a bound set is undefined behaviour whether or not it is read.
        const VkBuffer clusterBuffer = m_bufferAllocator.getBuffer(m_lightClusterBufferHandle);
        if (clusterBuffer != VK_NULL_HANDLE) {
            VkBufferDeviceAddressInfo clusterAddrInfo{};
            clusterAddrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            clusterAddrInfo.buffer = clusterBuffer;
            const VkDeviceAddress clusterAddress = vkGetBufferDeviceAddress(m_device, &clusterAddrInfo);
            writeDescriptorBufferStorage(
                m_mainBufferSet, region, mainOffset(15), clusterAddress, m_lightClusterBufferSize);
        }
        const VkBuffer contactMaskBuffer =
            m_bufferAllocator.getBuffer(m_contactShadowFullMaskBufferHandle);
        if (contactMaskBuffer != VK_NULL_HANDLE) {
            VkBufferDeviceAddressInfo addressInfo{};
            addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addressInfo.buffer = contactMaskBuffer;
            writeDescriptorBufferStorage(
                m_mainBufferSet, region, mainOffset(16),
                vkGetBufferDeviceAddress(m_device, &addressInfo),
                m_contactShadowFullMaskBufferSize);
        }
        const std::uint32_t screenSpaceGiCurrent = m_screenSpaceGiHistoryIndex ^ 1u;
        BufferHandle screenSpaceGiHandle =
            m_screenSpaceGiRecordBufferHandles[screenSpaceGiCurrent];
        VkDeviceSize screenSpaceGiSize = m_screenSpaceGiRecordBufferSize;
        if (screenSpaceGiHandle == kInvalidBufferHandle) {
            screenSpaceGiHandle = m_contactShadowHalfBufferHandle;
            screenSpaceGiSize = m_contactShadowHalfBufferSize;
        }
        const VkBuffer screenSpaceGiBuffer =
            m_bufferAllocator.getBuffer(screenSpaceGiHandle);
        if (screenSpaceGiBuffer != VK_NULL_HANDLE) {
            VkBufferDeviceAddressInfo addressInfo{};
            addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addressInfo.buffer = screenSpaceGiBuffer;
            writeDescriptorBufferStorage(
                m_mainBufferSet, region, mainOffset(17),
                vkGetBufferDeviceAddress(m_device, &addressInfo), screenSpaceGiSize);
        }
    }

    if (m_lightClusterAvailable && m_lightClusterBufferSet.valid()) {
        const uint32_t region = m_currentFrame;
        const VkDescriptorSetLayout layout = m_lightClusterDescriptorSetLayout;
        writeDescriptorBufferUniform(
            m_lightClusterBufferSet, region, descriptorBufferBindingOffset(layout, 0),
            cameraDeviceAddress, cameraBufferInfo.range);
        const VkBuffer clusterBuffer = m_bufferAllocator.getBuffer(m_lightClusterBufferHandle);
        if (clusterBuffer != VK_NULL_HANDLE) {
            VkBufferDeviceAddressInfo clusterAddrInfo{};
            clusterAddrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            clusterAddrInfo.buffer = clusterBuffer;
            const VkDeviceAddress clusterAddress = vkGetBufferDeviceAddress(m_device, &clusterAddrInfo);
            writeDescriptorBufferStorage(
                m_lightClusterBufferSet, region, descriptorBufferBindingOffset(layout, 1),
                clusterAddress, m_lightClusterBufferSize);
        }
    }

    if (m_contactShadowAvailable && m_contactShadowBufferSet.valid()) {
        const uint32_t region = m_currentFrame;
        const VkDescriptorSetLayout layout = m_contactShadowDescriptorSetLayout;
        const auto offset = [&](uint32_t binding) {
            return descriptorBufferBindingOffset(layout, binding);
        };
        writeDescriptorBufferUniform(
            m_contactShadowBufferSet, region, offset(0),
            cameraDeviceAddress, cameraBufferInfo.range);
        writeDescriptorBufferCombinedImageSampler(
            m_contactShadowBufferSet, region, offset(1), 0u,
            normalDepthImageInfo.imageView, normalDepthImageInfo.sampler,
            normalDepthImageInfo.imageLayout);
        const auto storage = [&](uint32_t binding, BufferHandle handle, VkDeviceSize size) {
            const VkBuffer buffer = m_bufferAllocator.getBuffer(handle);
            if (buffer == VK_NULL_HANDLE) {
                return;
            }
            VkBufferDeviceAddressInfo addressInfo{};
            addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addressInfo.buffer = buffer;
            writeDescriptorBufferStorage(
                m_contactShadowBufferSet, region, offset(binding),
                vkGetBufferDeviceAddress(m_device, &addressInfo), size);
        };
        storage(2u, m_lightClusterBufferHandle, m_lightClusterBufferSize);
        storage(3u, m_contactShadowDepthBufferHandle, m_contactShadowDepthBufferSize);
        storage(4u, m_contactShadowHalfBufferHandle, m_contactShadowHalfBufferSize);
        storage(5u, m_contactShadowFullMaskBufferHandle, m_contactShadowFullMaskBufferSize);
    }

    if (m_screenSpaceGiAvailable && m_screenSpaceGiBufferSet.valid()) {
        const uint32_t region = m_currentFrame;
        const VkDescriptorSetLayout layout = m_screenSpaceGiDescriptorSetLayout;
        const auto offset = [&](uint32_t binding) {
            return descriptorBufferBindingOffset(layout, binding);
        };
        writeDescriptorBufferUniform(
            m_screenSpaceGiBufferSet, region, offset(0),
            cameraDeviceAddress, cameraBufferInfo.range);
        writeDescriptorBufferCombinedImageSampler(
            m_screenSpaceGiBufferSet, region, offset(1), 0u,
            normalDepthImageInfo.imageView, normalDepthImageInfo.sampler,
            normalDepthImageInfo.imageLayout);
        writeDescriptorBufferCombinedImageSampler(
            m_screenSpaceGiBufferSet, region, offset(2), 0u,
            m_taaImageViews[m_taaHistoryIndex], m_ssaoSampler,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const auto storage = [&](uint32_t binding, BufferHandle handle, VkDeviceSize size) {
            const VkBuffer buffer = m_bufferAllocator.getBuffer(handle);
            if (buffer == VK_NULL_HANDLE) {
                return;
            }
            VkBufferDeviceAddressInfo addressInfo{};
            addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addressInfo.buffer = buffer;
            writeDescriptorBufferStorage(
                m_screenSpaceGiBufferSet, region, offset(binding),
                vkGetBufferDeviceAddress(m_device, &addressInfo), size);
        };
        storage(3u, m_contactShadowDepthBufferHandle, m_contactShadowDepthBufferSize);
        storage(4u, m_screenSpaceGiRecordBufferHandles[m_screenSpaceGiHistoryIndex],
                m_screenSpaceGiRecordBufferSize);
        storage(5u, m_screenSpaceGiRecordBufferHandles[m_screenSpaceGiHistoryIndex ^ 1u],
                m_screenSpaceGiRecordBufferSize);
    }

    if (m_voxelGiComputeAvailable && m_voxelGiBufferSet.valid()) {
        const uint32_t region = m_currentFrame;
        const VkDescriptorSetLayout layout = m_voxelGiDescriptorSetLayout;
        auto bindOffset = [&](uint32_t binding) {
            return descriptorBufferBindingOffset(layout, binding);
        };
        auto bufferAddress = [&](const VkDescriptorBufferInfo& info) -> VkDeviceAddress {
            if (info.buffer == VK_NULL_HANDLE) {
                return 0;
            }
            VkBufferDeviceAddressInfo addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addrInfo.buffer = info.buffer;
            return vkGetBufferDeviceAddress(m_device, &addrInfo) + info.offset;
        };

        // Camera UBO + shadow sampler.
        writeDescriptorBufferUniform(m_voxelGiBufferSet, region, bindOffset(0), cameraDeviceAddress, cameraBufferInfo.range);
        writeDescriptorBufferCombinedImageSampler(m_voxelGiBufferSet, region, bindOffset(1), 0,
            shadowMapImageInfo.imageView, shadowMapImageInfo.sampler, shadowMapImageInfo.imageLayout);
        // Radiance A (storage) / read (sampled) / B (storage).
        writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(2), m_voxelGiImageViews[0], VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptorBufferSampledImage(m_voxelGiBufferSet, region, bindOffset(3), m_voxelGiImageViews[0], VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(4), m_voxelGiImageViews[1], VK_IMAGE_LAYOUT_GENERAL);
        // Occupancy sampled (read-only) then the 6 surface faces (storage).
        writeDescriptorBufferSampledImage(m_voxelGiBufferSet, region, bindOffset(5), m_voxelGiOccupancyImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        for (uint32_t faceIndex = 0; faceIndex < 6u; ++faceIndex) {
            writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(6u + faceIndex),
                m_voxelGiSurfaceFaceImageViews[faceIndex], VK_IMAGE_LAYOUT_GENERAL);
        }
        // Sky exposure + occupancy storage view.
        writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(12), m_voxelGiSkyExposureImageView, VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(13), m_voxelGiOccupancyImageView, VK_IMAGE_LAYOUT_GENERAL);
        // Chunk meta / voxel storage buffers (fall back to the exposure buffer when absent).
        VkDescriptorBufferInfo voxelGiFallbackStorageInfo{};
        voxelGiFallbackStorageInfo.buffer = autoExposureStateBufferInfo.buffer;
        voxelGiFallbackStorageInfo.offset = 0;
        voxelGiFallbackStorageInfo.range = autoExposureStateBufferInfo.range;
        const VkDescriptorBufferInfo& voxelGiChunkMetaInfo =
            (voxelGiChunkMetaBufferInfo != nullptr) ? *voxelGiChunkMetaBufferInfo : voxelGiFallbackStorageInfo;
        const VkDescriptorBufferInfo& voxelGiChunkVoxelInfo =
            (voxelGiChunkVoxelBufferInfo != nullptr) ? *voxelGiChunkVoxelBufferInfo : voxelGiFallbackStorageInfo;
        writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(14),
            bufferAddress(voxelGiChunkMetaInfo), voxelGiChunkMetaInfo.range);
        writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(15),
            bufferAddress(voxelGiChunkVoxelInfo), voxelGiChunkVoxelInfo.range);

        // Ray-traced surface tracing + ReSTIR reservoirs (only when RT is live).
        const bool hasVoxelGiRayTracingSceneDescriptor =
            m_rayTracingRuntimeEnabled && m_rtTlas.handle != VK_NULL_HANDLE;
        if (hasVoxelGiRayTracingSceneDescriptor) {
            writeDescriptorBufferAccelerationStructure(m_voxelGiBufferSet, region, bindOffset(16), m_rtTlas.deviceAddress);
        }
        const VkBuffer restirCurrent = m_bufferAllocator.getBuffer(m_voxelGiRestirReservoirCurrentBufferHandle);
        const VkBuffer restirPrevious = m_bufferAllocator.getBuffer(m_voxelGiRestirReservoirPreviousBufferHandle);
        const VkBuffer restirScratch = m_bufferAllocator.getBuffer(m_voxelGiRestirReservoirScratchBufferHandle);
        if (m_rayTracingRuntimeEnabled &&
            restirCurrent != VK_NULL_HANDLE && restirPrevious != VK_NULL_HANDLE && restirScratch != VK_NULL_HANDLE) {
            writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(17),
                m_bufferAllocator.getDeviceAddress(m_voxelGiRestirReservoirCurrentBufferHandle),
                m_bufferAllocator.getSize(m_voxelGiRestirReservoirCurrentBufferHandle));
            writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(18),
                m_bufferAllocator.getDeviceAddress(m_voxelGiRestirReservoirPreviousBufferHandle),
                m_bufferAllocator.getSize(m_voxelGiRestirReservoirPreviousBufferHandle));
            writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(19),
                m_bufferAllocator.getDeviceAddress(m_voxelGiRestirReservoirScratchBufferHandle),
                m_bufferAllocator.getSize(m_voxelGiRestirReservoirScratchBufferHandle));
        }
    }

    if (m_autoExposureComputeAvailable &&
        m_autoExposureBufferSet.valid() &&
        autoExposureHistogramBuffer != VK_NULL_HANDLE &&
        autoExposureStateBuffer != VK_NULL_HANDLE) {
        // Descriptor-buffer writes are cheap (a memcpy into mapped memory), so we
        // write the region unconditionally each frame rather than diff against a key.
        const uint32_t region = m_currentFrame;
        const VkDeviceSize hdrOffset = descriptorBufferBindingOffset(m_autoExposureDescriptorSetLayout, 0);
        const VkDeviceSize histogramOffset = descriptorBufferBindingOffset(m_autoExposureDescriptorSetLayout, 1);
        const VkDeviceSize stateOffset = descriptorBufferBindingOffset(m_autoExposureDescriptorSetLayout, 2);
        writeDescriptorBufferCombinedImageSampler(
            m_autoExposureBufferSet, region, hdrOffset, 0,
            hdrSceneImageInfo.imageView, hdrSceneImageInfo.sampler, hdrSceneImageInfo.imageLayout);
        writeDescriptorBufferStorage(
            m_autoExposureBufferSet, region, histogramOffset,
            m_bufferAllocator.getDeviceAddress(m_autoExposureHistogramBufferHandle),
            static_cast<VkDeviceSize>(kAutoExposureHistogramBins * sizeof(uint32_t)));
        writeDescriptorBufferStorage(
            m_autoExposureBufferSet, region, stateOffset,
            m_bufferAllocator.getDeviceAddress(m_autoExposureStateBufferHandle),
            sizeof(float) * 4u);
    }

    if (m_sunShaftComputeAvailable &&
        m_sunShaftBufferSet.valid() &&
        aoFrameIndex < m_sunShaftImageViews.size() &&
        m_sunShaftImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        const uint32_t region = m_currentFrame;
        const VkDeviceSize cameraOffset = descriptorBufferBindingOffset(m_sunShaftDescriptorSetLayout, 0);
        const VkDeviceSize normalDepthOffset = descriptorBufferBindingOffset(m_sunShaftDescriptorSetLayout, 1);
        const VkDeviceSize shadowOffset = descriptorBufferBindingOffset(m_sunShaftDescriptorSetLayout, 2);
        const VkDeviceSize outputOffset = descriptorBufferBindingOffset(m_sunShaftDescriptorSetLayout, 3);
        writeDescriptorBufferUniform(
            m_sunShaftBufferSet, region, cameraOffset, cameraDeviceAddress, cameraBufferInfo.range);
        writeDescriptorBufferCombinedImageSampler(
            m_sunShaftBufferSet, region, normalDepthOffset, 0,
            normalDepthImageInfo.imageView, normalDepthImageInfo.sampler, normalDepthImageInfo.imageLayout);
        writeDescriptorBufferCombinedImageSampler(
            m_sunShaftBufferSet, region, shadowOffset, 0,
            shadowMapImageInfo.imageView, shadowMapImageInfo.sampler, shadowMapImageInfo.imageLayout);
        writeDescriptorBufferStorageImage(
            m_sunShaftBufferSet, region, outputOffset,
            m_sunShaftImageViews[aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }

    if (m_taaEnabled && m_taaBufferSet.valid() &&
        aoFrameIndex < m_hdrResolveImageViews.size() &&
        m_hdrResolveImageViews[aoFrameIndex] != VK_NULL_HANDLE &&
        m_taaImageViews[0] != VK_NULL_HANDLE && m_taaImageViews[1] != VK_NULL_HANDLE) {
        // TAA's small uniform lives in a fresh FrameArena slice each frame:
        // it changes every frame (prev matrices), and the arena is exactly the
        // per-frame upload path everything else uses for that.
        //
        // 256-byte alignment for the same reason the bone matrices use it --
        // this address feeds a shader binding directly, so it must satisfy
        // any device's minUniformBufferOffsetAlignment, not the struct's.
        TaaUniformData uniformData{};
        std::memcpy(uniformData.invView, m_taaInvViewColumnMajor.m, sizeof(uniformData.invView));
        std::memcpy(
            uniformData.prevViewProj, m_taaPrevViewProjColumnMajor.m,
            sizeof(uniformData.prevViewProj));
        // History weight 0.88: high enough to converge shimmer in ~8 frames,
        // low enough that the clamp can pull a stale region back quickly.
        // ODAI_UPSCALE_HISTORY overrides it. 0 isolates the spatial
        // reconstruction with no temporal accumulation at all, which is the
        // control that separates "the filter is wrong" from "the accumulation is
        // wrong" -- they look identical in a finished frame.
        static const float s_historyWeight = []() {
            const char* env = std::getenv("ODAI_UPSCALE_HISTORY");
            if (env == nullptr) {
                return 0.88f;
            }
            return std::clamp(static_cast<float>(std::atof(env)), 0.0f, 0.99f);
        }();
        uniformData.params[0] = s_historyWeight;
        uniformData.params[1] = (m_taaHistoryValid && m_taaPrevViewProjValid) ? 1.0f : 0.0f;
        // THIS frame's jitter, in input PIXELS, plus the input extent. The
        // upscaler needs the jitter in pixels rather than NDC because it works
        // on the input sample grid -- it has to know where each low-res texel
        // actually landed to weight it. m_taaJitterNdc is NDC (2 units across
        // the extent), so the conversion is the inverse of the one that
        // produced it.
        const float inputWidth = static_cast<float>(std::max(1u, m_renderExtent.width));
        const float inputHeight = static_cast<float>(std::max(1u, m_renderExtent.height));
        uniformData.jitterAndInputExtent[0] = (m_taaJitterNdc[0] * inputWidth) * 0.5f;
        uniformData.jitterAndInputExtent[1] = (m_taaJitterNdc[1] * inputHeight) * 0.5f;
        uniformData.jitterAndInputExtent[2] = inputWidth;
        uniformData.jitterAndInputExtent[3] = inputHeight;
        // ODAI_UPSCALE_CLAMP / ODAI_UPSCALE_BLEND, for sweeping the two values
        // that trade sharpness against ghosting. Defaults are the measured ones.
        static const float s_clampStatic = []() {
            const char* env = std::getenv("ODAI_UPSCALE_CLAMP");
            return (env != nullptr) ? static_cast<float>(std::atof(env)) : 4.0f;
        }();
        static const float s_maxBlend = []() {
            const char* env = std::getenv("ODAI_UPSCALE_BLEND");
            return (env != nullptr) ? static_cast<float>(std::atof(env)) : 0.5f;
        }();
        uniformData.upscaleTuning[0] = s_clampStatic;
        uniformData.upscaleTuning[1] = 1.25f;
        uniformData.upscaleTuning[2] = s_maxBlend;
        uniformData.upscaleTuning[3] = 0.0f;
        // LAST frame's jitter, which the shader takes back out of the
        // reprojected UV. prevViewProj is the matrix that frame actually
        // rendered with, so it carries that frame's jitter -- but the history
        // TEXTURE is the resolved image on the fixed output grid, which carries
        // none. Sampling it at a UV that still has the jitter in it lands
        // ~half a pixel off, every frame, in a direction that changes every
        // frame; the bilinear resample that results compounds into a blur that
        // looks exactly like a soft mip and gets worse the longer the camera
        // holds still. Measured before this correction: jitter made a static
        // 0.6-scale frame visibly softer than no jitter at all, which is the
        // opposite of what jitter is for.
        uniformData.params[2] = m_taaPrevJitterNdc[0];
        uniformData.params[3] = m_taaPrevJitterNdc[1];
        const std::optional<FrameArenaSlice> uniformSlice = m_frameArena.allocateUpload(
            sizeof(TaaUniformData), 256u, FrameArenaUploadKind::Unknown);
        if (uniformSlice.has_value() && uniformSlice->mapped != nullptr) {
            std::memcpy(uniformSlice->mapped, &uniformData, sizeof(uniformData));
            const VkDeviceAddress uniformAddress =
                m_bufferAllocator.getDeviceAddress(uniformSlice->buffer) + uniformSlice->offset;
            const uint32_t region = m_currentFrame;
            // The pass writes into (historyIndex ^ 1) and reads historyIndex;
            // recordTaaPass flips the index after recording, so these must
            // agree with the pre-flip value.
            const std::uint32_t currentImage = m_taaHistoryIndex ^ 1u;
            const std::uint32_t historyImage = m_taaHistoryIndex;
            writeDescriptorBufferUniform(
                m_taaBufferSet, region,
                descriptorBufferBindingOffset(m_taaDescriptorSetLayout, 0),
                cameraDeviceAddress, cameraBufferInfo.range);
            writeDescriptorBufferCombinedImageSampler(
                m_taaBufferSet, region,
                descriptorBufferBindingOffset(m_taaDescriptorSetLayout, 1), 0,
                m_hdrResolveImageViews[aoFrameIndex], m_ssaoSampler,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writeDescriptorBufferCombinedImageSampler(
                m_taaBufferSet, region,
                descriptorBufferBindingOffset(m_taaDescriptorSetLayout, 2), 0,
                m_taaImageViews[historyImage], m_ssaoSampler,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writeDescriptorBufferCombinedImageSampler(
                m_taaBufferSet, region,
                descriptorBufferBindingOffset(m_taaDescriptorSetLayout, 3), 0,
                normalDepthImageInfo.imageView, normalDepthImageInfo.sampler,
                normalDepthImageInfo.imageLayout);
            writeDescriptorBufferStorageImage(
                m_taaBufferSet, region,
                descriptorBufferBindingOffset(m_taaDescriptorSetLayout, 4),
                m_taaImageViews[currentImage], VK_IMAGE_LAYOUT_GENERAL);
            writeDescriptorBufferUniform(
                m_taaBufferSet, region,
                descriptorBufferBindingOffset(m_taaDescriptorSetLayout, 5),
                uniformAddress, sizeof(TaaUniformData));
            // Binding 6: motion vectors. Written whether or not the active
            // pipeline reads them -- a descriptor the layout declares must be
            // valid even for a shader that ignores it. Falls back to the
            // normal-depth view when the velocity target does not exist, so the
            // binding is never null.
            const bool hasVelocity = aoFrameIndex < m_velocityImageViews.size() &&
                m_velocityImageViews[aoFrameIndex] != VK_NULL_HANDLE &&
                m_velocityImageInitialized[aoFrameIndex];
            writeDescriptorBufferCombinedImageSampler(
                m_taaBufferSet, region,
                descriptorBufferBindingOffset(m_taaDescriptorSetLayout, 6), 0,
                hasVelocity ? m_velocityImageViews[aoFrameIndex] : normalDepthImageInfo.imageView,
                m_ssaoSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
    }

    if (m_waterReflectionTemporalEnabled &&
        m_waterReflectionResolveBufferSet.valid() &&
        aoFrameIndex < m_waterReflectionImageViews.size() &&
        aoFrameIndex < m_waterReflectionDepthImageViews.size() &&
        m_waterReflectionImageViews[aoFrameIndex] != VK_NULL_HANDLE &&
        m_waterReflectionDepthImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        WaterReflectionResolveUniform uniformData{};
        std::memcpy(
            uniformData.invViewProj,
            m_waterReflectionInvViewProjColumnMajor.m,
            sizeof(uniformData.invViewProj));
        std::memcpy(
            uniformData.invView,
            m_waterReflectionInvViewColumnMajor.m,
            sizeof(uniformData.invView));
        std::memcpy(
            uniformData.view,
            m_waterReflectionViewColumnMajor.m,
            sizeof(uniformData.view));
        std::memcpy(
            uniformData.prevView,
            m_waterReflectionPrevViewColumnMajor.m,
            sizeof(uniformData.prevView));
        std::memcpy(
            uniformData.prevViewProj,
            m_waterReflectionPrevViewProjColumnMajor.m,
            sizeof(uniformData.prevViewProj));
        uniformData.temporalParams[0] = 0.88f;
        uniformData.temporalParams[1] = 0.80f;
        uniformData.temporalParams[2] =
            (m_waterReflectionHistoryValid &&
             m_waterReflectionPrevMatricesValid) ? 1.0f : 0.0f;
        uniformData.temporalParams[3] = m_waterReflectionPlaneHeight;
        uniformData.jitterParams[0] = m_taaJitterNdc[0];
        uniformData.jitterParams[1] = m_taaJitterNdc[1];
        uniformData.jitterParams[2] = m_taaPrevJitterNdc[0];
        uniformData.jitterParams[3] = m_taaPrevJitterNdc[1];
        uniformData.extentParams[0] =
            static_cast<float>(m_waterReflectionExtent.width);
        uniformData.extentParams[1] =
            static_cast<float>(m_waterReflectionExtent.height);
        uniformData.extentParams[2] = static_cast<float>(m_renderExtent.width);
        uniformData.extentParams[3] = static_cast<float>(m_renderExtent.height);
        uniformData.projectionParams[0] = m_waterReflectionProjection[0];
        uniformData.projectionParams[1] = m_waterReflectionProjection[1];

        const std::optional<FrameArenaSlice> uniformSlice =
            m_frameArena.allocateUpload(
                sizeof(WaterReflectionResolveUniform), 256u,
                FrameArenaUploadKind::Unknown);
        if (uniformSlice.has_value() && uniformSlice->mapped != nullptr) {
            std::memcpy(
                uniformSlice->mapped, &uniformData, sizeof(uniformData));
            const VkDeviceAddress uniformAddress =
                m_bufferAllocator.getDeviceAddress(uniformSlice->buffer) +
                uniformSlice->offset;
            const std::uint32_t region = m_currentFrame;
            const auto offset = [&](std::uint32_t binding) {
                return descriptorBufferBindingOffset(
                    m_waterReflectionResolveDescriptorSetLayout, binding);
            };
            const std::uint32_t current =
                m_waterReflectionHistoryIndex ^ 1u;
            const std::uint32_t history = m_waterReflectionHistoryIndex;
            writeDescriptorBufferCombinedImageSampler(
                m_waterReflectionResolveBufferSet, region, offset(0), 0u,
                m_waterReflectionImageViews[aoFrameIndex],
                m_hdrResolveSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writeDescriptorBufferCombinedImageSampler(
                m_waterReflectionResolveBufferSet, region, offset(1), 0u,
                m_waterReflectionDepthImageViews[aoFrameIndex],
                m_normalDepthSampler,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
            writeDescriptorBufferCombinedImageSampler(
                m_waterReflectionResolveBufferSet, region, offset(2), 0u,
                m_waterReflectionHistoryImageViews[history],
                m_hdrResolveSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writeDescriptorBufferCombinedImageSampler(
                m_waterReflectionResolveBufferSet, region, offset(3), 0u,
                m_waterReflectionHistoryDepthImageViews[history],
                m_normalDepthSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            writeDescriptorBufferStorageImage(
                m_waterReflectionResolveBufferSet, region, offset(4),
                m_waterReflectionHistoryImageViews[current],
                VK_IMAGE_LAYOUT_GENERAL);
            writeDescriptorBufferStorageImage(
                m_waterReflectionResolveBufferSet, region, offset(5),
                m_waterReflectionHistoryDepthImageViews[current],
                VK_IMAGE_LAYOUT_GENERAL);
            writeDescriptorBufferUniform(
                m_waterReflectionResolveBufferSet, region, offset(6),
                uniformAddress, sizeof(WaterReflectionResolveUniform));
        }
    }

    if (m_ssaoBufferSet.valid() &&
        aoFrameIndex < m_ssaoRawImageViews.size() &&
        m_ssaoRawImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        const uint32_t region = m_currentFrame;
        const VkDeviceSize cameraOffset = descriptorBufferBindingOffset(m_ssaoDescriptorSetLayout, 0);
        const VkDeviceSize normalDepthOffset = descriptorBufferBindingOffset(m_ssaoDescriptorSetLayout, 1);
        const VkDeviceSize outputOffset = descriptorBufferBindingOffset(m_ssaoDescriptorSetLayout, 2);
        writeDescriptorBufferUniform(
            m_ssaoBufferSet, region, cameraOffset, cameraDeviceAddress, cameraBufferInfo.range);
        writeDescriptorBufferCombinedImageSampler(
            m_ssaoBufferSet, region, normalDepthOffset, 0,
            normalDepthImageInfo.imageView, normalDepthImageInfo.sampler, normalDepthImageInfo.imageLayout);
        writeDescriptorBufferStorageImage(
            m_ssaoBufferSet, region, outputOffset,
            m_ssaoRawImageViews[aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }

    // XeGTAO shares the same per-frame inputs as the other estimators; its sets
    // are separate only because its bindings are.
    writeXeGtaoDescriptors(
        m_currentFrame, aoFrameIndex, cameraDeviceAddress, cameraBufferInfo.range,
        normalDepthImageInfo);

    if (m_ssaoBlurBufferSet.valid() &&
        aoFrameIndex < m_ssaoBlurImageViews.size() &&
        m_ssaoBlurImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        const uint32_t region = m_currentFrame;
        const VkDeviceSize normalDepthOffset = descriptorBufferBindingOffset(m_ssaoBlurDescriptorSetLayout, 0);
        const VkDeviceSize ssaoRawOffset = descriptorBufferBindingOffset(m_ssaoBlurDescriptorSetLayout, 1);
        const VkDeviceSize outputOffset = descriptorBufferBindingOffset(m_ssaoBlurDescriptorSetLayout, 2);
        writeDescriptorBufferCombinedImageSampler(
            m_ssaoBlurBufferSet, region, normalDepthOffset, 0,
            normalDepthImageInfo.imageView, normalDepthImageInfo.sampler, normalDepthImageInfo.imageLayout);
        writeDescriptorBufferCombinedImageSampler(
            m_ssaoBlurBufferSet, region, ssaoRawOffset, 0,
            ssaoRawImageInfo.imageView, ssaoRawImageInfo.sampler, ssaoRawImageInfo.imageLayout);
        writeDescriptorBufferStorageImage(
            m_ssaoBlurBufferSet, region, outputOffset,
            m_ssaoBlurImageViews[aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }

    const std::size_t bindlessTextureCount =
        kBindlessTextureStaticCount + std::min<std::size_t>(
            m_importedTextureResources.size(),
            (m_bindlessTextureCapacity > kBindlessTextureStaticCount)
                ? static_cast<std::size_t>(m_bindlessTextureCapacity - kBindlessTextureStaticCount)
                : 0u);
    if (m_bindlessBufferSet.valid() && m_bindlessTextureCapacity >= bindlessTextureCount) {
        std::vector<VkDescriptorImageInfo> bindlessImageInfos(bindlessTextureCount);
        bindlessImageInfos[kBindlessTextureIndexDiffuse] = diffuseTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexHdrResolved] = hdrSceneImageInfo;
        bindlessImageInfos[kBindlessTextureIndexShadowAtlas] = shadowMapImageInfo;
        bindlessImageInfos[kBindlessTextureIndexNormalDepth] = normalDepthImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoBlur] = ssaoBlurImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoRaw] = ssaoRawImageInfo;
        bindlessImageInfos[kBindlessTextureIndexPlantDiffuse] = plantDiffuseTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSkyDaylight] = morrowindSkyTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexWaterNormal] = waterNormalTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexTerrainDetail] = terrainDetailTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexFogMap] = fogMapTextureImageInfo;
        for (std::size_t textureIndex = 0; textureIndex < m_importedTextureResources.size(); ++textureIndex) {
            const std::size_t bindlessIndex = kBindlessTextureStaticCount + textureIndex;
            if (bindlessIndex >= bindlessImageInfos.size()) {
                break;
            }
            const ImportedTextureResource& texture = m_importedTextureResources[textureIndex];
            if (texture.imageView == VK_NULL_HANDLE || m_importedTextureSampler == VK_NULL_HANDLE) {
                continue;
            }
            bindlessImageInfos[bindlessIndex].sampler = m_importedTextureSampler;
            bindlessImageInfos[bindlessIndex].imageView = texture.imageView;
            bindlessImageInfos[bindlessIndex].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        // Bindless array binding 0: one combined-image-sampler descriptor per array
        // element (single region shared across frames). Empty slots stay unwritten
        // (partially bound); the shader never indexes them.
        const VkDeviceSize bindlessBindingOffset = descriptorBufferBindingOffset(m_bindlessDescriptorSetLayout, 0);
        for (std::size_t index = 0; index < bindlessImageInfos.size(); ++index) {
            const VkDescriptorImageInfo& info = bindlessImageInfos[index];
            if (info.imageView == VK_NULL_HANDLE || info.sampler == VK_NULL_HANDLE) {
                continue;
            }
            writeDescriptorBufferCombinedImageSamplerArray(
                m_bindlessBufferSet, 0u, bindlessBindingOffset,
                static_cast<uint32_t>(index), m_bindlessTextureCapacity,
                info.imageView, info.sampler, info.imageLayout);
        }
    }
}

} // namespace odai::render
