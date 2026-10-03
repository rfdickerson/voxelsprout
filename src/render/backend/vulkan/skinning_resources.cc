#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include "core/log.h"
#include "math/math.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// GPU skeletal skinning (Dragon Age: Origins touchstone; see
// docs/ROADMAP.md's Party RPG / Narrative section, and skinning.comp.slang
// for the compute shader this drives). Wired into the actual frame
// (createSkinningComputeResources/destroySkinningComputeResources from
// init.cc, recordSkinningPass from frame_run.cc before the shadow/prepass/
// main passes, uploadSkinnedActorPoseForFrame right after
// m_frameArena.beginFrame -- see those files); the bone-matrix upload
// transposes to match the camera-MVP path's established row/column-major
// convention. Supports up to kMaxSkinnedInstances (see renderer_types.h)
// independent instance slots -- a small party, not a mass-battle crowd (see
// docs/ROADMAP.md's explicit out-of-scope note). Still unverified against a
// real Vulkan build (this sandbox has neither vcpkg nor a GPU) -- see the
// Windows CI job for that signal.
namespace odai::render {

#include "render/renderer_shared.h"

namespace {

constexpr const char* kSkinningShaderPath = "../src/render/shaders/skinning.comp.slang.spv";

struct SkinningPushConstants {
    std::uint32_t vertexCount;
    std::uint32_t boneCount;
    std::uint32_t morphTargetCount;
    std::uint32_t pad0;
};

// GPU-side rest-pose vertex layout (mirrors skinning.comp.slang's
// SkinnedVertexIn): widens the CPU-side ImportedSkinnedMeshVertex's compact
// uint16 bone indices to uint32, the same "CPU import format differs from
// the GPU buffer format" split ImportedScenePackedVertex -> ImportedMeshVertex
// already uses (chunk_upload.cc).
struct GpuSkinnedVertexIn {
    float position[3];
    float normal[3];
    float color[3];
    float uv[2];
    std::uint32_t textureIndex;
    std::uint32_t flags;
    std::uint32_t boneIndices[4];
    float boneWeights[4];
    std::uint32_t normalTextureIndex;
    float modelNormalBasis[9];
    std::uint32_t skinSoftTexture, skinSpecularTexture;
    float skinSpecularStrength, skinGlossiness, skinSoftRolloff;
    std::uint32_t skinSpecularColor;
};
static_assert(sizeof(GpuSkinnedVertexIn) == 148);
static_assert(sizeof(ImportedSkinnedMeshTemplate::MorphDelta) == 16);

}  // namespace

bool RendererBackend::createSkinningComputeResources() {
    // Shared across every instance slot: one shader, one binding layout, one
    // pipeline. Per-slot buffers/descriptor-buffer-sets are created lazily in
    // uploadSkinnedMeshTemplate instead.
    if (m_skinningDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding restPoseBinding{};
        restPoseBinding.binding = 0;
        restPoseBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        restPoseBinding.descriptorCount = 1;
        restPoseBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding boneMatrixBinding{};
        boneMatrixBinding.binding = 1;
        boneMatrixBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        boneMatrixBinding.descriptorCount = 1;
        boneMatrixBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding outputBinding{};
        outputBinding.binding = 2;
        outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        outputBinding.descriptorCount = 1;
        outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const auto storageBinding = [](std::uint32_t binding) {
            VkDescriptorSetLayoutBinding result{};
            result.binding = binding;
            result.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            result.descriptorCount = 1;
            result.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            return result;
        };
        const std::array<VkDescriptorSetLayoutBinding, 6> bindings = {
            restPoseBinding, boneMatrixBinding, outputBinding,
            storageBinding(3), storageBinding(4), storageBinding(5)
        };

        if (!createDescriptorSetLayout(
                bindings,
                m_skinningDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(skinning)",
                "renderer.descriptorSetLayout.skinning",
                nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
            )) {
            destroySkinningComputeResources();
            return false;
        }
    }

    if (m_skinningPipeline != VK_NULL_HANDLE) {
        return true;  // Already fully created.
    }

    VkShaderModule skinningShaderModule = VK_NULL_HANDLE;
    if (!createShaderModuleFromFile(m_device, kSkinningShaderPath, "skinning.comp", skinningShaderModule)) {
        destroySkinningComputeResources();
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(SkinningPushConstants);
    const std::array<VkPushConstantRange, 1> pushConstantRanges = {pushConstantRange};

    if (!createComputePipelineLayout(
            m_skinningDescriptorSetLayout,
            pushConstantRanges,
            m_skinningPipelineLayout,
            "vkCreatePipelineLayout(skinning)",
            "renderer.pipelineLayout.skinning"
        )) {
        vkDestroyShaderModule(m_device, skinningShaderModule, nullptr);
        destroySkinningComputeResources();
        return false;
    }
    const bool pipelineCreated = createComputePipeline(
        m_skinningPipelineLayout,
        skinningShaderModule,
        m_skinningPipeline,
        "vkCreateComputePipelines(skinning)",
        "pipeline.skinning",
        VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
    );
    vkDestroyShaderModule(m_device, skinningShaderModule, nullptr);
    if (!pipelineCreated) {
        destroySkinningComputeResources();
        return false;
    }
    // Pose compute is optional: a missing shader retains CPU palettes.
    (void)createPoseComputeResources();
    return true;
}

void RendererBackend::destroySkinningComputeResources() {
    destroyPoseComputeResources();
    if (m_skinningPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_skinningPipeline, nullptr);
        m_skinningPipeline = VK_NULL_HANDLE;
    }
    if (m_skinningPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_skinningPipelineLayout, nullptr);
        m_skinningPipelineLayout = VK_NULL_HANDLE;
    }
    // Matches the original single-slot teardown order: descriptor-buffer-sets
    // (which only reference the layout at write time, not at destroy time)
    // before the layout itself.
    for (SkinnedInstanceSlot& slot : m_skinningInstances) {
        destroyDescriptorBufferSet(slot.bufferSet);
        destroyDescriptorBufferSet(slot.velocityBufferSet);
        m_bufferAllocator.destroyBuffer(slot.restPoseVertexBufferHandle);
        slot.restPoseVertexBufferHandle = kInvalidBufferHandle;
        m_bufferAllocator.destroyBuffer(slot.morphOffsetBufferHandle);
        slot.morphOffsetBufferHandle = kInvalidBufferHandle;
        m_bufferAllocator.destroyBuffer(slot.morphDeltaBufferHandle);
        slot.morphDeltaBufferHandle = kInvalidBufferHandle;
        m_bufferAllocator.destroyBuffer(slot.indexBufferHandle);
        slot.indexBufferHandle = kInvalidBufferHandle;
        m_bufferAllocator.destroyBuffer(slot.outputVertexBufferHandle);
        slot.outputVertexBufferHandle = kInvalidBufferHandle;
        slot.vertexCount = 0;
        slot.boneCount = 0;
        slot.morphTargetCount = 0;
        slot.visible = true;
        slot.meshDraws.clear();
        slot.pendingBoneMatrices.clear();
        slot.pendingMorphWeights.clear();
        slot.previousMorphWeights.clear();
        slot.poseHistoryValid = false;
        slot.previousBoneMatrices.clear();
        slot.currentBoneAddress = 0;
        slot.previousBoneAddress = 0;
        slot.boneBufferBytes = 0;
        slot.currentMorphAddress = 0;
        slot.previousMorphAddress = 0;
        slot.morphBufferBytes = 0;
        for (const std::uint32_t textureSlot : slot.textureSlots) {
            releaseImportedTexture(textureSlot);
        }
        slot.textureSlots.clear();
    }
    if (m_skinningDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_skinningDescriptorSetLayout, nullptr);
        m_skinningDescriptorSetLayout = VK_NULL_HANDLE;
    }
    m_skinningActiveInstanceCount = 0;
    m_skinningMeshDraws.clear();
}

bool RendererBackend::uploadSkinnedMeshTemplate(
    std::uint32_t instanceIndex, const ImportedSkinnedMeshTemplate& meshTemplate
) {
    if (instanceIndex >= kMaxSkinnedInstances) {
        VOX_LOGW("render") << "skinned mesh template upload skipped: instanceIndex "
                            << instanceIndex << " >= kMaxSkinnedInstances";
        return false;
    }
    if (meshTemplate.vertices.empty() || meshTemplate.indices.empty() || meshTemplate.boneCount == 0) {
        VOX_LOGW("render") << "skinned mesh template upload skipped: empty geometry or zero bones";
        return false;
    }
    const bool hasMorphs = meshTemplate.morphTargetCount != 0u;
    if (hasMorphs &&
        (meshTemplate.morphVertexOffsets.size() != meshTemplate.vertices.size() + 1u ||
         meshTemplate.morphVertexOffsets.back() != meshTemplate.morphDeltas.size())) {
        VOX_LOGW("render") << "skinned mesh template upload skipped: malformed morph CSR data";
        return false;
    }
    if (!hasMorphs && (!meshTemplate.morphVertexOffsets.empty() ||
                       !meshTemplate.morphDeltas.empty())) {
        VOX_LOGW("render") << "skinned mesh template upload skipped: morph data has zero targets";
        return false;
    }
    if (hasMorphs) {
        for (std::size_t vertex = 0; vertex < meshTemplate.vertices.size(); ++vertex) {
            if (meshTemplate.morphVertexOffsets[vertex] >
                meshTemplate.morphVertexOffsets[vertex + 1u]) {
                VOX_LOGW("render") << "skinned mesh template upload skipped: unsorted morph offsets";
                return false;
            }
        }
        for (const auto& delta : meshTemplate.morphDeltas) {
            if (delta.targetIndex >= meshTemplate.morphTargetCount ||
                !std::isfinite(delta.position[0]) || !std::isfinite(delta.position[1]) ||
                !std::isfinite(delta.position[2])) {
                VOX_LOGW("render") << "skinned mesh template upload skipped: invalid morph delta";
                return false;
            }
        }
    }
    // Creates the shared pipeline/descriptor-set-layout on first use across
    // any slot; a no-op if already created.
    if (!createSkinningComputeResources()) {
        return false;
    }

    SkinnedInstanceSlot& slot = m_skinningInstances[instanceIndex];

    std::vector<GpuSkinnedVertexIn> gpuVertices(meshTemplate.vertices.size());
    for (std::size_t i = 0; i < meshTemplate.vertices.size(); ++i) {
        const ImportedSkinnedMeshVertex& src = meshTemplate.vertices[i];
        GpuSkinnedVertexIn& dst = gpuVertices[i];
        dst.position[0] = src.position[0];
        dst.position[1] = src.position[1];
        dst.position[2] = src.position[2];
        dst.normal[0] = src.normal[0];
        dst.normal[1] = src.normal[1];
        dst.normal[2] = src.normal[2];
        dst.color[0] = src.color[0];
        dst.color[1] = src.color[1];
        dst.color[2] = src.color[2];
        dst.uv[0] = src.uv[0];
        dst.uv[1] = src.uv[1];
        dst.textureIndex = src.textureIndex;
        dst.flags = src.flags;
        dst.normalTextureIndex = src.normalTextureIndex;
        std::copy_n(src.modelNormalBasis, 9, dst.modelNormalBasis);
        dst.skinSoftTexture = src.skinSoftTexture;
        dst.skinSpecularTexture = src.skinSpecularTexture;
        dst.skinSpecularStrength = src.skinSpecularStrength;
        dst.skinGlossiness = src.skinGlossiness;
        dst.skinSoftRolloff = src.skinSoftRolloff;
        dst.skinSpecularColor = src.skinSpecularColor;
        for (int b = 0; b < 4; ++b) {
            dst.boneIndices[b] = static_cast<std::uint32_t>(src.boneIndices[b]);
            dst.boneWeights[b] = src.boneWeights[b];
        }
    }

    // Same staged device-local upload shape as uploadImportedSceneInternal's
    // local uploadDeviceLocalBuffer lambda (chunk_upload.cc) -- duplicated
    // rather than shared, matching how every other upload call site in this
    // backend already does its own local copy of this pattern. Retain the last
    // transfer value so failed multi-buffer uploads and replaced templates can
    // retire buffers only after every earlier queue use has completed.
    std::uint64_t latestUploadTimelineValue = 0u;
    auto uploadDeviceLocalBuffer = [&](
                                       const void* sourceData,
                                       VkDeviceSize bufferSize,
                                       VkBufferUsageFlags usage,
                                       const char* debugLabel,
                                       BufferHandle& outHandle
                                   ) -> bool {
        outHandle = kInvalidBufferHandle;
        if (sourceData == nullptr || bufferSize == 0u) {
            return false;
        }

        BufferCreateDesc stagingCreateDesc{};
        stagingCreateDesc.size = bufferSize;
        stagingCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingCreateDesc.memoryProperties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        stagingCreateDesc.initialData = sourceData;
        const BufferHandle stagingHandle = m_bufferAllocator.createBuffer(stagingCreateDesc);
        if (stagingHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << debugLabel << " staging buffer allocation failed";
            return false;
        }

        BufferCreateDesc deviceCreateDesc{};
        deviceCreateDesc.size = bufferSize;
        deviceCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage;
        deviceCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        outHandle = m_bufferAllocator.createBuffer(deviceCreateDesc);
        if (outHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << debugLabel << " device-local buffer allocation failed";
            m_bufferAllocator.destroyBuffer(stagingHandle);
            return false;
        }

        bool uploadFailed = false;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo commandPoolCreateInfo{};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        commandPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
        VkResult result = vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &commandPool);
        if (result != VK_SUCCESS) {
            logVkFailure("vkCreateCommandPool(skinnedMeshUpload)", result);
            uploadFailed = true;
        }

        if (!uploadFailed) {
            VkCommandBufferAllocateInfo allocateInfo{};
            allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocateInfo.commandPool = commandPool;
            allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocateInfo.commandBufferCount = 1;
            result = vkAllocateCommandBuffers(m_device, &allocateInfo, &commandBuffer);
            if (result != VK_SUCCESS) {
                logVkFailure("vkAllocateCommandBuffers(skinnedMeshUpload)", result);
                uploadFailed = true;
            }
        }

        if (!uploadFailed) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
            if (result != VK_SUCCESS) {
                logVkFailure("vkBeginCommandBuffer(skinnedMeshUpload)", result);
                uploadFailed = true;
            }
        }

        if (!uploadFailed) {
            VkBufferCopy copyRegion{};
            copyRegion.size = bufferSize;
            vkCmdCopyBuffer(
                commandBuffer,
                m_bufferAllocator.getBuffer(stagingHandle),
                m_bufferAllocator.getBuffer(outHandle),
                1,
                &copyRegion);
            result = vkEndCommandBuffer(commandBuffer);
            if (result != VK_SUCCESS) {
                logVkFailure("vkEndCommandBuffer(skinnedMeshUpload)", result);
                uploadFailed = true;
            }
        }

        std::uint64_t uploadTimelineValue = 0u;
        if (!uploadFailed) {
            uploadTimelineValue = m_nextTimelineValue++;
            VkSemaphoreSubmitInfo signalInfo{};
            signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signalInfo.semaphore = m_renderTimelineSemaphore;
            signalInfo.value = uploadTimelineValue;
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkCommandBufferSubmitInfo commandBufferInfo{};
            commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            commandBufferInfo.commandBuffer = commandBuffer;
            VkSubmitInfo2 submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            submitInfo.commandBufferInfoCount = 1u;
            submitInfo.pCommandBufferInfos = &commandBufferInfo;
            submitInfo.signalSemaphoreInfoCount = 1u;
            submitInfo.pSignalSemaphoreInfos = &signalInfo;
            result = vkQueueSubmit2(m_graphicsQueue, 1u, &submitInfo, VK_NULL_HANDLE);
            if (result != VK_SUCCESS) {
                logVkFailure("vkQueueSubmit2(skinnedMeshUpload)", result);
                uploadTimelineValue = 0u;
                uploadFailed = true;
            } else {
                latestUploadTimelineValue = uploadTimelineValue;
                m_pendingTransferTimelineValue =
                    std::max(m_pendingTransferTimelineValue, uploadTimelineValue);
            }
        }

        scheduleCommandPoolRelease(commandPool, uploadTimelineValue);
        scheduleBufferRelease(stagingHandle, uploadTimelineValue);
        if (uploadFailed) {
            m_bufferAllocator.destroyBuffer(outHandle);
            outHandle = kInvalidBufferHandle;
            return false;
        }
        return true;
    };

    if (!slot.bufferSet.valid()) {
        if (!createDescriptorBufferSet(
                m_skinningDescriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.skinning",
                slot.bufferSet
            )) {
            return false;
        }
    }
    // Velocity set. Not fatal if it fails: the actor still renders, it just has
    // no motion vector and its pixels fall back to depth reprojection.
    if (!slot.velocityBufferSet.valid() && m_skinnedVelocityDescriptorSetLayout != VK_NULL_HANDLE) {
        if (!createDescriptorBufferSet(
                m_skinnedVelocityDescriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.skinnedVelocity",
                slot.velocityBufferSet
            )) {
            VOX_LOGW("render") << "skinned velocity descriptor set unavailable for instance "
                               << instanceIndex;
        }
    }

    BufferHandle newRestPoseHandle = kInvalidBufferHandle;
    if (!uploadDeviceLocalBuffer(
            gpuVertices.data(),
            static_cast<VkDeviceSize>(gpuVertices.size() * sizeof(GpuSkinnedVertexIn)),
            // SHADER_DEVICE_ADDRESS is required, not optional: this buffer is
            // reached through a descriptor BUFFER, and writeDescriptorBufferStorage
            // below takes its device address. Without the usage bit
            // vkGetBufferDeviceAddress is invalid -- validation reports
            // VUID-VkBufferDeviceAddressInfo-buffer-02601 and the address that
            // comes back does not point at the buffer, so the compute pass reads
            // garbage and the whole frame renders black.
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            "skinned mesh rest-pose vertex",
            newRestPoseHandle)) {
        return false;
    }

    // Morph topology and sparse deltas are immutable and device-local. Empty
    // templates still bind one harmless element at each binding because Vulkan
    // descriptors must be valid even though morphTargetCount makes the shader
    // skip every read.
    std::vector<std::uint32_t> zeroMorphOffsets(gpuVertices.size() + 1u, 0u);
    const ImportedSkinnedMeshTemplate::MorphDelta zeroMorphDelta{};
    const std::span<const std::uint32_t> morphOffsets = hasMorphs
        ? meshTemplate.morphVertexOffsets
        : std::span<const std::uint32_t>(zeroMorphOffsets);
    const std::span<const ImportedSkinnedMeshTemplate::MorphDelta> morphDeltas =
        hasMorphs && !meshTemplate.morphDeltas.empty()
        ? meshTemplate.morphDeltas
        : std::span<const ImportedSkinnedMeshTemplate::MorphDelta>(&zeroMorphDelta, 1u);
    BufferHandle newMorphOffsetHandle = kInvalidBufferHandle;
    if (!uploadDeviceLocalBuffer(
            morphOffsets.data(),
            static_cast<VkDeviceSize>(morphOffsets.size_bytes()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            "skinned mesh morph offsets", newMorphOffsetHandle)) {
        scheduleBufferRelease(newRestPoseHandle, latestUploadTimelineValue);
        return false;
    }
    BufferHandle newMorphDeltaHandle = kInvalidBufferHandle;
    if (!uploadDeviceLocalBuffer(
            morphDeltas.data(),
            static_cast<VkDeviceSize>(morphDeltas.size_bytes()),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            "skinned mesh morph deltas", newMorphDeltaHandle)) {
        scheduleBufferRelease(newRestPoseHandle, latestUploadTimelineValue);
        scheduleBufferRelease(newMorphOffsetHandle, latestUploadTimelineValue);
        return false;
    }

    BufferHandle newIndexHandle = kInvalidBufferHandle;
    if (!uploadDeviceLocalBuffer(
            meshTemplate.indices.data(),
            static_cast<VkDeviceSize>(meshTemplate.indices.size() * sizeof(std::uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            "skinned mesh index",
            newIndexHandle)) {
        scheduleBufferRelease(newRestPoseHandle, latestUploadTimelineValue);
        scheduleBufferRelease(newMorphOffsetHandle, latestUploadTimelineValue);
        scheduleBufferRelease(newMorphDeltaHandle, latestUploadTimelineValue);
        return false;
    }

    // Persistent skinned output: rewritten every frame by recordSkinningPass and
    // read every frame as a plain ImportedMeshVertex vertex buffer.
    //
    // SEEDED with the rest pose rather than left uninitialized. The comment here
    // used to say there was "nothing to initialize it with yet", which is not
    // true -- the rest pose is right there, and it is exactly what this buffer
    // should contain before the first dispatch. Left uninitialized, any frame
    // that draws before a dispatch lands (the debug bypass, a slot whose pose
    // has not been set, the frame a template is uploaded on) feeds the vertex
    // stage whatever the allocator handed back. Garbage floats become NaN
    // positions, NaN reaches the auto-exposure histogram, and the tonemapper
    // takes the ENTIRE frame with it -- the world included. That failure looks
    // nothing like "the character is missing"; it looks like the renderer
    // broke.
    std::vector<ImportedMeshVertex> restOutputVertices(gpuVertices.size());
    for (std::size_t i = 0; i < gpuVertices.size(); ++i) {
        const ImportedSkinnedMeshVertex& src = meshTemplate.vertices[i];
        ImportedMeshVertex& dst = restOutputVertices[i];
        dst.position[0] = src.position[0];
        dst.position[1] = src.position[1];
        dst.position[2] = src.position[2];
        dst.packedNormal = odai::importer::packImportedVertexNormal(src.normal);
        // Opaque: ImportedSkinnedMeshVertex carries no authored alpha. The
        // channel exists to feather placed world geometry (see
        // ImportedSceneVertex::colorAlpha) and no skinned actor part uses it,
        // so the skinned vertex is not widened for it.
        dst.packedColor = odai::importer::packImportedVertexColor(src.color);
        dst.uv[0] = src.uv[0];
        dst.uv[1] = src.uv[1];
        dst.textureIndex = src.textureIndex;
        dst.flags = src.flags;
        dst.packedLayerTexture01 = 0xffff0000u | std::min(src.normalTextureIndex, 0xffffu);
        dst.terrainSurface01 = odai::importer::packImportedVertexNormal(src.modelNormalBasis);
        dst.terrainSurface23 = odai::importer::packImportedVertexNormal(src.modelNormalBasis + 3);
        dst.terrainSurface45 = odai::importer::packImportedVertexNormal(src.modelNormalBasis + 6);
        if ((src.flags & odai::importer::kImportedSceneMaterialFlagSkinnedModelNormals) != 0u) {
            dst.packedLayerTexture01 = std::min(src.normalTextureIndex, 0xffffu) | (std::min(src.skinSoftTexture, 0xffffu) << 16);
            dst.packedLayerTexture23 = 0xffff0000u | std::min(src.skinSpecularTexture, 0xffffu);
            dst.packedTerrainNormal01 = std::bit_cast<std::uint32_t>(src.skinSpecularStrength);
            dst.packedTerrainNormal23 = std::bit_cast<std::uint32_t>(src.skinGlossiness);
            dst.packedTerrainNormal45 = std::bit_cast<std::uint32_t>(src.skinSoftRolloff);
            dst.layerWeights = src.skinSpecularColor;
        }
    }
    BufferHandle newOutputHandle = kInvalidBufferHandle;
    if (!uploadDeviceLocalBuffer(
            restOutputVertices.data(),
            static_cast<VkDeviceSize>(restOutputVertices.size() * sizeof(ImportedMeshVertex)),
            // Storage (written by the compute pass through a descriptor buffer,
            // hence SHADER_DEVICE_ADDRESS -- see the rest-pose buffer above)
            // plus vertex (read by the main pass as plain geometry).
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            "skinned mesh output",
            newOutputHandle)) {
        VOX_LOGE("render") << "skinned mesh output buffer allocation failed";
        scheduleBufferRelease(newRestPoseHandle, latestUploadTimelineValue);
        scheduleBufferRelease(newMorphOffsetHandle, latestUploadTimelineValue);
        scheduleBufferRelease(newMorphDeltaHandle, latestUploadTimelineValue);
        scheduleBufferRelease(newIndexHandle, latestUploadTimelineValue);
        return false;
    }

    // Rest-pose (binding 0) and output (binding 2) buffers are stable across
    // frames -- write every region once here rather than every frame.
    for (std::uint32_t region = 0; region < slot.bufferSet.regionCount; ++region) {
        writeDescriptorBufferStorage(
            slot.bufferSet, region, descriptorBufferBindingOffset(m_skinningDescriptorSetLayout, 0),
            m_bufferAllocator.getDeviceAddress(newRestPoseHandle),
            static_cast<VkDeviceSize>(gpuVertices.size() * sizeof(GpuSkinnedVertexIn)));
        writeDescriptorBufferStorage(
            slot.bufferSet, region, descriptorBufferBindingOffset(m_skinningDescriptorSetLayout, 2),
            m_bufferAllocator.getDeviceAddress(newOutputHandle),
            static_cast<VkDeviceSize>(restOutputVertices.size() * sizeof(ImportedMeshVertex)));
        writeDescriptorBufferStorage(
            slot.bufferSet, region, descriptorBufferBindingOffset(m_skinningDescriptorSetLayout, 3),
            m_bufferAllocator.getDeviceAddress(newMorphOffsetHandle),
            static_cast<VkDeviceSize>(morphOffsets.size_bytes()));
        writeDescriptorBufferStorage(
            slot.bufferSet, region, descriptorBufferBindingOffset(m_skinningDescriptorSetLayout, 4),
            m_bufferAllocator.getDeviceAddress(newMorphDeltaHandle),
            static_cast<VkDeviceSize>(morphDeltas.size_bytes()));
    }
    if (slot.velocityBufferSet.valid() && m_skinnedVelocityDescriptorSetLayout != VK_NULL_HANDLE) {
        for (std::uint32_t region = 0; region < slot.velocityBufferSet.regionCount; ++region) {
            writeDescriptorBufferStorage(
                slot.velocityBufferSet, region,
                descriptorBufferBindingOffset(m_skinnedVelocityDescriptorSetLayout, 2),
                m_bufferAllocator.getDeviceAddress(newMorphOffsetHandle),
                static_cast<VkDeviceSize>(morphOffsets.size_bytes()));
            writeDescriptorBufferStorage(
                slot.velocityBufferSet, region,
                descriptorBufferBindingOffset(m_skinnedVelocityDescriptorSetLayout, 3),
                m_bufferAllocator.getDeviceAddress(newMorphDeltaHandle),
                static_cast<VkDeviceSize>(morphDeltas.size_bytes()));
        }
    }

    // The old template may still be referenced by an in-flight frame. The
    // upload submissions are ordered after that frame on the same graphics
    // queue, so their final timeline signal is also a safe retirement point.
    scheduleBufferRelease(slot.restPoseVertexBufferHandle, latestUploadTimelineValue);
    scheduleBufferRelease(slot.morphOffsetBufferHandle, latestUploadTimelineValue);
    scheduleBufferRelease(slot.morphDeltaBufferHandle, latestUploadTimelineValue);
    scheduleBufferRelease(slot.indexBufferHandle, latestUploadTimelineValue);
    scheduleBufferRelease(slot.outputVertexBufferHandle, latestUploadTimelineValue);

    slot.restPoseVertexBufferHandle = newRestPoseHandle;
    slot.morphOffsetBufferHandle = newMorphOffsetHandle;
    slot.morphDeltaBufferHandle = newMorphDeltaHandle;
    slot.indexBufferHandle = newIndexHandle;
    slot.outputVertexBufferHandle = newOutputHandle;
    slot.vertexCount = static_cast<std::uint32_t>(gpuVertices.size());
    slot.boneCount = meshTemplate.boneCount;
    slot.morphTargetCount = meshTemplate.morphTargetCount;
    slot.pendingMorphWeights.assign(slot.morphTargetCount, 0.0f);

    slot.meshDraws.clear();
    slot.meshDraws.reserve(meshTemplate.draws.size());
    for (const auto& draw : meshTemplate.draws) {
        ImportedMeshDraw meshDraw{};
        meshDraw.vertexBufferHandle = slot.outputVertexBufferHandle;
        meshDraw.indexBufferHandle = slot.indexBufferHandle;
        meshDraw.firstIndex = draw.firstIndex;
        meshDraw.indexCount = draw.indexCount;
        // Authored material state, carried through the same way the static and
        // actor paths do it: the threshold off the packed draw, blend and
        // two-sidedness off the draw's first vertex (per-vertex flags, but
        // uniform across a draw because a draw is one NIF shape). Without this
        // every skinned part alpha-tested at the default 0.5 and none was ever
        // two-sided.
        meshDraw.alphaThreshold = draw.alphaThreshold;
        if (draw.firstIndex < meshTemplate.indices.size()) {
            const std::uint32_t vertexIndex = meshTemplate.indices[draw.firstIndex];
            if (vertexIndex < meshTemplate.vertices.size()) {
                const std::uint32_t flags = meshTemplate.vertices[vertexIndex].flags;
                meshDraw.blended =
                    (flags & odai::importer::kImportedSceneMaterialFlagAlphaBlend) != 0u;
                meshDraw.twoSided =
                    (flags & odai::importer::kImportedSceneMaterialFlagTwoSided) != 0u;
            }
        }
        // Body parts carry their texture in each vertex, not in draw state.
        // When adjacent index ranges agree on the small amount of state that
        // really is per draw, joining them is therefore exact: the GPU sees
        // one indexed range whose vertices still select the same bindless
        // textures. Skyrim humanoids commonly arrive as five to seven parts;
        // leaving those splits intact multiplied them through main, depth and
        // every shadow cascade even though no material bind separated them.
        if (!slot.meshDraws.empty()) {
            ImportedMeshDraw& previous = slot.meshDraws.back();
            if (previous.firstIndex + previous.indexCount == meshDraw.firstIndex &&
                previous.alphaThreshold == meshDraw.alphaThreshold &&
                previous.blended == meshDraw.blended &&
                previous.twoSided == meshDraw.twoSided) {
                previous.indexCount += meshDraw.indexCount;
                continue;
            }
        }
        slot.meshDraws.push_back(meshDraw);
    }

    m_skinningActiveInstanceCount = std::max(m_skinningActiveInstanceCount, instanceIndex + 1u);

    // Re-flatten every visible instance slot's draws into one contiguous
    // vector so frame_pass_shadow.cc/frame_pass_prepass.cc/frame_pass_main.cc/
    // frame_run.cc keep consuming a single std::span<const ImportedMeshDraw>
    // with no changes.
    rebuildVisibleSkinningDraws();
    return true;
}

void RendererBackend::rebuildVisibleSkinningDraws() {
    m_skinningMeshDraws.clear();
    for (std::uint32_t i = 0; i < m_skinningActiveInstanceCount; ++i) {
        const SkinnedInstanceSlot& slot = m_skinningInstances[i];
        if (!slot.visible) {
            continue;
        }
        m_skinningMeshDraws.insert(
            m_skinningMeshDraws.end(), slot.meshDraws.begin(), slot.meshDraws.end());
    }
}

// Uploads a skinned actor's textures into the shared bindless table and hands
// back one slot per input, in order.
//
// A skinned actor cannot reach a texture the way imported world geometry does.
// A chunk's textures are uploaded by addImportedSceneChunk, which remaps every
// vertex's scene-local texture index onto a bindless slot on the way in; the
// caller never sees the mapping and has no way to ask for it. A skinned
// template's vertices are uploaded verbatim, so whatever
// ImportedSkinnedMeshVertex::textureIndex holds has to ALREADY be a bindless
// slot. This is how a caller gets one.
//
// The shape (transient command pool, acquire, submit on the render timeline,
// defer the staging buffers to that value) is setWeatherClouds' -- the other
// caller that uploads textures outside any scene. Slots are reference counted
// by source path in the same table, so an actor sharing a texture with the
// world costs nothing extra.
std::vector<std::uint32_t> RendererBackend::uploadSkinnedActorTextures(
    std::uint32_t instanceIndex, const std::vector<odai::importer::ImportedSceneTexture>& textures
) {
    std::vector<std::uint32_t> slots(textures.size(), kInvalidImportedTextureSlot);
    if (instanceIndex >= kMaxSkinnedInstances || textures.empty()) {
        return slots;
    }
    // Released only after the new acquires have run, so a texture shared
    // between the old set and the new one keeps its reference and its image
    // rather than being destroyed and re-uploaded.
    SkinnedInstanceSlot& slot = m_skinningInstances[instanceIndex];
    const std::vector<std::uint32_t> previousSlots = std::move(slot.textureSlots);
    slot.textureSlots.clear();

    // The same gate acquireImportedTexture applies; checking it up front avoids
    // building a command pool only to have every acquire refuse.
    if (!m_supportsBindlessDescriptors || !m_bindlessBufferSet.valid() ||
        m_bindlessTextureCapacity <= kBindlessTextureStaticCount) {
        for (const std::uint32_t previous : previousSlots) {
            releaseImportedTexture(previous);
        }
        return slots;
    }

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    commandPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    if (vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
        VOX_LOGE("render") << "skinned actor texture upload command pool creation failed";
        for (const std::uint32_t previous : previousSlots) {
            releaseImportedTexture(previous);
        }
        return slots;
    }
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkAllocateCommandBuffers(m_device, &allocateInfo, &commandBuffer) != VK_SUCCESS ||
        vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        VOX_LOGE("render") << "skinned actor texture upload command buffer setup failed";
        scheduleCommandPoolRelease(commandPool, 0);
        for (const std::uint32_t previous : previousSlots) {
            releaseImportedTexture(previous);
        }
        return slots;
    }

    std::vector<BufferHandle> stagingBufferHandles;
    for (std::size_t i = 0; i < textures.size(); ++i) {
        const odai::importer::ImportedSceneTexture& texture = textures[i];
        if (texture.rgba8.empty()) {
            continue;
        }
        slots[i] = acquireImportedTexture(
            normalizedImportedTextureKey(texture), texture, commandBuffer,
            stagingBufferHandles);
        if (slots[i] != kInvalidImportedTextureSlot) {
            slot.textureSlots.push_back(slots[i]);
        }
    }

    for (const std::uint32_t previous : previousSlots) {
        releaseImportedTexture(previous);
    }

    std::uint64_t uploadTimelineValue = 0;
    if (vkEndCommandBuffer(commandBuffer) == VK_SUCCESS) {
        uploadTimelineValue = m_nextTimelineValue++;
        VkSemaphoreSubmitInfo signalInfo{};
        signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signalInfo.semaphore = m_renderTimelineSemaphore;
        signalInfo.value = uploadTimelineValue;
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkCommandBufferSubmitInfo commandBufferInfo{};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        commandBufferInfo.commandBuffer = commandBuffer;
        VkSubmitInfo2 submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandBufferInfo;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos = &signalInfo;
        const VkResult submitResult =
            vkQueueSubmit2(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        if (submitResult != VK_SUCCESS) {
            logVkFailure("vkQueueSubmit2(skinnedActorTextureUpload)", submitResult);
            uploadTimelineValue = 0;
        } else {
            m_pendingTransferTimelineValue =
                std::max(m_pendingTransferTimelineValue, uploadTimelineValue);
        }
    }
    for (const BufferHandle stagingHandle : stagingBufferHandles) {
        if (stagingHandle != kInvalidBufferHandle) {
            scheduleBufferRelease(stagingHandle, uploadTimelineValue);
        }
    }
    scheduleCommandPoolRelease(commandPool, uploadTimelineValue);
    return slots;
}

// Public entry point (Renderer::setSkinnedActorPose), called by app code
// *before* renderFrame() -- the natural "update game state, then render"
// order. It must NOT touch the FrameArena directly: m_frameArena.beginFrame()
// only runs once renderFrame() itself starts, so allocating here would write
// into whatever frame-in-flight slot was active last frame, not this frame's.
// This just stores the pose; uploadSkinnedActorPoseForFrame() (below) does
// the actual per-frame upload, called from frame_run.cc right after
// m_frameArena.beginFrame(m_currentFrame).
void RendererBackend::setSkinnedActorVisible(std::uint32_t instanceIndex, bool visible) {
    if (instanceIndex >= kMaxSkinnedInstances) {
        return;
    }
    SkinnedInstanceSlot& slot = m_skinningInstances[instanceIndex];
    if (slot.visible == visible) {
        return;
    }
    slot.visible = visible;
    if (!visible) {
        slot.poseHistoryValid = false;
        slot.currentBoneAddress = 0;
        slot.previousBoneAddress = 0;
        slot.boneBufferBytes = 0;
    }
    rebuildVisibleSkinningDraws();
}

void RendererBackend::setSkinnedActorPose(std::uint32_t instanceIndex, const ImportedSkinnedActorFrameData& pose) {
    if (instanceIndex >= kMaxSkinnedInstances) {
        return;
    }
    SkinnedInstanceSlot& slot = m_skinningInstances[instanceIndex];
    // Reject a pose that doesn't match the slot's bound template -- both
    // recordSkinningPass's push-constant boneCount and the descriptor range
    // written below are sized from this data, so a mismatch here would
    // otherwise become a GPU out-of-bounds read in the compute shader.
    if (pose.boneMatrices.size() != slot.boneCount) {
        VOX_LOGW("render") << "setSkinnedActorPose: instance " << instanceIndex << " pose has "
                            << pose.boneMatrices.size() << " matrices, expected " << slot.boneCount
                            << " -- ignoring";
        return;
    }
    if (pose.morphWeights.size() != slot.morphTargetCount) {
        VOX_LOGW("render") << "setSkinnedActorPose: instance " << instanceIndex << " pose has "
                            << pose.morphWeights.size() << " morph weights, expected "
                            << slot.morphTargetCount << " -- ignoring";
        return;
    }
    for (const float weight : pose.morphWeights) {
        if (!std::isfinite(weight) || weight < 0.0f || weight > 1.0f) {
            VOX_LOGW("render") << "setSkinnedActorPose: instance " << instanceIndex
                                << " has an invalid morph weight -- ignoring";
            return;
        }
    }
    slot.pendingBoneMatrices.assign(pose.boneMatrices.begin(), pose.boneMatrices.end());
    slot.pendingMorphWeights.assign(pose.morphWeights.begin(), pose.morphWeights.end());
    slot.pendingEvaluation.reset();
    if (pose.resetHistory) {
        slot.poseHistoryValid = false;
        slot.previousBoneMatrices.clear();
        slot.previousMorphWeights.clear();
    }
    if (pose.animationView && pose.evaluationPacket && pose.animationView->skeleton) {
        // Admit only packets whose evaluated palette matches the submitted
        // authoritative pose while the new execution path is being proven.
        // This also catches equipment corrections, ragdolls and interpolation.
        odai::anim::LocalPose local;
        std::string error;
        if (odai::anim::evaluatePosePacket(*pose.evaluationPacket, *pose.animationView->skeleton,
                pose.animationView->clips, local, error)) {
            auto world = odai::anim::composePoseWorld(*pose.animationView->skeleton, local);
            bool matches = world.size() == pose.boneMatrices.size() &&
                world.size() == pose.animationView->inverseBindMatrices.size();
            for (std::size_t bone = 0; matches && bone < world.size(); ++bone) {
                const auto expected = pose.actorWorld * world[bone] * pose.animationView->inverseBindMatrices[bone];
                for (std::size_t e = 0; e < 16; ++e)
                    if (!std::isfinite(expected.m[e]) || !std::isfinite(pose.boneMatrices[bone].m[e]) ||
                        std::abs(expected.m[e] - pose.boneMatrices[bone].m[e]) > 1.e-4f)
                        matches = false;
            }
            if (matches) {
                if (slot.poseView != pose.animationView) {
                    scheduleBufferRelease(slot.poseResourceBuffer, m_nextTimelineValue - 1);
                    slot.poseResourceBuffer = kInvalidBufferHandle;
                    slot.poseResource = {};
                    slot.poseHistoryValid = false;
                    slot.poseView = pose.animationView;
                    slot.previousBoneMatrices.clear();
                    slot.previousMorphWeights.clear();
                }
                slot.pendingEvaluation = pose.evaluationPacket;
                slot.poseActorWorld = pose.actorWorld;
            }
        }
    }
}

// Called from frame_run.cc immediately after m_frameArena.beginFrame(), so
// the FrameArena slice allocated here belongs to the frame recordSkinningPass
// is about to record for.
void RendererBackend::uploadSkinnedActorPoseForFrame() {
    for (std::uint32_t i = 0; i < m_skinningActiveInstanceCount; ++i) {
        SkinnedInstanceSlot& slot = m_skinningInstances[i];
        slot.poseComputeReady = false;
        slot.poseHistoryReady = false;
        if (!slot.visible) {
            slot.currentBoneAddress = 0;
            slot.previousBoneAddress = 0;
            slot.boneBufferBytes = 0;
            slot.currentMorphAddress = 0;
            slot.previousMorphAddress = 0;
            slot.morphBufferBytes = 0;
            continue;
        }
        if (slot.vertexCount == 0 || slot.pendingBoneMatrices.empty() || !slot.bufferSet.valid()) {
            continue;
        }

        const VkDeviceSize boneBufferSize =
            static_cast<VkDeviceSize>(slot.pendingBoneMatrices.size() * sizeof(odai::math::Matrix4));
        // 256-byte alignment, NOT alignof(float).
        //
        // This slice is handed to the shader as a StructuredBuffer<float4x4>
        // via its device address, so the base has to satisfy that type's
        // alignment (16 under std430), and separately any device's
        // minStorageBufferOffsetAlignment. alignof(float) is 4 and satisfies
        // neither. 256 covers every device's limit with a few bytes of padding
        // once per frame.
        //
        // The failure this caused is worth recording because it looks like
        // anything but an alignment bug: misaligned matrices read as garbage,
        // garbage matrices put NaN in the skinned positions, and NaN reaching
        // the auto-exposure histogram takes the tonemapper -- and therefore the
        // WHOLE FRAME, world and sky included -- to a single flat colour. The
        // symptom was "adding a character makes the renderer stop working".
        constexpr VkDeviceSize kBoneMatrixAlignment = 256u;
        const std::optional<FrameArenaSlice> boneSlice = m_frameArena.allocateUpload(
            boneBufferSize, kBoneMatrixAlignment, FrameArenaUploadKind::Unknown);
        if (!boneSlice.has_value() || boneSlice->mapped == nullptr) {
            VOX_LOGW("render") << "skinning: bone matrix upload failed for instance " << i
                                << ", skipping this frame's pose";
            continue;
        }

        // skinning.comp.slang compiles -matrix-layout-column-major; the camera
        // MVP path (frame_run.cc) always transposes odai::math::Matrix4
        // (row-major) via renderer_shared.h's transpose() before uploading to a
        // GPU float4x4 for exactly this reason -- mirror that here rather than
        // copying verbatim.
        auto* dstMatrices = static_cast<odai::math::Matrix4*>(boneSlice->mapped);
        for (std::size_t m = 0; m < slot.pendingBoneMatrices.size(); ++m) {
            dstMatrices[m] = transpose(slot.pendingBoneMatrices[m]);
        }

        const VkDeviceAddress boneAddress =
            m_bufferAllocator.getDeviceAddress(boneSlice->buffer) + boneSlice->offset;
        writeDescriptorBufferStorage(
            slot.bufferSet, m_currentFrame,
            descriptorBufferBindingOffset(m_skinningDescriptorSetLayout, 1),
            boneAddress, boneBufferSize);

        // Morph weights change with gameplay sliders while topology and deltas
        // remain device-local. Bind a single zero for non-morph templates so
        // every descriptor region is valid even when the shader skips reads.
        const float zeroMorphWeight = 0.0f;
        const void* morphSource = slot.pendingMorphWeights.empty()
            ? static_cast<const void*>(&zeroMorphWeight)
            : static_cast<const void*>(slot.pendingMorphWeights.data());
        const VkDeviceSize morphBufferSize = static_cast<VkDeviceSize>(
            std::max<std::size_t>(slot.pendingMorphWeights.size(), 1u) * sizeof(float));
        const std::optional<FrameArenaSlice> morphSlice = m_frameArena.allocateUpload(
            morphBufferSize, 256u, FrameArenaUploadKind::Unknown);
        if (!morphSlice.has_value() || morphSlice->mapped == nullptr) {
            VOX_LOGW("render") << "skinning: morph weight upload failed for instance " << i
                                << ", skipping this frame's pose";
            continue;
        }
        std::memcpy(morphSlice->mapped, morphSource, static_cast<std::size_t>(morphBufferSize));
        writeDescriptorBufferStorage(
            slot.bufferSet, m_currentFrame,
            descriptorBufferBindingOffset(m_skinningDescriptorSetLayout, 5),
            m_bufferAllocator.getDeviceAddress(morphSlice->buffer) + morphSlice->offset,
            morphBufferSize);
        const VkDeviceAddress morphAddress =
            m_bufferAllocator.getDeviceAddress(morphSlice->buffer) + morphSlice->offset;

        // Last frame's pose, uploaded alongside this frame's so the velocity
        // pass can skin the same rest vertex into both. On the very first frame
        // for a slot there is no previous pose, so it reuses the current one --
        // which yields a zero motion vector, i.e. "this actor did not move",
        // which is the right answer for a pose that has only just appeared.
        const std::vector<odai::math::Matrix4>& previousSource =
            (slot.previousBoneMatrices.size() == slot.pendingBoneMatrices.size())
                ? slot.previousBoneMatrices
                : slot.pendingBoneMatrices;
        const std::optional<FrameArenaSlice> previousSlice = m_frameArena.allocateUpload(
            boneBufferSize, kBoneMatrixAlignment, FrameArenaUploadKind::Unknown);
        if (previousSlice.has_value() && previousSlice->mapped != nullptr) {
            auto* dstPrevious = static_cast<odai::math::Matrix4*>(previousSlice->mapped);
            for (std::size_t m = 0; m < previousSource.size(); ++m) {
                dstPrevious[m] = transpose(previousSource[m]);
            }
            slot.currentBoneAddress = boneAddress;
            slot.previousBoneAddress =
                m_bufferAllocator.getDeviceAddress(previousSlice->buffer) + previousSlice->offset;
            slot.boneBufferBytes = boneBufferSize;
            const std::vector<float>& previousMorphSource =
                (slot.previousMorphWeights.size() == slot.pendingMorphWeights.size())
                    ? slot.previousMorphWeights : slot.pendingMorphWeights;
            const void* previousMorphData = previousMorphSource.empty()
                ? static_cast<const void*>(&zeroMorphWeight)
                : static_cast<const void*>(previousMorphSource.data());
            const std::optional<FrameArenaSlice> previousMorphSlice =
                m_frameArena.allocateUpload(morphBufferSize, 256u,
                    FrameArenaUploadKind::Unknown);
            if (previousMorphSlice.has_value() && previousMorphSlice->mapped != nullptr) {
                std::memcpy(previousMorphSlice->mapped, previousMorphData,
                    static_cast<std::size_t>(morphBufferSize));
                slot.currentMorphAddress = morphAddress;
                slot.previousMorphAddress = m_bufferAllocator.getDeviceAddress(
                    previousMorphSlice->buffer) + previousMorphSlice->offset;
                slot.morphBufferBytes = morphBufferSize;
            } else {
                slot.currentMorphAddress = 0;
                slot.previousMorphAddress = 0;
                slot.morphBufferBytes = 0;
            }
        } else {
            // No previous upload means no velocity draw this frame; the pixels
            // fall back to depth reprojection rather than reading a stale pose.
            slot.currentBoneAddress = 0;
            slot.previousBoneAddress = 0;
            slot.boneBufferBytes = 0;
            slot.currentMorphAddress = 0;
            slot.previousMorphAddress = 0;
            slot.morphBufferBytes = 0;
        }
        if (slot.velocityBufferSet.valid() && slot.currentBoneAddress != 0 &&
            m_skinnedVelocityDescriptorSetLayout != VK_NULL_HANDLE) {
            writeDescriptorBufferStorage(
                slot.velocityBufferSet, m_currentFrame,
                descriptorBufferBindingOffset(m_skinnedVelocityDescriptorSetLayout, 0),
                slot.currentBoneAddress, slot.boneBufferBytes);
            writeDescriptorBufferStorage(
                slot.velocityBufferSet, m_currentFrame,
                descriptorBufferBindingOffset(m_skinnedVelocityDescriptorSetLayout, 1),
                slot.previousBoneAddress, slot.boneBufferBytes);
            if (slot.currentMorphAddress != 0) {
                writeDescriptorBufferStorage(
                    slot.velocityBufferSet, m_currentFrame,
                    descriptorBufferBindingOffset(m_skinnedVelocityDescriptorSetLayout, 4),
                    slot.currentMorphAddress, slot.morphBufferBytes);
                writeDescriptorBufferStorage(
                    slot.velocityBufferSet, m_currentFrame,
                    descriptorBufferBindingOffset(m_skinnedVelocityDescriptorSetLayout, 5),
                    slot.previousMorphAddress, slot.morphBufferBytes);
            }
        }
        slot.previousBoneMatrices = slot.pendingBoneMatrices;
        slot.previousMorphWeights = slot.pendingMorphWeights;
        if (slot.currentBoneAddress && slot.previousBoneAddress)
            preparePoseCompute(i,boneAddress,boneBufferSize);
    }
}

}  // namespace odai::render
