#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include "core/log.h"
#include "math/math.h"
#include "sim/network_procedural.h"

#include <algorithm>
#include <array>
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
    float pad0;
    float pad1;
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
};

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

        const std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            restPoseBinding, boneMatrixBinding, outputBinding
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
    return true;
}

void RendererBackend::destroySkinningComputeResources() {
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
        m_bufferAllocator.destroyBuffer(slot.restPoseVertexBufferHandle);
        slot.restPoseVertexBufferHandle = kInvalidBufferHandle;
        m_bufferAllocator.destroyBuffer(slot.indexBufferHandle);
        slot.indexBufferHandle = kInvalidBufferHandle;
        m_bufferAllocator.destroyBuffer(slot.outputVertexBufferHandle);
        slot.outputVertexBufferHandle = kInvalidBufferHandle;
        slot.vertexCount = 0;
        slot.boneCount = 0;
        slot.meshDraws.clear();
        slot.pendingBoneMatrices.clear();
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
        for (int b = 0; b < 4; ++b) {
            dst.boneIndices[b] = static_cast<std::uint32_t>(src.boneIndices[b]);
            dst.boneWeights[b] = src.boneWeights[b];
        }
    }

    // Same staged device-local upload shape as uploadImportedSceneInternal's
    // local uploadDeviceLocalBuffer lambda (chunk_upload.cc) -- duplicated
    // rather than shared, matching how every other upload call site in this
    // backend already does its own local copy of this pattern.
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

        if (!uploadFailed) {
            result = submitCommandBufferOneShot(m_graphicsQueue, commandBuffer, VK_NULL_HANDLE);
            if (result != VK_SUCCESS) {
                logVkFailure("vkQueueSubmit2(skinnedMeshUpload)", result);
                uploadFailed = true;
            }
        }
        if (!uploadFailed) {
            result = vkQueueWaitIdle(m_graphicsQueue);
            if (result != VK_SUCCESS) {
                logVkFailure("vkQueueWaitIdle(skinnedMeshUpload)", result);
                uploadFailed = true;
            }
        }

        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, commandPool, nullptr);
        }
        m_bufferAllocator.destroyBuffer(stagingHandle);
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
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            "skinned mesh rest-pose vertex",
            newRestPoseHandle)) {
        return false;
    }

    BufferHandle newIndexHandle = kInvalidBufferHandle;
    if (!uploadDeviceLocalBuffer(
            meshTemplate.indices.data(),
            static_cast<VkDeviceSize>(meshTemplate.indices.size() * sizeof(std::uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            "skinned mesh index",
            newIndexHandle)) {
        m_bufferAllocator.destroyBuffer(newRestPoseHandle);
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
        m_bufferAllocator.destroyBuffer(newRestPoseHandle);
        m_bufferAllocator.destroyBuffer(newIndexHandle);
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
    }

    m_bufferAllocator.destroyBuffer(slot.restPoseVertexBufferHandle);
    m_bufferAllocator.destroyBuffer(slot.indexBufferHandle);
    m_bufferAllocator.destroyBuffer(slot.outputVertexBufferHandle);

    slot.restPoseVertexBufferHandle = newRestPoseHandle;
    slot.indexBufferHandle = newIndexHandle;
    slot.outputVertexBufferHandle = newOutputHandle;
    slot.vertexCount = static_cast<std::uint32_t>(gpuVertices.size());
    slot.boneCount = meshTemplate.boneCount;

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
        slot.meshDraws.push_back(meshDraw);
    }

    m_skinningActiveInstanceCount = std::max(m_skinningActiveInstanceCount, instanceIndex + 1u);

    // Re-flatten every active instance slot's draws into one contiguous
    // vector so frame_pass_shadow.cc/frame_pass_prepass.cc/frame_pass_main.cc/
    // frame_run.cc keep consuming a single std::span<const ImportedMeshDraw>
    // with no changes.
    m_skinningMeshDraws.clear();
    for (std::uint32_t i = 0; i < m_skinningActiveInstanceCount; ++i) {
        const auto& draws = m_skinningInstances[i].meshDraws;
        m_skinningMeshDraws.insert(m_skinningMeshDraws.end(), draws.begin(), draws.end());
    }
    return true;
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
            normalizedImportedTextureKey(texture.sourcePath), texture, commandBuffer,
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
    slot.pendingBoneMatrices.assign(pose.boneMatrices.begin(), pose.boneMatrices.end());
}

// Called from frame_run.cc immediately after m_frameArena.beginFrame(), so
// the FrameArena slice allocated here belongs to the frame recordSkinningPass
// is about to record for.
void RendererBackend::uploadSkinnedActorPoseForFrame() {
    for (std::uint32_t i = 0; i < m_skinningActiveInstanceCount; ++i) {
        SkinnedInstanceSlot& slot = m_skinningInstances[i];
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
    }
}

}  // namespace odai::render
