#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>
#include "core/grid3.h"
#include "core/log.h"
#include "math/math.h"
#include "sim/network_procedural.h"
#include "world/chunk_mesher.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

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
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace odai::render {

#include "render/renderer_shared.h"

namespace {

struct ImportedDrawBounds {
    float min[3] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    float max[3] = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    };
    bool valid = false;
};

void expandImportedBounds(
    ImportedDrawBounds& bounds,
    const odai::importer::ImportedScenePackedVertex& vertex
) {
    bounds.valid = true;
    bounds.min[0] = std::min(bounds.min[0], vertex.position[0]);
    bounds.min[1] = std::min(bounds.min[1], vertex.position[1]);
    bounds.min[2] = std::min(bounds.min[2], vertex.position[2]);
    bounds.max[0] = std::max(bounds.max[0], vertex.position[0]);
    bounds.max[1] = std::max(bounds.max[1], vertex.position[1]);
    bounds.max[2] = std::max(bounds.max[2], vertex.position[2]);
}

void expandImportedBounds(
    ImportedDrawBounds& bounds,
    float x,
    float y,
    float z
) {
    bounds.valid = true;
    bounds.min[0] = std::min(bounds.min[0], x);
    bounds.min[1] = std::min(bounds.min[1], y);
    bounds.min[2] = std::min(bounds.min[2], z);
    bounds.max[0] = std::max(bounds.max[0], x);
    bounds.max[1] = std::max(bounds.max[1], y);
    bounds.max[2] = std::max(bounds.max[2], z);
}

VkDeviceSize importedTextureMipOffset(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t mipLevel
) {
    VkDeviceSize offset = 0;
    for (std::uint32_t level = 0; level < mipLevel; ++level) {
        offset += static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4u;
        width = std::max(1u, width >> 1u);
        height = std::max(1u, height >> 1u);
    }
    return offset;
}

std::uint32_t blockBytesForImportedFormat(odai::importer::TextureFormat format) {
    switch (format) {
        case odai::importer::TextureFormat::BC1:
        case odai::importer::TextureFormat::BC4: return 8u;
        case odai::importer::TextureFormat::BC3:
        case odai::importer::TextureFormat::BC5:
        case odai::importer::TextureFormat::BC7: return 16u;
        default:                                 return 0u;
    }
}

VkFormat vkFormatForImportedTexture(odai::importer::TextureFormat format) {
    switch (format) {
        // Color albedo (BC1/BC3/BC7) holds sRGB-encoded bytes, so use the _SRGB views:
        // the sampler decodes sRGB -> linear, matching the linear lighting + tonemap
        // pipeline. Without this the raw sRGB values are read as linear (~2x too bright)
        // and terrain renders as washed-out pastel. BC4/BC5 are data (single/dual
        // channel — e.g. the water normal map) and must stay UNORM/linear.
        case odai::importer::TextureFormat::BC1: return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
        case odai::importer::TextureFormat::BC3: return VK_FORMAT_BC3_SRGB_BLOCK;
        case odai::importer::TextureFormat::BC4: return VK_FORMAT_BC4_UNORM_BLOCK;
        case odai::importer::TextureFormat::BC5: return VK_FORMAT_BC5_UNORM_BLOCK;
        case odai::importer::TextureFormat::BC7: return VK_FORMAT_BC7_SRGB_BLOCK;
        default:                                 return VK_FORMAT_R8G8B8A8_UNORM;
    }
}

// Byte offset of mipLevel in a packed mip chain, respecting block-compressed layout.
VkDeviceSize importedTextureMipOffsetFmt(
    std::uint32_t width, std::uint32_t height,
    std::uint32_t mipLevel, odai::importer::TextureFormat format
) {
    if (format == odai::importer::TextureFormat::RGBA8) {
        return importedTextureMipOffset(width, height, mipLevel);
    }
    const std::uint32_t bpb = blockBytesForImportedFormat(format);
    VkDeviceSize offset = 0;
    for (std::uint32_t level = 0; level < mipLevel; ++level) {
        offset += static_cast<VkDeviceSize>(std::max(1u, (width  + 3u) / 4u))
                * std::max(1u, (height + 3u) / 4u) * bpb;
        width  = std::max(1u, width  >> 1u);
        height = std::max(1u, height >> 1u);
    }
    return offset;
}

std::uint32_t inferImportedTextureMipLevelCount(
    std::uint32_t width,
    std::uint32_t height,
    std::size_t rgbaByteSize
) {
    if (width == 0u || height == 0u || rgbaByteSize == 0u) {
        return 0u;
    }
    std::size_t consumedBytes = 0u;
    std::uint32_t mipLevelCount = 0u;
    while (true) {
        const std::size_t mipByteSize =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
        consumedBytes += mipByteSize;
        ++mipLevelCount;
        if (consumedBytes == rgbaByteSize) {
            return mipLevelCount;
        }
        if (consumedBytes > rgbaByteSize) {
            return 0u;
        }
        if (width == 1u && height == 1u) {
            return 0u;
        }
        width = std::max(1u, width >> 1u);
        height = std::max(1u, height >> 1u);
    }
}

ImportedDrawBounds computeImportedDrawBounds(
    const std::vector<odai::importer::ImportedScenePackedVertex>& vertices,
    const std::vector<std::uint32_t>& indices,
    std::span<const odai::importer::ImportedScenePackedDraw> draws
) {
    ImportedDrawBounds bounds{};
    for (const odai::importer::ImportedScenePackedDraw& draw : draws) {
        const std::size_t indexEnd = static_cast<std::size_t>(draw.firstIndex) + static_cast<std::size_t>(draw.indexCount);
        if (draw.indexCount == 0 || indexEnd > indices.size()) {
            continue;
        }
        for (std::size_t i = draw.firstIndex; i < indexEnd; ++i) {
            const std::uint32_t vertexIndex = indices[i];
            if (vertexIndex >= vertices.size()) {
                continue;
            }
            expandImportedBounds(bounds, vertices[vertexIndex]);
        }
    }
    return bounds;
}

std::array<float, 3> sampleImportedTextureBaseColor(
    const std::vector<odai::importer::ImportedSceneTexture>& textures,
    const odai::importer::ImportedScenePackedVertex& vertex
) {
    if (vertex.textureIndex >= textures.size()) {
        return {vertex.color[0], vertex.color[1], vertex.color[2]};
    }
    const odai::importer::ImportedSceneTexture& texture = textures[vertex.textureIndex];
    if (texture.format != odai::importer::TextureFormat::RGBA8) {
        return {vertex.color[0], vertex.color[1], vertex.color[2]};
    }
    if (texture.width == 0u ||
        texture.height == 0u ||
        texture.rgba8.size() < static_cast<std::size_t>(texture.width) *
            static_cast<std::size_t>(texture.height) * 4u) {
        return {vertex.color[0], vertex.color[1], vertex.color[2]};
    }

    const float u = vertex.uv[0] - std::floor(vertex.uv[0]);
    const float v = vertex.uv[1] - std::floor(vertex.uv[1]);
    const std::uint32_t x = std::min(
        static_cast<std::uint32_t>(u * static_cast<float>(texture.width)),
        texture.width - 1u);
    const std::uint32_t y = std::min(
        static_cast<std::uint32_t>(v * static_cast<float>(texture.height)),
        texture.height - 1u);
    const std::size_t offset =
        ((static_cast<std::size_t>(y) * static_cast<std::size_t>(texture.width)) +
            static_cast<std::size_t>(x)) * 4u;
    return {
        static_cast<float>(texture.rgba8[offset + 0u]) / 255.0f,
        static_cast<float>(texture.rgba8[offset + 1u]) / 255.0f,
        static_cast<float>(texture.rgba8[offset + 2u]) / 255.0f
    };
}

ChunkResidentKey chunkResidentKeyForChunk(const odai::world::Chunk& chunk) {
    return ChunkResidentKey{
        chunk.chunkX(),
        chunk.chunkY(),
        chunk.chunkZ()
    };
}

bool chunkResidentKeyMatchesRecord(
    const ChunkResidentKey& key,
    const RtChunkSceneRecord& record
) {
    return key.chunkX == record.chunkX &&
           key.chunkY == record.chunkY &&
           key.chunkZ == record.chunkZ;
}

RtVertex decodePackedVoxelVertexPosition(std::uint32_t packedBits, float offsetX, float offsetY, float offsetZ) {
    const std::uint32_t x =
        (packedBits >> odai::world::PackedVoxelVertex::kShiftX) & odai::world::PackedVoxelVertex::kMask5;
    const std::uint32_t y =
        (packedBits >> odai::world::PackedVoxelVertex::kShiftY) & odai::world::PackedVoxelVertex::kMask5;
    const std::uint32_t z =
        (packedBits >> odai::world::PackedVoxelVertex::kShiftZ) & odai::world::PackedVoxelVertex::kMask5;
    const std::uint32_t face =
        (packedBits >> odai::world::PackedVoxelVertex::kShiftFace) & odai::world::PackedVoxelVertex::kMask3;
    const std::uint32_t corner =
        (packedBits >> odai::world::PackedVoxelVertex::kShiftCorner) & odai::world::PackedVoxelVertex::kMask2;

    RtVertex vertex{};
    vertex.position[0] = static_cast<float>(x) + offsetX;
    vertex.position[1] = static_cast<float>(y) + offsetY;
    vertex.position[2] = static_cast<float>(z) + offsetZ;
    if (face == 0u) {
        vertex.position[0] += 1.0f;
        vertex.position[1] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
        vertex.position[2] += (corner == 2u || corner == 3u) ? 1.0f : 0.0f;
        return vertex;
    }
    if (face == 1u) {
        vertex.position[1] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
        vertex.position[2] += (corner == 0u || corner == 1u) ? 1.0f : 0.0f;
        return vertex;
    }
    if (face == 2u) {
        vertex.position[0] += (corner == 2u || corner == 3u) ? 1.0f : 0.0f;
        vertex.position[1] += 1.0f;
        vertex.position[2] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
        return vertex;
    }
    if (face == 3u) {
        vertex.position[0] += (corner == 2u || corner == 3u) ? 1.0f : 0.0f;
        vertex.position[2] += (corner == 0u || corner == 3u) ? 1.0f : 0.0f;
        return vertex;
    }
    if (face == 4u) {
        vertex.position[0] += (corner == 0u || corner == 1u) ? 1.0f : 0.0f;
        vertex.position[1] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
        vertex.position[2] += 1.0f;
        return vertex;
    }

    vertex.position[0] += (corner == 2u || corner == 3u) ? 1.0f : 0.0f;
    vertex.position[1] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
    return vertex;
}

void destroyRtGeometryBuffers(BufferAllocator& allocator, RtGeometryBuffers& geometry) {
    if (geometry.indexBufferHandle != kInvalidBufferHandle) {
        allocator.destroyBuffer(geometry.indexBufferHandle);
        geometry.indexBufferHandle = kInvalidBufferHandle;
    }
    if (geometry.vertexBufferHandle != kInvalidBufferHandle) {
        allocator.destroyBuffer(geometry.vertexBufferHandle);
        geometry.vertexBufferHandle = kInvalidBufferHandle;
    }
    geometry.vertexCount = 0;
    geometry.indexCount = 0;
}

bool createRtGeometryBuffers(
    BufferAllocator& allocator,
    const std::vector<RtVertex>& vertices,
    const std::vector<std::uint32_t>& indices,
    RtGeometryBuffers& outGeometry
) {
    destroyRtGeometryBuffers(allocator, outGeometry);
    if (vertices.empty() || indices.empty()) {
        return true;
    }

    BufferCreateDesc vertexCreateDesc{};
    vertexCreateDesc.size = static_cast<VkDeviceSize>(vertices.size() * sizeof(RtVertex));
    vertexCreateDesc.usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vertexCreateDesc.initialData = vertices.data();
    outGeometry.vertexBufferHandle = allocator.createBuffer(vertexCreateDesc);
    if (outGeometry.vertexBufferHandle == kInvalidBufferHandle) {
        return false;
    }

    BufferCreateDesc indexCreateDesc{};
    indexCreateDesc.size = static_cast<VkDeviceSize>(indices.size() * sizeof(std::uint32_t));
    indexCreateDesc.usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    indexCreateDesc.initialData = indices.data();
    outGeometry.indexBufferHandle = allocator.createBuffer(indexCreateDesc);
    if (outGeometry.indexBufferHandle == kInvalidBufferHandle) {
        destroyRtGeometryBuffers(allocator, outGeometry);
        return false;
    }

    outGeometry.vertexCount = static_cast<std::uint32_t>(vertices.size());
    outGeometry.indexCount = static_cast<std::uint32_t>(indices.size());
    return true;
}

bool createImportedRtGeometryBuffers(
    BufferAllocator& allocator,
    const std::vector<odai::importer::ImportedScenePackedVertex>& packedVertices,
    const std::vector<std::uint32_t>& packedIndices,
    std::span<const odai::importer::ImportedScenePackedDraw> draws,
    RtGeometryBuffers& outGeometry
) {
    if (packedVertices.empty() || packedIndices.empty() || draws.empty()) {
        destroyRtGeometryBuffers(allocator, outGeometry);
        return true;
    }

    std::vector<RtVertex> rtVertices;
    rtVertices.reserve(packedVertices.size());
    for (const odai::importer::ImportedScenePackedVertex& packedVertex : packedVertices) {
        RtVertex rtVertex{};
        rtVertex.position[0] = packedVertex.position[0];
        rtVertex.position[1] = packedVertex.position[1];
        rtVertex.position[2] = packedVertex.position[2];
        rtVertices.push_back(rtVertex);
    }

    std::vector<std::uint32_t> rtIndices;
    rtIndices.reserve(packedIndices.size());
    for (const odai::importer::ImportedScenePackedDraw& draw : draws) {
        const std::size_t firstIndex = static_cast<std::size_t>(draw.firstIndex);
        const std::size_t indexCount = static_cast<std::size_t>(draw.indexCount);
        if (indexCount == 0 || firstIndex >= packedIndices.size()) {
            continue;
        }
        const std::size_t indexEnd = std::min(firstIndex + indexCount, packedIndices.size());
        rtIndices.insert(
            rtIndices.end(),
            packedIndices.begin() + static_cast<std::ptrdiff_t>(firstIndex),
            packedIndices.begin() + static_cast<std::ptrdiff_t>(indexEnd)
        );
    }

    if (rtIndices.empty()) {
        destroyRtGeometryBuffers(allocator, outGeometry);
        return true;
    }
    return createRtGeometryBuffers(allocator, rtVertices, rtIndices, outGeometry);
}

} // namespace

void RendererBackend::clearMagicaVoxelMeshes() {
    for (MagicaMeshDraw& draw : m_magicaMeshDraws) {
        if (draw.vertexBufferHandle != kInvalidBufferHandle) {
            scheduleBufferRelease(draw.vertexBufferHandle, m_lastGraphicsTimelineValue);
            draw.vertexBufferHandle = kInvalidBufferHandle;
        }
        if (draw.indexBufferHandle != kInvalidBufferHandle) {
            scheduleBufferRelease(draw.indexBufferHandle, m_lastGraphicsTimelineValue);
            draw.indexBufferHandle = kInvalidBufferHandle;
        }
        draw.indexCount = 0;
    }
    m_magicaMeshDraws.clear();
    for (RtGeometryBuffers& geometry : m_rtMagicaGeometries) {
        destroyRtGeometryBuffers(m_bufferAllocator, geometry);
    }
    m_rtMagicaGeometries.clear();
    markRayTracingSceneDirty();
}

void RendererBackend::clearGpuScene() {
    if (!m_importedTextureResources.empty() && m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
        for (ImportedTextureResource& texture : m_importedTextureResources) {
            if (texture.imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(m_device, texture.imageView, nullptr);
                texture.imageView = VK_NULL_HANDLE;
            }
            if (texture.image != VK_NULL_HANDLE) {
                if (m_vmaAllocator != VK_NULL_HANDLE && texture.allocation != VK_NULL_HANDLE) {
                    vmaDestroyImage(m_vmaAllocator, texture.image, texture.allocation);
                } else {
                    vkDestroyImage(m_device, texture.image, nullptr);
                }
                texture.image = VK_NULL_HANDLE;
                texture.allocation = VK_NULL_HANDLE;
            }
        }
        m_importedTextureResources.clear();
    }
    if (m_importedVertexBufferHandle != kInvalidBufferHandle) {
        if (m_lastGraphicsTimelineValue == 0) {
            m_bufferAllocator.destroyBuffer(m_importedVertexBufferHandle);
        } else {
            scheduleBufferRelease(m_importedVertexBufferHandle, m_lastGraphicsTimelineValue);
        }
        m_importedVertexBufferHandle = kInvalidBufferHandle;
    }
    if (m_importedIndexBufferHandle != kInvalidBufferHandle) {
        if (m_lastGraphicsTimelineValue == 0) {
            m_bufferAllocator.destroyBuffer(m_importedIndexBufferHandle);
        } else {
            scheduleBufferRelease(m_importedIndexBufferHandle, m_lastGraphicsTimelineValue);
        }
        m_importedIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_importedWaterVertexBufferHandle != kInvalidBufferHandle) {
        if (m_lastGraphicsTimelineValue == 0) {
            m_bufferAllocator.destroyBuffer(m_importedWaterVertexBufferHandle);
        } else {
            scheduleBufferRelease(m_importedWaterVertexBufferHandle, m_lastGraphicsTimelineValue);
        }
        m_importedWaterVertexBufferHandle = kInvalidBufferHandle;
    }
    if (m_importedWaterIndexBufferHandle != kInvalidBufferHandle) {
        if (m_lastGraphicsTimelineValue == 0) {
            m_bufferAllocator.destroyBuffer(m_importedWaterIndexBufferHandle);
        } else {
            scheduleBufferRelease(m_importedWaterIndexBufferHandle, m_lastGraphicsTimelineValue);
        }
        m_importedWaterIndexBufferHandle = kInvalidBufferHandle;
    }
    m_importedMeshDraws.clear();
    m_importedPageDrawRanges.clear();
    m_visibleImportedMeshDraws.clear();
    m_importedTextureSlots.clear();
    for (std::vector<ImportedMeshDraw>& shadowDraws : m_visibleImportedShadowMeshDraws) {
        shadowDraws.clear();
    }
    m_visibleImportedPageScratch.clear();
    m_visibleImportedTerrainDrawCount = 0;
    m_visibleImportedShadowTerrainDrawCounts.fill(0u);
    m_importedGiTriangles.clear();
    m_debugImportedGiTriangleCount = 0;
    m_debugImportedGiVoxelizedCellCount = 0;
    m_importedLocalLights.clear();
    m_debugImportedLightSelectedCount = 0;
    m_importedIndexCount = 0;
    m_importedTerrainDrawCount = 0;
    m_importedStaticDrawCount = 0;
    m_importedWaterIndexCount = 0;
    for (RtImportedSceneRecord& record : m_rtImportedSceneRecords) {
        if (record.blas.handle != VK_NULL_HANDLE && m_destroyAccelerationStructureKhr != nullptr) {
            m_destroyAccelerationStructureKhr(m_device, record.blas.handle, nullptr);
            record.blas.handle = VK_NULL_HANDLE;
        }
        if (record.blas.storageBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(record.blas.storageBufferHandle);
            record.blas.storageBufferHandle = kInvalidBufferHandle;
        }
        record.blas.deviceAddress = 0;
        record.blas.primitiveCount = 0;
        destroyRtGeometryBuffers(m_bufferAllocator, record.geometry);
        record.geometryResident = false;
        record.dirty = true;
    }
    if (!m_rtImportedSceneRecords.empty()) {
        markRayTracingSceneDirty();
    }
}

void RendererBackend::clearImportedSceneMeshes() {
    clearGpuScene();
}

void RendererBackend::clearHexTerrain() {
    const auto release = [&](BufferHandle& handle) {
        if (handle == kInvalidBufferHandle) {
            return;
        }
        if (m_lastGraphicsTimelineValue == 0) {
            m_bufferAllocator.destroyBuffer(handle);
        } else {
            scheduleBufferRelease(handle, m_lastGraphicsTimelineValue);
        }
        handle = kInvalidBufferHandle;
    };
    release(m_hexBaseVertexBufferHandle);
    release(m_hexBaseIndexBufferHandle);
    release(m_hexInstanceBufferHandle);
    m_hexIndexCount = 0;
    m_hexInstanceCount = 0;
}

bool RendererBackend::uploadHexTerrain(const odai::importer::HexTerrainData& data) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    clearHexTerrain();
    if (data.baseVertices.empty() || data.baseIndices.empty() || data.instances.empty()) {
        return true;  // e.g. an all-water map: nothing to displace, not an error.
    }

    BufferCreateDesc vertexDesc{};
    vertexDesc.size = static_cast<VkDeviceSize>(data.baseVertices.size() * sizeof(odai::importer::HexBaseVertex));
    vertexDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vertexDesc.initialData = data.baseVertices.data();
    const BufferHandle vertexHandle = m_bufferAllocator.createBuffer(vertexDesc);

    BufferCreateDesc indexDesc{};
    indexDesc.size = static_cast<VkDeviceSize>(data.baseIndices.size() * sizeof(std::uint32_t));
    indexDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    indexDesc.initialData = data.baseIndices.data();
    const BufferHandle indexHandle = m_bufferAllocator.createBuffer(indexDesc);

    // Remap each instance's packed terrain texture index (a scene index in classFlags
    // bits 16-31, written by the builder) to its bindless slot, resolved when the
    // imported-scene textures were uploaded. 0xFFFF keeps the fragment palette fallback.
    std::vector<odai::importer::HexTileInstance> instances = data.instances;
    for (odai::importer::HexTileInstance& inst : instances) {
        const std::uint32_t sceneIdx = (inst.classFlags >> 16u) & 0xFFFFu;
        std::uint32_t bindlessSlot = 0xFFFFu;
        if (sceneIdx != 0xFFFFu && sceneIdx < m_importedTextureSlots.size() &&
            m_importedTextureSlots[sceneIdx] != std::numeric_limits<std::uint32_t>::max()) {
            bindlessSlot = m_importedTextureSlots[sceneIdx] & 0xFFFFu;
        }
        inst.classFlags = (inst.classFlags & 0x0000FFFFu) | (bindlessSlot << 16u);
    }

    BufferCreateDesc instanceDesc{};
    instanceDesc.size = static_cast<VkDeviceSize>(instances.size() * sizeof(odai::importer::HexTileInstance));
    instanceDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    instanceDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    instanceDesc.initialData = instances.data();
    const BufferHandle instanceHandle = m_bufferAllocator.createBuffer(instanceDesc);

    if (vertexHandle == kInvalidBufferHandle || indexHandle == kInvalidBufferHandle ||
        instanceHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "hex terrain buffer allocation failed";
        if (vertexHandle != kInvalidBufferHandle) m_bufferAllocator.destroyBuffer(vertexHandle);
        if (indexHandle != kInvalidBufferHandle) m_bufferAllocator.destroyBuffer(indexHandle);
        if (instanceHandle != kInvalidBufferHandle) m_bufferAllocator.destroyBuffer(instanceHandle);
        return false;
    }

    m_hexBaseVertexBufferHandle = vertexHandle;
    m_hexBaseIndexBufferHandle = indexHandle;
    m_hexInstanceBufferHandle = instanceHandle;
    m_hexIndexCount = static_cast<uint32_t>(data.baseIndices.size());
    m_hexInstanceCount = static_cast<uint32_t>(data.instances.size());

    const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(vertexHandle);
    if (vertexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), "hex.terrain.baseVertex");
    }
    const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(indexHandle);
    if (indexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), "hex.terrain.baseIndex");
    }
    const VkBuffer instanceBuffer = m_bufferAllocator.getBuffer(instanceHandle);
    if (instanceBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(instanceBuffer), "hex.terrain.instance");
    }

    VOX_LOGI("render") << "uploaded hex terrain: instances=" << m_hexInstanceCount
                       << ", baseIndices=" << m_hexIndexCount;
    return true;
}

bool RendererBackend::uploadGpuScene(const odai::importer::GpuSceneAsset& scene) {
    odai::importer::ImportedScene compatibilityScene{};
    compatibilityScene.sourceTag = scene.sourceTag;
    compatibilityScene.textures = scene.renderCache.textures;
    compatibilityScene.waterPatches = scene.renderCache.waterPatches;
    compatibilityScene.lights = scene.renderCache.lights;
    compatibilityScene.packedVertices = scene.renderCache.packedVertices;
    compatibilityScene.packedIndices = scene.renderCache.packedIndices;
    compatibilityScene.packedDraws = scene.renderCache.packedDraws;
    compatibilityScene.sourceTextureCount = static_cast<std::uint32_t>(scene.textures.size());
    compatibilityScene.sourceMeshCount = static_cast<std::uint32_t>(scene.meshAssets.size());
    compatibilityScene.sourceInstanceCount = static_cast<std::uint32_t>(scene.instances.objectIndices.size());
    compatibilityScene.sourceLandscapeCellCount = scene.renderCache.terrainDrawCount;
    compatibilityScene.sourceWaterPatchCount = static_cast<std::uint32_t>(scene.waterPatches.size());
    compatibilityScene.sourceLightCount = static_cast<std::uint32_t>(scene.lights.size());
    compatibilityScene.boundsMin[0] = scene.sceneBounds.min[0];
    compatibilityScene.boundsMin[1] = scene.sceneBounds.min[1];
    compatibilityScene.boundsMin[2] = scene.sceneBounds.min[2];
    compatibilityScene.boundsMax[0] = scene.sceneBounds.max[0];
    compatibilityScene.boundsMax[1] = scene.sceneBounds.max[1];
    compatibilityScene.boundsMax[2] = scene.sceneBounds.max[2];
    return uploadImportedSceneInternal(compatibilityScene, &scene);
}

bool RendererBackend::uploadImportedScene(const odai::importer::ImportedScene& scene) {
    return uploadImportedSceneInternal(scene, nullptr);
}

bool RendererBackend::uploadImportedSceneInternal(
    const odai::importer::ImportedScene& scene,
    const odai::importer::GpuSceneAsset* gpuScene
) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }

    clearImportedSceneMeshes();
    destroyRayTracingScene();

    odai::importer::ImportedScene uploadScene = scene;
    const bool havePackedScene =
        !uploadScene.packedVertices.empty() &&
        !uploadScene.packedIndices.empty() &&
        !uploadScene.packedDraws.empty();
    if (!havePackedScene) {
        VOX_LOGI("render") << "imported scene missing packed geometry cache; rebuilding render stream on load";
        odai::importer::buildImportedScenePackedRenderData(uploadScene);
    } else {
        VOX_LOGI("render") << "imported scene using packed geometry cache (vertices="
                           << uploadScene.packedVertices.size()
                           << ", indices=" << uploadScene.packedIndices.size()
                           << ", draws=" << uploadScene.packedDraws.size() << ")";
    }

    std::vector<std::uint32_t> importedTextureSlots(
        uploadScene.textures.size(),
        std::numeric_limits<std::uint32_t>::max());
    if (m_supportsBindlessDescriptors &&
        m_bindlessDescriptorSet != VK_NULL_HANDLE &&
        m_bindlessTextureCapacity > kBindlessTextureStaticCount &&
        !uploadScene.textures.empty()) {
        if (m_importedTextureSampler == VK_NULL_HANDLE) {
            VkSamplerCreateInfo samplerCreateInfo{};
            samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
            samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
            samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerCreateInfo.mipLodBias = 0.0f;
            samplerCreateInfo.anisotropyEnable = m_supportsSamplerAnisotropy ? VK_TRUE : VK_FALSE;
            samplerCreateInfo.maxAnisotropy = m_supportsSamplerAnisotropy
                ? std::min(m_maxSamplerAnisotropy, 8.0f)
                : 1.0f;
            samplerCreateInfo.compareEnable = VK_FALSE;
            samplerCreateInfo.minLod = 0.0f;
            samplerCreateInfo.maxLod = 16.0f;
            samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
            if (vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_importedTextureSampler) != VK_SUCCESS) {
                VOX_LOGE("render") << "imported texture sampler creation failed";
                m_importedTextureSampler = VK_NULL_HANDLE;
                return false;
            }
            setObjectName(
                VK_OBJECT_TYPE_SAMPLER,
                vkHandleToUint64(m_importedTextureSampler),
                "imported.texture.sampler");
        }

        std::vector<BufferHandle> stagingBufferHandles;
        stagingBufferHandles.reserve(uploadScene.textures.size());
        bool textureUploadFailed = false;
        std::size_t uploadedTextureCount = 0u;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        if (!uploadScene.textures.empty()) {
            VkCommandPoolCreateInfo commandPoolCreateInfo{};
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            commandPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
            if (vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
                VOX_LOGE("render") << "imported texture upload command pool creation failed";
                textureUploadFailed = true;
            }
            if (!textureUploadFailed) {
                VkCommandBufferAllocateInfo allocateInfo{};
                allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                allocateInfo.commandPool = commandPool;
                allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocateInfo.commandBufferCount = 1;
                if (vkAllocateCommandBuffers(m_device, &allocateInfo, &commandBuffer) != VK_SUCCESS) {
                    VOX_LOGE("render") << "imported texture upload command buffer allocation failed";
                    textureUploadFailed = true;
                }
            }
            if (!textureUploadFailed) {
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
                    VOX_LOGE("render") << "imported texture upload command buffer begin failed";
                    textureUploadFailed = true;
                }
            }
        }

        const std::uint32_t maxImportedTextures =
            m_bindlessTextureCapacity - kBindlessTextureStaticCount;
        const std::size_t importedTextureLimit =
            std::min<std::size_t>(uploadScene.textures.size(), maxImportedTextures);
        for (std::size_t textureIndex = 0; textureIndex < importedTextureLimit && !textureUploadFailed; ++textureIndex) {
            const odai::importer::ImportedSceneTexture& srcTexture = uploadScene.textures[textureIndex];
            if (srcTexture.width == 0u || srcTexture.height == 0u || srcTexture.rgba8.empty()) {
                continue;
            }
            const std::uint32_t inferredMipLevelCount =
                inferImportedTextureMipLevelCount(srcTexture.width, srcTexture.height, srcTexture.rgba8.size());

            std::uint32_t mipLevelCount;
            if (srcTexture.format != odai::importer::TextureFormat::RGBA8) {
                // Block-compressed: trust the mip count stored by the DDS loader.
                if (srcTexture.mipLevelCount == 0u || srcTexture.rgba8.empty()) {
                    VOX_LOGW("render") << "block-compressed texture missing mip data: "
                                       << srcTexture.sourcePath << "; skipping";
                    continue;
                }
                mipLevelCount = srcTexture.mipLevelCount;
            } else {
                if (inferredMipLevelCount == 0u) {
                    VOX_LOGW("render") << "imported texture mip chain size invalid for "
                                       << srcTexture.sourcePath << "; skipping texture";
                    continue;
                }
                mipLevelCount = inferredMipLevelCount;
                if (srcTexture.mipLevelCount != 0u && srcTexture.mipLevelCount != inferredMipLevelCount) {
                    VOX_LOGW("render") << "imported texture mip metadata mismatch for "
                                       << srcTexture.sourcePath << "; stored=" << srcTexture.mipLevelCount
                                       << ", inferred=" << inferredMipLevelCount
                                       << " (using inferred chain)";
                }
            }

            BufferCreateDesc stagingCreateDesc{};
            stagingCreateDesc.size = static_cast<VkDeviceSize>(srcTexture.rgba8.size());
            stagingCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            stagingCreateDesc.memoryProperties =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            stagingCreateDesc.initialData = srcTexture.rgba8.data();
            const BufferHandle stagingHandle = m_bufferAllocator.createBuffer(stagingCreateDesc);
            if (stagingHandle == kInvalidBufferHandle) {
                VOX_LOGE("render") << "imported texture staging buffer allocation failed for "
                                   << srcTexture.sourcePath;
                textureUploadFailed = true;
                break;
            }
            stagingBufferHandles.push_back(stagingHandle);

            VkImageCreateInfo imageCreateInfo{};
            imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
            imageCreateInfo.format = vkFormatForImportedTexture(srcTexture.format);
            imageCreateInfo.extent = {srcTexture.width, srcTexture.height, 1};
            imageCreateInfo.mipLevels = mipLevelCount;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            ImportedTextureResource resource{};
            VmaAllocationCreateInfo allocationCreateInfo{};
            allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            const VkResult imageCreateResult = vmaCreateImage(
                m_vmaAllocator,
                &imageCreateInfo,
                &allocationCreateInfo,
                &resource.image,
                &resource.allocation,
                nullptr);
            if (imageCreateResult != VK_SUCCESS) {
                logVkFailure("vmaCreateImage(importedTexture)", imageCreateResult);
                textureUploadFailed = true;
                break;
            }

            VkImageViewCreateInfo viewCreateInfo{};
            viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewCreateInfo.image = resource.image;
            viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewCreateInfo.format = vkFormatForImportedTexture(srcTexture.format);
            viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewCreateInfo.subresourceRange.baseMipLevel = 0;
            viewCreateInfo.subresourceRange.levelCount = mipLevelCount;
            viewCreateInfo.subresourceRange.baseArrayLayer = 0;
            viewCreateInfo.subresourceRange.layerCount = 1;
            const VkResult imageViewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &resource.imageView);
            if (imageViewResult != VK_SUCCESS) {
                logVkFailure("vkCreateImageView(importedTexture)", imageViewResult);
                if (resource.image != VK_NULL_HANDLE) {
                    vmaDestroyImage(m_vmaAllocator, resource.image, resource.allocation);
                }
                textureUploadFailed = true;
                break;
            }

            setObjectName(
                VK_OBJECT_TYPE_IMAGE,
                vkHandleToUint64(resource.image),
                ("imported.texture.image." + std::to_string(textureIndex)).c_str());
            setObjectName(
                VK_OBJECT_TYPE_IMAGE_VIEW,
                vkHandleToUint64(resource.imageView),
                ("imported.texture.view." + std::to_string(textureIndex)).c_str());

            VkImageMemoryBarrier transitionToCopy{};
            transitionToCopy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            transitionToCopy.srcAccessMask = 0;
            transitionToCopy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            transitionToCopy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            transitionToCopy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            transitionToCopy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            transitionToCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            transitionToCopy.image = resource.image;
            transitionToCopy.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            transitionToCopy.subresourceRange.baseMipLevel = 0;
            transitionToCopy.subresourceRange.levelCount = mipLevelCount;
            transitionToCopy.subresourceRange.baseArrayLayer = 0;
            transitionToCopy.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &transitionToCopy);

            std::vector<VkBufferImageCopy> copyRegions;
            copyRegions.reserve(mipLevelCount);
            for (std::uint32_t mipLevel = 0; mipLevel < mipLevelCount; ++mipLevel) {
                VkBufferImageCopy copyRegion{};
                copyRegion.bufferOffset = importedTextureMipOffsetFmt(
                    srcTexture.width, srcTexture.height, mipLevel, srcTexture.format);
                copyRegion.bufferRowLength = 0;
                copyRegion.bufferImageHeight = 0;
                copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.imageSubresource.mipLevel = mipLevel;
                copyRegion.imageSubresource.baseArrayLayer = 0;
                copyRegion.imageSubresource.layerCount = 1;
                copyRegion.imageOffset = {0, 0, 0};
                copyRegion.imageExtent = {
                    std::max(1u, srcTexture.width >> mipLevel),
                    std::max(1u, srcTexture.height >> mipLevel),
                    1
                };
                copyRegions.push_back(copyRegion);
            }
            vkCmdCopyBufferToImage(
                commandBuffer,
                m_bufferAllocator.getBuffer(stagingHandle),
                resource.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                static_cast<std::uint32_t>(copyRegions.size()),
                copyRegions.data());

            VkImageMemoryBarrier transitionToRead{};
            transitionToRead.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            transitionToRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            transitionToRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            transitionToRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            transitionToRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            transitionToRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            transitionToRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            transitionToRead.image = resource.image;
            transitionToRead.subresourceRange = transitionToCopy.subresourceRange;
            vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &transitionToRead);

            importedTextureSlots[textureIndex] =
                static_cast<std::uint32_t>(kBindlessTextureStaticCount + m_importedTextureResources.size());
            m_importedTextureResources.push_back(resource);
            ++uploadedTextureCount;
        }

        if (!textureUploadFailed && commandBuffer != VK_NULL_HANDLE) {
            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                VOX_LOGE("render") << "imported texture upload command buffer end failed";
                textureUploadFailed = true;
            }
        }
        if (!textureUploadFailed && commandBuffer != VK_NULL_HANDLE) {
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
                vkQueueWaitIdle(m_graphicsQueue) != VK_SUCCESS) {
                VOX_LOGE("render") << "imported texture upload submit failed";
                textureUploadFailed = true;
            }
        }
        for (const BufferHandle stagingHandle : stagingBufferHandles) {
            if (stagingHandle != kInvalidBufferHandle) {
                m_bufferAllocator.destroyBuffer(stagingHandle);
            }
        }
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, commandPool, nullptr);
        }
        if (textureUploadFailed) {
            clearImportedSceneMeshes();
            return false;
        }
        if (uploadScene.textures.size() > importedTextureLimit) {
            VOX_LOGW("render") << "imported texture set truncated by bindless capacity: uploaded "
                               << uploadedTextureCount << " of " << uploadScene.textures.size();
        } else if (uploadedTextureCount > 0u) {
            VOX_LOGI("render") << "uploaded imported textures: " << uploadedTextureCount;
        }
    } else if (!uploadScene.textures.empty()) {
        VOX_LOGW("render") << "imported textures unavailable because bindless texture sampling is not ready";
    }
    m_importedTextureSlots = importedTextureSlots;

    std::vector<ImportedMeshVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<ImportedMeshDraw> draws;
    std::vector<ImportedWaterVertex> waterVertices;
    std::vector<std::uint32_t> waterIndices;
    vertices.reserve(uploadScene.packedVertices.size());
    indices.reserve(uploadScene.packedIndices.size());
    draws.reserve(uploadScene.packedDraws.size());
    waterVertices.reserve(uploadScene.waterPatches.size() * 4u);
    waterIndices.reserve(uploadScene.waterPatches.size() * 6u);
    m_importedLocalLights.clear();
    m_importedLocalLights.reserve(uploadScene.lights.size());
    for (const odai::importer::ImportedSceneLight& sceneLight : uploadScene.lights) {
        if (sceneLight.radius <= 0.0f || sceneLight.intensity <= 0.0f) {
            continue;
        }
        ImportedLocalLight light{};
        std::memcpy(light.position, sceneLight.position, sizeof(light.position));
        light.color[0] = std::clamp(sceneLight.color[0], 0.0f, 8.0f);
        light.color[1] = std::clamp(sceneLight.color[1], 0.0f, 8.0f);
        light.color[2] = std::clamp(sceneLight.color[2], 0.0f, 8.0f);
        light.radius = sceneLight.radius;
        light.intensity = sceneLight.intensity;
        m_importedLocalLights.push_back(light);
    }

    for (const odai::importer::ImportedScenePackedVertex& srcVertex : uploadScene.packedVertices) {
        ImportedMeshVertex dstVertex{};
        std::memcpy(dstVertex.position, srcVertex.position, sizeof(dstVertex.position));
        std::memcpy(dstVertex.normal, srcVertex.normal, sizeof(dstVertex.normal));
        std::memcpy(dstVertex.color, srcVertex.color, sizeof(dstVertex.color));
        std::memcpy(dstVertex.uv, srcVertex.uv, sizeof(dstVertex.uv));
        dstVertex.flags = srcVertex.flags;
        if (srcVertex.textureIndex < importedTextureSlots.size()) {
            dstVertex.textureIndex = importedTextureSlots[srcVertex.textureIndex];
        } else {
            dstVertex.textureIndex = std::numeric_limits<std::uint32_t>::max();
        }
        vertices.push_back(dstVertex);
    }
    indices.assign(uploadScene.packedIndices.begin(), uploadScene.packedIndices.end());
    const bool importedSceneIsInterior = uploadScene.sourceTag == "morrowind_interior";
    const std::uint32_t sourceTerrainDrawCount = importedSceneIsInterior
        ? 0u
        : std::min<std::uint32_t>(
            uploadScene.sourceLandscapeCellCount,
            static_cast<std::uint32_t>(uploadScene.packedDraws.size()));
    constexpr std::uint32_t kInvalidImportedPageRangeIndex = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> sourceDrawPageRangeIndices(
        uploadScene.packedDraws.size(),
        kInvalidImportedPageRangeIndex);
    std::vector<ImportedScenePageDrawRange> pageDrawRanges;
    if (gpuScene != nullptr && !gpuScene->renderCache.pageDrawRanges.empty() && !gpuScene->pages.empty()) {
        pageDrawRanges.reserve(gpuScene->renderCache.pageDrawRanges.size());
        for (const odai::importer::GpuScenePageDrawRange& sourceRange : gpuScene->renderCache.pageDrawRanges) {
            if (sourceRange.drawCount == 0u ||
                sourceRange.firstDraw >= uploadScene.packedDraws.size() ||
                sourceRange.pageIndex >= gpuScene->pages.size()) {
                continue;
            }
            ImportedScenePageDrawRange rendererRange{};
            const odai::importer::GpuSceneBounds& bounds = gpuScene->pages[sourceRange.pageIndex].bounds;
            std::memcpy(rendererRange.boundsMin, bounds.min, sizeof(rendererRange.boundsMin));
            std::memcpy(rendererRange.boundsMax, bounds.max, sizeof(rendererRange.boundsMax));
            const std::uint32_t rendererRangeIndex = static_cast<std::uint32_t>(pageDrawRanges.size());
            pageDrawRanges.push_back(rendererRange);

            const std::uint32_t sourceDrawEnd = static_cast<std::uint32_t>(std::min<std::size_t>(
                static_cast<std::size_t>(sourceRange.firstDraw) + static_cast<std::size_t>(sourceRange.drawCount),
                uploadScene.packedDraws.size()));
            for (std::uint32_t drawIndex = sourceRange.firstDraw; drawIndex < sourceDrawEnd; ++drawIndex) {
                sourceDrawPageRangeIndices[drawIndex] = rendererRangeIndex;
            }
        }

        bool pageRangesCoverDraws = !pageDrawRanges.empty();
        for (std::uint32_t drawIndex = 0; drawIndex < uploadScene.packedDraws.size(); ++drawIndex) {
            if (uploadScene.packedDraws[drawIndex].indexCount != 0u &&
                sourceDrawPageRangeIndices[drawIndex] == kInvalidImportedPageRangeIndex) {
                pageRangesCoverDraws = false;
                break;
            }
        }
        if (!pageRangesCoverDraws) {
            pageDrawRanges.clear();
            std::fill(
                sourceDrawPageRangeIndices.begin(),
                sourceDrawPageRangeIndices.end(),
                kInvalidImportedPageRangeIndex);
        }
    } else if (gpuScene == nullptr && !uploadScene.pageRanges.empty()) {
        // Native page ranges supplied directly on the ImportedScene (e.g. the hex
        // strategy map emits one page per chunk). Mirror the GpuScene translation so
        // the same downstream per-page frustum-cull consumer is reused unchanged.
        pageDrawRanges.reserve(uploadScene.pageRanges.size());
        for (const odai::importer::ImportedScenePageRange& sourceRange : uploadScene.pageRanges) {
            if (sourceRange.drawCount == 0u ||
                sourceRange.firstDraw >= uploadScene.packedDraws.size()) {
                continue;
            }
            ImportedScenePageDrawRange rendererRange{};
            std::memcpy(rendererRange.boundsMin, sourceRange.boundsMin, sizeof(rendererRange.boundsMin));
            std::memcpy(rendererRange.boundsMax, sourceRange.boundsMax, sizeof(rendererRange.boundsMax));
            const std::uint32_t rendererRangeIndex = static_cast<std::uint32_t>(pageDrawRanges.size());
            pageDrawRanges.push_back(rendererRange);

            const std::uint32_t sourceDrawEnd = static_cast<std::uint32_t>(std::min<std::size_t>(
                static_cast<std::size_t>(sourceRange.firstDraw) + static_cast<std::size_t>(sourceRange.drawCount),
                uploadScene.packedDraws.size()));
            for (std::uint32_t drawIndex = sourceRange.firstDraw; drawIndex < sourceDrawEnd; ++drawIndex) {
                sourceDrawPageRangeIndices[drawIndex] = rendererRangeIndex;
            }
        }

        bool pageRangesCoverDraws = !pageDrawRanges.empty();
        for (std::uint32_t drawIndex = 0; drawIndex < uploadScene.packedDraws.size(); ++drawIndex) {
            if (uploadScene.packedDraws[drawIndex].indexCount != 0u &&
                sourceDrawPageRangeIndices[drawIndex] == kInvalidImportedPageRangeIndex) {
                pageRangesCoverDraws = false;
                break;
            }
        }
        if (!pageRangesCoverDraws) {
            pageDrawRanges.clear();
            std::fill(
                sourceDrawPageRangeIndices.begin(),
                sourceDrawPageRangeIndices.end(),
                kInvalidImportedPageRangeIndex);
        }
    }
    std::uint32_t mergedTerrainDrawCount = 0;
    bool lastMergedDrawWasTerrain = false;
    std::uint32_t lastMergedPageRangeIndex = kInvalidImportedPageRangeIndex;
    auto appendMergedDraw = [&](
                                std::uint32_t firstIndex,
                                std::uint32_t indexCount,
                                bool terrainDraw,
                                std::uint32_t pageRangeIndex
                            ) {
        if (indexCount == 0) {
            return;
        }
        if (!draws.empty()) {
            ImportedMeshDraw& previous = draws.back();
            if (lastMergedDrawWasTerrain == terrainDraw &&
                lastMergedPageRangeIndex == pageRangeIndex &&
                previous.firstIndex + previous.indexCount == firstIndex) {
                previous.indexCount += indexCount;
                return;
            }
        }
        ImportedMeshDraw draw{};
        draw.firstIndex = firstIndex;
        draw.indexCount = indexCount;
        const std::uint32_t rendererDrawIndex = static_cast<std::uint32_t>(draws.size());
        draws.push_back(draw);
        if (terrainDraw) {
            ++mergedTerrainDrawCount;
        }
        if (pageRangeIndex != kInvalidImportedPageRangeIndex && pageRangeIndex < pageDrawRanges.size()) {
            ImportedScenePageDrawRange& pageRange = pageDrawRanges[pageRangeIndex];
            if (pageRange.drawCount == 0u) {
                pageRange.firstDraw = rendererDrawIndex;
            }
            ++pageRange.drawCount;
            if (terrainDraw) {
                ++pageRange.terrainDrawCount;
            }
        }
        lastMergedDrawWasTerrain = terrainDraw;
        lastMergedPageRangeIndex = pageRangeIndex;
    };

    for (std::uint32_t drawIndex = 0; drawIndex < uploadScene.packedDraws.size(); ++drawIndex) {
        const odai::importer::ImportedScenePackedDraw& srcDraw = uploadScene.packedDraws[drawIndex];
        if (srcDraw.indexCount == 0) {
            continue;
        }
        appendMergedDraw(
            srcDraw.firstIndex,
            srcDraw.indexCount,
            drawIndex < sourceTerrainDrawCount,
            sourceDrawPageRangeIndices[drawIndex]);
    }
    for (const odai::importer::ImportedSceneWaterPatch& patch : uploadScene.waterPatches) {
        const std::uint32_t baseVertex = static_cast<std::uint32_t>(waterVertices.size());
        ImportedWaterVertex vertex{};
        vertex.position[0] = patch.originX;
        vertex.position[1] = patch.waterLevel;
        vertex.position[2] = patch.originZ;
        vertex.uv[0] = 0.0f;
        vertex.uv[1] = 0.0f;
        waterVertices.push_back(vertex);

        vertex.position[0] = patch.originX + patch.sizeX;
        vertex.position[2] = patch.originZ;
        vertex.uv[0] = 1.0f;
        vertex.uv[1] = 0.0f;
        waterVertices.push_back(vertex);

        vertex.position[0] = patch.originX + patch.sizeX;
        vertex.position[2] = patch.originZ + patch.sizeZ;
        vertex.uv[0] = 1.0f;
        vertex.uv[1] = 1.0f;
        waterVertices.push_back(vertex);

        vertex.position[0] = patch.originX;
        vertex.position[2] = patch.originZ + patch.sizeZ;
        vertex.uv[0] = 0.0f;
        vertex.uv[1] = 1.0f;
        waterVertices.push_back(vertex);

        waterIndices.push_back(baseVertex + 0u);
        waterIndices.push_back(baseVertex + 2u);
        waterIndices.push_back(baseVertex + 1u);
        waterIndices.push_back(baseVertex + 0u);
        waterIndices.push_back(baseVertex + 3u);
        waterIndices.push_back(baseVertex + 2u);
    }

    if (vertices.empty() || indices.empty()) {
        VOX_LOGW("render") << "imported scene upload skipped because it produced no renderable geometry";
        return true;
    }

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
            logVkFailure("vkCreateCommandPool(importedGeometryUpload)", result);
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
                logVkFailure("vkAllocateCommandBuffers(importedGeometryUpload)", result);
                uploadFailed = true;
            }
        }

        if (!uploadFailed) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
            if (result != VK_SUCCESS) {
                logVkFailure("vkBeginCommandBuffer(importedGeometryUpload)", result);
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
                logVkFailure("vkEndCommandBuffer(importedGeometryUpload)", result);
                uploadFailed = true;
            }
        }

        if (!uploadFailed) {
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            result = vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            if (result != VK_SUCCESS) {
                logVkFailure("vkQueueSubmit(importedGeometryUpload)", result);
                uploadFailed = true;
            }
        }
        if (!uploadFailed) {
            result = vkQueueWaitIdle(m_graphicsQueue);
            if (result != VK_SUCCESS) {
                logVkFailure("vkQueueWaitIdle(importedGeometryUpload)", result);
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

    BufferHandle newVertexHandle = kInvalidBufferHandle;
    if (!uploadDeviceLocalBuffer(
            vertices.data(),
            static_cast<VkDeviceSize>(vertices.size() * sizeof(ImportedMeshVertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            "imported scene vertex",
            newVertexHandle)) {
        return false;
    }

    BufferHandle newIndexHandle = kInvalidBufferHandle;
    if (!uploadDeviceLocalBuffer(
            indices.data(),
            static_cast<VkDeviceSize>(indices.size() * sizeof(std::uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            "imported scene index",
            newIndexHandle)) {
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    m_importedVertexBufferHandle = newVertexHandle;
    m_importedIndexBufferHandle = newIndexHandle;
    m_importedIndexCount = static_cast<std::uint32_t>(indices.size());
    m_importedTerrainDrawCount = std::min<std::uint32_t>(mergedTerrainDrawCount, static_cast<std::uint32_t>(draws.size()));
    m_importedStaticDrawCount = static_cast<std::uint32_t>(draws.size()) - std::min(m_importedTerrainDrawCount, static_cast<std::uint32_t>(draws.size()));
    for (ImportedMeshDraw& draw : draws) {
        draw.vertexBufferHandle = newVertexHandle;
        draw.indexBufferHandle = newIndexHandle;
        m_importedMeshDraws.push_back(draw);
    }
    m_importedPageDrawRanges = std::move(pageDrawRanges);

    const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(newVertexHandle);
    const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(newIndexHandle);
    if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE) {
        VOX_LOGE("render") << "imported scene upload produced null Vulkan buffers"
                           << " (vertexHandle=" << newVertexHandle
                           << ", indexHandle=" << newIndexHandle << ")";
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        m_bufferAllocator.destroyBuffer(newIndexHandle);
        m_importedVertexBufferHandle = kInvalidBufferHandle;
        m_importedIndexBufferHandle = kInvalidBufferHandle;
        m_importedMeshDraws.clear();
        m_importedPageDrawRanges.clear();
        m_importedIndexCount = 0;
        m_importedTerrainDrawCount = 0;
        m_importedStaticDrawCount = 0;
        return false;
    }
    if (vertexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), "mesh.importedScene.vertex");
    }
    if (indexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), "mesh.importedScene.index");
    }
    if (!waterVertices.empty() && !waterIndices.empty()) {
        BufferHandle waterVertexHandle = kInvalidBufferHandle;
        if (!uploadDeviceLocalBuffer(
                waterVertices.data(),
                static_cast<VkDeviceSize>(waterVertices.size() * sizeof(ImportedWaterVertex)),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                "imported scene water vertex",
                waterVertexHandle)) {
            VOX_LOGE("render") << "imported scene water vertex buffer upload failed";
        } else {
            BufferHandle waterIndexHandle = kInvalidBufferHandle;
            if (!uploadDeviceLocalBuffer(
                    waterIndices.data(),
                    static_cast<VkDeviceSize>(waterIndices.size() * sizeof(std::uint32_t)),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    "imported scene water index",
                    waterIndexHandle)) {
                VOX_LOGE("render") << "imported scene water index buffer upload failed";
                m_bufferAllocator.destroyBuffer(waterVertexHandle);
            } else {
                m_importedWaterVertexBufferHandle = waterVertexHandle;
                m_importedWaterIndexBufferHandle = waterIndexHandle;
                m_importedWaterIndexCount = static_cast<std::uint32_t>(waterIndices.size());
                const VkBuffer waterVertexBuffer = m_bufferAllocator.getBuffer(waterVertexHandle);
                const VkBuffer waterIndexBuffer = m_bufferAllocator.getBuffer(waterIndexHandle);
                if (waterVertexBuffer != VK_NULL_HANDLE) {
                    setObjectName(
                        VK_OBJECT_TYPE_BUFFER,
                        vkHandleToUint64(waterVertexBuffer),
                        "mesh.importedScene.water.vertex");
                }
                if (waterIndexBuffer != VK_NULL_HANDLE) {
                    setObjectName(
                        VK_OBJECT_TYPE_BUFFER,
                        vkHandleToUint64(waterIndexBuffer),
                        "mesh.importedScene.water.index");
                }
            }
        }
    }
    const std::span<const odai::importer::ImportedScenePackedDraw> allDraws(uploadScene.packedDraws);
    const std::span<const odai::importer::ImportedScenePackedDraw> terrainDraws =
        allDraws.first(std::min<std::size_t>(sourceTerrainDrawCount, allDraws.size()));
    const std::span<const odai::importer::ImportedScenePackedDraw> staticDraws =
        allDraws.subspan(std::min<std::size_t>(sourceTerrainDrawCount, allDraws.size()));
    const ImportedDrawBounds terrainBounds =
        computeImportedDrawBounds(uploadScene.packedVertices, uploadScene.packedIndices, terrainDraws);
    const ImportedDrawBounds staticBounds =
        computeImportedDrawBounds(uploadScene.packedVertices, uploadScene.packedIndices, staticDraws);
    m_importedGiTriangles.clear();
    if (importedSceneIsInterior) {
        constexpr std::size_t kImportedGiTriangleLimit = 300000u;
        m_importedGiTriangles.reserve(
            std::min<std::size_t>(uploadScene.packedIndices.size() / 3u, kImportedGiTriangleLimit));
        for (const odai::importer::ImportedScenePackedDraw& draw : staticDraws) {
            const std::size_t indexEnd =
                static_cast<std::size_t>(draw.firstIndex) + static_cast<std::size_t>(draw.indexCount);
            if (draw.indexCount < 3u || indexEnd > uploadScene.packedIndices.size()) {
                continue;
            }
            for (std::size_t indexOffset = draw.firstIndex; indexOffset + 2u < indexEnd; indexOffset += 3u) {
                const std::uint32_t i0 = uploadScene.packedIndices[indexOffset + 0u];
                const std::uint32_t i1 = uploadScene.packedIndices[indexOffset + 1u];
                const std::uint32_t i2 = uploadScene.packedIndices[indexOffset + 2u];
                if (i0 >= uploadScene.packedVertices.size() ||
                    i1 >= uploadScene.packedVertices.size() ||
                    i2 >= uploadScene.packedVertices.size()) {
                    continue;
                }

                const odai::importer::ImportedScenePackedVertex& v0 = uploadScene.packedVertices[i0];
                const odai::importer::ImportedScenePackedVertex& v1 = uploadScene.packedVertices[i1];
                const odai::importer::ImportedScenePackedVertex& v2 = uploadScene.packedVertices[i2];
                ImportedGiTriangle triangle{};
                std::memcpy(triangle.p0, v0.position, sizeof(triangle.p0));
                std::memcpy(triangle.p1, v1.position, sizeof(triangle.p1));
                std::memcpy(triangle.p2, v2.position, sizeof(triangle.p2));
                const std::array<float, 3> c0 = sampleImportedTextureBaseColor(uploadScene.textures, v0);
                const std::array<float, 3> c1 = sampleImportedTextureBaseColor(uploadScene.textures, v1);
                const std::array<float, 3> c2 = sampleImportedTextureBaseColor(uploadScene.textures, v2);
                triangle.albedo[0] = (c0[0] + c1[0] + c2[0]) * (1.0f / 3.0f);
                triangle.albedo[1] = (c0[1] + c1[1] + c2[1]) * (1.0f / 3.0f);
                triangle.albedo[2] = (c0[2] + c1[2] + c2[2]) * (1.0f / 3.0f);
                m_importedGiTriangles.push_back(triangle);
                if (m_importedGiTriangles.size() >= kImportedGiTriangleLimit) {
                    break;
                }
            }
            if (m_importedGiTriangles.size() >= kImportedGiTriangleLimit) {
                break;
            }
        }
    }
    ImportedDrawBounds waterBounds{};
    for (const ImportedWaterVertex& vertex : waterVertices) {
        expandImportedBounds(
            waterBounds,
            vertex.position[0],
            vertex.position[1],
            vertex.position[2]);
    }
    VOX_LOGI("render") << "uploaded imported scene geometry (vertices=" << vertices.size()
                       << ", indices=" << indices.size()
                       << ", draws=" << m_importedMeshDraws.size()
                       << ", pageRanges=" << m_importedPageDrawRanges.size()
                       << ", instances=" << uploadScene.instances.size()
                       << ", terrainCells=" << uploadScene.landscapeCells.size()
                       << ", waterPatches=" << uploadScene.waterPatches.size()
                       << ", lights=" << m_importedLocalLights.size() << ")";
    if (terrainBounds.valid) {
        VOX_LOGI("render") << "imported terrain bounds min=("
                           << terrainBounds.min[0] << ", " << terrainBounds.min[1] << ", " << terrainBounds.min[2]
                           << ") max=("
                           << terrainBounds.max[0] << ", " << terrainBounds.max[1] << ", " << terrainBounds.max[2]
                           << ") draws=" << terrainDraws.size();
    }
    if (staticBounds.valid) {
        VOX_LOGI("render") << "imported static bounds min=("
                           << staticBounds.min[0] << ", " << staticBounds.min[1] << ", " << staticBounds.min[2]
                           << ") max=("
                           << staticBounds.max[0] << ", " << staticBounds.max[1] << ", " << staticBounds.max[2]
                           << ") draws=" << staticDraws.size();
    } else {
        VOX_LOGW("render") << "imported scene contained no static bounds after upload";
    }
    m_voxelGiWorldDirty = false;
    m_voxelGiOccupancyFullRebuildInProgress = false;
    m_voxelGiOccupancyFullRebuildNeedsClear = false;
    m_voxelGiOccupancyFullRebuildCursor = 0;
    m_voxelGiDirtyChunkIndices.clear();
    if (!m_importedGiTriangles.empty()) {
        m_debugImportedGiTriangleCount = static_cast<std::uint32_t>(
            std::min<std::size_t>(
                m_importedGiTriangles.size(),
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
        m_voxelGiWorldDirty = true;
        ++m_voxelGiWorldVersion;
        m_voxelGiOccupancyFullRebuildInProgress = true;
        m_voxelGiOccupancyFullRebuildNeedsClear = true;
        VOX_LOGI("render") << "imported interior GI source triangles="
                           << m_importedGiTriangles.size();
    }
    if (m_importedWaterIndexCount > 0) {
        VOX_LOGI("render") << "imported water geometry uploaded (vertices=" << waterVertices.size()
                           << ", indices=" << waterIndices.size()
                           << ", patches=" << uploadScene.waterPatches.size() << ")";
        if (waterBounds.valid) {
            VOX_LOGI("render") << "imported water bounds min=("
                               << waterBounds.min[0] << ", " << waterBounds.min[1] << ", " << waterBounds.min[2]
                               << ") max=("
                               << waterBounds.max[0] << ", " << waterBounds.max[1] << ", " << waterBounds.max[2]
                               << ")";
        }
    }
    if (m_rayTracingCapabilityProbe.rayTracingCoreReady) {
        auto appendImportedRtRecord = [&](
                                          std::span<const odai::importer::ImportedScenePackedDraw> sourceDraws,
                                          const char* debugName
                                      ) {
            if (sourceDraws.empty()) {
                return;
            }
            RtImportedSceneRecord record{};
            record.debugName = debugName;
            if (!createImportedRtGeometryBuffers(
                    m_bufferAllocator,
                    uploadScene.packedVertices,
                    uploadScene.packedIndices,
                    sourceDraws,
                    record.geometry
                )) {
                VOX_LOGE("render") << debugName << " RT geometry buffer allocation failed";
                return;
            }
            record.geometryResident =
                record.geometry.vertexCount > 0 && record.geometry.indexCount > 0;
            record.dirty = record.geometryResident;
            if (record.geometryResident) {
                m_rtImportedSceneRecords.push_back(record);
            }
        };

        appendImportedRtRecord(terrainDraws, "imported terrain");
        appendImportedRtRecord(staticDraws, "imported statics");
        if (!m_rtImportedSceneRecords.empty()) {
            VOX_LOGI("render") << "imported RT geometry prepared (records="
                               << m_rtImportedSceneRecords.size() << ")";
            markRayTracingSceneDirty();
        }
    } else {
        refreshShadowStats();
    }
    return true;
}


void RendererBackend::setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba) {
    m_voxelBaseColorPaletteRgba = paletteRgba;
}


bool RendererBackend::uploadMagicaVoxelMesh(
    const odai::world::ChunkMeshData& mesh,
    float worldOffsetX,
    float worldOffsetY,
    float worldOffsetZ
) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }

    if (mesh.vertices.empty() || mesh.indices.empty()) {
        return false;
    }

    BufferCreateDesc vertexCreateDesc{};
    vertexCreateDesc.size = static_cast<VkDeviceSize>(mesh.vertices.size() * sizeof(odai::world::PackedVoxelVertex));
    vertexCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vertexCreateDesc.initialData = mesh.vertices.data();
    const BufferHandle newVertexHandle = m_bufferAllocator.createBuffer(vertexCreateDesc);
    if (newVertexHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "magica voxel vertex buffer allocation failed";
        return false;
    }

    BufferCreateDesc indexCreateDesc{};
    indexCreateDesc.size = static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(std::uint32_t));
    indexCreateDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    indexCreateDesc.initialData = mesh.indices.data();
    const BufferHandle newIndexHandle = m_bufferAllocator.createBuffer(indexCreateDesc);
    if (newIndexHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "magica voxel index buffer allocation failed";
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(newVertexHandle);
    if (vertexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), "mesh.magicaVoxel.vertex");
    }
    const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(newIndexHandle);
    if (indexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), "mesh.magicaVoxel.index");
    }

    MagicaMeshDraw draw{};
    draw.vertexBufferHandle = newVertexHandle;
    draw.indexBufferHandle = newIndexHandle;
    draw.indexCount = static_cast<uint32_t>(mesh.indices.size());
    draw.offsetX = worldOffsetX;
    draw.offsetY = worldOffsetY;
    draw.offsetZ = worldOffsetZ;
    m_magicaMeshDraws.push_back(draw);
    if (m_rayTracingCapabilityProbe.rayTracingCoreReady) {
        std::vector<RtVertex> rtVertices;
        rtVertices.reserve(mesh.vertices.size());
        for (const odai::world::PackedVoxelVertex& vertex : mesh.vertices) {
            rtVertices.push_back(decodePackedVoxelVertexPosition(vertex.bits, worldOffsetX, worldOffsetY, worldOffsetZ));
        }
        RtGeometryBuffers rtGeometry{};
        if (!createRtGeometryBuffers(m_bufferAllocator, rtVertices, mesh.indices, rtGeometry)) {
            VOX_LOGE("render") << "magica voxel RT geometry buffer allocation failed";
        } else {
            m_rtMagicaGeometries.push_back(rtGeometry);
            markRayTracingSceneDirty();
            if (rayTracingRuntimeReady() && !rebuildRayTracingScene()) {
                VOX_LOGE("render") << "magica voxel RT scene rebuild failed";
            }
        }
    }
    return true;
}


bool RendererBackend::updateChunkMesh(const odai::world::ChunkGrid& chunkGrid) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    (void)chunkGrid;
    m_chunkMeshRebuildRequested = true;
    m_pendingChunkRemeshKeys.clear();
    m_voxelGiWorldDirty = true;
    ++m_voxelGiWorldVersion;
    m_voxelGiOccupancyFullRebuildInProgress = true;
    m_voxelGiOccupancyFullRebuildNeedsClear = true;
    m_voxelGiOccupancyFullRebuildCursor = 0;
    m_voxelGiDirtyChunkIndices.clear();
    return true;
}


bool RendererBackend::updateChunkMesh(const odai::world::ChunkGrid& chunkGrid, std::size_t chunkIndex) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    if (chunkIndex >= chunkGrid.chunks().size()) {
        return false;
    }
    if (m_chunkMeshRebuildRequested) {
        return true;
    }
    const ChunkResidentKey remeshKey = chunkResidentKeyForChunk(chunkGrid.chunks()[chunkIndex]);
    if (std::find(m_pendingChunkRemeshKeys.begin(), m_pendingChunkRemeshKeys.end(), remeshKey) ==
        m_pendingChunkRemeshKeys.end()) {
        m_pendingChunkRemeshKeys.push_back(remeshKey);
    }
    if (!m_voxelGiOccupancyFullRebuildInProgress &&
        std::find(m_voxelGiDirtyChunkIndices.begin(), m_voxelGiDirtyChunkIndices.end(), chunkIndex) ==
            m_voxelGiDirtyChunkIndices.end()) {
        m_voxelGiDirtyChunkIndices.push_back(chunkIndex);
    }
    m_voxelGiWorldDirty = true;
    ++m_voxelGiWorldVersion;
    return true;
}


bool RendererBackend::updateChunkMesh(const odai::world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    if (chunkIndices.empty()) {
        return true;
    }
    if (m_chunkMeshRebuildRequested) {
        return true;
    }
    for (const std::size_t chunkIndex : chunkIndices) {
        if (chunkIndex >= chunkGrid.chunks().size()) {
            return false;
        }
        const ChunkResidentKey remeshKey = chunkResidentKeyForChunk(chunkGrid.chunks()[chunkIndex]);
        if (std::find(m_pendingChunkRemeshKeys.begin(), m_pendingChunkRemeshKeys.end(), remeshKey) ==
            m_pendingChunkRemeshKeys.end()) {
            m_pendingChunkRemeshKeys.push_back(remeshKey);
        }
        if (!m_voxelGiOccupancyFullRebuildInProgress &&
            std::find(m_voxelGiDirtyChunkIndices.begin(), m_voxelGiDirtyChunkIndices.end(), chunkIndex) ==
                m_voxelGiDirtyChunkIndices.end()) {
            m_voxelGiDirtyChunkIndices.push_back(chunkIndex);
        }
    }
    m_voxelGiWorldDirty = true;
    ++m_voxelGiWorldVersion;
    return true;
}


bool RendererBackend::useSpatialPartitioningQueries() const {
    return m_debugEnableSpatialQueries;
}

odai::world::ClipmapConfig RendererBackend::clipmapQueryConfig() const {
    return m_debugClipmapConfig;
}


void RendererBackend::setSpatialQueryStats(
    bool used,
    const odai::world::SpatialQueryStats& stats,
    std::uint32_t visibleChunkCount
) {
    m_debugSpatialQueriesUsed = used;
    m_debugSpatialQueryStats = stats;
    m_debugSpatialVisibleChunkCount = visibleChunkCount;
}


bool RendererBackend::createChunkBuffers(const odai::world::ChunkGrid& chunkGrid, std::span<const std::size_t> remeshChunkIndices) {
    if (chunkGrid.chunks().empty()) {
        m_chunkDrawRanges.clear();
        m_chunkResidentKeys.clear();
        m_chunkLodMeshCache.clear();
        m_chunkGrassInstanceCache.clear();
        m_rtChunkSceneRecords.clear();
        m_chunkLodMeshCacheValid = false;
        m_debugChunkMeshVertexCount = 0;
        m_debugChunkMeshIndexCount = 0;
        m_debugChunkLastRemeshedChunkCount = 0;
        m_debugChunkLastRemeshActiveVertexCount = 0;
        m_debugChunkLastRemeshActiveIndexCount = 0;
        m_debugChunkLastRemeshNaiveVertexCount = 0;
        m_debugChunkLastRemeshNaiveIndexCount = 0;
        m_debugChunkLastRemeshMs = 0.0f;
        m_debugChunkLastRemeshReductionPercent = 0.0f;
        m_debugChunkLastFullRemeshMs = 0.0f;
        m_debugRtActiveChunkCount = 0;
        m_rtDirtyChunkCount = 0;

        collectCompletedBufferReleases();
        if (m_transferCommandBufferInFlightValue > 0 && !isTimelineValueReached(m_transferCommandBufferInFlightValue)) {
            return false;
        }
        m_transferCommandBufferInFlightValue = 0;

        if (m_chunkVertexBufferHandle != kInvalidBufferHandle) {
            if (m_lastGraphicsTimelineValue == 0) {
                m_bufferAllocator.destroyBuffer(m_chunkVertexBufferHandle);
            } else {
                scheduleBufferRelease(m_chunkVertexBufferHandle, m_lastGraphicsTimelineValue);
            }
            m_chunkVertexBufferHandle = kInvalidBufferHandle;
        }
        if (m_chunkIndexBufferHandle != kInvalidBufferHandle) {
            if (m_lastGraphicsTimelineValue == 0) {
                m_bufferAllocator.destroyBuffer(m_chunkIndexBufferHandle);
            } else {
                scheduleBufferRelease(m_chunkIndexBufferHandle, m_lastGraphicsTimelineValue);
            }
            m_chunkIndexBufferHandle = kInvalidBufferHandle;
        }
        if (m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
            if (m_lastGraphicsTimelineValue == 0) {
                m_bufferAllocator.destroyBuffer(m_grassBillboardInstanceBufferHandle);
            } else {
                scheduleBufferRelease(m_grassBillboardInstanceBufferHandle, m_lastGraphicsTimelineValue);
            }
            m_grassBillboardInstanceBufferHandle = kInvalidBufferHandle;
        }
        m_grassBillboardInstanceCount = 0;
        m_pendingTransferTimelineValue = 0;
        m_currentChunkReadyTimelineValue = 0;
        return true;
    }

    const std::vector<odai::world::Chunk>& chunks = chunkGrid.chunks();
    const std::vector<ChunkDrawRange> previousChunkDrawRanges = m_chunkDrawRanges;
    const std::uint32_t previousDebugChunkMeshVertexCount = m_debugChunkMeshVertexCount;
    const std::uint32_t previousDebugChunkMeshIndexCount = m_debugChunkMeshIndexCount;
    auto rollbackChunkDrawState = [&]() {
        m_chunkDrawRanges = previousChunkDrawRanges;
        m_debugChunkMeshVertexCount = previousDebugChunkMeshVertexCount;
        m_debugChunkMeshIndexCount = previousDebugChunkMeshIndexCount;
    };
    const std::size_t expectedDrawRangeCount = chunks.size() * odai::world::kChunkMeshLodCount;
    if (m_chunkDrawRanges.size() != expectedDrawRangeCount) {
        m_chunkDrawRanges.assign(expectedDrawRangeCount, ChunkDrawRange{});
    }
    const std::vector<ChunkResidentKey> previousResidentKeys = std::move(m_chunkResidentKeys);
    const std::vector<odai::world::ChunkLodMeshes> previousChunkLodMeshCache = std::move(m_chunkLodMeshCache);
    const std::vector<std::vector<GrassBillboardInstance>> previousChunkGrassInstanceCache = std::move(m_chunkGrassInstanceCache);
    std::vector<RtChunkSceneRecord> previousRtChunkSceneRecords = std::move(m_rtChunkSceneRecords);
    int previousResidentCenterChunkX = 0;
    int previousResidentCenterChunkZ = 0;
    if (!previousResidentKeys.empty()) {
        int previousMinChunkX = std::numeric_limits<int>::max();
        int previousMaxChunkX = std::numeric_limits<int>::min();
        int previousMinChunkZ = std::numeric_limits<int>::max();
        int previousMaxChunkZ = std::numeric_limits<int>::min();
        for (const ChunkResidentKey& key : previousResidentKeys) {
            previousMinChunkX = std::min(previousMinChunkX, key.chunkX);
            previousMaxChunkX = std::max(previousMaxChunkX, key.chunkX);
            previousMinChunkZ = std::min(previousMinChunkZ, key.chunkZ);
            previousMaxChunkZ = std::max(previousMaxChunkZ, key.chunkZ);
        }
        previousResidentCenterChunkX = (previousMinChunkX + previousMaxChunkX) / 2;
        previousResidentCenterChunkZ = (previousMinChunkZ + previousMaxChunkZ) / 2;
    }

    m_chunkResidentKeys.assign(chunks.size(), ChunkResidentKey{});
    m_chunkLodMeshCache.assign(chunks.size(), odai::world::ChunkLodMeshes{});
    m_chunkGrassInstanceCache.assign(chunks.size(), std::vector<GrassBillboardInstance>{});
    m_rtChunkSceneRecords.assign(chunks.size(), RtChunkSceneRecord{});

    std::vector<std::uint8_t> remeshMask(chunks.size(), 0u);
    bool reusedAnyChunkCache = false;
    bool residentSetChanged = previousResidentKeys.size() != chunks.size();
    for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
        const ChunkResidentKey key = chunkResidentKeyForChunk(chunks[chunkArrayIndex]);
        m_chunkResidentKeys[chunkArrayIndex] = key;

        const auto previousIt = std::find(previousResidentKeys.begin(), previousResidentKeys.end(), key);
        if (previousIt == previousResidentKeys.end()) {
            remeshMask[chunkArrayIndex] = 1u;
            residentSetChanged = true;
            continue;
        }

        const std::size_t previousIndex = static_cast<std::size_t>(std::distance(previousResidentKeys.begin(), previousIt));
        if (previousIndex < previousChunkLodMeshCache.size()) {
            m_chunkLodMeshCache[chunkArrayIndex] = previousChunkLodMeshCache[previousIndex];
            reusedAnyChunkCache = true;
        } else {
            remeshMask[chunkArrayIndex] = 1u;
        }
        if (previousIndex < previousChunkGrassInstanceCache.size()) {
            m_chunkGrassInstanceCache[chunkArrayIndex] = previousChunkGrassInstanceCache[previousIndex];
        }
        if (previousIndex != chunkArrayIndex) {
            residentSetChanged = true;
        }
        const auto previousRtIt = std::find_if(
            previousRtChunkSceneRecords.begin(),
            previousRtChunkSceneRecords.end(),
            [&](const RtChunkSceneRecord& record) { return chunkResidentKeyMatchesRecord(key, record); });
        if (previousRtIt != previousRtChunkSceneRecords.end()) {
            m_rtChunkSceneRecords[chunkArrayIndex] = std::move(*previousRtIt);
            previousRtIt->chunkX = std::numeric_limits<int>::min();
            previousRtIt->chunkY = std::numeric_limits<int>::min();
            previousRtIt->chunkZ = std::numeric_limits<int>::min();
        }
    }
    if (previousResidentKeys.empty() || !reusedAnyChunkCache) {
        m_chunkLodMeshCacheValid = false;
        std::fill(remeshMask.begin(), remeshMask.end(), 1u);
    }
    for (const std::size_t chunkIndex : remeshChunkIndices) {
        if (chunkIndex >= chunks.size()) {
            rollbackChunkDrawState();
            return false;
        }
        remeshMask[chunkIndex] = 1u;
    }

    int minChunkX = std::numeric_limits<int>::max();
    int maxChunkX = std::numeric_limits<int>::min();
    int minChunkZ = std::numeric_limits<int>::max();
    int maxChunkZ = std::numeric_limits<int>::min();
    for (const odai::world::Chunk& chunk : chunks) {
        minChunkX = std::min(minChunkX, chunk.chunkX());
        maxChunkX = std::max(maxChunkX, chunk.chunkX());
        minChunkZ = std::min(minChunkZ, chunk.chunkZ());
        maxChunkZ = std::max(maxChunkZ, chunk.chunkZ());
    }
    const int residentCenterChunkX = (minChunkX + maxChunkX) / 2;
    const int residentCenterChunkZ = (minChunkZ + maxChunkZ) / 2;

    auto rebuildGrassInstancesForChunk = [&](std::size_t chunkArrayIndex) {
        if (chunkArrayIndex >= chunks.size()) {
            return;
        }
        const odai::world::Chunk& chunk = chunks[chunkArrayIndex];
        const int grassDistanceX = std::abs(chunk.chunkX() - residentCenterChunkX);
        const int grassDistanceZ = std::abs(chunk.chunkZ() - residentCenterChunkZ);
        std::vector<GrassBillboardInstance>& grassInstances = m_chunkGrassInstanceCache[chunkArrayIndex];
        const bool previouslyGrassActive = !grassInstances.empty();
        grassInstances.clear();
        const int grassActiveRadius =
            previouslyGrassActive ? kGrassRetainedChunkRadius : kGrassActiveChunkRadius;
        if (grassDistanceX > grassActiveRadius || grassDistanceZ > grassActiveRadius) {
            return;
        }
        grassInstances.reserve(448);

        const float chunkWorldX = static_cast<float>(chunk.chunkX() * odai::world::Chunk::kSizeX);
        const float chunkWorldY = static_cast<float>(chunk.chunkY() * odai::world::Chunk::kSizeY);
        const float chunkWorldZ = static_cast<float>(chunk.chunkZ() * odai::world::Chunk::kSizeZ);

        for (int y = 0; y < odai::world::Chunk::kSizeY - 1; ++y) {
            for (int z = 0; z < odai::world::Chunk::kSizeZ; ++z) {
                for (int x = 0; x < odai::world::Chunk::kSizeX; ++x) {
                    if (chunk.voxelAt(x, y, z).type != odai::world::VoxelType::Grass) {
                        continue;
                    }
                    if (chunk.voxelAt(x, y + 1, z).type != odai::world::VoxelType::Empty) {
                        continue;
                    }

                    const std::uint32_t hash =
                        static_cast<std::uint32_t>(x * 73856093) ^
                        static_cast<std::uint32_t>(y * 19349663) ^
                        static_cast<std::uint32_t>(z * 83492791) ^
                        static_cast<std::uint32_t>((chunk.chunkX() + 101) * 2654435761u) ^
                        static_cast<std::uint32_t>((chunk.chunkZ() + 193) * 2246822519u);
                    // Keep grass sparse and deterministic so placement feels natural and stable.
                    if ((hash % 100u) >= 22u) {
                        continue;
                    }
                    const int clumpCount = 2 + static_cast<int>((hash >> 24u) & 0x1u);
                    for (int clumpIndex = 0; clumpIndex < clumpCount; ++clumpIndex) {
                        const std::uint32_t clumpHash = hash ^ (0x9E3779B9u * static_cast<std::uint32_t>(clumpIndex + 1));
                        const float rand0 = static_cast<float>(clumpHash & 0xFFu) / 255.0f;
                        const float rand1 = static_cast<float>((clumpHash >> 8u) & 0xFFu) / 255.0f;
                        const float rand2 = static_cast<float>((clumpHash >> 16u) & 0xFFu) / 255.0f;
                        const float rand3 = static_cast<float>((clumpHash >> 24u) & 0xFFu) / 255.0f;
                        const std::uint32_t tintHash = clumpHash ^ 0x85EBCA6Bu;
                        const float tintRand0 = static_cast<float>(tintHash & 0xFFu) / 255.0f;
                        const float tintRand1 = static_cast<float>((tintHash >> 8u) & 0xFFu) / 255.0f;
                        const float tintRand2 = static_cast<float>((tintHash >> 16u) & 0xFFu) / 255.0f;
                        const float radial = 0.06f + (0.18f * rand2);
                        const float angle = rand1 * (2.0f * 3.14159265f);
                        const float jitterX = std::cos(angle) * radial;
                        const float jitterZ = std::sin(angle) * radial;
                        const float yawRadians = rand0 * (2.0f * 3.14159265f);
                        const float yJitter = rand3 * 0.08f;

                        GrassBillboardInstance instance{};
                        instance.worldPosYaw[0] = chunkWorldX + static_cast<float>(x) + 0.5f + jitterX;
                        // Lift slightly above the supporting voxel top to avoid depth tie flicker.
                        instance.worldPosYaw[1] = chunkWorldY + static_cast<float>(y) + 1.02f + yJitter;
                        instance.worldPosYaw[2] = chunkWorldZ + static_cast<float>(z) + 0.5f + jitterZ;
                        instance.worldPosYaw[3] = yawRadians;
                        // Mostly green bushes, with some flowers.
                        const bool placeFlower = ((clumpHash >> 5u) % 100u) < 18u;
                        if (placeFlower) {
                            // Bias strongly toward poppies (tiles 1-2), with rarer lighter wildflowers (3-4).
                            const bool choosePoppy = ((clumpHash >> 13u) % 100u) < 74u;
                            const std::uint32_t flowerTile = choosePoppy
                                ? (1u + ((clumpHash >> 9u) & 0x1u))
                                : (3u + ((clumpHash >> 10u) & 0x1u));
                            if (choosePoppy) {
                                const float poppyBoost = 0.96f + (tintRand1 * 0.10f);
                                instance.colorTint[0] = (0.92f + (tintRand0 * 0.14f)) * poppyBoost;
                                instance.colorTint[1] = (0.92f + (tintRand2 * 0.14f)) * poppyBoost;
                                instance.colorTint[2] = (0.92f + (tintRand1 * 0.14f)) * poppyBoost;
                            } else {
                                const float flowerBoost = 0.94f + (tintRand1 * 0.12f);
                                instance.colorTint[0] = (0.94f + (tintRand0 * 0.14f)) * flowerBoost;
                                instance.colorTint[1] = (0.94f + (tintRand2 * 0.14f)) * flowerBoost;
                                instance.colorTint[2] = (0.94f + (tintRand1 * 0.14f)) * flowerBoost;
                            }
                            instance.colorTint[3] = static_cast<float>(flowerTile);
                        } else {
                            // Golden grass variation.
                            const float warmBias = 0.50f + (0.50f * tintRand0);
                            const float dryBias = tintRand2;
                            const float brightness = 0.82f + (tintRand1 * 0.32f);
                            const float redBase = std::lerp(0.90f, 1.28f, warmBias);
                            const float greenBase = std::lerp(0.98f, 1.36f, (warmBias * 0.70f) + (dryBias * 0.30f));
                            const float blueBase = std::lerp(0.56f, 0.20f, warmBias);
                            instance.colorTint[0] = redBase * brightness;
                            instance.colorTint[1] = greenBase * brightness;
                            instance.colorTint[2] = blueBase * brightness;
                            instance.colorTint[3] = 4.0f;
                        }
                        grassInstances.push_back(instance);
                    }
                }
            }
        }
    };

    std::size_t remeshedChunkCount = 0;
    std::size_t remeshedActiveVertexCount = 0;
    std::size_t remeshedActiveIndexCount = 0;
    std::size_t remeshedNaiveVertexCount = 0;
    std::size_t remeshedNaiveIndexCount = 0;
    const auto countMeshGeometry = [](const odai::world::ChunkLodMeshes& lodMeshes, std::size_t& outVertices, std::size_t& outIndices) {
        for (const odai::world::ChunkMeshData& lodMesh : lodMeshes.lodMeshes) {
            outVertices += lodMesh.vertices.size();
            outIndices += lodMesh.indices.size();
        }
    };
    const bool fullRemesh =
        !m_chunkLodMeshCacheValid ||
        std::all_of(remeshMask.begin(), remeshMask.end(), [](std::uint8_t dirty) { return dirty != 0u; });
    const auto remeshStart = std::chrono::steady_clock::now();
    if (fullRemesh) {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            m_chunkLodMeshCache[chunkArrayIndex] =
                odai::world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions);
            rebuildGrassInstancesForChunk(chunkArrayIndex);
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            if (m_chunkMeshingOptions.mode == odai::world::MeshingMode::Naive) {
                remeshedNaiveVertexCount = remeshedActiveVertexCount;
                remeshedNaiveIndexCount = remeshedActiveIndexCount;
            } else {
                const odai::world::ChunkLodMeshes naiveLodMeshes =
                    odai::world::buildChunkLodMeshes(chunks[chunkArrayIndex], odai::world::MeshingOptions{odai::world::MeshingMode::Naive});
                countMeshGeometry(naiveLodMeshes, remeshedNaiveVertexCount, remeshedNaiveIndexCount);
            }
        }
        remeshedChunkCount = chunks.size();
        m_chunkLodMeshCacheValid = true;
    } else {
        std::vector<std::size_t> uniqueRemeshChunkIndices;
        uniqueRemeshChunkIndices.reserve(chunks.size());
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < remeshMask.size(); ++chunkArrayIndex) {
            if (remeshMask[chunkArrayIndex] == 0u) {
                continue;
            }
            uniqueRemeshChunkIndices.push_back(chunkArrayIndex);
        }

        for (const std::size_t chunkArrayIndex : uniqueRemeshChunkIndices) {
            m_chunkLodMeshCache[chunkArrayIndex] =
                odai::world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions);
            rebuildGrassInstancesForChunk(chunkArrayIndex);
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            if (m_chunkMeshingOptions.mode == odai::world::MeshingMode::Naive) {
                remeshedNaiveVertexCount = remeshedActiveVertexCount;
                remeshedNaiveIndexCount = remeshedActiveIndexCount;
            } else {
                const odai::world::ChunkLodMeshes naiveLodMeshes =
                    odai::world::buildChunkLodMeshes(chunks[chunkArrayIndex], odai::world::MeshingOptions{odai::world::MeshingMode::Naive});
                countMeshGeometry(naiveLodMeshes, remeshedNaiveVertexCount, remeshedNaiveIndexCount);
            }
        }
        remeshedChunkCount = uniqueRemeshChunkIndices.size();
    }
    const auto remeshEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<float, std::milli> remeshMs = remeshEnd - remeshStart;
    m_debugChunkLastRemeshedChunkCount = static_cast<std::uint32_t>(remeshedChunkCount);
    m_debugChunkLastRemeshActiveVertexCount = static_cast<std::uint32_t>(remeshedActiveVertexCount);
    m_debugChunkLastRemeshActiveIndexCount = static_cast<std::uint32_t>(remeshedActiveIndexCount);
    m_debugChunkLastRemeshNaiveVertexCount = static_cast<std::uint32_t>(remeshedNaiveVertexCount);
    m_debugChunkLastRemeshNaiveIndexCount = static_cast<std::uint32_t>(remeshedNaiveIndexCount);
    m_debugChunkLastRemeshMs = remeshMs.count();
    if (remeshedNaiveIndexCount > 0) {
        const float ratio = static_cast<float>(remeshedActiveIndexCount) / static_cast<float>(remeshedNaiveIndexCount);
        m_debugChunkLastRemeshReductionPercent = std::clamp(100.0f * (1.0f - ratio), 0.0f, 100.0f);
    } else {
        m_debugChunkLastRemeshReductionPercent = 0.0f;
    }
    if (fullRemesh) {
        m_debugChunkLastFullRemeshMs = remeshMs.count();
    } else if (residentSetChanged) {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            if (remeshMask[chunkArrayIndex] == 0u) {
                const odai::world::Chunk& chunk = chunks[chunkArrayIndex];
                const std::vector<GrassBillboardInstance>& grassInstances = m_chunkGrassInstanceCache[chunkArrayIndex];
                const bool previouslyGrassActive = !grassInstances.empty();
                const bool currentlyGrassActive =
                    std::abs(chunk.chunkX() - residentCenterChunkX) <=
                        (previouslyGrassActive ? kGrassRetainedChunkRadius : kGrassActiveChunkRadius) &&
                    std::abs(chunk.chunkZ() - residentCenterChunkZ) <=
                        (previouslyGrassActive ? kGrassRetainedChunkRadius : kGrassActiveChunkRadius);
                const bool previouslyWithinRetainedRadius =
                    std::abs(chunk.chunkX() - previousResidentCenterChunkX) <= kGrassRetainedChunkRadius &&
                    std::abs(chunk.chunkZ() - previousResidentCenterChunkZ) <= kGrassRetainedChunkRadius;
                if (previouslyGrassActive != currentlyGrassActive ||
                    (currentlyGrassActive && !previouslyWithinRetainedRadius)) {
                    rebuildGrassInstancesForChunk(chunkArrayIndex);
                }
            }
        }
    }

    std::vector<GrassBillboardInstance> combinedGrassInstances;
    {
        std::size_t totalGrassInstanceCount = 0;
        for (const std::vector<GrassBillboardInstance>& chunkGrass : m_chunkGrassInstanceCache) {
            totalGrassInstanceCount += chunkGrass.size();
        }
        combinedGrassInstances.reserve(totalGrassInstanceCount);
        for (const std::vector<GrassBillboardInstance>& chunkGrass : m_chunkGrassInstanceCache) {
            combinedGrassInstances.insert(combinedGrassInstances.end(), chunkGrass.begin(), chunkGrass.end());
        }
    }
    if (combinedGrassInstances.empty()) {
        if (m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
            const uint64_t grassReleaseValue = m_lastGraphicsTimelineValue;
            scheduleBufferRelease(m_grassBillboardInstanceBufferHandle, grassReleaseValue);
            m_grassBillboardInstanceBufferHandle = kInvalidBufferHandle;
        }
        m_grassBillboardInstanceCount = 0;
    } else {
        if (fullRemesh) {
            float minR = std::numeric_limits<float>::max();
            float minG = std::numeric_limits<float>::max();
            float minB = std::numeric_limits<float>::max();
            float maxR = std::numeric_limits<float>::lowest();
            float maxG = std::numeric_limits<float>::lowest();
            float maxB = std::numeric_limits<float>::lowest();
            for (const GrassBillboardInstance& instance : combinedGrassInstances) {
                minR = std::min(minR, instance.colorTint[0]);
                minG = std::min(minG, instance.colorTint[1]);
                minB = std::min(minB, instance.colorTint[2]);
                maxR = std::max(maxR, instance.colorTint[0]);
                maxG = std::max(maxG, instance.colorTint[1]);
                maxB = std::max(maxB, instance.colorTint[2]);
            }
            VOX_LOGI("render") << "grass tint range rgb min=("
                              << minR << ", " << minG << ", " << minB
                              << "), max=("
                              << maxR << ", " << maxG << ", " << maxB
                              << "), instances=" << combinedGrassInstances.size() << "\n";
        }

        BufferCreateDesc grassInstanceCreateDesc{};
        grassInstanceCreateDesc.size = static_cast<VkDeviceSize>(combinedGrassInstances.size() * sizeof(GrassBillboardInstance));
        grassInstanceCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        grassInstanceCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        grassInstanceCreateDesc.initialData = combinedGrassInstances.data();

        const BufferHandle newGrassInstanceBufferHandle = m_bufferAllocator.createBuffer(grassInstanceCreateDesc);
        if (newGrassInstanceBufferHandle != kInvalidBufferHandle) {
            const VkBuffer grassInstanceBuffer = m_bufferAllocator.getBuffer(newGrassInstanceBufferHandle);
            if (grassInstanceBuffer != VK_NULL_HANDLE) {
                setObjectName(
                    VK_OBJECT_TYPE_BUFFER,
                    vkHandleToUint64(grassInstanceBuffer),
                    "mesh.grassBillboard.instances"
                );
            }
            if (m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
                const uint64_t grassReleaseValue = m_lastGraphicsTimelineValue;
                scheduleBufferRelease(m_grassBillboardInstanceBufferHandle, grassReleaseValue);
            }
            m_grassBillboardInstanceBufferHandle = newGrassInstanceBufferHandle;
            m_grassBillboardInstanceCount = static_cast<uint32_t>(combinedGrassInstances.size());
        } else {
            VOX_LOGE("render") << "grass billboard instance buffer allocation failed";
        }
    }

    std::vector<odai::world::PackedVoxelVertex> combinedVertices;
    std::vector<std::uint32_t> combinedIndices;
    auto destroyRtAs = [&](RtAccelerationStructure& accelerationStructure) {
        if (accelerationStructure.handle != VK_NULL_HANDLE && m_destroyAccelerationStructureKhr != nullptr) {
            m_destroyAccelerationStructureKhr(m_device, accelerationStructure.handle, nullptr);
            accelerationStructure.handle = VK_NULL_HANDLE;
        }
        if (accelerationStructure.storageBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(accelerationStructure.storageBufferHandle);
            accelerationStructure.storageBufferHandle = kInvalidBufferHandle;
        }
        accelerationStructure.deviceAddress = 0;
        accelerationStructure.primitiveCount = 0;
    };
    m_rtDirtyChunkCount = 0;
    m_debugRtActiveChunkCount = 0;
    std::size_t uploadedVertexCount = 0;
    std::size_t uploadedIndexCount = 0;

    for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
        const odai::world::Chunk& chunk = chunks[chunkArrayIndex];
        const odai::world::ChunkLodMeshes& chunkLodMeshes = m_chunkLodMeshCache[chunkArrayIndex];
        RtChunkSceneRecord& rtChunkRecord = m_rtChunkSceneRecords[chunkArrayIndex];
        const bool remeshChunk = fullRemesh || remeshMask[chunkArrayIndex] != 0u;
        const bool previousRtEligible = rtChunkRecord.rtEligible;
        const int rtActiveRadius =
            previousRtEligible ? kRtRetainedChunkRadius : kRtActiveChunkRadius;
        const bool rtEligible =
            std::abs(chunk.chunkX() - residentCenterChunkX) <= rtActiveRadius &&
            std::abs(chunk.chunkZ() - residentCenterChunkZ) <= rtActiveRadius;
        rtChunkRecord.chunkX = chunk.chunkX();
        rtChunkRecord.chunkY = chunk.chunkY();
        rtChunkRecord.chunkZ = chunk.chunkZ();
        rtChunkRecord.rtEligible = rtEligible;
        if (rtEligible) {
            ++m_debugRtActiveChunkCount;
        }

        for (std::size_t lodIndex = 0; lodIndex < odai::world::kChunkMeshLodCount; ++lodIndex) {
            const odai::world::ChunkMeshData& chunkMesh = chunkLodMeshes.lodMeshes[lodIndex];
            const std::size_t drawRangeArrayIndex = (chunkArrayIndex * odai::world::kChunkMeshLodCount) + lodIndex;
            ChunkDrawRange& drawRange = m_chunkDrawRanges[drawRangeArrayIndex];

            drawRange.offsetX = static_cast<float>(chunk.chunkX() * odai::world::Chunk::kSizeX);
            drawRange.offsetY = static_cast<float>(chunk.chunkY() * odai::world::Chunk::kSizeY);
            drawRange.offsetZ = static_cast<float>(chunk.chunkZ() * odai::world::Chunk::kSizeZ);
            drawRange.firstIndex = 0;
            drawRange.vertexOffset = 0;
            drawRange.indexCount = 0;

            if (chunkMesh.vertices.empty() || chunkMesh.indices.empty()) {
                continue;
            }

            const std::size_t baseVertexSize = combinedVertices.size();
            if (baseVertexSize > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
                VOX_LOGE("render") << "chunk mesh vertex offset exceeds int32 range";
                rollbackChunkDrawState();
                return false;
            }
            const uint32_t baseVertex = static_cast<uint32_t>(baseVertexSize);
            const uint32_t firstIndex = static_cast<uint32_t>(combinedIndices.size());

            combinedVertices.insert(combinedVertices.end(), chunkMesh.vertices.begin(), chunkMesh.vertices.end());
            combinedIndices.reserve(combinedIndices.size() + chunkMesh.indices.size());
            for (const std::uint32_t index : chunkMesh.indices) {
                combinedIndices.push_back(index + baseVertex);
            }
            if (lodIndex == 0u &&
                m_rayTracingCapabilityProbe.rayTracingCoreReady &&
                rtEligible &&
                (remeshChunk || !rtChunkRecord.geometryResident || previousRtEligible != rtEligible)) {
                // RT shadows should trace against the highest-detail chunk mesh.
                rtChunkRecord.vertexCount = static_cast<std::uint32_t>(chunkMesh.vertices.size());
                rtChunkRecord.indexCount = static_cast<std::uint32_t>(chunkMesh.indices.size());
                rtChunkRecord.geometryResident = !chunkMesh.vertices.empty() && !chunkMesh.indices.empty();
                std::vector<RtVertex> rtChunkVertices;
                rtChunkVertices.reserve(chunkMesh.vertices.size());
                for (const odai::world::PackedVoxelVertex& vertex : chunkMesh.vertices) {
                    rtChunkVertices.push_back(
                        decodePackedVoxelVertexPosition(vertex.bits, drawRange.offsetX, drawRange.offsetY, drawRange.offsetZ)
                    );
                }
                if (!createRtGeometryBuffers(m_bufferAllocator, rtChunkVertices, chunkMesh.indices, rtChunkRecord.geometry)) {
                    VOX_LOGE("render") << "chunk RT geometry buffer allocation failed for chunk ("
                                       << rtChunkRecord.chunkX << ","
                                       << rtChunkRecord.chunkY << ","
                                       << rtChunkRecord.chunkZ << ")";
                    rtChunkRecord.geometryResident = false;
                    rtChunkRecord.vertexCount = 0;
                    rtChunkRecord.indexCount = 0;
                }
            }

            drawRange.firstIndex = firstIndex;
            // Indices are already rebased into global vertex space.
            drawRange.vertexOffset = 0;
            drawRange.indexCount = static_cast<uint32_t>(chunkMesh.indices.size());
            uploadedVertexCount += chunkMesh.vertices.size();
            uploadedIndexCount += chunkMesh.indices.size();
        }

        if (!rtEligible) {
            destroyRtGeometryBuffers(m_bufferAllocator, rtChunkRecord.geometry);
            rtChunkRecord.geometryResident = false;
            rtChunkRecord.vertexCount = 0;
            rtChunkRecord.indexCount = 0;
        }
        rtChunkRecord.dirty =
            (rtEligible || previousRtEligible) &&
            (remeshChunk || previousRtEligible != rtEligible);
        if (rtChunkRecord.dirty) {
            ++m_rtDirtyChunkCount;
        }
    }
    for (RtChunkSceneRecord& previousRecord : previousRtChunkSceneRecords) {
        if (previousRecord.chunkX == std::numeric_limits<int>::min()) {
            continue;
        }
        destroyRtAs(previousRecord.blas);
        destroyRtGeometryBuffers(m_bufferAllocator, previousRecord.geometry);
    }
    m_debugChunkMeshVertexCount = static_cast<std::uint32_t>(uploadedVertexCount);
    m_debugChunkMeshIndexCount = static_cast<std::uint32_t>(uploadedIndexCount);

    std::array<uint32_t, 2> meshQueueFamilies = {
        m_graphicsQueueFamilyIndex,
        m_transferQueueFamilyIndex
    };
    if (meshQueueFamilies[0] == meshQueueFamilies[1]) {
        meshQueueFamilies[1] = UINT32_MAX;
    }

    BufferHandle newChunkVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle newChunkIndexBufferHandle = kInvalidBufferHandle;
    std::optional<FrameArenaSlice> chunkVertexUploadSliceOpt = std::nullopt;
    std::optional<FrameArenaSlice> chunkIndexUploadSliceOpt = std::nullopt;
    auto cleanupPendingAllocations = [&]() {
        if (newChunkVertexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(newChunkVertexBufferHandle);
            newChunkVertexBufferHandle = kInvalidBufferHandle;
        }
        if (newChunkIndexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(newChunkIndexBufferHandle);
            newChunkIndexBufferHandle = kInvalidBufferHandle;
        }
    };

    collectCompletedBufferReleases();

    if (m_transferCommandBufferInFlightValue > 0) {
        if (!isTimelineValueReached(m_transferCommandBufferInFlightValue)) {
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
    }
    m_transferCommandBufferInFlightValue = 0;
    collectCompletedBufferReleases();
    const uint64_t previousChunkReadyTimelineValue = m_currentChunkReadyTimelineValue;
    const bool hasChunkCopies = !combinedVertices.empty() && !combinedIndices.empty();

    if (hasChunkCopies) {
        const VkDeviceSize vertexBufferSize =
            static_cast<VkDeviceSize>(combinedVertices.size() * sizeof(odai::world::PackedVoxelVertex));
        const VkDeviceSize indexBufferSize =
            static_cast<VkDeviceSize>(combinedIndices.size() * sizeof(std::uint32_t));

        BufferCreateDesc vertexCreateDesc{};
        vertexCreateDesc.size = vertexBufferSize;
        vertexCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (meshQueueFamilies[1] != UINT32_MAX) {
            vertexCreateDesc.queueFamilyIndices = meshQueueFamilies.data();
            vertexCreateDesc.queueFamilyIndexCount = 2;
        }
        newChunkVertexBufferHandle = m_bufferAllocator.createBuffer(vertexCreateDesc);
        if (newChunkVertexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "chunk global vertex buffer allocation failed";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
        {
            const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(newChunkVertexBufferHandle);
            if (vertexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), "chunk.global.vertex");
            }
        }

        BufferCreateDesc indexCreateDesc{};
        indexCreateDesc.size = indexBufferSize;
        indexCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (meshQueueFamilies[1] != UINT32_MAX) {
            indexCreateDesc.queueFamilyIndices = meshQueueFamilies.data();
            indexCreateDesc.queueFamilyIndexCount = 2;
        }
        newChunkIndexBufferHandle = m_bufferAllocator.createBuffer(indexCreateDesc);
        if (newChunkIndexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "chunk global index buffer allocation failed";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
        {
            const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(newChunkIndexBufferHandle);
            if (indexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), "chunk.global.index");
            }
        }

        chunkVertexUploadSliceOpt = m_frameArena.allocateUpload(
            vertexBufferSize,
            static_cast<VkDeviceSize>(alignof(odai::world::PackedVoxelVertex)),
            FrameArenaUploadKind::Unknown
        );
        if (!chunkVertexUploadSliceOpt.has_value() || chunkVertexUploadSliceOpt->mapped == nullptr) {
            VOX_LOGE("render") << "chunk global vertex upload slice allocation failed";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
        std::memcpy(
            chunkVertexUploadSliceOpt->mapped,
            combinedVertices.data(),
            static_cast<size_t>(vertexBufferSize)
        );

        chunkIndexUploadSliceOpt = m_frameArena.allocateUpload(
            indexBufferSize,
            static_cast<VkDeviceSize>(alignof(std::uint32_t)),
            FrameArenaUploadKind::Unknown
        );
        if (!chunkIndexUploadSliceOpt.has_value() || chunkIndexUploadSliceOpt->mapped == nullptr) {
            VOX_LOGE("render") << "chunk global index upload slice allocation failed";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
        std::memcpy(
            chunkIndexUploadSliceOpt->mapped,
            combinedIndices.data(),
            static_cast<size_t>(indexBufferSize)
        );
    }

    uint64_t transferSignalValue = 0;
    const bool startupChunkUpload =
        m_lastGraphicsTimelineValue == 0 &&
        previousChunkReadyTimelineValue == 0 &&
        std::all_of(
            m_frameTimelineValues.begin(),
            m_frameTimelineValues.end(),
            [](uint64_t value) { return value == 0; }
        );
    if (hasChunkCopies) {
        const VkResult resetResult = vkResetCommandPool(m_device, m_transferCommandPool, 0);
        if (resetResult != VK_SUCCESS) {
            logVkFailure("vkResetCommandPool(transfer)", resetResult);
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_transferCommandBuffer, &beginInfo) != VK_SUCCESS) {
            VOX_LOGE("render") << "vkBeginCommandBuffer (transfer) failed\n";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        {
            const VkDeviceSize vertexBufferSize = m_bufferAllocator.getSize(newChunkVertexBufferHandle);
            const VkDeviceSize indexBufferSize = m_bufferAllocator.getSize(newChunkIndexBufferHandle);

            VkBufferCopy vertexCopy{};
            vertexCopy.srcOffset = chunkVertexUploadSliceOpt->offset;
            vertexCopy.size = vertexBufferSize;
            vkCmdCopyBuffer(
                m_transferCommandBuffer,
                m_bufferAllocator.getBuffer(chunkVertexUploadSliceOpt->buffer),
                m_bufferAllocator.getBuffer(newChunkVertexBufferHandle),
                1,
                &vertexCopy
            );

            VkBufferCopy indexCopy{};
            indexCopy.srcOffset = chunkIndexUploadSliceOpt->offset;
            indexCopy.size = indexBufferSize;
            vkCmdCopyBuffer(
                m_transferCommandBuffer,
                m_bufferAllocator.getBuffer(chunkIndexUploadSliceOpt->buffer),
                m_bufferAllocator.getBuffer(newChunkIndexBufferHandle),
                1,
                &indexCopy
            );
        }

        if (vkEndCommandBuffer(m_transferCommandBuffer) != VK_SUCCESS) {
            VOX_LOGE("render") << "vkEndCommandBuffer (transfer) failed\n";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        transferSignalValue = m_nextTimelineValue++;
        std::array<VkSemaphore, 1> transferWaitSemaphores{};
        std::array<VkPipelineStageFlags2, 1> transferWaitStages{};
        std::array<uint64_t, 1> transferWaitValues{};
        uint32_t transferWaitCount = 0;
        if (m_lastGraphicsTimelineValue > 0) {
            transferWaitSemaphores[0] = m_renderTimelineSemaphore;
            transferWaitStages[0] = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            transferWaitValues[0] = m_lastGraphicsTimelineValue;
            transferWaitCount = 1;
        }

        std::array<VkSemaphoreSubmitInfo, 1> transferWaitSemaphoreInfos{};
        if (transferWaitCount > 0) {
            transferWaitSemaphoreInfos[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            transferWaitSemaphoreInfos[0].semaphore = transferWaitSemaphores[0];
            transferWaitSemaphoreInfos[0].value = transferWaitValues[0];
            transferWaitSemaphoreInfos[0].stageMask = transferWaitStages[0];
            transferWaitSemaphoreInfos[0].deviceIndex = 0;
        }
        VkSemaphoreSubmitInfo transferSignalSemaphoreInfo{};
        transferSignalSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        transferSignalSemaphoreInfo.semaphore = m_renderTimelineSemaphore;
        transferSignalSemaphoreInfo.value = transferSignalValue;
        transferSignalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        transferSignalSemaphoreInfo.deviceIndex = 0;
        VkCommandBufferSubmitInfo transferCommandBufferInfo{};
        transferCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        transferCommandBufferInfo.commandBuffer = m_transferCommandBuffer;
        VkSubmitInfo2 transferSubmitInfo{};
        transferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        transferSubmitInfo.waitSemaphoreInfoCount = transferWaitCount;
        transferSubmitInfo.pWaitSemaphoreInfos =
            transferWaitCount > 0 ? transferWaitSemaphoreInfos.data() : nullptr;
        transferSubmitInfo.commandBufferInfoCount = 1;
        transferSubmitInfo.pCommandBufferInfos = &transferCommandBufferInfo;
        transferSubmitInfo.signalSemaphoreInfoCount = 1;
        transferSubmitInfo.pSignalSemaphoreInfos = &transferSignalSemaphoreInfo;

        const VkResult submitResult = vkQueueSubmit2(m_transferQueue, 1, &transferSubmitInfo, VK_NULL_HANDLE);
        if (submitResult != VK_SUCCESS) {
            logVkFailure("vkQueueSubmit2(transfer)", submitResult);
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        if (startupChunkUpload) {
            const VkResult transferWaitResult = vkQueueWaitIdle(m_transferQueue);
            if (transferWaitResult != VK_SUCCESS) {
                logVkFailure("vkQueueWaitIdle(startupChunkUpload)", transferWaitResult);
                cleanupPendingAllocations();
                rollbackChunkDrawState();
                return false;
            }
            m_currentChunkReadyTimelineValue = 0;
            m_pendingTransferTimelineValue = 0;
            m_transferCommandBufferInFlightValue = 0;
        } else {
            m_currentChunkReadyTimelineValue = transferSignalValue;
            m_pendingTransferTimelineValue = transferSignalValue;
            m_transferCommandBufferInFlightValue = transferSignalValue;
        }
    }

    const uint64_t oldChunkReleaseValue = std::max(m_lastGraphicsTimelineValue, previousChunkReadyTimelineValue);
    scheduleBufferRelease(m_chunkVertexBufferHandle, oldChunkReleaseValue);
    scheduleBufferRelease(m_chunkIndexBufferHandle, oldChunkReleaseValue);
    m_chunkVertexBufferHandle = newChunkVertexBufferHandle;
    m_chunkIndexBufferHandle = newChunkIndexBufferHandle;
    newChunkVertexBufferHandle = kInvalidBufferHandle;
    newChunkIndexBufferHandle = kInvalidBufferHandle;
    const bool rtSceneNeedsRefresh =
        m_rayTracingCapabilityProbe.rayTracingCoreReady &&
        (m_rtDirtyChunkCount > 0 || m_rtTlas.handle == VK_NULL_HANDLE);
    if (rtSceneNeedsRefresh) {
        markRayTracingSceneDirty();
    }

    VOX_LOGD("render") << "chunk upload queued (ranges=" << m_chunkDrawRanges.size()
                       << ", remeshedChunks=" << remeshedChunkCount
                       << ", meshingMode="
                       << (m_chunkMeshingOptions.mode == odai::world::MeshingMode::Greedy ? "greedy" : "naive")
                       << ", vertices=" << uploadedVertexCount
                       << ", indices=" << uploadedIndexCount
                       << ", rtResidentChunks=" << m_rtChunkSceneRecords.size()
                       << ", rtDirtyChunks=" << m_rtDirtyChunkCount
                       << (hasChunkCopies
                               ? (", timelineValue=" + std::to_string(transferSignalValue))
                               : ", immediate=true")
                       << ")";
    return true;
}

void RendererBackend::markRayTracingSceneDirty() {
    if (!m_rayTracingCapabilityProbe.rayTracingCoreReady) {
        return;
    }
    m_rtSceneDirty = true;
    refreshShadowStats();
}

void RendererBackend::destroyRayTracingScene() {
    const bool hasExistingScene =
        m_rtTlas.handle != VK_NULL_HANDLE ||
        !m_rtChunkSceneRecords.empty() ||
        !m_rtImportedSceneRecords.empty() ||
        !m_rtMagicaBlases.empty();
    if (hasExistingScene && m_device != VK_NULL_HANDLE && m_graphicsQueue != VK_NULL_HANDLE) {
        const VkResult waitResult = vkQueueWaitIdle(m_graphicsQueue);
        if (waitResult != VK_SUCCESS) {
            VOX_LOGW("render") << "destroyRayTracingScene: vkQueueWaitIdle failed before AS destruction";
        }
    }
    auto destroyAs = [&](RtAccelerationStructure& accelerationStructure) {
        if (accelerationStructure.handle != VK_NULL_HANDLE && m_destroyAccelerationStructureKhr != nullptr) {
            m_destroyAccelerationStructureKhr(m_device, accelerationStructure.handle, nullptr);
            accelerationStructure.handle = VK_NULL_HANDLE;
        }
        if (accelerationStructure.storageBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(accelerationStructure.storageBufferHandle);
            accelerationStructure.storageBufferHandle = kInvalidBufferHandle;
        }
        accelerationStructure.deviceAddress = 0;
        accelerationStructure.primitiveCount = 0;
    };

    destroyAs(m_rtTlas);
    if (m_rtTlasInstanceBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_rtTlasInstanceBufferHandle);
        m_rtTlasInstanceBufferHandle = kInvalidBufferHandle;
    }
    for (RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
        destroyAs(chunkRecord.blas);
        destroyRtGeometryBuffers(m_bufferAllocator, chunkRecord.geometry);
    }
    for (RtImportedSceneRecord& importedRecord : m_rtImportedSceneRecords) {
        destroyAs(importedRecord.blas);
        destroyRtGeometryBuffers(m_bufferAllocator, importedRecord.geometry);
    }
    m_rtImportedSceneRecords.clear();
    for (RtAccelerationStructure& blas : m_rtMagicaBlases) {
        destroyAs(blas);
    }
    m_rtMagicaBlases.clear();
    for (RtGeometryBuffers& geometry : m_rtMagicaGeometries) {
        destroyRtGeometryBuffers(m_bufferAllocator, geometry);
    }
    m_rtMagicaGeometries.clear();
    m_rtSceneDirty = false;
    m_rtSceneBuildCount = 0;
    m_rtBlasBuildCount = 0;
    m_rtTlasBuildCount = 0;
    m_rtDirtyChunkCount = 0;
    m_rtChunkSceneRecords.clear();
    refreshShadowStats();
}

bool RendererBackend::rebuildRayTracingScene() {
    if (!rayTracingRuntimeReady()) {
        refreshShadowStats();
        return false;
    }
    const VkDeviceAddress scratchAlignment = std::max<VkDeviceAddress>(
        1,
        static_cast<VkDeviceAddress>(m_rayTracingCapabilityProbe.scratchAlignment)
    );
    auto alignDeviceAddress = [&](VkDeviceAddress address) -> VkDeviceAddress {
        const VkDeviceAddress mask = scratchAlignment - 1;
        return (address + mask) & ~mask;
    };

    auto destroyAs = [&](RtAccelerationStructure& accelerationStructure) {
        if (accelerationStructure.handle != VK_NULL_HANDLE && m_destroyAccelerationStructureKhr != nullptr) {
            m_destroyAccelerationStructureKhr(m_device, accelerationStructure.handle, nullptr);
            accelerationStructure.handle = VK_NULL_HANDLE;
        }
        if (accelerationStructure.storageBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(accelerationStructure.storageBufferHandle);
            accelerationStructure.storageBufferHandle = kInvalidBufferHandle;
        }
        accelerationStructure.deviceAddress = 0;
        accelerationStructure.primitiveCount = 0;
    };
    auto createAsStorage = [&](VkAccelerationStructureTypeKHR type,
                               VkDeviceSize size,
                               RtAccelerationStructure& outAccelerationStructure) -> bool {
        destroyAs(outAccelerationStructure);
        BufferCreateDesc storageCreateDesc{};
        storageCreateDesc.size = size;
        storageCreateDesc.usage =
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        storageCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        outAccelerationStructure.storageBufferHandle = m_bufferAllocator.createBuffer(storageCreateDesc);
        if (outAccelerationStructure.storageBufferHandle == kInvalidBufferHandle) {
            return false;
        }
        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = m_bufferAllocator.getBuffer(outAccelerationStructure.storageBufferHandle);
        createInfo.size = size;
        createInfo.type = type;
        if (m_createAccelerationStructureKhr(m_device, &createInfo, nullptr, &outAccelerationStructure.handle) != VK_SUCCESS) {
            m_bufferAllocator.destroyBuffer(outAccelerationStructure.storageBufferHandle);
            outAccelerationStructure.storageBufferHandle = kInvalidBufferHandle;
            return false;
        }
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = outAccelerationStructure.handle;
        outAccelerationStructure.deviceAddress =
            m_getAccelerationStructureDeviceAddressKhr(m_device, &addressInfo);
        return outAccelerationStructure.deviceAddress != 0;
    };

    bool needsGraphicsIdle = false;
    for (const RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
        if (chunkRecord.blas.handle != VK_NULL_HANDLE &&
            (!chunkRecord.rtEligible || !chunkRecord.geometryResident || chunkRecord.dirty)) {
            needsGraphicsIdle = true;
            break;
        }
    }
    if (!needsGraphicsIdle) {
        for (const RtImportedSceneRecord& importedRecord : m_rtImportedSceneRecords) {
            if (importedRecord.blas.handle != VK_NULL_HANDLE &&
                (!importedRecord.geometryResident || importedRecord.dirty)) {
                needsGraphicsIdle = true;
                break;
            }
        }
    }
    if (!needsGraphicsIdle && m_rtTlas.handle != VK_NULL_HANDLE && m_rtSceneDirty) {
        needsGraphicsIdle = true;
    }
    if (needsGraphicsIdle) {
        const VkResult waitResult = vkQueueWaitIdle(m_graphicsQueue);
        if (waitResult != VK_SUCCESS) {
            VOX_LOGE("render") << "rebuildRayTracingScene: vkQueueWaitIdle failed before AS rebuild";
            refreshShadowStats();
            return false;
        }
    }

    std::vector<std::pair<RtGeometryBuffers*, RtAccelerationStructure*>> buildGeometries;
    for (RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
        if (!chunkRecord.rtEligible ||
            !chunkRecord.geometryResident ||
            chunkRecord.geometry.vertexCount == 0 ||
            chunkRecord.geometry.indexCount == 0) {
            destroyAs(chunkRecord.blas);
            continue;
        }
        if (chunkRecord.dirty || chunkRecord.blas.handle == VK_NULL_HANDLE) {
            destroyAs(chunkRecord.blas);
            buildGeometries.push_back({&chunkRecord.geometry, &chunkRecord.blas});
        }
    }
    for (RtImportedSceneRecord& importedRecord : m_rtImportedSceneRecords) {
        if (!importedRecord.geometryResident ||
            importedRecord.geometry.vertexCount == 0 ||
            importedRecord.geometry.indexCount == 0) {
            destroyAs(importedRecord.blas);
            continue;
        }
        if (importedRecord.dirty || importedRecord.blas.handle == VK_NULL_HANDLE) {
            destroyAs(importedRecord.blas);
            buildGeometries.push_back({&importedRecord.geometry, &importedRecord.blas});
        }
    }
    if (m_rtMagicaBlases.size() > m_rtMagicaGeometries.size()) {
        for (std::size_t i = m_rtMagicaGeometries.size(); i < m_rtMagicaBlases.size(); ++i) {
            destroyAs(m_rtMagicaBlases[i]);
        }
    }
    m_rtMagicaBlases.resize(m_rtMagicaGeometries.size());
    for (std::size_t i = 0; i < m_rtMagicaGeometries.size(); ++i) {
        if (m_rtMagicaGeometries[i].vertexCount == 0 || m_rtMagicaGeometries[i].indexCount == 0) {
            destroyAs(m_rtMagicaBlases[i]);
            continue;
        }
        if (m_rtMagicaBlases[i].handle == VK_NULL_HANDLE) {
            buildGeometries.push_back({&m_rtMagicaGeometries[i], &m_rtMagicaBlases[i]});
        }
    }

    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    commandPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
        return false;
    }
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        return false;
    }

    struct ScratchAllocation {
        BufferHandle handle = kInvalidBufferHandle;
        VkDeviceAddress alignedAddress = 0;
    };
    std::vector<ScratchAllocation> scratchBuffers;
    scratchBuffers.reserve(buildGeometries.size() + 1u);
    std::size_t estimatedInstanceCount = m_rtMagicaGeometries.size();
    for (const RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
        if (chunkRecord.rtEligible && chunkRecord.geometryResident && chunkRecord.geometry.indexCount > 0) {
            ++estimatedInstanceCount;
        }
    }
    for (const RtImportedSceneRecord& importedRecord : m_rtImportedSceneRecords) {
        if (importedRecord.geometryResident && importedRecord.geometry.indexCount > 0) {
            ++estimatedInstanceCount;
        }
    }
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(estimatedInstanceCount);
    bool buildOk = true;
    bool commandBufferBegun = false;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    buildOk = vkBeginCommandBuffer(commandBuffer, &beginInfo) == VK_SUCCESS;
    commandBufferBegun = buildOk;

    for (const auto& [geometry, outBlas] : buildGeometries) {
        if (!buildOk) {
            break;
        }
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = m_bufferAllocator.getDeviceAddress(geometry->vertexBufferHandle);
        triangles.vertexStride = sizeof(RtVertex);
        triangles.maxVertex = geometry->vertexCount > 0 ? (geometry->vertexCount - 1u) : 0u;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = m_bufferAllocator.getDeviceAddress(geometry->indexBufferHandle);

        VkAccelerationStructureGeometryKHR asGeometry{};
        asGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        asGeometry.geometry.triangles = triangles;

        const std::uint32_t primitiveCount = geometry->indexCount / 3u;
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &asGeometry;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        m_getAccelerationStructureBuildSizesKhr(
            m_device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo,
            &primitiveCount,
            &sizeInfo
        );
        buildOk = createAsStorage(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizeInfo.accelerationStructureSize, *outBlas);
        if (!buildOk) {
            break;
        }
        BufferCreateDesc scratchCreateDesc{};
        scratchCreateDesc.size = sizeInfo.buildScratchSize + scratchAlignment - 1;
        scratchCreateDesc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        scratchCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const BufferHandle scratchHandle = m_bufferAllocator.createBuffer(scratchCreateDesc);
        if (scratchHandle == kInvalidBufferHandle) {
            buildOk = false;
            break;
        }
        const VkDeviceAddress scratchAddress = alignDeviceAddress(m_bufferAllocator.getDeviceAddress(scratchHandle));
        if (scratchAddress == 0) {
            m_bufferAllocator.destroyBuffer(scratchHandle);
            buildOk = false;
            break;
        }
        scratchBuffers.push_back({scratchHandle, scratchAddress});

        buildInfo.dstAccelerationStructure = outBlas->handle;
        buildInfo.scratchData.deviceAddress = scratchAddress;
        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = primitiveCount;
        const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = {&rangeInfo};
        m_cmdBuildAccelerationStructuresKhr(commandBuffer, 1, &buildInfo, rangeInfos);
        outBlas->primitiveCount = primitiveCount;
    }

    if (buildOk && !buildGeometries.empty()) {
        VkMemoryBarrier2 blasBuildBarrier{};
        blasBuildBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        blasBuildBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        blasBuildBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        blasBuildBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        blasBuildBarrier.dstAccessMask =
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &blasBuildBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

    if (buildOk) {
        auto appendTlasInstance = [&](const RtAccelerationStructure& accelerationStructure) {
            if (accelerationStructure.handle == VK_NULL_HANDLE || accelerationStructure.deviceAddress == 0) {
                return;
            }
            VkAccelerationStructureInstanceKHR instance{};
            instance.transform.matrix[0][0] = 1.0f;
            instance.transform.matrix[1][1] = 1.0f;
            instance.transform.matrix[2][2] = 1.0f;
            instance.instanceCustomIndex = static_cast<std::uint32_t>(tlasInstances.size());
            instance.mask = 0xFFu;
            instance.instanceShaderBindingTableRecordOffset = 0;
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instance.accelerationStructureReference = accelerationStructure.deviceAddress;
            tlasInstances.push_back(instance);
        };

        for (const RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
            if (!chunkRecord.rtEligible || !chunkRecord.geometryResident) {
                continue;
            }
            appendTlasInstance(chunkRecord.blas);
        }
        for (const RtImportedSceneRecord& importedRecord : m_rtImportedSceneRecords) {
            if (!importedRecord.geometryResident) {
                continue;
            }
            appendTlasInstance(importedRecord.blas);
        }
        for (const RtAccelerationStructure& blas : m_rtMagicaBlases) {
            appendTlasInstance(blas);
        }
    }

    if (buildOk && !tlasInstances.empty()) {
        if (m_rtTlasInstanceBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(m_rtTlasInstanceBufferHandle);
            m_rtTlasInstanceBufferHandle = kInvalidBufferHandle;
        }
        BufferCreateDesc instanceCreateDesc{};
        instanceCreateDesc.size = static_cast<VkDeviceSize>(tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR));
        instanceCreateDesc.usage =
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        instanceCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        instanceCreateDesc.initialData = tlasInstances.data();
        m_rtTlasInstanceBufferHandle = m_bufferAllocator.createBuffer(instanceCreateDesc);
        buildOk = m_rtTlasInstanceBufferHandle != kInvalidBufferHandle;
        if (buildOk) {
            VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
            instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            instancesData.data.deviceAddress = m_bufferAllocator.getDeviceAddress(m_rtTlasInstanceBufferHandle);

            VkAccelerationStructureGeometryKHR tlasGeometry{};
            tlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            tlasGeometry.geometry.instances = instancesData;

            const std::uint32_t primitiveCount = static_cast<std::uint32_t>(tlasInstances.size());
            VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
            buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            buildInfo.geometryCount = 1;
            buildInfo.pGeometries = &tlasGeometry;

            VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
            sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            m_getAccelerationStructureBuildSizesKhr(
                m_device,
                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &buildInfo,
                &primitiveCount,
                &sizeInfo
            );
            buildOk = createAsStorage(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizeInfo.accelerationStructureSize, m_rtTlas);
            if (buildOk) {
                BufferCreateDesc scratchCreateDesc{};
                scratchCreateDesc.size = sizeInfo.buildScratchSize + scratchAlignment - 1;
                scratchCreateDesc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                scratchCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                const BufferHandle scratchHandle = m_bufferAllocator.createBuffer(scratchCreateDesc);
                VkDeviceAddress scratchAddress = 0;
                buildOk = scratchHandle != kInvalidBufferHandle;
                if (buildOk) {
                    scratchAddress = alignDeviceAddress(m_bufferAllocator.getDeviceAddress(scratchHandle));
                    buildOk = scratchAddress != 0;
                    if (!buildOk) {
                        m_bufferAllocator.destroyBuffer(scratchHandle);
                    } else {
                        scratchBuffers.push_back({scratchHandle, scratchAddress});
                    }
                }
                if (buildOk) {
                    buildInfo.dstAccelerationStructure = m_rtTlas.handle;
                    buildInfo.scratchData.deviceAddress = scratchAddress;
                    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
                    rangeInfo.primitiveCount = primitiveCount;
                    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = {&rangeInfo};
                    m_cmdBuildAccelerationStructuresKhr(commandBuffer, 1, &buildInfo, rangeInfos);
                    m_rtTlas.primitiveCount = primitiveCount;
                }
            }
        }
    } else {
        if (m_rtTlasInstanceBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(m_rtTlasInstanceBufferHandle);
            m_rtTlasInstanceBufferHandle = kInvalidBufferHandle;
        }
        destroyAs(m_rtTlas);
    }

    if (buildOk && m_rtTlas.handle != VK_NULL_HANDLE) {
        VkMemoryBarrier2 tlasBuildBarrier{};
        tlasBuildBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        tlasBuildBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        tlasBuildBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        tlasBuildBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        tlasBuildBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &tlasBuildBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

    if (buildOk) {
        buildOk = vkEndCommandBuffer(commandBuffer) == VK_SUCCESS;
    } else if (commandBufferBegun) {
        vkEndCommandBuffer(commandBuffer);
    }
    if (buildOk) {
        VkCommandBufferSubmitInfo commandBufferInfo{};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        commandBufferInfo.commandBuffer = commandBuffer;
        VkSubmitInfo2 submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandBufferInfo;
        buildOk = vkQueueSubmit2(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS;
    }
    if (buildOk) {
        buildOk = vkQueueWaitIdle(m_graphicsQueue) == VK_SUCCESS;
    }

    for (const ScratchAllocation& scratchAllocation : scratchBuffers) {
        if (scratchAllocation.handle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(scratchAllocation.handle);
        }
    }
    vkDestroyCommandPool(m_device, commandPool, nullptr);

    if (!buildOk) {
        destroyAs(m_rtTlas);
        for (RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
            destroyAs(chunkRecord.blas);
        }
        for (RtImportedSceneRecord& importedRecord : m_rtImportedSceneRecords) {
            destroyAs(importedRecord.blas);
        }
        for (RtAccelerationStructure& blas : m_rtMagicaBlases) {
            destroyAs(blas);
        }
        refreshShadowStats();
        return false;
    }

    m_rtSceneDirty = false;
    ++m_rtSceneBuildCount;
    m_rtBlasBuildCount = static_cast<std::uint32_t>(buildGeometries.size());
    m_rtTlasBuildCount = tlasInstances.empty() ? 0u : 1u;
    for (RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
        chunkRecord.dirty = false;
    }
    for (RtImportedSceneRecord& importedRecord : m_rtImportedSceneRecords) {
        importedRecord.dirty = false;
    }
    refreshShadowStats();
    VOX_LOGI("render") << "ray tracing scene rebuilt: blas=" << m_rtBlasBuildCount
                       << ", tlas=" << m_rtTlasBuildCount
                       << ", instances=" << tlasInstances.size()
                       << ", importedRecords=" << m_rtImportedSceneRecords.size()
                       << ", residentChunks=" << m_rtChunkSceneRecords.size()
                       << ", dirtyChunks=" << m_rtDirtyChunkCount
                       << ", sceneBuilds=" << m_rtSceneBuildCount << "\n";
    return true;
}
} // namespace odai::render
