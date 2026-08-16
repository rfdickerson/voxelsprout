#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>
#include "core/grid3.h"
#include "core/frame_profiler.h"
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
        case odai::importer::TextureFormat::BC2:
        case odai::importer::TextureFormat::BC3:
        case odai::importer::TextureFormat::BC5:
        case odai::importer::TextureFormat::BC7: return 16u;
        default:                                 return 0u;
    }
}

// normalizedImportedTextureKey moved to renderer_backend.h: the skinned-actor
// texture upload acquires from the same reference-counted table and has to
// normalize identically, which only one definition can guarantee.

VkFormat vkFormatForImportedTexture(odai::importer::TextureFormat format) {
    switch (format) {
        // Color albedo (BC1/BC3/BC7) holds sRGB-encoded bytes, so use the _SRGB views:
        // the sampler decodes sRGB -> linear, matching the linear lighting + tonemap
        // pipeline. Without this the raw sRGB values are read as linear (~2x too bright)
        // and terrain renders as washed-out pastel. BC4/BC5 are data (single/dual
        // channel — e.g. the water normal map) and must stay UNORM/linear.
        case odai::importer::TextureFormat::BC1: return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
        case odai::importer::TextureFormat::BC2: return VK_FORMAT_BC2_SRGB_BLOCK;
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
    if ((!m_importedTextureResources.empty() ||
         m_fogMapTextureResource.image != VK_NULL_HANDLE) && m_device != VK_NULL_HANDLE) {
        // This is the wholesale teardown path (scene swap / shutdown), so the
        // one big wait is still the right trade here -- it is not on the
        // streaming path, which evicts individual slots through
        // releaseImportedTexture() and never waits.
        vkDeviceWaitIdle(m_device);
        for (ImportedTextureResource& texture : m_importedTextureResources) {
            destroyImageResourceNow(texture.image, texture.allocation, texture.imageView);
            texture = ImportedTextureResource{};
        }
        m_importedTextureResources.clear();
        m_importedTextureSlotTable.clear();
        if (m_fogMapTextureResource.imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_fogMapTextureResource.imageView, nullptr);
            m_fogMapTextureResource.imageView = VK_NULL_HANDLE;
        }
        if (m_fogMapTextureResource.image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_fogMapTextureResource.image, m_fogMapTextureResource.allocation);
            m_fogMapTextureResource.image = VK_NULL_HANDLE;
            m_fogMapTextureResource.allocation = VK_NULL_HANDLE;
        }
        m_fogMapEnabled = false;
    }
    if (m_importedShadowVertexBufferHandle != kInvalidBufferHandle) {
        if (m_lastGraphicsTimelineValue == 0) {
            m_bufferAllocator.destroyBuffer(m_importedShadowVertexBufferHandle);
        } else {
            scheduleBufferRelease(m_importedShadowVertexBufferHandle, m_lastGraphicsTimelineValue);
        }
        m_importedShadowVertexBufferHandle = kInvalidBufferHandle;
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
    // Chunks and arenas go together: the buffers backing them were just
    // released above, so every recorded offset is now meaningless.
    m_importedSceneChunks.clear();
    m_freeImportedSceneChunks.clear();
    m_lastImportedChunkIndex = kInvalidImportedChunkIndex;
    m_importedVertexArena.reset(0);
    m_importedIndexArena.reset(0);
    m_visibleImportedMeshDraws.clear();
    m_importedTextureSlots.clear();
    for (std::vector<ImportedMeshDraw>& shadowDraws : m_visibleImportedShadowMeshDraws) {
        shadowDraws.clear();
    }
    m_visibleImportedPageScratch.clear();
    m_visibleImportedTerrainDrawCount = 0;
    m_visibleImportedNearTerrainDrawCount = 0;
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
    // m_rtImportedSceneRecords (BLAS + geometry buffers) are intentionally NOT
    // touched here. Every real caller of clearGpuScene() (via
    // clearImportedSceneMeshes()) calls destroyRayTracingScene() immediately
    // afterward, which tears down the same records safely behind a
    // vkQueueWaitIdle(). Destroying them here first — with no wait — used to
    // race an in-flight frame that could still be reading the BLAS during
    // acceleration-structure builds or RT shading, and rendered that later
    // wait useless since the handles were already gone. On heavier scenes
    // (longer frame times widen the race window) this surfaced as GPU hangs,
    // including on shutdown's subsequent vkDeviceWaitIdle().
}

void RendererBackend::clearImportedSceneMeshes() {
    m_importedSceneBoundsValid = false;
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
    return uploadImportedSceneInternal(scene, nullptr, /*appendChunk=*/false);
}

std::size_t RendererBackend::addImportedSceneChunk(const odai::importer::ImportedScene& scene) {
    // Cleared first so a success that produced no chunk -- an empty scene, which
    // returns true -- is distinguishable. Comparing the vector's size before and
    // after used to serve for that, and stopped being able to once a chunk could
    // land in a recycled slot without the vector growing at all.
    m_lastImportedChunkIndex = kInvalidImportedChunkIndex;
    if (!uploadImportedSceneInternal(scene, nullptr, /*appendChunk=*/true)) {
        return kInvalidImportedChunkIndex;
    }
    if (m_lastImportedChunkIndex == kInvalidImportedChunkIndex) {
        return kInvalidImportedChunkIndex;
    }
    // When ANY resident chunk carries page ranges, per-frame draw selection is
    // built from pages ONLY (see frame_run's importedPageCullingEnabled), so a
    // chunk whose pages were dropped -- e.g. by the coverage check above --
    // has every draw silently excluded. Worth a line at add time; that failure
    // renders as "uploaded fine, never drawn".
    rebuildImportedWaterBuffers();
    const ImportedSceneChunk& added = m_importedSceneChunks[m_lastImportedChunkIndex];
    VOX_LOGI("render") << "chunk " << m_lastImportedChunkIndex << " added: draws="
                       << added.draws.size() << " pages=" << added.pageRanges.size();
    return m_lastImportedChunkIndex;
}

void RendererBackend::removeImportedSceneChunkAt(std::size_t chunkIndex) {
    removeImportedSceneChunk(chunkIndex);
}

std::size_t RendererBackend::importedLocalLightCount() const {
    return m_importedLocalLights.size();
}

std::size_t RendererBackend::liveImportedSceneChunkCount() const {
    std::size_t liveCount = 0;
    for (const ImportedSceneChunk& chunk : m_importedSceneChunks) {
        if (chunk.alive) {
            ++liveCount;
        }
    }
    return liveCount;
}

bool RendererBackend::uploadIntoBufferRange(
    BufferHandle destination,
    VkDeviceSize destinationByteOffset,
    const void* sourceData,
    VkDeviceSize byteSize,
    const char* debugLabel) {
    if (destination == kInvalidBufferHandle || sourceData == nullptr || byteSize == 0u) {
        return byteSize == 0u;  // nothing to copy is success, not failure
    }

    BufferCreateDesc stagingCreateDesc{};
    stagingCreateDesc.size = byteSize;
    stagingCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingCreateDesc.memoryProperties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    stagingCreateDesc.initialData = sourceData;
    const BufferHandle stagingHandle = m_bufferAllocator.createBuffer(stagingCreateDesc);
    if (stagingHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << debugLabel << " staging buffer allocation failed";
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
        logVkFailure("vkCreateCommandPool(importedArenaUpload)", result);
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
            logVkFailure("vkAllocateCommandBuffers(importedArenaUpload)", result);
            uploadFailed = true;
        }
    }
    if (!uploadFailed) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
        if (result != VK_SUCCESS) {
            logVkFailure("vkBeginCommandBuffer(importedArenaUpload)", result);
            uploadFailed = true;
        }
    }
    if (!uploadFailed) {
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = destinationByteOffset;
        copyRegion.size = byteSize;
        vkCmdCopyBuffer(
            commandBuffer,
            m_bufferAllocator.getBuffer(stagingHandle),
            m_bufferAllocator.getBuffer(destination),
            1,
            &copyRegion);
        result = vkEndCommandBuffer(commandBuffer);
        if (result != VK_SUCCESS) {
            logVkFailure("vkEndCommandBuffer(importedArenaUpload)", result);
            uploadFailed = true;
        }
    }
    // Same reasoning as the texture path: signal a timeline value the next
    // frame's graphics submit already waits on, rather than draining the whole
    // graphics queue here. Three of these run per streamed cell (vertices,
    // shadow vertices, indices), so three full queue drains were being paid for
    // every cell that arrived.
    uint64_t uploadTimelineValue = 0;
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
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandBufferInfo;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos = &signalInfo;

        result = vkQueueSubmit2(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        if (result != VK_SUCCESS) {
            logVkFailure("vkQueueSubmit2(importedArenaUpload)", result);
            uploadFailed = true;
            uploadTimelineValue = 0;
        } else {
            m_pendingTransferTimelineValue =
                std::max(m_pendingTransferTimelineValue, uploadTimelineValue);
        }
    }

    scheduleCommandPoolRelease(commandPool, uploadTimelineValue);
    scheduleBufferRelease(stagingHandle, uploadTimelineValue);
    return !uploadFailed;
}

bool RendererBackend::copyBufferRange(
    BufferHandle source, BufferHandle destination, VkDeviceSize byteSize, const char* debugLabel,
    uint64_t& outTimelineValue) {
    outTimelineValue = 0;
    if (source == kInvalidBufferHandle || destination == kInvalidBufferHandle || byteSize == 0u) {
        return byteSize == 0u;
    }

    bool copyFailed = false;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    commandPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    VkResult result = vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateCommandPool(importedArenaGrow)", result);
        copyFailed = true;
    }
    if (!copyFailed) {
        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = commandPool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        result = vkAllocateCommandBuffers(m_device, &allocateInfo, &commandBuffer);
        if (result != VK_SUCCESS) {
            logVkFailure("vkAllocateCommandBuffers(importedArenaGrow)", result);
            copyFailed = true;
        }
    }
    if (!copyFailed) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
        if (result != VK_SUCCESS) {
            logVkFailure("vkBeginCommandBuffer(importedArenaGrow)", result);
            copyFailed = true;
        }
    }
    if (!copyFailed) {
        VkBufferCopy copyRegion{};
        copyRegion.size = byteSize;
        vkCmdCopyBuffer(
            commandBuffer,
            m_bufferAllocator.getBuffer(source),
            m_bufferAllocator.getBuffer(destination),
            1,
            &copyRegion);
        result = vkEndCommandBuffer(commandBuffer);
        if (result != VK_SUCCESS) {
            logVkFailure("vkEndCommandBuffer(importedArenaGrow)", result);
            copyFailed = true;
        }
    }
    // The old arena is scheduled for release on m_lastGraphicsTimelineValue by
    // the caller, and the new one is only ever drawn from after the graphics
    // submit waits on this value, so the copy needs no CPU wait either.
    uint64_t copyTimelineValue = 0;
    if (!copyFailed) {
        copyTimelineValue = m_nextTimelineValue++;

        VkSemaphoreSubmitInfo signalInfo{};
        signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signalInfo.semaphore = m_renderTimelineSemaphore;
        signalInfo.value = copyTimelineValue;
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

        result = vkQueueSubmit2(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        if (result != VK_SUCCESS) {
            logVkFailure("vkQueueSubmit2(importedArenaGrow)", result);
            copyFailed = true;
            copyTimelineValue = 0;
        } else {
            m_pendingTransferTimelineValue =
                std::max(m_pendingTransferTimelineValue, copyTimelineValue);
        }
    }
    scheduleCommandPoolRelease(commandPool, copyTimelineValue);
    outTimelineValue = copyTimelineValue;
    if (copyFailed) {
        VOX_LOGE("render") << debugLabel << " arena copy failed";
    }
    return !copyFailed;
}

bool RendererBackend::ensureImportedArenaCapacity(
    std::uint64_t vertexCount, std::uint64_t indexCount, bool pastCapacity) {
    // Vertex and index arenas grow independently; either may already be big
    // enough. "Big enough" is measured against used() normally and against
    // capacity() when the caller is here BECAUSE an allocation already failed --
    // see the header. Measuring a post-fragmentation retry against used() makes
    // this function a no-op, which is exactly the bug the pastCapacity flag
    // replaced: the caller then grew the suballocator by hand, the GPU buffer
    // stayed its old size, and allocate() started handing back offsets past the
    // end of it.
    const std::uint64_t vertexBase =
        pastCapacity ? m_importedVertexArena.capacity() : m_importedVertexArena.used();
    const std::uint64_t indexBase =
        pastCapacity ? m_importedIndexArena.capacity() : m_importedIndexArena.used();
    const std::uint64_t neededVertices = vertexBase + vertexCount;
    const std::uint64_t neededIndices = indexBase + indexCount;
    const bool growVertices = neededVertices > m_importedVertexArena.capacity();
    const bool growIndices = neededIndices > m_importedIndexArena.capacity();
    if (!growVertices && !growIndices) {
        return true;
    }

    // First fill is sized exactly; only *growth* doubles. Doubling from empty
    // would round a 3.29M-vertex scene up to 4.19M and cost ~110 MB across the
    // three arenas that a one-shot uploadImportedScene() never needs.
    //
    // A fragmentation retry is sized exactly too. Doubling is the right policy
    // for a working set that is genuinely getting bigger, but this arena never
    // shrinks, so paying it for a hole in the free list ratchets the buffer up
    // permanently -- at ~100 bytes per vertex across the main and shadow streams,
    // one doubling of an 8M-vertex arena is 800 MB the session never gives back.
    const auto nextCapacity = [pastCapacity](std::uint64_t current, std::uint64_t needed) {
        if (current == 0 || pastCapacity) {
            return needed;
        }
        std::uint64_t capacity = current;
        while (capacity < needed) {
            capacity *= 2u;
        }
        return capacity;
    };

    const std::uint64_t newVertexCapacity = growVertices
        ? nextCapacity(m_importedVertexArena.capacity(), neededVertices)
        : m_importedVertexArena.capacity();
    const std::uint64_t newIndexCapacity = growIndices
        ? nextCapacity(m_importedIndexArena.capacity(), neededIndices)
        : m_importedIndexArena.capacity();

    // Recreate each buffer at the new size and copy the old contents across at
    // the same offsets, so every live chunk's firstVertex/firstIndex stays
    // valid and nothing has to be re-uploaded from the CPU.
    struct ArenaBuffer {
        BufferHandle* handle;
        std::uint64_t oldCapacity;
        std::uint64_t newCapacity;
        std::uint64_t stride;
        VkBufferUsageFlags usage;
        const char* label;
    };
    const std::uint64_t oldVertexCapacity = m_importedVertexArena.capacity();
    const std::uint64_t oldIndexCapacity = m_importedIndexArena.capacity();
    const ArenaBuffer arenaBuffers[] = {
        {&m_importedVertexBufferHandle, oldVertexCapacity, newVertexCapacity,
         sizeof(ImportedMeshVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "imported vertex arena"},
        {&m_importedShadowVertexBufferHandle, oldVertexCapacity, newVertexCapacity,
         sizeof(ImportedShadowVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
         "imported shadow vertex arena"},
        {&m_importedIndexBufferHandle, oldIndexCapacity, newIndexCapacity,
         sizeof(std::uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, "imported index arena"},
    };

    std::vector<BufferHandle> createdHandles;
    createdHandles.reserve(std::size(arenaBuffers));
    bool failed = false;
    for (const ArenaBuffer& arena : arenaBuffers) {
        if (arena.newCapacity == arena.oldCapacity && *arena.handle != kInvalidBufferHandle) {
            createdHandles.push_back(kInvalidBufferHandle);  // unchanged
            continue;
        }
        BufferCreateDesc desc{};
        desc.size = static_cast<VkDeviceSize>(arena.newCapacity * arena.stride);
        desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | arena.usage;
        desc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const BufferHandle created = m_bufferAllocator.createBuffer(desc);
        if (created == kInvalidBufferHandle) {
            VOX_LOGE("render") << arena.label << " growth to " << arena.newCapacity
                               << " elements failed";
            failed = true;
            break;
        }
        createdHandles.push_back(created);
    }
    if (failed) {
        for (const BufferHandle handle : createdHandles) {
            if (handle != kInvalidBufferHandle) {
                m_bufferAllocator.destroyBuffer(handle);
            }
        }
        return false;
    }

    // Copy old -> new before swapping, so a failure leaves the arenas intact.
    uint64_t arenaCopyTimelineValue = 0;
    for (std::size_t i = 0; i < std::size(arenaBuffers) && !failed; ++i) {
        const ArenaBuffer& arena = arenaBuffers[i];
        if (createdHandles[i] == kInvalidBufferHandle) {
            continue;
        }
        const VkDeviceSize copyBytes =
            static_cast<VkDeviceSize>(arena.oldCapacity * arena.stride);
        if (copyBytes == 0u || *arena.handle == kInvalidBufferHandle) {
            continue;
        }
        uint64_t copyTimelineValue = 0;
        if (!copyBufferRange(
                *arena.handle, createdHandles[i], copyBytes, arena.label, copyTimelineValue)) {
            failed = true;
        }
        arenaCopyTimelineValue = std::max(arenaCopyTimelineValue, copyTimelineValue);
    }
    if (failed) {
        for (const BufferHandle handle : createdHandles) {
            if (handle != kInvalidBufferHandle) {
                m_bufferAllocator.destroyBuffer(handle);
            }
        }
        return false;
    }

    for (std::size_t i = 0; i < std::size(arenaBuffers); ++i) {
        if (createdHandles[i] == kInvalidBufferHandle) {
            continue;
        }
        // Deferred past BOTH the frames that may still be reading the old arena
        // and the copy that is reading it right now -- the copy signals a later
        // value than any submitted frame, so the frame value alone is not enough.
        scheduleBufferRelease(
            *arenaBuffers[i].handle,
            std::max(m_lastGraphicsTimelineValue, arenaCopyTimelineValue));
        *arenaBuffers[i].handle = createdHandles[i];
    }
    m_importedVertexArena.grow(newVertexCapacity);
    m_importedIndexArena.grow(newIndexCapacity);
    VOX_LOGI("render") << "imported geometry arena grown: vertices=" << newVertexCapacity
                       << ", indices=" << newIndexCapacity;
    return true;
}

void RendererBackend::removeImportedSceneChunk(std::size_t chunkIndex) {
    if (chunkIndex >= m_importedSceneChunks.size()) {
        return;
    }
    ImportedSceneChunk& chunk = m_importedSceneChunks[chunkIndex];
    if (!chunk.alive) {
        return;
    }

    m_importedVertexArena.free(chunk.firstVertex, chunk.vertexCount);
    m_importedIndexArena.free(chunk.firstIndex, chunk.indexCount);
    // Only the last chunk referencing a texture actually frees it; the slot
    // table owns that decision.
    for (const std::uint32_t slot : chunk.textureSlots) {
        releaseImportedTexture(slot);
    }

    chunk.alive = false;
    chunk.draws.clear();
    chunk.draws.shrink_to_fit();
    chunk.pageRanges.clear();
    chunk.pageRanges.shrink_to_fit();
    chunk.textureSlots.clear();
    chunk.textureSlots.shrink_to_fit();
    // Lights were being left on the dead chunk. rebuildImportedDrawTables skips
    // dead chunks so they never reached the frame, but the storage stayed for the
    // rest of the session -- and an interior cell carries a lot of them.
    chunk.lights.clear();
    chunk.lights.shrink_to_fit();
    chunk.waterPatches.clear();
    chunk.waterPatches.shrink_to_fit();
    chunk.vertexCount = 0;
    chunk.indexCount = 0;
    chunk.terrainDrawCount = 0;
    // Recycle the slot. Without this the vector only ever grows: a session that
    // streams thousands of cells leaves thousands of dead entries behind, and
    // rebuildImportedDrawTables walks all of them on every single add and remove.
    m_freeImportedSceneChunks.push_back(chunkIndex);

    rebuildImportedDrawTables();
    rebuildImportedWaterBuffers();
}

void RendererBackend::rebuildImportedDrawTables() {
    // The caster set is changing; every cached shadow-cascade tile may show
    // geometry that no longer exists (or miss geometry that now does).
    m_shadowRenderedValid = {};
    m_importedMeshDraws.clear();
    m_importedPageDrawRanges.clear();
    // Rebuilt, not appended to, so evicting a chunk drops its lights. This runs
    // on both add and remove, which is what makes that true in both directions.
    m_importedLocalLights.clear();
    std::uint32_t terrainDrawTotal = 0;
    std::uint32_t staticDrawTotal = 0;
    for (const ImportedSceneChunk& chunk : m_importedSceneChunks) {
        if (!chunk.alive) {
            continue;
        }
        // Each page range's firstDraw is chunk-relative on the chunk; rebase it
        // onto where this chunk's draws land in the flat table.
        const std::uint32_t chunkDrawBase = static_cast<std::uint32_t>(m_importedMeshDraws.size());
        m_importedMeshDraws.insert(
            m_importedMeshDraws.end(), chunk.draws.begin(), chunk.draws.end());
        for (const ImportedScenePageDrawRange& pageRange : chunk.pageRanges) {
            ImportedScenePageDrawRange rebased = pageRange;
            rebased.firstDraw += chunkDrawBase;
            m_importedPageDrawRanges.push_back(rebased);
        }
        m_importedLocalLights.insert(
            m_importedLocalLights.end(), chunk.lights.begin(), chunk.lights.end());
        terrainDrawTotal += chunk.terrainDrawCount;
        staticDrawTotal +=
            static_cast<std::uint32_t>(chunk.draws.size()) - chunk.terrainDrawCount;
    }
    m_importedTerrainDrawCount = terrainDrawTotal;
    m_importedStaticDrawCount = staticDrawTotal;
    // Total live indices across every chunk. Several callers use a non-zero
    // value purely as "is there imported geometry at all", so it has to fall to
    // zero when the last chunk is evicted.
    std::uint64_t liveIndexCount = 0;
    for (const ImportedSceneChunk& chunk : m_importedSceneChunks) {
        if (chunk.alive) {
            liveIndexCount += chunk.indexCount;
        }
    }
    m_importedIndexCount = static_cast<std::uint32_t>(liveIndexCount);
}

void RendererBackend::rebuildImportedWaterBuffers() {
    // Bethesda authors no water geometry at all: a cell states one height and
    // the engine fills that cell's 4096-unit footprint at it. So the entire
    // resident water surface is four vertices per water-bearing cell -- 81 cells
    // at the default load radius, and most of them dry. Regenerating the whole
    // thing is cheaper than any scheme for patching it in place.
    std::vector<ImportedWaterVertex> waterVertices;
    std::vector<std::uint32_t> waterIndices;
    for (const ImportedSceneChunk& chunk : m_importedSceneChunks) {
        if (!chunk.alive) {
            continue;
        }
        for (const odai::importer::ImportedSceneWaterPatch& patch : chunk.waterPatches) {
            const std::uint32_t baseVertex = static_cast<std::uint32_t>(waterVertices.size());
            ImportedWaterVertex vertex{};
            vertex.position[1] = patch.waterLevel;
            vertex.position[0] = patch.originX;
            vertex.position[2] = patch.originZ;
            vertex.uv[0] = 0.0f;
            vertex.uv[1] = 0.0f;
            waterVertices.push_back(vertex);

            vertex.position[0] = patch.originX + patch.sizeX;
            vertex.position[2] = patch.originZ;
            vertex.uv[0] = 1.0f;
            waterVertices.push_back(vertex);

            vertex.position[0] = patch.originX + patch.sizeX;
            vertex.position[2] = patch.originZ + patch.sizeZ;
            vertex.uv[1] = 1.0f;
            waterVertices.push_back(vertex);

            vertex.position[0] = patch.originX;
            vertex.position[2] = patch.originZ + patch.sizeZ;
            vertex.uv[0] = 0.0f;
            waterVertices.push_back(vertex);

            waterIndices.push_back(baseVertex + 0u);
            waterIndices.push_back(baseVertex + 2u);
            waterIndices.push_back(baseVertex + 1u);
            waterIndices.push_back(baseVertex + 0u);
            waterIndices.push_back(baseVertex + 3u);
            waterIndices.push_back(baseVertex + 2u);
        }
    }

    // The early-out is what keeps this affordable. Streaming across dry ground
    // -- which is most of the Mojave and most of Tamriel -- adds and evicts
    // cells constantly while the water set never changes, and the rebuild below
    // stalls the device. Comparing the built vertices rather than the patch
    // count catches a same-sized set at a different height too.
    if (waterVertices.size() == m_importedWaterVerticesResident.size() &&
        std::memcmp(
            waterVertices.data(), m_importedWaterVerticesResident.data(),
            waterVertices.size() * sizeof(ImportedWaterVertex)) == 0) {
        return;
    }

    // Destroying a buffer the GPU may still be reading is the one real hazard
    // here, and this takes the same way out the fog-map re-upload does: wait for
    // idle first. Affordable only because of the early-out above -- crossing a
    // shoreline is a handful of events in a session, not a per-cell cost.
    if (m_importedWaterVertexBufferHandle != kInvalidBufferHandle ||
        m_importedWaterIndexBufferHandle != kInvalidBufferHandle) {
        vkDeviceWaitIdle(m_device);
        if (m_importedWaterVertexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(m_importedWaterVertexBufferHandle);
            m_importedWaterVertexBufferHandle = kInvalidBufferHandle;
        }
        if (m_importedWaterIndexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(m_importedWaterIndexBufferHandle);
            m_importedWaterIndexBufferHandle = kInvalidBufferHandle;
        }
    }
    m_importedWaterIndexCount = 0;
    m_importedWaterVerticesResident = waterVertices;
    if (waterVertices.empty() || waterIndices.empty()) {
        return;
    }

    // Host-visible rather than device-local with a staging copy: this is a few
    // kilobytes read once per pass, so the transfer machinery would cost more
    // than the slower reads ever will.
    const auto createHostBuffer = [this](
                                      const void* data,
                                      VkDeviceSize bytes,
                                      VkBufferUsageFlags usage) -> BufferHandle {
        BufferCreateDesc desc{};
        desc.size = bytes;
        desc.usage = usage;
        desc.memoryProperties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        desc.initialData = data;
        return m_bufferAllocator.createBuffer(desc);
    };

    const BufferHandle vertexHandle = createHostBuffer(
        waterVertices.data(),
        static_cast<VkDeviceSize>(waterVertices.size() * sizeof(ImportedWaterVertex)),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if (vertexHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "streamed water vertex buffer allocation failed";
        m_importedWaterVerticesResident.clear();
        return;
    }
    const BufferHandle indexHandle = createHostBuffer(
        waterIndices.data(),
        static_cast<VkDeviceSize>(waterIndices.size() * sizeof(std::uint32_t)),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    if (indexHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "streamed water index buffer allocation failed";
        m_bufferAllocator.destroyBuffer(vertexHandle);
        m_importedWaterVerticesResident.clear();
        return;
    }

    m_importedWaterVertexBufferHandle = vertexHandle;
    m_importedWaterIndexBufferHandle = indexHandle;
    m_importedWaterIndexCount = static_cast<std::uint32_t>(waterIndices.size());
    VOX_LOGI("render") << "streamed water: " << (waterVertices.size() / 4u)
                       << " cell patch(es)";
}

bool RendererBackend::ensureImportedTextureSampler() {
    if (m_importedTextureSampler != VK_NULL_HANDLE) {
        return true;
    }
    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    // MIP BIAS FOR UPSCALING. Rendering at a fraction of the output resolution
    // means every texture lookup picks a mip chosen for the SMALLER grid, so a
    // surface that would have sampled mip 1 at native samples mip 2 -- and the
    // upscaler is then asked to reconstruct detail that was never rasterized.
    // The result reads as "low resolution" in a way no amount of temporal
    // accumulation can recover, because the information is not in any frame.
    //
    // log2(renderScale) is the standard correction and is what Intel's XeSS
    // guide means by "adjust mip bias": at 0.5 scale it is -1, restoring the mip
    // the native-resolution frame would have picked. Clamped at -2 because
    // beyond that the sampler is fetching detail the low-res grid genuinely
    // cannot hold and it just aliases.
    //
    // 0 when not upscaling, so every other game and the native path are
    // unaffected.
    const float renderScaleForMip =
        (m_swapchainExtent.width > 0u && m_renderExtent.width > 0u)
            ? (static_cast<float>(m_renderExtent.width) /
               static_cast<float>(m_swapchainExtent.width))
            : 1.0f;
    //
    // ODAI_UPSCALE_MIPBIAS overrides the computed value. The full log2 bias is
    // the textbook answer but it assumes the upscaler can resolve the extra
    // detail temporally, and distant terrain is exactly where it cannot: a
    // sub-pixel sliver of hillside is a different surface every jitter phase, so
    // the restored detail arrives as aliasing that never converges. The right
    // value is therefore a measured trade, not a derivation.
    // -0.5 rather than the textbook log2(0.5) = -1, measured on the distant
    // Goodsprings skyline against a native-resolution reference:
    //
    //   bias    far-band detail   error vs native
    //    0.0        10.01              4.936
    //   -0.5        10.87              4.824   <- best on both
    //   -1.0        11.75              4.926
    //
    // The full bias does restore more high-frequency energy, but past -0.5 the
    // extra is aliasing the temporal pass cannot resolve -- distant sub-pixel
    // geometry is a different surface every jitter phase -- so error rises again
    // even as "detail" does. Half the textbook value is where the restored
    // detail is still real.
    static const float s_mipBiasOverride = []() {
        const char* env = std::getenv("ODAI_UPSCALE_MIPBIAS");
        return (env != nullptr) ? static_cast<float>(std::atof(env)) : 1.0f;
    }();
    samplerCreateInfo.mipLodBias = (renderScaleForMip < 0.999f)
        ? ((s_mipBiasOverride <= 0.5f) ? s_mipBiasOverride : -0.5f)
        : 0.0f;
    VOX_LOGI("render") << "imported texture sampler: mip bias " << samplerCreateInfo.mipLodBias
                       << " (render " << m_renderExtent.width << "x" << m_renderExtent.height
                       << ", swapchain " << m_swapchainExtent.width << "x"
                       << m_swapchainExtent.height << ")";
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
    return true;
}

// Uploads the cloud layers of one weather. Called when the weather changes, not
// per frame, so the one-shot command pool here is not a hot path.
//
// Slots come from the same refcounted table imported geometry uses, which is
// what makes the shared case free: two weathers naming the same cloud texture
// (and they do -- "sky\alpha.dds", the empty-layer placeholder, appears in
// nearly every record) resolve to one upload.
static_assert(
    kInvalidImportedTextureSlot == ~0u,
    "m_weatherCloudSlots in renderer_backend.h initializes to ~0u because that header "
    "cannot see this constant; if it changes, that initializer must change with it");

void RendererBackend::setWeatherClouds(const WeatherCloudTextures& clouds) {
    // Release first, then acquire: the refcount makes re-acquiring a texture the
    // previous weather also used a no-op rather than an upload, and doing it in
    // this order keeps the count from briefly dropping to zero and freeing the
    // image we are about to ask for again.
    std::uint32_t previousSlots[kWeatherCloudLayerCount];
    for (int layer = 0; layer < kWeatherCloudLayerCount; ++layer) {
        previousSlots[layer] = m_weatherCloudSlots[layer];
        m_weatherCloudSlots[layer] = kInvalidImportedTextureSlot;
        m_weatherCloudLayers[layer] = clouds.layers[layer];
        // The pixels are already in the bindless table (or about to be); the
        // copy here is only the drawing parameters, so drop the payload rather
        // than keeping a second copy of every cloud texture alive per frame.
        m_weatherCloudLayers[layer].texture = odai::importer::ImportedSceneTexture{};
    }

    // Same gate acquireImportedTexture applies; checking it here avoids
    // building a command pool only to have every acquire refuse.
    if (!m_supportsBindlessDescriptors || !m_bindlessBufferSet.valid() ||
        m_bindlessTextureCapacity <= kBindlessTextureStaticCount) {
        for (const std::uint32_t slot : previousSlots) {
            releaseImportedTexture(slot);
        }
        return;
    }

    bool anyLayer = false;
    for (const auto& layer : clouds.layers) {
        anyLayer = anyLayer || !layer.texture.rgba8.empty();
    }
    if (!anyLayer) {
        for (const std::uint32_t slot : previousSlots) {
            releaseImportedTexture(slot);
        }
        return;
    }

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    commandPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    if (vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
        VOX_LOGE("render") << "weather cloud upload command pool creation failed";
        for (const std::uint32_t slot : previousSlots) {
            releaseImportedTexture(slot);
        }
        return;
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
        VOX_LOGE("render") << "weather cloud upload command buffer setup failed";
        scheduleCommandPoolRelease(commandPool, 0);
        for (const std::uint32_t slot : previousSlots) {
            releaseImportedTexture(slot);
        }
        return;
    }

    std::vector<BufferHandle> stagingBufferHandles;
    for (int layer = 0; layer < kWeatherCloudLayerCount; ++layer) {
        const odai::importer::ImportedSceneTexture& texture = clouds.layers[layer].texture;
        if (texture.rgba8.empty()) {
            continue;
        }
        m_weatherCloudSlots[layer] = acquireImportedTexture(
            normalizedImportedTextureKey(texture.sourcePath), texture, commandBuffer,
            stagingBufferHandles);
    }

    // Now that every acquire has run, drop the old references.
    for (const std::uint32_t slot : previousSlots) {
        releaseImportedTexture(slot);
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
            logVkFailure("vkQueueSubmit2(weatherCloudUpload)", submitResult);
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
}

std::uint32_t RendererBackend::acquireImportedTexture(
    const std::string& key,
    const odai::importer::ImportedSceneTexture& srcTexture,
    VkCommandBuffer commandBuffer,
    std::vector<BufferHandle>& stagingBuffers) {
    if (!m_supportsBindlessDescriptors ||
        !m_bindlessBufferSet.valid() ||
        m_bindlessTextureCapacity <= kBindlessTextureStaticCount ||
        commandBuffer == VK_NULL_HANDLE) {
        return kInvalidImportedTextureSlot;
    }

    if (srcTexture.width == 0u || srcTexture.height == 0u || srcTexture.rgba8.empty()) {
        return kInvalidImportedTextureSlot;
    }

    const std::uint32_t inferredMipLevelCount =
        inferImportedTextureMipLevelCount(srcTexture.width, srcTexture.height, srcTexture.rgba8.size());
    std::uint32_t mipLevelCount = 0;
    if (srcTexture.format != odai::importer::TextureFormat::RGBA8) {
        // Block-compressed: trust the mip count stored by the DDS loader.
        if (srcTexture.mipLevelCount == 0u) {
            VOX_LOGW("render") << "block-compressed texture missing mip data: "
                               << srcTexture.sourcePath << "; skipping";
            return kInvalidImportedTextureSlot;
        }
        mipLevelCount = srcTexture.mipLevelCount;
    } else {
        if (inferredMipLevelCount == 0u) {
            VOX_LOGW("render") << "imported texture mip chain size invalid for "
                               << srcTexture.sourcePath << "; skipping texture";
            return kInvalidImportedTextureSlot;
        }
        mipLevelCount = inferredMipLevelCount;
        if (srcTexture.mipLevelCount != 0u && srcTexture.mipLevelCount != inferredMipLevelCount) {
            VOX_LOGW("render") << "imported texture mip metadata mismatch for "
                               << srcTexture.sourcePath << "; stored=" << srcTexture.mipLevelCount
                               << ", inferred=" << inferredMipLevelCount
                               << " (using inferred chain)";
        }
    }

    if (!ensureImportedTextureSampler()) {
        return kInvalidImportedTextureSlot;
    }

    // Slot choice, reference counting and key lookup all belong to the table.
    // If it says the key is already resident there is nothing to upload -- this
    // is the whole point of keying by path, since streaming asks for the same
    // ground/rock diffuse from every cell that uses it.
    m_importedTextureSlotTable.setCapacity(m_bindlessTextureCapacity - kBindlessTextureStaticCount);
    const BindlessSlotTable::Acquisition acquisition = m_importedTextureSlotTable.acquire(key);
    if (acquisition.slotIndex == kInvalidSlotIndex) {
        return kInvalidImportedTextureSlot;
    }
    const std::uint32_t slotIndex = acquisition.slotIndex;
    if (!acquisition.needsUpload) {
        return static_cast<std::uint32_t>(kBindlessTextureStaticCount + slotIndex);
    }
    // Every failure from here on must abandon the slot, or a key that failed to
    // upload would stay addressable and resolve to a null image forever.
    if (m_importedTextureResources.size() <= slotIndex) {
        m_importedTextureResources.resize(slotIndex + 1u);
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
        m_importedTextureSlotTable.abandon(slotIndex);
        return kInvalidImportedTextureSlot;
    }
    stagingBuffers.push_back(stagingHandle);

    const VkFormat textureFormat = vkFormatForImportedTexture(srcTexture.format);

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = textureFormat;
    imageCreateInfo.extent = {srcTexture.width, srcTexture.height, 1};
    imageCreateInfo.mipLevels = mipLevelCount;
    // ODAI_DEBUG_TEXTURE_MIPS=1: census of what mip chains actually reach the
    // GPU. "The loader decodes mips" and "the sampler has maxLod=16" are
    // necessary but not sufficient -- a texture arriving here with
    // mipLevelCount 1 shimmers under minification no matter what the sampler
    // is configured to do, and nothing else in the frame says so.
    if (std::getenv("ODAI_DEBUG_TEXTURE_MIPS") != nullptr) {
        static std::uint32_t s_single = 0;
        static std::uint32_t s_chained = 0;
        ((mipLevelCount <= 1u) ? s_single : s_chained)++;
        if (((s_single + s_chained) % 64u) == 0u) {
            VOX_LOGI("render") << "texture mip census: singleMip=" << s_single
                               << " withChain=" << s_chained
                               << " (last: " << srcTexture.sourcePath << " mips=" << mipLevelCount
                               << " " << srcTexture.width << "x" << srcTexture.height << ")";
        }
    }
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
        m_vmaAllocator, &imageCreateInfo, &allocationCreateInfo,
        &resource.image, &resource.allocation, nullptr);
    if (imageCreateResult != VK_SUCCESS) {
        logVkFailure("vmaCreateImage(importedTexture)", imageCreateResult);
        m_importedTextureSlotTable.abandon(slotIndex);
        return kInvalidImportedTextureSlot;
    }

    VkImageViewCreateInfo viewCreateInfo{};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = resource.image;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = textureFormat;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = mipLevelCount;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;
    const VkResult imageViewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &resource.imageView);
    if (imageViewResult != VK_SUCCESS) {
        logVkFailure("vkCreateImageView(importedTexture)", imageViewResult);
        vmaDestroyImage(m_vmaAllocator, resource.image, resource.allocation);
        m_importedTextureSlotTable.abandon(slotIndex);
        return kInvalidImportedTextureSlot;
    }

    setObjectName(
        VK_OBJECT_TYPE_IMAGE,
        vkHandleToUint64(resource.image),
        ("imported.texture.image." + std::to_string(slotIndex)).c_str());
    setObjectName(
        VK_OBJECT_TYPE_IMAGE_VIEW,
        vkHandleToUint64(resource.imageView),
        ("imported.texture.view." + std::to_string(slotIndex)).c_str());

    transitionImageLayout(
        commandBuffer, resource.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, mipLevelCount);

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

    transitionImageLayout(
        commandBuffer, resource.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, mipLevelCount);

    m_importedTextureResources[slotIndex] = resource;
    return static_cast<std::uint32_t>(kBindlessTextureStaticCount + slotIndex);
}

void RendererBackend::releaseImportedTexture(std::uint32_t slot) {
    if (slot == kInvalidImportedTextureSlot || slot < kBindlessTextureStaticCount) {
        return;
    }
    const std::uint32_t slotIndex = slot - kBindlessTextureStaticCount;
    if (slotIndex >= m_importedTextureResources.size()) {
        return;
    }
    if (!m_importedTextureSlotTable.release(slotIndex)) {
        return;  // still referenced by another resident cell
    }
    ImportedTextureResource& resource = m_importedTextureResources[slotIndex];
    // Deferred: an in-flight frame may still be sampling this image.
    scheduleImageRelease(
        resource.image, resource.allocation, resource.imageView, m_lastGraphicsTimelineValue);
    resource = ImportedTextureResource{};
}

bool RendererBackend::uploadImportedSceneInternal(
    const odai::importer::ImportedScene& scene,
    const odai::importer::GpuSceneAsset* gpuScene,
    bool appendChunk
) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }

    // Append mode keeps every resident chunk and adds one more -- this is the
    // path streaming uses. Replace mode is the original whole-scene load and
    // still tears everything down first.
    if (!appendChunk) {
        clearImportedSceneMeshes();
        destroyRayTracingScene();
    }

    // Copy ONLY when the packed stream has to be rebuilt.
    //
    // This used to deep-copy every scene unconditionally so that
    // buildImportedScenePackedRenderData could mutate it -- the one and only
    // mutation in this whole function. A streamed cell always arrives with its
    // packed stream already built (the cooker and CellSceneBuilder both finish
    // with it), so that copy was ~10 MB of memcpy per cell, including a full
    // duplicate of every texture, thrown away moments later.
    // Per-phase timing for the chunk apply, env-gated. Worth keeping: it is what
    // showed that fusing the two vertex passes did NOT help, and that the cost
    // is spread rather than concentrated in one loop.
    const bool logChunkTiming = std::getenv("ODAI_DEBUG_CHUNK_TIMING") != nullptr;
    const core::Stopwatch chunkPhaseTimer;
    float textureMs = 0.0f;
    float vertexMs = 0.0f;
    float geometryUploadMs = 0.0f;
    float drawBuildMs = 0.0f;
    core::Stopwatch phaseTimer;

    const bool havePackedScene =
        !scene.packedVertices.empty() &&
        !scene.packedIndices.empty() &&
        !scene.packedDraws.empty();
    odai::importer::ImportedScene rebuiltScene;
    if (!havePackedScene) {
        VOX_LOGI("render") << "imported scene missing packed geometry cache; rebuilding render stream on load";
        rebuiltScene = scene;
        odai::importer::buildImportedScenePackedRenderData(rebuiltScene);
    }
    const odai::importer::ImportedScene& uploadScene = havePackedScene ? scene : rebuiltScene;
    if (havePackedScene) {
        VOX_LOGI("render") << "imported scene using packed geometry cache (vertices="
                           << uploadScene.packedVertices.size()
                           << ", indices=" << uploadScene.packedIndices.size()
                           << ", draws=" << uploadScene.packedDraws.size() << ")";
    }

    phaseTimer.restart();
    std::vector<std::uint32_t> importedTextureSlots(
        uploadScene.textures.size(),
        std::numeric_limits<std::uint32_t>::max());

    // The chunk built at the bottom of this function is what OWNS these slots:
    // ImportedSceneChunk::textureSlots is the only thing removeImportedSceneChunk
    // can release from. Every return between the acquires below and that chunk
    // existing therefore pins each acquired texture at refcount >= 1 for the rest
    // of the process -- there is no other handle to it. Streaming makes that
    // unbounded rather than merely untidy: a cell that fails to upload is marked
    // evicted, not unavailable, so the planner offers it again, and each attempt
    // leaves another reference behind. This guard undoes the acquires on any path
    // that does not reach the chunk.
    //
    // Disarmed (committed) in two places: when the chunk takes the slots, and
    // before clearImportedSceneMeshes(), which tears the whole table down and
    // would otherwise be double-released.
    struct TextureSlotAcquireGuard {
        RendererBackend* backend = nullptr;
        const std::vector<std::uint32_t>* slots = nullptr;
        bool committed = false;
        void commit() { committed = true; }
        ~TextureSlotAcquireGuard() {
            if (committed || backend == nullptr || slots == nullptr) {
                return;
            }
            for (const std::uint32_t slot : *slots) {
                backend->releaseImportedTexture(slot);
            }
        }
    } textureSlotGuard{this, &importedTextureSlots, false};

    if (m_supportsBindlessDescriptors &&
        m_bindlessBufferSet.valid() &&
        m_bindlessTextureCapacity > kBindlessTextureStaticCount &&
        !uploadScene.textures.empty()) {
        if (!ensureImportedTextureSampler()) {
            return false;
        }

        std::vector<BufferHandle> stagingBufferHandles;
        stagingBufferHandles.reserve(uploadScene.textures.size());
        bool textureUploadFailed = false;
        std::size_t uploadedTextureCount = 0u;   // textures this call actually created
        std::size_t acquiredTextureCount = 0u;   // slots resolved, shared or new
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

        // Slot assignment, refcounting and the image upload itself now live in
        // acquireImportedTexture. Keying by source path means a scene naming the
        // same .dds from several materials uploads it once, and it is the same
        // path streaming uses to share a texture across cells.
        // Distinguish acquires from actual uploads. A streamed chunk normally
        // shares most of its textures with chunks already resident, so counting
        // acquires here would report a full upload every time and make working
        // deduplication look broken.
        const std::size_t residentBeforeUploads = m_importedTextureSlotTable.residentCount();
        for (std::size_t textureIndex = 0; textureIndex < uploadScene.textures.size(); ++textureIndex) {
            const odai::importer::ImportedSceneTexture& srcTexture = uploadScene.textures[textureIndex];
            const std::uint32_t slot = acquireImportedTexture(
                normalizedImportedTextureKey(srcTexture.sourcePath),
                srcTexture,
                commandBuffer,
                stagingBufferHandles);
            if (slot == kInvalidImportedTextureSlot) {
                continue;
            }
            importedTextureSlots[textureIndex] = slot;
            ++acquiredTextureCount;
        }
        uploadedTextureCount =
            m_importedTextureSlotTable.residentCount() - residentBeforeUploads;

        if (!textureUploadFailed && commandBuffer != VK_NULL_HANDLE) {
            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                VOX_LOGE("render") << "imported texture upload command buffer end failed";
                textureUploadFailed = true;
            }
        }
        // Submit WITHOUT blocking the CPU.
        //
        // This used to be submit + vkQueueWaitIdle(m_graphicsQueue), and the
        // wait is what made a streamed cell cost hundreds of milliseconds:
        // vkQueueWaitIdle drains the ENTIRE graphics queue, so uploading one
        // cell's textures waited on every frame already in flight, not just on
        // the copy that was actually issued.
        //
        // Instead signal a timeline value and record it in
        // m_pendingTransferTimelineValue, which the next frame's graphics
        // submit already waits on (frame_run.cc). The GPU still orders the copy
        // before any sampling; the CPU simply stops standing there.
        //
        // Staying on the graphics queue is deliberate. The transfer ring would
        // be the obvious home, but m_transferQueueFamilyIndex can be a DISTINCT
        // family, and these uploads end with a barrier to SHADER_READ_ONLY at
        // VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT -- not a legal stage on a
        // transfer-only queue, and crossing families would additionally need
        // ownership transfers on every image.
        uint64_t textureUploadTimelineValue = 0;
        if (!textureUploadFailed && commandBuffer != VK_NULL_HANDLE) {
            textureUploadTimelineValue = m_nextTimelineValue++;

            VkSemaphoreSubmitInfo signalInfo{};
            signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signalInfo.semaphore = m_renderTimelineSemaphore;
            signalInfo.value = textureUploadTimelineValue;
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
                logVkFailure("vkQueueSubmit2(importedTextureUpload)", submitResult);
                textureUploadFailed = true;
                textureUploadTimelineValue = 0;
            } else {
                m_pendingTransferTimelineValue =
                    std::max(m_pendingTransferTimelineValue, textureUploadTimelineValue);
            }
        }

        // The staging buffers and the pool are still being read by that submit,
        // so they retire on its timeline value rather than here. A zero value
        // (nothing was submitted) frees them immediately.
        for (const BufferHandle stagingHandle : stagingBufferHandles) {
            if (stagingHandle != kInvalidBufferHandle) {
                scheduleBufferRelease(stagingHandle, textureUploadTimelineValue);
            }
        }
        scheduleCommandPoolRelease(commandPool, textureUploadTimelineValue);
        commandPool = VK_NULL_HANDLE;
        if (textureUploadFailed) {
            // Full teardown clears the slot table itself, so the guard must not
            // also release into it.
            textureSlotGuard.commit();
            clearImportedSceneMeshes();
            return false;
        }
        // uploadedTextureCount counts distinct slots, so a scene that reuses one
        // .dds across materials now legitimately reports fewer uploads than it
        // has texture entries. Only warn when slots actually ran out.
        const std::size_t uniqueTextureCount = m_importedTextureSlotTable.residentCount();
        if (m_importedTextureSlotTable.slotCount() >= m_importedTextureSlotTable.capacity() &&
            uploadedTextureCount < uploadScene.textures.size()) {
            VOX_LOGW("render") << "imported texture set truncated by bindless capacity: uploaded "
                               << uploadedTextureCount << " of " << uploadScene.textures.size();
        } else if (acquiredTextureCount > 0u) {
            VOX_LOGI("render") << "imported textures: uploaded " << uploadedTextureCount
                               << ", shared " << (acquiredTextureCount - uploadedTextureCount)
                               << " (resident=" << uniqueTextureCount
                               << ", entries=" << uploadScene.textures.size() << ")";
        }
    } else if (!uploadScene.textures.empty()) {
        VOX_LOGW("render") << "imported textures unavailable because bindless texture sampling is not ready";
    }
    m_importedTextureSlots = importedTextureSlots;

    // Upload fog-of-war visibility texture when the scene carries one.
    m_fogMapEnabled = false;
    m_fogMapInvExtentX = 0.0f;
    m_fogMapInvExtentZ = 0.0f;
    if (!uploadScene.fogMap.empty() &&
        uploadScene.fogMapW > 0 && uploadScene.fogMapH > 0 &&
        m_supportsBindlessDescriptors && m_bindlessBufferSet.valid() &&
        m_vmaAllocator != VK_NULL_HANDLE) {

        // Destroy previous fog texture so we always use the latest.
        if (m_fogMapTextureResource.imageView != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(m_device);
            vkDestroyImageView(m_device, m_fogMapTextureResource.imageView, nullptr);
            m_fogMapTextureResource.imageView = VK_NULL_HANDLE;
        }
        if (m_fogMapTextureResource.image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_fogMapTextureResource.image, m_fogMapTextureResource.allocation);
            m_fogMapTextureResource.image = VK_NULL_HANDLE;
            m_fogMapTextureResource.allocation = VK_NULL_HANDLE;
        }

        // Linear, clamp-to-edge sampler (created once, reused on re-uploads).
        if (m_fogMapSampler == VK_NULL_HANDLE) {
            VkSamplerCreateInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            si.magFilter = VK_FILTER_LINEAR;
            si.minFilter = VK_FILTER_LINEAR;
            si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            si.maxLod = 0.0f;
            si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            if (vkCreateSampler(m_device, &si, nullptr, &m_fogMapSampler) != VK_SUCCESS) {
                VOX_LOGE("render") << "fog map sampler creation failed";
            }
        }

        const std::uint32_t fogW = uploadScene.fogMapW;
        const std::uint32_t fogH = uploadScene.fogMapH;
        bool fogUploadOk = false;

        BufferCreateDesc fogStagingDesc{};
        fogStagingDesc.size = static_cast<VkDeviceSize>(uploadScene.fogMap.size());
        fogStagingDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        fogStagingDesc.memoryProperties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        fogStagingDesc.initialData = uploadScene.fogMap.data();
        const BufferHandle fogStaging = m_bufferAllocator.createBuffer(fogStagingDesc);

        if (fogStaging != kInvalidBufferHandle) {
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = VK_FORMAT_R8_UNORM;
            ici.extent = {fogW, fogH, 1};
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo fogAllocInfo{};
            fogAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            fogAllocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

            if (vmaCreateImage(m_vmaAllocator, &ici, &fogAllocInfo,
                               &m_fogMapTextureResource.image,
                               &m_fogMapTextureResource.allocation, nullptr) == VK_SUCCESS) {
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = m_fogMapTextureResource.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format = VK_FORMAT_R8_UNORM;
                vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                if (vkCreateImageView(m_device, &vci, nullptr,
                                      &m_fogMapTextureResource.imageView) == VK_SUCCESS) {
                    // Upload via a transient command buffer.
                    VkCommandPool fogPool = VK_NULL_HANDLE;
                    VkCommandPoolCreateInfo pi{};
                    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                    pi.queueFamilyIndex = m_graphicsQueueFamilyIndex;
                    pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
                    if (vkCreateCommandPool(m_device, &pi, nullptr, &fogPool) == VK_SUCCESS) {
                        VkCommandBuffer fogCmd = VK_NULL_HANDLE;
                        VkCommandBufferAllocateInfo cai{};
                        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                        cai.commandPool = fogPool;
                        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                        cai.commandBufferCount = 1;
                        if (vkAllocateCommandBuffers(m_device, &cai, &fogCmd) == VK_SUCCESS) {
                            VkCommandBufferBeginInfo bi{};
                            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                            if (vkBeginCommandBuffer(fogCmd, &bi) == VK_SUCCESS) {
                                transitionImageLayout(
                                    fogCmd,
                                    m_fogMapTextureResource.image,
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_PIPELINE_STAGE_2_NONE,
                                    VK_ACCESS_2_NONE,
                                    VK_PIPELINE_STAGE_2_COPY_BIT,
                                    VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT);

                                VkBufferImageCopy bic{};
                                bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                                bic.imageExtent = {fogW, fogH, 1};
                                vkCmdCopyBufferToImage(
                                    fogCmd,
                                    m_bufferAllocator.getBuffer(fogStaging),
                                    m_fogMapTextureResource.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);

                                transitionImageLayout(
                                    fogCmd,
                                    m_fogMapTextureResource.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_PIPELINE_STAGE_2_COPY_BIT,
                                    VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                    VK_IMAGE_ASPECT_COLOR_BIT);

                                if (vkEndCommandBuffer(fogCmd) == VK_SUCCESS) {
                                    if (submitCommandBufferOneShot(m_graphicsQueue, fogCmd, VK_NULL_HANDLE) == VK_SUCCESS &&
                                        vkQueueWaitIdle(m_graphicsQueue) == VK_SUCCESS) {
                                        fogUploadOk = true;
                                    }
                                }
                            }
                        }
                        vkDestroyCommandPool(m_device, fogPool, nullptr);
                    }
                }
            }
            m_bufferAllocator.destroyBuffer(fogStaging);
        }

        if (fogUploadOk) {
            m_fogMapEnabled = true;
            m_fogMapInvExtentX = uploadScene.fogMapInvExtentX;
            m_fogMapInvExtentZ = uploadScene.fogMapInvExtentZ;
            setObjectName(VK_OBJECT_TYPE_IMAGE,
                          vkHandleToUint64(m_fogMapTextureResource.image), "fog.map.image");
            setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW,
                          vkHandleToUint64(m_fogMapTextureResource.imageView), "fog.map.view");
            VOX_LOGI("render") << "uploaded fog map: " << fogW << "x" << fogH;
        } else {
            if (m_fogMapTextureResource.imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(m_device, m_fogMapTextureResource.imageView, nullptr);
                m_fogMapTextureResource.imageView = VK_NULL_HANDLE;
            }
            if (m_fogMapTextureResource.image != VK_NULL_HANDLE) {
                vmaDestroyImage(m_vmaAllocator, m_fogMapTextureResource.image, m_fogMapTextureResource.allocation);
                m_fogMapTextureResource.image = VK_NULL_HANDLE;
                m_fogMapTextureResource.allocation = VK_NULL_HANDLE;
            }
            VOX_LOGE("render") << "fog map upload failed";
        }
    }

    textureMs = phaseTimer.elapsedMs();
    phaseTimer.restart();

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
    // Collected per chunk, not appended to m_importedLocalLights: the flat list
    // is rebuilt from the live chunks below, so an evicted cell's lights go away
    // with it. Appending here instead leaked every light a streaming session had
    // ever loaded, since removeImportedSceneChunk could not identify them.
    std::vector<ImportedLocalLight> chunkLights;
    chunkLights.reserve(uploadScene.lights.size());
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
        chunkLights.push_back(light);
    }

    // Packed-source -> GPU-vertex conversion, in parallel. This is pure
    // per-vertex work -- an octahedral normal encode, an sRGB colour encode
    // (three pow() calls), and the bindless-slot remaps -- and it was the
    // single largest cost of applying a streamed chunk: measured with
    // ODAI_DEBUG_CHUNK_TIMING on Skyrim's persistent cell (2.06M vertices),
    // 85 ms of a 119 ms apply was this loop, run single-threaded on the
    // render thread while the frame stalled. Plain std::thread rather than a
    // job system on purpose: src/render/ has no job-system dependency, the
    // fan-out is once per streamed cell, and every worker writes a disjoint
    // range of a pre-sized vector, so there is nothing to synchronize but
    // join().
    core::Stopwatch subTimer;
    vertices.resize(uploadScene.packedVertices.size());
    std::vector<ImportedShadowVertex> shadowVertices(uploadScene.packedVertices.size());
    const float subResizeMs = subTimer.elapsedMs();
    subTimer.restart();
    {
        const auto convertRange = [&](std::size_t begin, std::size_t end) {
            for (std::size_t v = begin; v < end; ++v) {
                const odai::importer::ImportedScenePackedVertex& srcVertex =
                    uploadScene.packedVertices[v];
                ImportedMeshVertex dstVertex{};
                std::memcpy(dstVertex.position, srcVertex.position, sizeof(dstVertex.position));
                dstVertex.packedNormal =
                    odai::importer::packImportedVertexNormal(srcVertex.normal);
                dstVertex.packedColor = odai::importer::packImportedVertexColor(srcVertex.color);
                std::memcpy(dstVertex.uv, srcVertex.uv, sizeof(dstVertex.uv));
                dstVertex.flags = srcVertex.flags;
                if (srcVertex.textureIndex < importedTextureSlots.size()) {
                    dstVertex.textureIndex = importedTextureSlots[srcVertex.textureIndex];
                } else {
                    dstVertex.textureIndex = std::numeric_limits<std::uint32_t>::max();
                }
                // Terrain layer slots need the same scene-index -> bindless-slot
                // remap as textureIndex above. An unmapped layer becomes the
                // invalid slot and the shader skips it, rather than sampling
                // whatever descriptor happens to sit at the unremapped index.
                std::uint32_t remappedLayers[4] = {};
                for (std::size_t layer = 0; layer < 4; ++layer) {
                    const std::uint32_t sourceIndex = srcVertex.layerTextureIndex[layer];
                    remappedLayers[layer] = sourceIndex < importedTextureSlots.size()
                        ? importedTextureSlots[sourceIndex]
                        : std::numeric_limits<std::uint32_t>::max();
                }
                dstVertex.packedLayerTexture01 = odai::importer::packImportedVertexLayerPair(
                    remappedLayers[0], remappedLayers[1]);
                dstVertex.packedLayerTexture23 = odai::importer::packImportedVertexLayerPair(
                    remappedLayers[2], remappedLayers[3]);
                dstVertex.layerWeights = srcVertex.layerWeights;
                vertices[v] = dstVertex;
                // The compact shadow stream is derived in the same pass: it is
                // a strict projection of the vertex just built, and deriving it
                // in a second 2M-iteration loop afterwards was measured inside
                // the same phase this parallelism exists to shrink.
                ImportedShadowVertex shadowVertex{};
                shadowVertex.position[0] = dstVertex.position[0];
                shadowVertex.position[1] = dstVertex.position[1];
                shadowVertex.position[2] = dstVertex.position[2];
                shadowVertex.uv[0] = dstVertex.uv[0];
                shadowVertex.uv[1] = dstVertex.uv[1];
                shadowVertex.textureIndex = dstVertex.textureIndex;
                shadowVertex.flags = dstVertex.flags;
                shadowVertices[v] = shadowVertex;
            }
        };
        const std::size_t vertexCount = uploadScene.packedVertices.size();
        // Fan out only when it can pay for the thread launches: a typical
        // exterior cell is 30-70k vertices and converts in a few ms, and eight
        // thread spawns cost real time on their own. The threshold is where the
        // single-threaded loop starts to visibly outrun a frame.
        constexpr std::size_t kParallelConvertThreshold = 200000u;
        const std::size_t workerCount = std::min<std::size_t>(
            {std::size_t{8},
             std::max<std::size_t>(std::thread::hardware_concurrency(), 2u) - 1u,
             vertexCount / (kParallelConvertThreshold / 2u)});
        if (vertexCount >= kParallelConvertThreshold && workerCount >= 2u) {
            std::vector<std::thread> workers;
            workers.reserve(workerCount);
            const std::size_t stride = (vertexCount + workerCount - 1u) / workerCount;
            for (std::size_t worker = 0; worker < workerCount; ++worker) {
                const std::size_t begin = worker * stride;
                const std::size_t end = std::min(vertexCount, begin + stride);
                if (begin < end) {
                    workers.emplace_back(convertRange, begin, end);
                }
            }
            for (std::thread& worker : workers) {
                worker.join();
            }
        } else {
            convertRange(0u, vertexCount);
        }
    }
    const float subConvertMs = subTimer.elapsedMs();
    subTimer.restart();
    indices.assign(uploadScene.packedIndices.begin(), uploadScene.packedIndices.end());
    const bool importedSceneIsInterior =
        odai::importer::importedSceneSourceTagIsInterior(uploadScene.sourceTag);
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
                                bool blendedDraw,
                                bool twoSidedDraw,
                                std::uint8_t alphaThreshold,
                                const float (&drawCenter)[3],
                                std::uint32_t pageRangeIndex
                            ) {
        if (indexCount == 0) {
            return;
        }
        if (!draws.empty()) {
            ImportedMeshDraw& previous = draws.back();
            // Blendedness joins the merge key: an opaque run and a blended run
            // are recorded through different pipelines, so they can never be
            // folded into one vkCmdDrawIndexed even when their index ranges abut.
            if (lastMergedDrawWasTerrain == terrainDraw &&
                previous.blended == blendedDraw &&
                previous.twoSided == twoSidedDraw &&
                previous.alphaThreshold == alphaThreshold &&
                lastMergedPageRangeIndex == pageRangeIndex &&
                previous.firstIndex + previous.indexCount == firstIndex) {
                // Weight the merged centre by index count so a large shape does
                // not get dragged around by a small one folded in beside it.
                if (blendedDraw) {
                    const float previousWeight = static_cast<float>(previous.indexCount);
                    const float addedWeight = static_cast<float>(indexCount);
                    const float totalWeight = previousWeight + addedWeight;
                    for (int axis = 0; axis < 3; ++axis) {
                        previous.center[axis] =
                            ((previous.center[axis] * previousWeight) +
                             (drawCenter[axis] * addedWeight)) / totalWeight;
                    }
                }
                previous.indexCount += indexCount;
                return;
            }
        }
        ImportedMeshDraw draw{};
        draw.firstIndex = firstIndex;
        draw.indexCount = indexCount;
        draw.blended = blendedDraw;
        draw.twoSided = twoSidedDraw;
        draw.alphaThreshold = alphaThreshold;
        draw.center[0] = drawCenter[0];
        draw.center[1] = drawCenter[1];
        draw.center[2] = drawCenter[2];
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

    // Blendedness is a per-vertex flag, but it is uniform across a packed draw
    // (a draw is one NIF shape's triangles), so the first vertex decides.
    auto packedDrawIsBlended = [&](const odai::importer::ImportedScenePackedDraw& srcDraw) {
        if (srcDraw.firstIndex >= uploadScene.packedIndices.size()) {
            return false;
        }
        const std::uint32_t vertexIndex = uploadScene.packedIndices[srcDraw.firstIndex];
        if (vertexIndex >= uploadScene.packedVertices.size()) {
            return false;
        }
        return (uploadScene.packedVertices[vertexIndex].flags &
                odai::importer::kImportedSceneMaterialFlagAlphaBlend) != 0u;
    };
    auto packedDrawIsTwoSided = [&](const odai::importer::ImportedScenePackedDraw& srcDraw) {
        if (srcDraw.firstIndex >= uploadScene.packedIndices.size()) {
            return false;
        }
        const std::uint32_t vertexIndex = uploadScene.packedIndices[srcDraw.firstIndex];
        if (vertexIndex >= uploadScene.packedVertices.size()) {
            return false;
        }
        return (uploadScene.packedVertices[vertexIndex].flags &
                odai::importer::kImportedSceneMaterialFlagTwoSided) != 0u;
    };
    // AABB centre over the draw's own vertices. Only computed for blended draws
    // -- it exists purely to sort them, and walking every opaque draw's indices
    // to fill in a field nothing reads would be a full extra pass over the
    // scene's geometry for nothing.
    auto packedDrawCenter = [&](const odai::importer::ImportedScenePackedDraw& srcDraw,
                                float (&outCenter)[3]) {
        outCenter[0] = 0.0f;
        outCenter[1] = 0.0f;
        outCenter[2] = 0.0f;
        float boundsMin[3] = {
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()};
        float boundsMax[3] = {
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest()};
        const std::size_t lastIndex = std::min<std::size_t>(
            static_cast<std::size_t>(srcDraw.firstIndex) + srcDraw.indexCount,
            uploadScene.packedIndices.size());
        bool sawVertex = false;
        for (std::size_t i = srcDraw.firstIndex; i < lastIndex; ++i) {
            const std::uint32_t vertexIndex = uploadScene.packedIndices[i];
            if (vertexIndex >= uploadScene.packedVertices.size()) {
                continue;
            }
            const auto& position = uploadScene.packedVertices[vertexIndex].position;
            for (int axis = 0; axis < 3; ++axis) {
                boundsMin[axis] = std::min(boundsMin[axis], position[axis]);
                boundsMax[axis] = std::max(boundsMax[axis], position[axis]);
            }
            sawVertex = true;
        }
        if (!sawVertex) {
            return;
        }
        for (int axis = 0; axis < 3; ++axis) {
            outCenter[axis] = (boundsMin[axis] + boundsMax[axis]) * 0.5f;
        }
    };
    for (std::uint32_t drawIndex = 0; drawIndex < uploadScene.packedDraws.size(); ++drawIndex) {
        const odai::importer::ImportedScenePackedDraw& srcDraw = uploadScene.packedDraws[drawIndex];
        if (srcDraw.indexCount == 0) {
            continue;
        }
        const bool blendedDraw = packedDrawIsBlended(srcDraw);
        float drawCenter[3] = {0.0f, 0.0f, 0.0f};
        if (blendedDraw) {
            packedDrawCenter(srcDraw, drawCenter);
        }
        appendMergedDraw(
            srcDraw.firstIndex,
            srcDraw.indexCount,
            drawIndex < sourceTerrainDrawCount,
            blendedDraw,
            packedDrawIsTwoSided(srcDraw),
            srcDraw.alphaThreshold,
            drawCenter,
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
            result = submitCommandBufferOneShot(m_graphicsQueue, commandBuffer, VK_NULL_HANDLE);
            if (result != VK_SUCCESS) {
                logVkFailure("vkQueueSubmit2(importedGeometryUpload)", result);
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

    // Compact shadow stream: the same vertices with only what the cascades read.
    // Allocated from the same vertex range as the main stream, so one
    // ImportedMeshDraw::vertexOffset addresses both despite the differing stride.
    vertexMs = phaseTimer.elapsedMs();
    if (logChunkTiming) {
        VOX_LOGI("render") << "  vertexStreams sub (ms): resize=" << subResizeMs
                           << " convert=" << subConvertMs
                           << " rest=" << subTimer.elapsedMs();
    }
    phaseTimer.restart();

    // Carve this scene's geometry out of the shared arenas. Growing first and
    // retrying once covers the case where capacity was fine but fragmentation
    // left no single range big enough.
    const std::uint64_t chunkVertexCount = vertices.size();
    const std::uint64_t chunkIndexCount = indices.size();
    if (!ensureImportedArenaCapacity(chunkVertexCount, chunkIndexCount)) {
        return false;
    }
    std::uint64_t firstVertex = m_importedVertexArena.allocate(chunkVertexCount, 1);
    if (firstVertex == GpuArenaAllocator::kInvalidOffset && chunkVertexCount > 0) {
        // Capacity was already sufficient (the call above saw to that), so this
        // is fragmentation: no single free range fits. Grow PAST capacity so the
        // arena gains a contiguous tail block, and grow the GPU buffer with it --
        // ensureImportedArenaCapacity does both. Growing only the suballocator
        // here, which is what this used to do, produced an offset the buffer did
        // not actually reach.
        if (!ensureImportedArenaCapacity(chunkVertexCount, 0, /*pastCapacity=*/true)) {
            return false;
        }
        firstVertex = m_importedVertexArena.allocate(chunkVertexCount, 1);
    }
    std::uint64_t firstIndexSlot = m_importedIndexArena.allocate(chunkIndexCount, 1);
    if (firstIndexSlot == GpuArenaAllocator::kInvalidOffset && chunkIndexCount > 0) {
        if (!ensureImportedArenaCapacity(0, chunkIndexCount, /*pastCapacity=*/true)) {
            return false;
        }
        firstIndexSlot = m_importedIndexArena.allocate(chunkIndexCount, 1);
    }
    if ((chunkVertexCount > 0 && firstVertex == GpuArenaAllocator::kInvalidOffset) ||
        (chunkIndexCount > 0 && firstIndexSlot == GpuArenaAllocator::kInvalidOffset)) {
        VOX_LOGE("render") << "imported geometry arena could not satisfy scene of "
                           << chunkVertexCount << " vertices / " << chunkIndexCount << " indices";
        return false;
    }
    if (chunkVertexCount == 0) {
        firstVertex = 0;
    }
    if (chunkIndexCount == 0) {
        firstIndexSlot = 0;
    }

    const bool geometryUploaded =
        uploadIntoBufferRange(
            m_importedVertexBufferHandle,
            static_cast<VkDeviceSize>(firstVertex * sizeof(ImportedMeshVertex)),
            vertices.data(),
            static_cast<VkDeviceSize>(vertices.size() * sizeof(ImportedMeshVertex)),
            "imported scene vertex") &&
        uploadIntoBufferRange(
            m_importedShadowVertexBufferHandle,
            static_cast<VkDeviceSize>(firstVertex * sizeof(ImportedShadowVertex)),
            shadowVertices.data(),
            static_cast<VkDeviceSize>(shadowVertices.size() * sizeof(ImportedShadowVertex)),
            "imported scene shadow vertex") &&
        uploadIntoBufferRange(
            m_importedIndexBufferHandle,
            static_cast<VkDeviceSize>(firstIndexSlot * sizeof(std::uint32_t)),
            indices.data(),
            static_cast<VkDeviceSize>(indices.size() * sizeof(std::uint32_t)),
            "imported scene index");
    if (!geometryUploaded) {
        m_importedVertexArena.free(firstVertex, chunkVertexCount);
        m_importedIndexArena.free(firstIndexSlot, chunkIndexCount);
        return false;
    }

    // m_importedIndexCount / terrain / static counts are derived from the live
    // chunk set by rebuildImportedDrawTables() below, not set here.
    const std::uint32_t chunkTerrainDrawCount =
        std::min<std::uint32_t>(mergedTerrainDrawCount, static_cast<std::uint32_t>(draws.size()));

    geometryUploadMs = phaseTimer.elapsedMs();
    phaseTimer.restart();

    ImportedSceneChunk chunk{};
    chunk.alive = true;
    chunk.firstVertex = firstVertex;
    chunk.vertexCount = chunkVertexCount;
    chunk.firstIndex = firstIndexSlot;
    chunk.indexCount = chunkIndexCount;
    chunk.terrainDrawCount = chunkTerrainDrawCount;
    // Ownership of every acquired slot passes to the chunk here; from this point
    // removeImportedSceneChunk is what releases them.
    chunk.textureSlots = importedTextureSlots;
    textureSlotGuard.commit();
    chunk.lights = std::move(chunkLights);
    chunk.waterPatches = uploadScene.waterPatches;
    chunk.draws.reserve(draws.size());
    for (ImportedMeshDraw& draw : draws) {
        draw.vertexBufferHandle = m_importedVertexBufferHandle;
        draw.indexBufferHandle = m_importedIndexBufferHandle;
        // Rebase onto this chunk's arena ranges: indices are stored chunk-local,
        // so the arena offset is added here rather than baked into the data.
        draw.firstIndex += static_cast<std::uint32_t>(firstIndexSlot);
        draw.vertexOffset = static_cast<std::int32_t>(firstVertex);
        chunk.draws.push_back(draw);
    }
    chunk.pageRanges = std::move(pageDrawRanges);
    // Reuse a slot an evicted chunk left behind, or append if there is none.
    // Indices stay valid for the lifetime of a live chunk either way -- a caller
    // holding one has already been told the chunk was removed.
    if (!m_freeImportedSceneChunks.empty()) {
        m_lastImportedChunkIndex = m_freeImportedSceneChunks.back();
        m_freeImportedSceneChunks.pop_back();
        m_importedSceneChunks[m_lastImportedChunkIndex] = std::move(chunk);
    } else {
        m_lastImportedChunkIndex = m_importedSceneChunks.size();
        m_importedSceneChunks.push_back(std::move(chunk));
    }
    rebuildImportedDrawTables();
    drawBuildMs = phaseTimer.elapsedMs();
    phaseTimer.restart();

    const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(m_importedVertexBufferHandle);
    const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(m_importedIndexBufferHandle);
    if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE) {
        VOX_LOGE("render") << "imported scene upload produced null Vulkan buffers"
                           << " (vertexHandle=" << m_importedVertexBufferHandle
                           << ", indexHandle=" << m_importedIndexBufferHandle << ")";
        // The arenas are shared state now, so tear them down as a unit rather
        // than destroying handles this call happens to be holding.
        // As above: the teardown owns the slot table from here.
        textureSlotGuard.commit();
        clearImportedSceneMeshes();
        return false;
    }
    if (vertexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), "mesh.importedScene.vertex");
    }
    if (indexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), "mesh.importedScene.index");
    }
    // A streamed chunk's water is not uploaded here. Its patches were stored on
    // the chunk above and the whole buffer pair is regenerated from every live
    // chunk by rebuildImportedWaterBuffers(), because one exact-fit pair cannot
    // be appended to -- which is why this used to warn and drop them, and why
    // every coast in every streamed worldspace was a hole.
    //
    // The whole-scene path below is unchanged: a cooked scene or a strategy map
    // arrives complete, so it can size the buffers exactly once.
    if (appendChunk) {
        // Nothing to do; see above.
    } else if (!waterVertices.empty() && !waterIndices.empty()) {
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
    if (!appendChunk) {
        m_importedGiTriangles.clear();
    }
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
    {
        ImportedDrawBounds sceneBounds = terrainBounds;
        if (staticBounds.valid) {
            if (sceneBounds.valid) {
                for (int axis = 0; axis < 3; ++axis) {
                    sceneBounds.min[axis] = std::min(sceneBounds.min[axis], staticBounds.min[axis]);
                    sceneBounds.max[axis] = std::max(sceneBounds.max[axis], staticBounds.max[axis]);
                }
            } else {
                sceneBounds = staticBounds;
            }
        }
        if (sceneBounds.valid) {
            float chunkCenter[3] = {};
            float radiusSq = 0.0f;
            for (int axis = 0; axis < 3; ++axis) {
                chunkCenter[axis] = (sceneBounds.min[axis] + sceneBounds.max[axis]) * 0.5f;
                const float halfExtent = (sceneBounds.max[axis] - sceneBounds.min[axis]) * 0.5f;
                radiusSq += halfExtent * halfExtent;
            }
            const float chunkRadius = std::sqrt(radiusSq);

            if (!appendChunk || !m_importedSceneBoundsValid) {
                std::copy(std::begin(chunkCenter), std::end(chunkCenter), m_importedSceneBoundsCenter);
                m_importedSceneBoundsRadius = chunkRadius;
            } else {
                // Merge the new chunk's bounding sphere into the resident one.
                // Streaming grows the world a cell at a time, so replacing the
                // bounds would shrink them to whichever chunk loaded last and
                // mis-fit everything derived from them.
                float delta[3] = {
                    chunkCenter[0] - m_importedSceneBoundsCenter[0],
                    chunkCenter[1] - m_importedSceneBoundsCenter[1],
                    chunkCenter[2] - m_importedSceneBoundsCenter[2]};
                const float distance = std::sqrt(
                    delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
                if (distance + chunkRadius <= m_importedSceneBoundsRadius) {
                    // New sphere already enclosed; nothing to do.
                } else if (distance + m_importedSceneBoundsRadius <= chunkRadius) {
                    std::copy(std::begin(chunkCenter), std::end(chunkCenter), m_importedSceneBoundsCenter);
                    m_importedSceneBoundsRadius = chunkRadius;
                } else {
                    const float mergedRadius =
                        (distance + m_importedSceneBoundsRadius + chunkRadius) * 0.5f;
                    if (distance > 1e-5f) {
                        const float t =
                            (mergedRadius - m_importedSceneBoundsRadius) / distance;
                        for (int axis = 0; axis < 3; ++axis) {
                            m_importedSceneBoundsCenter[axis] += delta[axis] * t;
                        }
                    }
                    m_importedSceneBoundsRadius = mergedRadius;
                }
            }
            m_importedSceneBoundsValid = true;
        } else if (!appendChunk) {
            m_importedSceneBoundsValid = false;
        }
    }
    if (logChunkTiming) {
        VOX_LOGI("render") << "chunk upload phases (ms): textures=" << textureMs
                           << " vertexStreams=" << vertexMs
                           << " geometryUpload=" << geometryUploadMs
                           << " drawTables=" << drawBuildMs
                           << " tail(bounds/gi/rt)=" << phaseTimer.elapsedMs()
                           << " total=" << chunkPhaseTimer.elapsedMs();
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
    // Gate on rayTracingRuntimeReady() (extensions loaded + function pointers resolved),
    // not just rayTracingCoreReady() (hardware/driver merely supports the extensions).
    // rebuildRayTracingScene() -- the only consumer of RtImportedSceneRecord::geometry --
    // already requires rayTracingRuntimeReady() before it will touch these buffers, so on
    // hardware that advertises RT support but fails to load the runtime function pointers
    // (e.g. runtimeEnabled=no in the ray tracing runtime probe log), building this GPU-resident
    // geometry on every uploadImportedScene() call was pure wasted work -- a measured 40-70ms
    // stutter per call with no reader on the other end.
    if (rayTracingRuntimeReady()) {
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
    m_debugMacroCellStatsDirty = true;
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
    m_debugMacroCellStatsDirty = true;
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
    m_debugMacroCellStatsDirty = true;
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


bool RendererBackend::uploadChunkMeshes(
    const odai::world::ChunkGrid& chunkGrid,
    std::vector<odai::world::ChunkMeshResult> results
) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    if (results.empty()) {
        return true;
    }
    if (m_chunkMeshRebuildRequested) {
        // A queued full rebuild re-meshes everything synchronously anyway.
        return true;
    }
    // Map each result to its resident chunk index; drop results for chunks
    // that streamed out between meshing and upload.
    std::vector<std::size_t> chunkIndices;
    chunkIndices.reserve(results.size());
    for (const odai::world::ChunkMeshResult& result : results) {
        for (std::size_t chunkIndex = 0; chunkIndex < chunkGrid.chunks().size(); ++chunkIndex) {
            const odai::world::Chunk& chunk = chunkGrid.chunks()[chunkIndex];
            if (chunk.chunkX() == result.key.x &&
                chunk.chunkY() == result.key.y &&
                chunk.chunkZ() == result.key.z) {
                chunkIndices.push_back(chunkIndex);
                break;
            }
        }
    }
    if (chunkIndices.empty()) {
        return true;
    }
    for (odai::world::ChunkMeshResult& result : results) {
        // Replace any older un-consumed result for the same chunk.
        const auto existingIt = std::find_if(
            m_externalChunkMeshResults.begin(),
            m_externalChunkMeshResults.end(),
            [&](const odai::world::ChunkMeshResult& existing) { return existing.key == result.key; });
        if (existingIt != m_externalChunkMeshResults.end()) {
            *existingIt = std::move(result);
        } else {
            m_externalChunkMeshResults.push_back(std::move(result));
        }
    }
    // Queue through the existing per-frame remesh path so staging happens
    // inside renderFrame with the frame arena in a valid state; the remesh
    // loop consumes the stored meshes instead of running the mesher inline.
    return updateChunkMesh(chunkGrid, std::span<const std::size_t>(chunkIndices));
}


bool RendererBackend::consumeExternalChunkMeshResult(
    std::size_t chunkArrayIndex,
    odai::world::ChunkMeshingStats& outStats
) {
    if (chunkArrayIndex >= m_chunkResidentKeys.size()) {
        return false;
    }
    const ChunkResidentKey& residentKey = m_chunkResidentKeys[chunkArrayIndex];
    const auto resultIt = std::find_if(
        m_externalChunkMeshResults.begin(),
        m_externalChunkMeshResults.end(),
        [&](const odai::world::ChunkMeshResult& result) {
            return result.key.x == residentKey.chunkX &&
                   result.key.y == residentKey.chunkY &&
                   result.key.z == residentKey.chunkZ;
        });
    if (resultIt == m_externalChunkMeshResults.end()) {
        return false;
    }
    if (chunkArrayIndex < m_chunkLodMeshCache.size()) {
        m_chunkLodMeshCache[chunkArrayIndex] = std::move(resultIt->meshes);
    }
    outStats = resultIt->stats;
    m_externalChunkMeshResults.erase(resultIt);
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
        if (anyTransferSlotInFlight()) {
            return false;
        }

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
    std::vector<RtChunkSceneRecord> previousRtChunkSceneRecords = std::move(m_rtChunkSceneRecords);

    m_chunkResidentKeys.assign(chunks.size(), ChunkResidentKey{});
    m_chunkLodMeshCache.assign(chunks.size(), odai::world::ChunkLodMeshes{});
    m_rtChunkSceneRecords.assign(chunks.size(), RtChunkSceneRecord{});

    std::vector<std::uint8_t> remeshMask(chunks.size(), 0u);
    bool reusedAnyChunkCache = false;
    for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
        const ChunkResidentKey key = chunkResidentKeyForChunk(chunks[chunkArrayIndex]);
        m_chunkResidentKeys[chunkArrayIndex] = key;

        const auto previousIt = std::find(previousResidentKeys.begin(), previousResidentKeys.end(), key);
        if (previousIt == previousResidentKeys.end()) {
            remeshMask[chunkArrayIndex] = 1u;
            continue;
        }

        const std::size_t previousIndex = static_cast<std::size_t>(std::distance(previousResidentKeys.begin(), previousIt));
        if (previousIndex < previousChunkLodMeshCache.size()) {
            m_chunkLodMeshCache[chunkArrayIndex] = previousChunkLodMeshCache[previousIndex];
            reusedAnyChunkCache = true;
        } else {
            remeshMask[chunkArrayIndex] = 1u;
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

    // Centre chunk of the resident set — the ray-tracing active/retained radii below
    // measure against it.
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

    // (removed) grass billboard scatter. The billboards read as invisible from the
    // camera while their shadow casters smeared dark streaks across the ground, and
    // the feature is not wanted for the voxel game. world::buildGrassInstances and
    // its tests are untouched -- the generator still works if it is ever wanted back.
    // The PREVIOUS frame's resident centre went with it -- the scatter's retained
    // radius was its only reader; the ray-tracing path only needs the current one.

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
            odai::world::ChunkMeshingStats meshingStats{};
            if (!consumeExternalChunkMeshResult(chunkArrayIndex, meshingStats)) {
                m_chunkLodMeshCache[chunkArrayIndex] =
                    odai::world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions, &meshingStats);
            }
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            remeshedNaiveVertexCount += meshingStats.exposedFaceCount * 4u;
            remeshedNaiveIndexCount += meshingStats.exposedFaceCount * 6u;
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
            odai::world::ChunkMeshingStats meshingStats{};
            if (!consumeExternalChunkMeshResult(chunkArrayIndex, meshingStats)) {
                m_chunkLodMeshCache[chunkArrayIndex] =
                    odai::world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions, &meshingStats);
            }
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            remeshedNaiveVertexCount += meshingStats.exposedFaceCount * 4u;
            remeshedNaiveIndexCount += meshingStats.exposedFaceCount * 6u;
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
    TransferCommandSlot* transferSlot = nullptr;
    if (hasChunkCopies) {
        transferSlot = acquireTransferCommandSlot(nullptr);
        if (transferSlot == nullptr) {
            VOX_LOGE("render") << "no transfer command slot available for chunk upload";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
        const VkResult resetResult = vkResetCommandBuffer(transferSlot->commandBuffer, 0);
        if (resetResult != VK_SUCCESS) {
            logVkFailure("vkResetCommandBuffer(transfer)", resetResult);
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(transferSlot->commandBuffer, &beginInfo) != VK_SUCCESS) {
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
                transferSlot->commandBuffer,
                m_bufferAllocator.getBuffer(chunkVertexUploadSliceOpt->buffer),
                m_bufferAllocator.getBuffer(newChunkVertexBufferHandle),
                1,
                &vertexCopy
            );

            VkBufferCopy indexCopy{};
            indexCopy.srcOffset = chunkIndexUploadSliceOpt->offset;
            indexCopy.size = indexBufferSize;
            vkCmdCopyBuffer(
                transferSlot->commandBuffer,
                m_bufferAllocator.getBuffer(chunkIndexUploadSliceOpt->buffer),
                m_bufferAllocator.getBuffer(newChunkIndexBufferHandle),
                1,
                &indexCopy
            );
        }

        if (vkEndCommandBuffer(transferSlot->commandBuffer) != VK_SUCCESS) {
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
        transferCommandBufferInfo.commandBuffer = transferSlot->commandBuffer;
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
            transferSlot->inFlightTimelineValue = 0;
        } else {
            m_currentChunkReadyTimelineValue = transferSignalValue;
            m_pendingTransferTimelineValue = std::max(m_pendingTransferTimelineValue, transferSignalValue);
            transferSlot->inFlightTimelineValue = transferSignalValue;
            transferSlot->stagingFrameIndex = m_currentFrame;
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
