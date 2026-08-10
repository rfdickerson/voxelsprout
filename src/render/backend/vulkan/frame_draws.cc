#include "render/backend/vulkan/renderer_backend.h"

#include <cstdint>

namespace odai::render {

void RendererBackend::drawIndirectChunkRanges(
    VkCommandBuffer commandBuffer,
    std::uint32_t& passDrawCounter,
    const FrameChunkDrawData& frameChunkDrawData
) {
    if (!frameChunkDrawData.canDrawChunksIndirect) {
        return;
    }

    const uint32_t drawCount = frameChunkDrawData.chunkIndirectDrawCount;
    const std::optional<FrameArenaSlice>& indirectSlice = frameChunkDrawData.chunkIndirectSliceOpt;
    if (!indirectSlice.has_value()) {
        return;
    }

    m_debugDrawCallsTotal += drawCount;
    passDrawCounter += drawCount;

    if (m_supportsMultiDrawIndirect) {
        vkCmdDrawIndexedIndirect(
            commandBuffer,
            frameChunkDrawData.chunkIndirectBuffer,
            indirectSlice->offset,
            drawCount,
            sizeof(VkDrawIndexedIndirectCommand)
        );
        return;
    }

    const VkDeviceSize stride = static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand));
    VkDeviceSize drawOffset = indirectSlice->offset;
    for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex) {
        vkCmdDrawIndexedIndirect(commandBuffer, frameChunkDrawData.chunkIndirectBuffer, drawOffset, 1, static_cast<uint32_t>(stride));
        drawOffset += stride;
    }
}

void RendererBackend::drawIndirectShadowChunkRanges(
    VkCommandBuffer commandBuffer,
    std::uint32_t& passDrawCounter,
    std::uint32_t cascadeIndex,
    const FrameChunkDrawData& frameChunkDrawData
) {
    if (cascadeIndex >= kShadowCascadeCount || !frameChunkDrawData.canDrawShadowChunksIndirectByCascade[cascadeIndex]) {
        return;
    }

    const uint32_t drawCount = frameChunkDrawData.shadowCascadeIndirectDrawCounts[cascadeIndex];
    const VkBuffer indirectBuffer = frameChunkDrawData.shadowCascadeIndirectBuffers[cascadeIndex];
    const std::optional<FrameArenaSlice>& indirectSlice = frameChunkDrawData.shadowCascadeIndirectSliceOpts[cascadeIndex];
    if (!indirectSlice.has_value()) {
        return;
    }

    m_debugDrawCallsTotal += drawCount;
    passDrawCounter += drawCount;

    if (m_supportsMultiDrawIndirect) {
        vkCmdDrawIndexedIndirect(
            commandBuffer,
            indirectBuffer,
            indirectSlice->offset,
            drawCount,
            sizeof(VkDrawIndexedIndirectCommand)
        );
        return;
    }

    const VkDeviceSize stride = static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand));
    VkDeviceSize drawOffset = indirectSlice->offset;
    for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex) {
        vkCmdDrawIndexedIndirect(commandBuffer, indirectBuffer, drawOffset, 1, static_cast<uint32_t>(stride));
        drawOffset += stride;
    }
}


// Groups imported draws into indirect commands, one group per distinct
// alpha-test threshold. See ImportedIndirectBatch in renderer_backend.h for why
// the threshold is the grouping key and nothing else needs to be.
//
// Order within a group is not preserved relative to other groups, which is safe
// for OPAQUE draws only -- they resolve by depth test. Blended draws must keep
// their back-to-front sequence and are deliberately not routed through here.
bool RendererBackend::buildImportedIndirectBatches(
    std::span<const ImportedMeshDraw> draws,
    const std::function<bool(std::size_t)>& include,
    VkBuffer& outBuffer,
    VkDeviceSize& outBaseOffset) {
    m_importedIndirectScratch.clear();
    m_importedIndirectBatches.clear();
    if (draws.empty()) {
        return false;
    }

    // Bucket by threshold in one pass. 256 possible values, and retail data
    // uses two or three, so a flat array beats a map and avoids allocating.
    constexpr std::uint32_t kNoCommand = 0xffffffffu;
    std::uint32_t mergedCommands = 0;
    // 512 buckets: alpha threshold in the low 8 bits, two-sidedness in bit 8.
    // Both are state that cannot change inside one indirect call -- the
    // threshold is a push constant, two-sidedness is the pipeline -- so both
    // have to be part of the grouping key.
    constexpr std::size_t kBucketCount = 512;
    const auto bucketOf = [](const ImportedMeshDraw& draw) -> std::size_t {
        return static_cast<std::size_t>(draw.alphaThreshold) | (draw.twoSided ? 256u : 0u);
    };
    std::array<std::uint32_t, kBucketCount> countPerThreshold{};
    std::size_t totalIncluded = 0;
    for (std::size_t i = 0; i < draws.size(); ++i) {
        if (draws[i].blended || !include(i)) {
            continue;
        }
        ++countPerThreshold[bucketOf(draws[i])];
        ++totalIncluded;
    }
    if (totalIncluded == 0) {
        return false;
    }

    // Lay the groups out contiguously, then fill them by walking the draws once
    // more and writing each into its group's cursor.
    std::array<std::uint32_t, kBucketCount> groupStart{};
    std::uint32_t cursor = 0;
    for (std::size_t threshold = 0; threshold < countPerThreshold.size(); ++threshold) {
        if (countPerThreshold[threshold] == 0u) {
            continue;
        }
        groupStart[threshold] = cursor;
        ImportedIndirectBatch batch{};
        batch.drawCount = countPerThreshold[threshold];
        batch.alphaThreshold = static_cast<std::uint8_t>(threshold & 0xffu);
        batch.twoSided = (threshold & 256u) != 0u;
        batch.bufferOffset = static_cast<VkDeviceSize>(cursor) * sizeof(VkDrawIndexedIndirectCommand);
        m_importedIndirectBatches.push_back(batch);
        cursor += countPerThreshold[threshold];
    }

    m_importedIndirectScratch.resize(totalIncluded);
    std::array<std::uint32_t, kBucketCount> writeCursor = groupStart;
    // Per group, the command written most recently, so an incoming draw can be
    // MERGED into it instead of becoming its own command.
    std::array<std::uint32_t, kBucketCount> lastWritten{};
    lastWritten.fill(kNoCommand);

    for (std::size_t i = 0; i < draws.size(); ++i) {
        if (draws[i].blended || !include(i)) {
            continue;
        }
        const ImportedMeshDraw& draw = draws[i];
        const std::size_t threshold = bucketOf(draw);

        // Merge adjacent index ranges into one command.
        //
        // This is the change that actually removes draw calls, and it is only
        // legal because of how imported geometry is laid out and shaded:
        // every draw indexes ONE shared vertex/index buffer, and the texture is
        // a per-vertex bindless slot rather than per-draw state. So two draws
        // that sit back to back in the index buffer and agree on the alpha-test
        // threshold differ in nothing the GPU can observe -- they are one draw
        // that was split only because the importer emitted a part per material.
        //
        // Deliberately a single ordered pass rather than sort-then-merge:
        // rebuildImportedDrawTables appends each chunk's draws in index order,
        // so runs are already adjacent, and sorting thousands of commands three
        // times a frame would spend more CPU than the merge saves.
        if (lastWritten[threshold] != kNoCommand) {
            VkDrawIndexedIndirectCommand& previous = m_importedIndirectScratch[lastWritten[threshold]];
            if (previous.vertexOffset == static_cast<int32_t>(draw.vertexOffset) &&
                previous.firstIndex + previous.indexCount == draw.firstIndex) {
                previous.indexCount += draw.indexCount;
                ++mergedCommands;
                continue;
            }
        }

        const std::uint32_t slot = writeCursor[threshold]++;
        VkDrawIndexedIndirectCommand& command = m_importedIndirectScratch[slot];
        command.indexCount = draw.indexCount;
        command.instanceCount = 1;
        command.firstIndex = draw.firstIndex;
        command.vertexOffset = static_cast<int32_t>(draw.vertexOffset);
        command.firstInstance = 0;
        lastWritten[threshold] = slot;
    }

    // Merging leaves each group shorter than the count it was sized for, so the
    // groups are now non-contiguous. Compact them, fixing up each batch's
    // offset and count to what was actually written.
    std::uint32_t compactCursor = 0;
    for (ImportedIndirectBatch& batch : m_importedIndirectBatches) {
        const std::size_t bucket =
            static_cast<std::size_t>(batch.alphaThreshold) | (batch.twoSided ? 256u : 0u);
        const std::uint32_t written = writeCursor[bucket] - groupStart[bucket];
        for (std::uint32_t k = 0; k < written; ++k) {
            m_importedIndirectScratch[compactCursor + k] =
                m_importedIndirectScratch[groupStart[bucket] + k];
        }
        batch.bufferOffset =
            static_cast<VkDeviceSize>(compactCursor) * sizeof(VkDrawIndexedIndirectCommand);
        batch.drawCount = written;
        compactCursor += written;
    }
    m_importedIndirectScratch.resize(compactCursor);
    std::erase_if(m_importedIndirectBatches, [](const ImportedIndirectBatch& batch) {
        return batch.drawCount == 0u;
    });
    if (m_importedIndirectScratch.empty()) {
        return false;
    }
    m_debugImportedDrawsMerged = mergedCommands;

    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(m_importedIndirectScratch.size() * sizeof(VkDrawIndexedIndirectCommand));
    const std::optional<FrameArenaSlice> slice = m_frameArena.allocateUpload(
        bytes,
        static_cast<VkDeviceSize>(alignof(VkDrawIndexedIndirectCommand)),
        FrameArenaUploadKind::Unknown);
    if (!slice.has_value() || slice->mapped == nullptr) {
        // Out of arena: the caller falls back to direct draws. Returning false
        // rather than drawing nothing keeps a full arena a frame-rate problem
        // instead of a disappearing world.
        m_importedIndirectBatches.clear();
        return false;
    }
    std::memcpy(slice->mapped, m_importedIndirectScratch.data(), static_cast<size_t>(bytes));
    // slice->buffer is a BufferHandle (an index), not a VkBuffer; the
    // allocator owns the mapping.
    outBuffer = m_bufferAllocator.getBuffer(slice->buffer);
    outBaseOffset = slice->offset;
    return true;
}

}  // namespace odai::render
