#include "render/backend/vulkan/renderer_backend.h"
#include "render/backend/vulkan/frame_math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace odai::render {

void RendererBackend::prepareImportedVisibility(
    const odai::math::Vector3& eye, const CameraPose& camera, float projectionYScale) {
    // These depend on the main camera, not on the shadow or mirrored clip
    // matrix. Compute them once before assembling the per-view draw lists.
    m_importedVegetationLodVisible.resize(m_importedMeshDraws.size());
    const float halfHeight = 0.5f * static_cast<float>(m_renderExtent.height);
    for (std::size_t i = 0; i < m_importedMeshDraws.size(); ++i) {
        const ImportedMeshDraw& draw = m_importedMeshDraws[i];
        if (draw.vegetationLod == 0u) {
            m_importedVegetationLodVisible[i] = 1u;
            continue;
        }
        const float dx = draw.center[0] - eye.x;
        const float dy = draw.center[1] - eye.y;
        const float dz = draw.center[2] - eye.z;
        const float distance = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.0f);
        const float projectedPixels = camera.orthographic
            ? draw.vegetationHeight *
                (static_cast<float>(m_renderExtent.height) /
                 std::max(camera.orthoHalfHeight * 2.0f, 1.0f))
            : draw.vegetationHeight * std::abs(projectionYScale) * halfHeight / distance;
        const std::uint8_t wanted = projectedPixels >= 80.0f * 1.15f
            ? 1u : (projectedPixels <= 20.0f * 0.85f ? 3u : 2u);
        m_importedVegetationLodVisible[i] = draw.vegetationLod == wanted ? 1u : 0u;
    }

    m_importedPageNearTerrain.resize(m_importedPageDrawRanges.size());
    for (std::size_t i = 0; i < m_importedPageDrawRanges.size(); ++i) {
        const ImportedScenePageDrawRange& page = m_importedPageDrawRanges[i];
        const float eyePosition[3] = {eye.x, eye.y, eye.z};
        float distanceSq = 0.0f;
        for (int axis = 0; axis < 3; ++axis) {
            const float clamped = std::clamp(
                eyePosition[axis], page.boundsMin[axis], page.boundsMax[axis]);
            const float delta = eyePosition[axis] - clamped;
            distanceSq += delta * delta;
        }
        const float range = page.distantLodTessellation ? 48000.0f : 10500.0f;
        m_importedPageNearTerrain[i] = distanceSq < range * range ? 1u : 0u;
    }
}

RendererBackend::VisibleImportedDrawCounts RendererBackend::buildVisibleImportedDraws(
    const odai::math::Matrix4& clipMatrix, float clipMargin,
    std::vector<ImportedMeshDraw>& outDraws, float minimumY) {
    outDraws.clear();
    if (outDraws.capacity() < m_importedMeshDraws.size()) {
        outDraws.reserve(m_importedMeshDraws.size());
    }
    m_visibleImportedPageOrder.clear();
    for (std::size_t pageIndex = 0; pageIndex < m_importedPageDrawRanges.size(); ++pageIndex) {
        const ImportedScenePageDrawRange& page = m_importedPageDrawRanges[pageIndex];
        if (page.drawCount == 0u || page.boundsMax[1] < minimumY - 32.0f ||
            !importedBoundsIntersectClip(page.boundsMin, page.boundsMax, clipMatrix, clipMargin)) {
            continue;
        }
        m_visibleImportedPageOrder.push_back(static_cast<std::uint32_t>(pageIndex));
    }

    const auto appendDrawRange = [&](std::uint32_t firstDraw, std::uint32_t drawCount) {
        if (drawCount == 0u || firstDraw >= m_importedMeshDraws.size()) {
            return std::uint32_t{0};
        }
        const std::uint32_t available = static_cast<std::uint32_t>(
            m_importedMeshDraws.size() - firstDraw);
        const std::uint32_t end = firstDraw + std::min(drawCount, available);
        std::uint32_t appended = 0u;
        for (std::uint32_t i = firstDraw; i < end; ++i) {
            if (m_importedVegetationLodVisible[i] == 0u) continue;
            outDraws.push_back(m_importedMeshDraws[i]);
            ++appended;
        }
        return appended;
    };

    // The terrain prefix is near then far; the passes route those ranges
    // through tessellated and flat pipelines respectively.
    VisibleImportedDrawCounts counts{};
    for (const std::uint32_t pageIndex : m_visibleImportedPageOrder) {
        if (m_importedPageNearTerrain[pageIndex] == 0u) continue;
        const ImportedScenePageDrawRange& page = m_importedPageDrawRanges[pageIndex];
        counts.nearTerrain += appendDrawRange(
            page.firstDraw, std::min(page.terrainDrawCount, page.drawCount));
    }
    counts.terrain = counts.nearTerrain;
    for (const std::uint32_t pageIndex : m_visibleImportedPageOrder) {
        if (m_importedPageNearTerrain[pageIndex] != 0u) continue;
        const ImportedScenePageDrawRange& page = m_importedPageDrawRanges[pageIndex];
        counts.terrain += appendDrawRange(
            page.firstDraw, std::min(page.terrainDrawCount, page.drawCount));
    }
    for (const std::uint32_t pageIndex : m_visibleImportedPageOrder) {
        const ImportedScenePageDrawRange& page = m_importedPageDrawRanges[pageIndex];
        const std::uint32_t terrain = std::min(page.terrainDrawCount, page.drawCount);
        appendDrawRange(page.firstDraw + terrain, page.drawCount - terrain);
    }
    return counts;
}

bool RendererBackend::sampleImportedRigidAnimationTransform(
    std::uint32_t animationIndex,
    float outTransform[3][4]
) const {
    if (animationIndex >= m_importedRigidAnimations.size()) {
        return false;
    }
    float delta[16] = {};
    if (!odai::importer::sampleImportedSceneRigidAnimation(
            m_importedRigidAnimations[animationIndex],
            m_importedRigidAnimationTimeSeconds,
            delta)) {
        return false;
    }
    for (int row = 0; row < 3; ++row) {
        std::memcpy(outTransform[row], delta + (row * 4), sizeof(float) * 4u);
    }
    return true;
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
    struct FadeBucket { std::size_t base; std::array<float, 4> state; };
    std::vector<FadeBucket> fades;
    const auto bucketOf = [&](const ImportedMeshDraw& draw) -> std::size_t {
        const std::size_t base = static_cast<std::size_t>(draw.alphaThreshold) | (draw.twoSided ? 256u : 0u);
        if (draw.lodTransition[1] == 0.0f) return base;
        for (std::size_t i = 0; i < fades.size(); ++i)
            if (fades[i].base == base && fades[i].state == draw.lodTransition) return kBucketCount + i;
        fades.push_back({base, draw.lodTransition});
        return kBucketCount + fades.size() - 1;
    };
    std::vector<std::uint32_t> countPerThreshold(kBucketCount);
    std::size_t totalIncluded = 0;
    for (std::size_t i = 0; i < draws.size(); ++i) {
        if (draws[i].blended || draws[i].rigidAnimationIndex != 0xffffffffu || !include(i)) {
            continue;
        }
        const auto bucket = bucketOf(draws[i]);
        if (bucket >= countPerThreshold.size()) countPerThreshold.resize(bucket + 1);
        ++countPerThreshold[bucket];
        ++totalIncluded;
    }
    if (totalIncluded == 0) {
        return false;
    }

    // Lay the groups out contiguously, then fill them by walking the draws once
    // more and writing each into its group's cursor.
    std::vector<std::uint32_t> groupStart(countPerThreshold.size());
    std::uint32_t cursor = 0;
    for (std::size_t threshold = 0; threshold < countPerThreshold.size(); ++threshold) {
        if (countPerThreshold[threshold] == 0u) {
            continue;
        }
        groupStart[threshold] = cursor;
        ImportedIndirectBatch batch{};
        batch.drawCount = countPerThreshold[threshold];
        batch.bucket = threshold;
        const auto base = threshold < kBucketCount ? threshold : fades[threshold - kBucketCount].base;
        if (threshold >= kBucketCount) batch.lodTransition = fades[threshold - kBucketCount].state;
        batch.alphaThreshold = static_cast<std::uint8_t>(base & 0xffu);
        batch.twoSided = (base & 256u) != 0u;
        batch.bufferOffset = static_cast<VkDeviceSize>(cursor) * sizeof(VkDrawIndexedIndirectCommand);
        m_importedIndirectBatches.push_back(batch);
        cursor += countPerThreshold[threshold];
    }

    m_importedIndirectScratch.resize(totalIncluded);
    auto writeCursor = groupStart;
    // Per group, the command written most recently, so an incoming draw can be
    // MERGED into it instead of becoming its own command.
    std::vector<std::uint32_t> lastWritten(countPerThreshold.size(), kNoCommand);

    for (std::size_t i = 0; i < draws.size(); ++i) {
        if (draws[i].blended || draws[i].rigidAnimationIndex != 0xffffffffu || !include(i)) {
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
        const std::size_t bucket = batch.bucket;
        const std::uint32_t written = writeCursor[bucket] - groupStart[bucket];
        for (std::uint32_t k = 0; k < written; ++k) {
            m_importedIndirectScratch[compactCursor + k] =
                m_importedIndirectScratch[groupStart[bucket] + k];
            batch.triangleCount += m_importedIndirectScratch[compactCursor + k].indexCount / 3u;
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
