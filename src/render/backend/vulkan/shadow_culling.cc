#include "render/backend/vulkan/renderer_backend.h"

#include "render/backend/vulkan/shadow_culling_utils.h"

namespace voxelsprout::render {

std::vector<std::uint8_t> RendererBackend::buildShadowCandidateMask(
    std::span<const voxelsprout::world::Chunk> chunks,
    std::span<const std::size_t> visibleChunkIndices
) const {
    return ::voxelsprout::render::buildShadowCandidateMask(
        chunks,
        visibleChunkIndices,
        m_shadowDebugSettings.enableOccluderCulling
    );
}

} // namespace voxelsprout::render
