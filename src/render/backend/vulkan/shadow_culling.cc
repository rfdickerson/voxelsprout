#include "render/backend/vulkan/renderer_backend.h"

#include "render/backend/vulkan/shadow_culling_utils.h"

namespace odai::render {

std::vector<std::uint8_t> RendererBackend::buildShadowCandidateMask(
    std::span<const odai::world::Chunk> chunks,
    std::span<const std::size_t> visibleChunkIndices
) const {
    return ::odai::render::buildShadowCandidateMask(
        chunks,
        visibleChunkIndices,
        m_shadowDebugSettings.enableOccluderCulling
    );
}

} // namespace odai::render
