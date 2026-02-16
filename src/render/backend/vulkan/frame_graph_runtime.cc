#include "render/backend/vulkan/frame_graph_runtime.h"

#include "core/log.h"

namespace voxelsprout::render {

CoreFrameGraphOrderValidator::CoreFrameGraphOrderValidator(const CoreFrameGraphPlan& plan)
    : m_passOrderById(plan.passOrderById) {}

void CoreFrameGraphOrderValidator::markPassEntered(FrameGraph::PassId passId, const char* passName) {
    if (passId >= m_passOrderById.size()) {
        return;
    }
    const std::uint32_t passOrderIndex = m_passOrderById[passId];
    if (m_lastPassOrderIndex.has_value() && passOrderIndex < *m_lastPassOrderIndex) {
        VOX_LOGE("render")
            << "core frame pass executed out of graph order: " << passName
            << ", orderIndex=" << passOrderIndex
            << ", previousOrderIndex=" << *m_lastPassOrderIndex;
    }
    m_lastPassOrderIndex = passOrderIndex;
}

}  // namespace voxelsprout::render
