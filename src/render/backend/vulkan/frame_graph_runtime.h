#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "render/backend/vulkan/frame_graph_core.h"

namespace voxelsprout::render {

class CoreFrameGraphOrderValidator {
public:
    explicit CoreFrameGraphOrderValidator(const CoreFrameGraphPlan& plan);

    void markPassEntered(FrameGraph::PassId passId, const char* passName);

private:
    std::vector<std::uint32_t> m_passOrderById;
    std::optional<std::uint32_t> m_lastPassOrderIndex = std::nullopt;
};

}  // namespace voxelsprout::render
