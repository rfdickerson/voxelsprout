#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "render/frame_graph.h"

namespace voxelsprout::render {

struct CoreFrameGraphPlan {
    FrameGraph::PassId shadow = 0;
    FrameGraph::PassId prepass = 0;
    FrameGraph::PassId main = 0;
    FrameGraph::PassId post = 0;
    std::vector<FrameGraph::PassId> executionOrder;
    std::vector<std::uint32_t> passOrderById;
};

std::optional<CoreFrameGraphPlan> buildCoreFrameGraphPlan(FrameGraph* frameGraph);

}  // namespace voxelsprout::render
