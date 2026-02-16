#include "render/backend/vulkan/frame_graph_core.h"

namespace voxelsprout::render {

std::optional<CoreFrameGraphPlan> buildCoreFrameGraphPlan(FrameGraph* frameGraph) {
    if (frameGraph == nullptr) {
        return std::nullopt;
    }

    frameGraph->reset();

    CoreFrameGraphPlan plan{};
    plan.shadow = frameGraph->addPass({"shadow", FrameGraphQueue::Graphics});
    plan.prepass = frameGraph->addPass({"prepass", FrameGraphQueue::Graphics});
    plan.main = frameGraph->addPass({"main", FrameGraphQueue::Graphics});
    plan.post = frameGraph->addPass({"post", FrameGraphQueue::Graphics});

    const FrameGraph::ResourceId shadowAtlas = frameGraph->addResource({"shadow_atlas_depth"});
    const FrameGraph::ResourceId sceneDepth = frameGraph->addResource({"scene_depth"});
    const FrameGraph::ResourceId hdrColor = frameGraph->addResource({"scene_hdr_color"});
    const FrameGraph::ResourceId swapchainColor = frameGraph->addResource({"swapchain_color"});

    frameGraph->addDependency(plan.shadow, plan.prepass);
    frameGraph->addDependency(plan.prepass, plan.main);
    frameGraph->addDependency(plan.main, plan.post);

    frameGraph->addResourceUse(plan.shadow, shadowAtlas, FrameGraphResourceAccess::Write);
    frameGraph->addResourceUse(plan.prepass, shadowAtlas, FrameGraphResourceAccess::Read);
    frameGraph->addResourceUse(plan.prepass, sceneDepth, FrameGraphResourceAccess::Write);
    frameGraph->addResourceUse(plan.main, shadowAtlas, FrameGraphResourceAccess::Read);
    frameGraph->addResourceUse(plan.main, sceneDepth, FrameGraphResourceAccess::Read);
    frameGraph->addResourceUse(plan.main, hdrColor, FrameGraphResourceAccess::Write);
    frameGraph->addResourceUse(plan.post, hdrColor, FrameGraphResourceAccess::Read);
    frameGraph->addResourceUse(plan.post, swapchainColor, FrameGraphResourceAccess::Write);

    if (!frameGraph->buildExecutionOrder(&plan.executionOrder)) {
        return std::nullopt;
    }

    plan.passOrderById.assign(frameGraph->passes().size(), 0u);
    for (std::uint32_t executionIndex = 0; executionIndex < plan.executionOrder.size(); ++executionIndex) {
        const FrameGraph::PassId passId = plan.executionOrder[executionIndex];
        if (passId < plan.passOrderById.size()) {
            plan.passOrderById[passId] = executionIndex;
        }
    }

    return plan;
}

}  // namespace voxelsprout::render
