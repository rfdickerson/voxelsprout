#include <gtest/gtest.h>

#include <vector>

#include "render/backend/vulkan/frame_graph_runtime.h"
#include "render/frame_graph.h"

namespace {

TEST(FrameGraphTest, BuildExecutionOrderRespectsDependencies) {
    odai::render::FrameGraph frameGraph;
    const auto shadow = frameGraph.addPass({"shadow", odai::render::FrameGraphQueue::Graphics});
    const auto prepass = frameGraph.addPass({"prepass", odai::render::FrameGraphQueue::Graphics});
    const auto main = frameGraph.addPass({"main", odai::render::FrameGraphQueue::Graphics});
    const auto post = frameGraph.addPass({"post", odai::render::FrameGraphQueue::Graphics});

    frameGraph.addDependency(shadow, prepass);
    frameGraph.addDependency(prepass, main);
    frameGraph.addDependency(main, post);

    std::vector<odai::render::FrameGraph::PassId> order;
    ASSERT_TRUE(frameGraph.buildExecutionOrder(&order));
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], shadow);
    EXPECT_EQ(order[1], prepass);
    EXPECT_EQ(order[2], main);
    EXPECT_EQ(order[3], post);
}

TEST(FrameGraphTest, BuildExecutionOrderDetectsCycle) {
    odai::render::FrameGraph frameGraph;
    const auto a = frameGraph.addPass({"a", odai::render::FrameGraphQueue::Graphics});
    const auto b = frameGraph.addPass({"b", odai::render::FrameGraphQueue::Graphics});

    frameGraph.addDependency(a, b);
    frameGraph.addDependency(b, a);

    std::vector<odai::render::FrameGraph::PassId> order;
    EXPECT_FALSE(frameGraph.buildExecutionOrder(&order));
}

TEST(FrameGraphTest, CoreFrameGraphOrderValidatorTracksMonotonicPassOrder) {
    odai::render::CoreFrameGraphPlan plan{};
    plan.shadow = 0u;
    plan.prepass = 1u;
    plan.main = 2u;
    plan.post = 3u;
    plan.passOrderById = {0u, 1u, 2u, 3u};

    odai::render::CoreFrameGraphOrderValidator validator(plan);
    validator.markPassEntered(plan.shadow, "shadow");
    validator.markPassEntered(plan.prepass, "prepass");
    validator.markPassEntered(plan.main, "main");
    validator.markPassEntered(plan.post, "post");

    SUCCEED();
}

}  // namespace
