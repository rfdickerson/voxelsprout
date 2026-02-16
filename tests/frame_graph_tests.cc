#include <gtest/gtest.h>

#include <vector>

#include "render/frame_graph.h"

namespace {

TEST(FrameGraphTest, BuildExecutionOrderRespectsDependencies) {
    voxelsprout::render::FrameGraph frameGraph;
    const auto shadow = frameGraph.addPass({"shadow", voxelsprout::render::FrameGraphQueue::Graphics});
    const auto prepass = frameGraph.addPass({"prepass", voxelsprout::render::FrameGraphQueue::Graphics});
    const auto main = frameGraph.addPass({"main", voxelsprout::render::FrameGraphQueue::Graphics});
    const auto post = frameGraph.addPass({"post", voxelsprout::render::FrameGraphQueue::Graphics});

    frameGraph.addDependency(shadow, prepass);
    frameGraph.addDependency(prepass, main);
    frameGraph.addDependency(main, post);

    std::vector<voxelsprout::render::FrameGraph::PassId> order;
    ASSERT_TRUE(frameGraph.buildExecutionOrder(&order));
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], shadow);
    EXPECT_EQ(order[1], prepass);
    EXPECT_EQ(order[2], main);
    EXPECT_EQ(order[3], post);
}

TEST(FrameGraphTest, BuildExecutionOrderDetectsCycle) {
    voxelsprout::render::FrameGraph frameGraph;
    const auto a = frameGraph.addPass({"a", voxelsprout::render::FrameGraphQueue::Graphics});
    const auto b = frameGraph.addPass({"b", voxelsprout::render::FrameGraphQueue::Graphics});

    frameGraph.addDependency(a, b);
    frameGraph.addDependency(b, a);

    std::vector<voxelsprout::render::FrameGraph::PassId> order;
    EXPECT_FALSE(frameGraph.buildExecutionOrder(&order));
}

}  // namespace
