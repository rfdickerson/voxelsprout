#include <cstdlib>
#include <iostream>
#include <vector>

#include "render/backend/vulkan/frame_graph_runtime.h"
#include "render/frame_graph.h"

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void testExecutionOrder() {
    odai::render::FrameGraph graph;
    const auto shadow = graph.addPass({"shadow", odai::render::FrameGraphQueue::Graphics});
    const auto prepass = graph.addPass({"prepass", odai::render::FrameGraphQueue::Graphics});
    const auto main = graph.addPass({"main", odai::render::FrameGraphQueue::Graphics});
    const auto post = graph.addPass({"post", odai::render::FrameGraphQueue::Graphics});
    graph.addDependency(shadow, prepass);
    graph.addDependency(prepass, main);
    graph.addDependency(main, post);

    std::vector<odai::render::FrameGraph::PassId> order;
    expect(graph.buildExecutionOrder(&order), "acyclic graph must produce an order");
    expect(order == std::vector{shadow, prepass, main, post},
           "execution order must respect pass dependencies");
}

void testCycleDetection() {
    odai::render::FrameGraph graph;
    const auto a = graph.addPass({"a", odai::render::FrameGraphQueue::Graphics});
    const auto b = graph.addPass({"b", odai::render::FrameGraphQueue::Graphics});
    graph.addDependency(a, b);
    graph.addDependency(b, a);

    std::vector<odai::render::FrameGraph::PassId> order;
    expect(!graph.buildExecutionOrder(&order), "cycle must be rejected");
}

void testOrderValidator() {
    odai::render::CoreFrameGraphPlan plan{};
    plan.shadow = 0;
    plan.prepass = 1;
    plan.main = 2;
    plan.post = 3;
    plan.passOrderById = {0, 1, 2, 3};

    odai::render::CoreFrameGraphOrderValidator validator(plan);
    validator.markPassEntered(plan.shadow, "shadow");
    validator.markPassEntered(plan.prepass, "prepass");
    validator.markPassEntered(plan.main, "main");
    validator.markPassEntered(plan.post, "post");
}

}  // namespace

int main() {
    testExecutionOrder();
    testCycleDetection();
    testOrderValidator();
    std::cout << "frame graph tests passed\n";
    return 0;
}
