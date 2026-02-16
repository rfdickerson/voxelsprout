#include "render/frame_graph.h"

namespace voxelsprout::render {

void FrameGraph::reset() {
    m_passes.clear();
    m_dependencies.clear();
}

FrameGraph::PassId FrameGraph::addPass(const FrameGraphPassDesc& desc) {
    FrameGraphPass pass{};
    pass.name = (desc.name != nullptr && desc.name[0] != '\0') ? desc.name : "unnamed";
    pass.queue = desc.queue;
    const PassId id = static_cast<PassId>(m_passes.size());
    m_passes.push_back(std::move(pass));
    return id;
}

void FrameGraph::addDependency(PassId producer, PassId consumer) {
    if (producer == consumer) {
        return;
    }
    m_dependencies.emplace_back(producer, consumer);
}

std::span<const FrameGraphPass> FrameGraph::passes() const {
    return std::span<const FrameGraphPass>(m_passes.data(), m_passes.size());
}

std::span<const std::pair<FrameGraph::PassId, FrameGraph::PassId>> FrameGraph::dependencies() const {
    return std::span<const std::pair<PassId, PassId>>(m_dependencies.data(), m_dependencies.size());
}

} // namespace voxelsprout::render
