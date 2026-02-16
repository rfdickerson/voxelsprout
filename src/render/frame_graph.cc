#include "render/frame_graph.h"

#include <queue>

namespace voxelsprout::render {

void FrameGraph::reset() {
    m_passes.clear();
    m_resources.clear();
    m_dependencies.clear();
    m_resourceUses.clear();
}

FrameGraph::PassId FrameGraph::addPass(const FrameGraphPassDesc& desc) {
    FrameGraphPass pass{};
    pass.name = (desc.name != nullptr && desc.name[0] != '\0') ? desc.name : "unnamed";
    pass.queue = desc.queue;
    const PassId id = static_cast<PassId>(m_passes.size());
    m_passes.push_back(std::move(pass));
    return id;
}

FrameGraph::ResourceId FrameGraph::addResource(const FrameGraphResourceDesc& desc) {
    FrameGraphResource resource{};
    resource.name = (desc.name != nullptr && desc.name[0] != '\0') ? desc.name : "unnamed";
    const ResourceId id = static_cast<ResourceId>(m_resources.size());
    m_resources.push_back(std::move(resource));
    return id;
}

void FrameGraph::addDependency(PassId producer, PassId consumer) {
    if (producer == consumer) {
        return;
    }
    m_dependencies.emplace_back(producer, consumer);
}

void FrameGraph::addResourceUse(PassId pass, ResourceId resource, FrameGraphResourceAccess access) {
    if (pass >= m_passes.size() || resource >= m_resources.size()) {
        return;
    }
    m_resourceUses.emplace_back(pass, resource, access);
}

std::span<const FrameGraphPass> FrameGraph::passes() const {
    return std::span<const FrameGraphPass>(m_passes.data(), m_passes.size());
}

std::span<const FrameGraphResource> FrameGraph::resources() const {
    return std::span<const FrameGraphResource>(m_resources.data(), m_resources.size());
}

std::span<const std::pair<FrameGraph::PassId, FrameGraph::PassId>> FrameGraph::dependencies() const {
    return std::span<const std::pair<PassId, PassId>>(m_dependencies.data(), m_dependencies.size());
}

std::span<const std::tuple<FrameGraph::PassId, FrameGraph::ResourceId, FrameGraphResourceAccess>>
FrameGraph::resourceUses() const {
    return std::span<const std::tuple<PassId, ResourceId, FrameGraphResourceAccess>>(
        m_resourceUses.data(),
        m_resourceUses.size()
    );
}

bool FrameGraph::buildExecutionOrder(std::vector<PassId>* outOrder) const {
    if (outOrder == nullptr) {
        return false;
    }
    outOrder->clear();
    if (m_passes.empty()) {
        return true;
    }

    std::vector<std::vector<PassId>> adjacency(m_passes.size());
    std::vector<std::uint32_t> indegree(m_passes.size(), 0u);
    for (const auto& dep : m_dependencies) {
        const PassId producer = dep.first;
        const PassId consumer = dep.second;
        if (producer >= m_passes.size() || consumer >= m_passes.size()) {
            continue;
        }
        adjacency[producer].push_back(consumer);
        ++indegree[consumer];
    }

    std::queue<PassId> ready;
    for (PassId passId = 0; passId < m_passes.size(); ++passId) {
        if (indegree[passId] == 0u) {
            ready.push(passId);
        }
    }

    outOrder->reserve(m_passes.size());
    while (!ready.empty()) {
        const PassId passId = ready.front();
        ready.pop();
        outOrder->push_back(passId);
        for (const PassId consumer : adjacency[passId]) {
            if (indegree[consumer] > 0u) {
                --indegree[consumer];
            }
            if (indegree[consumer] == 0u) {
                ready.push(consumer);
            }
        }
    }
    return outOrder->size() == m_passes.size();
}

} // namespace voxelsprout::render
