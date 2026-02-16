#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/grid3.h"

// Simulation NetworkGraph subsystem
// Responsible for: storing deterministic transport graphs used by pipes, belts, and rails.
// Should NOT do: tick-based simulation, path search heuristics, or rendering.
namespace voxelsprout::sim {

using NodeId = std::uint32_t;
using EdgeId = std::uint32_t;

inline constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::max();
inline constexpr EdgeId kInvalidEdgeId = std::numeric_limits<EdgeId>::max();

enum class NetworkKind : std::uint8_t {
    Pipe = 0,
    Belt = 1,
    Rail = 2
};

struct Socket {
    voxelsprout::core::Cell3i cell{};
    voxelsprout::core::Dir6 face = voxelsprout::core::Dir6::PosY;
    std::uint8_t lane = 0;

    constexpr bool operator==(const Socket&) const = default;
};

struct EdgeSpan {
    voxelsprout::core::Cell3i start{};
    voxelsprout::core::Dir6 dir = voxelsprout::core::Dir6::PosY;
    std::uint16_t lengthVoxels = 1;

    constexpr bool operator==(const EdgeSpan&) const = default;
};

inline constexpr bool isValidEdgeSpan(const EdgeSpan& span) {
    return span.lengthVoxels > 0;
}

inline constexpr voxelsprout::core::Cell3i spanEndCell(const EdgeSpan& span) {
    if (!isValidEdgeSpan(span)) {
        return span.start;
    }
    const std::int32_t steps = static_cast<std::int32_t>(span.lengthVoxels) - 1;
    return span.start + (voxelsprout::core::dirToOffset(span.dir) * steps);
}

struct NetworkNode {
    Socket socket{};
};

struct NetworkEdge {
    NodeId a = kInvalidNodeId;
    NodeId b = kInvalidNodeId;
    EdgeSpan span{};
    NetworkKind kind = NetworkKind::Pipe;
    std::uint16_t typeId = 0;
};

class NetworkGraph {
public:
    NodeId addNode(const Socket& socket);
    EdgeId addEdge(NodeId a, NodeId b, const EdgeSpan& span, NetworkKind kind, std::uint16_t typeId = 0);
    void clear();

    std::size_t nodeCount() const;
    std::size_t edgeCount() const;
    const std::vector<NetworkNode>& nodes() const;
    const std::vector<NetworkEdge>& edges() const;
    const std::vector<EdgeId>& edgesForNode(NodeId nodeId) const;
    bool edgeConnectsNode(EdgeId edgeId, NodeId nodeId) const;

private:
    std::vector<NetworkNode> m_nodes;
    std::vector<NetworkEdge> m_edges;
    std::vector<std::vector<EdgeId>> m_nodeEdges;
};

inline NodeId NetworkGraph::addNode(const Socket& socket) {
    const NodeId id = static_cast<NodeId>(m_nodes.size());
    m_nodes.push_back(NetworkNode{socket});
    m_nodeEdges.emplace_back();
    return id;
}

inline EdgeId NetworkGraph::addEdge(NodeId a, NodeId b, const EdgeSpan& span, NetworkKind kind, std::uint16_t typeId) {
    if (a >= m_nodes.size() || b >= m_nodes.size() || !isValidEdgeSpan(span)) {
        return kInvalidEdgeId;
    }

    const EdgeId id = static_cast<EdgeId>(m_edges.size());
    m_edges.push_back(NetworkEdge{a, b, span, kind, typeId});
    m_nodeEdges[a].push_back(id);
    if (b != a) {
        m_nodeEdges[b].push_back(id);
    }
    return id;
}

inline void NetworkGraph::clear() {
    m_nodes.clear();
    m_edges.clear();
    m_nodeEdges.clear();
}

inline std::size_t NetworkGraph::nodeCount() const {
    return m_nodes.size();
}

inline std::size_t NetworkGraph::edgeCount() const {
    return m_edges.size();
}

inline const std::vector<NetworkNode>& NetworkGraph::nodes() const {
    return m_nodes;
}

inline const std::vector<NetworkEdge>& NetworkGraph::edges() const {
    return m_edges;
}

inline const std::vector<EdgeId>& NetworkGraph::edgesForNode(NodeId nodeId) const {
    static const std::vector<EdgeId> kEmpty;
    if (nodeId >= m_nodeEdges.size()) {
        return kEmpty;
    }
    return m_nodeEdges[nodeId];
}

inline bool NetworkGraph::edgeConnectsNode(EdgeId edgeId, NodeId nodeId) const {
    if (edgeId >= m_edges.size()) {
        return false;
    }
    const NetworkEdge& edge = m_edges[edgeId];
    return edge.a == nodeId || edge.b == nodeId;
}

struct PipeEdgeData {
    std::uint16_t diameterTier = 0;
    std::uint16_t fluidTypeId = 0;
    std::uint16_t capacityUnitsPerTick = 0;
    std::uint16_t pressureClass = 0;
};

struct BeltEdgeData {
    std::uint16_t speedTier = 0;
    std::uint8_t laneCount = 1;
    bool reversed = false;
    std::uint16_t slotSpacingQ8 = 256;
};

enum class RailSegmentClass : std::uint8_t {
    Straight = 0,
    Curve = 1,
    Slope = 2,
    Switch = 3
};

struct RailEdgeData {
    RailSegmentClass segmentClass = RailSegmentClass::Straight;
    std::uint16_t switchGroupId = 0;
    std::uint16_t blockId = 0;
};

struct TrackParam {
    EdgeId edgeId = kInvalidEdgeId;
    std::uint16_t distanceAlongQ8 = 0;
    bool forward = true;
};

struct TrainCar {
    std::uint32_t carId = 0;
    std::uint16_t carTypeId = 0;
    std::uint16_t lengthQ8 = 512;
    std::uint16_t maxFrontCouplingQ8 = 128;
    std::uint16_t maxRearCouplingQ8 = 128;
    std::uint32_t cargoComponentId = 0;
};

struct PipeEdgeRuntimeState {
    std::uint32_t inUnits = 0;
    std::uint32_t outUnits = 0;
};

struct BeltSlot {
    std::uint32_t itemId = 0;
    bool occupied = false;
};

struct RailBlockState {
    std::uint32_t reservedByTrainId = 0;
    bool reserved = false;
};

} // namespace voxelsprout::sim
