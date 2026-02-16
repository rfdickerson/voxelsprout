#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace voxelsprout::render {

enum class FrameGraphQueue : std::uint8_t {
    Graphics,
    Compute,
    Transfer,
};

struct FrameGraphPassDesc {
    const char* name = "unnamed";
    FrameGraphQueue queue = FrameGraphQueue::Graphics;
};

struct FrameGraphPass {
    std::string name;
    FrameGraphQueue queue = FrameGraphQueue::Graphics;
};

enum class FrameGraphResourceAccess : std::uint8_t {
    Read,
    Write,
    ReadWrite,
};

struct FrameGraphResourceDesc {
    const char* name = "unnamed";
};

struct FrameGraphResource {
    std::string name;
};

class FrameGraph {
public:
    using PassId = std::uint32_t;
    using ResourceId = std::uint32_t;

    void reset();
    PassId addPass(const FrameGraphPassDesc& desc);
    ResourceId addResource(const FrameGraphResourceDesc& desc);
    void addDependency(PassId producer, PassId consumer);
    void addResourceUse(PassId pass, ResourceId resource, FrameGraphResourceAccess access);

    [[nodiscard]] std::span<const FrameGraphPass> passes() const;
    [[nodiscard]] std::span<const FrameGraphResource> resources() const;
    [[nodiscard]] std::span<const std::pair<PassId, PassId>> dependencies() const;
    [[nodiscard]] std::span<const std::tuple<PassId, ResourceId, FrameGraphResourceAccess>> resourceUses() const;
    [[nodiscard]] bool buildExecutionOrder(std::vector<PassId>* outOrder) const;

private:
    std::vector<FrameGraphPass> m_passes;
    std::vector<FrameGraphResource> m_resources;
    std::vector<std::pair<PassId, PassId>> m_dependencies;
    std::vector<std::tuple<PassId, ResourceId, FrameGraphResourceAccess>> m_resourceUses;
};

} // namespace voxelsprout::render
