#pragma once

#include <cstdint>
#include <span>
#include <string>
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

class FrameGraph {
public:
    using PassId = std::uint32_t;

    void reset();
    PassId addPass(const FrameGraphPassDesc& desc);
    void addDependency(PassId producer, PassId consumer);

    [[nodiscard]] std::span<const FrameGraphPass> passes() const;
    [[nodiscard]] std::span<const std::pair<PassId, PassId>> dependencies() const;

private:
    std::vector<FrameGraphPass> m_passes;
    std::vector<std::pair<PassId, PassId>> m_dependencies;
};

} // namespace voxelsprout::render
