#pragma once

#include "bethesda/runtime_ids.h"
#include "bethesda/runtime_transform.h"

#include <array>
#include <cstdint>
#include <vector>

namespace odai::bethesda {

enum RuntimeRenderChange : std::uint32_t {
    RuntimeRenderTransform = 1u << 0u,
    RuntimeRenderVisibility = 1u << 1u,
    RuntimeRenderSkinning = 1u << 2u,
    RuntimeRenderMaterial = 1u << 3u,
    RuntimeRenderLight = 1u << 4u,
    RuntimeRenderParticle = 1u << 5u,
    RuntimeRenderDecal = 1u << 6u,
};

// Presentation-only mutation. Static scene geometry remains in ImportedScene;
// gameplay publishes compact changes keyed by stable ObjectId.
struct RuntimeRenderDelta {
    ObjectId object;
    std::uint32_t changes = 0u;
    RuntimeTransform transform;
    bool visible = true;
    std::uint32_t skinningSlot = 0xffffffffu;
    std::uint32_t materialIndex = 0u;
    std::array<float, 4> lightColorIntensity{};
};

using RuntimeRenderDeltaBatch = std::vector<RuntimeRenderDelta>;

}  // namespace odai::bethesda
