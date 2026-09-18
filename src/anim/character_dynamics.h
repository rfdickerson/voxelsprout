#pragma once

#include "math/math.h"

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace odai::anim {

struct MorphVertexDelta {
    std::uint32_t vertex = 0;
    odai::math::Vector3 position{};
};

struct BodyMorphTarget {
    std::string id;
    std::vector<MorphVertexDelta> deltas;
};

// Immutable, topology-bound native morph data. Outfit entries explicitly list
// the targets authored for that topology; an outfit with no entry is rejected
// instead of silently deforming mismatched clothing.
struct BodyMorphProgram {
    static constexpr std::uint32_t version = 1;
    std::uint32_t vertexCount = 0;
    std::string topologyFingerprint;
    std::vector<BodyMorphTarget> targets;
    std::map<std::string, std::vector<std::string>> outfitMappings;
};

struct BodyMorphSnapshot {
    std::string topologyFingerprint;
    std::map<std::string, float> sliders;
    friend bool operator==(const BodyMorphSnapshot&, const BodyMorphSnapshot&) = default;
};

bool compileBodyMorphProgram(
    const std::string& jsonText, BodyMorphProgram& out, std::string& error);
bool applyBodyMorphs(const BodyMorphProgram& program,
    const BodyMorphSnapshot& state, std::span<const odai::math::Vector3> basePositions,
    std::string_view outfit, std::vector<odai::math::Vector3>& outPositions,
    std::string& error);

struct HairCollisionSphere {
    int bone = -1;
    float radius = 0.0f;
};

struct HairChainConfig {
    std::vector<int> bones;
    float damping = 0.18f;
    float gravity = 9.81f;
    float maximumStretch = 1.05f;
    float teleportDistance = 64.0f;
    std::vector<HairCollisionSphere> collisions;
};

struct HairChainSnapshot {
    std::vector<odai::math::Vector3> positions;
    std::vector<odai::math::Vector3> previousPositions;
};

// Deterministic fixed-step secondary motion. The authored pose remains the
// fallback and is also the reset source after teleports or streaming changes.
class HairChainSimulator {
public:
    bool configure(const HairChainConfig& config, std::string& error);
    void reset(std::span<const odai::math::Matrix4> authoredBoneWorld);
    bool update(std::span<const odai::math::Matrix4> authoredBoneWorld,
        float fixedDeltaSeconds, bool enabled,
        std::vector<odai::math::Matrix4>& inOutBoneWorld);
    [[nodiscard]] HairChainSnapshot snapshot() const;
    [[nodiscard]] bool configured() const { return !m_config.bones.empty(); }

private:
    HairChainConfig m_config;
    std::vector<float> m_lengths;
    HairChainSnapshot m_state;
    odai::math::Vector3 m_lastRoot{};
    bool m_initialized = false;
};

} // namespace odai::anim
