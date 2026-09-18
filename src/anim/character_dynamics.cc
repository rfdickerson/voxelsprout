#include "anim/character_dynamics.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <set>

namespace odai::anim {
namespace {
using json = nlohmann::json;

bool finite(const odai::math::Vector3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

odai::math::Vector3 translation(const odai::math::Matrix4& matrix) {
    return {matrix(0, 3), matrix(1, 3), matrix(2, 3)};
}

void setTranslation(odai::math::Matrix4& matrix, const odai::math::Vector3& value) {
    matrix(0, 3) = value.x;
    matrix(1, 3) = value.y;
    matrix(2, 3) = value.z;
}
}

bool compileBodyMorphProgram(
    const std::string& jsonText, BodyMorphProgram& out, std::string& error) {
    error.clear();
    BodyMorphProgram result;
    try {
        const json root = json::parse(jsonText);
        if (root.value("version", 0u) != BodyMorphProgram::version) {
            error = "unsupported native body morph version";
            return false;
        }
        result.vertexCount = root.at("vertex_count").get<std::uint32_t>();
        result.topologyFingerprint = root.at("topology_fingerprint").get<std::string>();
        if (result.vertexCount == 0 || result.topologyFingerprint.empty()) {
            error = "body morph topology is empty";
            return false;
        }
        std::set<std::string> ids;
        for (const json& encoded : root.at("targets")) {
            BodyMorphTarget target;
            target.id = encoded.at("id").get<std::string>();
            if (target.id.empty() || !ids.insert(target.id).second) {
                error = "body morph target id is empty or duplicated";
                return false;
            }
            std::set<std::uint32_t> vertices;
            for (const json& encodedDelta : encoded.at("deltas")) {
                MorphVertexDelta delta;
                delta.vertex = encodedDelta.at("vertex").get<std::uint32_t>();
                const auto values = encodedDelta.at("position").get<std::vector<float>>();
                if (values.size() != 3) {
                    error = "body morph delta must have three components";
                    return false;
                }
                delta.position = {values[0], values[1], values[2]};
                if (delta.vertex >= result.vertexCount || !vertices.insert(delta.vertex).second ||
                    !finite(delta.position)) {
                    error = "body morph delta has invalid topology or value";
                    return false;
                }
                target.deltas.push_back(delta);
            }
            result.targets.push_back(std::move(target));
        }
        if (root.contains("outfit_mappings")) {
            for (auto it = root.at("outfit_mappings").begin();
                 it != root.at("outfit_mappings").end(); ++it) {
                auto targets = it.value().get<std::vector<std::string>>();
                if (it.key().empty() || targets.empty()) {
                    error = "body morph outfit mapping is empty";
                    return false;
                }
                for (const std::string& id : targets) {
                    if (!ids.contains(id)) {
                        error = "body morph outfit mapping names unknown target: " + id;
                        return false;
                    }
                }
                result.outfitMappings.emplace(it.key(), std::move(targets));
            }
        }
    } catch (const std::exception& exception) {
        error = std::string("malformed native body morph pack: ") + exception.what();
        return false;
    }
    out = std::move(result);
    return true;
}

bool applyBodyMorphs(const BodyMorphProgram& program,
    const BodyMorphSnapshot& state, std::span<const odai::math::Vector3> basePositions,
    std::string_view outfit, std::vector<odai::math::Vector3>& outPositions,
    std::string& error) {
    error.clear();
    if (basePositions.size() != program.vertexCount) {
        error = "body morph topology vertex count mismatch";
        return false;
    }
    if (!state.topologyFingerprint.empty() &&
        state.topologyFingerprint != program.topologyFingerprint) {
        error = "body morph topology fingerprint mismatch";
        return false;
    }
    std::set<std::string> allowed;
    if (!outfit.empty()) {
        const auto mapping = program.outfitMappings.find(std::string(outfit));
        if (mapping == program.outfitMappings.end()) {
            error = "outfit has no explicit body morph mapping: " + std::string(outfit);
            return false;
        }
        allowed.insert(mapping->second.begin(), mapping->second.end());
    }
    outPositions.assign(basePositions.begin(), basePositions.end());
    for (const BodyMorphTarget& target : program.targets) {
        const auto slider = state.sliders.find(target.id);
        if (slider == state.sliders.end() || slider->second == 0.0f) continue;
        if (!std::isfinite(slider->second) || slider->second < 0.0f || slider->second > 1.0f) {
            error = "body morph slider is outside [0, 1]: " + target.id;
            return false;
        }
        if (!outfit.empty() && !allowed.contains(target.id)) continue;
        for (const MorphVertexDelta& delta : target.deltas)
            outPositions[delta.vertex] = outPositions[delta.vertex] + delta.position * slider->second;
    }
    return true;
}

bool HairChainSimulator::configure(const HairChainConfig& config, std::string& error) {
    error.clear();
    if (config.bones.size() < 2 || !std::isfinite(config.damping) ||
        config.damping < 0.0f || config.damping >= 1.0f ||
        !std::isfinite(config.gravity) || config.gravity < 0.0f ||
        !std::isfinite(config.maximumStretch) || config.maximumStretch < 1.0f ||
        !std::isfinite(config.teleportDistance) || config.teleportDistance <= 0.0f) {
        error = "invalid hair chain settings";
        return false;
    }
    std::set<int> bones;
    for (int bone : config.bones) {
        if (bone < 0 || !bones.insert(bone).second) {
            error = "hair chain contains an invalid or duplicate bone";
            return false;
        }
    }
    for (const HairCollisionSphere& sphere : config.collisions) {
        if (sphere.bone < 0 || !std::isfinite(sphere.radius) || sphere.radius <= 0.0f) {
            error = "invalid hair collision sphere";
            return false;
        }
    }
    m_config = config;
    m_lengths.clear();
    m_state = {};
    m_initialized = false;
    return true;
}

void HairChainSimulator::reset(std::span<const odai::math::Matrix4> authoredBoneWorld) {
    m_initialized = false;
    if (m_config.bones.empty()) return;
    for (int bone : m_config.bones)
        if (static_cast<std::size_t>(bone) >= authoredBoneWorld.size()) return;
    m_state.positions.clear();
    for (int bone : m_config.bones)
        m_state.positions.push_back(translation(authoredBoneWorld[static_cast<std::size_t>(bone)]));
    m_state.previousPositions = m_state.positions;
    m_lengths.clear();
    for (std::size_t i = 1; i < m_state.positions.size(); ++i)
        m_lengths.push_back(std::max(0.001f,
            odai::math::length(m_state.positions[i] - m_state.positions[i - 1])));
    m_lastRoot = m_state.positions.front();
    m_initialized = true;
}

bool HairChainSimulator::update(std::span<const odai::math::Matrix4> authoredBoneWorld,
    float fixedDeltaSeconds, bool enabled,
    std::vector<odai::math::Matrix4>& inOutBoneWorld) {
    if (m_config.bones.empty() || !std::isfinite(fixedDeltaSeconds) ||
        fixedDeltaSeconds <= 0.0f || fixedDeltaSeconds > 0.25f) return false;
    for (int bone : m_config.bones)
        if (static_cast<std::size_t>(bone) >= authoredBoneWorld.size()) return false;
    inOutBoneWorld.assign(authoredBoneWorld.begin(), authoredBoneWorld.end());
    const odai::math::Vector3 authoredRoot =
        translation(authoredBoneWorld[static_cast<std::size_t>(m_config.bones.front())]);
    if (!enabled) {
        reset(authoredBoneWorld);
        return true;
    }
    if (!m_initialized || odai::math::length(authoredRoot - m_lastRoot) > m_config.teleportDistance) {
        reset(authoredBoneWorld);
    }
    const odai::math::Vector3 rootDelta = authoredRoot - m_lastRoot;
    for (std::size_t i = 1; i < m_state.positions.size(); ++i) {
        m_state.positions[i] = m_state.positions[i] + rootDelta;
        m_state.previousPositions[i] = m_state.previousPositions[i] + rootDelta;
    }
    m_state.positions.front() = authoredRoot;
    m_state.previousPositions.front() = authoredRoot;
    const odai::math::Vector3 acceleration{0.0f, -m_config.gravity, 0.0f};
    const float dt2 = fixedDeltaSeconds * fixedDeltaSeconds;
    for (std::size_t i = 1; i < m_state.positions.size(); ++i) {
        const odai::math::Vector3 current = m_state.positions[i];
        const odai::math::Vector3 velocity =
            (current - m_state.previousPositions[i]) * (1.0f - m_config.damping);
        m_state.positions[i] = current + velocity + acceleration * dt2;
        m_state.previousPositions[i] = current;
    }
    for (int iteration = 0; iteration < 4; ++iteration) {
        m_state.positions.front() = authoredRoot;
        for (std::size_t i = 1; i < m_state.positions.size(); ++i) {
            odai::math::Vector3 offset = m_state.positions[i] - m_state.positions[i - 1];
            const float distance = odai::math::length(offset);
            if (distance <= 1.0e-6f) continue;
            const float limit = m_lengths[i - 1] * m_config.maximumStretch;
            const float target = std::min(distance, limit);
            m_state.positions[i] = m_state.positions[i - 1] + offset * (target / distance);
        }
        for (std::size_t i = 1; i < m_state.positions.size(); ++i) {
            for (const HairCollisionSphere& sphere : m_config.collisions) {
                if (static_cast<std::size_t>(sphere.bone) >= authoredBoneWorld.size()) continue;
                const odai::math::Vector3 centre =
                    translation(authoredBoneWorld[static_cast<std::size_t>(sphere.bone)]);
                odai::math::Vector3 offset = m_state.positions[i] - centre;
                const float distance = odai::math::length(offset);
                if (distance < sphere.radius) {
                    if (distance <= 1.0e-6f) offset = {0.0f, 1.0f, 0.0f};
                    else offset = offset * (1.0f / distance);
                    m_state.positions[i] = centre + offset * sphere.radius;
                }
            }
        }
    }
    for (std::size_t i = 1; i < m_config.bones.size(); ++i)
        setTranslation(inOutBoneWorld[static_cast<std::size_t>(m_config.bones[i])],
            m_state.positions[i]);
    m_lastRoot = authoredRoot;
    return true;
}

HairChainSnapshot HairChainSimulator::snapshot() const { return m_state; }

} // namespace odai::anim
