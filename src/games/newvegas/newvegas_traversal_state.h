#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace odai::games::newvegas {

struct TraversalDiscovery {
    std::uint32_t sourceReferenceFormId = 0u;
    std::string worldspaceEditorId;
    std::string name;
};

struct TraversalState {
    static constexpr std::uint32_t kVersion = 1u;

    bool interior = false;
    std::string worldspaceEditorId;
    std::string interiorEditorId;
    float position[3] = {};
    float yawDegrees = 0.0f;
    float pitchDegrees = 0.0f;
    float timeOfDayHours = 9.5f;
    std::string weatherEditorId;
    std::string loadOrderFingerprint;
    std::vector<TraversalDiscovery> discoveries;
};

[[nodiscard]] std::filesystem::path defaultTraversalStatePath();
bool loadTraversalState(
    const std::filesystem::path& path, TraversalState& outState, std::string& outError);
bool saveTraversalStateAtomic(
    const std::filesystem::path& path, const TraversalState& state, std::string& outError);

}  // namespace odai::games::newvegas
