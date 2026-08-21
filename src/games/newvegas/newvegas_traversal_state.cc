#include "games/newvegas/newvegas_traversal_state.h"

#include <cstdlib>
#include <fstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace odai::games::newvegas {

std::filesystem::path defaultTraversalStatePath() {
    if (const char* xdg = std::getenv("XDG_STATE_HOME")) {
        if (*xdg != '\0') {
            return std::filesystem::path(xdg) / "odai" / "traversal.json";
        }
    }
    if (const char* home = std::getenv("HOME")) {
        if (*home != '\0') {
            return std::filesystem::path(home) / ".local" / "state" / "odai" /
                "traversal.json";
        }
    }
    return std::filesystem::path("odai-traversal.json");
}

bool loadTraversalState(
    const std::filesystem::path& path, TraversalState& outState, std::string& outError) {
    outError.clear();
    std::ifstream input(path);
    if (!input) {
        outError = "cannot open traversal state " + path.string();
        return false;
    }
    try {
        nlohmann::json json;
        input >> json;
        if (json.value("version", 0u) != TraversalState::kVersion) {
            outError = "unsupported traversal state version";
            return false;
        }
        TraversalState parsed;
        parsed.interior = json.value("space_kind", std::string("exterior")) == "interior";
        parsed.worldspaceEditorId = json.value("worldspace", std::string{});
        parsed.interiorEditorId = json.value("interior", std::string{});
        const auto position = json.at("camera").at("position");
        if (!position.is_array() || position.size() != 3u) {
            outError = "traversal camera position must contain three numbers";
            return false;
        }
        for (std::size_t i = 0; i < 3u; ++i) {
            parsed.position[i] = position.at(i).get<float>();
        }
        parsed.yawDegrees = json.at("camera").value("yaw", 0.0f);
        parsed.pitchDegrees = json.at("camera").value("pitch", 0.0f);
        parsed.timeOfDayHours = json.value("time", 9.5f);
        parsed.weatherEditorId = json.value("weather", std::string{});
        parsed.loadOrderFingerprint = json.value("load_order_fingerprint", std::string{});
        for (const auto& item : json.value("discoveries", nlohmann::json::array())) {
            TraversalDiscovery discovery;
            discovery.sourceReferenceFormId = item.value("source_reference", 0u);
            discovery.worldspaceEditorId = item.value("worldspace", std::string{});
            discovery.name = item.value("name", std::string{});
            if (!discovery.name.empty()) {
                parsed.discoveries.push_back(std::move(discovery));
            }
        }
        outState = std::move(parsed);
        return true;
    } catch (const std::exception& exception) {
        outError = "invalid traversal state: " + std::string(exception.what());
        return false;
    }
}

bool saveTraversalStateAtomic(
    const std::filesystem::path& path, const TraversalState& state, std::string& outError) {
    outError.clear();
    std::error_code filesystemError;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), filesystemError);
        if (filesystemError) {
            outError = "cannot create traversal state directory: " + filesystemError.message();
            return false;
        }
    }
    nlohmann::json discoveries = nlohmann::json::array();
    for (const TraversalDiscovery& discovery : state.discoveries) {
        discoveries.push_back({
            {"source_reference", discovery.sourceReferenceFormId},
            {"worldspace", discovery.worldspaceEditorId},
            {"name", discovery.name}});
    }
    const nlohmann::json json = {
        {"version", TraversalState::kVersion},
        {"space_kind", state.interior ? "interior" : "exterior"},
        {"worldspace", state.worldspaceEditorId},
        {"interior", state.interiorEditorId},
        {"camera", {
            {"position", {state.position[0], state.position[1], state.position[2]}},
            {"yaw", state.yawDegrees}, {"pitch", state.pitchDegrees}}},
        {"time", state.timeOfDayHours},
        {"weather", state.weatherEditorId},
        {"load_order_fingerprint", state.loadOrderFingerprint},
        {"discoveries", std::move(discoveries)}};

    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            outError = "cannot write traversal state " + temporary.string();
            return false;
        }
        output << json.dump(2) << '\n';
        output.flush();
        if (!output) {
            outError = "failed while writing traversal state " + temporary.string();
            return false;
        }
    }
    std::filesystem::rename(temporary, path, filesystemError);
    if (filesystemError) {
        outError = "cannot atomically replace traversal state: " + filesystemError.message();
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    return true;
}

}  // namespace odai::games::newvegas
