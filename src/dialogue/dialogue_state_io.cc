#include "dialogue/dialogue_state_io.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iterator>

namespace odai::dialogue {

namespace {

using json = nlohmann::json;

std::string g_lastError;

void setLastError(std::string message) {
    g_lastError = std::move(message);
}

}  // namespace

std::string saveDialogueStateToJson(const MapDialogueContext& ctx) {
    json root;
    json flags = json::object();
    for (const auto& [name, value] : ctx.flags()) {
        flags[name] = value;
    }
    json approvals = json::object();
    for (const auto& [companionId, value] : ctx.approvals()) {
        approvals[companionId] = value;
    }
    root["flags"] = std::move(flags);
    root["approvals"] = std::move(approvals);
    return root.dump(2);
}

bool loadDialogueStateFromJson(const std::string& jsonText, MapDialogueContext& ctx) {
    json root;
    try {
        root = json::parse(jsonText);
    } catch (const json::exception& e) {
        setLastError(std::string("dialogue state: JSON parse error: ") + e.what());
        return false;
    }
    if (!root.is_object()) {
        setLastError("dialogue state: root is not an object");
        return false;
    }

    if (root.contains("flags") && root["flags"].is_object()) {
        for (const auto& [name, value] : root["flags"].items()) {
            ctx.setFlag(name, value.is_boolean() ? value.get<bool>() : false);
        }
    }
    if (root.contains("approvals") && root["approvals"].is_object()) {
        for (const auto& [companionId, value] : root["approvals"].items()) {
            ctx.setApproval(companionId, value.is_number_integer() ? value.get<int>() : 0);
        }
    }

    setLastError("");
    return true;
}

bool saveDialogueState(const MapDialogueContext& ctx, const std::filesystem::path& outputPath) {
    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file) {
        setLastError("dialogue state: cannot open '" + outputPath.string() + "' for writing");
        return false;
    }
    const std::string text = saveDialogueStateToJson(ctx);
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!file) {
        setLastError("dialogue state: write failed for '" + outputPath.string() + "'");
        return false;
    }
    setLastError("");
    return true;
}

bool loadDialogueState(const std::filesystem::path& inputPath, MapDialogueContext& ctx) {
    std::ifstream file(inputPath, std::ios::binary);
    if (!file) {
        setLastError("dialogue state: cannot open '" + inputPath.string() + "' for reading");
        return false;
    }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return loadDialogueStateFromJson(text, ctx);
}

const std::string& getDialogueStateLastError() {
    return g_lastError;
}

}  // namespace odai::dialogue
