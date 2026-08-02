#include "dialogue/dialogue_io.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iterator>

namespace odai::dialogue {

namespace {

using json = nlohmann::json;

DialogueCondition parseCondition(const json& j) {
    DialogueCondition c;
    if (!j.is_object()) return c;
    c.flag = j.value("flag", std::string());
    c.requiredValue = j.value("requiredValue", true);
    c.companionId = j.value("companion", std::string());
    c.minApproval = j.value("minApproval", 0);
    return c;
}

DialogueEffect parseEffect(const json& j, std::vector<std::string>& errors, const std::string& where) {
    DialogueEffect e;
    if (!j.is_object()) {
        errors.push_back("dialogue: effect in " + where + " is not an object, skipped");
        return e;
    }
    e.setFlag = j.value("setFlag", std::string());
    e.flagValue = j.value("value", true);
    e.companionId = j.value("companion", std::string());
    e.approvalDelta = j.value("approvalDelta", 0);
    return e;
}

DialogueChoice parseChoice(const json& j, std::vector<std::string>& errors, const std::string& where) {
    DialogueChoice choice;
    if (!j.is_object()) {
        errors.push_back("dialogue: choice in " + where + " is not an object, skipped");
        return choice;
    }
    choice.text = j.value("text", std::string());
    choice.targetNode = j.value("next", std::string());
    if (j.contains("condition")) choice.condition = parseCondition(j["condition"]);
    if (j.contains("effects") && j["effects"].is_array()) {
        for (const json& effectJson : j["effects"]) {
            choice.effects.push_back(parseEffect(effectJson, errors, where));
        }
    }
    return choice;
}

DialogueNode parseNode(const std::string& id, const json& j, std::vector<std::string>& errors) {
    DialogueNode node;
    node.id = id;
    if (!j.is_object()) {
        errors.push_back("dialogue: node '" + id + "' is not an object, skipped");
        return node;
    }
    node.speaker = j.value("speaker", std::string());
    node.text = j.value("text", std::string());
    node.autoNext = j.value("autoNext", std::string());
    if (j.contains("choices") && j["choices"].is_array()) {
        const std::string where = "node '" + id + "'";
        for (const json& choiceJson : j["choices"]) {
            node.choices.push_back(parseChoice(choiceJson, errors, where));
        }
    }
    return node;
}

// Records an error for any choice target / autoNext that doesn't name a real
// node, so broken authoring is caught at load time instead of surfacing as a
// silent dead end during play.
void validateReferences(DialogueLoadResult& result, const std::string& sourceLabel) {
    for (const auto& [nodeId, node] : result.tree.nodes) {
        for (const DialogueChoice& choice : node.choices) {
            if (!choice.targetNode.empty() &&
                result.tree.nodes.find(choice.targetNode) == result.tree.nodes.end()) {
                result.errors.push_back("dialogue: " + sourceLabel + " node '" + nodeId +
                                         "' choice targets unknown node '" + choice.targetNode + "'");
            }
        }
        if (!node.autoNext.empty() && result.tree.nodes.find(node.autoNext) == result.tree.nodes.end()) {
            result.errors.push_back("dialogue: " + sourceLabel + " node '" + nodeId +
                                     "' autoNext targets unknown node '" + node.autoNext + "'");
        }
    }
    if (!result.tree.startNode.empty() &&
        result.tree.nodes.find(result.tree.startNode) == result.tree.nodes.end()) {
        result.errors.push_back("dialogue: " + sourceLabel + " start node '" + result.tree.startNode +
                                 "' not found");
    }
}

}  // namespace

DialogueLoadResult loadDialogueTreeFromJson(const std::string& jsonText, const std::string& sourceLabel) {
    DialogueLoadResult result;
    json root;
    try {
        root = json::parse(jsonText);
    } catch (const json::exception& e) {
        result.errors.push_back("dialogue: JSON parse error in " + sourceLabel + ": " + e.what());
        return result;
    }
    if (!root.is_object()) {
        result.errors.push_back("dialogue: " + sourceLabel + " root is not an object");
        return result;
    }

    result.tree.id = root.value("id", std::string());
    result.tree.startNode = root.value("start", std::string());
    if (root.contains("nodes") && root["nodes"].is_object()) {
        for (const auto& [nodeId, nodeJson] : root["nodes"].items()) {
            result.tree.nodes.emplace(nodeId, parseNode(nodeId, nodeJson, result.errors));
        }
    } else {
        result.errors.push_back("dialogue: " + sourceLabel + " has no 'nodes' object");
    }

    validateReferences(result, sourceLabel);
    return result;
}

DialogueLoadResult loadDialogueTreeFromFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        DialogueLoadResult result;
        result.errors.push_back("dialogue: cannot open " + path.string());
        return result;
    }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return loadDialogueTreeFromJson(text, path.string());
}

}  // namespace odai::dialogue
