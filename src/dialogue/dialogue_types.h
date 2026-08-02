#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// Data model for branching dialogue trees — the Dragon Age: Origins touchstone
// (see docs/ROADMAP.md's Party RPG / Narrative section). Pure flat data, no UI
// and no dependency on any specific game's state: conditions/effects operate
// through the abstract DialogueContext (dialogue_context.h) instead of reaching
// into a particular GameState.
namespace odai::dialogue {

// A choice is offered only when its condition passes. An empty flag/companionId
// skips that half of the check (always true).
struct DialogueCondition {
    std::string flag;
    bool requiredValue = true;
    std::string companionId;
    int minApproval = 0;
};

// Applied when the player picks the choice, before moving to targetNode.
struct DialogueEffect {
    std::string setFlag;
    bool flagValue = true;
    std::string companionId;
    int approvalDelta = 0;
};

struct DialogueChoice {
    std::string text;
    std::string targetNode;       // "" ends the dialogue
    DialogueCondition condition;  // default-constructed condition = always available
    std::vector<DialogueEffect> effects;
};

struct DialogueNode {
    std::string id;
    std::string speaker;  // "" = narrator/player-only line
    std::string text;     // rich-text markup, rendered as-is by the UI
    std::vector<DialogueChoice> choices;
    // If choices is empty and autoNext is non-empty, the runtime advances here
    // immediately (a narration beat). Empty choices + empty autoNext is a
    // terminal line — the conversation's last thing said.
    std::string autoNext;
};

struct DialogueTree {
    std::string id;
    std::string startNode;
    std::unordered_map<std::string, DialogueNode> nodes;
};

}  // namespace odai::dialogue
