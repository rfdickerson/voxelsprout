#pragma once

#include "dialogue/dialogue_context.h"
#include "dialogue/dialogue_types.h"

#include <vector>

// Walks a DialogueTree one node at a time, filtering choices against a
// DialogueContext and applying a choice's effects before advancing. This is
// the state machine a game's dialogue UI (e.g. ui::DialoguePanel) drives.
namespace odai::dialogue {

class DialogueRuntime {
public:
    // Enters tree at its startNode (auto-advancing through any autoNext chain)
    // and binds ctx for the rest of the conversation. Both must outlive the
    // runtime's use.
    void begin(const DialogueTree& tree, DialogueContext& ctx);

    // Null once isFinished() is true.
    [[nodiscard]] const DialogueNode* currentNode() const;

    // Only the current node's choices whose condition currently passes
    // against ctx.
    [[nodiscard]] std::vector<const DialogueChoice*> availableChoices() const;

    // Applies choice.effects, then follows targetNode (auto-advancing through
    // any autoNext chain). choice should be one returned by
    // availableChoices() for the current node.
    void choose(const DialogueChoice& choice);

    [[nodiscard]] bool isFinished() const { return finished_; }

private:
    void enterNode(const std::string& startId);
    [[nodiscard]] bool conditionPasses(const DialogueCondition& cond) const;

    const DialogueTree* tree_ = nullptr;
    DialogueContext* ctx_ = nullptr;
    const DialogueNode* current_ = nullptr;
    bool finished_ = true;
};

}  // namespace odai::dialogue
