#include "dialogue/dialogue_runtime.h"

namespace odai::dialogue {

namespace {
// Guards a malformed/cyclic autoNext chain in authored data from hanging the
// runtime instead of surfacing as a visible dead end.
constexpr int kMaxAutoHops = 64;
}  // namespace

void DialogueRuntime::begin(const DialogueTree& tree, DialogueContext& ctx) {
    tree_ = &tree;
    ctx_ = &ctx;
    enterNode(tree.startNode);
}

void DialogueRuntime::enterNode(const std::string& startId) {
    std::string nodeId = startId;
    for (int hops = 0; hops < kMaxAutoHops; ++hops) {
        if (tree_ == nullptr || nodeId.empty()) {
            current_ = nullptr;
            finished_ = true;
            return;
        }
        const auto it = tree_->nodes.find(nodeId);
        if (it == tree_->nodes.end()) {
            current_ = nullptr;
            finished_ = true;
            return;
        }
        current_ = &it->second;
        if (!current_->choices.empty() || current_->autoNext.empty()) {
            finished_ = false;
            return;
        }
        nodeId = current_->autoNext;
    }
    // Hop cap exceeded — treat as a dead end rather than hang.
    current_ = nullptr;
    finished_ = true;
}

bool DialogueRuntime::conditionPasses(const DialogueCondition& cond) const {
    if (ctx_ == nullptr) return true;
    if (!cond.flag.empty() && ctx_->flag(cond.flag) != cond.requiredValue) return false;
    if (!cond.companionId.empty() && ctx_->approval(cond.companionId) < cond.minApproval) return false;
    return true;
}

const DialogueNode* DialogueRuntime::currentNode() const {
    return finished_ ? nullptr : current_;
}

std::vector<const DialogueChoice*> DialogueRuntime::availableChoices() const {
    std::vector<const DialogueChoice*> out;
    if (finished_ || current_ == nullptr) return out;
    for (const DialogueChoice& choice : current_->choices) {
        if (conditionPasses(choice.condition)) out.push_back(&choice);
    }
    return out;
}

void DialogueRuntime::choose(const DialogueChoice& choice) {
    if (finished_ || ctx_ == nullptr) return;
    for (const DialogueEffect& effect : choice.effects) {
        if (!effect.setFlag.empty()) ctx_->setFlag(effect.setFlag, effect.flagValue);
        if (!effect.companionId.empty() && effect.approvalDelta != 0) {
            ctx_->adjustApproval(effect.companionId, effect.approvalDelta);
        }
    }
    enterNode(choice.targetNode);
}

}  // namespace odai::dialogue
