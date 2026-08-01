#pragma once

#include <string>
#include <unordered_map>

// Minimal read/write surface for the world state dialogue conditions/effects
// touch: boolean flags and per-companion approval. Kept abstract so the
// dialogue module has no dependency on any specific game's data model
// (game::GameState, citybuilder's own state, etc.) — a concrete game wires its
// own flag/relationship storage through this interface.
namespace odai::dialogue {

class DialogueContext {
public:
    virtual ~DialogueContext() = default;

    virtual bool flag(const std::string& name) const = 0;
    virtual void setFlag(const std::string& name, bool value) = 0;

    virtual int approval(const std::string& companionId) const = 0;
    virtual void adjustApproval(const std::string& companionId, int delta) = 0;
};

// In-memory DialogueContext, good enough for tests, tools, and any game that
// doesn't yet have its own flag/relationship storage to hook up.
class MapDialogueContext : public DialogueContext {
public:
    bool flag(const std::string& name) const override {
        const auto it = flags_.find(name);
        return it != flags_.end() && it->second;
    }
    void setFlag(const std::string& name, bool value) override { flags_[name] = value; }

    int approval(const std::string& companionId) const override {
        const auto it = approval_.find(companionId);
        return it != approval_.end() ? it->second : 0;
    }
    void adjustApproval(const std::string& companionId, int delta) override {
        approval_[companionId] += delta;
    }

private:
    std::unordered_map<std::string, bool> flags_;
    std::unordered_map<std::string, int> approval_;
};

}  // namespace odai::dialogue
