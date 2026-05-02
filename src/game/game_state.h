#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace odai::game {

struct DialogueChoice {
    std::string id;
    std::string text;
};

struct DialogueResult {
    bool handled = false;
    std::string text;
    std::vector<DialogueChoice> choices;
};

struct ScriptCallResult {
    bool handled = false;
    std::string message;
};

class GameState {
public:
    void clear();

    [[nodiscard]] int gold() const;
    void addGold(int amount);
    [[nodiscard]] bool spendGold(int amount);

    [[nodiscard]] int itemCount(const std::string& itemId) const;
    void addItem(const std::string& itemId, int count);
    [[nodiscard]] bool removeItem(const std::string& itemId, int count);

    [[nodiscard]] int journalStage(const std::string& questId) const;
    void setJournalStage(const std::string& questId, int stage);

    [[nodiscard]] bool isRefDead(const std::string& refId) const;
    void setRefDead(const std::string& refId, bool dead);
    [[nodiscard]] bool isRefDisabled(const std::string& refId) const;
    void setRefDisabled(const std::string& refId, bool disabled);

    void appendLog(std::string message);
    [[nodiscard]] const std::vector<std::string>& log() const;

private:
    int m_gold = 0;
    std::unordered_map<std::string, int> m_inventory;
    std::unordered_map<std::string, int> m_journalStages;
    std::unordered_set<std::string> m_deadRefs;
    std::unordered_set<std::string> m_disabledRefs;
    std::vector<std::string> m_log;
};

}  // namespace odai::game
