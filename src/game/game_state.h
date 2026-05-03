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

struct DialogueTopic {
    std::string id;
    std::string text;
};

struct DialogueResult {
    bool handled = false;
    std::string text;
    std::vector<DialogueTopic> topics;
    std::vector<DialogueChoice> choices;
};

struct ScriptCallResult {
    bool handled = false;
    std::string message;
};

struct PlayerStats {
    int health = 100;
    int maxHealth = 100;
    int magicka = 60;
    int maxMagicka = 60;
    int fatigue = 100;
    int maxFatigue = 100;
};

struct InventoryEntry {
    std::string id;
    std::string name;
    int count = 0;
};

struct QuestEntry {
    std::string id;
    std::string name;
    std::string objective;
    int stage = 0;
    bool completed = false;
};

class GameState {
public:
    void clear();

    [[nodiscard]] const PlayerStats& playerStats() const;
    void setPlayerMaxStats(int health, int magicka, int fatigue);
    void setPlayerStats(int health, int magicka, int fatigue);
    void damagePlayer(int amount);
    void restorePlayer(int health, int magicka, int fatigue);
    [[nodiscard]] bool spendPlayerMagicka(int amount);
    [[nodiscard]] bool spendPlayerFatigue(int amount);
    void recoverPlayer();
    [[nodiscard]] bool isPlayerDead() const;

    [[nodiscard]] int gold() const;
    void addGold(int amount);
    [[nodiscard]] bool spendGold(int amount);

    [[nodiscard]] int itemCount(const std::string& itemId) const;
    [[nodiscard]] std::string itemName(const std::string& itemId) const;
    void setItemName(const std::string& itemId, std::string name);
    void addItem(const std::string& itemId, int count);
    [[nodiscard]] bool removeItem(const std::string& itemId, int count);
    [[nodiscard]] std::vector<InventoryEntry> inventoryEntries() const;

    [[nodiscard]] int journalStage(const std::string& questId) const;
    void setJournalStage(const std::string& questId, int stage);
    void defineQuest(const std::string& questId, std::string name, std::string objective);
    void setQuestObjective(const std::string& questId, std::string objective);
    void completeQuest(const std::string& questId);
    [[nodiscard]] bool isQuestCompleted(const std::string& questId) const;
    [[nodiscard]] std::vector<QuestEntry> questEntries() const;

    [[nodiscard]] bool isRefDead(const std::string& refId) const;
    void setRefDead(const std::string& refId, bool dead);
    [[nodiscard]] bool isRefDisabled(const std::string& refId) const;
    void setRefDisabled(const std::string& refId, bool disabled);
    [[nodiscard]] int refHealth(const std::string& refId) const;
    void setRefHealth(const std::string& refId, int health);
    void damageRef(const std::string& refId, int amount);

    void appendLog(std::string message);
    [[nodiscard]] const std::vector<std::string>& log() const;

private:
    PlayerStats m_playerStats{};
    int m_gold = 0;
    std::unordered_map<std::string, int> m_inventory;
    std::unordered_map<std::string, std::string> m_itemNames;
    std::unordered_map<std::string, int> m_journalStages;
    std::unordered_map<std::string, QuestEntry> m_quests;
    std::unordered_set<std::string> m_deadRefs;
    std::unordered_set<std::string> m_disabledRefs;
    std::unordered_map<std::string, int> m_refHealth;
    std::vector<std::string> m_log;
};

}  // namespace odai::game
