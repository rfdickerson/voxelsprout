#include "game/game_state.h"

#include <algorithm>
#include <cctype>

namespace odai::game {
namespace {

std::string titleFromId(const std::string& id) {
    std::string result = id;
    bool capitalizeNext = true;
    for (char& c : result) {
        if (c == '_' || c == '-') {
            c = ' ';
            capitalizeNext = true;
            continue;
        }
        if (capitalizeNext) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            capitalizeNext = false;
        }
    }
    return result;
}

}  // namespace

void GameState::clear() {
    *this = GameState{};
}

const PlayerStats& GameState::playerStats() const {
    return m_playerStats;
}

void GameState::setPlayerMaxStats(int health, int magicka, int fatigue) {
    m_playerStats.maxHealth = std::max(1, health);
    m_playerStats.maxMagicka = std::max(0, magicka);
    m_playerStats.maxFatigue = std::max(1, fatigue);
    setPlayerStats(m_playerStats.health, m_playerStats.magicka, m_playerStats.fatigue);
}

void GameState::setPlayerStats(int health, int magicka, int fatigue) {
    m_playerStats.health = std::clamp(health, 0, m_playerStats.maxHealth);
    m_playerStats.magicka = std::clamp(magicka, 0, m_playerStats.maxMagicka);
    m_playerStats.fatigue = std::clamp(fatigue, 0, m_playerStats.maxFatigue);
}

void GameState::damagePlayer(int amount) {
    if (amount <= 0) {
        return;
    }
    setPlayerStats(m_playerStats.health - amount, m_playerStats.magicka, m_playerStats.fatigue);
}

void GameState::restorePlayer(int health, int magicka, int fatigue) {
    setPlayerStats(
        m_playerStats.health + std::max(0, health),
        m_playerStats.magicka + std::max(0, magicka),
        m_playerStats.fatigue + std::max(0, fatigue)
    );
}

bool GameState::spendPlayerMagicka(int amount) {
    if (amount < 0 || m_playerStats.magicka < amount) {
        return false;
    }
    m_playerStats.magicka -= amount;
    return true;
}

bool GameState::spendPlayerFatigue(int amount) {
    if (amount < 0 || m_playerStats.fatigue < amount) {
        return false;
    }
    m_playerStats.fatigue -= amount;
    return true;
}

void GameState::recoverPlayer() {
    setPlayerStats(m_playerStats.maxHealth, m_playerStats.maxMagicka, m_playerStats.maxFatigue);
}

bool GameState::isPlayerDead() const {
    return m_playerStats.health <= 0;
}

int GameState::gold() const {
    return m_gold;
}

void GameState::addGold(int amount) {
    m_gold = std::max(0, m_gold + amount);
}

bool GameState::spendGold(int amount) {
    if (amount < 0 || m_gold < amount) {
        return false;
    }
    m_gold -= amount;
    return true;
}

int GameState::itemCount(const std::string& itemId) const {
    const auto it = m_inventory.find(itemId);
    return it == m_inventory.end() ? 0 : it->second;
}

std::string GameState::itemName(const std::string& itemId) const {
    const auto it = m_itemNames.find(itemId);
    return it == m_itemNames.end() ? titleFromId(itemId) : it->second;
}

void GameState::setItemName(const std::string& itemId, std::string name) {
    if (!itemId.empty() && !name.empty()) {
        m_itemNames[itemId] = std::move(name);
    }
}

void GameState::addItem(const std::string& itemId, int count) {
    if (itemId.empty() || count <= 0) {
        return;
    }
    m_inventory[itemId] = std::max(0, itemCount(itemId) + count);
}

bool GameState::removeItem(const std::string& itemId, int count) {
    if (itemId.empty() || count <= 0) {
        return false;
    }
    auto it = m_inventory.find(itemId);
    if (it == m_inventory.end() || it->second < count) {
        return false;
    }
    it->second -= count;
    if (it->second == 0) {
        m_inventory.erase(it);
    }
    return true;
}

std::vector<InventoryEntry> GameState::inventoryEntries() const {
    std::vector<InventoryEntry> entries;
    entries.reserve(m_inventory.size());
    for (const auto& [id, count] : m_inventory) {
        if (count <= 0) {
            continue;
        }
        entries.push_back({id, itemName(id), count});
    }
    std::sort(entries.begin(), entries.end(), [](const InventoryEntry& a, const InventoryEntry& b) {
        return a.name < b.name;
    });
    return entries;
}

int GameState::journalStage(const std::string& questId) const {
    const auto it = m_journalStages.find(questId);
    return it == m_journalStages.end() ? 0 : it->second;
}

void GameState::setJournalStage(const std::string& questId, int stage) {
    if (questId.empty()) {
        return;
    }
    m_journalStages[questId] = std::max(0, stage);
    auto questIt = m_quests.find(questId);
    if (questIt != m_quests.end()) {
        questIt->second.stage = std::max(questIt->second.stage, std::max(0, stage));
    }
}

void GameState::defineQuest(const std::string& questId, std::string name, std::string objective) {
    if (questId.empty()) {
        return;
    }
    QuestEntry& quest = m_quests[questId];
    quest.id = questId;
    if (!name.empty()) {
        quest.name = std::move(name);
    } else if (quest.name.empty()) {
        quest.name = titleFromId(questId);
    }
    quest.objective = std::move(objective);
    quest.stage = journalStage(questId);
}

void GameState::setQuestObjective(const std::string& questId, std::string objective) {
    if (questId.empty()) {
        return;
    }
    QuestEntry& quest = m_quests[questId];
    quest.id = questId;
    if (quest.name.empty()) {
        quest.name = titleFromId(questId);
    }
    quest.objective = std::move(objective);
    quest.stage = journalStage(questId);
}

void GameState::completeQuest(const std::string& questId) {
    if (questId.empty()) {
        return;
    }
    QuestEntry& quest = m_quests[questId];
    quest.id = questId;
    if (quest.name.empty()) {
        quest.name = titleFromId(questId);
    }
    quest.stage = std::max(quest.stage, journalStage(questId));
    quest.completed = true;
}

bool GameState::isQuestCompleted(const std::string& questId) const {
    const auto it = m_quests.find(questId);
    return it != m_quests.end() && it->second.completed;
}

std::vector<QuestEntry> GameState::questEntries() const {
    std::vector<QuestEntry> entries;
    entries.reserve(m_quests.size());
    for (const auto& [id, quest] : m_quests) {
        QuestEntry entry = quest;
        entry.stage = std::max(entry.stage, journalStage(id));
        if (entry.stage > 0 || !entry.objective.empty() || entry.completed) {
            entries.push_back(std::move(entry));
        }
    }
    std::sort(entries.begin(), entries.end(), [](const QuestEntry& a, const QuestEntry& b) {
        if (a.completed != b.completed) {
            return !a.completed;
        }
        return a.name < b.name;
    });
    return entries;
}

bool GameState::isRefDead(const std::string& refId) const {
    return m_deadRefs.find(refId) != m_deadRefs.end();
}

void GameState::setRefDead(const std::string& refId, bool dead) {
    if (refId.empty()) {
        return;
    }
    if (dead) {
        m_deadRefs.insert(refId);
    } else {
        m_deadRefs.erase(refId);
    }
}

bool GameState::isRefDisabled(const std::string& refId) const {
    return m_disabledRefs.find(refId) != m_disabledRefs.end();
}

void GameState::setRefDisabled(const std::string& refId, bool disabled) {
    if (refId.empty()) {
        return;
    }
    if (disabled) {
        m_disabledRefs.insert(refId);
    } else {
        m_disabledRefs.erase(refId);
    }
}

int GameState::refHealth(const std::string& refId) const {
    const auto it = m_refHealth.find(refId);
    return it == m_refHealth.end() ? 0 : it->second;
}

void GameState::setRefHealth(const std::string& refId, int health) {
    if (refId.empty()) {
        return;
    }
    m_refHealth[refId] = std::max(0, health);
    if (health <= 0) {
        setRefDead(refId, true);
    }
}

void GameState::damageRef(const std::string& refId, int amount) {
    if (refId.empty() || amount <= 0 || isRefDead(refId)) {
        return;
    }
    const int current = std::max(1, refHealth(refId));
    setRefHealth(refId, current - amount);
}

void GameState::appendLog(std::string message) {
    if (!message.empty()) {
        m_log.push_back(std::move(message));
    }
}

const std::vector<std::string>& GameState::log() const {
    return m_log;
}

}  // namespace odai::game
