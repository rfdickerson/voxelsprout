#include "game/game_state.h"

#include <algorithm>

namespace odai::game {

void GameState::clear() {
    *this = GameState{};
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

int GameState::journalStage(const std::string& questId) const {
    const auto it = m_journalStages.find(questId);
    return it == m_journalStages.end() ? 0 : it->second;
}

void GameState::setJournalStage(const std::string& questId, int stage) {
    if (questId.empty()) {
        return;
    }
    m_journalStages[questId] = std::max(0, stage);
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

void GameState::appendLog(std::string message) {
    if (!message.empty()) {
        m_log.push_back(std::move(message));
    }
}

const std::vector<std::string>& GameState::log() const {
    return m_log;
}

}  // namespace odai::game
