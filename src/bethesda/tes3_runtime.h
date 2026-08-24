#pragma once

#include "bethesda/tes3_content.h"
#include "bethesda/tes3_script.h"
#include "bethesda/runtime_transform.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace odai::bethesda {

enum class Tes3JournalQuestClassification : std::uint8_t {
    Legacy,
    Active,
    Completed,
};

struct Tes3JournalVisit {
    std::uint64_t sequence = 0u;
    std::uint64_t tick = 0u;
    RecordKey quest;
    RecordKey info;
    std::int32_t index = 0;
    Tes3QuestStatus status = Tes3QuestStatus::None;
    std::string sourcePlugin;
    friend bool operator==(const Tes3JournalVisit&, const Tes3JournalVisit&) = default;
};

struct Tes3JournalQuestState {
    RecordKey quest;
    std::string id;
    std::int32_t currentIndex = 0;
    std::vector<RecordKey> visitedEntries;
    Tes3JournalQuestClassification classification = Tes3JournalQuestClassification::Legacy;
    bool hasStatusFlags = false;
    friend bool operator==(const Tes3JournalQuestState&, const Tes3JournalQuestState&) = default;
};

class Tes3Journal {
public:
    bool addEntry(
        const Tes3DialogueDefinition& quest, std::int32_t index,
        std::uint64_t tick, std::string& outError);
    bool setIndex(
        const Tes3DialogueDefinition& quest, std::int32_t index,
        std::string& outError);
    [[nodiscard]] std::int32_t index(std::string_view questId) const;
    [[nodiscard]] const Tes3JournalQuestState* find(std::string_view questId) const;
    [[nodiscard]] const std::map<RecordKey, Tes3JournalQuestState>& quests() const {
        return m_quests;
    }
    [[nodiscard]] std::map<RecordKey, Tes3JournalQuestState>& questsForRestore() {
        return m_quests;
    }
    [[nodiscard]] const std::vector<Tes3JournalVisit>& chronology() const {
        return m_chronology;
    }
    [[nodiscard]] std::vector<Tes3JournalVisit>& chronologyForRestore() {
        return m_chronology;
    }
    [[nodiscard]] std::uint64_t nextSequence() const { return m_nextSequence; }
    void setNextSequence(std::uint64_t value) { m_nextSequence = value == 0u ? 1u : value; }
    void clear();

private:
    std::map<RecordKey, Tes3JournalQuestState> m_quests;
    std::vector<Tes3JournalVisit> m_chronology;
    std::uint64_t m_nextSequence = 1u;
};

struct Tes3DialogueActorState {
    ObjectId object;
    std::string id;
    std::string race;
    std::string actorClass;
    std::string faction;
    std::string cell;
    std::int8_t rank = -1;
    std::int8_t gender = -1;
    float disposition = 50.0f;
    std::map<std::string, double> locals;
};

struct Tes3DialoguePlayerState {
    ObjectId object;
    std::map<std::string, std::int8_t> factionRanks;
    std::map<std::string, double> numericFilters;
    std::map<RecordKey, std::int32_t> inventory;
    std::map<std::string, std::int32_t> deathCounts;
};

struct Tes3DialogueChoice {
    std::string label;
    std::int32_t value = 0;
    friend bool operator==(const Tes3DialogueChoice&, const Tes3DialogueChoice&) = default;
};

struct Tes3DialogueResponse {
    bool accepted = false;
    bool goodbye = false;
    RecordKey topic;
    RecordKey info;
    std::string text;
    std::vector<Tes3DialogueChoice> choices;
    std::vector<std::string> discoveredTopics;
    std::vector<std::string> diagnostics;
};

struct Tes3DialogueState {
    bool active = false;
    Tes3DialogueActorState actor;
    Tes3DialoguePlayerState player;
    RecordKey currentTopic;
    RecordKey currentInfo;
    std::set<RecordKey> exhaustedInfos;
    std::int32_t choice = -1;
    bool goodbye = false;
    std::vector<Tes3DialogueChoice> choices;
    friend bool operator==(const Tes3DialogueState&, const Tes3DialogueState&) = default;
};

struct Tes3ScriptCheckReport {
    std::uint64_t scripts = 0u;
    std::uint64_t resultScripts = 0u;
    std::uint64_t compiled = 0u;
    std::map<std::string, std::uint64_t> commandUse;
    std::set<std::string> unsupportedCommands;
    std::vector<std::string> diagnostics;
    [[nodiscard]] bool strictPass() const {
        return diagnostics.empty() && unsupportedCommands.empty() &&
            compiled == scripts + resultScripts;
    }
};

struct Tes3ReferenceOverride {
    std::optional<bool> enabled;
    bool deleted = false;
    std::optional<RuntimeTransform> transform;
    std::map<std::string, Tes3Value> locals;
    friend bool operator==(const Tes3ReferenceOverride&, const Tes3ReferenceOverride&) = default;
};

struct Tes3ActiveSpellEffect {
    std::int16_t effectId = -1;
    std::int8_t skill = -1;
    std::int8_t attribute = -1;
    double magnitude = 0.0;
    std::uint64_t expiresTick = 0u;
    friend bool operator==(const Tes3ActiveSpellEffect&, const Tes3ActiveSpellEffect&) = default;
};

struct Tes3ActiveSpell {
    RecordKey spell;
    ObjectId caster;
    std::uint64_t appliedTick = 0u;
    std::vector<Tes3ActiveSpellEffect> effects;
    friend bool operator==(const Tes3ActiveSpell&, const Tes3ActiveSpell&) = default;
};

using Tes3ExternalNativeExecutor = std::function<Tes3NativeResult(const Tes3NativeCall&)>;

class Tes3Runtime {
public:
    bool configure(
        std::shared_ptr<const Tes3ContentStore> content, ObjectId player,
        std::string& outError);
    void setExternalNativeExecutor(Tes3ExternalNativeExecutor executor) {
        m_externalNative = std::move(executor);
    }
    [[nodiscard]] Tes3VmStepResult step(
        std::uint64_t tick, std::uint32_t instructionBudget = 10000u);

    [[nodiscard]] Tes3DialogueResponse startDialogue(
        Tes3DialogueActorState actor, Tes3DialoguePlayerState player,
        bool strict = true);
    [[nodiscard]] std::vector<std::string> availableTopics(bool strict = true) const;
    [[nodiscard]] Tes3DialogueResponse selectTopic(
        std::string_view topicId, bool strict = true);
    [[nodiscard]] Tes3DialogueResponse answerChoice(
        std::int32_t value, bool strict = true);
    void endDialogue();
    void dispatchGameplayEvent(
        std::string eventName, ObjectId target,
        Tes3Value value = Tes3Value::fromNumber(1.0));

    bool addTopic(std::string_view topicId);
    [[nodiscard]] const std::set<RecordKey>& knownTopics() const { return m_knownTopics; }
    [[nodiscard]] std::set<RecordKey>& knownTopicsForRestore() { return m_knownTopics; }
    [[nodiscard]] Tes3Journal& journal() { return m_journal; }
    [[nodiscard]] const Tes3Journal& journal() const { return m_journal; }
    [[nodiscard]] Tes3ScriptVm& scripts() { return m_scripts; }
    [[nodiscard]] const Tes3ScriptVm& scripts() const { return m_scripts; }
    [[nodiscard]] const Tes3DialogueState& dialogue() const { return m_dialogue; }
    [[nodiscard]] Tes3DialogueState& dialogueForRestore() { return m_dialogue; }
    [[nodiscard]] const Tes3ScriptCheckReport& scriptCheckReport() const { return m_scriptCheck; }
    [[nodiscard]] const std::shared_ptr<const Tes3ContentStore>& content() const { return m_content; }
    [[nodiscard]] const std::map<ObjectId, Tes3ReferenceOverride>& referenceOverrides() const {
        return m_referenceOverrides;
    }
    [[nodiscard]] const std::set<std::string>& activeSounds() const { return m_activeSounds; }
    [[nodiscard]] std::set<std::string>& activeSoundsForRestore() { return m_activeSounds; }
    [[nodiscard]] std::map<ObjectId, Tes3ReferenceOverride>& referenceOverridesForRestore() {
        return m_referenceOverrides;
    }
    [[nodiscard]] const std::map<ObjectId, std::vector<Tes3ActiveSpell>>& activeSpells() const {
        return m_activeSpells;
    }
    [[nodiscard]] std::map<ObjectId, std::vector<Tes3ActiveSpell>>& activeSpellsForRestore() {
        return m_activeSpells;
    }
    [[nodiscard]] ObjectId playerObject() const { return m_player; }
    [[nodiscard]] Tes3DialoguePlayerState& playerState() { return m_playerState; }
    [[nodiscard]] const Tes3DialoguePlayerState& playerState() const { return m_playerState; }
    void clear();

private:
    [[nodiscard]] bool matches(
        const Tes3DialogueInfo& info, const Tes3DialogueActorState& actor,
        const Tes3DialoguePlayerState& player, bool strict) const;
    [[nodiscard]] const Tes3DialogueInfo* selectInfo(
        const Tes3DialogueDefinition& topic, bool strict) const;
    [[nodiscard]] Tes3DialogueResponse activateInfo(
        const Tes3DialogueDefinition& topic, const Tes3DialogueInfo& info,
        bool strict);
    void discoverTopics(std::string_view response, std::vector<std::string>& outDiscovered);
    [[nodiscard]] Tes3NativeResult executeNative(const Tes3NativeCall& call);
    [[nodiscard]] std::string resultProgramId(const RecordKey& info) const;

    std::shared_ptr<const Tes3ContentStore> m_content;
    ObjectId m_player;
    Tes3DialoguePlayerState m_playerState;
    Tes3Journal m_journal;
    Tes3ScriptVm m_scripts;
    Tes3NativeRegistry m_nativeRegistry;
    Tes3ScriptCheckReport m_scriptCheck;
    Tes3DialogueState m_dialogue;
    std::set<RecordKey> m_knownTopics;
    std::map<ObjectId, Tes3ReferenceOverride> m_referenceOverrides;
    std::map<ObjectId, std::vector<Tes3ActiveSpell>> m_activeSpells;
    std::set<std::string> m_activeSounds;
    Tes3ExternalNativeExecutor m_externalNative;
    std::uint64_t m_currentTick = 0u;
};

}  // namespace odai::bethesda
