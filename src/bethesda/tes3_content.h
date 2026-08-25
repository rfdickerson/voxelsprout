#pragma once

#include "bethesda/runtime_ids.h"
#include "import/fnv/plugin_load_order.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <utility>

namespace odai::bethesda {

enum class Tes3DialogueType : std::int8_t {
    Unknown = -1,
    Topic = 0,
    Voice = 1,
    Greeting = 2,
    Persuasion = 3,
    Journal = 4,
};

enum class Tes3QuestStatus : std::uint8_t {
    None,
    Name,
    Finished,
    Restart,
};

enum class Tes3ConditionFunction : std::int16_t {
    Invalid = -1,
    FacReactionLowest = 0,
    // Native actor/player functions occupy the OpenMW-compatible 0..73 range.
    PcWerewolfKills = 73,
    Global = 74,
    Local,
    Journal,
    Item,
    Dead,
    NotId,
    NotFaction,
    NotClass,
    NotRace,
    NotCell,
    NotLocal,
};

struct Tes3DialogueCondition {
    std::uint8_t index = 0u;
    Tes3ConditionFunction function = Tes3ConditionFunction::Invalid;
    char comparison = ' ';
    std::string variable;
    std::variant<std::int32_t, float> value = std::int32_t{0};
    std::string rawRule;
    bool valid = false;
};

struct Tes3DialogueInfo {
    RecordKey record;
    std::string id;
    std::string previousId;
    std::string nextId;
    std::int32_t dispositionOrJournalIndex = 0;
    std::int8_t rank = -1;
    std::int8_t gender = -1;
    std::int8_t playerRank = -1;
    std::string actor;
    std::string race;
    std::string actorClass;
    std::string faction;
    std::string playerFaction;
    std::string cell;
    bool factionless = false;
    std::string sound;
    std::string response;
    std::string resultScript;
    std::vector<Tes3DialogueCondition> conditions;
    Tes3QuestStatus questStatus = Tes3QuestStatus::None;
    std::string sourcePlugin;
    std::uint64_t sourceOrdinal = 0u;
};

struct Tes3DialogueDefinition {
    RecordKey record;
    std::string id;
    Tes3DialogueType type = Tes3DialogueType::Unknown;
    std::vector<Tes3DialogueInfo> infos;
    std::string sourcePlugin;
};

struct Tes3ScriptDefinition {
    RecordKey record;
    std::string id;
    std::uint32_t shortCount = 0u;
    std::uint32_t longCount = 0u;
    std::uint32_t floatCount = 0u;
    std::vector<std::string> variableNames;
    std::vector<std::uint8_t> bytecode;
    std::string source;
    std::string sourcePlugin;
};

struct Tes3GlobalDefinition {
    RecordKey record;
    std::string id;
    char valueType = 'f';
    float value = 0.0f;
    std::string sourcePlugin;
};

struct Tes3ActorDefinition {
    struct TravelDestination {
        std::string cell;
        float position[3] = {};
        float rotationRadians[3] = {};
    };
    RecordKey record;
    std::string id;
    std::string name;
    std::string race;
    std::string actorClass;
    RecordKey faction;
    RecordKey script;
    std::int32_t level = 1;
    std::int32_t rank = -1;
    float health = 100.0f;
    float magicka = 100.0f;
    float fatigue = 100.0f;
    std::map<std::string, float> attributes;
    std::map<std::string, float> skills;
    bool creature = false;
    bool autoCalculate = false;
    std::uint32_t serviceFlags = 0u;
    std::vector<TravelDestination> travelDestinations;
    std::vector<std::pair<RecordKey, std::int32_t>> inventory;
    std::string sourcePlugin;
};

struct Tes3SpellEffect {
    std::int16_t effectId = -1;
    std::int8_t skill = -1;
    std::int8_t attribute = -1;
    std::int32_t range = 0;
    std::int32_t area = 0;
    std::int32_t duration = 0;
    std::int32_t magnitudeMin = 0;
    std::int32_t magnitudeMax = 0;
};

struct Tes3SpellDefinition {
    RecordKey record;
    std::string id;
    std::string name;
    std::int32_t type = 0;
    std::int32_t cost = 0;
    std::int32_t flags = 0;
    std::vector<Tes3SpellEffect> effects;
    std::string sourcePlugin;
};

struct Tes3SubrecordData {
    std::string type;
    std::vector<std::uint8_t> data;
};

// Generic immutable definition retained for gameplay records not requiring a
// specialized view yet (FACT, NPC_/CREA, items, cells, PGRD, and scripts).
struct Tes3NamedRecord {
    RecordKey record;
    std::string id;
    std::vector<Tes3SubrecordData> subrecords;
    std::string sourcePlugin;
};

struct Tes3ReferenceDefinition {
    ObjectId id;
    RecordKey cell;
    // Named TES3 exterior cells (Balmora, Ald'ruhn, etc.) do not use the
    // synthetic "#x,y" identity, so the CELL flag must remain explicit.
    bool interior = false;
    RecordKey base;
    std::string baseId;
    bool enabled = true;
    bool deleted = false;
    std::optional<float> scale;
    std::optional<std::int32_t> lockLevel;
    float position[3] = {};
    float rotationRadians[3] = {};
    bool hasTransform = false;
    std::string sourcePlugin;
    std::vector<Tes3SubrecordData> subrecords;
};

struct Tes3ContentStats {
    std::uint64_t recordsRead = 0u;
    std::uint64_t namedRecords = 0u;
    std::uint64_t dialogues = 0u;
    std::uint64_t infos = 0u;
    std::uint64_t scripts = 0u;
    std::uint64_t globals = 0u;
    std::uint64_t references = 0u;
    std::uint64_t deletions = 0u;
};

// Immutable, load-order-wide ESM3 content graph. Construction applies
// case-insensitive later-wins overrides and deletions before publishing maps.
class Tes3ContentStore {
public:
    bool load(const importer::fnv::FalloutLoadOrder& order, std::string encoding,
              std::string& outError);

    [[nodiscard]] const std::string& encoding() const { return m_encoding; }
    [[nodiscard]] const std::map<RecordKey, Tes3DialogueDefinition>& dialogues() const {
        return m_dialogues;
    }
    [[nodiscard]] const std::map<RecordKey, Tes3ScriptDefinition>& scripts() const {
        return m_scripts;
    }
    [[nodiscard]] const std::map<RecordKey, Tes3GlobalDefinition>& globals() const {
        return m_globals;
    }
    [[nodiscard]] const std::map<RecordKey, Tes3ActorDefinition>& actors() const {
        return m_actors;
    }
    [[nodiscard]] const std::map<RecordKey, Tes3SpellDefinition>& spells() const {
        return m_spells;
    }
    [[nodiscard]] const std::map<RecordKey, Tes3NamedRecord>& namedRecords() const {
        return m_namedRecords;
    }
    [[nodiscard]] const std::map<ObjectId, Tes3ReferenceDefinition>& references() const {
        return m_references;
    }
    [[nodiscard]] const Tes3ContentStats& stats() const { return m_stats; }

    [[nodiscard]] const Tes3DialogueDefinition* findDialogue(std::string_view id) const;
    [[nodiscard]] const Tes3ScriptDefinition* findScript(std::string_view id) const;
    [[nodiscard]] const Tes3ActorDefinition* findActor(
        std::string_view type, std::string_view id) const;
    [[nodiscard]] const Tes3SpellDefinition* findSpell(std::string_view id) const;
    [[nodiscard]] const Tes3NamedRecord* findRecord(
        std::string_view type, std::string_view id) const;

private:
    std::string m_encoding = "windows-1252";
    std::map<RecordKey, Tes3DialogueDefinition> m_dialogues;
    std::map<RecordKey, Tes3ScriptDefinition> m_scripts;
    std::map<RecordKey, Tes3GlobalDefinition> m_globals;
    std::map<RecordKey, Tes3ActorDefinition> m_actors;
    std::map<RecordKey, Tes3SpellDefinition> m_spells;
    std::map<RecordKey, Tes3NamedRecord> m_namedRecords;
    std::map<ObjectId, Tes3ReferenceDefinition> m_references;
    Tes3ContentStats m_stats;
};

// Decodes a TES3 byte string without depending on host locale. Western
// profiles support the complete Windows-1252 mapping used by Morrowind/TR.
[[nodiscard]] std::string decodeTes3Text(std::string_view bytes, std::string_view encoding);

}  // namespace odai::bethesda
