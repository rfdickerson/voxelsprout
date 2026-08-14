#include "games/newvegas/newvegas_character.h"

#include "import/fnv/esm_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace odai::games::newvegas {

namespace {

std::string subrecordString(const importer::fnv::EsmSubrecordView& sub) {
    if (sub.data == nullptr || sub.size == 0u) {
        return {};
    }
    std::string value(reinterpret_cast<const char*>(sub.data), static_cast<std::size_t>(sub.size));
    while (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

// The seven attributes are CONSECUTIVE records, AVStrength at 0x3E8 through
// AVLuck at 0x3EE, and that sequence is already S.P.E.C.I.A.L. order -- so the
// order the screen shows them in is the game's, not a choice made here.
constexpr std::uint32_t kFirstAttributeFormId = 0x3E8u;
constexpr std::uint32_t kLastAttributeFormId = 0x3EEu;

// The thirteen skills, by EDITOR ID, which is the stable half of the record.
// Their display names are read from FULL and deliberately not written here --
// that is the whole point, since two of these do not say what they mean:
// AVSmallGuns is "Guns" and AVThrowing is "Survival". AVBigGuns is omitted; it
// still exists in the file and its own FULL reads "Big Guns - OBSOLETE".
constexpr const char* kSkillEditorIds[] = {
    "AVBarter",   "AVEnergyWeapons", "AVExplosives", "AVLockpick", "AVMedicine",
    "AVMeleeWeapons", "AVRepair",    "AVScience",    "AVSmallGuns", "AVSneak",
    "AVSpeech",   "AVThrowing",      "AVUnarmed",
};

// Which attribute governs which skill. The ONE table here that is not read from
// the records: AVIF carries no governing-attribute field, so this is Fallout's
// documented pairing, keyed by the same editor IDs above.
const std::unordered_map<std::string, int>& governingAttribute() {
    // Indices are into the S.P.E.C.I.A.L. order above.
    static const std::unordered_map<std::string, int> table = {
        {"AVBarter", 3},        // Charisma
        {"AVEnergyWeapons", 1}, // Perception
        {"AVExplosives", 1},    // Perception
        {"AVLockpick", 1},      // Perception
        {"AVMedicine", 4},      // Intelligence
        {"AVMeleeWeapons", 0},  // Strength
        {"AVRepair", 4},        // Intelligence
        {"AVScience", 4},       // Intelligence
        {"AVSmallGuns", 5},     // Agility
        {"AVSneak", 5},         // Agility
        {"AVSpeech", 3},        // Charisma
        {"AVThrowing", 2},      // Endurance
        {"AVUnarmed", 2},       // Endurance
    };
    return table;
}

}  // namespace

bool loadCharacterDefinitions(
    const std::filesystem::path& pluginPath,
    CharacterDefinitions& outDefinitions,
    std::string& outError
) {
    outDefinitions = CharacterDefinitions{};
    outError.clear();

    importer::fnv::EsmReader reader;
    if (!reader.open(pluginPath)) {
        outError = reader.lastError();
        return false;
    }

    std::unordered_map<std::string, CharacterStatDefinition> skillsByEditorId;
    std::vector<std::pair<std::uint32_t, CharacterStatDefinition>> attributes;

    importer::fnv::EsmReader::Visitor visitor;
    // Without this every record in the file is materialized to be handed to
    // onRecord, which for FalloutNV.esm means inflating its 29363 compressed
    // LAND records to go looking for seven attributes. See CLAUDE.md.
    visitor.onRecordHeader = [](const importer::fnv::EsmRecordHeaderView& header) {
        return header.type == "AVIF" || header.type == "PERK";
    };
    visitor.onRecord = [&](const importer::fnv::EsmRecordView& record) {
        CharacterStatDefinition entry;
        entry.formId = record.formId;
        bool isTrait = false;
        bool playable = true;
        for (const auto& sub : record.subrecords) {
            if (sub.type == "EDID") {
                entry.editorId = subrecordString(sub);
            } else if (sub.type == "FULL") {
                entry.name = subrecordString(sub);
            } else if (sub.type == "DESC") {
                entry.description = subrecordString(sub);
            } else if (record.type == "PERK" && sub.type == "DATA" && sub.size >= 4u) {
                // PERK DATA: trait u8, minLevel u8, ranks u8, playable u8,
                // hidden u8. A trait is the subset a character takes at
                // creation instead of earning on level-up.
                isTrait = sub.data[0] != 0u;
                playable = sub.data[3] != 0u;
            }
        }
        if (entry.name.empty()) {
            entry.name = entry.editorId;  // unnamed records still beat a blank row
        }
        if (record.type == "AVIF") {
            if (record.formId >= kFirstAttributeFormId && record.formId <= kLastAttributeFormId) {
                attributes.emplace_back(record.formId, entry);
                return;
            }
            for (const char* wanted : kSkillEditorIds) {
                if (entry.editorId == wanted) {
                    skillsByEditorId.emplace(entry.editorId, entry);
                    break;
                }
            }
            return;
        }
        if (isTrait && playable) {
            outDefinitions.traits.push_back(entry);
        }
    };
    if (!reader.walk(visitor)) {
        outError = reader.lastError();
        return false;
    }

    // By formID, which is S.P.E.C.I.A.L. order. Records do not stream in any
    // guaranteed order, so sorting is what makes the sequence the game's rather
    // than the file's.
    std::sort(attributes.begin(), attributes.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    outDefinitions.attributes.reserve(attributes.size());
    for (auto& [formId, entry] : attributes) {
        outDefinitions.attributes.push_back(std::move(entry));
    }

    // In the order kSkillEditorIds lists them, which is alphabetical BY DISPLAY
    // NAME for eleven of the thirteen -- the two exceptions being the renamed
    // pair, and a player reading down the column wants "Guns" where G belongs.
    for (const char* wanted : kSkillEditorIds) {
        const auto found = skillsByEditorId.find(wanted);
        if (found != skillsByEditorId.end()) {
            outDefinitions.skills.push_back(found->second);
        }
    }
    std::sort(outDefinitions.skills.begin(), outDefinitions.skills.end(),
              [](const CharacterStatDefinition& a, const CharacterStatDefinition& b) {
                  return a.name < b.name;
              });
    std::sort(outDefinitions.traits.begin(), outDefinitions.traits.end(),
              [](const CharacterStatDefinition& a, const CharacterStatDefinition& b) {
                  return a.name < b.name;
              });
    return true;
}

bool PlayerCharacter::hasTag(std::size_t skillIndex) const {
    return std::find(taggedSkills.begin(), taggedSkills.end(), skillIndex) != taggedSkills.end();
}

bool PlayerCharacter::hasTrait(std::size_t traitIndex) const {
    return std::find(traits.begin(), traits.end(), traitIndex) != traits.end();
}

bool PlayerCharacter::toggleTag(std::size_t skillIndex) {
    const auto found = std::find(taggedSkills.begin(), taggedSkills.end(), skillIndex);
    if (found != taggedSkills.end()) {
        taggedSkills.erase(found);
        return true;
    }
    if (taggedSkills.size() >= kTagSkillCount) {
        return false;  // full: the UI declines rather than silently swapping one out
    }
    taggedSkills.push_back(skillIndex);
    return true;
}

bool PlayerCharacter::toggleTrait(std::size_t traitIndex) {
    const auto found = std::find(traits.begin(), traits.end(), traitIndex);
    if (found != traits.end()) {
        traits.erase(found);
        return true;
    }
    if (traits.size() >= kMaxTraits) {
        return false;
    }
    traits.push_back(traitIndex);
    return true;
}

int PlayerCharacter::skillValue(
    const CharacterDefinitions& definitions, std::size_t skillIndex
) const {
    if (skillIndex >= definitions.skills.size()) {
        return 0;
    }
    const auto governing = governingAttribute().find(definitions.skills[skillIndex].editorId);
    const int attribute =
        governing != governingAttribute().end() ? special[governing->second] : 5;
    // Fallout's own formula. Luck contributes to every skill, which is why a
    // Luck-heavy character is broadly competent rather than good at one thing.
    const int luck = special[6];
    const int base = 2 + (2 * attribute) + static_cast<int>(std::ceil(luck / 2.0));
    return base + (hasTag(skillIndex) ? 15 : 0);
}

}  // namespace odai::games::newvegas
