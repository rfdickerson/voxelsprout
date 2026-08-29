#include "bethesda/tes3_content.h"

#include "import/fnv/esm_reader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstring>
#include <set>
#include <sstream>
#include <unordered_map>

namespace odai::bethesda {
namespace {

using importer::fnv::EsmReader;
using importer::fnv::EsmRecordView;
using importer::fnv::EsmSubrecordView;
using importer::fnv::FalloutLoadOrder;
using importer::fnv::FalloutLoadOrderEntry;

std::string lowerAscii(std::string value) {
    for (char& ch : value) {
        if (static_cast<unsigned char>(ch) < 0x80u) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }
    return value;
}

std::string rawString(const EsmSubrecordView& sub) {
    if (sub.data == nullptr || sub.size == 0u) return {};
    std::string value(reinterpret_cast<const char*>(sub.data), sub.size);
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

std::uint32_t readU32(const std::uint8_t* bytes) {
    std::uint32_t value = 0u;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::int32_t readI32(const std::uint8_t* bytes) {
    std::int32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::int16_t readI16(const std::uint8_t* bytes) {
    std::int16_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

float readF32(const std::uint8_t* bytes) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

void appendUtf8(std::string& out, std::uint32_t codepoint) {
    if (codepoint <= 0x7fu) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffu) {
        out.push_back(static_cast<char>(0xc0u | (codepoint >> 6u)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    } else {
        out.push_back(static_cast<char>(0xe0u | (codepoint >> 12u)));
        out.push_back(static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    }
}

Tes3SubrecordData copySubrecord(const EsmSubrecordView& sub) {
    Tes3SubrecordData result;
    result.type = sub.type;
    if (sub.data != nullptr && sub.size != 0u) {
        result.data.assign(sub.data, sub.data + sub.size);
    }
    return result;
}

std::string decodedDataString(
    const Tes3SubrecordData& subrecord, std::string_view encoding) {
    std::string bytes(reinterpret_cast<const char*>(subrecord.data.data()),
                      subrecord.data.size());
    while (!bytes.empty() && bytes.back() == '\0') bytes.pop_back();
    return decodeTes3Text(bytes, encoding);
}

Tes3ActorDefinition parseActorDefinition(
    const Tes3NamedRecord& record, std::string_view encoding) {
    constexpr std::array<std::string_view, 8u> attributeNames = {
        "strength", "intelligence", "willpower", "agility",
        "speed", "endurance", "personality", "luck"};
    constexpr std::array<std::string_view, 27u> skillNames = {
        "block", "armorer", "mediumarmor", "heavyarmor", "bluntweapon",
        "longblade", "axe", "spear", "athletics", "enchant", "destruction",
        "alteration", "illusion", "conjuration", "mysticism", "restoration",
        "alchemy", "unarmored", "security", "sneak", "acrobatics",
        "lightarmor", "shortblade", "marksman", "mercantile", "speechcraft",
        "handtohand"};
    Tes3ActorDefinition result;
    result.record = record.record;
    result.id = record.id;
    result.creature = record.record.recordType == "CREA";
    result.sourcePlugin = record.sourcePlugin;
    for (const Tes3SubrecordData& sub : record.subrecords) {
        if (sub.type == "FNAM") result.name = decodedDataString(sub, encoding);
        else if (sub.type == "RNAM") result.race = decodedDataString(sub, encoding);
        else if (sub.type == "CNAM") result.actorClass = decodedDataString(sub, encoding);
        else if (sub.type == "ANAM") {
            const std::string faction = decodedDataString(sub, encoding);
            if (!faction.empty()) result.faction = makeTes3RecordKey("FACT", faction);
        } else if (sub.type == "SCRI") {
            const std::string script = decodedDataString(sub, encoding);
            if (!script.empty()) result.script = makeTes3RecordKey("SCPT", script);
        } else if (sub.type == "FLAG" && sub.data.size() >= 4u) {
            result.autoCalculate = (readU32(sub.data.data()) & 0x10u) != 0u;
        } else if (sub.type == "NPDT") {
            if (!result.creature && sub.data.size() >= 52u) {
                result.level = readI16(sub.data.data());
                for (std::size_t i = 0u; i < attributeNames.size(); ++i) {
                    result.attributes.emplace(std::string(attributeNames[i]), sub.data[2u + i]);
                }
                for (std::size_t i = 0u; i < skillNames.size(); ++i) {
                    result.skills.emplace(std::string(skillNames[i]), sub.data[10u + i]);
                }
                result.health = readI16(sub.data.data() + 38u);
                result.magicka = readI16(sub.data.data() + 40u);
                result.fatigue = readI16(sub.data.data() + 42u);
                result.rank = static_cast<std::int8_t>(sub.data[46u]);
            } else if (!result.creature && sub.data.size() >= 12u) {
                result.level = readI16(sub.data.data());
                result.rank = static_cast<std::int8_t>(sub.data[4u]);
                result.autoCalculate = true;
            } else if (result.creature && sub.data.size() >= 96u) {
                result.level = readI32(sub.data.data() + 4u);
                result.health = static_cast<float>(readI32(sub.data.data() + 40u));
                result.magicka = static_cast<float>(readI32(sub.data.data() + 44u));
                result.fatigue = static_cast<float>(readI32(sub.data.data() + 48u));
            }
        } else if (sub.type == "NPCO" && sub.data.size() >= 36u) {
            const std::int32_t count = readI32(sub.data.data());
            std::string bytes(reinterpret_cast<const char*>(sub.data.data() + 4u), 32u);
            const std::size_t nul = bytes.find('\0');
            if (nul != std::string::npos) bytes.resize(nul);
            const std::string item = decodeTes3Text(bytes, encoding);
            if (!item.empty() && count != 0) {
                result.inventory.emplace_back(makeTes3RecordKey("REFR", item), count);
            }
        } else if (sub.type == "AIDT" && sub.data.size() >= 12u) {
            result.serviceFlags = readU32(sub.data.data() + 8u);
        } else if (sub.type == "DODT" && sub.data.size() >= 24u) {
            Tes3ActorDefinition::TravelDestination destination;
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                destination.position[axis] = readF32(sub.data.data() + (axis * 4u));
                destination.rotationRadians[axis] =
                    readF32(sub.data.data() + 12u + (axis * 4u));
            }
            result.travelDestinations.push_back(std::move(destination));
        } else if (sub.type == "DNAM" && !result.travelDestinations.empty()) {
            result.travelDestinations.back().cell = decodedDataString(sub, encoding);
        }
    }
    return result;
}

Tes3SpellDefinition parseSpellDefinition(
    const Tes3NamedRecord& record, std::string_view encoding) {
    Tes3SpellDefinition result;
    result.record = record.record;
    result.id = record.id;
    result.sourcePlugin = record.sourcePlugin;
    for (const Tes3SubrecordData& sub : record.subrecords) {
        if (sub.type == "FNAM") result.name = decodedDataString(sub, encoding);
        else if (sub.type == "SPDT" && sub.data.size() >= 12u) {
            result.type = readI32(sub.data.data());
            result.cost = readI32(sub.data.data() + 4u);
            result.flags = readI32(sub.data.data() + 8u);
        } else if (sub.type == "ENAM" && sub.data.size() >= 24u) {
            Tes3SpellEffect effect;
            effect.effectId = readI16(sub.data.data());
            effect.skill = static_cast<std::int8_t>(sub.data[2u]);
            effect.attribute = static_cast<std::int8_t>(sub.data[3u]);
            effect.range = readI32(sub.data.data() + 4u);
            effect.area = readI32(sub.data.data() + 8u);
            effect.duration = readI32(sub.data.data() + 12u);
            effect.magnitudeMin = readI32(sub.data.data() + 16u);
            effect.magnitudeMax = readI32(sub.data.data() + 20u);
            result.effects.push_back(effect);
        }
    }
    return result;
}

bool hasDeletion(const EsmRecordView& record) {
    for (const EsmSubrecordView& sub : record.subrecords) {
        // A CELL owns an embedded sequence of references. DELE after the first
        // FRMR deletes that reference, not the cell record itself.
        if (record.type == "CELL" && sub.type == "FRMR") return false;
        if (sub.type == "DELE") return true;
    }
    return false;
}

std::string decodedSubrecord(const EsmSubrecordView& sub, std::string_view encoding) {
    return decodeTes3Text(rawString(sub), encoding);
}

std::string firstString(
    const EsmRecordView& record, std::string_view type, std::string_view encoding) {
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == type) return decodedSubrecord(sub, encoding);
    }
    return {};
}

std::string scriptId(const EsmRecordView& record, std::string_view encoding) {
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type != "SCHD" || sub.size == 0u) continue;
        const std::size_t size = std::min<std::size_t>(32u, sub.size);
        std::string id(reinterpret_cast<const char*>(sub.data), size);
        const std::size_t nul = id.find('\0');
        if (nul != std::string::npos) id.resize(nul);
        return decodeTes3Text(id, encoding);
    }
    return {};
}

std::string cellId(const EsmRecordView& record, std::string_view encoding) {
    std::string name;
    bool interior = false;
    std::int32_t x = 0;
    std::int32_t y = 0;
    bool hasData = false;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "FRMR") break;
        if (sub.type == "NAME") name = decodedSubrecord(sub, encoding);
        else if (sub.type == "DATA" && sub.size >= 12u) {
            interior = (readU32(sub.data) & 0x1u) != 0u;
            x = readI32(sub.data + 4u);
            y = readI32(sub.data + 8u);
            hasData = true;
        }
    }
    if (interior || !name.empty()) return name;
    if (!hasData) return {};
    return "#" + std::to_string(x) + "," + std::to_string(y);
}

bool cellIsInterior(const EsmRecordView& record) {
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "FRMR") break;
        if (sub.type == "DATA" && sub.size >= 4u) {
            return (readU32(sub.data) & 0x1u) != 0u;
        }
    }
    return false;
}

bool cellGrid(
    const EsmRecordView& record, std::int32_t& outX, std::int32_t& outZ) {
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "FRMR") break;
        if (sub.type == "DATA" && sub.size >= 12u) {
            if ((readU32(sub.data) & 0x1u) != 0u) return false;
            outX = readI32(sub.data + 4u);
            outZ = readI32(sub.data + 8u);
            return true;
        }
    }
    return false;
}

std::string recordId(const EsmRecordView& record, std::string_view encoding) {
    if (record.type == "SCPT") return scriptId(record, encoding);
    if (record.type == "CELL") return cellId(record, encoding);
    if (record.type == "LAND") {
        for (const EsmSubrecordView& sub : record.subrecords) {
            if (sub.type == "INTV" && sub.size >= 8u) {
                return "#" + std::to_string(readI32(sub.data)) + "," +
                    std::to_string(readI32(sub.data + 4u));
            }
        }
        return {};
    }
    return firstString(record, "NAME", encoding);
}

Tes3DialogueCondition parseCondition(
    const std::string& rule, const EsmSubrecordView* valueSubrecord) {
    Tes3DialogueCondition result;
    result.rawRule = rule;
    if (rule.size() < 5u || valueSubrecord == nullptr ||
        (valueSubrecord->type != "INTV" && valueSubrecord->type != "FLTV") ||
        valueSubrecord->size < 4u || rule[4] < '0' || rule[4] > '5') {
        return result;
    }
    result.index = rule[0] >= '0' && rule[0] <= '9'
        ? static_cast<std::uint8_t>(rule[0] - '0') : 0u;
    if (rule[1] == '1') {
        int function = -1;
        const auto parsed = std::from_chars(rule.data() + 2u, rule.data() + 4u, function);
        if (parsed.ec != std::errc{} || function < 0 || function > 73) return result;
        result.function = static_cast<Tes3ConditionFunction>(function);
    } else {
        switch (rule[1]) {
            case '2': result.function = Tes3ConditionFunction::Global; break;
            case '3': result.function = Tes3ConditionFunction::Local; break;
            case '4': result.function = Tes3ConditionFunction::Journal; break;
            case '5': result.function = Tes3ConditionFunction::Item; break;
            case '6': result.function = Tes3ConditionFunction::Dead; break;
            case '7': result.function = Tes3ConditionFunction::NotId; break;
            case '8': result.function = Tes3ConditionFunction::NotFaction; break;
            case '9': result.function = Tes3ConditionFunction::NotClass; break;
            case 'A': result.function = Tes3ConditionFunction::NotRace; break;
            case 'B': result.function = Tes3ConditionFunction::NotCell; break;
            case 'C': result.function = Tes3ConditionFunction::NotLocal; break;
            default: return result;
        }
    }
    result.comparison = rule[4];
    result.variable = rule.substr(5u);
    if (valueSubrecord->type == "FLTV") result.value = readF32(valueSubrecord->data);
    else result.value = readI32(valueSubrecord->data);
    result.valid = true;
    return result;
}

struct TopicWork {
    Tes3DialogueDefinition definition;
    std::map<std::string, Tes3DialogueInfo> infos;
    std::vector<std::string> insertionOrder;
};

Tes3DialogueInfo parseInfo(
    const EsmRecordView& record, std::string_view encoding,
    const std::string& plugin, std::uint64_t ordinal) {
    Tes3DialogueInfo info;
    info.sourcePlugin = plugin;
    info.sourceOrdinal = ordinal;
    for (std::size_t i = 0u; i < record.subrecords.size(); ++i) {
        const EsmSubrecordView& sub = record.subrecords[i];
        if (sub.type == "INAM") info.id = decodedSubrecord(sub, encoding);
        else if (sub.type == "PNAM") info.previousId = decodedSubrecord(sub, encoding);
        else if (sub.type == "NNAM") info.nextId = decodedSubrecord(sub, encoding);
        else if (sub.type == "DATA" && sub.size >= 12u) {
            info.dispositionOrJournalIndex = readI32(sub.data + 4u);
            info.rank = static_cast<std::int8_t>(sub.data[8u]);
            info.gender = static_cast<std::int8_t>(sub.data[9u]);
            info.playerRank = static_cast<std::int8_t>(sub.data[10u]);
        } else if (sub.type == "ONAM") info.actor = decodedSubrecord(sub, encoding);
        else if (sub.type == "RNAM") info.race = decodedSubrecord(sub, encoding);
        else if (sub.type == "CNAM") info.actorClass = decodedSubrecord(sub, encoding);
        else if (sub.type == "FNAM") {
            info.faction = decodedSubrecord(sub, encoding);
            info.factionless = lowerAscii(info.faction) == "ffff";
        } else if (sub.type == "ANAM") info.cell = decodedSubrecord(sub, encoding);
        else if (sub.type == "DNAM") info.playerFaction = decodedSubrecord(sub, encoding);
        else if (sub.type == "SNAM") info.sound = decodedSubrecord(sub, encoding);
        else if (sub.type == "NAME") info.response = decodedSubrecord(sub, encoding);
        else if (sub.type == "BNAM") info.resultScript = decodedSubrecord(sub, encoding);
        else if (sub.type == "QSTN") info.questStatus = Tes3QuestStatus::Name;
        else if (sub.type == "QSTF") info.questStatus = Tes3QuestStatus::Finished;
        else if (sub.type == "QSTR") info.questStatus = Tes3QuestStatus::Restart;
        else if (sub.type == "SCVR") {
            const std::string rule = rawString(sub);
            const EsmSubrecordView* value = i + 1u < record.subrecords.size()
                ? &record.subrecords[i + 1u] : nullptr;
            info.conditions.push_back(parseCondition(rule, value));
        }
    }
    info.record = makeTes3RecordKey("INFO", info.id);
    return info;
}

Tes3ScriptDefinition parseScript(
    const EsmRecordView& record, std::string_view encoding, const std::string& plugin) {
    Tes3ScriptDefinition result;
    result.id = scriptId(record, encoding);
    result.record = makeTes3RecordKey("SCPT", result.id);
    result.sourcePlugin = plugin;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "SCHD" && sub.size >= 44u) {
            result.shortCount = readU32(sub.data + 32u);
            result.longCount = readU32(sub.data + 36u);
            result.floatCount = readU32(sub.data + 40u);
        } else if (sub.type == "SCVR") {
            std::string names = decodeTes3Text(
                std::string_view(reinterpret_cast<const char*>(sub.data), sub.size), encoding);
            std::size_t begin = 0u;
            while (begin < names.size()) {
                const std::size_t end = names.find('\0', begin);
                const std::string name = names.substr(begin, end == std::string::npos
                    ? std::string::npos : end - begin);
                if (!name.empty()) result.variableNames.push_back(name);
                if (end == std::string::npos) break;
                begin = end + 1u;
            }
        } else if (sub.type == "SCDT" && sub.data != nullptr) {
            result.bytecode.assign(sub.data, sub.data + sub.size);
        } else if (sub.type == "SCTX") {
            result.source = decodedSubrecord(sub, encoding);
        }
    }
    return result;
}

Tes3GlobalDefinition parseGlobal(
    const EsmRecordView& record, std::string_view encoding, const std::string& plugin) {
    Tes3GlobalDefinition result;
    result.id = firstString(record, "NAME", encoding);
    result.record = makeTes3RecordKey("GLOB", result.id);
    result.sourcePlugin = plugin;
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "FNAM" && sub.size >= 1u) result.valueType = static_cast<char>(sub.data[0]);
        else if (sub.type == "FLTV" && sub.size >= 4u) result.value = readF32(sub.data);
    }
    return result;
}

RecordKey referenceOwnerKey(
    const FalloutLoadOrder& order, std::size_t pluginIndex, std::uint32_t rawFrmr) {
    const FalloutLoadOrderEntry& source = order.entries()[pluginIndex];
    const std::uint8_t localIndex = static_cast<std::uint8_t>(rawFrmr >> 24u);
    std::string owner = source.header.fileName;
    if (localIndex > 0u && static_cast<std::size_t>(localIndex - 1u) < source.header.masters.size()) {
        owner = source.header.masters[localIndex - 1u];
    }
    return makeTes3ReferenceKey(owner, rawFrmr & 0x00ffffffu);
}

void parseCellReferences(
    const EsmRecordView& record, const RecordKey& cell, const FalloutLoadOrder& order,
    bool interior, std::size_t pluginIndex, std::string_view encoding, const std::string& plugin,
    std::map<ObjectId, Tes3ReferenceDefinition>& references, Tes3ContentStats& stats) {
    std::int32_t cellGridX = 0;
    std::int32_t cellGridZ = 0;
    const bool hasCellGrid = cellGrid(record, cellGridX, cellGridZ);
    std::optional<Tes3ReferenceDefinition> current;
    const auto flush = [&]() {
        if (!current.has_value() || !current->id.valid()) return;
        if (current->deleted) {
            references.erase(current->id);
            ++stats.deletions;
        } else {
            references.insert_or_assign(current->id, std::move(*current));
            ++stats.references;
        }
        current.reset();
    };
    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "FRMR") {
            flush();
            if (sub.size < 4u) continue;
            current.emplace();
            current->id = ObjectId::persistent(referenceOwnerKey(order, pluginIndex, readU32(sub.data)));
            current->cell = cell;
            current->interior = interior;
            current->hasCellGrid = hasCellGrid;
            current->cellGridX = cellGridX;
            current->cellGridZ = cellGridZ;
            current->sourcePlugin = plugin;
            current->subrecords.push_back(copySubrecord(sub));
            continue;
        }
        if (!current.has_value()) continue;
        current->subrecords.push_back(copySubrecord(sub));
        if (sub.type == "NAME") {
            current->baseId = decodedSubrecord(sub, encoding);
            current->base = makeTes3RecordKey("REFR", current->baseId);
        } else if (sub.type == "DATA" && sub.size >= 24u) {
            for (std::size_t i = 0u; i < 3u; ++i) {
                current->position[i] = readF32(sub.data + i * 4u);
                current->rotationRadians[i] = readF32(sub.data + 12u + i * 4u);
            }
            current->hasTransform = true;
        } else if (sub.type == "XSCL" && sub.size >= 4u) {
            current->scale = readF32(sub.data);
        } else if (sub.type == "FLTV" && sub.size >= 4u) {
            current->lockLevel = readI32(sub.data);
        } else if (sub.type == "DELE") {
            current->deleted = true;
            current->enabled = false;
        }
    }
    flush();
}

void orderInfos(TopicWork& topic) {
    topic.definition.infos.clear();
    std::set<std::string> emitted;
    const auto emitChain = [&](const std::string& start) {
        std::string id = start;
        while (!id.empty() && !emitted.contains(id)) {
            const auto found = topic.infos.find(id);
            if (found == topic.infos.end()) break;
            topic.definition.infos.push_back(found->second);
            emitted.insert(id);
            id = lowerAscii(found->second.nextId);
        }
    };
    for (const std::string& id : topic.insertionOrder) {
        const auto found = topic.infos.find(id);
        if (found == topic.infos.end()) continue;
        const std::string previous = lowerAscii(found->second.previousId);
        if (previous.empty() || topic.infos.find(previous) == topic.infos.end()) emitChain(id);
    }
    for (const std::string& id : topic.insertionOrder) emitChain(id);
}

}  // namespace

std::string decodeTes3Text(std::string_view bytes, std::string_view encoding) {
    static constexpr std::array<std::uint16_t, 32> windows1252 = {
        0x20acu, 0x0081u, 0x201au, 0x0192u, 0x201eu, 0x2026u, 0x2020u, 0x2021u,
        0x02c6u, 0x2030u, 0x0160u, 0x2039u, 0x0152u, 0x008du, 0x017du, 0x008fu,
        0x0090u, 0x2018u, 0x2019u, 0x201cu, 0x201du, 0x2022u, 0x2013u, 0x2014u,
        0x02dcu, 0x2122u, 0x0161u, 0x203au, 0x0153u, 0x009du, 0x017eu, 0x0178u};
    const std::string normalized = lowerAscii(std::string(encoding));
    const bool western = normalized.empty() || normalized == "win1252" ||
        normalized == "windows-1252" || normalized == "cp1252";
    if (!western) return std::string(bytes);
    std::string out;
    out.reserve(bytes.size());
    for (const unsigned char byte : bytes) {
        std::uint32_t codepoint = byte;
        if (byte >= 0x80u && byte <= 0x9fu) codepoint = windows1252[byte - 0x80u];
        appendUtf8(out, codepoint);
    }
    return out;
}

bool Tes3ContentStore::load(
    const FalloutLoadOrder& order, std::string encoding, std::string& outError) {
    outError.clear();
    m_encoding = encoding.empty() ? "windows-1252" : lowerAscii(std::move(encoding));
    m_dialogues.clear();
    m_scripts.clear();
    m_globals.clear();
    m_actors.clear();
    m_spells.clear();
    m_namedRecords.clear();
    m_references.clear();
    m_stats = {};
    if (order.empty()) {
        outError = "TES3 content store requires a non-empty load order";
        return false;
    }
    for (const FalloutLoadOrderEntry& entry : order.entries()) {
        if (entry.header.format != importer::fnv::EsmPluginFormat::kMorrowind) {
            outError = "TES3 content store cannot load non-Morrowind plugin " + entry.header.fileName;
            return false;
        }
    }

    std::map<RecordKey, TopicWork> topics;
    std::uint64_t ordinal = 0u;
    for (std::size_t pluginIndex = 0u; pluginIndex < order.entries().size(); ++pluginIndex) {
        const FalloutLoadOrderEntry& entry = order.entries()[pluginIndex];
        EsmReader reader;
        if (!reader.open(entry.path)) {
            outError = "cannot open TES3 plugin " + entry.path.string() + ": " + reader.lastError();
            return false;
        }
        std::optional<RecordKey> currentDialogue;
        EsmReader::Visitor visitor;
        visitor.onRecordHeader = [](const importer::fnv::EsmRecordHeaderView&) { return true; };
        visitor.onRecord = [&](const EsmRecordView& record) {
            ++m_stats.recordsRead;
            ++ordinal;
            if (record.type == "TES3") return;
            if (record.type == "DIAL") {
                const std::string id = firstString(record, "NAME", m_encoding);
                if (id.empty()) { currentDialogue.reset(); return; }
                const RecordKey key = makeTes3RecordKey("DIAL", id);
                currentDialogue = key;
                if (hasDeletion(record)) {
                    topics.erase(key);
                    ++m_stats.deletions;
                    return;
                }
                TopicWork& topic = topics[key];
                topic.definition.record = key;
                topic.definition.id = id;
                topic.definition.sourcePlugin = entry.header.fileName;
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "DATA" && sub.size == 1u) {
                        topic.definition.type = static_cast<Tes3DialogueType>(
                            static_cast<std::int8_t>(sub.data[0]));
                    }
                }
                return;
            }
            if (record.type == "INFO") {
                if (!currentDialogue.has_value()) return;
                auto topic = topics.find(*currentDialogue);
                if (topic == topics.end()) return;
                Tes3DialogueInfo info = parseInfo(
                    record, m_encoding, entry.header.fileName, ordinal);
                if (info.id.empty()) return;
                const std::string normalizedId = lowerAscii(info.id);
                if (hasDeletion(record)) {
                    topic->second.infos.erase(normalizedId);
                    ++m_stats.deletions;
                } else {
                    if (!topic->second.infos.contains(normalizedId)) {
                        topic->second.insertionOrder.push_back(normalizedId);
                    }
                    topic->second.infos.insert_or_assign(normalizedId, std::move(info));
                    ++m_stats.infos;
                }
                return;
            }

            const std::string id = recordId(record, m_encoding);
            if (id.empty()) return;
            const RecordKey key = makeTes3RecordKey(record.type, id);
            if (hasDeletion(record)) {
                m_namedRecords.erase(key);
                if (record.type == "SCPT") m_scripts.erase(key);
                if (record.type == "GLOB") m_globals.erase(key);
                ++m_stats.deletions;
                return;
            }
            Tes3NamedRecord named;
            named.record = key;
            named.id = id;
            named.sourcePlugin = entry.header.fileName;
            named.subrecords.reserve(record.subrecords.size());
            for (const EsmSubrecordView& sub : record.subrecords) {
                named.subrecords.push_back(copySubrecord(sub));
            }
            m_namedRecords.insert_or_assign(key, std::move(named));
            ++m_stats.namedRecords;
            if (record.type == "SCPT") {
                m_scripts.insert_or_assign(key, parseScript(record, m_encoding, entry.header.fileName));
                ++m_stats.scripts;
            } else if (record.type == "GLOB") {
                m_globals.insert_or_assign(key, parseGlobal(record, m_encoding, entry.header.fileName));
                ++m_stats.globals;
            } else if (record.type == "CELL") {
                parseCellReferences(record, key, order, cellIsInterior(record), pluginIndex, m_encoding,
                    entry.header.fileName, m_references, m_stats);
            }
        };
        if (!reader.walk(visitor)) {
            outError = "failed while walking TES3 plugin " + entry.path.string() +
                ": " + reader.lastError();
            return false;
        }
    }

    for (auto& [key, topic] : topics) {
        orderInfos(topic);
        m_dialogues.insert_or_assign(key, std::move(topic.definition));
    }
    m_stats.dialogues = m_dialogues.size();

    // Resolve each reference's string base ID to the later-wins named record.
    // A malformed duplicate string ID across types remains an explicit REFR
    // pseudo-type instead of silently choosing the wrong gameplay definition.
    std::map<std::string, RecordKey> uniqueBase;
    std::set<std::string> ambiguousBase;
    for (const auto& [key, record] : m_namedRecords) {
        (void)record;
        if (key.recordType == "CELL" || key.recordType == "LAND" || key.recordType == "PGRD") continue;
        const auto [it, inserted] = uniqueBase.emplace(key.textId, key);
        if (!inserted && it->second != key) ambiguousBase.insert(key.textId);
    }
    for (auto& [id, reference] : m_references) {
        (void)id;
        const std::string normalized = makeTes3RecordKey("REFR", reference.baseId).textId;
        const auto found = uniqueBase.find(normalized);
        if (found != uniqueBase.end() && !ambiguousBase.contains(normalized)) reference.base = found->second;
    }
    for (const auto& [key, record] : m_namedRecords) {
        if (key.recordType == "NPC_" || key.recordType == "CREA") {
            Tes3ActorDefinition actor = parseActorDefinition(record, m_encoding);
            for (auto& [item, count] : actor.inventory) {
                (void)count;
                const auto found = uniqueBase.find(item.textId);
                if (found != uniqueBase.end() && !ambiguousBase.contains(item.textId)) {
                    item = found->second;
                }
            }
            m_actors.insert_or_assign(key, std::move(actor));
        } else if (key.recordType == "SPEL") {
            m_spells.insert_or_assign(key, parseSpellDefinition(record, m_encoding));
        }
    }
    return true;
}

const Tes3DialogueDefinition* Tes3ContentStore::findDialogue(std::string_view id) const {
    const auto found = m_dialogues.find(makeTes3RecordKey("DIAL", std::string(id)));
    return found == m_dialogues.end() ? nullptr : &found->second;
}

const Tes3ScriptDefinition* Tes3ContentStore::findScript(std::string_view id) const {
    const auto found = m_scripts.find(makeTes3RecordKey("SCPT", std::string(id)));
    return found == m_scripts.end() ? nullptr : &found->second;
}

const Tes3ActorDefinition* Tes3ContentStore::findActor(
    std::string_view type, std::string_view id) const {
    const auto found = m_actors.find(makeTes3RecordKey(std::string(type), std::string(id)));
    return found == m_actors.end() ? nullptr : &found->second;
}

const Tes3SpellDefinition* Tes3ContentStore::findSpell(std::string_view id) const {
    const auto found = m_spells.find(makeTes3RecordKey("SPEL", std::string(id)));
    return found == m_spells.end() ? nullptr : &found->second;
}

const Tes3NamedRecord* Tes3ContentStore::findRecord(
    std::string_view type, std::string_view id) const {
    const auto found = m_namedRecords.find(makeTes3RecordKey(std::string(type), std::string(id)));
    return found == m_namedRecords.end() ? nullptr : &found->second;
}

}  // namespace odai::bethesda
