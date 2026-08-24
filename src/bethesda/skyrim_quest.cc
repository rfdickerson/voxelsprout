#include "bethesda/skyrim_quest.h"

#include <algorithm>
#include <cstring>
#include <span>

namespace odai::bethesda {
namespace {

std::uint16_t u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8u));
}

std::uint32_t u32(const std::uint8_t* bytes) {
    std::uint32_t value = 0u;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::int32_t i32(const std::uint8_t* bytes) {
    std::int32_t value = 0;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::string zstring(const importer::fnv::EsmSubrecordView& subrecord) {
    std::string value(reinterpret_cast<const char*>(subrecord.data), subrecord.size);
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

void addReference(std::vector<std::uint32_t>& references, std::uint32_t formId) {
    if (formId != 0u) references.push_back(formId);
}

}  // namespace

bool readSkyrimQuest(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimQuestDefinition& out,
    std::string& outError) {
    if (record.type != "QUST") {
        outError = "record is " + record.type + ", not QUST";
        return false;
    }
    if (!stableRecord.valid()) {
        outError = "Skyrim quest requires a stable RecordKey";
        return false;
    }

    SkyrimQuestDefinition parsed;
    parsed.record = std::move(stableRecord);
    SkyrimQuestStageDefinition* currentStage = nullptr;
    SkyrimQuestLogEntryDefinition* currentLogEntry = nullptr;
    SkyrimQuestObjectiveDefinition* currentObjective = nullptr;
    SkyrimQuestAliasDefinition* currentAlias = nullptr;
    Condition* currentCondition = nullptr;
    Condition* currentLogCondition = nullptr;
    for (const importer::fnv::EsmSubrecordView& subrecord : record.subrecords) {
        const std::span<const std::uint8_t> bytes(subrecord.data, subrecord.size);
        if (subrecord.type == "EDID") {
            parsed.editorId = zstring(subrecord);
        } else if (subrecord.type == "DNAM") {
            if (bytes.size() < 4u) {
                outError = "QUST DNAM is shorter than 4 bytes";
                return false;
            }
            parsed.questFlags = u16(bytes.data());
            parsed.priority = bytes[2];
        } else if (subrecord.type == "INDX") {
            if (bytes.size() < 2u) {
                outError = "QUST INDX is shorter than 2 bytes";
                return false;
            }
            const std::uint16_t index = u16(bytes.data());
            const auto existing = std::find_if(parsed.stages.begin(), parsed.stages.end(),
                [&](const SkyrimQuestStageDefinition& stage) { return stage.index == index; });
            if (existing == parsed.stages.end()) {
                SkyrimQuestStageDefinition stage;
                stage.index = index;
                parsed.stages.push_back(std::move(stage));
                currentStage = &parsed.stages.back();
            } else {
                currentStage = &*existing;
            }
            currentObjective = nullptr;
            currentAlias = nullptr;
            currentLogEntry = nullptr;
        } else if (subrecord.type == "QSDT") {
            if (bytes.empty() || currentStage == nullptr) {
                outError = "QUST QSDT appears without a stage or has no flags";
                return false;
            }
            currentStage->logEntryFlags.push_back(bytes[0]);
            currentStage->logEntries.push_back(SkyrimQuestLogEntryDefinition{bytes[0], {}});
            currentLogEntry = &currentStage->logEntries.back();
        } else if (subrecord.type == "QOBJ") {
            if (bytes.size() < 2u) {
                outError = "QUST QOBJ is shorter than 2 bytes";
                return false;
            }
            SkyrimQuestObjectiveDefinition objective;
            objective.index = u16(bytes.data());
            parsed.objectives.push_back(std::move(objective));
            currentObjective = &parsed.objectives.back();
            currentStage = nullptr;
            currentAlias = nullptr;
            currentLogEntry = nullptr;
        } else if (subrecord.type == "NNAM" && currentObjective != nullptr) {
            if (bytes.size() >= 4u) currentObjective->displayTextId = u32(bytes.data());
        } else if (subrecord.type == "ALST" || subrecord.type == "ALLS") {
            if (bytes.size() < 4u) {
                outError = "QUST ALST is shorter than 4 bytes";
                return false;
            }
            parsed.aliases.push_back(SkyrimQuestAliasDefinition{});
            currentAlias = &parsed.aliases.back();
            currentAlias->id = i32(bytes.data());
            currentAlias->location = subrecord.type == "ALLS";
            currentStage = nullptr;
            currentObjective = nullptr;
            currentLogEntry = nullptr;
        } else if (subrecord.type == "ALID" && currentAlias != nullptr) {
            currentAlias->name = zstring(subrecord);
        } else if (subrecord.type == "FNAM" && currentAlias != nullptr) {
            if (bytes.size() < 4u) {
                outError = "QUST alias FNAM is shorter than 4 bytes";
                return false;
            }
            currentAlias->flags = u32(bytes.data());
        } else if (subrecord.type == "ALFL" && currentAlias != nullptr) {
            if (bytes.size() < 4u) {
                outError = "QUST ALFL is shorter than 4 bytes";
                return false;
            }
            if (currentAlias->location) {
                currentAlias->forcedLocationFormId = u32(bytes.data());
                addReference(parsed.referencedFormIds, currentAlias->forcedLocationFormId);
            } else {
                currentAlias->flags = u32(bytes.data());
            }
        } else if (subrecord.type == "ALFR" && currentAlias != nullptr) {
            if (bytes.size() < 4u) {
                outError = "QUST ALFR is shorter than 4 bytes";
                return false;
            }
            currentAlias->forcedReferenceFormId = u32(bytes.data());
            addReference(parsed.referencedFormIds, currentAlias->forcedReferenceFormId);
        } else if (subrecord.type == "ALUA" && currentAlias != nullptr) {
            if (bytes.size() < 4u) {
                outError = "QUST ALUA is shorter than 4 bytes";
                return false;
            }
            currentAlias->uniqueActorFormId = u32(bytes.data());
            addReference(parsed.referencedFormIds, currentAlias->uniqueActorFormId);
        } else if (subrecord.type == "ALFA" && currentAlias != nullptr) {
            if (bytes.size() < 4u) {
                outError = "QUST ALFA is shorter than 4 bytes";
                return false;
            }
            currentAlias->findMatchingReferenceInAliasId = i32(bytes.data());
        } else if (subrecord.type == "ALRT" && currentAlias != nullptr) {
            if (bytes.size() < 4u) {
                outError = "QUST ALRT is shorter than 4 bytes";
                return false;
            }
            currentAlias->referenceTypeFormId = u32(bytes.data());
            addReference(parsed.referencedFormIds, currentAlias->referenceTypeFormId);
        } else if (subrecord.type == "ALCO" && currentAlias != nullptr) {
            if (bytes.size() < 4u) {
                outError = "QUST ALCO is shorter than 4 bytes";
                return false;
            }
            currentAlias->createdObjectFormId = u32(bytes.data());
            addReference(parsed.referencedFormIds, currentAlias->createdObjectFormId);
        } else if (subrecord.type == "ALCA" && currentAlias != nullptr) {
            if (bytes.size() < 4u) {
                outError = "QUST ALCA is shorter than 4 bytes";
                return false;
            }
            currentAlias->createdInAliasId = static_cast<std::int32_t>(
                u32(bytes.data()) & 0x7fffffffu);
        } else if (subrecord.type == "ALCL" && currentAlias != nullptr) {
            if (bytes.size() < 4u) {
                outError = "QUST ALCL is shorter than 4 bytes";
                return false;
            }
            currentAlias->createdLevel = i32(bytes.data());
        } else if (subrecord.type == "CTDA") {
            Condition condition;
            if (!readCondition(bytes, condition, outError)) {
                outError = "malformed QUST CTDA: " + outError;
                return false;
            }
            currentCondition = nullptr;
            currentLogCondition = nullptr;
            if (currentAlias != nullptr) {
                currentAlias->conditions.push_back(condition);
                currentCondition = &currentAlias->conditions.back();
            }
            else if (currentStage != nullptr) {
                currentStage->conditions.push_back(condition);
                currentCondition = &currentStage->conditions.back();
                if (currentLogEntry != nullptr) {
                    currentLogEntry->conditions.push_back(condition);
                    currentLogCondition = &currentLogEntry->conditions.back();
                }
            }
            addReference(parsed.referencedFormIds, condition.parameter1);
            if (condition.function != 629u) {
                addReference(parsed.referencedFormIds, condition.parameter2);
            }
            addReference(parsed.referencedFormIds, condition.reference);
        } else if (subrecord.type == "CIS1") {
            const std::string value = zstring(subrecord);
            if (currentCondition != nullptr) currentCondition->stringParameter1 = value;
            if (currentLogCondition != nullptr) currentLogCondition->stringParameter1 = value;
        } else if (subrecord.type == "CIS2") {
            const std::string value = zstring(subrecord);
            if (currentCondition != nullptr) currentCondition->stringParameter2 = value;
            if (currentLogCondition != nullptr) currentLogCondition->stringParameter2 = value;
        } else if (subrecord.type == "VMAD") {
            VmadQuestAttachments questVmad;
            if (!readVmadQuestAttachments(bytes, questVmad, outError)) {
                outError = "malformed QUST VMAD: " + outError;
                return false;
            }
            parsed.scripts = std::move(questVmad.common);
            parsed.stageFragments = std::move(questVmad.fragments);
            parsed.aliasScripts = std::move(questVmad.aliases);
            for (const VmadScriptAttachment& script : parsed.scripts.scripts) {
                for (const VmadProperty& property : script.properties) {
                    const auto collect = [&](const VmadValue& value, const auto& collectSelf) -> void {
                        if (value.type == VmadValueType::Object) {
                            addReference(parsed.referencedFormIds, value.object.formId);
                        } else if (value.type == VmadValueType::ObjectArray) {
                            for (const VmadValue& element : value.array) collectSelf(element, collectSelf);
                        }
                    };
                    collect(property.value, collect);
                }
            }
        }
    }
    if (parsed.editorId.empty()) {
        outError = "QUST record has no EDID";
        return false;
    }
    std::sort(parsed.stages.begin(), parsed.stages.end(),
        [](const auto& left, const auto& right) { return left.index < right.index; });
    std::sort(parsed.objectives.begin(), parsed.objectives.end(),
        [](const auto& left, const auto& right) { return left.index < right.index; });
    std::sort(parsed.aliases.begin(), parsed.aliases.end(),
        [](const auto& left, const auto& right) { return left.id < right.id; });
    std::sort(parsed.referencedFormIds.begin(), parsed.referencedFormIds.end());
    parsed.referencedFormIds.erase(
        std::unique(parsed.referencedFormIds.begin(), parsed.referencedFormIds.end()),
        parsed.referencedFormIds.end());
    out = std::move(parsed);
    outError.clear();
    return true;
}

}  // namespace odai::bethesda
