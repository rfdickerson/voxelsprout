#include "bethesda/skyrim_dialogue.h"

#include <cstring>
#include <span>

namespace odai::bethesda {
namespace {

std::uint32_t u32(const std::uint8_t* bytes) {
    std::uint32_t value = 0u;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::string zstring(const importer::fnv::EsmSubrecordView& subrecord) {
    std::string value(reinterpret_cast<const char*>(subrecord.data), subrecord.size);
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

}  // namespace

bool readSkyrimDialogueBranch(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimDialogueBranchDefinition& out,
    std::string& outError) {
    if (record.type != "DLBR") {
        outError = "record is " + record.type + ", not DLBR";
        return false;
    }
    if (!stableRecord.valid()) {
        outError = "Skyrim dialogue branch requires a stable RecordKey";
        return false;
    }
    SkyrimDialogueBranchDefinition parsed;
    parsed.record = std::move(stableRecord);
    for (const importer::fnv::EsmSubrecordView& subrecord : record.subrecords) {
        if (subrecord.type == "EDID") parsed.editorId = zstring(subrecord);
        else if (subrecord.type == "QNAM" && subrecord.size >= 4u) {
            parsed.rawQuestFormId = u32(subrecord.data);
        } else if (subrecord.type == "SNAM" && subrecord.size >= 4u) {
            parsed.rawStartingTopicFormId = u32(subrecord.data);
        } else if (subrecord.type == "DNAM" && subrecord.size >= 4u) {
            parsed.flags = u32(subrecord.data);
        }
    }
    if (parsed.rawQuestFormId == 0u || parsed.rawStartingTopicFormId == 0u) {
        outError = "Skyrim DLBR has no QNAM quest owner or SNAM starting topic";
        return false;
    }
    out = std::move(parsed);
    outError.clear();
    return true;
}

bool readSkyrimDialogueTopic(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimDialogueTopicDefinition& out,
    std::string& outError) {
    if (record.type != "DIAL") {
        outError = "record is " + record.type + ", not DIAL";
        return false;
    }
    if (!stableRecord.valid()) {
        outError = "Skyrim dialogue topic requires a stable RecordKey";
        return false;
    }
    SkyrimDialogueTopicDefinition parsed;
    parsed.record = std::move(stableRecord);
    for (const importer::fnv::EsmSubrecordView& subrecord : record.subrecords) {
        if (subrecord.type == "EDID") parsed.editorId = zstring(subrecord);
        else if (subrecord.type == "FULL" && subrecord.size >= 4u) {
            parsed.promptStringId = u32(subrecord.data);
        } else if (subrecord.type == "QNAM" && subrecord.size >= 4u) {
            parsed.rawQuestFormId = u32(subrecord.data);
        } else if (subrecord.type == "BNAM" && subrecord.size >= 4u) {
            parsed.rawBranchFormId = u32(subrecord.data);
        } else if (subrecord.type == "DATA" && subrecord.size >= 4u) {
            parsed.flags = u32(subrecord.data);
        }
    }
    if (parsed.rawQuestFormId == 0u) {
        outError = "Skyrim DIAL has no QNAM quest owner";
        return false;
    }
    out = std::move(parsed);
    outError.clear();
    return true;
}

bool readSkyrimDialogueInfo(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    RecordKey stableTopic,
    RecordKey stableQuest,
    SkyrimDialogueInfoDefinition& out,
    std::string& outError) {
    if (record.type != "INFO") {
        outError = "record is " + record.type + ", not INFO";
        return false;
    }
    if (!stableRecord.valid() || !stableTopic.valid() || !stableQuest.valid()) {
        outError = "Skyrim INFO requires stable record, topic, and quest identities";
        return false;
    }
    SkyrimDialogueInfoDefinition parsed;
    parsed.record = std::move(stableRecord);
    parsed.topic = std::move(stableTopic);
    parsed.quest = std::move(stableQuest);
    SkyrimDialogueResponseDefinition* currentResponse = nullptr;
    Condition* currentCondition = nullptr;
    for (const importer::fnv::EsmSubrecordView& subrecord : record.subrecords) {
        const std::span<const std::uint8_t> bytes(subrecord.data, subrecord.size);
        if (subrecord.type == "EDID") parsed.editorId = zstring(subrecord);
        else if (subrecord.type == "RNAM" && bytes.size() >= 4u) {
            parsed.promptStringId = u32(bytes.data());
        }
        else if (subrecord.type == "ENAM" && bytes.size() >= 4u) {
            parsed.flags = u32(bytes.data());
        } else if (subrecord.type == "DNAM" && bytes.size() >= 4u) {
            parsed.rawResponseInfoFormId = u32(bytes.data());
        } else if (subrecord.type == "PNAM" && bytes.size() >= 4u) {
            parsed.rawPreviousInfoFormId = u32(bytes.data());
        } else if (subrecord.type == "TCLT" && bytes.size() >= 4u) {
            parsed.rawLinkedTopicFormIds.push_back(u32(bytes.data()));
        } else if (subrecord.type == "TRDT") {
            parsed.responses.emplace_back();
            currentResponse = &parsed.responses.back();
            if (bytes.size() >= 20u) {
                currentResponse->responseNumber = u32(bytes.data() + 12u);
                currentResponse->soundFormId = u32(bytes.data() + 16u);
            }
        } else if (subrecord.type == "NAM1" && bytes.size() >= 4u) {
            if (currentResponse == nullptr) {
                parsed.responses.emplace_back();
                currentResponse = &parsed.responses.back();
            }
            currentResponse->textStringId = u32(bytes.data());
        } else if (subrecord.type == "CTDA") {
            Condition condition;
            if (!readCondition(bytes, condition, outError)) {
                outError = "malformed INFO CTDA: " + outError;
                return false;
            }
            parsed.conditions.push_back(condition);
            currentCondition = &parsed.conditions.back();
        } else if (subrecord.type == "CIS1" && currentCondition != nullptr) {
            currentCondition->stringParameter1 = zstring(subrecord);
        } else if (subrecord.type == "CIS2" && currentCondition != nullptr) {
            currentCondition->stringParameter2 = zstring(subrecord);
        } else if (subrecord.type == "VMAD") {
            if (!readVmadInfoAttachments(bytes, parsed.scripts, outError)) {
                outError = "malformed INFO VMAD: " + outError;
                return false;
            }
        }
    }
    out = std::move(parsed);
    outError.clear();
    return true;
}

}  // namespace odai::bethesda
