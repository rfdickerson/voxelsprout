#pragma once

#include "bethesda/condition.h"
#include "bethesda/runtime_ids.h"
#include "bethesda/vmad_reader.h"
#include "import/fnv/esm_reader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace odai::bethesda {

struct SkyrimDialogueResponseDefinition {
    std::uint32_t textStringId = 0u;
    std::string text;
    // TRDT is kept as opaque authored metadata until lip/emotion playback
    // consumes all of it; the response number and sound form are stable fields.
    std::uint32_t responseNumber = 0u;
    std::uint32_t soundFormId = 0u;
};

struct SkyrimDialogueTopicDefinition {
    RecordKey record;
    std::string editorId;
    std::uint32_t promptStringId = 0u;
    std::string prompt;
    std::uint32_t rawQuestFormId = 0u;
    std::uint32_t rawBranchFormId = 0u;
    std::uint32_t flags = 0u;
    RecordKey quest;
    RecordKey branch;
};

struct SkyrimDialogueBranchDefinition {
    RecordKey record;
    RecordKey quest;
    RecordKey startingTopic;
    std::string editorId;
    std::uint32_t rawQuestFormId = 0u;
    std::uint32_t rawStartingTopicFormId = 0u;
    std::uint32_t flags = 0u;
};

struct SkyrimDialogueInfoDefinition {
    RecordKey record;
    RecordKey topic;
    RecordKey quest;
    std::string editorId;
    std::uint32_t promptStringId = 0u;  // RNAM player-line override
    std::string prompt;
    std::uint64_t authoredOrder = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t rawResponseInfoFormId = 0u;  // DNAM
    std::uint32_t rawPreviousInfoFormId = 0u;  // PNAM
    std::vector<std::uint32_t> rawLinkedTopicFormIds;
    // Resolved by the load-order adapter after parsing. These are the only
    // identities consumed by the runtime and remain stable across slot moves.
    RecordKey responseInfo;
    RecordKey previousInfo;
    std::vector<RecordKey> linkedTopics;
    std::vector<SkyrimDialogueResponseDefinition> responses;
    std::vector<Condition> conditions;
    VmadInfoAttachments scripts;
};

bool readSkyrimDialogueBranch(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimDialogueBranchDefinition& out,
    std::string& outError);

bool readSkyrimDialogueTopic(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimDialogueTopicDefinition& out,
    std::string& outError);

bool readSkyrimDialogueInfo(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    RecordKey stableTopic,
    RecordKey stableQuest,
    SkyrimDialogueInfoDefinition& out,
    std::string& outError);

}  // namespace odai::bethesda
