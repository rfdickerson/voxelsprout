#pragma once

#include "bethesda/condition.h"
#include "bethesda/runtime_ids.h"
#include "bethesda/vmad_reader.h"
#include "import/fnv/esm_reader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace odai::bethesda {

struct SkyrimQuestLogEntryDefinition {
    std::uint8_t flags = 0u;
    std::vector<Condition> conditions;
};

struct SkyrimQuestStageDefinition {
    std::uint16_t index = 0u;
    std::vector<std::uint8_t> logEntryFlags;
    std::vector<Condition> conditions;
    std::vector<SkyrimQuestLogEntryDefinition> logEntries;
};

struct SkyrimQuestObjectiveDefinition {
    std::uint16_t index = 0u;
    std::uint32_t displayTextId = 0u;
    // Resolved from the source plugin's localized .STRINGS table by the
    // scenario content loader. The numeric ID remains available to probes.
    std::string displayText;
};

struct SkyrimQuestAliasDefinition {
    std::int32_t id = -1;
    std::string name;
    bool location = false;
    std::uint32_t flags = 0u;
    std::uint32_t forcedReferenceFormId = 0u;
    std::uint32_t uniqueActorFormId = 0u;
    // Location aliases use ALFL for a forced LCTN. Reference aliases use
    // ALFA/ALRT to find one reference of a given LCRT inside that location.
    std::uint32_t forcedLocationFormId = 0u;
    std::int32_t findMatchingReferenceInAliasId = -1;
    std::uint32_t referenceTypeFormId = 0u;
    // ALCO/ALCA/ALCL: create this base object in another reference alias.
    // ALCA's high bit is a CK encoding flag; the low 31 bits name the owner
    // alias used by the runtime inventory materializer.
    std::uint32_t createdObjectFormId = 0u;
    std::int32_t createdInAliasId = -1;
    std::int32_t createdLevel = 0;
    std::vector<Condition> conditions;
};

// Compiled TES5 quest semantics used by the session adapter. Raw FormIDs are
// kept only for dependency discovery; saved identity always uses RecordKey.
struct SkyrimQuestDefinition {
    RecordKey record;
    std::string editorId;
    std::uint16_t questFlags = 0u;
    std::uint8_t priority = 0u;
    std::vector<SkyrimQuestStageDefinition> stages;
    std::vector<SkyrimQuestObjectiveDefinition> objectives;
    std::vector<SkyrimQuestAliasDefinition> aliases;
    VmadAttachments scripts;
    std::vector<VmadQuestFragment> stageFragments;
    std::vector<VmadQuestAliasAttachment> aliasScripts;
    std::vector<std::uint32_t> referencedFormIds;
};

bool readSkyrimQuest(
    const importer::fnv::EsmRecordView& record,
    RecordKey stableRecord,
    SkyrimQuestDefinition& out,
    std::string& outError);

}  // namespace odai::bethesda
