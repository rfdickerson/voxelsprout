#pragma once

#include "bethesda/bethesda_session.h"
#include "bethesda/scenario.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/plugin_load_order.h"

#include <cstddef>
#include <string>
#include <vector>

namespace odai::bethesda {

struct ScenarioQuestLoadDetail {
    std::string editorId;
    std::size_t stages = 0u;
    std::size_t objectives = 0u;
    std::size_t aliases = 0u;
    std::size_t stageFragments = 0u;
    std::size_t aliasScriptAttachments = 0u;
    std::size_t referencedRecords = 0u;
    std::size_t scripts = 0u;
    std::size_t unresolvedCalls = 0u;
    std::vector<std::string> scriptClasses;
};

struct SkyrimScenarioContentReport {
    std::vector<ScenarioQuestLoadDetail> quests;
    std::vector<std::string> transitiveScriptClasses;
    std::size_t transitiveScriptInstances = 0u;
    std::size_t locationsRegistered = 0u;
    std::size_t globalVariablesRegistered = 0u;
    std::size_t dialogueTopicsRegistered = 0u;
    std::size_t dialogueBranchesRegistered = 0u;
    std::size_t dialogueInfosRegistered = 0u;
    std::size_t dialogueFragmentsLoaded = 0u;
    std::vector<std::string> runtimeBlockers;
    std::vector<std::string> diagnostics;
    std::vector<std::string> unresolvedCallBindings;
};

// Loads the retail QUST -> VMAD -> PEX closure into an already configured
// Skyrim session. This is shared by the graphical runtime and the headless
// compatibility probe so they cannot disagree about scenario readiness.
bool loadSkyrimScenarioContent(
    const ScenarioDefinition& scenario,
    const importer::fnv::FalloutLoadOrder& loadOrder,
    const importer::fnv::FalloutAssetSource& assets,
    BethesdaSession& session,
    SkyrimScenarioContentReport& outReport,
    std::string& outError);

}  // namespace odai::bethesda
