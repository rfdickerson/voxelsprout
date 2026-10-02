#pragma once

#include "import/bethesda/content_profile.h"

#include <optional>
#include <string>
#include <vector>

namespace odai::bethesda {

struct ScenarioQuestSeed {
    std::string editorId;
    std::int32_t stage = 0;
    bool completed = false;
};

struct ScenarioQuestRecord {
    std::string editorId;
    std::string plugin;
    std::uint32_t localFormId = 0u;
    bool scriptsRequired = true;
};

struct ScenarioDefinition {
    std::string id;
    importer::bethesda::BethesdaGame game = importer::bethesda::BethesdaGame::Unknown;
    std::string basePlugin;
    std::string worldspace;
    std::string startMarker;
    std::vector<ScenarioQuestRecord> questRecords;
    std::vector<ScenarioQuestSeed> prerequisiteQuests;
    std::uint32_t startDoorFormId = 0u;
    std::uint32_t companionReferenceFormId = 0u;
};

[[nodiscard]] const ScenarioDefinition& skyrimBleakFallsScenario();
[[nodiscard]] const ScenarioDefinition& skyrimHelgenRalofScenario();
[[nodiscard]] const ScenarioDefinition& skyrimWhiterunShowcaseScenario();
[[nodiscard]] const ScenarioDefinition& skyrimRiftenShowcaseScenario();
[[nodiscard]] const ScenarioDefinition* findScenario(const std::string& id);

}  // namespace odai::bethesda
