#pragma once

#include "import/fnv/content_profile.h"

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
    importer::fnv::BethesdaGame game = importer::fnv::BethesdaGame::Unknown;
    std::string basePlugin;
    std::string worldspace;
    std::string startMarker;
    std::vector<ScenarioQuestRecord> questRecords;
    std::vector<ScenarioQuestSeed> prerequisiteQuests;
};

[[nodiscard]] const ScenarioDefinition& skyrimBleakFallsScenario();
[[nodiscard]] const ScenarioDefinition& skyrimWhiterunShowcaseScenario();
[[nodiscard]] const ScenarioDefinition& skyrimRiftenShowcaseScenario();
[[nodiscard]] const ScenarioDefinition* findScenario(const std::string& id);

}  // namespace odai::bethesda
