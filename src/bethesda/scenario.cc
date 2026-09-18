#include "bethesda/scenario.h"

namespace odai::bethesda {

const ScenarioDefinition& skyrimBleakFallsScenario() {
    // The post-Helgen start completes MQ101 and starts MQ102 at its authored
    // Riverwood entry stage. Golden Claw, the Whiterun handoff, and Bleak Falls
    // remain untouched and must advance through retail records/scripts.
    static const ScenarioDefinition scenario{
        "skyrim-bleak-falls",
        importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "Skyrim.esm",
        "Tamriel",
        "Riverwood",
        {{"MQ101", "Skyrim.esm", 0x0003372bu, false},
         {"MQ102", "Skyrim.esm", 0x0004e50du},
         // MQ102 delegates the actual Riverwood arrival/help conversations to
         // the mutually-exclusive Hadvar and Ralof branch quests. They remain
         // unstarted until the scenario's post-Helgen branch is selected, but
         // both definitions belong to the content closure so their authored
         // DIAL/INFO records and VMAD dependencies can be diagnosed.
         {"MQ102A", "Skyrim.esm", 0x0002bf9cu},
         {"MQ102B", "Skyrim.esm", 0x0002610au},
         {"MS13", "Skyrim.esm", 0x00039645u},
         {"MQ103", "Skyrim.esm", 0x000d0800u}},
        {// Branch selection is part of the post-Helgen bootstrap, not later
         // quest progress. ODAI deliberately chooses the Imperial/Hadvar
         // route and replays the retail startup topology in authored order;
         // MQ102B remains at 0 and no Riverwood conversation stage is faked.
         {"MQ101", 900, true},
         {"MQ102", 10, false},
         {"MQ102A", 0, false},
         {"MQ102A", 1, false},
         {"MQ102A", 5, false},
         {"MQ102A", 10, false},
         {"MQ102A", 20, false}},
    };
    return scenario;
}

const ScenarioDefinition& skyrimHelgenRalofScenario() {
    // Keep the versioned Hadvar route intact. This is the Stormcloak branch
    // at the authored cave-exit marker, before the introductory conversation.
    static const ScenarioDefinition scenario = [] {
        auto value = skyrimBleakFallsScenario();
        value.id = "skyrim-helgen-ralof";
        value.startMarker.clear();
        value.startDoorFormId = 0x00091608u; // HelgenKeep01toExteriorExitREF.XTEL
        value.companionReferenceFormId = 0x0002bf9eu; // RalofRef
        // Stage 0 is a test-start teleport back inside the keep. Stage 20
        // belongs to the introductory dialogue, not the departure bootstrap.
        value.prerequisiteQuests = {{"MQ101", 900, true}, {"MQ102", 10, false},
                                    {"MQ102B", 5, false}, {"MQ102B", 10, false}, {"MQ102", 15, false}};
        return value;
    }();
    return scenario;
}

const ScenarioDefinition& skyrimWhiterunShowcaseScenario() {
    // Enter Whiterun after Dragon Rising's prerequisite chain, while leaving
    // the next main-quest beat uncompleted. The city itself, its actors and
    // their packages still come entirely from the resolved retail load order.
    static const ScenarioDefinition scenario{
        "skyrim-whiterun-showcase",
        importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "Skyrim.esm",
        "WhiterunWorld",
        {},
        {},
        {{"MQ101", 900, true},
         {"MQ102", 160, true},
         {"MQ103", 190, true},
         {"MQ104", 160, true},
         {"MQ105", 10, false}},
    };
    return scenario;
}

const ScenarioDefinition& skyrimRiftenShowcaseScenario() {
    // Presentation-only third-person start inside the authored walled city.
    // The paired Tamriel gate supplies the arrival pose at runtime; the city
    // and its residents remain entirely retail content.
    static const ScenarioDefinition scenario{
        "skyrim-riften-showcase",
        importer::fnv::BethesdaGame::SkyrimSpecialEdition,
        "Skyrim.esm",
        "RiftenWorld",
        {},
        {},
        {},
    };
    return scenario;
}

const ScenarioDefinition* findScenario(const std::string& id) {
    const ScenarioDefinition& bleakFalls = skyrimBleakFallsScenario();
    if (id == bleakFalls.id) return &bleakFalls;
    const ScenarioDefinition& ralof = skyrimHelgenRalofScenario();
    if (id == ralof.id) return &ralof;
    const ScenarioDefinition& whiterun = skyrimWhiterunShowcaseScenario();
    if (id == whiterun.id) return &whiterun;
    const ScenarioDefinition& riften = skyrimRiftenShowcaseScenario();
    return id == riften.id ? &riften : nullptr;
}

}  // namespace odai::bethesda
