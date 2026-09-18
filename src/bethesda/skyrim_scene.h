#pragma once

#include "bethesda/skyrim_dialogue.h"
#include <map>
#include <set>

namespace odai::bethesda {

struct SkyrimSceneFragment {
    std::uint8_t flags = 0;
    std::uint32_t phase = 0xffffffffu;
    std::string scriptClass, function;
};
struct SkyrimScenePhase {
    std::string name;
    std::vector<Condition> startConditions, completionConditions;
};
struct SkyrimSceneAction {
    std::uint16_t type = 0;
    std::int32_t actorAlias = -1;
    std::uint32_t index = 0, startPhase = 0, endPhase = 0, flags = 0;
    std::uint32_t rawTopic = 0;
    RecordKey topic;
    std::vector<std::uint32_t> rawPackages;
    std::vector<RecordKey> packages;
    float seconds = 0;
};
struct SkyrimSceneDefinition {
    RecordKey record, quest;
    std::string editorId;
    std::uint32_t rawQuest = 0, flags = 0;
    std::vector<SkyrimScenePhase> phases;
    std::vector<SkyrimSceneAction> actions;
    std::map<std::int32_t, std::vector<RecordKey>> aliasPackages;
    VmadAttachments scripts;
    std::vector<SkyrimSceneFragment> fragments;
};
struct SkyrimSceneProgress {
    std::uint32_t phase = 0;
    bool entered = false, begun = false;
    std::set<std::uint32_t> completed;
    std::map<std::uint32_t, double> timers;
};
struct SkyrimSceneSpeech {
    RecordKey scene, info, responseInfo;
    ObjectId speaker;
    std::uint32_t action = 0, response = 0;
    std::uint64_t sequence = 0;
    std::string text, voiceKey;
    bool presented = false;
    double remainingSeconds = 0;
};
struct SkyrimScenePackage {
    RecordKey record;
    RecordKey quest;
    ObjectId destination;
    ObjectId escortTarget;
    float escortWaitDistance = 0;
    std::int32_t destinationAlias = -1;
    std::vector<Condition> conditions;
    float radius = 128;
    bool patrol = false;
    VmadInfoAttachments scripts;
};
struct SkyrimScriptTrigger {
    ObjectId object;
    std::array<float, 3> halfExtents{};
    std::vector<std::string> scripts;
    std::set<ObjectId> occupants;
};
bool readSkyrimScene(const importer::fnv::EsmRecordView& record,
    RecordKey stable, SkyrimSceneDefinition& out, std::string& error);
bool readVmadSceneFragments(std::span<const std::uint8_t> bytes,
    VmadAttachments& common, std::vector<SkyrimSceneFragment>& fragments,
    std::string& error);
} // namespace odai::bethesda
