#pragma once

// Victor, the Goodsprings Securitron: placed in the world, talked to with E,
// and speaking his own recorded lines.
//
// Kept out of newvegas_app so the four moving parts stay readable together --
// where he stands (ACRE reference), what he looks like (securitron_static.nif),
// what he says (DIAL/INFO), and what he sounds like (Ogg Vorbis in a BSA).
//
// Everything here is optional and silent when unavailable: no Fallout install,
// no Victor, no voice archive -- each just leaves the feature off rather than
// failing the viewer.
//
// He is a STATIC mesh on a rigid transform. Fallout keeps skeletal animation in
// .kf files and this engine has no reader for them, so the rigged mesh would
// stand in bind pose; the static one is the same robot, and a Securitron rides
// a single wheel, so nothing about him is meant to articulate while idle.

#include "audio/audio.h"
#include "dialogue/dialogue_context.h"
#include "dialogue/dialogue_runtime.h"
#include "dialogue/dialogue_types.h"
#include "import/imported_scene.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::games::newvegas {

// Flags/approval storage the dialogue runtime needs. Fallout's own gating lives
// in CTDA conditions this engine cannot evaluate (no quest system), so nothing
// ever sets these -- the runtime's conditions are all default-constructed and
// therefore always pass. Present because DialogueRuntime requires a context.
class VictorDialogueContext final : public odai::dialogue::DialogueContext {
public:
    bool flag(const std::string& name) const override {
        const auto it = flags_.find(name);
        return it != flags_.end() && it->second;
    }
    void setFlag(const std::string& name, bool value) override { flags_[name] = value; }
    int approval(const std::string& companionId) const override {
        const auto it = approval_.find(companionId);
        return it == approval_.end() ? 0 : it->second;
    }
    void adjustApproval(const std::string& companionId, int delta) override {
        approval_[companionId] += delta;
    }

private:
    std::unordered_map<std::string, bool> flags_;
    std::unordered_map<std::string, int> approval_;
};

struct VictorState {
    bool placed = false;
    // Engine space, already converted from the reference's Bethesda coords.
    float position[3] = {};
    odai::dialogue::DialogueTree tree;
    VictorDialogueContext context;
    odai::dialogue::DialogueRuntime runtime;
    bool talking = false;
    // Node whose voice line has been started, so a line plays once rather than
    // restarting every frame the node is current.
    std::string spokenNodeId;
    std::size_t chunkIndex = static_cast<std::size_t>(-1);
    std::string status;  // one line for the startup log, success or reason
};

// Reads Victor's placement and dialogue, and builds a one-mesh ImportedScene
// standing him at his reference. Returns false (with state.status set) when
// anything is missing. `outScene` is what the caller hands to
// Renderer::addImportedSceneChunk.
// `positionOverride` (engine space, feet on the ground) stands him somewhere
// other than his ACRE reference. His real spot is ~7400 units from the usual
// spawn, which makes every "walk over and talk to him" test a hike across
// Goodsprings; passing the spawn point instead puts him where he can actually
// be exercised. Null uses his authored position.
bool loadVictor(
    const std::filesystem::path& dataFilesPath,
    const std::filesystem::path& pluginPath,
    VictorState& outState,
    odai::importer::ImportedScene& outScene,
    const float* positionOverride = nullptr);

// True when the camera is close enough and facing him for "press E to talk".
[[nodiscard]] bool victorIsInReach(
    const VictorState& state, const float cameraPosition[3], float cameraYawRadians);

// Starts a line's recorded audio, if the voice archive has one for it.
//
// Fallout names a voice file after the INFO record it belongs to
// (<quest>_<topic>_<infoFormId>_1.ogg), and the dialogue importer uses that same
// formID as its node id, so the lookup is a substring match on the node. The
// file is Ogg Vorbis, which miniaudio cannot decode, so it is converted to a
// .wav in `cacheDirectory` on first use -- the same trick the weather ambients
// use (see newvegas_ogg.h).
void speakVictorLine(
    VictorState& state,
    const std::filesystem::path& dataFilesPath,
    const std::filesystem::path& cacheDirectory,
    odai::audio::Audio& audioSystem);

}  // namespace odai::games::newvegas
