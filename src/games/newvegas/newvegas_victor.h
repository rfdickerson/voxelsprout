#pragma once

// Victor, the Goodsprings Securitron: placed in the world, animated from the
// game's own .kf clips, talked to with E, and speaking his own recorded lines.
//
// Kept out of newvegas_app so the five moving parts stay readable together --
// where he stands (ACRE reference), what he looks like (skeleton + skinned body
// parts), how he moves (mtidle.kf / specialidle_talk01.kf), what he says
// (DIAL/INFO), and what he sounds like (Ogg Vorbis in a BSA).
//
// Everything here is optional and silent when unavailable: no Fallout install,
// no Victor, no voice archive -- each just leaves the feature off rather than
// failing the viewer. He degrades in steps rather than all at once: no clips
// still gives a posed static robot, no voice still gives readable dialogue.

#include "anim/animation_clip.h"
#include "anim/animation_sampler.h"
#include "audio/audio.h"
#include "dialogue/dialogue_context.h"
#include "dialogue/dialogue_runtime.h"
#include "dialogue/dialogue_types.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/character_builder.h"
#include "import/imported_scene.h"
#include "math/math.h"

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

// Victor's voice lines, indexed by the dialogue node they belong to.
//
// Fallout names a voice file after the INFO record it belongs to
// (<quest>_<topic>_<infoFormId>_1.ogg) and the dialogue importer uses that same
// formID as its node id, so the index is built once by scanning the voice
// archive for his folder and pulling the formID out of each filename. Scanning
// is what this needs and lookup is not enough: only the formID half of the name
// is known here, never the quest/topic half.
struct VictorVoiceIndex {
    // Node id -> virtual path of its .ogg inside the BSA.
    std::unordered_map<std::string, std::string> pathByNodeId;
    std::string status;
};

struct VictorState {
    bool placed = false;
    // The CREA this actor instances, so the generic town population can skip
    // him -- he is loaded, posed and animated by this file instead.
    std::uint32_t baseFormId = 0;
    // Engine space, already converted from the reference's Bethesda coords.
    float position[3] = {};
    // Which way he faces, engine yaw. Folded into the actor's world matrix
    // rather than an instance transform, because a skinned actor has no
    // instance transform -- placement rides on the bone matrices.
    float yawRadians = 0.0f;

    odai::dialogue::DialogueTree tree;
    VictorDialogueContext context;
    odai::dialogue::DialogueRuntime runtime;
    bool talking = false;
    // Node whose voice line has been started, so a line plays once rather than
    // restarting every frame the node is current.
    std::string spokenNodeId;
    VictorVoiceIndex voice;

    // The skinned actor: rest-pose geometry, its textures, and its draws. Held
    // rather than consumed at upload because ImportedSkinnedMeshTemplate takes
    // spans -- a local would dangle the moment loadVictor returned.
    odai::importer::fnv::FalloutCharacter character;
    std::vector<odai::importer::ImportedSceneTexture> textures;
    std::vector<odai::importer::ImportedScenePackedDraw> draws;

    // Clips, and the pose they produce. idle plays whenever he is not in
    // conversation; talk plays while he is.
    odai::anim::AnimationClip idleClip;
    odai::anim::AnimationClip talkClip;
    odai::anim::AnimationSampler sampler;
    std::vector<odai::math::Matrix4> poseScratch;
    float animationSeconds = 0.0f;
    // Which clip the last pose came from, so a switch can restart the clock.
    bool posedTalking = false;
    bool uploaded = false;

    std::string status;      // one line for the startup log, success or reason
    std::string animationStatus;
    // Per-phase load cost, "placement 12ms  dialogue 900ms  ...". See
    // loadVictor: this is the only thing that makes a slow launch attributable.
    std::string timing;
};

// Reads Victor's placement, dialogue, skinned geometry, textures and animation
// clips. Returns false (with state.status set) when anything essential is
// missing; a missing CLIP is not essential and leaves him posed but still.
//
// Nothing here touches the renderer. The caller uploads
// `state.textures` (Renderer::uploadSkinnedActorTextures), rewrites each
// vertex's textureIndex to the bindless slot it gets back, and then uploads the
// template -- see remapVictorTextureSlots below.
//
// `assets` is the caller's already-open archive index -- pass the streamer's.
// Opening a second one here cost ~50 ms of redundant indexing, and worse, a
// private one carries no --mod directories, so Victor silently ignored any
// installed texture pack while the world around him used it.
//
// `positionOverride` (engine space, feet on the ground) stands him somewhere
// other than his ACRE reference, which is ~7400 units from the usual spawn.
bool loadVictor(
    const std::filesystem::path& dataFilesPath,
    const std::filesystem::path& pluginPath,
    const odai::importer::fnv::FalloutAssetSource& assets,
    VictorState& outState,
    const float* positionOverride = nullptr);

// Rewrites every vertex's textureIndex from an index into `state.textures` to
// the bindless slot the renderer handed back for it.
//
// This exists because a skinned template's vertices reach the GPU verbatim: the
// scene-index-to-bindless-slot remap that addImportedSceneChunk performs for
// world geometry has no equivalent on the skinned path, so the caller has to do
// it. Must run after Renderer::uploadSkinnedActorTextures and before
// Renderer::uploadSkinnedMeshTemplate.
void remapVictorTextureSlots(VictorState& state, const std::vector<std::uint32_t>& bindlessSlots);

// Advances the clip clock and writes this frame's bone matrices, world
// placement already folded in, into state.poseScratch. Hand that straight to
// Renderer::setSkinnedActorPose.
void updateVictorPose(VictorState& state, float deltaSeconds);

// True when the camera is close enough and facing him for "press E to talk".
[[nodiscard]] bool victorIsInReach(
    const VictorState& state, const float cameraPosition[3], float cameraYawRadians);

// Starts the current dialogue node's recorded audio, if the voice archive has
// one for it. Does nothing when the node has already been spoken, so it is safe
// to call every frame.
//
// The file is Ogg Vorbis, which miniaudio cannot decode, so it is converted to
// a .wav in `cacheDirectory` on first use -- the same trick the weather
// ambients use (see newvegas_ogg.h).
// Takes no Data directory: the voice archive is opened once by loadVictor and
// held open, because a line can only be found by scanning the full name list
// (see VictorVoiceIndex) and re-indexing 105517 entries per spoken line is not
// an option.
void speakVictorLine(
    VictorState& state,
    const std::filesystem::path& cacheDirectory,
    odai::audio::Audio& audioSystem);

}  // namespace odai::games::newvegas
