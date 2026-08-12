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

#include "audio/audio.h"
#include "games/newvegas/newvegas_actors.h"
#include "import/fnv/asset_source.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace odai::games::newvegas {

// Victor is an ordinary SkinnedActor with a conversation attached. These names
// are kept because this file is about HIM specifically -- where he stands, what
// he says, what he sounds like -- while everything about being a posable actor
// now lives in newvegas_actors.h and is shared with the rest of the town.
using VictorDialogueContext = ActorDialogueContext;
using VictorVoiceIndex = ActorVoiceIndex;
using VictorState = SkinnedActor;

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

// Uploading, posing and the "press E to talk" reach test are not Victor-specific
// and live in newvegas_actors.h: remapActorTextureSlots, updateActorPoses,
// actorIsInReach / findActorInReach.

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
