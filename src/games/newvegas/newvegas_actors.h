#pragma once

// The rest of Goodsprings: every actor the plugin places near the player,
// built into GPU-skinned instances the same way Victor is.
//
// Victor (newvegas_victor.h) stays separate because he is the one actor with a
// conversation, a voice and a scripted place to stand. Everything here is the
// generic case -- find them, build them, pose them -- and the two share
// buildSkinnedActor so the geometry/texture/clip path exists once.

#include "anim/animation_clip.h"
#include "anim/animation_sampler.h"
#include "import/fnv/actor_records.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/character_builder.h"
#include "import/imported_scene.h"
#include "math/math.h"

#include <cstdint>
#include <string>
#include <vector>

namespace odai::games::newvegas {

// One built, posable actor. Geometry and textures are held rather than consumed
// at upload because ImportedSkinnedMeshTemplate takes spans.
struct SkinnedActor {
    std::string name;
    odai::importer::fnv::FalloutCharacter character;
    std::vector<odai::importer::ImportedSceneTexture> textures;
    std::vector<odai::importer::ImportedScenePackedDraw> draws;

    odai::anim::AnimationClip idleClip;
    odai::anim::AnimationSampler sampler;
    std::vector<odai::math::Matrix4> poseScratch;
    float animationSeconds = 0.0f;

    // Engine space, feet on the ground.
    float position[3] = {};
    float yawRadians = 0.0f;

    std::uint32_t instanceSlot = 0;
    bool uploaded = false;
};

// Builds the skinned geometry, textures and draws for one actor from its
// skeleton and body-part list. Does not touch the renderer.
//
// `bodyPartPaths` are filenames relative to the skeleton's own directory, as
// NIFZ stores them. Returns false with `outWhy` set when nothing drawable came
// out; a partly-readable actor still succeeds, because one unreadable body part
// is not a reason to drop the whole character.
bool buildSkinnedActor(
    const odai::importer::fnv::FalloutAssetSource& assets,
    const std::string& skeletonPath,
    const std::vector<std::string>& bodyPartPaths,
    odai::importer::fnv::FalloutCharacter& outCharacter,
    std::vector<odai::importer::ImportedSceneTexture>& outTextures,
    std::vector<odai::importer::ImportedScenePackedDraw>& outDraws,
    std::string& outWhy);

// Loads a standing idle for an actor, resolved against `skeleton`. Silent
// failure is fine and expected: not every actor has one, and an actor without a
// clip stands in bind pose.
//
// TWO CONVENTIONS, because a creature and a human do not store this the same
// way. A creature keeps "mtidle.kf" beside its skeleton and the game finds it
// by name. A human's equivalent does not exist: `characters\_male\h2hidle.kf`
// is the unarmed Idle sequence and it animates exactly ONE node, because a
// standing human in Fallout is posed by an IDLE record pointing into
// `idleanims\`. Loading it and stopping there leaves the whole town in the
// T-pose the bodies are authored in.
//
// `variant` picks among the human idles so a row of settlers is not one pose
// repeated -- the tell that a crowd is one actor drawn many times.
bool loadActorIdleClip(
    const odai::importer::fnv::FalloutAssetSource& assets,
    const std::string& skeletonPath,
    const odai::anim::Skeleton& skeleton,
    std::size_t variant,
    odai::anim::AnimationClip& outClip,
    std::string& outWhy);

struct ActorPopulationStats {
    std::size_t placementsConsidered = 0;
    std::size_t skippedDisabled = 0;
    std::size_t skippedNoGeometry = 0;   // nothing in the record chain names a mesh
    std::size_t skippedBuildFailed = 0;
    std::size_t skippedNoSlots = 0;
    std::size_t skippedExcluded = 0;
    std::size_t built = 0;
    std::size_t animated = 0;
    std::string detail;
};

// Finds every actor placed within `radius` of the player's Bethesda XY and
// builds the ones whose geometry can be resolved today.
//
// `firstInstanceSlot` is where slot assignment starts -- Victor and the
// --character harness own lower slots. `maxActors` caps how many are built;
// placements are considered nearest-first, so the cap spends slots on the
// actors the player is standing among.
//
// `bethesdaCentre` is XY in Bethesda coordinates. A placement's own Z is kept
// as its standing height rather than being snapped to this engine's terrain:
// the terrain here is built from the same LAND records the editor placed the
// actor against, and snapping would drop the ones authored on a porch, a
// rooftop or a rock to the ground under them.
bool loadGoodspringsActors(
    const std::filesystem::path& pluginPath,
    const odai::importer::fnv::FalloutAssetSource& assets,
    const float bethesdaCentre[2],
    float radius,
    std::uint32_t firstInstanceSlot,
    std::size_t maxActors,
    // Actor bases already handled elsewhere. Victor is loaded by
    // newvegas_victor.cc with dialogue and a voice; without this he is built a
    // second time here, at his authored position, and the town has two of him.
    const std::vector<std::uint32_t>& excludeBaseFormIds,
    std::vector<SkinnedActor>& outActors,
    ActorPopulationStats& outStats);

// Advances every actor's clip and writes this frame's bone matrices, world
// placement folded in. Hand each actor's poseScratch to setSkinnedActorPose.
void updateActorPoses(std::vector<SkinnedActor>& actors, float deltaSeconds);

}  // namespace odai::games::newvegas
