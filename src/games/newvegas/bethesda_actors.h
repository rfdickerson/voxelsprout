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
#include "audio/audio.h"
#include "dialogue/dialogue_context.h"
#include "dialogue/dialogue_runtime.h"
#include "dialogue/dialogue_types.h"
#include "import/fnv/actor_records.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/character_builder.h"
#include "import/imported_scene.h"
#include "bethesda/navigation_world.h"
#include "bethesda/runtime_ids.h"
#include "math/math.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::games::newvegas {

// How close, and how squarely faced, an actor has to be for "press E to talk".
// Generous on both: a conversation the player has to hunt for the exact angle
// of is worse than one that occasionally offers itself a step early.
inline constexpr float kActorTalkRange = 500.0f;
inline constexpr float kActorTalkFacingDot = 0.25f;

// Flags/approval storage the dialogue runtime needs. Fallout's own gating lives
// in CTDA conditions this engine only partly evaluates, so little ever sets
// these -- the runtime's conditions are mostly default-constructed and
// therefore pass. Present because DialogueRuntime requires a context.
class ActorDialogueContext final : public odai::dialogue::DialogueContext {
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

// An actor's voice lines, indexed by the dialogue node they belong to.
//
// Fallout names a voice file after the INFO record it belongs to
// (<quest>_<topic>_<infoFormId>_1.ogg) and the dialogue importer uses that same
// formID as its node id, so the index is built by scanning the voice archive
// for the actor's VOICE TYPE folder and pulling the formID out of each
// filename. Scanning is what this needs and lookup is not enough: only the
// formID half of the name is known here, never the quest/topic half.
struct ActorVoiceIndex {
    // The VTYP this actor speaks with. Also the key a CTDA names when it binds
    // a GENERIC line to a voice type rather than to one actor, which is the
    // only dialogue most of a town has -- see dialogue_records.h.
    std::uint32_t voiceTypeFormId = 0;
    // The VTYP EditorID whose folder these lines live under. Actors sharing a
    // voice type share one index -- a town of ten has perhaps four.
    std::string voiceFolder;
    // Which plugin's voice tree these lines live under. Empty means "the one
    // the caller passed", which is the base game's. A companion's lines are
    // under ITS plugin (sound\\voice\\NVWillow.esp\\WillowsVoice\\), so this is
    // per ACTOR, not per run -- the header note below about the first component
    // not being a constant is now true within a single load order too.
    std::string voicePlugin;
    // Which open archive set holds them: "<plugin>\\<folder>", lowercased.
    // Keyed by plugin as well as folder because FemaleAdult01Default exists in
    // both games and means a different set of recordings in each.
    std::string archiveKey;
    // Node id -> virtual path of its .ogg inside the BSA.
    std::unordered_map<std::string, std::string> pathByNodeId;
    // Node id -> absolute path of its .ogg as a LOOSE file. A mod often ships
    // its voice as a plain tree rather than an archive -- Willow's 1474 lines
    // are loose under Sound\\Voice\\NVWillow.esp\\WillowsVoice\\ -- and both the
    // index and playback used to look only inside .bsa files, so those lines
    // were invisible and the actor read as having no voice at all.
    std::unordered_map<std::string, std::filesystem::path> loosePathByNodeId;
    std::string status;
};

// One built, posable actor. Geometry and textures are held rather than consumed
// at upload because ImportedSkinnedMeshTemplate takes spans.
//
// ONE TYPE FOR EVERY ACTOR, talkative or not. Victor was a separate struct
// until the town arrived, and the two had already drifted -- the conversation,
// the camera framing and the activation prompt each existed once for him and
// would have had to be written a second time for everybody else. An actor with
// nothing to say simply carries an empty `tree`, which costs a few empty
// containers per actor and is what makes "talk to anyone" a change to one code
// path instead of two.
struct SkinnedActor {
    // EditorID -- stable, unique, and what a diagnostic env var names.
    std::string name;
    // FULL -- what the game shows a player. Empty for most creatures.
    std::string fullName;
    // The actor BASE this instances. Dialogue is attributed by this formID (see
    // dialogue_records.h), and the generic town scan skips bases handled
    // elsewhere by it.
    std::uint32_t baseFormId = 0;
    // The placed ACHR/ACRE reference after load-order remapping. This is the
    // persistent gameplay identity; baseFormId identifies what the actor is.
    std::uint32_t referenceFormId = 0;
    // Unified runtime identity. TES3 placements are FRMR ObjectIds and have no
    // numeric form ID; later games lazily derive this from referenceFormId.
    odai::bethesda::ObjectId runtimeObjectId;
    std::vector<std::uint32_t> referenceTypeFormIds;
    // Deterministically materialized CNTO inventory. LVLI tokens are expanded
    // during actor construction so runtime never exposes a list record as if
    // it were a takeable item.
    std::vector<std::uint32_t> inventoryFormIds;
    odai::importer::fnv::FalloutCharacter character;
    std::vector<odai::importer::ImportedSceneTexture> textures;
    std::vector<odai::importer::ImportedScenePackedDraw> draws;

    // idle plays whenever the actor is not in conversation; talk plays while it
    // is. An actor with no talk clip keeps idling through the conversation.
    odai::anim::AnimationClip idleClip;
    odai::anim::AnimationClip talkClip;
    // The game's own mtforward, with the root motion taken out of the pose and
    // handed to walkSpeedUnitsPerSecond instead -- see loadActorWalkClip. Empty
    // for anything with no locomotion clip beside its skeleton.
    odai::anim::AnimationClip walkClip;
    float walkSpeedUnitsPerSecond = 0.0f;
    odai::anim::AnimationSampler sampler;
    // GPU-ready local pose before actor world placement. Clip changes blend
    // from a frozen copy of this pose, avoiding an idle/walk hard cut while
    // sampling only the destination clip once per frame.
    std::vector<odai::math::Matrix4> localPoseScratch;
    std::vector<odai::math::Matrix4> poseTransitionSource;
    std::vector<odai::math::Matrix4> poseScratch;
    float animationSeconds = 0.0f;
    float poseTransitionElapsedSeconds = 0.0f;
    // Which clip the last pose came from, so a switch can restart the clock and
    // capture the displayed pose as the blend source.
    bool posedTalking = false;
    bool posedWalking = false;

    // Engine space, feet on the ground.
    float position[3] = {};
    float yawRadians = 0.0f;

    // ---- Wandering ---------------------------------------------------
    // A townsperson walks between connected authored navmesh triangles near
    // where the plugin placed them. The route stores shared-edge midpoints, so
    // following it cannot take a straight-line shortcut across scenery.
    bool wanders = false;   // has a locomotion clip, so it can move at all
    bool walking = false;   // moving right now, which picks walkClip over idle
    bool projectedToNavigation = false;
    float wanderOrigin[3] = {};
    float wanderTarget[3] = {};
    std::vector<ActorNavigationStep> wanderPath;
    std::size_t wanderPathIndex = 0u;
    float wanderPauseSeconds = 0.0f;
    std::uint32_t wanderRng = 0;
    bool scriptedMoveActive = false;
    bool scriptedMoveArrived = false;
    std::uint64_t scriptedMoveRevision = 0u;
    // Skyrim scenario actors hand translation to BethesdaSession/Jolt after
    // their initial authored-navmesh projection. The legacy actor planner then
    // produces velocity intent only; these fields bridge that intent without
    // giving presentation a second transform authority.
    bool runtimeControllerOwned = false;
    bool runtimeControllerNeedsRelocation = false;
    bool runtimeControllerBlocked = false;
    bool runtimeDead = false;
    odai::math::Vector3 runtimeRequestedVelocity{};
    // Top of the actor's own rest-pose geometry, above its feet. A conversation
    // aims at a fraction of this rather than at a constant, because a bighorner,
    // a settler and a Securitron are not the same height and a placement's
    // origin is at the FEET -- aiming at the origin points the camera at the
    // ground.
    float standingHeightUnits = 0.0f;
    // NPC_ records have an upright humanoid body even when their legacy rig
    // has no vertices dominantly weighted to a bone literally named "Head".
    // That distinction matters to the dialogue fallback: 65% is sensible for
    // a low creature head, but is a humanoid's pelvis.
    bool humanoid = false;
    // Where the actor's HEAD is, as a point the live pose can carry.
    //
    // A bone index plus the centroid of that bone's own vertices, in the space
    // those vertices are authored in, so the head's WORLD position each frame is
    // one matrix-vector multiply against poseScratch. Static heights measured off
    // the bind pose are not enough: an animation moves a head, and Willow's idle
    // moves hers far enough that a bind-pose number framed her at the waist.
    // -1 means the rig names no head; headHeightUnits is then the fallback.
    int headAnchorBone = -1;
    float headAnchorLocal[3] = {};
    float headHeightUnits = 0.0f;

    // The conversation, when this actor has one. An empty `tree` means it
    // cannot be talked to.
    odai::dialogue::DialogueTree tree;
    ActorDialogueContext context;
    odai::dialogue::DialogueRuntime runtime;
    bool talking = false;
    // Node whose voice line has been started, so a line plays once rather than
    // restarting every frame the node is current.
    std::string spokenNodeId;
    ActorVoiceIndex voice;

    bool placed = false;
    std::uint32_t instanceSlot = 0;
    bool uploaded = false;
    // Game-side mirror of the renderer slot's culling state. Invisible actors
    // keep advancing their animation clock, but do not wander, sample or upload
    // a pose until they can contribute to the frame again.
    bool renderVisible = true;

    std::string status;           // one line for the startup log, success or reason
    std::string animationStatus;
    // Per-phase load cost, "placement 12ms  dialogue 900ms  ...". This is the
    // only thing that makes a slow launch attributable, and a slow launch is the
    // only symptom a forgotten onRecordHeader filter produces.
    std::string timing;

    [[nodiscard]] bool canTalk() const {
        return placed && !runtimeDead && !tree.nodes.empty();
    }
    // What to put on screen. Falls back to the EditorID rather than to nothing,
    // because an unnamed actor still has to be addressable in a prompt.
    [[nodiscard]] const std::string& displayName() const {
        return fullName.empty() ? name : fullName;
    }
};

// Called on a copy of one immutable catalog placement. Returning false makes
// it non-resident; returning true may also replace position/rotation with the
// persistent runtime transform before the presentation actor is built.
using ActorPlacementRuntimeResolver =
    std::function<bool(odai::importer::fnv::FalloutActorPlacement&)>;

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
    std::string& outWhy,
    // TES3 BODY records carry the rigid attachment slot outside the NIF. One
    // optional skeleton-node name per bodyPartPaths entry restores that
    // authored pivot; later Bethesda formats keep deriving it from the NIF.
    const std::vector<std::string>* rigidAttachmentBones = nullptr);

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

// Loads ordinary forward locomotion and removes its horizontal root motion so
// navigation remains the sole transform authority. TES3 humanoids keep this
// interval inside base_anim.nif; later games use external KF files (or the
// existing procedural HKX fallback).
bool loadActorWalkClip(
    const odai::importer::fnv::FalloutAssetSource& assets,
    const std::string& skeletonPath,
    const odai::anim::Skeleton& skeleton,
    bool female,
    odai::anim::AnimationClip& outClip,
    float& outSpeedUnitsPerSecond,
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
    // Actors that resolved a locomotion clip, and so can wander rather than
    // hold one spot. A creature with no mtforward.kf is not a failure.
    std::size_t walking = 0;
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
    // Optional; when non-null and non-empty, actors are discovered across the
    // whole load order instead of from pluginPath alone. A companion mod's NPC,
    // placement, race and armour all live in its own plugin.
    const odai::importer::fnv::FalloutLoadOrder* loadOrder,
    const odai::importer::fnv::FalloutAssetSource& assets,
    const float bethesdaCentre[3],
    float radius,
    std::uint32_t firstInstanceSlot,
    std::size_t maxActors,
    // Actor bases already handled elsewhere. Victor is loaded by
    // newvegas_victor.cc with dialogue and a voice; without this he is built a
    // second time here, at his authored position, and the town has two of him.
    const std::vector<std::uint32_t>& excludeBaseFormIds,
    // Optional ownership filter applied after load-order remapping. Distance
    // alone is insufficient for Skyrim because Solstheim and Tamriel reuse
    // overlapping coordinate ranges.
    const std::function<bool(std::uint32_t referenceFormId)>& placementFilter,
    std::vector<SkinnedActor>& outActors,
    ActorPopulationStats& outStats,
    // Optional immutable full-placement catalog. When present, `radius` is not
    // applied to authored coordinates; placementRuntimeResolver owns residency
    // and may substitute the saved/runtime transform. This is the Skyrim path.
    const odai::importer::fnv::FalloutActorScan* actorCatalog = nullptr,
    const std::unordered_map<std::uint32_t, std::string>* catalogVoiceFolderPlugin = nullptr,
    const ActorPlacementRuntimeResolver& placementRuntimeResolver = {});

// Attaches a conversation to every actor that has one, in ONE walk over the
// plugin. Actors that already carry a tree are left alone, and actors the
// plugin gives no lines are simply left unable to talk.
//
// Separate from loadGoodspringsActors because it is a different question about
// the same actors, and because it must run after every actor is in the list --
// including any the caller added itself, like Victor.
//
// Returns the number of actors that gained a conversation; `outDetail` is one
// line for the startup log.
std::size_t loadActorDialogue(
    const std::filesystem::path& pluginPath,
    // Optional; non-null reads DIAL/INFO across the whole load order, which is
    // the only way a companion mod's own lines are reachable.
    const odai::importer::fnv::FalloutLoadOrder* loadOrder,
    std::vector<SkinnedActor>& actors,
    std::string& outDetail);

// Indexes the recorded lines for every voice type the actors use, ONCE per
// distinct folder, and hands each actor the index for its own.
//
// Indexing is what this needs and lookup is not enough: only the formID half of
// a voice filename is derivable from a dialogue node, never the quest/topic
// half, so the file can only be found by scanning the archive's name list. That
// list is 105517 entries in Fallout - Voices1.bsa alone, which is why
// BsaArchive::open's folder-prefix filter is load-bearing here and why the
// index is shared rather than built per actor.
//
// Actors with no dialogue are skipped -- an index nothing will ever look up is
// pure cost. Silent failure is fine: no archive means readable dialogue with no
// audio, which is how this degrades everywhere else too.
// `pluginFileName` is the plugin the lines belong to ("FalloutNV.esm",
// "Fallout3.esm"), because a voice path is
// sound\voice\<plugin>\<VTYP EditorID>\ and that first component is NOT a
// constant. Hardcoding it to falloutnv.esm is why Fallout 3 loaded 46 actors,
// resolved four voice types for them, and voiced none of them.
std::size_t loadActorVoices(
    const std::filesystem::path& dataFilesPath,
    const std::string& pluginFileName,
    // Mod roots, searched for LOOSE voice trees alongside the archives.
    const std::vector<std::string>& modDirectories,
    std::vector<SkinnedActor>& actors,
    std::string& outDetail);

// Starts the current dialogue node's recorded audio, if the actor's voice index
// has one for it. Does nothing when the node has already been spoken, so it is
// safe to call every frame.
//
// The file is Ogg Vorbis, which miniaudio cannot decode, so it is converted to
// a .wav in `cacheDirectory` on first use -- the same trick the weather
// ambients use (see newvegas_ogg.h).
void speakActorLine(
    SkinnedActor& actor,
    const std::filesystem::path& cacheDirectory,
    odai::audio::Audio& audioSystem);

// Advances every actor's clip and writes this frame's bone matrices, world
// placement folded in. Hand each actor's poseScratch to setSkinnedActorPose.
void updateActorPoses(std::vector<SkinnedActor>& actors, float deltaSeconds);

// Walks the ones that can walk: chooses a connected authored-navmesh route near
// where they were placed, turns toward each shared-edge waypoint, and moves at
// the speed their own animation was authored for. With no resident navmesh the
// legacy local wander remains available for games/content that have none.
//
// `groundHeightAt(x, z, referenceY, outHeight)` clamps feet to whatever they
// should be standing on, and must answer false where no cell is resident -- an
// actor over a hole keeps its current height rather than dropping to zero,
// which is the difference between a townsperson at the edge of the streamed set
// and one falling through the world.
//
// referenceY is the ACTOR's own feet, not the player's. Passing the player's
// puts everyone at the player's altitude: on a hillside town that stands the
// whole population in the air over the valley, which is exactly what the
// diagnostic parade used to do.
//
// `skipIndex` is the actor the player is talking to: someone who walks off
// mid-sentence is worse than someone who stands still.
//
// `slideOutOfWalls` takes (x, z, feetY, headY, radius) and pushes the position
// out of anything solid. Without it an actor has NO horizontal collision at all
// -- the wander radius was the entire strategy, on the theory that staying near
// an authored spot keeps everyone indoors-or-outdoors as the game intended. It
// does not: a companion spawned beside the player has no authored spot, and
// walks through the nearest wall and out through the floor beyond it.
//
void updateActorWandering(
    std::vector<SkinnedActor>& actors,
    float deltaSeconds,
    const ActorNavigationWorld* navigation,
    const std::function<bool(float, float, float, float&)>& groundHeightAt,
    const std::function<void(float&, float&, float, float, float)>& slideOutOfWalls,
    int skipIndex);

// Converts a horizontal engine-space direction into the yaw convention shared
// by actor world transforms and locomotion. Imported actors face engine -Z at
// yaw zero (Bethesda asset +Y after the Z-up -> Y-up basis conversion).
[[nodiscard]] float actorYawForDirection(float dx, float dz);
[[nodiscard]] odai::math::Vector3 actorFacing(float yawRadians);

// Rewrites every vertex's textureIndex from an index into `actor.textures` to
// the bindless slot the renderer handed back for it.
//
// This exists because a skinned template's vertices reach the GPU verbatim: the
// scene-index-to-bindless-slot remap that addImportedSceneChunk performs for
// world geometry has no equivalent on the skinned path, so the caller has to do
// it. Must run after Renderer::uploadSkinnedActorTextures and before
// Renderer::uploadSkinnedMeshTemplate.
void remapActorTextureSlots(SkinnedActor& actor, const std::vector<std::uint32_t>& bindlessSlots);

// Rest-pose height above the actor's own origin. The origin is its FEET, so
// this is what a caller needs to aim at anything above the ground.
[[nodiscard]] float actorStandingHeight(const odai::importer::fnv::FalloutCharacter& character);

// Locates the rig's head: the bone, the centroid of its vertices in their
// authored space, and the bind-pose height of that centroid above the actor's
// feet. outBone is -1 when the skeleton names no head. Walks every vertex, so
// call it once per built base rather than per frame.
void findActorHeadAnchor(
    const odai::importer::fnv::FalloutCharacter& character,
    int& outBone,
    float outLocal[3],
    float& outBindHeight);

// Where a conversation should look: below the speaker's face by enough that the
// dialogue card, which sits centred, does not cover the person talking.
[[nodiscard]] float conversationFaceHeight(const SkinnedActor& actor);

// True when the camera is close enough to `actor` and facing it -- the test
// behind "press E to talk".
[[nodiscard]] bool actorIsInReach(
    const SkinnedActor& actor, const float cameraPosition[3], float cameraYawRadians);

// The actor the player is looking at and close enough to talk to, or -1. When
// two are in reach the nearer one wins, so standing between two townsfolk picks
// the one being faced rather than whichever was built first.
[[nodiscard]] int findActorInReach(
    const std::vector<SkinnedActor>& actors, const float cameraPosition[3],
    float cameraYawRadians);

}  // namespace odai::games::newvegas
