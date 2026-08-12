#include "games/newvegas/newvegas_victor.h"

#include "games/newvegas/newvegas_ogg.h"
#include "import/dds.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/bsa_archive.h"
#include "import/fnv/dialogue_records.h"
#include "import/fnv/kf_animation.h"
#include "import/fnv/nif_scene.h"

#include "core/frame_profiler.h"
#include "core/log.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <system_error>

namespace odai::games::newvegas {

namespace {

constexpr const char* kVictorEditorId = "Victor";
// mtidle is his standing idle: 13.3s, looping, and it moves the whole robot --
// torso sway, arm drift, the wheel correcting under him. specialidle_talk01 is
// a gesture cycle authored for conversation; it is flagged one-shot in the
// file and looped here deliberately, because a conversation lasts as long as
// the player reads and a 5s clip that stops dead looks broken.
constexpr const char* kIdleClipPath = "creatures\\NVSecuritron\\mtidle.kf";
constexpr const char* kTalkClipPath =
    "creatures\\NVSecuritron\\idleanims\\specialidle_talk01.kf";
// His VTYP EditorID, and so his voice folder. Named here rather than resolved
// from his VTCK because he is deliberately EXCLUDED from the generic town scan
// -- which is what owns the VTYP table -- so there is nothing to look it up in.
// Verified against that table: his VTCK resolves to exactly this.
constexpr const char* kVictorVoiceFolder = "RobotVictor";


std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

// Loads one .kf and resolves it against the skeleton. A missing or unreadable
// clip is not an error worth failing the character over -- he simply stands
// still -- so this reports through `outWhy` and returns false.
bool loadClip(
    const importer::fnv::FalloutAssetSource& assets,
    const anim::Skeleton& skeleton,
    const char* virtualPath,
    bool forceLoop,
    anim::AnimationClip& outClip,
    std::string& outWhy
) {
    std::vector<std::uint8_t> bytes;
    if (!assets.resolveMesh(virtualPath, bytes, outWhy)) {
        return false;
    }
    importer::fnv::KfAnimation animation;
    if (!importer::fnv::parseKfAnimation(bytes, animation, outWhy)) {
        return false;
    }
    importer::fnv::FalloutAnimationStats stats;
    if (!importer::fnv::buildFalloutAnimationClip(animation, skeleton, outClip, stats)) {
        outWhy = "no track bound to this skeleton";
        return false;
    }
    if (forceLoop) {
        outClip.loop = true;
    }
    outWhy = std::to_string(stats.boundTracks) + "/" + std::to_string(stats.tracks) +
             " tracks bound, " + std::to_string(animation.stats.unsupportedInterpolators) +
             " B-spline skipped, " + std::to_string(outClip.duration) + "s";
    return true;
}

// Indexes Victor's voice folder inside the archives.
//
}  // namespace

bool loadVictor(
    const std::filesystem::path& pluginPath,
    const importer::fnv::FalloutAssetSource& assets,
    VictorState& outState,
    const float* positionOverride
) {
    outState = VictorState{};

    // Phase timing, reported through outState.timing. loadVictor dominated
    // startup (7.0 s of an 8.2 s launch when this was first measured) and the
    // work is spread over six unrelated subsystems -- plugin records, archive
    // indexing, NIF parsing, DDS decode, .kf parsing, voice scanning -- so
    // "Victor is slow" is not actionable without knowing which.
    odai::core::Stopwatch phaseTimer;
    const auto lap = [&phaseTimer](const char* label, std::string& out) {
        char buffer[64] = {};
        std::snprintf(buffer, sizeof(buffer), "%s %.0fms  ", label, static_cast<double>(phaseTimer.lapMs()));
        out += buffer;
    };

    importer::fnv::SpeakerPlacement placement;
    std::string error;
    if (!importer::fnv::findSpeakerPlacement(pluginPath, kVictorEditorId, placement, error)) {
        outState.status = "not placed: " + error;
        return false;
    }
    // Bethesda is Z-up; this engine is Y-up. Same conversion cell_builder makes
    // for every other reference: (x, y, z) -> (x, z, -y).
    if (positionOverride != nullptr) {
        outState.position[0] = positionOverride[0];
        outState.position[1] = positionOverride[1];
        outState.position[2] = positionOverride[2];
    } else {
        outState.position[0] = placement.position[0];
        outState.position[1] = placement.position[2];
        outState.position[2] = -placement.position[1];
    }

    lap("placement", outState.timing);

    outState.baseFormId = placement.baseFormId;

    importer::fnv::DialogueImportStats stats;
    if (!importer::fnv::buildSpeakerDialogueTree(
            pluginPath, kVictorEditorId, outState.tree, stats, error)) {
        outState.status = "no dialogue: " + error;
        return false;
    }

    lap("dialogue", outState.timing);

    if (placement.skeletonPath.empty() || placement.bodyPartPaths.empty()) {
        outState.status = "actor record names no skeleton or body parts";
        return false;
    }

    lap("archives", outState.timing);

    // Skeleton first: a CREA's MODL is the skeleton, and every body part is
    // skinned to it.
    std::vector<std::uint8_t> bytes;
    importer::fnv::NifSkeleton nifSkeleton;
    if (!assets.resolveMesh(placement.skeletonPath, bytes, error) ||
        !importer::fnv::parseNifSkeleton(bytes, nifSkeleton, error)) {
        outState.status = "skeleton unavailable: " + error;
        return false;
    }
    if (!importer::fnv::buildFalloutSkeleton(nifSkeleton, outState.character.skeleton)) {
        outState.status = "skeleton conversion failed";
        return false;
    }

    const std::size_t slash = placement.skeletonPath.find_last_of("\\/");
    const std::string modelDirectory =
        slash == std::string::npos ? std::string() : placement.skeletonPath.substr(0, slash + 1);
    // A Securitron's NIFZ list carries EVERY face screen the model can wear --
    // Victor's cowboy portrait and the "no signal" static screen sit in it side
    // by side, and the game swaps between them at runtime. Drawing the list
    // literally puts both on the same quad, where the static screen wins and
    // Victor wears colour bars for a face. He is being placed as Victor, so his
    // own screen is the one to keep.
    bool hasNamedScreen = false;
    for (const std::string& partName : placement.bodyPartPaths) {
        if (toLowerAscii(partName).find("victorscreen") != std::string::npos) {
            hasNamedScreen = true;
            break;
        }
    }
    for (const std::string& partName : placement.bodyPartPaths) {
        const std::string lowered = toLowerAscii(partName);
        // An effect billboard, not body geometry; the opaque path draws it as a
        // grey sheet.
        if (lowered.find("smoketrail") != std::string::npos) {
            continue;
        }
        if (hasNamedScreen && lowered.find("screenstatic") != std::string::npos) {
            continue;
        }
        const std::string partPath = modelDirectory + partName;
        importer::fnv::NifSkinnedModel model;
        if (!assets.resolveMesh(partPath, bytes, error) ||
            !importer::fnv::parseNifSkinnedMesh(bytes, model, error)) {
            continue;
        }
        std::string bindError;
        if (importer::fnv::appendFalloutCharacterMesh(model, outState.character, bindError)) {
            continue;
        }
        // No skinned shapes: this part is a rigid prop parented to a bone
        // rather than weighted geometry, which is how Victor's face screen is
        // authored. Re-read it as a static mesh and hang it off the bone named
        // by its own root node. Skipping this is what left him with an empty
        // black box for a head.
        importer::fnv::NifSkeleton partNodes;
        importer::fnv::NifModel staticModel;
        std::string rigidError;
        if (!importer::fnv::parseNifSkeleton(bytes, partNodes, rigidError) ||
            partNodes.bones.empty() ||
            !importer::fnv::parseNifStaticMesh(bytes, staticModel, rigidError)) {
            continue;
        }
        // Drop the alpha-blended shapes before they are merged in. The skinned
        // draw path is opaque-only, so a blended shape does not blend -- it
        // draws as a solid slab. Victor's face screen ships with a glare quad
        // (textures\pipboy3000\screenglare.dds) mounted a few units in FRONT of
        // the portrait itself, which opaque means his face renders as a dark
        // panel and looks like a texture that failed to load.
        std::erase_if(staticModel.shapes, [](const importer::fnv::NifShape& shape) {
            return shape.alphaBlend;
        });
        if (staticModel.shapes.empty()) {
            continue;
        }
        importer::fnv::appendFalloutCharacterRigidMesh(
            staticModel, partNodes.bones.front().name, outState.character, rigidError);
    }
    if (outState.character.vertices.empty()) {
        outState.status = "no skinned geometry bound to the skeleton";
        return false;
    }

    lap("geometry", outState.timing);

    // Textures, and the per-vertex material state that goes with them.
    //
    // Both are written onto the VERTICES here, walking each part's index range,
    // because a skinned template has no per-part metadata on the GPU: the
    // skinning pass copies textureIndex and flags straight through to the
    // output vertex, and the main pass reads them from there. The parts do not
    // share vertices, so every vertex is reached exactly once.
    std::unordered_map<std::string, std::uint32_t> localTextureIndexByPath;
    for (const importer::fnv::FalloutCharacterPart& part : outState.character.parts) {
        if (part.indexCount == 0u) {
            continue;
        }
        std::uint32_t localTextureIndex = 0xffffffffu;
        if (!part.diffuseTexturePath.empty()) {
            const std::string key = toLowerAscii(part.diffuseTexturePath);
            const auto existing = localTextureIndexByPath.find(key);
            if (existing != localTextureIndexByPath.end()) {
                localTextureIndex = existing->second;
            } else {
                std::vector<std::uint8_t> ddsBytes;
                odai::importer::ImportedSceneTexture texture;
                if (assets.resolveTexture(part.diffuseTexturePath, ddsBytes, error) &&
                    odai::importer::loadDdsFromMemory(ddsBytes.data(), ddsBytes.size(), texture)) {
                    texture.sourcePath = part.diffuseTexturePath;
                    localTextureIndex = static_cast<std::uint32_t>(outState.textures.size());
                    localTextureIndexByPath.emplace(key, localTextureIndex);
                    outState.textures.push_back(std::move(texture));
                }
            }
        }

        std::uint32_t flags = 0u;
        if (part.alphaTest) {
            flags |= odai::importer::kImportedSceneMaterialFlagAlphaTest;
        }
        if (part.alphaBlend) {
            flags |= odai::importer::kImportedSceneMaterialFlagAlphaBlend;
        }
        if (part.twoSided) {
            flags |= odai::importer::kImportedSceneMaterialFlagTwoSided;
        }
        if (part.unlit) {
            flags |= odai::importer::kImportedSceneMaterialFlagUnlit;
        }
        for (std::uint32_t i = part.firstIndex; i < part.firstIndex + part.indexCount; ++i) {
            if (i >= outState.character.indices.size()) {
                break;
            }
            const std::uint32_t vertexIndex = outState.character.indices[i];
            if (vertexIndex >= outState.character.vertices.size()) {
                continue;
            }
            outState.character.vertices[vertexIndex].textureIndex = localTextureIndex;
            outState.character.vertices[vertexIndex].flags = flags;
        }

        odai::importer::ImportedScenePackedDraw draw{};
        draw.firstIndex = part.firstIndex;
        draw.indexCount = part.indexCount;
        draw.alphaThreshold = part.alphaThreshold;
        outState.draws.push_back(draw);
    }
    if (outState.draws.empty()) {
        outState.status = "no drawable parts";
        return false;
    }

    lap("textures", outState.timing);

    // Animation. The sampler is bound with the character's OWN inverse bind
    // matrices, from NiSkinData, not with ones derived from the skeleton -- see
    // AnimationSampler::bindSkeleton's overload comment.
    outState.sampler.bindSkeleton(
        outState.character.skeleton, outState.character.inverseBindMatrices);
    std::string idleWhy;
    std::string talkWhy;
    const bool haveIdle = loadClip(
        assets, outState.character.skeleton, kIdleClipPath, true, outState.idleClip, idleWhy);
    const bool haveTalk = loadClip(
        assets, outState.character.skeleton, kTalkClipPath, true, outState.talkClip, talkWhy);
    outState.animationStatus =
        std::string("idle ") + (haveIdle ? idleWhy : "UNAVAILABLE (" + idleWhy + ")") + "; talk " +
        (haveTalk ? talkWhy : "UNAVAILABLE (" + talkWhy + ")");

    lap("clips", outState.timing);

    // The index itself is built later, once, per voice type -- see
    // loadActorVoices. All this decides is WHICH folder his lines are under.
    if (outState.voice.voiceFolder.empty()) {
        outState.voice.voiceFolder = kVictorVoiceFolder;
    }

    outState.placed = true;
    // He is an ordinary actor from here on, so he needs what every actor has:
    // a name the activation prompt can offer, and a height the conversation
    // camera can aim by.
    outState.name = kVictorEditorId;
    outState.fullName = kVictorEditorId;
    outState.standingHeightUnits = actorStandingHeight(outState.character);
    outState.status = "placed at (" + std::to_string(outState.position[0]) + ", " +
                      std::to_string(outState.position[1]) + ", " +
                      std::to_string(outState.position[2]) + ") -- " +
                      std::to_string(outState.character.vertices.size()) + " verts, " +
                      std::to_string(outState.draws.size()) + " parts, " +
                      std::to_string(outState.textures.size()) + " textures, " +
                      std::to_string(outState.character.skeleton.bones.size()) + " bones, " +
                      std::to_string(outState.tree.nodes.size()) + " dialogue nodes";
    return true;
}

}  // namespace odai::games::newvegas
