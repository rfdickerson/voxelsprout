#include "games/newvegas/newvegas_actors.h"

#include "core/log.h"
#include "import/dds.h"
#include "games/newvegas/newvegas_ogg.h"
#include "import/fnv/bsa_archive.h"
#include "import/fnv/dialogue_records.h"
#include "import/fnv/kf_animation.h"
#include "import/fnv/nif_scene.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_map>

namespace odai::games::newvegas {

namespace {

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string directoryOf(const std::string& path) {
    const std::size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}


// One open archive set per voice folder, held for the process.
//
// Per FOLDER rather than per actor: three settlers sharing MaleAdult01DefaultB
// share one index, and the archive behind it can only be searched by name list.
// A vector because a voice type's lines can be split across the base game's
// archive and a DLC's.
std::unordered_map<std::string, std::vector<importer::fnv::BsaArchive>>& voiceArchives() {
    static std::unordered_map<std::string, std::vector<importer::fnv::BsaArchive>> archives;
    return archives;
}

// Pulls the 8-hex-digit formID field out of "<quest>_<topic>_<infoFormId>_1.ogg"
// and turns it into the id the dialogue importer used for that node.
//
// Scanning for the field rather than counting underscores is deliberate: quest
// and topic names contain underscores of their own.
std::string voiceNodeIdFromLeaf(const std::string& loweredLeaf) {
    std::string formIdHex;
    std::size_t start = 0;
    while (start < loweredLeaf.size()) {
        const std::size_t end = loweredLeaf.find('_', start);
        const std::string field = loweredLeaf.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (field.size() == 8u && std::all_of(field.begin(), field.end(), [](unsigned char c) {
                return std::isxdigit(c) != 0;
            })) {
            formIdHex = field;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1u;
    }
    if (formIdHex.empty()) {
        return {};
    }
    // DialogueTree ids are "info_%08X" -- UPPERCASE hex -- while the filenames
    // are lowercase. Keying without this conversion builds an index that never
    // matches anything and fails silently.
    std::string nodeId = "info_";
    for (const char c : formIdHex) {
        nodeId.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    return nodeId;
}

// Indexes one voice folder across every archive that carries voices.
void buildVoiceIndexForFolder(
    const std::filesystem::path& dataFilesPath,
    const std::string& voiceFolder,
    ActorVoiceIndex& outIndex
) {
    outIndex.voiceFolder = voiceFolder;
    const std::string folderNoSlash =
        toLowerAscii("sound\\voice\\falloutnv.esm\\" + voiceFolder);
    // Trailing separator included: this is a prefix test, and without it
    // "maleadult01" would also match "maleadult01defaultb".
    const std::string folderPrefix = folderNoSlash + "\\";

    std::error_code listError;
    std::vector<std::filesystem::path> archivePaths;
    for (const auto& entry : std::filesystem::directory_iterator(dataFilesPath, listError)) {
        if (entry.is_regular_file() && toLowerAscii(entry.path().extension().string()) == ".bsa") {
            archivePaths.push_back(entry.path());
        }
    }
    if (listError) {
        outIndex.status = "cannot list archives";
        return;
    }
    std::sort(archivePaths.begin(), archivePaths.end());

    std::vector<importer::fnv::BsaArchive> matched;
    std::vector<std::string> matchedNames;
    for (const auto& archivePath : archivePaths) {
        std::uint32_t contentFlags = 0;
        if (!importer::fnv::peekBsaContentFlags(archivePath, contentFlags) ||
            (contentFlags & importer::fnv::kBsaContentVoices) == 0u) {
            continue;
        }
        importer::fnv::BsaArchive archive;
        // Index only this one folder. Unfiltered, keeping the ~500 lines an
        // actor needs pulls all 105517 entries of Fallout - Voices1.bsa into
        // memory -- ~120 ms and tens of MB, per voice type.
        if (!archive.open(archivePath, folderNoSlash)) {
            continue;
        }
        std::size_t foundHere = 0;
        for (const importer::fnv::BsaFileEntry& entry : archive.files()) {
            const std::string lowered = toLowerAscii(entry.virtualPath);
            if (lowered.compare(0, folderPrefix.size(), folderPrefix) != 0) {
                continue;
            }
            if (lowered.size() < 4u || lowered.compare(lowered.size() - 4u, 4u, ".ogg") != 0) {
                continue;
            }
            std::string nodeId =
                voiceNodeIdFromLeaf(lowered.substr(lowered.find_last_of('\\') + 1u));
            if (nodeId.empty()) {
                continue;
            }
            outIndex.pathByNodeId.emplace(std::move(nodeId), entry.virtualPath);
            ++foundHere;
        }
        if (foundHere != 0u) {
            matched.push_back(std::move(archive));
            matchedNames.push_back(archivePath.filename().string());
        }
    }

    if (outIndex.pathByNodeId.empty()) {
        outIndex.status = "no voice archive holds " + folderPrefix;
        return;
    }
    voiceArchives()[toLowerAscii(voiceFolder)] = std::move(matched);
    outIndex.status = std::to_string(outIndex.pathByNodeId.size()) + " lines from ";
    for (std::size_t i = 0; i < matchedNames.size(); ++i) {
        outIndex.status += (i == 0 ? "" : ", ") + matchedNames[i];
    }
}

}  // namespace

bool buildSkinnedActor(
    const importer::fnv::FalloutAssetSource& assets,
    const std::string& skeletonPath,
    const std::vector<std::string>& bodyPartPaths,
    importer::fnv::FalloutCharacter& outCharacter,
    std::vector<odai::importer::ImportedSceneTexture>& outTextures,
    std::vector<odai::importer::ImportedScenePackedDraw>& outDraws,
    std::string& outWhy
) {
    outCharacter = importer::fnv::FalloutCharacter{};
    outTextures.clear();
    outDraws.clear();

    std::vector<std::uint8_t> bytes;
    std::string error;
    importer::fnv::NifSkeleton nifSkeleton;
    if (!assets.resolveMesh(skeletonPath, bytes, error) ||
        !importer::fnv::parseNifSkeleton(bytes, nifSkeleton, error)) {
        outWhy = "skeleton unavailable: " + error;
        return false;
    }
    if (!importer::fnv::buildFalloutSkeleton(nifSkeleton, outCharacter.skeleton)) {
        outWhy = "skeleton conversion failed";
        return false;
    }

    // A creature's part list can name every face/screen variant the model can
    // wear; drawing them all stacks them on one quad. Keep a named variant when
    // one exists, the way Victor's own screen is chosen.
    bool hasNamedScreen = false;
    for (const std::string& partName : bodyPartPaths) {
        if (toLowerAscii(partName).find("screen") != std::string::npos &&
            toLowerAscii(partName).find("static") == std::string::npos) {
            hasNamedScreen = true;
            break;
        }
    }

    for (const std::string& partPath : bodyPartPaths) {
        const std::string lowered = toLowerAscii(partPath);
        // Effect billboards: the opaque path draws them as grey sheets.
        if (lowered.find("smoketrail") != std::string::npos) {
            continue;
        }
        if (hasNamedScreen && lowered.find("screenstatic") != std::string::npos) {
            continue;
        }
        importer::fnv::NifSkinnedModel model;
        if (!assets.resolveMesh(partPath, bytes, error) ||
            !importer::fnv::parseNifSkinnedMesh(bytes, model, error)) {
            continue;
        }
        // Every human body NIF ships its own DISMEMBERMENT CAPS -- "bodycaps",
        // "limbcaps", "meatneck01", "meathead01", the raw meat the game reveals
        // when a limb comes off. They are ordinary skinned shapes in the same
        // file as the skin, so drawing the file literally hangs slabs of gore
        // off an otherwise fine settler. The one thing they all share is the
        // gore texture folder.
        std::erase_if(model.shapes, [](const importer::fnv::NifSkinnedShape& shape) {
            return toLowerAscii(shape.diffuseTexturePath).find("\\gore\\") != std::string::npos;
        });
        std::string bindError;
        if (importer::fnv::appendFalloutCharacterMesh(model, outCharacter, bindError)) {
            continue;
        }
        // No skinned shapes: a rigid prop parented to a bone. Re-read it as a
        // static mesh and hang it off the bone its own root node names.
        importer::fnv::NifSkeleton partNodes;
        importer::fnv::NifModel staticModel;
        std::string rigidError;
        if (!importer::fnv::parseNifSkeleton(bytes, partNodes, rigidError) ||
            partNodes.bones.empty() ||
            !importer::fnv::parseNifStaticMesh(bytes, staticModel, rigidError)) {
            continue;
        }
        std::erase_if(staticModel.shapes, [](const importer::fnv::NifShape& shape) {
            return shape.alphaBlend;
        });
        if (staticModel.shapes.empty()) {
            continue;
        }
        importer::fnv::appendFalloutCharacterRigidMesh(
            staticModel, partNodes.bones.front().name, outCharacter, rigidError);
    }
    if (outCharacter.vertices.empty()) {
        outWhy = "no geometry bound to the skeleton";
        return false;
    }

    // Textures and per-vertex material state, written onto the vertices by
    // walking each part's index range -- a skinned template carries no per-part
    // metadata on the GPU.
    std::unordered_map<std::string, std::uint32_t> localTextureIndexByPath;
    for (const importer::fnv::FalloutCharacterPart& part : outCharacter.parts) {
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
                    localTextureIndex = static_cast<std::uint32_t>(outTextures.size());
                    localTextureIndexByPath.emplace(key, localTextureIndex);
                    outTextures.push_back(std::move(texture));
                }
            }
        }

        std::uint32_t flags = 0u;
        if (part.alphaTest) { flags |= odai::importer::kImportedSceneMaterialFlagAlphaTest; }
        if (part.alphaBlend) { flags |= odai::importer::kImportedSceneMaterialFlagAlphaBlend; }
        if (part.twoSided) { flags |= odai::importer::kImportedSceneMaterialFlagTwoSided; }
        if (part.unlit) { flags |= odai::importer::kImportedSceneMaterialFlagUnlit; }
        for (std::uint32_t i = part.firstIndex; i < part.firstIndex + part.indexCount; ++i) {
            if (i >= outCharacter.indices.size()) {
                break;
            }
            const std::uint32_t vertexIndex = outCharacter.indices[i];
            if (vertexIndex >= outCharacter.vertices.size()) {
                continue;
            }
            outCharacter.vertices[vertexIndex].textureIndex = localTextureIndex;
            outCharacter.vertices[vertexIndex].flags = flags;
        }

        odai::importer::ImportedScenePackedDraw draw{};
        draw.firstIndex = part.firstIndex;
        draw.indexCount = part.indexCount;
        draw.alphaThreshold = part.alphaThreshold;
        outDraws.push_back(draw);
    }
    if (outDraws.empty()) {
        outWhy = "no drawable parts";
        return false;
    }
    return true;
}

float actorStandingHeight(const importer::fnv::FalloutCharacter& character) {
    // Rest-pose extent above the actor's own origin, which is its FEET. Good
    // enough to aim a camera at without posing anything: an idle moves a head
    // by a few units, and the alternative is a per-frame bounds recompute over
    // every skinned vertex.
    float highest = 0.0f;
    for (const odai::render::ImportedSkinnedMeshVertex& vertex : character.vertices) {
        highest = std::max(highest, vertex.position[1]);
    }
    return highest;
}

bool loadActorIdleClip(
    const importer::fnv::FalloutAssetSource& assets,
    const std::string& skeletonPath,
    const anim::Skeleton& skeleton,
    std::size_t variant,
    anim::AnimationClip& outClip,
    std::string& outWhy
) {
    // The conversation idles, which are what a Fallout human does when standing
    // still and not walking anywhere. The "listen" variants hold a pose; the
    // "talk" ones gesture, which reads as odd on someone standing alone.
    static const char* const kStandingIdles[] = {
        "idleanims\\ttnpchappysubtlelistena.kf",
        "idleanims\\ttnpchappyarmslooselistena.kf",
        "idleanims\\ttpdgstarmscrossedlistena.kf",
    };
    constexpr std::size_t kStandingIdleCount =
        sizeof(kStandingIdles) / sizeof(kStandingIdles[0]);

    const std::string directory = directoryOf(skeletonPath);
    // "mtidle" first: it is the creature convention, it is right when it
    // resolves, and no human skeleton has one.
    std::vector<std::string> candidates{directory + "mtidle.kf"};
    for (std::size_t i = 0; i < kStandingIdleCount; ++i) {
        candidates.push_back(directory + kStandingIdles[(variant + i) % kStandingIdleCount]);
    }

    for (const std::string& clipPath : candidates) {
        std::vector<std::uint8_t> bytes;
        if (!assets.resolveMesh(clipPath, bytes, outWhy)) {
            continue;
        }
        importer::fnv::KfAnimation animation;
        if (!importer::fnv::parseKfAnimation(bytes, animation, outWhy)) {
            continue;
        }
        importer::fnv::FalloutAnimationStats stats;
        if (!importer::fnv::buildFalloutAnimationClip(animation, skeleton, outClip, stats)) {
            outWhy = "no track bound";
            continue;
        }
        // A clip that binds a handful of bones is not a pose -- the unarmed
        // "Idle" sequence binds one node and would otherwise count as success
        // while leaving the actor in bind pose.
        if (outClip.tracks.size() < 8u) {
            outWhy = "only " + std::to_string(outClip.tracks.size()) + " tracks bound";
            outClip = anim::AnimationClip{};
            continue;
        }
        // Authored one-shot, because the game plays them between dialogue
        // lines. Standing around is the loop.
        outClip.loop = true;
        return true;
    }
    return false;
}

bool loadGoodspringsActors(
    const std::filesystem::path& pluginPath,
    const importer::fnv::FalloutAssetSource& assets,
    const float bethesdaCentre[2],
    float radius,
    std::uint32_t firstInstanceSlot,
    std::size_t maxActors,
    const std::vector<std::uint32_t>& excludeBaseFormIds,
    std::vector<SkinnedActor>& outActors,
    ActorPopulationStats& outStats
) {
    outActors.clear();
    outStats = ActorPopulationStats{};

    importer::fnv::FalloutActorScan scan;
    std::string error;
    if (!importer::fnv::findActorsNear(
            pluginPath, bethesdaCentre[0], bethesdaCentre[1], radius, scan, error)) {
        outStats.detail = "scan failed: " + error;
        return false;
    }
    outStats.placementsConsidered = scan.placements.size();

    // One build per distinct base: Goodsprings places six bighorners, and
    // parsing the same NIF six times is six times the cost for one result. Each
    // still gets its own GPU instance slot -- the renderer's template is
    // per-slot -- but the CPU-side assembly is shared.
    struct BuiltBase {
        bool ok = false;
        importer::fnv::FalloutCharacter character;
        std::vector<odai::importer::ImportedSceneTexture> textures;
        std::vector<odai::importer::ImportedScenePackedDraw> draws;
        anim::AnimationClip idleClip;
        bool hasClip = false;
    };
    std::unordered_map<std::uint32_t, BuiltBase> builtByBase;
    std::uint32_t nextSlot = firstInstanceSlot;

    for (const importer::fnv::FalloutActorPlacement& placement : scan.placements) {
        if (placement.initiallyDisabled) {
            ++outStats.skippedDisabled;
            continue;
        }
        if (std::find(excludeBaseFormIds.begin(), excludeBaseFormIds.end(), placement.baseFormId) !=
            excludeBaseFormIds.end()) {
            ++outStats.skippedExcluded;
            continue;
        }
        const importer::fnv::ResolvedActorBase resolved = scan.resolve(placement.baseFormId);
        if (resolved.geometrySource == importer::fnv::ActorGeometrySource::None ||
            resolved.bodyPartPaths.empty()) {
            ++outStats.skippedNoGeometry;
            continue;
        }
        if (outActors.size() >= maxActors) {
            ++outStats.skippedNoSlots;
            continue;
        }

        BuiltBase& built = builtByBase[resolved.resolvedBaseFormId];
        if (built.character.vertices.empty() && !built.ok) {
            std::string why;
            built.ok = buildSkinnedActor(
                assets, resolved.skeletonPath, resolved.bodyPartPaths, built.character,
                built.textures, built.draws, why);
            if (built.ok) {
                std::string clipWhy;
                built.hasClip = loadActorIdleClip(
                    assets, resolved.skeletonPath, built.character.skeleton, builtByBase.size(),
                    built.idleClip, clipWhy);
            }
        }
        if (!built.ok) {
            ++outStats.skippedBuildFailed;
            continue;
        }

        SkinnedActor actor;
        actor.name = resolved.base != nullptr ? resolved.base->editorId : std::string("actor");
        actor.fullName = resolved.base != nullptr ? resolved.base->fullName : std::string();
        // Which folder this actor's recorded lines live under. Resolved here
        // because this is where the scan is; the index itself is built later,
        // once per distinct folder, by loadActorVoices.
        actor.voice.voiceTypeFormId = scan.voiceTypeFormIdFor(placement.baseFormId);
        actor.voice.voiceFolder = scan.voiceFolderFor(placement.baseFormId);
        actor.baseFormId = placement.baseFormId;
        actor.placed = true;
        actor.character = built.character;
        actor.standingHeightUnits = actorStandingHeight(built.character);
        actor.textures = built.textures;
        actor.draws = built.draws;
        actor.idleClip = built.idleClip;
        actor.instanceSlot = nextSlot++;
        // Bethesda Z-up -> engine Y-up: (x, y, z) -> (x, z, -y).
        actor.position[0] = placement.position[0];
        actor.position[1] = placement.position[2];
        actor.position[2] = -placement.position[1];
        // Bethesda's Z rotation is the actor's heading; negated because the
        // basis change flips the handedness of the horizontal plane.
        actor.yawRadians = -placement.rotationRadians[2];
        // Staggered so a row of identical creatures does not breathe in unison,
        // which is the tell that they are one animation rather than a herd.
        actor.animationSeconds =
            static_cast<float>(outActors.size()) * 0.7f;
        actor.sampler.bindSkeleton(actor.character.skeleton, actor.character.inverseBindMatrices);
        if (built.hasClip) {
            ++outStats.animated;
        }
        outActors.push_back(std::move(actor));
        ++outStats.built;
    }

    outStats.detail =
        std::to_string(outStats.built) + " built (" + std::to_string(outStats.animated) +
        " animated) of " + std::to_string(outStats.placementsConsidered) + " placements; skipped " +
        std::to_string(outStats.skippedDisabled) + " disabled, " +
        std::to_string(outStats.skippedNoGeometry) + " without geometry, " +
        std::to_string(outStats.skippedBuildFailed) + " unbuildable, " +
        std::to_string(outStats.skippedNoSlots) + " over slot budget, " +
        std::to_string(outStats.skippedExcluded) + " handled elsewhere";
    return true;
}

std::size_t loadActorDialogue(
    const std::filesystem::path& pluginPath,
    std::vector<SkinnedActor>& actors,
    std::string& outDetail
) {
    // One request per DISTINCT base. Goodsprings places two ravens off the same
    // record; asking twice would read the same INFOs twice and hand back the
    // same tree.
    std::vector<importer::fnv::SpeakerDialogueRequest> requests;
    for (const SkinnedActor& actor : actors) {
        if (actor.baseFormId == 0u || !actor.tree.nodes.empty()) {
            continue;
        }
        const bool alreadyRequested = std::any_of(
            requests.begin(), requests.end(),
            [&](const importer::fnv::SpeakerDialogueRequest& request) {
                return request.baseFormId == actor.baseFormId;
            });
        if (!alreadyRequested) {
            requests.push_back(importer::fnv::SpeakerDialogueRequest{
                actor.baseFormId, actor.voice.voiceTypeFormId, actor.displayName()});
        }
    }
    if (requests.empty()) {
        outDetail = "no actors needed dialogue";
        return 0;
    }

    std::unordered_map<std::uint32_t, odai::dialogue::DialogueTree> trees;
    std::unordered_map<std::uint32_t, importer::fnv::DialogueImportStats> stats;
    std::string error;
    if (!importer::fnv::buildSpeakerDialogueTrees(pluginPath, requests, trees, stats, error)) {
        outDetail = "dialogue scan failed: " + error;
        return 0;
    }

    std::size_t attached = 0;
    for (SkinnedActor& actor : actors) {
        if (!actor.tree.nodes.empty()) {
            continue;
        }
        const auto tree = trees.find(actor.baseFormId);
        if (tree == trees.end() || tree->second.nodes.empty()) {
            continue;
        }
        // Copied rather than moved: two placements of the same base are two
        // actors who each need their own runtime to talk from.
        actor.tree = tree->second;
        ++attached;
    }
    outDetail = std::to_string(attached) + " of " + std::to_string(actors.size()) +
        " actors can talk (" + std::to_string(requests.size()) + " bases asked, one plugin walk)";
    return attached;
}

std::size_t loadActorVoices(
    const std::filesystem::path& dataFilesPath,
    std::vector<SkinnedActor>& actors,
    std::string& outDetail
) {
    // One index per DISTINCT folder, built once and copied to everyone using
    // it. Building per actor would re-scan the same archive for the same ~500
    // lines once per settler.
    std::unordered_map<std::string, ActorVoiceIndex> indexByFolder;
    std::size_t voiced = 0;
    for (SkinnedActor& actor : actors) {
        // Skip actors with nothing to say: an index nothing will look up is
        // pure cost, and most of a town is in that state.
        if (!actor.canTalk() || actor.voice.voiceFolder.empty() ||
            !actor.voice.pathByNodeId.empty()) {
            continue;
        }
        const std::string key = toLowerAscii(actor.voice.voiceFolder);
        auto existing = indexByFolder.find(key);
        if (existing == indexByFolder.end()) {
            ActorVoiceIndex index;
            buildVoiceIndexForFolder(dataFilesPath, actor.voice.voiceFolder, index);
            existing = indexByFolder.emplace(key, std::move(index)).first;
        }
        actor.voice = existing->second;
        if (!actor.voice.pathByNodeId.empty()) {
            ++voiced;
        }
    }

    if (indexByFolder.empty()) {
        outDetail = "no speaking actor has a voice type";
        return 0;
    }
    outDetail = std::to_string(voiced) + " actors voiced from " +
        std::to_string(indexByFolder.size()) + " voice type(s):";
    for (const auto& [folder, index] : indexByFolder) {
        outDetail += " " + index.voiceFolder + "(" +
            std::to_string(index.pathByNodeId.size()) + ")";
    }
    return voiced;
}

void speakActorLine(
    SkinnedActor& actor,
    const std::filesystem::path& cacheDirectory,
    odai::audio::Audio& audioSystem
) {
    if (!actor.talking || cacheDirectory.empty()) {
        return;
    }
    const dialogue::DialogueNode* node = actor.runtime.currentNode();
    if (node == nullptr || node->id == actor.spokenNodeId) {
        return;
    }
    actor.spokenNodeId = node->id;

    const auto found = actor.voice.pathByNodeId.find(node->id);
    if (found == actor.voice.pathByNodeId.end()) {
        return;  // a line the game never recorded, or a topic node
    }
    const auto archives = voiceArchives().find(toLowerAscii(actor.voice.voiceFolder));
    if (archives == voiceArchives().end()) {
        return;
    }
    importer::fnv::BsaArchive* holder = nullptr;
    const importer::fnv::BsaFileEntry* entry = nullptr;
    for (importer::fnv::BsaArchive& archive : archives->second) {
        if (const importer::fnv::BsaFileEntry* hit = archive.find(found->second)) {
            holder = &archive;
            entry = hit;
            break;
        }
    }
    if (entry == nullptr || holder == nullptr) {
        return;
    }

    // Cached by leaf name, so a line costs one extract + one Vorbis decode per
    // install rather than one per playback.
    std::string leaf = found->second;
    const std::size_t lastSeparator = leaf.find_last_of("\\/");
    if (lastSeparator != std::string::npos) {
        leaf = leaf.substr(lastSeparator + 1u);
    }
    const std::filesystem::path oggPath = cacheDirectory / leaf;
    std::filesystem::path wavPath = oggPath;
    wavPath.replace_extension(".wav");

    std::error_code existsError;
    if (!std::filesystem::exists(wavPath, existsError) || existsError) {
        std::error_code createError;
        std::filesystem::create_directories(cacheDirectory, createError);
        std::vector<std::uint8_t> oggBytes;
        std::string extractError;
        if (!holder->extract(*entry, oggBytes, extractError) || oggBytes.empty()) {
            VOX_LOGW("newvegas") << actor.displayName() << " voice extract failed for " << node->id
                                 << ": " << extractError;
            return;
        }
        {
            std::ofstream out(oggPath, std::ios::binary | std::ios::trunc);
            if (!out) {
                return;
            }
            out.write(
                reinterpret_cast<const char*>(oggBytes.data()),
                static_cast<std::streamsize>(oggBytes.size()));
        }
        if (!decodeOggToWav(oggPath, wavPath)) {
            VOX_LOGW("newvegas") << actor.displayName() << " voice decode failed for " << node->id;
            return;
        }
    }

    // Ui, not Ambient: there is no Voice bus in SoundCategory, and Ui is the
    // non-spatialized 2D one, which is what a conversation line is here -- the
    // camera is a step away and the line should not duck with distance.
    const odai::audio::SoundHandle clip =
        audioSystem.loadSound(wavPath, odai::audio::SoundCategory::Ui);
    if (clip.valid()) {
        audioSystem.playSound(clip);
    }
}

void updateActorPoses(std::vector<SkinnedActor>& actors, float deltaSeconds) {
    // ODAI_FNV_NOANIM=1 freezes every actor at bind pose while leaving the rest
    // of the path running -- same upload, same per-frame pose submission, same
    // draws. It is the control for "is this actually animating": a screenshot
    // diff of an actor's own pixels across two moments is otherwise impossible
    // to attribute, because the world is streaming in behind it and the light is
    // moving, so SOMETHING always changes.
    static const bool freezeAtBindPose = std::getenv("ODAI_FNV_NOANIM") != nullptr ||
        std::getenv("ODAI_FNV_VICTOR_NOANIM") != nullptr;

    for (SkinnedActor& actor : actors) {
        if (actor.character.skeleton.bones.empty()) {
            continue;
        }
        // Restart the clock when the clip changes, so a conversation opens on
        // the first frame of the gesture cycle rather than wherever the idle had
        // got to -- entering talk 4 seconds into a 5-second clip reads as an
        // actor finishing a gesture it never started.
        const bool wantsTalkClip = actor.talking && !actor.talkClip.tracks.empty();
        if (wantsTalkClip != actor.posedTalking) {
            actor.posedTalking = wantsTalkClip;
            actor.animationSeconds = 0.0f;
        }
        actor.animationSeconds += deltaSeconds;

        const anim::AnimationClip& clip = wantsTalkClip ? actor.talkClip : actor.idleClip;
        if (freezeAtBindPose || clip.tracks.empty() || clip.duration <= 0.0f) {
            // No clip: hold the bind pose rather than leaving the previous
            // frame's matrices, which on the first frame would be none at all.
            importer::fnv::computeFalloutBindPose(actor.character, actor.poseScratch);
        } else {
            actor.sampler.sample(
                actor.character.skeleton, clip, actor.animationSeconds, actor.poseScratch);
        }

        // World placement rides on the bone matrices, pre-multiplied: the
        // skinning pass consumes bone matrices and nothing else, so there is no
        // separate instance transform to put it in.
        const odai::math::Matrix4 actorWorld =
            odai::math::Matrix4::translation(odai::math::Vector3{
                actor.position[0], actor.position[1], actor.position[2]}) *
            odai::math::Matrix4::rotationY(actor.yawRadians);
        for (odai::math::Matrix4& matrix : actor.poseScratch) {
            matrix = actorWorld * matrix;
        }
    }
}

void remapActorTextureSlots(SkinnedActor& actor, const std::vector<std::uint32_t>& bindlessSlots) {
    for (odai::render::ImportedSkinnedMeshVertex& vertex : actor.character.vertices) {
        vertex.textureIndex = (vertex.textureIndex < bindlessSlots.size())
            ? bindlessSlots[vertex.textureIndex]
            : 0xffffffffu;
    }
}

float conversationFaceHeight(const SkinnedActor& actor) {
    // Victor's tuned value was 150 units against a ~230-unit Securitron, so
    // ~0.65 of standing height -- roughly the shoulders, which is where the
    // camera wants to sit for a portrait. Falling back to his old constant
    // keeps an actor whose geometry never measured from aiming at its feet.
    constexpr float kFaceFraction = 0.65f;
    constexpr float kFallbackUnits = 150.0f;
    return actor.standingHeightUnits > 1.0f ? (actor.standingHeightUnits * kFaceFraction)
                                            : kFallbackUnits;
}

bool actorIsInReach(
    const SkinnedActor& actor, const float cameraPosition[3], float cameraYawRadians
) {
    if (!actor.placed) {
        return false;
    }
    const float dx = actor.position[0] - cameraPosition[0];
    const float dy = actor.position[1] - cameraPosition[1];
    const float dz = actor.position[2] - cameraPosition[2];
    if (((dx * dx) + (dy * dy) + (dz * dz)) > (kActorTalkRange * kActorTalkRange)) {
        return false;
    }
    const float horizontal = std::sqrt((dx * dx) + (dz * dz));
    if (horizontal < 1e-3f) {
        return true;  // standing on top of them counts as facing them
    }
    // Same basis the camera uses: forward is (cos(yaw), sin(yaw)) in XZ.
    const float forwardX = std::cos(cameraYawRadians);
    const float forwardZ = std::sin(cameraYawRadians);
    return (((dx / horizontal) * forwardX) + ((dz / horizontal) * forwardZ)) >= kActorTalkFacingDot;
}

int findActorInReach(
    const std::vector<SkinnedActor>& actors, const float cameraPosition[3], float cameraYawRadians
) {
    int best = -1;
    float bestDistanceSquared = 0.0f;
    for (std::size_t i = 0; i < actors.size(); ++i) {
        if (!actors[i].canTalk() || !actorIsInReach(actors[i], cameraPosition, cameraYawRadians)) {
            continue;
        }
        const float dx = actors[i].position[0] - cameraPosition[0];
        const float dy = actors[i].position[1] - cameraPosition[1];
        const float dz = actors[i].position[2] - cameraPosition[2];
        const float distanceSquared = (dx * dx) + (dy * dy) + (dz * dz);
        if (best < 0 || distanceSquared < bestDistanceSquared) {
            best = static_cast<int>(i);
            bestDistanceSquared = distanceSquared;
        }
    }
    return best;
}

}  // namespace odai::games::newvegas
