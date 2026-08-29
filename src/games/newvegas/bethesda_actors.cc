#include "games/newvegas/bethesda_actors.h"

#include "import/fnv/skyrim_animation_assets.h"
#include "bethesda/navigation_world.h"

#include "core/lcg.h"
#include "core/log.h"
#include "import/dds.h"
#include "games/newvegas/newvegas_ogg.h"
#include "import/fnv/bsa_archive.h"
#include "import/fnv/dialogue_records.h"
#include "import/fnv/kf_animation.h"
#include "import/fnv/nif_scene.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

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

bool textKeyContainsLabel(std::string_view text, std::string_view wanted) {
    std::size_t lineStart = 0u;
    while (lineStart <= text.size()) {
        const std::size_t lineEnd = text.find_first_of("\r\n", lineStart);
        std::string line(text.substr(
            lineStart, lineEnd == std::string_view::npos
                ? text.size() - lineStart
                : lineEnd - lineStart));
        const auto first = std::find_if_not(line.begin(), line.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        });
        const auto last = std::find_if_not(line.rbegin(), line.rend(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }).base();
        line = first < last ? std::string(first, last) : std::string();
        if (toLowerAscii(std::move(line)) == wanted) {
            return true;
        }
        if (lineEnd == std::string_view::npos) {
            break;
        }
        lineStart = lineEnd + 1u;
        while (lineStart < text.size() &&
               (text[lineStart] == '\r' || text[lineStart] == '\n')) {
            ++lineStart;
        }
    }
    return false;
}

template <typename Key, typename Interpolate>
std::vector<Key> sliceKeyRange(
    const std::vector<Key>& keys, float startTime, float stopTime,
    Interpolate interpolate) {
    if (keys.empty() || !(stopTime > startTime)) {
        return {};
    }
    const auto sample = [&](float time) {
        if (time <= keys.front().time) return keys.front().value;
        if (time >= keys.back().time) return keys.back().value;
        const auto upper = std::upper_bound(
            keys.begin(), keys.end(), time,
            [](float value, const Key& key) { return value < key.time; });
        const Key& right = *upper;
        const Key& left = *(upper - 1);
        const float span = right.time - left.time;
        const float fraction = span > 0.0f ? (time - left.time) / span : 0.0f;
        return interpolate(left.value, right.value, fraction);
    };

    const float duration = stopTime - startTime;
    std::vector<Key> sliced;
    sliced.reserve(keys.size() + 2u);
    sliced.push_back(Key{0.0f, sample(startTime)});
    for (const Key& key : keys) {
        if (key.time > startTime && key.time < stopTime) {
            Key copied = key;
            copied.time -= startTime;
            sliced.push_back(std::move(copied));
        }
    }
    sliced.push_back(Key{duration, sample(stopTime)});
    return sliced;
}

void sliceTes3Track(
    importer::fnv::KfBoneTrack& track, float startTime, float stopTime) {
    const auto lerpVector = [](const odai::math::Vector3& a,
                               const odai::math::Vector3& b, float t) {
        return a + ((b - a) * t);
    };
    track.translationKeys = sliceKeyRange(
        track.translationKeys, startTime, stopTime, lerpVector);
    track.rotationKeys = sliceKeyRange(
        track.rotationKeys, startTime, stopTime, odai::math::slerp);
    track.scaleKeys = sliceKeyRange(
        track.scaleKeys, startTime, stopTime, lerpVector);
}

odai::math::Quaternion multiplyQuaternion(
    const odai::math::Quaternion& a, const odai::math::Quaternion& b) {
    return odai::math::normalize(odai::math::Quaternion{
        (a.w * b.x) + (a.x * b.w) + (a.y * b.z) - (a.z * b.y),
        (a.w * b.y) - (a.x * b.z) + (a.y * b.w) + (a.z * b.x),
        (a.w * b.z) + (a.x * b.y) - (a.y * b.x) + (a.z * b.w),
        (a.w * b.w) - (a.x * b.x) - (a.y * b.y) - (a.z * b.z)});
}

odai::math::Quaternion rotationX(float degrees) {
    const float half = odai::math::radians(degrees) * 0.5f;
    return odai::math::Quaternion{std::sin(half), 0.0f, 0.0f, std::cos(half)};
}

void addProceduralRotationTrack(
    anim::AnimationClip& clip, const anim::Skeleton& skeleton, const char* boneName,
    std::initializer_list<std::pair<float, float>> keys) {
    const int boneIndex = skeleton.findBone(boneName);
    if (boneIndex < 0) {
        return;
    }
    const odai::math::Quaternion bind =
        skeleton.bones[static_cast<std::size_t>(boneIndex)].localRotation;
    anim::BoneTrack track;
    track.boneIndex = boneIndex;
    for (const auto& [time, degrees] : keys) {
        // Delta AFTER the authored bind rotation: Skyrim's limbs have
        // non-identity bind transforms, so replacing rather than composing
        // would point them along the generic rig's axes and tear the actor.
        track.rotationKeys.push_back(
            anim::QuaternionKey{time, multiplyQuaternion(bind, rotationX(degrees))});
    }
    clip.tracks.push_back(std::move(track));
}

bool makeProceduralSkyrimIdle(
    const anim::Skeleton& skeleton, anim::AnimationClip& outClip) {
    const int com = skeleton.findBone("NPC COM [COM ]");
    if (com < 0 || skeleton.findBone("NPC L Thigh [LThg]") < 0 ||
        skeleton.findBone("NPC R Thigh [RThg]") < 0) {
        return false;
    }
    outClip = anim::AnimationClip{};
    outClip.name = "procedural skyrim idle";
    outClip.duration = 2.4f;
    outClip.loop = true;
    anim::BoneTrack track;
    track.boneIndex = com;
    const odai::math::Vector3 bind =
        skeleton.bones[static_cast<std::size_t>(com)].localTranslation;
    track.translationKeys = {
        {0.0f, bind}, {1.2f, bind + odai::math::Vector3{0.0f, 1.25f, 0.0f}}, {2.4f, bind}};
    outClip.tracks.push_back(std::move(track));
    // A COM-only sub-unit bob was technically animated but visually static,
    // especially on the smaller child meshes. These bind-relative breathing
    // and balance tracks use bones shared by the adult and child Skyrim rigs.
    addProceduralRotationTrack(outClip, skeleton, "NPC Spine1 [Spn1]",
        {{0.0f, -0.8f}, {1.2f, 1.2f}, {2.4f, -0.8f}});
    addProceduralRotationTrack(outClip, skeleton, "NPC Spine2 [Spn2]",
        {{0.0f, 0.6f}, {1.2f, -1.0f}, {2.4f, 0.6f}});
    addProceduralRotationTrack(outClip, skeleton, "NPC Head [Head]",
        {{0.0f, -0.7f}, {1.2f, 0.9f}, {2.4f, -0.7f}});
    addProceduralRotationTrack(outClip, skeleton, "NPC L UpperArm [LUar]",
        {{0.0f, -1.5f}, {1.2f, 1.8f}, {2.4f, -1.5f}});
    addProceduralRotationTrack(outClip, skeleton, "NPC R UpperArm [RUar]",
        {{0.0f, 1.5f}, {1.2f, -1.8f}, {2.4f, 1.5f}});
    return outClip.tracks.size() >= 4u;
}

bool makeProceduralSkyrimWalk(
    const anim::Skeleton& skeleton, anim::AnimationClip& outClip) {
    if (skeleton.findBone("NPC L Thigh [LThg]") < 0 ||
        skeleton.findBone("NPC R Thigh [RThg]") < 0 ||
        skeleton.findBone("NPC L Calf [LClf]") < 0 ||
        skeleton.findBone("NPC R Calf [RClf]") < 0) {
        return false;
    }
    outClip = anim::AnimationClip{};
    outClip.name = "procedural skyrim walk";
    outClip.duration = 0.86f;
    outClip.loop = true;
    addProceduralRotationTrack(outClip, skeleton, "NPC L Thigh [LThg]",
        {{0.0f, -24.0f}, {0.215f, 0.0f}, {0.43f, 24.0f}, {0.645f, 0.0f}, {0.86f, -24.0f}});
    addProceduralRotationTrack(outClip, skeleton, "NPC R Thigh [RThg]",
        {{0.0f, 24.0f}, {0.215f, 0.0f}, {0.43f, -24.0f}, {0.645f, 0.0f}, {0.86f, 24.0f}});
    addProceduralRotationTrack(outClip, skeleton, "NPC L Calf [LClf]",
        {{0.0f, 4.0f}, {0.215f, 34.0f}, {0.43f, 4.0f}, {0.645f, 8.0f}, {0.86f, 4.0f}});
    addProceduralRotationTrack(outClip, skeleton, "NPC R Calf [RClf]",
        {{0.0f, 4.0f}, {0.215f, 8.0f}, {0.43f, 4.0f}, {0.645f, 34.0f}, {0.86f, 4.0f}});
    addProceduralRotationTrack(outClip, skeleton, "NPC L UpperArm [LUar]",
        {{0.0f, 13.0f}, {0.43f, -13.0f}, {0.86f, 13.0f}});
    addProceduralRotationTrack(outClip, skeleton, "NPC R UpperArm [RUar]",
        {{0.0f, -13.0f}, {0.43f, 13.0f}, {0.86f, -13.0f}});
    return outClip.tracks.size() >= 4u;
}

bool loadSkyrimHkxSkeleton(
    const importer::fnv::FalloutAssetSource& assets, bool female,
    anim::HkxDecodedSkeleton& outSkeleton, std::string& outError) {
    const char* path = female
        ? "meshes\\actors\\character\\character assets female\\skeleton_female.hkx"
        : "meshes\\actors\\character\\character assets\\skeleton.hkx";
    std::vector<std::uint8_t> bytes;
    return assets.resolveAsset(path, bytes, outError) &&
        anim::decodeHkxAnimationSkeleton(bytes, outSkeleton, outError);
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
// A voice filename carries the INFO's formID with its MOD-INDEX BYTE ZEROED:
// Willow's info_0109F030 is recorded as AWillowD_HELLO_0009F030_1.ogg. The byte
// is a load-order position, which is not a property of the line and is not
// stable, so the GECK leaves it out -- and a lookup that keeps it matches
// nothing while looking exactly like a mod that shipped no audio.
//
// A no-op for the base game, whose records are index 00 already, which is why
// this went unnoticed: every vanilla actor's voice resolved regardless.
std::string voiceKeyForNodeId(const std::string& nodeId) {
    constexpr std::size_t kPrefix = 5u;  // "info_"
    if (nodeId.size() != kPrefix + 8u || nodeId.compare(0, kPrefix, "info_") != 0) {
        return nodeId;
    }
    std::string key = nodeId;
    key[kPrefix] = '0';
    key[kPrefix + 1u] = '0';
    return key;
}

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
    const std::string& pluginFileName,
    const std::string& voiceFolder,
    const std::vector<std::string>& modDirectories,
    ActorVoiceIndex& outIndex
) {
    outIndex.voiceFolder = voiceFolder;
    // The plugin's own file name is a path component: Fallout 3's lines live
    // under sound\voice\fallout3.esm\, New Vegas's under falloutnv.esm\.
    const std::string folderNoSlash =
        toLowerAscii("sound\\voice\\" + pluginFileName + "\\" + voiceFolder);
    // Trailing separator included: this is a prefix test, and without it
    // "maleadult01" would also match "maleadult01defaultb".
    const std::string folderPrefix = folderNoSlash + "\\";

    // LOOSE FIRST, and it wins: a mod shipping a plain voice tree is the common
    // case for a companion, and the game applies the same loose-beats-archive
    // rule. Walking the whole mod root is affordable -- Willow's is 3367 files
    // -- and it sidesteps having to guess the on-disk CASE of every component,
    // which is "Sound\Voice\NVWillow.esp\WillowsVoice" on disk against an
    // all-lowercase lookup.
    for (const std::string& modDirectory : modDirectories) {
        std::error_code walkError;
        std::filesystem::recursive_directory_iterator walk(
            std::filesystem::path(modDirectory),
            std::filesystem::directory_options::skip_permission_denied, walkError);
        if (walkError) {
            continue;
        }
        for (const auto& entry : walk) {
            std::error_code typeError;
            if (!entry.is_regular_file(typeError) || typeError) {
                continue;
            }
            std::string relative =
                std::filesystem::relative(entry.path(), std::filesystem::path(modDirectory))
                    .string();
            for (char& c : relative) {
                if (c == '/') {
                    c = '\\';
                }
            }
            const std::string lowered = toLowerAscii(relative);
            if (lowered.size() < folderPrefix.size() ||
                lowered.compare(0, folderPrefix.size(), folderPrefix) != 0) {
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
            outIndex.loosePathByNodeId.emplace(std::move(nodeId), entry.path());
        }
    }

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
    outIndex.archiveKey = toLowerAscii(pluginFileName + "\\" + voiceFolder);
    voiceArchives()[outIndex.archiveKey] = std::move(matched);
    outIndex.status =
        std::to_string(outIndex.pathByNodeId.size() + outIndex.loosePathByNodeId.size()) +
        " lines from ";
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
    std::string& outWhy,
    const std::vector<std::string>* rigidAttachmentBones
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

    std::unordered_set<std::string> appendedSkinnedPaths;
    for (std::size_t bodyPartIndex = 0u;
         bodyPartIndex < bodyPartPaths.size(); ++bodyPartIndex) {
        const std::string& partPath = bodyPartPaths[bodyPartIndex];
        const std::string authoredAttachment =
            rigidAttachmentBones != nullptr &&
                bodyPartIndex < rigidAttachmentBones->size()
                ? (*rigidAttachmentBones)[bodyPartIndex]
                : std::string();
        const std::string lowered = toLowerAscii(partPath);
        // Effect billboards: the opaque path draws them as grey sheets.
        if (lowered.find("smoketrail") != std::string::npos) {
            continue;
        }
        if (hasNamedScreen && lowered.find("screenstatic") != std::string::npos) {
            continue;
        }
        importer::fnv::NifSkinnedModel model;
        if (!assets.resolveMesh(partPath, bytes, error)) {
            continue;
        }
        const std::size_t firstAddedPart = outCharacter.parts.size();
        std::string bindError;
        const bool parsedSkinned =
            importer::fnv::parseNifSkinnedMesh(bytes, model, bindError);
        if (parsedSkinned) {
            // Every human body NIF ships its own DISMEMBERMENT CAPS --
            // "bodycaps", "limbcaps", "meatneck01", "meathead01", the raw
            // meat the game reveals when a limb comes off. They are ordinary
            // skinned shapes in the same file as the skin, so drawing the file
            // literally hangs slabs of gore off an otherwise fine settler.
            std::erase_if(model.shapes, [](const importer::fnv::NifSkinnedShape& shape) {
                return toLowerAscii(shape.diffuseTexturePath).find("\\gore\\") !=
                    std::string::npos;
            });
            // TES3 equipment can name the same aggregate weighted mesh for
            // several covered BODY slots (a cuirass skin commonly includes
            // chest and both hands). Its slot attachment is irrelevant once
            // the NIF's real weights bind, so appending it once per slot stacks
            // duplicate surfaces and wastes the actor palette/upload budget.
            if (!model.shapes.empty() && appendedSkinnedPaths.contains(lowered)) {
                continue;
            }
        }
        if (parsedSkinned &&
            importer::fnv::appendFalloutCharacterMesh(model, outCharacter, bindError)) {
            appendedSkinnedPaths.insert(lowered);
            for (std::size_t part = firstAddedPart; part < outCharacter.parts.size(); ++part) {
                outCharacter.parts[part].sourcePath = partPath;
            }
            continue;
        }
        // No skinned shapes: a rigid prop parented to a bone. Re-read it as a
        // static mesh and hang it off the bone its own root node names.
        importer::fnv::NifSkeleton partNodes;
        importer::fnv::NifModel staticModel;
        std::string rigidError;
        if (!importer::fnv::parseNifStaticMesh(bytes, staticModel, rigidError)) {
            continue;
        }
        if (authoredAttachment.empty() &&
            (!importer::fnv::parseNifSkeleton(bytes, partNodes, rigidError) ||
             partNodes.bones.empty())) {
            continue;
        }
        // Fallout glare/effect cards are not body parts. TES3 hair and several
        // armour fringes are also alpha blended, but their BODY slot supplies
        // an explicit bone attachment and makes them unambiguously wearable.
        if (authoredAttachment.empty()) {
            std::erase_if(staticModel.shapes, [](const importer::fnv::NifShape& shape) {
                return shape.alphaBlend;
            });
        }
        if (staticModel.shapes.empty()) {
            continue;
        }
        if (importer::fnv::appendFalloutCharacterRigidMesh(
                staticModel,
                authoredAttachment.empty()
                    ? partNodes.bones.front().name
                    : authoredAttachment,
                outCharacter, rigidError)) {
            for (std::size_t part = firstAddedPart; part < outCharacter.parts.size(); ++part) {
                outCharacter.parts[part].sourcePath = partPath;
            }
        }
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
    // Extent above the actor's own origin, which is its FEET, measured in the
    // BIND POSE -- see actorHeadHeight for why the raw vertex data is a
    // different space than the drawn body and reads far too short.
    //
    // Still bind rather than per-frame: an idle moves a head by a few units, and
    // the alternative is a bounds recompute over every skinned vertex every
    // frame.
    std::vector<odai::math::Matrix4> bindPose;
    importer::fnv::computeFalloutBindPose(character, bindPose);
    float highest = 0.0f;
    for (const odai::render::ImportedSkinnedMeshVertex& vertex : character.vertices) {
        int dominant = -1;
        float bestWeight = 0.0f;
        for (int influence = 0; influence < 4; ++influence) {
            if (vertex.boneWeights[influence] > bestWeight) {
                bestWeight = vertex.boneWeights[influence];
                dominant = static_cast<int>(vertex.boneIndices[influence]);
            }
        }
        if (dominant < 0 || static_cast<std::size_t>(dominant) >= bindPose.size()) {
            continue;
        }
        const odai::math::Matrix4& matrix = bindPose[static_cast<std::size_t>(dominant)];
        highest = std::max(highest,
            (matrix(1, 0) * vertex.position[0]) + (matrix(1, 1) * vertex.position[1]) +
                (matrix(1, 2) * vertex.position[2]) + matrix(1, 3));
    }
    return highest;
}

void findActorHeadAnchor(
    const importer::fnv::FalloutCharacter& character,
    int& outBone,
    float outLocal[3],
    float& outBindHeight
) {
    outBone = -1;
    outLocal[0] = outLocal[1] = outLocal[2] = 0.0f;
    outBindHeight = 0.0f;

    const anim::Skeleton& skeleton = character.skeleton;
    // Bethesda's rigs all carry a Bip01 hierarchy; the exact name first, then a
    // suffix match for the creature rigs that decorate it. "Bip01 HeadNub" is
    // deliberately NOT matched -- it is the tip past the head, and aiming there
    // overshoots.
    int head = skeleton.findBone("Bip01 Head");
    if (head < 0) {
        for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
            const std::string name = toLowerAscii(skeleton.bones[i].name);
            if (name.size() >= 4u && name.compare(name.size() - 4u, 4u, "head") == 0) {
                head = static_cast<int>(i);
                break;
            }
        }
    }
    if (head < 0) {
        return;
    }

    // The centroid of the head bone's own vertices, in the space they are
    // authored in -- which is what makes it a point the LIVE pose can carry:
    // poseScratch[head] is exactly the matrix the skinning shader applies to
    // these vertices, so one matrix-vector multiply per frame gives the head's
    // real world position, animation included.
    //
    // The centroid rather than the top of the skull: the top is where a hat
    // sits, and a conversation wants the face.
    double sum[3] = {0.0, 0.0, 0.0};
    std::size_t counted = 0;
    for (const odai::render::ImportedSkinnedMeshVertex& vertex : character.vertices) {
        // Dominant bone only. A vertex on the jaw is partly the neck's, and
        // averaging over every influence drags the centroid down the throat.
        int dominant = -1;
        float bestWeight = 0.0f;
        for (int influence = 0; influence < 4; ++influence) {
            if (vertex.boneWeights[influence] > bestWeight) {
                bestWeight = vertex.boneWeights[influence];
                dominant = static_cast<int>(vertex.boneIndices[influence]);
            }
        }
        if (dominant != head) {
            continue;
        }
        for (int axis = 0; axis < 3; ++axis) {
            sum[axis] += static_cast<double>(vertex.position[axis]);
        }
        ++counted;
    }
    if (counted == 0u) {
        return;  // a rig with a head bone that nothing is weighted to
    }
    for (int axis = 0; axis < 3; ++axis) {
        outLocal[axis] = static_cast<float>(sum[axis] / static_cast<double>(counted));
    }
    outBone = head;

    // The same point through the BIND pose, as the fallback for the frames
    // before the actor has ever been posed.
    std::vector<odai::math::Matrix4> bindPose;
    importer::fnv::computeFalloutBindPose(character, bindPose);
    if (static_cast<std::size_t>(head) < bindPose.size()) {
        const odai::math::Matrix4& matrix = bindPose[static_cast<std::size_t>(head)];
        outBindHeight = (matrix(1, 0) * outLocal[0]) + (matrix(1, 1) * outLocal[1]) +
            (matrix(1, 2) * outLocal[2]) + matrix(1, 3);
    }
}

bool loadActorIdleClip(
    const importer::fnv::FalloutAssetSource& assets,
    const std::string& skeletonPath,
    const anim::Skeleton& skeleton,
    bool female,
    std::size_t variant,
    anim::AnimationClip& outClip,
    std::string& outWhy
) {
    // TES3 keeps the humanoid animation bank inside base_anim*.nif rather than
    // in external KF files. The suffix variants are beast-race rigs with extra
    // bones, not a different storage convention. Every bone's
    // NiKeyframeController spans the whole bank, while NiTextKeyExtraData is
    // the authoritative clip directory.
    // Merge the tracks, select the exact authored Idle interval, and rebase its
    // keys to zero. Sampling an assumed interval leaves the actor in the first
    // bank pose and is especially visible as horizontal upper arms.
    std::string normalizedSkeleton = toLowerAscii(skeletonPath);
    std::replace(normalizedSkeleton.begin(), normalizedSkeleton.end(), '/', '\\');
    const std::size_t skeletonName = normalizedSkeleton.find_last_of('\\');
    const std::string_view skeletonFile = skeletonName == std::string::npos
        ? std::string_view(normalizedSkeleton)
        : std::string_view(normalizedSkeleton).substr(skeletonName + 1u);
    const bool tes3AnimationBank = skeletonFile.starts_with("base_anim") &&
        skeletonFile.ends_with(".nif");
    if (tes3AnimationBank) {
        std::vector<std::uint8_t> skeletonBytes;
        if (assets.resolveMesh(skeletonPath, skeletonBytes, outWhy)) {
            importer::fnv::NifModel skeletonModel;
            if (importer::fnv::parseNifStaticMesh(
                    skeletonBytes, skeletonModel, outWhy) &&
                !skeletonModel.embeddedAnimations.empty()) {
                float idleStart = -1.0f;
                float idleStop = -1.0f;
                for (const importer::fnv::NifTextKey& key : skeletonModel.textKeys) {
                    if (idleStart < 0.0f &&
                        textKeyContainsLabel(key.text, "idle: start")) {
                        idleStart = key.time;
                        continue;
                    }
                    if (idleStart >= 0.0f && key.time > idleStart &&
                        textKeyContainsLabel(key.text, "idle: stop")) {
                        idleStop = key.time;
                        break;
                    }
                }
                if (!(idleStart >= 0.0f && idleStop > idleStart)) {
                    outWhy = skeletonPath + " has no complete Idle text-key interval";
                    return false;
                }
                importer::fnv::KfAnimation idle;
                idle.name = "Morrowind Idle";
                idle.startTime = 0.0f;
                idle.stopTime = idleStop - idleStart;
                idle.cycleType = 0u;
                for (const importer::fnv::KfAnimation& controller :
                     skeletonModel.embeddedAnimations) {
                    for (const importer::fnv::KfBoneTrack& sourceTrack :
                         controller.tracks) {
                        importer::fnv::KfBoneTrack track = sourceTrack;
                        sliceTes3Track(track, idleStart, idleStop);
                        idle.tracks.push_back(std::move(track));
                    }
                }
                importer::fnv::FalloutAnimationStats stats;
                if (importer::fnv::buildFalloutAnimationClip(
                        idle, skeleton, outClip, stats) &&
                    outClip.tracks.size() >= 8u) {
                    outClip.loop = true;
                    outWhy = std::to_string(outClip.tracks.size()) +
                        " authored TES3 idle tracks";
                    return true;
                }
            }
        }
    }

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
    std::vector<std::string> candidates{
        directory + "mtidle.kf",
        // Oblivion uses the plain action name beside the humanoid skeleton.
        directory + "idle.kf"};
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
    // Skyrim SE's authored animation is a reflected x64 Havok packfile. Decode
    // the immutable retail clip directly into the same AnimationClip used by
    // KF/NIF animation; the NIF skeleton remains the skinning authority.
    {
        const std::string sex = female ? "female" : "male";
        const std::string hkxPath =
            "meshes\\actors\\character\\animations\\" + sex + "\\mt_idle.hkx";
        std::vector<std::uint8_t> hkxBytes;
        std::string hkxError;
        anim::HkxDecodedSkeleton hkxSkeleton;
        if (assets.resolveAsset(hkxPath, hkxBytes, hkxError) &&
            loadSkyrimHkxSkeleton(assets, female, hkxSkeleton, hkxError)) {
            anim::HkxDecodedClipMetadata metadata;
            if (anim::decodeHkxAnimationClip(hkxBytes, skeleton,
                    "Skyrim " + sex + " idle", outClip, metadata, hkxError,
                    &hkxSkeleton) &&
                metadata.boundTracks >= 8u) {
                outWhy = hkxPath + ": " + std::to_string(metadata.boundTracks) +
                    " retail HKX tracks";
                return true;
            }
            outWhy = hkxPath + ": " + hkxError;
            outClip = anim::AnimationClip{};
        }
    }
    // A malformed or mod-incompatible clip remains recoverable for ordinary
    // launches. The explicit cross-game showcase reports this fallback.
    if (makeProceduralSkyrimIdle(skeleton, outClip)) {
        outWhy = "procedural Skyrim humanoid idle";
        return true;
    }
    return false;
}

bool loadActorIdleClip(
    const importer::fnv::FalloutAssetSource& assets,
    const std::string& skeletonPath,
    const anim::Skeleton& skeleton,
    std::size_t variant,
    anim::AnimationClip& outClip,
    std::string& outWhy
) {
    return loadActorIdleClip(
        assets, skeletonPath, skeleton, false, variant, outClip, outWhy);
}

bool loadActorWalkClip(
    const importer::fnv::FalloutAssetSource& assets,
    const std::string& skeletonPath,
    const anim::Skeleton& skeleton,
    bool female,
    anim::AnimationClip& outClip,
    float& outSpeedUnitsPerSecond,
    std::string& outWhy
) {
    outSpeedUnitsPerSecond = 0.0f;
    const std::string directory = directoryOf(skeletonPath);

    // TES3 stores one animation bank in base_anim*.nif. The text keys are the
    // clip directory; use the repeating portion of WalkForward rather than the
    // intro/outro so the cycle is seamless. This is deliberately resolved
    // before the external-KF convention below, because no mtforward.kf exists
    // for the retail Morrowind humanoid rig.
    std::string normalizedSkeleton = toLowerAscii(skeletonPath);
    std::replace(normalizedSkeleton.begin(), normalizedSkeleton.end(), '/', '\\');
    const std::size_t skeletonName = normalizedSkeleton.find_last_of('\\');
    const std::string_view skeletonFile = skeletonName == std::string::npos
        ? std::string_view(normalizedSkeleton)
        : std::string_view(normalizedSkeleton).substr(skeletonName + 1u);
    const bool tes3AnimationBank = skeletonFile.starts_with("base_anim") &&
        skeletonFile.ends_with(".nif");
    if (tes3AnimationBank) {
        std::vector<std::uint8_t> skeletonBytes;
        if (assets.resolveMesh(skeletonPath, skeletonBytes, outWhy)) {
            importer::fnv::NifModel skeletonModel;
            if (importer::fnv::parseNifStaticMesh(
                    skeletonBytes, skeletonModel, outWhy) &&
                !skeletonModel.embeddedAnimations.empty()) {
                float walkStart = -1.0f;
                float walkStop = -1.0f;
                for (const importer::fnv::NifTextKey& key : skeletonModel.textKeys) {
                    if (walkStart < 0.0f &&
                        textKeyContainsLabel(key.text, "walkforward: loop start")) {
                        walkStart = key.time;
                        continue;
                    }
                    if (walkStart >= 0.0f && key.time > walkStart &&
                        textKeyContainsLabel(key.text, "walkforward: loop stop")) {
                        walkStop = key.time;
                        break;
                    }
                }
                if (!(walkStart >= 0.0f && walkStop > walkStart)) {
                    // A modded bank may omit loop markers while retaining the
                    // outer interval. It is still preferable to no locomotion.
                    walkStart = -1.0f;
                    walkStop = -1.0f;
                    for (const importer::fnv::NifTextKey& key : skeletonModel.textKeys) {
                        if (walkStart < 0.0f &&
                            textKeyContainsLabel(key.text, "walkforward: start")) {
                            walkStart = key.time;
                            continue;
                        }
                        if (walkStart >= 0.0f && key.time > walkStart &&
                            textKeyContainsLabel(key.text, "walkforward: stop")) {
                            walkStop = key.time;
                            break;
                        }
                    }
                }
                if (walkStart >= 0.0f && walkStop > walkStart) {
                    importer::fnv::KfAnimation walk;
                    walk.name = "Morrowind WalkForward";
                    walk.startTime = 0.0f;
                    walk.stopTime = walkStop - walkStart;
                    walk.cycleType = 0u;
                    for (const importer::fnv::KfAnimation& controller :
                         skeletonModel.embeddedAnimations) {
                        for (const importer::fnv::KfBoneTrack& sourceTrack :
                             controller.tracks) {
                            importer::fnv::KfBoneTrack track = sourceTrack;
                            sliceTes3Track(track, walkStart, walkStop);
                            walk.tracks.push_back(std::move(track));
                        }
                    }
                    importer::fnv::FalloutAnimationStats stats;
                    if (!importer::fnv::buildFalloutAnimationClip(
                            walk, skeleton, outClip, stats)) {
                        outWhy = "TES3 WalkForward interval bound no tracks";
                    }
                } else {
                    outWhy = skeletonPath + " has no complete WalkForward interval";
                }
            }
        }
    }

    // "mt" is Bethesda's movement type prefix and "forward" the direction, so
    // mtforward.kf is the ordinary walk for anything that walks. Humans keep
    // theirs one level down under locomotion\<sex>\; creatures keep a single
    // clip beside the skeleton or under a flat locomotion\.
    std::vector<std::string> candidates;
    if (female) {
        candidates.push_back(directory + "locomotion\\female\\mtforward.kf");
    }
    candidates.push_back(directory + "locomotion\\male\\mtforward.kf");
    candidates.push_back(directory + "locomotion\\mtforward.kf");
    candidates.push_back(directory + "mtforward.kf");
    // TES4 predates the movement-type prefix used by Fallout. Its humanoid
    // archive calls the same authored cycle walkforward.kf; creatures often
    // use the shorter forward.kf spelling.
    candidates.push_back(directory + "walkforward.kf");
    candidates.push_back(directory + "forward.kf");

    std::string clipPath = skeletonPath + ":WalkForward";
    for (std::size_t candidateIndex = 0u;
         candidateIndex <= candidates.size(); ++candidateIndex) {
        if (candidateIndex > 0u) {
            clipPath = candidates[candidateIndex - 1u];
        }
        if (candidateIndex == 0u && outClip.tracks.empty()) {
            continue;
        }
        if (candidateIndex > 0u) {
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
        }
        if (outClip.tracks.size() < 8u || outClip.duration <= 0.0f) {
            outWhy = "only " + std::to_string(outClip.tracks.size()) + " tracks bound";
            outClip = anim::AnimationClip{};
            continue;
        }

        // A LOCOMOTION CLIP CARRIES ITS OWN ROOT MOTION. The game moves the
        // actor by consuming that translation; here the wander code moves it
        // instead, so playing the clip untouched adds the walk twice -- the
        // actor slides forward through the cycle and snaps back at the loop.
        //
        // Take the horizontal displacement of the root over the clip as the
        // speed the animation was authored for, then flatten the root's
        // horizontal channel to its first key. Driving the actor at exactly
        // that speed is what keeps the feet planted instead of skating: the
        // stride length and the ground speed are then the same number.
        // Vertical is left alone -- that is the walk's bob, and it belongs to
        // the pose rather than to the path.
        // WHICH bone carries it is not obvious and getting it wrong is silent:
        // Bethesda accumulates on "Bip01" while the skeleton's actual root is a
        // scene node above it that no clip ever animates. Taking the topmost
        // bone therefore finds a track with no keys, derives a speed of zero,
        // and leaves the actor standing still forever.
        int rootBone = -1;
        for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
            if (toLowerAscii(skeleton.bones[i].name) == "bip01") {
                rootBone = static_cast<int>(i);
                break;
            }
        }
        if (rootBone < 0) {
            for (std::size_t i = 0; i < skeleton.bones.size(); ++i) {
                if (skeleton.bones[i].parentIndex < 0) {
                    rootBone = static_cast<int>(i);
                    break;
                }
            }
        }
        for (anim::BoneTrack& track : outClip.tracks) {
            if (track.boneIndex != rootBone || track.translationKeys.size() < 2u) {
                continue;
            }
            const odai::math::Vector3 first = track.translationKeys.front().value;
            const odai::math::Vector3 last = track.translationKeys.back().value;
            const float dx = last.x - first.x;
            const float dz = last.z - first.z;
            outSpeedUnitsPerSecond = std::sqrt((dx * dx) + (dz * dz)) / outClip.duration;
            for (anim::Vector3Key& key : track.translationKeys) {
                key.value.x = first.x;
                key.value.z = first.z;
            }
            break;
        }
        // A clip with no root translation is an in-place walk -- some creature
        // sets are authored that way -- so fall back rather than refusing to
        // move. ~100 units/s is a 1.4 m/s stroll at ~70 units per metre.
        const bool derived = outSpeedUnitsPerSecond > 1.0f;
        if (!derived) {
            outSpeedUnitsPerSecond = 100.0f;
        }
        VOX_LOGD("newvegas") << "walk clip " << clipPath << ": " << outClip.tracks.size()
                             << " tracks, " << outClip.duration << "s, "
                             << outSpeedUnitsPerSecond << " u/s"
                             << (derived ? " (from root motion)" : " (no root motion, assumed)");
        outClip.loop = true;
        return true;
    }
    // Skyrim ships mt_walkforward.hkx rather than a KF. Decode it after the
    // legacy KF conventions have been exhausted, then feed it through the same
    // root-motion measurement/flattening used by Gamebryo locomotion above.
    {
        const std::string sex = female ? "female" : "male";
        const std::string hkxPath =
            "meshes\\actors\\character\\animations\\" + sex + "\\mt_walkforward.hkx";
        std::vector<std::uint8_t> hkxBytes;
        std::string hkxError;
        anim::HkxDecodedSkeleton hkxSkeleton;
        if (assets.resolveAsset(hkxPath, hkxBytes, hkxError) &&
            loadSkyrimHkxSkeleton(assets, female, hkxSkeleton, hkxError)) {
            anim::HkxDecodedClipMetadata metadata;
            if (anim::decodeHkxAnimationClip(hkxBytes, skeleton,
                    "Skyrim " + sex + " walk forward", outClip, metadata, hkxError,
                    &hkxSkeleton) &&
                metadata.boundTracks >= 8u && outClip.duration > 0.0f) {
                int rootBone = skeleton.findBone("NPC Root [Root]");
                if (rootBone < 0) rootBone = skeleton.findBone("NPC COM [COM ]");
                for (anim::BoneTrack& track : outClip.tracks) {
                    if (track.boneIndex != rootBone || track.translationKeys.size() < 2u) continue;
                    const odai::math::Vector3 first = track.translationKeys.front().value;
                    const odai::math::Vector3 last = track.translationKeys.back().value;
                    const float dx = last.x - first.x;
                    const float dz = last.z - first.z;
                    outSpeedUnitsPerSecond = std::sqrt(dx * dx + dz * dz) / outClip.duration;
                    for (anim::Vector3Key& key : track.translationKeys) {
                        key.value.x = first.x;
                        key.value.z = first.z;
                    }
                    break;
                }
                if (outSpeedUnitsPerSecond <= 1.0f) outSpeedUnitsPerSecond = 100.0f;
                outClip.loop = true;
                outWhy = hkxPath + ": " + std::to_string(metadata.boundTracks) +
                    " retail HKX tracks";
                return true;
            }
            outWhy = hkxPath + ": " + hkxError;
            outClip = anim::AnimationClip{};
        }
    }
    // 100 units/s is the measured fallback used for an in-place creature clip
    // above and reads as a normal city patrol pace at Bethesda scale.
    if (makeProceduralSkyrimWalk(skeleton, outClip)) {
        outSpeedUnitsPerSecond = 100.0f;
        outWhy = "procedural Skyrim humanoid walk (HKX fallback)";
        return true;
    }
    return false;
}

// Facing for a yaw, and the yaw for a facing.
//
// Bethesda actors are authored facing +Y in their Z-up asset space. The
// character importer's basis change (x, y, z) -> (x, z, -y) therefore makes
// the rendered skeleton's unrotated forward axis engine -Z, not +X. Rotating
// local -Z by Matrix4::rotationY(yaw) yields (-sin(yaw), -cos(yaw)). Keeping
// navigation on any other local axis makes every actor walk sideways.
odai::math::Vector3 actorFacing(float yawRadians) {
    return odai::math::Vector3{-std::sin(yawRadians), 0.0f, -std::cos(yawRadians)};
}

float actorYawForDirection(float dx, float dz) { return std::atan2(-dx, -dz); }

// Shortest signed angle from `from` to `to`, in radians.
float angleDelta(float from, float to) {
    constexpr float kTwoPi = 6.28318530718f;
    float delta = std::fmod(to - from + 3.14159265f, kTwoPi);
    if (delta < 0.0f) {
        delta += kTwoPi;
    }
    return delta - 3.14159265f;
}

bool loadGoodspringsActors(
    const std::filesystem::path& pluginPath,
    const importer::fnv::FalloutLoadOrder* loadOrder,
    const importer::fnv::FalloutAssetSource& assets,
    const float bethesdaCentre[3],
    float radius,
    std::uint32_t firstInstanceSlot,
    std::size_t maxActors,
    const std::vector<std::uint32_t>& excludeBaseFormIds,
    const std::function<bool(std::uint32_t referenceFormId)>& placementFilter,
    std::vector<SkinnedActor>& outActors,
    ActorPopulationStats& outStats,
    const importer::fnv::FalloutActorScan* actorCatalog,
    const std::unordered_map<std::uint32_t, std::string>* catalogVoiceFolderPlugin,
    const ActorPlacementRuntimeResolver& placementRuntimeResolver
) {
    outActors.clear();
    outStats = ActorPopulationStats{};

    importer::fnv::FalloutActorScan scannedContent;
    std::string error;
    // A companion mod defines its NPC, its placement, its race and its armour in
    // ITS OWN plugin, so scanning only the worldspace's finds nothing of it.
    // voiceFolderPlugin remembers which file each base came from, because a
    // voice path's first component is the DEFINING plugin's own name
    // (sound\voice\NVWillow.esp\...), not the load order's first entry.
    std::unordered_map<std::uint32_t, std::string> scannedVoiceFolderPlugin;
    if (actorCatalog == nullptr) {
        const bool scanned =
            (loadOrder != nullptr && !loadOrder->empty())
                ? importer::fnv::findActorsNearAcrossOrder(
                      *loadOrder, bethesdaCentre[0], bethesdaCentre[1], radius,
                      scannedContent, scannedVoiceFolderPlugin, error)
                : importer::fnv::findActorsNear(
                      pluginPath, bethesdaCentre[0], bethesdaCentre[1], radius,
                      scannedContent, error);
        if (!scanned) {
            outStats.detail = "scan failed: " + error;
            return false;
        }
        actorCatalog = &scannedContent;
        catalogVoiceFolderPlugin = &scannedVoiceFolderPlugin;
    }
    const importer::fnv::FalloutActorScan& scan = *actorCatalog;
    static const std::unordered_map<std::uint32_t, std::string> kNoVoicePlugins;
    const auto& voiceFolderPlugin = catalogVoiceFolderPlugin != nullptr
        ? *catalogVoiceFolderPlugin : kNoVoicePlugins;
    std::vector<importer::fnv::FalloutActorPlacement> placements = scan.placements;
    if (placementFilter) {
        std::erase_if(placements, [&](const importer::fnv::FalloutActorPlacement& placement) {
            return placement.refFormId != 0u && !placementFilter(placement.refFormId);
        });
    }
    if (placementRuntimeResolver) {
        std::erase_if(placements, [&](importer::fnv::FalloutActorPlacement& placement) {
            return !placementRuntimeResolver(placement);
        });
    }
    std::sort(placements.begin(), placements.end(), [&](const auto& left, const auto& right) {
        const float leftX = left.position[0] - bethesdaCentre[0];
        const float leftY = left.position[1] - bethesdaCentre[1];
        const float rightX = right.position[0] - bethesdaCentre[0];
        const float rightY = right.position[1] - bethesdaCentre[1];
        const float leftDistance = (leftX * leftX) + (leftY * leftY);
        const float rightDistance = (rightX * rightX) + (rightY * rightY);
        if (leftDistance != rightDistance) return leftDistance < rightDistance;
        return left.refFormId < right.refFormId;
    });

    // ODAI_FNV_SPAWN_ACTOR=<EditorID>[,<EditorID>...] places a named actor in
    // front of the player whether or not the world places it anywhere.
    //
    // This is what a companion mod needs. Its NPC is parked in a private
    // interior and moved into the world by a quest script -- Willow's placement
    // is in WillowsCell and her 7187-byte script does the rest -- so there is
    // nothing in the worldspace to find and no door to arrive through. The base
    // record itself is complete, and the scan already collects every base
    // wholesale rather than only the ones something places, so building one is
    // just a matter of asking for it.
    //
    // A SYNTHETIC PLACEMENT rather than a separate build path: everything
    // downstream -- geometry, armour, skeleton, animation, dialogue, voice --
    // keys off a placement, so injecting one gets all of it for free and cannot
    // drift from how a placed actor is built.
    if (const char* spawnEnv = std::getenv("ODAI_FNV_SPAWN_ACTOR")) {
        std::vector<std::string> wanted;
        std::string current;
        for (const char* cursor = spawnEnv;; ++cursor) {
            if (*cursor == ',' || *cursor == '\0') {
                if (!current.empty()) {
                    wanted.push_back(toLowerAscii(current));
                    current.clear();
                }
                if (*cursor == '\0') {
                    break;
                }
                continue;
            }
            current.push_back(*cursor);
        }
        // Fan them out sideways so several named actors do not occupy one spot.
        int placed = 0;
        for (const std::string& name : wanted) {
            const importer::fnv::FalloutActorBase* match = nullptr;
            for (const auto& [formId, base] : scan.bases) {
                if (toLowerAscii(base.editorId) == name) {
                    match = &base;
                    break;
                }
            }
            if (match == nullptr) {
                VOX_LOGW("newvegas") << "spawn actor: no base with EditorID \"" << name
                                     << "\" in the loaded plugins";
                continue;
            }
            // Reuse a real resident reference when the cell already places
            // this actor. Apart from producing a visible duplicate, injecting
            // a base-only copy here loses the placed reference identity used
            // by face data, dialogue and runtime state. Explicit spawning still
            // relocates that authored actor in front of the player.
            importer::fnv::FalloutActorPlacement placement{};
            const auto resident = std::find_if(
                placements.begin(), placements.end(), [&](const auto& candidate) {
                    return candidate.baseFormId == match->formId;
                });
            const bool reusedAuthoredReference = resident != placements.end();
            if (reusedAuthoredReference) {
                placement = *resident;
                placements.erase(resident);
            } else {
                placement.refFormId = 0u;  // no authored placement is resident
                placement.baseFormId = match->formId;
            }
            placement.position[0] = bethesdaCentre[0] + 160.0f + (static_cast<float>(placed) * 140.0f);
            placement.position[1] = bethesdaCentre[1] + 160.0f;
            // Ground height is resolved per actor every frame, so this only has
            // to be close enough not to start underneath the terrain.
            placement.position[2] = bethesdaCentre[2] + 40.0f;
            placement.initiallyDisabled = false;
            // Front of the list: placements are nearest-first and the slot
            // budget cuts from the back, so a deliberately requested actor must
            // not lose its slot to whatever happens to be standing nearby.
            placements.insert(placements.begin(), placement);
            VOX_LOGI("newvegas") << "spawn actor: " << match->editorId
                                 << (match->fullName.empty() ? "" : (" \"" + match->fullName + "\""))
                                 << " base 0x" << std::hex << match->formId << std::dec
                                 << (reusedAuthoredReference ? " (authored reference relocated)" : "")
                                 << (match->isFemale ? " female" : "");
            ++placed;
        }
    }
    outStats.placementsConsidered = placements.size();

    // One build per distinct base: Goodsprings places six bighorners, and
    // parsing the same NIF six times is six times the cost for one result. Each
    // still gets its own GPU instance slot -- the renderer's template is
    // per-slot -- but the CPU-side assembly is shared.
    struct BuiltBase {
        bool ok = false;
        // Whichever skeleton actually bound -- the race's own, or the human one
        // substituted for it. The animation clips must be resolved against the
        // same rig the vertices were weighted to.
        std::string skeletonPath;
        importer::fnv::FalloutCharacter character;
        std::vector<odai::importer::ImportedSceneTexture> textures;
        std::vector<odai::importer::ImportedScenePackedDraw> draws;
        anim::AnimationClip idleClip;
        bool hasClip = false;
        anim::AnimationClip walkClip;
        float walkSpeed = 0.0f;
        bool hasWalk = false;
    };
    std::unordered_map<std::uint32_t, BuiltBase> builtByBase;
    std::uint32_t nextSlot = firstInstanceSlot;

    for (const importer::fnv::FalloutActorPlacement& placement : placements) {
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
        std::string why;
        if (built.character.vertices.empty() && !built.ok) {
            std::string skeletonPath = resolved.skeletonPath;
            built.ok = buildSkinnedActor(
                assets, skeletonPath, resolved.bodyPartPaths, built.character,
                built.textures, built.draws, why);
            if (!built.ok) {
                // A CUSTOM RACE OFTEN DECLARES A SKELETON IT DOES NOT SHIP.
                // Willow's race names characters\willow race\skeleton.nif and
                // her 258 MB archive contains 154 meshes and no skeleton at all:
                // the race is a copy of the human one, and the human skeleton is
                // what it expects to be rigged against. Every human body part in
                // the game is weighted to that same skeleton, so substituting it
                // is not a guess -- it is the only rig those parts can bind to.
                //
                // Only after the declared path has actually been tried, so a
                // race that does ship its own is never overridden by this.
                constexpr const char* kHumanSkeleton = "characters\\_male\\skeleton.nif";
                if (skeletonPath != kHumanSkeleton) {
                    std::string fallbackWhy;
                    const bool fallbackOk = buildSkinnedActor(
                        assets, kHumanSkeleton, resolved.bodyPartPaths, built.character,
                        built.textures, built.draws, fallbackWhy);
                    if (fallbackOk) {
                        VOX_LOGI("newvegas")
                            << "actor "
                            << (resolved.base != nullptr ? resolved.base->editorId : "<unnamed>")
                            << ": skeleton " << skeletonPath
                            << " unavailable, rigged against " << kHumanSkeleton;
                        skeletonPath = kHumanSkeleton;
                        built.ok = true;
                    } else {
                        why += "; human-skeleton fallback also failed: " + fallbackWhy;
                    }
                }
            }
            built.skeletonPath = skeletonPath;
            if (built.ok) {
                std::string clipWhy;
                const bool female = resolved.base != nullptr && resolved.base->isFemale;
                built.hasClip = loadActorIdleClip(
                    assets, built.skeletonPath, built.character.skeleton, female,
                    builtByBase.size(),
                    built.idleClip, clipWhy);
                built.hasWalk = loadActorWalkClip(
                    assets, built.skeletonPath, built.character.skeleton, female,
                    built.walkClip, built.walkSpeed, clipWhy);
            }
        }
        if (!built.ok) {
            ++outStats.skippedBuildFailed;
            // The reason was being collected and thrown away, which made an
            // unbuildable actor a number with no cause -- and for one the user
            // deliberately asked to spawn, the cause is the whole message.
            VOX_LOGW("newvegas")
                << "actor base 0x" << std::hex << placement.baseFormId << std::dec
                << " ("
                << (resolved.base != nullptr && !resolved.base->editorId.empty()
                        ? resolved.base->editorId
                        : "<unnamed>")
                << ") could not be built: " << (why.empty() ? "no reason given" : why)
                << "\n    skeleton: " << resolved.skeletonPath << "\n    parts ("
                << resolved.bodyPartPaths.size() << "):";
            for (const std::string& part : resolved.bodyPartPaths) {
                VOX_LOGW("newvegas") << "      " << part;
            }
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
        // Which plugin DEFINED this actor's base -- its voice tree lives under
        // that plugin's name, not the worldspace's.
        if (const auto voicePluginIt = voiceFolderPlugin.find(placement.baseFormId);
            voicePluginIt != voiceFolderPlugin.end()) {
            actor.voice.voicePlugin = voicePluginIt->second;
        }
        actor.baseFormId = placement.baseFormId;
        actor.referenceFormId = placement.refFormId;
        actor.referenceTypeFormIds = placement.referenceTypeFormIds;
        actor.inventoryFormIds =
            scan.materializeInventory(placement.baseFormId, placement.refFormId);
        actor.placed = true;
        actor.character = built.character;
        actor.standingHeightUnits = actorStandingHeight(built.character);
        actor.humanoid = resolved.base != nullptr && resolved.base->recordType == "NPC_";
        findActorHeadAnchor(built.character, actor.headAnchorBone, actor.headAnchorLocal,
                            actor.headHeightUnits);
        actor.textures = built.textures;
        actor.draws = built.draws;
        actor.idleClip = built.idleClip;
        actor.walkClip = built.walkClip;
        actor.walkSpeedUnitsPerSecond = built.walkSpeed;
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
        // Wander from where the plugin put them. The authored position is the
        // anchor rather than the destination: a town whose people each hold one
        // spot reads as a diorama, and one whose people walk anywhere reads as
        // a crowd that has lost its buildings.
        actor.wanders = built.hasWalk && actor.walkSpeedUnitsPerSecond > 1.0f;
        actor.wanderOrigin[0] = actor.position[0];
        actor.wanderOrigin[1] = actor.position[1];
        actor.wanderOrigin[2] = actor.position[2];
        // Target starts AT the actor, so the first update takes the arrival
        // branch and picks a real destination. Left at the default it is the
        // world origin, tens of thousands of units away, and the whole town
        // sets off toward the same point on the map.
        actor.wanderTarget[0] = actor.position[0];
        actor.wanderTarget[1] = actor.position[1];
        actor.wanderTarget[2] = actor.position[2];
        // Distinct per actor and stable across runs: same seed, same walk.
        actor.wanderRng = 0x9e3779b9u ^ (placement.refFormId * 2654435761u);
        // Staggered starts, or the whole town steps off together on frame one.
        actor.wanderPauseSeconds = static_cast<float>(outActors.size() % 7u) * 0.9f;
        if (built.hasWalk) {
            ++outStats.walking;
        }
        if (built.hasClip) {
            ++outStats.animated;
        }
        outActors.push_back(std::move(actor));
        ++outStats.built;
    }

    // ODAI_FNV_ACTORS_LIST names what was actually built. Counts alone cannot
    // answer "is the companion I just installed among these" -- and with a mod
    // loaded the answer decides whether the problem is discovery, the build, or
    // only the look.
    if (std::getenv("ODAI_FNV_ACTORS_LIST") != nullptr) {
        for (const SkinnedActor& built : outActors) {
            const importer::fnv::FalloutActorBase* base = nullptr;
            const auto found = scan.bases.find(built.baseFormId);
            if (found != scan.bases.end()) {
                base = &found->second;
            }
            VOX_LOGI("newvegas")
                << "  actor base 0x" << std::hex << built.baseFormId << std::dec << " "
                << (base != nullptr && !base->editorId.empty() ? base->editorId : "<unnamed>")
                << (base != nullptr && !base->fullName.empty() ? (" \"" + base->fullName + "\"")
                                                               : std::string())
                << (base != nullptr && base->isFemale ? " female" : "")
                << " voice=" << (built.voice.voiceFolder.empty() ? "<none>" : built.voice.voiceFolder)
                << " height=" << built.standingHeightUnits
                << " headHeight=" << built.headHeightUnits;
            const importer::fnv::ResolvedActorBase resolved =
                scan.resolve(built.baseFormId);
            for (const std::string& part : resolved.bodyPartPaths) {
                VOX_LOGI("newvegas") << "      part: " << part;
            }
        }
    }

    outStats.detail =
        std::to_string(outStats.built) + " built (" + std::to_string(outStats.animated) +
        " animated, " + std::to_string(outStats.walking) + " able to walk) of " + std::to_string(outStats.placementsConsidered) + " placements; skipped " +
        std::to_string(outStats.skippedDisabled) + " disabled, " +
        std::to_string(outStats.skippedNoGeometry) + " without geometry, " +
        std::to_string(outStats.skippedBuildFailed) + " unbuildable, " +
        std::to_string(outStats.skippedNoSlots) + " over slot budget, " +
        std::to_string(outStats.skippedExcluded) + " handled elsewhere";
    return true;
}

bool loadSkyrimGuardShowcase(
    const std::filesystem::path& skyrimDataDirectory,
    const float engineCentre[3],
    std::uint32_t firstInstanceSlot,
    std::size_t guardCount,
    std::vector<SkinnedActor>& outGuards,
    std::string& outDetail) {
    outGuards.clear();
    outDetail.clear();
    if (guardCount == 0u) return true;

    const std::filesystem::path skyrimPlugin =
        skyrimDataDirectory / "Skyrim.esm";
    if (!std::filesystem::is_regular_file(skyrimPlugin)) {
        outDetail = "Skyrim.esm is unavailable under " +
            skyrimDataDirectory.string();
        return false;
    }

    importer::fnv::FalloutAssetSource skyrimAssets;
    if (!skyrimAssets.open(skyrimDataDirectory)) {
        outDetail = "could not index Skyrim actor assets under " +
            skyrimDataDirectory.string();
        return false;
    }
    importer::fnv::FalloutActorScan catalog;
    std::string error;
    if (!importer::fnv::findActorsNear(
            skyrimPlugin, 0.0f, 0.0f, 1.0f, catalog, error)) {
        outDetail = "could not scan Skyrim guard records: " + error;
        return false;
    }

    constexpr const char* kPreferredGuards[] = {
        "guardwhiterunsonsjail", "guardwhiterunimperialjail"};
    std::uint32_t guardBaseFormId = 0u;
    for (const char* preferred : kPreferredGuards) {
        const auto found = std::find_if(catalog.bases.begin(), catalog.bases.end(),
            [&](const auto& entry) {
                return toLowerAscii(entry.second.editorId) == preferred;
            });
        if (found != catalog.bases.end() &&
            catalog.resolve(found->first).bodyPartPaths.size() >= 4u) {
            guardBaseFormId = found->first;
            break;
        }
    }
    if (guardBaseFormId == 0u) {
        const auto fallback = std::find_if(catalog.bases.begin(), catalog.bases.end(),
            [&](const auto& entry) {
                const std::string editorId = toLowerAscii(entry.second.editorId);
                return editorId.starts_with("guardwhiterun") &&
                    catalog.resolve(entry.first).bodyPartPaths.size() >= 4u;
            });
        if (fallback != catalog.bases.end()) guardBaseFormId = fallback->first;
    }
    if (guardBaseFormId == 0u) {
        outDetail = "Skyrim.esm contains no buildable Whiterun guard base";
        return false;
    }

    // Synthetic authored-space placements feed the ordinary actor builder so
    // guard armor, skinning, clips and texture packing share exactly the same
    // path as native residents. The caller replaces these approximate points
    // with Balmora navmesh projections after construction.
    catalog.placements.clear();
    const float bethesdaCentre[3] = {
        engineCentre[0], -engineCentre[2], engineCentre[1]};
    for (std::size_t index = 0u; index < guardCount; ++index) {
        importer::fnv::FalloutActorPlacement placement;
        placement.baseFormId = guardBaseFormId;
        placement.position[0] = bethesdaCentre[0] +
            180.0f + static_cast<float>(index / 2u) * 140.0f;
        placement.position[1] = bethesdaCentre[1] +
            (index % 2u == 0u ? -110.0f : 110.0f);
        placement.position[2] = bethesdaCentre[2];
        catalog.placements.push_back(placement);
    }

    ActorPopulationStats stats;
    if (!loadGoodspringsActors(
            skyrimPlugin, nullptr, skyrimAssets, bethesdaCentre, 1000.0f,
            firstInstanceSlot, guardCount, {}, {}, outGuards, stats, &catalog)) {
        outDetail = stats.detail;
        return false;
    }
    for (std::size_t index = 0u; index < outGuards.size(); ++index) {
        SkinnedActor& guard = outGuards[index];
        guard.fullName = "Skyrim Guard";
        guard.referenceFormId = 0u;
        guard.runtimeObjectId = bethesda::ObjectId::persistent(
            bethesda::makeTes3ReferenceKey(
                "odai-skyrim-guard-showcase", static_cast<std::uint32_t>(index + 1u)));
        guard.wanderRng = 0x68bc21ebu ^
            (static_cast<std::uint32_t>(index + 1u) * 2654435761u);
        guard.wanderPauseSeconds = static_cast<float>(index) * 0.55f;
    }
    outDetail = std::to_string(outGuards.size()) +
        " Whiterun guard(s): " + stats.detail;
    return !outGuards.empty();
}

bool loadSkyrimPlayerAvatar(
    const std::filesystem::path& skyrimDataDirectory,
    std::string_view outfitEditorId,
    std::uint32_t instanceSlot,
    SkinnedActor& outAvatar,
    std::string& outDetail) {
    outAvatar = SkinnedActor{};
    outDetail.clear();
    const std::filesystem::path plugin = skyrimDataDirectory / "Skyrim.esm";
    if (!std::filesystem::is_regular_file(plugin)) {
        outDetail = "Skyrim.esm is unavailable under " + skyrimDataDirectory.string();
        return false;
    }
    importer::fnv::FalloutAssetSource assets;
    if (!assets.open(skyrimDataDirectory)) {
        outDetail = "could not index Skyrim avatar assets under " +
            skyrimDataDirectory.string();
        return false;
    }
    std::string error;
    importer::fnv::SkyrimAnimationAssetReport animationBundle;
    if (!importer::fnv::inspectSkyrimAnimationBundle(
            assets, animationBundle, true, error)) {
        outDetail = "Skyrim locomotion bundle is not usable: " + error;
        for (const std::string& diagnostic : animationBundle.diagnostics) {
            outDetail += "; " + diagnostic;
        }
        return false;
    }
    importer::fnv::FalloutActorScan catalog;
    error.clear();
    if (!importer::fnv::findActorsNear(plugin, 0.0f, 0.0f, 1.0f, catalog, error)) {
        outDetail = "could not scan Skyrim avatar records: " + error;
        return false;
    }
    const auto player = std::find_if(catalog.bases.begin(), catalog.bases.end(),
        [](const auto& entry) {
            return toLowerAscii(entry.second.editorId) == "player";
        });
    if (player == catalog.bases.end()) {
        outDetail = "Skyrim.esm has no NPC_ record with EditorID Player";
        return false;
    }
    const std::string wantedOutfit = toLowerAscii(std::string(outfitEditorId));
    const auto outfit = std::find_if(
        catalog.outfitEditorIds.begin(), catalog.outfitEditorIds.end(),
        [&](const auto& entry) { return toLowerAscii(entry.second) == wantedOutfit; });
    if (outfit == catalog.outfitEditorIds.end()) {
        outDetail = "Skyrim outfit '" + std::string(outfitEditorId) +
            "' was not found in Skyrim.esm";
        return false;
    }
    // NPC heads are pre-generated under FaceGeom, but form 0x7 is the
    // customizable Player and intentionally has no baked mesh in the retail
    // archives. Assemble one deterministic stock male Nord face from the
    // chargen source meshes instead. Every piece is required: silently
    // skipping one is how the previous avatar rendered without a head at all.
    static constexpr std::array<std::string_view, 6> kStockMaleNordFaceParts{
        "actors\\character\\character assets\\malehead.nif",
        "actors\\character\\character assets\\eyesmale.nif",
        "actors\\character\\character assets\\mouth\\mouthhuman.nif",
        "actors\\character\\character assets\\faceparts\\malebrows.nif",
        "actors\\character\\character assets\\hair\\male\\hairline01.nif",
        "actors\\character\\character assets\\hair\\male\\hair01.nif",
    };
    bool generatedFaceAvailable = false;
    for (const std::string& facePath : player->second.faceGeometryPaths) {
        std::vector<std::uint8_t> faceBytes;
        std::string faceError;
        if (assets.resolveMesh(facePath, faceBytes, faceError)) {
            generatedFaceAvailable = true;
            break;
        }
    }
    std::vector<std::string> requiredStockFaceParts;
    if (!generatedFaceAvailable) {
        player->second.faceGeometryPaths.clear();
        requiredStockFaceParts.reserve(kStockMaleNordFaceParts.size());
        for (const std::string_view facePathView : kStockMaleNordFaceParts) {
            const std::string facePath(facePathView);
            std::vector<std::uint8_t> faceBytes;
            std::string faceError;
            importer::fnv::NifSkinnedModel faceModel;
            if (!assets.resolveMesh(facePath, faceBytes, faceError)) {
                outDetail = "required stock male Nord face mesh is unavailable: " +
                    facePath + " (" + faceError + ")";
                return false;
            }
            if (!importer::fnv::parseNifSkinnedMesh(
                    faceBytes, faceModel, faceError) || faceModel.shapes.empty()) {
                outDetail = "required stock male Nord face mesh is not decodable: " +
                    facePath + " (" + faceError + ")";
                return false;
            }
            player->second.faceGeometryPaths.emplace_back(facePath);
            requiredStockFaceParts.emplace_back(facePath);
        }
    }
    // The mutable copy is an avatar compilation catalog only. It cannot leak
    // into BethesdaSession or alter Morrowind's inventory/quests/scripts.
    player->second.isFemale = false;
    player->second.defaultOutfitFormId = outfit->first;
    catalog.placements.clear();
    importer::fnv::FalloutActorPlacement placement;
    placement.baseFormId = player->first;
    catalog.placements.push_back(placement);

    std::vector<SkinnedActor> avatars;
    ActorPopulationStats stats;
    const float origin[3] = {};
    if (!loadGoodspringsActors(
            plugin, nullptr, assets, origin, 1.0f, instanceSlot, 1u, {}, {},
            avatars, stats, &catalog) || avatars.size() != 1u) {
        outDetail = "Skyrim player avatar build failed: " + stats.detail;
        return false;
    }
    outAvatar = std::move(avatars.front());
    for (const std::string& requiredPath : requiredStockFaceParts) {
        const std::string loweredRequired = toLowerAscii(requiredPath);
        const bool contributed = std::any_of(
            outAvatar.character.parts.begin(), outAvatar.character.parts.end(),
            [&](const importer::fnv::FalloutCharacterPart& part) {
                return toLowerAscii(part.sourcePath) == loweredRequired;
            });
        if (!contributed) {
            outDetail = "required stock male Nord face mesh did not bind to the avatar: " +
                requiredPath;
            outAvatar = SkinnedActor{};
            return false;
        }
    }
    outAvatar.name = "SkyrimPlayerAvatar";
    outAvatar.fullName = "Player";
    outAvatar.referenceFormId = 0u;
    outAvatar.runtimeObjectId = {};
    outAvatar.wanders = false;
    outAvatar.walking = false;
    outAvatar.instanceSlot = instanceSlot;
    if (const auto items = catalog.outfits.find(outfit->first);
        items != catalog.outfits.end()) {
        outAvatar.inventoryFormIds = items->second;
    }
    if (outAvatar.idleClip.name.starts_with("procedural") ||
        outAvatar.walkClip.name.starts_with("procedural")) {
        outDetail = "Skyrim player avatar requires decodable retail male idle and walk HKX clips";
        outAvatar = SkinnedActor{};
        return false;
    }
    static constexpr std::array additionalClips{
        std::pair{"meshes\\actors\\character\\animations\\male\\mt_runforward.hkx",
                  "Skyrim male run forward"},
        std::pair{"meshes\\actors\\character\\animations\\male\\mt_sprintforward.hkx",
                  "Skyrim male sprint forward"},
        std::pair{"meshes\\actors\\character\\animations\\mt_jump.hkx",
                  "Skyrim jump"},
        std::pair{"meshes\\actors\\character\\animations\\mt_jumpfall.hkx",
                  "Skyrim fall"},
        std::pair{"meshes\\actors\\character\\animations\\mt_jumpland.hkx",
                  "Skyrim landing"},
        std::pair{"meshes\\actors\\character\\animations\\male\\npc_turnleft90.hkx",
                  "Skyrim turn left"},
        std::pair{"meshes\\actors\\character\\animations\\male\\npc_turnright90.hkx",
                  "Skyrim turn right"}};
    anim::HkxDecodedSkeleton hkxSkeleton;
    if (!loadSkyrimHkxSkeleton(assets, false, hkxSkeleton, error)) {
        outDetail = "required Skyrim avatar skeleton HKX could not decode: " + error;
        outAvatar = SkinnedActor{};
        return false;
    }
    for (const auto& [path, name] : additionalClips) {
        std::vector<std::uint8_t> bytes;
        if (!assets.resolveAsset(path, bytes, error)) {
            outDetail = "required Skyrim avatar HKX is unavailable: " +
                std::string(path) + " (" + error + ")";
            outAvatar = SkinnedActor{};
            return false;
        }
        anim::AnimationClip clip;
        anim::HkxDecodedClipMetadata metadata;
        if (!anim::decodeHkxAnimationClip(
                bytes, outAvatar.character.skeleton, name, clip, metadata, error,
                &hkxSkeleton) ||
            metadata.boundTracks < 8u) {
            outDetail = "required Skyrim avatar HKX could not decode: " +
                std::string(path) + " (" + error + ")";
            outAvatar = SkinnedActor{};
            return false;
        }
        outAvatar.authoredLocomotionClips.push_back(std::move(clip));
    }
    outDetail = "male Nord Player with complete stock head wearing " +
        std::string(outfitEditorId) +
        " (" + std::to_string(outAvatar.character.parts.size()) +
        " body pieces); 9 retail HKX locomotion clips decoded";
    return true;
}

std::size_t loadActorDialogue(
    const std::filesystem::path& pluginPath,
    const importer::fnv::FalloutLoadOrder* loadOrder,
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
    const bool built =
        (loadOrder != nullptr && !loadOrder->empty())
            ? importer::fnv::buildSpeakerDialogueTreesAcrossOrder(
                  *loadOrder, requests, trees, stats, error)
            : importer::fnv::buildSpeakerDialogueTrees(pluginPath, requests, trees, stats, error);
    if (!built) {
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
    const std::string& pluginFileName,
    const std::vector<std::string>& modDirectories,
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
        // Keyed by plugin AND folder: FemaleAdult01Default exists in both
        // games and means a different set of recordings in each -- and now, in
        // both the base game and a companion mod within ONE load order.
        const std::string voicePlugin =
            actor.voice.voicePlugin.empty() ? pluginFileName : actor.voice.voicePlugin;
        const std::string key = toLowerAscii(voicePlugin + "\\" + actor.voice.voiceFolder);
        auto existing = indexByFolder.find(key);
        if (existing == indexByFolder.end()) {
            ActorVoiceIndex index;
            buildVoiceIndexForFolder(
                dataFilesPath, voicePlugin, actor.voice.voiceFolder, modDirectories, index);
            existing = indexByFolder.emplace(key, std::move(index)).first;
        }
        // Preserve which plugin this actor's lines came from: the shared index
        // is reused across actors, and overwriting it would make the next
        // actor's lookup key wrong.
        const std::string keepPlugin = actor.voice.voicePlugin;
        actor.voice = existing->second;
        actor.voice.voicePlugin = keepPlugin;
        if (!actor.voice.pathByNodeId.empty() || !actor.voice.loosePathByNodeId.empty()) {
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
        // Both sources: a mod's lines are usually loose, and reporting only the
        // archive count showed a fully voiced companion as "(0)".
        outDetail += " " + index.voiceFolder + "(" +
            std::to_string(index.pathByNodeId.size() + index.loosePathByNodeId.size()) + ")";
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

    // A loose line short-circuits the archive lookup entirely: there is nothing
    // to extract, the bytes are already a file on disk.
    const std::string voiceKey = voiceKeyForNodeId(node->id);
    const auto looseFound = actor.voice.loosePathByNodeId.find(voiceKey);
    const bool haveLoose = looseFound != actor.voice.loosePathByNodeId.end();
    const auto found = actor.voice.pathByNodeId.find(voiceKey);
    if (!haveLoose && found == actor.voice.pathByNodeId.end()) {
        return;  // a line the game never recorded, or a topic node
    }
    importer::fnv::BsaArchive* holder = nullptr;
    const importer::fnv::BsaFileEntry* entry = nullptr;
    if (!haveLoose) {
        const auto archives = voiceArchives().find(actor.voice.archiveKey);
        if (archives == voiceArchives().end()) {
            return;
        }
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
    }

    // Cached by leaf name, so a line costs one extract + one Vorbis decode per
    // install rather than one per playback.
    std::string leaf = haveLoose ? looseFound->second.string() : found->second;
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
        if (haveLoose) {
            std::ifstream in(looseFound->second, std::ios::binary);
            if (in) {
                oggBytes.assign(
                    std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            }
            if (oggBytes.empty()) {
                VOX_LOGW("newvegas") << actor.displayName() << " voice read failed for "
                                     << node->id << ": " << looseFound->second.string();
                return;
            }
        } else if (!holder->extract(*entry, oggBytes, extractError) || oggBytes.empty()) {
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

void updateActorPoses(std::span<SkinnedActor> actors, float deltaSeconds) {
    // ODAI_FNV_NOANIM=1 freezes every actor at bind pose while leaving the rest
    // of the path running -- same upload, same per-frame pose submission, same
    // draws. It is the control for "is this actually animating": a screenshot
    // diff of an actor's own pixels across two moments is otherwise impossible
    // to attribute, because the world is streaming in behind it and the light is
    // moving, so SOMETHING always changes.
    static const bool freezeAtBindPose = std::getenv("ODAI_FNV_NOANIM") != nullptr ||
        std::getenv("ODAI_FNV_VICTOR_NOANIM") != nullptr;
    constexpr float kPoseTransitionSeconds = 0.20f;

    for (SkinnedActor& actor : actors) {
        if (actor.character.skeleton.bones.empty()) {
            continue;
        }
        const bool wantsTalkClip = actor.talking && !actor.talkClip.tracks.empty();
        const bool wantsWalkClip =
            !wantsTalkClip && actor.walking && !actor.walkClip.tracks.empty();
        // Restart the destination clip and retain the exact pose currently on
        // screen. Blending from that snapshot also handles a rapid stop/start
        // during an unfinished transition without snapping back to either
        // clip's raw pose.
        if (wantsTalkClip != actor.posedTalking ||
            wantsWalkClip != actor.posedWalking) {
            actor.posedTalking = wantsTalkClip;
            actor.posedWalking = wantsWalkClip;
            actor.animationSeconds = 0.0f;
            actor.poseTransitionElapsedSeconds = 0.0f;
            actor.poseTransitionSource = actor.localPoseScratch;
        }
        actor.animationSeconds += deltaSeconds;

        if (!actor.renderVisible) {
            continue;
        }

        const anim::AnimationClip& clip = wantsTalkClip ? actor.talkClip
            : (wantsWalkClip ? actor.walkClip : actor.idleClip);
        if (freezeAtBindPose || clip.tracks.empty() || clip.duration <= 0.0f) {
            // No clip: hold the bind pose rather than leaving the previous
            // frame's matrices, which on the first frame would be none at all.
            importer::fnv::computeFalloutBindPose(actor.character, actor.poseScratch);
        } else {
            actor.sampler.sample(
                actor.character.skeleton, clip, actor.animationSeconds, actor.poseScratch);
        }

        const bool canBlend =
            actor.poseTransitionSource.size() == actor.poseScratch.size() &&
            !actor.poseTransitionSource.empty() &&
            actor.poseTransitionElapsedSeconds < kPoseTransitionSeconds;
        if (canBlend) {
            actor.poseTransitionElapsedSeconds += std::max(deltaSeconds, 0.0f);
            const float linear = std::clamp(
                actor.poseTransitionElapsedSeconds / kPoseTransitionSeconds, 0.0f, 1.0f);
            const float blend = linear * linear * (3.0f - (2.0f * linear));
            actor.localPoseScratch.resize(actor.poseScratch.size());
            for (std::size_t bone = 0u; bone < actor.poseScratch.size(); ++bone) {
                for (std::size_t component = 0u; component < 16u; ++component) {
                    actor.localPoseScratch[bone].m[component] =
                        actor.poseTransitionSource[bone].m[component] +
                        ((actor.poseScratch[bone].m[component] -
                             actor.poseTransitionSource[bone].m[component]) * blend);
                }
            }
            if (linear >= 1.0f) actor.poseTransitionSource.clear();
        } else {
            actor.localPoseScratch = actor.poseScratch;
            actor.poseTransitionSource.clear();
        }
        actor.poseScratch = actor.localPoseScratch;

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

void updateActorWandering(
    std::vector<SkinnedActor>& actors,
    float deltaSeconds,
    const ActorNavigationWorld* navigation,
    const std::function<bool(float, float, float, float&)>& groundHeightAt,
    const std::function<void(float&, float&, float, float, float)>& slideOutOfWalls,
    int skipIndex
) {
    // ODAI_FNV_NOWANDER=1 pins everyone to their authored spot. The control for
    // "is this actor in the wrong place because the wander put it there or
    // because the placement did".
    //
    // IT MUST NOT ALSO STOP THE GROUND SETTLE BELOW, which is what it used to
    // do by returning from here: an actor repositioned from outside keeps a
    // height that came from wherever the PLAYER was standing, so the flag left
    // a spawned companion hanging in the air. That defeats the flag's own
    // purpose -- it exists to isolate a placement bug, and instead it
    // manufactured one. Ground settling is not wandering.
    static const bool s_wanderDisabled = std::getenv("ODAI_FNV_NOWANDER") != nullptr;
    // How far from the authored spot a townsperson will stray, and how close
    // counts as arrived. The radius stays small -- ~14 m -- because an authored
    // spot is somewhere the game already believed an actor could stand, and
    // staying near one keeps a townsperson in the part of town they belong to.
    // It is no longer the ONLY thing keeping them out of walls; see
    // slideOutOfWalls below.
    constexpr float kWanderRadius = 950.0f;
    constexpr float kArriveDistance = 55.0f;
    constexpr float kTurnRateRadiansPerSecond = 2.6f;
    // Beyond this the actor is off-screen for any camera that matters, and a
    // pose it walks into is a pose nobody sees. Cheap because the alternative
    // is a ground query per actor per frame.
    constexpr float kMaxStepUnits = 400.0f;

    // Residency changes outside this update. Evaluate once for the whole
    // actor batch and include TES3's collision-derived navigation, which has
    // generated nodes but deliberately no authored NAVM meshes.
    const bool navigationAvailable =
        navigation != nullptr && navigation->hasNavigation();

    for (std::size_t i = 0; i < actors.size(); ++i) {
        SkinnedActor& actor = actors[i];
        actor.runtimeRequestedVelocity = {};
        if (!actor.placed) {
            continue;
        }

        // A hidden townsperson has no screen-space motion to preserve. Freeze
        // its local wander state until it enters the conservative visibility
        // margin. This also prevents off-screen walkers from tunnelling through
        // walls while they are not being simulated.
        if (!actor.renderVisible) {
            actor.walking = false;
            continue;
        }
        if (actor.runtimeDead) {
            actor.walking = false;
            continue;
        }

        // Project onto the walkable surface before ordinary collision.
        // Collision knows that a rock is a surface; NAVM knows it is not a path.
        // The wider first projection moves an authored rock-top placement to
        // the nearby street. Subsequent projections are a small height settle
        // that cannot jump the actor sideways to a different route.
        const odai::math::Vector3 positionBeforeSettle{
            actor.position[0], actor.position[1], actor.position[2]};
        bool settledByNavigation = false;
        if (navigationAvailable && !actor.projectedToNavigation) {
            odai::math::Vector3 point;
            if (navigation->projectPoint(
                    actor.position[0], actor.position[1], actor.position[2],
                    500.0f, 700.0f, point)) {
                actor.position[0] = point.x;
                actor.position[1] = point.y;
                actor.position[2] = point.z;
                settledByNavigation = true;
                if (std::getenv("ODAI_FNV_LOG_NAVIGATION") != nullptr) {
                    VOX_LOGI("newvegas") << "nav projected " << actor.name << " from ("
                                         << actor.wanderOrigin[0] << "," << actor.wanderOrigin[1]
                                         << "," << actor.wanderOrigin[2] << ") to ("
                                         << point.x << "," << point.y << "," << point.z << ")";
                }
                actor.projectedToNavigation = true;
                actor.wanderOrigin[0] = point.x;
                actor.wanderOrigin[1] = point.y;
                actor.wanderOrigin[2] = point.z;
                actor.wanderTarget[0] = point.x;
                actor.wanderTarget[1] = point.y;
                actor.wanderTarget[2] = point.z;
                actor.wanderPath.clear();
                actor.wanderPathIndex = 0u;
            }
        } else if (navigationAvailable && actor.projectedToNavigation) {
            // The route itself is the constraint. Avoid a global closest-point
            // search over every resident triangle for every actor every frame.
            settledByNavigation = true;
        }

        // Settle every visible actor onto collision when there is not yet an
        // authored navmesh. This remains the fallback for Morrowind and cooked
        // scenes, which do not provide NAVM records to the streamer. Two things
        // need it: a placement whose cell was not resident when the
        // actor was built has no ground to stand on yet and would hold its
        // authored height forever, and anything repositioned from outside (the
        // diagnostic parade, the spawn-side placement) arrives with a height
        // that came from wherever the PLAYER was standing.
        if (!settledByNavigation && !actor.runtimeControllerOwned) {
            float ground = 0.0f;
            if (groundHeightAt(actor.position[0], actor.position[2], actor.position[1], ground)) {
                actor.position[1] = ground;
            }
        }
        if (actor.runtimeControllerOwned &&
            (actor.position[0] != positionBeforeSettle.x ||
             actor.position[1] != positionBeforeSettle.y ||
             actor.position[2] != positionBeforeSettle.z)) {
            actor.runtimeControllerNeedsRelocation = true;
        }

        // Everything past here is the walk itself, and only it is what these
        // three suppress. Each used to skip the settle above as well.
        //
        // skipIndex is the actor being talked to, and it mattered most: a
        // conversation opens on the first tick, so a companion spawned into one
        // was excluded from the settle before it ever ran and hung in the air
        // for the whole conversation. "Someone who walks off mid-sentence is
        // worse than someone who stands still" -- but standing still means on
        // the ground.
        //
        // A zero timestep is the same case: with no time to move there is
        // nothing to advance, but the settle has already run, which is what puts
        // an actor on the ground while a conversation holds the clock still.
        if (static_cast<int>(i) == skipIndex || s_wanderDisabled || deltaSeconds <= 0.0f) {
            actor.walking = false;
            continue;
        }

        if (!actor.wanders || actor.talking ||
            (navigationAvailable && !actor.projectedToNavigation)) {
            actor.walking = false;
            continue;
        }

        if (actor.wanderPauseSeconds > 0.0f) {
            actor.wanderPauseSeconds -= deltaSeconds;
            actor.walking = false;
            continue;
        }

        if (actor.runtimeControllerOwned && actor.runtimeControllerBlocked) {
            actor.wanderPath.clear();
            actor.wanderPathIndex = 0u;
            actor.wanderTarget[0] = actor.position[0];
            actor.wanderTarget[1] = actor.position[1];
            actor.wanderTarget[2] = actor.position[2];
            actor.wanderPauseSeconds = 0.0f;
            if (actor.scriptedMoveActive) {
                // Force the package bridge to request a fresh path on the next
                // frame instead of declaring arrival at the obstruction.
                actor.scriptedMoveActive = false;
                actor.scriptedMoveRevision = 0u;
            }
            actor.walking = false;
            continue;
        }

        const float toTargetX = actor.wanderTarget[0] - actor.position[0];
        const float toTargetZ = actor.wanderTarget[2] - actor.position[2];
        const float distance = std::sqrt((toTargetX * toTargetX) + (toTargetZ * toTargetZ));
        if (distance < kArriveDistance) {
            if (navigationAvailable && actor.wanderPathIndex > 0u &&
                actor.wanderPathIndex <= actor.wanderPath.size()) {
                ActorNavigationStep& reached =
                    actor.wanderPath[actor.wanderPathIndex - 1u];
                if (reached.kind == ActorNavigationStepKind::ActivateDoor) {
                    // This is an authored off-mesh link, not movement through
                    // the gap. Relocate the character controller to XTEL's
                    // projected arrival and consume the action exactly once.
                    actor.position[0] = reached.arrivalPosition.x;
                    actor.position[1] = reached.arrivalPosition.y;
                    actor.position[2] = reached.arrivalPosition.z;
                    actor.runtimeControllerNeedsRelocation =
                        actor.runtimeControllerOwned;
                    reached.kind = ActorNavigationStepKind::Walk;
                    if (actor.wanderPathIndex < actor.wanderPath.size()) {
                        const ActorNavigationStep& waypoint =
                            actor.wanderPath[actor.wanderPathIndex++];
                        actor.wanderTarget[0] = waypoint.position.x;
                        actor.wanderTarget[1] = waypoint.position.y;
                        actor.wanderTarget[2] = waypoint.position.z;
                    } else {
                        actor.wanderTarget[0] = actor.position[0];
                        actor.wanderTarget[1] = actor.position[1];
                        actor.wanderTarget[2] = actor.position[2];
                    }
                    actor.walking = false;
                    continue;
                }
            }
            if (navigationAvailable &&
                actor.wanderPathIndex < actor.wanderPath.size()) {
                const ActorNavigationStep& waypoint =
                    actor.wanderPath[actor.wanderPathIndex++];
                actor.wanderTarget[0] = waypoint.position.x;
                actor.wanderTarget[1] = waypoint.position.y;
                actor.wanderTarget[2] = waypoint.position.z;
                // This is one continuous stride across a route waypoint, not
                // an arrival. Selecting idle for this bookkeeping frame makes
                // the skeleton visibly hitch at every shared navmesh edge.
                actor.walking = true;
                continue;
            }

            if (actor.scriptedMoveActive) {
                actor.scriptedMoveActive = false;
                actor.scriptedMoveArrived = true;
                actor.walking = false;
                continue;
            }

            // Arrived (or never had a target): stand for a moment, then pick
            // the next spot. The pause is most of what makes this read as
            // people rather than as patrol routes -- a town where everyone is
            // permanently in motion looks as wrong as one where nobody is.
            core::Lcg32 rng(actor.wanderRng);
            if (navigationAvailable) {
                const odai::math::Vector3 start{
                    actor.position[0], actor.position[1], actor.position[2]};
                const odai::math::Vector3 origin{
                    actor.wanderOrigin[0], actor.wanderOrigin[1], actor.wanderOrigin[2]};
                actor.wanderPath.clear();
                actor.wanderPathIndex = 0u;
                if (navigation->buildWanderPath(
                        start, origin, kWanderRadius, rng.next24(), actor.wanderPath) &&
                    !actor.wanderPath.empty()) {
                    const ActorNavigationStep& waypoint = actor.wanderPath.front();
                    actor.wanderTarget[0] = waypoint.position.x;
                    actor.wanderTarget[1] = waypoint.position.y;
                    actor.wanderTarget[2] = waypoint.position.z;
                    actor.wanderPathIndex = 1u;
                }
            } else {
                const float angle =
                    static_cast<float>(rng.next24() % 3600u) * (3.14159265f / 1800.0f);
                const float reach =
                    0.35f + (static_cast<float>(rng.next24() % 1000u) * 0.00065f);
                actor.wanderTarget[0] =
                    actor.wanderOrigin[0] + (std::cos(angle) * kWanderRadius * reach);
                actor.wanderTarget[2] =
                    actor.wanderOrigin[2] + (std::sin(angle) * kWanderRadius * reach);
            }
            actor.wanderPauseSeconds =
                1.5f + (static_cast<float>(rng.next24() % 1000u) * 0.0055f);  // 1.5..7.0 s
            actor.wanderRng = rng.state();
            actor.walking = false;
            continue;
        }

        // Turn first, then walk along the facing rather than straight at the
        // target: an actor that translates toward a point it is not yet facing
        // moon-walks sideways for the length of the turn.
        const float desiredYaw = actorYawForDirection(toTargetX, toTargetZ);
        const float delta = angleDelta(actor.yawRadians, desiredYaw);
        const float maxTurn = kTurnRateRadiansPerSecond * deltaSeconds;
        actor.yawRadians += std::clamp(delta, -maxTurn, maxTurn);

        const odai::math::Vector3 facing = actorFacing(actor.yawRadians);
        const float step =
            std::min(actor.walkSpeedUnitsPerSecond * deltaSeconds, kMaxStepUnits);
        // Ramp naturally from turning in place to a full stride. The old
        // binary 15%/100% choice produced a sevenfold velocity jump on the
        // first sufficiently aligned frame, visible as a positional jerk even
        // when the walk pose itself was smooth.
        const float facingDot = distance > 1e-3f
            ? std::clamp(((facing.x * toTargetX) + (facing.z * toTargetZ)) / distance,
                  -1.0f, 1.0f)
            : 1.0f;
        const float alignmentLinear = std::clamp((facingDot - 0.35f) / 0.60f, 0.0f, 1.0f);
        const float alignment = alignmentLinear * alignmentLinear *
            (3.0f - (2.0f * alignmentLinear));
        const float fromX = actor.position[0];
        const float fromZ = actor.position[2];
        const float stride = step * alignment;
        if (actor.runtimeControllerOwned) {
            actor.runtimeRequestedVelocity = {
                facing.x * actor.walkSpeedUnitsPerSecond * alignment,
                0.0f,
                facing.z * actor.walkSpeedUnitsPerSecond * alignment};
        } else {
            actor.position[0] = fromX + (facing.x * stride);
            actor.position[2] = fromZ + (facing.z * stride);
            if (navigationAvailable && distance > 1e-3f) {
                const float routeFraction = std::min(stride / distance, 1.0f);
                actor.position[1] +=
                    (actor.wanderTarget[1] - actor.position[1]) * routeFraction;
            }
        }

        // Push back out of anything solid. The capsule is the actor's own: the
        // player's is 34 units around a 120-unit body, so scaling by height
        // keeps a bighorner wide and a radroach narrow rather than giving every
        // creature a person's girth. Clamped because a rig that failed to
        // measure would otherwise be either intangible or a moving wall.
        if (!actor.runtimeControllerOwned && slideOutOfWalls) {
            const float height =
                actor.standingHeightUnits > 1.0f ? actor.standingHeightUnits : 120.0f;
            const float radius = std::clamp(height * 0.28f, 12.0f, 48.0f);
            slideOutOfWalls(
                actor.position[0], actor.position[2], actor.position[1],
                actor.position[1] + height, radius);
        }

        // A blocked actor picks somewhere else to go rather than leaning on the
        // wall for the rest of its stride. Without this the slide above is
        // silent and permanent: the walk animation keeps playing against a
        // surface the actor can never get past, which reads worse than the
        // clipping did.
        const float movedX = actor.position[0] - fromX;
        const float movedZ = actor.position[2] - fromZ;
        const float moved = std::sqrt((movedX * movedX) + (movedZ * movedZ));
        if (!actor.runtimeControllerOwned && stride > 1e-3f && moved < (stride * 0.35f)) {
            actor.wanderPauseSeconds = 0.0f;
            actor.wanderTarget[0] = actor.position[0];
            actor.wanderTarget[2] = actor.position[2];  // counts as arrived: repick next tick
        }
        actor.walking = alignment > 0.02f;
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
    // WHERE THE HEAD ACTUALLY IS THIS FRAME, not where the bind pose put it.
    //
    // poseScratch holds the matrices the skinning shader itself applies, with
    // the actor's world placement already folded in, so pushing the head
    // centroid through its own bone matrix gives the head's world position --
    // the one on screen. Subtracting the placement turns it back into a height
    // above the feet, which is what the caller adds it to.
    //
    // This is what a static measurement could not do. Willow's idle displaces
    // her body far enough that a bind-pose height framed her at the waist while
    // every number in the camera arithmetic checked out; freezing her with
    // ODAI_FNV_NOANIM=1 framed her correctly and was what isolated it. A head
    // moves when an actor animates, so the aim has to move with it.
    if (actor.headAnchorBone >= 0 &&
        static_cast<std::size_t>(actor.headAnchorBone) < actor.poseScratch.size()) {
        const odai::math::Matrix4& matrix =
            actor.poseScratch[static_cast<std::size_t>(actor.headAnchorBone)];
        const float worldY = (matrix(1, 0) * actor.headAnchorLocal[0]) +
            (matrix(1, 1) * actor.headAnchorLocal[1]) +
            (matrix(1, 2) * actor.headAnchorLocal[2]) + matrix(1, 3);
        const float aboveFeet = worldY - actor.position[1];
        // A positive result alone is not enough. TES3 animation controllers
        // can expose a skin matrix whose pivot is valid for the vertices but
        // not an anatomical bone-space point. Such a matrix put a nominal
        // "head" anchor around the pelvis: dialogue then narrowed the lens
        // around the actor's crotch while the real face left the top edge.
        //
        // The bind-pose head height was measured through the same skinned
        // geometry at import time, so it is the stable anatomical reference.
        // An idle may bob or turn the head, but it must not move it by a large
        // fraction of the actor's entire standing height. Keep creature rigs
        // working too by validating displacement from their own authored head
        // rather than imposing a human-only height fraction.
        const float allowedPoseDisplacement = std::max(
            12.0f, (actor.standingHeightUnits > 1.0f
                       ? actor.standingHeightUnits
                       : actor.headHeightUnits) * 0.20f);
        const bool agreesWithBindPose = actor.headHeightUnits <= 1.0f ||
            std::abs(aboveFeet - actor.headHeightUnits) <= allowedPoseDisplacement;
        // Also guards a pose that has not been written yet, or a rig whose
        // head lands below its own feet: either would aim at the ground.
        if (aboveFeet > 1.0f && agreesWithBindPose) {
            return aboveFeet;
        }
    }
    if (actor.headHeightUnits > 1.0f) {
        return actor.headHeightUnits;  // measured, but bind pose only
    }
    // NO USABLE HEAD BONE. Legacy TES3 body parts can leave the named head bone
    // with no dominantly weighted vertices, so this path is common for NPCs,
    // not just radroaches and eyebots. An upright person's face is near 88% of
    // standing height; 65% remains the useful fallback for creatures and for
    // Victor's low screen-on-a-wheel silhouette.
    constexpr float kHumanoidFaceFraction = 0.88f;
    constexpr float kCreatureFaceFraction = 0.65f;
    constexpr float kFallbackUnits = 150.0f;
    const float faceFraction =
        actor.humanoid ? kHumanoidFaceFraction : kCreatureFaceFraction;
    return actor.standingHeightUnits > 1.0f
        ? (actor.standingHeightUnits * faceFraction)
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
