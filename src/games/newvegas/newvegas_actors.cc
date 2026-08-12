#include "games/newvegas/newvegas_actors.h"

#include "core/log.h"
#include "import/dds.h"
#include "import/fnv/kf_animation.h"
#include "import/fnv/nif_scene.h"

#include <algorithm>
#include <cctype>
#include <cmath>
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
        actor.character = built.character;
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

void updateActorPoses(std::vector<SkinnedActor>& actors, float deltaSeconds) {
    for (SkinnedActor& actor : actors) {
        if (actor.character.skeleton.bones.empty()) {
            continue;
        }
        actor.animationSeconds += deltaSeconds;
        if (actor.idleClip.tracks.empty() || actor.idleClip.duration <= 0.0f) {
            importer::fnv::computeFalloutBindPose(actor.character, actor.poseScratch);
        } else {
            actor.sampler.sample(
                actor.character.skeleton, actor.idleClip, actor.animationSeconds,
                actor.poseScratch);
        }
        const odai::math::Matrix4 actorWorld =
            odai::math::Matrix4::translation(odai::math::Vector3{
                actor.position[0], actor.position[1], actor.position[2]}) *
            odai::math::Matrix4::rotationY(actor.yawRadians);
        for (odai::math::Matrix4& matrix : actor.poseScratch) {
            matrix = actorWorld * matrix;
        }
    }
}

}  // namespace odai::games::newvegas
