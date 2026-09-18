#include <cstdlib>
#include "anim/skeleton_io.h"
#include "import/fnv/skyrim_animation_assets.h"
#include "import/fnv/character_asset_manifest.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <set>
#include <functional>
#include <cmath>
#include <tuple>

namespace odai::importer::fnv {
namespace {

std::string lowerAscii(std::string text) {
    for (char& ch : text) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return text;
}

odai::anim::HkxGeneratorIdentity providerGenerator(
    const FalloutAssetSource::ResolvedAsset& asset) {
    const std::string identity = lowerAscii(asset.providerId + " " + asset.providerName + " " +
        asset.providerRoot.generic_string());
    if (identity.find("nemesis") != std::string::npos) return odai::anim::HkxGeneratorIdentity::Nemesis;
    if (identity.find("fnis") != std::string::npos) return odai::anim::HkxGeneratorIdentity::Fnis;
    return odai::anim::HkxGeneratorIdentity::Unknown;
}

}  // namespace

bool inspectSkyrimAnimationBundle(
    const FalloutAssetSource& assets, SkyrimAnimationAssetReport& out,
    bool strict, std::string& outError) {
    out = SkyrimAnimationAssetReport{};
    outError.clear();
    static constexpr std::array roots{
        "meshes\\actors\\character\\behaviors\\0_master.hkx",
        "meshes\\actors\\character\\characters\\defaultmale.hkx",
        "meshes\\actors\\character\\characters female\\defaultfemale.hkx"};
    std::set<odai::anim::HkxGeneratorIdentity> identities;
    for (const char* path : roots) {
        FalloutAssetSource::ResolvedAsset asset;
        std::string error;
        if (!assets.resolveAssetWithProvider(path, asset, error)) {
            out.missingAssets.emplace_back(path);
            out.diagnostics.push_back(std::string("missing generated root: ") + path + " (" + error + ")");
            continue;
        }
        odai::anim::HkxPackfileSummary summary;
        if (!odai::anim::inspectHkxPackfile(asset.bytes, summary, error)) {
            out.diagnostics.push_back(std::string("invalid generated root: ") + path + " (" + error + ")");
            continue;
        }
        odai::anim::HkxGeneratorIdentity identity = providerGenerator(asset);
        if (identity == odai::anim::HkxGeneratorIdentity::Unknown) identity = summary.generator;
        // A root that contains only references/string data may carry no class
        // proving which generator wrote it. Unknown is absence of evidence,
        // not a second generator identity; counting it made three files from
        // the same retail BSA report as a mixed generated-root installation.
        if (identity != odai::anim::HkxGeneratorIdentity::Unknown) {
            identities.insert(identity);
        }
        out.unsupportedClasses.insert(out.unsupportedClasses.end(),
            summary.unsupportedBehaviorClasses.begin(), summary.unsupportedBehaviorClasses.end());
        if (summary.containsBehaviorGraph) {
            odai::anim::HkxDecodedBehaviorGraph graph;
            std::string decodeError;
            if (!odai::anim::decodeHkxBehaviorGraph(asset.bytes, graph, decodeError)) {
                out.diagnostics.push_back(std::string("undecodable behavior graph: ") +
                    path + " (" + decodeError + ")");
            } else {
                out.behaviorGraphs.push_back(std::move(graph));
            }
        }
        out.roots.push_back(std::move(asset));
    }
    std::sort(out.unsupportedClasses.begin(), out.unsupportedClasses.end());
    out.unsupportedClasses.erase(
        std::unique(out.unsupportedClasses.begin(), out.unsupportedClasses.end()),
        out.unsupportedClasses.end());
    out.coherent = out.roots.size() == roots.size();
    if (!out.roots.empty()) out.generatorProvider = out.roots.front().providerName;
    if (identities.size() == 1u) out.generator = *identities.begin();
    // Provider labels describe provenance, not binary compatibility. Generated
    // output can legitimately override only part of a retail character bundle.
    for (const auto& root : out.roots) {
        if (root.canonicalVirtualPath.find("defaultmale.hkx") == std::string::npos &&
            root.canonicalVirtualPath.find("defaultfemale.hkx") == std::string::npos) continue;
        odai::anim::HkxCharacterAssets character;
        std::string error;
        if (!odai::anim::decodeHkxCharacterAssets(root.bytes, character, error)) {
            out.coherent = false;
            out.diagnostics.push_back(root.canonicalVirtualPath + ": " + error);
            continue;
        }
        const bool female = root.canonicalVirtualPath.find("defaultfemale.hkx") != std::string::npos;
        const std::string prefix = "meshes\\actors\\character\\";
        std::string skeleton = character.skeletonPath.empty()
            ? (female ? "character assets female\\skeleton_female.hkx" : "character assets\\skeleton.hkx")
            : character.skeletonPath;
        skeleton = lowerAscii(normalizeModelPath(skeleton));
        if (!skeleton.starts_with("meshes\\")) skeleton = prefix + skeleton;
        FalloutAssetSource::ResolvedAsset asset;
        odai::anim::HkxDecodedSkeleton rig;
        if (!assets.resolveAssetWithProvider(skeleton, asset, error)) {
            out.coherent = false; out.missingAssets.push_back(skeleton);
            out.diagnostics.push_back(skeleton + ": " + error);
        } else if (!odai::anim::decodeHkxAnimationSkeleton(asset.bytes, rig, error)) {
            out.coherent = false; out.diagnostics.push_back(skeleton + ": " + error);
        }
        auto behavior = lowerAscii(normalizeModelPath(character.behaviorPath.empty()
            ? "behaviors\\0_master.hkx" : character.behaviorPath));
        if (!behavior.starts_with("meshes\\")) behavior = prefix + behavior;
        std::string fingerprint;
        if (!loadSkyrimBehaviorProgram(assets, behavior, fingerprint, error)) {
            out.coherent = false; out.diagnostics.push_back(behavior + ": " + error);
        }
    }
    out.strictCompatible = out.coherent && out.unsupportedClasses.empty() &&
        !out.behaviorGraphs.empty();
    if (strict && !out.strictCompatible) {
        outError = !out.coherent ? "incoherent Skyrim generated animation bundle" :
            !out.unsupportedClasses.empty() ?
                "unsupported gameplay behavior classes in Skyrim animation bundle" :
                "Skyrim master behavior graph could not be decoded";
        return false;
    }
    return true;
}

std::shared_ptr<const odai::anim::BehaviorProgram> loadSkyrimBehaviorProgram(
    const FalloutAssetSource& assets, const std::string& rootPath,
    std::string& fingerprint, std::string& error) {
    using namespace odai::anim;
    fingerprint.clear();
    error.clear();
    HkxDecodedBehaviorGraph combined;
    std::map<std::string, std::uint32_t> roots;
    std::set<std::string> visiting;
    std::function<bool(std::string, std::uint32_t&)> load = [&](std::string path, std::uint32_t& root) {
        path = lowerAscii(normalizeModelPath(std::move(path)));
        if (visiting.contains(path)) { error = "cyclic behavior reference: " + path; return false; }
        if (auto found = roots.find(path); found != roots.end()) { root = found->second; return true; }
        if (roots.size() >= 64) { error = "behavior reference limit exceeded"; return false; }
        FalloutAssetSource::ResolvedAsset asset;
        if (!assets.resolveAssetWithProvider(path, asset, error)) return false;
        HkxDecodedBehaviorGraph graph;
        if (!decodeHkxBehaviorGraph(asset.bytes, graph, error)) return false;
        if (combined.nodes.size() + graph.nodes.size() > 65536u) {
            error = "expanded behavior node limit exceeded"; return false;
        }
        fingerprint += asset.canonicalVirtualPath + "@" + asset.contentFingerprint + ";";
        const auto base = static_cast<std::uint32_t>(combined.nodes.size());
        root = base + graph.rootNode;
        roots.emplace(path, root);
        visiting.insert(path);
        combined.unsupportedVariables.insert(combined.unsupportedVariables.end(),
            graph.unsupportedVariables.begin(), graph.unsupportedVariables.end());
        for (const auto& [name, value] : graph.variableDefaults) {
            const auto [entry, inserted] = combined.variableDefaults.emplace(name, value);
            if (!inserted && entry->second != value)
                combined.unsupportedVariables.push_back(name + ": referenced graphs have differing defaults");
        }
        std::vector<std::int32_t> variableMap;
        for (const auto& variable : graph.variableNames) {
            auto found = std::find(combined.variableNames.begin(), combined.variableNames.end(), variable);
            if (found == combined.variableNames.end()) {
                variableMap.push_back(static_cast<std::int32_t>(combined.variableNames.size()));
                combined.variableNames.push_back(variable);
            } else variableMap.push_back(static_cast<std::int32_t>(found - combined.variableNames.begin()));
        }
        std::vector<std::int32_t> eventMap;
        for (const auto& name : graph.eventNames) {
            auto found = std::find(combined.eventNames.begin(), combined.eventNames.end(), name);
            if (found == combined.eventNames.end()) {
                eventMap.push_back(static_cast<std::int32_t>(combined.eventNames.size()));
                combined.eventNames.push_back(name);
            } else eventMap.push_back(static_cast<std::int32_t>(found - combined.eventNames.begin()));
        }
        const auto remapEvent = [&](std::int32_t& event) {
            if (event >= 0) event = static_cast<std::size_t>(event) < eventMap.size() ? eventMap[event] : -2;
        };
        for (auto& node : graph.nodes) {
            for (auto& binding : node.bindings)
                if (binding.bindingType == 0)
                    binding.variableIndex = binding.variableIndex >= 0 &&
                        static_cast<std::size_t>(binding.variableIndex) < variableMap.size()
                        ? variableMap[binding.variableIndex] : -1;
            for (auto& child : node.children) child += base;
            for (auto& rule : node.transitions) {
                if (rule.effectNode >= 0) rule.effectNode += base;
                remapEvent(rule.eventId);
                remapEvent(rule.triggerInterval.enterEventId); remapEvent(rule.triggerInterval.exitEventId);
                remapEvent(rule.initiateInterval.enterEventId); remapEvent(rule.initiateInterval.exitEventId);
            }
            for (auto& trigger : node.triggers) remapEvent(trigger.eventId);
            if (node.kind == HkxBehaviorNodeKind::Clip) {
                node.assetPath = lowerAscii(normalizeModelPath(node.assetPath));
                if (!node.assetPath.starts_with("meshes\\"))
                    node.assetPath = "meshes\\actors\\character\\" + node.assetPath;
            }
            combined.nodes.push_back(std::move(node));
        }
        for (std::uint32_t i = base; i < base + graph.nodes.size(); ++i) {
            if (combined.nodes[i].kind != HkxBehaviorNodeKind::BehaviorReference) continue;
            std::string reference = lowerAscii(normalizeModelPath(combined.nodes[i].assetPath));
            if (!reference.starts_with("meshes\\")) {
                if (!reference.starts_with("behaviors\\")) reference = "behaviors\\" + reference;
                reference = "meshes\\actors\\character\\" + reference;
            }
            std::uint32_t child = 0;
            if (!load(reference, child)) return false;
            // A reference becomes a graph wrapper after name/index linking.
            combined.nodes[i].kind = HkxBehaviorNodeKind::Graph;
            combined.nodes[i].children = {child};
        }
        visiting.erase(path);
        return true;
    };
    std::uint32_t root = 0;
    if (!load(rootPath, root)) return nullptr;
    combined.rootNode = root;
    combined.name = rootPath;
    return compileBehaviorProgram(std::move(combined));
}

void loadSkyrimNativeAnimationPacks(const FalloutAssetSource& assets,
    odai::anim::AnimationView& view, const odai::anim::HkxDecodedSkeleton* sourceSkeleton) {
    using namespace odai::anim;
    std::string error;
    if (const char* mode = std::getenv("ODAI_SKYRIM_ANIMATION_MODE"); mode && std::string_view(mode) == "havok")
        view.executionMode = AnimationExecutionMode::Havok;
    // Decode each source once, then copy it into independent playback views.
    // A jump policy must never leak into an animation-driven use of the same file.
    std::map<std::string, AnimationClip> sources;
    std::map<std::tuple<std::string, bool, bool>, std::string> playbacks;
    const auto loadClip = [&](const std::string& path, bool loop, bool controllerJump,
                              std::string& playbackName) {
        if (!view.skeleton) return false;
        const auto key = std::make_tuple(path, loop, controllerJump);
        if (auto found = playbacks.find(key); found != playbacks.end()) {
            playbackName = found->second;
            return true;
        }
        if (!sources.contains(path)) {
            FalloutAssetSource::ResolvedAsset asset;
            AnimationClip authored;
            HkxDecodedClipMetadata metadata;
            if (!loadCharacterSourceClip(assets, path, *view.skeleton, sourceSkeleton,
                    true, authored, metadata, asset, error)) return false;
            view.sourceFingerprint += path + "@" + asset.providerId + ":" + asset.contentFingerprint + ";";
            sources.emplace(path, std::move(authored));
        }
        auto clip = sources.at(path);
        clip.loop = loop;
        clip.extractedMotionBone = view.skeleton->findBone("NPC Root [Root]");
        if (controllerJump) {
            const int com = view.skeleton->findBone("NPC COM [COM ]");
            for (auto& track : clip.tracks) {
                if (track.boneIndex != com || track.translationKeys.empty()) continue;
                const float height = track.translationKeys.front().value.y;
                for (auto& frame : track.translationKeys) frame.value.y = height;
            }
        }
        clip.name = path;
        if (std::any_of(view.clips.begin(), view.clips.end(), [&](const auto& existing) { return existing.name == clip.name; }))
            clip.name += std::string("#native:") + (loop ? "loop" : "once") + (controllerJump ? ":controller-jump" : ":authored");
        playbackName = clip.name;
        playbacks.emplace(key, playbackName);
        view.clips.push_back(std::move(clip));
        return true;
    };
    auto native = std::make_shared<NativeAnimationProgram>();
    auto paths = assets.virtualPaths();
    std::sort(paths.begin(), paths.end());
    for (const auto& path : paths) {
        if (!path.starts_with("odai\\animations\\") || !path.ends_with(".json")) continue;
        FalloutAssetSource::ResolvedAsset pack;
        NativeAnimationProgram compiled;
        if (!assets.resolveAssetWithProvider(path, pack, error)) continue;
        view.sourceFingerprint += path + "@" + pack.providerId + ":" + std::to_string(pack.layerPriority) + ":" + pack.contentFingerprint + ";";
        if (!compileNativeAnimationPack(std::string_view(reinterpret_cast<const char*>(pack.bytes.data()), pack.bytes.size()),
                pack.providerId, pack.layerPriority, compiled, error)) {
            view.diagnostics.push_back({AnimationDiagnosticSeverity::Warning, "native.pack_invalid", path + ": " + error});
            continue;
        }
        for (auto& rule : compiled.rules) {
            // File-qualified IDs are stable and unique across independent packs.
            rule.id = path + "#" + rule.id;
            for (auto& variant : rule.variants) {
                if (variant.clip.find_first_of("/\\") == std::string::npos) {
                    const auto existing = std::find_if(view.clips.begin(), view.clips.end(),
                        [&](const auto& clip) { return clip.name == variant.clip; });
                    if (existing != view.clips.end() && existing->loop != variant.loop) {
                        auto playback = *existing;
                        playback.name += std::string("#native:") + (variant.loop ? "loop" : "once");
                        playback.loop = variant.loop;
                        variant.clip = playback.name;
                        if (std::none_of(view.clips.begin(), view.clips.end(),
                            [&](const auto& clip) { return clip.name == playback.name; })) view.clips.push_back(std::move(playback));
                    }
                    continue;
                }
                const auto path = characterAssetPath(variant.clip);
                std::string playbackName;
                if (path.empty() || !path.starts_with("meshes\\") ||
                    !loadClip(path, variant.loop, !rule.animationDriven &&
                        (rule.state == "jump" || rule.state.starts_with("jump_")), playbackName)) {
                    view.diagnostics.push_back({AnimationDiagnosticSeverity::Warning, "native.clip_unavailable", variant.clip + ": " + error});
                } else variant.clip = playbackName;
            }
            const auto loadAuxiliary = [&](std::string& name) {
                if (name.empty() || name.find_first_of("/\\") == std::string::npos) return;
                const auto path = characterAssetPath(name);
                std::string playbackName;
                if (path.empty() || !path.starts_with("meshes\\") || !loadClip(path, true, false, playbackName))
                    view.diagnostics.push_back({AnimationDiagnosticSeverity::Warning, "native.clip_unavailable", name + ": " + error});
                else name = playbackName;
            };
            for (auto& sample : rule.blendSamples) loadAuxiliary(sample.clip);
            for (auto& layer : rule.layers) { loadAuxiliary(layer.clip); loadAuxiliary(layer.referenceClip); }
            if (rule.graph) {
                auto graph = std::make_shared<PoseGraphProgram>(*rule.graph);
                for (auto& [id, node] : graph->nodes) {
                    loadAuxiliary(node.clip);
                    loadAuxiliary(node.reference);
                }
                rule.graph = std::move(graph);
            }
            native->rules.push_back(std::move(rule));
        }
    }
    view.nativeProgram = std::move(native);
    view.sourceFingerprint += view.executionMode == AnimationExecutionMode::Native ? "mode:native-v2-source-playback;" : "mode:havok;";
}

std::shared_ptr<const odai::anim::AnimationView> loadSkyrimNpcAnimationView(
    const FalloutAssetSource& assets, const odai::anim::Skeleton& skeleton,
    const std::vector<odai::math::Matrix4>& inverseBind, bool female,
    const odai::anim::AnimationClip& idle, const odai::anim::AnimationClip& walk,
    std::shared_ptr<const odai::anim::BehaviorProgram> behavior,
    const std::string& graphFingerprint) {
    using namespace odai::anim;
    auto view = std::make_shared<AnimationView>();
    if (const char* mode = std::getenv("ODAI_SKYRIM_ANIMATION_MODE"); mode && std::string_view(mode) == "havok")
        view->executionMode = AnimationExecutionMode::Havok;
    view->selectorDefaults.values["view"] = std::string("third_person");
    view->selectorDefaults.values["sex"] = std::string(female ? "female" : "male");
    view->skeleton = std::make_shared<const Skeleton>(skeleton);
    view->inverseBindMatrices = inverseBind;
    auto rigMapping = std::make_shared<HumanoidRigMapping>();
    std::string rigError;
    if (canonicalizeHumanoidRig(skeleton, *rigMapping, rigError)) view->humanoidRig = std::move(rigMapping);
    view->clips = {idle, walk};
    view->stateClips = {{"idle", idle.name}, {"locomotion", walk.name}, {"walk_forward", walk.name}};
    view->behavior = std::move(behavior);
    view->sourceFingerprint = graphFingerprint;
    std::uint64_t rigHash = 14695981039346656037ull;
    const auto hashWord = [&](std::uint32_t word) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            rigHash ^= (word >> shift) & 0xffu; rigHash *= 1099511628211ull;
        }
    };
    for (const auto& bone : skeleton.bones) {
        hashWord(static_cast<std::uint32_t>(bone.name.size()));
        for (unsigned char c : bone.name) hashWord(c);
        hashWord(static_cast<std::uint32_t>(bone.parentIndex));
        for (float value : {bone.localTranslation.x, bone.localTranslation.y, bone.localTranslation.z,
                bone.localRotation.x, bone.localRotation.y, bone.localRotation.z, bone.localRotation.w,
                bone.localScale.x, bone.localScale.y, bone.localScale.z}) hashWord(std::bit_cast<std::uint32_t>(value));
    }
    for (const auto& matrix : inverseBind)
        for (float value : matrix.m) hashWord(std::bit_cast<std::uint32_t>(value));
    view->sourceFingerprint += "render-rig@" + std::to_string(rigHash) + ";";
    for (const auto& clip : view->clips) {
        hashWord(static_cast<std::uint32_t>(clip.name.size()));
        for (unsigned char c : clip.name) hashWord(c);
        hashWord(std::bit_cast<std::uint32_t>(clip.duration));
        hashWord(clip.loop); hashWord(clip.additive);
        hashWord(static_cast<std::uint32_t>(clip.tracks.size()));
        for (const auto& track : clip.tracks) {
            hashWord(static_cast<std::uint32_t>(track.boneIndex));
            const auto vectors = [&](const auto& keys) {
                hashWord(static_cast<std::uint32_t>(keys.size()));
                for (const auto& key : keys) for (float value : {key.time, key.value.x, key.value.y, key.value.z})
                    hashWord(std::bit_cast<std::uint32_t>(value));
            };
            vectors(track.translationKeys); vectors(track.scaleKeys);
            hashWord(static_cast<std::uint32_t>(track.rotationKeys.size()));
            for (const auto& key : track.rotationKeys)
                for (float value : {key.time, key.value.x, key.value.y, key.value.z, key.value.w})
                    hashWord(std::bit_cast<std::uint32_t>(value));
        }
        hashWord(static_cast<std::uint32_t>(clip.annotations.size()));
        for (const auto& marker : clip.annotations) {
            hashWord(std::bit_cast<std::uint32_t>(marker.time)); hashWord(marker.firstCycleOnly);
            hashWord(static_cast<std::uint32_t>(marker.name.size()));
            for (unsigned char c : marker.name) hashWord(c);
        }
    }
    view->sourceFingerprint += "builtin-clips@" + std::to_string(rigHash) + ";";

    view->providerId = female ? "skyrim-npc-female" : "skyrim-npc-male";
    view->socketBoneNames = {"NPC R Hand [RHnd]", "NPC L Hand [LHnd]", "WEAPON", "SHIELD", "QUIVER",
        "WeaponSword", "WeaponDagger", "WeaponAxe", "WeaponMace", "WeaponBack", "WeaponBow"};
    const std::string characterRoot = "meshes\\actors\\character\\";
    const std::string sex = female ? "female" : "male";
    FalloutAssetSource::ResolvedAsset source;
    HkxDecodedSkeleton sourceSkeleton;
    std::string error;
    const std::string characterPath = characterRoot + (female
        ? "characters female\\defaultfemale.hkx" : "characters\\defaultmale.hkx");
    HkxCharacterAssets character;
    if (!assets.resolveAssetWithProvider(characterPath, source, error) ||
        !decodeHkxCharacterAssets(source.bytes, character, error)) {
        view->diagnostics.push_back({AnimationDiagnosticSeverity::Error, "character.invalid", error});
        view->behavior.reset();
        loadSkyrimNativeAnimationPacks(assets, *view);
        return view;
    }
    view->sourceFingerprint += source.canonicalVirtualPath + "@" + source.contentFingerprint;
    const auto assetPath = [&](std::string path) {
        path = lowerAscii(normalizeModelPath(std::move(path)));
        return path.starts_with("meshes\\") ? path : characterRoot + path;
    };
    const std::string skeletonPath = character.skeletonPath.empty()
        ? characterRoot + "character assets" + (female ? " female\\skeleton_female.hkx" : "\\skeleton.hkx")
        : assetPath(character.skeletonPath);
    const std::string behaviorPath = character.behaviorPath.empty()
        ? characterRoot + "behaviors\\0_master.hkx" : assetPath(character.behaviorPath);
    view->characterAssets = character;
    view->characterDefinitionPath = characterPath;
    view->animationSkeletonPath = skeletonPath;
    if (behaviorPath != characterRoot + "behaviors\\0_master.hkx") {
        std::string fingerprint;
        view->behavior = loadSkyrimBehaviorProgram(assets, behaviorPath, fingerprint, error);
        view->sourceFingerprint += fingerprint;
        if (!view->behavior) view->diagnostics.push_back({AnimationDiagnosticSeverity::Error,
            "character.behavior_missing", error});
    }
    if (!assets.resolveAssetWithProvider(skeletonPath, source, error) ||
        !decodeHkxAnimationSkeleton(source.bytes, sourceSkeleton, error)) {
        view->diagnostics.push_back({AnimationDiagnosticSeverity::Warning, "rig.hkx_missing", error});
        view->behavior.reset();
        loadSkyrimNativeAnimationPacks(assets, *view);
        return view;
    }
    view->sourceFingerprint += source.contentFingerprint;
    const auto loadClip = [&](const std::string& path, const std::string& state, bool loop) {
        FalloutAssetSource::ResolvedAsset asset;
        AnimationClip clip;
        HkxDecodedClipMetadata metadata;
        if (!loadCharacterSourceClip(assets, path, skeleton, &sourceSkeleton,
                false, clip, metadata, asset, error)) return false;
        if (metadata.missingTracks) view->diagnostics.push_back({AnimationDiagnosticSeverity::Warning,
            "rig.unbound_tracks", path + ": " + std::to_string(metadata.missingTracks) +
                " source tracks have no render-rig bone; omitted without index substitution"});
        view->sourceFingerprint += asset.canonicalVirtualPath + "@" + asset.contentFingerprint + ";";
        clip.loop = loop;
        if (view->behavior) {
            for (const auto& node : view->behavior->graph.nodes) {
                if (node.kind != HkxBehaviorNodeKind::Clip || node.assetPath != path) continue;
                for (const auto& trigger : node.triggers) {
                    if (trigger.eventId < 0 || static_cast<std::size_t>(trigger.eventId) >=
                        view->behavior->graph.eventNames.size() || trigger.hasPayload) continue;
                    const float time = trigger.relativeToEnd ? clip.duration + trigger.time : trigger.time;
                    const auto& name = view->behavior->graph.eventNames[trigger.eventId];
                    if (time < 0 || time > clip.duration) continue;
                    if (std::none_of(clip.annotations.begin(), clip.annotations.end(), [&](const auto& existing) {
                        return existing.name == name && std::abs(existing.time - time) < 0.0001f;
                    })) clip.annotations.push_back({time, name, trigger.acyclic});
                }
                break;
            }
        }
        std::stable_sort(clip.annotations.begin(), clip.annotations.end(),
            [](const auto& a, const auto& b) { return a.time < b.time; });
        // Navigation owns translation for these ordinary NPC states. Preserve
        // vertical bob; remove horizontal root drift from the sampled pose.
        int root = skeleton.findBone("NPC Root [Root]");
        if (root < 0) root = skeleton.findBone("NPC COM [COM ]");
        for (auto& track : clip.tracks) {
            if (track.boneIndex != root || track.translationKeys.empty()) continue;
            const auto first = track.translationKeys.front().value;
            for (auto& key : track.translationKeys) { key.value.x = first.x; key.value.z = first.z; }
        }
        // Retail mt_jump includes an authored COM lift. The character controller
        // supplies the world-space jump arc; retain the limb rotations without
        // adding that lift a second time. Landing compression remains authored.
        if (view->executionMode == AnimationExecutionMode::Native && state == "jump") {
            const int com = skeleton.findBone("NPC COM [COM ]");
            for (auto& track : clip.tracks) {
                if (track.boneIndex != com || track.translationKeys.empty()) continue;
                const float takeoffHeight = track.translationKeys.front().value.y;
                for (auto& key : track.translationKeys) key.value.y = takeoffHeight;
            }
        }
        view->sourceFingerprint += asset.canonicalVirtualPath + "@" + asset.contentFingerprint + ";";
        view->stateClips[state] = path;
        view->clips.push_back(std::move(clip));
        return true;
    };
    if (view->behavior && view->behavior->executable()) {
        bool allLoaded = true;
        std::set<std::string> loaded;
        for (const auto& node : view->behavior->graph.nodes) {
            if (node.kind != HkxBehaviorNodeKind::Clip || !loaded.insert(node.assetPath).second) continue;
            if (!loadClip(node.assetPath, node.assetPath, node.playbackMode == 1)) {
                allLoaded = false;
                view->diagnostics.push_back({AnimationDiagnosticSeverity::Warning, "graph.clip_missing", node.assetPath});
            }
        }
        if (allLoaded) {
            view->supportedBehaviorGraph = true;
            if (view->executionMode == AnimationExecutionMode::Havok) return view;
        } else view->behavior.reset();
    }
    // Explicit recovery catalog, not a replacement claimed as graph parity.
    // Sex-specific paths precede shared paths; missing variants remain visible
    // in diagnostics and never synthesize an arbitrary gesture.
    struct Candidate { const char* state; const char* file; bool loop; };
    static constexpr Candidate candidates[]{
        {"walk_back", "mt_walkbackward.hkx", true},
        {"walk_left", "mt_walkleft.hkx", true}, {"walk_right", "mt_walkright.hkx", true},
        {"run_forward", "mt_runforward.hkx", true}, {"run_back", "mt_runbackward.hkx", true},
        {"run_left", "mt_runleft.hkx", true}, {"run_right", "mt_runright.hkx", true},
        {"sprint", "mt_sprintforward.hkx", true},
        {"turn_left", "npc_turnleft90.hkx", false}, {"turn_right", "npc_turnright90.hkx", false},
        {"jump", "mt_jump.hkx", false}, {"fall", "mt_jumpfall.hkx", true},
        {"landing", "mt_jumpland.hkx", false},
        {"sneak_idle", "sneakmtidle.hkx", true}, {"sneak", "sneakwalk_forward.hkx", true},
        {"swim_idle", "swimidle.hkx", true}, {"swim", "swimforward.hkx", true},
        {"swim_back", "swimbackward.hkx", true}, {"swim_left", "swimleft.hkx", true},
        {"swim_right", "swimright.hkx", true},
        {"sneak_back", "sneakwalk_bckward.hkx", true}, {"sneak_left", "sneakwalk_left.hkx", true},
        {"sneak_right", "sneakwalk_right.hkx", true},
        {"attack", "h2h_attackright.hkx", false}, {"combat_idle", "h2h_idle.hkx", true},
        {"equip", "h2h_equip.hkx", false}, {"unequip", "h2h_unequip.hkx", false},
        {"stagger", "h2h_staggerbacksmall.hkx", false},
        {"death", "deathanimationa.hkx", false},
        {"talk", "dialogueneutralexpressivea.hkx", false}
    };
    for (const auto& candidate : candidates) {
        const std::string base = characterRoot + "animations\\";
        if (!loadClip(base + sex + "\\" + candidate.file, candidate.state, candidate.loop) &&
            !loadClip(base + candidate.file, candidate.state, candidate.loop))
            view->diagnostics.push_back({AnimationDiagnosticSeverity::Info, "clip.unavailable", candidate.state});
    }

    for (const std::string style : {"1hm", "2hm", "2hw"}) {
        for (const auto& [state, suffix, loop] : std::vector<std::tuple<std::string, std::string, bool>>{
                {"attack", "attackright", false}, {"combat_idle", "idle", true},
                {"equip", "equip", false}, {"unequip", "unequip", false}, {"block", "blockidle", true},
                {"walk_forward", "walkforward", true}, {"walk_back", "walkbackward", true},
                {"walk_left", "walkleft", true}, {"walk_right", "walkright", true},
                {"run_forward", "runforward", true}, {"run_back", "runbackward", true},
                {"run_left", "runleft", true}, {"run_right", "runright", true}}) {
            const auto file = style + "_" + suffix + ".hkx";
            if (!loadClip(characterRoot + "animations\\" + file, state + "_" + style, loop))
                view->diagnostics.push_back({AnimationDiagnosticSeverity::Info, "clip.unavailable", file});
        }
    }
    loadSkyrimNativeAnimationPacks(assets, *view, &sourceSkeleton);
    return view;
}

bool SkyrimAnimationAssetCache::resolve(
    const FalloutAssetSource& source, const std::string& virtualPath, Bytes& outBytes,
    FalloutAssetSource::ResolvedAsset& outResolution, std::string& outError) {
    if (!source.resolveAssetWithProvider(virtualPath, outResolution, outError)) return false;
    const std::string key = outResolution.canonicalVirtualPath + "@" + outResolution.contentFingerprint;
    const auto found = m_assets.find(key);
    if (found != m_assets.end()) { outBytes = found->second; return true; }
    auto immutable = std::make_shared<const std::vector<std::uint8_t>>(outResolution.bytes);
    outBytes = immutable;
    m_assets.emplace(key, std::move(immutable));
    return true;
}

}  // namespace odai::importer::fnv
