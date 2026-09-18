#include "import/fnv/character_asset_manifest.h"
#include "anim/skeleton_io.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>

namespace odai::importer::fnv {
std::string characterAssetBundleRoot(const std::string& path) {
    // Bundles can nest (ambient/chicken, character/_1stperson), and animated
    // objects also use the same directory convention outside actors/.
    std::size_t start = 0;
    while (start < path.size()) {
        const auto end = path.find('\\', start);
        if (end == std::string::npos) break;
        const auto component = path.substr(start, end - start);
        if (component.starts_with("behaviors") || component.starts_with("characters") ||
            component == "animations" || component == "character assets" || component == "characterassets")
            return path.substr(0, start);
        start = end + 1;
    }
    return {};
}

std::string characterAssetPath(std::string path, const std::string& owner, bool behavior) {
    path = normalizeModelPath(std::move(path));
    std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c) { return std::tolower(c); });
    if (path.empty() || path.front() == '\\' || path.find(':') != std::string::npos) return {};
    const bool absolute = path.starts_with("meshes\\") || path.starts_with("textures\\") || path.starts_with("odai\\");
    if (!absolute) {
        const auto canonicalOwner = owner.empty() ? std::string{} : characterAssetPath(owner);
        auto root = characterAssetBundleRoot(canonicalOwner);
        if (root.empty() || path.starts_with("actors\\")) root = "meshes\\";
        if (behavior && path.find('\\') == std::string::npos) {
            auto directory = std::string("behaviors\\");
            if (canonicalOwner.substr(root.size()).starts_with("behaviors")) {
                const auto end = canonicalOwner.find('\\', root.size());
                if (end != std::string::npos) directory = canonicalOwner.substr(root.size(), end + 1 - root.size());
            }
            path = directory + path;
        }
        path = root + path;
    }
    std::vector<std::string> components;
    for (std::size_t begin = 0; begin < path.size();) {
        const auto end = path.find('\\', begin);
        auto part = path.substr(begin, end == std::string::npos ? end : end - begin);
        if (part.empty()) return {};
        if (part == "..") {
            if (components.size() <= 1) return {}; // never escape virtual Data root
            components.pop_back();
        } else if (part != ".") components.push_back(std::move(part));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    std::string result;
    for (const auto& component : components) { if (!result.empty()) result += '\\'; result += component; }
    return result;
}

bool loadCharacterSourceClip(const FalloutAssetSource& assets, const std::string& path,
    const odai::anim::Skeleton& target, const odai::anim::HkxDecodedSkeleton* source,
    bool retarget, odai::anim::AnimationClip& clip, odai::anim::HkxDecodedClipMetadata& metadata,
    FalloutAssetSource::ResolvedAsset& resolution, std::string& error) {
    using namespace odai::anim;
    clip = {}; metadata = {}; resolution = {}; error.clear();
    const auto canonical = characterAssetPath(path);
    if (canonical.empty()) { error = "invalid virtual clip path"; return false; }
    if (!assets.resolveAssetWithProvider(canonical, resolution, error)) return false;
    AnimationClip decoded;
    if (canonical.ends_with(".json")) {
        auto result = loadAnimationClipFromJson(std::string(resolution.bytes.begin(), resolution.bytes.end()), target, canonical);
        if (!result.ok()) { error = "invalid native clip"; return false; }
        decoded = std::move(result.clip); decoded.name = canonical;
    } else {
        if (!source) { error = "animation skeleton unavailable"; return false; }
        if (retarget && !source->referenceSkeleton.bones.empty()) {
            ClipRigBinding binding;
            AnimationClip authored;
            if (!bindHumanoidClipRig(source->referenceSkeleton, target, TranslationScalePolicy::LimbLength, binding, error) ||
                !decodeHkxAnimationClip(resolution.bytes, source->referenceSkeleton, canonical, authored, metadata, error, source) ||
                !retargetHumanoidClip(authored, binding, decoded, error)) return false;
        } else if (!decodeHkxAnimationClip(resolution.bytes, target, canonical, decoded, metadata, error, source)) return false;
        if (!metadata.boundTracks) { error = "no source tracks bind to target rig"; return false; }
    }
    if (!std::isfinite(decoded.duration) || decoded.duration <= 0) { error = "invalid clip duration"; return false; }
    clip = std::move(decoded);
    return true;
}

CharacterAssetManifest discoverCharacterAssets(const FalloutAssetSource& source,
    const std::map<std::string, std::set<std::string>>& seeds, std::size_t maxAssets, std::size_t maxDepth) {
    using namespace odai::anim;
    CharacterAssetManifest result;
    std::set<std::string> visiting, done;
    std::function<void(const std::string&, const std::string&, std::size_t)> visit;
    visit = [&](const std::string& input, const std::string& parent, std::size_t depth) {
        auto path = characterAssetPath(input);
        if (path.empty()) { result.diagnostics.push_back("invalid path: " + input); result.complete = false; return; }
        if (!result.assets.contains(path) && result.assets.size() >= maxAssets) {
            result.complete = false; return;
        }
        auto& entry = result.assets[path]; entry.path = path;
        if (entry.kind.empty()) entry.kind = path.ends_with(".nif") ? "nif" : path.ends_with(".tri") ? "facial_morph" :
            path.ends_with(".hkx") ? "hkx_unknown" : path.ends_with(".json") ? "native_clip" : "other";
        if (!parent.empty()) entry.references.insert(parent);
        if (visiting.contains(path)) { result.diagnostics.push_back("dependency cycle: " + parent + " -> " + path); return; }
        if (done.contains(path)) return;
        if (depth > maxDepth) { result.complete = false; entry.status = "resource_limit"; return; }
        visiting.insert(path);
        FalloutAssetSource::ResolvedAsset asset;
        if (!source.resolveAssetWithProvider(path, asset, entry.error)) entry.status = "missing_dependency";
        else {
            entry.provider = asset.providerId; entry.providerName = asset.providerName;
            entry.archive = asset.archiveName; entry.physicalPath = asset.physicalPath.generic_string(); entry.fingerprint = asset.contentFingerprint; entry.bytes = asset.bytes.size();
            entry.status = "resolved";
            auto add = [&](const std::string& raw, bool behavior = false) {
                if (raw.empty()) return;
                auto dependency = characterAssetPath(raw, path, behavior);
                if (dependency.empty()) { result.complete = false; result.diagnostics.push_back(path + ": invalid dependency " + raw); return; }
                entry.dependencies.push_back(dependency);
            };
            if (path.ends_with(".hkx")) {
                HkxPackfileSummary summary;
                if (!inspectHkxPackfile(asset.bytes, summary, entry.error)) entry.status = "decode_failed_unclassified";
                else {
                    entry.status = "inspected"; entry.kind = "hkx_other";
                    entry.unsupportedClasses = summary.unsupportedBehaviorClasses;
                    if (std::any_of(summary.objects.begin(), summary.objects.end(), [](const auto& object) { return object.className == "hkbCharacterStringData"; })) {
                        entry.kind = "character_definition";
                        HkxCharacterAssets character;
                        if (!decodeHkxCharacterAssets(asset.bytes, character, entry.error)) entry.status = "decode_failed_unclassified";
                        else {
                            entry.status = "decoded";
                            add(character.skeletonPath); add(character.behaviorPath, true);
                            for (const auto& clip : character.animationPaths) add(clip);
                        }
                    }
                    if (summary.containsBehaviorGraph) {
                        if (entry.kind != "character_definition") entry.kind = "behavior_graph";
                        HkxDecodedBehaviorGraph graph;
                        if (!decodeHkxBehaviorGraph(asset.bytes, graph, entry.error)) entry.status = "decode_failed_unclassified";
                        else {
                            entry.status = "decoded";
                            for (const auto& node : graph.nodes) {
                                if (node.kind == HkxBehaviorNodeKind::Clip) add(node.assetPath);
                                if (node.kind == HkxBehaviorNodeKind::BehaviorReference) add(node.assetPath, true);
                            }
                        }
                    } else if (summary.containsSkeleton) entry.kind = "animation_skeleton";
                    else if (summary.containsAnimation) entry.kind = "animation_clip";
                }
            } else if (path.starts_with("odai\\animations\\") && path.ends_with(".json")) {
                entry.kind = "native_pack";
                NativeAnimationProgram pack;
                if (!compileNativeAnimationPack(std::string_view(reinterpret_cast<const char*>(asset.bytes.data()), asset.bytes.size()),
                    asset.providerId, asset.layerPriority, pack, entry.error)) entry.status = "malformed_data";
                else {
                    entry.status = "decoded";
                    auto clip = [&](const std::string& value) {
                        // Bare names are catalog aliases, not missing virtual files.
                        if (value.find_first_of("/\\") != std::string::npos) add(value);
                    };
                    for (const auto& rule : pack.rules) {
                        for (const auto& v : rule.variants) clip(v.clip);
                        for (const auto& v : rule.blendSamples) clip(v.clip);
                        for (const auto& v : rule.layers) { clip(v.clip); clip(v.referenceClip); }
                        if (rule.graph) for (const auto& [id, node] : rule.graph->nodes) { clip(node.clip); clip(node.reference); }
                    }
                }
            }
            std::sort(entry.dependencies.begin(), entry.dependencies.end());
            entry.dependencies.erase(std::unique(entry.dependencies.begin(), entry.dependencies.end()), entry.dependencies.end());
            for (const auto& dependency : entry.dependencies) visit(dependency, path, depth + 1);
        }
        visiting.erase(path); done.insert(path);
    };
    for (const auto& [path, refs] : seeds) {
        visit(path, {}, 0);
        auto found = result.assets.find(characterAssetPath(path));
        if (found != result.assets.end()) found->second.references.insert(refs.begin(), refs.end());
    }
    if (!result.complete) result.diagnostics.push_back("manifest incomplete: invalid dependency or resource limit");
    return result;
}
}
