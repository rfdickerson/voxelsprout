#pragma once
#include "import/fnv/skyrim_animation_assets.h"
#include <map>
#include <set>

namespace odai::importer::fnv {
struct CharacterAssetEntry {
    std::string path, kind, status = "unassessed", provider, providerName, archive, physicalPath, fingerprint, error;
    std::set<std::string> references;
    std::vector<std::string> dependencies;
    std::vector<std::string> unsupportedClasses;
    std::size_t bytes = 0;
};
struct CharacterAssetManifest {
    std::map<std::string, CharacterAssetEntry> assets;
    std::vector<std::string> diagnostics;
    bool complete = true;
};
// Resolves a Data-relative reference against the owning actor bundle, not always
// the humanoid directory. Empty/escaping paths return an empty string.
std::string characterAssetBundleRoot(const std::string& path);
std::string characterAssetPath(std::string path, const std::string& owner = {}, bool behavior = false);
// Seeds may include unreferenced inventory entries. Expansion does not require
// graph execution admission. Limits bound distinct files and recursive edges.
CharacterAssetManifest discoverCharacterAssets(const FalloutAssetSource& source,
    const std::map<std::string, std::set<std::string>>& seeds,
    std::size_t maxAssets = 100000, std::size_t maxDepth = 64);
// Source decoding/binding only: never changes looping, root motion or markers.
bool loadCharacterSourceClip(const FalloutAssetSource& assets, const std::string& path,
    const odai::anim::Skeleton& target, const odai::anim::HkxDecodedSkeleton* source,
    bool retarget, odai::anim::AnimationClip& clip, odai::anim::HkxDecodedClipMetadata& metadata,
    FalloutAssetSource::ResolvedAsset& resolution, std::string& error);
}
