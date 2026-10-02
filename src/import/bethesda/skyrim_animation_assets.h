#pragma once

#include "anim/hkx_packfile.h"
#include "anim/skyrim_animation.h"
#include "import/bethesda/asset_source.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::importer::bethesda {

struct SkyrimAnimationAssetReport {
    bool coherent = false;
    bool strictCompatible = false;
    odai::anim::HkxGeneratorIdentity generator = odai::anim::HkxGeneratorIdentity::Unknown;
    std::string generatorProvider;
    std::vector<FalloutAssetSource::ResolvedAsset> roots;
    std::vector<odai::anim::HkxDecodedBehaviorGraph> behaviorGraphs;
    std::vector<std::string> missingAssets;
    std::vector<std::string> unsupportedClasses;
    std::vector<std::string> diagnostics;
};

// Checks only immutable virtual-Data output. It never launches FNIS/Nemesis.
bool inspectSkyrimAnimationBundle(
    const FalloutAssetSource& assets, SkyrimAnimationAssetReport& out,
    bool strict, std::string& outError);

// Resolves references through virtual Data, with bounded graph expansion. The
// returned program retains explicit admission gaps instead of claiming parity.
std::shared_ptr<const odai::anim::BehaviorProgram> loadSkyrimBehaviorProgram(
    const FalloutAssetSource& assets, const std::string& rootPath,
    std::string& fingerprint, std::string& error);

std::shared_ptr<const odai::anim::AnimationView> loadSkyrimNpcAnimationView(
    const FalloutAssetSource& assets, const odai::anim::Skeleton& skeleton,
    const std::vector<odai::math::Matrix4>& inverseBind, bool female,
    const odai::anim::AnimationClip& idle, const odai::anim::AnimationClip& walk,
    std::shared_ptr<const odai::anim::BehaviorProgram> behavior,
    const std::string& graphFingerprint);

// Also used by player views that already own their imported clip catalog.
void loadSkyrimNativeAnimationPacks(const FalloutAssetSource& assets,
    odai::anim::AnimationView& view, const odai::anim::HkxDecodedSkeleton* sourceSkeleton = nullptr);

class SkyrimAnimationAssetCache {
public:
    using Bytes = std::shared_ptr<const std::vector<std::uint8_t>>;
    bool resolve(const FalloutAssetSource& source, const std::string& virtualPath,
        Bytes& outBytes, FalloutAssetSource::ResolvedAsset& outResolution,
        std::string& outError);
    void clear() { m_assets.clear(); }
    [[nodiscard]] std::size_t size() const { return m_assets.size(); }

private:
    std::unordered_map<std::string, Bytes> m_assets;
};

}  // namespace odai::importer::bethesda
