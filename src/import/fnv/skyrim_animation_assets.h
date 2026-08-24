#pragma once

#include "anim/hkx_packfile.h"
#include "import/fnv/asset_source.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::importer::fnv {

struct SkyrimAnimationAssetReport {
    bool coherent = false;
    bool strictCompatible = false;
    odai::anim::HkxGeneratorIdentity generator = odai::anim::HkxGeneratorIdentity::Unknown;
    std::string generatorProvider;
    std::vector<FalloutAssetSource::ResolvedAsset> roots;
    std::vector<std::string> missingAssets;
    std::vector<std::string> unsupportedClasses;
    std::vector<std::string> diagnostics;
};

// Checks only immutable virtual-Data output. It never launches FNIS/Nemesis.
bool inspectSkyrimAnimationBundle(
    const FalloutAssetSource& assets, SkyrimAnimationAssetReport& out,
    bool strict, std::string& outError);

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

}  // namespace odai::importer::fnv
