#include "import/fnv/skyrim_animation_assets.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <set>

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
    std::set<std::string> generatorProviders;
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
        identities.insert(identity);
        generatorProviders.insert(asset.providerId);
        out.unsupportedClasses.insert(out.unsupportedClasses.end(),
            summary.unsupportedBehaviorClasses.begin(), summary.unsupportedBehaviorClasses.end());
        out.roots.push_back(std::move(asset));
    }
    std::sort(out.unsupportedClasses.begin(), out.unsupportedClasses.end());
    out.unsupportedClasses.erase(
        std::unique(out.unsupportedClasses.begin(), out.unsupportedClasses.end()),
        out.unsupportedClasses.end());
    out.coherent = out.roots.size() == roots.size() && generatorProviders.size() == 1u &&
        identities.size() == 1u;
    if (!out.roots.empty()) out.generatorProvider = out.roots.front().providerName;
    if (identities.size() == 1u) out.generator = *identities.begin();
    if (!out.coherent && out.roots.size() == roots.size()) {
        out.diagnostics.push_back("generated root HKX files resolve from inconsistent providers/generators");
    }
    out.strictCompatible = out.coherent && out.unsupportedClasses.empty();
    if (strict && !out.strictCompatible) {
        outError = !out.coherent ? "incoherent Skyrim generated animation bundle" :
            "unsupported gameplay behavior classes in Skyrim animation bundle";
        return false;
    }
    return true;
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
