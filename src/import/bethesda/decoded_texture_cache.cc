#include "import/bethesda/decoded_texture_cache.h"

#include <cctype>
#include <algorithm>

#include "import/dds.h"

namespace odai::importer::bethesda {

namespace {

// Same key the renderer dedups GPU textures by: the normalized path,
// lowercased. The ESM, the NIF texture sets and the BSA index disagree on
// casing and separators for the same file.
std::string cacheKey(const std::string& texturePath, std::uint32_t maxSize) {
    std::string key = normalizeTexturePath(texturePath);
    for (char& c : key) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    // The mip ceiling is part of the identity: the same file decoded to a
    // different maximum size is a different texture, and returning the wrong
    // one would silently change how a surface looks.
    return key + "|" + std::to_string(maxSize);
}

bool decodeTexture(
    const FalloutAssetSource& assets, const std::string& texturePath,
    std::uint32_t maxSize, ImportedSceneTexture& outTexture,
    std::string& error, bool linearData) {
    FalloutAssetSource::ResolvedAsset asset;
    return assets.resolveTextureWithProvider(texturePath, asset, error) &&
        loadTextureFromMemory(asset.bytes.data(), asset.bytes.size(),
            asset.canonicalVirtualPath, outTexture, maxSize, error, linearData);
}

}  // namespace

const ImportedSceneTexture* DecodedTextureCache::get(
    const FalloutAssetSource& assets,
    const std::string& texturePath,
    std::uint32_t maxSize,
    ImportedSceneTexture& outOwned, std::string* outError, bool linearData) {
    if (texturePath.empty()) {
        return nullptr;
    }
    const std::string key = std::to_string(assets.cacheIdentity()) + "|" +
        cacheKey(texturePath, maxSize) + (linearData ? "|linear" : "|color");
    std::string error;

    Entry* entry = nullptr;
    bool overBudget = false;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto existing = m_entries.find(key);
        if (existing != m_entries.end()) {
            ++m_stats.hits;
            entry = existing->second.get();
        } else if (m_stats.residentBytes >= m_byteBudget) {
            // Past the budget: decode for this caller but do not retain, so the
            // cache stays bounded without evicting entries other threads may
            // still be holding pointers into.
            ++m_stats.overBudgetDecodes;
            overBudget = true;
        } else {
            ++m_stats.misses;
            entry = m_entries.emplace(key, std::make_unique<Entry>()).first->second.get();
        }
    }

    if (overBudget) {
        if (!decodeTexture(assets, texturePath, maxSize, outOwned, error, linearData)) {
            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_stats.failures;
            if (outError) *outError = error;
            return nullptr;
        }
        return &outOwned;
    }

    // Outside the map mutex: two threads missing on DIFFERENT textures decode
    // concurrently, two missing on the SAME texture decode once.
    std::call_once(entry->once, [&]() {
        entry->valid = decodeTexture(assets, texturePath, maxSize, entry->texture, entry->error, linearData);
        std::lock_guard<std::mutex> lock(m_mutex);
        if (entry->valid) {
            // Concurrent misses must not grow the retained budget without bound.
            if (entry->texture.rgba8.size() <= m_byteBudget - std::min(m_byteBudget, m_stats.residentBytes)) {
                m_stats.residentBytes += entry->texture.rgba8.size();
                ++m_stats.residentCount;
            } else {
                entry->valid = false;
                entry->error = "decoded cache budget";
                entry->texture = {};
                ++m_stats.overBudgetDecodes;
            }
        } else {
            ++m_stats.failures;
        }
    });

    if (entry->error == "decoded cache budget") {
        // call_once entry stays as a nonresident marker; callers own decodes.
        if (decodeTexture(assets, texturePath, maxSize, outOwned, error, linearData)) return &outOwned;
        if (outError) *outError = error;
        return nullptr;
    }
    if (outError) *outError = entry->error;
    return entry->valid ? &entry->texture : nullptr;
}

DecodedTextureCacheStats DecodedTextureCache::stats() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_mutex));
    return m_stats;
}

}  // namespace odai::importer::bethesda
