#include "import/fnv/decoded_texture_cache.h"

#include <cctype>

#include "import/dds.h"

namespace odai::importer::fnv {

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
    const FalloutAssetSource& assets,
    const std::string& texturePath,
    std::uint32_t maxSize,
    ImportedSceneTexture& outTexture) {
    std::vector<std::uint8_t> ddsBytes;
    std::string error;
    if (!assets.resolveTexture(texturePath, ddsBytes, error)) {
        return false;
    }
    if (!loadDdsFromMemory(ddsBytes.data(), ddsBytes.size(), outTexture)) {
        return false;
    }
    if (maxSize != 0u) {
        dropDdsMipLevels(outTexture, maxSize);
    }
    outTexture.sourcePath = texturePath;
    return true;
}

}  // namespace

const ImportedSceneTexture* DecodedTextureCache::get(
    const FalloutAssetSource& assets,
    const std::string& texturePath,
    std::uint32_t maxSize,
    ImportedSceneTexture& outOwned) {
    if (texturePath.empty()) {
        return nullptr;
    }
    const std::string key = cacheKey(texturePath, maxSize);

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
        if (!decodeTexture(assets, texturePath, maxSize, outOwned)) {
            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_stats.failures;
            return nullptr;
        }
        return &outOwned;
    }

    // Outside the map mutex: two threads missing on DIFFERENT textures decode
    // concurrently, two missing on the SAME texture decode once.
    std::call_once(entry->once, [&]() {
        entry->valid = decodeTexture(assets, texturePath, maxSize, entry->texture);
        std::lock_guard<std::mutex> lock(m_mutex);
        if (entry->valid) {
            m_stats.residentBytes += static_cast<std::uint64_t>(entry->texture.rgba8.size());
            ++m_stats.residentCount;
        } else {
            ++m_stats.failures;
        }
    });

    return entry->valid ? &entry->texture : nullptr;
}

DecodedTextureCacheStats DecodedTextureCache::stats() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_mutex));
    return m_stats;
}

}  // namespace odai::importer::fnv
