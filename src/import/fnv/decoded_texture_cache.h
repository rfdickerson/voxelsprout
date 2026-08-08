#pragma once

// Decoded textures, shared across every cell being built.
//
// Measured motivation: building one exterior cell costs ~270 ms, of which
// ~170 ms is DDS decode for its ~37 textures. Adjacent cells reuse most of that
// set -- the same ground, rock and building diffuse maps -- but a
// CellSceneBuilder owns its own caches, so four workers building four cells
// decode the same textures four times. This is the one piece deliberately
// shared between them.
//
// THREAD SAFE. Map lookups are serialized on a mutex; the decode itself happens
// under a per-entry std::once_flag OUTSIDE that mutex, so concurrent misses on
// different textures decode in parallel while concurrent misses on the SAME
// texture decode exactly once and the losers wait.
//
// Entries are never evicted and pointers into the cache stay valid for its
// lifetime. That is safe because the working set is bounded by the number of
// distinct textures the game ships, not by how far the player walks -- but it
// is bounded, not free, so there is a byte budget past which new textures are
// decoded per-call instead of retained.

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "import/fnv/asset_source.h"
#include "import/imported_scene.h"

namespace odai::importer::fnv {

struct DecodedTextureCacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t failures = 0;
    // Textures decoded but not retained because the budget was already spent.
    std::uint64_t overBudgetDecodes = 0;
    std::uint64_t residentBytes = 0;
    std::size_t residentCount = 0;
};

class DecodedTextureCache {
public:
    explicit DecodedTextureCache(std::uint64_t byteBudget = 512ull * 1024ull * 1024ull)
        : m_byteBudget(byteBudget) {}

    DecodedTextureCache(const DecodedTextureCache&) = delete;
    DecodedTextureCache& operator=(const DecodedTextureCache&) = delete;

    // Returns a decoded texture for `texturePath`, or nullptr if it cannot be
    // resolved or decoded. `outOwned` receives the texture when the cache
    // declined to retain it (over budget), in which case the returned pointer
    // aims at outOwned.
    //
    // `maxSize` is the mip-drop ceiling; a request that differs from the cached
    // entry's ceiling is decoded fresh rather than returning the wrong size.
    const ImportedSceneTexture* get(
        const FalloutAssetSource& assets,
        const std::string& texturePath,
        std::uint32_t maxSize,
        ImportedSceneTexture& outOwned);

    [[nodiscard]] DecodedTextureCacheStats stats() const;

private:
    struct Entry {
        std::once_flag once;
        ImportedSceneTexture texture;
        bool valid = false;
    };

    std::mutex m_mutex;
    std::unordered_map<std::string, std::unique_ptr<Entry>> m_entries;
    std::uint64_t m_byteBudget = 0;
    DecodedTextureCacheStats m_stats;
};

}  // namespace odai::importer::fnv
