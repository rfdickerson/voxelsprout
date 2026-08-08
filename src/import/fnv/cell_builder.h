#pragma once

// Builds one exterior cell's geometry -- terrain plus placed statics -- into an
// ImportedScene.
//
// Extracted from the cooker so the runtime streamer can produce the same
// geometry directly from FalloutNV.esm and the BSAs, instead of loading a .bin
// that the cooker produced earlier. Both now go through this one path, so the
// two cannot drift: the cooker is a batch driver over many cells, the streamer
// a per-cell driver over one, and the geometry, texture resolution and
// coordinate conventions are shared.
//
// THREADING: a CellSceneBuilder is not thread safe and owns mutable caches
// (textures, per-static meshes). Give each worker its own, or drive one from a
// single thread. The FalloutAssetSource underneath IS safe to share.

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "import/fnv/asset_source.h"
#include "import/fnv/decoded_texture_cache.h"
#include "import/fnv/fallout_records.h"
#include "import/imported_scene.h"

namespace odai::importer::fnv {

// Plugin-wide lookups the per-cell build needs, gathered once. A cell's REFR
// names a STAT by formID and a LAND quadrant names an LTEX by formID; neither
// record carries the path, so these have to come from a pass over the plugin
// that is NOT per-cell.
struct FalloutWorldTables {
    // STAT formID -> MODL path, relative to Data\Meshes.
    std::unordered_map<std::uint32_t, std::string> staticModelPaths;
    // STAT formID -> editor ID, used only to name meshes readably.
    std::unordered_map<std::uint32_t, std::string> staticEditorIds;
    // LTEX formID -> diffuse texture path, already resolved through TXST.
    std::unordered_map<std::uint32_t, std::string> landTexturePaths;
    // Worldspace editor ID -> formID, so a streamer can select one by name.
    std::unordered_map<std::string, std::uint32_t> worldspaceFormIdsByEditorId;
};

// One pass over the plugin that materializes no cell contents: it rejects every
// worldspace group and every cell's children, so LAND records are never
// decompressed. This is what makes it affordable at startup.
bool buildFalloutWorldTables(
    const std::filesystem::path& esmPath, FalloutWorldTables& outTables, std::string& outError);

struct CellBuildStats {
    std::size_t placedInstances = 0;
    std::size_t totalShapes = 0;
    std::size_t untexturedShapes = 0;
    std::size_t shapesWithNoTexturePath = 0;
    std::size_t shadowDecalShapesSkipped = 0;
    std::size_t editorMarkerModelsSkipped = 0;
    std::size_t untexturedShapesGivenModelTexture = 0;
    std::size_t droppedTerrainLayers = 0;
    std::uint32_t skippedGeometryShapes = 0;
    std::size_t terrainPartsEmitted = 0;
    // Where a cell's build time actually goes. Texture decode and NIF parse are
    // the two candidates for caching, and they are very differently sized.
    float nifParseMs = 0.0f;
    float textureDecodeMs = 0.0f;
    std::size_t texturesDecoded = 0;
    std::size_t nifsParsed = 0;
    std::size_t extremeUvShapes = 0;
    bool textureBudgetExceeded = false;

    // Diagnostic name sets the cooker reports. Kept here rather than dropped in
    // the extraction: "half the rock is grey" being a list of model paths rather
    // than a guess is the reason they exist.
    std::unordered_set<std::string> extremeUvModelPaths;
    std::unordered_set<std::string> untexturedModelPaths;
    std::unordered_set<std::string> unresolvedTexturePaths;
};

class CellSceneBuilder {
public:
    // Neither reference is owned; both must outlive the builder.
    // `textureCache` is optional but strongly recommended when several builders
    // run concurrently: it is the only state they share, and without it each
    // decodes the same textures independently (~170 ms of a ~270 ms cell build).
    CellSceneBuilder(
        const FalloutAssetSource& assets,
        const FalloutWorldTables& tables,
        DecodedTextureCache* textureCache = nullptr);

    // Terrain draws must all precede static draws in the finished scene, which
    // is what the renderer's terrain/static draw split relies on. Call
    // addCellTerrain for every cell first, then addCellStatics for every cell.
    void addCellTerrain(const FalloutCellRecord& cell);
    void addCellStatics(const FalloutCellRecord& cell);

    // Convenience for the single-cell (streaming) case.
    void addCell(const FalloutCellRecord& cell) {
        addCellTerrain(cell);
        addCellStatics(cell);
    }

    // Finalizes packed render data and page ranges, and hands over the scene.
    // The builder is left empty and must not be reused.
    void finish(ImportedScene& outScene);

    [[nodiscard]] const CellBuildStats& stats() const { return m_stats; }
    [[nodiscard]] const ImportedScene& scene() const { return m_scene; }
    [[nodiscard]] ImportedScene& scene() { return m_scene; }

    // Resolves a texture path to a scene texture index, decoding and caching on
    // first use. Public because the LOD cooker needs the same behaviour.
    std::uint32_t resolveTextureIndex(const std::string& texturePath);

    // Most-used BTXT base texture across `cells`, as a scene texture index, for
    // feeding back into setFallbackLandTexture(). Resolves (and so caches) the
    // texture as a side effect.
    std::uint32_t dominantLandTexture(const std::vector<const FalloutCellRecord*>& cells);

    // Texture a LAND quadrant with no BTXT falls back to. A quadrant without a
    // BTXT does NOT mean "untextured" -- it means the worldspace default, which
    // this importer does not parse. The closest honest stand-in is the most-used
    // land texture in the area being built; treating it as untextured left a
    // large fraction of terrain vertices shading from vertex colour alone.
    //
    // The caller picks it because the right answer differs: the cooker uses the
    // dominant texture across the whole region it is cooking, a streamer only
    // has one cell to go on.
    void setFallbackLandTexture(std::uint32_t sceneTextureIndex) {
        m_fallbackLandTexture = sceneTextureIndex;
    }

    void setTextureBudget(std::size_t budget) { m_textureBudget = budget; }
    void setMaxTextureSize(std::uint32_t maxSize) { m_maxTextureSize = maxSize; }

private:
    std::uint32_t resolveLandTexture(std::uint32_t landTextureFormId, bool exact);

    const FalloutAssetSource& m_assets;
    const FalloutWorldTables& m_tables;
    DecodedTextureCache* m_textureCache = nullptr;
    ImportedScene m_scene;
    CellBuildStats m_stats{};

    // Index of the single merged terrain mesh in m_scene.meshes, or npos until
    // a cell with LAND is added.
    std::size_t m_terrainMeshIndex = static_cast<std::size_t>(-1);
    std::unordered_map<std::string, std::uint32_t> m_textureIndexByPath;
    std::unordered_set<std::string> m_failedTexturePaths;
    std::unordered_map<std::uint32_t, std::uint32_t> m_meshIndexByStaticFormId;
    std::unordered_set<std::uint32_t> m_failedStatics;
    std::uint32_t m_fallbackLandTexture = 0xFFFFFFFFu;
    std::size_t m_textureBudget = 1000u;
    std::uint32_t m_maxTextureSize = 512u;
    bool m_warnedTextureBudget = false;
};

}  // namespace odai::importer::fnv
