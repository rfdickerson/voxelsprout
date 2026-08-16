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
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "import/fnv/asset_source.h"
#include "import/fnv/plugin_load_order.h"
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
    // Base formID -> its record type (STAT, MSTT, ACTI, ...), for attributing
    // geometry back to the kind of record that placed it.
    std::unordered_map<std::uint32_t, std::string> staticRecordTypes;
    // LTEX formID -> diffuse texture path, already resolved through TXST.
    std::unordered_map<std::uint32_t, std::string> landTexturePaths;
    // REGN formID -> the name to show the player (RDMP). Only discoverable
    // regions are in here: a region with no map name is deliberately absent
    // rather than present-with-an-empty-string, so a lookup miss means "do not
    // announce this" without the caller having to re-check.
    std::unordered_map<std::uint32_t, std::string> regionNamesByFormId;
    // Worldspace editor ID -> formID, so a streamer can select one by name.
    std::unordered_map<std::string, std::uint32_t> worldspaceFormIdsByEditorId;
    // Every worldspace by formID, with its DNAM default land/water heights
    // already INHERITED down the WNAM parent chain by
    // resolveWorldspaceInheritance(). hasDefaultHeights stays false only when
    // neither the worldspace nor any ancestor declares any, which is every
    // Oblivion one. See FalloutWorldspaceRecord::hasDefaultHeights for why a
    // cell with no LAND needs this, and parentWorldspaceFormId for what a
    // Skyrim city looks like without it.
    std::unordered_map<std::uint32_t, FalloutWorldspaceRecord> worldspaceDefaultsByFormId;

    [[nodiscard]] const FalloutWorldspaceRecord* findWorldspace(std::uint32_t formId) const {
        const auto found = worldspaceDefaultsByFormId.find(formId);
        return found == worldspaceDefaultsByFormId.end() ? nullptr : &found->second;
    }
    // MORROWIND REFERENCES NAME THEIR BASE BY STRING, so this is how a placed
    // reference reaches its model. Lowercased id -> the synthetic formID the
    // scan handed that record. Empty for every later generation, where a
    // reference carries the formID directly.
    std::unordered_map<std::string, std::uint32_t> baseFormIdsByEditorId;
    // LIGH formID -> its light parameters. A LIGH also appears in the maps
    // above when it carries a MODL (29 of 501 do), because the lamp mesh and
    // the light it casts are both wanted.
    std::unordered_map<std::uint32_t, FalloutLightRecord> lightsByFormId;
};

// One pass over the plugin that materializes no cell contents: it rejects every
// worldspace group and every cell's children, so LAND records are never
// decompressed. This is what makes it affordable at startup.
bool buildFalloutWorldTables(
    const std::filesystem::path& esmPath, FalloutWorldTables& outTables, std::string& outError);

// As above, across a whole load order. Every plugin's records are rewritten from
// its own local mod-index space into the order's global one, and a later plugin
// REPLACES an earlier one's record with the same formID -- which is the whole
// mechanism an override patch works by. A plugin that fails to read is skipped
// with a warning rather than failing the build: losing a patch's records is a
// degraded scene, losing the base game's is no scene.
bool buildFalloutWorldTables(
    const FalloutLoadOrder& order, FalloutWorldTables& outTables, std::string& outError);

// True for meshes that only make sense alpha-blended or additive: dust, glow
// billboards, light beams, sand. The imported static path draws opaque, so
// these render as solid pale sheets standing in the landscape.
//
// A path heuristic rather than a flag test because the flag is not reliable
// here: NiAlphaProperty's blend bit catches FXDustWhirlWind01 but not
// SandDust02, which signals its transparency some other way. Skipping is a
// stopgap for whatever the blended pass does not pick up.
bool isEffectOnlyModelPath(std::string_view modelPath);

// True for the game's own sky objects (Skyrim places sky\clouddistant*.nif and
// friends as ordinary references in Tamriel's persistent cell). See the
// definition: drawn as scenery they are a white plane over the landscape.
bool isSkyOnlyModelPath(std::string_view modelPath);

// Appends this cell's water surface to `outScene`, and reports whether it did.
//
// Free rather than a CellSceneBuilder member because the offline cooker builds
// its ImportedScene directly and does not go through the builder -- the terrain
// append is already duplicated between the two, and duplicating this as well is
// how the cooked and streamed worlds drift apart.
//
// `worldspace` supplies the implied water height for a cell that states none --
// resolved up the WNAM parent chain by resolveWorldspaceInheritance(). May be
// null, which means "no default": a cell with neither an XCLW nor terrain then
// contributes nothing, rather than a quad at height 0.
bool appendCellWaterPatch(
    odai::importer::ImportedScene& outScene,
    const FalloutCellRecord& cell,
    const FalloutWorldspaceRecord* worldspace = nullptr);

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
    // See NifModel's fields of the same names (import/fnv/nif_scene.h).
    std::uint32_t nodeParseFailures = 0;
    std::uint32_t unhandledNodeTypes = 0;
    std::size_t terrainPartsEmitted = 0;
    // Where a cell's build time actually goes. Texture decode and NIF parse are
    // the two candidates for caching, and they are very differently sized.
    float nifParseMs = 0.0f;
    float textureDecodeMs = 0.0f;
    std::size_t texturesDecoded = 0;
    std::size_t nifsParsed = 0;
    std::size_t extremeUvShapes = 0;
    std::size_t effectMeshesSkipped = 0;
    // REFR header flag 0x800: quest objects hidden until a script enables
    // them. Skipped, because an unstarted game does not show them -- and some
    // are worldspace-sized (Skyrim's MG07 blizzard barrier).
    std::size_t disabledReferencesSkipped = 0;
    // LIGH references turned into ImportedScene lights, and those rejected for
    // having a zero radius (exactly one LIGH in FalloutNV.esm does).
    std::size_t lightsPlaced = 0;
    std::size_t lightsSkippedZeroRadius = 0;
    // Cells that contributed a water surface. Zero across most of the Mojave
    // and nonzero along any coast, lake or river.
    std::size_t waterPatchesEmitted = 0;

    // References that were placed in the cell and then drew nothing.
    //
    // Every one of these paths used to `continue` with no counter at all, so
    // "this town has holes in it" had no way to be asked as a question. They are
    // split by CAUSE because the causes have opposite fixes: a formID that
    // resolves to no record is a load-order or remap problem, a record with no
    // MODL is usually correct (a trigger, a marker, an activator with no mesh),
    // and a MODL naming a file that is not there is a missing-asset problem.
    //
    // Counted per REFERENCE, not per base record, so a hundred placements of one
    // missing rock read as a hundred holes -- which is what a hole in a town
    // actually looks like.
    //
    // Deliberate skips (effect meshes, editor markers) are NOT counted here;
    // they have their own counters above and folding them in would bury the
    // signal in known-good noise.
    std::size_t referencesDroppedBaseNotFound = 0;
    std::size_t referencesDroppedBaseHasNoModel = 0;
    std::size_t referencesDroppedMeshUnresolved = 0;
    std::size_t referencesDroppedMeshUnreadable = 0;
    // Base record type -> how many references it dropped, e.g. {"ACTI": 4}.
    // "<base record not found>" for a formID with no record at all.
    std::unordered_map<std::string, std::size_t> droppedReferencesByBaseType;
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
    // Emits one ImportedScene light for a REFR whose base is a LIGH. Called
    // from addCellStatics, and additive to the lamp mesh rather than instead
    // of it.
    void addCellLight(const FalloutPlacedReference& ref, const FalloutLightRecord& light);

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
public:
    // Why a base record stopped producing geometry, so a REPEAT reference to it
    // can be attributed to the same cause instead of only the first one being
    // explained. kIntentional covers the deliberate skips, which are counted
    // elsewhere and must not be counted again here. Public because
    // failedStatics() below hands the map to diagnostics.
    enum class StaticDropReason : std::uint8_t {
        kIntentional,
        kBaseNotFound,
        kBaseHasNoModel,
        kMeshUnresolved,
        kMeshUnreadable,
    };

private:
    void noteDroppedReference(std::uint32_t baseFormId, StaticDropReason reason);
    std::unordered_map<std::uint32_t, StaticDropReason> m_failedStatics;

public:
    // Which base records produced no geometry, and why -- for diagnostics that
    // want to name the culprits rather than only count them.
    [[nodiscard]] const std::unordered_map<std::uint32_t, StaticDropReason>& failedStatics() const {
        return m_failedStatics;
    }

private:
    std::uint32_t m_fallbackLandTexture = 0xFFFFFFFFu;
    std::size_t m_textureBudget = 1000u;
    std::uint32_t m_maxTextureSize = 512u;
    bool m_warnedTextureBudget = false;
};

}  // namespace odai::importer::fnv
