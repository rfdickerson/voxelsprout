#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Material-index bit layout and the GPU table record. Kept in its own header so
// tests (which never link Vulkan) and the renderer can share one definition.
#include "import/imported_material.h"

namespace odai::importer {

struct ImportedSceneVertex {
    float position[3] = {};
    float normal[3] = {};
    float uv[2] = {};
    // Authored vertex tint, multiplied into the diffuse texture by the shader
    // when kImportedSceneMaterialFlagVertexColorTint is set on the vertex.
    // White is neutral, so geometry that never sets it shades unchanged --
    // which is also what scenes cooked before file version 19 decode to.
    //
    // The FNV importer fills this from LAND's VCLR: the per-post terrain tint
    // carrying baked ambient and the regional palette, which is what makes the
    // Mojave read sunbleached instead of like a generic lit texture.
    float color[3] = {1.0f, 1.0f, 1.0f};
    // Terrain layer blending. Up to kImportedSceneMaxTerrainLayers textures
    // blended over the one named by the mesh part, each with a per-vertex
    // weight -- this is Fallout's ATXT/VTXT, which is what draws roads, gravel
    // and the transitions between ground types. kImportedSceneNoTerrainLayer
    // means the slot is unused; a weight of 0 means the layer is absent here.
    //
    // Per-vertex rather than a splat texture because it is lossless: VTXT
    // defines opacity per landscape post and the cooked terrain mesh keeps one
    // vertex per post, so the resolutions already match exactly.
    std::uint32_t layerTextureIndex[4] = {
        0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    float layerWeight[4] = {};
};

// Four rather than three: at three, a 169-cell Mojave cook dropped 535 layer
// contributions on quadrants declaring more layers than a vertex could hold.
// Each extra slot costs 8 bytes per packed vertex (one index plus one weight
// byte), which is the whole reason this is a fixed budget and not a list.
inline constexpr int kImportedSceneMaxTerrainLayers = 4;
inline constexpr std::uint32_t kImportedSceneNoTerrainLayer = 0xffffffffu;

struct ImportedSceneMeshPart {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t textureIndex = 0;
    bool alphaTest = false;
    // NiAlphaProperty's blend bit, distinct from the test bit above: alphaTest
    // discards fragments in the opaque pass, alphaBlend moves the whole draw
    // into the blended pass. See kImportedSceneMaterialFlagAlphaBlend.
    bool alphaBlend = false;
    // NiStencilProperty DRAW_BOTH. See kImportedSceneMaterialFlagTwoSided.
    bool twoSided = false;
};

struct ImportedSceneMesh {
    std::string name;
    std::vector<ImportedSceneVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<ImportedSceneMeshPart> parts;
};

// A teleport door: stand near it, look at it, and it takes you to another
// cooked scene. Position is in this scene's space; arrival position/rotation
// are in the TARGET scene's space, straight from Fallout's XTEL.
//
// The target is named by cell EditorID rather than by file path, so a scene
// stays valid however the cooked .bin files are laid out on disk -- the loader
// applies whatever naming convention the cooker used (see
// importedSceneInteriorFileName).
struct ImportedSceneDoor {
    float position[3] = {};
    float arrivalPosition[3] = {};
    float arrivalYawDegrees = 0.0f;
    std::string targetCellEditorId;
};

// Where a cooked interior lives relative to its exterior scene. One convention,
// stated once, so the cooker that writes the files and the app that opens them
// cannot drift: "<exterior stem>_<CellEditorID>.bin", beside the exterior.
[[nodiscard]] std::string importedSceneInteriorFileName(
    const std::string& exteriorStem, const std::string& cellEditorId);

struct ImportedSceneInstance {
    std::uint32_t meshIndex = 0;
    float transform[16] = {};
    std::string sourceId;
    std::string modelPath;
};

// Pixel or block-compression format of a texture's data blob.
// RGBA8: 4 bytes per pixel, mip levels packed largest-first.
// BC*: 4×4-texel blocks; 8 bytes/block for BC1/BC4, 16 bytes/block for BC3/BC5/BC7.
enum class TextureFormat : std::uint8_t {
    RGBA8 = 0,  // 4 bytes per pixel
    BC1   = 1,  // DXT1 — 8 bytes per block (opaque or 1-bit alpha)
    BC3   = 2,  // DXT5 — 16 bytes per block (RGBA)
    BC4   = 3,  // ATI1 — 8 bytes per block (single channel)
    BC5   = 4,  // ATI2 — 16 bytes per block (dual channel)
    BC7   = 5,  // 16 bytes per block (high-quality RGBA)
    // Appended, NOT inserted in numeric order next to BC1/BC3. This value is
    // serialized as a uint8 and bounded by kImportedSceneMaxTextureFormat;
    // renumbering the existing entries would silently reinterpret every
    // already-cooked scene's textures as the wrong format.
    BC2   = 6,  // DXT3 — 16 bytes per block (RGBA, 4-bit explicit alpha)
};

struct ImportedSceneTexture {
    std::string sourcePath;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevelCount = 1;
    TextureFormat format = TextureFormat::RGBA8;
    std::vector<std::uint8_t> rgba8; // pixel or block data, mip chain packed largest-first
};

struct ImportedScenePackedVertex {
    float position[3] = {};
    float normal[3] = {};
    float color[3] = {};
    float uv[2] = {};
    std::uint32_t textureIndex = 0xffffffffu;
    std::uint32_t flags = 0u;
    // Terrain layer blend, mirroring ImportedSceneVertex's. Indices stay a full
    // uint32 each rather than being bit-packed because the renderer remaps them
    // to bindless slots, and those run past what 10 bits would hold
    // (kBindlessTargetTextureCapacity is 1024 on top of the static entries).
    // Weights are quantized to 8 bits, which is well past what landscape alpha
    // needs and keeps this to one extra word.
    std::uint32_t layerTextureIndex[4] = {
        0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    std::uint32_t layerWeights = 0u;  // 4x 8-bit, layer 0 in the low byte
};

// Quantization for ImportedScenePackedVertex::layerWeights. Byte n holds layer
// n's weight over [0,1] -- four layers exactly fills the word; mirrored in
// imported_static.frag.slang.
inline constexpr std::uint32_t packImportedSceneTerrainLayerWeights(const float weights[4]) {
    std::uint32_t packed = 0u;
    for (int layer = 0; layer < kImportedSceneMaxTerrainLayers; ++layer) {
        const float clamped = weights[layer] < 0.0f ? 0.0f : (weights[layer] > 1.0f ? 1.0f : weights[layer]);
        const std::uint32_t quantized = static_cast<std::uint32_t>((clamped * 255.0f) + 0.5f);
        packed |= (quantized & 0xffu) << (layer * 8);
    }
    return packed;
}

// ---------------------------------------------------------------------------
// Packed material flags — the canonical layout for ImportedScenePackedVertex::flags
// and ImportedSceneMeshPart-derived draw flags. Mirrored in
// src/render/shaders/imported_static.frag.slang; change both together.
//
//   bit 0      alpha test
//   bit 1      reserved — terrain slope blend (docs/stylized_low_poly.md §1)
//   bit 2      PBR material present: bits 8..23 carry metallic/roughness
//   bit 3      modulate the diffuse texture by the vertex colour
//   bit 4      terrain layer blend: layerTextureIndex/layerWeights are live
//   bits 5-7   free
//   bits 8-15  roughness, 8-bit quantized over [0,1]
//   bits 16-23 metallic, 8-bit quantized over [0,1]
//   bits 24-31 free
//
// packedVertices is written to disk as a raw struct blit with no per-field
// version gate, so a zero `flags` must always keep meaning "legacy default".
// Bit 2 is the opt-in that makes that work: scenes cooked before materials
// existed decode as a fully rough dielectric and shade exactly as before.
inline constexpr std::uint32_t kImportedSceneMaterialFlagAlphaTest = 1u << 0;
inline constexpr std::uint32_t kImportedSceneMaterialFlagTerrainSlopeBlend = 1u << 1;
inline constexpr std::uint32_t kImportedSceneMaterialFlagPbr = 1u << 2;
// Opt-in for the same reason bit 2 is: untextured geometry has always used the
// vertex colour AS its albedo, and textured geometry has always ignored it. This
// bit adds a third case -- textured, and tinted by a colour that means something
// -- without changing how anything already cooked shades. Only set it where the
// colour is authored data, never where it is a synthesized fallback.
inline constexpr std::uint32_t kImportedSceneMaterialFlagVertexColorTint = 1u << 3;
// Opt-in for the same reason: without it, every scene cooked before terrain
// layers existed would have the shader reading layer slots that were never
// written. Set only where the cooker actually filled them.
inline constexpr std::uint32_t kImportedSceneMaterialFlagTerrainLayers = 1u << 4;

// Surface is alpha-BLENDED, not merely alpha-tested. Fallout marks glass, dust
// sheets, god rays and vulture billboards this way (NiAlphaProperty bit 0); the
// renderer pulls these draws out of the opaque pass and replays them afterwards
// through a blend-enabled pipeline with depth writes off. Without it they show
// up as solid pale slabs standing in the landscape.
inline constexpr std::uint32_t kImportedSceneMaterialFlagAlphaBlend = 1u << 5;

// Surface is authored two-sided (Fallout's NiStencilProperty DRAW_BOTH). The
// renderer draws these with back-face culling off. Thin alpha geometry --
// window glass, foliage cards, awnings -- is the overwhelming majority of it,
// which is why it rides alongside the blend flag rather than in a general
// material system.
inline constexpr std::uint32_t kImportedSceneMaterialFlagTwoSided = 1u << 6;

inline constexpr int kImportedSceneMaterialRoughnessShift = 8;
inline constexpr int kImportedSceneMaterialMetallicShift = 16;
inline constexpr std::uint32_t kImportedSceneMaterialChannelMask = 0xffu;

// Metallic-roughness surface parameters. The defaults are the legacy response:
// a fully rough dielectric, which the shader treats as "no PBR" and shades with
// the pre-existing diffuse chain.
struct ImportedSceneSurfaceMaterial {
    float metallic = 0.0f;
    float roughness = 1.0f;
};

inline bool importedSceneMaterialIsDefault(const ImportedSceneSurfaceMaterial& material) {
    return material.metallic <= 0.0f && material.roughness >= 1.0f;
}

// Quantizes to the bit layout above. A default material packs to 0 — no opt-in
// bit, no material bits — so authoring sites that never set a material leave
// their geometry shading bit-for-bit as it did before PBR existed.
inline std::uint32_t packImportedSceneMaterialFlags(const ImportedSceneSurfaceMaterial& material) {
    if (importedSceneMaterialIsDefault(material)) {
        return 0u;
    }
    const auto quantize = [](float value) -> std::uint32_t {
        const float clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        return static_cast<std::uint32_t>((clamped * 255.0f) + 0.5f) & kImportedSceneMaterialChannelMask;
    };
    return kImportedSceneMaterialFlagPbr |
           (quantize(material.roughness) << kImportedSceneMaterialRoughnessShift) |
           (quantize(material.metallic) << kImportedSceneMaterialMetallicShift);
}

// Library-material variant: stamps a material-table index into bits 24-31 and
// forces the PBR opt-in bit whenever that index is nonzero, so the shader's
// "does this surface use PBR" test stays the single unchanged condition it
// already is. The quantized metallic/roughness are still written into bits
// 8-23 as a fallback, so a scene whose material table failed to load shades
// approximately right instead of going flat.
//
// Index 0 means "no library material" and this reduces exactly to the overload
// above — in particular a default material with index 0 still packs to 0u,
// which is what keeps every pre-existing cooked scene bit-identical.
inline std::uint32_t packImportedSceneMaterialFlags(const ImportedSceneSurfaceMaterial& material,
                                                    std::uint32_t materialIndex) {
    const std::uint32_t indexBits = (materialIndex & kImportedSceneMaterialIndexMask)
                                    << kImportedSceneMaterialIndexShift;
    if (indexBits == 0u) {
        return packImportedSceneMaterialFlags(material);
    }
    const auto quantize = [](float value) -> std::uint32_t {
        const float clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        return static_cast<std::uint32_t>((clamped * 255.0f) + 0.5f) & kImportedSceneMaterialChannelMask;
    };
    return kImportedSceneMaterialFlagPbr | indexBits |
           (quantize(material.roughness) << kImportedSceneMaterialRoughnessShift) |
           (quantize(material.metallic) << kImportedSceneMaterialMetallicShift);
}

// One named entry in a scene's material library. Names are carried in the
// cooked .bin so nothing needs the authoring JSON at runtime, and so a live
// editor can match by name across a hot reload rather than by position — a
// reorder in the source file is then harmless.
//
// Entry 0 is a reserved sentinel with an empty name; see imported_material.h.
struct ImportedSceneMaterial {
    std::string name;
    float baseColorTint[3] = {1.0f, 1.0f, 1.0f};  // multiplies sampled/vertex albedo
    float metallic = 0.0f;
    float roughness = 1.0f;
    float emissive[3] = {0.0f, 0.0f, 0.0f};
    float emissiveStrength = 0.0f;
};

// Inverse of the above. Flags without the PBR bit decode to the legacy default
// regardless of what the material bit ranges happen to hold.
inline ImportedSceneSurfaceMaterial unpackImportedSceneMaterialFlags(std::uint32_t flags) {
    if ((flags & kImportedSceneMaterialFlagPbr) == 0u) {
        return ImportedSceneSurfaceMaterial{};
    }
    ImportedSceneSurfaceMaterial material;
    material.roughness =
        static_cast<float>((flags >> kImportedSceneMaterialRoughnessShift) & kImportedSceneMaterialChannelMask) /
        255.0f;
    material.metallic =
        static_cast<float>((flags >> kImportedSceneMaterialMetallicShift) & kImportedSceneMaterialChannelMask) /
        255.0f;
    return material;
}

struct ImportedScenePackedDraw {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
};

// Optional spatial grouping of packed draws for per-chunk frustum culling.
// When non-empty, the renderer treats each entry as a cullable page covering the
// contiguous draw range [firstDraw, firstDraw + drawCount). Empty => no culling
// (the whole scene draws every frame, legacy behavior).
struct ImportedScenePageRange {
    std::uint32_t firstDraw = 0;
    std::uint32_t drawCount = 0;
    std::uint32_t terrainDrawCount = 0;
    float boundsMin[3] = {};
    float boundsMax[3] = {};
};

struct ImportedSceneCellRef {
    std::string refId;
    std::string modelPath;
    float position[3] = {};
    float rotationRadians[3] = {};
    float scale = 1.0f;
};

struct ImportedSceneLandscapeCell {
    int gridX = 0;
    int gridY = 0;
    std::vector<float> heights;
    std::vector<std::uint16_t> textureIndices;
};

struct ImportedSceneWaterPatch {
    float originX = 0.0f;
    float originZ = 0.0f;
    float sizeX = 0.0f;
    float sizeZ = 0.0f;
    float waterLevel = 0.0f;
};

struct ImportedSceneLight {
    std::string sourceId;
    float position[3] = {};
    float color[3] = {1.0f, 1.0f, 1.0f};
    float radius = 0.0f;
    float intensity = 1.0f;
    std::uint32_t flags = 0u;
};

struct ImportedScene {
    std::string sourceTag;
    std::vector<ImportedSceneTexture> textures;
    std::vector<ImportedSceneMesh> meshes;
    std::vector<ImportedSceneInstance> instances;
    std::vector<ImportedSceneLandscapeCell> landscapeCells;
    std::vector<ImportedSceneWaterPatch> waterPatches;
    std::vector<ImportedSceneLight> lights;
    std::vector<ImportedSceneDoor> doors;
    std::vector<ImportedSceneCellRef> unresolvedRefs;
    std::vector<ImportedScenePackedVertex> packedVertices;
    std::vector<std::uint32_t> packedIndices;
    std::vector<ImportedScenePackedDraw> packedDraws;
    std::vector<ImportedScenePageRange> pageRanges;
    // Named material library, indexed by vertex flag bits 24-31. Entry 0 is a
    // reserved sentinel so the flag index and the vector index are the same
    // number; empty means "this scene predates materials" and every surface
    // takes the legacy per-vertex path. Never exceeds
    // kImportedSceneMaterialTableCapacity.
    std::vector<ImportedSceneMaterial> materials;
    std::uint32_t sourceTextureCount = 0;
    std::uint32_t sourceFileVersion = 0;
    std::uint32_t sourceMeshCount = 0;
    std::uint32_t sourceInstanceCount = 0;
    std::uint32_t sourceLandscapeCellCount = 0;
    std::uint32_t sourceWaterPatchCount = 0;
    std::uint32_t sourceLightCount = 0;
    std::uint32_t sourceUnresolvedRefCount = 0;
    float boundsMin[3] = {};
    float boundsMax[3] = {};

    // Fog-of-war visibility map for the strategy map. R8 data, fogMapW×fogMapH
    // texels (one per hex tile). Values: 0=hidden, 100=explored, 255=visible,
    // blurred across 2 passes so bilinear sampling gives smooth fog edges.
    // fogMapInvExtentX/Z are the world-space UV scale factors (1/(extent in X/Z).
    // Empty when fog of war is disabled.
    std::vector<std::uint8_t> fogMap;
    std::uint32_t fogMapW = 0;
    std::uint32_t fogMapH = 0;
    float fogMapInvExtentX = 0.0f;
    float fogMapInvExtentZ = 0.0f;
};

// True when the scene's sourceTag marks it as an interior cell (no exterior
// terrain/landscape draws). Cookers for different source games tag interiors
// with their own "<game>_interior" sourceTag; this is the single place that
// enumerates the recognized tags so terrain-draw classification stays
// consistent across imported_scene.cc, the renderer upload path, and app-side
// scene inspection.
bool importedSceneSourceTagIsInterior(std::string_view sourceTag);

bool saveImportedScene(const ImportedScene& scene, const std::filesystem::path& outputPath);
bool loadImportedScene(const std::filesystem::path& inputPath, ImportedScene& outScene);
bool loadImportedSceneRuntime(const std::filesystem::path& inputPath, ImportedScene& outScene);
const std::string& getImportedSceneLastError();
void buildImportedScenePackedRenderData(ImportedScene& scene);

// One Morrowind exterior cell (8192 units) — the natural culling granularity
// for cooked exterior scenes.
inline constexpr float kImportedSceneDefaultPageSize = 8192.0f;

// Rebuilds pageRanges by partitioning packedDraws into XZ tiles of pageSize
// world units. Reorders packedDraws (and rebuilds packedIndices to match) so
// every page covers a contiguous draw range, keeping terrain draws in the
// leading [0, terrainDrawCount) slots the renderer expects. Terrain and static
// draws land in separate pages so the terrain-first invariant survives the
// reorder. Both loaders call this automatically when a file carries no pages.
void buildImportedScenePageRanges(
    ImportedScene& scene,
    float pageSize = kImportedSceneDefaultPageSize);

bool exportImportedSceneTerrainObj(const ImportedScene& scene, const std::filesystem::path& outputObjPath);

}  // namespace odai::importer
