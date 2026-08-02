#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace odai::importer {

struct ImportedSceneVertex {
    float position[3] = {};
    float normal[3] = {};
    float uv[2] = {};
};

struct ImportedSceneMeshPart {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t textureIndex = 0;
    bool alphaTest = false;
};

struct ImportedSceneMesh {
    std::string name;
    std::vector<ImportedSceneVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<ImportedSceneMeshPart> parts;
};

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
};

// ---------------------------------------------------------------------------
// Packed material flags — the canonical layout for ImportedScenePackedVertex::flags
// and ImportedSceneMeshPart-derived draw flags. Mirrored in
// src/render/shaders/imported_static.frag.slang; change both together.
//
//   bit 0      alpha test
//   bit 1      reserved — terrain slope blend (docs/stylized_low_poly.md §1)
//   bit 2      PBR material present: bits 8..23 carry metallic/roughness
//   bits 3-7   free
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
    std::vector<ImportedSceneCellRef> unresolvedRefs;
    std::vector<ImportedScenePackedVertex> packedVertices;
    std::vector<std::uint32_t> packedIndices;
    std::vector<ImportedScenePackedDraw> packedDraws;
    std::vector<ImportedScenePageRange> pageRanges;
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
