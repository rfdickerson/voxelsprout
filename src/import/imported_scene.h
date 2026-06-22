#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
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

bool saveImportedScene(const ImportedScene& scene, const std::filesystem::path& outputPath);
bool loadImportedScene(const std::filesystem::path& inputPath, ImportedScene& outScene);
bool loadImportedSceneRuntime(const std::filesystem::path& inputPath, ImportedScene& outScene);
const std::string& getImportedSceneLastError();
void buildImportedScenePackedRenderData(ImportedScene& scene);

bool exportImportedSceneTerrainObj(const ImportedScene& scene, const std::filesystem::path& outputObjPath);

}  // namespace odai::importer
