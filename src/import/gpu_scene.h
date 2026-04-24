#pragma once

#include "import/imported_scene.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace odai::importer {

struct GpuSceneBounds {
    float min[3] = {};
    float max[3] = {};
    float center[3] = {};
    float extent[3] = {};
};

struct GpuSceneMeshAssetPart {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t textureIndex = 0;
    std::uint32_t flags = 0;
};

struct GpuSceneMeshAsset {
    std::string name;
    std::uint32_t firstVertex = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t firstPart = 0;
    std::uint32_t partCount = 0;
    GpuSceneBounds localBounds{};
};

struct GpuSceneTransformTable {
    std::vector<std::int32_t> parentIndices;
    std::vector<float> localTranslationX;
    std::vector<float> localTranslationY;
    std::vector<float> localTranslationZ;
    std::vector<float> localRotationX;
    std::vector<float> localRotationY;
    std::vector<float> localRotationZ;
    std::vector<float> localRotationW;
    std::vector<float> localScaleX;
    std::vector<float> localScaleY;
    std::vector<float> localScaleZ;
    std::vector<std::array<float, 16>> worldMatrices;
    std::vector<std::uint32_t> flags;
};

struct GpuSceneObjectTable {
    std::string nameBlob;
    std::vector<std::uint32_t> nameOffsets;
    std::vector<std::uint32_t> rootTransformIndices;
    std::vector<std::uint32_t> firstComponentIndices;
    std::vector<std::uint32_t> componentCounts;
    std::vector<std::uint32_t> pageIndices;
    std::vector<std::uint32_t> flags;
    std::vector<GpuSceneBounds> worldBounds;
    std::vector<std::array<float, 16>> appliedTransforms;
};

struct GpuSceneMeshComponentTable {
    std::vector<std::uint32_t> objectIndices;
    std::vector<std::uint32_t> transformIndices;
    std::vector<std::uint32_t> meshAssetIndices;
    std::vector<std::uint32_t> instanceIndices;
    std::vector<std::uint32_t> flags;
    std::vector<GpuSceneBounds> localBounds;
    std::vector<GpuSceneBounds> worldBounds;
};

struct GpuSceneInstanceTable {
    std::vector<std::uint32_t> objectIndices;
    std::vector<std::uint32_t> componentIndices;
    std::vector<std::uint32_t> meshAssetIndices;
    std::vector<std::uint32_t> transformIndices;
    std::vector<std::uint32_t> pageIndices;
    std::vector<std::uint32_t> flags;
    std::vector<GpuSceneBounds> worldBounds;
};

struct GpuScenePageRecord {
    std::int32_t gridX = 0;
    std::int32_t gridY = 0;
    std::int32_t gridZ = 0;
    std::uint32_t firstObject = 0;
    std::uint32_t objectCount = 0;
    std::uint32_t firstInstance = 0;
    std::uint32_t instanceCount = 0;
    GpuSceneBounds bounds{};
};

struct GpuSceneRenderCache {
    std::vector<ImportedSceneTexture> textures;
    std::vector<ImportedSceneWaterPatch> waterPatches;
    std::vector<ImportedScenePackedVertex> packedVertices;
    std::vector<std::uint32_t> packedIndices;
    std::vector<ImportedScenePackedDraw> packedDraws;
    std::vector<std::uint32_t> drawInstanceIndices;
    std::uint32_t terrainDrawCount = 0;
};

struct GpuSceneAsset {
    std::string sourceTag;
    std::vector<ImportedSceneTexture> textures;
    std::vector<ImportedSceneWaterPatch> waterPatches;
    std::vector<ImportedSceneVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<GpuSceneMeshAssetPart> meshParts;
    std::vector<GpuSceneMeshAsset> meshAssets;
    GpuSceneTransformTable transforms;
    GpuSceneObjectTable objects;
    GpuSceneMeshComponentTable meshComponents;
    GpuSceneInstanceTable instances;
    std::vector<GpuScenePageRecord> pages;
    GpuSceneRenderCache renderCache;
    GpuSceneBounds sceneBounds{};
};

struct GpuSceneRuntime {
    GpuSceneTransformTable transforms;
    std::vector<std::uint32_t> dirtyTransformIndices;
};

struct GpuSceneObjectView {
    std::string_view name;
    const float* appliedTransform = nullptr;
    std::uint32_t rootTransformIndex = 0;
    std::uint32_t firstComponentIndex = 0;
    std::uint32_t componentCount = 0;
    std::uint32_t pageIndex = 0;
};

bool buildGpuSceneAssetFromImportedScene(const ImportedScene& scene, GpuSceneAsset& outAsset);
void buildGpuSceneRenderCache(GpuSceneAsset& scene);
GpuSceneRuntime createGpuSceneRuntime(const GpuSceneAsset& scene);
void rebuildGpuSceneWorldTransforms(GpuSceneRuntime& runtime);
GpuSceneObjectView gpuSceneObjectView(const GpuSceneAsset& scene, std::uint32_t objectIndex);

} // namespace odai::importer
