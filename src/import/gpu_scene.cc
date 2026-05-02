#include "import/gpu_scene.h"

#include "math/math.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace odai::importer {

namespace {

constexpr std::uint32_t kGpuSceneComponentFlagTerrain = 1u << 0;
constexpr std::uint32_t kGpuSceneComponentFlagStatic = 1u << 1;
constexpr std::uint32_t kGpuSceneComponentFlagAlphaTest = 1u << 2;
constexpr std::uint32_t kImportedSceneMaterialFlagAlphaTest = 1u;
constexpr std::uint32_t kImportedSceneMaterialFlagSiltStrider = 1u << 1u;
constexpr std::uint32_t kImportedSceneMaterialFlagSiltStriderMaskShift = 8u;
constexpr std::uint32_t kImportedSceneMaterialFlagSiltStriderSide = 1u << 16u;
constexpr float kGpuScenePageSize = 4096.0f;

struct Quaternion {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct PackedRenderColor {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

PackedRenderColor packedRenderColorFromHash(std::string_view key) {
    std::uint32_t hash = 2166136261u;
    for (const char ch : key) {
        hash ^= static_cast<std::uint8_t>(ch);
        hash *= 16777619u;
    }
    PackedRenderColor color{};
    color.r = 0.30f + (static_cast<float>((hash >> 0) & 0xffu) / 255.0f) * 0.55f;
    color.g = 0.28f + (static_cast<float>((hash >> 8) & 0xffu) / 255.0f) * 0.52f;
    color.b = 0.25f + (static_cast<float>((hash >> 16) & 0xffu) / 255.0f) * 0.50f;
    return color;
}

std::string lowerCopy(std::string_view value) {
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return result;
}

std::uint32_t siltStriderAnimationFlags(const ImportedSceneVertex& vertex) {
    const float lateral = std::abs(vertex.position[0]);
    const float lowMask = std::clamp((125.0f - vertex.position[1]) / 320.0f, 0.0f, 1.0f);
    const float outerMask = std::clamp((lateral - 18.0f) / 85.0f, 0.0f, 1.0f);
    const float influence = std::clamp(lowMask * outerMask, 0.0f, 1.0f);
    const std::uint32_t packedInfluence = static_cast<std::uint32_t>((influence * 255.0f) + 0.5f);
    const std::uint32_t side =
        vertex.position[0] >= 0.0f ? kImportedSceneMaterialFlagSiltStriderSide : 0u;
    return kImportedSceneMaterialFlagSiltStrider |
        (packedInfluence << kImportedSceneMaterialFlagSiltStriderMaskShift) |
        side;
}

PackedRenderColor packedTerrainColor(float height) {
    PackedRenderColor color{};
    const float normalized = std::clamp((height + 256.0f) / 1024.0f, 0.0f, 1.0f);
    color.r = 0.22f + (normalized * 0.32f);
    color.g = 0.24f + (normalized * 0.34f);
    color.b = 0.18f + (normalized * 0.14f);
    return color;
}

bool textureUsesAlphaCutout(const ImportedSceneTexture& texture) {
    const std::size_t basePixelCount = static_cast<std::size_t>(texture.width) * texture.height;
    const std::size_t baseByteCount = basePixelCount * 4u;
    if (texture.width == 0u || texture.height == 0u || texture.rgba8.size() < baseByteCount) {
        return false;
    }

    bool sawTransparent = false;
    bool sawVisible = false;
    for (std::size_t pixelIndex = 0; pixelIndex < basePixelCount; ++pixelIndex) {
        const std::uint8_t alpha = texture.rgba8[(pixelIndex * 4u) + 3u];
        sawTransparent = sawTransparent || alpha < 250u;
        sawVisible = sawVisible || alpha > 8u;
        if (sawTransparent && sawVisible) {
            return true;
        }
    }
    return false;
}

std::vector<bool> buildTextureAlphaCutoutMask(const std::vector<ImportedSceneTexture>& textures) {
    std::vector<bool> mask(textures.size(), false);
    for (std::size_t textureIndex = 0; textureIndex < textures.size(); ++textureIndex) {
        mask[textureIndex] = textureUsesAlphaCutout(textures[textureIndex]);
    }
    return mask;
}

bool textureIndexUsesAlphaCutout(const std::vector<bool>& mask, std::uint32_t textureIndex) {
    return textureIndex < mask.size() && mask[textureIndex];
}

GpuSceneBounds makeEmptyBounds() {
    GpuSceneBounds bounds{};
    bounds.min[0] = bounds.min[1] = bounds.min[2] = std::numeric_limits<float>::max();
    bounds.max[0] = bounds.max[1] = bounds.max[2] = std::numeric_limits<float>::lowest();
    return bounds;
}

void finalizeBounds(GpuSceneBounds& bounds) {
    for (int axis = 0; axis < 3; ++axis) {
        bounds.center[axis] = (bounds.min[axis] + bounds.max[axis]) * 0.5f;
        bounds.extent[axis] = std::max(0.0f, (bounds.max[axis] - bounds.min[axis]) * 0.5f);
    }
}

void expandBounds(GpuSceneBounds& bounds, float x, float y, float z) {
    bounds.min[0] = std::min(bounds.min[0], x);
    bounds.min[1] = std::min(bounds.min[1], y);
    bounds.min[2] = std::min(bounds.min[2], z);
    bounds.max[0] = std::max(bounds.max[0], x);
    bounds.max[1] = std::max(bounds.max[1], y);
    bounds.max[2] = std::max(bounds.max[2], z);
}

GpuSceneBounds computeMeshBounds(
    const std::vector<ImportedSceneVertex>& vertices,
    std::uint32_t firstVertex,
    std::uint32_t vertexCount
) {
    GpuSceneBounds bounds = makeEmptyBounds();
    for (std::uint32_t i = 0; i < vertexCount; ++i) {
        const ImportedSceneVertex& vertex = vertices[firstVertex + i];
        expandBounds(bounds, vertex.position[0], vertex.position[1], vertex.position[2]);
    }
    finalizeBounds(bounds);
    return bounds;
}

std::array<float, 3> transformPoint(const std::array<float, 16>& transform, const float point[3]) {
    return {
        (transform[0] * point[0]) + (transform[1] * point[1]) + (transform[2] * point[2]) + transform[3],
        (transform[4] * point[0]) + (transform[5] * point[1]) + (transform[6] * point[2]) + transform[7],
        (transform[8] * point[0]) + (transform[9] * point[1]) + (transform[10] * point[2]) + transform[11]
    };
}

std::array<float, 3> transformDirection(const std::array<float, 16>& transform, const float direction[3]) {
    return {
        (transform[0] * direction[0]) + (transform[1] * direction[1]) + (transform[2] * direction[2]),
        (transform[4] * direction[0]) + (transform[5] * direction[1]) + (transform[6] * direction[2]),
        (transform[8] * direction[0]) + (transform[9] * direction[1]) + (transform[10] * direction[2])
    };
}

std::array<float, 3> normalizeVector(std::array<float, 3> value) {
    const float length = std::sqrt(
        (value[0] * value[0]) +
        (value[1] * value[1]) +
        (value[2] * value[2]));
    if (length > 1e-6f) {
        value[0] /= length;
        value[1] /= length;
        value[2] /= length;
    }
    return value;
}

GpuSceneBounds transformBounds(const GpuSceneBounds& localBounds, const std::array<float, 16>& transform) {
    GpuSceneBounds bounds = makeEmptyBounds();
    const float minX = localBounds.min[0];
    const float minY = localBounds.min[1];
    const float minZ = localBounds.min[2];
    const float maxX = localBounds.max[0];
    const float maxY = localBounds.max[1];
    const float maxZ = localBounds.max[2];
    const std::array<std::array<float, 3>, 8> corners{{
        {minX, minY, minZ},
        {maxX, minY, minZ},
        {minX, maxY, minZ},
        {maxX, maxY, minZ},
        {minX, minY, maxZ},
        {maxX, minY, maxZ},
        {minX, maxY, maxZ},
        {maxX, maxY, maxZ},
    }};
    for (const std::array<float, 3>& corner : corners) {
        const std::array<float, 3> point = transformPoint(transform, corner.data());
        expandBounds(bounds, point[0], point[1], point[2]);
    }
    finalizeBounds(bounds);
    return bounds;
}

Quaternion quaternionFromMatrix(const std::array<float, 16>& matrix) {
    const float trace = matrix[0] + matrix[5] + matrix[10];
    Quaternion q{};
    if (trace > 0.0f) {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (matrix[9] - matrix[6]) / s;
        q.y = (matrix[2] - matrix[8]) / s;
        q.z = (matrix[4] - matrix[1]) / s;
    } else if (matrix[0] > matrix[5] && matrix[0] > matrix[10]) {
        const float s = std::sqrt(1.0f + matrix[0] - matrix[5] - matrix[10]) * 2.0f;
        q.w = (matrix[9] - matrix[6]) / s;
        q.x = 0.25f * s;
        q.y = (matrix[1] + matrix[4]) / s;
        q.z = (matrix[2] + matrix[8]) / s;
    } else if (matrix[5] > matrix[10]) {
        const float s = std::sqrt(1.0f + matrix[5] - matrix[0] - matrix[10]) * 2.0f;
        q.w = (matrix[2] - matrix[8]) / s;
        q.x = (matrix[1] + matrix[4]) / s;
        q.y = 0.25f * s;
        q.z = (matrix[6] + matrix[9]) / s;
    } else {
        const float s = std::sqrt(1.0f + matrix[10] - matrix[0] - matrix[5]) * 2.0f;
        q.w = (matrix[4] - matrix[1]) / s;
        q.x = (matrix[2] + matrix[8]) / s;
        q.y = (matrix[6] + matrix[9]) / s;
        q.z = 0.25f * s;
    }
    return q;
}

void decomposeTransform(
    const std::array<float, 16>& matrix,
    float& outTx,
    float& outTy,
    float& outTz,
    Quaternion& outRotation,
    float& outSx,
    float& outSy,
    float& outSz
) {
    outTx = matrix[3];
    outTy = matrix[7];
    outTz = matrix[11];

    const odai::math::Vector3 row0{matrix[0], matrix[1], matrix[2]};
    const odai::math::Vector3 row1{matrix[4], matrix[5], matrix[6]};
    const odai::math::Vector3 row2{matrix[8], matrix[9], matrix[10]};
    outSx = std::max(odai::math::length(row0), 1e-6f);
    outSy = std::max(odai::math::length(row1), 1e-6f);
    outSz = std::max(odai::math::length(row2), 1e-6f);

    std::array<float, 16> rotationMatrix = matrix;
    rotationMatrix[0] /= outSx;
    rotationMatrix[1] /= outSx;
    rotationMatrix[2] /= outSx;
    rotationMatrix[4] /= outSy;
    rotationMatrix[5] /= outSy;
    rotationMatrix[6] /= outSy;
    rotationMatrix[8] /= outSz;
    rotationMatrix[9] /= outSz;
    rotationMatrix[10] /= outSz;
    rotationMatrix[3] = rotationMatrix[7] = rotationMatrix[11] = 0.0f;
    rotationMatrix[12] = rotationMatrix[13] = rotationMatrix[14] = 0.0f;
    rotationMatrix[15] = 1.0f;
    outRotation = quaternionFromMatrix(rotationMatrix);
}

std::array<float, 16> multiplyMatrices(const std::array<float, 16>& lhs, const std::array<float, 16>& rhs) {
    std::array<float, 16> out{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                sum += lhs[row * 4 + i] * rhs[i * 4 + col];
            }
            out[row * 4 + col] = sum;
        }
    }
    return out;
}

std::array<float, 16> composeTransform(
    float tx,
    float ty,
    float tz,
    float rx,
    float ry,
    float rz,
    float rw,
    float sx,
    float sy,
    float sz
) {
    const float xx = rx * rx;
    const float yy = ry * ry;
    const float zz = rz * rz;
    const float xy = rx * ry;
    const float xz = rx * rz;
    const float yz = ry * rz;
    const float wx = rw * rx;
    const float wy = rw * ry;
    const float wz = rw * rz;
    return {
        (1.0f - (2.0f * (yy + zz))) * sx, (2.0f * (xy - wz)) * sx, (2.0f * (xz + wy)) * sx, tx,
        (2.0f * (xy + wz)) * sy, (1.0f - (2.0f * (xx + zz))) * sy, (2.0f * (yz - wx)) * sy, ty,
        (2.0f * (xz - wy)) * sz, (2.0f * (yz + wx)) * sz, (1.0f - (2.0f * (xx + yy))) * sz, tz,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

std::uint32_t appendName(std::string& blob, const std::string& name) {
    const std::uint32_t offset = static_cast<std::uint32_t>(blob.size());
    blob.append(name);
    blob.push_back('\0');
    return offset;
}

std::int32_t pageCoord(float value) {
    return static_cast<std::int32_t>(std::floor(value / kGpuScenePageSize));
}

} // namespace

bool buildGpuSceneAssetFromImportedScene(const ImportedScene& scene, GpuSceneAsset& outAsset) {
    outAsset = {};
    outAsset.sourceTag = scene.sourceTag;
    outAsset.textures = scene.textures;
    outAsset.waterPatches = scene.waterPatches;
    outAsset.lights = scene.lights;
    outAsset.sceneBounds = makeEmptyBounds();
    const std::vector<bool> textureAlphaCutoutMask = buildTextureAlphaCutoutMask(scene.textures);

    std::uint32_t globalFirstVertex = 0u;
    std::uint32_t globalFirstIndex = 0u;
    std::uint32_t globalFirstPart = 0u;
    for (const ImportedSceneMesh& mesh : scene.meshes) {
        GpuSceneMeshAsset asset{};
        asset.name = mesh.name;
        asset.firstVertex = globalFirstVertex;
        asset.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
        asset.firstIndex = globalFirstIndex;
        asset.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
        asset.firstPart = globalFirstPart;
        asset.partCount = static_cast<std::uint32_t>(mesh.parts.size());
        asset.localBounds = computeMeshBounds(
            mesh.vertices,
            0u,
            static_cast<std::uint32_t>(mesh.vertices.size()));
        outAsset.meshAssets.push_back(asset);
        outAsset.vertices.insert(outAsset.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        for (const std::uint32_t index : mesh.indices) {
            outAsset.indices.push_back(index + globalFirstVertex);
        }
        for (const ImportedSceneMeshPart& part : mesh.parts) {
            GpuSceneMeshAssetPart dstPart{};
            dstPart.firstIndex = part.firstIndex + globalFirstIndex;
            dstPart.indexCount = part.indexCount;
            dstPart.textureIndex = part.textureIndex;
            dstPart.flags = (part.alphaTest || textureIndexUsesAlphaCutout(textureAlphaCutoutMask, part.textureIndex))
                ? kGpuSceneComponentFlagAlphaTest
                : 0u;
            outAsset.meshParts.push_back(dstPart);
        }
        globalFirstVertex += asset.vertexCount;
        globalFirstIndex += asset.indexCount;
        globalFirstPart += asset.partCount;
    }

    auto appendStaticObject = [&](std::string objectName,
                                  const std::array<float, 16>& worldTransform,
                                  std::uint32_t meshAssetIndex,
                                  std::uint32_t componentFlags) {
        const std::uint32_t transformIndex = static_cast<std::uint32_t>(outAsset.transforms.parentIndices.size());
        outAsset.transforms.parentIndices.push_back(-1);
        float tx = 0.0f;
        float ty = 0.0f;
        float tz = 0.0f;
        float sx = 1.0f;
        float sy = 1.0f;
        float sz = 1.0f;
        Quaternion rotation{};
        decomposeTransform(worldTransform, tx, ty, tz, rotation, sx, sy, sz);
        outAsset.transforms.localTranslationX.push_back(tx);
        outAsset.transforms.localTranslationY.push_back(ty);
        outAsset.transforms.localTranslationZ.push_back(tz);
        outAsset.transforms.localRotationX.push_back(rotation.x);
        outAsset.transforms.localRotationY.push_back(rotation.y);
        outAsset.transforms.localRotationZ.push_back(rotation.z);
        outAsset.transforms.localRotationW.push_back(rotation.w);
        outAsset.transforms.localScaleX.push_back(sx);
        outAsset.transforms.localScaleY.push_back(sy);
        outAsset.transforms.localScaleZ.push_back(sz);
        outAsset.transforms.worldMatrices.push_back(worldTransform);
        outAsset.transforms.flags.push_back(0u);

        const std::uint32_t objectIndex = static_cast<std::uint32_t>(outAsset.objects.rootTransformIndices.size());
        const std::uint32_t componentIndex = static_cast<std::uint32_t>(outAsset.meshComponents.objectIndices.size());
        const std::uint32_t instanceIndex = static_cast<std::uint32_t>(outAsset.instances.objectIndices.size());
        const GpuSceneBounds worldBounds = transformBounds(outAsset.meshAssets[meshAssetIndex].localBounds, worldTransform);

        outAsset.objects.nameOffsets.push_back(appendName(outAsset.objects.nameBlob, objectName));
        outAsset.objects.rootTransformIndices.push_back(transformIndex);
        outAsset.objects.firstComponentIndices.push_back(componentIndex);
        outAsset.objects.componentCounts.push_back(1u);
        outAsset.objects.pageIndices.push_back(0u);
        outAsset.objects.flags.push_back(componentFlags);
        outAsset.objects.worldBounds.push_back(worldBounds);
        outAsset.objects.appliedTransforms.push_back(worldTransform);

        outAsset.meshComponents.objectIndices.push_back(objectIndex);
        outAsset.meshComponents.transformIndices.push_back(transformIndex);
        outAsset.meshComponents.meshAssetIndices.push_back(meshAssetIndex);
        outAsset.meshComponents.instanceIndices.push_back(instanceIndex);
        outAsset.meshComponents.flags.push_back(componentFlags);
        outAsset.meshComponents.localBounds.push_back(outAsset.meshAssets[meshAssetIndex].localBounds);
        outAsset.meshComponents.worldBounds.push_back(worldBounds);

        outAsset.instances.objectIndices.push_back(objectIndex);
        outAsset.instances.componentIndices.push_back(componentIndex);
        outAsset.instances.meshAssetIndices.push_back(meshAssetIndex);
        outAsset.instances.transformIndices.push_back(transformIndex);
        outAsset.instances.pageIndices.push_back(0u);
        outAsset.instances.flags.push_back(componentFlags);
        outAsset.instances.worldBounds.push_back(worldBounds);

        for (int axis = 0; axis < 3; ++axis) {
            outAsset.sceneBounds.min[axis] = std::min(outAsset.sceneBounds.min[axis], worldBounds.min[axis]);
            outAsset.sceneBounds.max[axis] = std::max(outAsset.sceneBounds.max[axis], worldBounds.max[axis]);
        }
    };

    const bool hasTerrainMesh =
        !scene.meshes.empty() &&
        !outAsset.meshAssets.empty() &&
        scene.meshes.front().name == "terrain";
    if (hasTerrainMesh) {
        appendStaticObject("terrain", {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        }, 0u, kGpuSceneComponentFlagTerrain);
    }

    for (const ImportedSceneInstance& instance : scene.instances) {
        if (instance.meshIndex >= outAsset.meshAssets.size()) {
            continue;
        }
        std::array<float, 16> worldTransform{};
        std::copy(std::begin(instance.transform), std::end(instance.transform), worldTransform.begin());
        const std::string objectName = !instance.sourceId.empty() ? instance.sourceId : instance.modelPath;
        appendStaticObject(objectName, worldTransform, instance.meshIndex, kGpuSceneComponentFlagStatic);
    }

    finalizeBounds(outAsset.sceneBounds);

    using PageKey = std::tuple<std::int32_t, std::int32_t, std::int32_t>;
    std::map<PageKey, std::uint32_t> pageIndexByKey;
    std::vector<std::vector<std::uint32_t>> pageObjects;
    std::vector<std::vector<std::uint32_t>> pageInstances;
    for (std::uint32_t objectIndex = 0; objectIndex < outAsset.objects.worldBounds.size(); ++objectIndex) {
        const GpuSceneBounds& bounds = outAsset.objects.worldBounds[objectIndex];
        const PageKey key{
            pageCoord(bounds.center[0]),
            pageCoord(bounds.center[1]),
            pageCoord(bounds.center[2])
        };
        auto [it, inserted] = pageIndexByKey.emplace(key, static_cast<std::uint32_t>(outAsset.pages.size()));
        if (inserted) {
            GpuScenePageRecord page{};
            page.gridX = std::get<0>(key);
            page.gridY = std::get<1>(key);
            page.gridZ = std::get<2>(key);
            page.bounds = makeEmptyBounds();
            outAsset.pages.push_back(page);
            pageObjects.emplace_back();
            pageInstances.emplace_back();
        }
        const std::uint32_t pageIndex = it->second;
        outAsset.objects.pageIndices[objectIndex] = pageIndex;
        pageObjects[pageIndex].push_back(objectIndex);
        const std::uint32_t instanceIndex = outAsset.meshComponents.instanceIndices[outAsset.objects.firstComponentIndices[objectIndex]];
        outAsset.instances.pageIndices[instanceIndex] = pageIndex;
        pageInstances[pageIndex].push_back(instanceIndex);
        GpuScenePageRecord& page = outAsset.pages[pageIndex];
        for (int axis = 0; axis < 3; ++axis) {
            page.bounds.min[axis] = std::min(page.bounds.min[axis], bounds.min[axis]);
            page.bounds.max[axis] = std::max(page.bounds.max[axis], bounds.max[axis]);
        }
    }

    std::uint32_t runningFirstObject = 0u;
    std::uint32_t runningFirstInstance = 0u;
    for (std::uint32_t pageIndex = 0; pageIndex < outAsset.pages.size(); ++pageIndex) {
        GpuScenePageRecord& page = outAsset.pages[pageIndex];
        page.firstObject = runningFirstObject;
        page.objectCount = static_cast<std::uint32_t>(pageObjects[pageIndex].size());
        page.firstInstance = runningFirstInstance;
        page.instanceCount = static_cast<std::uint32_t>(pageInstances[pageIndex].size());
        runningFirstObject += page.objectCount;
        runningFirstInstance += page.instanceCount;
        finalizeBounds(page.bounds);
    }

    buildGpuSceneRenderCache(outAsset);
    return true;
}

void buildGpuSceneRenderCache(GpuSceneAsset& scene) {
    scene.renderCache = {};
    scene.renderCache.textures = scene.textures;
    scene.renderCache.waterPatches = scene.waterPatches;
    scene.renderCache.lights = scene.lights;
    scene.renderCache.packedVertices.reserve(scene.vertices.size() + (scene.instances.objectIndices.size() * 64u));
    scene.renderCache.packedIndices.reserve(scene.indices.size() + (scene.instances.objectIndices.size() * 128u));

    auto appendPageDrawRange = [&](std::uint32_t pageIndex, std::uint32_t drawIndex, bool terrainMesh) {
        if (!scene.renderCache.pageDrawRanges.empty()) {
            GpuScenePageDrawRange& lastRange = scene.renderCache.pageDrawRanges.back();
            if (lastRange.pageIndex == pageIndex &&
                lastRange.firstDraw + lastRange.drawCount == drawIndex) {
                ++lastRange.drawCount;
                if (terrainMesh) {
                    ++lastRange.terrainDrawCount;
                }
                return;
            }
        }
        GpuScenePageDrawRange range{};
        range.pageIndex = pageIndex;
        range.firstDraw = drawIndex;
        range.drawCount = 1u;
        range.terrainDrawCount = terrainMesh ? 1u : 0u;
        scene.renderCache.pageDrawRanges.push_back(range);
    };

    auto appendMeshInstance = [&](const GpuSceneMeshAsset& mesh,
                                  const std::array<float, 16>& transform,
                                  std::uint32_t instanceIndex,
                                  std::uint32_t pageIndex,
                                  bool transformedVertices,
                                  bool terrainMesh,
                                  const PackedRenderColor& flatColor) {
        const bool animateSiltStrider =
            lowerCopy(mesh.name).find("siltstrider") != std::string::npos;
        auto appendVertex = [&](std::uint32_t sourceVertexIndex,
                                std::uint32_t textureIndex,
                                std::uint32_t flags) {
            const ImportedSceneVertex& srcVertex = scene.vertices[sourceVertexIndex];
            ImportedScenePackedVertex dstVertex{};
            if (transformedVertices) {
                const std::array<float, 3> position = transformPoint(transform, srcVertex.position);
                const std::array<float, 3> normal = normalizeVector(transformDirection(transform, srcVertex.normal));
                dstVertex.position[0] = position[0];
                dstVertex.position[1] = position[1];
                dstVertex.position[2] = position[2];
                dstVertex.normal[0] = normal[0];
                dstVertex.normal[1] = normal[1];
                dstVertex.normal[2] = normal[2];
            } else {
                const std::array<float, 3> normal = normalizeVector({
                    srcVertex.normal[0],
                    srcVertex.normal[1],
                    srcVertex.normal[2]
                });
                dstVertex.position[0] = srcVertex.position[0];
                dstVertex.position[1] = srcVertex.position[1];
                dstVertex.position[2] = srcVertex.position[2];
                dstVertex.normal[0] = normal[0];
                dstVertex.normal[1] = normal[1];
                dstVertex.normal[2] = normal[2];
            }
            const PackedRenderColor color = terrainMesh ? packedTerrainColor(dstVertex.position[1]) : flatColor;
            dstVertex.color[0] = color.r;
            dstVertex.color[1] = color.g;
            dstVertex.color[2] = color.b;
            dstVertex.uv[0] = srcVertex.uv[0];
            dstVertex.uv[1] = srcVertex.uv[1];
            dstVertex.textureIndex = textureIndex;
            dstVertex.flags = flags | (animateSiltStrider ? siltStriderAnimationFlags(srcVertex) : 0u);
            const std::uint32_t packedVertexIndex = static_cast<std::uint32_t>(scene.renderCache.packedVertices.size());
            scene.renderCache.packedVertices.push_back(dstVertex);
            return packedVertexIndex;
        };

        auto appendIndexRange = [&](std::uint32_t firstIndex,
                                    std::uint32_t indexCount,
                                    std::uint32_t textureIndex,
                                    std::uint32_t flags) {
            const std::uint32_t drawFirstIndex = static_cast<std::uint32_t>(scene.renderCache.packedIndices.size());
            std::vector<std::uint32_t> remappedVertexIndices(
                mesh.vertexCount,
                std::numeric_limits<std::uint32_t>::max());
            const std::uint32_t meshLastIndex = mesh.firstIndex + mesh.indexCount;
            const std::uint32_t requestedLastIndex = firstIndex + indexCount;
            const std::uint32_t lastIndex = std::min(requestedLastIndex, meshLastIndex);
            for (std::uint32_t indexOffset = firstIndex; indexOffset < lastIndex; ++indexOffset) {
                const std::uint32_t sourceVertexIndex = scene.indices[indexOffset];
                if (sourceVertexIndex < mesh.firstVertex ||
                    sourceVertexIndex >= mesh.firstVertex + mesh.vertexCount) {
                    continue;
                }
                const std::uint32_t localVertexIndex = sourceVertexIndex - mesh.firstVertex;
                std::uint32_t& remappedIndex = remappedVertexIndices[localVertexIndex];
                if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                    remappedIndex = appendVertex(sourceVertexIndex, textureIndex, flags);
                }
                scene.renderCache.packedIndices.push_back(remappedIndex);
            }
            const std::uint32_t drawIndexCount =
                static_cast<std::uint32_t>(scene.renderCache.packedIndices.size() - drawFirstIndex);
            if (drawIndexCount == 0u) {
                return 0u;
            }
            ImportedScenePackedDraw draw{};
            draw.firstIndex = drawFirstIndex;
            draw.indexCount = drawIndexCount;
            const std::uint32_t drawIndex = static_cast<std::uint32_t>(scene.renderCache.packedDraws.size());
            scene.renderCache.packedDraws.push_back(draw);
            scene.renderCache.drawInstanceIndices.push_back(instanceIndex);
            appendPageDrawRange(pageIndex, drawIndex, terrainMesh);
            return 1u;
        };

        std::uint32_t appendedDrawCount = 0u;
        if (mesh.partCount == 0u) {
            appendedDrawCount += appendIndexRange(
                mesh.firstIndex,
                mesh.indexCount,
                std::numeric_limits<std::uint32_t>::max(),
                0u);
            return appendedDrawCount;
        }
        for (std::uint32_t partIndex = 0; partIndex < mesh.partCount; ++partIndex) {
            const GpuSceneMeshAssetPart& part = scene.meshParts[mesh.firstPart + partIndex];
            const std::uint32_t flags = (part.flags & kGpuSceneComponentFlagAlphaTest) != 0u
                ? kImportedSceneMaterialFlagAlphaTest
                : 0u;
            appendedDrawCount += appendIndexRange(part.firstIndex, part.indexCount, part.textureIndex, flags);
        }
        return appendedDrawCount;
    };

    std::uint32_t firstStaticInstanceIndex = 0u;
    if (!scene.instances.objectIndices.empty() &&
        !scene.instances.flags.empty() &&
        (scene.instances.flags.front() & kGpuSceneComponentFlagTerrain) != 0u) {
        const std::uint32_t terrainPageIndex = scene.instances.pageIndices.front();
        scene.renderCache.terrainDrawCount = appendMeshInstance(
            scene.meshAssets[scene.instances.meshAssetIndices.front()],
            scene.objects.appliedTransforms.front(),
            0u,
            terrainPageIndex,
            false,
            true,
            {});
        firstStaticInstanceIndex = 1u;
    }
    for (std::uint32_t instanceIndex = firstStaticInstanceIndex; instanceIndex < scene.instances.objectIndices.size(); ++instanceIndex) {
        const std::uint32_t meshAssetIndex = scene.instances.meshAssetIndices[instanceIndex];
        if (meshAssetIndex >= scene.meshAssets.size()) {
            continue;
        }
        const std::uint32_t objectIndex = scene.instances.objectIndices[instanceIndex];
        std::string_view objectName;
        if (objectIndex < scene.objects.nameOffsets.size()) {
            const std::uint32_t nameOffset = scene.objects.nameOffsets[objectIndex];
            if (nameOffset < scene.objects.nameBlob.size()) {
                objectName = std::string_view(scene.objects.nameBlob.c_str() + nameOffset);
            }
        }
        appendMeshInstance(
            scene.meshAssets[meshAssetIndex],
            scene.objects.appliedTransforms[objectIndex],
            instanceIndex,
            scene.instances.pageIndices[instanceIndex],
            true,
            false,
            packedRenderColorFromHash(objectName));
    }
}

GpuSceneRuntime createGpuSceneRuntime(const GpuSceneAsset& scene) {
    GpuSceneRuntime runtime{};
    runtime.transforms = scene.transforms;
    runtime.dirtyTransformIndices.resize(scene.transforms.parentIndices.size());
    for (std::uint32_t i = 0; i < runtime.dirtyTransformIndices.size(); ++i) {
        runtime.dirtyTransformIndices[i] = i;
    }
    return runtime;
}

void rebuildGpuSceneWorldTransforms(GpuSceneRuntime& runtime) {
    for (std::size_t transformIndex = 0; transformIndex < runtime.transforms.parentIndices.size(); ++transformIndex) {
        const std::array<float, 16> localMatrix = composeTransform(
            runtime.transforms.localTranslationX[transformIndex],
            runtime.transforms.localTranslationY[transformIndex],
            runtime.transforms.localTranslationZ[transformIndex],
            runtime.transforms.localRotationX[transformIndex],
            runtime.transforms.localRotationY[transformIndex],
            runtime.transforms.localRotationZ[transformIndex],
            runtime.transforms.localRotationW[transformIndex],
            runtime.transforms.localScaleX[transformIndex],
            runtime.transforms.localScaleY[transformIndex],
            runtime.transforms.localScaleZ[transformIndex]);
        const std::int32_t parentIndex = runtime.transforms.parentIndices[transformIndex];
        if (parentIndex >= 0) {
            runtime.transforms.worldMatrices[transformIndex] = multiplyMatrices(
                runtime.transforms.worldMatrices[static_cast<std::size_t>(parentIndex)],
                localMatrix);
        } else {
            runtime.transforms.worldMatrices[transformIndex] = localMatrix;
        }
    }
}

GpuSceneObjectView gpuSceneObjectView(const GpuSceneAsset& scene, std::uint32_t objectIndex) {
    GpuSceneObjectView view{};
    if (objectIndex >= scene.objects.nameOffsets.size()) {
        return view;
    }
    const std::uint32_t nameOffset = scene.objects.nameOffsets[objectIndex];
    view.name = std::string_view(scene.objects.nameBlob.c_str() + nameOffset);
    view.appliedTransform = scene.objects.appliedTransforms[objectIndex].data();
    view.rootTransformIndex = scene.objects.rootTransformIndices[objectIndex];
    view.firstComponentIndex = scene.objects.firstComponentIndices[objectIndex];
    view.componentCount = scene.objects.componentCounts[objectIndex];
    view.pageIndex = scene.objects.pageIndices[objectIndex];
    return view;
}

} // namespace odai::importer
