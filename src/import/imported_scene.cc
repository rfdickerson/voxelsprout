#include "import/imported_scene.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace odai::importer {

namespace {

constexpr std::uint32_t kImportedSceneMagic = 0x4E435356u;  // VSCN
constexpr std::uint32_t kImportedSceneVersion = 16u;
constexpr std::uint32_t kMinSupportedImportedSceneVersion = 15u;
constexpr std::uint32_t kImportedSceneMaterialFlagAlphaTest = 1u;

std::string g_lastImportedSceneError;

void setLastImportedSceneError(std::string message) {
    g_lastImportedSceneError = std::move(message);
}

struct DebugBounds {
    std::array<float, 3> min{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    std::array<float, 3> max{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    };
    bool valid = false;
};

[[maybe_unused]] void expandBounds(DebugBounds& bounds, const std::array<float, 3>& point) {
    bounds.valid = true;
    for (int axis = 0; axis < 3; ++axis) {
        bounds.min[axis] = std::min(bounds.min[axis], point[axis]);
        bounds.max[axis] = std::max(bounds.max[axis], point[axis]);
    }
}

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

void applyTextureAlphaCutoutFlags(ImportedScene& scene) {
    const std::vector<bool> textureAlphaCutoutMask = buildTextureAlphaCutoutMask(scene.textures);
    for (ImportedSceneMesh& mesh : scene.meshes) {
        for (ImportedSceneMeshPart& part : mesh.parts) {
            if (textureIndexUsesAlphaCutout(textureAlphaCutoutMask, part.textureIndex)) {
                part.alphaTest = true;
            }
        }
    }
    for (ImportedScenePackedVertex& vertex : scene.packedVertices) {
        if (textureIndexUsesAlphaCutout(textureAlphaCutoutMask, vertex.textureIndex)) {
            vertex.flags |= kImportedSceneMaterialFlagAlphaTest;
        }
    }
}

bool readExact(std::istream& input, void* dst, std::size_t size) {
    input.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
    return input.good();
}

bool skipExact(std::istream& input, std::size_t size) {
    input.seekg(static_cast<std::streamoff>(size), std::ios::cur);
    return input.good();
}

template <typename T>
bool readValue(std::istream& input, T& out) {
    static_assert(std::is_trivially_copyable_v<T>);
    return readExact(input, &out, sizeof(T));
}

template <typename T>
void writeValue(std::ostream& output, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    output.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
}

void writeString(std::ostream& output, const std::string& value) {
    const std::uint32_t size = static_cast<std::uint32_t>(value.size());
    writeValue(output, size);
    if (!value.empty()) {
        output.write(value.data(), static_cast<std::streamsize>(value.size()));
    }
}

bool readString(std::istream& input, std::string& out) {
    std::uint32_t size = 0;
    if (!readValue(input, size)) {
        return false;
    }
    out.resize(size);
    return size == 0 || readExact(input, out.data(), size);
}

bool skipString(std::istream& input) {
    std::uint32_t size = 0;
    if (!readValue(input, size)) {
        return false;
    }
    return size == 0 || skipExact(input, size);
}

std::array<float, 3> transformPoint(
    const std::array<float, 16>& matrix,
    const std::array<float, 3>& point
) {
    return {
        (matrix[0] * point[0]) + (matrix[1] * point[1]) + (matrix[2] * point[2]) + matrix[3],
        (matrix[4] * point[0]) + (matrix[5] * point[1]) + (matrix[6] * point[2]) + matrix[7],
        (matrix[8] * point[0]) + (matrix[9] * point[1]) + (matrix[10] * point[2]) + matrix[11]
    };
}

std::array<float, 3> transformDirection(
    const std::array<float, 16>& matrix,
    const std::array<float, 3>& direction
) {
    return {
        (matrix[0] * direction[0]) + (matrix[1] * direction[1]) + (matrix[2] * direction[2]),
        (matrix[4] * direction[0]) + (matrix[5] * direction[1]) + (matrix[6] * direction[2]),
        (matrix[8] * direction[0]) + (matrix[9] * direction[1]) + (matrix[10] * direction[2])
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

}  // namespace

void buildImportedScenePackedRenderData(ImportedScene& scene) {
    applyTextureAlphaCutoutFlags(scene);
    scene.packedVertices.clear();
    scene.packedIndices.clear();
    scene.packedDraws.clear();
    scene.boundsMin[0] = std::numeric_limits<float>::max();
    scene.boundsMin[1] = std::numeric_limits<float>::max();
    scene.boundsMin[2] = std::numeric_limits<float>::max();
    scene.boundsMax[0] = std::numeric_limits<float>::lowest();
    scene.boundsMax[1] = std::numeric_limits<float>::lowest();
    scene.boundsMax[2] = std::numeric_limits<float>::lowest();

    auto expandBounds = [&](const ImportedScenePackedVertex& vertex) {
        scene.boundsMin[0] = std::min(scene.boundsMin[0], vertex.position[0]);
        scene.boundsMin[1] = std::min(scene.boundsMin[1], vertex.position[1]);
        scene.boundsMin[2] = std::min(scene.boundsMin[2], vertex.position[2]);
        scene.boundsMax[0] = std::max(scene.boundsMax[0], vertex.position[0]);
        scene.boundsMax[1] = std::max(scene.boundsMax[1], vertex.position[1]);
        scene.boundsMax[2] = std::max(scene.boundsMax[2], vertex.position[2]);
    };

    auto appendMesh = [&](const ImportedSceneMesh& mesh,
                          const std::array<float, 16>& transform,
                          const PackedRenderColor& color) {
        if (mesh.vertices.empty() || mesh.indices.empty()) {
            return;
        }
        const std::uint32_t firstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
        const auto appendVertex = [&](const ImportedSceneVertex& srcVertex,
                                      std::uint32_t textureIndex,
                                      std::uint32_t flags) {
            ImportedScenePackedVertex dstVertex{};
            const std::array<float, 3> localPosition{
                srcVertex.position[0],
                srcVertex.position[1],
                srcVertex.position[2]
            };
            const std::array<float, 3> localNormal{
                srcVertex.normal[0],
                srcVertex.normal[1],
                srcVertex.normal[2]
            };
            const std::array<float, 3> worldPosition = transformPoint(transform, localPosition);
            const std::array<float, 3> worldNormal = normalizeVector(transformDirection(transform, localNormal));
            dstVertex.position[0] = worldPosition[0];
            dstVertex.position[1] = worldPosition[1];
            dstVertex.position[2] = worldPosition[2];
            dstVertex.normal[0] = worldNormal[0];
            dstVertex.normal[1] = worldNormal[1];
            dstVertex.normal[2] = worldNormal[2];
            dstVertex.color[0] = color.r;
            dstVertex.color[1] = color.g;
            dstVertex.color[2] = color.b;
            dstVertex.uv[0] = srcVertex.uv[0];
            dstVertex.uv[1] = srcVertex.uv[1];
            dstVertex.textureIndex = textureIndex;
            dstVertex.flags = flags;
            const std::uint32_t packedVertexIndex = static_cast<std::uint32_t>(scene.packedVertices.size());
            scene.packedVertices.push_back(dstVertex);
            expandBounds(dstVertex);
            return packedVertexIndex;
        };

        if (mesh.parts.empty()) {
            const std::uint32_t invalidTextureIndex = std::numeric_limits<std::uint32_t>::max();
            std::vector<std::uint32_t> remappedVertexIndices(
                mesh.vertices.size(),
                std::numeric_limits<std::uint32_t>::max());
            for (const std::uint32_t index : mesh.indices) {
                if (index >= mesh.vertices.size()) {
                    continue;
                }
                std::uint32_t& remappedIndex = remappedVertexIndices[index];
                if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                    remappedIndex = appendVertex(mesh.vertices[index], invalidTextureIndex, 0u);
                }
                scene.packedIndices.push_back(remappedIndex);
            }
        } else {
            for (const ImportedSceneMeshPart& part : mesh.parts) {
                if (part.indexCount == 0u || part.firstIndex >= mesh.indices.size()) {
                    continue;
                }
                std::vector<std::uint32_t> remappedVertexIndices(
                    mesh.vertices.size(),
                    std::numeric_limits<std::uint32_t>::max());
                const std::size_t firstPartIndex = static_cast<std::size_t>(part.firstIndex);
                const std::size_t lastPartIndex = std::min(
                    firstPartIndex + static_cast<std::size_t>(part.indexCount),
                    mesh.indices.size());
                const std::uint32_t partFlags =
                    part.alphaTest ? kImportedSceneMaterialFlagAlphaTest : 0u;
                for (std::size_t indexOffset = firstPartIndex; indexOffset < lastPartIndex; ++indexOffset) {
                    const std::uint32_t index = mesh.indices[indexOffset];
                    if (index >= mesh.vertices.size()) {
                        continue;
                    }
                    std::uint32_t& remappedIndex = remappedVertexIndices[index];
                    if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                        remappedIndex = appendVertex(mesh.vertices[index], part.textureIndex, partFlags);
                    }
                    scene.packedIndices.push_back(remappedIndex);
                }
            }
        }
        const std::uint32_t indexCount = static_cast<std::uint32_t>(scene.packedIndices.size() - firstIndex);
        if (indexCount != 0u) {
            scene.packedDraws.push_back(ImportedScenePackedDraw{firstIndex, indexCount});
        }
    };

    const bool hasTerrainMesh = !scene.meshes.empty() && scene.meshes.front().name == "terrain";
    if (hasTerrainMesh) {
        const ImportedSceneMesh& terrainMesh = scene.meshes.front();
        if (!terrainMesh.vertices.empty() && !terrainMesh.indices.empty()) {
            for (const ImportedSceneMeshPart& part : terrainMesh.parts) {
                if (part.indexCount == 0u || part.firstIndex >= terrainMesh.indices.size()) {
                    continue;
                }
                const std::uint32_t firstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
                std::vector<std::uint32_t> remappedVertexIndices(
                    terrainMesh.vertices.size(),
                    std::numeric_limits<std::uint32_t>::max());
                const std::size_t firstPartIndex = static_cast<std::size_t>(part.firstIndex);
                const std::size_t lastPartIndex = std::min(
                    firstPartIndex + static_cast<std::size_t>(part.indexCount),
                    terrainMesh.indices.size());
                for (std::size_t indexOffset = firstPartIndex; indexOffset < lastPartIndex; ++indexOffset) {
                    const std::uint32_t index = terrainMesh.indices[indexOffset];
                    if (index >= terrainMesh.vertices.size()) {
                        continue;
                    }
                    std::uint32_t& remappedIndex = remappedVertexIndices[index];
                    if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                        const ImportedSceneVertex& srcVertex = terrainMesh.vertices[index];
                        ImportedScenePackedVertex dstVertex{};
                        const std::array<float, 3> worldNormal = normalizeVector({
                            srcVertex.normal[0],
                            srcVertex.normal[1],
                            srcVertex.normal[2]
                        });
                        const PackedRenderColor color = packedTerrainColor(srcVertex.position[1]);
                        dstVertex.position[0] = srcVertex.position[0];
                        dstVertex.position[1] = srcVertex.position[1];
                        dstVertex.position[2] = srcVertex.position[2];
                        dstVertex.normal[0] = worldNormal[0];
                        dstVertex.normal[1] = worldNormal[1];
                        dstVertex.normal[2] = worldNormal[2];
                        dstVertex.color[0] = color.r;
                        dstVertex.color[1] = color.g;
                        dstVertex.color[2] = color.b;
                        dstVertex.uv[0] = srcVertex.uv[0];
                        dstVertex.uv[1] = srcVertex.uv[1];
                        dstVertex.textureIndex = part.textureIndex;
                        dstVertex.flags = 0u;
                        remappedIndex = static_cast<std::uint32_t>(scene.packedVertices.size());
                        scene.packedVertices.push_back(dstVertex);
                        expandBounds(dstVertex);
                    }
                    scene.packedIndices.push_back(remappedIndex);
                }
                const std::uint32_t indexCount =
                    static_cast<std::uint32_t>(scene.packedIndices.size() - firstIndex);
                if (indexCount != 0u) {
                    scene.packedDraws.push_back(ImportedScenePackedDraw{firstIndex, indexCount});
                }
            }
        }

    }

    for (const ImportedSceneInstance& instance : scene.instances) {
        if ((hasTerrainMesh && instance.meshIndex == 0u) || instance.meshIndex >= scene.meshes.size()) {
            continue;
        }
        std::array<float, 16> transform{};
        std::copy(std::begin(instance.transform), std::end(instance.transform), transform.begin());
        appendMesh(
            scene.meshes[instance.meshIndex],
            transform,
            packedRenderColorFromHash(instance.modelPath));
    }

    if (scene.packedVertices.empty()) {
        scene.boundsMin[0] = 0.0f;
        scene.boundsMin[1] = 0.0f;
        scene.boundsMin[2] = 0.0f;
        scene.boundsMax[0] = 0.0f;
        scene.boundsMax[1] = 0.0f;
        scene.boundsMax[2] = 0.0f;
    }
}

void computeImportedSceneBoundsFromPackedData(ImportedScene& scene) {
    if (scene.packedVertices.empty()) {
        scene.boundsMin[0] = 0.0f;
        scene.boundsMin[1] = 0.0f;
        scene.boundsMin[2] = 0.0f;
        scene.boundsMax[0] = 0.0f;
        scene.boundsMax[1] = 0.0f;
        scene.boundsMax[2] = 0.0f;
        return;
    }
    scene.boundsMin[0] = std::numeric_limits<float>::max();
    scene.boundsMin[1] = std::numeric_limits<float>::max();
    scene.boundsMin[2] = std::numeric_limits<float>::max();
    scene.boundsMax[0] = std::numeric_limits<float>::lowest();
    scene.boundsMax[1] = std::numeric_limits<float>::lowest();
    scene.boundsMax[2] = std::numeric_limits<float>::lowest();
    for (const ImportedScenePackedVertex& vertex : scene.packedVertices) {
        scene.boundsMin[0] = std::min(scene.boundsMin[0], vertex.position[0]);
        scene.boundsMin[1] = std::min(scene.boundsMin[1], vertex.position[1]);
        scene.boundsMin[2] = std::min(scene.boundsMin[2], vertex.position[2]);
        scene.boundsMax[0] = std::max(scene.boundsMax[0], vertex.position[0]);
        scene.boundsMax[1] = std::max(scene.boundsMax[1], vertex.position[1]);
        scene.boundsMax[2] = std::max(scene.boundsMax[2], vertex.position[2]);
    }
}

const std::string& getImportedSceneLastError() {
    return g_lastImportedSceneError;
}

bool saveImportedScene(const ImportedScene& scene, const std::filesystem::path& outputPath) {
    g_lastImportedSceneError.clear();
    const std::filesystem::path parentPath = outputPath.parent_path();
    if (!parentPath.empty()) {
        std::error_code mkdirError;
        std::filesystem::create_directories(parentPath, mkdirError);
        if (mkdirError) {
            setLastImportedSceneError(
                "Failed to create output directory " + parentPath.string() + ": " + mkdirError.message());
            return false;
        }
    }

    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        setLastImportedSceneError("Failed to open output file for writing: " + outputPath.string());
        return false;
    }

    writeValue(output, kImportedSceneMagic);
    writeValue(output, kImportedSceneVersion);
    writeString(output, scene.sourceTag);

    const std::uint32_t textureCount = static_cast<std::uint32_t>(scene.textures.size());
    const std::uint32_t meshCount = static_cast<std::uint32_t>(scene.meshes.size());
    const std::uint32_t instanceCount = static_cast<std::uint32_t>(scene.instances.size());
    const std::uint32_t landscapeCellCount = static_cast<std::uint32_t>(scene.landscapeCells.size());
    const std::uint32_t waterPatchCount = static_cast<std::uint32_t>(scene.waterPatches.size());
    const std::uint32_t lightCount = static_cast<std::uint32_t>(scene.lights.size());
    const std::uint32_t unresolvedRefCount = static_cast<std::uint32_t>(scene.unresolvedRefs.size());
    const std::uint32_t packedVertexCount = static_cast<std::uint32_t>(scene.packedVertices.size());
    const std::uint32_t packedIndexCount = static_cast<std::uint32_t>(scene.packedIndices.size());
    const std::uint32_t packedDrawCount = static_cast<std::uint32_t>(scene.packedDraws.size());
    writeValue(output, textureCount);
    writeValue(output, meshCount);
    writeValue(output, instanceCount);
    writeValue(output, landscapeCellCount);
    writeValue(output, waterPatchCount);
    writeValue(output, lightCount);
    writeValue(output, unresolvedRefCount);
    writeValue(output, packedVertexCount);
    writeValue(output, packedIndexCount);
    writeValue(output, packedDrawCount);
    output.write(reinterpret_cast<const char*>(scene.boundsMin), static_cast<std::streamsize>(sizeof(scene.boundsMin)));
    output.write(reinterpret_cast<const char*>(scene.boundsMax), static_cast<std::streamsize>(sizeof(scene.boundsMax)));

    for (const ImportedSceneTexture& texture : scene.textures) {
        writeString(output, texture.sourcePath);
        writeValue(output, texture.width);
        writeValue(output, texture.height);
        writeValue(output, texture.mipLevelCount);
        const std::uint32_t rgbaSize = static_cast<std::uint32_t>(texture.rgba8.size());
        writeValue(output, rgbaSize);
        if (!texture.rgba8.empty()) {
            output.write(reinterpret_cast<const char*>(texture.rgba8.data()), static_cast<std::streamsize>(texture.rgba8.size()));
        }
    }

    for (const ImportedSceneMesh& mesh : scene.meshes) {
        writeString(output, mesh.name);
        const std::uint32_t vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
        const std::uint32_t indexCount = static_cast<std::uint32_t>(mesh.indices.size());
        const std::uint32_t partCount = static_cast<std::uint32_t>(mesh.parts.size());
        writeValue(output, vertexCount);
        writeValue(output, indexCount);
        writeValue(output, partCount);
        if (!mesh.vertices.empty()) {
            output.write(reinterpret_cast<const char*>(mesh.vertices.data()),
                         static_cast<std::streamsize>(mesh.vertices.size() * sizeof(ImportedSceneVertex)));
        }
        if (!mesh.indices.empty()) {
            output.write(reinterpret_cast<const char*>(mesh.indices.data()),
                         static_cast<std::streamsize>(mesh.indices.size() * sizeof(std::uint32_t)));
        }
        if (!mesh.parts.empty()) {
            output.write(reinterpret_cast<const char*>(mesh.parts.data()),
                         static_cast<std::streamsize>(mesh.parts.size() * sizeof(ImportedSceneMeshPart)));
        }
    }

    for (const ImportedSceneInstance& instance : scene.instances) {
        writeValue(output, instance.meshIndex);
        output.write(reinterpret_cast<const char*>(instance.transform), static_cast<std::streamsize>(sizeof(instance.transform)));
        writeString(output, instance.sourceId);
        writeString(output, instance.modelPath);
    }

    for (const ImportedSceneLandscapeCell& cell : scene.landscapeCells) {
        writeValue(output, cell.gridX);
        writeValue(output, cell.gridY);
        const std::uint32_t heightCount = static_cast<std::uint32_t>(cell.heights.size());
        const std::uint32_t textureIndexCount = static_cast<std::uint32_t>(cell.textureIndices.size());
        writeValue(output, heightCount);
        writeValue(output, textureIndexCount);
        if (!cell.heights.empty()) {
            output.write(reinterpret_cast<const char*>(cell.heights.data()),
                         static_cast<std::streamsize>(cell.heights.size() * sizeof(float)));
        }
        if (!cell.textureIndices.empty()) {
            output.write(reinterpret_cast<const char*>(cell.textureIndices.data()),
                         static_cast<std::streamsize>(cell.textureIndices.size() * sizeof(std::uint16_t)));
        }
    }

    if (!scene.waterPatches.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.waterPatches.data()),
            static_cast<std::streamsize>(scene.waterPatches.size() * sizeof(ImportedSceneWaterPatch)));
    }

    for (const ImportedSceneLight& light : scene.lights) {
        writeString(output, light.sourceId);
        output.write(reinterpret_cast<const char*>(light.position), static_cast<std::streamsize>(sizeof(light.position)));
        output.write(reinterpret_cast<const char*>(light.color), static_cast<std::streamsize>(sizeof(light.color)));
        writeValue(output, light.radius);
        writeValue(output, light.intensity);
        writeValue(output, light.flags);
    }

    for (const ImportedSceneCellRef& ref : scene.unresolvedRefs) {
        writeString(output, ref.refId);
        writeString(output, ref.modelPath);
        output.write(reinterpret_cast<const char*>(ref.position), static_cast<std::streamsize>(sizeof(ref.position)));
        output.write(reinterpret_cast<const char*>(ref.rotationRadians), static_cast<std::streamsize>(sizeof(ref.rotationRadians)));
        writeValue(output, ref.scale);
    }

    if (!scene.packedVertices.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.packedVertices.data()),
            static_cast<std::streamsize>(scene.packedVertices.size() * sizeof(ImportedScenePackedVertex)));
    }
    if (!scene.packedIndices.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.packedIndices.data()),
            static_cast<std::streamsize>(scene.packedIndices.size() * sizeof(std::uint32_t)));
    }
    if (!scene.packedDraws.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.packedDraws.data()),
            static_cast<std::streamsize>(scene.packedDraws.size() * sizeof(ImportedScenePackedDraw)));
    }

    if (!output.good()) {
        setLastImportedSceneError("Failed while writing output file: " + outputPath.string());
        return false;
    }
    return true;
}

bool loadImportedScene(const std::filesystem::path& inputPath, ImportedScene& outScene) {
    g_lastImportedSceneError.clear();
    std::ifstream input(inputPath, std::ios::binary);
    if (!input) {
        setLastImportedSceneError("Failed to open imported scene file: " + inputPath.string());
        return false;
    }

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!readValue(input, magic) || !readValue(input, version) || magic != kImportedSceneMagic) {
        setLastImportedSceneError("Invalid imported scene file header: " + inputPath.string());
        return false;
    }
    if (version < kMinSupportedImportedSceneVersion || version > kImportedSceneVersion) {
        setLastImportedSceneError(
            "Imported scene file version " + std::to_string(version) +
            " is unsupported; recook with the current odai_balmora_cooker (supported versions " +
            std::to_string(kMinSupportedImportedSceneVersion) + "-" +
            std::to_string(kImportedSceneVersion) + ")");
        return false;
    }

    ImportedScene scene{};
    scene.sourceFileVersion = version;
    if (!readString(input, scene.sourceTag)) {
        setLastImportedSceneError("Failed to read imported scene source tag: " + inputPath.string());
        return false;
    }

    std::uint32_t textureCount = 0;
    std::uint32_t meshCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t landscapeCellCount = 0;
    std::uint32_t waterPatchCount = 0;
    std::uint32_t lightCount = 0;
    std::uint32_t unresolvedRefCount = 0;
    std::uint32_t packedVertexCount = 0;
    std::uint32_t packedIndexCount = 0;
    std::uint32_t packedDrawCount = 0;
    if (!readValue(input, textureCount) ||
        !readValue(input, meshCount) ||
        !readValue(input, instanceCount) ||
        !readValue(input, landscapeCellCount) ||
        !readValue(input, waterPatchCount) ||
        !readValue(input, lightCount) ||
        !readValue(input, unresolvedRefCount)) {
        return false;
    }
    if (version >= 2u &&
        (!readValue(input, packedVertexCount) ||
         !readValue(input, packedIndexCount) ||
         !readValue(input, packedDrawCount))) {
        return false;
    }
    if (version >= 3u &&
        (!readExact(input, scene.boundsMin, sizeof(scene.boundsMin)) ||
         !readExact(input, scene.boundsMax, sizeof(scene.boundsMax)))) {
        return false;
    }
    scene.sourceTextureCount = textureCount;
    scene.sourceMeshCount = meshCount;
    scene.sourceInstanceCount = instanceCount;
    scene.sourceLandscapeCellCount = landscapeCellCount;
    scene.sourceWaterPatchCount = waterPatchCount;
    scene.sourceLightCount = lightCount;
    scene.sourceUnresolvedRefCount = unresolvedRefCount;

    scene.textures.resize(textureCount);
    for (ImportedSceneTexture& texture : scene.textures) {
        if (!readString(input, texture.sourcePath) ||
            !readValue(input, texture.width) ||
            !readValue(input, texture.height) ||
            !readValue(input, texture.mipLevelCount)) {
            return false;
        }
        std::uint32_t rgbaSize = 0;
        if (!readValue(input, rgbaSize)) {
            return false;
        }
        texture.rgba8.resize(rgbaSize);
        if (rgbaSize != 0 && !readExact(input, texture.rgba8.data(), rgbaSize)) {
            return false;
        }
    }

    scene.meshes.resize(meshCount);
    for (ImportedSceneMesh& mesh : scene.meshes) {
        std::uint32_t vertexCount = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t partCount = 0;
        if (!readString(input, mesh.name) ||
            !readValue(input, vertexCount) ||
            !readValue(input, indexCount) ||
            !readValue(input, partCount)) {
            return false;
        }
        mesh.vertices.resize(vertexCount);
        mesh.indices.resize(indexCount);
        mesh.parts.resize(partCount);
        if (vertexCount != 0 &&
            !readExact(input, mesh.vertices.data(), mesh.vertices.size() * sizeof(ImportedSceneVertex))) {
            return false;
        }
        if (indexCount != 0 &&
            !readExact(input, mesh.indices.data(), mesh.indices.size() * sizeof(std::uint32_t))) {
            return false;
        }
        if (partCount != 0 &&
            !readExact(input, mesh.parts.data(), mesh.parts.size() * sizeof(ImportedSceneMeshPart))) {
            return false;
        }
    }

    scene.instances.resize(instanceCount);
    for (ImportedSceneInstance& instance : scene.instances) {
        if (!readValue(input, instance.meshIndex) ||
            !readExact(input, instance.transform, sizeof(instance.transform)) ||
            !readString(input, instance.sourceId) ||
            !readString(input, instance.modelPath)) {
            return false;
        }
    }

    scene.landscapeCells.resize(landscapeCellCount);
    for (ImportedSceneLandscapeCell& cell : scene.landscapeCells) {
        std::uint32_t heightCount = 0;
        std::uint32_t textureIndexCount = 0;
        if (!readValue(input, cell.gridX) ||
            !readValue(input, cell.gridY) ||
            !readValue(input, heightCount) ||
            !readValue(input, textureIndexCount)) {
            return false;
        }
        cell.heights.resize(heightCount);
        cell.textureIndices.resize(textureIndexCount);
        if (heightCount != 0 &&
            !readExact(input, cell.heights.data(), cell.heights.size() * sizeof(float))) {
            return false;
        }
        if (textureIndexCount != 0 &&
            !readExact(input, cell.textureIndices.data(), cell.textureIndices.size() * sizeof(std::uint16_t))) {
            return false;
        }
    }

    scene.waterPatches.resize(waterPatchCount);
    if (waterPatchCount != 0 &&
        !readExact(input, scene.waterPatches.data(), scene.waterPatches.size() * sizeof(ImportedSceneWaterPatch))) {
        return false;
    }

    scene.lights.resize(lightCount);
    for (ImportedSceneLight& light : scene.lights) {
        if (!readString(input, light.sourceId) ||
            !readExact(input, light.position, sizeof(light.position)) ||
            !readExact(input, light.color, sizeof(light.color)) ||
            !readValue(input, light.radius) ||
            !readValue(input, light.intensity) ||
            !readValue(input, light.flags)) {
            return false;
        }
        if (version < 16u) {
            const float morrowindY = light.position[1];
            light.position[1] = light.position[2];
            light.position[2] = morrowindY;
        }
    }

    scene.unresolvedRefs.resize(unresolvedRefCount);
    for (ImportedSceneCellRef& ref : scene.unresolvedRefs) {
        if (!readString(input, ref.refId) ||
            !readString(input, ref.modelPath) ||
            !readExact(input, ref.position, sizeof(ref.position)) ||
            !readExact(input, ref.rotationRadians, sizeof(ref.rotationRadians)) ||
            !readValue(input, ref.scale)) {
            return false;
        }
    }

    if (version >= 2u) {
        scene.packedVertices.resize(packedVertexCount);
        scene.packedIndices.resize(packedIndexCount);
        scene.packedDraws.resize(packedDrawCount);
        if (packedVertexCount != 0 &&
            !readExact(input, scene.packedVertices.data(), scene.packedVertices.size() * sizeof(ImportedScenePackedVertex))) {
            return false;
        }
        if (packedIndexCount != 0 &&
            !readExact(input, scene.packedIndices.data(), scene.packedIndices.size() * sizeof(std::uint32_t))) {
            return false;
        }
        if (packedDrawCount != 0 &&
            !readExact(input, scene.packedDraws.data(), scene.packedDraws.size() * sizeof(ImportedScenePackedDraw))) {
            return false;
        }
    } else {
        buildImportedScenePackedRenderData(scene);
    }

    applyTextureAlphaCutoutFlags(scene);
    outScene = std::move(scene);
    return true;
}

bool loadImportedSceneRuntime(const std::filesystem::path& inputPath, ImportedScene& outScene) {
    g_lastImportedSceneError.clear();
    std::ifstream input(inputPath, std::ios::binary);
    if (!input) {
        setLastImportedSceneError("Failed to open imported scene file: " + inputPath.string());
        return false;
    }

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!readValue(input, magic) || !readValue(input, version) || magic != kImportedSceneMagic) {
        setLastImportedSceneError("Invalid imported scene file header: " + inputPath.string());
        return false;
    }
    if (version < kMinSupportedImportedSceneVersion || version > kImportedSceneVersion) {
        setLastImportedSceneError(
            "Imported scene file version " + std::to_string(version) +
            " is unsupported; recook with the current odai_balmora_cooker (supported versions " +
            std::to_string(kMinSupportedImportedSceneVersion) + "-" +
            std::to_string(kImportedSceneVersion) + ")");
        return false;
    }

    ImportedScene scene{};
    scene.sourceFileVersion = version;
    if (!readString(input, scene.sourceTag)) {
        return false;
    }

    std::uint32_t textureCount = 0;
    std::uint32_t meshCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t landscapeCellCount = 0;
    std::uint32_t waterPatchCount = 0;
    std::uint32_t lightCount = 0;
    std::uint32_t unresolvedRefCount = 0;
    std::uint32_t packedVertexCount = 0;
    std::uint32_t packedIndexCount = 0;
    std::uint32_t packedDrawCount = 0;
    if (!readValue(input, textureCount) ||
        !readValue(input, meshCount) ||
        !readValue(input, instanceCount) ||
        !readValue(input, landscapeCellCount) ||
        !readValue(input, waterPatchCount) ||
        !readValue(input, lightCount) ||
        !readValue(input, unresolvedRefCount) ||
        !readValue(input, packedVertexCount) ||
        !readValue(input, packedIndexCount) ||
        !readValue(input, packedDrawCount)) {
        return false;
    }
    if (version >= 3u &&
        (!readExact(input, scene.boundsMin, sizeof(scene.boundsMin)) ||
         !readExact(input, scene.boundsMax, sizeof(scene.boundsMax)))) {
        return false;
    }
    scene.sourceTextureCount = textureCount;
    scene.sourceMeshCount = meshCount;
    scene.sourceInstanceCount = instanceCount;
    scene.sourceLandscapeCellCount = landscapeCellCount;
    scene.sourceWaterPatchCount = waterPatchCount;
    scene.sourceLightCount = lightCount;
    scene.sourceUnresolvedRefCount = unresolvedRefCount;

    scene.textures.resize(textureCount);
    for (ImportedSceneTexture& texture : scene.textures) {
        std::uint32_t rgbaSize = 0;
        if (!readString(input, texture.sourcePath) ||
            !readValue(input, texture.width) ||
            !readValue(input, texture.height) ||
            !readValue(input, texture.mipLevelCount) ||
            !readValue(input, rgbaSize)) {
            return false;
        }
        texture.rgba8.resize(rgbaSize);
        if (rgbaSize != 0 &&
            !readExact(input, texture.rgba8.data(), texture.rgba8.size())) {
            return false;
        }
    }

    for (std::uint32_t i = 0; i < meshCount; ++i) {
        std::uint32_t vertexCount = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t partCount = 0;
        if (!skipString(input) ||
            !readValue(input, vertexCount) ||
            !readValue(input, indexCount) ||
            !readValue(input, partCount)) {
            return false;
        }
        const std::size_t vertexBytes = static_cast<std::size_t>(vertexCount) * sizeof(ImportedSceneVertex);
        const std::size_t indexBytes = static_cast<std::size_t>(indexCount) * sizeof(std::uint32_t);
        const std::size_t partBytes = static_cast<std::size_t>(partCount) * sizeof(ImportedSceneMeshPart);
        if ((vertexBytes != 0 && !skipExact(input, vertexBytes)) ||
            (indexBytes != 0 && !skipExact(input, indexBytes)) ||
            (partBytes != 0 && !skipExact(input, partBytes))) {
            return false;
        }
    }

    for (std::uint32_t i = 0; i < instanceCount; ++i) {
        std::uint32_t meshIndex = 0;
        float transform[16] = {};
        if (!readValue(input, meshIndex) ||
            !readExact(input, transform, sizeof(transform)) ||
            !skipString(input) ||
            !skipString(input)) {
            return false;
        }
    }

    for (std::uint32_t i = 0; i < landscapeCellCount; ++i) {
        int gridX = 0;
        int gridY = 0;
        std::uint32_t heightCount = 0;
        std::uint32_t textureIndexCount = 0;
        if (!readValue(input, gridX) ||
            !readValue(input, gridY) ||
            !readValue(input, heightCount) ||
            !readValue(input, textureIndexCount)) {
            return false;
        }
        const std::size_t heightBytes = static_cast<std::size_t>(heightCount) * sizeof(float);
        const std::size_t textureIndexBytes = static_cast<std::size_t>(textureIndexCount) * sizeof(std::uint16_t);
        if ((heightBytes != 0 && !skipExact(input, heightBytes)) ||
            (textureIndexBytes != 0 && !skipExact(input, textureIndexBytes))) {
            return false;
        }
    }

    scene.waterPatches.resize(waterPatchCount);
    if (waterPatchCount != 0 &&
        !readExact(input, scene.waterPatches.data(), scene.waterPatches.size() * sizeof(ImportedSceneWaterPatch))) {
        return false;
    }

    scene.lights.resize(lightCount);
    for (ImportedSceneLight& light : scene.lights) {
        if (!readString(input, light.sourceId) ||
            !readExact(input, light.position, sizeof(light.position)) ||
            !readExact(input, light.color, sizeof(light.color)) ||
            !readValue(input, light.radius) ||
            !readValue(input, light.intensity) ||
            !readValue(input, light.flags)) {
            return false;
        }
        if (version < 16u) {
            const float morrowindY = light.position[1];
            light.position[1] = light.position[2];
            light.position[2] = morrowindY;
        }
    }

    for (std::uint32_t i = 0; i < unresolvedRefCount; ++i) {
        float position[3] = {};
        float rotation[3] = {};
        float scale = 1.0f;
        if (!skipString(input) ||
            !skipString(input) ||
            !readExact(input, position, sizeof(position)) ||
            !readExact(input, rotation, sizeof(rotation)) ||
            !readValue(input, scale)) {
            return false;
        }
    }

    scene.packedVertices.resize(packedVertexCount);
    scene.packedIndices.resize(packedIndexCount);
    scene.packedDraws.resize(packedDrawCount);
    if ((packedVertexCount != 0 &&
         !readExact(input, scene.packedVertices.data(), scene.packedVertices.size() * sizeof(ImportedScenePackedVertex))) ||
        (packedIndexCount != 0 &&
         !readExact(input, scene.packedIndices.data(), scene.packedIndices.size() * sizeof(std::uint32_t))) ||
        (packedDrawCount != 0 &&
         !readExact(input, scene.packedDraws.data(), scene.packedDraws.size() * sizeof(ImportedScenePackedDraw)))) {
        return false;
    }
    if (version < 3u) {
        computeImportedSceneBoundsFromPackedData(scene);
    }

    applyTextureAlphaCutoutFlags(scene);
    outScene = std::move(scene);
    return true;
}

bool exportImportedSceneTerrainObj(const ImportedScene& scene, const std::filesystem::path& outputObjPath) {
    g_lastImportedSceneError.clear();
    if (scene.meshes.empty()) {
        setLastImportedSceneError("Scene does not contain any meshes to export");
        return false;
    }
    const ImportedSceneMesh& mesh = scene.meshes.front();
    const std::filesystem::path parentPath = outputObjPath.parent_path();
    if (!parentPath.empty()) {
        std::error_code mkdirError;
        std::filesystem::create_directories(parentPath, mkdirError);
        if (mkdirError) {
            setLastImportedSceneError(
                "Failed to create OBJ output directory " + parentPath.string() + ": " + mkdirError.message());
            return false;
        }
    }
    std::ofstream output(outputObjPath, std::ios::trunc);
    if (!output) {
        setLastImportedSceneError("Failed to open OBJ output file for writing: " + outputObjPath.string());
        return false;
    }
    output << "o terrain\n";
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        output << "v " << vertex.position[0] << " " << vertex.position[1] << " " << vertex.position[2] << "\n";
    }
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        output << "vn " << vertex.normal[0] << " " << vertex.normal[1] << " " << vertex.normal[2] << "\n";
    }
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        output << "vt " << vertex.uv[0] << " " << (1.0f - vertex.uv[1]) << "\n";
    }
    for (const ImportedSceneMeshPart& part : mesh.parts) {
        output << "g terrain_part_" << part.textureIndex << "_" << part.firstIndex << "\n";
        for (std::uint32_t i = 0; i + 2u < part.indexCount; i += 3u) {
            const std::uint32_t i0 = mesh.indices[part.firstIndex + i] + 1u;
            const std::uint32_t i1 = mesh.indices[part.firstIndex + i + 1u] + 1u;
            const std::uint32_t i2 = mesh.indices[part.firstIndex + i + 2u] + 1u;
            output << "f "
                   << i0 << "/" << i0 << "/" << i0 << " "
                   << i1 << "/" << i1 << "/" << i1 << " "
                   << i2 << "/" << i2 << "/" << i2 << "\n";
        }
    }
    if (!output.good()) {
        setLastImportedSceneError("Failed while writing OBJ output file: " + outputObjPath.string());
        return false;
    }
    return true;
}


}  // namespace odai::importer
