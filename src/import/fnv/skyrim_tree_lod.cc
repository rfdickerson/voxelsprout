#include "import/fnv/skyrim_tree_lod.h"

#include "import/dds.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace odai::importer::fnv {
namespace {

constexpr std::uint32_t kMaxTreeTypes = 4096u;
constexpr std::uint32_t kMaxTileGroups = 4096u;
constexpr std::uint32_t kMaxTileInstances = 1'000'000u;
constexpr std::size_t kListRecordBytes = 32u;
constexpr std::size_t kInstanceBytes = 32u;

std::uint32_t readU32(const std::uint8_t* bytes) {
    std::uint32_t value = 0u;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

float readF32(const std::uint8_t* bytes) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

void appendCard(
    ImportedSceneMesh& mesh, const SkyrimTreeLodType& type,
    const SkyrimTreeLodInstance& tree, float angle, float phase) {
    const float halfWidth = 0.5f * type.width * tree.scale;
    const float height = type.height * tree.scale;
    const float dx = std::cos(angle) * halfWidth;
    const float dy = std::sin(angle) * halfWidth;
    const float nx = -std::sin(angle);
    const float ny = std::cos(angle);
    const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
    const float bx[4] = {
        tree.position[0] - dx, tree.position[0] + dx,
        tree.position[0] + dx, tree.position[0] - dx};
    const float by[4] = {
        tree.position[1] - dy, tree.position[1] + dy,
        tree.position[1] + dy, tree.position[1] - dy};
    const float bz[4] = {
        tree.position[2], tree.position[2],
        tree.position[2] + height, tree.position[2] + height};
    const float u[4] = {type.uvMin[0], type.uvMax[0], type.uvMax[0], type.uvMin[0]};
    const float v[4] = {type.uvMax[1], type.uvMax[1], type.uvMin[1], type.uvMin[1]};
    for (std::size_t corner = 0u; corner < 4u; ++corner) {
        ImportedSceneVertex vertex{};
        // BTT positions are already world-space Bethesda coordinates.
        vertex.position[0] = bx[corner];
        vertex.position[1] = bz[corner];
        vertex.position[2] = -by[corner];
        vertex.normal[0] = nx;
        vertex.normal[1] = 0.0f;
        vertex.normal[2] = -ny;
        vertex.uv[0] = u[corner];
        vertex.uv[1] = v[corner];
        vertex.layerTextureIndex[3] = kImportedSceneFoliageWindMarker;
        vertex.layerWeight[0] = corner >= 2u ? 1.0f : 0.0f;
        vertex.layerWeight[1] = phase;
        mesh.vertices.push_back(vertex);
    }
    const std::uint32_t indices[6] = {base, base + 1u, base + 2u,
                                      base, base + 2u, base + 3u};
    mesh.indices.insert(mesh.indices.end(), std::begin(indices), std::end(indices));
}

}  // namespace

bool parseSkyrimTreeLodList(
    const std::vector<std::uint8_t>& bytes,
    std::vector<SkyrimTreeLodType>& outTypes,
    std::string& outError) {
    outTypes.clear();
    if (bytes.size() < 4u) {
        outError = "Skyrim tree LST is truncated before its type count";
        return false;
    }
    const std::uint32_t count = readU32(bytes.data());
    if (count == 0u || count > kMaxTreeTypes) {
        outError = "Skyrim tree LST has invalid type count " + std::to_string(count);
        return false;
    }
    if (bytes.size() != 4u + static_cast<std::size_t>(count) * kListRecordBytes) {
        outError = "Skyrim tree LST byte length does not match its type count";
        return false;
    }
    std::unordered_map<std::uint32_t, bool> seen;
    outTypes.reserve(count);
    for (std::uint32_t i = 0u; i < count; ++i) {
        const std::uint8_t* record = bytes.data() + 4u + i * kListRecordBytes;
        SkyrimTreeLodType type;
        type.index = readU32(record);
        type.width = readF32(record + 4u);
        type.height = readF32(record + 8u);
        type.uvMin[0] = readF32(record + 12u);
        type.uvMin[1] = readF32(record + 16u);
        type.uvMax[0] = readF32(record + 20u);
        type.uvMax[1] = readF32(record + 24u);
        const bool finite = std::isfinite(type.width) && std::isfinite(type.height) &&
            std::isfinite(type.uvMin[0]) && std::isfinite(type.uvMin[1]) &&
            std::isfinite(type.uvMax[0]) && std::isfinite(type.uvMax[1]);
        if (!finite || type.width <= 0.0f || type.height <= 0.0f ||
            // Retail Tamriel expands a few rectangles by less than one atlas
            // texel (-0.00054..1.00098) to hide mip seams. Accept that authored
            // guard band, then clamp because Vulkan normalized sampling must
            // not wrap it onto the opposite side of the atlas.
            type.uvMin[0] < -0.01f || type.uvMin[1] < -0.01f ||
            type.uvMax[0] > 1.01f || type.uvMax[1] > 1.01f ||
            type.uvMin[0] >= type.uvMax[0] || type.uvMin[1] >= type.uvMax[1]) {
            outError = "Skyrim tree LST type " + std::to_string(type.index) +
                " has invalid dimensions or atlas coordinates";
            return false;
        }
        if (seen.contains(type.index)) {
            outError = "Skyrim tree LST repeats type index " + std::to_string(type.index);
            return false;
        }
        type.uvMin[0] = std::clamp(type.uvMin[0], 0.0f, 1.0f);
        type.uvMin[1] = std::clamp(type.uvMin[1], 0.0f, 1.0f);
        type.uvMax[0] = std::clamp(type.uvMax[0], 0.0f, 1.0f);
        type.uvMax[1] = std::clamp(type.uvMax[1], 0.0f, 1.0f);
        seen.emplace(type.index, true);
        outTypes.push_back(type);
    }
    return true;
}

bool parseSkyrimTreeLodTile(
    const std::vector<std::uint8_t>& bytes,
    std::vector<SkyrimTreeLodInstance>& outInstances,
    std::string& outError) {
    outInstances.clear();
    if (bytes.size() < 4u) {
        outError = "Skyrim BTT is truncated before its group count";
        return false;
    }
    const std::uint32_t groupCount = readU32(bytes.data());
    if (groupCount > kMaxTileGroups) {
        outError = "Skyrim BTT has invalid group count " + std::to_string(groupCount);
        return false;
    }
    std::size_t cursor = 4u;
    for (std::uint32_t group = 0u; group < groupCount; ++group) {
        if (cursor + 8u > bytes.size()) {
            outError = "Skyrim BTT is truncated in group header " + std::to_string(group);
            return false;
        }
        const std::uint32_t typeIndex = readU32(bytes.data() + cursor);
        const std::uint32_t count = readU32(bytes.data() + cursor + 4u);
        cursor += 8u;
        if (count > kMaxTileInstances ||
            count > (bytes.size() - cursor) / kInstanceBytes) {
            outError = "Skyrim BTT group " + std::to_string(group) +
                " has an invalid instance count";
            return false;
        }
        if (outInstances.size() + count > kMaxTileInstances) {
            outError = "Skyrim BTT exceeds the bounded instance budget";
            return false;
        }
        for (std::uint32_t i = 0u; i < count; ++i) {
            const std::uint8_t* record = bytes.data() + cursor;
            SkyrimTreeLodInstance tree;
            tree.typeIndex = typeIndex;
            tree.position[0] = readF32(record);
            tree.position[1] = readF32(record + 4u);
            tree.position[2] = readF32(record + 8u);
            tree.rotation = readF32(record + 12u);
            tree.scale = readF32(record + 16u);
            tree.referenceId = readU32(record + 20u);
            if (!std::isfinite(tree.position[0]) || !std::isfinite(tree.position[1]) ||
                !std::isfinite(tree.position[2]) || !std::isfinite(tree.rotation) ||
                !std::isfinite(tree.scale) || tree.scale <= 0.0f || tree.scale > 100.0f) {
                outError = "Skyrim BTT contains an invalid tree transform";
                return false;
            }
            outInstances.push_back(tree);
            cursor += kInstanceBytes;
        }
    }
    if (cursor != bytes.size()) {
        outError = "Skyrim BTT has trailing bytes after its final group";
        return false;
    }
    return true;
}

bool appendSkyrimTreeLod(
    const LandLodByteResolver& resolveMesh,
    const LandLodByteResolver& resolveTexture,
    const std::string& worldspaceEditorId,
    std::int32_t cellX0, std::int32_t cellZ0,
    std::int32_t cellX1, std::int32_t cellZ1,
    const SkyrimTreeDetailedCellPredicate& detailedCellResident,
    ImportedScene& out,
    SkyrimTreeLodStats& outStats,
    std::string& outError) {
    outStats = {};
    const std::string world = lowerCopy(worldspaceEditorId);
    std::vector<std::uint8_t> listBytes;
    const std::string listPath = "terrain\\" + world + "\\trees\\" + world + ".lst";
    if (!resolveMesh(listPath, listBytes)) {
        outError = "missing Skyrim tree type list " + listPath;
        return false;
    }
    std::vector<SkyrimTreeLodType> types;
    if (!parseSkyrimTreeLodList(listBytes, types, outError)) {
        outError = listPath + ": " + outError;
        return false;
    }
    std::unordered_map<std::uint32_t, const SkyrimTreeLodType*> typeByIndex;
    for (const SkyrimTreeLodType& type : types) typeByIndex.emplace(type.index, &type);

    const std::string atlasPath =
        "terrain\\" + world + "\\trees\\" + world + "treelod.dds";
    std::vector<std::uint8_t> atlasBytes;
    ImportedSceneTexture atlas;
    if (!resolveTexture(atlasPath, atlasBytes) ||
        !loadDdsFromMemory(atlasBytes.data(), atlasBytes.size(), atlas)) {
        outError = "missing or unreadable Skyrim tree atlas " + atlasPath;
        return false;
    }
    atlas.sourcePath = atlasPath;
    const std::uint32_t atlasIndex = static_cast<std::uint32_t>(out.textures.size());
    out.textures.push_back(std::move(atlas));

    ImportedSceneMesh mesh;
    mesh.name = "skyrim_tree_lod_" + world;

    constexpr std::int32_t kTileCells = kLandLodBlockCells;
    const std::int32_t x0 = landLodTileOrigin(std::min(cellX0, cellX1), kTileCells);
    const std::int32_t z0 = landLodTileOrigin(std::min(cellZ0, cellZ1), kTileCells);
    const std::int32_t x1 = landLodTileOrigin(std::max(cellX0, cellX1), kTileCells);
    const std::int32_t z1 = landLodTileOrigin(std::max(cellZ0, cellZ1), kTileCells);
    for (std::int32_t tz = z0; tz <= z1; tz += kTileCells) {
        for (std::int32_t tx = x0; tx <= x1; tx += kTileCells) {
            const std::string tilePath = "terrain\\" + world + "\\trees\\" + world +
                ".4." + std::to_string(tx) + "." + std::to_string(tz) + ".btt";
            std::vector<std::uint8_t> tileBytes;
            if (!resolveMesh(tilePath, tileBytes)) {
                ++outStats.tilesMissing;
                continue;
            }
            ++outStats.tilesResolved;
            std::vector<SkyrimTreeLodInstance> trees;
            std::string tileError;
            if (!parseSkyrimTreeLodTile(tileBytes, trees, tileError)) {
                outError = tilePath + ": " + tileError;
                return false;
            }
            ++outStats.tilesParsed;
            ImportedSceneMeshPart tilePart;
            tilePart.firstIndex = static_cast<std::uint32_t>(mesh.indices.size());
            tilePart.textureIndex = atlasIndex;
            tilePart.alphaTest = true;
            tilePart.twoSided = true;
            tilePart.alphaThreshold = 96u;
            for (const SkyrimTreeLodInstance& tree : trees) {
                const std::int32_t treeCellX = static_cast<std::int32_t>(
                    std::floor(tree.position[0] / kExteriorCellSize));
                const std::int32_t treeCellZ = static_cast<std::int32_t>(
                    std::floor(tree.position[1] / kExteriorCellSize));
                if (detailedCellResident && detailedCellResident(treeCellX, treeCellZ)) {
                    ++outStats.instancesTrimmed;
                    continue;
                }
                const auto typeIt = typeByIndex.find(tree.typeIndex);
                if (typeIt == typeByIndex.end()) {
                    outError = tilePath + ": references absent tree type " +
                        std::to_string(tree.typeIndex);
                    return false;
                }
                const float phase = static_cast<float>(
                    (tree.referenceId * 1103515245u + 12345u) & 0xffffu) / 65535.0f;
                // A two-plane cross exposes its construction whenever the
                // camera approaches either plane edge-on. Three radial planes
                // give the retail atlas a six-sided crown silhouette; at the
                // BTT handoff distance it reads as volume without inventing
                // textures or paying for full NIF branch geometry.
                constexpr float kThirdTurn = 1.0471975512f;
                appendCard(mesh, *typeIt->second, tree, tree.rotation, phase);
                appendCard(mesh, *typeIt->second, tree,
                           tree.rotation + kThirdTurn, phase);
                appendCard(mesh, *typeIt->second, tree,
                           tree.rotation + (2.0f * kThirdTurn), phase);
                ++outStats.instances;
            }
            tilePart.indexCount = static_cast<std::uint32_t>(mesh.indices.size()) -
                tilePart.firstIndex;
            if (tilePart.indexCount != 0u) mesh.parts.push_back(tilePart);
        }
    }
    if (outStats.tilesParsed == 0u) {
        outError = "no Skyrim BTT tiles found for worldspace \"" +
            worldspaceEditorId + "\"";
        return false;
    }
    if (!mesh.indices.empty()) {
        const std::uint32_t meshIndex = static_cast<std::uint32_t>(out.meshes.size());
        out.meshes.push_back(std::move(mesh));
        ImportedSceneInstance instance;
        instance.meshIndex = meshIndex;
        instance.transform[0] = 1.0f;
        instance.transform[5] = 1.0f;
        instance.transform[10] = 1.0f;
        instance.transform[15] = 1.0f;
        out.instances.push_back(instance);
    }
    outStats.triangles = outStats.instances * 6u;
    outStats.textures = 1u;
    out.alphaFlagsAuthored = true;
    return true;
}

}  // namespace odai::importer::fnv
