#pragma once

#include "import/fnv/land_lod.h"
#include "import/imported_scene.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace odai::importer::fnv {

struct SkyrimTreeLodType {
    std::uint32_t index = 0u;
    float width = 0.0f;
    float height = 0.0f;
    float uvMin[2] = {};
    float uvMax[2] = {};
};

struct SkyrimTreeLodInstance {
    std::uint32_t typeIndex = 0u;
    float position[3] = {};
    float rotation = 0.0f;
    float scale = 1.0f;
    std::uint32_t referenceId = 0u;
};

struct SkyrimTreeLodStats {
    std::size_t tilesResolved = 0u;
    std::size_t tilesParsed = 0u;
    std::size_t tilesMissing = 0u;
    std::size_t instances = 0u;
    std::size_t instancesTrimmed = 0u;
    std::size_t triangles = 0u;
    std::size_t textures = 0u;
};

bool parseSkyrimTreeLodList(
    const std::vector<std::uint8_t>& bytes,
    std::vector<SkyrimTreeLodType>& outTypes,
    std::string& outError);

bool parseSkyrimTreeLodTile(
    const std::vector<std::uint8_t>& bytes,
    std::vector<SkyrimTreeLodInstance>& outInstances,
    std::string& outError);

using SkyrimTreeDetailedCellPredicate =
    std::function<bool(std::int32_t cellX, std::int32_t cellZ)>;

// Appends Skyrim's authored BTT tree cards for the inclusive raw-cell range.
// Instances whose detailed cell is resident are omitted so their full NIF
// counterparts own the near field without double trees or a cutout/NIF pop.
bool appendSkyrimTreeLod(
    const LandLodByteResolver& resolveMesh,
    const LandLodByteResolver& resolveTexture,
    const std::string& worldspaceEditorId,
    std::int32_t cellX0, std::int32_t cellZ0,
    std::int32_t cellX1, std::int32_t cellZ1,
    const SkyrimTreeDetailedCellPredicate& detailedCellResident,
    ImportedScene& out,
    SkyrimTreeLodStats& outStats,
    std::string& outError);

}  // namespace odai::importer::fnv
