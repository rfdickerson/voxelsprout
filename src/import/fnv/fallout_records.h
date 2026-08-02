#pragma once

// Typed extraction of the record types needed to cook a Fallout: New Vegas
// exterior worldspace region or interior cell into an ImportedScene: TES4
// (plugin header/masters), STAT (static base model), CELL, WRLD, LAND
// (heightmap), and REFR (placed static instances).
//
// This intentionally covers a narrow slice of the ~700 Fallout record types —
// enough for terrain + static-mesh cooking, matching the scope of this
// project's (Windows-only, not in this repo) Morrowind balmora cooker. It is
// not a general-purpose ESM editor.
//
// Coordinate/scale notes carried over from public Gamebryo landscape
// documentation, NOT validated against a real .esm in this environment (no
// Fallout: New Vegas Data Files available here — see README for the local
// path convention once a user tests this against real data):
//   - Exterior cells are 4096 world units square.
//   - LAND VHGT posts form a 33x33 grid at 128-unit spacing (32 gaps * 128 = 4096).
//   - VHGT height deltas are accumulated in a fixed unit scale (kLandHeightScale);
//     this constant should be double-checked against a known in-game elevation
//     the first time this cooker runs against real assets.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "import/fnv/esm_reader.h"

namespace odai::importer::fnv {

// Height-post grid dimensions for one LAND record (one exterior cell).
constexpr int kLandGridSize = 33;
constexpr int kLandVertexCount = kLandGridSize * kLandGridSize;
constexpr float kLandPostSpacing = 128.0f;
// VHGT delta-byte -> world-unit height scale. See the file-level comment:
// this is a best-effort constant from public Gamebryo landscape format
// documentation, not verified against real Fallout: New Vegas data.
constexpr float kLandHeightScale = 8.0f;
constexpr float kExteriorCellSize = kLandPostSpacing * static_cast<float>(kLandGridSize - 1);  // 4096

struct FalloutStaticRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    std::string modelPath;  // relative to Data\Meshes, backslashes as stored in the plugin
};

struct FalloutPlacedReference {
    std::uint32_t formId = 0;
    std::uint32_t baseFormId = 0;  // NAME: the STAT (or other base record) this instance places
    float position[3] = {};        // DATA, world units
    float rotationRadians[3] = {};  // DATA
    float scale = 1.0f;             // XSCL, defaults to 1 when absent
};

struct FalloutLandRecord {
    std::uint32_t cellFormId = 0;
    bool hasHeights = false;
    float heights[kLandVertexCount] = {};   // row-major [row * 33 + col], world Z units
    bool hasNormals = false;
    float normals[kLandVertexCount * 3] = {};  // row-major, normalized
    // Base texture per quadrant (0=SW,1=SE... layout mirrors BTXT's own
    // quadrant index), 0 when the quadrant has no explicit BTXT record.
    std::uint32_t quadrantBaseTextureFormId[4] = {};
};

struct FalloutCellRecord {
    std::uint32_t formId = 0;
    std::string editorId;
    bool isInterior = false;
    bool hasGridCoords = false;
    std::int32_t gridX = 0;
    std::int32_t gridZ = 0;
    std::uint32_t worldspaceFormId = 0;  // 0 for interior cells
    std::vector<FalloutPlacedReference> references;
    std::optional<FalloutLandRecord> land;
};

struct FalloutWorldspaceRecord {
    std::uint32_t formId = 0;
    std::string editorId;
};

// Everything extracted from one plugin pass. Populated by extractFalloutScene
// as a flat pass over the whole file — the caller is expected to filter down
// to the cells/worldspace it actually wants to cook.
struct FalloutSceneData {
    std::vector<FalloutStaticRecord> statics;         // keyed by index; look up via staticsByFormId
    std::vector<FalloutWorldspaceRecord> worldspaces;
    std::vector<FalloutCellRecord> cells;
};

// Runs a single forward pass over the plugin, extracting TES4/STAT/WRLD/CELL/
// LAND/REFR records. Cell attribution for LAND/REFR records uses simple
// "most recently seen CELL record" tracking rather than fully validating the
// GRUP hierarchy — correct for well-formed retail plugins.
bool extractFalloutScene(const std::filesystem::path& esmPath, FalloutSceneData& outScene, std::string& outError);

}  // namespace odai::importer::fnv
