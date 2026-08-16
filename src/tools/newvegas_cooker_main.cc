// Offline cooker: Fallout: New Vegas Data Files (.esm + .bsa + .nif) -> the
// engine's native ImportedScene .bin format, mirroring the scope of this
// project's (Windows-only, not in this repo) Morrowind balmora cooker.
//
// Usage:
//   odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin>
//       --cell <EditorID>
//   odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin>
//       --worldspace <EditorID> <gridX0> <gridZ0> <gridX1> <gridZ1>
//
// Scope cuts made explicitly (not oversights — see the module headers under
// src/import/fnv/ for the reasoning behind each):
//   - No texture extraction. Static meshes render via the engine's existing
//     per-model hashed-color fallback (same path buildImportedScenePackedRenderData
//     already uses for any mesh with no parts); terrain renders via its
//     existing height-based color fallback. Wiring DDS textures through
//     BSShaderTextureSet is a follow-up, not attempted here (see nif_scene.h).
//   - Bethesda's coordinate space is Z-up; this engine (like its Morrowind
//     import path) is Y-up. Positions and rotations are converted with a
//     Y<->Z axis swap below. The exact Euler rotation order Bethesda encodes
//     in REFR DATA is not verified against a real plugin in this environment
//     — double check placed-object orientation against a known in-game
//     reference the first time this runs against real Data Files.

#include "core/frame_profiler.h"
#include "import/dds.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/bsa_archive.h"
#include "import/fnv/cell_builder.h"
#include "import/fnv/fallout_records.h"
#include "import/fnv/land_lod.h"
#include "import/fnv/nif_scene.h"
#include "import/imported_scene.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using odai::importer::ImportedScene;
using odai::importer::ImportedSceneInstance;
using odai::importer::ImportedSceneLandscapeCell;
using odai::importer::ImportedSceneMesh;
using odai::importer::ImportedSceneMeshPart;
using odai::importer::ImportedSceneVertex;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

// Bethesda's Gamebryo space is right-handed Z-up (X east, Y north, Z up);
// this engine is Y-up. The conversion is a -90 degree rotation about X:
//
//     (x, y, z)_bethesda -> (x, z, -y)_engine
//
// The negation matters. Plain (x, z, y) — what this used to do — swaps two
// axes, which is a reflection (determinant -1), not a rotation: it mirrors the
// entire world and inverts every triangle's winding, so back faces get drawn
// and front faces get culled. Determinant of the map below is +1.
//
// This same matrix is applied to model-space NIF vertices via the instance
// transform (see makeEngineInstanceRotation), because Gamebryo *model* space
// is Z-up too — the two must use one convention or every object lands rotated
// 90 degrees relative to the world it sits in.
Vec3 bethesdaToEngine(float x, float y, float z) {
    return Vec3{x, z, -y};
}

struct Mat3 {
    float m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

Mat3 rotationX(float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    Mat3 out{};
    out.m[0] = 1; out.m[1] = 0; out.m[2] = 0;
    out.m[3] = 0; out.m[4] = c; out.m[5] = -s;
    out.m[6] = 0; out.m[7] = s; out.m[8] = c;
    return out;
}
Mat3 rotationY(float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    Mat3 out{};
    out.m[0] = c;  out.m[1] = 0; out.m[2] = s;
    out.m[3] = 0;  out.m[4] = 1; out.m[5] = 0;
    out.m[6] = -s; out.m[7] = 0; out.m[8] = c;
    return out;
}
Mat3 rotationZ(float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    Mat3 out{};
    out.m[0] = c; out.m[1] = -s; out.m[2] = 0;
    out.m[3] = s; out.m[4] = c;  out.m[5] = 0;
    out.m[6] = 0; out.m[7] = 0;  out.m[8] = 1;
    return out;
}
Mat3 multiply(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum += a.m[(r * 3) + k] * b.m[(k * 3) + c];
            }
            out.m[(r * 3) + c] = sum;
        }
    }
    return out;
}

// Composes the rotation an instance's transform needs, given that its mesh
// vertices stay in raw Bethesda model space.
//
// The full chain for a vertex is engine = M * (T * R * S) * v_model, so the
// instance's 3x3 is M * R (and its translation is M * t, via
// bethesdaToEngine). Note the single M: this is NOT the similarity transform
// M * R * transpose(M), which is what you would want if the vertices had
// already been converted to engine space. Mixing the two — a similarity
// transform applied to un-converted model-space vertices — is what previously
// left every placed object rotated 90 degrees about X and mirrored.
//
//   M = [1  0  0]      so M * R takes rows (R0, R2, -R1).
//       [0  0  1]
//       [0 -1  0]
Mat3 makeEngineInstanceRotation(const Mat3& beth) {
    Mat3 out{};
    for (int c = 0; c < 3; ++c) {
        out.m[(0 * 3) + c] = beth.m[(0 * 3) + c];
        out.m[(1 * 3) + c] = beth.m[(2 * 3) + c];
        out.m[(2 * 3) + c] = -beth.m[(1 * 3) + c];
    }
    return out;
}

// REFR rotation order: Bethesda applies X, then Y, then Z (R = Rz*Ry*Rx),
// with the angles used as stored (not negated).
//
// The angle sign is settled: `odai_newvegas_probe --rotations FalloutNV.esm
// GSDocMitchellHouse` scores every candidate convention by how much the cell's
// modular pieces interpenetrate once placed, and positive angles beat negated
// ones by 1.6x (5.9e7 vs 9.6e7). The order is not separated by that test —
// ZYX, YZX and ZXY land within 0.5% of each other — so this keeps ZYX, which
// is both the documented convention and the best of the tied group. If a
// specific asset ever looks wrong about a diagonal axis, that near-tie is the
// place to look.
Mat3 eulerToMatrixBethesdaOrder(float rx, float ry, float rz) {
    return multiply(rotationZ(rz), multiply(rotationY(ry), rotationX(rx)));
}

void writeTransform(
    ImportedSceneInstance& instance, const Vec3& translation, const Mat3& rotation, float scale
) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            instance.transform[(r * 4) + c] = rotation.m[(r * 3) + c] * scale;
        }
    }
    instance.transform[3] = translation.x;
    instance.transform[7] = translation.y;
    instance.transform[11] = translation.z;
    instance.transform[15] = 1.0f;
}

std::string toLowerAsciiCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

// Canonical form of a texture path: backslashes, and the "textures\" prefix
// present exactly once. Both are needed because the two sources disagree —
// NIF BSShaderTextureSet paths carry the prefix, LTEX diffuse paths do not.
//
// Used for the dedup cache key AND the archive lookup, deliberately. Keying
// the cache on the raw path while the lookup added a prefix meant the same
// file stored under both spellings counted as two textures: 5 byte-identical
// duplicates in a 441-cell cook, each burning a bindless slot.
std::string normalizeTexturePath(const std::string& path);

std::string normalizeModelPath(std::string path) {
    for (char& c : path) {
        if (c == '/') {
            c = '\\';
        }
    }
    return path;
}

std::string normalizeTexturePath(const std::string& path) {
    std::string normalized = normalizeModelPath(path);
    if (toLowerAsciiCopy(normalized).rfind("textures\\", 0) != 0) {
        normalized = "textures\\" + normalized;
    }
    return normalized;
}

// Bethesda paths are backslash-separated regardless of host OS. On POSIX,
// std::filesystem::path::operator/ treats '\' as an ordinary filename
// character (not a separator), so appending a raw backslash-joined string
// via operator/ looks for one literal (and wrong) filename instead of
// walking subdirectories. Split explicitly and join each component so the
// resulting path is correct on every platform.
std::filesystem::path joinBackslashPath(std::filesystem::path base, const std::string& backslashPath) {
    std::string component;
    for (char c : backslashPath) {
        if (c == '\\' || c == '/') {
            if (!component.empty()) {
                base /= component;
                component.clear();
            }
        } else {
            component.push_back(c);
        }
    }
    if (!component.empty()) {
        base /= component;
    }
    return base;
}

// Resolves model/texture paths to bytes. The precedence and load-order rules
// live in import/fnv/asset_source.h so the runtime streamer uses the same ones
// -- they were arrived at by measurement and two copies would drift.
class AssetResolver {
public:
    explicit AssetResolver(std::filesystem::path dataFilesPath) {
        // Exclude audio-only archives: "Fallout - Sound.bsa" and
        // "Fallout - Voices1.bsa" hold 111982 files between them and cannot
        // contain a mesh or texture, and indexing them was ~85% of runtime.
        constexpr std::uint32_t kWanted = ~(
            odai::importer::fnv::kBsaContentSounds | odai::importer::fnv::kBsaContentVoices);
        if (!m_source.open(dataFilesPath, kWanted)) {
            std::cerr << "warning: could not read Data Files directory " << dataFilesPath << "\n";
        }
        for (const std::string& warning : m_source.warnings()) {
            std::cerr << "warning: " << warning << "\n";
        }
    }

    bool resolveTexture(const std::string& texturePath, std::vector<std::uint8_t>& outBytes) {
        std::string error;
        return m_source.resolveTexture(texturePath, outBytes, error);
    }

    bool resolveMesh(const std::string& modelPath, std::vector<std::uint8_t>& outBytes) {
        std::string error;
        return m_source.resolveMesh(modelPath, outBytes, error);
    }

private:
    odai::importer::fnv::FalloutAssetSource m_source;
};

std::string formIdHex(std::uint32_t formId) {
    std::ostringstream out;
    out << std::hex << std::setw(8) << std::setfill('0') << formId;
    return out.str();
}

void printUsage() {
    std::cerr <<
        "Usage:\n"
        "  odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin> --cell <EditorID>\n"
        "  odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin> --worldspace <EditorID> "
        "<gridX0> <gridZ0> <gridX1> <gridZ1>\n"
        // Distant LOD. Two sets, not one -- see cookLodTier. The plugin name is
        // ignored by these: LOD tiles come from the archives, not the ESM.
        "  odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin> --lod <WorldspaceID> "
        "<cellX0> <cellZ0> <cellX1> <cellZ1> [tier: 4|8|16|32, default 4]\n"
        "  odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin> --lodobjects "
        "<WorldspaceID> <cellX0> <cellZ0> <cellX1> <cellZ1>   (merged distant buildings, level4 only)\n";
}

// Appends one exterior cell's LAND heightmap as a 32x32-quad grid, all
// vertices already in final engine world space (terrain is not instanced —
// see the cooker's file-header note on how buildImportedScenePackedRenderData
// treats scene.meshes[0] when named "terrain").
// Sentinel meaning "no texture, shade from vertex color". NOTE this is NOT
// ImportedSceneMeshPart's default, which is 0 — a real, valid texture index.
// Terrain relied on that default plus an empty texture table; once the cook
// started emitting textures, index 0 became a real mesh texture and every
// terrain post sampled one corner texel of it, which is why the ground went
// black. Always set this explicitly.
constexpr std::uint32_t kNoTextureIndex = 0xffffffffu;

// True when the shape is a flat sheet: its thinnest axis is negligible compared
// to its own footprint.
//
// The test is RELATIVE, not exact. Exact equality was the obvious choice and it
// was wrong -- NVPoster10's decal quad spans y = -0.399997 to -0.399992, flat to
// any purpose that matters but not bit-identical, so it slipped through and kept
// rendering as a grey sheet. An absolute epsilon is no better: it would mean
// something different for a doorframe than for a building.
//
// A ratio separates the two cases cleanly. Decal quads come out around 1e-7;
// genuinely thin real geometry does not come close -- a 20-unit-thick wall
// panel 600 units across is 3.3e-2, five orders of magnitude away.
bool shapeIsPlanar(const odai::importer::fnv::NifShape& shape) {
    if (shape.positions.size() < 9u) {
        return false;  // fewer than 3 vertices: not a surface worth judging
    }
    float boundsMin[3] = {shape.positions[0], shape.positions[1], shape.positions[2]};
    float boundsMax[3] = {shape.positions[0], shape.positions[1], shape.positions[2]};
    for (std::size_t v = 0; (v * 3u) + 2u < shape.positions.size(); ++v) {
        for (int axis = 0; axis < 3; ++axis) {
            const float value = shape.positions[(v * 3u) + static_cast<std::size_t>(axis)];
            boundsMin[axis] = std::min(boundsMin[axis], value);
            boundsMax[axis] = std::max(boundsMax[axis], value);
        }
    }
    float thinnest = std::numeric_limits<float>::max();
    float largest = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const float extent = boundsMax[axis] - boundsMin[axis];
        thinnest = std::min(thinnest, extent);
        largest = std::max(largest, extent);
    }
    if (largest <= 0.0f) {
        return false;  // degenerate point/line, not a sheet
    }
    constexpr float kPlanarRatio = 1e-4f;
    return (thinnest / largest) < kPlanarRatio;
}

void appendTerrainCell(
    ImportedSceneMesh& terrainMesh,
    const odai::importer::fnv::FalloutCellRecord& cell,
    const std::function<std::uint32_t(std::uint32_t)>& resolveLandTexture,
    const std::function<std::uint32_t(std::uint32_t)>& resolveLandTextureExact,
    std::size_t& outDroppedLayerCount
) {
    if (cell.land == nullptr) {
        return;
    }
    const odai::importer::fnv::FalloutLandRecord& land = *cell.land;
    using odai::importer::fnv::kExteriorCellSize;
    using odai::importer::fnv::kLandGridSize;
    using odai::importer::fnv::kLandPostSpacing;
    using odai::importer::fnv::kLandQuadrantGridSize;
    using odai::importer::fnv::kLandTextureTilesPerCell;

    const float cellOriginX = static_cast<float>(cell.gridX) * kExteriorCellSize;
    const float cellOriginZ = static_cast<float>(cell.gridZ) * kExteriorCellSize;

    // One vertex block per quadrant rather than one shared 33x33 block per cell.
    //
    // Layers are declared PER QUADRANT by ATXT, so every post in a quadrant
    // shares one layer stack and only the opacity varies across it. Choosing the
    // stack per vertex instead -- picking each post's own strongest three --
    // broke that: neighbouring posts selected different subsets, and because the
    // shader carries layer texture indices as `nointerpolation`, each triangle
    // shaded with its provoking vertex's set while the weights interpolated from
    // all three. Wherever the selection changed, the blend jumped instead of
    // ramping, which is what drew hard square patches across the terrain.
    //
    // Per-quadrant vertices cost 4*17*17 = 1156 posts per cell against 1089
    // shared (+6%), and buy the property that matters: within a quadrant the
    // layer set is constant, so only the interpolating weights vary.
    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        const int colBegin = ((quadrant & 1) != 0) ? (kLandGridSize - 1) / 2 : 0;
        const int rowBegin = ((quadrant & 2) != 0) ? (kLandGridSize - 1) / 2 : 0;

        // This quadrant's layer stack, chosen once. Selection is by PEAK opacity
        // across the quadrant so a layer that is strong anywhere survives the
        // budget, then restored to ATXT order because the shader's lerp chain is
        // not commutative.
        struct QuadrantLayer {
            const odai::importer::fnv::FalloutLandTextureLayer* layer = nullptr;
            std::uint32_t textureIndex = kNoTextureIndex;
            float peakOpacity = 0.0f;
        };
        std::vector<QuadrantLayer> quadrantLayers;
        for (const auto& layer : land.textureLayers) {
            if (layer.quadrant != static_cast<std::uint8_t>(quadrant)) {
                continue;
            }
            // Exact, not the dominant-texture fallback: that fallback gives an
            // untextured BASE something plausible. Substituting it for a layer
            // would paint the region's commonest ground on top of itself.
            const std::uint32_t textureIndex = resolveLandTextureExact(layer.textureFormId);
            if (textureIndex == kNoTextureIndex) {
                continue;
            }
            float peakOpacity = 0.0f;
            for (const float opacity : layer.opacity) {
                peakOpacity = std::max(peakOpacity, opacity);
            }
            if (peakOpacity <= 0.0f) {
                continue;
            }
            quadrantLayers.push_back(QuadrantLayer{&layer, textureIndex, peakOpacity});
        }
        if (quadrantLayers.size() > static_cast<std::size_t>(odai::importer::kImportedSceneMaxTerrainLayers)) {
            std::stable_sort(
                quadrantLayers.begin(), quadrantLayers.end(),
                [](const QuadrantLayer& a, const QuadrantLayer& b) {
                    return a.peakOpacity != b.peakOpacity ? (a.peakOpacity > b.peakOpacity)
                                                          : (a.textureIndex < b.textureIndex);
                });
            outDroppedLayerCount +=
                quadrantLayers.size() - static_cast<std::size_t>(odai::importer::kImportedSceneMaxTerrainLayers);
            quadrantLayers.resize(static_cast<std::size_t>(odai::importer::kImportedSceneMaxTerrainLayers));
        }
        std::stable_sort(
            quadrantLayers.begin(), quadrantLayers.end(),
            [](const QuadrantLayer& a, const QuadrantLayer& b) {
                return a.layer->layerIndex < b.layer->layerIndex;
            });

        const std::uint32_t quadrantBaseVertex = static_cast<std::uint32_t>(terrainMesh.vertices.size());
        for (int quadrantRow = 0; quadrantRow < kLandQuadrantGridSize; ++quadrantRow) {
            for (int quadrantCol = 0; quadrantCol < kLandQuadrantGridSize; ++quadrantCol) {
                const int row = rowBegin + quadrantRow;
                const int col = colBegin + quadrantCol;
                const int postIndex = (row * kLandGridSize) + col;
                const float bethesdaX = cellOriginX + (static_cast<float>(col) * kLandPostSpacing);
                const float bethesdaY = cellOriginZ + (static_cast<float>(row) * kLandPostSpacing);
                const float bethesdaZ = land.hasHeights ? land.heights[postIndex] : 0.0f;
                const Vec3 world = bethesdaToEngine(bethesdaX, bethesdaY, bethesdaZ);

                ImportedSceneVertex vertex{};
                vertex.position[0] = world.x;
                vertex.position[1] = world.y;
                vertex.position[2] = world.z;
                // UVs stay keyed to the CELL, not the quadrant, so the base
                // texture tiles continuously across a quadrant boundary instead
                // of restarting at every seam.
                vertex.uv[0] = (static_cast<float>(col) / static_cast<float>(kLandGridSize - 1)) *
                    kLandTextureTilesPerCell;
                vertex.uv[1] = (static_cast<float>(row) / static_cast<float>(kLandGridSize - 1)) *
                    kLandTextureTilesPerCell;
                if (land.hasNormals) {
                    const Vec3 normal = bethesdaToEngine(
                        land.normals[(postIndex * 3) + 0],
                        land.normals[(postIndex * 3) + 1],
                        land.normals[(postIndex * 3) + 2]);
                    vertex.normal[0] = normal.x;
                    vertex.normal[1] = normal.y;
                    vertex.normal[2] = normal.z;
                } else {
                    vertex.normal[1] = 1.0f;
                }
                // VCLR is a colour, not a direction: no basis change, unlike the
                // normals above. Cells without it keep the vertex's white
                // default, which leaves their texture untinted.
                if (land.hasColors) {
                    vertex.color[0] = land.colors[(postIndex * 3) + 0];
                    vertex.color[1] = land.colors[(postIndex * 3) + 1];
                    vertex.color[2] = land.colors[(postIndex * 3) + 2];
                }

                const int quadrantPost = (quadrantRow * kLandQuadrantGridSize) + quadrantCol;
                for (std::size_t slot = 0; slot < quadrantLayers.size(); ++slot) {
                    vertex.layerTextureIndex[slot] = quadrantLayers[slot].textureIndex;
                    vertex.layerWeight[slot] = quadrantLayers[slot].layer->opacity[quadrantPost];
                }
                terrainMesh.vertices.push_back(vertex);
            }
        }

        const std::uint32_t quadrantFirstIndex = static_cast<std::uint32_t>(terrainMesh.indices.size());
        for (int quadrantRow = 0; quadrantRow < kLandQuadrantGridSize - 1; ++quadrantRow) {
            for (int quadrantCol = 0; quadrantCol < kLandQuadrantGridSize - 1; ++quadrantCol) {
                const std::uint32_t i00 = quadrantBaseVertex +
                    static_cast<std::uint32_t>((quadrantRow * kLandQuadrantGridSize) + quadrantCol);
                const std::uint32_t i10 = i00 + 1u;
                const std::uint32_t i01 = i00 + static_cast<std::uint32_t>(kLandQuadrantGridSize);
                const std::uint32_t i11 = i01 + 1u;
                // Winding note: the grid is laid out so that increasing `col`
                // moves +X in engine space but increasing `row` moves -Z,
                // because bethesdaToEngine negates Y. That row reversal flips
                // the sense of the quad, so the indices are emitted in the order
                // that leaves the surface normal pointing +Y (up).
                //
                // Getting this backwards is not subtle to spot and easy to
                // misread as a lighting bug: the terrain vanishes when viewed
                // from above and is solid when viewed from underneath.
                terrainMesh.indices.push_back(i00);
                terrainMesh.indices.push_back(i11);
                terrainMesh.indices.push_back(i01);
                terrainMesh.indices.push_back(i00);
                terrainMesh.indices.push_back(i10);
                terrainMesh.indices.push_back(i11);
            }
        }
        const std::uint32_t quadrantIndexCount =
            static_cast<std::uint32_t>(terrainMesh.indices.size()) - quadrantFirstIndex;
        if (quadrantIndexCount == 0u) {
            continue;
        }
        const std::uint32_t textureIndex = resolveLandTexture(land.quadrantBaseTextureFormId[quadrant]);
        terrainMesh.parts.push_back(
            ImportedSceneMeshPart{quadrantFirstIndex, quadrantIndexCount, textureIndex, false});
    }
}

}  // namespace

namespace {

// One cook: extract, select cells, build, write. Factored out of main so the
// same path can produce an exterior region and then each interior a door in it
// leads to, without the caller reaching into any of it.
//
// Re-extracting per interior rather than sharing one pass is deliberate. The
// interior filter rejects every worldspace outright, so a second extraction
// costs ~65 ms against the ~250 ms a worldspace pass takes -- cheaper than
// keeping every interior cell's references resident through the exterior build
// just in case a door points at them.
int cookOne(
    const std::filesystem::path& dataFilesPath,
    const std::filesystem::path& pluginName,
    const std::filesystem::path& outputPath,
    const std::optional<std::string>& targetCellEditorId,
    const std::optional<std::string>& targetWorldspaceEditorId,
    std::int32_t gridX0,
    std::int32_t gridZ0,
    std::int32_t gridX1,
    std::int32_t gridZ1,
    std::vector<std::string>* outDoorTargetCells
) {
    const std::filesystem::path esmPath = dataFilesPath / pluginName;
    // Phase timings. This is a content tool whose cost is dominated by data
    // volume, so where the time goes should be visible without a profiler.
    odai::core::Stopwatch phaseTimer;
    float extractMs = 0.0f;
    float resolverMs = 0.0f;

    odai::importer::fnv::FalloutSceneData scene;
    std::string extractError;

    // Materialize only the cells this cook actually selects. Every other
    // cell's REFR and LAND records are rejected from the record-header
    // callback, so they are never decompressed or parsed — which for a single
    // interior cell skips all 29363 LAND records in the plugin.
    //
    // These predicates must stay in step with the selection loops below; they
    // answer the same question, just early enough to save the work.
    odai::importer::fnv::FalloutExtractFilter filter;
    if (targetCellEditorId.has_value()) {
        filter.wantCellContents = [&](const odai::importer::fnv::FalloutCellRecord& cell) {
            return cell.isInterior && cell.editorId == *targetCellEditorId;
        };
        // Interior cells never live under a worldspace, so the walk can seek
        // straight past every world-children group — which is most of the file.
        filter.wantWorldspace = [](std::uint32_t) { return false; };
    } else {
        filter.wantCellContents = [&](const odai::importer::fnv::FalloutCellRecord& cell) {
            if (cell.isInterior) {
                return false;
            }
            // The worldspace's PERSISTENT cell is where teleport doors live --
            // every door into an interior is a persistent reference, not a
            // per-cell temporary one. FalloutNV.esm stores it with XCLC (0,0)
            // rather than omitting the coordinates, so a region that does not
            // happen to contain the origin skipped it: that is why an exterior
            // cook reported 6307 references and zero teleports. Its contents
            // span the whole worldspace and are clipped by position below.
            if (!cell.hasGridCoords || (cell.gridX == 0 && cell.gridZ == 0)) {
                for (const auto& ws : scene.worldspaces) {
                    if (ws.editorId == *targetWorldspaceEditorId) {
                        return cell.worldspaceFormId == ws.formId;
                    }
                }
                return false;
            }
            if (cell.gridX < gridX0 || cell.gridX > gridX1 || cell.gridZ < gridZ0 || cell.gridZ > gridZ1) {
                return false;
            }
            // Resolved against what has been parsed so far, which is safe: a
            // worldspace's WRLD record always precedes its world-children group.
            for (const auto& ws : scene.worldspaces) {
                if (ws.editorId == *targetWorldspaceEditorId) {
                    return cell.worldspaceFormId == ws.formId;
                }
            }
            return false;
        };
        // Skip every other worldspace's children outright. Same lookup as
        // above, and safe for the same reason: a WRLD record precedes its
        // world-children group.
        filter.wantWorldspace = [&](std::uint32_t worldspaceFormId) {
            for (const auto& ws : scene.worldspaces) {
                if (ws.editorId == *targetWorldspaceEditorId) {
                    return worldspaceFormId == ws.formId;
                }
            }
            return false;
        };
    }

    if (!odai::importer::fnv::extractFalloutScene(esmPath, filter, scene, extractError)) {
        std::cerr << "Failed to extract plugin data: " << extractError << "\n";
        return 1;
    }
    extractMs = phaseTimer.lapMs();
    std::cout << "Extracted " << scene.statics.size() << " statics, " << scene.worldspaces.size()
              << " worldspaces, " << scene.cells.size() << " cells from " << esmPath << "\n";

    std::vector<const odai::importer::fnv::FalloutCellRecord*> selectedCells;
    if (targetCellEditorId.has_value()) {
        for (const auto& cell : scene.cells) {
            if (cell.isInterior && cell.editorId == *targetCellEditorId) {
                selectedCells.push_back(&cell);
            }
        }
    } else {
        std::uint32_t worldspaceFormId = 0;
        bool foundWorldspace = false;
        for (const auto& ws : scene.worldspaces) {
            if (ws.editorId == *targetWorldspaceEditorId) {
                worldspaceFormId = ws.formId;
                foundWorldspace = true;
                break;
            }
        }
        if (!foundWorldspace) {
            std::cerr << "Worldspace not found: " << *targetWorldspaceEditorId << "\n";
            return 1;
        }
        for (const auto& cell : scene.cells) {
            if (!cell.isInterior && cell.worldspaceFormId == worldspaceFormId && cell.hasGridCoords &&
                cell.gridX >= gridX0 && cell.gridX <= gridX1 && cell.gridZ >= gridZ0 && cell.gridZ <= gridZ1) {
                selectedCells.push_back(&cell);
            }
        }
    }
    if (selectedCells.empty()) {
        std::cerr << "No matching cells found for the requested selection.\n";
        return 1;
    }
    std::cout << "Cooking " << selectedCells.size() << " cell(s).\n";
    // An interior's lighting is the whole rig for the room -- see XCLL in
    // fallout_records.h -- so report it rather than letting a black interior be
    // a mystery.
    for (const auto* cell : selectedCells) {
        if (cell == nullptr || !cell->isInterior) {
            continue;
        }
        if (!cell->hasLighting) {
            std::cout << "  " << cell->editorId << ": interior with NO XCLL lighting\n";
            continue;
        }
        std::cout << "  " << cell->editorId << ": ambient ("
                  << static_cast<int>(cell->ambientColor[0] * 255.0f) << ","
                  << static_cast<int>(cell->ambientColor[1] * 255.0f) << ","
                  << static_cast<int>(cell->ambientColor[2] * 255.0f) << ") directional ("
                  << static_cast<int>(cell->directionalColor[0] * 255.0f) << ","
                  << static_cast<int>(cell->directionalColor[1] * 255.0f) << ","
                  << static_cast<int>(cell->directionalColor[2] * 255.0f) << ") fog ("
                  << static_cast<int>(cell->fogColor[0] * 255.0f) << ","
                  << static_cast<int>(cell->fogColor[1] * 255.0f) << ","
                  << static_cast<int>(cell->fogColor[2] * 255.0f) << ") near " << cell->fogNear
                  << " far " << cell->fogFar << "\n";
    }

    std::unordered_map<std::uint32_t, const odai::importer::fnv::FalloutStaticRecord*> staticsByFormId;
    for (const auto& stat : scene.statics) {
        staticsByFormId[stat.formId] = &stat;
    }

    const bool anyInterior = selectedCells.front()->isInterior;

    ImportedScene outScene;
    outScene.sourceTag = anyInterior ? "fnv_interior" : "fnv_exterior";

    // Terrain is built after the resolver/texture cache below exist, because
    // each quadrant needs its landscape texture resolved. Placeholder mesh 0
    // keeps the "meshes[0] is terrain" invariant that
    // buildImportedScenePackedRenderData relies on.
    const std::size_t terrainMeshIndex = outScene.meshes.size();
    if (!anyInterior) {
        ImportedSceneMesh terrainMesh;
        terrainMesh.name = "terrain";
        outScene.meshes.push_back(std::move(terrainMesh));
    }

    phaseTimer.restart();
    AssetResolver resolver(dataFilesPath);
    resolverMs = phaseTimer.lapMs();
    std::unordered_map<std::uint32_t, std::uint32_t> meshIndexByStaticFormId;  // formId -> outScene.meshes index
    std::unordered_set<std::uint32_t> failedStatics;
    std::size_t untexturedShapeCount = 0;
    std::size_t totalShapeCount = 0;
    std::size_t shapesWithNoTexturePath = 0;
    std::unordered_set<std::string> unresolvedTexturePaths;
    std::unordered_set<std::string> untexturedModelPaths;
    std::size_t shadowDecalShapesSkipped = 0;
    std::size_t extremeUvShapeCount = 0;
    std::size_t editorMarkerModelsSkipped = 0;
    std::size_t untexturedShapesGivenModelTexture = 0;
    std::unordered_set<std::string> extremeUvModelPaths;
    std::uint32_t skippedGeometryShapes = 0;
    std::uint32_t placedInstances = 0;

    // Texture table, deduplicated by lowercased virtual path so a wall texture
    // shared by fifty meshes is decoded and uploaded once.
    std::unordered_map<std::string, std::uint32_t> textureIndexByPath;
    std::unordered_set<std::string> failedTextures;
    // The renderer's bindless table is finite and silently truncates past its
    // limit, which would show up as a handful of arbitrary meshes losing their
    // textures. Track the count and say something instead.
    constexpr std::size_t kTextureBudget = 1000u;
    constexpr std::uint32_t maxTextureSize = 512u;
    bool warnedTextureBudget = false;

    auto resolveTextureIndex = [&](const std::string& texturePath) -> std::uint32_t {
        if (texturePath.empty()) {
            return kNoTextureIndex;
        }
        const std::string key = toLowerAsciiCopy(normalizeTexturePath(texturePath));
        if (const auto it = textureIndexByPath.find(key); it != textureIndexByPath.end()) {
            return it->second;
        }
        if (failedTextures.count(key) != 0u) {
            return kNoTextureIndex;
        }
        if (outScene.textures.size() >= kTextureBudget) {
            if (!warnedTextureBudget) {
                std::cerr << "warning: texture budget of " << kTextureBudget
                          << " reached; further textures will fall back to flat color. "
                             "Cook a smaller region.\n";
                warnedTextureBudget = true;
            }
            return kNoTextureIndex;
        }
        std::vector<std::uint8_t> ddsBytes;
        if (!resolver.resolveTexture(texturePath, ddsBytes)) {
            failedTextures.insert(key);
            return kNoTextureIndex;
        }
        odai::importer::ImportedSceneTexture texture;
        if (!odai::importer::loadDdsFromMemory(ddsBytes.data(), ddsBytes.size(), texture)) {
            failedTextures.insert(key);
            return kNoTextureIndex;
        }
        // Fallout ships 1024-square diffuse maps; a whole region of them at
        // full resolution is gigabytes of VRAM for detail no one sees from
        // outdoors. Drop the top mips — for block-compressed data that is a
        // pointer bump, not a resample.
        odai::importer::dropDdsMipLevels(texture, maxTextureSize);
        texture.sourcePath = key;
        const auto index = static_cast<std::uint32_t>(outScene.textures.size());
        outScene.textures.push_back(std::move(texture));
        textureIndexByPath.emplace(key, index);
        return index;
    };

    // Now that textures can be resolved, fill the terrain mesh reserved above.
    if (!anyInterior && terrainMeshIndex < outScene.meshes.size()) {
        std::unordered_map<std::uint32_t, const odai::importer::fnv::FalloutLandTextureRecord*> landTexturesByFormId;
        for (const auto& landTexture : scene.landTextures) {
            landTexturesByFormId[landTexture.formId] = &landTexture;
        }
        // A LAND quadrant with no BTXT is not "no texture" -- it means the base
        // layer is the worldspace default, which lives outside the LAND record.
        // Treating it as untextured left 44% of terrain vertices shading from
        // vertex colour, which is what made the ground read as fragmented
        // patchwork with the wrong colours: the packed fallback for terrain is a
        // synthetic height ramp (packedTerrainColor in imported_scene.cc), not
        // anything Fallout ever shaded with.
        //
        // Without parsing the worldspace default, the closest honest stand-in is
        // the texture the cooked region itself uses most: on a Mojave grid that
        // is the dominant desert ground, which is what those quadrants are.
        std::unordered_map<std::uint32_t, std::size_t> landTextureUseCount;
        for (const auto* cell : selectedCells) {
            if (cell->land == nullptr) {
                continue;
            }
            for (std::uint32_t quadrantFormId : cell->land->quadrantBaseTextureFormId) {
                if (quadrantFormId != 0u && landTexturesByFormId.count(quadrantFormId) != 0u) {
                    ++landTextureUseCount[quadrantFormId];
                }
            }
        }
        std::uint32_t dominantLandTextureFormId = 0u;
        std::size_t dominantUseCount = 0;
        for (const auto& [formId, count] : landTextureUseCount) {
            // Ties broken by formID so a given cook is reproducible; iteration
            // order of an unordered_map is not.
            if (count > dominantUseCount ||
                (count == dominantUseCount && formId < dominantLandTextureFormId)) {
                dominantLandTextureFormId = formId;
                dominantUseCount = count;
            }
        }

        auto resolveLandTextureExact = [&](std::uint32_t landTextureFormId) -> std::uint32_t {
            const auto it = landTexturesByFormId.find(landTextureFormId);
            if (it == landTexturesByFormId.end()) {
                return kNoTextureIndex;
            }
            return resolveTextureIndex(it->second->diffuseTexturePath);
        };
        const std::uint32_t fallbackLandTexture = dominantLandTextureFormId != 0u
            ? resolveLandTextureExact(dominantLandTextureFormId)
            : kNoTextureIndex;
        if (fallbackLandTexture != kNoTextureIndex) {
            const auto dominantIt = landTexturesByFormId.find(dominantLandTextureFormId);
            std::cout << "Untextured land quadrants fall back to the region's most common texture ("
                      << dominantUseCount << " uses): \""
                      << (dominantIt != landTexturesByFormId.end()
                              ? dominantIt->second->diffuseTexturePath
                              : std::string("<unknown>"))
                      << "\"\n";
        } else {
            std::cout << "warning: no usable land textures; untextured quadrants will shade from vertex colour.\n";
        }
        auto resolveLandTexture = [&](std::uint32_t landTextureFormId) -> std::uint32_t {
            const std::uint32_t exact = resolveLandTextureExact(landTextureFormId);
            return exact != kNoTextureIndex ? exact : fallbackLandTexture;
        };
        ImportedSceneMesh& terrainMesh = outScene.meshes[terrainMeshIndex];
        std::size_t droppedLayerCount = 0;
        std::size_t totalLayerCount = 0;
        std::size_t waterPatchCount = 0;
        for (const auto* cell : selectedCells) {
            if (cell->land != nullptr) {
                totalLayerCount += cell->land->textureLayers.size();
            }
            // Shared with the streaming path on purpose: this is the one piece
            // of the terrain build that is NOT duplicated here, so a cooked
            // coastline and a streamed one cannot disagree about where the
            // water is.
            if (odai::importer::fnv::appendCellWaterPatch(outScene, *cell)) {
                ++waterPatchCount;
            }
            appendTerrainCell(
                terrainMesh, *cell, resolveLandTexture, resolveLandTextureExact, droppedLayerCount);
        }
        if (waterPatchCount != 0) {
            std::cout << "Water: " << waterPatchCount << " cell(s) carry a visible water surface\n";
        }
        std::cout << "Terrain texture layers: " << totalLayerCount << " ATXT/VTXT layers across "
                  << selectedCells.size() << " cell(s)\n";
        if (droppedLayerCount != 0) {
            // Not silent: a post covered by more layers than a vertex can hold
            // loses the weakest ones, and that is a visible difference from the
            // game rather than a rounding detail.
            std::cout << "warning: dropped " << droppedLayerCount
                      << " per-vertex layer contribution(s) beyond the "
                      << odai::importer::kImportedSceneMaxTerrainLayers << "-layer budget\n";
        }

        // One landscapeCells entry per emitted terrain PART, not per cell —
        // terrain is split into four quadrant parts so each can carry its own
        // BTXT landscape texture, and each part becomes one draw.
        //
        // This is load-bearing, not bookkeeping. sourceLandscapeCellCount is
        // DERIVED: saveImportedScene persists landscapeCells.size() and
        // loadImportedScene overwrites sourceLandscapeCellCount from it. The
        // renderer requires terrain to occupy the leading [0, terrainDrawCount)
        // draws, and loadImportedScene re-runs buildImportedScenePageRanges to
        // enforce that — so if this count disagrees with the real terrain draw
        // count, terrain draws get sorted out of that range and shaded as
        // ordinary static geometry.
        outScene.landscapeCells.clear();
        outScene.landscapeCells.resize(terrainMesh.parts.size());
        outScene.sourceLandscapeCellCount = static_cast<std::uint32_t>(outScene.landscapeCells.size());
    }

    for (const auto* cell : selectedCells) {
        for (const auto& ref : cell->references) {
            if (failedStatics.count(ref.baseFormId) != 0u) {
                continue;
            }
            auto meshIt = meshIndexByStaticFormId.find(ref.baseFormId);
            if (meshIt == meshIndexByStaticFormId.end()) {
                const auto statIt = staticsByFormId.find(ref.baseFormId);
                if (statIt == staticsByFormId.end() || statIt->second->modelPath.empty()) {
                    failedStatics.insert(ref.baseFormId);
                    continue;
                }
                std::vector<std::uint8_t> nifBytes;
                if (!resolver.resolveMesh(statIt->second->modelPath, nifBytes)) {
                    std::cerr << "warning: could not resolve mesh " << statIt->second->modelPath << "\n";
                    failedStatics.insert(ref.baseFormId);
                    continue;
                }
                // Editor markers are level-design furniture, not world geometry:
                // the GECK draws them, the game does not. marker_radiation.nif is
                // one shape whose UVs are a single constant point, so it has no
                // sensible texture and never had one -- it was rendering as a
                // grey slab in mid-air.
                const std::string lowerModelPath = toLowerAsciiCopy(statIt->second->modelPath);
                const std::size_t lastSlash = lowerModelPath.find_last_of('\\');
                const std::string modelBaseName = lastSlash == std::string::npos
                    ? lowerModelPath
                    : lowerModelPath.substr(lastSlash + 1u);
                // "marker" with no underscore required: FNV ships both
                // marker_radiation.nif and markerxheading.nif. The root-level
                // check keeps the rule tight -- editor markers live at the top of
                // meshes\, so a model in a content directory whose name happens to
                // start with those six letters is not caught.
                const bool isRootLevelModel = lastSlash == std::string::npos;
                if ((isRootLevelModel && modelBaseName.rfind("marker", 0) == 0) ||
                    lowerModelPath.find("\\markers\\") != std::string::npos) {
                    ++editorMarkerModelsSkipped;
                    failedStatics.insert(ref.baseFormId);
                    continue;
                }

                odai::importer::fnv::NifModel nifModel;
                std::string nifError;
                if (!odai::importer::fnv::parseNifStaticMesh(nifBytes, nifModel, nifError) || nifModel.shapes.empty()) {
                    std::cerr << "warning: failed to parse NIF " << statIt->second->modelPath << ": " << nifError << "\n";
                    failedStatics.insert(ref.baseFormId);
                    continue;
                }
                odai::importer::fnv::applyNifBannerGravityRestPose(
                    statIt->second->modelPath, nifModel);
                skippedGeometryShapes += nifModel.skippedShapeCount;

                // A model's own most-used texture, for sub-shapes that carry no
                // diffuse of their own. Same reasoning as the land-quadrant
                // fallback: a 53-vertex piece of OTBldgDesCorner04 sitting among
                // six shapes that all use the building's wall texture wants that
                // texture, not a hash of the model path -- which now reads as
                // flat grey, since the slope tint applies to untextured surfaces.
                std::unordered_map<std::uint32_t, std::size_t> modelTextureUse;
                for (const auto& shape : nifModel.shapes) {
                    const std::uint32_t shapeTexture = resolveTextureIndex(shape.diffuseTexturePath);
                    if (shapeTexture != kNoTextureIndex) {
                        ++modelTextureUse[shapeTexture];
                    }
                }
                std::uint32_t modelDominantTexture = kNoTextureIndex;
                std::size_t modelDominantUse = 0;
                for (const auto& [textureIndex, useCount] : modelTextureUse) {
                    // Ties broken by index so a cook stays reproducible.
                    if (useCount > modelDominantUse ||
                        (useCount == modelDominantUse && textureIndex < modelDominantTexture)) {
                        modelDominantTexture = textureIndex;
                        modelDominantUse = useCount;
                    }
                }

                ImportedSceneMesh mesh;
                mesh.name = statIt->second->editorId.empty() ? statIt->second->modelPath : statIt->second->editorId;
                for (const auto& shape : nifModel.shapes) {
                    // Drop baked shadow decals.
                    //
                    // Bethesda models carry a flat quad at the base, spanning the
                    // model's own footprint, holding a pre-baked ground shadow
                    // that the game alpha-blends onto the terrain. It is
                    // recognisable without guessing at names: untextured (its
                    // shader property is not one that names a diffuse through a
                    // texture set) AND perfectly planar -- every vertex sharing
                    // one coordinate on some axis, which no real piece of
                    // building geometry does.
                    //
                    // Keeping them was the "duplicate object, one textured and
                    // one grey" artifact: with no texture they take the
                    // per-model hashed colour, and with no alpha blending on the
                    // static path they draw as an opaque grey sheet lying across
                    // the real geometry and the ground.
                    if (shape.diffuseTexturePath.empty() && shapeIsPlanar(shape)) {
                        ++shadowDecalShapesSkipped;
                        continue;
                    }
                    const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
                    const std::uint32_t partFirstIndex = static_cast<std::uint32_t>(mesh.indices.size());
                    for (std::size_t v = 0; v * 3u < shape.positions.size(); ++v) {
                        ImportedSceneVertex vertex{};
                        // Kept in raw Bethesda model space (Z-up) on purpose: the
                        // Z-up -> Y-up conversion is folded into the instance
                        // transform by makeEngineInstanceRotation, so it happens
                        // once per instance instead of once per vertex.
                        vertex.position[0] = shape.positions[v * 3u];
                        vertex.position[1] = shape.positions[(v * 3u) + 1];
                        vertex.position[2] = shape.positions[(v * 3u) + 2];
                        if (!shape.normals.empty()) {
                            vertex.normal[0] = shape.normals[v * 3u];
                            vertex.normal[1] = shape.normals[(v * 3u) + 1];
                            vertex.normal[2] = shape.normals[(v * 3u) + 2];
                        }
                        if ((v * 2u) + 1u < shape.uvs.size()) {
                            vertex.uv[0] = shape.uvs[v * 2u];
                            vertex.uv[1] = shape.uvs[(v * 2u) + 1u];
                        }
                        mesh.vertices.push_back(vertex);
                    }
                    for (const std::uint32_t index : shape.triangleIndices) {
                        mesh.indices.push_back(baseVertex + index);
                    }
                    // One part per shape. Without these the mesh has no parts
                    // at all and every surface falls back to the per-model
                    // hashed color, which is why cooked scenes used to look
                    // like flat pastel blocks.
                    const auto partIndexCount =
                        static_cast<std::uint32_t>(mesh.indices.size()) - partFirstIndex;
                    if (partIndexCount != 0u) {
                        ImportedSceneMeshPart part{};
                        part.firstIndex = partFirstIndex;
                        part.indexCount = partIndexCount;
                        part.textureIndex = resolveTextureIndex(shape.diffuseTexturePath);
                        if (part.textureIndex == kNoTextureIndex &&
                            modelDominantTexture != kNoTextureIndex) {
                            part.textureIndex = modelDominantTexture;
                            ++untexturedShapesGivenModelTexture;
                        }
                        part.alphaTest = shape.alphaTest;
                        // A shape with no diffuse texture is not an error the
                        // resolver reports -- it silently shades from the
                        // per-model hashed colour, which reads as flat grey-brown
                        // patches on an otherwise textured model. Count them so
                        // "half the rock is grey" is a number rather than a
                        // guess.
                        // Extreme UVs only matter if a triangle that actually
                        // covers pixels has a huge UV span across it: that is what
                        // drives the mip selector to the smallest level and makes
                        // the surface shade as one flat colour.
                        //
                        // Measured per TRIANGLE, not per shape. Retail meshes carry
                        // degenerate stitching vertices whose UVs are junk -- the
                        // same pair repeated exactly, interleaved with sane ones --
                        // and a zero-area triangle emits no fragments, so its UVs
                        // cannot smear anything. Reporting per shape counted those
                        // and cried wolf.
                        for (std::size_t tri = 0; (tri * 3u) + 2u < shape.triangleIndices.size(); ++tri) {
                            const std::uint32_t ia = shape.triangleIndices[tri * 3u];
                            const std::uint32_t ib = shape.triangleIndices[(tri * 3u) + 1u];
                            const std::uint32_t ic = shape.triangleIndices[(tri * 3u) + 2u];
                            if ((static_cast<std::size_t>(ic) * 3u) + 2u >= shape.positions.size() ||
                                (static_cast<std::size_t>(ic) * 2u) + 1u >= shape.uvs.size()) {
                                continue;
                            }
                            const auto position = [&](std::uint32_t vi, int axis) {
                                return shape.positions[(static_cast<std::size_t>(vi) * 3u) +
                                                       static_cast<std::size_t>(axis)];
                            };
                            const float e0[3] = {position(ib, 0) - position(ia, 0),
                                                 position(ib, 1) - position(ia, 1),
                                                 position(ib, 2) - position(ia, 2)};
                            const float e1[3] = {position(ic, 0) - position(ia, 0),
                                                 position(ic, 1) - position(ia, 1),
                                                 position(ic, 2) - position(ia, 2)};
                            const float cx = (e0[1] * e1[2]) - (e0[2] * e1[1]);
                            const float cy = (e0[2] * e1[0]) - (e0[0] * e1[2]);
                            const float cz = (e0[0] * e1[1]) - (e0[1] * e1[0]);
                            const float doubleArea = std::sqrt((cx * cx) + (cy * cy) + (cz * cz));
                            if (doubleArea <= 1e-3f) {
                                continue;  // degenerate: rasterizes to nothing
                            }
                            float uvMin[2] = {0.0f, 0.0f};
                            float uvMax[2] = {0.0f, 0.0f};
                            for (int corner = 0; corner < 3; ++corner) {
                                const std::uint32_t vi = shape.triangleIndices[(tri * 3u) +
                                                                              static_cast<std::size_t>(corner)];
                                for (int c = 0; c < 2; ++c) {
                                    const float value =
                                        shape.uvs[(static_cast<std::size_t>(vi) * 2u) +
                                                  static_cast<std::size_t>(c)];
                                    if (corner == 0) {
                                        uvMin[c] = value;
                                        uvMax[c] = value;
                                    } else {
                                        uvMin[c] = std::min(uvMin[c], value);
                                        uvMax[c] = std::max(uvMax[c], value);
                                    }
                                }
                            }
                            const float uvSpan =
                                std::max(uvMax[0] - uvMin[0], uvMax[1] - uvMin[1]);
                            if (uvSpan > 64.0f) {
                                ++extremeUvShapeCount;
                                extremeUvModelPaths.insert(
                                    toLowerAsciiCopy(statIt->second->modelPath) +
                                    "  (worst triangle UV span " +
                                    std::to_string(static_cast<int>(uvSpan)) + ")");
                                break;
                            }
                        }
                        if (part.textureIndex == kNoTextureIndex) {
                            ++untexturedShapeCount;
                            if (shape.diffuseTexturePath.empty()) {
                                ++shapesWithNoTexturePath;
                                untexturedModelPaths.insert(
                                    toLowerAsciiCopy(statIt->second->modelPath));
                            } else {
                                unresolvedTexturePaths.insert(
                                    toLowerAsciiCopy(shape.diffuseTexturePath));
                            }
                        }
                        ++totalShapeCount;
                        mesh.parts.push_back(part);
                    }
                }
                if (mesh.vertices.empty()) {
                    failedStatics.insert(ref.baseFormId);
                    continue;
                }
                const std::uint32_t meshIndex = static_cast<std::uint32_t>(outScene.meshes.size());
                outScene.meshes.push_back(std::move(mesh));
                meshIt = meshIndexByStaticFormId.emplace(ref.baseFormId, meshIndex).first;
            }

            ImportedSceneInstance instance;
            instance.meshIndex = meshIt->second;
            instance.sourceId = "refr_" + formIdHex(ref.formId);
            instance.modelPath = staticsByFormId.at(ref.baseFormId)->modelPath;
            const Vec3 worldPos = bethesdaToEngine(ref.position[0], ref.position[1], ref.position[2]);
            const Mat3 bethRotation = eulerToMatrixBethesdaOrder(
                ref.rotationRadians[0], ref.rotationRadians[1], ref.rotationRadians[2]);
            const Mat3 engineRotation = makeEngineInstanceRotation(bethRotation);
            writeTransform(instance, worldPos, engineRotation, ref.scale);
            outScene.instances.push_back(instance);
            ++placedInstances;
        }
    }

    if (skippedGeometryShapes != 0u) {
        std::cout << "warning: " << skippedGeometryShapes
                  << " mesh shape(s) had unreadable NiTriShapeData and were dropped (see nif_scene.h).\n";
    }
    std::cout << "Placed " << placedInstances << " instance(s) across " << outScene.meshes.size() << " mesh(es).\n";
    if (extremeUvShapeCount != 0) {
        std::cout << "Shapes whose visible triangles span >64 texture tiles (they shade flat): "
                  << extremeUvShapeCount << " across " << extremeUvModelPaths.size() << " model(s)\n";
    }
    {
        std::size_t shown = 0;
        for (const std::string& path : extremeUvModelPaths) {
            std::cout << "  " << path << "\n";
            if (++shown >= 12) {
                std::cout << "  ... and " << (extremeUvModelPaths.size() - shown) << " more\n";
                break;
            }
        }
    }
    std::cout << "Skipped " << editorMarkerModelsSkipped << " editor-marker model(s); "
              << untexturedShapesGivenModelTexture
              << " untextured shape(s) fell back to their model's own texture\n";
    std::cout << "Skipped " << shadowDecalShapesSkipped
              << " flat untextured shape(s) as baked shadow decals\n";
    std::cout << "Static shapes: " << totalShapeCount << ", untextured " << untexturedShapeCount
              << " (" << shapesWithNoTexturePath << " with no diffuse path, "
              << unresolvedTexturePaths.size() << " distinct path(s) that failed to resolve)\n";
    if (!untexturedModelPaths.empty()) {
        std::size_t shownModels = 0;
        for (const std::string& path : untexturedModelPaths) {
            std::cout << "  shape with no diffuse path in: " << path << "\n";
            if (++shownModels >= 12) {
                std::cout << "  ... and " << (untexturedModelPaths.size() - shownModels) << " more model(s)\n";
                break;
            }
        }
    }
    if (!unresolvedTexturePaths.empty()) {
        std::size_t shown = 0;
        for (const std::string& path : unresolvedTexturePaths) {
            std::cout << "  unresolved texture: " << path << "\n";
            if (++shown >= 10) {
                std::cout << "  ... and " << (unresolvedTexturePaths.size() - shown) << " more\n";
                break;
            }
        }
    }
    const float meshMs = phaseTimer.lapMs();

    // Same statement the streaming path makes in CellSceneBuilder::finish():
    // every part's alpha mode came off its NiAlphaProperty, so the texture-
    // content cutout guess must not run on load. Without it a cooked scene is
    // written at v24 with the flag clear, applyTextureAlphaCutoutFlags() runs
    // over it exactly as before, and `--scene` shows the ragged holes the
    // streamed path no longer has -- the two paths disagreeing about the same
    // geometry.
    outScene.alphaFlagsAuthored = true;
    odai::importer::buildImportedScenePackedRenderData(outScene);
    odai::importer::buildImportedScenePageRanges(outScene);
    const float packMs = phaseTimer.lapMs();

    // Teleport doors. XTEL names the door reference on the far side and gives
    // the arrival transform, but not which cell that reference lives in --
    // cellIndexByReferenceFormId, built from every REFR header during
    // extraction, is what resolves it.
    //
    // A door whose target is not an interior with an EditorID is dropped: the
    // file naming convention is keyed on that ID, so a target without one
    // cannot be cooked or found again.
    const bool targetCellIsInteriorCook = targetCellEditorId.has_value();
    std::unordered_set<std::string> doorTargetCellSet;
    // The persistent cell spans the whole worldspace, so its doors are clipped
    // to the cooked grid by position -- otherwise a region cook would drag in
    // every interior in the Mojave.
    std::vector<const odai::importer::fnv::FalloutCellRecord*> doorSourceCells = selectedCells;
    const float regionMinX = static_cast<float>(gridX0) * odai::importer::fnv::kExteriorCellSize;
    const float regionMaxX = static_cast<float>(gridX1 + 1) * odai::importer::fnv::kExteriorCellSize;
    const float regionMinY = static_cast<float>(gridZ0) * odai::importer::fnv::kExteriorCellSize;
    const float regionMaxY = static_cast<float>(gridZ1 + 1) * odai::importer::fnv::kExteriorCellSize;
    std::unordered_set<const odai::importer::fnv::FalloutCellRecord*> persistentSources;
    if (targetWorldspaceEditorId.has_value()) {
        for (const auto& cell : scene.cells) {
            const bool isOriginCell = cell.hasGridCoords && cell.gridX == 0 && cell.gridZ == 0;
            if (cell.isInterior || cell.references.empty()) {
                continue;
            }
            if ((!cell.hasGridCoords || isOriginCell) &&
                std::find(selectedCells.begin(), selectedCells.end(), &cell) == selectedCells.end()) {
                doorSourceCells.push_back(&cell);
                persistentSources.insert(&cell);
            }
        }
    }
    for (const auto* cell : doorSourceCells) {
        const bool clipToRegion = persistentSources.count(cell) != 0u;
        for (const auto& ref : cell->references) {
            if (clipToRegion &&
                (ref.position[0] < regionMinX || ref.position[0] > regionMaxX ||
                 ref.position[1] < regionMinY || ref.position[1] > regionMaxY)) {
                continue;
            }
            if (!ref.hasTeleport || ref.teleportTargetRefFormId == 0u) {
                continue;
            }
            const auto targetIt = scene.cellIndexByReferenceFormId.find(ref.teleportTargetRefFormId);
            const bool targetResolved = targetIt != scene.cellIndexByReferenceFormId.end() &&
                targetIt->second < scene.cells.size();
            if (!targetResolved) {
                // An interior cook walks every interior but skips worldspace
                // groups outright, so an unresolved target can only be a
                // reference in a worldspace -- which is the door back out. That
                // is the exit, and dropping it made every interior a one-way
                // trip. For an exterior cook there is no such inference: an
                // unresolved target there is genuinely unknown.
                if (!targetCellIsInteriorCook) {
                    continue;
                }
                odai::importer::ImportedSceneDoor exitDoor{};
                const Vec3 exitWorld = bethesdaToEngine(ref.position[0], ref.position[1], ref.position[2]);
                exitDoor.position[0] = exitWorld.x;
                exitDoor.position[1] = exitWorld.y;
                exitDoor.position[2] = exitWorld.z;
                const Vec3 exitArrival = bethesdaToEngine(
                    ref.teleportPosition[0], ref.teleportPosition[1], ref.teleportPosition[2]);
                exitDoor.arrivalPosition[0] = exitArrival.x;
                exitDoor.arrivalPosition[1] = exitArrival.y;
                exitDoor.arrivalPosition[2] = exitArrival.z;
                exitDoor.arrivalYawDegrees =
                    -ref.teleportRotationRadians[2] * (180.0f / 3.14159265358979323846f);
                outScene.doors.push_back(std::move(exitDoor));
                continue;
            }
            const auto& targetCell = scene.cells[targetIt->second];
            // An EXTERIOR target is the way back out. It gets an empty
            // targetCellEditorId, which the loader reads as "the exterior scene
            // this interior was cooked beside" -- exterior cells have no useful
            // EditorID to name a file with, and without this an interior would
            // be a one-way trip.
            if (targetCell.isInterior && targetCell.editorId.empty()) {
                continue;
            }
            odai::importer::ImportedSceneDoor door{};
            const Vec3 doorWorld = bethesdaToEngine(ref.position[0], ref.position[1], ref.position[2]);
            door.position[0] = doorWorld.x;
            door.position[1] = doorWorld.y;
            door.position[2] = doorWorld.z;
            const Vec3 arrivalWorld = bethesdaToEngine(
                ref.teleportPosition[0], ref.teleportPosition[1], ref.teleportPosition[2]);
            door.arrivalPosition[0] = arrivalWorld.x;
            door.arrivalPosition[1] = arrivalWorld.y;
            door.arrivalPosition[2] = arrivalWorld.z;
            // Bethesda's Z rotation is the compass heading; the engine camera's
            // yaw is measured in the XZ plane from +X, and bethesdaToEngine maps
            // Bethesda +Y (north) onto engine -Z. Negating converts between them.
            door.arrivalYawDegrees =
                -ref.teleportRotationRadians[2] * (180.0f / 3.14159265358979323846f);
            door.targetCellEditorId = targetCell.isInterior ? targetCell.editorId : std::string();
            outScene.doors.push_back(std::move(door));
            if (targetCell.isInterior) {
                doorTargetCellSet.insert(targetCell.editorId);
            }
        }
    }
    {
        std::size_t navMeshCount = 0, navTriangles = 0, navVertices = 0, navPortals = 0, navBorderEdges = 0;
        for (const auto* cell : selectedCells) {
            for (const auto& navMesh : cell->navMeshes) {
                ++navMeshCount;
                navVertices += navMesh.vertices.size() / 3u;
                navTriangles += navMesh.triangles.size();
                navPortals += navMesh.doorPortals.size();
                for (const auto& tri : navMesh.triangles) {
                    for (int e = 0; e < 3; ++e) {
                        if (tri.neighbour[e] == odai::importer::fnv::kNavMeshNoNeighbour) ++navBorderEdges;
                    }
                }
            }
        }
        if (navMeshCount != 0) {
            std::cout << "Navmesh: " << navMeshCount << " NAVM, " << navVertices << " verts, "
                      << navTriangles << " tris, " << navPortals << " door portal(s), "
                      << navBorderEdges << " border edges ("
                      << (navTriangles == 0 ? 0 : (navBorderEdges * 100) / (navTriangles * 3)) << "% of edges)\n";
        }
    }
    if (!outScene.doors.empty()) {
        std::cout << "Doors: " << outScene.doors.size() << " teleport(s) into "
                  << doorTargetCellSet.size() << " interior cell(s)\n";
    }
    if (outDoorTargetCells != nullptr) {
        outDoorTargetCells->assign(doorTargetCellSet.begin(), doorTargetCellSet.end());
        std::sort(outDoorTargetCells->begin(), outDoorTargetCells->end());
    }

    if (!odai::importer::saveImportedScene(outScene, outputPath)) {
        std::cerr << "Failed to save output scene: " << odai::importer::getImportedSceneLastError() << "\n";
        return 1;
    }
    const float writeMs = phaseTimer.lapMs();

    std::cout << "Wrote " << outputPath << "\n";
    std::cout << "Timings: extract " << extractMs << " ms, archives " << resolverMs << " ms, meshes " << meshMs
              << " ms, pack " << packMs << " ms, write " << writeMs << " ms (total "
              << (extractMs + resolverMs + meshMs + packMs + writeMs) << " ms)\n";
    return 0;
}

}  // namespace


// Cooks one distant-landscape LOD tier for a worldspace into one scene.
//
// Layout and coordinate space are MEASURED, not assumed -- see the
// kLandLodTierCellCounts comment in fallout_records.h, which is the authority.
// The two things that matter here:
//
//   * There are TWO sets, terrain and objects, and which one you get is a
//     parameter rather than something inferred from the tier. This function
//     used to derive the directory from `tierCells == 4`, so `--lod ... 4`
//     cooked distant BUILDINGS while tiers 8/16/32 cooked terrain -- one flag
//     silently meaning two different things, and no way to ask for terrain
//     level4 (the finest terrain tier, 1024 tiles for WastelandNV) at all.
//   * Terrain is a real four-tier pyramid; the object set is level4 only.
//
// The tiles' vertices are already in WORLD units, unlike static models which
// are in model space and placed by an instance transform. So each tile gets an
// identity placement carrying only the Bethesda Z-up -> engine Y-up change,
// which is folded into the instance rotation exactly as makeEngineInstanceRotation
// does for statics.
int cookLodTier(
    const std::filesystem::path& dataFilesPath,
    const std::filesystem::path& outputPath,
    const std::string& worldspaceEditorId,
    odai::importer::fnv::LandLodSet lodSet,
    std::int32_t tierCells,
    std::int32_t blockX0, std::int32_t blockZ0, std::int32_t blockX1, std::int32_t blockZ1
) {
    const auto totalStart = std::chrono::steady_clock::now();
    AssetResolver assets(dataFilesPath);

    ImportedScene scene;
    scene.sourceTag = "fnv_lod";

    // The tile walk itself lives in import/fnv/land_lod.cc so the runtime
    // streamer builds these the same way. It used to live here, inside a
    // function that also parsed argv and wrote a file, which is why there was
    // no way to reach it from a game.
    odai::importer::fnv::LandLodTierStats stats;
    std::string error;
    const bool ok = odai::importer::fnv::appendLandLodTier(
        [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
            return assets.resolveMesh(path, bytes);
        },
        [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
            return assets.resolveTexture(path, bytes);
        },
        worldspaceEditorId, lodSet, tierCells, blockX0, blockZ0, blockX1, blockZ1,
        /*sinkUnits=*/0.0f, scene, stats, error);
    if (!ok) {
        std::cerr << "error: " << error << "\n";
        return 1;
    }

    odai::importer::buildImportedScenePackedRenderData(scene);
    odai::importer::buildImportedScenePageRanges(scene);
    if (!odai::importer::saveImportedScene(scene, outputPath)) {
        std::cerr << "error: failed to write " << outputPath << ": "
                  << odai::importer::getImportedSceneLastError() << "\n";
        return 1;
    }

    const double totalMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - totalStart).count();
    std::cout << "LOD " << (lodSet == odai::importer::fnv::LandLodSet::Objects ? "objects" : "terrain")
              << " level" << tierCells << ": " << stats.tilesParsed << " tiles parsed of "
              << stats.tilesResolved << " resolved (" << stats.tilesMissing
              << " absent from the sparse grid), " << stats.triangles << " triangles, "
              << stats.textures << " textures\n";
    std::cout << "Wrote " << outputPath << " in " << totalMs << " ms\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printUsage();
        return 1;
    }
    const std::filesystem::path dataFilesPath = argv[1];
    const std::filesystem::path pluginName = argv[2];
    const std::filesystem::path outputPath = argv[3];
    const std::string mode = argv[4];

    std::optional<std::string> targetCellEditorId;
    std::optional<std::string> targetWorldspaceEditorId;
    std::int32_t gridX0 = 0, gridZ0 = 0, gridX1 = 0, gridZ1 = 0;
    bool withInteriors = false;

    if (mode == "--cell" && argc >= 6) {
        targetCellEditorId = argv[5];
    } else if (mode == "--worldspace" && argc >= 10) {
        targetWorldspaceEditorId = argv[5];
        gridX0 = std::atoi(argv[6]);
        gridZ0 = std::atoi(argv[7]);
        gridX1 = std::atoi(argv[8]);
        gridZ1 = std::atoi(argv[9]);
        if (gridX0 > gridX1) std::swap(gridX0, gridX1);
        if (gridZ0 > gridZ1) std::swap(gridZ0, gridZ1);
    } else if ((mode == "--lod" || mode == "--lodobjects") && argc >= 10) {
        // Tile coordinates, which step by the tier width. A range given in cell
        // coordinates still works: the corners are floored onto the tile grid
        // below and the loop only visits multiples of the step from there.
        //
        // --lod is TERRAIN and --lodobjects is the merged distant buildings.
        // Separate modes rather than a set argument because they are not
        // interchangeable: the object set exists only at level4, so the tier
        // argument means different things for each.
        const auto lodSet = (mode == "--lodobjects")
            ? odai::importer::fnv::LandLodSet::Objects
            : odai::importer::fnv::LandLodSet::Terrain;
        const std::string lodWorldspace = argv[5];
        std::int32_t bx0 = std::atoi(argv[6]);
        std::int32_t bz0 = std::atoi(argv[7]);
        std::int32_t bx1 = std::atoi(argv[8]);
        std::int32_t bz1 = std::atoi(argv[9]);
        if (bx0 > bx1) std::swap(bx0, bx1);
        if (bz0 > bz1) std::swap(bz0, bz1);
        // Optional tier (cells per tile): 4, 8, 16 or 32 for terrain, 4 only
        // for objects. Defaults to the finest.
        std::int32_t tierCells = odai::importer::fnv::kLandLodBlockCells;
        if (argc >= 11) {
            tierCells = std::atoi(argv[10]);
        }
        if (!odai::importer::fnv::landLodTierExists(lodSet, tierCells)) {
            std::cerr << "error: " << mode << " has no level" << tierCells << " tier"
                      << (lodSet == odai::importer::fnv::LandLodSet::Objects
                              ? " (object LOD ships level4 only)\n"
                              : " (terrain tiers are 4, 8, 16, 32)\n");
            return 1;
        }
        using odai::importer::fnv::landLodTileOrigin;
        return cookLodTier(
            dataFilesPath, outputPath, lodWorldspace, lodSet, tierCells,
            landLodTileOrigin(bx0, tierCells), landLodTileOrigin(bz0, tierCells),
            landLodTileOrigin(bx1, tierCells), landLodTileOrigin(bz1, tierCells));
    } else {
        printUsage();
        return 1;
    }
    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--with-interiors") == 0) {
            withInteriors = true;
        }
    }

    std::vector<std::string> doorTargetCells;
    const int result = cookOne(
        dataFilesPath, pluginName, outputPath, targetCellEditorId, targetWorldspaceEditorId,
        gridX0, gridZ0, gridX1, gridZ1, withInteriors ? &doorTargetCells : nullptr);
    if (result != 0 || !withInteriors) {
        return result;
    }

    // Every interior a door in the region opens onto, beside the exterior and
    // named by the one convention importedSceneInteriorFileName owns. Failing
    // to cook one is reported and skipped rather than fatal: a door into a cell
    // this plugin does not define should not lose the region that does.
    const std::string exteriorStem = outputPath.stem().string();
    const std::filesystem::path outputDirectory = outputPath.parent_path();
    std::size_t cooked = 0;
    for (const std::string& cellEditorId : doorTargetCells) {
        const std::filesystem::path interiorPath =
            outputDirectory / odai::importer::importedSceneInteriorFileName(exteriorStem, cellEditorId);
        std::cout << "\n=== interior: " << cellEditorId << " -> " << interiorPath.filename().string() << "\n";
        if (cookOne(dataFilesPath, pluginName, interiorPath, cellEditorId, std::nullopt,
                    0, 0, 0, 0, nullptr) == 0) {
            ++cooked;
        } else {
            std::cerr << "warning: could not cook interior " << cellEditorId << "\n";
        }
    }
    std::cout << "\nCooked " << cooked << " of " << doorTargetCells.size() << " reachable interior(s).\n";
    return 0;
}
