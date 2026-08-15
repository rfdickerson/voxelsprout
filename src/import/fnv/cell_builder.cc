#include "import/fnv/cell_builder.h"

#include <cstdlib>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>

#include "core/frame_profiler.h"
#include "import/dds.h"
#include "import/fnv/nif_scene.h"

namespace odai::importer::fnv {

namespace {

constexpr std::uint32_t kNoTextureIndex = 0xFFFFFFFFu;

std::string formIdHex(std::uint32_t formId) {
    std::ostringstream out;
    out << std::hex << std::uppercase << formId;
    return out.str();
}

std::string toLowerAsciiCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

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

// Morrowind's terrain, which is splatted rather than blended.
//
// There are no quadrants and no per-post opacity layers: VTEX is a 16x16 grid of
// land-texture indices over the cell, each entry covering a 4x4 block of quads.
// So a cell is textured by drawing whole blocks, and the natural mesh is one
// part per distinct texture, gathering every block that uses it -- typically two
// to eight parts rather than the four a Fallout cell emits.
//
// Each block gets its OWN 5x5 vertex block rather than sharing posts with its
// neighbours. That costs 16*16*25 = 6400 posts against 4225 shared (+51%) and
// buys the thing that matters: the texture tiles once across each block, so a
// shared post would need two different UVs. Same trade the Fallout path makes
// for its quadrants, for the same reason.
void appendMorrowindTerrainCell(
    ImportedSceneMesh& terrainMesh,
    const odai::importer::fnv::FalloutCellRecord& cell,
    const std::function<std::uint32_t(std::uint32_t)>& resolveLandTexture,
    const std::function<std::uint32_t(std::uint32_t)>& resolveLandTextureExact
) {
    using odai::importer::fnv::kLandPostSpacing;
    using odai::importer::fnv::kMorrowindTextureBlockQuads;
    using odai::importer::fnv::kMorrowindTextureGridSize;
    const odai::importer::fnv::FalloutLandRecord& land = *cell.land;
    const int gridSize = land.gridSize;
    const float cellWorldSize = land.cellWorldSize();
    const float cellOriginX = static_cast<float>(cell.gridX) * cellWorldSize;
    const float cellOriginY = static_cast<float>(cell.gridZ) * cellWorldSize;

    // Group blocks by the texture they resolve to, so one part covers every
    // block sharing a texture rather than one part per block (256 draws a cell).
    std::unordered_map<std::uint32_t, std::vector<int>> blocksByTexture;
    for (int block = 0; block < kMorrowindTextureGridSize * kMorrowindTextureGridSize; ++block) {
        std::uint32_t textureIndex = kNoTextureIndex;
        if (static_cast<std::size_t>(block) < land.morrowindTextureGrid.size()) {
            const std::uint16_t stored = land.morrowindTextureGrid[static_cast<std::size_t>(block)];
            // Stored value is the LTEX index PLUS ONE; 0 means the worldspace
            // default, which this importer serves from the fallback texture.
            if (stored != 0u) {
                textureIndex = resolveLandTextureExact(stored);
            }
        }
        if (textureIndex == kNoTextureIndex) {
            textureIndex = resolveLandTexture(0u);
        }
        blocksByTexture[textureIndex].push_back(block);
    }

    for (const auto& [textureIndex, blocks] : blocksByTexture) {
        const std::uint32_t firstIndex = static_cast<std::uint32_t>(terrainMesh.indices.size());
        for (const int block : blocks) {
            const int blockRow = block / kMorrowindTextureGridSize;
            const int blockCol = block % kMorrowindTextureGridSize;
            const int postRow0 = blockRow * kMorrowindTextureBlockQuads;
            const int postCol0 = blockCol * kMorrowindTextureBlockQuads;
            const std::uint32_t baseVertex = static_cast<std::uint32_t>(terrainMesh.vertices.size());
            for (int row = 0; row <= kMorrowindTextureBlockQuads; ++row) {
                for (int col = 0; col <= kMorrowindTextureBlockQuads; ++col) {
                    const int postRow = std::min(postRow0 + row, gridSize - 1);
                    const int postCol = std::min(postCol0 + col, gridSize - 1);
                    const int postIndex = (postRow * gridSize) + postCol;
                    const float bethesdaX = cellOriginX + (static_cast<float>(postCol) * kLandPostSpacing);
                    const float bethesdaY = cellOriginY + (static_cast<float>(postRow) * kLandPostSpacing);
                    const float bethesdaZ =
                        land.hasHeights ? land.heights[static_cast<std::size_t>(postIndex)] : 0.0f;
                    const Vec3 world = bethesdaToEngine(bethesdaX, bethesdaY, bethesdaZ);

                    ImportedSceneVertex vertex{};
                    vertex.position[0] = world.x;
                    vertex.position[1] = world.y;
                    vertex.position[2] = world.z;
                    // One full tile of the texture across the block.
                    vertex.uv[0] = static_cast<float>(col) / static_cast<float>(kMorrowindTextureBlockQuads);
                    vertex.uv[1] = static_cast<float>(row) / static_cast<float>(kMorrowindTextureBlockQuads);
                    if (land.hasNormals) {
                        const Vec3 normal = bethesdaToEngine(
                            land.normals[static_cast<std::size_t>(postIndex) * 3u],
                            land.normals[(static_cast<std::size_t>(postIndex) * 3u) + 1u],
                            land.normals[(static_cast<std::size_t>(postIndex) * 3u) + 2u]);
                        vertex.normal[0] = normal.x;
                        vertex.normal[1] = normal.y;
                        vertex.normal[2] = normal.z;
                    } else {
                        vertex.normal[1] = 1.0f;
                    }
                    // VCLR is Morrowind's baked lighting, and it is doing more
                    // work here than VCLR does in Fallout: this is a game with
                    // no dynamic terrain shadowing, so the shading under trees
                    // and against cliffs lives entirely in these bytes.
                    if (land.hasColors) {
                        vertex.color[0] = land.colors[static_cast<std::size_t>(postIndex) * 3u];
                        vertex.color[1] = land.colors[(static_cast<std::size_t>(postIndex) * 3u) + 1u];
                        vertex.color[2] = land.colors[(static_cast<std::size_t>(postIndex) * 3u) + 2u];
                    }
                    terrainMesh.vertices.push_back(vertex);
                }
            }
            constexpr int kStride = kMorrowindTextureBlockQuads + 1;
            for (int row = 0; row < kMorrowindTextureBlockQuads; ++row) {
                for (int col = 0; col < kMorrowindTextureBlockQuads; ++col) {
                    const std::uint32_t i00 =
                        baseVertex + static_cast<std::uint32_t>((row * kStride) + col);
                    const std::uint32_t i10 = i00 + 1u;
                    const std::uint32_t i01 = i00 + static_cast<std::uint32_t>(kStride);
                    const std::uint32_t i11 = i01 + 1u;
                    // Same winding as the Fallout path, and for the same reason:
                    // bethesdaToEngine negates Y, which flips the quad's sense.
                    terrainMesh.indices.push_back(i00);
                    terrainMesh.indices.push_back(i11);
                    terrainMesh.indices.push_back(i01);
                    terrainMesh.indices.push_back(i00);
                    terrainMesh.indices.push_back(i10);
                    terrainMesh.indices.push_back(i11);
                }
            }
        }
        const std::uint32_t indexCount =
            static_cast<std::uint32_t>(terrainMesh.indices.size()) - firstIndex;
        if (indexCount != 0u) {
            terrainMesh.parts.push_back(
                ImportedSceneMeshPart{firstIndex, indexCount, textureIndex, false});
        }
    }
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
    if (cell.land->gridSize != odai::importer::fnv::kLandGridSize) {
        appendMorrowindTerrainCell(terrainMesh, cell, resolveLandTexture, resolveLandTextureExact);
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

bool isEffectOnlyModelPath(std::string_view modelPath) {
    std::string lowered(modelPath);
    for (char& c : lowered) {
        if (c == '/') {
            c = '\\';
        } else {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    if (lowered.find("\\effects\\") != std::string::npos || lowered.rfind("effects\\", 0) == 0) {
        return true;
    }
    // fx-prefixed basenames living outside effects\ (fxvultures, fxglow...).
    const std::size_t lastSlash = lowered.find_last_of('\\');
    const std::string baseName =
        (lastSlash == std::string::npos) ? lowered : lowered.substr(lastSlash + 1u);
    return baseName.rfind("fx", 0) == 0;
}

namespace {

// Fills `outTables` from one plugin's records, rewriting every formID from that
// plugin's local mod-index space into the load order's global one.
//
// `laterWins` is what makes an override plugin work. The single-plugin path
// keeps first-wins (emplace) because it is the same either way with one file,
// and changing it would be a behaviour change for no reason.
void mergeWorldTablesFromScene(
    const FalloutSceneData& data,
    const FalloutLoadOrder& order,
    std::size_t pluginIndex,
    bool laterWins,
    FalloutWorldTables& outTables) {
    const auto remap = [&](std::uint32_t formId) {
        return order.remapFormId(pluginIndex, formId);
    };
    const auto put = [&](auto& map, const auto& key, const auto& value) {
        if (laterWins) {
            map.insert_or_assign(key, value);
        } else {
            map.emplace(key, value);
        }
    };
    for (const FalloutStaticRecord& entry : data.statics) {
        const std::uint32_t formId = remap(entry.formId);
        if (!entry.modelPath.empty()) {
            put(outTables.staticModelPaths, formId, entry.modelPath);
        }
        if (!entry.editorId.empty()) {
            put(outTables.staticEditorIds, formId, entry.editorId);
        }
        if (!entry.recordType.empty()) {
            put(outTables.staticRecordTypes, formId, entry.recordType);
        }
    }
    for (const FalloutLightRecord& entry : data.lights) {
        FalloutLightRecord light = entry;
        light.formId = remap(entry.formId);
        put(outTables.lightsByFormId, light.formId, light);
    }
    for (const FalloutLandTextureRecord& entry : data.landTextures) {
        if (!entry.diffuseTexturePath.empty()) {
            put(outTables.landTexturePaths, remap(entry.formId), entry.diffuseTexturePath);
        }
    }
    for (const FalloutRegionRecord& entry : data.regions) {
        if (entry.isDiscoverable()) {
            put(outTables.regionNamesByFormId, remap(entry.formId), entry.mapName);
        }
    }
    for (const FalloutWorldspaceRecord& entry : data.worldspaces) {
        if (!entry.editorId.empty()) {
            put(outTables.worldspaceFormIdsByEditorId, toLowerAsciiCopy(entry.editorId),
                remap(entry.formId));
        }
        if (entry.hasDefaultHeights) {
            FalloutWorldspaceRecord remapped = entry;
            remapped.formId = remap(entry.formId);
            put(outTables.worldspaceDefaultsByFormId, remapped.formId, remapped);
        }
    }
}

// The filter both builders use: reject every worldspace group and every cell's
// contents, so no LAND record is ever decompressed. That is what makes this
// affordable at startup.
FalloutExtractFilter worldTableFilter() {
    FalloutExtractFilter filter{};
    filter.wantWorldspace = [](std::uint32_t) { return false; };
    filter.wantCellContents = [](const FalloutCellRecord&) { return false; };
    return filter;
}

}  // namespace

bool buildFalloutWorldTables(
    const FalloutLoadOrder& order, FalloutWorldTables& outTables, std::string& outError) {
    outTables = FalloutWorldTables{};
    if (order.empty()) {
        outError = "empty load order";
        return false;
    }
    // Ascending load order: each plugin's records replace what an earlier one
    // offered, which is what an override plugin is for. A base record a patch
    // fixes -- a corrected MODL, a light's parameters -- takes effect here.
    for (std::size_t pluginIndex = 0; pluginIndex < order.entries().size(); ++pluginIndex) {
        const FalloutLoadOrderEntry& entry = order.entries()[pluginIndex];
        FalloutSceneData data;
        const FalloutExtractFilter filter = worldTableFilter();
        std::string error;
        if (!extractFalloutScene(entry.path, filter, data, error)) {
            // One unreadable plugin must not take the whole world down: the
            // base game's records are already in, and losing a patch's is a
            // degraded scene rather than no scene.
            //
            // std::cerr rather than VOX_LOGW for the reason the alpha-test log
            // below states: this file links into odai_newvegas_probe and
            // odai_newvegas_cooker, neither of which links core/log.cc.
            std::cerr << "[fnv] world tables: skipping " << entry.header.fileName << ": " << error
                      << "\n";
            continue;
        }
        mergeWorldTablesFromScene(data, order, pluginIndex, /*laterWins=*/true, outTables);
    }
    return true;
}

bool buildFalloutWorldTables(
    const std::filesystem::path& esmPath, FalloutWorldTables& outTables, std::string& outError) {
    outTables = FalloutWorldTables{};

    // Reject every worldspace group and every cell's contents: this pass wants
    // only the top-level STAT / LIGH / LTEX / TXST / WRLD / REGN records. Nothing per-cell is
    // materialized, so no LAND record is ever decompressed -- which is what
    // makes it cheap enough to run at game startup.
    FalloutExtractFilter filter{};
    filter.wantWorldspace = [](std::uint32_t) { return false; };
    filter.wantCellContents = [](const FalloutCellRecord&) { return false; };

    FalloutSceneData data;
    if (!extractFalloutScene(esmPath, filter, data, outError)) {
        return false;
    }

    for (const FalloutStaticRecord& entry : data.statics) {
        if (!entry.modelPath.empty()) {
            outTables.staticModelPaths.emplace(entry.formId, entry.modelPath);
        }
        if (!entry.editorId.empty()) {
            outTables.staticEditorIds.emplace(entry.formId, entry.editorId);
        }
        if (!entry.recordType.empty()) {
            outTables.staticRecordTypes.emplace(entry.formId, entry.recordType);
        }
        if (!entry.editorId.empty()) {
            outTables.baseFormIdsByEditorId.emplace(
                toLowerAsciiCopy(entry.editorId), entry.formId);
        }
    }
    for (const FalloutLightRecord& entry : data.lights) {
        outTables.lightsByFormId.emplace(entry.formId, entry);
    }
    for (const FalloutLandTextureRecord& entry : data.landTextures) {
        if (!entry.diffuseTexturePath.empty()) {
            outTables.landTexturePaths.emplace(entry.formId, entry.diffuseTexturePath);
        }
    }
    for (const FalloutRegionRecord& entry : data.regions) {
        if (entry.isDiscoverable()) {
            outTables.regionNamesByFormId.emplace(entry.formId, entry.mapName);
        }
    }
    for (const FalloutWorldspaceRecord& entry : data.worldspaces) {
        if (!entry.editorId.empty()) {
            outTables.worldspaceFormIdsByEditorId.emplace(
                toLowerAsciiCopy(entry.editorId), entry.formId);
        }
        if (entry.hasDefaultHeights) {
            outTables.worldspaceDefaultsByFormId.emplace(entry.formId, entry);
        }
    }
    return true;
}

CellSceneBuilder::CellSceneBuilder(
    const FalloutAssetSource& assets,
    const FalloutWorldTables& tables,
    DecodedTextureCache* textureCache)
    : m_assets(assets), m_tables(tables), m_textureCache(textureCache) {}

std::uint32_t CellSceneBuilder::resolveTextureIndex(const std::string& texturePath) {
    if (texturePath.empty()) {
        return kNoTextureIndex;
    }
    // One normalizer, the one asset resolution uses. The cooker used to key this
    // cache with its own private copy, so a path spelled differently from the
    // one that resolved could occupy a second slot.
    const std::string key = toLowerAsciiCopy(normalizeTexturePath(texturePath));
    if (const auto it = m_textureIndexByPath.find(key); it != m_textureIndexByPath.end()) {
        return it->second;
    }
    if (m_failedTexturePaths.count(key) != 0u) {
        return kNoTextureIndex;
    }
    if (m_scene.textures.size() >= m_textureBudget) {
        if (!m_warnedTextureBudget) {
            m_warnedTextureBudget = true;
            m_stats.textureBudgetExceeded = true;
        }
        return kNoTextureIndex;
    }

    const core::Stopwatch decodeTimer;
    ImportedSceneTexture texture;
    if (m_textureCache != nullptr) {
        // Shared across every builder: the decode happens once per distinct
        // texture no matter how many cells are being built at the same time.
        ImportedSceneTexture owned;
        const ImportedSceneTexture* cached =
            m_textureCache->get(m_assets, texturePath, m_maxTextureSize, owned);
        if (cached == nullptr) {
            m_failedTexturePaths.insert(key);
            return kNoTextureIndex;
        }
        // ImportedScene embeds its textures by value, so the scene still gets
        // its own copy -- what the cache saves is the decode, not the bytes.
        texture = *cached;
    } else {
        std::vector<std::uint8_t> ddsBytes;
        std::string resolveError;
        if (!m_assets.resolveTexture(texturePath, ddsBytes, resolveError)) {
            m_failedTexturePaths.insert(key);
            return kNoTextureIndex;
        }
        if (!loadDdsFromMemory(ddsBytes.data(), ddsBytes.size(), texture)) {
            m_failedTexturePaths.insert(key);
            return kNoTextureIndex;
        }
        if (m_maxTextureSize != 0u) {
            dropDdsMipLevels(texture, m_maxTextureSize);
        }
        texture.sourcePath = texturePath;
    }
    m_stats.textureDecodeMs += decodeTimer.elapsedMs();
    ++m_stats.texturesDecoded;
    const auto index = static_cast<std::uint32_t>(m_scene.textures.size());
    m_scene.textures.push_back(std::move(texture));
    m_textureIndexByPath.emplace(key, index);
    return index;
}

std::uint32_t CellSceneBuilder::resolveLandTexture(std::uint32_t landTextureFormId, bool exact) {
    // "exact" is the ATXT path: an additional layer that names a texture which
    // cannot be resolved must contribute nothing, NOT the fallback, or the layer
    // paints the base texture over itself at partial opacity.
    std::uint32_t resolved = kNoTextureIndex;
    if (landTextureFormId != 0u) {
        const auto it = m_tables.landTexturePaths.find(landTextureFormId);
        if (it != m_tables.landTexturePaths.end()) {
            resolved = resolveTextureIndex(it->second);
        }
    }
    if (resolved == kNoTextureIndex && !exact) {
        return m_fallbackLandTexture;
    }
    return resolved;
}

std::uint32_t CellSceneBuilder::dominantLandTexture(
    const std::vector<const FalloutCellRecord*>& cells) {
    std::unordered_map<std::uint32_t, std::size_t> useCounts;
    for (const FalloutCellRecord* cell : cells) {
        if (cell == nullptr || cell->land == nullptr) {
            continue;
        }
        for (const std::uint32_t formId : cell->land->quadrantBaseTextureFormId) {
            if (formId != 0u) {
                ++useCounts[formId];
            }
        }
    }
    // Most-used candidate that ACTUALLY RESOLVES, not simply most-used.
    //
    // A BTXT is free to name a formID that is not a land texture at all --
    // Fallout 3's MegatonWorld cell -1,-6 names 0xa8b, which is a levelled NPC
    // list. Picking it because it was the only candidate and then failing to
    // resolve it leaves the whole cell with no fallback, so every quadrant
    // draws untextured. Bethesda never saw it because Megaton's crater floor is
    // hidden under the town.
    std::uint32_t best = kNoTextureIndex;
    std::uint32_t bestFormId = 0;
    std::size_t bestUse = 0;
    for (const auto& [formId, count] : useCounts) {
        // Ties broken by formID so a cook stays reproducible.
        if (count < bestUse || (count == bestUse && formId >= bestFormId)) {
            continue;
        }
        const std::uint32_t resolved = resolveLandTexture(formId, /*exact=*/true);
        if (resolved == kNoTextureIndex) {
            continue;
        }
        best = resolved;
        bestFormId = formId;
        bestUse = count;
    }
    return best;
}

// One water quad per exterior cell whose water surface is high enough to be
// seen. Bethesda has no water GEOMETRY: a cell states a height and the engine
// fills the cell's whole 4096-unit footprint at it, so this is the entire
// representation, not an approximation of one.
//
// Without it every coast, lake and river in every worldspace is a hole. Anvil
// is the case that surfaced it -- a port city on the Abecean Sea, where the
// missing ocean occupies the left third of the frame and reads as "the terrain
// just ends into a grey void", which is a very convincing impression of a
// streaming bug. It is not: the sea was never imported.
bool appendCellWaterPatch(ImportedScene& outScene, const FalloutCellRecord& cell) {
    if (cell.isInterior || !cell.hasGridCoords) {
        return false;
    }
    // See FalloutCellRecord::hasWater: an absent XCLW is Oblivion's "sea level",
    // not "no water". The dry case is a sentinel VALUE and was rejected at parse.
    const float waterHeight = cell.hasWater ? cell.waterHeight : 0.0f;
    if (cell.land != nullptr && cell.land->hasHeights) {
        // Water strictly below every post in the cell is water under a solid
        // floor -- true of most of the Mojave, where sea level sits far beneath
        // the desert. Emitting it anyway would put a full-cell alpha-blended
        // quad under every cell in the worldspace, at real fill cost, for
        // nothing visible.
        const float lowestPost =
            *std::min_element(std::begin(cell.land->heights), std::end(cell.land->heights));
        if (waterHeight <= lowestPost) {
            return false;
        }
    }
    // Bethesda (x, y) -> engine (x, -y), so the cell's +Y edge becomes its
    // MINIMUM engine z and the origin moves to the far corner.
    ImportedSceneWaterPatch patch{};
    patch.originX = static_cast<float>(cell.gridX) * kExteriorCellSize;
    patch.originZ = -(static_cast<float>(cell.gridZ) + 1.0f) * kExteriorCellSize;
    patch.sizeX = kExteriorCellSize;
    patch.sizeZ = kExteriorCellSize;
    patch.waterLevel = waterHeight;
    outScene.waterPatches.push_back(patch);
    return true;
}

// Records that one placed reference produced no geometry, and why.
//
// Both halves matter. The memo (m_failedStatics) is what lets a REPEAT of an
// already-failed base be attributed to the same cause instead of only the first
// one being explained, and the counters are what turn "this town has holes" from
// an impression into a number with a cause attached to it.
void CellSceneBuilder::noteDroppedReference(
    std::uint32_t baseFormId, StaticDropReason reason) {
    m_failedStatics[baseFormId] = reason;
    if (reason == StaticDropReason::kIntentional) {
        return;  // counted by effectMeshesSkipped / editorMarkerModelsSkipped
    }
    switch (reason) {
        case StaticDropReason::kBaseNotFound:
            ++m_stats.referencesDroppedBaseNotFound;
            break;
        case StaticDropReason::kBaseHasNoModel:
            ++m_stats.referencesDroppedBaseHasNoModel;
            break;
        case StaticDropReason::kMeshUnresolved:
            ++m_stats.referencesDroppedMeshUnresolved;
            break;
        case StaticDropReason::kMeshUnreadable:
            ++m_stats.referencesDroppedMeshUnreadable;
            break;
        case StaticDropReason::kIntentional:
            break;
    }
    const auto typeIt = m_tables.staticRecordTypes.find(baseFormId);
    ++m_stats.droppedReferencesByBaseType[
        typeIt == m_tables.staticRecordTypes.end() ? std::string("<base record not found>")
                                                   : typeIt->second];
}

void CellSceneBuilder::addCellTerrain(const FalloutCellRecord& cell) {
    // Before the LAND guard: a cell can be open ocean with no terrain at all.
    // Tamriel (0,0) is exactly that, so gating water on terrain would drop the
    // sea precisely where there is nothing else to draw.
    if (appendCellWaterPatch(m_scene, cell)) {
        ++m_stats.waterPatchesEmitted;
    }
    if (cell.land == nullptr) {
        return;
    }
    if (m_terrainMeshIndex == static_cast<std::size_t>(-1)) {
        ImportedSceneMesh terrainMesh;
        terrainMesh.name = "terrain";
        m_terrainMeshIndex = m_scene.meshes.size();
        m_scene.meshes.push_back(std::move(terrainMesh));
    }
    const auto resolveInherited = [this](std::uint32_t formId) {
        return resolveLandTexture(formId, /*exact=*/false);
    };
    const auto resolveExact = [this](std::uint32_t formId) {
        return resolveLandTexture(formId, /*exact=*/true);
    };
    appendTerrainCell(
        m_scene.meshes[m_terrainMeshIndex], cell, resolveInherited, resolveExact,
        m_stats.droppedTerrainLayers);
}

// A LIGH reference becomes an ImportedSceneLight. The renderer's punctual-light
// path (chunk_upload -> frame_run's 64-light budget -> evaluateImportedLocalLights)
// has always been complete; nothing in this importer had ever written to it, so
// every lamp in the Mojave was an unlit prop.
//
// Rotation is ignored on purpose: the flag census over all 501 LIGH records
// found no spotlight bit anywhere in the base game, so every one of them is
// omnidirectional and its REFR orientation cannot matter.
void CellSceneBuilder::addCellLight(
    const FalloutPlacedReference& ref, const FalloutLightRecord& light) {
    if (light.radius <= 0.0f) {
        // Exactly one LIGH in FalloutNV.esm has radius 0. chunk_upload would
        // drop it anyway; counting it here is what makes that visible.
        ++m_stats.lightsSkippedZeroRadius;
        return;
    }
    ImportedSceneLight entry{};
    entry.sourceId = light.editorId;
    const Vec3 position = bethesdaToEngine(ref.position[0], ref.position[1], ref.position[2]);
    entry.position[0] = position.x;
    entry.position[1] = position.y;
    entry.position[2] = position.z;
    entry.color[0] = light.color[0];
    entry.color[1] = light.color[1];
    entry.color[2] = light.color[2];
    // XSCL scales the placed instance, and for a light the only thing there is
    // to scale is its reach.
    entry.radius = light.radius * (ref.scale > 0.0f ? ref.scale : 1.0f);
    // FNAM, the GECK's fade value, is the only authored brightness in the
    // record. The shader multiplies by its own global intensity on top, so this
    // stays a plain pass-through rather than being pre-tuned here.
    entry.intensity = light.fadeValue > 0.0f ? light.fadeValue : 1.0f;
    entry.flags = light.flags;
    m_scene.lights.push_back(std::move(entry));
    ++m_stats.lightsPlaced;
}

void CellSceneBuilder::addCellStatics(const FalloutCellRecord& cell) {
    // Diagnostic sets the cooker kept (unresolved texture paths, extreme-UV
    // model names, per-model untextured lists) are not carried here: they are
    // reporting for a batch cook, not something a streaming build can act on.
    // The counters that matter are in CellBuildStats.
    for (const auto& rawRef : cell.references) {
        // Morrowind names its base by string, so the formID a reference carries
        // is the one this builder's own scan handed out. Resolved here, once,
        // rather than threaded through every lookup below -- everything after
        // this point works in formIDs whichever game the cell came from.
        FalloutPlacedReference resolvedRef;
        const FalloutPlacedReference* refPtr = &rawRef;
        if (rawRef.baseFormId == 0u && !rawRef.baseEditorId.empty()) {
            resolvedRef = rawRef;
            const auto nameIt =
                m_tables.baseFormIdsByEditorId.find(toLowerAsciiCopy(rawRef.baseEditorId));
            resolvedRef.baseFormId =
                (nameIt == m_tables.baseFormIdsByEditorId.end()) ? 0u : nameIt->second;
            refPtr = &resolvedRef;
        }
        const FalloutPlacedReference& ref = *refPtr;
            // Lights first, and deliberately ahead of the m_failedStatics gate:
            // only 29 of 501 LIGH records carry a MODL, so the other 472 have no
            // model path, land in m_failedStatics on first sight, and would be
            // skipped before ever being looked at as a light.
            if (const auto lightIt = m_tables.lightsByFormId.find(ref.baseFormId);
                lightIt != m_tables.lightsByFormId.end()) {
                addCellLight(ref, lightIt->second);
                // No `continue`: a LIGH that does have a mesh still needs its
                // lamp placed, so fall through into the static path below.
            }
            if (const auto failedIt = m_failedStatics.find(ref.baseFormId);
                failedIt != m_failedStatics.end()) {
                // A repeat of a base that already failed. Counted again, because
                // the question this answers is "how many placements drew
                // nothing", and one missing rock placed a hundred times is a
                // hundred holes.
                noteDroppedReference(ref.baseFormId, failedIt->second);
                continue;
            }
            auto meshIt = m_meshIndexByStaticFormId.find(ref.baseFormId);
            if (meshIt == m_meshIndexByStaticFormId.end()) {
                const auto statIt = m_tables.staticModelPaths.find(ref.baseFormId);
                if (statIt == m_tables.staticModelPaths.end() || statIt->second.empty()) {
                    // Two different failures wearing one shape: a formID that
                    // names no record at all (a load-order or remap fault) and a
                    // record that simply has no MODL (a trigger or an activator,
                    // usually correct). staticRecordTypes is what separates them.
                    const bool baseExists =
                        m_tables.staticRecordTypes.find(ref.baseFormId) !=
                        m_tables.staticRecordTypes.end();
                    noteDroppedReference(
                        ref.baseFormId,
                        baseExists ? StaticDropReason::kBaseHasNoModel
                                   : StaticDropReason::kBaseNotFound);
                    continue;
                }
                const std::string& staticModelPath = statIt->second;
                std::vector<std::uint8_t> nifBytes;
                if (isEffectOnlyModelPath(staticModelPath)) {
                    ++m_stats.effectMeshesSkipped;
                    noteDroppedReference(ref.baseFormId, StaticDropReason::kIntentional);
                    continue;
                }
                std::string meshError;
                if (!m_assets.resolveMesh(staticModelPath, nifBytes, meshError)) {
                    std::cerr << "warning: could not resolve mesh " << staticModelPath << "\n";
                    noteDroppedReference(ref.baseFormId, StaticDropReason::kMeshUnresolved);
                    continue;
                }
                // Editor markers are level-design furniture, not world geometry:
                // the GECK draws them, the game does not. marker_radiation.nif is
                // one shape whose UVs are a single constant point, so it has no
                // sensible texture and never had one -- it was rendering as a
                // grey slab in mid-air.
                const std::string lowerModelPath = toLowerAsciiCopy(staticModelPath);
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
                    ++m_stats.editorMarkerModelsSkipped;
                    noteDroppedReference(ref.baseFormId, StaticDropReason::kIntentional);
                    continue;
                }

                const core::Stopwatch nifTimer;
                odai::importer::fnv::NifModel nifModel;
                std::string nifError;
                if (!odai::importer::fnv::parseNifStaticMesh(nifBytes, nifModel, nifError) || nifModel.shapes.empty()) {
                    std::cerr << "warning: failed to parse NIF " << staticModelPath << ": " << nifError << "\n";
                    noteDroppedReference(ref.baseFormId, StaticDropReason::kMeshUnreadable);
                    continue;
                }
                m_stats.skippedGeometryShapes += nifModel.skippedShapeCount;
                // Node-recognition health. Nonzero nodeParseFailures means a
                // subtree was dropped rather than -- as it used to be --
                // silently relocated to the model origin and drawn in the sky.
                m_stats.nodeParseFailures += nifModel.nodeParseFailedCount;
                m_stats.unhandledNodeTypes += nifModel.unhandledNodeTypeCount;

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
                const auto editorIdIt = m_tables.staticEditorIds.find(ref.baseFormId);
                mesh.name = (editorIdIt == m_tables.staticEditorIds.end() || editorIdIt->second.empty())
                    ? staticModelPath
                    : editorIdIt->second;
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
                        ++m_stats.shadowDecalShapesSkipped;
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
                            ++m_stats.untexturedShapesGivenModelTexture;
                        }
                        part.alphaTest = shape.alphaTest;
                        part.alphaBlend = shape.alphaBlend;
                        part.unlit = shape.unlit;
                        part.twoSided = shape.twoSided;
                        part.alphaThreshold = shape.alphaThreshold;
                        // ODAI_FNV_LOG_ALPHATEST=1 names every surface that will
                        // run the discard, with the threshold and texture it
                        // will run against. "Some walls have holes" is otherwise
                        // a hunt through hundreds of shapes for the few that
                        // actually alpha-test.
                        static const bool s_logAlphaTest =
                            std::getenv("ODAI_FNV_LOG_ALPHATEST") != nullptr;
                        if (s_logAlphaTest && part.alphaTest) {
                            // std::cout rather than VOX_LOGI: this file is
                            // linked into odai_newvegas_probe and
                            // odai_newvegas_cooker, neither of which links
                            // core/log.cc, and a debug line is not worth
                            // growing their link closure.
                            std::cout << "alphaTest model=" << staticModelPath
                                      << " shape=" << shape.name
                                      << " thr=" << static_cast<int>(part.alphaThreshold)
                                      << " tex=" << shape.diffuseTexturePath << "\n";
                        }
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
                                ++m_stats.extremeUvShapes;
                                m_stats.extremeUvModelPaths.insert(
                                    toLowerAsciiCopy(staticModelPath) +
                                    "  (worst triangle UV span " +
                                    std::to_string(static_cast<int>(uvSpan)) + ")");
                                break;
                            }
                        }
                        if (part.textureIndex == kNoTextureIndex) {
                            ++m_stats.untexturedShapes;
                            if (shape.diffuseTexturePath.empty()) {
                                ++m_stats.shapesWithNoTexturePath;
                                m_stats.untexturedModelPaths.insert(
                                    toLowerAsciiCopy(staticModelPath));
                            } else {
                                m_stats.unresolvedTexturePaths.insert(
                                    toLowerAsciiCopy(shape.diffuseTexturePath));
                            }
                        }
                        ++m_stats.totalShapes;
                        mesh.parts.push_back(part);
                    }
                }
                if (mesh.vertices.empty()) {
                    // The one drop path with no diagnostic at all: the NIF
                    // parsed, its shapes were all filtered out (decals, gore
                    // caps, effect-only sheets, extreme UVs), and the reference
                    // then vanished without a word. Naming it is the difference
                    // between "the importer is dropping things" and knowing
                    // which asset and which filter.
                    std::cerr << "warning: " << staticModelPath
                              << " built no geometry (every shape was filtered)\n";
                    noteDroppedReference(ref.baseFormId, StaticDropReason::kMeshUnreadable);
                    continue;
                }
                const std::uint32_t meshIndex = static_cast<std::uint32_t>(m_scene.meshes.size());
                m_scene.meshes.push_back(std::move(mesh));
                meshIt = m_meshIndexByStaticFormId.emplace(ref.baseFormId, meshIndex).first;
            }

            ImportedSceneInstance instance;
            instance.meshIndex = meshIt->second;
            instance.sourceId = "refr_" + formIdHex(ref.formId);
            if (const auto modelIt = m_tables.staticModelPaths.find(ref.baseFormId);
                modelIt != m_tables.staticModelPaths.end()) {
                instance.modelPath = modelIt->second;
            }
            const Vec3 worldPos = bethesdaToEngine(ref.position[0], ref.position[1], ref.position[2]);
            const Mat3 bethRotation = eulerToMatrixBethesdaOrder(
                ref.rotationRadians[0], ref.rotationRadians[1], ref.rotationRadians[2]);
            const Mat3 engineRotation = makeEngineInstanceRotation(bethRotation);
            writeTransform(instance, worldPos, engineRotation, ref.scale);
            m_scene.instances.push_back(instance);
            ++m_stats.placedInstances;
    }
}

void CellSceneBuilder::finish(ImportedScene& outScene) {
    if (m_terrainMeshIndex != static_cast<std::size_t>(-1)) {
        const ImportedSceneMesh& terrainMesh = m_scene.meshes[m_terrainMeshIndex];
        // Deliberately NO instance for the terrain mesh.
        //
        // buildImportedScenePackedRenderData emits terrain from meshes[0]
        // directly, via the landscapeCells path -- the invariant is "meshes[0]
        // is terrain", not "terrain has an instance". Adding one produced a
        // scene with one more instance than the cooker's for the same cell.
        // See the cooker's note: one landscapeCells entry per emitted terrain
        // PART, not per cell. sourceLandscapeCellCount is derived from it and
        // the renderer requires terrain to occupy the leading draws.
        m_scene.landscapeCells.clear();
        m_scene.landscapeCells.resize(terrainMesh.parts.size());
        m_scene.sourceLandscapeCellCount =
            static_cast<std::uint32_t>(m_scene.landscapeCells.size());
        m_stats.terrainPartsEmitted = terrainMesh.parts.size();
    }

    // Every part's alpha mode came off its NiAlphaProperty (or its absence),
    // so the texture-content cutout guess must not run -- it forces alpha test
    // onto authored-opaque shapes that share a cutout's texture. Set before
    // packing: buildImportedScenePackedRenderData is a caller of the guess.
    m_scene.alphaFlagsAuthored = true;
    buildImportedScenePackedRenderData(m_scene);
    buildImportedScenePageRanges(m_scene);
    outScene = std::move(m_scene);
    m_scene = ImportedScene{};
}

}  // namespace odai::importer::fnv
