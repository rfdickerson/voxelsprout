#include "import/fnv/cell_builder.h"

#include <cstdlib>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>

#include "core/frame_profiler.h"
#include "import/dds.h"
#include "import/fnv/nif_scene.h"

namespace odai::importer::fnv {

float sampleLandLayerOpacity(
    const FalloutLandTextureLayer& layer, float quadrantRow, float quadrantCol) {
    constexpr int kSide = kLandQuadrantGridSize;
    const auto rawPost = [&](int row, int col) {
        row = std::clamp(row, 0, kSide - 1);
        col = std::clamp(col, 0, kSide - 1);
        return std::clamp(layer.opacity[(row * kSide) + col], 0.0f, 1.0f);
    };
    // A narrow positive reconstruction kernel gives a painted layer a small
    // shoulder outside its last non-zero post. Without it, interpolation can
    // only shrink an authored shape: zero-weight triangles are skipped by the
    // shader and their grid-aligned support boundary can never move.
    const auto filteredPost = [&](int row, int col) {
        static constexpr float kKernel[3] = {1.0f, 2.0f, 1.0f};
        float blurred = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                blurred += rawPost(row + dy, col + dx) *
                    kKernel[dy + 1] * kKernel[dx + 1];
            }
        }
        blurred *= 1.0f / 16.0f;
        // Keep authored centres dominant while feathering only the boundary.
        return std::clamp((rawPost(row, col) * 0.65f) + (blurred * 0.35f), 0.0f, 1.0f);
    };

    const float rowClamped = std::clamp(quadrantRow, 0.0f, static_cast<float>(kSide - 1));
    const float colClamped = std::clamp(quadrantCol, 0.0f, static_cast<float>(kSide - 1));
    const int row0 = std::min(static_cast<int>(std::floor(rowClamped)), kSide - 2);
    const int col0 = std::min(static_cast<int>(std::floor(colClamped)), kSide - 2);
    const float rowFraction = rowClamped - static_cast<float>(row0);
    const float colFraction = colClamped - static_cast<float>(col0);
    const float rowT = rowFraction * rowFraction * (3.0f - (2.0f * rowFraction));
    const float colT = colFraction * colFraction * (3.0f - (2.0f * colFraction));
    const float top = std::lerp(filteredPost(row0, col0), filteredPost(row0, col0 + 1), colT);
    const float bottom =
        std::lerp(filteredPost(row0 + 1, col0), filteredPost(row0 + 1, col0 + 1), colT);
    return std::clamp(std::lerp(top, bottom, rowT), 0.0f, 1.0f);
}

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
// The angle sign is settled: `odai_bethesda_probe --rotations FalloutNV.esm
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

    // Resolve every block's texture up front, because a block now needs to know
    // what its NEIGHBOURS use as well as what it uses itself.
    constexpr int kBlockCount = kMorrowindTextureGridSize * kMorrowindTextureGridSize;
    std::array<std::uint32_t, kBlockCount> blockTexture{};
    for (int block = 0; block < kBlockCount; ++block) {
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
        blockTexture[static_cast<std::size_t>(block)] = textureIndex;
    }

    // Group blocks by the texture they resolve to, so one part covers every
    // block sharing a texture rather than one part per block (256 draws a cell).
    std::unordered_map<std::uint32_t, std::vector<int>> blocksByTexture;
    for (int block = 0; block < kBlockCount; ++block) {
        blocksByTexture[blockTexture[static_cast<std::size_t>(block)]].push_back(block);
    }

    // MORROWIND HAS NO PER-VERTEX TEXTURE WEIGHTS AT ALL. Where Oblivion and
    // Fallout author ATXT/VTXT -- an opacity per layer per post -- Morrowind's
    // VTEX names ONE texture per 512-unit block and stops. Drawn literally that
    // is what it looks like: a staircase of hard-edged squares, most visible
    // where a path crosses grass, and it is the single blockiest thing about
    // this terrain.
    //
    // The fix is to synthesize the weights the format never stored. Treat the
    // block textures as samples on a lattice at block CENTRES and bilinearly
    // interpolate between the four nearest, which is exactly the 4-slot layer
    // blend the shader already implements for the other two games -- own
    // texture, horizontal neighbour, vertical neighbour, diagonal.
    //
    // Two things make this work out cleanly rather than approximately:
    //
    //  * UVs ARE ALREADY CONTINUOUS ACROSS A BLOCK EDGE. Each block tiles its
    //    texture exactly once, 0..1, so at a shared edge this block's u=1 and
    //    the neighbour's u=0 are the same point in a tiling texture. Sampling a
    //    neighbour's texture at our own UV therefore lands where the neighbour
    //    itself draws it, with no phase seam to hide.
    //  * The weights are a partition of unity by construction, and the shader's
    //    chain is a sequence of lerps, so each layer's weight is divided by the
    //    running total (w_i = a_i / sum(a_0..a_i)). That reproduces the
    //    normalized blend exactly rather than approximately -- see the loop.
    //
    // Neighbours OUTSIDE this cell are not reachable here (the adjacent LAND
    // record is a different extract), so an out-of-range neighbour falls back to
    // this block's own texture. That leaves the 8192-unit cell seams unblended
    // while fixing every 512-unit block seam inside them, which is 15 of every
    // 16 boundaries in each axis.
    const auto textureAt = [&](int blockRow, int blockCol, std::uint32_t fallback) {
        if (blockRow < 0 || blockRow >= kMorrowindTextureGridSize || blockCol < 0 ||
            blockCol >= kMorrowindTextureGridSize) {
            return fallback;
        }
        return blockTexture[static_cast<std::size_t>((blockRow * kMorrowindTextureGridSize) + blockCol)];
    };

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
                    const float blockU =
                        static_cast<float>(col) / static_cast<float>(kMorrowindTextureBlockQuads);
                    const float blockV =
                        static_cast<float>(row) / static_cast<float>(kMorrowindTextureBlockQuads);
                    vertex.uv[0] = blockU;
                    vertex.uv[1] = blockV;

                    // Bilinear blend toward the neighbouring blocks' textures.
                    // Offset from this block's CENTRE, so the influence is zero
                    // in the middle of a block and half at its edge -- where the
                    // neighbour computes the mirrored half and the two meet
                    // continuously.
                    const float offsetU = blockU - 0.5f;
                    const float offsetV = blockV - 0.5f;
                    const float fracU = std::abs(offsetU);
                    const float fracV = std::abs(offsetV);
                    const int stepCol = offsetU < 0.0f ? -1 : 1;
                    const int stepRow = offsetV < 0.0f ? -1 : 1;
                    const std::uint32_t ownTexture = textureIndex;
                    const std::uint32_t neighbourU =
                        textureAt(blockRow, blockCol + stepCol, ownTexture);
                    const std::uint32_t neighbourV =
                        textureAt(blockRow + stepRow, blockCol, ownTexture);
                    const std::uint32_t neighbourUv =
                        textureAt(blockRow + stepRow, blockCol + stepCol, ownTexture);
                    // Partition of unity over the four nearest block centres.
                    // amount[0] is this block's own share and is carried by the
                    // base texture rather than by a layer slot.
                    const float amount[4] = {
                        (1.0f - fracU) * (1.0f - fracV),
                        fracU * (1.0f - fracV),
                        (1.0f - fracU) * fracV,
                        fracU * fracV,
                    };
                    const std::uint32_t neighbourTexture[3] = {
                        neighbourU, neighbourV, neighbourUv};
                    // The shader composites as a chain of lerps from the base
                    // sample, so a layer's weight has to be its share of the
                    // running total, not its share of the whole. Dividing by the
                    // cumulative sum makes the chain reproduce the normalized
                    // blend exactly.
                    float runningTotal = amount[0];
                    for (int slot = 0; slot < 3; ++slot) {
                        const float share = amount[slot + 1];
                        runningTotal += share;
                        // A neighbour using the same texture as this block is
                        // not a transition and must not consume a layer slot --
                        // blending a texture with itself is a no-op that costs a
                        // sample, and there are only four slots for what can be
                        // four genuinely different textures at a corner.
                        if (neighbourTexture[slot] == ownTexture || share <= 0.0f ||
                            runningTotal <= 0.0f) {
                            continue;
                        }
                        vertex.layerTextureIndex[slot] = neighbourTexture[slot];
                        vertex.layerWeight[slot] = share / runningTotal;
                    }
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
    // VTXT is only 17x17 per quadrant. Reconstruct it on a 2x denser mesh so
    // the smooth per-quad opacity field below is not immediately reduced back
    // to one pair of linear triangles per 128-unit post square. Terrain height
    // remains on the authored triangle planes; only normals, tint and blend
    // weights gain intermediate samples.
    constexpr int kBlendSubdivision = 2;
    constexpr int kRefinedQuadrantGridSize =
        ((kLandQuadrantGridSize - 1) * kBlendSubdivision) + 1;

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
    // Per-quadrant vertices are deliberately duplicated, and the 2x blend
    // reconstruction below makes that 4*33*33 = 4356 vertices per cell. The
    // property it buys is the one that matters: within a quadrant the layer set
    // is constant, so only the interpolating weights vary.
    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        const int colBegin = ((quadrant & 1) != 0) ? (kLandGridSize - 1) / 2 : 0;
        const int rowBegin = ((quadrant & 2) != 0) ? (kLandGridSize - 1) / 2 : 0;

        // This quadrant's layer stack, chosen once. Skyrim commonly authors five
        // fully opaque overlays in a quadrant, while the packed terrain vertex
        // carries four. Peak opacity cannot rank that case: every layer ties at
        // 1.0, and the old form-ID tie-break happened to reject Whiterun's
        // DirtPath01 because its unrelated LTEX form ID was largest.
        //
        // ATXT is an ordered paint stack. When it overflows, retain the latest
        // four layers: those are the layers the author painted last and the ones
        // that would cover an earlier layer wherever they overlap. The selected
        // subset is restored to ascending order below because the shader's lerp
        // chain is not commutative.
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
                    return a.layer->layerIndex != b.layer->layerIndex
                        ? (a.layer->layerIndex > b.layer->layerIndex)
                        : (a.peakOpacity > b.peakOpacity);
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

        const auto sampleAuthoredTriangle = [&](const std::vector<float>& values, int channels,
                                                int channel, float row, float col,
                                                float fallback) {
            if (values.empty()) {
                return fallback;
            }
            int row0 = std::min(static_cast<int>(std::floor(row)), kLandGridSize - 2);
            int col0 = std::min(static_cast<int>(std::floor(col)), kLandGridSize - 2);
            row0 = std::clamp(row0, 0, kLandGridSize - 2);
            col0 = std::clamp(col0, 0, kLandGridSize - 2);
            const float fy = std::clamp(row - static_cast<float>(row0), 0.0f, 1.0f);
            const float fx = std::clamp(col - static_cast<float>(col0), 0.0f, 1.0f);
            const auto at = [&](int r, int c) {
                const std::size_t index =
                    (static_cast<std::size_t>((r * kLandGridSize) + c) *
                     static_cast<std::size_t>(channels)) + static_cast<std::size_t>(channel);
                return index < values.size() ? values[index] : fallback;
            };
            const float v00 = at(row0, col0);
            const float v10 = at(row0, col0 + 1);
            const float v01 = at(row0 + 1, col0);
            const float v11 = at(row0 + 1, col0 + 1);
            // Match the two triangles emitted below: 00-11-01 above the
            // diagonal and 00-10-11 below it. This makes the refined mesh
            // geometrically identical to the old one when tessellation is off.
            return fy >= fx
                ? ((1.0f - fy) * v00) + (fx * v11) + ((fy - fx) * v01)
                : ((1.0f - fx) * v00) + (fy * v11) + ((fx - fy) * v10);
        };

        const std::uint32_t quadrantBaseVertex = static_cast<std::uint32_t>(terrainMesh.vertices.size());
        for (int refinedRow = 0; refinedRow < kRefinedQuadrantGridSize; ++refinedRow) {
            for (int refinedCol = 0; refinedCol < kRefinedQuadrantGridSize; ++refinedCol) {
                const float quadrantRow =
                    static_cast<float>(refinedRow) / static_cast<float>(kBlendSubdivision);
                const float quadrantCol =
                    static_cast<float>(refinedCol) / static_cast<float>(kBlendSubdivision);
                const float row = static_cast<float>(rowBegin) + quadrantRow;
                const float col = static_cast<float>(colBegin) + quadrantCol;
                const float bethesdaX = cellOriginX + (col * kLandPostSpacing);
                const float bethesdaY = cellOriginZ + (row * kLandPostSpacing);
                const float bethesdaZ = land.hasHeights
                    ? sampleAuthoredTriangle(land.heights, 1, 0, row, col, 0.0f)
                    : 0.0f;
                const Vec3 world = bethesdaToEngine(bethesdaX, bethesdaY, bethesdaZ);

                ImportedSceneVertex vertex{};
                vertex.position[0] = world.x;
                vertex.position[1] = world.y;
                vertex.position[2] = world.z;
                // Keep the authored 512-unit landscape-texture scale, but
                // phase it in WORLD coordinates rather than restarting from
                // (0,0) in every extracted CELL. The old local coordinate was
                // continuous inside its four quadrants yet repeated the same
                // eight-by-eight texture stamp every 4096 units. On Riverwood's
                // riverbank that turns a natural sequence of Dirt02 / grass
                // layers into immediately visible square repetitions whenever
                // the camera sees two cells at once.
                //
                // bethesdaX/Y are deliberately used here instead of engine X/Z:
                // terrain's authored UV orientation lives in the Bethesda grid,
                // and the engine-space conversion negates its second horizontal
                // axis. One texture repeat remains four 128-unit LAND quads.
                constexpr float kTerrainTextureWorldPeriod =
                    (kLandPostSpacing * static_cast<float>(kLandGridSize - 1)) /
                    kLandTextureTilesPerCell;
                vertex.uv[0] = bethesdaX / kTerrainTextureWorldPeriod;
                vertex.uv[1] = bethesdaY / kTerrainTextureWorldPeriod;
                if (land.hasNormals) {
                    Vec3 normal = bethesdaToEngine(
                        sampleAuthoredTriangle(land.normals, 3, 0, row, col, 0.0f),
                        sampleAuthoredTriangle(land.normals, 3, 1, row, col, 0.0f),
                        sampleAuthoredTriangle(land.normals, 3, 2, row, col, 1.0f));
                    const float length = std::sqrt(
                        (normal.x * normal.x) + (normal.y * normal.y) + (normal.z * normal.z));
                    if (length > 1.0e-6f) {
                        normal.x /= length;
                        normal.y /= length;
                        normal.z /= length;
                    }
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
                    vertex.color[0] = sampleAuthoredTriangle(land.colors, 3, 0, row, col, 1.0f);
                    vertex.color[1] = sampleAuthoredTriangle(land.colors, 3, 1, row, col, 1.0f);
                    vertex.color[2] = sampleAuthoredTriangle(land.colors, 3, 2, row, col, 1.0f);
                }

                for (std::size_t slot = 0; slot < quadrantLayers.size(); ++slot) {
                    vertex.layerTextureIndex[slot] = quadrantLayers[slot].textureIndex;
                    vertex.layerWeight[slot] = sampleLandLayerOpacity(
                        *quadrantLayers[slot].layer, quadrantRow, quadrantCol);
                }
                terrainMesh.vertices.push_back(vertex);
            }
        }

        const std::uint32_t quadrantFirstIndex = static_cast<std::uint32_t>(terrainMesh.indices.size());
        for (int quadrantRow = 0; quadrantRow < kRefinedQuadrantGridSize - 1; ++quadrantRow) {
            for (int quadrantCol = 0; quadrantCol < kRefinedQuadrantGridSize - 1; ++quadrantCol) {
                const std::uint32_t i00 = quadrantBaseVertex +
                    static_cast<std::uint32_t>((quadrantRow * kRefinedQuadrantGridSize) + quadrantCol);
                const std::uint32_t i10 = i00 + 1u;
                const std::uint32_t i01 = i00 + static_cast<std::uint32_t>(kRefinedQuadrantGridSize);
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

bool isFireParticleEffectModelPath(std::string_view modelPath) {
    if (!isEffectOnlyModelPath(modelPath)) {
        return false;
    }
    std::string lowered(modelPath);
    for (char& c : lowered) {
        if (c == '/') {
            c = '\\';
        } else {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    const bool fireNamed = lowered.find("fire") != std::string::npos ||
        lowered.find("flame") != std::string::npos ||
        lowered.find("ember") != std::string::npos;
    if (!fireNamed) {
        return false;
    }
    // Moving/gameplay effects need an owner, duration and collision semantics;
    // a CELL reference alone cannot supply those. This pass is for the looping
    // environmental fire NIFs used by hearths, braziers and campfires.
    constexpr std::array<std::string_view, 7> kDynamicTokens = {
        "firefly", "projectile", "fireball", "firebolt", "weapon",
        "impact", "magic"
    };
    return std::none_of(kDynamicTokens.begin(), kDynamicTokens.end(), [&](std::string_view token) {
        return lowered.find(token) != std::string::npos;
    });
}

// THE SKY IS NOT WORLD GEOMETRY, AND SKYRIM PLACES IT AS IF IT WERE.
//
// Tamriel's persistent cell (0,0) holds 14643 references, and among them are
// the game's own sky objects: sky\clouddistant01.nif and its ten siblings, the
// aurora, the cloud shapes. Those are sky-dome scale -- tens of thousands of
// units across -- and Skyrim's own renderer draws them on the sky dome, keyed
// to the weather, never as scenery.
//
// Imported literally they become enormous opaque quads hanging over the
// landscape. The symptom is not "a floating mesh": it is a flat, near-white
// plane covering the ground with the real terrain visible only where it pokes
// out at the edges, appearing a moment AFTER the terrain because the persistent
// cell is slow to build. The material-flags and untextured-highlight views both
// call it ordinary geometry, because that is exactly what it now is. The
// giveaway is the NORMAL view: one uniform up-facing normal across a region
// that ought to be a hillside.
//
// This engine draws its own sky from the WTHR record (see
// src/import/fnv/weather_records.h), so these meshes have no job here at all.
// True for Bethesda's distant-LOD stand-in meshes, which are named by suffix:
// wrcastlemainbuilding01LOD.nif, wrjorvaskr01lod.nif, wrskyforge01lod.nif.
// Matched on the stem rather than anywhere in the path so a directory called
// "lod" full of real geometry is not swept up.
bool isDistantLodModelPath(std::string_view modelPath) {
    std::string lowered(modelPath);
    for (char& c : lowered) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    constexpr std::string_view kSuffix = "lod.nif";
    return lowered.size() > kSuffix.size() &&
        lowered.compare(lowered.size() - kSuffix.size(), kSuffix.size(), kSuffix) == 0;
}

bool isSkyOnlyModelPath(std::string_view modelPath) {
    std::string lowered(modelPath);
    for (char& c : lowered) {
        if (c == '/') {
            c = '\\';
        } else {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    // Anchored at the model root rather than matched anywhere in the path: a
    // "sky" component deeper in a path is a building's skylight or an interior
    // named for one, not the firmament.
    return lowered.rfind("sky\\", 0) == 0 || lowered.rfind("meshes\\sky\\", 0) == 0;
}

namespace {

// Fills `outTables` from one plugin's records, rewriting every formID from that
// plugin's local mod-index space into the load order's global one.
//
// `laterWins` is what makes an override plugin work. The single-plugin path
// keeps first-wins (emplace) because it is the same either way with one file,
// and changing it would be a behaviour change for no reason.
// Defined below; declared here because mergeWorldTablesFromScene calls it.
void resolveWorldspaceInheritance(FalloutWorldTables& outTables);

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
            if (entry.mapNameStringId != 0u) {
                put(outTables.regionNameStringIdsByFormId, remap(entry.formId),
                    entry.mapNameStringId);
            }
        }
    }
    for (const FalloutWorldspaceRecord& entry : data.worldspaces) {
        if (!entry.editorId.empty()) {
            put(outTables.worldspaceFormIdsByEditorId, toLowerAsciiCopy(entry.editorId),
                remap(entry.formId));
        }
        // Every worldspace, not only the ones carrying DNAM: a child inherits
        // its defaults from its WNAM parent, and the parent has to be in the
        // map to be found. resolveWorldspaceInheritance() pushes them down.
        FalloutWorldspaceRecord remapped = entry;
        remapped.formId = remap(entry.formId);
        if (remapped.parentWorldspaceFormId != 0u) {
            remapped.parentWorldspaceFormId = remap(entry.parentWorldspaceFormId);
        }
        put(outTables.worldspaceDefaultsByFormId, remapped.formId, remapped);
    }
    resolveWorldspaceInheritance(outTables);
}

// Pushes DNAM defaults down the WNAM parent chain, so a child worldspace that
// declares none answers with its parent's.
//
// Skyrim needs this and the earlier games do not: WhiterunWorld's entire WRLD
// record is an EDID plus a WNAM, and every walled city is the same shape. The
// chain is walked with a visit cap rather than a visited set because it is
// two or three links long in practice and a cycle must not hang a load.
void resolveWorldspaceInheritance(FalloutWorldTables& outTables) {
    constexpr int kMaxParentHops = 8;
    for (auto& entry : outTables.worldspaceDefaultsByFormId) {
        FalloutWorldspaceRecord& worldspace = entry.second;
        if (worldspace.hasDefaultHeights) {
            continue;
        }
        std::uint32_t parentFormId = worldspace.parentWorldspaceFormId;
        for (int hop = 0; hop < kMaxParentHops && parentFormId != 0u; ++hop) {
            const auto found = outTables.worldspaceDefaultsByFormId.find(parentFormId);
            if (found == outTables.worldspaceDefaultsByFormId.end()) {
                break;
            }
            if (found->second.hasDefaultHeights) {
                worldspace.hasDefaultHeights = true;
                worldspace.defaultLandHeight = found->second.defaultLandHeight;
                worldspace.defaultWaterHeight = found->second.defaultWaterHeight;
                break;
            }
            parentFormId = found->second.parentWorldspaceFormId;
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
    std::uint32_t nextMorrowindBaseFormId = 0x80000000u;
    std::unordered_map<std::uint64_t, std::string> morrowindPaletteEditorIds;
    std::unordered_map<std::string, std::string> morrowindTexturePathsByEditorId;

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
            // below states: this file links into odai_bethesda_probe and
            // odai_newvegas_cooker, neither of which links core/log.cc.
            std::cerr << "[fnv] world tables: skipping " << entry.header.fileName << ": " << error
                      << "\n";
            continue;
        }
        if (entry.header.format != EsmPluginFormat::kMorrowind) {
            mergeWorldTablesFromScene(data, order, pluginIndex, /*laterWins=*/true, outTables);
            continue;
        }

        // TES3 base records are keyed by text, not by a plugin-local formID.
        // Reuse one collision-free synthetic ID for every case-insensitive key
        // and let later records replace the model/type behind it.
        for (const FalloutStaticRecord& source : data.statics) {
            if (source.editorId.empty()) {
                continue;
            }
            const std::string key = toLowerAsciiCopy(source.editorId);
            auto found = outTables.baseFormIdsByEditorId.find(key);
            if (found == outTables.baseFormIdsByEditorId.end()) {
                found = outTables.baseFormIdsByEditorId.emplace(
                    key, nextMorrowindBaseFormId++).first;
            }
            const std::uint32_t globalId = found->second;
            if (!source.modelPath.empty()) {
                outTables.staticModelPaths.insert_or_assign(globalId, source.modelPath);
            }
            outTables.staticEditorIds.insert_or_assign(globalId, source.editorId);
            if (!source.recordType.empty()) {
                outTables.staticRecordTypes.insert_or_assign(globalId, source.recordType);
            }
        }
        for (const FalloutLandTextureRecord& texture : data.landTextures) {
            if (texture.editorId.empty() || texture.formId == 0u) {
                continue;
            }
            const std::string editorKey = toLowerAsciiCopy(texture.editorId);
            const std::uint64_t paletteKey =
                (static_cast<std::uint64_t>(pluginIndex) << 32u) | texture.formId;
            morrowindPaletteEditorIds[paletteKey] = editorKey;
            if (!texture.diffuseTexturePath.empty()) {
                morrowindTexturePathsByEditorId.insert_or_assign(
                    editorKey, texture.diffuseTexturePath);
            }
        }
        for (const FalloutWorldspaceRecord& worldspace : data.worldspaces) {
            if (!worldspace.editorId.empty()) {
                outTables.worldspaceFormIdsByEditorId.insert_or_assign(
                    toLowerAsciiCopy(worldspace.editorId), worldspace.formId);
            }
            outTables.worldspaceDefaultsByFormId.insert_or_assign(
                worldspace.formId, worldspace);
        }
    }
    for (const auto& [paletteKey, editorKey] : morrowindPaletteEditorIds) {
        const auto path = morrowindTexturePathsByEditorId.find(editorKey);
        if (path != morrowindTexturePathsByEditorId.end()) {
            outTables.morrowindLandTexturePaths.emplace(paletteKey, path->second);
        }
    }
    resolveWorldspaceInheritance(outTables);
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
            outTables.morrowindLandTexturePaths.emplace(
                static_cast<std::uint64_t>(entry.formId), entry.diffuseTexturePath);
        }
    }
    for (const FalloutRegionRecord& entry : data.regions) {
        if (entry.isDiscoverable()) {
            outTables.regionNamesByFormId.emplace(entry.formId, entry.mapName);
            if (entry.mapNameStringId != 0u) {
                outTables.regionNameStringIdsByFormId.emplace(entry.formId, entry.mapNameStringId);
            }
        }
    }
    for (const FalloutWorldspaceRecord& entry : data.worldspaces) {
        if (!entry.editorId.empty()) {
            outTables.worldspaceFormIdsByEditorId.emplace(
                toLowerAsciiCopy(entry.editorId), entry.formId);
        }
        outTables.worldspaceDefaultsByFormId.emplace(entry.formId, entry);
    }
    resolveWorldspaceInheritance(outTables);
    return true;
}

CellSceneBuilder::CellSceneBuilder(
    const FalloutAssetSource& assets,
    const FalloutWorldTables& tables,
    DecodedTextureCache* textureCache)
    : m_assets(assets), m_tables(tables), m_textureCache(textureCache) {
    m_syntheticStaticModelPaths.emplace(
        0xfff00001u, "Clutter\\Lumbermill\\LumbermillSaw01\\LumbermillSaw01.nif");
    m_syntheticStaticModelPaths.emplace(
        0xfff00002u, "Clutter\\Lumbermill\\LumbermillSash01\\LumbermillSash01.nif");
}

const std::string* CellSceneBuilder::staticModelPathFor(std::uint32_t baseFormId) const {
    if (const auto it = m_syntheticStaticModelPaths.find(baseFormId);
        it != m_syntheticStaticModelPaths.end()) {
        return &it->second;
    }
    if (const auto it = m_tables.staticModelPaths.find(baseFormId);
        it != m_tables.staticModelPaths.end()) {
        return &it->second;
    }
    return nullptr;
}

std::uint32_t CellSceneBuilder::resolveTextureIndex(
    const std::string& texturePath, bool linearData) {
    if (texturePath.empty()) {
        return kNoTextureIndex;
    }
    // One normalizer, the one asset resolution uses. The cooker used to key this
    // cache with its own private copy, so a path spelled differently from the
    // one that resolved could occupy a second slot.
    std::string key = toLowerAsciiCopy(normalizeTexturePath(texturePath));
    if (linearData) {
        key += "|linear-data";
    }
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
    // DDS does not carry colour-space intent. Most DXT1 assets are albedo and
    // use an sRGB image view, but Skyrim's DefaultWater.dds is vector data.
    if (linearData && texture.format == TextureFormat::BC1) {
        texture.format = TextureFormat::BC1Linear;
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

// Water is authored as a height rather than geometry. A fully submerged cell
// can therefore use one quad, but a river cell must follow the LAND shoreline:
// one 4096-unit rectangle makes a narrow river look like a cyan city block and
// leaves adjacent boardwalks apparently hanging in mid-air. Partial cells are
// emitted as contiguous LAND-post runs (128 units wide in TES4/TES5), keeping
// the coastline faithful without adding a new serialized water-mesh format.
//
// Without it every coast, lake and river in every worldspace is a hole. Anvil
// is the case that surfaced it -- a port city on the Abecean Sea, where the
// missing ocean occupies the left third of the frame and reads as "the terrain
// just ends into a grey void", which is a very convincing impression of a
// streaming bug. It is not: the sea was never imported.
bool appendCellWaterPatch(
    ImportedScene& outScene, const FalloutCellRecord& cell,
    const FalloutWorldspaceRecord* worldspace) {
    if (cell.isInterior || !cell.hasGridCoords) {
        return false;
    }
    // See FalloutCellRecord::hasWater: an absent XCLW is Oblivion's "sea level",
    // not "no water". The dry case is a sentinel VALUE and was rejected at parse.
    //
    // BUT "SEA LEVEL" IS ONLY MEANINGFUL WHERE THERE IS A SEA TO BE LEVEL WITH.
    // A CITY WORLDSPACE HAS NO LAND RECORD AT ALL -- WhiterunWorld, and every
    // Imperial City district -- so the lowest-post guard below cannot run, and
    // the implied height of 0 was emitted unconditionally. Whiterun's ground is
    // placed statics sitting at engine y about -3120, which put a full-cell
    // alpha-blended water quad 3120 units ABOVE the city, slicing through the
    // houses. On screen it is a flat grey wedge across the frame that reads as
    // broken geometry rather than as water in the wrong place.
    //
    // So an absent XCLW needs land to mean sea level. An EXPLICIT XCLW is still
    // honoured with or without land, because that is authored intent -- which is
    // what keeps a landless open-water cell wet.
    // The implied height is the WORLDSPACE's default water height, resolved up
    // the WNAM parent chain -- not zero. Tamriel declares -14000; WhiterunWorld
    // declares nothing at all and inherits it. Falling back to 0 put the sea
    // 14000 units too high, which in a city standing at y -3120 is a full-cell
    // quad through the rooftops.
    // A LANDLESS CELL IS STILL EMITTED. Tamriel (0,0) has no LAND and is open
    // ocean; requiring terrain would drop the sea exactly where there is
    // nothing else to draw. Oblivion declares no DNAM on any of its 84
    // worldspaces, so there the implied height stays 0 -- its sea level -- and
    // this changes nothing. It is Skyrim that needed the lookup.
    const bool hasImpliedHeight = worldspace != nullptr && worldspace->hasDefaultHeights;
    const float impliedHeight = hasImpliedHeight ? worldspace->defaultWaterHeight : 0.0f;
    const float waterHeight = cell.hasWater ? cell.waterHeight : impliedHeight;
    const float cellWorldSize = cell.land != nullptr
        ? cell.land->cellWorldSize()
        : kExteriorCellSize;
    if (cell.land != nullptr && cell.land->hasHeights) {
        // Water strictly below every post in the cell is water under a solid
        // floor -- true of most of the Mojave, where sea level sits far beneath
        // the desert. Emitting it anyway would put a full-cell alpha-blended
        // quad under every cell in the worldspace, at real fill cost, for
        // nothing visible.
        const auto [lowest, highest] =
            std::minmax_element(cell.land->heights.begin(), cell.land->heights.end());
        const float lowestPost = *lowest;
        if (waterHeight <= lowestPost) {
            return false;
        }
        // A lake/ocean lying above every terrain post is exactly the old
        // full-cell case. Keep one patch here rather than needlessly splitting
        // open water into a LAND-grid checkerboard.
        if (waterHeight < *highest) {
            const int side = cell.land->gridSize;
            const float postSpacing = cellWorldSize / static_cast<float>(side - 1);
            bool appended = false;
            for (int row = 0; row < side - 1; ++row) {
                int col = 0;
                while (col < side - 1) {
                    const auto quadIsWet = [&](int quadCol) {
                        const auto heightAt = [&](int r, int c) {
                            return cell.land->heights[static_cast<std::size_t>((r * side) + c)];
                        };
                        // A terrain triangle crossing the water plane belongs
                        // to the river. The small one-post overreach is vastly
                        // less visible than a 4096-unit square and cannot make
                        // a dry, wholly-above-water quad wet.
                        return std::min({
                            heightAt(row, quadCol), heightAt(row, quadCol + 1),
                            heightAt(row + 1, quadCol), heightAt(row + 1, quadCol + 1)}) < waterHeight;
                    };
                    while (col < side - 1 && !quadIsWet(col)) {
                        ++col;
                    }
                    const int firstWetCol = col;
                    while (col < side - 1 && quadIsWet(col)) {
                        ++col;
                    }
                    if (firstWetCol == col) {
                        continue;
                    }
                    ImportedSceneWaterPatch patch{};
                    patch.originX = (static_cast<float>(cell.gridX) * cellWorldSize) +
                        (static_cast<float>(firstWetCol) * postSpacing);
                    // Bethesda +Y maps to engine -Z. A row's lower engine-Z
                    // edge is consequently its *next* Bethesda post.
                    patch.originZ = -((static_cast<float>(cell.gridZ) * cellWorldSize) +
                        (static_cast<float>(row + 1) * postSpacing));
                    patch.sizeX = static_cast<float>(col - firstWetCol) * postSpacing;
                    patch.sizeZ = postSpacing;
                    patch.waterLevel = waterHeight;
                    outScene.waterPatches.push_back(patch);
                    appended = true;
                }
            }
            return appended;
        }
    }
    // Bethesda (x, y) -> engine (x, -y), so the cell's +Y edge becomes its
    // MINIMUM engine z and the origin moves to the far corner.
    ImportedSceneWaterPatch patch{};
    patch.originX = static_cast<float>(cell.gridX) * cellWorldSize;
    patch.originZ = -(static_cast<float>(cell.gridZ) + 1.0f) * cellWorldSize;
    patch.sizeX = cellWorldSize;
    patch.sizeZ = cellWorldSize;
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
    const std::size_t firstWaterPatch = m_scene.waterPatches.size();
    if (appendCellWaterPatch(m_scene, cell, m_tables.findWorldspace(cell.worldspaceFormId))) {
        // Skyrim supplies a 64x64 flow vector texture for each exterior water
        // cell. Resolve it first: its presence is the format discriminator, so
        // Fallout/Oblivion/Morrowind retain their existing generic normal.
        // Tamriel's default WATR (form 0x18) names DefaultWater.dds for all
        // three normal layers; sampling it at three scales happens in Slang.
        const std::string flowPath =
            "textures\\water\\skyrim.esm\\flow." + std::to_string(cell.gridX) +
            "." + std::to_string(cell.gridZ) + ".dds";
        const std::uint32_t flowTexture = resolveTextureIndex(flowPath, /*linearData=*/true);
        if (flowTexture != kNoTextureIndex) {
            const std::uint32_t normalTexture = resolveTextureIndex(
                "Data\\Textures\\Water\\DefaultWater.dds", /*linearData=*/true);
            for (std::size_t index = firstWaterPatch; index < m_scene.waterPatches.size(); ++index) {
                ImportedSceneWaterPatch& water = m_scene.waterPatches[index];
                water.flowTextureIndex = flowTexture;
                water.normalTextureIndex = normalTexture;
            }
        }
        m_stats.waterPatchesEmitted += m_scene.waterPatches.size() - firstWaterPatch;
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
    const auto resolveExact = [this, &cell](std::uint32_t formId) {
        if (cell.land != nullptr &&
            cell.land->gridSize == kMorrowindLandGridSize && formId != 0u) {
            const std::uint64_t key =
                (static_cast<std::uint64_t>(cell.land->sourcePluginIndex) << 32u) | formId;
            const auto found = m_tables.morrowindLandTexturePaths.find(key);
            if (found != m_tables.morrowindLandTexturePaths.end()) {
                return resolveTextureIndex(found->second);
            }
            return kNoTextureIndex;
        }
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

void CellSceneBuilder::addCellFireEmitter(
    const FalloutPlacedReference& ref, std::string_view modelPath) {
    ImportedSceneParticleEmitter emitter{};
    emitter.sourceId = "refr_" + formIdHex(ref.formId);
    const Vec3 position = bethesdaToEngine(ref.position[0], ref.position[1], ref.position[2]);
    emitter.position[0] = position.x;
    emitter.position[1] = position.y;
    emitter.position[2] = position.z;
    emitter.seed = ref.formId ^ (ref.baseFormId * 0x9e3779b9u);

    std::string lowered(modelPath);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    const float placedScale = ref.scale > 0.0f ? ref.scale : 1.0f;
    float presetScale = 1.0f;
    if (lowered.find("large") != std::string::npos ||
        lowered.find("heavy") != std::string::npos) {
        presetScale = 1.55f;
        emitter.particleCount = 72u;
    } else if (lowered.find("small") != std::string::npos ||
               lowered.find("sconce") != std::string::npos ||
               lowered.find("candle") != std::string::npos) {
        presetScale = 0.52f;
        emitter.particleCount = 32u;
    } else if (lowered.find("medium") != std::string::npos) {
        presetScale = 0.82f;
        emitter.particleCount = 48u;
    }
    const float effectScale = presetScale * placedScale;
    emitter.spawnRadius *= effectScale;
    // REFR scale expands the footprint of a fire but not each flame tongue.
    // Scaling all three dimensions made a 3x hearth marker produce individual
    // five-metre billboards. Keep lobe size and rise in a physically useful
    // range while still allowing named large/small variants to read apart.
    emitter.upwardSpeed *= std::sqrt(presetScale);
    emitter.particleSize *= std::sqrt(presetScale);
    if (lowered.find("_lite") != std::string::npos) {
        emitter.intensity = 0.38f;
        emitter.spawnRadius = 9.0f * placedScale;
        emitter.particleLifetime = 0.72f;
        emitter.upwardSpeed = 58.0f;
        emitter.particleSize = 8.0f;
        emitter.particleCount = 20u;
    }
    if (std::getenv("ODAI_DEBUG_FIRE_EMITTERS") != nullptr) {
        std::cerr << "[fire-emitter] model=" << modelPath
                  << " ref=" << formIdHex(ref.formId)
                  << " position=(" << emitter.position[0] << ", "
                  << emitter.position[1] << ", " << emitter.position[2] << ")"
                  << " scale=" << effectScale << '\n';
    }
    m_scene.particleEmitters.push_back(emitter);
    ++m_stats.particleEmittersPlaced;
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
        // INITIALLY DISABLED REFERENCES DO NOT RENDER, and nothing here had
        // ever checked. These are quest objects waiting for a script -- and
        // some are enormous: Skyrim's MG07 blizzard barrier is a dome measured
        // at 246723 x 341884 units, parked in Tamriel's persistent cell. Drawn
        // literally it is a flat near-white plane over the whole landscape,
        // appearing a beat after the terrain because the persistent cell is
        // slow to build -- which reads as a renderer bug, not as a flag.
        //
        // XESP (enable-parent) state is NOT resolved: the parent can live in
        // any cell, and its runtime state does not exist in a viewer with no
        // quest engine. The flag alone is the authored "hidden until story
        // says otherwise", and honouring just it matches what an unstarted
        // save shows. A ref that is enable-parented to a disabled parent
        // WITHOUT carrying the flag itself still draws; measured across the
        // Skyrim spawn cells that is scenery (ferns under a bridge), not
        // barriers.
        if ((ref.recordFlags & 0x00000800u) != 0u) {
            ++m_stats.disabledReferencesSkipped;
            continue;
        }
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
            // Effect-only NIFs are placements too. The opaque mesh path cannot
            // draw their particles, but a stationary fire reference supplies
            // an exact cross-game emitter origin. This must run before the
            // failed-base cache: the same fire base can be placed many times
            // and every REFR needs its own emitter.
            const std::string* placedModelPath = staticModelPathFor(ref.baseFormId);
            if (placedModelPath != nullptr && isEffectOnlyModelPath(*placedModelPath)) {
                if (isFireParticleEffectModelPath(*placedModelPath)) {
                    addCellFireEmitter(ref, *placedModelPath);
                }
                ++m_stats.effectMeshesSkipped;
                noteDroppedReference(ref.baseFormId, StaticDropReason::kIntentional);
                continue;
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
                const std::string* resolvedModelPath = staticModelPathFor(ref.baseFormId);
                if (resolvedModelPath == nullptr || resolvedModelPath->empty()) {
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
                const std::string& staticModelPath = *resolvedModelPath;
                std::vector<std::uint8_t> nifBytes;
                if (isSkyOnlyModelPath(staticModelPath)) {
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
                if (applyNifBannerGravityRestPose(staticModelPath, nifModel)) {
                    ++m_stats.clothMeshesSettled;
                }
                std::vector<NifCollisionTriangle> modelCollision =
                    nifModel.collisionTriangles;
                if (modelCollision.empty()) {
                    // Per-NIF fallback, not a global mode: one unsupported
                    // Havok wrapper must not disable authored collision for
                    // every other building in the cell.
                    for (const NifShape& shape : nifModel.shapes) {
                        if (shape.alphaBlend ||
                            (shape.diffuseTexturePath.empty() && shapeIsPlanar(shape))) {
                            continue;
                        }
                        for (std::size_t tri = 0;
                             (tri * 3u) + 2u < shape.triangleIndices.size(); ++tri) {
                            const std::uint32_t indices[3] = {
                                shape.triangleIndices[tri * 3u],
                                shape.triangleIndices[(tri * 3u) + 1u],
                                shape.triangleIndices[(tri * 3u) + 2u]};
                            NifCollisionTriangle collision;
                            bool valid = true;
                            for (int point = 0; point < 3; ++point) {
                                const std::size_t source =
                                    static_cast<std::size_t>(indices[point]) * 3u;
                                if (source + 2u >= shape.positions.size()) {
                                    valid = false;
                                    break;
                                }
                                std::copy_n(&shape.positions[source], 3u,
                                            &collision.vertices[point * 3]);
                            }
                            if (valid) {
                                modelCollision.push_back(collision);
                            }
                        }
                    }
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
                // Skyrim's environmental machines carry ordinary
                // NiControllerSequence tracks in their geometry NIFs. Actor
                // behavior remains out of scope, but these three assets are a
                // closed rigid hierarchy and can be played without Papyrus or
                // Havok state. Water wheels use their authored Idle loop; the
                // saw and sash normally receive Activate events from the mill
                // script, so the passive showcase repeats that work cycle.
                const KfAnimation* selectedMachineryClip = nullptr;
                const bool waterWheel =
                    lowerModelPath.find("lumbermill01waterwheel") != std::string::npos;
                const bool sawMachine =
                    lowerModelPath.find("\\lumbermillsaw01\\lumbermillsaw01.nif") !=
                    std::string::npos;
                const bool sawSash =
                    lowerModelPath.find("\\lumbermillsash01\\lumbermillsash01.nif") !=
                    std::string::npos;
                const std::string_view wantedClip = waterWheel ? "Idle" : "Activate";
                if (waterWheel || sawMachine || sawSash) {
                    const auto clipIt = std::find_if(
                        nifModel.embeddedAnimations.begin(), nifModel.embeddedAnimations.end(),
                        [&](const KfAnimation& animation) {
                            return animation.name == wantedClip;
                        });
                    if (clipIt != nifModel.embeddedAnimations.end()) {
                        selectedMachineryClip = &*clipIt;
                    }
                }
                std::vector<const KfAnimation*> selectedRigidClips;
                if (nifModel.autoPlayEmbeddedAnimations) {
                    selectedRigidClips.reserve(nifModel.embeddedAnimations.size());
                    for (const KfAnimation& animation : nifModel.embeddedAnimations) {
                        selectedRigidClips.push_back(&animation);
                    }
                }
                if (selectedMachineryClip != nullptr &&
                    std::find(
                        selectedRigidClips.begin(), selectedRigidClips.end(),
                        selectedMachineryClip) == selectedRigidClips.end()) {
                    selectedRigidClips.push_back(selectedMachineryClip);
                }
                std::unordered_map<std::string, std::uint32_t> rigidAnimationByNode;
                for (const KfAnimation* selectedClip : selectedRigidClips) {
                    for (const KfBoneTrack& track : selectedClip->tracks) {
                        if (rigidAnimationByNode.contains(track.nodeName)) {
                            continue;
                        }
                        const auto shapeIt = std::find_if(
                            nifModel.shapes.begin(), nifModel.shapes.end(),
                            [&](const NifShape& shape) {
                                return shape.animationNodeName == track.nodeName;
                            });
                        if (shapeIt == nifModel.shapes.end()) {
                            continue;
                        }
                        ImportedSceneRigidAnimation animation;
                        animation.nodeName = track.nodeName;
                        animation.duration = selectedClip->duration();
                        // The scripted Skyrim Activate cycle is intentionally
                        // repeated by this renderer-only showcase. Morrowind's
                        // direct controllers keep their authored cycle mode.
                        animation.cycleType = selectedClip == selectedMachineryClip
                            ? 0u
                            : selectedClip->cycleType;
                        std::memcpy(
                            animation.parentTransform, shapeIt->animationParentTransform,
                            sizeof(animation.parentTransform));
                        std::memcpy(
                            animation.bindTransform, shapeIt->animationBindTransform,
                            sizeof(animation.bindTransform));
                        animation.translationKeys.reserve(track.translationKeys.size());
                        for (const KfVector3Key& key : track.translationKeys) {
                            ImportedSceneVectorKey copied;
                            copied.time = key.time;
                            copied.value[0] = key.value.x;
                            copied.value[1] = key.value.y;
                            copied.value[2] = key.value.z;
                            animation.translationKeys.push_back(copied);
                        }
                        animation.rotationKeys.reserve(track.rotationKeys.size());
                        for (const KfQuaternionKey& key : track.rotationKeys) {
                            ImportedSceneQuaternionKey copied;
                            copied.time = key.time;
                            copied.value[0] = key.value.x;
                            copied.value[1] = key.value.y;
                            copied.value[2] = key.value.z;
                            copied.value[3] = key.value.w;
                            animation.rotationKeys.push_back(copied);
                        }
                        animation.scaleKeys.reserve(track.scaleKeys.size());
                        for (const KfVector3Key& key : track.scaleKeys) {
                            ImportedSceneVectorKey copied;
                            copied.time = key.time;
                            copied.value[0] = key.value.x;
                            copied.value[1] = key.value.y;
                            copied.value[2] = key.value.z;
                            animation.scaleKeys.push_back(copied);
                        }
                        const auto animationIndex =
                            static_cast<std::uint32_t>(mesh.rigidAnimations.size());
                        rigidAnimationByNode.emplace(track.nodeName, animationIndex);
                        mesh.rigidAnimations.push_back(std::move(animation));
                    }
                }
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
                        // Alpha only. The RGB of a Bethesda vertex colour is
                        // usually a baked ambient-occlusion tint that this
                        // renderer already gets from its own AO pass, and
                        // multiplying it in on top would double-darken every
                        // corner -- so it is deliberately left out while the
                        // channel that has no other source is taken. See
                        // ImportedSceneVertex::colorAlpha.
                        if ((v * 4u) + 3u < shape.colors.size()) {
                            vertex.colorAlpha = shape.colors[(v * 4u) + 3u];
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
                        // A DISTANT-LOD SHELL IS A HOLLOW, SINGLE-SIDED HULL,
                        // and back-face culling eats half of it.
                        //
                        // Bethesda places `*LOD.nif` stand-ins in the PARENT
                        // worldspace for anything whose real geometry lives in
                        // a child one: Dragonsreach in Tamriel is
                        // WRCastleMainBuilding01LOD, the whole building in 1324
                        // triangles off a shared LOD atlas, with no interior
                        // faces because the game only ever shows it from far
                        // outside. Drawn one-sided here, every angle that looks
                        // along a wall sees straight through it, and the
                        // symptom is exactly "half of Dragonsreach is missing"
                        // -- with nothing dropped and no reference unresolved,
                        // which is what makes it so hard to place.
                        //
                        // Forcing them two-sided costs nothing (the shell is
                        // tiny) and makes the silhouette solid from every
                        // angle, which is all a stand-in has to be.
                        part.twoSided = shape.twoSided || isDistantLodModelPath(staticModelPath);
                        part.alphaThreshold = shape.alphaThreshold;
                        if (const auto animationIt =
                                rigidAnimationByNode.find(shape.animationNodeName);
                            animationIt != rigidAnimationByNode.end()) {
                            part.rigidAnimationIndex = animationIt->second;
                        }
                        // ODAI_FNV_LOG_ALPHATEST=1 names every surface that will
                        // run the discard, with the threshold and texture it
                        // will run against. "Some walls have holes" is otherwise
                        // a hunt through hundreds of shapes for the few that
                        // actually alpha-test.
                        static const bool s_logAlphaTest =
                            std::getenv("ODAI_FNV_LOG_ALPHATEST") != nullptr;
                        if (s_logAlphaTest && part.alphaTest) {
                            // std::cout rather than VOX_LOGI: this file is
                            // linked into odai_bethesda_probe and
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
                m_collisionByStaticFormId.emplace(ref.baseFormId, std::move(modelCollision));
                meshIt = m_meshIndexByStaticFormId.emplace(ref.baseFormId, meshIndex).first;
            }

            ImportedSceneInstance instance;
            instance.meshIndex = meshIt->second;
            instance.sourceId = "refr_" + formIdHex(ref.formId);
            if (const std::string* modelPath = staticModelPathFor(ref.baseFormId)) {
                instance.modelPath = *modelPath;
            }
            const Vec3 worldPos = bethesdaToEngine(ref.position[0], ref.position[1], ref.position[2]);
            const Mat3 bethRotation = eulerToMatrixBethesdaOrder(
                ref.rotationRadians[0], ref.rotationRadians[1], ref.rotationRadians[2]);
            const Mat3 engineRotation = makeEngineInstanceRotation(bethRotation);
            writeTransform(instance, worldPos, engineRotation, ref.scale);
            if (const auto collisionIt = m_collisionByStaticFormId.find(ref.baseFormId);
                collisionIt != m_collisionByStaticFormId.end()) {
                for (const NifCollisionTriangle& local : collisionIt->second) {
                    ImportedSceneCollisionTriangle world;
                    for (int point = 0; point < 3; ++point) {
                        for (int row = 0; row < 3; ++row) {
                            world.vertices[(point * 3) + row] =
                                instance.transform[(row * 4) + 0] *
                                    local.vertices[(point * 3) + 0] +
                                instance.transform[(row * 4) + 1] *
                                    local.vertices[(point * 3) + 1] +
                                instance.transform[(row * 4) + 2] *
                                    local.vertices[(point * 3) + 2] +
                                instance.transform[(row * 4) + 3];
                        }
                    }
                    m_scene.collisionTriangles.push_back(world);
                }
            }
            m_scene.instances.push_back(instance);
            ++m_stats.placedInstances;

            // Skyrim's mill furniture graph instantiates the moving saw and
            // sash at the lumber-mill reference's origin; neither component is
            // a separate REFR in Skyrim.esm. A passive world viewer has no
            // Papyrus/Havok behavior graph to perform that spawn, so reproduce
            // this one closed composition explicitly and feed the authored
            // Activate tracks through the same rigid-animation path as the
            // separately placed water wheel.
            if (toLowerAsciiCopy(instance.modelPath) ==
                "architecture\\farmhouse\\lumbermill01.nif") {
                FalloutCellRecord components{};
                for (const std::uint32_t componentBase : {0xfff00001u, 0xfff00002u}) {
                    FalloutPlacedReference componentRef = ref;
                    componentRef.baseFormId = componentBase;
                    componentRef.formId = ref.formId ^ componentBase;
                    components.references.push_back(std::move(componentRef));
                }
                addCellStatics(components);
            }
    }
}

void CellSceneBuilder::finish(ImportedScene& outScene) {
    // Bethesda normally places a LIGH beside each stationary fire. Preserve
    // that authored light instead of doubling it; synthesize a clustered,
    // flickering fallback only for effects that have no nearby light. This is
    // also what lets sparse Oblivion cells get emissive fire without making a
    // light-rich Skyrim interior twice as bright.
    for (const ImportedSceneParticleEmitter& emitter : m_scene.particleEmitters) {
        bool hasNearbyAuthoredLight = false;
        for (const ImportedSceneLight& light : m_scene.lights) {
            const float dx = light.position[0] - emitter.position[0];
            const float dy = light.position[1] - emitter.position[1];
            const float dz = light.position[2] - emitter.position[2];
            if ((dx * dx) + (dy * dy) + (dz * dz) <= 320.0f * 320.0f) {
                hasNearbyAuthoredLight = true;
                break;
            }
        }
        if (hasNearbyAuthoredLight) {
            continue;
        }
        ImportedSceneLight light{};
        light.sourceId = emitter.sourceId + "_firelight";
        light.position[0] = emitter.position[0];
        light.position[1] = emitter.position[1] + 24.0f;
        light.position[2] = emitter.position[2];
        light.color[0] = 1.0f;
        light.color[1] = 0.24f;
        light.color[2] = 0.045f;
        light.radius = 440.0f;
        light.intensity = 1.15f;
        light.flags = 0x08u;
        m_scene.lights.push_back(std::move(light));
        ++m_stats.lightsPlaced;
    }
    m_scene.sourceLightCount = static_cast<std::uint32_t>(m_scene.lights.size());
    m_scene.sourceParticleEmitterCount =
        static_cast<std::uint32_t>(m_scene.particleEmitters.size());
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
