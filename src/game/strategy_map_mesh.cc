#include "game/strategy_map_mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace odai::game {

namespace {

using odai::importer::ImportedScene;
using odai::importer::ImportedScenePackedDraw;
using odai::importer::ImportedScenePackedVertex;
using odai::math::Vector3;

constexpr std::uint32_t kInvalidTextureIndex = 0xffffffffu;

// Key for the per-corner vertex deduplication map.
// Includes both world-space position (quantised) and terrain texture index so
// two adjacent tiles that share a physical corner but render different terrain
// textures each get their own vertex — preventing the first tile's texture from
// bleeding onto its neighbour's geometry.
struct CornerTexKey {
    std::uint64_t pos;
    std::uint32_t tex;
    bool operator==(const CornerTexKey& o) const noexcept {
        return pos == o.pos && tex == o.tex;
    }
};
struct CornerTexKeyHash {
    std::size_t operator()(const CornerTexKey& k) const noexcept {
        return k.pos ^ (static_cast<std::uint64_t>(k.tex + 1u) * 0x9e3779b97f4a7c15ULL);
    }
};

// Distinct, muted faction colors for territory borders and settlement markers.
constexpr std::array<TileColor, 8> kOwnerColors = {{
    {0.78f, 0.24f, 0.22f},  // 0 (also used as "unowned" border fallback).
    {0.24f, 0.40f, 0.78f},
    {0.86f, 0.70f, 0.20f},
    {0.30f, 0.66f, 0.36f},
    {0.66f, 0.32f, 0.72f},
    {0.86f, 0.46f, 0.18f},
    {0.24f, 0.68f, 0.70f},
    {0.80f, 0.42f, 0.58f},
}};

TileColor ownerColor(std::uint8_t owner) {
    return kOwnerColors[owner % kOwnerColors.size()];
}

TileColor blend(const TileColor& a, const TileColor& b, float t) {
    return {
        a.r + ((b.r - a.r) * t),
        a.g + ((b.g - a.g) * t),
        a.b + ((b.b - a.b) * t),
    };
}

// Terrain color with river/road overlays folded in, so the slice shows the flags
// without extra geometry.
TileColor tileTopColor(const MapTile& tile) {
    TileColor color = terrainColor(tile.terrain);
    if ((tile.flags & TileFlag_River) != 0u) {
        color = blend(color, {0.18f, 0.42f, 0.74f}, 0.55f);
    }
    if ((tile.flags & TileFlag_Road) != 0u) {
        color = blend(color, {0.50f, 0.38f, 0.24f}, 0.55f);
    }
    return color;
}

Vector3 faceNormal(const Vector3& a, const Vector3& b, const Vector3& c) {
    return odai::math::normalize(odai::math::cross(b - a, c - a));
}

// Accumulates packed geometry and tracks scene bounds.
struct MeshBuilder {
    ImportedScene& scene;

    explicit MeshBuilder(ImportedScene& target) : scene(target) {}

    std::uint32_t addVertex(
        const Vector3& position,
        const Vector3& normal,
        const TileColor& color,
        float u = 0.0f,
        float v = 0.0f,
        std::uint32_t textureIndex = kInvalidTextureIndex)
    {
        ImportedScenePackedVertex vertex{};
        vertex.position[0] = position.x;
        vertex.position[1] = position.y;
        vertex.position[2] = position.z;
        vertex.normal[0] = normal.x;
        vertex.normal[1] = normal.y;
        vertex.normal[2] = normal.z;
        vertex.color[0] = color.r;
        vertex.color[1] = color.g;
        vertex.color[2] = color.b;
        vertex.uv[0] = u;
        vertex.uv[1] = v;
        vertex.textureIndex = textureIndex;
        vertex.flags = 0u;
        const auto index = static_cast<std::uint32_t>(scene.packedVertices.size());
        scene.packedVertices.push_back(vertex);
        expandBounds(position);
        return index;
    }

    void addTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        scene.packedIndices.push_back(a);
        scene.packedIndices.push_back(b);
        scene.packedIndices.push_back(c);
    }

    // Convenience for a flat-shaded triangle with a computed normal.
    void addFlatTriangle(const Vector3& a, const Vector3& b, const Vector3& c, const TileColor& color) {
        const Vector3 normal = faceNormal(a, b, c);
        addTriangle(addVertex(a, normal, color), addVertex(b, normal, color), addVertex(c, normal, color));
    }

    void addQuad(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d, const TileColor& color) {
        addFlatTriangle(a, b, c, color);
        addFlatTriangle(a, c, d, color);
    }

    void expandBounds(const Vector3& p) {
        scene.boundsMin[0] = std::min(scene.boundsMin[0], p.x);
        scene.boundsMin[1] = std::min(scene.boundsMin[1], p.y);
        scene.boundsMin[2] = std::min(scene.boundsMin[2], p.z);
        scene.boundsMax[0] = std::max(scene.boundsMax[0], p.x);
        scene.boundsMax[1] = std::max(scene.boundsMax[1], p.y);
        scene.boundsMax[2] = std::max(scene.boundsMax[2], p.z);
    }

    // Close out the indices appended since `firstIndex` as one draw.
    void finishDraw(std::uint32_t firstIndex) {
        const auto indexCount = static_cast<std::uint32_t>(scene.packedIndices.size() - firstIndex);
        if (indexCount != 0u) {
            scene.packedDraws.push_back(ImportedScenePackedDraw{firstIndex, indexCount});
        }
    }
};

}  // namespace

static ImportedScene buildStrategyMapSceneImpl(const StrategyMap& map,
                                               const std::vector<Unit>& units,
                                               StrategyMapMeshOptions options) {
    ImportedScene scene{};
    scene.sourceTag = "strategy_map";
    scene.boundsMin[0] = scene.boundsMin[1] = scene.boundsMin[2] = std::numeric_limits<float>::max();
    scene.boundsMax[0] = scene.boundsMax[1] = scene.boundsMax[2] = std::numeric_limits<float>::lowest();

    if (map.width == 0 || map.height == 0 || map.tiles.empty()) {
        scene.boundsMin[0] = scene.boundsMin[1] = scene.boundsMin[2] = 0.0f;
        scene.boundsMax[0] = scene.boundsMax[1] = scene.boundsMax[2] = 0.0f;
        return scene;
    }

    // Resolve class -> scene-texture-index (shared with the hex-terrain builder), then
    // move each unique texture into scene.textures in that index order.
    const std::vector<std::uint32_t> terrainToTexIdx = terrainTextureSceneIndices(options.terrainTextures);
    {
        const std::size_t inputCount = std::min(options.terrainTextures.size(), terrainToTexIdx.size());
        for (std::size_t i = 0; i < inputCount; ++i) {
            // First occurrence of a unique texture: its scene index equals the current
            // count, so push it now; duplicates already point at an earlier entry.
            if (terrainToTexIdx[i] != kInvalidTextureIndex &&
                terrainToTexIdx[i] == static_cast<std::uint32_t>(scene.textures.size())) {
                scene.textures.push_back(std::move(options.terrainTextures[i]));
            }
        }
    }

    MeshBuilder builder(scene);
    const Vector3 up{0.0f, 1.0f, 0.0f};
    const std::uint32_t chunkSize = std::max<std::uint32_t>(options.chunkSize, 1u);
    const bool extruded = options.extruded;

    // Single-level board: all land sits one step above a single sea level; per-tile
    // elevation drives only color, not height, so the map reads as one plateau over a
    // flat continuous sea (3D) or a flat board (2D). Heights scale with hexSize.
    const float H = map.hexSize;
    // World-space UV scale: one texture repeat per hexSize so terrain tiles naturally.
    const float invHexSize = (H > 0.0f) ? (1.0f / H) : 0.0f;
    const float seaFloorY = 0.0f;
    const float waterSurfaceY = extruded ? 0.06f * H : 0.04f * H;
    const float baseY = extruded ? -0.20f * H : 0.0f;
    const float seaDepthPerStep = extruded ? 0.20f * H : 0.0f;
    const float seaBaseY = extruded ? -0.85f * H : 0.0f;
    // Lowest land sits just above the water surface so coastlines read as raised plateaus.
    const float landFloorY = 0.12f * H;

    const auto seaFloorAt = [&](const MapTile& t) -> float {
        const int stepsBelowCoast = 1 - std::clamp<int>(t.elevation, -2, 1);
        return seaFloorY - (static_cast<float>(stepsBelowCoast) * seaDepthPerStep);
    };

    // Elevation range over land tiles only (drives the welded continuous surface).
    int minElev = std::numeric_limits<int>::max();
    int maxElev = std::numeric_limits<int>::min();
    for (const MapTile& tile : map.tiles) {
        if (!terrainIsWater(tile.terrain)) {
            minElev = std::min(minElev, static_cast<int>(tile.elevation));
            maxElev = std::max(maxElev, static_cast<int>(tile.elevation));
        }
    }
    if (minElev > maxElev) { minElev = maxElev = 0; }
    const float elevRange = static_cast<float>(std::max(1, maxElev - minElev));
    const float reliefWorld =
        static_cast<float>(maxElev - minElev) * map.elevationStep * options.heightExaggeration;

    // Top-face Y for a tile: elevation-scaled for land, depth-sunk sea floor for water.
    const auto tileTopY = [&](const MapTile& t) -> float {
        if (!extruded) return 0.0f;
        if (terrainIsWater(t.terrain)) return seaFloorAt(t);
        const float norm =
            (static_cast<float>(t.elevation) - static_cast<float>(minElev)) / elevRange;
        return landFloorY + std::clamp(norm, 0.0f, 1.0f) * reliefWorld;
    };
    const auto atY = [](Vector3 v, float y) -> Vector3 { v.y = y; return v; };

    // Weld: for each land-tile corner world-XZ, accumulate heights from every tile
    // that claims that corner (2-3 tiles per interior corner). The averaged height
    // makes adjacent tiles agree on corner Y so the surface is crack-free and reads
    // as one continuous terrain instead of isolated per-tile plateaus.
    // Water tiles are intentionally excluded so coastline corners are only pulled by
    // the land-tile height (not dragged down to sea-floor level).
    // Quantise world-XZ to a 64-bit key for corner deduplication. Precision is
    // relative to hexSize (8 steps per hex) so the key stays valid regardless of
    // the map scale — unlike a fixed 1/8-unit grid, which breaks at small hexSize.
    auto cornerKey64 = [invHexSize](float x, float z) -> std::uint64_t {
        const auto ix = static_cast<std::uint32_t>(static_cast<std::int32_t>(x * invHexSize * 8.0f));
        const auto iz = static_cast<std::uint32_t>(static_cast<std::int32_t>(z * invHexSize * 8.0f));
        return static_cast<std::uint64_t>(ix) | (static_cast<std::uint64_t>(iz) << 32);
    };
    struct CornerAccum { float sum = 0.0f; int count = 0; };
    std::unordered_map<std::uint64_t, CornerAccum> cornerHeightMap;
    cornerHeightMap.reserve(map.width * map.height * 3u);
    for (std::uint32_t r = 0; r < map.height; ++r) {
        for (std::uint32_t c = 0; c < map.width; ++c) {
            const MapTile& t = map.at(c, r);
            if (terrainIsWater(t.terrain)) { continue; }
            // Hidden land tiles render as a flat dark surface (not welded terrain),
            // so exclude them from height averaging to avoid dragging adjacent
            // visible tiles' shared corners down to sea level.
            if (options.fogOfWar && t.visibility == TileVisibility::Hidden) { continue; }
            const float y = tileTopY(t);
            for (int k = 0; k < 6; ++k) {
                const Vector3 cp = tileCornerWorld(map, c, r, k);
                auto& acc = cornerHeightMap[cornerKey64(cp.x, cp.z)];
                acc.sum += y;
                acc.count++;
            }
        }
    }
    const auto avgCornerY = [&](float x, float z, float fallback) -> float {
        const auto it = cornerHeightMap.find(cornerKey64(x, z));
        return (it != cornerHeightMap.end()) ? (it->second.sum / it->second.count) : fallback;
    };

    // Global corner vertex deduplication: a corner shared by adjacent tiles (and
    // chunks) is stored once in scene.packedVertices, referenced by both draws.
    // Smooth normals (computed after all terrain triangles are emitted) will then
    // accumulate contributions from all triangles sharing each vertex.
    // The key includes terrainTexIdx so different-terrain neighbours each get their
    // own vertex — preventing the first tile's texture from bleeding onto adjacent
    // terrain of a different type at shared boundary corners.
    std::unordered_map<CornerTexKey, std::uint32_t, CornerTexKeyHash> cornerVertexMap;
    cornerVertexMap.reserve(map.width * map.height * 3u);

    // Track the terrain vertex/index range for the post-pass normal computation.
    const auto terrainVertBase = static_cast<std::uint32_t>(scene.packedVertices.size());
    const auto terrainIdxBase  = static_cast<std::uint32_t>(scene.packedIndices.size());

    // Per-chunk axis-aligned bounds, accumulated as a chunk's tiles are emitted, so
    // the renderer can frustum-cull each chunk page independently.
    struct ChunkBounds {
        Vector3 lo{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        Vector3 hi{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
        void expand(const Vector3& p) {
            lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
            hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
        }
    };

    // --- Terrain: one packed draw + one cullable page per NxN chunk. ---
    const std::uint32_t chunkCols = (map.width + chunkSize - 1u) / chunkSize;
    const std::uint32_t chunkRows = (map.height + chunkSize - 1u) / chunkSize;
    for (std::uint32_t chunkRow = 0; chunkRow < chunkRows; ++chunkRow) {
        for (std::uint32_t chunkCol = 0; chunkCol < chunkCols; ++chunkCol) {
            const std::uint32_t firstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
            const std::size_t drawsBefore = scene.packedDraws.size();
            ChunkBounds bounds{};

            const std::uint32_t rowEnd = std::min(map.height, (chunkRow + 1u) * chunkSize);
            const std::uint32_t colEnd = std::min(map.width, (chunkCol + 1u) * chunkSize);
            for (std::uint32_t row = chunkRow * chunkSize; row < rowEnd; ++row) {
                for (std::uint32_t col = chunkCol * chunkSize; col < colEnd; ++col) {
                    const MapTile& tile = map.at(col, row);
                    // When the GPU hex-terrain path owns the land, skip land tops/skirts
                    // here; water tiles still emit their sea-floor + water skirts.
                    if (!options.emitLandSurface && !terrainIsWater(tile.terrain)) {
                        continue;
                    }
                    // Hidden land tiles: emit a flat dark shroud at just above the
                    // water surface so the player sees featureless darkness — no
                    // elevation, no terrain type, no hex outline (grid skips them).
                    if (options.fogOfWar && !terrainIsWater(tile.terrain) &&
                        tile.visibility == TileVisibility::Hidden) {
                        constexpr TileColor kShroud{0.03f, 0.03f, 0.05f};
                        const float shroudY = waterSurfaceY + std::max(H * 0.005f, 0.2f);
                        const Vector3 sc = atY(tileCenterWorld(map, col, row), shroudY);
                        bounds.expand(sc);
                        const std::uint32_t ci = builder.addVertex(sc, up, kShroud);
                        std::array<std::uint32_t, 6> si{};
                        for (int k = 0; k < 6; ++k) {
                            const Vector3 cp = atY(tileCornerWorld(map, col, row, k), shroudY);
                            bounds.expand(cp);
                            si[static_cast<std::size_t>(k)] = builder.addVertex(cp, up, kShroud);
                        }
                        for (int k = 0; k < 6; ++k) {
                            const std::size_t nxt = static_cast<std::size_t>((k + 1) % 6);
                            builder.addTriangle(ci, si[nxt], si[static_cast<std::size_t>(k)]);
                        }
                        continue;
                    }

                    TileColor topColor = tileTopColor(tile);
                    if (options.fogOfWar && tile.visibility == TileVisibility::Explored) {
                        const float luma = topColor.r * 0.299f
                                         + topColor.g * 0.587f
                                         + topColor.b * 0.114f;
                        topColor = {luma * 0.38f, luma * 0.40f, luma * 0.48f};
                    }
                    const float topY = tileTopY(tile);
                    const Vector3 center = atY(tileCenterWorld(map, col, row), topY);

                    // Land corners are welded: use the mean height across all tiles that
                    // share each corner so adjacent tiles agree on corner Y (no cracks).
                    // Water corners keep the tile's own sea-floor Y (not averaged with land).
                    std::array<Vector3, 6> corners{};
                    for (int corner = 0; corner < 6; ++corner) {
                        const Vector3 rawCorner = tileCornerWorld(map, col, row, corner);
                        const float cornerY = terrainIsWater(tile.terrain)
                            ? topY
                            : avgCornerY(rawCorner.x, rawCorner.z, topY);
                        corners[static_cast<std::size_t>(corner)] = atY(rawCorner, cornerY);
                    }

                    // Top face as a fan from the center.
                    const std::uint32_t terrainTexIdx =
                        static_cast<std::size_t>(tile.terrain) < terrainToTexIdx.size()
                        ? terrainToTexIdx[static_cast<std::size_t>(tile.terrain)]
                        : kInvalidTextureIndex;
                    bounds.expand(center);
                    const std::uint32_t centerIndex = builder.addVertex(
                        center, up, topColor,
                        center.x * invHexSize, center.z * invHexSize,
                        terrainTexIdx);
                    std::array<std::uint32_t, 6> cornerIndices{};
                    for (int corner = 0; corner < 6; ++corner) {
                        const Vector3& cp = corners[static_cast<std::size_t>(corner)];
                        bounds.expand(cp);
                        if (!terrainIsWater(tile.terrain)) {
                            // Land: deduplicate so the vertex is shared with adjacent tiles
                            // of the same terrain type. Different-terrain neighbours each
                            // get their own vertex (keyed by texIdx) so the first tile's
                            // texture cannot bleed onto its neighbour at a shared corner.
                            const CornerTexKey ckey{cornerKey64(cp.x, cp.z), terrainTexIdx};
                            const auto it = cornerVertexMap.find(ckey);
                            if (it != cornerVertexMap.end()) {
                                cornerIndices[static_cast<std::size_t>(corner)] = it->second;
                            } else {
                                const std::uint32_t idx = builder.addVertex(
                                    cp, up, topColor,
                                    cp.x * invHexSize, cp.z * invHexSize,
                                    terrainTexIdx);
                                cornerVertexMap[ckey] = idx;
                                cornerIndices[static_cast<std::size_t>(corner)] = idx;
                            }
                        } else {
                            // Water: isolated corners; sea-floor geometry is rarely visible
                            // and not shared across tile boundaries.
                            cornerIndices[static_cast<std::size_t>(corner)] = builder.addVertex(
                                cp, up, topColor,
                                cp.x * invHexSize, cp.z * invHexSize,
                                terrainTexIdx);
                        }
                    }
                    // Wind the top fan so the face points up (+Y).
                    for (int corner = 0; corner < 6; ++corner) {
                        const std::size_t next = static_cast<std::size_t>((corner + 1) % 6);
                        builder.addTriangle(centerIndex, cornerIndices[next], cornerIndices[static_cast<std::size_t>(corner)]);
                    }

                    // Side skirts down to the slab base give the plateau/coast cliffs.
                    // Backface culling drops the inner of any two coincident skirts, so
                    // only the exposed land-against-water (and map-edge) faces show.
                    // 2D board mode is flat, so skirts are skipped.
                    if (extruded) {
                        // Water skirts drop to the deep sea base so depth steps between
                        // adjacent sea-floor tiles form continuous underwater walls
                        // rather than gaps; land keeps the shallower slab base.
                        const float skirtBottomY = terrainIsWater(tile.terrain) ? seaBaseY : baseY;
                        const TileColor skirtColor = blend(topColor, {0.0f, 0.0f, 0.0f}, 0.45f);
                        for (int corner = 0; corner < 6; ++corner) {
                            const Vector3 topA = corners[static_cast<std::size_t>(corner)];
                            const Vector3 topB = corners[static_cast<std::size_t>((corner + 1) % 6)];
                            const Vector3 bottomA{topA.x, skirtBottomY, topA.z};
                            const Vector3 bottomB{topB.x, skirtBottomY, topB.z};
                            bounds.expand(bottomA);
                            builder.addQuad(topA, topB, bottomB, bottomA, skirtColor);
                        }
                    }
                }
            }

            builder.finishDraw(firstIndex);
            // finishDraw appends one draw only when the chunk emitted geometry.
            if (scene.packedDraws.size() > drawsBefore) {
                odai::importer::ImportedScenePageRange page{};
                page.firstDraw = static_cast<std::uint32_t>(drawsBefore);
                page.drawCount = 1u;
                page.terrainDrawCount = 1u;
                page.boundsMin[0] = bounds.lo.x; page.boundsMin[1] = bounds.lo.y; page.boundsMin[2] = bounds.lo.z;
                page.boundsMax[0] = bounds.hi.x; page.boundsMax[1] = bounds.hi.y; page.boundsMax[2] = bounds.hi.z;
                scene.pageRanges.push_back(page);
            }
        }
    }
    // --- Post-pass: smooth per-vertex normals for all terrain geometry. ---
    // Each vertex accumulates the area-weighted cross-products of every triangle
    // that references it. After normalization, shared corner vertices get a proper
    // blend of their neighbouring faces (smooth shading at elevation transitions)
    // and the hex fan centre gets a stable near-up normal from all 6 fan triangles.
    // This eliminates the tessellation-path topology singularity entirely.
    {
        const auto terrainVertEnd = static_cast<std::uint32_t>(scene.packedVertices.size());
        const auto terrainIdxEnd  = static_cast<std::uint32_t>(scene.packedIndices.size());
        const std::uint32_t nVerts = terrainVertEnd - terrainVertBase;
        if (nVerts > 0u) {
            std::vector<float> normAccum(static_cast<std::size_t>(nVerts) * 3u, 0.0f);
            for (std::uint32_t i = terrainIdxBase; i + 2u < terrainIdxEnd; i += 3u) {
                const std::uint32_t i0 = scene.packedIndices[i];
                const std::uint32_t i1 = scene.packedIndices[i + 1u];
                const std::uint32_t i2 = scene.packedIndices[i + 2u];
                if (i0 < terrainVertBase || i0 >= terrainVertEnd ||
                    i1 < terrainVertBase || i1 >= terrainVertEnd ||
                    i2 < terrainVertBase || i2 >= terrainVertEnd) { continue; }
                const auto& v0 = scene.packedVertices[i0];
                const auto& v1 = scene.packedVertices[i1];
                const auto& v2 = scene.packedVertices[i2];
                const float ax = v1.position[0] - v0.position[0];
                const float ay = v1.position[1] - v0.position[1];
                const float az = v1.position[2] - v0.position[2];
                const float bx = v2.position[0] - v0.position[0];
                const float by = v2.position[1] - v0.position[1];
                const float bz = v2.position[2] - v0.position[2];
                const float nx = (ay * bz) - (az * by);
                const float ny = (az * bx) - (ax * bz);
                const float nz = (ax * by) - (ay * bx);
                const std::array<std::uint32_t, 3> vis = {i0, i1, i2};
                for (std::uint32_t vi : vis) {
                    const std::uint32_t off = (vi - terrainVertBase) * 3u;
                    normAccum[off]      += nx;
                    normAccum[off + 1u] += ny;
                    normAccum[off + 2u] += nz;
                }
            }
            for (std::uint32_t vi = terrainVertBase; vi < terrainVertEnd; ++vi) {
                const std::uint32_t off = (vi - terrainVertBase) * 3u;
                const float nx = normAccum[off];
                const float ny = normAccum[off + 1u];
                const float nz = normAccum[off + 2u];
                const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                auto& v = scene.packedVertices[vi];
                if (len > 1e-6f) {
                    v.normal[0] = nx / len;
                    v.normal[1] = ny / len;
                    v.normal[2] = nz / len;
                } else {
                    v.normal[0] = 0.0f;
                    v.normal[1] = 1.0f;
                    v.normal[2] = 0.0f;
                }
            }
        }
    }

    // Terrain chunk draws occupy the contiguous prefix [0, terrainDrawCount); the
    // renderer keys its terrain/static split off sourceLandscapeCellCount.
    const std::uint32_t terrainDrawCount = static_cast<std::uint32_t>(scene.packedDraws.size());
    scene.sourceLandscapeCellCount = terrainDrawCount;

    // --- Water: flat quads over ocean/coast tiles for the animated water shader. ---
    // Every patch sits at the SAME waterSurfaceY so adjacent quads are coplanar and
    // read as one continuous sea (the per-tile bounding squares overlap, which is fine
    // when coplanar). Land plateaus rise above this surface for the coastline.
    if (options.emitWaterPatches) {
        const float halfWidth = 0.5f * std::sqrt(3.0f) * map.hexSize;  // pointy-top x extent / 2
        const float halfHeight = map.hexSize;                          // pointy-top z extent / 2
        for (std::uint32_t row = 0; row < map.height; ++row) {
            for (std::uint32_t col = 0; col < map.width; ++col) {
                const MapTile& tile = map.at(col, row);
                if (!terrainIsWater(tile.terrain)) {
                    continue;
                }
                const Vector3 center = tileCenterWorld(map, col, row);
                odai::importer::ImportedSceneWaterPatch patch{};
                patch.originX = center.x - halfWidth;
                patch.originZ = center.z - halfHeight;
                patch.sizeX = 2.0f * halfWidth;
                patch.sizeZ = 2.0f * halfHeight;
                patch.waterLevel = waterSurfaceY;
                scene.waterPatches.push_back(patch);
            }
        }
    }

    // --- Grid overlay (thin edge quads, border edges colored by owner). ---
    if (options.drawGridOverlay) {
        const auto gridFirstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
        const float halfWidth = options.gridLineWidth * 0.5f;
        const float lift = std::max(H * 0.008f, 0.3f);  // just clear of the tile top
        const TileColor darkTint{0.06f, 0.07f, 0.08f};
        for (std::uint32_t row = 0; row < map.height; ++row) {
            for (std::uint32_t col = 0; col < map.width; ++col) {
                const MapTile& tile = map.at(col, row);
                // Skip water tiles and Hidden land tiles: water has the animated
                // surface above it; Hidden tiles get the flat shroud (no hex outline).
                if (terrainIsWater(tile.terrain)) { continue; }
                if (options.fogOfWar && tile.visibility == TileVisibility::Hidden) { continue; }
                const bool isBorder = (tile.flags & TileFlag_Border) != 0u;
                const TileColor tileColor = tileTopColor(tile);
                // Extremely subtle: darken the tile's own color just 8-12%, with a
                // light faction tint on territory borders. Borders blend in rather
                // than drawing as separate dark lines.
                const TileColor lineColor = isBorder
                    ? blend(blend(tileColor, darkTint, 0.12f), ownerColor(tile.owner), 0.18f)
                    : blend(tileColor, darkTint, 0.08f);
                const float topY = tileTopY(tile);
                for (int corner = 0; corner < 6; ++corner) {
                    const Vector3 rawA = tileCornerWorld(map, col, row, corner);
                    const Vector3 rawB = tileCornerWorld(map, col, row, (corner + 1) % 6);
                    Vector3 a{rawA.x, avgCornerY(rawA.x, rawA.z, topY) + lift, rawA.z};
                    Vector3 b{rawB.x, avgCornerY(rawB.x, rawB.z, topY) + lift, rawB.z};
                    const Vector3 edge = odai::math::normalize(b - a);
                    const Vector3 side = odai::math::cross(up, edge) * halfWidth;
                    builder.addQuad(a - side, b - side, b + side, a + side, lineColor);
                }
            }
        }
        builder.finishDraw(gridFirstIndex);
    }

    // --- Draw 2: settlement markers (box pillars colored by owner / tier). ---
    if (options.drawSettlements && !map.settlements.empty()) {
        const auto settlementFirstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
        for (const Settlement& settlement : map.settlements) {
            if (!map.inBounds(static_cast<int>(settlement.col), static_cast<int>(settlement.row))) {
                continue;
            }
            const MapTile& tile = map.at(settlement.col, settlement.row);
            // Don't reveal settlements the player hasn't discovered.
            if (options.fogOfWar && tile.visibility == TileVisibility::Hidden) { continue; }
            const Vector3 center = atY(tileCenterWorld(map, settlement.col, settlement.row), tileTopY(tile));
            const float halfExtent = map.hexSize * 0.35f;
            const float height = options.settlementHeight * static_cast<float>(std::max<std::uint8_t>(settlement.tier, 1));
            const TileColor color = ownerColor(settlement.owner);

            const float minX = center.x - halfExtent;
            const float maxX = center.x + halfExtent;
            const float minZ = center.z - halfExtent;
            const float maxZ = center.z + halfExtent;
            const float minY = center.y;
            const float maxY = center.y + height;

            const Vector3 corners[8] = {
                {minX, minY, minZ}, {maxX, minY, minZ}, {maxX, minY, maxZ}, {minX, minY, maxZ},
                {minX, maxY, minZ}, {maxX, maxY, minZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ},
            };
            // Six faces wound counter-clockwise when viewed from outside.
            builder.addQuad(corners[4], corners[5], corners[6], corners[7], color);  // top
            builder.addQuad(corners[3], corners[2], corners[1], corners[0], color);  // bottom
            builder.addQuad(corners[0], corners[1], corners[5], corners[4], color);  // -Z
            builder.addQuad(corners[2], corners[3], corners[7], corners[6], color);  // +Z
            builder.addQuad(corners[3], corners[0], corners[4], corners[7], color);  // -X
            builder.addQuad(corners[1], corners[2], corners[6], corners[5], color);  // +X
        }
        builder.finishDraw(settlementFirstIndex);
    }

    // --- Draw 3: unit tokens (short boxes colored by owner, reddened by wounds). ---
    if (!units.empty()) {
        const auto unitFirstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
        for (const Unit& unit : units) {
            if (!unit.alive() ||
                !map.inBounds(static_cast<int>(unit.col), static_cast<int>(unit.row))) {
                continue;
            }
            const MapTile& tile = map.at(unit.col, unit.row);
            // Hide enemy units on tiles that aren't currently visible to the player.
            if (options.fogOfWar && options.playerOwner != 0 &&
                unit.owner != options.playerOwner &&
                tile.visibility != TileVisibility::Visible) {
                continue;
            }
            const Vector3 base = atY(tileCenterWorld(map, unit.col, unit.row), tileTopY(tile));
            const float halfExtent = map.hexSize * 0.22f;
            const float boxHeight = map.hexSize * 0.5f;
            const float lift = std::max(H * 0.02f, 0.5f);  // clear of the tile top + grid lines

            // Owner color, tinted toward red as the unit's health falls.
            const float hpFrac = unit.maxHp > 0
                ? std::clamp(static_cast<float>(unit.hp) / static_cast<float>(unit.maxHp), 0.0f, 1.0f)
                : 1.0f;
            const TileColor color = blend(ownerColor(unit.owner), {0.85f, 0.05f, 0.05f}, (1.0f - hpFrac) * 0.7f);

            const float minX = base.x - halfExtent;
            const float maxX = base.x + halfExtent;
            const float minZ = base.z - halfExtent;
            const float maxZ = base.z + halfExtent;
            const float minY = base.y + lift;
            const float maxY = base.y + lift + boxHeight;
            const Vector3 corners[8] = {
                {minX, minY, minZ}, {maxX, minY, minZ}, {maxX, minY, maxZ}, {minX, minY, maxZ},
                {minX, maxY, minZ}, {maxX, maxY, minZ}, {maxX, maxY, maxZ}, {minX, maxY, maxZ},
            };
            builder.addQuad(corners[4], corners[5], corners[6], corners[7], color);  // top
            builder.addQuad(corners[3], corners[2], corners[1], corners[0], color);  // bottom
            builder.addQuad(corners[0], corners[1], corners[5], corners[4], color);  // -Z
            builder.addQuad(corners[2], corners[3], corners[7], corners[6], color);  // +Z
            builder.addQuad(corners[3], corners[0], corners[4], corners[7], color);  // -X
            builder.addQuad(corners[1], corners[2], corners[6], corners[5], color);  // +X
        }
        builder.finishDraw(unitFirstIndex);
    }

    // --- Draw N: observation fort markers (narrow stone watchtowers). ---
    {
        const auto fortFirstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
        const TileColor fortWall{0.50f, 0.47f, 0.44f};
        const TileColor fortTop{0.60f, 0.58f, 0.54f};
        for (std::uint32_t row = 0; row < map.height; ++row) {
            for (std::uint32_t col = 0; col < map.width; ++col) {
                const MapTile& tile = map.at(col, row);
                if (!(tile.flags & TileFlag_Fort)) continue;
                if (options.fogOfWar && tile.visibility == TileVisibility::Hidden) continue;
                const Vector3 base = atY(tileCenterWorld(map, col, row), tileTopY(tile));
                const float he = map.hexSize * 0.12f;  // narrow footprint
                const float towerH = map.hexSize * 1.1f;
                const float lift = std::max(H * 0.02f, 0.5f);
                const float minX = base.x - he;   const float maxX = base.x + he;
                const float minZ = base.z - he;   const float maxZ = base.z + he;
                const float minY = base.y + lift;  const float maxY = base.y + lift + towerH;
                const Vector3 fc[8] = {
                    {minX,minY,minZ},{maxX,minY,minZ},{maxX,minY,maxZ},{minX,minY,maxZ},
                    {minX,maxY,minZ},{maxX,maxY,minZ},{maxX,maxY,maxZ},{minX,maxY,maxZ},
                };
                builder.addQuad(fc[4],fc[5],fc[6],fc[7], fortTop);   // top
                builder.addQuad(fc[3],fc[2],fc[1],fc[0], fortWall);  // bottom
                builder.addQuad(fc[0],fc[1],fc[5],fc[4], fortWall);  // -Z
                builder.addQuad(fc[2],fc[3],fc[7],fc[6], fortWall);  // +Z
                builder.addQuad(fc[3],fc[0],fc[4],fc[7], fortWall);  // -X
                builder.addQuad(fc[1],fc[2],fc[6],fc[5], fortWall);  // +X
            }
        }
        builder.finishDraw(fortFirstIndex);
    }

    // Grid overlay + settlement draws are not spatially chunked; cover them with a
    // single whole-map page so the renderer's cull validation accepts the page set
    // (it discards ALL pages unless every non-empty draw is covered). Whole-map
    // bounds mean this page always passes the frustum test, so overlays never cull.
    const auto overlayDrawEnd = static_cast<std::uint32_t>(scene.packedDraws.size());
    if (!scene.pageRanges.empty() && overlayDrawEnd > terrainDrawCount) {
        odai::importer::ImportedScenePageRange overlay{};
        overlay.firstDraw = terrainDrawCount;
        overlay.drawCount = overlayDrawEnd - terrainDrawCount;
        overlay.terrainDrawCount = 0u;
        for (int i = 0; i < 3; ++i) {
            overlay.boundsMin[i] = scene.boundsMin[i];
            overlay.boundsMax[i] = scene.boundsMax[i];
        }
        scene.pageRanges.push_back(overlay);
    }

    scene.sourceMeshCount = 1u;
    scene.sourceInstanceCount = static_cast<std::uint32_t>(map.settlements.size());
    return scene;
}

std::vector<std::uint32_t> terrainTextureSceneIndices(
    const std::vector<odai::importer::ImportedSceneTexture>& terrainTextures) {
    constexpr std::size_t kTerrainCount = static_cast<std::size_t>(TerrainType::Count);
    std::vector<std::uint32_t> indices(kTerrainCount, kInvalidTextureIndex);
    std::unordered_map<std::string, std::uint32_t> pathToSceneIdx;
    std::uint32_t nextSceneIdx = 0u;
    const std::size_t inputCount = std::min(terrainTextures.size(), kTerrainCount);
    for (std::size_t i = 0; i < inputCount; ++i) {
        const odai::importer::ImportedSceneTexture& tex = terrainTextures[i];
        if (tex.width == 0u || tex.height == 0u || tex.rgba8.empty()) {
            continue;
        }
        const auto it = pathToSceneIdx.find(tex.sourcePath);
        if (it != pathToSceneIdx.end()) {
            indices[i] = it->second;
        } else {
            indices[i] = nextSceneIdx;
            pathToSceneIdx.emplace(tex.sourcePath, nextSceneIdx);
            ++nextSceneIdx;
        }
    }
    return indices;
}

ImportedScene buildStrategyMapScene(const StrategyMap& map, StrategyMapMeshOptions options) {
    return buildStrategyMapSceneImpl(map, {}, std::move(options));
}

ImportedScene buildStrategyMapScene(const StrategyMap& map, const std::vector<Unit>& units,
                                    StrategyMapMeshOptions options) {
    return buildStrategyMapSceneImpl(map, units, std::move(options));
}

}  // namespace odai::game
