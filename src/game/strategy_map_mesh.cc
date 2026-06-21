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
    const float landTopY = extruded ? 0.45f * H : 0.0f;       // single land plateau top
    const float seaFloorY = 0.0f;                             // coastal shallows floor (just under the surface)
    // Water is a single continuous surface; the sea FLOOR sinks with depth (below).
    // The land plateau towers above the surface for the "land higher than water" read.
    const float waterSurfaceY = extruded ? 0.06f * H : 0.04f * H;
    const float baseY = extruded ? -0.20f * H : 0.0f;        // land slab bottom for skirts
    // Sea floor drop per elevation step below coast, and the deep slab bottom that
    // the sea-floor skirts descend to (kept below the deepest possible floor).
    const float seaDepthPerStep = extruded ? 0.20f * H : 0.0f;
    const float seaBaseY = extruded ? -0.85f * H : 0.0f;

    // Sea floor sinks with depth so open ocean reads deep-cobalt while coastal
    // shallows stay azure — this is what drives the water shader's depth-based
    // absorption/tint gradient. Coast (elevation 1) sits at the shallow seaFloorY;
    // each elevation step below that drops the floor by seaDepthPerStep.
    const auto seaFloorAt = [&](const MapTile& t) -> float {
        const int stepsBelowCoast = 1 - std::clamp<int>(t.elevation, -2, 1);
        return seaFloorY - (static_cast<float>(stepsBelowCoast) * seaDepthPerStep);
    };
    // Top-face Y for a tile: raised for land, depth-sunk sea floor for water (flat in 2D).
    const auto tileTopY = [&](const MapTile& t) -> float {
        if (!extruded) return 0.0f;
        return terrainIsWater(t.terrain) ? seaFloorAt(t) : landTopY;
    };
    const auto atY = [](Vector3 v, float y) -> Vector3 { v.y = y; return v; };

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
                    const TileColor topColor = tileTopColor(tile);
                    const float topY = tileTopY(tile);
                    const Vector3 center = atY(tileCenterWorld(map, col, row), topY);

                    std::array<Vector3, 6> corners{};
                    for (int corner = 0; corner < 6; ++corner) {
                        corners[static_cast<std::size_t>(corner)] = atY(tileCornerWorld(map, col, row, corner), topY);
                    }

                    // Top face as a fan from the center.
                    // Use world-space XZ as UV (one tile = one repeat) and bind the
                    // terrain texture when one was loaded for this terrain type.
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
                        cornerIndices[static_cast<std::size_t>(corner)] = builder.addVertex(
                            cp, up, topColor,
                            cp.x * invHexSize, cp.z * invHexSize,
                            terrainTexIdx);
                    }
                    // Wind the top fan so the face points up (+Y) — front-facing for the
                    // imported-static pipeline's CCW/back-cull convention.
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
                // Skip water tiles: the animated water surface covers them and
                // thin edge quads create visible artifacts on the reflective surface.
                if (terrainIsWater(tile.terrain)) {
                    continue;
                }
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
                    Vector3 a = atY(tileCornerWorld(map, col, row, corner), topY);
                    Vector3 b = atY(tileCornerWorld(map, col, row, (corner + 1) % 6), topY);
                    a.y += lift;
                    b.y += lift;
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
