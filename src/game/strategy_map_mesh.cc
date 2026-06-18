#include "game/strategy_map_mesh.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

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

    std::uint32_t addVertex(const Vector3& position, const Vector3& normal, const TileColor& color) {
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
        vertex.uv[0] = 0.0f;
        vertex.uv[1] = 0.0f;
        vertex.textureIndex = kInvalidTextureIndex;
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

float lowestWorldY(const StrategyMap& map) {
    std::int16_t minElevation = 0;
    bool any = false;
    for (const MapTile& tile : map.tiles) {
        if (!any || tile.elevation < minElevation) {
            minElevation = tile.elevation;
            any = true;
        }
    }
    return (static_cast<float>(minElevation) * map.elevationStep) - (map.elevationStep * 3.0f);
}

}  // namespace

ImportedScene buildStrategyMapScene(const StrategyMap& map, const StrategyMapMeshOptions& options) {
    ImportedScene scene{};
    scene.sourceTag = "strategy_map";
    scene.boundsMin[0] = scene.boundsMin[1] = scene.boundsMin[2] = std::numeric_limits<float>::max();
    scene.boundsMax[0] = scene.boundsMax[1] = scene.boundsMax[2] = std::numeric_limits<float>::lowest();

    if (map.width == 0 || map.height == 0 || map.tiles.empty()) {
        scene.boundsMin[0] = scene.boundsMin[1] = scene.boundsMin[2] = 0.0f;
        scene.boundsMax[0] = scene.boundsMax[1] = scene.boundsMax[2] = 0.0f;
        return scene;
    }

    MeshBuilder builder(scene);
    const float baseY = lowestWorldY(map);
    const Vector3 up{0.0f, 1.0f, 0.0f};

    // --- Draw 0: terrain prisms (top hex face + side skirts down to the base). ---
    const auto terrainFirstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
    for (std::uint32_t row = 0; row < map.height; ++row) {
        for (std::uint32_t col = 0; col < map.width; ++col) {
            const MapTile& tile = map.at(col, row);
            const TileColor topColor = tileTopColor(tile);
            const Vector3 center = tileCenterWorld(map, col, row);

            std::array<Vector3, 6> corners{};
            for (int corner = 0; corner < 6; ++corner) {
                corners[static_cast<std::size_t>(corner)] = tileCornerWorld(map, col, row, corner);
            }

            // Top face as a fan from the center.
            const std::uint32_t centerIndex = builder.addVertex(center, up, topColor);
            std::array<std::uint32_t, 6> cornerIndices{};
            for (int corner = 0; corner < 6; ++corner) {
                cornerIndices[static_cast<std::size_t>(corner)] = builder.addVertex(corners[static_cast<std::size_t>(corner)], up, topColor);
            }
            for (int corner = 0; corner < 6; ++corner) {
                const std::size_t next = static_cast<std::size_t>((corner + 1) % 6);
                builder.addTriangle(centerIndex, cornerIndices[static_cast<std::size_t>(corner)], cornerIndices[next]);
            }

            // Side skirts so elevation changes read as solid land, not gaps.
            const TileColor skirtColor = blend(topColor, {0.0f, 0.0f, 0.0f}, 0.45f);
            for (int corner = 0; corner < 6; ++corner) {
                const Vector3 topA = corners[static_cast<std::size_t>(corner)];
                const Vector3 topB = corners[static_cast<std::size_t>((corner + 1) % 6)];
                const Vector3 bottomA{topA.x, baseY, topA.z};
                const Vector3 bottomB{topB.x, baseY, topB.z};
                builder.addQuad(topA, topB, bottomB, bottomA, skirtColor);
            }
        }
    }
    builder.finishDraw(terrainFirstIndex);

    // --- Draw 1: grid overlay (thin edge quads, border edges colored by owner). ---
    if (options.drawGridOverlay) {
        const auto gridFirstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
        const float halfWidth = options.gridLineWidth * 0.5f;
        const float lift = std::max(map.elevationStep * 0.25f, 1.0f);
        const TileColor defaultLine{0.06f, 0.06f, 0.07f};
        for (std::uint32_t row = 0; row < map.height; ++row) {
            for (std::uint32_t col = 0; col < map.width; ++col) {
                const MapTile& tile = map.at(col, row);
                const bool isBorder = (tile.flags & TileFlag_Border) != 0u;
                const TileColor lineColor = isBorder ? ownerColor(tile.owner) : defaultLine;
                for (int corner = 0; corner < 6; ++corner) {
                    Vector3 a = tileCornerWorld(map, col, row, corner);
                    Vector3 b = tileCornerWorld(map, col, row, (corner + 1) % 6);
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
            const Vector3 center = tileCenterWorld(map, settlement.col, settlement.row);
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

    scene.sourceMeshCount = 1u;
    scene.sourceInstanceCount = static_cast<std::uint32_t>(map.settlements.size());
    return scene;
}

}  // namespace odai::game
