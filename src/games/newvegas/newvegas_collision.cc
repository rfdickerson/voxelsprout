#include "games/newvegas/newvegas_collision.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "import/fnv/fallout_records.h"

namespace odai::games::newvegas {

namespace {

constexpr int kTerrainGrid = odai::importer::fnv::kLandGridSize;       // 33
constexpr float kPostSpacing = odai::importer::fnv::kLandPostSpacing;  // 128
constexpr float kCellSize = odai::importer::fnv::kExteriorCellSize;    // 4096

// A cell's engine-space corner. Engine X is Fallout X; engine Z is -Fallout Y,
// so the cell spanning Fallout Y [z, z+1) spans engine Z [-(z+1), -z) -- the
// corner is the MORE NEGATIVE end, not z * size.
void cellEngineOrigin(const importer::CellCoord& cell, float& outX, float& outZ) {
    outX = static_cast<float>(cell.x) * kCellSize;
    outZ = -static_cast<float>(cell.z + 1) * kCellSize;
}

// Closest point on segment ab to p, in 2D.
void closestPointOnSegment(
    float px, float pz, float ax, float az, float bx, float bz, float& outX, float& outZ) {
    const float abx = bx - ax;
    const float abz = bz - az;
    const float lengthSq = (abx * abx) + (abz * abz);
    if (lengthSq <= 1e-6f) {
        outX = ax;
        outZ = az;
        return;
    }
    float t = (((px - ax) * abx) + ((pz - az) * abz)) / lengthSq;
    t = std::clamp(t, 0.0f, 1.0f);
    outX = ax + (abx * t);
    outZ = az + (abz * t);
}

}  // namespace

void CollisionWorld::addCell(
    const importer::CellCoord& cell, const importer::ImportedScene& scene) {
    CellCollision collision;
    cellEngineOrigin(cell, collision.originX, collision.originZ);

    // Terrain, read from the scene's own terrain mesh rather than the LAND
    // record: the mesh is what is drawn, and reading it here means a cell
    // restored from the disk cache produces identical collision to one just
    // built. No second decode path to drift.
    if (!scene.meshes.empty() && scene.meshes[0].name == "terrain" &&
        !scene.meshes[0].vertices.empty()) {
        collision.heights.assign(
            static_cast<std::size_t>(kTerrainGrid) * kTerrainGrid,
            -std::numeric_limits<float>::max());
        for (const importer::ImportedSceneVertex& vertex : scene.meshes[0].vertices) {
            const int gx =
                static_cast<int>(std::lround((vertex.position[0] - collision.originX) / kPostSpacing));
            const int gz =
                static_cast<int>(std::lround((vertex.position[2] - collision.originZ) / kPostSpacing));
            if (gx < 0 || gx >= kTerrainGrid || gz < 0 || gz >= kTerrainGrid) {
                continue;
            }
            float& slot = collision.heights[(static_cast<std::size_t>(gz) * kTerrainGrid) + gx];
            // Quadrants share their middle row and column, so a post can be
            // written twice; keep the highest so a seam is never a hole.
            slot = std::max(slot, vertex.position[1]);
        }
        float fallback = 0.0f;
        for (const float height : collision.heights) {
            if (height != -std::numeric_limits<float>::max()) {
                fallback = height;
                break;
            }
        }
        for (float& height : collision.heights) {
            if (height == -std::numeric_limits<float>::max()) {
                height = fallback;
            }
        }
    }

    // Static geometry as triangles, in world space.
    //
    // Whole-mesh boxes were tried first and are the wrong shape for this world:
    // a rock's box covers the walkable ground around it, so the player hits an
    // invisible wall with open terrain visibly in front of them. Triangles are
    // what let "walk up and over this rock" and "do not pass through this wall"
    // be different answers, which a single box per mesh fundamentally cannot
    // express.
    collision.buckets.assign(kBucketGrid * kBucketGrid, {});
    for (const importer::ImportedSceneInstance& instance : scene.instances) {
        if (instance.meshIndex >= scene.meshes.size()) {
            continue;
        }
        const importer::ImportedSceneMesh& mesh = scene.meshes[instance.meshIndex];
        if (mesh.name == "terrain" || mesh.vertices.empty() || mesh.indices.size() < 3u) {
            continue;  // terrain is a height field, not a triangle soup
        }

        const float* t = instance.transform;
        const auto toWorld = [&](const float local[3], float out[3]) {
            for (int row = 0; row < 3; ++row) {
                out[row] = (t[(row * 4) + 0] * local[0]) + (t[(row * 4) + 1] * local[1]) +
                           (t[(row * 4) + 2] * local[2]) + t[(row * 4) + 3];
            }
        };

        for (std::size_t i = 0; (i + 2u) < mesh.indices.size(); i += 3u) {
            const std::uint32_t i0 = mesh.indices[i];
            const std::uint32_t i1 = mesh.indices[i + 1u];
            const std::uint32_t i2 = mesh.indices[i + 2u];
            if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() ||
                i2 >= mesh.vertices.size()) {
                continue;
            }
            Triangle triangle;
            toWorld(mesh.vertices[i0].position, &triangle.v[0]);
            toWorld(mesh.vertices[i1].position, &triangle.v[3]);
            toWorld(mesh.vertices[i2].position, &triangle.v[6]);

            const float e1[3] = {triangle.v[3] - triangle.v[0], triangle.v[4] - triangle.v[1],
                                 triangle.v[5] - triangle.v[2]};
            const float e2[3] = {triangle.v[6] - triangle.v[0], triangle.v[7] - triangle.v[1],
                                 triangle.v[8] - triangle.v[2]};
            float normal[3] = {(e1[1] * e2[2]) - (e1[2] * e2[1]),
                               (e1[2] * e2[0]) - (e1[0] * e2[2]),
                               (e1[0] * e2[1]) - (e1[1] * e2[0])};
            const float length =
                std::sqrt((normal[0] * normal[0]) + (normal[1] * normal[1]) + (normal[2] * normal[2]));
            if (length <= 1e-6f) {
                continue;  // degenerate
            }
            triangle.normalY = normal[1] / length;

            // Bucket by the triangle's XZ footprint so a query only looks at
            // what is actually near the player.
            const float minX = std::min({triangle.v[0], triangle.v[3], triangle.v[6]});
            const float maxX = std::max({triangle.v[0], triangle.v[3], triangle.v[6]});
            const float minZ = std::min({triangle.v[2], triangle.v[5], triangle.v[8]});
            const float maxZ = std::max({triangle.v[2], triangle.v[5], triangle.v[8]});
            const auto bucketIndex = [&](float value, float origin) {
                return std::clamp(
                    static_cast<int>((value - origin) / kBucketSize), 0, kBucketGrid - 1);
            };
            const int bx0 = bucketIndex(minX, collision.originX);
            const int bx1 = bucketIndex(maxX, collision.originX);
            const int bz0 = bucketIndex(minZ, collision.originZ);
            const int bz1 = bucketIndex(maxZ, collision.originZ);
            // A triangle spanning an absurd area is a skybox or a terrain-sized
            // decal; bucketing it everywhere would make every query O(all).
            if ((bx1 - bx0) > 4 || (bz1 - bz0) > 4) {
                continue;
            }

            const auto index = static_cast<std::uint32_t>(collision.triangles.size());
            collision.triangles.push_back(triangle);
            for (int bz = bz0; bz <= bz1; ++bz) {
                for (int bx = bx0; bx <= bx1; ++bx) {
                    collision.buckets[(static_cast<std::size_t>(bz) * kBucketGrid) + bx]
                        .push_back(index);
                }
            }
        }
    }

    m_cells[cell] = std::move(collision);
}

void CollisionWorld::removeCell(const importer::CellCoord& cell) {
    m_cells.erase(cell);
}

void CollisionWorld::clear() {
    m_cells.clear();
}

std::size_t CollisionWorld::triangleCount() const {
    std::size_t total = 0;
    for (const auto& [cell, collision] : m_cells) {
        (void)cell;
        total += collision.triangles.size();
    }
    return total;
}

void CollisionWorld::forEachNearbyTriangle(
    float worldX, float worldZ, const std::function<void(const Triangle&)>& visit) const {
    // The player can stand near a boundary while the geometry belongs to the
    // neighbouring cell, so walk the 3x3 block.
    const importer::CellCoord centre{
        static_cast<std::int32_t>(std::floor(worldX / kCellSize)),
        static_cast<std::int32_t>(std::floor(-worldZ / kCellSize))};
    for (std::int32_t dz = -1; dz <= 1; ++dz) {
        for (std::int32_t dx = -1; dx <= 1; ++dx) {
            const auto found = m_cells.find(importer::CellCoord{centre.x + dx, centre.z + dz});
            if (found == m_cells.end() || found->second.buckets.empty()) {
                continue;
            }
            const CellCollision& collision = found->second;
            const int bx =
                static_cast<int>(std::floor((worldX - collision.originX) / kBucketSize));
            const int bz =
                static_cast<int>(std::floor((worldZ - collision.originZ) / kBucketSize));
            // One bucket ring, so a query near a bucket edge still sees the
            // geometry just across it.
            for (int oz = -1; oz <= 1; ++oz) {
                for (int ox = -1; ox <= 1; ++ox) {
                    const int cx = bx + ox;
                    const int cz = bz + oz;
                    if (cx < 0 || cx >= kBucketGrid || cz < 0 || cz >= kBucketGrid) {
                        continue;
                    }
                    for (const std::uint32_t index :
                         collision.buckets[(static_cast<std::size_t>(cz) * kBucketGrid) + cx]) {
                        visit(collision.triangles[index]);
                    }
                }
            }
        }
    }
}

bool CollisionWorld::terrainHeight(float worldX, float worldZ, float& outHeight) const {
    const importer::CellCoord cell{
        static_cast<std::int32_t>(std::floor(worldX / kCellSize)),
        static_cast<std::int32_t>(std::floor(-worldZ / kCellSize))};
    const auto found = m_cells.find(cell);
    if (found == m_cells.end() || found->second.heights.empty()) {
        return false;
    }
    const CellCollision& collision = found->second;
    const float gx = std::clamp(
        (worldX - collision.originX) / kPostSpacing, 0.0f, static_cast<float>(kTerrainGrid - 1));
    const float gz = std::clamp(
        (worldZ - collision.originZ) / kPostSpacing, 0.0f, static_cast<float>(kTerrainGrid - 1));
    const int x0 = static_cast<int>(gx);
    const int z0 = static_cast<int>(gz);
    const int x1 = std::min(x0 + 1, kTerrainGrid - 1);
    const int z1 = std::min(z0 + 1, kTerrainGrid - 1);
    const float fx = gx - static_cast<float>(x0);
    const float fz = gz - static_cast<float>(z0);
    const auto at = [&](int x, int z) {
        return collision.heights[(static_cast<std::size_t>(z) * kTerrainGrid) + x];
    };
    outHeight = (at(x0, z0) * (1.0f - fx) * (1.0f - fz)) + (at(x1, z0) * fx * (1.0f - fz)) +
                (at(x0, z1) * (1.0f - fx) * fz) + (at(x1, z1) * fx * fz);
    return true;
}

bool CollisionWorld::groundHeight(
    float worldX, float worldZ, float referenceY, float& outHeight) const {
    float height = 0.0f;
    const bool haveTerrain = terrainHeight(worldX, worldZ, height);
    if (!haveTerrain) {
        return false;
    }

    // Surfaces the player can stand on: floors, rock tops, steps. Take the
    // highest one at or below head height, so walking onto a rock raises the
    // player rather than stopping them.
    const float headY = referenceY + m_tuning.stepHeight;
    forEachNearbyTriangle(worldX, worldZ, [&](const Triangle& triangle) {
        if (triangle.normalY < m_tuning.minWalkableNormalY) {
            return;  // a wall, not a floor
        }
        // Barycentric containment and interpolated height, in XZ.
        const float x1 = triangle.v[0];
        const float z1 = triangle.v[2];
        const float x2 = triangle.v[3];
        const float z2 = triangle.v[5];
        const float x3 = triangle.v[6];
        const float z3 = triangle.v[8];
        const float denominator = ((z2 - z3) * (x1 - x3)) + ((x3 - x2) * (z1 - z3));
        if (std::abs(denominator) <= 1e-6f) {
            return;
        }
        const float a =
            (((z2 - z3) * (worldX - x3)) + ((x3 - x2) * (worldZ - z3))) / denominator;
        const float b =
            (((z3 - z1) * (worldX - x3)) + ((x1 - x3) * (worldZ - z3))) / denominator;
        const float c = 1.0f - a - b;
        if (a < 0.0f || b < 0.0f || c < 0.0f) {
            return;
        }
        const float surfaceY = (a * triangle.v[1]) + (b * triangle.v[4]) + (c * triangle.v[7]);
        if (surfaceY <= headY && surfaceY > height) {
            height = surfaceY;
        }
    });

    outHeight = height;
    return true;
}

void CollisionWorld::resolveHorizontal(float& worldX, float eyeY, float& worldZ) const {
    resolveHorizontalFor(
        worldX, worldZ, eyeY - m_tuning.eyeHeight, eyeY, m_tuning.radius, m_tuning.stepHeight);
}

void CollisionWorld::resolveHorizontalFor(
    float& worldX,
    float& worldZ,
    float feetY,
    float headY,
    float radius,
    float stepHeight
) const {
    // Anything the body can step onto is not a wall, however steep its
    // triangles are. Without this, the lip of every rock and kerb is solid.
    const float blockingFloorY = feetY + stepHeight;

    for (int pass = 0; pass < 2; ++pass) {
        float pushX = 0.0f;
        float pushZ = 0.0f;
        bool blocked = false;
        const float queryX = worldX;
        const float queryZ = worldZ;

        forEachNearbyTriangle(queryX, queryZ, [&](const Triangle& triangle) {
            if (triangle.normalY >= m_tuning.minWalkableNormalY) {
                return;  // walkable surface, handled by groundHeight
            }
            const float minY = std::min({triangle.v[1], triangle.v[4], triangle.v[7]});
            const float maxY = std::max({triangle.v[1], triangle.v[4], triangle.v[7]});
            // Only walls that actually span the player's body block them: a
            // parapet at knee height or a beam overhead does not.
            if (maxY <= blockingFloorY || minY >= headY) {
                return;
            }

            // Closest point on the triangle's XZ outline to the player.
            float bestX = 0.0f;
            float bestZ = 0.0f;
            float bestDistanceSq = std::numeric_limits<float>::max();
            for (int edge = 0; edge < 3; ++edge) {
                const int a = edge * 3;
                const int b = ((edge + 1) % 3) * 3;
                float cx = 0.0f;
                float cz = 0.0f;
                closestPointOnSegment(
                    queryX, queryZ, triangle.v[a], triangle.v[a + 2], triangle.v[b],
                    triangle.v[b + 2], cx, cz);
                const float dx = queryX - cx;
                const float dz = queryZ - cz;
                const float distanceSq = (dx * dx) + (dz * dz);
                if (distanceSq < bestDistanceSq) {
                    bestDistanceSq = distanceSq;
                    bestX = cx;
                    bestZ = cz;
                }
            }
            if (bestDistanceSq >= radius * radius) {
                return;
            }

            const float distance = std::sqrt(bestDistanceSq);
            float dirX = queryX - bestX;
            float dirZ = queryZ - bestZ;
            if (distance > 1e-4f) {
                dirX /= distance;
                dirZ /= distance;
            } else {
                // Dead centre on the edge: push along the triangle's own
                // horizontal normal instead of dividing by zero.
                dirX = 1.0f;
                dirZ = 0.0f;
            }
            const float penetration = radius - distance;
            // Accumulate the deepest push rather than summing every triangle of
            // a wall, which would launch the player away from flat surfaces.
            if ((pushX * dirX) + (pushZ * dirZ) < penetration) {
                pushX = dirX * penetration;
                pushZ = dirZ * penetration;
            }
            blocked = true;
        });

        if (!blocked) {
            break;
        }
        worldX += pushX;
        worldZ += pushZ;
    }
}

}  // namespace odai::games::newvegas
