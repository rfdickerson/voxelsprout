#include "games/newvegas/bethesda_collision.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "import/fnv/fallout_records.h"

namespace odai::games::newvegas {

namespace {

constexpr int kTerrainGrid = odai::importer::fnv::kLandGridSize;       // 33
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

    // Terrain, read from the scene's own terrain mesh rather than the LAND
    // record: the mesh is what is drawn, and reading it here means a cell
    // restored from the disk cache produces identical collision to one just
    // built. No second decode path to drift.
    if (!scene.meshes.empty() && scene.meshes[0].name == "terrain" &&
        !scene.meshes[0].vertices.empty()) {
        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minZ = std::numeric_limits<float>::max();
        float maxZ = std::numeric_limits<float>::lowest();
        for (const importer::ImportedSceneVertex& vertex : scene.meshes[0].vertices) {
            minX = std::min(minX, vertex.position[0]);
            maxX = std::max(maxX, vertex.position[0]);
            minZ = std::min(minZ, vertex.position[2]);
            maxZ = std::max(maxZ, vertex.position[2]);
        }
        const float span = std::max(maxX - minX, maxZ - minZ);
        if (std::isfinite(span) && span > 1.0f) {
            collision.originX = minX;
            collision.originZ = minZ;
            collision.cellSize = span;
            collision.postSpacing = span / static_cast<float>(kTerrainGrid - 1);
            collision.bucketSize = span / static_cast<float>(kBucketGrid);
        }
        collision.heights.assign(
            static_cast<std::size_t>(kTerrainGrid) * kTerrainGrid,
            -std::numeric_limits<float>::max());
        for (const importer::ImportedSceneVertex& vertex : scene.meshes[0].vertices) {
            const int gx =
                static_cast<int>(std::lround(
                    (vertex.position[0] - collision.originX) / collision.postSpacing));
            const int gz =
                static_cast<int>(std::lround(
                    (vertex.position[2] - collision.originZ) / collision.postSpacing));
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
    const auto addTriangle = [&](const float* vertices) {
        Triangle triangle;
        std::copy_n(vertices, 9u, triangle.v);

        const float e1[3] = {triangle.v[3] - triangle.v[0], triangle.v[4] - triangle.v[1],
                             triangle.v[5] - triangle.v[2]};
        const float e2[3] = {triangle.v[6] - triangle.v[0], triangle.v[7] - triangle.v[1],
                             triangle.v[8] - triangle.v[2]};
        float normal[3] = {(e1[1] * e2[2]) - (e1[2] * e2[1]),
                           (e1[2] * e2[0]) - (e1[0] * e2[2]),
                           (e1[0] * e2[1]) - (e1[1] * e2[0])};
        const float length =
            std::sqrt((normal[0] * normal[0]) + (normal[1] * normal[1]) +
                      (normal[2] * normal[2]));
        if (length <= 1e-6f) {
            return;
        }
        triangle.normalY = normal[1] / length;

        const float minX = std::min({triangle.v[0], triangle.v[3], triangle.v[6]});
        const float maxX = std::max({triangle.v[0], triangle.v[3], triangle.v[6]});
        const float minZ = std::min({triangle.v[2], triangle.v[5], triangle.v[8]});
        const float maxZ = std::max({triangle.v[2], triangle.v[5], triangle.v[8]});
        const auto bucketIndex = [&](float value, float origin) {
            return std::clamp(
                static_cast<int>((value - origin) / collision.bucketSize), 0, kBucketGrid - 1);
        };
        const int bx0 = bucketIndex(minX, collision.originX);
        const int bx1 = bucketIndex(maxX, collision.originX);
        const int bz0 = bucketIndex(minZ, collision.originZ);
        const int bz1 = bucketIndex(maxZ, collision.originZ);
        if ((bx1 - bx0) > 4 || (bz1 - bz0) > 4) {
            return;
        }

        const auto index = static_cast<std::uint32_t>(collision.triangles.size());
        collision.triangles.push_back(triangle);
        for (int bz = bz0; bz <= bz1; ++bz) {
            for (int bx = bx0; bx <= bx1; ++bx) {
                collision.buckets[(static_cast<std::size_t>(bz) * kBucketGrid) + bx]
                    .push_back(index);
            }
        }
    };

    // CellSceneBuilder emits a complete world-space soup: authored Havok for
    // each NIF that has a supported fixed/keyframed graph, opaque visible
    // triangles for that individual NIF otherwise. Cached cells therefore use
    // exactly the same collision as freshly built ones.
    if (!scene.collisionTriangles.empty()) {
        for (const importer::ImportedSceneCollisionTriangle& triangle :
             scene.collisionTriangles) {
            addTriangle(triangle.vertices);
        }
        m_cells[cell] = std::move(collision);
        return;
    }

    // Compatibility fallback for synthetic/test scenes that predate the
    // separate collision stream.
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
            float vertices[9];
            toWorld(mesh.vertices[i0].position, &vertices[0]);
            toWorld(mesh.vertices[i1].position, &vertices[3]);
            toWorld(mesh.vertices[i2].position, &vertices[6]);
            addTriangle(vertices);
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
    // Derive coverage from each imported scene rather than assuming the later
    // games' 4096-unit grid. Only the resident ring is examined (normally 49
    // cells), after which the per-cell buckets keep triangle work local.
    for (const auto& [cell, collision] : m_cells) {
            (void)cell;
            if (collision.buckets.empty() ||
                worldX < collision.originX - collision.bucketSize ||
                worldX > collision.originX + collision.cellSize + collision.bucketSize ||
                worldZ < collision.originZ - collision.bucketSize ||
                worldZ > collision.originZ + collision.cellSize + collision.bucketSize) {
                continue;
            }
            const int bx =
                static_cast<int>(std::floor((worldX - collision.originX) / collision.bucketSize));
            const int bz =
                static_cast<int>(std::floor((worldZ - collision.originZ) / collision.bucketSize));
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

bool CollisionWorld::terrainHeight(float worldX, float worldZ, float& outHeight) const {
    const CellCollision* found = nullptr;
    for (const auto& [cell, collision] : m_cells) {
        (void)cell;
        if (!collision.heights.empty() && worldX >= collision.originX &&
            worldX <= collision.originX + collision.cellSize && worldZ >= collision.originZ &&
            worldZ <= collision.originZ + collision.cellSize) {
            found = &collision;
            break;
        }
    }
    if (found == nullptr) return false;
    const CellCollision& collision = *found;
    const float gx = std::clamp(
        (worldX - collision.originX) / collision.postSpacing,
        0.0f, static_cast<float>(kTerrainGrid - 1));
    const float gz = std::clamp(
        (worldZ - collision.originZ) / collision.postSpacing,
        0.0f, static_cast<float>(kTerrainGrid - 1));
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
    float height = -std::numeric_limits<float>::infinity();
    bool haveGround = terrainHeight(worldX, worldZ, height);

    // Surfaces the player can stand on: floors, rock tops, steps. Take the
    // highest one at or below head height, so walking onto a rock raises the
    // player rather than stopping them. A LAND heightfield is not required:
    // Skyrim's city child-world cells commonly contain only static streets,
    // floors and battlements. Returning early when terrain was absent left
    // every actor in those cells at its unsnapped authored Y, visibly hovering
    // over the rendered street even though valid walkable triangles existed.
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
            haveGround = true;
        }
    });

    if (!haveGround) {
        return false;
    }
    outHeight = height;
    return true;
}

bool CollisionWorld::recoverFeetAboveIntersectingFloor(
    float worldX,
    float worldZ,
    float feetY,
    float headY,
    float stepHeight,
    float& outFeetY
) const {
    if (!(headY > feetY) || stepHeight < 0.0f) return false;
    float surfaceY = 0.0f;
    // groundHeight accepts a feet reference and searches through
    // reference+stepHeight. Shift that reference so this recovery query's
    // explicit ceiling is the top of the capsule.
    if (!groundHeight(worldX, worldZ, headY - stepHeight, surfaceY)) return false;
    if (surfaceY <= feetY + stepHeight || surfaceY >= headY) return false;
    outFeetY = surfaceY;
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
