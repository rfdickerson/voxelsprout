#include "bethesda/navigation_world.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>

namespace odai::games::newvegas {

namespace {

using odai::math::Vector3;

float distanceSquaredXZ(const Vector3& a, const Vector3& b) {
    const float dx = a.x - b.x;
    const float dz = a.z - b.z;
    return (dx * dx) + (dz * dz);
}

Vector3 closestPointOnSegmentXZ(
    const Vector3& point, const Vector3& a, const Vector3& b) {
    const float dx = b.x - a.x;
    const float dz = b.z - a.z;
    const float lengthSquared = (dx * dx) + (dz * dz);
    const float t = lengthSquared > 1e-8f
        ? std::clamp(((point.x - a.x) * dx + (point.z - a.z) * dz) / lengthSquared,
                     0.0f, 1.0f)
        : 0.0f;
    return Vector3{
        a.x + (dx * t),
        a.y + ((b.y - a.y) * t),
        a.z + (dz * t)};
}

// Closest point in the horizontal projection, with Y reconstructed from the
// same barycentric/edge weights. Navmesh triangles describe a height field in
// the small, so this is the point a walking actor wants; a full 3D closest-point
// query can slide sideways toward a steep face solely to reduce vertical error.
Vector3 closestPointOnTriangleXZ(
    const Vector3& point, const Vector3& a, const Vector3& b, const Vector3& c) {
    const float v0x = b.x - a.x;
    const float v0z = b.z - a.z;
    const float v1x = c.x - a.x;
    const float v1z = c.z - a.z;
    const float v2x = point.x - a.x;
    const float v2z = point.z - a.z;
    const float d00 = (v0x * v0x) + (v0z * v0z);
    const float d01 = (v0x * v1x) + (v0z * v1z);
    const float d11 = (v1x * v1x) + (v1z * v1z);
    const float d20 = (v2x * v0x) + (v2z * v0z);
    const float d21 = (v2x * v1x) + (v2z * v1z);
    const float denominator = (d00 * d11) - (d01 * d01);
    if (std::abs(denominator) > 1e-8f) {
        const float v = ((d11 * d20) - (d01 * d21)) / denominator;
        const float w = ((d00 * d21) - (d01 * d20)) / denominator;
        const float u = 1.0f - v - w;
        if (u >= 0.0f && v >= 0.0f && w >= 0.0f) {
            return Vector3{
                (a.x * u) + (b.x * v) + (c.x * w),
                (a.y * u) + (b.y * v) + (c.y * w),
                (a.z * u) + (b.z * v) + (c.z * w)};
        }
    }

    Vector3 closest = closestPointOnSegmentXZ(point, a, b);
    float best = distanceSquaredXZ(point, closest);
    for (const Vector3 candidate : {
             closestPointOnSegmentXZ(point, b, c),
             closestPointOnSegmentXZ(point, c, a)}) {
        const float distance = distanceSquaredXZ(point, candidate);
        if (distance < best) {
            best = distance;
            closest = candidate;
        }
    }
    return closest;
}

struct RasterTriangle {
    Vector3 vertex[3]{};
    Vector3 normal{};
};

bool sampleTriangleHeightXZ(
    const RasterTriangle& triangle, float x, float z, float& outHeight) {
    const Vector3& a = triangle.vertex[0];
    const Vector3& b = triangle.vertex[1];
    const Vector3& c = triangle.vertex[2];
    const float denominator = ((b.z - c.z) * (a.x - c.x)) +
        ((c.x - b.x) * (a.z - c.z));
    if (std::abs(denominator) <= 1.0e-6f) return false;
    const float u = (((b.z - c.z) * (x - c.x)) +
        ((c.x - b.x) * (z - c.z))) / denominator;
    const float v = (((c.z - a.z) * (x - c.x)) +
        ((a.x - c.x) * (z - c.z))) / denominator;
    const float w = 1.0f - u - v;
    constexpr float kEdgeTolerance = 1.0e-4f;
    if (u < -kEdgeTolerance || v < -kEdgeTolerance || w < -kEdgeTolerance) return false;
    outHeight = (a.y * u) + (b.y * v) + (c.y * w);
    return true;
}

float pointSegmentDistanceSquaredXZ(
    const Vector3& point, const Vector3& a, const Vector3& b) {
    return distanceSquaredXZ(point, closestPointOnSegmentXZ(point, a, b));
}

float pointTriangleDistanceSquaredXZ(
    const Vector3& point, const RasterTriangle& triangle) {
    float ignored = 0.0f;
    if (sampleTriangleHeightXZ(triangle, point.x, point.z, ignored)) return 0.0f;
    return std::min({
        pointSegmentDistanceSquaredXZ(point, triangle.vertex[0], triangle.vertex[1]),
        pointSegmentDistanceSquaredXZ(point, triangle.vertex[1], triangle.vertex[2]),
        pointSegmentDistanceSquaredXZ(point, triangle.vertex[2], triangle.vertex[0])});
}

float orientationXZ(const Vector3& a, const Vector3& b, const Vector3& c) {
    return ((b.x - a.x) * (c.z - a.z)) - ((b.z - a.z) * (c.x - a.x));
}

bool segmentsIntersectXZ(
    const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d) {
    const float abC = orientationXZ(a, b, c);
    const float abD = orientationXZ(a, b, d);
    const float cdA = orientationXZ(c, d, a);
    const float cdB = orientationXZ(c, d, b);
    constexpr float kEpsilon = 1.0e-5f;
    return ((abC > kEpsilon && abD < -kEpsilon) ||
               (abC < -kEpsilon && abD > kEpsilon)) &&
        ((cdA > kEpsilon && cdB < -kEpsilon) ||
            (cdA < -kEpsilon && cdB > kEpsilon));
}

float segmentSegmentDistanceSquaredXZ(
    const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d) {
    if (segmentsIntersectXZ(a, b, c, d)) return 0.0f;
    return std::min({pointSegmentDistanceSquaredXZ(a, c, d),
        pointSegmentDistanceSquaredXZ(b, c, d),
        pointSegmentDistanceSquaredXZ(c, a, b),
        pointSegmentDistanceSquaredXZ(d, a, b)});
}

float segmentTriangleDistanceSquaredXZ(
    const Vector3& a, const Vector3& b, const RasterTriangle& triangle) {
    float ignored = 0.0f;
    if (sampleTriangleHeightXZ(triangle, a.x, a.z, ignored) ||
        sampleTriangleHeightXZ(triangle, b.x, b.z, ignored)) return 0.0f;
    return std::min({
        segmentSegmentDistanceSquaredXZ(a, b, triangle.vertex[0], triangle.vertex[1]),
        segmentSegmentDistanceSquaredXZ(a, b, triangle.vertex[1], triangle.vertex[2]),
        segmentSegmentDistanceSquaredXZ(a, b, triangle.vertex[2], triangle.vertex[0])});
}

std::optional<RasterTriangle> makeRasterTriangle(const float* vertices) {
    RasterTriangle triangle;
    for (std::size_t corner = 0u; corner < 3u; ++corner) {
        triangle.vertex[corner] = Vector3{vertices[corner * 3u],
            vertices[(corner * 3u) + 1u], vertices[(corner * 3u) + 2u]};
    }
    const Vector3 normal = odai::math::cross(
        triangle.vertex[1] - triangle.vertex[0],
        triangle.vertex[2] - triangle.vertex[0]);
    const float length = odai::math::length(normal);
    if (!std::isfinite(length) || length <= 1.0e-5f) return std::nullopt;
    triangle.normal = normal / length;
    return triangle;
}

template <typename MeshType, typename TriangleType>
Vector3 triangleCentroid(const MeshType& mesh, const TriangleType& triangle) {
    const Vector3& a = mesh.vertices[triangle.vertex[0]];
    const Vector3& b = mesh.vertices[triangle.vertex[1]];
    const Vector3& c = mesh.vertices[triangle.vertex[2]];
    return Vector3{
        (a.x + b.x + c.x) / 3.0f,
        (a.y + b.y + c.y) / 3.0f,
        (a.z + b.z + c.z) / 3.0f};
}

}  // namespace

void ActorNavigationWorld::addCell(
    const importer::CellCoord& cell,
    const std::vector<importer::fnv::FalloutNavMeshRecord>& records) {
    std::vector<Mesh> meshes;
    meshes.reserve(records.size());
    for (const importer::fnv::FalloutNavMeshRecord& record : records) {
        const std::size_t vertexCount = record.vertices.size() / 3u;
        if (vertexCount == 0u || record.triangles.empty()) {
            continue;
        }
        Mesh mesh;
        mesh.formId = record.formId;
        mesh.cell = cell;
        mesh.doorPortals = record.doorPortals;
        mesh.vertices.reserve(vertexCount);
        for (std::size_t i = 0; i < vertexCount; ++i) {
            const float* source = &record.vertices[i * 3u];
            // Bethesda xyz is Z-up; engine xyz is Y-up with the horizontal Y
            // axis negated, matching CellSceneBuilder's world conversion.
            mesh.vertices.push_back(Vector3{source[0], source[2], -source[1]});
        }
        mesh.triangles.reserve(record.triangles.size());
        for (const importer::fnv::FalloutNavMeshTriangle& source : record.triangles) {
            if (source.vertex[0] >= vertexCount || source.vertex[1] >= vertexCount ||
                source.vertex[2] >= vertexCount) {
                continue;
            }
            Triangle triangle;
            for (int corner = 0; corner < 3; ++corner) {
                triangle.vertex[corner] = source.vertex[corner];
                triangle.neighbour[corner] = source.neighbour[corner];
            }
            mesh.triangles.push_back(triangle);
        }
        if (!mesh.triangles.empty()) {
            meshes.push_back(std::move(mesh));
        }
    }
    if (meshes.empty()) {
        m_cells.erase(cell);
    } else {
        m_cells[cell] = std::move(meshes);
        // Authored navigation is always preferred when it exists.
        m_generatedCells.erase(cell);
    }
}

void ActorNavigationWorld::addGeneratedCell(
    const importer::CellCoord& cell,
    const importer::ImportedScene& scene,
    const GeneratedNavigationConfig& requestedConfig) {
    GeneratedNavigationConfig config = requestedConfig;
    config.cellSize = std::clamp(config.cellSize, 16.0f, 256.0f);
    config.agentRadius = std::clamp(config.agentRadius, 1.0f, config.cellSize * 0.49f);
    config.agentHeight = std::max(config.agentHeight, 1.0f);
    config.stepHeight = std::clamp(config.stepHeight, 0.0f, config.agentHeight);
    config.maxSlopeDegrees = std::clamp(config.maxSlopeDegrees, 0.0f, 89.0f);
    const float minimumNormalY = std::cos(
        config.maxSlopeDegrees * (odai::math::kPi / 180.0f));

    std::vector<RasterTriangle> triangles;
    triangles.reserve(scene.collisionTriangles.size() + 1024u);
    for (const importer::ImportedSceneCollisionTriangle& source : scene.collisionTriangles) {
        if (auto triangle = makeRasterTriangle(source.vertices)) {
            triangles.push_back(*triangle);
        }
    }
    if (!scene.meshes.empty() && scene.meshes.front().name == "terrain") {
        const importer::ImportedSceneMesh& terrain = scene.meshes.front();
        for (std::size_t offset = 0u; offset + 2u < terrain.indices.size(); offset += 3u) {
            const std::uint32_t indices[3] = {terrain.indices[offset],
                terrain.indices[offset + 1u], terrain.indices[offset + 2u]};
            if (indices[0] >= terrain.vertices.size() || indices[1] >= terrain.vertices.size() ||
                indices[2] >= terrain.vertices.size()) continue;
            float vertices[9]{};
            for (std::size_t corner = 0u; corner < 3u; ++corner) {
                std::copy_n(terrain.vertices[indices[corner]].position, 3u,
                    vertices + (corner * 3u));
            }
            if (auto triangle = makeRasterTriangle(vertices)) triangles.push_back(*triangle);
        }
    }

    struct FloorSample {
        float height = 0.0f;
        float normalY = 1.0f;
    };
    using RasterKey = std::pair<std::int64_t, std::int64_t>;
    std::map<RasterKey, std::vector<FloorSample>> samples;
    GeneratedCell generated;
    generated.config = config;

    for (const RasterTriangle& triangle : triangles) {
        const float minX = std::min({triangle.vertex[0].x,
            triangle.vertex[1].x, triangle.vertex[2].x});
        const float maxX = std::max({triangle.vertex[0].x,
            triangle.vertex[1].x, triangle.vertex[2].x});
        const float minZ = std::min({triangle.vertex[0].z,
            triangle.vertex[1].z, triangle.vertex[2].z});
        const float maxZ = std::max({triangle.vertex[0].z,
            triangle.vertex[1].z, triangle.vertex[2].z});
        const std::int64_t gridMinX = static_cast<std::int64_t>(
            std::floor(minX / config.cellSize));
        const std::int64_t gridMaxX = static_cast<std::int64_t>(
            std::floor(maxX / config.cellSize));
        const std::int64_t gridMinZ = static_cast<std::int64_t>(
            std::floor(minZ / config.cellSize));
        const std::int64_t gridMaxZ = static_cast<std::int64_t>(
            std::floor(maxZ / config.cellSize));
        if (triangle.normal.y >= minimumNormalY) {
            for (std::int64_t gridZ = gridMinZ; gridZ <= gridMaxZ; ++gridZ) {
                for (std::int64_t gridX = gridMinX; gridX <= gridMaxX; ++gridX) {
                    const float x = (static_cast<float>(gridX) + 0.5f) * config.cellSize;
                    const float z = (static_cast<float>(gridZ) + 0.5f) * config.cellSize;
                    float height = 0.0f;
                    if (sampleTriangleHeightXZ(triangle, x, z, height)) {
                        samples[{gridX, gridZ}].push_back({height, triangle.normal.y});
                    }
                }
            }
            continue;
        }

        GeneratedCell::Obstacle obstacle;
        std::copy_n(triangle.vertex, 3u, obstacle.vertex);
        obstacle.minY = std::min({triangle.vertex[0].y,
            triangle.vertex[1].y, triangle.vertex[2].y});
        obstacle.maxY = std::max({triangle.vertex[0].y,
            triangle.vertex[1].y, triangle.vertex[2].y});
        const std::uint32_t obstacleIndex =
            static_cast<std::uint32_t>(generated.obstacles.size());
        generated.obstacles.push_back(obstacle);
        const std::int64_t obstacleMinX = static_cast<std::int64_t>(
            std::floor((minX - config.agentRadius) / config.cellSize));
        const std::int64_t obstacleMaxX = static_cast<std::int64_t>(
            std::floor((maxX + config.agentRadius) / config.cellSize));
        const std::int64_t obstacleMinZ = static_cast<std::int64_t>(
            std::floor((minZ - config.agentRadius) / config.cellSize));
        const std::int64_t obstacleMaxZ = static_cast<std::int64_t>(
            std::floor((maxZ + config.agentRadius) / config.cellSize));
        for (std::int64_t gridZ = obstacleMinZ; gridZ <= obstacleMaxZ; ++gridZ) {
            for (std::int64_t gridX = obstacleMinX; gridX <= obstacleMaxX; ++gridX) {
                generated.obstacleBuckets[{gridX, gridZ}].push_back(obstacleIndex);
            }
        }
    }

    const float radiusSquared = config.agentRadius * config.agentRadius;
    for (auto& [key, heights] : samples) {
        std::sort(heights.begin(), heights.end(), [](const FloorSample& left,
                                                      const FloorSample& right) {
            return std::tie(left.height, left.normalY) <
                std::tie(right.height, right.normalY);
        });
        std::vector<FloorSample> unique;
        for (const FloorSample& sample : heights) {
            if (!unique.empty() && std::abs(unique.back().height - sample.height) <= 1.0f) {
                unique.back().height = std::max(unique.back().height, sample.height);
                unique.back().normalY = std::max(unique.back().normalY, sample.normalY);
            } else {
                unique.push_back(sample);
            }
        }
        const Vector3 horizontalPoint{
            (static_cast<float>(key.first) + 0.5f) * config.cellSize,
            0.0f,
            (static_cast<float>(key.second) + 0.5f) * config.cellSize};
        for (const FloorSample& sample : unique) {
            bool blocked = false;
            const auto bucket = generated.obstacleBuckets.find(key);
            if (bucket != generated.obstacleBuckets.end()) {
                for (const std::uint32_t obstacleIndex : bucket->second) {
                    const GeneratedCell::Obstacle& obstacle =
                        generated.obstacles[obstacleIndex];
                    if (obstacle.maxY <= sample.height + config.stepHeight + 0.5f ||
                        obstacle.minY >= sample.height + config.agentHeight) continue;
                    RasterTriangle raster;
                    std::copy_n(obstacle.vertex, 3u, raster.vertex);
                    if (pointTriangleDistanceSquaredXZ(horizontalPoint, raster) <
                        radiusSquared) {
                        blocked = true;
                        break;
                    }
                }
            }
            if (!blocked) {
                generated.nodes.push_back(GeneratedNode{key.first, key.second,
                    Vector3{horizontalPoint.x, sample.height, horizontalPoint.z},
                    sample.normalY});
            }
        }
    }
    std::sort(generated.nodes.begin(), generated.nodes.end(),
        [](const GeneratedNode& left, const GeneratedNode& right) {
            return std::tie(left.gridX, left.gridZ, left.position.y) <
                std::tie(right.gridX, right.gridZ, right.position.y);
        });
    if (generated.nodes.empty()) m_generatedCells.erase(cell);
    else m_generatedCells.insert_or_assign(cell, std::move(generated));
}

void ActorNavigationWorld::removeCell(const importer::CellCoord& cell) {
    m_cells.erase(cell);
    m_generatedCells.erase(cell);
}

void ActorNavigationWorld::clear() {
    m_cells.clear();
    m_generatedCells.clear();
    m_residentDoors.clear();
}

void ActorNavigationWorld::setResidentDoors(
    const std::vector<importer::ImportedSceneDoor>& doors) {
    m_residentDoors = doors;
    std::sort(m_residentDoors.begin(), m_residentDoors.end(),
        [](const importer::ImportedSceneDoor& left,
           const importer::ImportedSceneDoor& right) {
            return std::tie(left.sourceReferenceFormId,
                       left.arrivalPosition[0], left.arrivalPosition[1], left.arrivalPosition[2]) <
                std::tie(right.sourceReferenceFormId,
                       right.arrivalPosition[0], right.arrivalPosition[1], right.arrivalPosition[2]);
        });
}

bool ActorNavigationWorld::findNearest(
    const Vector3& point,
    float maxHorizontalDistance,
    float maxVerticalDistance,
    Location& outLocation) const {
    const float maxHorizontalSquared = maxHorizontalDistance * maxHorizontalDistance;
    float bestScore = std::numeric_limits<float>::infinity();
    bool found = false;
    for (const auto& [cell, meshes] : m_cells) {
        (void)cell;
        for (const Mesh& mesh : meshes) {
            for (std::size_t index = 0; index < mesh.triangles.size(); ++index) {
                const Triangle& triangle = mesh.triangles[index];
                const Vector3 candidate = closestPointOnTriangleXZ(
                    point,
                    mesh.vertices[triangle.vertex[0]],
                    mesh.vertices[triangle.vertex[1]],
                    mesh.vertices[triangle.vertex[2]]);
                const float horizontalSquared = distanceSquaredXZ(point, candidate);
                const float vertical = std::abs(candidate.y - point.y);
                if (horizontalSquared > maxHorizontalSquared || vertical > maxVerticalDistance) {
                    continue;
                }
                // Prefer the actor's authored storey, then the closest path in
                // plan view. This keeps battlement guards on battlements while
                // still moving a rock-top guard sideways onto the street.
                const float score = horizontalSquared + (vertical * vertical * 4.0f);
                const bool stableTieBreak = found && std::abs(score - bestScore) <= 1.0e-5f &&
                    std::tie(mesh.cell.x, mesh.cell.z, mesh.formId, index) <
                    std::tie(outLocation.mesh->cell.x, outLocation.mesh->cell.z,
                        outLocation.mesh->formId, outLocation.triangle);
                if (score < bestScore - 1.0e-5f || stableTieBreak) {
                    bestScore = score;
                    outLocation = Location{&mesh, index, candidate, score};
                    found = true;
                }
            }
        }
    }
    return found;
}

bool ActorNavigationWorld::findNearestGenerated(
    const Vector3& point,
    float maxHorizontalDistance,
    float maxVerticalDistance,
    GeneratedLocation& outLocation) const {
    const float maxHorizontalSquared = maxHorizontalDistance * maxHorizontalDistance;
    float bestScore = std::numeric_limits<float>::infinity();
    bool found = false;
    for (const auto& [cellCoord, cell] : m_generatedCells) {
        for (std::size_t index = 0u; index < cell.nodes.size(); ++index) {
            const GeneratedNode& node = cell.nodes[index];
            const float horizontalSquared = distanceSquaredXZ(point, node.position);
            const float vertical = std::abs(point.y - node.position.y);
            if (horizontalSquared > maxHorizontalSquared || vertical > maxVerticalDistance) {
                continue;
            }
            const float score = horizontalSquared + (vertical * vertical * 4.0f);
            const bool stableTieBreak = found && std::abs(score - bestScore) <= 1.0e-5f &&
                std::tie(cellCoord.x, cellCoord.z, index) <
                std::tie(outLocation.cellCoord.x, outLocation.cellCoord.z,
                    outLocation.node);
            if (score < bestScore - 1.0e-5f || stableTieBreak) {
                bestScore = score;
                outLocation = GeneratedLocation{
                    &cell, cellCoord, index, node.position, score};
                found = true;
            }
        }
    }
    return found;
}

bool ActorNavigationWorld::projectPoint(
    float worldX,
    float worldY,
    float worldZ,
    float maxHorizontalDistance,
    float maxVerticalDistance,
    Vector3& outPoint) const {
    Location location;
    GeneratedLocation generated;
    const Vector3 requested{worldX, worldY, worldZ};
    const bool foundAuthored = findNearest(requested,
        maxHorizontalDistance, maxVerticalDistance, location);
    const bool foundGenerated = findNearestGenerated(requested,
        maxHorizontalDistance, maxVerticalDistance, generated);
    if (!foundAuthored && !foundGenerated) return false;
    if (foundGenerated && (!foundAuthored || generated.score < location.score)) {
        outPoint = generated.point;
    } else {
        outPoint = location.point;
    }
    return true;
}

bool ActorNavigationWorld::buildGeneratedPath(
    const Vector3& start,
    const Vector3& goal,
    std::vector<ActorNavigationStep>& outWaypoints) const {
    outWaypoints.clear();
    GeneratedLocation startLocation;
    GeneratedLocation goalLocation;
    if (!findNearestGenerated(start, 220.0f, 300.0f, startLocation) ||
        !findNearestGenerated(goal, 500.0f, 700.0f, goalLocation)) return false;

    struct FlatNode {
        const GeneratedCell* cell = nullptr;
        importer::CellCoord cellCoord{};
        std::size_t localIndex = 0u;
        const GeneratedNode* node = nullptr;
    };
    std::vector<std::pair<importer::CellCoord, const GeneratedCell*>> cells;
    cells.reserve(m_generatedCells.size());
    for (const auto& [coord, cell] : m_generatedCells) cells.emplace_back(coord, &cell);
    std::sort(cells.begin(), cells.end(), [](const auto& left, const auto& right) {
        return std::tie(left.first.x, left.first.z) <
            std::tie(right.first.x, right.first.z);
    });
    std::vector<FlatNode> nodes;
    std::map<std::pair<std::int64_t, std::int64_t>, std::vector<std::size_t>> byGrid;
    std::size_t startNode = std::numeric_limits<std::size_t>::max();
    std::size_t goalNode = std::numeric_limits<std::size_t>::max();
    for (const auto& [coord, cell] : cells) {
        for (std::size_t local = 0u; local < cell->nodes.size(); ++local) {
            const std::size_t global = nodes.size();
            nodes.push_back(FlatNode{cell, coord, local, &cell->nodes[local]});
            byGrid[{cell->nodes[local].gridX, cell->nodes[local].gridZ}].push_back(global);
            if (cell == startLocation.cell && local == startLocation.node) startNode = global;
            if (cell == goalLocation.cell && local == goalLocation.node) goalNode = global;
        }
    }
    if (startNode >= nodes.size() || goalNode >= nodes.size()) return false;

    struct Link {
        std::size_t target = 0u;
        ActorNavigationStepKind kind = ActorNavigationStepKind::Walk;
        Vector3 position{};
        Vector3 arrival{};
        std::uint32_t doorReferenceFormId = 0u;
    };
    std::map<std::size_t, std::vector<Link>> doorLinks;
    const auto globalFor = [&](const GeneratedLocation& location) {
        for (std::size_t index = 0u; index < nodes.size(); ++index) {
            if (nodes[index].cell == location.cell &&
                nodes[index].localIndex == location.node) return index;
        }
        return std::numeric_limits<std::size_t>::max();
    };
    for (const importer::ImportedSceneDoor& door : m_residentDoors) {
        GeneratedLocation source;
        GeneratedLocation arrival;
        const Vector3 sourcePoint{door.position[0], door.position[1], door.position[2]};
        const Vector3 arrivalPoint{
            door.arrivalPosition[0], door.arrivalPosition[1], door.arrivalPosition[2]};
        if (!findNearestGenerated(sourcePoint, 180.0f, 250.0f, source) ||
            !findNearestGenerated(arrivalPoint, 500.0f, 700.0f, arrival)) continue;
        const std::size_t sourceGlobal = globalFor(source);
        const std::size_t arrivalGlobal = globalFor(arrival);
        if (sourceGlobal >= nodes.size() || arrivalGlobal >= nodes.size() ||
            sourceGlobal == arrivalGlobal) continue;
        doorLinks[sourceGlobal].push_back(Link{arrivalGlobal,
            ActorNavigationStepKind::ActivateDoor, source.point, arrival.point,
            door.sourceReferenceFormId});
    }

    const auto linkClear = [&](const FlatNode& from, const FlatNode& to) {
        const float feetY = std::min(from.node->position.y, to.node->position.y);
        const float headY = std::max(from.node->position.y, to.node->position.y) +
            std::min(from.cell->config.agentHeight, to.cell->config.agentHeight);
        const float stepHeight = std::min(
            from.cell->config.stepHeight, to.cell->config.stepHeight);
        const float radius = std::max(
            from.cell->config.agentRadius, to.cell->config.agentRadius);
        const float radiusSquared = radius * radius;
        std::set<std::pair<const GeneratedCell*, std::uint32_t>> candidates;
        for (const GeneratedCell* owner : {from.cell, to.cell}) {
            for (const auto key : {
                     std::pair{from.node->gridX, from.node->gridZ},
                     std::pair{to.node->gridX, to.node->gridZ}}) {
                const auto bucket = owner->obstacleBuckets.find(key);
                if (bucket == owner->obstacleBuckets.end()) continue;
                for (const std::uint32_t index : bucket->second) {
                    candidates.emplace(owner, index);
                }
            }
        }
        for (const auto& [owner, index] : candidates) {
            const GeneratedCell::Obstacle& obstacle = owner->obstacles[index];
            if (obstacle.maxY <= feetY + stepHeight + 0.5f ||
                obstacle.minY >= headY) continue;
            RasterTriangle raster;
            std::copy_n(obstacle.vertex, 3u, raster.vertex);
            if (segmentTriangleDistanceSquaredXZ(
                    from.node->position, to.node->position, raster) < radiusSquared) {
                return false;
            }
        }
        return true;
    };

    const auto canTraverse = [&](const FlatNode& from, const FlatNode& to) {
        const float dx = to.node->position.x - from.node->position.x;
        const float dz = to.node->position.z - from.node->position.z;
        const float horizontal = std::sqrt((dx * dx) + (dz * dz));
        const float expected = std::max(
            from.cell->config.cellSize, to.cell->config.cellSize);
        if (horizontal > expected * 1.1f || horizontal < expected * 0.75f) return false;
        const float rise = std::abs(to.node->position.y - from.node->position.y);
        float allowedRise = std::min(
            from.cell->config.stepHeight, to.cell->config.stepHeight) + 0.5f;
        if (from.node->normalY < 0.995f && to.node->normalY < 0.995f) {
            const float slope = std::min(
                from.cell->config.maxSlopeDegrees, to.cell->config.maxSlopeDegrees);
            allowedRise = std::max(allowedRise,
                std::tan(slope * (odai::math::kPi / 180.0f)) * horizontal + 0.5f);
        }
        return rise <= allowedRise && linkClear(from, to);
    };

    struct QueueEntry {
        float estimated = 0.0f;
        float cost = 0.0f;
        std::size_t node = 0u;
        bool operator>(const QueueEntry& right) const {
            return std::tie(estimated, cost, node) >
                std::tie(right.estimated, right.cost, right.node);
        }
    };
    const auto heuristic = [&](std::size_t node) {
        return std::sqrt(distanceSquaredXZ(
            nodes[node].node->position, nodes[goalNode].node->position));
    };
    std::vector<float> cost(nodes.size(), std::numeric_limits<float>::infinity());
    std::vector<std::ptrdiff_t> parent(nodes.size(), -2);
    std::vector<Link> parentLink(nodes.size());
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> frontier;
    cost[startNode] = 0.0f;
    parent[startNode] = -1;
    frontier.push({heuristic(startNode), 0.0f, startNode});
    while (!frontier.empty()) {
        const QueueEntry current = frontier.top();
        frontier.pop();
        if (current.cost > cost[current.node] + 1.0e-5f) continue;
        if (current.node == goalNode) break;
        std::vector<Link> links;
        const GeneratedNode& source = *nodes[current.node].node;
        for (const auto offset : {std::pair{-1, 0}, std::pair{0, -1},
                 std::pair{0, 1}, std::pair{1, 0}}) {
            const auto found = byGrid.find(
                {source.gridX + offset.first, source.gridZ + offset.second});
            if (found == byGrid.end()) continue;
            for (const std::size_t target : found->second) {
                if (canTraverse(nodes[current.node], nodes[target])) {
                    links.push_back(Link{target, ActorNavigationStepKind::Walk,
                        nodes[target].node->position});
                }
            }
        }
        if (const auto doors = doorLinks.find(current.node); doors != doorLinks.end()) {
            links.insert(links.end(), doors->second.begin(), doors->second.end());
        }
        std::sort(links.begin(), links.end(), [](const Link& left, const Link& right) {
            return std::tie(left.target, left.kind, left.doorReferenceFormId) <
                std::tie(right.target, right.kind, right.doorReferenceFormId);
        });
        for (const Link& link : links) {
            const Vector3 delta = nodes[link.target].node->position -
                nodes[current.node].node->position;
            const float edgeCost = link.kind == ActorNavigationStepKind::ActivateDoor
                ? 1.0f
                : std::max(odai::math::length(delta), 1.0f);
            const float nextCost = current.cost + edgeCost;
            if (nextCost + 1.0e-5f >= cost[link.target]) continue;
            cost[link.target] = nextCost;
            parent[link.target] = static_cast<std::ptrdiff_t>(current.node);
            parentLink[link.target] = link;
            frontier.push({nextCost + heuristic(link.target), nextCost, link.target});
        }
    }
    if (parent[goalNode] == -2) return false;

    std::vector<std::size_t> route;
    for (std::ptrdiff_t current = static_cast<std::ptrdiff_t>(goalNode); current >= 0;
         current = parent[static_cast<std::size_t>(current)]) {
        route.push_back(static_cast<std::size_t>(current));
    }
    std::reverse(route.begin(), route.end());
    for (std::size_t index = 1u; index < route.size(); ++index) {
        const Link& link = parentLink[route[index]];
        outWaypoints.push_back(ActorNavigationStep{link.kind, link.position,
            link.arrival, link.doorReferenceFormId});
    }
    if (outWaypoints.empty() ||
        distanceSquaredXZ(outWaypoints.back().position, goalLocation.point) > 1.0e-4f) {
        outWaypoints.push_back(
            ActorNavigationStep{ActorNavigationStepKind::Walk, goalLocation.point});
    }
    return true;
}

bool ActorNavigationWorld::buildGeneratedWanderPath(
    const Vector3& start,
    const Vector3& origin,
    float radius,
    std::uint32_t randomValue,
    std::vector<ActorNavigationStep>& outWaypoints) const {
    GeneratedLocation startLocation;
    if (!findNearestGenerated(start, 180.0f, 250.0f, startLocation)) return false;

    // A wander is deliberately local. Sending it through buildGeneratedPath
    // flattened every resident cell and rebuilt the full door-link graph for
    // every NPC, then repeated that A* for each unreachable random candidate.
    // Balmora's nine-cell ring is ~139k raster nodes: a dozen residents could
    // turn a local stroll into minutes of startup work. Keep the search inside
    // the actor's wander disc; scripted travel and doors still use full A*.
    struct LocalNode {
        const GeneratedCell* cell = nullptr;
        const GeneratedNode* node = nullptr;
        std::size_t localIndex = 0u;
    };
    const float radiusSquared = radius * radius;
    constexpr float kMinimumTripSquared = 180.0f * 180.0f;
    std::vector<std::pair<importer::CellCoord, const GeneratedCell*>> cells;
    cells.reserve(m_generatedCells.size());
    for (const auto& [coord, cell] : m_generatedCells) cells.emplace_back(coord, &cell);
    std::sort(cells.begin(), cells.end(), [](const auto& left, const auto& right) {
        return std::tie(left.first.x, left.first.z) <
            std::tie(right.first.x, right.first.z);
    });

    std::vector<LocalNode> nodes;
    std::map<std::pair<std::int64_t, std::int64_t>, std::vector<std::size_t>> byGrid;
    std::size_t startNode = std::numeric_limits<std::size_t>::max();
    for (const auto& [coord, cell] : cells) {
        (void)coord;
        for (std::size_t local = 0u; local < cell->nodes.size(); ++local) {
            const GeneratedNode& node = cell->nodes[local];
            if (distanceSquaredXZ(node.position, origin) > radiusSquared &&
                !(cell == startLocation.cell && local == startLocation.node)) {
                continue;
            }
            const std::size_t index = nodes.size();
            nodes.push_back(LocalNode{cell, &node, local});
            byGrid[{node.gridX, node.gridZ}].push_back(index);
            if (cell == startLocation.cell && local == startLocation.node) startNode = index;
        }
    }
    if (startNode >= nodes.size()) return false;

    const auto linkClear = [&](const LocalNode& from, const LocalNode& to) {
        const float feetY = std::min(from.node->position.y, to.node->position.y);
        const float headY = std::max(from.node->position.y, to.node->position.y) +
            std::min(from.cell->config.agentHeight, to.cell->config.agentHeight);
        const float stepHeight = std::min(
            from.cell->config.stepHeight, to.cell->config.stepHeight);
        const float radiusAroundActor = std::max(
            from.cell->config.agentRadius, to.cell->config.agentRadius);
        const float clearanceSquared = radiusAroundActor * radiusAroundActor;
        std::set<std::pair<const GeneratedCell*, std::uint32_t>> obstacles;
        for (const GeneratedCell* owner : {from.cell, to.cell}) {
            for (const auto key : {
                     std::pair{from.node->gridX, from.node->gridZ},
                     std::pair{to.node->gridX, to.node->gridZ}}) {
                const auto bucket = owner->obstacleBuckets.find(key);
                if (bucket == owner->obstacleBuckets.end()) continue;
                for (const std::uint32_t index : bucket->second) {
                    obstacles.emplace(owner, index);
                }
            }
        }
        for (const auto& [owner, index] : obstacles) {
            const GeneratedCell::Obstacle& obstacle = owner->obstacles[index];
            if (obstacle.maxY <= feetY + stepHeight + 0.5f ||
                obstacle.minY >= headY) continue;
            RasterTriangle raster;
            std::copy_n(obstacle.vertex, 3u, raster.vertex);
            if (segmentTriangleDistanceSquaredXZ(
                    from.node->position, to.node->position, raster) < clearanceSquared) {
                return false;
            }
        }
        return true;
    };
    const auto canTraverse = [&](const LocalNode& from, const LocalNode& to) {
        const float dx = to.node->position.x - from.node->position.x;
        const float dz = to.node->position.z - from.node->position.z;
        const float horizontal = std::sqrt((dx * dx) + (dz * dz));
        const float expected = std::max(
            from.cell->config.cellSize, to.cell->config.cellSize);
        if (horizontal > expected * 1.1f || horizontal < expected * 0.75f) return false;
        const float rise = std::abs(to.node->position.y - from.node->position.y);
        float allowedRise = std::min(
            from.cell->config.stepHeight, to.cell->config.stepHeight) + 0.5f;
        if (from.node->normalY < 0.995f && to.node->normalY < 0.995f) {
            const float slope = std::min(
                from.cell->config.maxSlopeDegrees, to.cell->config.maxSlopeDegrees);
            allowedRise = std::max(allowedRise,
                std::tan(slope * (odai::math::kPi / 180.0f)) * horizontal + 0.5f);
        }
        return rise <= allowedRise && linkClear(from, to);
    };

    std::vector<std::ptrdiff_t> parent(nodes.size(), -2);
    std::queue<std::size_t> frontier;
    std::vector<std::size_t> candidates;
    parent[startNode] = -1;
    frontier.push(startNode);
    while (!frontier.empty()) {
        const std::size_t current = frontier.front();
        frontier.pop();
        if (distanceSquaredXZ(nodes[current].node->position, start) >=
            kMinimumTripSquared) {
            candidates.push_back(current);
        }
        const GeneratedNode& source = *nodes[current].node;
        for (const auto offset : {std::pair{-1, 0}, std::pair{0, -1},
                 std::pair{0, 1}, std::pair{1, 0}}) {
            const auto found = byGrid.find(
                {source.gridX + offset.first, source.gridZ + offset.second});
            if (found == byGrid.end()) continue;
            for (const std::size_t target : found->second) {
                if (parent[target] != -2 || !canTraverse(nodes[current], nodes[target])) {
                    continue;
                }
                parent[target] = static_cast<std::ptrdiff_t>(current);
                frontier.push(target);
            }
        }
    }
    if (candidates.empty()) return false;

    const std::size_t target = candidates[randomValue % candidates.size()];
    std::vector<std::size_t> route;
    for (std::ptrdiff_t current = static_cast<std::ptrdiff_t>(target); current >= 0;
         current = parent[static_cast<std::size_t>(current)]) {
        route.push_back(static_cast<std::size_t>(current));
    }
    std::reverse(route.begin(), route.end());
    for (std::size_t step = 1u; step < route.size(); ++step) {
        outWaypoints.push_back(ActorNavigationStep{
            ActorNavigationStepKind::Walk, nodes[route[step]].node->position});
    }
    return !outWaypoints.empty();
}

bool ActorNavigationWorld::buildWanderPath(
    const Vector3& start,
    const Vector3& origin,
    float radius,
    std::uint32_t randomValue,
    std::vector<ActorNavigationStep>& outWaypoints) const {
    outWaypoints.clear();
    GeneratedLocation generatedStart;
    if (findNearestGenerated(start, 180.0f, 250.0f, generatedStart)) {
        return buildGeneratedWanderPath(start, origin, radius, randomValue, outWaypoints);
    }
    Location startLocation;
    if (!findNearest(start, 180.0f, 250.0f, startLocation)) {
        return false;
    }

    const Mesh& mesh = *startLocation.mesh;
    const std::size_t count = mesh.triangles.size();
    std::vector<int> parent(count, -2);
    std::queue<std::size_t> frontier;
    parent[startLocation.triangle] = -1;
    frontier.push(startLocation.triangle);
    std::vector<std::size_t> candidates;
    const float radiusSquared = radius * radius;
    constexpr float kMinimumTripSquared = 180.0f * 180.0f;

    while (!frontier.empty()) {
        const std::size_t current = frontier.front();
        frontier.pop();
        const Vector3 centroid = triangleCentroid(mesh, mesh.triangles[current]);
        if (distanceSquaredXZ(centroid, origin) <= radiusSquared &&
            distanceSquaredXZ(centroid, startLocation.point) >= kMinimumTripSquared) {
            candidates.push_back(current);
        }
        for (const std::uint16_t neighbour : mesh.triangles[current].neighbour) {
            if (neighbour == importer::fnv::kNavMeshNoNeighbour || neighbour >= count ||
                parent[neighbour] != -2) {
                continue;
            }
            parent[neighbour] = static_cast<int>(current);
            frontier.push(neighbour);
        }
    }
    if (candidates.empty()) {
        return false;
    }

    const std::size_t target = candidates[randomValue % candidates.size()];
    std::vector<std::size_t> triangles;
    for (int current = static_cast<int>(target); current >= 0; current = parent[current]) {
        triangles.push_back(static_cast<std::size_t>(current));
    }
    std::reverse(triangles.begin(), triangles.end());

    // Route through each shared edge midpoint. A triangle centroid connected
    // directly to another triangle's centroid is also valid, but explicit edge
    // crossings keep every finite-precision segment visibly centred on paths.
    for (std::size_t step = 1; step < triangles.size(); ++step) {
        const Triangle& before = mesh.triangles[triangles[step - 1u]];
        const Triangle& after = mesh.triangles[triangles[step]];
        std::uint16_t shared[2] = {};
        int sharedCount = 0;
        for (const std::uint16_t a : before.vertex) {
            for (const std::uint16_t b : after.vertex) {
                if (a == b && sharedCount < 2) {
                    shared[sharedCount++] = a;
                }
            }
        }
        if (sharedCount == 2) {
            const Vector3& a = mesh.vertices[shared[0]];
            const Vector3& b = mesh.vertices[shared[1]];
            outWaypoints.push_back(ActorNavigationStep{
                ActorNavigationStepKind::Walk,
                Vector3{(a.x + b.x) * 0.5f,
                    (a.y + b.y) * 0.5f,
                    (a.z + b.z) * 0.5f}});
        }
    }
    outWaypoints.push_back(ActorNavigationStep{ActorNavigationStepKind::Walk,
        triangleCentroid(mesh, mesh.triangles[target])});
    return !outWaypoints.empty();
}

bool ActorNavigationWorld::buildPath(
    const Vector3& start,
    const Vector3& goal,
    std::vector<ActorNavigationStep>& outWaypoints) const {
    outWaypoints.clear();
    GeneratedLocation generatedStart;
    GeneratedLocation generatedGoal;
    if (findNearestGenerated(start, 220.0f, 300.0f, generatedStart) &&
        findNearestGenerated(goal, 500.0f, 700.0f, generatedGoal)) {
        return buildGeneratedPath(start, goal, outWaypoints);
    }
    Location startLocation;
    Location goalLocation;
    if (!findNearest(start, 220.0f, 300.0f, startLocation) ||
        !findNearest(goal, 500.0f, 700.0f, goalLocation)) {
        return false;
    }

    // Flatten resident meshes in authored identity order. Streaming completion
    // order and unordered-map bucket order must not affect path selection.
    std::vector<const Mesh*> meshes;
    for (const auto& [cell, cellMeshes] : m_cells) {
        (void)cell;
        for (const Mesh& mesh : cellMeshes) meshes.push_back(&mesh);
    }
    std::sort(meshes.begin(), meshes.end(), [](const Mesh* left, const Mesh* right) {
        return std::tie(left->cell.x, left->cell.z, left->formId) <
            std::tie(right->cell.x, right->cell.z, right->formId);
    });
    std::unordered_map<const Mesh*, std::size_t> meshIndices;
    std::vector<std::size_t> offsets(meshes.size() + 1u, 0u);
    for (std::size_t index = 0u; index < meshes.size(); ++index) {
        meshIndices.emplace(meshes[index], index);
        offsets[index + 1u] = offsets[index] + meshes[index]->triangles.size();
    }
    const auto startMesh = meshIndices.find(startLocation.mesh);
    const auto goalMesh = meshIndices.find(goalLocation.mesh);
    if (startMesh == meshIndices.end() || goalMesh == meshIndices.end()) return false;
    const std::size_t startNode = offsets[startMesh->second] + startLocation.triangle;
    const std::size_t goalNode = offsets[goalMesh->second] + goalLocation.triangle;

    struct Link {
        std::size_t target = 0u;
        Vector3 crossing{};
        ActorNavigationStepKind kind = ActorNavigationStepKind::Walk;
        Vector3 arrival{};
        std::uint32_t doorReferenceFormId = 0u;
    };
    std::vector<std::vector<Link>> graph(offsets.back());
    const auto sharedEdgeMidpoint = [](const Mesh& mesh,
                                        const Triangle& left,
                                        const Triangle& right) {
        std::array<std::uint16_t, 2u> shared{};
        std::size_t count = 0u;
        for (const std::uint16_t a : left.vertex) {
            for (const std::uint16_t b : right.vertex) {
                if (a == b && count < shared.size()) shared[count++] = a;
            }
        }
        if (count == 2u) {
            const Vector3& a = mesh.vertices[shared[0]];
            const Vector3& b = mesh.vertices[shared[1]];
            return Vector3{(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f,
                (a.z + b.z) * 0.5f};
        }
        return triangleCentroid(mesh, right);
    };
    for (std::size_t meshIndex = 0u; meshIndex < meshes.size(); ++meshIndex) {
        const Mesh& mesh = *meshes[meshIndex];
        for (std::size_t triangleIndex = 0u;
             triangleIndex < mesh.triangles.size(); ++triangleIndex) {
            const Triangle& triangle = mesh.triangles[triangleIndex];
            const std::size_t node = offsets[meshIndex] + triangleIndex;
            for (const std::uint16_t neighbour : triangle.neighbour) {
                if (neighbour == importer::fnv::kNavMeshNoNeighbour ||
                    neighbour >= mesh.triangles.size()) continue;
                graph[node].push_back(Link{offsets[meshIndex] + neighbour,
                    sharedEdgeMidpoint(mesh, triangle, mesh.triangles[neighbour])});
            }
        }
    }

    // Skyrim splits exterior navigation at cell/NAVM boundaries. Match border
    // edges at 1/16 Bethesda-unit precision; this tolerates harmless float
    // roundoff without bridging visible gaps or different floors.
    using QuantizedPoint = std::array<std::int64_t, 3u>;
    using EdgeKey = std::array<std::int64_t, 6u>;
    struct EdgeOccurrence {
        std::size_t mesh = 0u;
        std::size_t node = 0u;
        Vector3 midpoint{};
    };
    const auto quantize = [](const Vector3& value) {
        return QuantizedPoint{static_cast<std::int64_t>(std::llround(value.x * 16.0f)),
            static_cast<std::int64_t>(std::llround(value.y * 16.0f)),
            static_cast<std::int64_t>(std::llround(value.z * 16.0f))};
    };
    std::map<EdgeKey, std::vector<EdgeOccurrence>> edgeOccurrences;
    for (std::size_t meshIndex = 0u; meshIndex < meshes.size(); ++meshIndex) {
        const Mesh& mesh = *meshes[meshIndex];
        for (std::size_t triangleIndex = 0u;
             triangleIndex < mesh.triangles.size(); ++triangleIndex) {
            const Triangle& triangle = mesh.triangles[triangleIndex];
            for (std::size_t edge = 0u; edge < 3u; ++edge) {
                const Vector3& a = mesh.vertices[triangle.vertex[edge]];
                const Vector3& b = mesh.vertices[triangle.vertex[(edge + 1u) % 3u]];
                QuantizedPoint qa = quantize(a);
                QuantizedPoint qb = quantize(b);
                if (qb < qa) std::swap(qa, qb);
                EdgeKey key{qa[0], qa[1], qa[2], qb[0], qb[1], qb[2]};
                edgeOccurrences[key].push_back(EdgeOccurrence{meshIndex,
                    offsets[meshIndex] + triangleIndex,
                    Vector3{(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f,
                        (a.z + b.z) * 0.5f}});
            }
        }
    }
    for (const auto& [edge, occurrences] : edgeOccurrences) {
        (void)edge;
        for (std::size_t left = 0u; left < occurrences.size(); ++left) {
            const std::size_t leftCount = static_cast<std::size_t>(std::count_if(
                occurrences.begin(), occurrences.end(), [&](const EdgeOccurrence& value) {
                    return value.mesh == occurrences[left].mesh;
                }));
            if (leftCount != 1u) continue;  // an interior edge in this mesh
            for (std::size_t right = left + 1u; right < occurrences.size(); ++right) {
                if (occurrences[left].mesh == occurrences[right].mesh) continue;
                const std::size_t rightCount = static_cast<std::size_t>(std::count_if(
                    occurrences.begin(), occurrences.end(), [&](const EdgeOccurrence& value) {
                        return value.mesh == occurrences[right].mesh;
                    }));
                if (rightCount != 1u) continue;
                const Vector3 crossing{
                    (occurrences[left].midpoint.x + occurrences[right].midpoint.x) * 0.5f,
                    (occurrences[left].midpoint.y + occurrences[right].midpoint.y) * 0.5f,
                    (occurrences[left].midpoint.z + occurrences[right].midpoint.z) * 0.5f};
                graph[occurrences[left].node].push_back(
                    Link{occurrences[right].node, crossing});
                graph[occurrences[right].node].push_back(
                    Link{occurrences[left].node, crossing});
            }
        }
    }

    // NAVM's NVDP table identifies the exact triangle at a source door. The
    // imported door resolves its XTEL arrival into engine space. When that
    // arrival projects onto another resident NAVM, connect the two with a
    // typed one-way edge rather than a fake segment across the world.
    std::unordered_map<std::uint32_t, std::vector<std::size_t>> portalNodes;
    for (std::size_t meshIndex = 0u; meshIndex < meshes.size(); ++meshIndex) {
        const Mesh& mesh = *meshes[meshIndex];
        for (const importer::fnv::FalloutNavMeshDoorPortal& portal : mesh.doorPortals) {
            if (portal.doorRefFormId == 0u || portal.triangleIndex >= mesh.triangles.size()) {
                continue;
            }
            portalNodes[portal.doorRefFormId].push_back(
                offsets[meshIndex] + portal.triangleIndex);
        }
    }
    for (auto& [reference, nodes] : portalNodes) {
        (void)reference;
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    }
    for (const importer::ImportedSceneDoor& door : m_residentDoors) {
        const auto sources = portalNodes.find(door.sourceReferenceFormId);
        if (sources == portalNodes.end()) continue;
        Location arrivalLocation;
        const Vector3 authoredArrival{
            door.arrivalPosition[0], door.arrivalPosition[1], door.arrivalPosition[2]};
        if (!findNearest(authoredArrival, 500.0f, 700.0f, arrivalLocation)) continue;
        const auto arrivalMesh = meshIndices.find(arrivalLocation.mesh);
        if (arrivalMesh == meshIndices.end()) continue;
        const std::size_t arrivalNode =
            offsets[arrivalMesh->second] + arrivalLocation.triangle;
        for (const std::size_t sourceNode : sources->second) {
            const std::size_t sourceMeshIndex = static_cast<std::size_t>(
                std::upper_bound(offsets.begin(), offsets.end(), sourceNode) - offsets.begin() - 1);
            const Mesh& sourceMesh = *meshes[sourceMeshIndex];
            const Vector3 sourcePoint = triangleCentroid(
                sourceMesh, sourceMesh.triangles[sourceNode - offsets[sourceMeshIndex]]);
            graph[sourceNode].push_back(Link{arrivalNode, sourcePoint,
                ActorNavigationStepKind::ActivateDoor, arrivalLocation.point,
                door.sourceReferenceFormId});
        }
    }
    for (std::vector<Link>& links : graph) {
        std::sort(links.begin(), links.end(), [](const Link& left, const Link& right) {
            return std::tie(left.target, left.kind, left.doorReferenceFormId,
                       left.crossing.x, left.crossing.y, left.crossing.z) <
                std::tie(right.target, right.kind, right.doorReferenceFormId,
                       right.crossing.x, right.crossing.y, right.crossing.z);
        });
        links.erase(std::unique(links.begin(), links.end(), [](const Link& left, const Link& right) {
            return left.target == right.target && left.kind == right.kind &&
                left.doorReferenceFormId == right.doorReferenceFormId;
        }), links.end());
    }

    std::vector<std::ptrdiff_t> parent(graph.size(), -2);
    std::vector<Link> parentLink(graph.size());
    std::queue<std::size_t> frontier;
    parent[startNode] = -1;
    frontier.push(startNode);
    while (!frontier.empty() && parent[goalNode] == -2) {
        const std::size_t current = frontier.front();
        frontier.pop();
        for (const Link& link : graph[current]) {
            if (parent[link.target] != -2) continue;
            parent[link.target] = static_cast<std::ptrdiff_t>(current);
            parentLink[link.target] = link;
            frontier.push(link.target);
        }
    }
    if (parent[goalNode] == -2) return false;

    std::vector<std::size_t> nodes;
    for (std::ptrdiff_t current = static_cast<std::ptrdiff_t>(goalNode); current >= 0;
         current = parent[static_cast<std::size_t>(current)]) {
        nodes.push_back(static_cast<std::size_t>(current));
    }
    std::reverse(nodes.begin(), nodes.end());
    for (std::size_t step = 1u; step < nodes.size(); ++step) {
        const Link& link = parentLink[nodes[step]];
        outWaypoints.push_back(ActorNavigationStep{link.kind, link.crossing,
            link.arrival, link.doorReferenceFormId});
    }
    outWaypoints.push_back(
        ActorNavigationStep{ActorNavigationStepKind::Walk, goalLocation.point});
    return true;
}

std::size_t ActorNavigationWorld::meshCount() const {
    std::size_t count = 0u;
    for (const auto& [cell, meshes] : m_cells) {
        (void)cell;
        count += meshes.size();
    }
    return count;
}

std::size_t ActorNavigationWorld::triangleCount() const {
    std::size_t count = 0u;
    for (const auto& [cell, meshes] : m_cells) {
        (void)cell;
        for (const Mesh& mesh : meshes) {
            count += mesh.triangles.size();
        }
    }
    return count;
}

std::size_t ActorNavigationWorld::generatedNodeCount() const {
    std::size_t count = 0u;
    for (const auto& [cell, generated] : m_generatedCells) {
        (void)cell;
        count += generated.nodes.size();
    }
    return count;
}

}  // namespace odai::games::newvegas
