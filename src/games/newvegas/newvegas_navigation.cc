#include "games/newvegas/newvegas_navigation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
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
    }
}

void ActorNavigationWorld::removeCell(const importer::CellCoord& cell) {
    m_cells.erase(cell);
}

void ActorNavigationWorld::clear() {
    m_cells.clear();
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

bool ActorNavigationWorld::projectPoint(
    float worldX,
    float worldY,
    float worldZ,
    float maxHorizontalDistance,
    float maxVerticalDistance,
    Vector3& outPoint) const {
    Location location;
    if (!findNearest(
            Vector3{worldX, worldY, worldZ},
            maxHorizontalDistance, maxVerticalDistance, location)) {
        return false;
    }
    outPoint = location.point;
    return true;
}

bool ActorNavigationWorld::buildWanderPath(
    const Vector3& start,
    const Vector3& origin,
    float radius,
    std::uint32_t randomValue,
    std::vector<ActorNavigationStep>& outWaypoints) const {
    outWaypoints.clear();
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

}  // namespace odai::games::newvegas
