#include "import/imported_scene_query.h"

#include <algorithm>

#include "import/imported_material.h"

namespace odai::importer {

ImportedSceneRayHit raycastImportedScene(const ImportedScene& scene, const math::Ray& ray,
                                         float maxDistance) {
    ImportedSceneRayHit result{};
    if (scene.packedVertices.empty() || scene.packedIndices.empty() ||
        scene.packedDraws.empty() || maxDistance <= 0.0f) {
        return result;
    }
    if (math::lengthSquared(ray.direction) <= 0.0f) {
        return result;
    }

    const auto packedPosition = [](const ImportedScenePackedVertex& vertex) {
        return math::Vector3{vertex.position[0], vertex.position[1], vertex.position[2]};
    };

    constexpr float kRayEpsilon = 1e-4f;
    float bestDistance = maxDistance;
    std::uint32_t triangleOrdinal = 0;

    for (std::uint32_t drawIndex = 0; drawIndex < scene.packedDraws.size(); ++drawIndex) {
        const ImportedScenePackedDraw& draw = scene.packedDraws[drawIndex];
        const std::uint32_t lastIndex =
            std::min<std::uint32_t>(draw.firstIndex + draw.indexCount,
                                    static_cast<std::uint32_t>(scene.packedIndices.size()));
        for (std::uint32_t index = draw.firstIndex; index + 2u < lastIndex;
             index += 3u, ++triangleOrdinal) {
            const std::uint32_t i0 = scene.packedIndices[index];
            const std::uint32_t i1 = scene.packedIndices[index + 1u];
            const std::uint32_t i2 = scene.packedIndices[index + 2u];
            if (i0 >= scene.packedVertices.size() || i1 >= scene.packedVertices.size() ||
                i2 >= scene.packedVertices.size()) {
                continue;
            }
            const math::RayTriangleHit triangleHit = math::intersectRayTriangle(
                ray, packedPosition(scene.packedVertices[i0]),
                packedPosition(scene.packedVertices[i1]),
                packedPosition(scene.packedVertices[i2]), kRayEpsilon);
            if (!triangleHit.hit || triangleHit.distance >= bestDistance) {
                continue;
            }

            // i0 rather than an interpolation: inFlags is `nointerpolation`, so
            // the provoking vertex is what the GPU actually shades this
            // triangle with. Reading any other vertex could report a material
            // the surface is not drawn in.
            const ImportedScenePackedVertex& provoking = scene.packedVertices[i0];
            bestDistance = triangleHit.distance;
            result.hit = true;
            result.distance = triangleHit.distance;
            result.position = ray.origin + (ray.direction * triangleHit.distance);
            result.drawIndex = drawIndex;
            result.triangleIndex = triangleOrdinal;
            result.textureIndex = provoking.textureIndex;
            result.flags = provoking.flags;
            result.materialIndex = importedSceneMaterialIndex(provoking.flags);
        }
    }
    return result;
}

}  // namespace odai::importer
