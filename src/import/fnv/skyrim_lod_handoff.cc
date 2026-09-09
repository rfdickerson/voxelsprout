#include "import/fnv/skyrim_lod_handoff.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace odai::importer::fnv {
void applySkyrimLodHandoff(ImportedScene& scene, const SkyrimLodHandoff& h) {
    std::vector<bool> remove(scene.meshes.size(), h.dropAll);
    for (std::size_t n = 0; n < scene.meshes.size(); ++n) {
        auto& mesh = scene.meshes[n];
        if (h.dropAll) continue;
        if (!h.clipResident) {
            remove[n] = h.dropRegular && !mesh.name.ends_with("_largeref");
            continue;
        }
        std::vector<std::uint32_t> indices;
        std::vector<ImportedSceneMeshPart> parts;
        indices.reserve(mesh.indices.size());
        for (const auto& source : mesh.parts) {
            auto part = source;
            part.firstIndex = static_cast<std::uint32_t>(indices.size());
            const auto end = std::min(mesh.indices.size(),
                std::size_t(source.firstIndex) + source.indexCount);
            for (std::size_t i = source.firstIndex; i + 2 < end; i += 3) {
                const auto a = mesh.indices[i], b = mesh.indices[i+1], c = mesh.indices[i+2];
                if (a >= mesh.vertices.size() || b >= mesh.vertices.size() || c >= mesh.vertices.size()) continue;
                const auto cell = [&](int axis) {
                    return static_cast<std::int32_t>(std::floor(
                        (mesh.vertices[a].position[axis] + mesh.vertices[b].position[axis] +
                         mesh.vertices[c].position[axis]) / (3.0f * 4096.0f)));
                };
                const int x = cell(0) - h.tileX, z = cell(1) - h.tileZ;
                if (x >= 0 && x < 4 && z >= 0 && z < 4 &&
                    (h.residentMask & (1u << (z * 4 + x)))) continue;
                indices.insert(indices.end(), {a, b, c});
            }
            part.indexCount = static_cast<std::uint32_t>(indices.size()) - part.firstIndex;
            if (part.indexCount) parts.push_back(part);
        }
        mesh.indices = std::move(indices);
        mesh.parts = std::move(parts);
        remove[n] = mesh.indices.empty();
    }
    std::erase_if(scene.instances, [&](const ImportedSceneInstance& instance) {
        return instance.meshIndex >= remove.size() || remove[instance.meshIndex];
    });
    buildImportedScenePackedRenderData(scene);
    buildImportedScenePageRanges(scene);
}
}
