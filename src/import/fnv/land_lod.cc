#include "import/fnv/land_lod.h"

#include "import/dds.h"
#include "import/fnv/nif_scene.h"

#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <vector>

namespace odai::importer::fnv {

namespace {

// Same sentinel ImportedSceneMeshPart defaults to. Spelled here rather than
// shared with cell_builder.cc's private copy, since a second file reaching into
// the first for a constant is worse than two files agreeing with the header.
constexpr std::uint32_t kNoTextureIndex = 0xffffffffu;

std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

// A LOD tile's vertices are already in WORLD units and in Bethesda's Z-up
// space, unlike a static model's, which are in model space and placed by a REFR
// transform. So the instance carries no translation and no rotation of its own
// -- only the Z-up -> Y-up basis change, written directly rather than composed,
// because there is nothing to compose it with.
//
// The mapping is (x, y, z) -> (x, z, -y), the same one bethesdaToEngine applies
// to positions elsewhere. Doing it here instead of on every vertex keeps the
// tile's geometry byte-identical to what the file stores, which is what makes a
// mismatch debuggable against the source asset.
//
// Written literally rather than by negating an identity basis, which is what
// the cooker used to do. Those two differ: negating produces NEGATIVE ZERO in
// the row-2 entries, so a cooked scene from the old path and this one disagree
// in 59206 bytes at identical file size. Every one of them is the sign bit of a
// float that is zero in both, so the scenes are numerically identical -- worth
// knowing before treating a hash mismatch on this file as a regression.
void writeLodInstanceTransform(ImportedSceneInstance& instance, float sinkUnits) {
    instance.transform[0] = 1.0f;
    instance.transform[6] = 1.0f;
    instance.transform[9] = -1.0f;
    instance.transform[15] = 1.0f;
    // Sink applies in ENGINE space, where +y is up, so it lands in the
    // translation row rather than being folded into the basis.
    //
    // Guarded against zero so a sink of 0 writes +0.0 rather than -0.0.
    if (sinkUnits != 0.0f) {
        instance.transform[7] = -sinkUnits;
    }
}

}  // namespace

bool appendLandLodTier(
    const LandLodByteResolver& resolveMesh,
    const LandLodByteResolver& resolveTexture,
    const std::string& worldspaceEditorId,
    LandLodSet set,
    std::int32_t tierCells,
    std::int32_t cellX0, std::int32_t cellZ0, std::int32_t cellX1, std::int32_t cellZ1,
    float sinkUnits,
    ImportedScene& out,
    LandLodTierStats& outStats,
    std::string& outError) {
    outStats = LandLodTierStats{};
    if (!landLodTierExists(set, tierCells)) {
        outError = "no LOD tier " + std::to_string(tierCells) + " exists for that set";
        return false;
    }

    // Texture indices are memoized across the whole tier, INCLUDING the misses.
    // A terrain tile names its own per-tile diffuse, so a tier is one texture
    // per tile and the cache mostly does not hit -- but the object set shares a
    // single atlas across every tile, where it saves decoding it hundreds of
    // times.
    std::unordered_map<std::string, std::uint32_t> textureIndexByPath;
    const auto textureIndexFor = [&](const std::string& texturePath) -> std::uint32_t {
        if (texturePath.empty()) {
            return kNoTextureIndex;
        }
        const std::string key = toLowerCopy(texturePath);
        const auto existing = textureIndexByPath.find(key);
        if (existing != textureIndexByPath.end()) {
            return existing->second;
        }
        std::vector<std::uint8_t> ddsBytes;
        ImportedSceneTexture texture;
        if (!resolveTexture(texturePath, ddsBytes) ||
            !loadDdsFromMemory(ddsBytes.data(), ddsBytes.size(), texture)) {
            textureIndexByPath.emplace(key, kNoTextureIndex);
            return kNoTextureIndex;
        }
        texture.sourcePath = texturePath;
        const auto index = static_cast<std::uint32_t>(out.textures.size());
        out.textures.push_back(std::move(texture));
        textureIndexByPath.emplace(key, index);
        return index;
    };

    const std::string loweredWorldspace = toLowerCopy(worldspaceEditorId);
    // Snap to the tile lattice: a name built from an arbitrary cell coordinate
    // resolves to nothing, which is indistinguishable from the sparse-grid hole
    // it is not.
    const std::int32_t x0 = landLodTileOrigin(std::min(cellX0, cellX1), tierCells);
    const std::int32_t z0 = landLodTileOrigin(std::min(cellZ0, cellZ1), tierCells);
    const std::int32_t x1 = landLodTileOrigin(std::max(cellX0, cellX1), tierCells);
    const std::int32_t z1 = landLodTileOrigin(std::max(cellZ0, cellZ1), tierCells);

    for (std::int32_t tz = z0; tz <= z1; tz += tierCells) {
        for (std::int32_t tx = x0; tx <= x1; tx += tierCells) {
            const std::string tilePath =
                landLodTilePath(loweredWorldspace, set, tierCells, tx, tz);
            std::vector<std::uint8_t> nifBytes;
            if (!resolveMesh(tilePath, nifBytes)) {
                ++outStats.tilesMissing;  // sparse grid; normal
                continue;
            }
            ++outStats.tilesResolved;

            NifModel nifModel;
            std::string nifError;
            if (!parseNifStaticMesh(nifBytes, nifModel, nifError) || nifModel.shapes.empty()) {
                continue;
            }

            ImportedSceneMesh mesh;
            mesh.name = "lod" + std::to_string(tierCells) + "_" + std::to_string(tx) + "_" +
                        std::to_string(tz);
            for (const NifShape& shape : nifModel.shapes) {
                const auto baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
                const auto partFirstIndex = static_cast<std::uint32_t>(mesh.indices.size());
                for (std::size_t v = 0; (v * 3u) < shape.positions.size(); ++v) {
                    ImportedSceneVertex vertex{};
                    vertex.position[0] = shape.positions[v * 3u];
                    vertex.position[1] = shape.positions[(v * 3u) + 1u];
                    vertex.position[2] = shape.positions[(v * 3u) + 2u];
                    if ((v * 3u) + 2u < shape.normals.size()) {
                        vertex.normal[0] = shape.normals[v * 3u];
                        vertex.normal[1] = shape.normals[(v * 3u) + 1u];
                        vertex.normal[2] = shape.normals[(v * 3u) + 2u];
                    }
                    if ((v * 2u) + 1u < shape.uvs.size()) {
                        vertex.uv[0] = shape.uvs[v * 2u];
                        vertex.uv[1] = shape.uvs[(v * 2u) + 1u];
                    }
                    mesh.vertices.push_back(vertex);
                }
                for (const std::uint32_t index : shape.triangleIndices) {
                    mesh.indices.push_back(baseVertex + index);
                }
                const auto partIndexCount =
                    static_cast<std::uint32_t>(mesh.indices.size()) - partFirstIndex;
                if (partIndexCount == 0u) {
                    continue;
                }
                ImportedSceneMeshPart part{};
                part.firstIndex = partFirstIndex;
                part.indexCount = partIndexCount;
                part.textureIndex = textureIndexFor(shape.diffuseTexturePath);
                part.alphaTest = shape.alphaTest;
                mesh.parts.push_back(part);
                outStats.triangles += partIndexCount / 3u;
            }
            if (mesh.indices.empty()) {
                continue;
            }

            const auto meshIndex = static_cast<std::uint32_t>(out.meshes.size());
            out.meshes.push_back(std::move(mesh));
            ImportedSceneInstance instance{};
            instance.meshIndex = meshIndex;
            writeLodInstanceTransform(instance, sinkUnits);
            out.instances.push_back(std::move(instance));
            ++outStats.tilesParsed;
        }
    }

    outStats.textures = out.textures.size();
    if (outStats.tilesParsed == 0u) {
        outError = "no LOD tiles parsed for worldspace \"" + worldspaceEditorId + "\" at tier " +
                   std::to_string(tierCells);
        return false;
    }
    // Every part's alpha mode came off its own NiAlphaProperty, so the
    // texture-content cutout guess must not run over this on load -- the same
    // statement CellSceneBuilder::finish() makes, and for the same reason.
    out.alphaFlagsAuthored = true;
    return true;
}

}  // namespace odai::importer::fnv
