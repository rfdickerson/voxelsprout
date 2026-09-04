#include "import/fnv/land_lod.h"

#include "import/dds.h"
#include "import/fnv/nif_scene.h"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <utility>
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
void writeLodInstanceTransform(
    ImportedSceneInstance& instance, float sinkUnits,
    float bethesdaOffsetX = 0.0f, float bethesdaOffsetY = 0.0f) {
    instance.transform[0] = 1.0f;
    instance.transform[6] = 1.0f;
    instance.transform[9] = -1.0f;
    instance.transform[15] = 1.0f;
    // Skyrim BTR terrain is local to its filename's corner cell. Convert that
    // tile origin through the same (x, y, z) -> (x, z, -y) basis as vertices.
    instance.transform[3] = bethesdaOffsetX;
    instance.transform[11] = -bethesdaOffsetY;
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
    const auto textureIndexFor = [&](const std::string& texturePath, bool linearData = false)
        -> std::uint32_t {
        if (texturePath.empty()) {
            return kNoTextureIndex;
        }
        std::string key = toLowerCopy(texturePath);
        if (linearData) {
            key += "|linear-data";
        }
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
        if (linearData && texture.format == TextureFormat::BC1) {
            texture.format = TextureFormat::BC1Linear;
        }
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

    struct PendingMesh {
        ImportedSceneMesh mesh;
        std::int32_t tileX = 0;
        std::int32_t tileZ = 0;
    };
    std::vector<PendingMesh> tessellatedMeshes;
    std::vector<PendingMesh> ordinaryMeshes;
    // The renderer's tessellated range is a leading draw prefix. All runtime
    // and cooker callers build one LOD set into a fresh scene; retain safe
    // append semantics for tooling by declining the optimization if a caller
    // has already placed unrelated geometry ahead of us.
    const bool canEstablishDistantTessellationPrefix =
        out.meshes.empty() && out.instances.empty() &&
        out.sourceLandscapeCellCount == 0u;

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

            const std::string tileName =
                "lod" + std::to_string(tierCells) + "_" + std::to_string(tx) + "_" +
                std::to_string(tz);
            ImportedSceneMesh detailMesh;
            detailMesh.name = set == LandLodSet::SkyrimObjects
                ? tileName + "_detail"
                : tileName;
            ImportedSceneMesh largeRefMesh;
            largeRefMesh.name = tileName + "_largeref";
            ImportedSceneMesh mountainMesh;
            mountainMesh.name = tileName + "_mountain";
            const auto appendShape = [&](ImportedSceneMesh& mesh, const NifShape& shape,
                                         bool distantMountain, bool distantMountainSnow) {
                const auto baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
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
                    if ((v * 4u) + 3u < shape.colors.size()) {
                        vertex.colorAlpha = shape.colors[(v * 4u) + 3u];
                    }
                    mesh.vertices.push_back(vertex);
                }
                ImportedSceneMeshPart part{};
                part.firstIndex = static_cast<std::uint32_t>(mesh.indices.size());
                part.textureIndex = textureIndexFor(shape.diffuseTexturePath);
                const std::uint32_t normalTextureIndex =
                    textureIndexFor(shape.normalTexturePath, /*linearData=*/true);
                if (part.textureIndex != kNoTextureIndex &&
                    normalTextureIndex != kNoTextureIndex) {
                    out.normalTextureByDiffuseIndex.emplace(
                        part.textureIndex, normalTextureIndex);
                }
                if (distantMountain && canEstablishDistantTessellationPrefix) {
                    part.vegetationReserved[0] |=
                        kImportedSceneMeshPartDistantLodTessellation;
                    if (distantMountainSnow) {
                        part.vegetationReserved[0] |=
                            kImportedSceneMeshPartDistantLodSnow;
                    }
                }
                // Skyrim's generated object atlas uses alpha as auxiliary
                // material data. Trees have their own BTT/TreeLod atlas and
                // are not present in these BTO shapes, so inferring cutouts
                // from Tamriel.Objects.DDS punches holes in roofs and walls.
                // Fallout/Oblivion object LOD still gets an authored true here
                // when its NIF explicitly carries NiAlphaProperty.
                part.alphaTest = shape.alphaTest;
                // A generated BTO building is a hollow exterior shell. Its
                // merged triangles do not consistently face the camera from
                // every authored approach, so ordinary back-face culling cuts
                // roofs and walls out of the skyline. This is the same rule
                // CellSceneBuilder applies to placed *LOD.nif shells, extended
                // to Skyrim's generated object-LOD container.
                part.twoSided = shape.twoSided || set == LandLodSet::SkyrimObjects ||
                    set == LandLodSet::SkyrimTerrain;
                for (const std::uint32_t index : shape.triangleIndices) {
                    mesh.indices.push_back(baseVertex + index);
                }
                part.indexCount =
                    static_cast<std::uint32_t>(mesh.indices.size()) - part.firstIndex;
                if (part.indexCount != 0u) {
                    mesh.parts.push_back(part);
                    outStats.triangles += part.indexCount / 3u;
                }
            };
            for (const NifShape& shape : nifModel.shapes) {
                // A Skyrim BTR can carry a small, flat WATER shape alongside
                // the textured land mesh. It has no diffuse texture and needs
                // the runtime water material/reflection path, not the opaque
                // static path used by this LOD ring. Emitting it here paints a
                // featureless white slab across valleys and lakes.
                if (set == LandLodSet::SkyrimTerrain &&
                    shape.diffuseTexturePath.empty()) {
                    continue;
                }
                // Skyrim deliberately separates large references from the
                // ordinary per-tile objects inside a BTO. The runtime hands
                // ordinary detail back to streamed cells near the camera, but
                // a walled child worldspace's distant building shell exists
                // only in the LargeRef shape. Keeping both in one mesh made
                // that handoff delete roofs and towers together with rocks and
                // props -- the partial-Whiterun skyline symptom.
                const std::string loweredShapeName = toLowerCopy(shape.name);
                const bool largeReference =
                    set == LandLodSet::SkyrimObjects &&
                    loweredShapeName.find("largeref") != std::string::npos &&
                    // The HD variants are mountains/rocks with their own
                    // vertex-alpha handoff, not a child-worldspace city shell.
                    loweredShapeName.find("hd-largeref") == std::string::npos;
                const bool distantMountain =
                    set == LandLodSet::SkyrimObjects &&
                    loweredShapeName.find("hd-largeref") != std::string::npos;
                const bool distantMountainSnow =
                    distantMountain && loweredShapeName.find("snow") != std::string::npos;
                appendShape(
                    distantMountain ? mountainMesh
                                    : (largeReference ? largeRefMesh : detailMesh),
                    shape, distantMountain, distantMountainSnow);
            }

            bool emittedTile = false;
            const auto queueMesh = [&](ImportedSceneMesh& mesh, bool tessellated) {
                if (mesh.indices.empty()) {
                    return;
                }
                (tessellated ? tessellatedMeshes : ordinaryMeshes).push_back(
                    PendingMesh{std::move(mesh), tx, tz});
                emittedTile = true;
            };
            queueMesh(mountainMesh, true);
            queueMesh(detailMesh, false);
            queueMesh(largeRefMesh, false);
            outStats.tilesParsed += emittedTile ? 1u : 0u;
        }
    }

    const auto emitPending = [&](std::vector<PendingMesh>& pending) {
        for (PendingMesh& item : pending) {
            const auto meshIndex = static_cast<std::uint32_t>(out.meshes.size());
            out.meshes.push_back(std::move(item.mesh));
            ImportedSceneInstance instance{};
            instance.meshIndex = meshIndex;
            const bool localTerrain = set == LandLodSet::SkyrimTerrain;
            writeLodInstanceTransform(
                instance, sinkUnits,
                localTerrain ? static_cast<float>(item.tileX) * kExteriorCellSize : 0.0f,
                localTerrain ? static_cast<float>(item.tileZ) * kExteriorCellSize : 0.0f);
            out.instances.push_back(std::move(instance));
        }
    };
    // Packed rendering emits instances in this order. Keeping the selected
    // mountain proxies first preserves the renderer's contiguous
    // [0, sourceLandscapeCellCount) tessellation invariant.
    emitPending(tessellatedMeshes);
    if (canEstablishDistantTessellationPrefix) {
        out.sourceLandscapeCellCount = static_cast<std::uint32_t>(
            std::accumulate(
                out.meshes.begin(), out.meshes.end(), std::size_t{0},
                [](std::size_t count, const ImportedSceneMesh& mesh) {
                    return count + static_cast<std::size_t>(std::count_if(
                        mesh.parts.begin(), mesh.parts.end(),
                        [](const ImportedSceneMeshPart& part) {
                            return (part.vegetationReserved[0] &
                                    kImportedSceneMeshPartDistantLodTessellation) != 0u;
                        }));
                }));
    }
    emitPending(ordinaryMeshes);

    outStats.textures = out.textures.size();
    if (outStats.tilesParsed == 0u) {
        outError = "no LOD tiles parsed for worldspace \"" + worldspaceEditorId + "\" at tier " +
                   std::to_string(tierCells);
        return false;
    }
    // Every part's alpha mode is explicit: Fallout/Oblivion get it from
    // NiAlphaProperty, while Skyrim BTO architecture is opaque. Do not let
    // ImportedScene's whole-texture inference alpha-test the combined atlas.
    out.alphaFlagsAuthored = true;
    return true;
}

}  // namespace odai::importer::fnv
