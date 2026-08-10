#include "games/newvegas/newvegas_victor.h"

#include "import/fnv/asset_source.h"
#include "import/fnv/dialogue_records.h"
#include "import/fnv/nif_scene.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <system_error>

namespace odai::games::newvegas {

namespace {

constexpr const char* kVictorEditorId = "Victor";
constexpr const char* kSecuritronModel = "creatures\\securitron\\securitron_static.nif";
// He stands ~2.6 m tall; talk range is generous enough that you do not have to
// hunt for a spot, tight enough that you cannot start a conversation from
// across the road. Bethesda units, ~70 per metre.
constexpr float kTalkRange = 260.0f;
constexpr float kTalkFacingDot = 0.25f;

}  // namespace

bool loadVictor(
    const std::filesystem::path& dataFilesPath,
    const std::filesystem::path& pluginPath,
    VictorState& outState,
    odai::importer::ImportedScene& outScene
) {
    outState = VictorState{};
    outScene = odai::importer::ImportedScene{};

    importer::fnv::SpeakerPlacement placement;
    std::string error;
    if (!importer::fnv::findSpeakerPlacement(pluginPath, kVictorEditorId, placement, error)) {
        outState.status = "not placed: " + error;
        return false;
    }
    // Bethesda is Z-up; this engine is Y-up. Same conversion cell_builder makes
    // for every other reference: (x, y, z) -> (x, z, -y).
    outState.position[0] = placement.position[0];
    outState.position[1] = placement.position[2];
    outState.position[2] = -placement.position[1];

    importer::fnv::DialogueImportStats stats;
    if (!importer::fnv::buildSpeakerDialogueTree(
            pluginPath, kVictorEditorId, outState.tree, stats, error)) {
        outState.status = "no dialogue: " + error;
        return false;
    }

    importer::fnv::FalloutAssetSource assets;
    if (!assets.open(dataFilesPath, importer::fnv::kBsaContentMeshes)) {
        outState.status = "cannot index meshes under " + dataFilesPath.string();
        return false;
    }
    std::vector<std::uint8_t> nifBytes;
    if (!assets.resolveMesh(kSecuritronModel, nifBytes, error)) {
        outState.status = "mesh not found: " + error;
        return false;
    }
    importer::fnv::NifModel model;
    if (!importer::fnv::parseNifStaticMesh(nifBytes, model, error) || model.shapes.empty()) {
        outState.status = "mesh failed to parse: " + (error.empty() ? "no shapes" : error);
        return false;
    }

    // One mesh, one instance, placed at his reference. The vertices arrive in
    // raw Bethesda model space (Z-up), so the instance transform carries the
    // same basis change the cell path applies -- rows (R0, R2, -R1) of identity
    // -- rather than the vertices being rotated here.
    odai::importer::ImportedSceneMesh mesh;
    mesh.name = "victor";
    for (const importer::fnv::NifShape& shape : model.shapes) {
        const std::size_t vertexCount = shape.positions.size() / 3u;
        if (vertexCount == 0u || shape.triangleIndices.empty()) {
            continue;
        }
        const auto baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
        const auto firstIndex = static_cast<std::uint32_t>(mesh.indices.size());
        for (std::size_t i = 0; i < vertexCount; ++i) {
            odai::importer::ImportedSceneVertex vertex{};
            for (int axis = 0; axis < 3; ++axis) {
                vertex.position[axis] = shape.positions[(i * 3u) + static_cast<std::size_t>(axis)];
            }
            if (shape.normals.size() >= (i * 3u) + 3u) {
                for (int axis = 0; axis < 3; ++axis) {
                    vertex.normal[axis] = shape.normals[(i * 3u) + static_cast<std::size_t>(axis)];
                }
            } else {
                vertex.normal[2] = 1.0f;
            }
            if (shape.uvs.size() >= (i * 2u) + 2u) {
                vertex.uv[0] = shape.uvs[i * 2u];
                vertex.uv[1] = shape.uvs[(i * 2u) + 1u];
            }
            mesh.vertices.push_back(vertex);
        }
        for (const std::uint32_t index : shape.triangleIndices) {
            // Shape-local and never validated by the parser against the vertex
            // count; a whole triangle is dropped rather than one index, which
            // would re-tuple every triangle after it.
            if (index >= vertexCount) {
                continue;
            }
            mesh.indices.push_back(baseVertex + index);
        }
        while (((mesh.indices.size() - firstIndex) % 3u) != 0u) {
            mesh.indices.pop_back();
        }
        odai::importer::ImportedSceneMeshPart part{};
        part.firstIndex = firstIndex;
        part.indexCount = static_cast<std::uint32_t>(mesh.indices.size()) - firstIndex;
        part.textureIndex = 0xffffffffu;  // vertex colour; no texture table here
        part.alphaTest = shape.alphaTest;
        part.alphaBlend = shape.alphaBlend;
        part.twoSided = shape.twoSided;
        part.alphaThreshold = shape.alphaThreshold;
        if (part.indexCount != 0u) {
            mesh.parts.push_back(part);
        }
    }
    if (mesh.vertices.empty() || mesh.parts.empty()) {
        outState.status = "mesh parsed but produced no drawable geometry";
        return false;
    }
    outScene.sourceTag = "victor";
    outScene.meshes.push_back(std::move(mesh));

    odai::importer::ImportedSceneInstance instance{};
    instance.meshIndex = 0;
    instance.modelPath = kSecuritronModel;
    instance.sourceId = "victor";
    // M = [1 0 0; 0 0 1; 0 -1 0] applied to identity, translation in engine space.
    instance.transform[0] = 1.0f;
    instance.transform[6] = 1.0f;
    instance.transform[9] = -1.0f;
    instance.transform[3] = outState.position[0];
    instance.transform[7] = outState.position[1];
    instance.transform[11] = outState.position[2];
    instance.transform[15] = 1.0f;
    outScene.instances.push_back(instance);
    // The NIF stated every shape's alpha mode, so the texture-content cutout
    // guess must not run over this scene.
    outScene.alphaFlagsAuthored = true;
    odai::importer::buildImportedScenePackedRenderData(outScene);

    outState.placed = true;
    outState.status = "placed at (" + std::to_string(outState.position[0]) + ", " +
                      std::to_string(outState.position[1]) + ", " +
                      std::to_string(outState.position[2]) + ") with " +
                      std::to_string(outState.tree.nodes.size()) + " dialogue nodes";
    return true;
}

bool victorIsInReach(
    const VictorState& state, const float cameraPosition[3], float cameraYawRadians
) {
    if (!state.placed) {
        return false;
    }
    const float dx = state.position[0] - cameraPosition[0];
    const float dy = state.position[1] - cameraPosition[1];
    const float dz = state.position[2] - cameraPosition[2];
    const float distanceSquared = (dx * dx) + (dy * dy) + (dz * dz);
    if (distanceSquared > kTalkRange * kTalkRange) {
        return false;
    }
    const float horizontal = std::sqrt((dx * dx) + (dz * dz));
    if (horizontal < 1e-3f) {
        return true;  // standing on top of him counts as facing him
    }
    // Same basis the camera uses: forward is (cos(yaw), sin(yaw)) in XZ.
    const float forwardX = std::cos(cameraYawRadians);
    const float forwardZ = std::sin(cameraYawRadians);
    return (((dx / horizontal) * forwardX) + ((dz / horizontal) * forwardZ)) >= kTalkFacingDot;
}

void speakVictorLine(
    VictorState& state,
    const std::filesystem::path& dataFilesPath,
    const std::filesystem::path& cacheDirectory,
    odai::audio::Audio& audioSystem
) {
    // NOT WIRED. Everything except one API is in place, and the missing piece is
    // specific: the voice file for a line is named
    // <questEdid>_<topicEdid>_<infoFormId>_1.ogg, and only the formID half is
    // known here (it is the dialogue node's id). Resolving the rest means
    // ENUMERATING the archive for a name containing that formID, and
    // FalloutAssetSource/BsaArchive expose only find() on an exact path -- the
    // entry list is private with no iterator.
    //
    // So this needs a small enumeration API on BsaArchive first. The rest is
    // already settled and measured: the files live in
    // sound\voice\falloutnv.esm\robotvictor\, inside Fallout - Voices1.bsa,
    // which needs kBsaContentVoices because the default mask indexes meshes and
    // textures only; and decodeOggToWav (newvegas_ogg.h) already exists because
    // miniaudio has no Vorbis decoder.
    //
    // Left as a no-op rather than a guess: a wrong path here would fail silently
    // every line, which is worse than a feature that is visibly absent.
    (void)state;
    (void)dataFilesPath;
    (void)cacheDirectory;
    (void)audioSystem;
}

}  // namespace odai::games::newvegas
