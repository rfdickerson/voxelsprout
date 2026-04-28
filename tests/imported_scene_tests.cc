#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "import/gpu_scene.h"
#include "import/imported_scene.h"
#include "import/morrowind_nif.h"
#include "world/imported_scene_collision.h"

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[imported scene test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectNear(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) > epsilon) {
        std::cerr << "[imported scene test] FAIL: " << message
                  << " (expected " << expected
                  << ", got " << actual << ")\n";
        ++g_failures;
    }
}

template <typename T>
void appendBinaryValue(std::vector<std::uint8_t>& bytes, const T& value) {
    const auto* raw = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(T));
}

void appendSizedString(std::vector<std::uint8_t>& bytes, const std::string& value) {
    appendBinaryValue(bytes, static_cast<std::uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendBool(std::vector<std::uint8_t>& bytes, bool value) {
    appendBinaryValue(bytes, static_cast<std::int32_t>(value ? 1 : 0));
}

void appendRefList(std::vector<std::uint8_t>& bytes, std::initializer_list<std::int32_t> refs) {
    appendBinaryValue(bytes, static_cast<std::uint32_t>(refs.size()));
    for (const std::int32_t ref : refs) {
        appendBinaryValue(bytes, ref);
    }
}

void appendIdentityAvObject(std::vector<std::uint8_t>& bytes, const std::string& name) {
    appendSizedString(bytes, name);
    appendBinaryValue(bytes, static_cast<std::int32_t>(-1));
    appendBinaryValue(bytes, static_cast<std::int32_t>(-1));
    appendBinaryValue(bytes, static_cast<std::uint16_t>(0));

    const float translation[3] = {0.0f, 0.0f, 0.0f};
    for (const float value : translation) {
        appendBinaryValue(bytes, value);
    }
    const float rotation[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    for (const float value : rotation) {
        appendBinaryValue(bytes, value);
    }
    appendBinaryValue(bytes, 1.0f);
    const float center[3] = {0.0f, 0.0f, 0.0f};
    for (const float value : center) {
        appendBinaryValue(bytes, value);
    }
    appendRefList(bytes, {});
    appendBool(bytes, false);
}

void appendNifNode(
    std::vector<std::uint8_t>& bytes,
    const std::string& name,
    std::initializer_list<std::int32_t> children
) {
    appendSizedString(bytes, "NiNode");
    appendIdentityAvObject(bytes, name);
    appendRefList(bytes, children);
    appendRefList(bytes, {});
}

void appendNifTriShape(
    std::vector<std::uint8_t>& bytes,
    const std::string& name,
    std::int32_t dataRef
) {
    appendSizedString(bytes, "NiTriShape");
    appendIdentityAvObject(bytes, name);
    appendBinaryValue(bytes, dataRef);
    appendBinaryValue(bytes, static_cast<std::int32_t>(-1));
}

void appendNifTriShapeData(
    std::vector<std::uint8_t>& bytes,
    const float (&positions)[9],
    const std::uint16_t (&indices)[3]
) {
    appendSizedString(bytes, "NiTriShapeData");
    appendBinaryValue(bytes, static_cast<std::uint16_t>(3));
    appendBool(bytes, true);
    for (const float value : positions) {
        appendBinaryValue(bytes, value);
    }
    appendBool(bytes, true);
    const float normals[9] = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f
    };
    for (const float value : normals) {
        appendBinaryValue(bytes, value);
    }
    const float bounds[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (const float value : bounds) {
        appendBinaryValue(bytes, value);
    }
    appendBool(bytes, false);
    appendBinaryValue(bytes, static_cast<std::uint16_t>(1));
    appendBool(bytes, true);
    const float uvs[6] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    for (const float value : uvs) {
        appendBinaryValue(bytes, value);
    }
    appendBinaryValue(bytes, static_cast<std::uint16_t>(1));
    appendBinaryValue(bytes, static_cast<std::uint32_t>(3));
    for (const std::uint16_t index : indices) {
        appendBinaryValue(bytes, index);
    }
    appendBinaryValue(bytes, static_cast<std::uint16_t>(0));
}

std::filesystem::path writeSyntheticRootCollisionNif() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai_named_root_collision_node.nif";
    std::vector<std::uint8_t> bytes;
    const std::string header = "NetImmerse File Format, Version 4.0.0.2\n";
    bytes.insert(bytes.end(), header.begin(), header.end());
    appendBinaryValue(bytes, static_cast<std::uint32_t>(0x04000002u));
    appendBinaryValue(bytes, static_cast<std::uint32_t>(6));

    appendNifNode(bytes, "Root", {1, 2});
    appendNifTriShape(bytes, "visible_triangle", 3);
    appendNifNode(bytes, "RootCollisionNode", {4});
    const float visiblePositions[9] = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f
    };
    const std::uint16_t triangleIndices[3] = {0, 1, 2};
    appendNifTriShapeData(bytes, visiblePositions, triangleIndices);
    appendNifTriShape(bytes, "collision_triangle", 5);
    const float collisionPositions[9] = {
        10.0f, 0.0f, 0.0f,
        11.0f, 0.0f, 0.0f,
        10.0f, 1.0f, 0.0f
    };
    appendNifTriShapeData(bytes, collisionPositions, triangleIndices);

    appendBinaryValue(bytes, static_cast<std::uint32_t>(1));
    appendBinaryValue(bytes, static_cast<std::int32_t>(0));

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return path;
}

void testImportedSceneSerialization() {
    namespace fs = std::filesystem;
    using odai::importer::ImportedScene;
    using odai::importer::ImportedSceneCellRef;
    using odai::importer::ImportedSceneInstance;
    using odai::importer::ImportedSceneLandscapeCell;
    using odai::importer::ImportedSceneLight;
    using odai::importer::ImportedSceneMesh;
    using odai::importer::ImportedSceneMeshPart;
    using odai::importer::ImportedSceneTexture;
    using odai::importer::ImportedSceneVertex;
    using odai::importer::ImportedSceneWaterPatch;

    ImportedScene scene{};
    scene.sourceTag = "synthetic_scene";

    ImportedSceneTexture texture{};
    texture.sourcePath = "textures/terrain/test.dds";
    texture.width = 2;
    texture.height = 2;
    texture.rgba8 = {
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255,
        255, 255, 255, 255
    };
    scene.textures.push_back(texture);

    ImportedSceneMesh mesh{};
    mesh.name = "terrain";
    mesh.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    mesh.indices = {0u, 1u, 2u};
    mesh.parts = {ImportedSceneMeshPart{0u, 3u, 0u, false}};
    scene.meshes.push_back(mesh);

    ImportedSceneLandscapeCell landscape{};
    landscape.gridX = -1;
    landscape.gridY = 2;
    landscape.heights = {0.0f, 8.0f, 16.0f, 24.0f};
    landscape.textureIndices = {3u, 4u, 5u, 6u};
    scene.landscapeCells.push_back(landscape);

    ImportedSceneCellRef unresolved{};
    unresolved.refId = "flora_bittergreen_01";
    unresolved.modelPath = "f/flora_bittergreen_01.nif";
    unresolved.position[0] = 12.0f;
    unresolved.position[1] = 24.0f;
    unresolved.position[2] = 36.0f;
    unresolved.rotationRadians[1] = 1.5707963f;
    unresolved.scale = 0.8f;
    scene.unresolvedRefs.push_back(unresolved);

    ImportedSceneWaterPatch waterPatch{};
    waterPatch.originX = 64.0f;
    waterPatch.originZ = 96.0f;
    waterPatch.sizeX = 128.0f;
    waterPatch.sizeZ = 64.0f;
    waterPatch.waterLevel = 4.0f;
    scene.waterPatches.push_back(waterPatch);

    ImportedSceneLight light{};
    light.sourceId = "light_de_lantern_05_128";
    light.position[0] = 16.0f;
    light.position[1] = 72.0f;
    light.position[2] = 24.0f;
    light.color[0] = 1.0f;
    light.color[1] = 0.72f;
    light.color[2] = 0.42f;
    light.radius = 384.0f;
    light.intensity = 1.0f;
    light.flags = 0x018u;
    scene.lights.push_back(light);

    ImportedSceneInstance instance{};
    instance.meshIndex = 0u;
    instance.transform[0] = 1.0f;
    instance.transform[5] = 1.0f;
    instance.transform[10] = 1.0f;
    instance.transform[15] = 1.0f;
    instance.transform[3] = 32.0f;
    instance.transform[7] = 48.0f;
    instance.transform[11] = 64.0f;
    instance.sourceId = "ex_hlaalu_b_01";
    instance.modelPath = "x/ex_hlaalu_b_01.nif";
    scene.instances.push_back(instance);
    odai::importer::buildImportedScenePackedRenderData(scene);

    const fs::path scenePath = fs::temp_directory_path() / "odai_imported_scene_roundtrip.bin";
    const fs::path objPath = fs::temp_directory_path() / "odai_imported_scene_roundtrip.obj";

    expectTrue(odai::importer::saveImportedScene(scene, scenePath), "Imported scene saves");

    ImportedScene loaded{};
    expectTrue(odai::importer::loadImportedScene(scenePath, loaded), "Imported scene loads");
    expectTrue(loaded.sourceTag == scene.sourceTag, "Imported scene source tag round-trips");
    expectTrue(loaded.textures.size() == 1u, "Imported scene texture count round-trips");
    expectTrue(loaded.meshes.size() == 1u, "Imported scene mesh count round-trips");
    expectTrue(loaded.meshes.front().vertices.size() == 3u, "Imported scene vertex count round-trips");
    expectTrue(loaded.meshes.front().indices == mesh.indices, "Imported scene indices round-trip");
    expectTrue(loaded.instances.size() == 1u, "Imported scene instance count round-trips");
    expectNear(loaded.instances.front().transform[11], instance.transform[11], 1e-6f, "Imported scene instance transform round-trips");
    expectTrue(!loaded.packedVertices.empty(), "Imported scene packed vertices round-trip");
    expectTrue(!loaded.packedIndices.empty(), "Imported scene packed indices round-trip");
    expectTrue(!loaded.packedDraws.empty(), "Imported scene packed draws round-trip");
    expectTrue(loaded.landscapeCells.size() == 1u, "Imported scene landscape cell count round-trips");
    expectTrue(loaded.landscapeCells.front().gridX == -1 && loaded.landscapeCells.front().gridY == 2, "Imported scene landscape cell coords round-trip");
    expectTrue(loaded.unresolvedRefs.size() == 1u, "Imported scene unresolved ref count round-trips");
    expectNear(loaded.unresolvedRefs.front().rotationRadians[1], unresolved.rotationRadians[1], 1e-6f, "Imported scene unresolved ref rotation round-trips");
    expectNear(loaded.unresolvedRefs.front().scale, unresolved.scale, 1e-6f, "Imported scene unresolved ref scale round-trips");
    expectTrue(loaded.waterPatches.size() == 1u, "Imported scene water patch count round-trips");
    expectNear(loaded.waterPatches.front().waterLevel, waterPatch.waterLevel, 1e-6f, "Imported scene water patch level round-trips");
    expectTrue(loaded.lights.size() == 1u, "Imported scene light count round-trips");
    expectTrue(loaded.lights.front().sourceId == light.sourceId, "Imported scene light id round-trips");
    expectNear(loaded.lights.front().position[1], light.position[1], 1e-6f, "Imported scene light position round-trips");
    expectNear(loaded.lights.front().color[1], light.color[1], 1e-6f, "Imported scene light color round-trips");
    expectNear(loaded.lights.front().radius, light.radius, 1e-6f, "Imported scene light radius round-trips");

    ImportedScene runtimeLoaded{};
    expectTrue(odai::importer::loadImportedSceneRuntime(scenePath, runtimeLoaded), "Imported scene runtime loader works");
    expectTrue(runtimeLoaded.sourceMeshCount == 1u, "Imported scene runtime loader keeps mesh count metadata");
    expectTrue(runtimeLoaded.sourceInstanceCount == 1u, "Imported scene runtime loader keeps instance count metadata");
    expectTrue(runtimeLoaded.meshes.empty(), "Imported scene runtime loader skips full meshes");
    expectTrue(runtimeLoaded.unresolvedRefs.empty(), "Imported scene runtime loader skips unresolved refs");
    expectTrue(runtimeLoaded.instances.empty(), "Imported scene runtime loader skips instance transforms");
    expectTrue(runtimeLoaded.landscapeCells.empty(), "Imported scene runtime loader skips landscape cells");
    expectTrue(runtimeLoaded.waterPatches.size() == 1u, "Imported scene runtime loader keeps water patches");
    expectTrue(runtimeLoaded.lights.size() == 1u, "Imported scene runtime loader keeps lights");
    expectTrue(!runtimeLoaded.packedVertices.empty(), "Imported scene runtime loader reads packed vertices");
    expectTrue(!runtimeLoaded.packedIndices.empty(), "Imported scene runtime loader reads packed indices");
    expectTrue(!runtimeLoaded.packedDraws.empty(), "Imported scene runtime loader reads packed draws");
    expectTrue(runtimeLoaded.boundsMax[0] >= runtimeLoaded.boundsMin[0], "Imported scene runtime loader reads bounds");

    expectTrue(odai::importer::exportImportedSceneTerrainObj(loaded, objPath), "Imported scene OBJ export succeeds");
    expectTrue(fs::exists(objPath), "Imported scene OBJ export writes a file");
    expectTrue(fs::file_size(objPath) > 0u, "Imported scene OBJ export file is non-empty");

    fs::remove(scenePath);
    fs::remove(objPath);
}

void testGpuSceneBuildFromImportedScene() {
    using odai::importer::GpuSceneAsset;
    using odai::importer::GpuSceneObjectView;
    using odai::importer::GpuSceneRuntime;
    using odai::importer::ImportedScene;
    using odai::importer::ImportedSceneInstance;
    using odai::importer::ImportedSceneLight;
    using odai::importer::ImportedSceneMesh;
    using odai::importer::ImportedSceneMeshPart;
    using odai::importer::ImportedSceneTexture;
    using odai::importer::ImportedSceneVertex;
    using odai::importer::ImportedSceneWaterPatch;

    ImportedScene scene{};
    scene.sourceTag = "gpu_scene_synthetic";

    ImportedSceneTexture texture{};
    texture.sourcePath = "textures/test/wall.dds";
    texture.width = 1;
    texture.height = 1;
    texture.rgba8 = {255, 255, 255, 255};
    scene.textures.push_back(texture);
    ImportedSceneTexture leafTexture{};
    leafTexture.sourcePath = "textures/test/leaves.dds";
    leafTexture.width = 2;
    leafTexture.height = 1;
    leafTexture.rgba8 = {16, 48, 16, 255, 16, 48, 16, 0};
    scene.textures.push_back(leafTexture);

    ImportedSceneMesh terrain{};
    terrain.name = "terrain";
    terrain.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{4.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 0.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    terrain.indices = {0u, 1u, 2u};
    terrain.parts = {ImportedSceneMeshPart{0u, 3u, 0u, false}};
    scene.meshes.push_back(terrain);

    ImportedSceneMesh wall{};
    wall.name = "wall";
    wall.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 3.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{3.0f, 3.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        ImportedSceneVertex{{3.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}
    };
    wall.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    wall.parts = {
        ImportedSceneMeshPart{0u, 3u, 0u, false},
        ImportedSceneMeshPart{3u, 3u, 1u, false}
    };
    scene.meshes.push_back(wall);

    ImportedSceneInstance instance{};
    instance.meshIndex = 1u;
    instance.transform[0] = 1.0f;
    instance.transform[5] = 1.0f;
    instance.transform[10] = 1.0f;
    instance.transform[15] = 1.0f;
    instance.transform[3] = 32.0f;
    instance.transform[7] = 8.0f;
    instance.transform[11] = 48.0f;
    instance.sourceId = "ex_hlaalu_wall_01";
    instance.modelPath = "x/ex_hlaalu_wall_01.nif";
    scene.instances.push_back(instance);

    ImportedSceneWaterPatch patch{};
    patch.originX = 8.0f;
    patch.originZ = 12.0f;
    patch.sizeX = 16.0f;
    patch.sizeZ = 16.0f;
    patch.waterLevel = 2.0f;
    scene.waterPatches.push_back(patch);

    ImportedSceneLight light{};
    light.sourceId = "light_de_lantern_05_128";
    light.position[1] = 72.0f;
    light.radius = 256.0f;
    scene.lights.push_back(light);

    GpuSceneAsset gpuScene{};
    expectTrue(
        odai::importer::buildGpuSceneAssetFromImportedScene(scene, gpuScene),
        "GPU scene asset builds from imported scene");
    expectTrue(gpuScene.meshAssets.size() == 2u, "GPU scene preserves mesh asset count");
    expectTrue(gpuScene.objects.rootTransformIndices.size() == 2u, "GPU scene creates terrain and static objects");
    expectTrue(gpuScene.instances.objectIndices.size() == 2u, "GPU scene creates terrain and static instances");
    expectTrue(!gpuScene.pages.empty(), "GPU scene partitions objects into at least one page");
    expectTrue(!gpuScene.renderCache.packedVertices.empty(), "GPU scene render cache packs vertices");
    expectTrue(!gpuScene.renderCache.packedIndices.empty(), "GPU scene render cache packs indices");
    expectTrue(!gpuScene.renderCache.packedDraws.empty(), "GPU scene render cache packs draws");
    expectTrue(gpuScene.renderCache.terrainDrawCount == 1u, "GPU scene marks terrain draw count");
    expectTrue(gpuScene.renderCache.drawInstanceIndices.size() == gpuScene.renderCache.packedDraws.size(),
               "GPU scene draw-instance mapping matches draw count");
    expectTrue(gpuScene.renderCache.packedDraws.size() == 3u, "GPU scene preserves mesh parts as separate draws");
    expectTrue(gpuScene.lights.size() == 1u, "GPU scene keeps imported lights");
    expectTrue(gpuScene.renderCache.lights.size() == 1u, "GPU scene render cache keeps imported lights");
    expectTrue(!gpuScene.renderCache.pageDrawRanges.empty(), "GPU scene render cache records page draw ranges");
    expectTrue(gpuScene.renderCache.pageDrawRanges.front().firstDraw == 0u,
               "GPU scene page draw ranges start at the first draw");
    expectTrue(gpuScene.renderCache.pageDrawRanges.front().drawCount >= 1u,
               "GPU scene page draw ranges cover at least one draw");
    expectTrue(gpuScene.renderCache.pageDrawRanges.front().terrainDrawCount == 1u,
               "GPU scene page draw ranges record terrain draws");
    const auto& opaqueWallDraw = gpuScene.renderCache.packedDraws[1];
    const auto& alphaWallDraw = gpuScene.renderCache.packedDraws[2];
    const std::uint32_t opaqueWallVertex =
        gpuScene.renderCache.packedIndices[opaqueWallDraw.firstIndex];
    const std::uint32_t alphaWallVertex =
        gpuScene.renderCache.packedIndices[alphaWallDraw.firstIndex];
    expectTrue(
        gpuScene.renderCache.packedVertices[opaqueWallVertex].flags == 0u,
        "GPU scene render cache keeps opaque part flags isolated");
    expectTrue(
        gpuScene.renderCache.packedVertices[alphaWallVertex].flags == 1u,
        "GPU scene render cache derives alpha-test flags from texture alpha");

    const GpuSceneObjectView wallView = odai::importer::gpuSceneObjectView(gpuScene, 1u);
    expectTrue(wallView.name == "ex_hlaalu_wall_01", "GPU scene object view exposes object name");
    expectTrue(wallView.componentCount == 1u, "GPU scene object view exposes component count");
    expectNear(wallView.appliedTransform[3], 32.0f, 1e-6f, "GPU scene object view exposes applied transform");

    GpuSceneRuntime runtime = odai::importer::createGpuSceneRuntime(gpuScene);
    odai::importer::rebuildGpuSceneWorldTransforms(runtime);
    expectTrue(runtime.transforms.worldMatrices.size() == gpuScene.transforms.worldMatrices.size(),
               "GPU scene runtime keeps transform count");
    expectNear(runtime.transforms.worldMatrices[1][3], 32.0f, 1e-6f, "GPU scene runtime rebuild keeps translation");
}

void testGpuSceneBuildFromInteriorSceneDoesNotCreateTerrain() {
    using odai::importer::GpuSceneAsset;
    using odai::importer::ImportedScene;
    using odai::importer::ImportedSceneInstance;
    using odai::importer::ImportedSceneMesh;
    using odai::importer::ImportedSceneMeshPart;
    using odai::importer::ImportedSceneVertex;

    ImportedScene scene{};
    scene.sourceTag = "morrowind_interior";

    ImportedSceneMesh roomMesh{};
    roomMesh.name = "in_hlaalu_room";
    roomMesh.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{4.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 0.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    roomMesh.indices = {0u, 1u, 2u};
    roomMesh.parts = {ImportedSceneMeshPart{0u, 3u, 0u, false}};
    scene.meshes.push_back(roomMesh);

    ImportedSceneInstance roomInstance{};
    roomInstance.meshIndex = 0u;
    roomInstance.transform[0] = 1.0f;
    roomInstance.transform[5] = 1.0f;
    roomInstance.transform[10] = 1.0f;
    roomInstance.transform[15] = 1.0f;
    roomInstance.transform[3] = 12.0f;
    roomInstance.transform[7] = 3.0f;
    roomInstance.transform[11] = 5.0f;
    roomInstance.sourceId = "in_hlaalu_room_ref";
    scene.instances.push_back(roomInstance);

    GpuSceneAsset gpuScene{};
    expectTrue(
        odai::importer::buildGpuSceneAssetFromImportedScene(scene, gpuScene),
        "GPU scene asset builds from interior scene");
    expectTrue(gpuScene.objects.rootTransformIndices.size() == 1u,
               "Interior GPU scene does not synthesize terrain object");
    expectTrue(gpuScene.instances.objectIndices.size() == 1u,
               "Interior GPU scene only creates placed static instances");
    expectTrue(gpuScene.renderCache.terrainDrawCount == 0u,
               "Interior GPU scene does not mark any terrain draws");
    expectTrue(gpuScene.renderCache.packedDraws.size() == 1u,
               "Interior GPU scene keeps the placed room draw");

    const odai::importer::ImportedScenePackedDraw& draw = gpuScene.renderCache.packedDraws.front();
    const std::uint32_t packedVertexIndex = gpuScene.renderCache.packedIndices[draw.firstIndex];
    expectNear(
        gpuScene.renderCache.packedVertices[packedVertexIndex].position[0],
        12.0f,
        1e-6f,
        "Interior GPU scene transforms placed mesh vertices");
    expectNear(
        gpuScene.renderCache.packedVertices[packedVertexIndex].position[1],
        3.0f,
        1e-6f,
        "Interior GPU scene keeps placed mesh height");

    odai::importer::buildImportedScenePackedRenderData(scene);
    expectTrue(scene.packedDraws.size() == 1u,
               "Interior packed scene does not synthesize terrain draw");
    const std::uint32_t runtimePackedVertexIndex = scene.packedIndices[scene.packedDraws.front().firstIndex];
    expectNear(
        scene.packedVertices[runtimePackedVertexIndex].position[0],
        12.0f,
        1e-6f,
        "Interior packed scene transforms placed mesh vertices");
    expectNear(
        scene.packedVertices[runtimePackedVertexIndex].position[1],
        3.0f,
        1e-6f,
        "Interior packed scene keeps placed mesh height");
}

void testMorrowindNifSkipsNamedRootCollisionNode() {
    const std::filesystem::path nifPath = writeSyntheticRootCollisionNif();

    odai::importer::ImportedNifResult result{};
    std::string error;
    expectTrue(
        odai::importer::loadMorrowindStaticNif(nifPath, result, error),
        "Morrowind NIF loader imports synthetic fixture");
    expectTrue(result.mesh.vertices.size() == 3u,
               "Morrowind NIF loader skips NiNode named RootCollisionNode vertices");
    expectTrue(result.mesh.indices.size() == 3u,
               "Morrowind NIF loader skips NiNode named RootCollisionNode indices");
    expectTrue(result.mesh.parts.size() == 1u,
               "Morrowind NIF loader skips NiNode named RootCollisionNode parts");
    for (const odai::importer::ImportedSceneVertex& vertex : result.mesh.vertices) {
        expectTrue(vertex.position[0] < 10.0f,
                   "Morrowind NIF loader excludes collision-only triangle positions");
    }

    std::filesystem::remove(nifPath);
}

void testImportedSceneCollision() {
    using odai::importer::GpuSceneAsset;
    using odai::importer::ImportedScene;
    using odai::importer::ImportedSceneInstance;
    using odai::importer::ImportedSceneMesh;
    using odai::importer::ImportedSceneMeshPart;
    using odai::importer::ImportedSceneVertex;
    using odai::world::ImportedSceneCollision;

    ImportedScene scene{};
    scene.sourceTag = "collision_synthetic";

    ImportedSceneMesh terrain{};
    terrain.name = "terrain";
    terrain.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{10.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
        ImportedSceneVertex{{0.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    terrain.indices = {0u, 2u, 1u, 0u, 3u, 2u};
    terrain.parts = {ImportedSceneMeshPart{0u, 6u, 0u, false}};
    scene.meshes.push_back(terrain);

    ImportedSceneMesh raisedFloor{};
    raisedFloor.name = "raised_floor";
    raisedFloor.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{4.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{4.0f, 0.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
        ImportedSceneVertex{{0.0f, 0.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    raisedFloor.indices = {0u, 2u, 1u, 0u, 3u, 2u};
    raisedFloor.parts = {ImportedSceneMeshPart{0u, 6u, 0u, false}};
    scene.meshes.push_back(raisedFloor);

    ImportedSceneMesh wall{};
    wall.name = "wall";
    wall.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{4.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{4.0f, 6.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        ImportedSceneVertex{{0.0f, 6.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}
    };
    wall.indices = {0u, 1u, 2u, 0u, 2u, 3u};
    wall.parts = {ImportedSceneMeshPart{0u, 6u, 0u, false}};
    scene.meshes.push_back(wall);

    ImportedSceneMesh narrowTread{};
    narrowTread.name = "narrow_tread";
    narrowTread.vertices = {
        ImportedSceneVertex{{0.0f, 2.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{0.4f, 2.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.4f, 2.0f, 0.4f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
        ImportedSceneVertex{{0.0f, 2.0f, 0.4f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    narrowTread.indices = {0u, 2u, 1u, 0u, 3u, 2u};
    narrowTread.parts = {ImportedSceneMeshPart{0u, 6u, 0u, false}};
    scene.meshes.push_back(narrowTread);

    ImportedSceneInstance floorInstance{};
    floorInstance.meshIndex = 1u;
    floorInstance.transform[0] = 1.0f;
    floorInstance.transform[5] = 1.0f;
    floorInstance.transform[10] = 1.0f;
    floorInstance.transform[15] = 1.0f;
    floorInstance.transform[3] = 20.0f;
    floorInstance.transform[7] = 8.0f;
    scene.instances.push_back(floorInstance);

    ImportedSceneInstance wallInstance{};
    wallInstance.meshIndex = 2u;
    wallInstance.transform[0] = 1.0f;
    wallInstance.transform[5] = 1.0f;
    wallInstance.transform[10] = 1.0f;
    wallInstance.transform[15] = 1.0f;
    wallInstance.transform[3] = 40.0f;
    scene.instances.push_back(wallInstance);

    ImportedSceneInstance narrowTreadInstance{};
    narrowTreadInstance.meshIndex = 3u;
    narrowTreadInstance.transform[0] = 1.0f;
    narrowTreadInstance.transform[5] = 1.0f;
    narrowTreadInstance.transform[10] = 1.0f;
    narrowTreadInstance.transform[15] = 1.0f;
    narrowTreadInstance.transform[3] = 61.6f;
    narrowTreadInstance.transform[11] = 0.1f;
    scene.instances.push_back(narrowTreadInstance);

    GpuSceneAsset gpuScene{};
    expectTrue(
        odai::importer::buildGpuSceneAssetFromImportedScene(scene, gpuScene),
        "GPU scene asset builds for collision test");

    ImportedSceneCollision collision{};
    expectTrue(collision.build(gpuScene), "Imported scene collision builds");
    const ImportedSceneCollision::BuildStats stats = collision.stats();
    expectTrue(stats.triangleCount == 8u, "Imported scene collision keeps terrain and static triangles");

    ImportedSceneCollision::GroundHit groundHit{};
    expectTrue(
        collision.findGroundSupport(2.0f, 5.0f, 2.0f, 0.5f, 10.0f, 1.0f, 0.65f, groundHit),
        "Imported scene collision finds terrain support");
    expectNear(groundHit.y, 0.0f, 1e-5f, "Imported scene terrain support height is correct");

    expectTrue(
        collision.findGroundSupport(22.0f, 12.0f, 2.0f, 0.5f, 10.0f, 1.0f, 0.65f, groundHit),
        "Imported scene collision finds transformed static floor support");
    expectNear(groundHit.y, 8.0f, 1e-5f, "Imported scene transformed floor support height is correct");

    expectTrue(
        !collision.findGroundSupport(30.0f, 12.0f, 2.0f, 0.5f, 10.0f, 1.0f, 0.65f, groundHit),
        "Imported scene collision rejects support outside triangles");

    expectTrue(
        collision.findGroundSupport(60.0f, 0.0f, 0.0f, 2.0f, 0.5f, 3.0f, 0.65f, groundHit),
        "Imported scene collision finds narrow stair treads inside the player footprint");
    expectNear(groundHit.y, 2.0f, 1e-5f, "Imported scene narrow stair support height is correct");

    ImportedSceneCollision::CeilingHit ceilingHit{};
    expectTrue(
        collision.findCeiling(22.0f, 8.5f, 2.0f, 0.5f, 1.0f, ceilingHit),
        "Imported scene collision finds a penetrated ceiling plane");
    expectNear(ceilingHit.y, 8.0f, 1e-5f, "Imported scene ceiling height is correct");

    odai::math::Vector3 correction{};
    expectTrue(
        collision.resolveHorizontalCylinder(42.0f, 1.0f, 0.25f, 1.0f, 4.0f, 0.65f, correction),
        "Imported scene collision resolves a wall overlap");
    expectTrue(correction.z > 0.0f, "Imported scene wall correction pushes away from wall");
}

}  // namespace

int main() {
    testImportedSceneSerialization();
    testGpuSceneBuildFromImportedScene();
    testGpuSceneBuildFromInteriorSceneDoesNotCreateTerrain();
    testMorrowindNifSkipsNamedRootCollisionNode();
    testImportedSceneCollision();

    if (g_failures != 0) {
        std::cerr << "[imported scene test] " << g_failures << " failures\n";
        return 1;
    }

    std::cout << "[imported scene test] all checks passed\n";
    return 0;
}
