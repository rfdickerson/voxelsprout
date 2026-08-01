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

void testTextureFormatRoundTrip() {
    namespace fs = std::filesystem;
    using odai::importer::ImportedScene;
    using odai::importer::ImportedSceneTexture;
    using odai::importer::TextureFormat;

    ImportedScene scene{};
    scene.sourceTag = "format_roundtrip";
    ImportedSceneTexture texture{};
    texture.sourcePath = "textures/tx_stone.dds";
    texture.width = 4;
    texture.height = 4;
    texture.mipLevelCount = 1;
    texture.format = TextureFormat::BC3;
    texture.rgba8.assign(16u, 0xffu);  // one opaque BC3 block
    scene.textures.push_back(texture);

    const fs::path scenePath = fs::temp_directory_path() / "odai_imported_scene_format.bin";
    expectTrue(odai::importer::saveImportedScene(scene, scenePath), "BC scene saves");

    ImportedScene loaded{};
    expectTrue(odai::importer::loadImportedScene(scenePath, loaded), "BC scene loads");
    expectTrue(loaded.textures.size() == 1u, "BC scene texture count round-trips");
    expectTrue(loaded.textures.front().format == TextureFormat::BC3,
               "Texture format survives the save/load round trip");

    ImportedScene runtimeLoaded{};
    expectTrue(odai::importer::loadImportedSceneRuntime(scenePath, runtimeLoaded),
               "BC scene loads via runtime loader");
    expectTrue(runtimeLoaded.textures.front().format == TextureFormat::BC3,
               "Texture format survives the runtime load path");

    fs::remove(scenePath);
}

void testBlockCompressedAlphaCutoutDetection() {
    using odai::importer::ImportedScene;
    using odai::importer::ImportedSceneMesh;
    using odai::importer::ImportedSceneMeshPart;
    using odai::importer::ImportedSceneTexture;
    using odai::importer::ImportedSceneVertex;
    using odai::importer::TextureFormat;

    // Texture 0: BC1 punch-through block (color0 <= color1, all indices 3).
    ImportedSceneTexture bc1Cutout{};
    bc1Cutout.sourcePath = "textures/tx_leaves.dds";
    bc1Cutout.width = 4;
    bc1Cutout.height = 4;
    bc1Cutout.format = TextureFormat::BC1;
    bc1Cutout.rgba8 = {0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff};

    // Texture 1: BC3 block mixing fully transparent and fully opaque texels
    // (alpha0=255 > alpha1=0, texel 0 selects alpha0, texel 1 selects alpha1).
    ImportedSceneTexture bc3Cutout{};
    bc3Cutout.sourcePath = "textures/tx_banner.dds";
    bc3Cutout.width = 4;
    bc3Cutout.height = 4;
    bc3Cutout.format = TextureFormat::BC3;
    bc3Cutout.rgba8 = {
        0xff, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,  // alpha block
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00   // color block
    };

    // Texture 2: fully opaque BC3 block (both alpha endpoints 255).
    ImportedSceneTexture bc3Opaque{};
    bc3Opaque.sourcePath = "textures/tx_wall.dds";
    bc3Opaque.width = 4;
    bc3Opaque.height = 4;
    bc3Opaque.format = TextureFormat::BC3;
    bc3Opaque.rgba8.assign(16u, 0x00);
    bc3Opaque.rgba8[0] = 0xff;
    bc3Opaque.rgba8[1] = 0xff;

    ImportedScene scene{};
    scene.sourceTag = "bc_cutout";
    scene.textures = {bc1Cutout, bc3Cutout, bc3Opaque};

    ImportedSceneMesh mesh{};
    mesh.name = "props";
    mesh.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    mesh.indices = {0u, 1u, 2u, 0u, 1u, 2u, 0u, 1u, 2u};
    mesh.parts = {
        ImportedSceneMeshPart{0u, 3u, 0u, false},
        ImportedSceneMeshPart{3u, 3u, 1u, false},
        ImportedSceneMeshPart{6u, 3u, 2u, false}
    };
    scene.meshes.push_back(mesh);

    odai::importer::ImportedSceneInstance instance{};
    instance.meshIndex = 0u;
    instance.transform[0] = 1.0f;
    instance.transform[5] = 1.0f;
    instance.transform[10] = 1.0f;
    instance.transform[15] = 1.0f;
    scene.instances.push_back(instance);

    odai::importer::buildImportedScenePackedRenderData(scene);
    expectTrue(scene.meshes.front().parts[0].alphaTest,
               "BC1 punch-through texture is detected as alpha cutout");
    expectTrue(scene.meshes.front().parts[1].alphaTest,
               "BC3 texture with transparent texels is detected as alpha cutout");
    expectTrue(!scene.meshes.front().parts[2].alphaTest,
               "Fully opaque BC3 texture is not flagged as alpha cutout");
}

void testPageRangeBuildAndRoundTrip() {
    namespace fs = std::filesystem;
    using odai::importer::ImportedScene;
    using odai::importer::ImportedScenePackedDraw;
    using odai::importer::ImportedScenePackedVertex;

    // Two Morrowind cells ~2.5 cells apart on X: terrain + one static in each.
    // Draw order deliberately interleaves the cells for the statics so the
    // builder has to reorder draws to make page members contiguous.
    ImportedScene scene{};
    scene.sourceTag = "morrowind_balmora";
    scene.sourceLandscapeCellCount = 2u;

    auto addTriangle = [&scene](float baseX) {
        const std::uint32_t baseVertex = static_cast<std::uint32_t>(scene.packedVertices.size());
        for (int corner = 0; corner < 3; ++corner) {
            ImportedScenePackedVertex vertex{};
            vertex.position[0] = baseX + static_cast<float>(corner) * 10.0f;
            vertex.position[1] = 0.0f;
            vertex.position[2] = static_cast<float>(corner % 2) * 10.0f;
            scene.packedVertices.push_back(vertex);
        }
        const std::uint32_t firstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
        scene.packedIndices.push_back(baseVertex);
        scene.packedIndices.push_back(baseVertex + 1u);
        scene.packedIndices.push_back(baseVertex + 2u);
        scene.packedDraws.push_back(ImportedScenePackedDraw{firstIndex, 3u});
    };

    addTriangle(0.0f);      // draw 0: terrain, cell A
    addTriangle(20000.0f);  // draw 1: terrain, cell B
    addTriangle(20100.0f);  // draw 2: static, cell B
    addTriangle(100.0f);    // draw 3: static, cell A

    odai::importer::buildImportedScenePageRanges(scene);

    expectTrue(scene.pageRanges.size() == 4u,
               "Page builder emits one page per (terrain/static, cell) group");
    expectTrue(scene.packedDraws.size() == 4u, "Page builder keeps every draw");
    expectTrue(scene.packedIndices.size() == 12u, "Page builder keeps every index");
    std::uint32_t coveredDraws = 0;
    std::uint32_t terrainPageDraws = 0;
    for (std::size_t pageIndex = 0; pageIndex < scene.pageRanges.size(); ++pageIndex) {
        const auto& page = scene.pageRanges[pageIndex];
        expectTrue(page.firstDraw == coveredDraws, "Pages cover contiguous draw ranges in order");
        coveredDraws += page.drawCount;
        terrainPageDraws += page.terrainDrawCount;
    }
    expectTrue(coveredDraws == 4u, "Pages cover every draw exactly once");
    expectTrue(terrainPageDraws == 2u, "Pages record both terrain draws");
    expectTrue(scene.pageRanges[0].terrainDrawCount == 1u && scene.pageRanges[1].terrainDrawCount == 1u,
               "Terrain draws stay in the leading pages");
    expectTrue(scene.pageRanges[2].terrainDrawCount == 0u && scene.pageRanges[3].terrainDrawCount == 0u,
               "Static pages carry no terrain draws");

    // After the reorder, draw 2 must be the cell-A static (x ~ 100), so both
    // cell-A draws sit in the leading half of their groups.
    const std::uint32_t staticAVertex = scene.packedIndices[scene.packedDraws[2].firstIndex];
    expectNear(scene.packedVertices[staticAVertex].position[0], 100.0f, 1e-3f,
               "Statics are reordered so same-cell draws are adjacent");
    expectTrue(scene.pageRanges[3].boundsMin[0] >= 20000.0f - 1.0f,
               "Page bounds contain only that page's geometry");
    expectTrue(scene.pageRanges[1].boundsMax[0] >= 20020.0f - 1.0f,
               "Terrain page bounds cover the cell's vertices");

    // v17 round trip: pages survive save/load on both loaders.
    const fs::path scenePath = fs::temp_directory_path() / "odai_imported_scene_pages.bin";
    expectTrue(odai::importer::saveImportedScene(scene, scenePath), "Paged scene saves");
    ImportedScene loaded{};
    expectTrue(odai::importer::loadImportedScene(scenePath, loaded), "Paged scene loads");
    expectTrue(loaded.pageRanges.size() == 4u, "Page ranges survive the save/load round trip");
    expectTrue(loaded.pageRanges[3].firstDraw == scene.pageRanges[3].firstDraw &&
               loaded.pageRanges[3].drawCount == scene.pageRanges[3].drawCount,
               "Page draw ranges round-trip exactly");
    ImportedScene runtimeLoaded{};
    expectTrue(odai::importer::loadImportedSceneRuntime(scenePath, runtimeLoaded),
               "Paged scene loads via runtime loader");
    expectTrue(runtimeLoaded.pageRanges.size() == 4u, "Page ranges survive the runtime load path");

    // A file saved without pages gets culling pages rebuilt at load time.
    ImportedScene unpaged = scene;
    unpaged.pageRanges.clear();
    expectTrue(odai::importer::saveImportedScene(unpaged, scenePath), "Unpaged scene saves");
    ImportedScene rebuilt{};
    expectTrue(odai::importer::loadImportedScene(scenePath, rebuilt), "Unpaged scene loads");
    expectTrue(!rebuilt.pageRanges.empty(), "Loader rebuilds culling pages for unpaged files");
    std::uint32_t rebuiltCovered = 0;
    for (const auto& page : rebuilt.pageRanges) {
        rebuiltCovered += page.drawCount;
    }
    expectTrue(rebuiltCovered == static_cast<std::uint32_t>(rebuilt.packedDraws.size()),
               "Rebuilt pages cover every draw");

    fs::remove(scenePath);
}

}  // namespace

int main() {
    testImportedSceneSerialization();
    testGpuSceneBuildFromImportedScene();
    testGpuSceneBuildFromInteriorSceneDoesNotCreateTerrain();
    testTextureFormatRoundTrip();
    testBlockCompressedAlphaCutoutDetection();
    testPageRangeBuildAndRoundTrip();

    if (g_failures != 0) {
        std::cerr << "[imported scene test] " << g_failures << " failures\n";
        return 1;
    }

    std::cout << "[imported scene test] all checks passed\n";
    return 0;
}
