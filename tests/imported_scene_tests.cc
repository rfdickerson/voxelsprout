#include <cmath>
#include <filesystem>
#include <iostream>

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

    ImportedScene runtimeLoaded{};
    expectTrue(odai::importer::loadImportedSceneRuntime(scenePath, runtimeLoaded), "Imported scene runtime loader works");
    expectTrue(runtimeLoaded.sourceMeshCount == 1u, "Imported scene runtime loader keeps mesh count metadata");
    expectTrue(runtimeLoaded.sourceInstanceCount == 1u, "Imported scene runtime loader keeps instance count metadata");
    expectTrue(runtimeLoaded.meshes.empty(), "Imported scene runtime loader skips full meshes");
    expectTrue(runtimeLoaded.unresolvedRefs.empty(), "Imported scene runtime loader skips unresolved refs");
    expectTrue(runtimeLoaded.instances.empty(), "Imported scene runtime loader skips instance transforms");
    expectTrue(runtimeLoaded.landscapeCells.empty(), "Imported scene runtime loader skips landscape cells");
    expectTrue(runtimeLoaded.waterPatches.size() == 1u, "Imported scene runtime loader keeps water patches");
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
        ImportedSceneMeshPart{3u, 3u, 0u, true}
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
        "GPU scene render cache keeps alpha-test part flags isolated");

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

}  // namespace

int main() {
    testImportedSceneSerialization();
    testGpuSceneBuildFromImportedScene();

    if (g_failures != 0) {
        std::cerr << "[imported scene test] " << g_failures << " failures\n";
        return 1;
    }

    std::cout << "[imported scene test] all checks passed\n";
    return 0;
}
