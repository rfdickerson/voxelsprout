#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "import/dds.h"
#include "import/imported_scene.h"
#include "import/imported_scene_query.h"

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
    using odai::importer::ImportedSceneParticleEmitter;
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
    waterPatch.normalTextureIndex = 3u;
    waterPatch.flowTextureIndex = 5u;
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

    ImportedSceneParticleEmitter emitter{};
    emitter.sourceId = "refr_fire_01";
    emitter.position[0] = 20.0f;
    emitter.position[1] = 4.0f;
    emitter.position[2] = 28.0f;
    emitter.spawnRadius = 72.0f;
    emitter.particleLifetime = 1.4f;
    emitter.particleCount = 64u;
    emitter.seed = 0x12345u;
    scene.particleEmitters.push_back(emitter);

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
    instance.sourceReferenceFormId = 0x01001234u;
    instance.sourceReferenceIdentity = "frmr:morrowind.esm:0x00001234";
    instance.initiallyVisible = true;
    scene.instances.push_back(instance);

    odai::importer::ImportedSceneDoor door{};
    door.position[0] = 10.0f;
    door.arrivalPosition[2] = 40.0f;
    door.arrivalYawDegrees = 90.0f;
    door.targetCellEditorId = "WhiterunBanneredMare";
    door.sourceReferenceFormId = 0xfe001234u;
    door.targetCellFormId = 0x000165a8u;
    door.targetWorldspaceFormId = 0x0000003cu;
    door.targetWorldspaceEditorId = "WhiterunWorld";
    door.targetKind = odai::importer::ImportedSceneDoorTargetKind::Interior;
    door.locked = true;
    door.lockLevel = 50u;
    scene.doors.push_back(door);

    odai::importer::ImportedSceneCollisionTriangle collision{};
    collision.vertices[0] = 1.0f;
    collision.vertices[4] = 2.0f;
    collision.vertices[8] = 3.0f;
    collision.sourceReferenceFormId = 0xfe009876u;
    scene.collisionTriangles.push_back(collision);
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
    expectTrue(loaded.instances.front().sourceReferenceFormId == instance.sourceReferenceFormId &&
                   loaded.instances.front().sourceReferenceIdentity == instance.sourceReferenceIdentity &&
                   loaded.instances.front().initiallyVisible,
               "Imported scene v31 source identity and initial visibility round-trip");
    expectTrue(loaded.doors.size() == 1u, "Expanded door metadata round-trips");
    expectTrue(loaded.doors.front().targetKind ==
                   odai::importer::ImportedSceneDoorTargetKind::Interior,
               "Door target kind round-trips");
    expectTrue(loaded.doors.front().sourceReferenceFormId == 0xfe001234u &&
                   loaded.doors.front().locked && loaded.doors.front().lockLevel == 50u,
               "Door source and lock state round-trip");
    expectTrue(loaded.collisionTriangles.size() == 1u,
               "Authored collision triangle count round-trips");
    expectNear(loaded.collisionTriangles.front().vertices[8], 3.0f, 1e-6f,
               "Authored collision coordinates round-trip");
    expectTrue(loaded.collisionTriangles.front().sourceReferenceFormId == 0xfe009876u,
               "Authored collision reference attribution round-trips");
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
    expectTrue(loaded.waterPatches.front().normalTextureIndex == 3u &&
                   loaded.waterPatches.front().flowTextureIndex == 5u,
               "Imported scene water texture indices round-trip");
    expectTrue(loaded.lights.size() == 1u, "Imported scene light count round-trips");
    expectTrue(loaded.lights.front().sourceId == light.sourceId, "Imported scene light id round-trips");
    expectNear(loaded.lights.front().position[1], light.position[1], 1e-6f, "Imported scene light position round-trips");
    expectNear(loaded.lights.front().color[1], light.color[1], 1e-6f, "Imported scene light color round-trips");
    expectNear(loaded.lights.front().radius, light.radius, 1e-6f, "Imported scene light radius round-trips");
    expectTrue(loaded.particleEmitters.size() == 1u,
               "Imported scene particle emitter count round-trips");
    expectTrue(loaded.particleEmitters.front().sourceId == emitter.sourceId,
               "Imported scene particle emitter id round-trips");
    expectNear(loaded.particleEmitters.front().spawnRadius, emitter.spawnRadius, 1e-6f,
               "Imported scene particle emitter radius round-trips");
    expectTrue(loaded.particleEmitters.front().particleCount == emitter.particleCount,
               "Imported scene particle emitter capacity round-trips");

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
    expectTrue(runtimeLoaded.particleEmitters.size() == 1u,
               "Imported scene runtime loader keeps particle emitters");
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

void testImportedSceneSourceTagInteriorClassification() {
    using odai::importer::importedSceneSourceTagIsInterior;
    expectTrue(importedSceneSourceTagIsInterior("morrowind_interior"),
               "Morrowind interior tag is classified as interior");
    expectTrue(importedSceneSourceTagIsInterior("fnv_interior"),
               "Fallout: New Vegas interior tag is classified as interior");
    expectTrue(!importedSceneSourceTagIsInterior("morrowind_balmora"),
               "Exterior scene tags are not classified as interior");
    expectTrue(!importedSceneSourceTagIsInterior(""),
               "Empty source tag is not classified as interior");
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

void testLegacyArgbWaterFlowDds() {
    using odai::importer::ImportedSceneTexture;
    using odai::importer::TextureFormat;

    // Minimal 1x1 legacy DDS matching Skyrim's per-cell flow maps: 32-bit
    // ARGB masks, stored as BGRA bytes. No filesystem or retail data required.
    std::vector<std::uint8_t> dds(132u, 0u);
    const auto putU32 = [&](std::size_t offset, std::uint32_t value) {
        std::memcpy(dds.data() + offset, &value, sizeof(value));
    };
    putU32(0u, 0x20534444u);   // "DDS "
    putU32(4u, 124u);
    putU32(8u, 0x0000100fu);
    putU32(12u, 1u);           // height
    putU32(16u, 1u);           // width
    putU32(20u, 4u);           // pitch
    putU32(28u, 1u);           // mip count
    putU32(76u, 32u);          // pixel format size
    putU32(80u, 0x41u);        // RGB | alpha pixels
    putU32(88u, 32u);
    putU32(92u, 0x00ff0000u);
    putU32(96u, 0x0000ff00u);
    putU32(100u, 0x000000ffu);
    putU32(104u, 0xff000000u);
    putU32(108u, 0x1000u);     // texture caps
    dds[128u] = 11u;           // B
    dds[129u] = 22u;           // G
    dds[130u] = 33u;           // R
    dds[131u] = 44u;           // A

    ImportedSceneTexture texture{};
    expectTrue(odai::importer::loadDdsFromMemory(dds.data(), dds.size(), texture),
               "Skyrim legacy ARGB water-flow DDS decodes");
    expectTrue(texture.width == 1u && texture.height == 1u &&
                   texture.format == TextureFormat::RGBA8,
               "Legacy water-flow DDS keeps dimensions and becomes RGBA8");
    expectTrue(texture.rgba8 == std::vector<std::uint8_t>({33u, 22u, 11u, 44u}),
               "Legacy water-flow DDS swizzles BGRA storage to RGBA sampling");

    // Generated object LOD atlases use RGBA bytes and ABGR masks instead.
    // Reuse the same header and pixel so the only changing variable is the
    // channel declaration.
    putU32(92u, 0x000000ffu);
    putU32(96u, 0x0000ff00u);
    putU32(100u, 0x00ff0000u);
    dds[128u] = 33u;  // R
    dds[129u] = 22u;  // G
    dds[130u] = 11u;  // B
    dds[131u] = 44u;  // A
    texture = {};
    expectTrue(odai::importer::loadDdsFromMemory(dds.data(), dds.size(), texture),
               "Skyrim generated-object RGBA atlas DDS decodes");
    expectTrue(texture.format == TextureFormat::RGBA8Srgb,
               "Generated-object colour atlas is marked for sRGB sampling");
    expectTrue(texture.rgba8 == std::vector<std::uint8_t>({33u, 22u, 11u, 44u}),
               "Generated-object atlas keeps RGBA storage in RGBA order");

    // Generated terrain LOD uses opaque BGRX: the same BGR masks as the flow
    // map, but no alpha mask. The fourth stored byte is padding and must not
    // make the terrain transparent.
    putU32(92u, 0x00ff0000u);
    putU32(96u, 0x0000ff00u);
    putU32(100u, 0x000000ffu);
    putU32(104u, 0u);
    dds[128u] = 11u;  // B
    dds[129u] = 22u;  // G
    dds[130u] = 33u;  // R
    dds[131u] = 0u;   // X
    texture = {};
    expectTrue(odai::importer::loadDdsFromMemory(dds.data(), dds.size(), texture),
               "Skyrim generated-terrain BGRX DDS decodes");
    expectTrue(texture.format == TextureFormat::RGBA8Srgb,
               "Generated-terrain colour atlas is marked for sRGB sampling");
    expectTrue(texture.rgba8 == std::vector<std::uint8_t>({33u, 22u, 11u, 255u}),
               "Generated-terrain DDS swizzles BGR and synthesizes opaque alpha");
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

    ImportedScene authoredScene = scene;
    authoredScene.sourceTag = "authored_bc_cutout";
    authoredScene.alphaFlagsAuthored = true;

    odai::importer::buildImportedScenePackedRenderData(scene);
    expectTrue(scene.meshes.front().parts[0].alphaTest,
               "BC1 punch-through texture is detected as alpha cutout");
    expectTrue(scene.meshes.front().parts[1].alphaTest,
               "BC3 texture with transparent texels is detected as alpha cutout");
    expectTrue(!scene.meshes.front().parts[2].alphaTest,
               "Fully opaque BC3 texture is not flagged as alpha cutout");

    odai::importer::buildImportedScenePackedRenderData(authoredScene);
    expectTrue(
        !authoredScene.meshes.front().parts[0].alphaTest &&
            !authoredScene.meshes.front().parts[1].alphaTest &&
            !authoredScene.meshes.front().parts[2].alphaTest,
        "Authored opaque parts are not overridden by texture alpha inference");
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

// The v18 material library, through both loaders, plus the back-compatibility
// contract that makes the version bump safe: an older file must load to an
// empty table with its vertex flags untouched.
void testMaterialLibraryRoundTrip() {
    namespace fs = std::filesystem;
    using namespace odai::importer;

    ImportedScene scene;
    scene.sourceTag = "materials";
    scene.packedVertices.resize(3);
    for (std::size_t i = 0; i < scene.packedVertices.size(); ++i) {
        scene.packedVertices[i].position[0] = static_cast<float>(i);
        // Every vertex references library slot 2 and keeps fallback coefficients.
        scene.packedVertices[i].flags =
            packImportedSceneMaterialFlags(ImportedSceneSurfaceMaterial{0.5f, 0.25f}, 2u);
    }
    scene.packedIndices = {0u, 1u, 2u};
    ImportedScenePackedDraw draw{};
    draw.indexCount = 3u;
    scene.packedDraws.push_back(draw);

    scene.materials.resize(3);
    scene.materials[1].name = "brick_1890";
    scene.materials[1].roughness = 0.82f;
    scene.materials[1].baseColorTint[0] = 0.62f;
    scene.materials[2].name = "mullion_aluminum";
    scene.materials[2].metallic = 0.90f;
    scene.materials[2].roughness = 0.34f;
    scene.materials[2].emissive[0] = 0.25f;
    scene.materials[2].emissiveStrength = 4.0f;

    const fs::path scenePath =
        fs::temp_directory_path() / "odai_material_library_roundtrip.bin";
    expectTrue(saveImportedScene(scene, scenePath), "Material scene saves");

    const auto verify = [](const ImportedScene& s, const char* who) {
        expectTrue(s.materials.size() == 3, who);
        if (s.materials.size() != 3) return;
        expectTrue(s.materials[0].name.empty(), "slot 0 stays the reserved sentinel");
        expectTrue(s.materials[1].name == "brick_1890", "material name round trip");
        expectNear(s.materials[1].roughness, 0.82f, 1e-6f, "roughness round trip");
        expectNear(s.materials[1].baseColorTint[0], 0.62f, 1e-6f, "base color round trip");
        expectTrue(s.materials[2].name == "mullion_aluminum", "second material name");
        expectNear(s.materials[2].metallic, 0.90f, 1e-6f, "metallic round trip");
        expectNear(s.materials[2].emissiveStrength, 4.0f, 1e-6f, "emissive strength round trip");
        expectTrue(!s.packedVertices.empty() &&
                       importedSceneMaterialIndex(s.packedVertices[0].flags) == 2u,
                   "vertex material index survives the round trip");
    };

    ImportedScene loaded;
    expectTrue(loadImportedScene(scenePath, loaded), "Material scene loads (full loader)");
    verify(loaded, "full loader sees 3 materials");

    ImportedScene runtime;
    expectTrue(loadImportedSceneRuntime(scenePath, runtime),
               "Material scene loads (runtime loader)");
    verify(runtime, "runtime loader sees 3 materials");

    // A DOCTORED-OLD HEADER MUST BE REJECTED. This rewrites a current file's
    // version word to 17 and nothing else, so its bytes are still perfectly
    // valid -- the point is that the version alone decides, and the reader does
    // not attempt a layout it no longer knows how to read. See
    // kMinSupportedImportedSceneVersion.
    const fs::path legacyPath = fs::temp_directory_path() / "odai_material_library_v17.bin";
    {
        std::ifstream in(scenePath, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
        const std::uint32_t seventeen = 17u;
        std::memcpy(bytes.data() + sizeof(std::uint32_t), &seventeen, sizeof(seventeen));
        std::ofstream out(legacyPath, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    ImportedScene legacy;
    expectTrue(!loadImportedScene(legacyPath, legacy), "a v17 file no longer loads");
    expectTrue(legacy.packedVertices.empty(), "a rejected file leaves the scene untouched");

    fs::remove(scenePath);
    fs::remove(legacyPath);
}

// Ray picking against packed geometry -- the material editor's selection path,
// exercised with no window and no Vulkan.
void testImportedSceneRaycast() {
    using namespace odai::importer;

    ImportedScene scene;
    // Two quads facing +Y at different heights, so a downward ray must pick the
    // nearer one. Each carries a distinct material index.
    const auto addQuad = [&scene](float y, std::uint32_t materialIndex) {
        const std::uint32_t base = static_cast<std::uint32_t>(scene.packedVertices.size());
        const float xs[4] = {-1.0f, 1.0f, 1.0f, -1.0f};
        const float zs[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
        for (int i = 0; i < 4; ++i) {
            ImportedScenePackedVertex v{};
            v.position[0] = xs[i];
            v.position[1] = y;
            v.position[2] = zs[i];
            v.normal[1] = 1.0f;
            v.flags = packImportedSceneMaterialFlags(ImportedSceneSurfaceMaterial{}, materialIndex);
            scene.packedVertices.push_back(v);
        }
        for (const std::uint32_t o : {0u, 1u, 2u, 0u, 2u, 3u}) {
            scene.packedIndices.push_back(base + o);
        }
    };
    addQuad(0.0f, 3u);   // lower
    addQuad(5.0f, 7u);   // upper -- the one a downward ray from above hits first
    ImportedScenePackedDraw draw{};
    draw.firstIndex = 0u;
    draw.indexCount = static_cast<std::uint32_t>(scene.packedIndices.size());
    scene.packedDraws.push_back(draw);

    const odai::math::Ray down{{0.0f, 10.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};
    const ImportedSceneRayHit hit = raycastImportedScene(scene, down, 100.0f);
    expectTrue(hit.hit, "ray hits the stacked quads");
    expectNear(hit.distance, 5.0f, 1e-3f, "ray picks the NEAREST surface");
    expectTrue(hit.materialIndex == 7u, "hit reports the material index of the nearest surface");
    expectNear(hit.position.y, 5.0f, 1e-3f, "hit position lands on the surface");

    // maxDistance must actually bound the search.
    const ImportedSceneRayHit shortRay = raycastImportedScene(scene, down, 2.0f);
    expectTrue(!shortRay.hit, "a ray shorter than the surface distance misses");

    // A ray pointing away hits nothing.
    const odai::math::Ray up{{0.0f, 10.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    expectTrue(!raycastImportedScene(scene, up, 100.0f).hit, "a ray pointing away misses");

    // Degenerate inputs are handled rather than crashing.
    const odai::math::Ray zero{{0.0f, 10.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    expectTrue(!raycastImportedScene(scene, zero, 100.0f).hit, "a zero-length ray misses");
    expectTrue(!raycastImportedScene(ImportedScene{}, down, 100.0f).hit, "an empty scene misses");
}

}  // namespace

// v19 added a colour to ImportedSceneVertex, widening a struct that both loaders
// handle as a raw array: the full loader blits it, and the runtime loader SKIPS
// it by computing vertexCount * sizeof(ImportedSceneVertex). Get the stride
// wrong on an older file and nothing errors -- the mesh block is mis-sized and
// every section after it is read from the wrong offset.
//
// Rather than hand-roll a fixture (which would only prove the test author and
// the loader share a misunderstanding), this saves a real scene and rewrites the
// bytes: version 19 -> 18, and each 11-float vertex down to the old 8. The rest
// of the file is copied verbatim, so the sections after the mesh block are
// genuinely the ones the writer produced.
void testPreV19VertexLayoutCompatibility() {
    namespace fs = std::filesystem;
    using namespace odai::importer;

    // No textures and no instances, so the packed block lands immediately after
    // the mesh block on disk — which is what lets this rewrite both arrays and
    // then use the packed data as proof the mesh block was sized right.
    ImportedScene scene{};
    scene.sourceTag = "v18_compat";

    ImportedSceneMesh mesh{};
    mesh.name = "terrain";
    mesh.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{4.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 0.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    mesh.indices = {0u, 1u, 2u};
    mesh.parts = {ImportedSceneMeshPart{0u, 3u, 0xffffffffu, false}};
    scene.meshes.push_back(mesh);

    buildImportedScenePackedRenderData(scene);
    expectTrue(!scene.packedVertices.empty(), "downgrade fixture has packed vertices to check");

    const fs::path v19Path = fs::temp_directory_path() / "odai_v19_source.bin";
    const fs::path v18Path = fs::temp_directory_path() / "odai_v18_compat.bin";
    expectTrue(saveImportedScene(scene, v19Path), "v19 scene saves for the downgrade fixture");

    std::vector<std::uint8_t> bytes;
    {
        std::ifstream input(v19Path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    expectTrue(!bytes.empty(), "downgrade fixture source is non-empty");

    // Header: magic, version, sourceTag string, 10 counts, two float[3] bounds.
    // Then 0 textures, then the mesh block: name, three counts, vertices.
    std::size_t cursor = sizeof(std::uint32_t) * 2;
    const std::uint32_t tagLength = *reinterpret_cast<const std::uint32_t*>(bytes.data() + cursor);
    cursor += sizeof(std::uint32_t) + tagLength;
    cursor += sizeof(std::uint32_t) * 10;
    cursor += sizeof(float) * 6;
    const std::uint32_t meshNameLength = *reinterpret_cast<const std::uint32_t*>(bytes.data() + cursor);
    cursor += sizeof(std::uint32_t) + meshNameLength;
    const std::uint32_t vertexCount = *reinterpret_cast<const std::uint32_t*>(bytes.data() + cursor);
    const std::uint32_t indexCount = *reinterpret_cast<const std::uint32_t*>(bytes.data() + cursor + 4);
    const std::uint32_t partCount = *reinterpret_cast<const std::uint32_t*>(bytes.data() + cursor + 8);
    cursor += sizeof(std::uint32_t) * 3;
    expectTrue(vertexCount == 3u, "downgrade fixture locates the mesh vertex block");

    const std::size_t vertexBlockBegin = cursor;
    const std::size_t currentVertexBytes = vertexCount * sizeof(ImportedSceneVertex);

    std::vector<std::uint8_t> downgraded(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(vertexBlockBegin));
    *reinterpret_cast<std::uint32_t*>(downgraded.data() + sizeof(std::uint32_t)) = 18u;
    for (std::uint32_t i = 0; i < vertexCount; ++i) {
        const auto* vertex = reinterpret_cast<const ImportedSceneVertex*>(
            bytes.data() + vertexBlockBegin + (i * sizeof(ImportedSceneVertex)));
        const float legacy[8] = {
            vertex->position[0], vertex->position[1], vertex->position[2],
            vertex->normal[0], vertex->normal[1], vertex->normal[2],
            vertex->uv[0], vertex->uv[1]
        };
        const auto* raw = reinterpret_cast<const std::uint8_t*>(legacy);
        downgraded.insert(downgraded.end(), raw, raw + sizeof(legacy));
    }
    // Everything between the mesh vertices and the packed vertices is copied as
    // is: indices, parts, then (all empty here) instances, landscape, water,
    // lights, unresolved refs.
    const std::size_t tailBegin = vertexBlockBegin + currentVertexBytes;
    const std::size_t packedBlockBegin = tailBegin +
        (static_cast<std::size_t>(indexCount) * sizeof(std::uint32_t)) +
        (static_cast<std::size_t>(partCount) * sizeof(ImportedSceneMeshPart));
    expectTrue(packedBlockBegin < bytes.size(), "downgrade fixture locates the packed vertex block");
    downgraded.insert(
        downgraded.end(),
        bytes.begin() + static_cast<std::ptrdiff_t>(tailBegin),
        bytes.begin() + static_cast<std::ptrdiff_t>(packedBlockBegin));

    // Packed vertices were 52 bytes through v19; v20 appended three layer
    // indices and the weight word.
    for (const ImportedScenePackedVertex& packed : scene.packedVertices) {
        float head[11] = {
            packed.position[0], packed.position[1], packed.position[2],
            packed.normal[0], packed.normal[1], packed.normal[2],
            packed.color[0], packed.color[1], packed.color[2],
            packed.uv[0], packed.uv[1]
        };
        const auto* headBytes = reinterpret_cast<const std::uint8_t*>(head);
        downgraded.insert(downgraded.end(), headBytes, headBytes + sizeof(head));
        const std::uint32_t tail[2] = {packed.textureIndex, packed.flags};
        const auto* tailBytes = reinterpret_cast<const std::uint8_t*>(tail);
        downgraded.insert(downgraded.end(), tailBytes, tailBytes + sizeof(tail));
    }
    // Packed indices copy as is, but packed draws were a bare
    // {firstIndex, indexCount} pair through v22; v23 appended the alpha-test
    // threshold and its padding. A v18 fixture has to carry the narrow form or
    // the reader walks off the end of the draw block and everything after it.
    const std::size_t packedIndicesBegin =
        packedBlockBegin + (scene.packedVertices.size() * sizeof(ImportedScenePackedVertex));
    const std::size_t packedDrawsBegin =
        packedIndicesBegin + (scene.packedIndices.size() * sizeof(std::uint32_t));
    const std::size_t packedDrawsEnd =
        packedDrawsBegin + (scene.packedDraws.size() * sizeof(ImportedScenePackedDraw));
    expectTrue(packedDrawsEnd <= bytes.size(), "downgrade fixture locates the packed draw block");
    downgraded.insert(
        downgraded.end(),
        bytes.begin() + static_cast<std::ptrdiff_t>(packedIndicesBegin),
        bytes.begin() + static_cast<std::ptrdiff_t>(packedDrawsBegin));
    for (const ImportedScenePackedDraw& packedDraw : scene.packedDraws) {
        const std::uint32_t legacyDraw[2] = {packedDraw.firstIndex, packedDraw.indexCount};
        const auto* drawBytes = reinterpret_cast<const std::uint8_t*>(legacyDraw);
        downgraded.insert(downgraded.end(), drawBytes, drawBytes + sizeof(legacyDraw));
    }
    downgraded.insert(
        downgraded.end(),
        bytes.begin() + static_cast<std::ptrdiff_t>(packedDrawsEnd),
        bytes.end());
    {
        std::ofstream output(v18Path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(downgraded.data()),
                     static_cast<std::streamsize>(downgraded.size()));
    }

    // A BELOW-FLOOR FILE MUST BE REJECTED, NOT EXPANDED. The reader used to
    // carry every layout back to v15; it now supports exactly
    // kMinSupportedImportedSceneVersion and up, because a scene older than that
    // is one no current build can produce or consult -- a cell cache is keyed by
    // kCellBuildVersion and the plugin's size and mtime, so the caller rebuilds
    // it either way.
    //
    // The failure has to be CLEAN. This fixture is a v19 file with its header
    // rewritten to 18 and its vertex block narrowed, so every byte after the
    // mesh block sits at the wrong offset: a reader that tried anyway would
    // report success on garbage rather than error, which is the whole reason
    // this fixture exists.
    ImportedScene loaded{};
    expectTrue(!loadImportedScene(v18Path, loaded),
               "a file below the supported version floor is rejected");
    expectTrue(!getImportedSceneLastError().empty(),
               "rejecting an old file names a reason");
    ImportedScene runtimeLoaded{};
    expectTrue(!loadImportedSceneRuntime(v18Path, runtimeLoaded),
               "the runtime loader rejects it too");

    std::error_code cleanupError;
    fs::remove(v19Path, cleanupError);
    fs::remove(v18Path, cleanupError);
}

// Authored vertex colour must reach the packed stream AND opt into the tint bit;
// synthesized fallback colours must not set it, or every cooked scene without
// VCLR would start multiplying its textures by a height ramp.
void testVertexColorTintFlag() {
    using namespace odai::importer;

    ImportedScene scene{};
    scene.sourceTag = "fnv_exterior";

    ImportedSceneMesh mesh{};
    mesh.name = "terrain";
    mesh.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {0.5f, 0.25f, 0.125f}},
        ImportedSceneVertex{{4.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, 0.25f, 0.125f}},
        ImportedSceneVertex{{0.0f, 0.0f, 4.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {0.5f, 0.25f, 0.125f}}
    };
    mesh.indices = {0u, 1u, 2u};
    mesh.parts = {ImportedSceneMeshPart{0u, 3u, 0u, false}};
    scene.meshes.push_back(mesh);
    scene.landscapeCells.resize(1);

    buildImportedScenePackedRenderData(scene);
    expectTrue(!scene.packedVertices.empty(), "tinted terrain packs vertices");
    if (!scene.packedVertices.empty()) {
        const ImportedScenePackedVertex& packed = scene.packedVertices[0];
        expectTrue((packed.flags & kImportedSceneMaterialFlagTerrainSlopeBlend) != 0u,
                   "terrain packing marks terrain for runtime material presets");
        expectNear(packed.color[0], 0.5f, 1e-5f, "authored vertex colour reaches the packed stream");
        expectNear(packed.color[2], 0.125f, 1e-5f, "authored vertex colour keeps its blue channel");
        expectTrue((packed.flags & kImportedSceneMaterialFlagVertexColorTint) != 0u,
                   "authored vertex colour sets the tint flag");
    }

    // Same geometry with the default white: the fallback ramp applies and the
    // bit stays clear, so the shader leaves those textures alone.
    ImportedScene untinted{};
    untinted.sourceTag = "fnv_exterior";
    ImportedSceneMesh plainMesh = mesh;
    for (ImportedSceneVertex& vertex : plainMesh.vertices) {
        vertex.color[0] = 1.0f;
        vertex.color[1] = 1.0f;
        vertex.color[2] = 1.0f;
    }
    untinted.meshes.push_back(plainMesh);
    untinted.landscapeCells.resize(1);
    buildImportedScenePackedRenderData(untinted);
    expectTrue(!untinted.packedVertices.empty(), "untinted terrain packs vertices");
    if (!untinted.packedVertices.empty()) {
        expectTrue((untinted.packedVertices[0].flags & kImportedSceneMaterialFlagVertexColorTint) == 0u,
                   "white vertex colour does not set the tint flag");
    }
}

// Terrain layers must survive packing with their weights quantized but their
// ordering and identity intact, and must opt in via the flag exactly when a
// layer is actually present.
void testTerrainLayerPacking() {
    using namespace odai::importer;

    ImportedScene scene{};
    scene.sourceTag = "fnv_exterior";

    ImportedSceneMesh mesh{};
    mesh.name = "terrain";
    ImportedSceneVertex layered{};
    layered.position[0] = 1.0f;
    layered.layerTextureIndex[0] = 7u;
    layered.layerWeight[0] = 1.0f;
    layered.layerTextureIndex[1] = 9u;
    layered.layerWeight[1] = 0.5f;
    layered.layerTextureIndex[3] = kImportedSceneTerrainNormalizedBlendMarker;
    ImportedSceneVertex declaredZero{};
    declaredZero.position[0] = 2.0f;
    declaredZero.layerTextureIndex[0] = 12u;
    ImportedSceneVertex plain{};
    plain.position[0] = 3.0f;

    mesh.vertices = {layered, layered, declaredZero, plain};
    mesh.indices = {0u, 1u, 2u, 1u, 2u, 3u};
    mesh.parts = {ImportedSceneMeshPart{0u, 6u, 0u, false}};
    scene.meshes.push_back(mesh);
    scene.landscapeCells.resize(1);

    buildImportedScenePackedRenderData(scene);
    expectTrue(scene.packedVertices.size() == 4, "layered terrain packs one vertex per source vertex");
    if (scene.packedVertices.size() != 4) {
        return;
    }

    const ImportedScenePackedVertex& packedLayered = scene.packedVertices[0];
    expectTrue((packedLayered.flags & kImportedSceneMaterialFlagTerrainLayers) != 0u,
               "a vertex with a layer sets the terrain layer flag");
    expectTrue((packedLayered.flags &
                    kImportedSceneMaterialFlagTerrainNormalizedLayers) != 0u,
               "the TES3 normalized-layer marker becomes a packed material semantic");
    expectTrue(packedLayered.layerTextureIndex[0] == 7u, "layer 0 texture index survives packing");
    expectTrue(packedLayered.layerTextureIndex[1] == 9u, "layer 1 texture index survives packing");
    expectTrue(packedLayered.layerTextureIndex[2] == kImportedSceneNoTerrainLayer,
               "unused layer slot stays empty");
    expectTrue(packedLayered.layerTextureIndex[3] == kImportedSceneNoTerrainLayer,
               "the transient normalized-layer marker is not serialized as a texture");
    // 8-bit quantization: 1.0 -> 255, 0.5 -> 128 (round-half-up).
    expectTrue((packedLayered.layerWeights & 0xffu) == 255u, "layer 0 weight quantizes to full");
    expectTrue(((packedLayered.layerWeights >> 8) & 0xffu) == 128u, "layer 1 weight quantizes to half");
    expectTrue(((packedLayered.layerWeights >> 16) & 0xffu) == 0u, "unused layer weight is zero");

    const ImportedScenePackedVertex& packedDeclaredZero = scene.packedVertices[2];
    expectTrue((packedDeclaredZero.flags & kImportedSceneMaterialFlagTerrainLayers) != 0u,
               "a declared zero-weight layer keeps the flat triangle semantic enabled");
    expectTrue((packedDeclaredZero.layerWeights & 0xffu) == 0u,
               "a declared zero-weight layer remains zero after packing");

    const ImportedScenePackedVertex& packedPlain = scene.packedVertices[3];
    expectTrue((packedPlain.flags & kImportedSceneMaterialFlagTerrainLayers) == 0u,
               "a vertex with no layers does not set the terrain layer flag");

    // And the whole thing round-trips at the current version.
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() / "odai_terrain_layers.bin";
    expectTrue(saveImportedScene(scene, path), "layered scene saves");
    ImportedScene loaded{};
    expectTrue(loadImportedScene(path, loaded), "layered scene loads");
    if (loaded.packedVertices.size() == 4) {
        expectTrue(loaded.packedVertices[0].layerTextureIndex[1] == 9u,
                   "layer texture index round-trips through the file");
        expectTrue(loaded.packedVertices[0].layerWeights == packedLayered.layerWeights,
                   "packed layer weights round-trip through the file");
        expectTrue((loaded.packedVertices[0].flags &
                        kImportedSceneMaterialFlagTerrainNormalizedLayers) != 0u,
                   "normalized TES3 terrain semantics round-trip through the packed flags");
    }
    expectTrue(loaded.meshes.size() == 1 && loaded.meshes[0].vertices.size() == 4,
               "layered mesh vertices round-trip");
    if (loaded.meshes.size() == 1 && loaded.meshes[0].vertices.size() == 4) {
        expectNear(loaded.meshes[0].vertices[0].layerWeight[1], 0.5f, 1e-5f,
                   "source layer weight round-trips unquantized");
    }
    std::error_code cleanupError;
    fs::remove(path, cleanupError);
}

// The alpha-test threshold is authored per surface and rides the packed draw
// rather than the vertex. Two things have to hold for that to be usable: the
// value has to survive packing (which reorders draws to build page ranges) and
// it has to survive the round trip to disk.
void testAlphaThresholdRoundTrip() {
    namespace fs = std::filesystem;
    using namespace odai::importer;

    ImportedScene scene;
    scene.sourceTag = "alpha_threshold";

    // Two parts of one mesh with different thresholds, so a single shared
    // threshold or a dropped one both show up as a failure.
    ImportedSceneMesh mesh{};
    mesh.name = "cutouts";
    mesh.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
        ImportedSceneVertex{{2.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{3.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{2.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}
    };
    mesh.indices = {0u, 1u, 2u, 3u, 4u, 5u};
    ImportedSceneMeshPart lowPart{};
    lowPart.firstIndex = 0u;
    lowPart.indexCount = 3u;
    lowPart.textureIndex = 0xffffffffu;
    lowPart.alphaTest = true;
    lowPart.alphaThreshold = 32u;
    ImportedSceneMeshPart highPart{};
    highPart.firstIndex = 3u;
    highPart.indexCount = 3u;
    highPart.textureIndex = 0xffffffffu;
    highPart.alphaTest = true;
    highPart.alphaThreshold = 200u;
    mesh.parts = {lowPart, highPart};
    scene.meshes.push_back(mesh);

    // Packing walks instances, not meshes; a mesh nothing places produces
    // nothing.
    ImportedSceneInstance instance{};
    instance.meshIndex = 0u;
    instance.modelPath = "cutouts.nif";
    instance.transform[0] = 1.0f;
    instance.transform[5] = 1.0f;
    instance.transform[10] = 1.0f;
    instance.transform[15] = 1.0f;
    scene.instances.push_back(instance);

    buildImportedScenePackedRenderData(scene);
    expectTrue(scene.packedDraws.size() == 2u, "one packed draw per part");
    if (scene.packedDraws.size() != 2u) {
        return;
    }
    expectTrue(scene.packedDraws[0].alphaThreshold == 32u, "low threshold packs onto its draw");
    expectTrue(scene.packedDraws[1].alphaThreshold == 200u, "high threshold packs onto its draw");

    // Page building rewrites the draw array wholesale; the threshold has to be
    // carried across rather than reset to the default.
    buildImportedScenePageRanges(scene, 64.0f);
    expectTrue(scene.packedDraws.size() == 2u, "page build keeps both draws");
    if (scene.packedDraws.size() == 2u) {
        expectTrue(scene.packedDraws[0].alphaThreshold == 32u, "page build keeps the low threshold");
        expectTrue(scene.packedDraws[1].alphaThreshold == 200u, "page build keeps the high threshold");
    }

    const fs::path scenePath = fs::temp_directory_path() / "odai_alpha_threshold.bin";
    expectTrue(saveImportedScene(scene, scenePath), "threshold scene saves");
    ImportedScene loaded{};
    expectTrue(loadImportedScene(scenePath, loaded), "threshold scene loads");
    expectTrue(loaded.packedDraws.size() == 2u, "threshold scene round trips both draws");
    if (loaded.packedDraws.size() == 2u) {
        expectTrue(loaded.packedDraws[0].alphaThreshold == 32u, "low threshold survives disk");
        expectTrue(loaded.packedDraws[1].alphaThreshold == 200u, "high threshold survives disk");
    }
    fs::remove(scenePath);
}

// The CPU half of ImportedMeshVertex's packing against the shader half in
// shaders/imported_vertex_pack.slang. Nothing else can catch those two
// drifting: a wrong decode does not fail, it shades slightly wrong forever.
//
// The decoders below are transcribed from that Slang module deliberately -- if
// someone edits the shader without editing this, the point is that this test
// keeps asserting the OLD contract and starts failing.
void testImportedVertexPacking() {
    using odai::importer::packImportedVertexColor;
    using odai::importer::packImportedVertexLayerPair;
    using odai::importer::packImportedVertexNormal;

    const auto decodeNormal = [](std::uint32_t packed) {
        const auto half = [](std::uint32_t bits) {
            return static_cast<float>(static_cast<std::int16_t>(bits & 0xffffu)) / 32767.0f;
        };
        float x = half(packed);
        float z = half(packed >> 16);
        float y = 1.0f - std::abs(x) - std::abs(z);
        if (y < 0.0f) {
            const float foldedX = (1.0f - std::abs(z)) * (x >= 0.0f ? 1.0f : -1.0f);
            const float foldedZ = (1.0f - std::abs(x)) * (z >= 0.0f ? 1.0f : -1.0f);
            x = foldedX;
            z = foldedZ;
        }
        const float length = std::sqrt((x * x) + (y * y) + (z * z));
        return std::array<float, 3>{x / length, y / length, z / length};
    };
    const auto decodeSrgbByte = [](std::uint32_t byte) {
        const float c = static_cast<float>(byte) / 255.0f;
        return (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
    };

    // Normals: worst-case angular error over a deterministic sweep of the sphere.
    double worstDegrees = 0.0;
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            const float theta = static_cast<float>(i) * 3.14159265f / 63.0f;
            const float phi = static_cast<float>(j) * 6.28318531f / 64.0f;
            const float normal[3] = {
                std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
            const std::array<float, 3> decoded = decodeNormal(packImportedVertexNormal(normal));
            const float dot = (normal[0] * decoded[0]) + (normal[1] * decoded[1]) +
                              (normal[2] * decoded[2]);
            worstDegrees = std::max<double>(
                worstDegrees,
                std::acos(static_cast<double>(std::clamp(dot, -1.0f, 1.0f))) * 180.0 / 3.14159265);
        }
    }
    // 0.05, against a measured worst case of 0.034 degrees. The worst case is
    // NOT uniform over the sphere -- it sits just below the equator near the
    // fold diagonals, around (0.98, -0.07, 0.19) -- so a random-sample estimate
    // understates it by an order of magnitude. This grid sweep hits it.
    expectTrue(worstDegrees < 0.05,
               "octahedral normal packing stays under 0.05 degrees of error");

    // A degenerate normal must not produce NaN -- the shader normalizes what it
    // gets and would propagate one straight into the lighting.
    const float zeroNormal[3] = {0.0f, 0.0f, 0.0f};
    const std::array<float, 3> degenerate = decodeNormal(packImportedVertexNormal(zeroNormal));
    expectTrue(std::isfinite(degenerate[0]) && std::isfinite(degenerate[1]) &&
                   std::isfinite(degenerate[2]),
               "a zero-length normal packs to a finite direction");

    // Colour: every one of the 256 sRGB source bytes must survive the round trip
    // exactly. That is the case that matters -- these values are authored as
    // sRGB bytes, not as arbitrary linear floats.
    int inexact = 0;
    for (std::uint32_t byte = 0; byte < 256u; ++byte) {
        const float linear = decodeSrgbByte(byte);
        const float color[3] = {linear, linear, linear};
        const std::uint32_t packed = packImportedVertexColor(color);
        if ((packed & 0xffu) != byte || ((packed >> 8) & 0xffu) != byte ||
            ((packed >> 16) & 0xffu) != byte) {
            ++inexact;
        }
    }
    expectTrue(inexact == 0, "all 256 sRGB source bytes round-trip exactly through vertex colour");

    // Out-of-range input clamps rather than wrapping into a bright colour.
    const float overbright[3] = {4.0f, -1.0f, 0.0f};
    const std::uint32_t clamped = packImportedVertexColor(overbright);
    expectTrue((clamped & 0xffu) == 255u, "colour above 1.0 clamps to white");
    expectTrue(((clamped >> 8) & 0xffu) == 0u, "colour below 0.0 clamps to black");

    // Layer slots: two per word, and anything that will not fit in 16 bits must
    // land on the sentinel instead of truncating into a valid-looking slot.
    expectTrue(packImportedVertexLayerPair(3u, 9u) == (3u | (9u << 16)),
               "layer slots pack low half first");
    expectTrue(packImportedVertexLayerPair(0xffffffffu, 0x1ffffu) == 0xffffffffu,
               "unrepresentable layer slots become the 0xffff sentinel, not a truncated index");
}

void testRigidAnimationPackingSamplingAndRoundTrip() {
    namespace fs = std::filesystem;
    using namespace odai::importer;

    ImportedScene scene{};
    scene.sourceTag = "synthetic_rigid_animation";

    ImportedSceneMesh mesh{};
    mesh.name = "wheel";
    mesh.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}};
    mesh.indices = {0u, 1u, 2u};
    ImportedSceneMeshPart part{};
    part.indexCount = 3u;
    part.rigidAnimationIndex = 0u;
    mesh.parts.push_back(part);

    ImportedSceneRigidAnimation animation{};
    animation.nodeName = "wheel axle";
    animation.duration = 2.0f;
    animation.cycleType = 0u;
    animation.translationKeys = {
        ImportedSceneVectorKey{0.0f, {0.0f, 0.0f, 0.0f}},
        ImportedSceneVectorKey{2.0f, {10.0f, 0.0f, 0.0f}}};
    mesh.rigidAnimations.push_back(animation);
    scene.meshes.push_back(std::move(mesh));

    ImportedSceneInstance instance{};
    instance.meshIndex = 0u;
    instance.transform[0] = 1.0f;
    instance.transform[5] = 1.0f;
    instance.transform[10] = 1.0f;
    instance.transform[15] = 1.0f;
    instance.transform[3] = 100.0f;
    scene.instances.push_back(instance);

    buildImportedScenePackedRenderData(scene);
    expectTrue(scene.rigidAnimations.size() == 1u,
               "rigid animation templates are placed with their mesh instance");
    expectTrue(scene.packedDraws.size() == 1u &&
                   scene.packedDraws.front().rigidAnimationIndex == 0u,
               "packed draw points at its placed rigid animation");

    float sampled[16] = {};
    expectTrue(sampleImportedSceneRigidAnimation(scene.rigidAnimations.front(), 1.0f, sampled),
               "placed rigid animation samples");
    expectNear(sampled[3], 5.0f, 1.0e-4f,
               "rigid animation produces a bind-relative midpoint transform");
    expectTrue(sampleImportedSceneRigidAnimation(scene.rigidAnimations.front(), 2.5f, sampled),
               "looping rigid animation samples after its duration");
    expectNear(sampled[3], 2.5f, 1.0e-4f,
               "looping rigid animation wraps on the authored duration");

    const fs::path path = fs::temp_directory_path() / "odai_rigid_animation_roundtrip.bin";
    expectTrue(saveImportedScene(scene, path), "rigid animation scene saves");
    ImportedScene loaded{};
    expectTrue(loadImportedSceneRuntime(path, loaded), "rigid animation scene loads at runtime");
    expectTrue(loaded.rigidAnimations.size() == 1u,
               "runtime scene retains packed rigid animations");
    expectTrue(loaded.packedDraws.size() == 1u &&
                   loaded.packedDraws.front().rigidAnimationIndex == 0u,
               "runtime scene retains packed rigid animation draw indices");
    expectTrue(loaded.rigidAnimations.front().translationKeys.size() == 2u,
               "runtime scene retains rigid animation keys");
    fs::remove(path);
}

void testStaticNormalMapSidecarPackingAndRoundTrip() {
    namespace fs = std::filesystem;
    using namespace odai::importer;

    ImportedScene scene{};
    scene.sourceTag = "synthetic_static_normal_map";
    scene.textures.resize(2u);
    scene.textures[0].sourcePath = "textures\\architecture\\gate.dds";
    scene.textures[1].sourcePath = "textures\\architecture\\gate_n.dds";
    scene.normalTextureByDiffuseIndex.emplace(0u, 1u);

    ImportedSceneMesh mesh{};
    mesh.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}};
    mesh.indices = {0u, 1u, 2u};
    ImportedSceneMeshPart part{};
    part.indexCount = 3u;
    part.textureIndex = 0u;
    mesh.parts.push_back(part);
    scene.meshes.push_back(std::move(mesh));

    ImportedSceneInstance instance{};
    instance.meshIndex = 0u;
    instance.transform[0] = 1.0f;
    instance.transform[5] = 1.0f;
    instance.transform[10] = 1.0f;
    instance.transform[15] = 1.0f;
    scene.instances.push_back(instance);

    buildImportedScenePackedRenderData(scene);
    expectTrue(!scene.packedVertices.empty() &&
                   scene.packedVertices.front().layerTextureIndex[0] == 1u,
               "static authored normal map occupies the existing non-terrain layer slot");
    expectTrue((scene.packedVertices.front().flags &
                kImportedSceneMaterialFlagTerrainLayers) == 0u,
               "static normal-map sidecar does not masquerade as terrain layering");

    const fs::path path = fs::temp_directory_path() / "odai_static_normal_map_roundtrip.bin";
    expectTrue(saveImportedScene(scene, path), "normal-map sidecar scene saves");
    ImportedScene loaded{};
    expectTrue(loadImportedSceneRuntime(path, loaded), "normal-map sidecar scene loads");
    expectTrue(!loaded.packedVertices.empty() &&
                   loaded.packedVertices.front().layerTextureIndex[0] == 1u,
               "normal-map slot survives through the unchanged packed-vertex layout");
    fs::remove(path);
}

void testDistantLodTessellationMarkerPacking() {
    using namespace odai::importer;

    ImportedScene scene{};
    scene.sourceTag = "skyrim_object_lod:Tamriel";
    ImportedSceneMesh mesh{};
    mesh.name = "lod4_4_-12_mountain";
    mesh.vertices = {
        ImportedSceneVertex{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
        ImportedSceneVertex{{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
        ImportedSceneVertex{{0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}};
    mesh.indices = {0u, 1u, 2u};
    ImportedSceneMeshPart part{};
    part.indexCount = 3u;
    part.vegetationReserved[0] =
        kImportedSceneMeshPartDistantLodTessellation |
        kImportedSceneMeshPartDistantLodSnow;
    mesh.parts.push_back(part);
    scene.meshes.push_back(std::move(mesh));

    ImportedSceneInstance instance{};
    instance.meshIndex = 0u;
    instance.transform[0] = instance.transform[5] =
        instance.transform[10] = instance.transform[15] = 1.0f;
    scene.instances.push_back(instance);
    scene.sourceLandscapeCellCount = 1u;

    buildImportedScenePackedRenderData(scene);
    expectTrue(!scene.packedVertices.empty() &&
                   (scene.packedVertices.front().flags &
                    kImportedSceneMaterialFlagDistantLodTessellation) != 0u,
               "distant mountain part marker reaches packed vertex flags");
    expectTrue((scene.packedVertices.front().flags &
                kImportedSceneMaterialFlagDistantLodSnow) != 0u,
               "authored distant snow coverage reaches the fragment material flags");
    buildImportedScenePageRanges(scene);
    expectTrue(scene.pageRanges.size() == 1u &&
                   scene.pageRanges.front().terrainDrawCount == 1u,
               "distant mountain remains in the tessellated draw prefix after paging");
}

int main() {
    testImportedSceneSerialization();
    testPreV19VertexLayoutCompatibility();
    testVertexColorTintFlag();
    testTerrainLayerPacking();
    testImportedSceneSourceTagInteriorClassification();
    testTextureFormatRoundTrip();
    testLegacyArgbWaterFlowDds();
    testBlockCompressedAlphaCutoutDetection();
    testPageRangeBuildAndRoundTrip();
    testMaterialLibraryRoundTrip();
    testImportedSceneRaycast();
    testAlphaThresholdRoundTrip();
    testImportedVertexPacking();
    testRigidAnimationPackingSamplingAndRoundTrip();
    testStaticNormalMapSidecarPackingAndRoundTrip();
    testDistantLodTessellationMarkerPacking();

    if (g_failures != 0) {
        std::cerr << "[imported scene test] " << g_failures << " failures\n";
        return 1;
    }

    std::cout << "[imported scene test] all checks passed\n";
    return 0;
}
