#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "app/morrowind_actor_system.h"
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

std::array<float, 16> multiplyMatrices(
    const std::array<float, 16>& lhs,
    const std::array<float, 16>& rhs
) {
    std::array<float, 16> out{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                sum += lhs[row * 4 + i] * rhs[i * 4 + col];
            }
            out[row * 4 + col] = sum;
        }
    }
    return out;
}

std::array<float, 16> matrixFromFloats(const float matrix[16]) {
    std::array<float, 16> out{};
    std::copy(matrix, matrix + 16, out.begin());
    return out;
}

float maxAbsDifferenceFromIdentity(const std::array<float, 16>& matrix) {
    const std::array<float, 16> identity{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float maxDifference = 0.0f;
    for (std::size_t i = 0; i < matrix.size(); ++i) {
        maxDifference = std::max(maxDifference, std::fabs(matrix[i] - identity[i]));
    }
    return maxDifference;
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

void appendAvObjectWithTransform(
    std::vector<std::uint8_t>& bytes,
    const std::string& name,
    const float (&translation)[3],
    const float (&rotation)[9],
    float scale = 1.0f
) {
    appendSizedString(bytes, name);
    appendBinaryValue(bytes, static_cast<std::int32_t>(-1));
    appendBinaryValue(bytes, static_cast<std::int32_t>(-1));
    appendBinaryValue(bytes, static_cast<std::uint16_t>(0));

    for (const float value : translation) {
        appendBinaryValue(bytes, value);
    }
    for (const float value : rotation) {
        appendBinaryValue(bytes, value);
    }
    appendBinaryValue(bytes, scale);
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

void appendNifNodeWithTransform(
    std::vector<std::uint8_t>& bytes,
    const std::string& name,
    std::initializer_list<std::int32_t> children,
    const float (&translation)[3],
    const float (&rotation)[9]
) {
    appendSizedString(bytes, "NiNode");
    appendAvObjectWithTransform(bytes, name, translation, rotation);
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

void appendNifSkinnedTriShape(
    std::vector<std::uint8_t>& bytes,
    const std::string& name,
    std::int32_t dataRef,
    std::int32_t skinInstanceRef
) {
    appendSizedString(bytes, "NiTriShape");
    appendIdentityAvObject(bytes, name);
    appendBinaryValue(bytes, dataRef);
    appendBinaryValue(bytes, skinInstanceRef);
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

void appendSkinTransform(
    std::vector<std::uint8_t>& bytes,
    const float (&translation)[3],
    const float (&rotation)[9],
    float scale = 1.0f
) {
    for (const float value : rotation) {
        appendBinaryValue(bytes, value);
    }
    for (const float value : translation) {
        appendBinaryValue(bytes, value);
    }
    appendBinaryValue(bytes, scale);
}

void appendNifSkinInstance(
    std::vector<std::uint8_t>& bytes,
    std::int32_t skinDataRef,
    std::int32_t skeletonRootRef,
    std::initializer_list<std::int32_t> boneRefs
) {
    appendSizedString(bytes, "NiSkinInstance");
    appendBinaryValue(bytes, skinDataRef);
    appendBinaryValue(bytes, skeletonRootRef);
    appendRefList(bytes, boneRefs);
}

void appendNifSkinDataSingleBone(
    std::vector<std::uint8_t>& bytes,
    const float (&skinTranslation)[3],
    const float (&skinRotation)[9],
    const float (&boneTranslation)[3],
    const float (&boneRotation)[9]
) {
    appendSizedString(bytes, "NiSkinData");
    appendSkinTransform(bytes, skinTranslation, skinRotation);
    appendBinaryValue(bytes, static_cast<std::uint32_t>(1));
    appendBinaryValue(bytes, static_cast<std::int32_t>(1));
    appendSkinTransform(bytes, boneTranslation, boneRotation);
    const float bounds[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    for (const float value : bounds) {
        appendBinaryValue(bytes, value);
    }
    appendBinaryValue(bytes, static_cast<std::uint16_t>(3));
    for (std::uint16_t vertexIndex = 0; vertexIndex < 3u; ++vertexIndex) {
        appendBinaryValue(bytes, vertexIndex);
        appendBinaryValue(bytes, 1.0f);
    }
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

std::filesystem::path writeSyntheticSkinnedTransformOrderNif() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai_skinned_transform_order_actor_part.nif";
    std::vector<std::uint8_t> bytes;
    const std::string header = "NetImmerse File Format, Version 4.0.0.2\n";
    bytes.insert(bytes.end(), header.begin(), header.end());
    appendBinaryValue(bytes, static_cast<std::uint32_t>(0x04000002u));
    appendBinaryValue(bytes, static_cast<std::uint32_t>(6));

    const float identityTranslation[3] = {0.0f, 0.0f, 0.0f};
    const float identityRotation[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    const float rotateZ90[9] = {
        0.0f, -1.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    const float skinTranslation[3] = {2.0f, 3.0f, 5.0f};

    appendNifNode(bytes, "Root", {1, 3});
    appendNifSkinnedTriShape(bytes, "tri Chest", 2, 4);
    const float positions[9] = {
        1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        1.0f, 0.0f, 1.0f
    };
    const std::uint16_t triangleIndices[3] = {0, 1, 2};
    appendNifTriShapeData(bytes, positions, triangleIndices);
    appendNifNodeWithTransform(bytes, "Bip01 Chest", {}, identityTranslation, rotateZ90);
    appendNifSkinInstance(bytes, 5, 0, {3});
    appendNifSkinDataSingleBone(
        bytes,
        skinTranslation,
        identityRotation,
        identityTranslation,
        identityRotation);

    appendBinaryValue(bytes, static_cast<std::uint32_t>(1));
    appendBinaryValue(bytes, static_cast<std::int32_t>(0));

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return path;
}

void appendTextKeyExtraData(std::vector<std::uint8_t>& bytes) {
    appendSizedString(bytes, "NiTextKeyExtraData");
    appendBinaryValue(bytes, static_cast<std::int32_t>(-1));
    appendBinaryValue(bytes, static_cast<std::uint32_t>(0));
    appendBinaryValue(bytes, static_cast<std::uint32_t>(2));
    appendBinaryValue(bytes, 0.0f);
    appendSizedString(bytes, "idle: start");
    appendBinaryValue(bytes, 1.0f);
    appendSizedString(bytes, "idle: stop");
}

std::filesystem::path writeSyntheticSkeletonOnlyNif() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "odai_skeleton_only_actor.nif";
    std::vector<std::uint8_t> bytes;
    const std::string header = "NetImmerse File Format, Version 4.0.0.2\n";
    bytes.insert(bytes.end(), header.begin(), header.end());
    appendBinaryValue(bytes, static_cast<std::uint32_t>(0x04000002u));
    appendBinaryValue(bytes, static_cast<std::uint32_t>(3));
    appendNifNode(bytes, "Bip01", {1});
    appendNifNode(bytes, "Bip01 Pelvis", {});
    appendTextKeyExtraData(bytes);
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
    unresolved.refNum = 12345u;
    unresolved.hasRefNum = true;
    unresolved.cellX = -2;
    unresolved.cellY = -9;
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
    expectTrue(loaded.unresolvedRefs.front().hasRefNum, "Imported scene unresolved ref number presence round-trips");
    expectTrue(loaded.unresolvedRefs.front().refNum == unresolved.refNum, "Imported scene unresolved ref number round-trips");
    expectTrue(
        loaded.unresolvedRefs.front().cellX == unresolved.cellX &&
        loaded.unresolvedRefs.front().cellY == unresolved.cellY,
        "Imported scene unresolved ref source cell round-trips");
    expectTrue(!loaded.unresolvedRefs.front().deleted, "Imported scene unresolved ref deleted flag defaults false");
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

void testMorrowindSkeletonOnlyNifLoadsAndExtractsTextKeys() {
    const std::filesystem::path nifPath = writeSyntheticSkeletonOnlyNif();

    odai::importer::ImportedAnimatedNifResult animated{};
    std::string error;
    expectTrue(
        odai::importer::loadMorrowindAnimatedNif(nifPath, animated, error),
        "Morrowind animated NIF loader accepts skeleton-only NIFs");
    expectTrue(animated.vertices.empty(), "Skeleton-only animated NIF has no geometry");
    expectTrue(animated.nodes.size() == 2u, "Skeleton-only animated NIF exposes nodes");
    expectTrue(animated.textKeys.size() == 2u, "Skeleton-only animated NIF extracts text keys");

    odai::importer::ImportedSkinnedActorAsset actor{};
    expectTrue(
        odai::importer::loadMorrowindSkinnedActorSkeleton(nifPath, actor, error),
        "Skeleton-only NIF loads as actor skeleton");
    expectTrue(actor.skeleton.size() == 2u, "Actor skeleton keeps skeleton-only nodes");
    expectTrue(!actor.animationClips.empty(), "Actor skeleton builds clips from text keys");
    expectNear(actor.skeleton[0].inverseBindWorldTransform[0], 1.0f, 0.0001f,
               "Actor skeleton computes inverse bind matrices");

    std::filesystem::remove(nifPath);
}

void testMorrowindSkinnedActorPartUsesOpenMwMatrixOrder() {
    const std::filesystem::path nifPath = writeSyntheticSkinnedTransformOrderNif();

    odai::importer::ImportedNifResult result{};
    std::string error;
    expectTrue(
        odai::importer::loadMorrowindActorPartNif(nifPath, result, error),
        "Skinned actor part matrix-order fixture loads");
    expectTrue(result.mesh.vertices.size() == 3u, "Matrix-order fixture keeps all vertices");
    expectTrue(result.mesh.indices.size() == 3u, "Matrix-order fixture keeps triangle indices");
    if (result.mesh.vertices.size() >= 3u) {
        // OpenMW evaluates sourceVertex * inverseBind * boneWorld * skinTransform
        // in row-vector OSG space. In our column-vector convention the same vertex
        // becomes skinTransform * boneWorld * inverseBind * sourceVertex, then the
        // Morrowind Y/Z axes are remapped into engine space.
        expectNear(result.mesh.vertices[0].position[0], 2.0f, 0.0001f,
                   "Skinned bind-pose bake applies skin translation after bone rotation (x)");
        expectNear(result.mesh.vertices[0].position[1], 5.0f, 0.0001f,
                   "Skinned bind-pose bake remaps NIF z to engine y");
        expectNear(result.mesh.vertices[0].position[2], 4.0f, 0.0001f,
                   "Skinned bind-pose bake applies OpenMW-compatible transform order");
        expectNear(result.mesh.vertices[1].position[0], 1.0f, 0.0001f,
                   "Skinned bind-pose bake rotates the second vertex");
        expectNear(result.mesh.vertices[2].position[1], 6.0f, 0.0001f,
                   "Skinned bind-pose bake preserves post-rotation height");
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

void testMorrowindFargothActorPartsLoadWhenDataFilesAvailable() {
    const std::filesystem::path dataFilesPath = "C:/GOG Games/Morrowind/Data Files";
    const std::filesystem::path meshesPath = dataFilesPath / "Meshes";
    if (!std::filesystem::exists(meshesPath)) {
        std::cout << "[imported scene test] skipping Fargoth actor part load: "
                  << meshesPath.string() << " not found\n";
        return;
    }

    const std::vector<std::string> fargothModelParts = {
        "b/B_N_Wood Elf_M_Skins.nif",
        "b/B_N_Wood Elf_M_Neck.NIF",
        "b/B_N_Wood Elf_M_Head_02.nif",
        "b/B_N_Wood Elf_M_Hair_03.nif",
        "c/C_M_Shirt_C_commonL04.NIF",
        "c/C_M_Shirt_UA_commonL04.nif",
        "c/C_M_Shirt_FA_commonL04.nif",
        "c/C_M_Shirt_W_commonL04.nif",
        "c/C_M_Pants_G_common02.nif",
        "c/C_M_pants_A_common02.nif",
        "c/C_M_Pants_UL_common02.nif",
        "c/C_M_Pants_K_common02.nif",
        "c/C_shoes_common_4.NIF"
    };

    for (const std::string& relativeModelPath : fargothModelParts) {
        const std::filesystem::path nifPath = meshesPath / std::filesystem::path(relativeModelPath);
        expectTrue(std::filesystem::exists(nifPath), "Fargoth actor part NIF exists");
        if (!std::filesystem::exists(nifPath)) {
            continue;
        }

        odai::importer::ImportedNifResult result{};
        std::string error;
        const bool loaded = odai::importer::loadMorrowindActorPartNif(nifPath, result, error);
        if (!loaded) {
            std::cerr << "[imported scene test] actor part failed: "
                      << relativeModelPath << " (" << error << ")\n";
        }
        expectTrue(loaded, "Fargoth actor part NIF loads");
        if (!loaded) {
            continue;
        }
        expectTrue(!result.mesh.vertices.empty(), "Fargoth actor part has vertices");
        expectTrue(!result.mesh.indices.empty(), "Fargoth actor part has indices");
        expectTrue(!result.mesh.parts.empty(), "Fargoth actor part has draw parts");
        expectTrue(
            !result.diffuseTexturePath.empty() || !result.partDiffuseTexturePaths.empty(),
            "Fargoth actor part exposes texture paths");
        for (const odai::importer::ImportedSceneMeshPart& part : result.mesh.parts) {
            expectTrue(
                part.firstIndex + part.indexCount <= result.mesh.indices.size(),
                "Fargoth actor part draw range stays inside index buffer");
        }
    }
}

void testMorrowindFargothSkinnedActorLoadsWhenDataFilesAvailable() {
    const std::filesystem::path dataFilesPath = "C:/GOG Games/Morrowind/Data Files";
    const std::filesystem::path meshesPath = dataFilesPath / "Meshes";
    if (!std::filesystem::exists(meshesPath)) {
        std::cout << "[imported scene test] skipping Fargoth skinned actor load: "
                  << meshesPath.string() << " not found\n";
        return;
    }

    odai::importer::ImportedSkinnedActorAsset actor{};
    std::string error;
    const bool skeletonLoaded =
        odai::importer::loadMorrowindSkinnedActorSkeleton(meshesPath / "base_anim.nif", actor, error);
    if (!skeletonLoaded) {
        std::cerr << "[imported scene test] base_anim.nif failed: " << error << '\n';
    }
    expectTrue(skeletonLoaded, "base_anim.nif loads as a skinned actor skeleton");
    expectTrue(!actor.skeleton.empty(), "base_anim.nif exposes skeleton nodes");
    expectTrue(!actor.nodeAnimations.empty(), "base_anim.nif exposes keyframe animation data");
    expectTrue(!actor.animationClips.empty(), "base_anim.nif exposes animation clip ranges");
    if (!skeletonLoaded) {
        return;
    }

    odai::importer::MorrowindActorCatalog actorCatalog{};
    odai::importer::MorrowindEquipmentCatalog equipmentCatalog{};
    std::vector<odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart> fargothParts;
    if (odai::importer::loadMorrowindActorCatalog(dataFilesPath, actorCatalog) &&
        odai::importer::loadMorrowindEquipmentCatalog(dataFilesPath, equipmentCatalog)) {
        const auto fargothIt = actorCatalog.actorsById.find("fargoth");
        if (fargothIt != actorCatalog.actorsById.end()) {
            fargothParts =
                odai::importer::resolveMorrowindNpcParts(fargothIt->second, equipmentCatalog);
        }
    }
    expectTrue(!fargothParts.empty(), "Fargoth resolves catalog-driven actor parts");
    for (const odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart& part : fargothParts) {
        const std::string& relativeModelPath = part.modelPath;
        const std::size_t firstNewVertex = actor.vertices.size();
        const bool loaded = odai::importer::appendMorrowindSkinnedActorPartNif(
            meshesPath / std::filesystem::path(relativeModelPath),
            odai::importer::toMorrowindActorPartMetadata(part),
            actor,
            error);
        if (!loaded) {
            std::cerr << "[imported scene test] skinned actor part failed: "
                      << relativeModelPath << " (" << error << ")\n";
        }
        expectTrue(loaded, "Fargoth skinned actor part loads");
        if (loaded &&
            (part.slot == "head" || part.slot == "hair")) {
            expectTrue(
                actor.vertices.size() > firstNewVertex,
                "Fargoth head/hair actor part appends vertices");
            float minY = std::numeric_limits<float>::max();
            float maxY = -std::numeric_limits<float>::max();
            for (std::size_t vertexIndex = firstNewVertex; vertexIndex < actor.vertices.size(); ++vertexIndex) {
                minY = std::min(minY, actor.vertices[vertexIndex].position[1]);
                maxY = std::max(maxY, actor.vertices[vertexIndex].position[1]);
            }
            if (maxY <= 95.0f || minY >= 180.0f) {
                std::cerr << "[imported scene test] Fargoth head/hair part "
                          << relativeModelPath << " y-bounds min=" << minY
                          << " max=" << maxY << '\n';
            }
            expectTrue(maxY > 95.0f && minY < 180.0f,
                       "Fargoth head/hair actor part is positioned near the upper body");
        }
    }

    expectTrue(!actor.vertices.empty(), "Fargoth skinned actor has vertices");
    expectTrue(!actor.indices.empty(), "Fargoth skinned actor has indices");
    expectTrue(!actor.draws.empty(), "Fargoth skinned actor has draws");
    expectTrue(actor.boneIndices.size() == actor.vertices.size(), "bone index stream matches vertex count");
    expectTrue(actor.boneWeights.size() == actor.vertices.size(), "bone weight stream matches vertex count");
    expectTrue(actor.weightedVertexCount > 0u, "Fargoth skinned actor has weighted vertices");
    expectTrue(actor.unweightedVertexCount == 0u, "Fargoth skinned actor has no unweighted vertices after actor build");
    if (!actor.skeleton.empty()) {
        std::vector<std::array<float, 16>> worldMatrices(actor.skeleton.size());
        float maxBindPaletteError = 0.0f;
        for (std::size_t nodeIndex = 0; nodeIndex < actor.skeleton.size(); ++nodeIndex) {
            const std::array<float, 16> local = matrixFromFloats(actor.skeleton[nodeIndex].localTransform);
            const std::int32_t parentIndex = actor.skeleton[nodeIndex].parentIndex;
            if (parentIndex >= 0 && static_cast<std::size_t>(parentIndex) < worldMatrices.size()) {
                worldMatrices[nodeIndex] =
                    multiplyMatrices(worldMatrices[static_cast<std::size_t>(parentIndex)], local);
            } else {
                worldMatrices[nodeIndex] = local;
            }
            const std::array<float, 16> inverseBind =
                matrixFromFloats(actor.skeleton[nodeIndex].inverseBindWorldTransform);
            const std::array<float, 16> bindPalette =
                multiplyMatrices(worldMatrices[nodeIndex], inverseBind);
            maxBindPaletteError =
                std::max(maxBindPaletteError, maxAbsDifferenceFromIdentity(bindPalette));
        }
        if (maxBindPaletteError > 0.001f) {
            std::cerr << "[imported scene test] Fargoth bind palette max identity error="
                      << maxBindPaletteError << '\n';
        }
        expectTrue(maxBindPaletteError <= 0.001f, "Fargoth bind pose palette evaluates to identity");
    }
    std::size_t invalidIndexCount = 0u;
    for (const std::uint32_t index : actor.indices) {
        if (index >= actor.vertices.size()) {
            ++invalidIndexCount;
        }
    }
    if (invalidIndexCount != 0u) {
        std::cerr << "[imported scene test] Fargoth invalid actor indices="
                  << invalidIndexCount << '\n';
    }
    expectTrue(invalidIndexCount == 0u, "Fargoth assembled actor indices stay inside vertex buffer");
    if (!actor.vertices.empty()) {
        float minX = actor.vertices.front().position[0];
        float minY = actor.vertices.front().position[1];
        float minZ = actor.vertices.front().position[2];
        float maxX = minX;
        float maxY = minY;
        float maxZ = minZ;
        for (const odai::importer::ImportedScenePackedVertex& vertex : actor.vertices) {
            minX = std::min(minX, vertex.position[0]);
            minY = std::min(minY, vertex.position[1]);
            minZ = std::min(minZ, vertex.position[2]);
            maxX = std::max(maxX, vertex.position[0]);
            maxY = std::max(maxY, vertex.position[1]);
            maxZ = std::max(maxZ, vertex.position[2]);
        }
        const float spanX = maxX - minX;
        const float spanY = maxY - minY;
        const float spanZ = maxZ - minZ;
        if (spanY <= 100.0f || spanX >= 220.0f || spanZ >= 140.0f || spanY <= spanZ) {
            std::cerr << "[imported scene test] Fargoth assembled spans x=" << spanX
                      << " y=" << spanY
                      << " z=" << spanZ << '\n';
        }
        expectTrue(spanY > 80.0f, "Fargoth assembled actor has upright vertical extent");
        expectTrue(spanX < 220.0f && spanZ < 140.0f, "Fargoth assembled actor parts stay near the skeleton");
        expectTrue(spanY > 100.0f && spanY > spanZ, "Fargoth assembled actor has a sane T-pose envelope");
        float maxTriangleEdge = 0.0f;
        auto edgeLength = [&](std::uint32_t a, std::uint32_t b) {
            const odai::importer::ImportedScenePackedVertex& va = actor.vertices[a];
            const odai::importer::ImportedScenePackedVertex& vb = actor.vertices[b];
            const float dx = va.position[0] - vb.position[0];
            const float dy = va.position[1] - vb.position[1];
            const float dz = va.position[2] - vb.position[2];
            return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
        };
        for (std::size_t indexOffset = 0; indexOffset + 2u < actor.indices.size(); indexOffset += 3u) {
            const std::uint32_t i0 = actor.indices[indexOffset + 0u];
            const std::uint32_t i1 = actor.indices[indexOffset + 1u];
            const std::uint32_t i2 = actor.indices[indexOffset + 2u];
            if (i0 >= actor.vertices.size() || i1 >= actor.vertices.size() || i2 >= actor.vertices.size()) {
                continue;
            }
            maxTriangleEdge = std::max(maxTriangleEdge, edgeLength(i0, i1));
            maxTriangleEdge = std::max(maxTriangleEdge, edgeLength(i1, i2));
            maxTriangleEdge = std::max(maxTriangleEdge, edgeLength(i2, i0));
        }
        if (maxTriangleEdge >= 96.0f) {
            std::cerr << "[imported scene test] Fargoth assembled max triangle edge="
                      << maxTriangleEdge << '\n';
        }
        expectTrue(maxTriangleEdge < 96.0f, "Fargoth assembled actor triangles stay local");
    }
    for (std::size_t vertexIndex = 0; vertexIndex < actor.boneWeights.size(); ++vertexIndex) {
        float totalWeight = 0.0f;
        for (std::size_t influenceIndex = 0; influenceIndex < 4u; ++influenceIndex) {
            totalWeight += actor.boneWeights[vertexIndex][influenceIndex];
            expectTrue(
                actor.boneIndices[vertexIndex][influenceIndex] < actor.skeleton.size(),
                "Fargoth bone influence uses a valid skeleton node index");
        }
        if (totalWeight > 0.0f) {
            expectNear(totalWeight, 1.0f, 0.001f, "Fargoth bone weights normalize to one");
        }
    }
}

void testMorrowindHumanoidActorSystemPacksAndDeduplicatesFargothWhenDataFilesAvailable() {
    const std::filesystem::path dataFilesPath = "C:/GOG Games/Morrowind/Data Files";
    if (!std::filesystem::exists(dataFilesPath / "Meshes" / "base_anim.nif")) {
        std::cout << "[imported scene test] skipping humanoid actor system test: "
                  << dataFilesPath.string() << " not found\n";
        return;
    }

    odai::importer::MorrowindActorCatalog actorCatalog{};
    odai::importer::MorrowindEquipmentCatalog equipmentCatalog{};
    expectTrue(
        odai::importer::loadMorrowindActorCatalog(dataFilesPath, actorCatalog),
        "Actor system test loads Morrowind actor catalog");
    expectTrue(
        odai::importer::loadMorrowindEquipmentCatalog(dataFilesPath, equipmentCatalog),
        "Actor system test loads Morrowind equipment catalog");
    const auto fargothIt = actorCatalog.actorsById.find("fargoth");
    expectTrue(fargothIt != actorCatalog.actorsById.end(), "Actor system test finds Fargoth");
    if (fargothIt == actorCatalog.actorsById.end()) {
        return;
    }

    odai::app::MorrowindActorSystem actorSystem{};
    std::unordered_set<std::string> texturePaths;
    const auto textureSlotFn = [&](const std::string& texturePath) -> std::uint32_t {
        if (texturePath.empty()) {
            return std::numeric_limits<std::uint32_t>::max();
        }
        const auto inserted = texturePaths.insert(texturePath);
        if (inserted.second) {
            return static_cast<std::uint32_t>(texturePaths.size() - 1u);
        }
        return static_cast<std::uint32_t>(
            std::distance(texturePaths.begin(), texturePaths.find(texturePath)));
    };

    std::string error;
    const std::uint32_t firstPrototype =
        actorSystem.findOrBuildHumanoidPrototype(
            dataFilesPath,
            fargothIt->second,
            equipmentCatalog,
            textureSlotFn,
            error);
    if (firstPrototype == std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "[imported scene test] actor system failed: " << error << '\n';
    }
    expectTrue(firstPrototype != std::numeric_limits<std::uint32_t>::max(),
               "Actor system builds Fargoth prototype");
    const std::uint32_t secondPrototype =
        actorSystem.findOrBuildHumanoidPrototype(
            dataFilesPath,
            fargothIt->second,
            equipmentCatalog,
            textureSlotFn,
            error);
    expectTrue(secondPrototype == firstPrototype,
               "Actor system deduplicates repeated Fargoth appearance");
    expectTrue(actorSystem.prototypeCount() == 1u, "Actor system stores one Fargoth prototype");
    expectTrue(actorSystem.hasRenderableAsset(), "Actor system has packed renderable actor asset");
    expectTrue(actorSystem.stats().unweightedVertexCount == 0u,
               "Actor system Fargoth prototype has no unweighted vertices");
    expectTrue(actorSystem.walkClip() != nullptr, "Actor system finds a humanoid walk clip");
    expectTrue(actorSystem.idleClip() != nullptr, "Actor system finds a humanoid idle clip");

    const odai::render::ImportedActorRenderAssetData renderData = actorSystem.renderAssetData();
    expectTrue(renderData.vertices.size() == renderData.boneIndices.size(),
               "Actor system packed bone index stream matches vertices");
    expectTrue(renderData.vertices.size() == renderData.boneWeights.size(),
               "Actor system packed bone weight stream matches vertices");
    for (const odai::importer::ImportedScenePackedDraw& draw : renderData.draws) {
        expectTrue(draw.firstIndex < renderData.indices.size(),
                   "Actor system draw starts inside packed index buffer");
        expectTrue(draw.firstIndex + draw.indexCount <= renderData.indices.size(),
                   "Actor system draw remains inside packed index buffer");
    }

    actorSystem.beginFrame(1u, true);
    odai::app::MorrowindActorSystem::FrameInput frameInput{};
    frameInput.prototypeIndex = firstPrototype;
    frameInput.position = {0.0f, 0.0f, 0.0f};
    frameInput.moving = true;
    frameInput.animationTime = 0.25f;
    actorSystem.appendFrameInstance(
        frameInput,
        odai::app::MorrowindActorSystem::PoseMode::Movement,
        0.25f);
    const odai::render::ImportedActorFrameData frameData = actorSystem.frameData();
    expectTrue(frameData.instances.size() == 1u, "Actor system emits one frame instance");
    expectTrue(frameData.bonePalette.size() == actorSystem.skeletonNodeCount(),
               "Actor system emits one contiguous palette per instance");
    expectTrue(!frameData.debugBoneLines.empty(), "Actor system emits debug bone lines");

    actorSystem.beginFrame(1u, false);
    frameInput.moving = false;
    frameInput.animationTime = 0.0f;
    actorSystem.appendFrameInstance(
        frameInput,
        odai::app::MorrowindActorSystem::PoseMode::Movement,
        12.5f);
    const odai::render::ImportedActorFrameData idleFrameData = actorSystem.frameData();
    expectTrue(idleFrameData.instances.size() == 1u, "Actor system emits idle frame instance");
    expectNear(idleFrameData.instances.front().animationTime, 12.5f, 0.0001f,
               "Actor system uses wall-clock time for idle actors");
    expectTrue(idleFrameData.bonePalette.size() == actorSystem.skeletonNodeCount(),
               "Actor system emits one contiguous idle palette per instance");

    odai::app::MorrowindActorSystem strictFailureSystem{};
    std::vector<odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart> invalidParts(1u);
    invalidParts[0].modelPath = "missing/actor_part_that_should_not_exist.nif";
    invalidParts[0].slot = "head";
    invalidParts[0].attachBone = "Head";
    invalidParts[0].meshFilter = "Head";
    const std::uint32_t invalidPrototype =
        strictFailureSystem.findOrBuildHumanoidPrototypeFromParts(
            dataFilesPath,
            "strict_failure_actor",
            invalidParts,
            textureSlotFn,
            error);
    expectTrue(invalidPrototype == std::numeric_limits<std::uint32_t>::max(),
               "Actor system hard-fails missing humanoid actor parts");
    expectTrue(error.find("missing file") != std::string::npos,
               "Actor system strict failure reports missing file");
}

void testMorrowindActorCatalogLoadsNpcAndCreatureRecordsWhenDataFilesAvailable() {
    const std::filesystem::path dataFilesPath = "C:/GOG Games/Morrowind/Data Files";
    const std::filesystem::path masterPath = dataFilesPath / "Morrowind.esm";
    if (!std::filesystem::exists(masterPath)) {
        std::cout << "[imported scene test] skipping actor catalog load: "
                  << masterPath.string() << " not found\n";
        return;
    }

    odai::importer::MorrowindActorCatalog catalog{};
    expectTrue(
        odai::importer::loadMorrowindActorCatalog(dataFilesPath, catalog),
        "Morrowind actor catalog loads from Morrowind.esm");
    expectTrue(!catalog.actorsById.empty(), "Morrowind actor catalog contains actor records");

    const auto fargothIt = catalog.actorsById.find("fargoth");
    expectTrue(fargothIt != catalog.actorsById.end(), "Morrowind actor catalog contains Fargoth");
    if (fargothIt != catalog.actorsById.end()) {
        const odai::importer::MorrowindActorRecord& fargoth = fargothIt->second;
        expectTrue(fargoth.kind == odai::importer::MorrowindActorKind::Npc, "Fargoth is an NPC record");
        expectTrue(fargoth.raceId == "wood elf", "Fargoth keeps his Wood Elf race id");
        expectTrue(!fargoth.headBodyPartId.empty(), "Fargoth exposes a head body-part id");
        expectTrue(!fargoth.hairBodyPartId.empty(), "Fargoth exposes a hair body-part id");
        expectTrue(!fargoth.inventoryItemIds.empty(), "Fargoth exposes inventory item ids");
    }

    bool foundCreatureWithMesh = false;
    for (const auto& [actorId, actor] : catalog.actorsById) {
        (void)actorId;
        if (actor.kind == odai::importer::MorrowindActorKind::Creature && !actor.modelPath.empty()) {
            foundCreatureWithMesh = true;
            break;
        }
    }
    expectTrue(foundCreatureWithMesh, "Morrowind actor catalog contains creature mesh paths");
}

void testMorrowindEquipmentCatalogLoadsKnownClothingWhenDataFilesAvailable() {
    const std::filesystem::path dataFilesPath = "C:/GOG Games/Morrowind/Data Files";
    const std::filesystem::path masterPath = dataFilesPath / "Morrowind.esm";
    if (!std::filesystem::exists(masterPath)) {
        std::cout << "[imported scene test] skipping equipment catalog load: "
                  << masterPath.string() << " not found\n";
        return;
    }

    odai::importer::MorrowindEquipmentCatalog equipmentCatalog{};
    expectTrue(
        odai::importer::loadMorrowindEquipmentCatalog(dataFilesPath, equipmentCatalog),
        "Morrowind equipment catalog loads from Morrowind.esm");
    const auto headIt = equipmentCatalog.modelPathByBodyPartId.find("b_n_wood elf_m_head_02");
    expectTrue(headIt != equipmentCatalog.modelPathByBodyPartId.end(), "Equipment catalog contains Wood Elf male head BODY record");
    if (headIt != equipmentCatalog.modelPathByBodyPartId.end()) {
        expectTrue(!headIt->second.empty(), "Wood Elf male head resolves through BODY MODL");
    }
    const auto shirtIt = equipmentCatalog.bodyPartModelPathsByItemId.find("common_shirt_04");
    expectTrue(shirtIt != equipmentCatalog.bodyPartModelPathsByItemId.end(), "Equipment catalog contains common_shirt_04");
    if (shirtIt != equipmentCatalog.bodyPartModelPathsByItemId.end()) {
        expectTrue(!shirtIt->second.empty(), "common_shirt_04 resolves body part model paths");
        std::unordered_set<std::string> uniquePaths(shirtIt->second.begin(), shirtIt->second.end());
        expectTrue(uniquePaths.size() == shirtIt->second.size(), "common_shirt_04 body part model paths are deduplicated");
    }
}

void testMorrowindGenericHumanoidSkinnedActorLoadsWhenDataFilesAvailable() {
    const std::filesystem::path dataFilesPath = "C:/GOG Games/Morrowind/Data Files";
    const std::filesystem::path meshesPath = dataFilesPath / "Meshes";
    if (!std::filesystem::exists(meshesPath)) {
        std::cout << "[imported scene test] skipping generic humanoid skinned actor load: "
                  << meshesPath.string() << " not found\n";
        return;
    }

    odai::importer::MorrowindActorCatalog actorCatalog{};
    odai::importer::MorrowindEquipmentCatalog equipmentCatalog{};
    expectTrue(
        odai::importer::loadMorrowindActorCatalog(dataFilesPath, actorCatalog),
        "Generic humanoid actor catalog loads");
    expectTrue(
        odai::importer::loadMorrowindEquipmentCatalog(dataFilesPath, equipmentCatalog),
        "Generic humanoid equipment catalog loads");

    const odai::importer::MorrowindActorRecord* selectedActor = nullptr;
    std::vector<odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart> parts;
    for (const auto& [actorId, actor] : actorCatalog.actorsById) {
        if (actor.kind != odai::importer::MorrowindActorKind::Npc || actorId == "fargoth") {
            continue;
        }
        std::vector<odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart> candidateParts =
            odai::importer::resolveMorrowindNpcParts(actor, equipmentCatalog);
        std::size_t existingPartCount = 0;
        for (const auto& part : candidateParts) {
            if (!part.modelPath.empty() && std::filesystem::exists(meshesPath / std::filesystem::path(part.modelPath))) {
                ++existingPartCount;
            }
        }
        if (existingPartCount >= 5u) {
            selectedActor = &actor;
            parts = std::move(candidateParts);
            break;
        }
    }
    expectTrue(selectedActor != nullptr, "Actor catalog contains a non-Fargoth humanoid with resolved parts");
    if (selectedActor == nullptr) {
        return;
    }
    expectTrue(parts.size() >= 5u, "Generic humanoid resolves multiple body/equipment parts");
    std::unordered_set<std::string> uniquePartKeys;
    bool hasHead = false;
    bool hasHair = false;
    bool hasTorso = false;
    bool hasLegsOrFeet = false;
    for (const odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart& part : parts) {
        uniquePartKeys.insert(
            part.modelPath + "|" +
            part.slot + "|" +
            part.side + "|" +
            part.bodyPartId + "|" +
            part.attachBone + "|" +
            part.meshFilter + "|" +
            std::to_string(part.partReferenceType));
        hasHead = hasHead || part.slot == "head";
        hasHair = hasHair || part.slot == "hair";
        hasTorso = hasTorso || part.slot == "torso" || part.slot == "body";
        hasLegsOrFeet = hasLegsOrFeet ||
            part.slot == "groin" ||
            part.slot == "upper_leg" ||
            part.slot == "knee" ||
            part.slot == "ankle" ||
            part.slot == "foot";
    }
    expectTrue(uniquePartKeys.size() == parts.size(), "Generic humanoid resolved actor part keys are deduplicated");
    expectTrue(hasHead, "Generic humanoid resolves a head part");
    expectTrue(hasHair, "Generic humanoid resolves a hair part");
    expectTrue(hasTorso, "Generic humanoid resolves torso/body coverage");
    expectTrue(hasLegsOrFeet, "Generic humanoid resolves lower-body coverage");

    odai::importer::ImportedSkinnedActorAsset actor{};
    std::string error;
    expectTrue(
        odai::importer::loadMorrowindSkinnedActorSkeleton(meshesPath / "base_anim.nif", actor, error),
        "Generic humanoid base skeleton loads");
    for (const odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart& part : parts) {
        const std::filesystem::path nifPath = meshesPath / std::filesystem::path(part.modelPath);
        if (!std::filesystem::exists(nifPath)) {
            continue;
        }
        (void)odai::importer::appendMorrowindSkinnedActorPartNif(
            nifPath,
            odai::importer::toMorrowindActorPartMetadata(part),
            actor,
            error);
    }

    expectTrue(!actor.vertices.empty(), "Generic humanoid skinned actor has vertices");
    expectTrue(!actor.indices.empty(), "Generic humanoid skinned actor has indices");
    expectTrue(!actor.draws.empty(), "Generic humanoid skinned actor has draws");
    expectTrue(actor.boneIndices.size() == actor.vertices.size(), "Generic humanoid bone index stream matches vertex count");
    expectTrue(actor.boneWeights.size() == actor.vertices.size(), "Generic humanoid bone weight stream matches vertex count");
    expectTrue(actor.weightedVertexCount > 0u, "Generic humanoid has weighted vertices");
}

}  // namespace

int main() {
    testImportedSceneSerialization();
    testGpuSceneBuildFromImportedScene();
    testGpuSceneBuildFromInteriorSceneDoesNotCreateTerrain();
    testMorrowindNifSkipsNamedRootCollisionNode();
    testMorrowindSkeletonOnlyNifLoadsAndExtractsTextKeys();
    testMorrowindSkinnedActorPartUsesOpenMwMatrixOrder();
    testImportedSceneCollision();
    testMorrowindFargothActorPartsLoadWhenDataFilesAvailable();
    testMorrowindFargothSkinnedActorLoadsWhenDataFilesAvailable();
    testMorrowindHumanoidActorSystemPacksAndDeduplicatesFargothWhenDataFilesAvailable();
    testMorrowindActorCatalogLoadsNpcAndCreatureRecordsWhenDataFilesAvailable();
    testMorrowindEquipmentCatalogLoadsKnownClothingWhenDataFilesAvailable();
    testMorrowindGenericHumanoidSkinnedActorLoadsWhenDataFilesAvailable();

    if (g_failures != 0) {
        std::cerr << "[imported scene test] " << g_failures << " failures\n";
        return 1;
    }

    std::cout << "[imported scene test] all checks passed\n";
    return 0;
}
