#pragma once

#include "import/imported_scene.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace odai::importer {

struct ImportedNifResult {
    ImportedSceneMesh mesh;
    std::string diffuseTexturePath;
    std::vector<std::string> partDiffuseTexturePaths;
    std::uint32_t skippedAlphaBlendOnlyParts = 0;
    bool alphaTest = false;
};

struct ImportedNifVec3Key {
    float time = 0.0f;
    float value[3] = {};
};

struct ImportedNifQuatKey {
    float time = 0.0f;
    // Stored as x, y, z, w.
    float value[4] = {0.0f, 0.0f, 0.0f, 1.0f};
};

struct ImportedNifFloatKey {
    float time = 0.0f;
    float value = 0.0f;
};

struct ImportedNifNodeAnimation {
    std::uint32_t nodeIndex = 0;
    float startTime = 0.0f;
    float stopTime = 0.0f;
    float frequency = 1.0f;
    float phase = 0.0f;
    std::vector<ImportedNifVec3Key> translationKeys;
    std::vector<ImportedNifQuatKey> rotationKeys;
    std::vector<ImportedNifFloatKey> scaleKeys;
    // Some Morrowind-era NIFs use XYZ rotation key maps instead of quaternion maps.
    std::vector<ImportedNifFloatKey> xRotationKeys;
    std::vector<ImportedNifFloatKey> yRotationKeys;
    std::vector<ImportedNifFloatKey> zRotationKeys;
};

struct ImportedAnimatedNifPart {
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t nodeIndex = 0;
    std::uint32_t textureIndex = 0xffffffffu;
    bool alphaTest = false;
};

struct ImportedAnimatedNifNode {
    std::string name;
    std::int32_t parentIndex = -1;
    float localTransform[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
};

struct ImportedAnimatedNifResult {
    std::string name;
    std::vector<ImportedSceneVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<ImportedAnimatedNifPart> parts;
    std::vector<ImportedAnimatedNifNode> nodes;
    std::vector<ImportedNifNodeAnimation> nodeAnimations;
    std::string diffuseTexturePath;
    std::vector<std::string> partDiffuseTexturePaths;
    std::uint32_t skippedAlphaBlendOnlyParts = 0;
    bool alphaTest = false;
};

bool loadMorrowindStaticNif(
    const std::filesystem::path& nifPath,
    ImportedNifResult& outResult,
    std::string& outError
);

bool loadMorrowindActorPartNif(
    const std::filesystem::path& nifPath,
    ImportedNifResult& outResult,
    std::string& outError
);

bool loadMorrowindAnimatedNif(
    const std::filesystem::path& nifPath,
    ImportedAnimatedNifResult& outResult,
    std::string& outError
);

}  // namespace odai::importer
