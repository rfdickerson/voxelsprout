#pragma once

#include "import/imported_scene.h"

#include <filesystem>
#include <string>

namespace voxelsprout::importer {

struct ImportedNifResult {
    ImportedSceneMesh mesh;
    std::string diffuseTexturePath;
    bool alphaTest = false;
};

bool loadMorrowindStaticNif(
    const std::filesystem::path& nifPath,
    ImportedNifResult& outResult,
    std::string& outError
);

}  // namespace voxelsprout::importer
