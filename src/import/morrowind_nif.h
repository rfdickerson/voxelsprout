#pragma once

#include "import/imported_scene.h"

#include <filesystem>
#include <string>
#include <vector>

namespace odai::importer {

struct ImportedNifResult {
    ImportedSceneMesh mesh;
    std::string diffuseTexturePath;
    std::vector<std::string> partDiffuseTexturePaths;
    bool alphaTest = false;
};

bool loadMorrowindStaticNif(
    const std::filesystem::path& nifPath,
    ImportedNifResult& outResult,
    std::string& outError
);

}  // namespace odai::importer
