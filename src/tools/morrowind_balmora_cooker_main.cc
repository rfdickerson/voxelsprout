#include "import/imported_scene.h"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

void printUsage() {
    std::cerr
        << "Usage: odai_balmora_cooker <Data Files path> <scene output path> [terrain obj output path]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        printUsage();
        return 1;
    }

    const std::filesystem::path dataFilesPath = argv[1];
    const std::filesystem::path sceneOutputPath = argv[2];
    const std::filesystem::path terrainObjOutputPath = argc >= 4 ? std::filesystem::path(argv[3]) : std::filesystem::path{};

    odai::importer::MorrowindBalmoraCookResult result{};
    if (!odai::importer::cookMorrowindBalmoraScene(dataFilesPath, result)) {
        std::cerr << "Failed to cook Balmora scene from " << dataFilesPath << "\n";
        const std::string& detail = odai::importer::getImportedSceneLastError();
        if (!detail.empty()) {
            std::cerr << "Reason: " << detail << "\n";
        }
        std::cerr << "Expected a valid Morrowind Data Files directory containing Morrowind.esm and Balmora exterior cells.\n";
        return 2;
    }

    if (!odai::importer::saveImportedScene(result.scene, sceneOutputPath)) {
        std::cerr << "Failed to save cooked scene to " << sceneOutputPath << "\n";
        const std::string& detail = odai::importer::getImportedSceneLastError();
        if (!detail.empty()) {
            std::cerr << "Reason: " << detail << "\n";
        }
        return 3;
    }

    if (!terrainObjOutputPath.empty() &&
        !odai::importer::exportImportedSceneTerrainObj(result.scene, terrainObjOutputPath)) {
        std::cerr << "Failed to export terrain OBJ to " << terrainObjOutputPath << "\n";
        const std::string& detail = odai::importer::getImportedSceneLastError();
        if (!detail.empty()) {
            std::cerr << "Reason: " << detail << "\n";
        }
        return 4;
    }

    std::cout << "Cooked Balmora scene\n";
    std::cout << "  Balmora cells: " << result.balmoraCells.size() << "\n";
    std::cout << "  Included cells: " << result.includedCells.size() << "\n";
    std::cout << "  Terrain cells: " << result.scene.landscapeCells.size() << "\n";
    std::cout << "  Unresolved refs: " << result.scene.unresolvedRefs.size() << "\n";
    std::cout << "  Model records: " << result.modelPathById.size() << "\n";
    std::cout << "  Landscape textures: " << result.texturePathByLandscapeIndex.size() << "\n";
    std::cout << "  Scene output: " << sceneOutputPath << "\n";
    if (!terrainObjOutputPath.empty()) {
        std::cout << "  Terrain OBJ: " << terrainObjOutputPath << "\n";
    }
    return 0;
}
