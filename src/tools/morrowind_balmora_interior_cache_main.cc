#include "import/imported_scene.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

void printUsage() {
    std::cerr
        << "Usage: odai_balmora_interior_cache <Data Files path> <cache output directory> [--force]\n";
}

std::string lowerPathCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        if (ch == '\\') {
            return '/';
        }
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::uint64_t fnv1a64(std::string_view value) {
    std::uint64_t hash = 14695981039346656037ull;
    for (const char ch : value) {
        hash ^= static_cast<std::uint8_t>(ch);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::filesystem::path cachePathForCell(
    const std::filesystem::path& cacheDirectory,
    std::string_view normalizedCellName
) {
    return cacheDirectory / ("interior_" + std::to_string(fnv1a64(normalizedCellName)) + ".bin");
}

std::filesystem::path doorCachePath(const std::filesystem::path& cacheDirectory) {
    return cacheDirectory / "balmora_doors.bin";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        printUsage();
        return 1;
    }

    const std::filesystem::path dataFilesPath = argv[1];
    const std::filesystem::path cacheDirectory = argv[2];
    const bool force = argc == 4 && std::string_view(argv[3]) == "--force";
    if (argc == 4 && !force) {
        printUsage();
        return 1;
    }

    std::vector<odai::importer::MorrowindDoorReference> doors;
    if (!odai::importer::loadMorrowindBalmoraDoorReferences(dataFilesPath, doors)) {
        std::cerr << "Failed to scan Balmora doors from " << dataFilesPath << "\n";
        const std::string& detail = odai::importer::getImportedSceneLastError();
        if (!detail.empty()) {
            std::cerr << "Reason: " << detail << "\n";
        }
        return 2;
    }

    std::vector<std::string> destinationCells;
    for (const odai::importer::MorrowindDoorReference& door : doors) {
        const std::string normalizedCell = lowerPathCopy(door.destination.destinationCell);
        if (normalizedCell.empty() ||
            std::find(destinationCells.begin(), destinationCells.end(), normalizedCell) != destinationCells.end()) {
            continue;
        }
        destinationCells.push_back(normalizedCell);
    }

    odai::importer::MorrowindDoorCache doorCache{};
    doorCache.exteriorDoors = doors;
    for (const std::string& destinationCell : destinationCells) {
        std::vector<odai::importer::MorrowindDoorReference> interiorDoors;
        if (!odai::importer::loadMorrowindInteriorDoorReferences(dataFilesPath, destinationCell, interiorDoors)) {
            std::cerr << "Warning: failed to scan interior doors for '" << destinationCell
                      << "': " << odai::importer::getImportedSceneLastError() << "\n";
            continue;
        }
        doorCache.interiorDoorsByCell.emplace(destinationCell, std::move(interiorDoors));
    }

    std::error_code mkdirError;
    std::filesystem::create_directories(cacheDirectory, mkdirError);
    if (mkdirError) {
        std::cerr << "Failed to create cache directory " << cacheDirectory
                  << ": " << mkdirError.message() << "\n";
        return 3;
    }

    const auto start = std::chrono::steady_clock::now();
    std::uint32_t cookedCount = 0;
    std::uint32_t skippedCount = 0;
    std::uint32_t failedCount = 0;
    for (const std::string& destinationCell : destinationCells) {
        const std::filesystem::path outputPath = cachePathForCell(cacheDirectory, destinationCell);
        if (!force && std::filesystem::exists(outputPath)) {
            ++skippedCount;
            continue;
        }

        odai::importer::ImportedScene scene{};
        if (!odai::importer::cookMorrowindInteriorCellScene(dataFilesPath, destinationCell, scene)) {
            ++failedCount;
            std::cerr << "Failed to cook interior '" << destinationCell << "': "
                      << odai::importer::getImportedSceneLastError() << "\n";
            continue;
        }
        odai::importer::ImportedScene runtimeScene = scene;
        runtimeScene.meshes.clear();
        runtimeScene.instances.clear();
        runtimeScene.landscapeCells.clear();
        runtimeScene.unresolvedRefs.clear();
        if (!odai::importer::saveImportedScene(runtimeScene, outputPath)) {
            ++failedCount;
            std::cerr << "Failed to save interior cache '" << destinationCell
                      << "' to " << outputPath << ": "
                      << odai::importer::getImportedSceneLastError() << "\n";
            continue;
        }
        ++cookedCount;
        std::cout << "Cached interior '" << destinationCell << "' -> "
                  << outputPath << "\n";
    }

    if (!odai::importer::saveMorrowindDoorCache(doorCache, doorCachePath(cacheDirectory))) {
        ++failedCount;
        std::cerr << "Failed to save Balmora door cache to " << doorCachePath(cacheDirectory)
                  << ": " << odai::importer::getImportedSceneLastError() << "\n";
    }

    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "Balmora interior cache complete\n";
    std::cout << "  Doors: " << doors.size() << "\n";
    std::cout << "  Unique interiors: " << destinationCells.size() << "\n";
    std::cout << "  Cooked: " << cookedCount << "\n";
    std::cout << "  Skipped valid cache: " << skippedCount << "\n";
    std::cout << "  Failed: " << failedCount << "\n";
    std::cout << "  Door manifest: " << doorCachePath(cacheDirectory) << "\n";
    std::cout << "  Cache directory: " << cacheDirectory << "\n";
    std::cout << "  Elapsed ms: " << elapsedMs << "\n";
    return failedCount == 0 ? 0 : 4;
}
