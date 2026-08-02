// Offline cooker: Fallout: New Vegas Data Files (.esm + .bsa + .nif) -> the
// engine's native ImportedScene .bin format, mirroring the scope of this
// project's (Windows-only, not in this repo) Morrowind balmora cooker.
//
// Usage:
//   odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin>
//       --cell <EditorID>
//   odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin>
//       --worldspace <EditorID> <gridX0> <gridZ0> <gridX1> <gridZ1>
//
// Scope cuts made explicitly (not oversights — see the module headers under
// src/import/fnv/ for the reasoning behind each):
//   - No texture extraction. Static meshes render via the engine's existing
//     per-model hashed-color fallback (same path buildImportedScenePackedRenderData
//     already uses for any mesh with no parts); terrain renders via its
//     existing height-based color fallback. Wiring DDS textures through
//     BSShaderTextureSet is a follow-up, not attempted here (see nif_scene.h).
//   - Bethesda's coordinate space is Z-up; this engine (like its Morrowind
//     import path) is Y-up. Positions and rotations are converted with a
//     Y<->Z axis swap below. The exact Euler rotation order Bethesda encodes
//     in REFR DATA is not verified against a real plugin in this environment
//     — double check placed-object orientation against a known in-game
//     reference the first time this runs against real Data Files.

#include "import/fnv/bsa_archive.h"
#include "import/fnv/fallout_records.h"
#include "import/fnv/nif_scene.h"
#include "import/imported_scene.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using odai::importer::ImportedScene;
using odai::importer::ImportedSceneInstance;
using odai::importer::ImportedSceneLandscapeCell;
using odai::importer::ImportedSceneMesh;
using odai::importer::ImportedSceneMeshPart;
using odai::importer::ImportedSceneVertex;

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

// Bethesda's Gamebryo world space is Z-up; this engine (matching its
// Morrowind import path) is Y-up.
Vec3 bethesdaToEngine(float x, float y, float z) {
    return Vec3{x, z, y};
}

struct Mat3 {
    float m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

Mat3 rotationX(float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    Mat3 out{};
    out.m[0] = 1; out.m[1] = 0; out.m[2] = 0;
    out.m[3] = 0; out.m[4] = c; out.m[5] = -s;
    out.m[6] = 0; out.m[7] = s; out.m[8] = c;
    return out;
}
Mat3 rotationY(float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    Mat3 out{};
    out.m[0] = c;  out.m[1] = 0; out.m[2] = s;
    out.m[3] = 0;  out.m[4] = 1; out.m[5] = 0;
    out.m[6] = -s; out.m[7] = 0; out.m[8] = c;
    return out;
}
Mat3 rotationZ(float radians) {
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    Mat3 out{};
    out.m[0] = c; out.m[1] = -s; out.m[2] = 0;
    out.m[3] = s; out.m[4] = c;  out.m[5] = 0;
    out.m[6] = 0; out.m[7] = 0;  out.m[8] = 1;
    return out;
}
Mat3 multiply(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum += a.m[(r * 3) + k] * b.m[(k * 3) + c];
            }
            out.m[(r * 3) + c] = sum;
        }
    }
    return out;
}

// Remaps a Bethesda-space (Z-up) rotation matrix into this engine's Y-up
// space via the similarity transform for a Y<->Z axis swap: engineR[r][c] =
// bethR[perm(r)][perm(c)] with perm = {0:0, 1:2, 2:1}.
Mat3 remapRotationToEngineSpace(const Mat3& beth) {
    constexpr int perm[3] = {0, 2, 1};
    Mat3 out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out.m[(r * 3) + c] = beth.m[(perm[r] * 3) + perm[c]];
        }
    }
    return out;
}

// REFR rotation order: Bethesda applies X, then Y, then Z (R = Rz*Ry*Rx).
// Not verified against a real plugin in this environment — see file header.
Mat3 eulerToMatrixBethesdaOrder(float rx, float ry, float rz) {
    return multiply(rotationZ(rz), multiply(rotationY(ry), rotationX(rx)));
}

void writeTransform(
    ImportedSceneInstance& instance, const Vec3& translation, const Mat3& rotation, float scale
) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            instance.transform[(r * 4) + c] = rotation.m[(r * 3) + c] * scale;
        }
    }
    instance.transform[3] = translation.x;
    instance.transform[7] = translation.y;
    instance.transform[11] = translation.z;
    instance.transform[15] = 1.0f;
}

std::string normalizeModelPath(std::string path) {
    for (char& c : path) {
        if (c == '/') {
            c = '\\';
        }
    }
    return path;
}

// Bethesda paths are backslash-separated regardless of host OS. On POSIX,
// std::filesystem::path::operator/ treats '\' as an ordinary filename
// character (not a separator), so appending a raw backslash-joined string
// via operator/ looks for one literal (and wrong) filename instead of
// walking subdirectories. Split explicitly and join each component so the
// resulting path is correct on every platform.
std::filesystem::path joinBackslashPath(std::filesystem::path base, const std::string& backslashPath) {
    std::string component;
    for (char c : backslashPath) {
        if (c == '\\' || c == '/') {
            if (!component.empty()) {
                base /= component;
                component.clear();
            }
        } else {
            component.push_back(c);
        }
    }
    if (!component.empty()) {
        base /= component;
    }
    return base;
}

// Resolves a model-relative path (as stored in a STAT's MODL, relative to
// Data\Meshes) to raw NIF bytes: loose files under Data Files\Meshes win
// over BSA archives, matching Bethesda's own load-order precedence.
class AssetResolver {
public:
    explicit AssetResolver(std::filesystem::path dataFilesPath) : m_dataFilesPath(std::move(dataFilesPath)) {
        std::error_code listError;
        for (const auto& entry : std::filesystem::directory_iterator(m_dataFilesPath, listError)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bsa") {
                odai::importer::fnv::BsaArchive archive;
                if (archive.open(entry.path())) {
                    m_archives.push_back(std::move(archive));
                } else {
                    std::cerr << "warning: failed to open BSA archive " << entry.path() << "\n";
                }
            }
        }
    }

    bool resolveMesh(const std::string& modelPath, std::vector<std::uint8_t>& outBytes) {
        const std::string normalized = normalizeModelPath(modelPath);
        const std::filesystem::path loosePath = joinBackslashPath(m_dataFilesPath / "meshes", normalized);
        if (std::filesystem::exists(loosePath)) {
            std::ifstream file(loosePath, std::ios::binary | std::ios::ate);
            if (file) {
                const auto size = static_cast<std::size_t>(file.tellg());
                file.seekg(0);
                outBytes.resize(size);
                if (size == 0 || file.read(reinterpret_cast<char*>(outBytes.data()), static_cast<std::streamsize>(size))) {
                    return true;
                }
            }
        }
        const std::string virtualPath = "meshes\\" + normalized;
        for (odai::importer::fnv::BsaArchive& archive : m_archives) {
            const auto* entry = archive.find(virtualPath);
            if (entry != nullptr && archive.extract(*entry, outBytes)) {
                return true;
            }
        }
        return false;
    }

private:
    std::filesystem::path m_dataFilesPath;
    std::vector<odai::importer::fnv::BsaArchive> m_archives;
};

std::string formIdHex(std::uint32_t formId) {
    std::ostringstream out;
    out << std::hex << std::setw(8) << std::setfill('0') << formId;
    return out.str();
}

void printUsage() {
    std::cerr <<
        "Usage:\n"
        "  odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin> --cell <EditorID>\n"
        "  odai_newvegas_cooker <DataFilesPath> <PluginName.esm> <output.bin> --worldspace <EditorID> "
        "<gridX0> <gridZ0> <gridX1> <gridZ1>\n";
}

// Appends one exterior cell's LAND heightmap as a 32x32-quad grid, all
// vertices already in final engine world space (terrain is not instanced —
// see the cooker's file-header note on how buildImportedScenePackedRenderData
// treats scene.meshes[0] when named "terrain").
void appendTerrainCell(
    ImportedSceneMesh& terrainMesh, const odai::importer::fnv::FalloutCellRecord& cell
) {
    if (!cell.land.has_value()) {
        return;
    }
    const odai::importer::fnv::FalloutLandRecord& land = *cell.land;
    using odai::importer::fnv::kExteriorCellSize;
    using odai::importer::fnv::kLandGridSize;
    using odai::importer::fnv::kLandPostSpacing;

    const float cellOriginX = static_cast<float>(cell.gridX) * kExteriorCellSize;
    const float cellOriginZ = static_cast<float>(cell.gridZ) * kExteriorCellSize;

    const std::uint32_t baseVertex = static_cast<std::uint32_t>(terrainMesh.vertices.size());
    for (int row = 0; row < kLandGridSize; ++row) {
        for (int col = 0; col < kLandGridSize; ++col) {
            const int postIndex = (row * kLandGridSize) + col;
            const float bethesdaX = cellOriginX + (static_cast<float>(col) * kLandPostSpacing);
            const float bethesdaY = cellOriginZ + (static_cast<float>(row) * kLandPostSpacing);
            const float bethesdaZ = land.hasHeights ? land.heights[postIndex] : 0.0f;
            const Vec3 world = bethesdaToEngine(bethesdaX, bethesdaY, bethesdaZ);

            ImportedSceneVertex vertex{};
            vertex.position[0] = world.x;
            vertex.position[1] = world.y;
            vertex.position[2] = world.z;
            if (land.hasNormals) {
                const Vec3 normal = bethesdaToEngine(
                    land.normals[(postIndex * 3) + 0], land.normals[(postIndex * 3) + 1], land.normals[(postIndex * 3) + 2]);
                vertex.normal[0] = normal.x;
                vertex.normal[1] = normal.y;
                vertex.normal[2] = normal.z;
            } else {
                vertex.normal[1] = 1.0f;
            }
            terrainMesh.vertices.push_back(vertex);
        }
    }

    const std::uint32_t firstIndex = static_cast<std::uint32_t>(terrainMesh.indices.size());
    for (int row = 0; row < kLandGridSize - 1; ++row) {
        for (int col = 0; col < kLandGridSize - 1; ++col) {
            const std::uint32_t i00 = baseVertex + static_cast<std::uint32_t>((row * kLandGridSize) + col);
            const std::uint32_t i10 = i00 + 1u;
            const std::uint32_t i01 = i00 + static_cast<std::uint32_t>(kLandGridSize);
            const std::uint32_t i11 = i01 + 1u;
            terrainMesh.indices.push_back(i00);
            terrainMesh.indices.push_back(i01);
            terrainMesh.indices.push_back(i11);
            terrainMesh.indices.push_back(i00);
            terrainMesh.indices.push_back(i11);
            terrainMesh.indices.push_back(i10);
        }
    }
    const std::uint32_t indexCount = static_cast<std::uint32_t>(terrainMesh.indices.size()) - firstIndex;
    // Texture index 0 with an empty scene.textures[] safely degrades to the
    // height-based vertex-color fallback at upload time — see file header.
    terrainMesh.parts.push_back(ImportedSceneMeshPart{firstIndex, indexCount, 0u, false});
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        printUsage();
        return 1;
    }
    const std::filesystem::path dataFilesPath = argv[1];
    const std::filesystem::path pluginName = argv[2];
    const std::filesystem::path outputPath = argv[3];
    const std::string mode = argv[4];

    std::optional<std::string> targetCellEditorId;
    std::optional<std::string> targetWorldspaceEditorId;
    std::int32_t gridX0 = 0, gridZ0 = 0, gridX1 = 0, gridZ1 = 0;

    if (mode == "--cell" && argc >= 6) {
        targetCellEditorId = argv[5];
    } else if (mode == "--worldspace" && argc >= 10) {
        targetWorldspaceEditorId = argv[5];
        gridX0 = std::atoi(argv[6]);
        gridZ0 = std::atoi(argv[7]);
        gridX1 = std::atoi(argv[8]);
        gridZ1 = std::atoi(argv[9]);
        if (gridX0 > gridX1) std::swap(gridX0, gridX1);
        if (gridZ0 > gridZ1) std::swap(gridZ0, gridZ1);
    } else {
        printUsage();
        return 1;
    }

    const std::filesystem::path esmPath = dataFilesPath / pluginName;
    odai::importer::fnv::FalloutSceneData scene;
    std::string extractError;
    if (!odai::importer::fnv::extractFalloutScene(esmPath, scene, extractError)) {
        std::cerr << "Failed to extract plugin data: " << extractError << "\n";
        return 1;
    }
    std::cout << "Extracted " << scene.statics.size() << " statics, " << scene.worldspaces.size()
              << " worldspaces, " << scene.cells.size() << " cells from " << esmPath << "\n";

    std::vector<const odai::importer::fnv::FalloutCellRecord*> selectedCells;
    if (targetCellEditorId.has_value()) {
        for (const auto& cell : scene.cells) {
            if (cell.isInterior && cell.editorId == *targetCellEditorId) {
                selectedCells.push_back(&cell);
            }
        }
    } else {
        std::uint32_t worldspaceFormId = 0;
        bool foundWorldspace = false;
        for (const auto& ws : scene.worldspaces) {
            if (ws.editorId == *targetWorldspaceEditorId) {
                worldspaceFormId = ws.formId;
                foundWorldspace = true;
                break;
            }
        }
        if (!foundWorldspace) {
            std::cerr << "Worldspace not found: " << *targetWorldspaceEditorId << "\n";
            return 1;
        }
        for (const auto& cell : scene.cells) {
            if (!cell.isInterior && cell.worldspaceFormId == worldspaceFormId && cell.hasGridCoords &&
                cell.gridX >= gridX0 && cell.gridX <= gridX1 && cell.gridZ >= gridZ0 && cell.gridZ <= gridZ1) {
                selectedCells.push_back(&cell);
            }
        }
    }
    if (selectedCells.empty()) {
        std::cerr << "No matching cells found for the requested selection.\n";
        return 1;
    }
    std::cout << "Cooking " << selectedCells.size() << " cell(s).\n";

    std::unordered_map<std::uint32_t, const odai::importer::fnv::FalloutStaticRecord*> staticsByFormId;
    for (const auto& stat : scene.statics) {
        staticsByFormId[stat.formId] = &stat;
    }

    const bool anyInterior = selectedCells.front()->isInterior;

    ImportedScene outScene;
    outScene.sourceTag = anyInterior ? "fnv_interior" : "fnv_exterior";

    if (!anyInterior) {
        ImportedSceneMesh terrainMesh;
        terrainMesh.name = "terrain";
        for (const auto* cell : selectedCells) {
            appendTerrainCell(terrainMesh, *cell);
        }
        outScene.sourceLandscapeCellCount = static_cast<std::uint32_t>(terrainMesh.parts.size());
        outScene.meshes.push_back(std::move(terrainMesh));
    }

    AssetResolver resolver(dataFilesPath);
    std::unordered_map<std::uint32_t, std::uint32_t> meshIndexByStaticFormId;  // formId -> outScene.meshes index
    std::unordered_set<std::uint32_t> failedStatics;
    std::uint32_t skippedGeometryShapes = 0;
    std::uint32_t placedInstances = 0;

    for (const auto* cell : selectedCells) {
        for (const auto& ref : cell->references) {
            if (failedStatics.count(ref.baseFormId) != 0u) {
                continue;
            }
            auto meshIt = meshIndexByStaticFormId.find(ref.baseFormId);
            if (meshIt == meshIndexByStaticFormId.end()) {
                const auto statIt = staticsByFormId.find(ref.baseFormId);
                if (statIt == staticsByFormId.end() || statIt->second->modelPath.empty()) {
                    failedStatics.insert(ref.baseFormId);
                    continue;
                }
                std::vector<std::uint8_t> nifBytes;
                if (!resolver.resolveMesh(statIt->second->modelPath, nifBytes)) {
                    std::cerr << "warning: could not resolve mesh " << statIt->second->modelPath << "\n";
                    failedStatics.insert(ref.baseFormId);
                    continue;
                }
                odai::importer::fnv::NifModel nifModel;
                std::string nifError;
                if (!odai::importer::fnv::parseNifStaticMesh(nifBytes, nifModel, nifError) || nifModel.shapes.empty()) {
                    std::cerr << "warning: failed to parse NIF " << statIt->second->modelPath << ": " << nifError << "\n";
                    failedStatics.insert(ref.baseFormId);
                    continue;
                }
                skippedGeometryShapes += nifModel.skippedShapeCount;

                ImportedSceneMesh mesh;
                mesh.name = statIt->second->editorId.empty() ? statIt->second->modelPath : statIt->second->editorId;
                for (const auto& shape : nifModel.shapes) {
                    const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
                    for (std::size_t v = 0; v * 3u < shape.positions.size(); ++v) {
                        ImportedSceneVertex vertex{};
                        // NIF model-space is already Y-up in this engine's convention
                        // (Gamebryo meshes are authored per-model, not world-space
                        // Z-up) — no axis swap here, only for placement/terrain data.
                        vertex.position[0] = shape.positions[v * 3u];
                        vertex.position[1] = shape.positions[(v * 3u) + 1];
                        vertex.position[2] = shape.positions[(v * 3u) + 2];
                        if (!shape.normals.empty()) {
                            vertex.normal[0] = shape.normals[v * 3u];
                            vertex.normal[1] = shape.normals[(v * 3u) + 1];
                            vertex.normal[2] = shape.normals[(v * 3u) + 2];
                        }
                        mesh.vertices.push_back(vertex);
                    }
                    for (const std::uint32_t index : shape.triangleIndices) {
                        mesh.indices.push_back(baseVertex + index);
                    }
                }
                if (mesh.vertices.empty()) {
                    failedStatics.insert(ref.baseFormId);
                    continue;
                }
                const std::uint32_t meshIndex = static_cast<std::uint32_t>(outScene.meshes.size());
                outScene.meshes.push_back(std::move(mesh));
                meshIt = meshIndexByStaticFormId.emplace(ref.baseFormId, meshIndex).first;
            }

            ImportedSceneInstance instance;
            instance.meshIndex = meshIt->second;
            instance.sourceId = "refr_" + formIdHex(ref.formId);
            instance.modelPath = staticsByFormId.at(ref.baseFormId)->modelPath;
            const Vec3 worldPos = bethesdaToEngine(ref.position[0], ref.position[1], ref.position[2]);
            const Mat3 bethRotation = eulerToMatrixBethesdaOrder(
                ref.rotationRadians[0], ref.rotationRadians[1], ref.rotationRadians[2]);
            const Mat3 engineRotation = remapRotationToEngineSpace(bethRotation);
            writeTransform(instance, worldPos, engineRotation, ref.scale);
            outScene.instances.push_back(instance);
            ++placedInstances;
        }
    }

    if (skippedGeometryShapes != 0u) {
        std::cout << "warning: " << skippedGeometryShapes
                  << " mesh shape(s) had unreadable NiTriShapeData and were dropped (see nif_scene.h).\n";
    }
    std::cout << "Placed " << placedInstances << " instance(s) across " << outScene.meshes.size() << " mesh(es).\n";

    odai::importer::buildImportedScenePackedRenderData(outScene);
    odai::importer::buildImportedScenePageRanges(outScene);

    if (!odai::importer::saveImportedScene(outScene, outputPath)) {
        std::cerr << "Failed to save output scene: " << odai::importer::getImportedSceneLastError() << "\n";
        return 1;
    }
    std::cout << "Wrote " << outputPath << "\n";
    return 0;
}
