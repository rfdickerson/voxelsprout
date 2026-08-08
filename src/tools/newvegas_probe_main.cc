// Real-data harness for the Fallout: New Vegas readers.
//
// odai_fnv_import_tests is pinned to synthetic fixtures only (see CLAUDE.md),
// which is the right call for CI but means nothing in the test suite can catch
// a wrong assumption about the retail file format — and several such bugs did
// ship. This tool is the other half: point it at a real Data directory and it
// reports what the readers actually manage to read.
//
// It is deliberately not registered with ctest. It needs game data that only
// exists on a machine with the game installed.
//
//   odai_newvegas_probe <DataFilesPath> --archives
//   odai_newvegas_probe <DataFilesPath> --nifs [limit]
//   odai_newvegas_probe <DataFilesPath> --nif <virtualPath>
//   odai_newvegas_probe <DataFilesPath> --plugin <Plugin.esm>

#include "import/fnv/asset_source.h"
#include "import/fnv/bsa_archive.h"
#include "import/fnv/cell_builder.h"
#include "import/fnv/esm_reader.h"
#include "import/fnv/fallout_records.h"
#include "import/fnv/nif_scene.h"
#include "import/imported_scene.h"

#include <algorithm>
#include <array>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using odai::importer::fnv::BsaArchive;
using odai::importer::fnv::BsaFileEntry;

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::vector<std::filesystem::path> listArchives(const std::filesystem::path& dataPath) {
    std::vector<std::filesystem::path> out;
    std::error_code listError;
    for (const auto& entry : std::filesystem::directory_iterator(dataPath, listError)) {
        if (entry.is_regular_file() && toLowerAscii(entry.path().extension().string()) == ".bsa") {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Lists archive entries whose virtual path contains `needle`. Exists because
// every format in this importer that was reasoned from documentation instead of
// measured has been wrong at least once, and the distant-LOD asset layout is the
// next thing about to be built against. Measure it, then write the code.
int findArchiveEntries(
    const std::filesystem::path& dataPath, const std::string& needle, std::size_t limit) {
    const std::string loweredNeedle = toLowerAscii(needle);
    std::size_t matches = 0;
    std::size_t shown = 0;
    std::map<std::string, std::size_t> matchesByArchive;
    std::map<std::string, std::size_t> matchesByExtension;

    for (const auto& archivePath : listArchives(dataPath)) {
        BsaArchive archive;
        if (!archive.open(archivePath)) {
            continue;
        }
        for (const BsaFileEntry& entry : archive.files()) {
            if (toLowerAscii(entry.virtualPath).find(loweredNeedle) == std::string::npos) {
                continue;
            }
            ++matches;
            ++matchesByArchive[archivePath.filename().string()];
            const std::size_t dot = entry.virtualPath.rfind('.');
            matchesByExtension[dot == std::string::npos
                                   ? "(none)"
                                   : toLowerAscii(entry.virtualPath.substr(dot))]++;
            if (shown < limit) {
                std::cout << "  " << entry.virtualPath << "  (" << entry.sizeOnDisk << " bytes"
                          << (entry.compressed ? ", compressed" : "") << ")\n";
                ++shown;
            }
        }
    }
    std::cout << matches << " entries matching \"" << needle << "\"";
    if (matches > shown) {
        std::cout << " (showing first " << shown << ")";
    }
    std::cout << "\n";
    for (const auto& [archive, count] : matchesByArchive) {
        std::cout << "  " << archive << ": " << count << "\n";
    }
    std::cout << "  by extension:";
    for (const auto& [extension, count] : matchesByExtension) {
        std::cout << " " << extension << "=" << count;
    }
    std::cout << "\n";
    return matches == 0 ? 1 : 0;
}

int probeArchives(const std::filesystem::path& dataPath) {
    std::size_t totalFiles = 0;
    std::size_t failures = 0;
    for (const auto& archivePath : listArchives(dataPath)) {
        BsaArchive archive;
        if (!archive.open(archivePath)) {
            std::cout << "  FAILED  " << archivePath.filename().string() << ": " << archive.lastError() << "\n";
            ++failures;
            continue;
        }
        // Extract the first and last entries as a cheap end-to-end check that
        // the data-block layout (embedded names, compression) is right.
        std::string extractNote = "ok";
        std::vector<std::uint8_t> bytes;
        for (const std::size_t index : {std::size_t{0}, archive.files().size() - 1u}) {
            if (archive.files().empty()) {
                break;
            }
            if (!archive.extract(archive.files()[index], bytes)) {
                extractNote = "EXTRACT FAILED: " + archive.lastError();
                ++failures;
                break;
            }
        }
        std::cout << "  " << archivePath.filename().string() << "\n"
                  << "      files " << archive.files().size() << ", contentFlags 0x" << std::hex
                  << archive.contentFlags() << std::dec << ", first \""
                  << (archive.files().empty() ? std::string("<none>") : archive.files().front().virtualPath)
                  << "\", extract " << extractNote << "\n";
        totalFiles += archive.files().size();
    }
    std::cout << "Archives: " << totalFiles << " files indexed, " << failures << " failure(s).\n";
    return failures == 0 ? 0 : 1;
}

// Opens the mesh archives and returns every .nif entry paired with its archive.
struct MeshIndex {
    std::vector<BsaArchive> archives;
    std::vector<std::pair<std::size_t, const BsaFileEntry*>> nifs;
};

MeshIndex buildMeshIndex(const std::filesystem::path& dataPath) {
    MeshIndex index;
    for (const auto& archivePath : listArchives(dataPath)) {
        std::uint32_t contentFlags = 0;
        if (odai::importer::fnv::peekBsaContentFlags(archivePath, contentFlags) &&
            (contentFlags & odai::importer::fnv::kBsaContentMeshes) == 0u) {
            continue;
        }
        BsaArchive archive;
        if (archive.open(archivePath)) {
            index.archives.push_back(std::move(archive));
        }
    }
    for (std::size_t a = 0; a < index.archives.size(); ++a) {
        for (const BsaFileEntry& entry : index.archives[a].files()) {
            if (entry.virtualPath.size() > 4u &&
                toLowerAscii(entry.virtualPath.substr(entry.virtualPath.size() - 4u)) == ".nif") {
                index.nifs.emplace_back(a, &entry);
            }
        }
    }
    return index;
}

int probeNifs(const std::filesystem::path& dataPath, std::size_t limit) {
    MeshIndex index = buildMeshIndex(dataPath);
    std::cout << "Found " << index.nifs.size() << " .nif entries across " << index.archives.size() << " archive(s).\n";

    std::size_t examined = 0;
    std::size_t extractFailures = 0;
    std::size_t parseFailures = 0;
    std::size_t emptyModels = 0;
    std::size_t withGeometry = 0;
    std::size_t withUvs = 0;
    std::size_t shapesWithDiffuse = 0;
    std::size_t shapesWithAlphaTest = 0;
    std::size_t totalShapes = 0;
    std::size_t totalTriangles = 0;
    std::size_t totalSkippedShapes = 0;
    std::map<std::string, std::size_t> parseErrors;

    std::vector<std::uint8_t> bytes;
    for (const auto& [archiveIndex, entry] : index.nifs) {
        if (examined >= limit) {
            break;
        }
        ++examined;
        if (!index.archives[archiveIndex].extract(*entry, bytes)) {
            ++extractFailures;
            continue;
        }
        odai::importer::fnv::NifModel model;
        std::string error;
        if (!odai::importer::fnv::parseNifStaticMesh(bytes, model, error)) {
            ++parseFailures;
            ++parseErrors[error];
            continue;
        }
        totalSkippedShapes += model.skippedShapeCount;
        if (model.shapes.empty()) {
            ++emptyModels;
            continue;
        }
        ++withGeometry;
        bool anyUvs = false;
        for (const odai::importer::fnv::NifShape& shape : model.shapes) {
            ++totalShapes;
            totalTriangles += shape.triangleIndices.size() / 3u;
            anyUvs = anyUvs || !shape.uvs.empty();
            shapesWithDiffuse += shape.diffuseTexturePath.empty() ? 0u : 1u;
            shapesWithAlphaTest += shape.alphaTest ? 1u : 0u;
        }
        if (anyUvs) {
            ++withUvs;
        }
    }

    std::cout << "Examined " << examined << " NIF(s):\n"
              << "  with geometry     " << withGeometry << "\n"
              << "  with UVs          " << withUvs << "\n"
              << "  empty (no shapes) " << emptyModels << "\n"
              << "  parse failures    " << parseFailures << "\n"
              << "  extract failures  " << extractFailures << "\n"
              << "  shapes            " << totalShapes << "\n"
              << "  triangles         " << totalTriangles << "\n"
              << "  skipped shapes    " << totalSkippedShapes << "\n"
              << "  shapes w/ diffuse " << shapesWithDiffuse << "\n"
              << "  shapes w/ alphaTest " << shapesWithAlphaTest << "\n";
    for (const auto& [message, count] : parseErrors) {
        std::cout << "  [" << count << "x] " << message << "\n";
    }
    return 0;
}

// Raw block inventory: what types the file actually contains, and the header
// string table. Exists because three separate attempts to locate a texture path
// by walking to a computed field offset all missed -- reading the file's own
// declared contents settles in one shot what guessing does not.
int dumpNifBlocks(const std::filesystem::path& dataPath, const std::string& virtualPath) {
    MeshIndex index = buildMeshIndex(dataPath);
    std::vector<std::uint8_t> bytes;
    for (const auto& [archiveIndex, entry] : index.nifs) {
        if (toLowerAscii(entry->virtualPath) != toLowerAscii(virtualPath)) {
            continue;
        }
        if (!index.archives[archiveIndex].extract(*entry, bytes)) {
            std::cout << "Extract failed.\n";
            return 1;
        }
        odai::importer::fnv::NifBlockSummary summary;
        std::string error;
        if (!odai::importer::fnv::parseNifBlockSummary(bytes, summary, error)) {
            std::cout << "Header parse FAILED: " << error << "\n";
            return 1;
        }
        std::cout << "blocks " << summary.blockTypeNames.size()
                  << ", strings " << summary.strings.size() << "\n";
        std::map<std::string, std::size_t> typeCounts;
        for (const std::string& typeName : summary.blockTypeNames) {
            ++typeCounts[typeName];
        }
        std::cout << "block types:\n";
        for (const auto& [typeName, count] : typeCounts) {
            std::cout << "  " << count << "x " << typeName << "\n";
        }
        std::cout << "block list:\n";
        for (std::size_t i = 0; i < summary.blockTypeNames.size(); ++i) {
            std::cout << "  [" << i << "] " << summary.blockTypeNames[i]
                      << " (" << summary.blockSizes[i] << " bytes)\n";
        }
        // Raw words of small blocks: the last resort when a field's location is
        // in question, which is exactly the situation these dumps exist for.
        std::cout << "small block words (index: value ...):\n";
        for (std::size_t i = 0; i < summary.blockTypeNames.size(); ++i) {
            if (summary.blockSizes[i] > 128u) {
                continue;
            }
            std::cout << "  [" << i << "] " << summary.blockTypeNames[i] << ":";
            const std::size_t wordCount = summary.blockSizes[i] / 4u;
            for (std::size_t w = 0; w < wordCount; ++w) {
                std::int32_t value = 0;
                std::memcpy(&value, bytes.data() + summary.blockStarts[i] + (w * 4u), 4u);
                std::cout << " " << value;
            }
            std::cout << "\n";
        }
        std::cout << "string table:\n";
        for (std::size_t i = 0; i < summary.strings.size(); ++i) {
            std::cout << "  [" << i << "] \"" << summary.strings[i] << "\"\n";
        }
        return 0;
    }
    std::cout << "Not found in any mesh archive: " << virtualPath << "\n";
    return 1;
}

int probeSingleNif(const std::filesystem::path& dataPath, const std::string& virtualPath) {
    MeshIndex index = buildMeshIndex(dataPath);
    for (std::size_t a = 0; a < index.archives.size(); ++a) {
        const BsaFileEntry* entry = index.archives[a].find(virtualPath);
        if (entry == nullptr) {
            continue;
        }
        std::vector<std::uint8_t> bytes;
        if (!index.archives[a].extract(*entry, bytes)) {
            std::cout << "extract failed: " << index.archives[a].lastError() << "\n";
            return 1;
        }
        std::cout << "Extracted " << bytes.size() << " bytes.\n";
        odai::importer::fnv::NifModel model;
        std::string error;
        const bool ok = odai::importer::fnv::parseNifStaticMesh(bytes, model, error);
        std::cout << "parse " << (ok ? "ok" : "FAILED") << (error.empty() ? "" : (": " + error)) << "\n"
                  << "shapes " << model.shapes.size() << ", skipped " << model.skippedShapeCount << "\n";
        for (const odai::importer::fnv::NifShape& shape : model.shapes) {
            std::cout << "  \"" << shape.name << "\" verts " << (shape.positions.size() / 3u) << ", tris "
                      << (shape.triangleIndices.size() / 3u) << ", uvs " << (shape.uvs.size() / 2u)
                      << ", alphaTest=" << (shape.alphaTest ? "yes" : "no")
                      << ", alphaBlend=" << (shape.alphaBlend ? "yes" : "no")
                      << ", diffuse=\"" << shape.diffuseTexturePath << "\"\n";
            if (!shape.uvs.empty()) {
                // Two ranges, and the difference between them is the point: a
                // vertex no triangle references never reaches a fragment, so
                // junk in one costs nothing. Only the referenced range can
                // affect what is drawn.
                float allMin[2] = {shape.uvs[0], shape.uvs[1]};
                float allMax[2] = {shape.uvs[0], shape.uvs[1]};
                for (std::size_t i = 0; i + 1u < shape.uvs.size(); i += 2u) {
                    for (int c = 0; c < 2; ++c) {
                        allMin[c] = std::min(allMin[c], shape.uvs[i + static_cast<std::size_t>(c)]);
                        allMax[c] = std::max(allMax[c], shape.uvs[i + static_cast<std::size_t>(c)]);
                    }
                }
                bool anyReferenced = false;
                float refMin[2] = {0.0f, 0.0f};
                float refMax[2] = {0.0f, 0.0f};
                for (const std::uint32_t vertexIndex : shape.triangleIndices) {
                    const std::size_t base = static_cast<std::size_t>(vertexIndex) * 2u;
                    if (base + 1u >= shape.uvs.size()) {
                        continue;
                    }
                    for (int c = 0; c < 2; ++c) {
                        const float value = shape.uvs[base + static_cast<std::size_t>(c)];
                        if (!anyReferenced) {
                            refMin[c] = value;
                            refMax[c] = value;
                        } else {
                            refMin[c] = std::min(refMin[c], value);
                            refMax[c] = std::max(refMax[c], value);
                        }
                    }
                    anyReferenced = true;
                }
                std::cout << "      uv all      u[" << allMin[0] << ", " << allMax[0]
                          << "] v[" << allMin[1] << ", " << allMax[1] << "]\n";
                if (anyReferenced) {
                    std::cout << "      uv referenced u[" << refMin[0] << ", " << refMax[0]
                              << "] v[" << refMin[1] << ", " << refMax[1] << "]\n";
                }
            }
            // Bounds per shape: an untextured shape that occupies the same space
            // as a textured one is a duplicate (a shadow or proxy mesh being
            // rendered), which is a different bug from a shape that is simply a
            // separate untextured part of the model.
            if (shape.positions.size() >= 3u) {
                float boundsMin[3] = {shape.positions[0], shape.positions[1], shape.positions[2]};
                float boundsMax[3] = {shape.positions[0], shape.positions[1], shape.positions[2]};
                for (std::size_t v = 0; v * 3u + 2u < shape.positions.size(); ++v) {
                    for (int c = 0; c < 3; ++c) {
                        const float value = shape.positions[(v * 3u) + static_cast<std::size_t>(c)];
                        boundsMin[c] = std::min(boundsMin[c], value);
                        boundsMax[c] = std::max(boundsMax[c], value);
                    }
                }
                std::cout << "      bounds min(" << boundsMin[0] << ", " << boundsMin[1] << ", " << boundsMin[2]
                          << ") max(" << boundsMax[0] << ", " << boundsMax[1] << ", " << boundsMax[2] << ")\n";
            }
        }
        if (!model.unresolvedPropertyTypes.empty()) {
            std::cout << "shape property types this parser got no texture set from:\n";
            for (const std::string& typeName : model.unresolvedPropertyTypes) {
                std::cout << "  " << typeName << "\n";
            }
        }
        return ok ? 0 : 1;
    }
    std::cout << "Not found in any mesh archive: " << virtualPath << "\n";
    return 1;
}

// Interior cell EditorIDs, which is what odai_newvegas_cooker --cell takes.
// Without this the flag was unusable: the only way to learn a valid ID was
// FNVEdit or the GECK, even though the extractor already parses every one of
// them. An optional substring filters the list, since a retail plugin has
// hundreds.
// Verifies the streaming index against the full extractor: build the offset
// index, then materialize a sample of cells through extractFalloutCellAt and
// require every field to match what a whole-file pass produces for the same
// cell. This is the gate on Phase 1 -- an index that is subtly wrong (an offset
// off by a group header, a cell attributed to the wrong worldspace) produces
// plausible-looking geometry in the wrong place rather than an error.
int probeCellIndex(
    const std::filesystem::path& esmPath, const std::string& worldspaceFilter, std::size_t sampleCount) {
    using namespace odai::importer::fnv;

    std::string error;
    const auto indexStart = std::chrono::steady_clock::now();
    FalloutCellIndex index;
    if (!buildFalloutCellIndex(esmPath, index, error)) {
        std::cout << "Index build FAILED: " << error << "\n";
        return 1;
    }
    const double indexMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - indexStart).count();

    std::size_t exteriorCount = 0;
    for (const FalloutCellIndexEntry& entry : index.cells) {
        if (!entry.isInterior && entry.hasGridCoords) {
            ++exteriorCount;
        }
    }
    const std::size_t indexBytes = index.cells.size() * sizeof(FalloutCellIndexEntry);
    std::cout << "cell index: " << index.cells.size() << " cells (" << exteriorCount
              << " exterior with grid coords), " << index.worldspaces.size() << " worldspaces, "
              << index.cellIndexByReferenceFormId.size() << " references mapped\n";
    std::cout << "  built in " << indexMs << " ms, entries occupy "
              << (indexBytes / 1024u) << " KB\n";

    // Per-worldspace cell counts: this is what sets the streaming budget, and
    // it is worth measuring rather than quoting -- the resident set has to be
    // sized against the worldspace actually being walked.
    std::map<std::uint32_t, std::size_t> cellsByWorldspace;
    for (const FalloutCellIndexEntry& entry : index.cells) {
        if (!entry.isInterior && entry.hasGridCoords) {
            ++cellsByWorldspace[entry.worldspaceFormId];
        }
    }
    for (const FalloutWorldspaceRecord& world : index.worldspaces) {
        const auto found = cellsByWorldspace.find(world.formId);
        if (found != cellsByWorldspace.end() && found->second > 0) {
            std::cout << "    " << world.editorId << ": " << found->second << " exterior cells\n";
        }
    }

    // Resolve the worldspace to sample from, by editor ID substring.
    std::uint32_t worldspaceFormId = 0;
    if (!worldspaceFilter.empty()) {
        const std::string lowered = toLowerAscii(worldspaceFilter);
        for (const FalloutWorldspaceRecord& world : index.worldspaces) {
            if (toLowerAscii(world.editorId).find(lowered) != std::string::npos) {
                worldspaceFormId = world.formId;
                std::cout << "  sampling worldspace " << world.editorId << "\n";
                break;
            }
        }
        if (worldspaceFormId == 0) {
            std::cout << "  no worldspace matching \"" << worldspaceFilter << "\"\n";
            return 1;
        }
    }

    // Pick the sample cells, preferring ones that actually have contents.
    std::vector<std::size_t> sampleIndices;
    for (std::size_t i = 0; i < index.cells.size() && sampleIndices.size() < sampleCount; ++i) {
        const FalloutCellIndexEntry& entry = index.cells[i];
        if (worldspaceFormId != 0 && entry.worldspaceFormId != worldspaceFormId) {
            continue;
        }
        if (entry.childrenGroupSize == 0u) {
            continue;
        }
        sampleIndices.push_back(i);
    }
    if (sampleIndices.empty()) {
        std::cout << "  no cells with contents to sample\n";
        return 1;
    }

    // Full-pass reference: materialize exactly the sampled cells.
    std::vector<std::uint32_t> wantedFormIds;
    wantedFormIds.reserve(sampleIndices.size());
    for (const std::size_t i : sampleIndices) {
        wantedFormIds.push_back(index.cells[i].cellFormId);
    }
    FalloutExtractFilter referenceFilter{};
    referenceFilter.wantCellContents = [&](const FalloutCellRecord& cell) {
        return std::find(wantedFormIds.begin(), wantedFormIds.end(), cell.formId) !=
               wantedFormIds.end();
    };
    FalloutSceneData reference;
    if (!extractFalloutScene(esmPath, referenceFilter, reference, error)) {
        std::cout << "Reference extract FAILED: " << error << "\n";
        return 1;
    }

    EsmReader reader;
    if (!reader.open(esmPath)) {
        std::cout << "Reader open FAILED: " << reader.lastError() << "\n";
        return 1;
    }

    std::size_t mismatches = 0;
    std::size_t compared = 0;
    double extractMsTotal = 0.0;
    for (const std::size_t i : sampleIndices) {
        const FalloutCellIndexEntry& entry = index.cells[i];
        const FalloutCellRecord* expected = nullptr;
        for (const FalloutCellRecord& cell : reference.cells) {
            if (cell.formId == entry.cellFormId) {
                expected = &cell;
                break;
            }
        }
        if (expected == nullptr) {
            std::cout << "  cell " << std::hex << entry.cellFormId << std::dec
                      << ": MISSING from the reference pass\n";
            ++mismatches;
            continue;
        }

        const auto extractStart = std::chrono::steady_clock::now();
        FalloutCellRecord actual;
        if (!extractFalloutCellAt(reader, entry, actual, error)) {
            std::cout << "  cell " << std::hex << entry.cellFormId << std::dec
                      << ": extract FAILED: " << error << "\n";
            ++mismatches;
            continue;
        }
        extractMsTotal +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - extractStart)
                .count();
        ++compared;

        std::vector<std::string> problems;
        if (actual.references.size() != expected->references.size()) {
            problems.push_back(
                "references " + std::to_string(actual.references.size()) + " vs " +
                std::to_string(expected->references.size()));
        }
        if (actual.navMeshes.size() != expected->navMeshes.size()) {
            problems.push_back(
                "navmeshes " + std::to_string(actual.navMeshes.size()) + " vs " +
                std::to_string(expected->navMeshes.size()));
        }
        const bool actualHasLand = actual.land != nullptr;
        const bool expectedHasLand = expected->land != nullptr;
        if (actualHasLand != expectedHasLand) {
            problems.push_back(std::string("land presence ") + (actualHasLand ? "yes" : "no") +
                               " vs " + (expectedHasLand ? "yes" : "no"));
        } else if (actualHasLand) {
            const FalloutLandRecord& a = *actual.land;
            const FalloutLandRecord& b = *expected->land;
            if (a.hasHeights != b.hasHeights ||
                (a.hasHeights && std::memcmp(a.heights, b.heights, sizeof(a.heights)) != 0)) {
                problems.push_back("VHGT heights differ");
            }
            if (a.hasNormals != b.hasNormals ||
                (a.hasNormals && std::memcmp(a.normals, b.normals, sizeof(a.normals)) != 0)) {
                problems.push_back("VNML normals differ");
            }
            if (a.hasColors != b.hasColors ||
                (a.hasColors && std::memcmp(a.colors, b.colors, sizeof(a.colors)) != 0)) {
                problems.push_back("VCLR colours differ");
            }
            if (std::memcmp(
                    a.quadrantBaseTextureFormId, b.quadrantBaseTextureFormId,
                    sizeof(a.quadrantBaseTextureFormId)) != 0) {
                problems.push_back("BTXT base textures differ");
            }
            if (a.textureLayers.size() != b.textureLayers.size()) {
                problems.push_back(
                    "layers " + std::to_string(a.textureLayers.size()) + " vs " +
                    std::to_string(b.textureLayers.size()));
            } else {
                for (std::size_t layer = 0; layer < a.textureLayers.size(); ++layer) {
                    const FalloutLandTextureLayer& la = a.textureLayers[layer];
                    const FalloutLandTextureLayer& lb = b.textureLayers[layer];
                    if (la.textureFormId != lb.textureFormId || la.quadrant != lb.quadrant ||
                        la.layerIndex != lb.layerIndex ||
                        std::memcmp(la.opacity, lb.opacity, sizeof(la.opacity)) != 0) {
                        problems.push_back("layer " + std::to_string(layer) + " differs");
                        break;
                    }
                }
            }
        }
        // References must match element-for-element, not just in count.
        const std::size_t refCompareCount =
            std::min(actual.references.size(), expected->references.size());
        for (std::size_t r = 0; r < refCompareCount; ++r) {
            const FalloutPlacedReference& ra = actual.references[r];
            const FalloutPlacedReference& rb = expected->references[r];
            if (ra.formId != rb.formId || ra.baseFormId != rb.baseFormId ||
                std::memcmp(ra.position, rb.position, sizeof(ra.position)) != 0 ||
                std::memcmp(ra.rotationRadians, rb.rotationRadians, sizeof(ra.rotationRadians)) != 0 ||
                ra.scale != rb.scale || ra.hasTeleport != rb.hasTeleport ||
                ra.teleportTargetRefFormId != rb.teleportTargetRefFormId) {
                problems.push_back("reference " + std::to_string(r) + " differs");
                break;
            }
        }

        if (!problems.empty()) {
            ++mismatches;
            std::cout << "  cell " << std::hex << entry.cellFormId << std::dec << " ("
                      << entry.gridX << "," << entry.gridZ << "): MISMATCH";
            for (const std::string& problem : problems) {
                std::cout << "\n      " << problem;
            }
            std::cout << "\n";
        }
    }

    std::cout << "compared " << compared << " cells, " << mismatches << " mismatches\n";
    if (compared > 0) {
        std::cout << "  extractFalloutCellAt averaged " << (extractMsTotal / static_cast<double>(compared))
                  << " ms per cell\n";
    }
    return mismatches == 0 ? 0 : 1;
}

// Builds one exterior cell through the SAME path the runtime streamer uses:
// world tables -> cell offset index -> extractFalloutCellAt -> CellSceneBuilder.
// Reporting the geometry counts here is what makes it possible to check the
// extracted library against what the cooker produces for the same cell.
int probeBuildCell(
    const std::filesystem::path& dataPath, const std::filesystem::path& esmPath,
    const std::string& worldspaceFilter, std::int32_t cellX, std::int32_t cellZ) {
    using namespace odai::importer::fnv;

    std::string error;
    const auto tablesStart = std::chrono::steady_clock::now();
    FalloutWorldTables tables;
    if (!buildFalloutWorldTables(esmPath, tables, error)) {
        std::cout << "world tables FAILED: " << error << "\n";
        return 1;
    }
    const double tablesMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tablesStart).count();
    std::cout << "world tables: " << tables.staticModelPaths.size() << " statics, "
              << tables.landTexturePaths.size() << " land textures, "
              << tables.worldspaceFormIdsByEditorId.size() << " worldspaces, built in "
              << tablesMs << " ms\n";

    FalloutCellIndex index;
    if (!buildFalloutCellIndex(esmPath, index, error)) {
        std::cout << "cell index FAILED: " << error << "\n";
        return 1;
    }

    std::uint32_t worldspaceFormId = 0;
    const auto worldIt = tables.worldspaceFormIdsByEditorId.find(toLowerAscii(worldspaceFilter));
    if (worldIt != tables.worldspaceFormIdsByEditorId.end()) {
        worldspaceFormId = worldIt->second;
    }

    const FalloutCellIndexEntry* entry = nullptr;
    for (const FalloutCellIndexEntry& candidate : index.cells) {
        if (candidate.isInterior || !candidate.hasGridCoords) {
            continue;
        }
        if (worldspaceFormId != 0 && candidate.worldspaceFormId != worldspaceFormId) {
            continue;
        }
        if (candidate.gridX == cellX && candidate.gridZ == cellZ) {
            entry = &candidate;
            break;
        }
    }
    if (entry == nullptr) {
        std::cout << "no cell at (" << cellX << "," << cellZ << ") in that worldspace\n";
        return 1;
    }

    EsmReader reader;
    if (!reader.open(esmPath)) {
        std::cout << "reader open FAILED: " << reader.lastError() << "\n";
        return 1;
    }
    FalloutCellRecord cell;
    const auto extractStart = std::chrono::steady_clock::now();
    if (!extractFalloutCellAt(reader, *entry, cell, error)) {
        std::cout << "extractFalloutCellAt FAILED: " << error << "\n";
        return 1;
    }
    const double extractMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - extractStart).count();

    FalloutAssetSource assets;
    if (!assets.open(dataPath)) {
        std::cout << "asset source FAILED to open " << dataPath << "\n";
        return 1;
    }

    const auto buildStart = std::chrono::steady_clock::now();
    CellSceneBuilder builder(assets, tables);
    const std::vector<const FalloutCellRecord*> cells{&cell};
    builder.setFallbackLandTexture(builder.dominantLandTexture(cells));
    builder.addCellTerrain(cell);
    builder.addCellStatics(cell);
    odai::importer::ImportedScene scene;
    builder.finish(scene);
    const double buildMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();

    // Optional: write it out so it can be diffed against a cooker-produced
    // scene for the same cell.
    if (const char* savePath = std::getenv("ODAI_PROBE_SAVE_CELL")) {
        if (odai::importer::saveImportedScene(scene, savePath)) {
            std::cout << "  saved to " << savePath << "\n";
        }
    }

    const CellBuildStats& stats = builder.stats();
    std::cout << "cell (" << cellX << "," << cellZ << "): " << cell.references.size()
              << " references, land=" << (cell.land != nullptr ? "yes" : "no") << "\n";
    std::cout << "  built: meshes=" << scene.meshes.size()
              << " instances=" << scene.instances.size()
              << " textures=" << scene.textures.size()
              << " packedVerts=" << scene.packedVertices.size()
              << " packedIndices=" << scene.packedIndices.size()
              << " packedDraws=" << scene.packedDraws.size()
              << " terrainParts=" << stats.terrainPartsEmitted << "\n";
    std::cout << "  shapes=" << stats.totalShapes
              << " untextured=" << stats.untexturedShapes
              << " placed=" << stats.placedInstances
              << " decalsSkipped=" << stats.shadowDecalShapesSkipped
              << " markersSkipped=" << stats.editorMarkerModelsSkipped
              << " droppedLayers=" << stats.droppedTerrainLayers << "\n";
    std::cout << "  timing: extract " << extractMs << " ms, build " << buildMs << " ms\n";
    std::cout << "    of which: NIF parse " << stats.nifParseMs << " ms (" << stats.nifsParsed
              << " meshes), texture decode " << stats.textureDecodeMs << " ms ("
              << stats.texturesDecoded << " textures)\n";
    return 0;
}

// Lists placed references by how far their origin sits above the cell's own
// terrain. Most statics rest on the ground, so a large positive offset is either
// a legitimately elevated object or a placement bug -- and printing the model
// path and rotation beside the offset is what separates the two.
int probeFloaters(
    const std::filesystem::path& dataPath, const std::filesystem::path& esmPath,
    const std::string& worldspaceFilter, std::int32_t cellX, std::int32_t cellZ) {
    using namespace odai::importer::fnv;

    std::string error;
    FalloutWorldTables tables;
    if (!buildFalloutWorldTables(esmPath, tables, error)) {
        std::cout << "world tables FAILED: " << error << "\n";
        return 1;
    }
    FalloutCellIndex index;
    if (!buildFalloutCellIndex(esmPath, index, error)) {
        std::cout << "cell index FAILED: " << error << "\n";
        return 1;
    }
    std::uint32_t worldspaceFormId = 0;
    const auto worldIt = tables.worldspaceFormIdsByEditorId.find(toLowerAscii(worldspaceFilter));
    if (worldIt != tables.worldspaceFormIdsByEditorId.end()) {
        worldspaceFormId = worldIt->second;
    }
    const FalloutCellIndexEntry* entry = nullptr;
    for (const FalloutCellIndexEntry& candidate : index.cells) {
        if (!candidate.isInterior && candidate.hasGridCoords &&
            (worldspaceFormId == 0 || candidate.worldspaceFormId == worldspaceFormId) &&
            candidate.gridX == cellX && candidate.gridZ == cellZ) {
            entry = &candidate;
            break;
        }
    }
    if (entry == nullptr) {
        std::cout << "no cell at (" << cellX << "," << cellZ << ")\n";
        return 1;
    }

    EsmReader reader;
    if (!reader.open(esmPath)) {
        return 1;
    }
    FalloutCellRecord cell;
    if (!extractFalloutCellAt(reader, *entry, cell, error)) {
        std::cout << "extract FAILED: " << error << "\n";
        return 1;
    }
    if (cell.land == nullptr || !cell.land->hasHeights) {
        std::cout << "cell has no LAND heights to compare against\n";
        return 1;
    }

    // Bilinear sample of the 33x33 VHGT grid at a world position inside the cell.
    const float cellOriginX = static_cast<float>(cellX) * kExteriorCellSize;
    const float cellOriginY = static_cast<float>(cellZ) * kExteriorCellSize;
    const auto terrainHeightAt = [&](float worldX, float worldY) {
        const float gx = std::clamp(
            (worldX - cellOriginX) / kLandPostSpacing, 0.0f, static_cast<float>(kLandGridSize - 1));
        const float gy = std::clamp(
            (worldY - cellOriginY) / kLandPostSpacing, 0.0f, static_cast<float>(kLandGridSize - 1));
        const int x0 = static_cast<int>(gx);
        const int y0 = static_cast<int>(gy);
        const int x1 = std::min(x0 + 1, kLandGridSize - 1);
        const int y1 = std::min(y0 + 1, kLandGridSize - 1);
        const float fx = gx - static_cast<float>(x0);
        const float fy = gy - static_cast<float>(y0);
        const auto at = [&](int x, int y) { return cell.land->heights[(y * kLandGridSize) + x]; };
        return (at(x0, y0) * (1 - fx) * (1 - fy)) + (at(x1, y0) * fx * (1 - fy)) +
               (at(x0, y1) * (1 - fx) * fy) + (at(x1, y1) * fx * fy);
    };

    // Ground truth for "is it floating": transform the mesh's own vertices into
    // world space and take the MINIMUM clearance over the terrain beneath them.
    // The reference origin sitting high means nothing on its own -- a sloped
    // road piece is authored with its origin at the raised end.
    FalloutAssetSource assets;
    const bool haveAssets = assets.open(dataPath);
    std::unordered_map<std::uint32_t, std::vector<std::array<float, 3>>> meshPointsByFormId;
    const auto meshPointsFor = [&](std::uint32_t baseFormId,
                                   const std::string& modelPath) -> const std::vector<std::array<float, 3>>& {
        static const std::vector<std::array<float, 3>> kEmpty;
        const auto cached = meshPointsByFormId.find(baseFormId);
        if (cached != meshPointsByFormId.end()) {
            return cached->second;
        }
        std::vector<std::array<float, 3>> points;
        std::vector<std::uint8_t> nifBytes;
        std::string meshError;
        if (haveAssets && assets.resolveMesh(modelPath, nifBytes, meshError)) {
            NifModel model;
            std::string nifError;
            if (parseNifStaticMesh(nifBytes, model, nifError)) {
                for (const auto& shape : model.shapes) {
                    // Sample sparsely; a few hundred points bound the footprint
                    // well enough and this runs per distinct model, not per ref.
                    const std::size_t vertexCount = shape.positions.size() / 3u;
                    const std::size_t stride = std::max<std::size_t>(1u, vertexCount / 200u);
                    for (std::size_t v = 0; v < vertexCount; v += stride) {
                        points.push_back({shape.positions[v * 3u], shape.positions[(v * 3u) + 1],
                                          shape.positions[(v * 3u) + 2]});
                    }
                }
            }
        }
        return meshPointsByFormId.emplace(baseFormId, std::move(points)).first->second;
    };

    struct Entry {
        float offset;
        float minClearance;
        float rotationMagnitudeDegrees;
        float scale;
        float localX;
        float localY;
        bool outsideCell;
        std::string model;
    };
    std::vector<Entry> entries;
    std::size_t unknownBaseCount = 0;
    for (const FalloutPlacedReference& ref : cell.references) {
        const auto modelIt = tables.staticModelPaths.find(ref.baseFormId);
        if (modelIt == tables.staticModelPaths.end()) {
            // The base record is not a STAT this importer knows. Every such
            // reference is silently dropped from the scene, which is how a road
            // can end up resting on nothing.
            ++unknownBaseCount;
            continue;
        }
        const float ground = terrainHeightAt(ref.position[0], ref.position[1]);
        const float rotationMagnitude =
            std::sqrt((ref.rotationRadians[0] * ref.rotationRadians[0]) +
                      (ref.rotationRadians[1] * ref.rotationRadians[1]) +
                      (ref.rotationRadians[2] * ref.rotationRadians[2])) *
            57.2957795f;
        const float localX = ref.position[0] - cellOriginX;
        const float localY = ref.position[1] - cellOriginY;
        // A reference can be listed in one cell but positioned outside it; the
        // height sampler clamps to the cell edge there, so the offset it reports
        // is against the wrong ground and must not be read as a placement bug.
        const bool outsideCell = localX < 0.0f || localX > kExteriorCellSize ||
                                 localY < 0.0f || localY > kExteriorCellSize;
        // Bethesda euler order, matching the cooker/cell builder.
        const float cx = std::cos(ref.rotationRadians[0]);
        const float sx = std::sin(ref.rotationRadians[0]);
        const float cy = std::cos(ref.rotationRadians[1]);
        const float sy = std::sin(ref.rotationRadians[1]);
        const float cz = std::cos(ref.rotationRadians[2]);
        const float sz = std::sin(ref.rotationRadians[2]);
        const float rot[9] = {
            cz * cy,  (cz * sy * sx) - (sz * cx),  (cz * sy * cx) + (sz * sx),
            sz * cy,  (sz * sy * sx) + (cz * cx),  (sz * sy * cx) - (cz * sx),
            -sy,      cy * sx,                     cy * cx};
        float minClearance = std::numeric_limits<float>::max();
        for (const std::array<float, 3>& local : meshPointsFor(ref.baseFormId, modelIt->second)) {
            const float wx = ref.position[0] + ref.scale * ((rot[0] * local[0]) + (rot[1] * local[1]) + (rot[2] * local[2]));
            const float wy = ref.position[1] + ref.scale * ((rot[3] * local[0]) + (rot[4] * local[1]) + (rot[5] * local[2]));
            const float wz = ref.position[2] + ref.scale * ((rot[6] * local[0]) + (rot[7] * local[1]) + (rot[8] * local[2]));
            minClearance = std::min(minClearance, wz - terrainHeightAt(wx, wy));
        }
        if (minClearance == std::numeric_limits<float>::max()) {
            minClearance = 0.0f;  // no mesh points; do not report it as floating
        }
        entries.push_back(Entry{ref.position[2] - ground, minClearance, rotationMagnitude,
                                ref.scale, localX, localY, outsideCell, modelIt->second});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.minClearance > b.minClearance;
    });

    // Which base record types actually placed geometry here.
    std::map<std::string, std::size_t> countByType;
    for (const FalloutPlacedReference& ref : cell.references) {
        const auto typeIt = tables.staticRecordTypes.find(ref.baseFormId);
        if (typeIt != tables.staticRecordTypes.end()) {
            ++countByType[typeIt->second];
        }
    }
    std::cout << "  placements by base record type:";
    for (const auto& [type, count] : countByType) {
        std::cout << " " << type << "=" << count;
    }
    std::cout << "\n";
    // Name what the non-STAT types actually place: a type count alone cannot
    // say whether those records carry real geometry or invisible markers.
    std::map<std::string, std::set<std::string>> modelsByType;
    for (const FalloutPlacedReference& ref : cell.references) {
        const auto typeIt = tables.staticRecordTypes.find(ref.baseFormId);
        const auto modelIt = tables.staticModelPaths.find(ref.baseFormId);
        if (typeIt == tables.staticRecordTypes.end() || typeIt->second == "STAT" ||
            modelIt == tables.staticModelPaths.end()) {
            continue;
        }
        modelsByType[typeIt->second].insert(modelIt->second);
    }
    for (const auto& [type, models] : modelsByType) {
        for (const std::string& model : models) {
            std::cout << "    " << type << "  " << model << "\n";
        }
    }

    std::cout << "cell (" << cellX << "," << cellZ << "): " << entries.size()
              << " placed statics, " << unknownBaseCount
              << " references DROPPED (base record is not a known STAT)\n";
    std::size_t floating = 0;
    for (const Entry& e : entries) {
        if (e.minClearance > 50.0f) {
            ++floating;
        }
    }
    std::cout << "  " << floating << " have their ENTIRE mesh more than 50 units clear of the "
              << "terrain (i.e. genuinely floating)\n";
    const std::size_t show = std::min<std::size_t>(entries.size(), 14u);
    for (std::size_t i = 0; i < show; ++i) {
        std::cout << "  clearance +" << static_cast<int>(entries[i].minClearance)
                  << "  origin +" << static_cast<int>(entries[i].offset) << " units  rot "
                  << static_cast<int>(entries[i].rotationMagnitudeDegrees) << " deg  scale "
                  << entries[i].scale << (entries[i].outsideCell ? "  OUTSIDE-CELL" : "")
                  << "  local(" << static_cast<int>(entries[i].localX) << ","
                  << static_cast<int>(entries[i].localY) << ")  " << entries[i].model << "\n";
    }
    return 0;
}

int listCells(const std::filesystem::path& esmPath, const std::string& filter) {
    odai::importer::fnv::FalloutSceneData scene;
    std::string error;
    odai::importer::fnv::FalloutExtractFilter extractFilter{};
    if (!odai::importer::fnv::extractFalloutScene(esmPath, extractFilter, scene, error)) {
        std::cout << "Extract FAILED: " << error << "\n";
        return 1;
    }
    const std::string loweredFilter = toLowerAscii(filter);
    std::size_t interiorCount = 0;
    std::size_t shown = 0;
    for (const auto& cell : scene.cells) {
        if (!cell.isInterior || cell.editorId.empty()) {
            continue;
        }
        ++interiorCount;
        if (!loweredFilter.empty() &&
            toLowerAscii(cell.editorId).find(loweredFilter) == std::string::npos) {
            continue;
        }
        std::cout << "  " << cell.editorId << "  (" << cell.references.size() << " refs)\n";
        ++shown;
    }
    std::cout << "Interior cells with an EditorID: " << interiorCount;
    if (!loweredFilter.empty()) {
        std::cout << ", " << shown << " matching \"" << filter << "\"";
    }
    std::cout << "\n";
    return 0;
}

// NAVM raw dump. The navmesh layout is being derived from real records rather
// than from memory: every format guess in this importer that was reasoned from
// documentation instead of bytes has been wrong at least once (the NIF vector
// flags, the shader property offsets, the byte-misaligned texture refs). This
// prints subrecord sizes and the leading words of NVNM so the structure can be
// read off actual data.
int probeNavmesh(const std::filesystem::path& pluginPath, std::size_t limit) {
    odai::importer::fnv::EsmReader reader;
    if (!reader.open(pluginPath)) {
        std::cout << "open failed: " << reader.lastError() << "\n";
        return 1;
    }
    std::size_t seen = 0;
    std::map<std::string, std::size_t> subrecordCounts;
    odai::importer::fnv::EsmReader::Visitor visitor;
    visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
        if (record.type != "NAVM") {
            return;
        }
        for (const auto& sub : record.subrecords) {
            ++subrecordCounts[std::string(sub.type)];
        }
        if (seen++ >= limit) {
            return;
        }
        std::cout << "NAVM formId=0x" << std::hex << record.formId << std::dec << "\n";
        for (const auto& sub : record.subrecords) {
            std::cout << "  " << sub.type << " size=" << sub.size;
            const std::size_t words = std::min<std::size_t>(sub.size / 4u, 12u);
            std::cout << "  words:";
            for (std::size_t w = 0; w < words; ++w) {
                std::uint32_t value = 0;
                std::memcpy(&value, sub.data + (w * 4u), 4u);
                std::cout << " " << value;
            }
            std::cout << "\n";
        }
    };
    if (!reader.walk(visitor)) {
        std::cout << "walk failed: " << reader.lastError() << "\n";
        return 1;
    }
    std::cout << "NAVM subrecord census:\n";
    for (const auto& [type, count] : subrecordCounts) {
        std::cout << "  " << count << "x " << type << "\n";
    }
    return 0;
}

int probePlugin(const std::filesystem::path& pluginPath) {
    odai::importer::fnv::EsmReader reader;
    if (!reader.open(pluginPath)) {
        std::cout << "open failed: " << reader.lastError() << "\n";
        return 1;
    }
    std::map<std::string, std::size_t> recordCounts;
    odai::importer::fnv::EsmReader::Visitor visitor;
    visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) { ++recordCounts[record.type]; };
    if (!reader.walk(visitor)) {
        std::cout << "walk failed: " << reader.lastError() << "\n";
        return 1;
    }
    std::size_t total = 0;
    for (const auto& [type, count] : recordCounts) {
        total += count;
    }
    std::cout << "Walked " << total << " records of " << recordCounts.size() << " types.\n";
    std::vector<std::pair<std::string, std::size_t>> sorted(recordCounts.begin(), recordCounts.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (std::size_t i = 0; i < sorted.size() && i < 20u; ++i) {
        std::cout << "  " << sorted[i].second << "  " << sorted[i].first << "\n";
    }
    std::cout << "Tolerated checksum failures: " << reader.toleratedChecksumFailures() << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Placement-convention search.
//
// Two things about REFR placement were never validated against real data and
// are flagged as guesses in the source: the Bethesda->engine axis map, and the
// Euler order the three DATA rotation angles compose in. Guessing is what
// produced the format bugs already found here, so this searches instead.
//
// The signal is that Bethesda interiors are hand-assembled from modular
// pieces: floors, walls and ceilings meet edge-to-edge with very little
// overlap. Compose the angles wrongly and the room shatters — pieces rotate
// through each other and the total pairwise intersection volume jumps. So:
// score every candidate convention by summed pairwise AABB overlap across a
// cell's instances, and the convention that assembles the room wins.
// ---------------------------------------------------------------------------

struct Mat3 {
    double m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

Mat3 multiply3(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += a.m[(r * 3) + k] * b.m[(k * 3) + c];
            }
            out.m[(r * 3) + c] = sum;
        }
    }
    return out;
}

Mat3 axisRotation(int axis, double radians) {
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    Mat3 out{};
    if (axis == 0) {
        out.m[0] = 1; out.m[4] = c; out.m[5] = -s; out.m[7] = s; out.m[8] = c;
        out.m[1] = out.m[2] = out.m[3] = out.m[6] = 0;
    } else if (axis == 1) {
        out.m[0] = c; out.m[2] = s; out.m[4] = 1; out.m[6] = -s; out.m[8] = c;
        out.m[1] = out.m[3] = out.m[5] = out.m[7] = 0;
    } else {
        out.m[0] = c; out.m[1] = -s; out.m[3] = s; out.m[4] = c; out.m[8] = 1;
        out.m[2] = out.m[5] = out.m[6] = out.m[7] = 0;
    }
    return out;
}

struct Aabb {
    double lo[3] = {1e30, 1e30, 1e30};
    double hi[3] = {-1e30, -1e30, -1e30};
    void add(const double p[3]) {
        for (int i = 0; i < 3; ++i) {
            lo[i] = std::min(lo[i], p[i]);
            hi[i] = std::max(hi[i], p[i]);
        }
    }
    bool valid() const { return lo[0] <= hi[0]; }
    double volume() const {
        if (!valid()) {
            return 0.0;
        }
        return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
    }
};

double overlapVolume(const Aabb& a, const Aabb& b) {
    double product = 1.0;
    for (int i = 0; i < 3; ++i) {
        const double span = std::min(a.hi[i], b.hi[i]) - std::max(a.lo[i], b.lo[i]);
        if (span <= 0.0) {
            return 0.0;
        }
        product *= span;
    }
    return product;
}

struct Convention {
    std::string name;
    int order[3];      // axis application order, innermost first
    double angleSign;  // +1 or -1
    bool mirrorMap;    // true = the current (x,z,y) reflection, false = (x,z,-y)
};

// Applies the Bethesda->engine axis map to a point.
void mapToEngine(bool mirrorMap, const double p[3], double out[3]) {
    out[0] = p[0];
    out[1] = p[2];
    out[2] = mirrorMap ? p[1] : -p[1];
}

int probeRotations(const std::filesystem::path& dataPath, const std::string& plugin, const std::string& cellId) {
    odai::importer::fnv::FalloutSceneData scene;
    std::string error;
    odai::importer::fnv::FalloutExtractFilter filter;
    filter.wantCellContents = [&](const odai::importer::fnv::FalloutCellRecord& cell) {
        return cell.editorId == cellId;
    };
    if (!odai::importer::fnv::extractFalloutScene(dataPath / plugin, filter, scene, error)) {
        std::cout << "extract failed: " << error << "\n";
        return 1;
    }
    const odai::importer::fnv::FalloutCellRecord* target = nullptr;
    for (const auto& cell : scene.cells) {
        if (cell.editorId == cellId) {
            target = &cell;
            break;
        }
    }
    if (target == nullptr) {
        std::cout << "cell not found: " << cellId << "\n";
        return 1;
    }

    std::map<std::uint32_t, const odai::importer::fnv::FalloutStaticRecord*> staticsByFormId;
    for (const auto& stat : scene.statics) {
        staticsByFormId[stat.formId] = &stat;
    }

    // Local-space AABB per base static, from its NIF.
    MeshIndex meshes = buildMeshIndex(dataPath);
    std::map<std::uint32_t, Aabb> localBounds;
    std::vector<std::uint8_t> bytes;
    for (const auto& ref : target->references) {
        if (localBounds.count(ref.baseFormId) != 0u) {
            continue;
        }
        const auto statIt = staticsByFormId.find(ref.baseFormId);
        if (statIt == staticsByFormId.end()) {
            continue;
        }
        std::string modelPath = "meshes\\" + statIt->second->modelPath;
        for (char& c : modelPath) {
            if (c == '/') {
                c = '\\';
            }
        }
        for (std::size_t a = 0; a < meshes.archives.size(); ++a) {
            const auto* entry = meshes.archives[a].find(modelPath);
            if (entry == nullptr || !meshes.archives[a].extract(*entry, bytes)) {
                continue;
            }
            odai::importer::fnv::NifModel model;
            std::string nifError;
            if (!odai::importer::fnv::parseNifStaticMesh(bytes, model, nifError)) {
                break;
            }
            Aabb box;
            for (const auto& shape : model.shapes) {
                for (std::size_t v = 0; v * 3u < shape.positions.size(); ++v) {
                    const double p[3] = {shape.positions[v * 3u], shape.positions[(v * 3u) + 1],
                                         shape.positions[(v * 3u) + 2]};
                    box.add(p);
                }
            }
            if (box.valid()) {
                localBounds[ref.baseFormId] = box;
            }
            break;
        }
    }
    std::cout << "Cell " << cellId << ": " << target->references.size() << " reference(s), " << localBounds.size()
              << " base mesh(es) resolved.\n";

    std::vector<Convention> candidates;
    const int orders[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    const char* axisName = "XYZ";
    for (const auto& order : orders) {
        for (const double sign : {1.0, -1.0}) {
            for (const bool mirror : {false, true}) {
                Convention convention;
                convention.name = std::string("R=") + axisName[order[2]] + axisName[order[1]] + axisName[order[0]] +
                    (sign > 0 ? " +ang" : " -ang") + (mirror ? " map(x,z,y)" : " map(x,z,-y)");
                convention.order[0] = order[0];
                convention.order[1] = order[1];
                convention.order[2] = order[2];
                convention.angleSign = sign;
                convention.mirrorMap = mirror;
                candidates.push_back(convention);
            }
        }
    }

    std::vector<std::pair<double, std::string>> results;
    for (const Convention& convention : candidates) {
        std::vector<Aabb> worldBoxes;
        for (const auto& ref : target->references) {
            const auto boundsIt = localBounds.find(ref.baseFormId);
            if (boundsIt == localBounds.end()) {
                continue;
            }
            Mat3 rotation{};
            for (int step = 0; step < 3; ++step) {
                const int axis = convention.order[step];
                const double angle = convention.angleSign * static_cast<double>(ref.rotationRadians[axis]);
                rotation = multiply3(axisRotation(axis, angle), rotation);
            }
            Aabb world;
            const Aabb& local = boundsIt->second;
            for (int corner = 0; corner < 8; ++corner) {
                const double localPoint[3] = {
                    (corner & 1) ? local.hi[0] : local.lo[0],
                    (corner & 2) ? local.hi[1] : local.lo[1],
                    (corner & 4) ? local.hi[2] : local.lo[2]};
                double rotated[3] = {0, 0, 0};
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        rotated[r] += rotation.m[(r * 3) + c] * localPoint[c] * static_cast<double>(ref.scale);
                    }
                }
                const double bethPoint[3] = {
                    rotated[0] + ref.position[0], rotated[1] + ref.position[1], rotated[2] + ref.position[2]};
                double enginePoint[3];
                mapToEngine(convention.mirrorMap, bethPoint, enginePoint);
                world.add(enginePoint);
            }
            worldBoxes.push_back(world);
        }

        double totalOverlap = 0.0;
        for (std::size_t i = 0; i < worldBoxes.size(); ++i) {
            for (std::size_t j = i + 1; j < worldBoxes.size(); ++j) {
                totalOverlap += overlapVolume(worldBoxes[i], worldBoxes[j]);
            }
        }
        results.emplace_back(totalOverlap, convention.name);
    }

    std::sort(results.begin(), results.end());
    std::cout << "Ranked by total pairwise overlap volume (lower assembles better):\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        std::cout << "  " << (i == 0 ? "-> " : "   ") << results[i].second << "   overlap=" << results[i].first << "\n";
    }
    return 0;
}

// Inspects a cooked .bin the way the renderer sees it: the [0,
// terrainDrawCount) / [terrainDrawCount, end) draw split, and what the vertices
// in each half actually carry. Written because "the terrain is black" produced
// four plausible explanations and no way to choose between them.
int probeScene(const std::filesystem::path& scenePath) {
    odai::importer::ImportedScene scene;
    if (!odai::importer::loadImportedScene(scenePath, scene)) {
        std::cout << "load failed: " << odai::importer::getImportedSceneLastError() << "\n";
        return 1;
    }
    const bool interior = odai::importer::importedSceneSourceTagIsInterior(scene.sourceTag);
    const auto terrainDrawCount = interior
        ? 0u
        : std::min<std::uint32_t>(
              scene.landscapeCells.empty() ? scene.sourceLandscapeCellCount
                                           : static_cast<std::uint32_t>(scene.landscapeCells.size()),
              static_cast<std::uint32_t>(scene.packedDraws.size()));

    // Texture inventory. An entry with no pixel data resolves to a valid index
    // at cook time and uploads as nothing; a two-channel data format (BC4/BC5)
    // used as a colour map reads as pale blue-white, which by eye is
    // indistinguishable from blown-out lighting.
    {
        std::size_t emptyTextures = 0;
        std::map<int, std::size_t> formatCounts;
        for (const auto& texture : scene.textures) {
            if (texture.width == 0u || texture.height == 0u || texture.rgba8.empty()) {
                ++emptyTextures;
                std::cout << "  EMPTY texture: \"" << texture.sourcePath << "\"\n";
            }
            ++formatCounts[static_cast<int>(texture.format)];
        }
        std::cout << "texture inventory: " << scene.textures.size() << " total, "
                  << emptyTextures << " empty; formats"
                  << " (0=RGBA8 1=BC1 2=BC2 3=BC3 4=BC4 5=BC5 6=BC7):";
        for (const auto& [format, count] : formatCounts) {
            std::cout << " " << format << "x" << count;
        }
        std::cout << "\n";
        for (const auto& texture : scene.textures) {
            const int format = static_cast<int>(texture.format);
            if (format == 4 || format == 5) {
                std::cout << "  data-format texture in the colour table: \"" << texture.sourcePath
                          << "\" format " << format << "\n";
            }
        }
        if (std::getenv("ODAI_PROBE_LIST_TEXTURES") != nullptr) {
            for (const auto& texture : scene.textures) {
                std::cout << "  tex fmt=" << static_cast<int>(texture.format)
                          << " " << texture.width << "x" << texture.height
                          << " mips=" << texture.mipLevelCount
                          << " \"" << texture.sourcePath << "\"\n";
            }
        }
    }
    std::cout << "sourceTag \"" << scene.sourceTag << "\" (interior=" << (interior ? "yes" : "no") << ")\n"
              << "meshes " << scene.meshes.size() << ", instances " << scene.instances.size() << ", textures "
              << scene.textures.size() << ", lights " << scene.lights.size() << "\n"
              << "packedVertices " << scene.packedVertices.size() << ", packedIndices " << scene.packedIndices.size()
              << ", packedDraws " << scene.packedDraws.size() << "\n"
              << "sourceLandscapeCellCount " << scene.sourceLandscapeCellCount << " -> terrainDrawCount "
              << terrainDrawCount << "\n";
    if (!scene.meshes.empty()) {
        std::cout << "meshes[0].name \"" << scene.meshes.front().name << "\" (terrain iff exactly \"terrain\")"
                  << ", verts " << scene.meshes.front().vertices.size() << ", parts "
                  << scene.meshes.front().parts.size() << "\n";
    }

    auto summarize = [&](const char* label, std::uint32_t first, std::uint32_t last) {
        std::size_t indexTotal = 0;
        std::size_t emptyDraws = 0;
        std::size_t outOfRange = 0;
        float minColor[3] = {1e30f, 1e30f, 1e30f};
        float maxColor[3] = {-1e30f, -1e30f, -1e30f};
        float minUv[2] = {1e30f, 1e30f};
        float maxUv[2] = {-1e30f, -1e30f};
        std::size_t texturedVerts = 0;
        std::size_t untexturedVerts = 0;
        std::size_t layeredVerts = 0;
        std::size_t layerSlotsUsed = 0;
        std::size_t tintedVerts = 0;
        std::uint32_t minLayerWeight = 255u;
        std::uint32_t maxLayerWeight = 0u;
        std::size_t sampled = 0;
        double sumColor[3] = {0.0, 0.0, 0.0};
        for (std::uint32_t d = first; d < last && d < scene.packedDraws.size(); ++d) {
            const auto& draw = scene.packedDraws[d];
            indexTotal += draw.indexCount;
            if (draw.indexCount == 0u) { ++emptyDraws; continue; }
            if (static_cast<std::size_t>(draw.firstIndex) + draw.indexCount > scene.packedIndices.size()) {
                ++outOfRange;
                continue;
            }
            for (std::uint32_t i = 0; i < draw.indexCount; ++i, ++sampled) {
                const std::uint32_t vi = scene.packedIndices[draw.firstIndex + i];
                if (vi >= scene.packedVertices.size()) { ++outOfRange; break; }
                const auto& v = scene.packedVertices[vi];
                for (int c = 0; c < 3; ++c) {
                    minColor[c] = std::min(minColor[c], v.color[c]);
                    maxColor[c] = std::max(maxColor[c], v.color[c]);
                    sumColor[c] += static_cast<double>(v.color[c]);
                }
                for (int c = 0; c < 2; ++c) {
                    minUv[c] = std::min(minUv[c], v.uv[c]);
                    maxUv[c] = std::max(maxUv[c], v.uv[c]);
                }
                (v.textureIndex == 0xffffffffu ? untexturedVerts : texturedVerts) += 1u;
                if ((v.flags & odai::importer::kImportedSceneMaterialFlagTerrainLayers) != 0u) {
                    ++layeredVerts;
                    for (int layer = 0; layer < odai::importer::kImportedSceneMaxTerrainLayers; ++layer) {
                        if (v.layerTextureIndex[layer] != odai::importer::kImportedSceneNoTerrainLayer) {
                            ++layerSlotsUsed;
                            const std::uint32_t weight =
                                (v.layerWeights >> (layer * 8)) & 0xffu;
                            minLayerWeight = std::min(minLayerWeight, weight);
                            maxLayerWeight = std::max(maxLayerWeight, weight);
                        }
                    }
                }
                if ((v.flags & odai::importer::kImportedSceneMaterialFlagVertexColorTint) != 0u) {
                    ++tintedVerts;
                }
            }
        }
        std::cout << label << ": draws " << (last - first) << ", indices " << indexTotal << ", emptyDraws "
                  << emptyDraws << ", outOfRange " << outOfRange << "\n";
        if (sampled == 0) { std::cout << "    (no vertices sampled)\n"; return; }
        std::cout << "    color  r[" << minColor[0] << "," << maxColor[0] << "] g[" << minColor[1] << ","
                  << maxColor[1] << "] b[" << minColor[2] << "," << maxColor[2] << "]\n"
                  << "    mean colour (" << (sumColor[0] / static_cast<double>(sampled)) << ", "
                  << (sumColor[1] / static_cast<double>(sampled)) << ", "
                  << (sumColor[2] / static_cast<double>(sampled)) << ")\n"
                  << "    uv     u[" << minUv[0] << "," << maxUv[0] << "] v[" << minUv[1] << "," << maxUv[1] << "]\n"
                  << "    verts  textured " << texturedVerts << ", untextured " << untexturedVerts << "\n"
                  << "    tinted " << tintedVerts << " (vertex-colour modulated)\n"
                  << "    layers " << layeredVerts << " verts, " << layerSlotsUsed << " slots";
        if (layerSlotsUsed != 0) {
            std::cout << ", weight [" << minLayerWeight << "," << maxLayerWeight << "]/255";
        }
        std::cout << "\n";
    };
    summarize("terrain draws", 0u, terrainDrawCount);
    summarize("static draws ", terrainDrawCount, static_cast<std::uint32_t>(scene.packedDraws.size()));
    return 0;
}

void printUsage() {
    std::cout << "Usage:\n"
              << "  odai_newvegas_probe <DataFilesPath> --archives\n"
              << "  odai_newvegas_probe <DataFilesPath> --nifs [limit]\n"
              << "  odai_newvegas_probe <DataFilesPath> --nif <virtualPath>\n"
              << "  odai_newvegas_probe <DataFilesPath> --nifblocks <virtualPath>\n"
              << "  odai_newvegas_probe <DataFilesPath> --plugin <Plugin.esm>\n"
              << "  odai_newvegas_probe <DataFilesPath> --cells <Plugin.esm> [filter]\n"
              << "  odai_newvegas_probe <DataFilesPath> --navm <Plugin.esm> [dumpCount]\n"
              << "  odai_newvegas_probe <DataFilesPath> --rotations <Plugin.esm> <CellEditorID>\n"
              << "  odai_newvegas_probe <anyDir> --scene <cooked.bin>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage();
        return 2;
    }
    const std::filesystem::path dataPath = argv[1];
    const std::string mode = argv[2];

    if (!std::filesystem::is_directory(dataPath)) {
        std::cout << "Not a directory: " << dataPath << "\n";
        return 2;
    }

    if (mode == "--archives") {
        return probeArchives(dataPath);
    }
    if (mode == "--nifs") {
        const std::size_t limit = argc >= 4 ? static_cast<std::size_t>(std::stoull(argv[3])) : 500u;
        return probeNifs(dataPath, limit);
    }
    if (mode == "--nif" && argc >= 4) {
        return probeSingleNif(dataPath, argv[3]);
    }
    if (mode == "--nifblocks" && argc >= 4) {
        return dumpNifBlocks(dataPath, argv[3]);
    }
    if (mode == "--plugin" && argc >= 4) {
        return probePlugin(dataPath / argv[3]);
    }
    if (mode == "--navm" && argc >= 4) {
        return probeNavmesh(dataPath / argv[3], argc >= 5 ? static_cast<std::size_t>(std::atoi(argv[4])) : 2u);
    }
    if (mode == "--cells" && argc >= 4) {
        return listCells(dataPath / argv[3], argc >= 5 ? argv[4] : "");
    }
    if (mode == "--find" && argc >= 4) {
        return findArchiveEntries(
            dataPath, argv[3], argc >= 5 ? static_cast<std::size_t>(std::atoi(argv[4])) : 25u);
    }
    if (mode == "--floaters" && argc >= 7) {
        return probeFloaters(
            dataPath, dataPath / argv[3], argv[4], std::atoi(argv[5]), std::atoi(argv[6]));
    }
    if (mode == "--buildcell" && argc >= 7) {
        return probeBuildCell(
            dataPath, dataPath / argv[3], argv[4], std::atoi(argv[5]), std::atoi(argv[6]));
    }
    if (mode == "--cellindex" && argc >= 4) {
        return probeCellIndex(
            dataPath / argv[3],
            argc >= 5 ? argv[4] : "",
            argc >= 6 ? static_cast<std::size_t>(std::atoi(argv[5])) : 8u);
    }
    if (mode == "--scene" && argc >= 4) {
        return probeScene(argv[3]);
    }
    if (mode == "--rotations" && argc >= 5) {
        return probeRotations(dataPath, argv[3], argv[4]);
    }
    printUsage();
    return 2;
}
