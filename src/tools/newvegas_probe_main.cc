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

#include "import/fnv/bsa_archive.h"
#include "import/fnv/esm_reader.h"
#include "import/fnv/fallout_records.h"
#include "import/fnv/nif_scene.h"
#include "import/imported_scene.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
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
    if (mode == "--scene" && argc >= 4) {
        return probeScene(argv[3]);
    }
    if (mode == "--rotations" && argc >= 5) {
        return probeRotations(dataPath, argv[3], argv[4]);
    }
    printUsage();
    return 2;
}
