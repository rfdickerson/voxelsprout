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
//   odai_bethesda_probe <DataFilesPath> --archives
//   odai_bethesda_probe <DataFilesPath> --nifs [limit]
//   odai_bethesda_probe <DataFilesPath> --nif <virtualPath>
//   odai_bethesda_probe <DataFilesPath> --plugin <Plugin.esm>

#include "bethesda/papyrus_vm.h"
#include "bethesda/save_game.h"
#include "bethesda/skyrim_quest.h"
#include "bethesda/skyrim_scenario_content.h"
#include "bethesda/tes3_content.h"
#include "bethesda/tes3_runtime.h"
#include "bethesda/vmad_reader.h"
#include "import/dds.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/bsa_archive.h"
#include "import/fnv/cell_builder.h"
#include "import/fnv/character_builder.h"
#include "import/fnv/content_profile.h"
#include "import/fnv/content_record_index.h"
#include "import/fnv/esm_reader.h"
#include "import/fnv/dialogue_records.h"
#include "import/fnv/actor_records.h"
#include "import/fnv/fallout_records.h"
#include "import/fnv/kf_animation.h"
#include "import/fnv/nif_scene.h"
#include "import/fnv/skyrim_animation_assets.h"
#include "import/fnv/strings_table.h"
#include "bethesda/bethesda_physics_world.h"
#include "import/imported_scene.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using odai::importer::fnv::BsaArchive;
using odai::importer::fnv::BsaFileEntry;

// Small hex formatter: the enable-parent report in the floater dump is built as
// a string, so it cannot lean on the stream's std::hex the way the other formID
// prints do.
std::string toHex(std::uint32_t value) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%x", value);
    return buffer;
}

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

const char* alphaSemanticName(odai::importer::fnv::NifAlphaSemantic semantic) {
    using Semantic = odai::importer::fnv::NifAlphaSemantic;
    switch (semantic) {
        case Semantic::Opaque: return "opaque";
        case Semantic::Cutout: return "cutout";
        case Semantic::ExplicitBlend: return "explicit-blend";
        case Semantic::VertexFade: return "vertex-fade";
    }
    return "unknown";
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
    // Node-recognition health. A nonzero failure count means a subtree was
    // dropped; before the else-branch existed it meant one was silently
    // relocated to the model origin instead.
    std::array<std::size_t, 8> alphaTestFunctions{};
    std::size_t modelsUsingFooterRoots = 0;
    std::size_t totalNodeParseFailures = 0;
    std::size_t modelsWithNodeParseFailure = 0;
    std::size_t totalUnhandledNodeTypes = 0;
    std::size_t totalMirroredShapes = 0;
    std::size_t totalReversedWinding = 0;
    std::size_t totalOutOfRangeTriangles = 0;
    std::size_t totalDegenerateTriangles = 0;
    std::size_t modelsWithOutOfRangeTriangles = 0;
    std::array<std::size_t, 4> stencilDrawModes{};
    std::map<std::string, std::size_t> parseErrors;
    // Which property types are costing us textures, aggregated across the whole
    // sample rather than one file at a time. "44 of 800 shapes have no diffuse"
    // is not actionable; "N of them carry a BSShaderNoLightingProperty" is, and
    // it turns a fix into a before/after number instead of a claim.
    std::map<std::string, std::size_t> unresolvedPropertyTypes;

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
        if (model.usedFooterRoots) { ++modelsUsingFooterRoots; }
        for (int f = 0; f < 8; ++f) { alphaTestFunctions[f] += model.alphaTestFunctionCounts[f]; }
        totalNodeParseFailures += model.nodeParseFailedCount;
        totalUnhandledNodeTypes += model.unhandledNodeTypeCount;
        if (model.nodeParseFailedCount != 0u) { ++modelsWithNodeParseFailure; }
        totalMirroredShapes += model.mirroredShapeCount;
        totalReversedWinding += model.reversedWindingShapeCount;
        totalOutOfRangeTriangles += model.outOfRangeTriangleCount;
        totalDegenerateTriangles += model.degenerateTriangleCount;
        if (model.outOfRangeTriangleCount != 0u) { ++modelsWithOutOfRangeTriangles; }
        for (int m = 0; m < 4; ++m) {
            stencilDrawModes[m] += model.stencilDrawModeCounts[m];
        }
        for (const std::string& typeName : model.unresolvedPropertyTypes) {
            ++unresolvedPropertyTypes[typeName];
        }
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
              << "  footer-rooted     " << modelsUsingFooterRoots << " model(s)\n"
              << "  alphaTest funcs   ALWAYS=" << alphaTestFunctions[0] << " LESS=" << alphaTestFunctions[1]
              << " EQUAL=" << alphaTestFunctions[2] << " LEQUAL=" << alphaTestFunctions[3]
              << " GREATER=" << alphaTestFunctions[4] << " NOTEQUAL=" << alphaTestFunctions[5]
              << " GEQUAL=" << alphaTestFunctions[6] << " NEVER=" << alphaTestFunctions[7] << "\n"
              << "  node parse fails  " << totalNodeParseFailures << " across "
              << modelsWithNodeParseFailure << " model(s); unhandled *Node types "
              << totalUnhandledNodeTypes << "\n"
              << "  mirrored shapes   " << totalMirroredShapes << "  (negative-determinant transform)\n"
              << "  DRAW_CW shapes    " << totalReversedWinding << "  (stencil says CW is the front face)\n"
              << "  bad triangles     " << totalOutOfRangeTriangles << " out-of-range in "
              << modelsWithOutOfRangeTriangles << " model(s), " << totalDegenerateTriangles
              << " degenerate  (dropped at parse)\n"
              << "  stencil drawModes ccwOrBoth=" << stencilDrawModes[0] << " ccw=" << stencilDrawModes[1]
              << " cw=" << stencilDrawModes[2] << " both=" << stencilDrawModes[3] << "\n"
              << "  shapes w/ diffuse " << shapesWithDiffuse << "\n"
              << "  shapes w/ alphaTest " << shapesWithAlphaTest << "\n"
              << "  shapes w/o diffuse  " << (totalShapes - shapesWithDiffuse) << "\n";
    for (const auto& [message, count] : parseErrors) {
        std::cout << "  [" << count << "x] " << message << "\n";
    }
    if (!unresolvedPropertyTypes.empty()) {
        std::cout << "unresolved property types (each one is a shape that lost its texture):\n";
        for (const auto& [typeName, count] : unresolvedPropertyTypes) {
            std::cout << "  " << count << "x " << typeName << "\n";
        }
    }
    return 0;
}

// Raw block inventory: what types the file actually contains, and the header
// string table. Exists because three separate attempts to locate a texture path
// by walking to a computed field offset all missed -- reading the file's own
// declared contents settles in one shot what guessing does not.
int dumpNifBlocks(const std::filesystem::path& dataPath, const std::string& virtualPath) {
    // Specific-file diagnostics must see loose mod assets too. The archive-only
    // mesh index used by --nifs made --nifblocks claim that every unpacked
    // Morrowind/Tamriel Data mesh was missing, even though the runtime resolved
    // that same path successfully a moment later.
    odai::importer::fnv::FalloutAssetSource assets;
    if (!assets.open(dataPath)) {
        std::cout << "cannot index assets under " << dataPath << "\n";
        return 1;
    }
    assets.addModDirectory(dataPath);
    std::vector<std::uint8_t> bytes;
    std::string resolveError;
    if (!assets.resolveMesh(virtualPath, bytes, resolveError)) {
        std::cout << "resolve failed: " << resolveError << "\n";
        return 1;
    }
    {
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
        // The FOOTER: "Num Roots" u32 then that many block refs, immediately
        // after the last block. NIF states its roots here explicitly, and the
        // importer does not read it -- it instead promotes every node that no
        // other node claims as a child, walking each with an identity
        // transform. That heuristic is what relocates a subtree to the model
        // origin when its parent is unrecognized or fails to parse.
        //
        // Printed before anything depends on it, so the claim "the footer says
        // which blocks are roots" can be checked against real archives rather
        // than assumed.
        std::cout << "footer:\n";
        if (summary.blockStarts.empty()) {
            std::cout << "  (no blocks)\n";
        } else {
            const std::size_t footerOffset =
                summary.blockStarts.back() + summary.blockSizes.back();
            if (footerOffset + 4u > bytes.size()) {
                std::cout << "  MISSING: last block ends at " << footerOffset << " but file is "
                          << bytes.size() << " bytes\n";
            } else {
                std::uint32_t rootCount = 0;
                std::memcpy(&rootCount, bytes.data() + footerOffset, 4u);
                std::cout << "  offset=" << footerOffset << " fileSize=" << bytes.size()
                          << " trailingBytes=" << (bytes.size() - footerOffset)
                          << " numRoots=" << rootCount << "\n";
                // 4 bytes for the count plus one ref each; anything else means
                // the offset is not actually the footer.
                const std::size_t expected = 4u + (static_cast<std::size_t>(rootCount) * 4u);
                if (rootCount > 64u || footerOffset + expected > bytes.size()) {
                    std::cout << "  IMPLAUSIBLE root count -- offset is probably not the footer\n";
                } else {
                    for (std::uint32_t r = 0; r < rootCount; ++r) {
                        std::int32_t rootRef = 0;
                        std::memcpy(&rootRef, bytes.data() + footerOffset + 4u + (r * 4u), 4u);
                        const bool inRange = rootRef >= 0 &&
                            static_cast<std::size_t>(rootRef) < summary.blockTypeNames.size();
                        std::cout << "  root[" << r << "] = block " << rootRef << " ("
                                  << (inRange ? summary.blockTypeNames[static_cast<std::size_t>(rootRef)]
                                              : std::string("OUT OF RANGE"))
                                  << ")\n";
                    }
                    std::cout << "  exactBytes=" << (expected == (bytes.size() - footerOffset) ? "yes" : "no")
                              << "\n";
                }
            }
        }
        // NiAlphaProperty decoded: flags u16 at block offset 12 (after nameRef,
        // numExtraData=0, controllerRef), threshold u8 at 14. The function
        // bits (10-12) are what the importer currently ignores; 4 is GREATER.
        std::cout << "NiAlphaProperty decode:\n";
        for (std::size_t i = 0; i < summary.blockTypeNames.size(); ++i) {
            if (summary.blockTypeNames[i] != "NiAlphaProperty" || summary.blockSizes[i] < 15u) {
                continue;
            }
            std::uint16_t flags = 0;
            std::memcpy(&flags, bytes.data() + summary.blockStarts[i] + 12u, 2u);
            const std::uint8_t threshold = bytes[summary.blockStarts[i] + 14u];
            static const char* kTestFunctions[8] = {
                "ALWAYS", "LESS", "EQUAL", "LEQUAL", "GREATER", "NOTEQUAL", "GEQUAL", "NEVER"};
            std::cout << "  [" << i << "] flags=0x" << std::hex << flags << std::dec
                      << " blend=" << (flags & 1u)
                      << " srcBlend=" << ((flags >> 1) & 0xFu)
                      << " dstBlend=" << ((flags >> 5) & 0xFu)
                      << " test=" << ((flags >> 9) & 1u)
                      << " testFunc=" << kTestFunctions[(flags >> 10) & 7u]
                      << " noSorter=" << ((flags >> 13) & 1u)
                      << " threshold=" << static_cast<int>(threshold) << "\n";
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
}

// Dumps a .kf animation file's block layout and the raw words of its
// NiControllerSequence.
//
// Exists for the same reason --nifblocks does: the ControlledBlock layout is
// the one part of the KF format that is genuinely version-conditional (a
// string-palette offset before 20.1.0.3, a header string index after), and
// guessing which branch retail New Vegas took is exactly the mistake this file
// header warns about. --nifblocks cannot be used instead because its index is
// filtered to ".nif" and a .kf is not one.
int dumpKfAnimation(const std::filesystem::path& dataPath, const std::string& virtualPath) {
    odai::importer::fnv::FalloutAssetSource assets;
    if (!assets.open(dataPath)) {
        std::cout << "cannot index archives under " << dataPath << "\n";
        return 1;
    }
    std::vector<std::uint8_t> bytes;
    std::string error;
    if (!assets.resolveMesh(virtualPath, bytes, error)) {
        std::cout << "resolve failed: " << error << "\n";
        return 1;
    }
    odai::importer::fnv::NifBlockSummary summary;
    if (!odai::importer::fnv::parseNifBlockSummary(bytes, summary, error)) {
        std::cout << "header parse FAILED: " << error << "\n";
        return 1;
    }
    std::cout << virtualPath << ": " << bytes.size() << " bytes, "
              << summary.blockTypeNames.size() << " blocks, " << summary.strings.size()
              << " strings\n";

    std::map<std::string, std::size_t> typeCounts;
    for (const std::string& typeName : summary.blockTypeNames) {
        ++typeCounts[typeName];
    }
    std::cout << "block types:\n";
    for (const auto& [typeName, count] : typeCounts) {
        std::cout << "  " << count << "x " << typeName << "\n";
    }

    // The sequence block, word by word. Its header is a handful of floats and
    // refs followed by the controlled-block array, and reading the words as
    // both int and float side by side is what makes the boundary visible --
    // start/stop time are recognisable floats, refs are small ints.
    for (std::size_t i = 0; i < summary.blockTypeNames.size(); ++i) {
        if (summary.blockTypeNames[i] != "NiControllerSequence") {
            continue;
        }
        const std::size_t start = summary.blockStarts[i];
        const std::size_t words = summary.blockSizes[i] / 4u;
        std::cout << "[" << i << "] NiControllerSequence, " << summary.blockSizes[i]
                  << " bytes (" << words << " words):\n";
        for (std::size_t w = 0; w < words && w < 96u; ++w) {
            std::int32_t asInt = 0;
            float asFloat = 0.0f;
            std::memcpy(&asInt, bytes.data() + start + (w * 4u), 4u);
            std::memcpy(&asFloat, bytes.data() + start + (w * 4u), 4u);
            std::cout << "  w" << w << " int=" << asInt << " float=" << asFloat;
            if (asInt >= 0 && static_cast<std::size_t>(asInt) < summary.strings.size()) {
                std::cout << " str=\"" << summary.strings[static_cast<std::size_t>(asInt)] << "\"";
            }
            std::cout << "\n";
        }
    }

    // First interpolator/data pair, same treatment.
    for (const char* wanted : {"NiTransformInterpolator", "NiTransformData"}) {
        for (std::size_t i = 0; i < summary.blockTypeNames.size(); ++i) {
            if (summary.blockTypeNames[i] != wanted) {
                continue;
            }
            const std::size_t start = summary.blockStarts[i];
            const std::size_t words = summary.blockSizes[i] / 4u;
            std::cout << "[" << i << "] " << wanted << ", " << summary.blockSizes[i]
                      << " bytes (" << words << " words):\n";
            for (std::size_t w = 0; w < words && w < 40u; ++w) {
                std::int32_t asInt = 0;
                float asFloat = 0.0f;
                std::memcpy(&asInt, bytes.data() + start + (w * 4u), 4u);
                std::memcpy(&asFloat, bytes.data() + start + (w * 4u), 4u);
                std::cout << "  w" << w << " int=" << asInt << " float=" << asFloat << "\n";
            }
            break;
        }
    }

    // And what the reader itself makes of it -- the dump above is only useful
    // next to the parse it is supposed to justify.
    odai::importer::fnv::KfAnimation animation;
    if (!odai::importer::fnv::parseKfAnimation(bytes, animation, error)) {
        std::cout << "parseKfAnimation FAILED: " << error << "\n";
        return 1;
    }
    std::cout << "parsed \"" << animation.name << "\": " << animation.duration() << "s, "
              << (animation.loops() ? "looping" : "one-shot") << ", " << animation.tracks.size()
              << " tracks (" << animation.stats.transformInterpolators << " transform, "
              << animation.stats.unsupportedInterpolators << " unsupported, of "
              << animation.stats.controlledBlocks << " controlled blocks)\n";
    if (!animation.stats.unsupportedNodes.empty()) {
        std::cout << "  undecoded (B-spline) nodes:";
        for (const std::string& node : animation.stats.unsupportedNodes) {
            std::cout << " " << node;
        }
        std::cout << "\n";
    }
    std::size_t shownTracks = 0;
    for (const auto& track : animation.tracks) {
        if (shownTracks++ >= 8u) {
            break;
        }
        std::cout << "  " << track.nodeName << ": " << track.rotationKeys.size() << " rot, "
                  << track.translationKeys.size() << " trans, " << track.scaleKeys.size()
                  << " scale";
        if (!track.rotationKeys.empty()) {
            std::cout << "  t[" << track.rotationKeys.front().time << ".."
                      << track.rotationKeys.back().time << "]"
                      << " q0=(" << track.rotationKeys.front().value.x << ","
                      << track.rotationKeys.front().value.y << ","
                      << track.rotationKeys.front().value.z << ","
                      << track.rotationKeys.front().value.w << ")"
                      << " q1=(" << track.rotationKeys.back().value.x << ","
                      << track.rotationKeys.back().value.y << ","
                      << track.rotationKeys.back().value.z << ","
                      << track.rotationKeys.back().value.w << ")";
        }
        if (!track.translationKeys.empty()) {
            std::cout << " p0=(" << track.translationKeys.front().value.x << ","
                      << track.translationKeys.front().value.y << ","
                      << track.translationKeys.front().value.z << ")"
                      << " p1=(" << track.translationKeys.back().value.x << ","
                      << track.translationKeys.back().value.y << ","
                      << track.translationKeys.back().value.z << ")";
        }
        std::cout << "\n";
    }

    std::vector<odai::importer::fnv::KfAnimation> embeddedAnimations;
    if (odai::importer::fnv::parseNifEmbeddedAnimations(bytes, embeddedAnimations, error) &&
        embeddedAnimations.size() > 1u) {
        std::cout << "all embedded sequences:\n";
        for (const auto& embedded : embeddedAnimations) {
            std::cout << "  \"" << embedded.name << "\": " << embedded.duration()
                      << "s, " << (embedded.loops() ? "looping" : "one-shot")
                      << ", " << embedded.tracks.size() << " tracks\n";
            for (const auto& track : embedded.tracks) {
                std::cout << "    " << track.nodeName << ": "
                          << track.rotationKeys.size() << " rot, "
                          << track.translationKeys.size() << " trans, "
                          << track.scaleKeys.size() << " scale\n";
            }
        }
    }

    std::cout << "string table:\n";
    for (std::size_t i = 0; i < summary.strings.size() && i < 40u; ++i) {
        std::cout << "  [" << i << "] \"" << summary.strings[i] << "\"\n";
    }
    return 0;
}

// Parses every .kf under a folder and reports what the reader made of each.
//
// One file parsing is not evidence the layout is right -- it is evidence it is
// right for that file. The securitron alone ships 89 clips spanning every key
// type and both interpolator families, and a stride error shows up as a
// scattering of failures across them rather than a clean break.
int probeKfFolder(const std::filesystem::path& dataPath, const std::string& folderNeedle) {
    const std::string loweredNeedle = toLowerAscii(folderNeedle);
    std::size_t parsed = 0;
    std::size_t failed = 0;
    std::size_t noTracks = 0;
    std::size_t totalTracks = 0;
    std::size_t totalUnsupported = 0;
    std::map<std::string, std::size_t> failureReasons;

    for (const auto& archivePath : listArchives(dataPath)) {
        std::uint32_t contentFlags = 0;
        if (odai::importer::fnv::peekBsaContentFlags(archivePath, contentFlags) &&
            (contentFlags & odai::importer::fnv::kBsaContentMeshes) == 0u) {
            continue;
        }
        BsaArchive archive;
        if (!archive.open(archivePath)) {
            continue;
        }
        for (const BsaFileEntry& entry : archive.files()) {
            const std::string lowered = toLowerAscii(entry.virtualPath);
            if (lowered.size() < 4u || lowered.compare(lowered.size() - 3u, 3u, ".kf") != 0) {
                continue;
            }
            if (lowered.find(loweredNeedle) == std::string::npos) {
                continue;
            }
            std::vector<std::uint8_t> bytes;
            std::string error;
            if (!archive.extract(entry, bytes, error)) {
                ++failed;
                ++failureReasons["extract: " + error];
                continue;
            }
            odai::importer::fnv::KfAnimation animation;
            if (!odai::importer::fnv::parseKfAnimation(bytes, animation, error)) {
                ++failed;
                ++failureReasons[error];
                continue;
            }
            ++parsed;
            totalTracks += animation.tracks.size();
            totalUnsupported += animation.stats.unsupportedInterpolators;
            if (animation.tracks.empty()) {
                ++noTracks;
                std::cout << "  NO TRACKS: " << entry.virtualPath << "\n";
            }
        }
    }
    std::cout << parsed << " parsed, " << failed << " failed, " << noTracks
              << " parsed but empty\n"
              << "  " << totalTracks << " tracks total, " << totalUnsupported
              << " unsupported interpolators skipped\n";
    for (const auto& [reason, count] : failureReasons) {
        std::cout << "  " << count << "x " << reason << "\n";
    }
    return failed == 0 ? 0 : 1;
}

// Dumps a skeleton NIF's bone hierarchy as an indented tree.
//
// The tree shape is the check that matters and it is one a human has to make:
// a bone array can be complete, correctly named and still wrong, if the parent
// links put the forearm under the pelvis. Indentation makes that obvious at a
// glance in a way a bone count never does.
int probeSkeleton(const std::filesystem::path& dataPath, const std::string& virtualPath) {
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
        odai::importer::fnv::NifSkeleton skeleton;
        std::string error;
        if (!odai::importer::fnv::parseNifSkeleton(bytes, skeleton, error)) {
            std::cout << "skeleton parse FAILED: " << error << "\n";
            return 1;
        }
        std::cout << skeleton.bones.size() << " bones, " << skeleton.orphanedBoneCount
                  << " orphaned\n";
        std::vector<int> depth(skeleton.bones.size(), 0);
        for (std::size_t b = 0; b < skeleton.bones.size(); ++b) {
            const auto& bone = skeleton.bones[b];
            // Safe as a plain lookup rather than a walk precisely because the
            // array is topologically ordered -- the parent's depth is always
            // already computed.
            depth[b] = (bone.parentIndex >= 0) ? depth[static_cast<std::size_t>(bone.parentIndex)] + 1 : 0;
            std::cout << "  " << std::string(static_cast<std::size_t>(depth[b]) * 2u, ' ')
                      << (bone.name.empty() ? "<unnamed>" : bone.name) << "  t("
                      << bone.translation[0] << ", " << bone.translation[1] << ", "
                      << bone.translation[2] << ")";
            if (bone.scale != 1.0f) {
                std::cout << " scale=" << bone.scale;
            }
            std::cout << "\n";
        }
        return 0;
    }
    std::cout << "Not found in any mesh archive: " << virtualPath << "\n";
    return 1;
}

// Dumps a skinned mesh's shapes: their bone bindings and weight distribution.
//
// The influence histogram is the part worth printing. Skinning quality lives
// or dies on whether truncating to four influences is throwing away real
// weight, and that is invisible in any per-shape summary -- a mesh where 3% of
// vertices lose a 0.01 influence is fine, and one where 30% lose a 0.2 is not.
int probeSkinnedNif(const std::filesystem::path& dataPath, const std::string& virtualPath) {
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
        odai::importer::fnv::NifSkinnedModel model;
        std::string error;
        if (!odai::importer::fnv::parseNifSkinnedMesh(bytes, model, error)) {
            std::cout << "skinned parse FAILED: " << error << "\n";
            return 1;
        }
        std::cout << model.shapes.size() << " skinned shape(s), " << model.unskinnedShapeCount
                  << " unskinned, " << model.truncatedInfluenceVertexCount
                  << " vertices truncated to 4 influences\n";
        for (const auto& shape : model.shapes) {
            const std::size_t vertexCount = shape.positions.size() / 3u;
            std::cout << "  \"" << shape.name << "\" verts " << vertexCount << ", tris "
                      << (shape.triangleIndices.size() / 3u) << ", bones " << shape.boneNames.size()
                      << ", diffuse \"" << shape.diffuseTexturePath << "\"\n";
            // Where the geometry actually sits, in the file's own space, and
            // the transform that is supposed to carry it into the skeleton's.
            // A part in the wrong place on screen is one or the other, and the
            // two need different fixes.
            float rawMin[3] = {1e30f, 1e30f, 1e30f};
            float rawMax[3] = {-1e30f, -1e30f, -1e30f};
            for (std::size_t v = 0; v + 2u < shape.positions.size(); v += 3u) {
                for (int a2 = 0; a2 < 3; ++a2) {
                    rawMin[a2] = std::min(rawMin[a2], shape.positions[v + static_cast<std::size_t>(a2)]);
                    rawMax[a2] = std::max(rawMax[a2], shape.positions[v + static_cast<std::size_t>(a2)]);
                }
            }
            std::cout << "      skin-space bounds (" << rawMin[0] << ".." << rawMax[0] << ", "
                      << rawMin[1] << ".." << rawMax[1] << ", " << rawMin[2] << ".." << rawMax[2]
                      << ")  skinTransform t(" << shape.skinTransform[3] << ", "
                      << shape.skinTransform[7] << ", " << shape.skinTransform[11] << ")\n";
            // How many bones each vertex actually uses, after truncation. A
            // column at 1 on a body mesh means the weights did not parse.
            std::size_t byCount[odai::importer::fnv::kNifMaxBoneInfluences + 1] = {};
            double weightSumMin = 2.0;
            double weightSumMax = 0.0;
            for (std::size_t v = 0; v < vertexCount; ++v) {
                int used = 0;
                double sum = 0.0;
                for (int k = 0; k < odai::importer::fnv::kNifMaxBoneInfluences; ++k) {
                    const float w = shape.boneWeights[(v * odai::importer::fnv::kNifMaxBoneInfluences) + static_cast<std::size_t>(k)];
                    if (w > 0.0f) {
                        ++used;
                        sum += w;
                    }
                }
                ++byCount[used];
                weightSumMin = std::min(weightSumMin, sum);
                weightSumMax = std::max(weightSumMax, sum);
            }
            std::cout << "      influences/vertex:";
            for (int k = 0; k <= odai::importer::fnv::kNifMaxBoneInfluences; ++k) {
                std::cout << " " << k << "=" << byCount[k];
            }
            std::cout << "   weight sum [" << weightSumMin << ", " << weightSumMax << "]\n";
            std::cout << "      bones:";
            for (std::size_t b = 0; b < shape.boneNames.size() && b < 8u; ++b) {
                std::cout << " " << shape.boneNames[b];
            }
            if (shape.boneNames.size() > 8u) {
                std::cout << " ...(+" << (shape.boneNames.size() - 8u) << ")";
            }
            std::cout << "\n";
        }
        return 0;
    }
    std::cout << "Not found in any mesh archive: " << virtualPath << "\n";
    return 1;
}

// The whole character path end to end: skeleton NIF + body-part NIFs -> one
// bound, engine-space, GPU-skinnable mesh.
//
// Every number printed here is a thing that can be wrong silently. The two that
// matter most: unresolvedBones > 0 means vertices are bound to bones the
// skeleton does not have (they collapse to the root), and a bind-pose bounding
// box that is not roughly human-sized and standing on y=0 means the basis
// change or the inverse binds are wrong -- which looks like an exploded mesh on
// screen and like nothing at all in a vertex count.
int probeCharacter(
    const std::filesystem::path& dataPath, const std::string& skeletonPath,
    const std::vector<std::string>& partPaths) {
    // Paths here are relative to meshes\, matching FalloutAssetSource and the
    // cooker -- NOT the full "meshes\..." virtual path --nif and --nifblocks
    // take. Those two go straight at the archive index; this goes through the
    // same resolver the game does, loose files and all.
    odai::importer::fnv::FalloutAssetSource assets;
    if (!assets.open(dataPath)) {
        std::cout << "could not index archives under " << dataPath << "\n";
        return 1;
    }

    std::string resolveError;
    std::vector<std::uint8_t> skeletonBytes;
    if (!assets.resolveMesh(skeletonPath, skeletonBytes, resolveError)) {
        std::cout << "could not resolve skeleton: " << skeletonPath << " (" << resolveError << ")\n";
        return 1;
    }
    odai::importer::fnv::NifSkeleton nifSkeleton;
    std::string error;
    if (!odai::importer::fnv::parseNifSkeleton(skeletonBytes, nifSkeleton, error)) {
        std::cout << "skeleton parse FAILED: " << error << "\n";
        return 1;
    }
    odai::importer::fnv::FalloutCharacter character;
    if (!odai::importer::fnv::buildFalloutSkeleton(nifSkeleton, character.skeleton)) {
        std::cout << "skeleton conversion FAILED\n";
        return 1;
    }
    std::cout << "skeleton: " << character.skeleton.bones.size() << " bones ("
              << nifSkeleton.orphanedBoneCount << " orphaned)\n";

    for (const std::string& partPath : partPaths) {
        std::vector<std::uint8_t> partBytes;
        if (!assets.resolveMesh(partPath, partBytes, resolveError)) {
            std::cout << "  could not resolve part: " << partPath << " (" << resolveError << ")\n";
            continue;
        }
        odai::importer::fnv::NifSkinnedModel model;
        if (!odai::importer::fnv::parseNifSkinnedMesh(partBytes, model, error)) {
            std::cout << "  part parse FAILED (" << partPath << "): " << error << "\n";
            continue;
        }
        const std::size_t partsBefore = character.parts.size();
        if (!odai::importer::fnv::appendFalloutCharacterMesh(model, character, error)) {
            std::cout << "  bind FAILED (" << partPath << "): " << error << "\n";
            continue;
        }
        std::cout << "  " << partPath << " -> " << (character.parts.size() - partsBefore)
                  << " part(s), " << model.truncatedInfluenceVertexCount << " truncated\n";
    }

    std::cout << "bound: " << character.vertices.size() << " vertices, "
              << (character.indices.size() / 3u) << " triangles, " << character.parts.size()
              << " parts\n"
              << "  unresolvedBones=" << character.unresolvedBoneCount
              << " conflictingInverseBinds=" << character.conflictingInverseBindCount << "\n";

    // Skin the bind pose on the CPU and measure the result. If the inverse
    // binds and the basis change agree, this reproduces the rest pose exactly,
    // so the skinned bounds must match the raw vertex bounds. They are printed
    // together because the comparison is the test.
    std::vector<odai::math::Matrix4> boneMatrices;
    odai::importer::fnv::computeFalloutBindPose(character, boneMatrices);

    float rawMin[3] = {1e30f, 1e30f, 1e30f};
    float rawMax[3] = {-1e30f, -1e30f, -1e30f};
    float skinnedMin[3] = {1e30f, 1e30f, 1e30f};
    float skinnedMax[3] = {-1e30f, -1e30f, -1e30f};
    double maxDrift = 0.0;
    for (const auto& vertex : character.vertices) {
        for (int a = 0; a < 3; ++a) {
            rawMin[a] = std::min(rawMin[a], vertex.position[a]);
            rawMax[a] = std::max(rawMax[a], vertex.position[a]);
        }
        // Exactly what the skinning compute shader does: a weighted sum of the
        // bone matrices applied to the rest position.
        odai::math::Vector3 skinned{0.0f, 0.0f, 0.0f};
        const odai::math::Vector3 rest{vertex.position[0], vertex.position[1], vertex.position[2]};
        for (int k = 0; k < odai::importer::fnv::kNifMaxBoneInfluences; ++k) {
            const float weight = vertex.boneWeights[k];
            if (weight <= 0.0f) {
                continue;
            }
            const std::size_t bone = vertex.boneIndices[k];
            if (bone >= boneMatrices.size()) {
                continue;
            }
            const odai::math::Vector3 contribution =
                odai::math::transformPoint(boneMatrices[bone], rest);
            skinned.x += contribution.x * weight;
            skinned.y += contribution.y * weight;
            skinned.z += contribution.z * weight;
        }
        const float skinnedArray[3] = {skinned.x, skinned.y, skinned.z};
        for (int a = 0; a < 3; ++a) {
            skinnedMin[a] = std::min(skinnedMin[a], skinnedArray[a]);
            skinnedMax[a] = std::max(skinnedMax[a], skinnedArray[a]);
            maxDrift = std::max(maxDrift, static_cast<double>(std::fabs(skinnedArray[a] - vertex.position[a])));
        }
    }
    // Per part, because a single merged bounding box hides the one part that is
    // in the wrong place -- and "one part misplaced" and "the whole rig is
    // wrong" need completely different fixes.
    std::cout << "  per-part skinned bounds:\n";
    for (const auto& part : character.parts) {
        float partMin[3] = {1e30f, 1e30f, 1e30f};
        float partMax[3] = {-1e30f, -1e30f, -1e30f};
        for (std::uint32_t idx = part.firstIndex; idx < part.firstIndex + part.indexCount; ++idx) {
            const std::uint32_t vertexIndex = character.indices[idx];
            const auto& vertex = character.vertices[vertexIndex];
            odai::math::Vector3 skinned{0.0f, 0.0f, 0.0f};
            const odai::math::Vector3 rest{vertex.position[0], vertex.position[1], vertex.position[2]};
            for (int k = 0; k < odai::importer::fnv::kNifMaxBoneInfluences; ++k) {
                const float weight = vertex.boneWeights[k];
                if (weight <= 0.0f) {
                    continue;
                }
                const std::size_t bone = vertex.boneIndices[k];
                if (bone >= boneMatrices.size()) {
                    continue;
                }
                const odai::math::Vector3 c = odai::math::transformPoint(boneMatrices[bone], rest);
                skinned.x += c.x * weight;
                skinned.y += c.y * weight;
                skinned.z += c.z * weight;
            }
            const float values[3] = {skinned.x, skinned.y, skinned.z};
            for (int a = 0; a < 3; ++a) {
                partMin[a] = std::min(partMin[a], values[a]);
                partMax[a] = std::max(partMax[a], values[a]);
            }
        }
        std::cout << "    \"" << part.name << "\"  y " << partMin[1] << ".." << partMax[1]
                  << "  x " << partMin[0] << ".." << partMax[0]
                  << "  z " << partMin[2] << ".." << partMax[2] << "\n";
    }
    std::cout << "  rest bounds     min(" << rawMin[0] << ", " << rawMin[1] << ", " << rawMin[2]
              << ") max(" << rawMax[0] << ", " << rawMax[1] << ", " << rawMax[2] << ")\n"
              << "  bind-pose skin  min(" << skinnedMin[0] << ", " << skinnedMin[1] << ", "
              << skinnedMin[2] << ") max(" << skinnedMax[0] << ", " << skinnedMax[1] << ", "
              << skinnedMax[2] << ")\n"
              << "  max per-vertex drift " << maxDrift
              << "   (near zero means the bind pose round-trips: inverse binds and"
                 " basis change agree)\n";
    return 0;
}

// Regions, and which cells belong to them.
//
// Prints the discoverable set (RDMP-bearing) against the total, because the gap
// between the two is the whole point: announcing every REGN would fire on
// weather and audio zones the player is not meant to see named.
int probeRegions(const std::filesystem::path& pluginPath, std::size_t limit) {
    using namespace odai::importer::fnv;
    std::string error;
    FalloutWorldTables tables;
    if (!buildFalloutWorldTables(pluginPath, tables, error)) {
        std::cout << "world tables FAILED: " << error << "\n";
        return 1;
    }
    FalloutCellIndex index;
    if (!buildFalloutCellIndex(pluginPath, index, error)) {
        std::cout << "cell index FAILED: " << error << "\n";
        return 1;
    }

    std::map<std::uint32_t, std::size_t> cellsPerRegion;
    std::size_t cellsWithAnyRegion = 0;
    std::size_t cellsWithDiscoverableRegion = 0;
    for (const FalloutCellIndexEntry& cell : index.cells) {
        if (cell.regionFormIds.empty()) {
            continue;
        }
        ++cellsWithAnyRegion;
        bool discoverable = false;
        for (const std::uint32_t regionFormId : cell.regionFormIds) {
            ++cellsPerRegion[regionFormId];
            discoverable = discoverable || tables.regionNamesByFormId.count(regionFormId) != 0u;
        }
        if (discoverable) {
            ++cellsWithDiscoverableRegion;
        }
    }
    std::cout << tables.regionNamesByFormId.size() << " discoverable region(s) (RDMP-bearing)\n"
              << index.cells.size() << " cells, " << cellsWithAnyRegion << " in at least one region, "
              << cellsWithDiscoverableRegion << " in at least one DISCOVERABLE region\n";

    // Cell-grid centroid per named region, so a traversal test can be aimed at
    // one. Without a location a region name is unusable for navigation --
    // "walk until something happens" is not a test.
    struct RegionExtent {
        std::string name;
        std::size_t cellCount = 0;
        long long sumX = 0;
        long long sumZ = 0;
    };
    std::map<std::uint32_t, RegionExtent> extents;
    for (const FalloutCellIndexEntry& cell : index.cells) {
        if (!cell.hasGridCoords) {
            continue;
        }
        for (const std::uint32_t regionFormId : cell.regionFormIds) {
            const auto named = tables.regionNamesByFormId.find(regionFormId);
            if (named == tables.regionNamesByFormId.end()) {
                continue;
            }
            RegionExtent& extent = extents[regionFormId];
            extent.name = named->second;
            ++extent.cellCount;
            extent.sumX += cell.gridX;
            extent.sumZ += cell.gridZ;
        }
    }
    std::vector<RegionExtent> sorted;
    sorted.reserve(extents.size());
    for (const auto& [formId, extent] : extents) {
        sorted.push_back(extent);
    }
    std::sort(sorted.begin(), sorted.end(), [](const RegionExtent& a, const RegionExtent& b) {
        return a.cellCount > b.cellCount;
    });
    for (std::size_t i = 0; i < sorted.size() && i < limit; ++i) {
        const RegionExtent& extent = sorted[i];
        std::cout << "  " << extent.cellCount << " cells  centroid cell ("
                  << (extent.sumX / static_cast<long long>(extent.cellCount)) << ","
                  << (extent.sumZ / static_cast<long long>(extent.cellCount)) << ")  \""
                  << extent.name << "\"\n";
    }
    return 0;
}

// Decoded alpha histogram for one texture.
//
// "The shape has alphaTest and vanishes" has two very different causes -- an
// authored cutout working as intended, or an alpha channel that decoded to
// nothing -- and only the decoded bytes tell them apart.
int probeTexture(const std::filesystem::path& dataPath, const std::string& texturePath) {
    using namespace odai::importer;
    fnv::FalloutAssetSource assets;
    if (!assets.open(dataPath)) {
        std::cout << "could not index archives under " << dataPath << "\n";
        return 1;
    }
    std::vector<std::uint8_t> ddsBytes;
    std::string error;
    if (!assets.resolveTexture(texturePath, ddsBytes, error)) {
        std::cout << "resolve failed: " << error << "\n";
        return 1;
    }
    // Same escape hatch --nif has, for the same reason: these files live inside
    // a BSA, so nothing outside this tool can look at one. A cloud layer's
    // LAYOUT -- fisheye dome map or tiling plane -- is not a number this mode
    // can print, and getting it wrong renders a seam rather than an error.
    if (const char* dumpPath = std::getenv("ODAI_NIF_DUMP")) {
        std::ofstream out(dumpPath, std::ios::binary);
        out.write(reinterpret_cast<const char*>(ddsBytes.data()),
                  static_cast<std::streamsize>(ddsBytes.size()));
        std::cout << "wrote " << ddsBytes.size() << " bytes to " << dumpPath << "\n";
    }
    ImportedSceneTexture texture;
    if (!loadDdsFromMemory(ddsBytes.data(), ddsBytes.size(), texture)) {
        std::cout << "DDS decode failed\n";
        return 1;
    }
    const char* formatName = "?";
    switch (texture.format) {
        case TextureFormat::RGBA8: formatName = "RGBA8"; break;
        case TextureFormat::RGBA8Srgb: formatName = "RGBA8-sRGB"; break;
        case TextureFormat::BC1: formatName = "BC1"; break;
        case TextureFormat::BC3: formatName = "BC3"; break;
        case TextureFormat::BC5: formatName = "BC5"; break;
        case TextureFormat::BC7: formatName = "BC7"; break;
        default: break;
    }
    std::cout << texturePath << ": " << texture.width << "x" << texture.height
              << " format=" << formatName << " mips=" << texture.mipLevelCount
              << " bytes=" << ddsBytes.size() << "\n";
    // The GPU samples the compressed data directly, so decode base mip 0 to
    // RGBA only when the loader already did; otherwise report what is known.
    if ((texture.format != TextureFormat::RGBA8 &&
         texture.format != TextureFormat::RGBA8Srgb) || texture.rgba8.empty()) {
        // Decode the block alpha into the same three bands the cutout
        // classifier uses (imported_scene.cc AlphaBandCounts), so this mode
        // answers "would the importer infer alpha test from this texture".
        const std::size_t blockCount =
            (static_cast<std::size_t>(texture.width) + 3u) / 4u *
            ((static_cast<std::size_t>(texture.height) + 3u) / 4u);
        std::size_t low = 0;
        std::size_t mid = 0;
        std::size_t high = 0;
        const auto addAlpha = [&](std::uint8_t a) {
            if (a < 32u) { ++low; } else if (a > 224u) { ++high; } else { ++mid; }
        };
        if (texture.format == TextureFormat::BC1 && texture.rgba8.size() >= blockCount * 8u) {
            for (std::size_t b = 0; b < blockCount; ++b) {
                const std::uint8_t* block = texture.rgba8.data() + (b * 8u);
                const std::uint16_t c0 = static_cast<std::uint16_t>(block[0] | (block[1] << 8));
                const std::uint16_t c1 = static_cast<std::uint16_t>(block[2] | (block[3] << 8));
                const bool punchThrough = c0 <= c1;
                for (int byteIndex = 4; byteIndex < 8; ++byteIndex) {
                    std::uint8_t bits = block[byteIndex];
                    for (int t = 0; t < 4; ++t) {
                        addAlpha((punchThrough && ((bits & 0x3u) == 0x3u)) ? 0u : 255u);
                        bits >>= 2;
                    }
                }
            }
        } else if (texture.format == TextureFormat::BC3 && texture.rgba8.size() >= blockCount * 16u) {
            for (std::size_t b = 0; b < blockCount; ++b) {
                const std::uint8_t* block = texture.rgba8.data() + (b * 16u);
                const std::uint8_t a0 = block[0];
                const std::uint8_t a1 = block[1];
                std::uint8_t palette[8] = {a0, a1};
                if (a0 > a1) {
                    for (int s = 1; s <= 6; ++s) {
                        palette[1 + s] = static_cast<std::uint8_t>(((7 - s) * a0 + s * a1) / 7);
                    }
                } else {
                    for (int s = 1; s <= 4; ++s) {
                        palette[1 + s] = static_cast<std::uint8_t>(((5 - s) * a0 + s * a1) / 5);
                    }
                    palette[6] = 0u;
                    palette[7] = 255u;
                }
                std::uint64_t indexBits = 0;
                for (int i = 0; i < 6; ++i) {
                    indexBits |= static_cast<std::uint64_t>(block[2 + i]) << (8 * i);
                }
                for (int t = 0; t < 16; ++t) {
                    addAlpha(palette[(indexBits >> (3 * t)) & 0x7u]);
                }
            }
        } else {
            std::cout << "  (compressed on the GPU; alpha lives in the block data)\n";
            return 0;
        }
        const std::size_t total = low + mid + high;
        const double lowFraction = total ? static_cast<double>(low) / static_cast<double>(total) : 0.0;
        const double midFraction = total ? static_cast<double>(mid) / static_cast<double>(total) : 0.0;
        std::cout << "  block alpha bands: low(<32)=" << low << " mid=" << mid
                  << " high(>224)=" << high << " of " << total << "\n"
                  << "  lowFraction=" << lowFraction << " midFraction=" << midFraction
                  << " -> classifier verdict: "
                  << ((lowFraction >= 0.01 && midFraction <= 0.20) ? "CUTOUT (alpha test inferred)"
                                                                   : "opaque")
                  << "\n";
        return 0;
    }
    std::array<std::size_t, 5> buckets{};
    const std::size_t pixelCount =
        static_cast<std::size_t>(texture.width) * static_cast<std::size_t>(texture.height);
    for (std::size_t i = 0; i < pixelCount && (i * 4u) + 3u < texture.rgba8.size(); ++i) {
        const std::uint8_t alpha = texture.rgba8[(i * 4u) + 3u];
        if (alpha == 0u) { ++buckets[0]; }
        else if (alpha < 64u) { ++buckets[1]; }
        else if (alpha < 128u) { ++buckets[2]; }
        else if (alpha < 255u) { ++buckets[3]; }
        else { ++buckets[4]; }
    }
    std::cout << "  alpha: zero=" << buckets[0] << " <64=" << buckets[1] << " <128=" << buckets[2]
              << " <255=" << buckets[3] << " opaque=" << buckets[4] << " of " << pixelCount << "\n";
    return 0;
}

// Census of the NIF FOOTER across the archives.
//
// The importer picks its roots by promoting every node no other node claims as
// a child, each walked with an identity transform -- so an unrecognized or
// unparsed parent silently relocates its whole subtree to the model origin.
// NIF states its roots explicitly in a footer instead. Before making root
// selection depend on that footer, this measures whether it is actually there
// and well-formed across real content, rather than trusting the format spec.
int probeFooters(const std::filesystem::path& dataPath, std::size_t limit) {
    MeshIndex index = buildMeshIndex(dataPath);
    std::cout << "Scanning footers across " << index.nifs.size() << " .nif entries (limit "
              << limit << ").\n";

    std::size_t examined = 0;
    std::size_t extractFailures = 0;
    std::size_t summaryFailures = 0;
    std::size_t noBlocks = 0;
    std::size_t footerPastEof = 0;
    std::size_t implausible = 0;
    std::size_t exactBytes = 0;
    std::size_t rootOutOfRange = 0;
    std::size_t rootIsBlockZero = 0;
    std::map<std::uint32_t, std::size_t> rootCountHistogram;
    std::map<std::string, std::size_t> rootBlockTypes;
    std::vector<std::string> anomalies;

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
        odai::importer::fnv::NifBlockSummary summary;
        std::string error;
        if (!odai::importer::fnv::parseNifBlockSummary(bytes, summary, error)) {
            ++summaryFailures;
            continue;
        }
        if (summary.blockStarts.empty()) {
            ++noBlocks;
            continue;
        }
        const std::size_t footerOffset = summary.blockStarts.back() + summary.blockSizes.back();
        if (footerOffset + 4u > bytes.size()) {
            ++footerPastEof;
            if (anomalies.size() < 12u) {
                anomalies.push_back("past-eof: " + entry->virtualPath);
            }
            continue;
        }
        std::uint32_t rootCount = 0;
        std::memcpy(&rootCount, bytes.data() + footerOffset, 4u);
        const std::size_t expected = 4u + (static_cast<std::size_t>(rootCount) * 4u);
        if (rootCount > 64u || footerOffset + expected > bytes.size()) {
            ++implausible;
            if (anomalies.size() < 12u) {
                anomalies.push_back("implausible numRoots=" + std::to_string(rootCount) + ": " +
                                    entry->virtualPath);
            }
            continue;
        }
        ++rootCountHistogram[rootCount];
        if (expected == (bytes.size() - footerOffset)) {
            ++exactBytes;
        }
        for (std::uint32_t r = 0; r < rootCount; ++r) {
            std::int32_t rootRef = 0;
            std::memcpy(&rootRef, bytes.data() + footerOffset + 4u + (r * 4u), 4u);
            if (rootRef < 0 || static_cast<std::size_t>(rootRef) >= summary.blockTypeNames.size()) {
                ++rootOutOfRange;
                if (anomalies.size() < 12u) {
                    anomalies.push_back("root out of range: " + entry->virtualPath);
                }
                continue;
            }
            if (rootRef == 0) {
                ++rootIsBlockZero;
            }
            ++rootBlockTypes[summary.blockTypeNames[static_cast<std::size_t>(rootRef)]];
        }
    }

    std::cout << "examined " << examined << ": extractFail=" << extractFailures
              << " summaryFail=" << summaryFailures << " noBlocks=" << noBlocks
              << " footerPastEof=" << footerPastEof << " implausible=" << implausible << "\n"
              << "  footers whose size is EXACTLY 4+4*numRoots: " << exactBytes << "\n"
              << "  root refs out of range: " << rootOutOfRange
              << ", roots that are block 0: " << rootIsBlockZero << "\n";
    std::cout << "numRoots histogram:";
    for (const auto& [count, files] : rootCountHistogram) {
        std::cout << " " << count << "x" << files;
    }
    std::cout << "\nroot block types:";
    for (const auto& [typeName, count] : rootBlockTypes) {
        std::cout << " " << typeName << "=" << count;
    }
    std::cout << "\n";
    if (!anomalies.empty()) {
        std::cout << "anomalies (first " << anomalies.size() << "):\n";
        for (const std::string& a : anomalies) {
            std::cout << "  " << a << "\n";
        }
    }
    return 0;
}

// Independent ground truth for the VHGT decode. The game ships its own distant
// terrain as meshes -- meshes\landscape\lod\<worldspace>\blocks\*.nif -- baked
// by the GECK from the same LAND records the streamer reads. Their vertices are
// already in world space (a level4 block's bounds land exactly on its 4x4 cell
// footprint), so the height they report at an XY can be compared directly with
// the height our own decode produces there.
//
// This is what answers "is the terrain wrong or is the reference high", which
// clearances alone cannot: every placed reference in a cell is positioned
// relative to the same terrain, so if that terrain is wrong they all agree with
// each other and still disagree with the game.
//
// Caveat worth stating up front: level4 is the FINEST LOD and is still ~20x
// coarser than the full 33x33-per-cell heightfield (855 vertices over 4x4
// cells, ~585 units apart). It resolves a hillside, not a doorstep -- so a
// disagreement of tens of units means nothing and one of ~170 means a great
// deal.
int lodHeightAt(const std::filesystem::path& dataPath, const std::string& nifPath,
                float worldX, float worldY) {
    odai::importer::fnv::FalloutAssetSource assets;
    if (!assets.open(dataPath)) {
        std::cout << "asset source open FAILED\n";
        return 1;
    }
    std::vector<std::uint8_t> nifBytes;
    std::string error;
    if (!assets.resolveMesh(nifPath, nifBytes, error)) {
        std::cout << "resolve FAILED: " << error << "\n";
        return 1;
    }
    odai::importer::fnv::NifModel model;
    if (!odai::importer::fnv::parseNifStaticMesh(nifBytes, model, error)) {
        std::cout << "parse FAILED: " << error << "\n";
        return 1;
    }
    struct Near {
        float distanceSquared;
        float x;
        float y;
        float z;
    };
    // A LOD block carries a SKIRT: a second ring of vertices at the block's
    // edge dropped far below the surface, so neighbouring blocks cannot show a
    // crack between them. They sit at the same XY as the surface vertex above
    // them and at the block's minimum Z (-14098 in the x-20.y0 block, against a
    // terrain around 8200), so averaging them in reports a height thousands of
    // units below anything real. Collapsing each XY to its HIGHEST vertex drops
    // the skirt and keeps the surface, without needing to know the skirt depth.
    std::map<std::pair<int, int>, float> surfaceByXy;
    for (const odai::importer::fnv::NifShape& shape : model.shapes) {
        const std::size_t vertexCount = shape.positions.size() / 3u;
        for (std::size_t v = 0; v < vertexCount; ++v) {
            const float vx = shape.positions[(v * 3u) + 0];
            const float vy = shape.positions[(v * 3u) + 1];
            const float vz = shape.positions[(v * 3u) + 2];
            const std::pair<int, int> key{static_cast<int>(std::lround(vx)),
                                          static_cast<int>(std::lround(vy))};
            const auto existing = surfaceByXy.find(key);
            if (existing == surfaceByXy.end() || vz > existing->second) {
                surfaceByXy[key] = vz;
            }
        }
    }
    std::vector<Near> nearest;
    nearest.reserve(surfaceByXy.size());
    for (const auto& [xy, z] : surfaceByXy) {
        const float dx = static_cast<float>(xy.first) - worldX;
        const float dy = static_cast<float>(xy.second) - worldY;
        nearest.push_back(
            Near{(dx * dx) + (dy * dy), static_cast<float>(xy.first),
                 static_cast<float>(xy.second), z});
    }
    if (nearest.empty()) {
        std::cout << "no vertices in " << nifPath << "\n";
        return 1;
    }
    std::sort(nearest.begin(), nearest.end(),
              [](const Near& a, const Near& b) { return a.distanceSquared < b.distanceSquared; });
    std::cout << nifPath << ": " << nearest.size() << " surface vertices (skirt removed)\n";
    std::cout << "  query (" << worldX << ", " << worldY << ")\n";
    const std::size_t show = std::min<std::size_t>(nearest.size(), 8u);
    // Inverse-distance blend of the nearest few, which is the best a scattered
    // LOD grid supports -- it is not a regular lattice this can bilerp on.
    double weightSum = 0.0;
    double heightSum = 0.0;
    for (std::size_t i = 0; i < show; ++i) {
        const float distance = std::sqrt(nearest[i].distanceSquared);
        std::cout << "    d=" << static_cast<int>(distance) << "  ("
                  << static_cast<int>(nearest[i].x) << ", " << static_cast<int>(nearest[i].y)
                  << ")  z=" << nearest[i].z << "\n";
        const double weight = 1.0 / std::max(1.0, static_cast<double>(distance));
        weightSum += weight;
        heightSum += weight * static_cast<double>(nearest[i].z);
    }
    std::cout << "  LOD height at query (inverse-distance over " << show
              << " nearest) = " << (heightSum / std::max(1e-9, weightSum)) << "\n";
    return 0;
}

int probeSingleNif(const std::filesystem::path& dataPath, const std::string& virtualPath) {
    odai::importer::fnv::FalloutAssetSource assets;
    if (!assets.open(dataPath)) {
        std::cout << "cannot index assets under " << dataPath << "\n";
        return 1;
    }
    assets.addModDirectory(dataPath);
    std::vector<std::uint8_t> bytes;
    std::string resolveError;
    if (!assets.resolveMesh(virtualPath, bytes, resolveError)) {
        std::cout << "resolve failed: " << resolveError << "\n";
        return 1;
    }
    {
        std::cout << "Extracted " << bytes.size() << " bytes.\n";
        // ODAI_NIF_DUMP writes the decompressed asset out as-is. Everything an
        // unsupported header needs is in its first hundred bytes, and there is
        // otherwise no way to look at them: these files live inside a BSA, so a
        // hex editor -- or NifSkope -- cannot reach one without this.
        if (const char* dumpPath = std::getenv("ODAI_NIF_DUMP")) {
            std::ofstream out(dumpPath, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
            std::cout << "wrote " << bytes.size() << " bytes to " << dumpPath << "\n";
        }
        odai::importer::fnv::NifModel model;
        std::string error;
        const bool ok = odai::importer::fnv::parseNifStaticMesh(bytes, model, error);
        std::cout << "parse " << (ok ? "ok" : "FAILED") << (error.empty() ? "" : (": " + error)) << "\n"
                  << "shapes " << model.shapes.size() << ", skipped " << model.skippedShapeCount
                  << ", editor markers " << model.editorMarkerShapeCount
                  << ", hidden " << model.hiddenShapeCount
                  << ", failed nodes " << model.nodeParseFailedCount
                  << ", authored collision triangles " << model.collisionTriangles.size() << "\n";
        if (!model.failedNodeTypes.empty()) {
            std::cout << "  failed node types:";
            for (const std::string& type : model.failedNodeTypes) {
                std::cout << " " << type;
            }
            std::cout << "\n";
        }
        for (const odai::importer::fnv::KfAnimation& animation : model.embeddedAnimations) {
            std::cout << "  animation \"" << animation.name << "\": "
                      << animation.duration() << "s, "
                      << (animation.loops() ? "looping" : "one-shot") << ", "
                      << animation.tracks.size() << " tracks\n";
        }
        for (const odai::importer::fnv::NifShape& shape : model.shapes) {
            std::cout << "  \"" << shape.name << "\" verts " << (shape.positions.size() / 3u) << ", tris "
                      << (shape.triangleIndices.size() / 3u) << ", uvs " << (shape.uvs.size() / 2u)
                      << ", block=" << shape.sourceBlockType
                      << ", sourceTris=" << shape.sourceTriangleCount
                      << ", rejectedTris=" << shape.rejectedTriangleCount
                      << ", alphaTest=" << (shape.alphaTest ? "yes" : "no")
                      << (shape.alphaTest
                              ? (" thr=" + std::to_string(static_cast<int>(shape.alphaThreshold)))
                              : std::string())
                      << ", twoSided=" << (shape.twoSided ? "yes" : "no")
                      << ", alphaBlend=" << (shape.alphaBlend ? "yes" : "no")
                      << ", alphaSemantic=" << alphaSemanticName(shape.alphaSemantic)
                      << ", diffuse=\"" << shape.diffuseTexturePath << "\""
                      << (shape.animationNodeName.empty()
                              ? std::string()
                              : (", animated-by=\"" + shape.animationNodeName + "\""))
                      << "\n";
            if (!shape.animationNodeName.empty()) {
                std::cout << "      animation parent translation=("
                          << shape.animationParentTransform[3] << ","
                          << shape.animationParentTransform[7] << ","
                          << shape.animationParentTransform[11] << ") bind translation=("
                          << shape.animationBindTransform[3] << ","
                          << shape.animationBindTransform[7] << ","
                          << shape.animationBindTransform[11] << ")\n";
            }
            // Vertex-alpha census. This is the channel that feathers a placed
            // road into the ground under it, so "does this shape have one, and
            // is it actually varying" is the question a hard-edged road asks.
            // A constant 1.0 is not a feather; a spread is.
            if (!shape.colors.empty()) {
                float minAlpha = 1.0F;
                float maxAlpha = 0.0F;
                std::size_t fadedVertices = 0;
                for (std::size_t v = 3u; v < shape.colors.size(); v += 4u) {
                    const float alpha = shape.colors[v];
                    minAlpha = std::min(minAlpha, alpha);
                    maxAlpha = std::max(maxAlpha, alpha);
                    if (alpha < 0.99F) {
                        ++fadedVertices;
                    }
                }
                std::cout << "      vertex color: yes, alpha [" << minAlpha << ", " << maxAlpha
                          << "] faded=" << fadedVertices << "/" << (shape.colors.size() / 4u)
                          << "\n";
            } else {
                std::cout << "      vertex color: none\n";
            }
            // Winding orientation: signed volume via the divergence theorem,
            // plus the fraction of faces whose geometric normal agrees with
            // the shape's authored vertex normals. A closed shell wound
            // consistently outward has a strongly positive signed volume and
            // an agreement fraction near 1; mixed winding sits near 0.5 and
            // is what back-face culling turns into localized holes.
            if (!shape.triangleIndices.empty() && !shape.positions.empty()) {
                double signedVolume6 = 0.0;
                std::size_t agree = 0;
                std::size_t counted = 0;
                for (std::size_t t = 0; t + 2 < shape.triangleIndices.size(); t += 3) {
                    const std::uint32_t ia = shape.triangleIndices[t];
                    const std::uint32_t ib = shape.triangleIndices[t + 1];
                    const std::uint32_t ic = shape.triangleIndices[t + 2];
                    if ((ic * 3u) + 2u >= shape.positions.size()) {
                        continue;
                    }
                    const float* a = shape.positions.data() + (ia * 3u);
                    const float* b = shape.positions.data() + (ib * 3u);
                    const float* c = shape.positions.data() + (ic * 3u);
                    const double abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
                    const double acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
                    const double nx = (aby * acz) - (abz * acy);
                    const double ny = (abz * acx) - (abx * acz);
                    const double nz = (abx * acy) - (aby * acx);
                    signedVolume6 += (a[0] * ((b[1] * c[2]) - (b[2] * c[1]))) -
                                     (a[1] * ((b[0] * c[2]) - (b[2] * c[0]))) +
                                     (a[2] * ((b[0] * c[1]) - (b[1] * c[0])));
                    if (shape.normals.size() == shape.positions.size()) {
                        const float* na = shape.normals.data() + (ia * 3u);
                        const double dot = (nx * na[0]) + (ny * na[1]) + (nz * na[2]);
                        ++counted;
                        if (dot > 0.0) {
                            ++agree;
                        }
                    }
                }
                std::cout << "      winding: signedVolume=" << (signedVolume6 / 6.0);
                if (counted != 0u) {
                    std::cout << " geomNormal-vs-authoredNormal agree="
                              << (static_cast<double>(agree) / static_cast<double>(counted));
                }
                std::cout << "\n";

                // Generated BTO shapes merge many unrelated placed objects
                // into one shape. Report their disconnected components so a
                // near-camera LOD handoff can distinguish one enormous proxy
                // from the smaller city buildings sharing its material.
                const std::size_t vertexCount = shape.positions.size() / 3u;
                std::vector<std::uint32_t> parent(vertexCount);
                for (std::size_t v = 0; v < vertexCount; ++v) {
                    parent[v] = static_cast<std::uint32_t>(v);
                }
                const auto findRoot = [&](std::uint32_t value) {
                    std::uint32_t root = value;
                    while (parent[root] != root) {
                        root = parent[root];
                    }
                    while (parent[value] != value) {
                        const std::uint32_t next = parent[value];
                        parent[value] = root;
                        value = next;
                    }
                    return root;
                };
                const auto unite = [&](std::uint32_t a, std::uint32_t b) {
                    a = findRoot(a);
                    b = findRoot(b);
                    if (a != b) {
                        parent[b] = a;
                    }
                };
                for (std::size_t t = 0; t + 2u < shape.triangleIndices.size(); t += 3u) {
                    const std::uint32_t a = shape.triangleIndices[t];
                    const std::uint32_t b = shape.triangleIndices[t + 1u];
                    const std::uint32_t c = shape.triangleIndices[t + 2u];
                    if (a < vertexCount && b < vertexCount && c < vertexCount) {
                        unite(a, b);
                        unite(a, c);
                    }
                }
                struct Component {
                    std::size_t triangles = 0u;
                    float min[3] = {std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max()};
                    float max[3] = {std::numeric_limits<float>::lowest(),
                                    std::numeric_limits<float>::lowest(),
                                    std::numeric_limits<float>::lowest()};
                };
                std::map<std::uint32_t, Component> components;
                for (std::size_t t = 0; t + 2u < shape.triangleIndices.size(); t += 3u) {
                    const std::uint32_t first = shape.triangleIndices[t];
                    if (first >= vertexCount) {
                        continue;
                    }
                    Component& component = components[findRoot(first)];
                    ++component.triangles;
                    for (std::size_t corner = 0; corner < 3u; ++corner) {
                        const std::uint32_t vertex = shape.triangleIndices[t + corner];
                        if (vertex >= vertexCount) {
                            continue;
                        }
                        for (std::size_t axis = 0; axis < 3u; ++axis) {
                            const float value = shape.positions[(vertex * 3u) + axis];
                            component.min[axis] = std::min(component.min[axis], value);
                            component.max[axis] = std::max(component.max[axis], value);
                        }
                    }
                }
                std::vector<Component> sortedComponents;
                for (const auto& [root, component] : components) {
                    (void)root;
                    sortedComponents.push_back(component);
                }
                std::sort(sortedComponents.begin(), sortedComponents.end(),
                          [](const Component& a, const Component& b) {
                              return a.triangles > b.triangles;
                          });
                if (sortedComponents.size() > 1u) {
                    std::cout << "      components: " << sortedComponents.size() << " (largest)\n";
                    for (std::size_t i = 0; i < std::min<std::size_t>(12u, sortedComponents.size()); ++i) {
                        const Component& component = sortedComponents[i];
                        std::cout << "        tris " << component.triangles << " bounds min("
                                  << component.min[0] << ", " << component.min[1] << ", "
                                  << component.min[2] << ") max(" << component.max[0] << ", "
                                  << component.max[1] << ", " << component.max[2] << ")\n";
                    }
                }
            }
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
}

// Interior cell EditorIDs, which is what odai_newvegas_cooker --cell takes.
// Without this the flag was unusable: the only way to learn a valid ID was
// FNVEdit or the GECK, even though the extractor already parses every one of
// them. An optional substring filters the list, since a retail plugin has
// hundreds.
// Verifies the streaming index against the full extractor: build the offset
// index, then materialize a sample of cells through extractFalloutCellAt. For
// TES4+ every field is required to match what a whole-file pass produces for
// the same cell. TES3's whole-file extractor intentionally scans object tables
// only, so its oracle is that every sampled indexed CELL/LAND range can be read
// and preserves the metadata carried by the index.
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

        // WHERE the worldspace's content sits, which is the first thing anyone
        // pointing a camera at an unfamiliar place needs and the slowest thing
        // to find by hand. A worldspace record's own NAM0/NAM9 corners are NOT
        // this: Bravil's span 52 cells of open bay rather than a city, so they
        // framed the water and not the town. Occupied cells cannot lie.
        //
        // Cells with an empty children group are excluded on purpose -- they are
        // the header-only overrides that make a populated district look present
        // at coordinates it has nothing at.
        std::int32_t minX = 0, maxX = 0, minZ = 0, maxZ = 0;
        std::size_t occupied = 0;
        std::vector<std::pair<std::uint32_t, const FalloutCellIndexEntry*>> byWeight;
        for (const FalloutCellIndexEntry& entry : index.cells) {
            if (entry.isInterior || !entry.hasGridCoords ||
                entry.worldspaceFormId != worldspaceFormId || entry.childrenGroupSize == 0u) {
                continue;
            }
            if (occupied == 0) {
                minX = maxX = entry.gridX;
                minZ = maxZ = entry.gridZ;
            }
            minX = std::min(minX, entry.gridX);
            maxX = std::max(maxX, entry.gridX);
            minZ = std::min(minZ, entry.gridZ);
            maxZ = std::max(maxZ, entry.gridZ);
            ++occupied;
            byWeight.emplace_back(entry.childrenGroupSize, &entry);
        }
        std::cout << "  occupied cells: " << occupied << ", grid x [" << minX << "," << maxX
                  << "] z [" << minZ << "," << maxZ << "]\n";
        // Children-group BYTES, not a reference count -- the index deliberately
        // never walks a cell's contents, and byte size ranks density well enough
        // to say "start looking here".
        std::sort(byWeight.begin(), byWeight.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        for (std::size_t i = 0; i < byWeight.size() && i < 5u; ++i) {
            std::cout << "    densest: (" << byWeight[i].second->gridX << ","
                      << byWeight[i].second->gridZ << ") " << byWeight[i].first
                      << " bytes of children\n";
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

    EsmReader reader;
    if (!reader.open(esmPath)) {
        std::cout << "Reader open FAILED: " << reader.lastError() << "\n";
        return 1;
    }

    if (reader.pluginFormat() == EsmPluginFormat::kMorrowind) {
        std::size_t failures = 0;
        std::size_t materialized = 0;
        std::size_t referenceCount = 0;
        std::size_t landCount = 0;
        double extractMsTotal = 0.0;
        for (const std::size_t i : sampleIndices) {
            const FalloutCellIndexEntry& entry = index.cells[i];
            const auto extractStart = std::chrono::steady_clock::now();
            FalloutCellRecord actual;
            if (!extractFalloutCellAt(reader, entry, actual, error)) {
                std::cout << "  cell " << std::hex << entry.cellFormId << std::dec
                          << ": extract FAILED: " << error << "\n";
                ++failures;
                continue;
            }
            extractMsTotal += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - extractStart)
                                  .count();
            ++materialized;
            referenceCount += actual.references.size();
            landCount += actual.land != nullptr ? 1u : 0u;
            if (actual.isInterior != entry.isInterior ||
                actual.hasGridCoords != entry.hasGridCoords ||
                actual.gridX != entry.gridX || actual.gridZ != entry.gridZ ||
                actual.editorId != entry.editorId) {
                std::cout << "  cell " << std::hex << entry.cellFormId << std::dec
                          << ": indexed metadata changed during extraction\n";
                ++failures;
            }
        }
        std::cout << "materialized " << materialized << " TES3 cells ("
                  << referenceCount << " references, " << landCount << " LAND records), "
                  << failures << " failures\n";
        if (materialized > 0) {
            std::cout << "  extractFalloutCellAt averaged "
                      << (extractMsTotal / static_cast<double>(materialized))
                      << " ms per cell\n";
        }
        return failures == 0 && materialized == sampleIndices.size() ? 0 : 1;
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
                (a.hasHeights && a.heights != b.heights)) {
                problems.push_back("VHGT heights differ");
            }
            if (a.hasNormals != b.hasNormals ||
                (a.hasNormals && a.normals != b.normals)) {
                problems.push_back("VNML normals differ");
            }
            if (a.hasColors != b.hasColors ||
                (a.hasColors && a.colors != b.colors)) {
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

    // A worldspace name that resolves to nothing used to leave this at 0, which
    // the filter below reads as "no filter" -- so a typo, or a worldspace this
    // build cannot see, silently reported some OTHER worldspace's cell at the
    // same grid coordinate. Since Tamriel is first in the index, every bad name
    // rendered Tamriel and looked entirely plausible. Fail instead.
    const auto worldIt = tables.worldspaceFormIdsByEditorId.find(toLowerAscii(worldspaceFilter));
    if (worldIt == tables.worldspaceFormIdsByEditorId.end()) {
        std::cout << "no worldspace named \"" << worldspaceFilter << "\" in " << esmPath << "\n";
        return 1;
    }
    const std::uint32_t worldspaceFormId = worldIt->second;

    // Take the entry that actually HAS a children group, not merely the first at
    // these coordinates. A worldspace can carry several CELL records for one
    // grid square -- an override that touches only the cell header leaves an
    // entry with no children at all -- and stopping at the first one reports a
    // populated city district as empty. cell_streamer.cc:274 has always made
    // this check; this tool did not, which is why the two disagreed.
    const FalloutCellIndexEntry* entry = nullptr;
    for (const FalloutCellIndexEntry& candidate : index.cells) {
        if (candidate.isInterior || !candidate.hasGridCoords ||
            candidate.worldspaceFormId != worldspaceFormId) {
            continue;
        }
        if (candidate.gridX != cellX || candidate.gridZ != cellZ) {
            continue;
        }
        if (entry == nullptr) {
            entry = &candidate;
        }
        if (candidate.childrenGroupSize != 0u) {
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
    // Match the runtime streamer's asset override path so --buildcell can
    // verify a loose replacer in the actual scene it changes. This is the same
    // colon-separated convention used by odai.
    if (const char* mods = std::getenv("ODAI_FNV_MODS")) {
        std::string root;
        for (const char* cursor = mods; ; ++cursor) {
            if (*cursor == ':' || *cursor == '\0') {
                if (!root.empty()) {
                    assets.addModDirectory(std::filesystem::path(root));
                    root.clear();
                }
                if (*cursor == '\0') {
                    break;
                }
            } else {
                root.push_back(*cursor);
            }
        }
    }

    // Per-quadrant BASE texture, which is what an untextured-looking terrain
    // comes down to: BTXT names an LTEX per quadrant, and a quadrant naming
    // none falls back to the cell set's dominant texture -- which is itself
    // nothing when no quadrant anywhere named one.
    if (cell.land != nullptr) {
        // VNML, as decoded. Printed because a terrain that shades flat and
        // white is almost never a texture problem: it is the surface being lit
        // as though it faced sideways, and nothing else in this output would
        // say so. Bethesda space here, so a level post reads (0, 0, 1).
        std::cout << "  land normals: " << (cell.land->hasNormals ? "present" : "ABSENT");
        if (cell.land->hasNormals && cell.land->normals.size() >= 9u) {
            const int centre = (cell.land->gridSize / 2) * cell.land->gridSize +
                (cell.land->gridSize / 2);
            const auto show = [&](const char* label, int post) {
                const std::size_t base = static_cast<std::size_t>(post) * 3u;
                if (base + 2u < cell.land->normals.size()) {
                    std::cout << " " << label << "(" << cell.land->normals[base] << ","
                              << cell.land->normals[base + 1] << ","
                              << cell.land->normals[base + 2] << ")";
                }
            };
            show("post0", 0);
            show("centre", centre);
        }
        std::cout << "\n";
        std::cout << "  land base textures (BTXT) per quadrant:";
        for (const std::uint32_t formId : cell.land->quadrantBaseTextureFormId) {
            std::cout << " ";
            if (formId == 0u) {
                std::cout << "<none>";
            } else {
                const auto path = tables.landTexturePaths.find(formId);
                std::cout << std::hex << formId << std::dec
                          << (path != tables.landTexturePaths.end()
                                  ? ("=" + path->second)
                                  : std::string("=UNRESOLVED"));
            }
        }
        std::cout << "\n  land texture LAYERS (ATXT): " << cell.land->textureLayers.size() << "\n";
        for (const auto& layer : cell.land->textureLayers) {
            float peak = 0.0f;
            float sum = 0.0f;
            std::size_t nonzero = 0u;
            for (const float opacity : layer.opacity) {
                peak = std::max(peak, opacity);
                sum += opacity;
                nonzero += opacity > 0.0f ? 1u : 0u;
            }
            const auto path = tables.landTexturePaths.find(layer.textureFormId);
            std::cout << "    q" << static_cast<unsigned>(layer.quadrant)
                      << " layer=" << layer.layerIndex
                      << " form=0x" << std::hex << layer.textureFormId << std::dec
                      << " peak=" << peak << " sum=" << sum << " posts=" << nonzero
                      << " " << (path != tables.landTexturePaths.end()
                                      ? path->second
                                      : std::string("UNRESOLVED"))
                      << "\n";
        }
    } else {
        std::cout << "  no LAND record\n";
    }

    const auto buildStart = std::chrono::steady_clock::now();
    CellSceneBuilder builder(assets, tables);
    const std::vector<const FalloutCellRecord*> cells{&cell};
    builder.setFallbackLandTexture(builder.dominantLandTexture(cells));
    builder.addCellTerrain(cell);
    builder.addCellStatics(cell);
    odai::importer::ImportedScene scene;
    builder.finish(scene);
    appendResolvedDoors(cell, index, scene);
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
              << " rigidAnimations=" << scene.rigidAnimations.size()
              << " doors=" << scene.doors.size()
              << " terrainParts=" << stats.terrainPartsEmitted << "\n";
    if (std::getenv("ODAI_PROBE_LIST_DOORS") != nullptr) {
        for (const auto& door : scene.doors) {
            std::cout << "  door ref=0x" << std::hex << door.sourceReferenceFormId
                      << std::dec << " pos=(" << door.position[0] << ", "
                      << door.position[1] << ", " << door.position[2]
                      << ") target=\"" << door.targetCellEditorId << "\"\n";
        }
    }
    for (const auto& animation : scene.rigidAnimations) {
        float start[16] = {};
        float quarter[16] = {};
        if (odai::importer::sampleImportedSceneRigidAnimation(animation, 0.0f, start) &&
            odai::importer::sampleImportedSceneRigidAnimation(
                animation, animation.duration * 0.25f, quarter)) {
            std::cout << "  animation \"" << animation.nodeName << "\" delta t0=("
                      << start[3] << "," << start[7] << "," << start[11]
                      << ") quarter=(" << quarter[3] << "," << quarter[7] << ","
                      << quarter[11] << ")\n";
        }
    }
    // The terrain mesh's normals AFTER the basis change, in engine space, where
    // a level post must read (0, 1, 0). Printed next to the raw VNML above so a
    // terrain that shades like a wall can be blamed on the decode or on the
    // basis change without guessing which.
    for (const auto& mesh : scene.meshes) {
        if (mesh.name != "terrain" || mesh.vertices.empty()) {
            continue;
        }
        double sum[3] = {0.0, 0.0, 0.0};
        for (const auto& vertex : mesh.vertices) {
            for (int axis = 0; axis < 3; ++axis) {
                sum[axis] += vertex.normal[axis];
            }
        }
        const double count = static_cast<double>(mesh.vertices.size());
        std::cout << "  terrain normals (engine space, mean over " << mesh.vertices.size()
                  << " posts): (" << (sum[0] / count) << "," << (sum[1] / count) << ","
                  << (sum[2] / count) << ")\n";
        // VCLR, which MULTIPLIES the sampled texture. A dark mean here is the
        // difference between "the land texture is wrong" and "the land texture
        // is fine and something is multiplying it to black".
        double colorSum[3] = {0.0, 0.0, 0.0};
        for (const auto& vertex : mesh.vertices) {
            for (int channel = 0; channel < 3; ++channel) {
                colorSum[channel] += vertex.color[channel];
            }
        }
        std::cout << "  terrain vertex colour (VCLR, mean): (" << (colorSum[0] / count) << ","
                  << (colorSum[1] / count) << "," << (colorSum[2] / count) << ")\n";
        break;
    }
    // And the same normals once they have been through the PACKED stream, which
    // is what the renderer actually uploads. The mesh above is the builder's
    // output; anything that goes wrong between the two shows up as a difference
    // here and nowhere else.
    {
        double packedSum[3] = {0.0, 0.0, 0.0};
        std::size_t packedCount = 0;
        for (const auto& vertex : scene.packedVertices) {
            if ((vertex.flags & odai::importer::kImportedSceneMaterialFlagTerrainLayers) == 0u) {
                continue;
            }
            for (int axis = 0; axis < 3; ++axis) {
                packedSum[axis] += vertex.normal[axis];
            }
            ++packedCount;
        }
        if (packedCount > 0u) {
            const double count = static_cast<double>(packedCount);
            std::cout << "  packed terrain normals (mean over " << packedCount << " verts): ("
                      << (packedSum[0] / count) << "," << (packedSum[1] / count) << ","
                      << (packedSum[2] / count) << ")\n";
        } else {
            std::cout << "  packed terrain normals: no packed vertex carries the terrain-layer"
                      << " flag\n";
        }
    }
    // The widest meshes in the built scene, in world units. A single object
    // spanning several cells is almost always the answer to "what is this
    // plane over the landscape", and nothing else in this output names it.
    {
        struct Footprint {
            float extentX;
            float extentZ;
            std::string name;
        };
        std::vector<Footprint> footprints;
        for (const auto& mesh : scene.meshes) {
            if (mesh.vertices.empty()) {
                continue;
            }
            float minX = 1e30f, maxX = -1e30f, minZ = 1e30f, maxZ = -1e30f;
            for (const auto& vertex : mesh.vertices) {
                minX = std::min(minX, vertex.position[0]);
                maxX = std::max(maxX, vertex.position[0]);
                minZ = std::min(minZ, vertex.position[2]);
                maxZ = std::max(maxZ, vertex.position[2]);
            }
            footprints.push_back({maxX - minX, maxZ - minZ, mesh.name});
        }
        std::sort(footprints.begin(), footprints.end(), [](const auto& a, const auto& b) {
            return std::max(a.extentX, a.extentZ) > std::max(b.extentX, b.extentZ);
        });
        std::cout << "  widest meshes (XZ extent, world units):\n";
        for (std::size_t i = 0; i < footprints.size() && i < 8u; ++i) {
            std::cout << "    " << footprints[i].extentX << " x " << footprints[i].extentZ
                      << "  " << footprints[i].name << "\n";
        }
    }
    std::cout << "  shapes=" << stats.totalShapes
              << " untextured=" << stats.untexturedShapes
              << " placed=" << stats.placedInstances
              << " decalsSkipped=" << stats.shadowDecalShapesSkipped
              << " markersSkipped=" << stats.editorMarkerModelsSkipped
              << " droppedLayers=" << stats.droppedTerrainLayers << "\n";
    // Water is the one cell property with no geometry of its own: the record
    // states a height and the engine fills the footprint. Printing the decision
    // and its inputs together is the only way to tell "this cell is dry" from
    // "the height decoded wrong" -- both render as no water.
    // A cell with no LAND is flat ground at the worldspace's DNAM default
    // height, not a hole -- so "land=no" is only alarming once you know whether
    // a default exists to stand in for it.
    if (cell.land == nullptr) {
        const auto defaultsIt = tables.worldspaceDefaultsByFormId.find(cell.worldspaceFormId);
        if (defaultsIt == tables.worldspaceDefaultsByFormId.end()) {
            std::cout << "  no LAND, and this worldspace declares no DNAM default land height\n";
        } else {
            std::cout << "  no LAND; worldspace DNAM default land height "
                      << defaultsIt->second.defaultLandHeight << ", default water "
                      << defaultsIt->second.defaultWaterHeight << "\n";
        }
    }
    std::cout << "  water: XCLW=" << (cell.hasWater ? std::to_string(cell.waterHeight) : "<absent>");
    if (cell.land != nullptr && cell.land->hasHeights) {
        const auto [lowest, highest] =
            std::minmax_element(std::begin(cell.land->heights), std::end(cell.land->heights));
        std::cout << " terrain z [" << *lowest << "," << *highest << "]";
    }
    std::cout << " -> patches=" << scene.waterPatches.size() << "\n";
    if (!scene.waterPatches.empty()) {
        const odai::importer::ImportedSceneWaterPatch& water = scene.waterPatches.front();
        const auto textureName = [&](std::uint32_t index) -> std::string {
            return index < scene.textures.size() ? scene.textures[index].sourcePath : "<generic>";
        };
        std::cout << "    water normal: " << textureName(water.normalTextureIndex) << "\n"
                  << "    water flow:   " << textureName(water.flowTextureIndex) << "\n";
    }
    // Every one of these used to be a silent `continue`. A town with holes in it
    // is the symptom; this is the only place that says which kind of hole.
    const std::size_t droppedReferences =
        stats.referencesDroppedBaseNotFound + stats.referencesDroppedBaseHasNoModel +
        stats.referencesDroppedMeshUnresolved + stats.referencesDroppedMeshUnreadable;
    std::cout << "  refs dropped: " << droppedReferences << " of " << cell.references.size()
              << "  (baseNotFound=" << stats.referencesDroppedBaseNotFound
              << " baseHasNoModel=" << stats.referencesDroppedBaseHasNoModel
              << " meshUnresolved=" << stats.referencesDroppedMeshUnresolved
              << " meshUnreadable=" << stats.referencesDroppedMeshUnreadable << ")\n";
    // Name the dropped bases, not only count them: "STAT=21 baseHasNoModel"
    // says a fifth of Whiterun is missing and nothing else; the formIDs are
    // what --formid can then answer for.
    if (!builder.failedStatics().empty()) {
        std::cout << "  dropped base records (formID reason [type] editorID/model):\n";
        std::size_t shown = 0;
        for (const auto& [baseFormId, reason] : builder.failedStatics()) {
            if (reason == CellSceneBuilder::StaticDropReason::kIntentional) {
                continue;
            }
            if (++shown > 24u) {
                std::cout << "    ...\n";
                break;
            }
            const char* reasonName = "?";
            switch (reason) {
                case CellSceneBuilder::StaticDropReason::kBaseNotFound: reasonName = "baseNotFound"; break;
                case CellSceneBuilder::StaticDropReason::kBaseHasNoModel: reasonName = "baseHasNoModel"; break;
                case CellSceneBuilder::StaticDropReason::kMeshUnresolved: reasonName = "meshUnresolved"; break;
                case CellSceneBuilder::StaticDropReason::kMeshUnreadable: reasonName = "meshUnreadable"; break;
                case CellSceneBuilder::StaticDropReason::kIntentional: break;
            }
            const auto typeIt = tables.staticRecordTypes.find(baseFormId);
            const auto editorIt = tables.staticEditorIds.find(baseFormId);
            const auto modelIt = tables.staticModelPaths.find(baseFormId);
            std::cout << "    0x" << std::hex << baseFormId << std::dec << "  " << reasonName
                      << "  [" << (typeIt != tables.staticRecordTypes.end() ? typeIt->second : "?")
                      << "]  "
                      << (editorIt != tables.staticEditorIds.end() ? editorIt->second : "")
                      << "  "
                      << (modelIt != tables.staticModelPaths.end() ? modelIt->second : "<no model>")
                      << "\n";
        }
    }
    if (!stats.droppedReferencesByBaseType.empty()) {
        // Sorted so two runs can be diffed, and by count so the one worth
        // chasing is first.
        std::vector<std::pair<std::string, std::size_t>> byType(
            stats.droppedReferencesByBaseType.begin(), stats.droppedReferencesByBaseType.end());
        std::sort(byType.begin(), byType.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        });
        std::cout << "    by base record type:";
        for (const auto& [typeName, count] : byType) {
            std::cout << " " << typeName << "=" << count;
        }
        std::cout << "\n";
    }
    // The other half of ODAI_DEBUG_UNTEXTURED_MAGENTA: the render says WHERE an
    // untextured surface is, this says WHICH asset it wanted. Neither is much
    // use alone.
    if (!stats.unresolvedTexturePaths.empty() || !stats.untexturedModelPaths.empty()) {
        std::cout << "  untextured: " << stats.untexturedShapes << " shape(s), "
                  << stats.untexturedShapesGivenModelTexture
                  << " rescued by a sibling shape's texture\n";
        std::vector<std::string> paths(
            stats.unresolvedTexturePaths.begin(), stats.unresolvedTexturePaths.end());
        std::sort(paths.begin(), paths.end());
        for (std::size_t i = 0; i < paths.size() && i < 12u; ++i) {
            std::cout << "    unresolved texture: " << paths[i] << "\n";
        }
        if (paths.size() > 12u) {
            std::cout << "    ... and " << (paths.size() - 12u) << " more\n";
        }
        std::vector<std::string> models(
            stats.untexturedModelPaths.begin(), stats.untexturedModelPaths.end());
        std::sort(models.begin(), models.end());
        for (std::size_t i = 0; i < models.size() && i < 12u; ++i) {
            std::cout << "    model with no texture: " << models[i] << "\n";
        }
        if (models.size() > 12u) {
            std::cout << "    ... and " << (models.size() - 12u) << " more\n";
        }
    }
    std::cout << "  lights=" << scene.lights.size()
              << " placed=" << stats.lightsPlaced
              << " zeroRadiusSkipped=" << stats.lightsSkippedZeroRadius
              << " (LIGH base records known: " << tables.lightsByFormId.size() << ")\n";
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
    // ODAI_FLOATERS_PLUGINS=<comma-separated plugin file names> builds the same
    // merged view of the world the streamer uses when extra plugins are loaded,
    // so a floater introduced BY a mod can be told apart from one the base game
    // ships. Without it this reads one plugin, which is what it always did.
    FalloutLoadOrder order;
    bool useOrder = false;
    if (const char* pluginsEnv = std::getenv("ODAI_FLOATERS_PLUGINS")) {
        std::vector<std::string> requested;
        requested.push_back(esmPath.filename().string());
        std::string current;
        for (const char* cursor = pluginsEnv; ; ++cursor) {
            if (*cursor == ',' || *cursor == '\0') {
                if (!current.empty()) {
                    requested.push_back(current);
                    current.clear();
                }
                if (*cursor == '\0') {
                    break;
                }
                continue;
            }
            current.push_back(*cursor);
        }
        if (const char* rootsEnv = std::getenv("ODAI_FLOATERS_MODROOTS")) {
            std::string root;
            for (const char* cursor = rootsEnv; ; ++cursor) {
                if (*cursor == ':' || *cursor == '\0') {
                    if (!root.empty()) {
                        order.addSearchRoot(std::filesystem::path(root));
                        root.clear();
                    }
                    if (*cursor == '\0') {
                        break;
                    }
                } else {
                    root.push_back(*cursor);
                }
            }
        }
        std::string orderError;
        if (!order.open(dataPath, requested, orderError)) {
            std::cout << "load order FAILED: " << orderError << "\n";
            return 1;
        }
        useOrder = true;
        std::cout << "load order:";
        for (const auto& loaded : order.entries()) {
            std::cout << " " << loaded.header.fileName;
        }
        std::cout << "\n";
    }

    FalloutWorldTables tables;
    const bool worldOk = useOrder ? buildFalloutWorldTables(order, tables, error)
                                  : buildFalloutWorldTables(esmPath, tables, error);
    if (!worldOk) {
        std::cout << "world tables FAILED: " << error << "\n";
        return 1;
    }
    FalloutCellIndex index;
    const bool indexOk = useOrder ? buildFalloutCellIndex(order, index, error)
                                  : buildFalloutCellIndex(esmPath, index, error);
    if (!indexOk) {
        std::cout << "cell index FAILED: " << error << "\n";
        return 1;
    }
    // Same two corrections as --buildcell above: an unresolvable name is an
    // error rather than "match any worldspace", and the entry that carries the
    // children group wins over the first one at these coordinates.
    const auto worldIt = tables.worldspaceFormIdsByEditorId.find(toLowerAscii(worldspaceFilter));
    if (worldIt == tables.worldspaceFormIdsByEditorId.end()) {
        std::cout << "no worldspace named \"" << worldspaceFilter << "\"\n";
        return 1;
    }
    const std::uint32_t worldspaceFormId = worldIt->second;
    const FalloutCellIndexEntry* entry = nullptr;
    for (const FalloutCellIndexEntry& candidate : index.cells) {
        if (candidate.isInterior || !candidate.hasGridCoords ||
            candidate.worldspaceFormId != worldspaceFormId || candidate.gridX != cellX ||
            candidate.gridZ != cellZ) {
            continue;
        }
        if (entry == nullptr) {
            entry = &candidate;
        }
        if (candidate.childrenGroupSize != 0u) {
            entry = &candidate;
            break;
        }
    }
    if (entry == nullptr) {
        std::cout << "no cell at (" << cellX << "," << cellZ << ")\n";
        return 1;
    }

    EsmReader reader;
    if (!useOrder && !reader.open(esmPath)) {
        return 1;
    }
    FalloutCellRecord cell;
    const bool extracted = useOrder
        ? extractFalloutCellMerged(index, order, *entry, cell, error)
        : extractFalloutCellAt(reader, *entry, cell, error);
    if (!extracted) {
        std::cout << "extract FAILED: " << error << "\n";
        return 1;
    }
    if (useOrder) {
        std::cout << "cell contributions: " << entry->contributions.size()
                  << "  merged references: " << cell.references.size() << "\n";
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

    // ODAI_FLOATERS_LAND prints the decoded 33x33 VHGT grid as a coarse map.
    // "Is the terrain wrong here" is otherwise unanswerable from clearances
    // alone: a floater over FLAT ground means the placement is high, a floater
    // over ground that should have a berm means the LAND decode lost it.
    if (std::getenv("ODAI_FLOATERS_LAND") != nullptr) {
        float minH = std::numeric_limits<float>::max();
        float maxH = std::numeric_limits<float>::lowest();
        for (int i = 0; i < kLandGridSize * kLandGridSize; ++i) {
            minH = std::min(minH, cell.land->heights[i]);
            maxH = std::max(maxH, cell.land->heights[i]);
        }
        std::cout << "  LAND 33x33 heights " << minH << " .. " << maxH
                  << " (row 0 = south edge, col 0 = west edge; 128 units per post)\n";
        for (int y = kLandGridSize - 1; y >= 0; --y) {
            std::cout << "   y" << (y < 10 ? " " : "") << y << " ";
            for (int x = 0; x < kLandGridSize; ++x) {
                std::cout << " " << static_cast<int>(cell.land->heights[(y * kLandGridSize) + x]);
            }
            std::cout << "\n";
        }
    }

    // ODAI_FLOATERS_LOD=<lod block virtual path> validates the VHGT decode
    // against the game's own baked distant terrain, across the whole cell
    // rather than at one point. If our heightfield were wrong by the ~170 units
    // a floating road implies, this sweep is where it would show as a bias; a
    // mean near zero with a spread of a few tens of units is just the LOD being
    // ~20x coarser than the heightfield it was baked from.
    if (const char* lodEnv = std::getenv("ODAI_FLOATERS_LOD")) {
        odai::importer::fnv::FalloutAssetSource lodAssets;
        std::vector<std::uint8_t> lodBytes;
        std::string lodError;
        odai::importer::fnv::NifModel lodModel;
        if (lodAssets.open(dataPath) && lodAssets.resolveMesh(lodEnv, lodBytes, lodError) &&
            odai::importer::fnv::parseNifStaticMesh(lodBytes, lodModel, lodError)) {
            // Collapse each XY to its highest vertex: LOD blocks carry a skirt
            // far below the surface at the block edge (see --lodheight).
            std::map<std::pair<int, int>, float> surfaceByXy;
            for (const odai::importer::fnv::NifShape& shape : lodModel.shapes) {
                const std::size_t vertexCount = shape.positions.size() / 3u;
                for (std::size_t v = 0; v < vertexCount; ++v) {
                    const std::pair<int, int> key{
                        static_cast<int>(std::lround(shape.positions[(v * 3u) + 0])),
                        static_cast<int>(std::lround(shape.positions[(v * 3u) + 1]))};
                    const float vz = shape.positions[(v * 3u) + 2];
                    const auto existing = surfaceByXy.find(key);
                    if (existing == surfaceByXy.end() || vz > existing->second) {
                        surfaceByXy[key] = vz;
                    }
                }
            }
            double sumDiff = 0.0;
            double sumAbsDiff = 0.0;
            float worstDiff = 0.0f;
            float worstX = 0.0f;
            float worstY = 0.0f;
            std::size_t samples = 0;
            for (int gy = 0; gy < kLandGridSize; gy += 2) {
                for (int gx = 0; gx < kLandGridSize; gx += 2) {
                    const float wx = cellOriginX + (static_cast<float>(gx) * kLandPostSpacing);
                    const float wy = cellOriginY + (static_cast<float>(gy) * kLandPostSpacing);
                    // Inverse-distance over the nearest few LOD surface points.
                    std::vector<std::pair<float, float>> byDistance;  // (d, z)
                    byDistance.reserve(surfaceByXy.size());
                    for (const auto& [xy, z] : surfaceByXy) {
                        const float dx = static_cast<float>(xy.first) - wx;
                        const float dy = static_cast<float>(xy.second) - wy;
                        byDistance.emplace_back(std::sqrt((dx * dx) + (dy * dy)), z);
                    }
                    std::partial_sort(byDistance.begin(),
                                      byDistance.begin() + std::min<std::size_t>(byDistance.size(), 4u),
                                      byDistance.end(),
                                      [](const auto& a, const auto& b) { return a.first < b.first; });
                    double weightSum = 0.0;
                    double heightSum = 0.0;
                    for (std::size_t i = 0; i < std::min<std::size_t>(byDistance.size(), 4u); ++i) {
                        const double weight = 1.0 / std::max(1.0f, byDistance[i].first);
                        weightSum += weight;
                        heightSum += weight * byDistance[i].second;
                    }
                    if (weightSum <= 0.0) {
                        continue;
                    }
                    const float lodHeight = static_cast<float>(heightSum / weightSum);
                    const float ourHeight = cell.land->heights[(gy * kLandGridSize) + gx];
                    const float diff = ourHeight - lodHeight;
                    sumDiff += diff;
                    sumAbsDiff += std::abs(diff);
                    if (std::abs(diff) > std::abs(worstDiff)) {
                        worstDiff = diff;
                        worstX = wx;
                        worstY = wy;
                    }
                    ++samples;
                }
            }
            if (samples > 0) {
                std::cout << "  LAND-vs-LOD over " << samples << " posts: mean(ours-lod)="
                          << (sumDiff / static_cast<double>(samples))
                          << "  mean|diff|=" << (sumAbsDiff / static_cast<double>(samples))
                          << "  worst=" << worstDiff << " at (" << static_cast<int>(worstX) << ","
                          << static_cast<int>(worstY) << ")\n";
            }
        } else {
            std::cout << "  LAND-vs-LOD skipped: " << lodError << "\n";
        }
    }

    // Ground truth for "is it floating": transform the mesh's own vertices into
    // world space and take the MINIMUM clearance over the terrain beneath them.
    // The reference origin sitting high means nothing on its own -- a sloped
    // road piece is authored with its origin at the raised end.
    FalloutAssetSource assets;
    bool haveAssets = assets.open(dataPath);
    // Mod directories must reach the ASSET source too, not just plugin
    // resolution. A reference whose mesh does not resolve scores zero clearance
    // and is silently never reported as floating -- so with the mod's meshes
    // missing, every placement the mod introduces is invisible to this check,
    // which is exactly the geometry a new mod is most likely to get wrong.
    if (haveAssets) {
        if (const char* rootsEnv = std::getenv("ODAI_FLOATERS_MODROOTS")) {
            std::string root;
            for (const char* cursor = rootsEnv; ; ++cursor) {
                if (*cursor == ':' || *cursor == '\0') {
                    if (!root.empty()) {
                        assets.addModDirectory(std::filesystem::path(root));
                        root.clear();
                    }
                    if (*cursor == '\0') {
                        break;
                    }
                } else {
                    root.push_back(*cursor);
                }
            }
        }
    }
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
        // Absolute values behind the two offsets above. A delta alone cannot
        // say WHICH side is wrong -- a road 167 units over the ground is either
        // a placement that is too high or terrain that is too low, and those
        // have opposite fixes.
        float originZ;
        float groundZ;
        bool initiallyDisabled;
        std::uint32_t refFormId;
        std::uint32_t recordFlags;
        bool hasEnableParent;
        std::uint32_t enableParentFormId;
        bool enableParentOpposite;
        float euler[3];
    };
    std::vector<Entry> entries;
    std::size_t unknownBaseCount = 0;
    std::map<std::string, std::vector<std::uint32_t>> droppedByType;
    for (const FalloutPlacedReference& ref : cell.references) {
        std::uint32_t baseFormId = ref.baseFormId;
        if (baseFormId == 0u && !ref.baseEditorId.empty()) {
            const auto resolved = tables.baseFormIdsByEditorId.find(toLowerAscii(ref.baseEditorId));
            if (resolved != tables.baseFormIdsByEditorId.end()) {
                baseFormId = resolved->second;
            }
        }
        const auto modelIt = tables.staticModelPaths.find(baseFormId);
        if (modelIt == tables.staticModelPaths.end()) {
            // The base record is not one this importer places. Every such
            // reference is silently dropped from the scene, which is how a road
            // can end up resting on nothing -- so name them rather than just
            // counting them. A count says something is missing; the TYPE says
            // whether it is geometry that should have been there.
            ++unknownBaseCount;
            const auto droppedTypeIt = tables.staticRecordTypes.find(baseFormId);
            droppedByType[droppedTypeIt == tables.staticRecordTypes.end()
                              ? std::string("<base record not found>")
                              : droppedTypeIt->second]
                .push_back(baseFormId);
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
        for (const std::array<float, 3>& local : meshPointsFor(baseFormId, modelIt->second)) {
            const float wx = ref.position[0] + ref.scale * ((rot[0] * local[0]) + (rot[1] * local[1]) + (rot[2] * local[2]));
            const float wy = ref.position[1] + ref.scale * ((rot[3] * local[0]) + (rot[4] * local[1]) + (rot[5] * local[2]));
            const float wz = ref.position[2] + ref.scale * ((rot[6] * local[0]) + (rot[7] * local[1]) + (rot[8] * local[2]));
            minClearance = std::min(minClearance, wz - terrainHeightAt(wx, wy));
        }
        if (minClearance == std::numeric_limits<float>::max()) {
            minClearance = 0.0f;  // no mesh points; do not report it as floating
        }
        entries.push_back(Entry{ref.position[2] - ground, minClearance, rotationMagnitude,
                                ref.scale, localX, localY, outsideCell, modelIt->second,
                                ref.position[2], ground,
                                (ref.recordFlags & 0x00000800u) != 0u, ref.formId, ref.recordFlags,
                                ref.hasEnableParent, ref.enableParentFormId,
                                ref.enableParentOpposite,
                                {ref.rotationRadians[0], ref.rotationRadians[1],
                                 ref.rotationRadians[2]}});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.minClearance > b.minClearance;
    });

    // Which base record types actually placed geometry here.
    std::map<std::string, std::size_t> countByType;
    for (const FalloutPlacedReference& ref : cell.references) {
        std::uint32_t baseFormId = ref.baseFormId;
        if (baseFormId == 0u && !ref.baseEditorId.empty()) {
            const auto resolved = tables.baseFormIdsByEditorId.find(toLowerAscii(ref.baseEditorId));
            if (resolved != tables.baseFormIdsByEditorId.end()) {
                baseFormId = resolved->second;
            }
        }
        const auto typeIt = tables.staticRecordTypes.find(baseFormId);
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
        std::uint32_t baseFormId = ref.baseFormId;
        if (baseFormId == 0u && !ref.baseEditorId.empty()) {
            const auto resolved = tables.baseFormIdsByEditorId.find(toLowerAscii(ref.baseEditorId));
            if (resolved != tables.baseFormIdsByEditorId.end()) {
                baseFormId = resolved->second;
            }
        }
        const auto typeIt = tables.staticRecordTypes.find(baseFormId);
        const auto modelIt = tables.staticModelPaths.find(baseFormId);
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
              << " references DROPPED (base record carries no model this importer places)\n";
    for (const auto& [type, formIds] : droppedByType) {
        std::cout << "    DROPPED " << type << " x" << formIds.size() << " :";
        for (std::size_t i = 0; i < formIds.size() && i < 8u; ++i) {
            std::cout << " 0x" << std::hex << formIds[i] << std::dec;
        }
        std::cout << "\n";
    }
    std::size_t floating = 0;
    for (const Entry& e : entries) {
        if (e.minClearance > 50.0f) {
            ++floating;
        }
    }
    std::cout << "  " << floating << " have their ENTIRE mesh more than 50 units clear of the "
              << "terrain (i.e. genuinely floating)\n";
    // ODAI_FLOATERS_ALL lists every placement, not just the worst 14. A floater
    // is only diagnosable against its NEIGHBOURS -- "is the terrain wrong here
    // or is this one reference high" is answered by what sits beside it.
    const bool showAll = std::getenv("ODAI_FLOATERS_ALL") != nullptr;
    const std::size_t show =
        showAll ? entries.size() : std::min<std::size_t>(entries.size(), 14u);
    for (std::size_t i = 0; i < show; ++i) {
        std::cout << "  ref 0x" << std::hex << entries[i].refFormId << std::dec
                  << "  clearance +" << static_cast<int>(entries[i].minClearance)
                  << "  origin +" << static_cast<int>(entries[i].offset) << " units  rot "
                  << static_cast<int>(entries[i].rotationMagnitudeDegrees) << " deg  scale "
                  << entries[i].scale << (entries[i].outsideCell ? "  OUTSIDE-CELL" : "")
                  << "  local(" << static_cast<int>(entries[i].localX) << ","
                  << static_cast<int>(entries[i].localY) << ")  z=" << entries[i].originZ
                  << " ground=" << entries[i].groundZ
                  << "  flags=0x" << toHex(entries[i].recordFlags)
                  << (entries[i].initiallyDisabled ? "  INITIALLY-DISABLED" : "")
                  << (entries[i].hasEnableParent
                          ? ("  ENABLE-PARENT=0x" + toHex(entries[i].enableParentFormId) +
                             (entries[i].enableParentOpposite ? "(opposite)" : ""))
                          : std::string())
                  << "  rotXYZ(" << static_cast<int>(entries[i].euler[0] * 57.2957795f) << ","
                  << static_cast<int>(entries[i].euler[1] * 57.2957795f) << ","
                  << static_cast<int>(entries[i].euler[2] * 57.2957795f) << ")"
                  << "  " << entries[i].model << "\n";
    }
    return 0;
}

int listCells(const std::filesystem::path& esmPath, const std::string& filter) {
    using namespace odai::importer::fnv;
    std::string error;
    EsmReader formatReader;
    if (!formatReader.open(esmPath)) {
        std::cout << "Open FAILED: " << formatReader.lastError() << "\n";
        return 1;
    }
    const std::string loweredFilter = toLowerAscii(filter);
    if (formatReader.pluginFormat() == EsmPluginFormat::kMorrowind) {
        // TES3's whole-file extractor intentionally scans object tables only;
        // its CELL records are exposed through the streaming index instead.
        FalloutCellIndex index;
        if (!buildFalloutCellIndex(esmPath, index, error)) {
            std::cout << "Index FAILED: " << error << "\n";
            return 1;
        }
        std::size_t interiorCount = 0;
        std::size_t shown = 0;
        for (const FalloutCellIndexEntry& entry : index.cells) {
            if (!entry.isInterior || entry.editorId.empty()) {
                continue;
            }
            ++interiorCount;
            if (!loweredFilter.empty() &&
                toLowerAscii(entry.editorId).find(loweredFilter) == std::string::npos) {
                continue;
            }
            FalloutCellRecord cell;
            if (!extractFalloutCellAt(formatReader, entry, cell, error)) {
                std::cout << "  " << entry.editorId << "  (extract FAILED: " << error << ")\n";
            } else {
                std::cout << "  " << entry.editorId << "  (" << cell.references.size()
                          << " refs)\n";
            }
            ++shown;
        }
        std::cout << "Interior cells with an EditorID: " << interiorCount;
        if (!loweredFilter.empty()) {
            std::cout << ", " << shown << " matching \"" << filter << "\"";
        }
        std::cout << "\n";
        return 0;
    }

    FalloutSceneData scene;
    FalloutExtractFilter extractFilter{};
    if (!extractFalloutScene(esmPath, extractFilter, scene, error)) {
        std::cout << "Extract FAILED: " << error << "\n";
        return 1;
    }
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

// Generic "what is actually in this record type" dump.
//
// Every typed reader in fallout_records.cc that got its field layout from
// documentation rather than from the file has been wrong at least once, and
// wrong silently -- a bad offset returns a plausible number, not an error. This
// mode exists so the first step of adding any record type is reading it, and it
// prints every interpretation at once (hex, u32, i32, f32, ASCII) so the right
// one is picked by recognising it rather than by assuming it.
//
// The subrecord census at the end is the other half: it shows which subrecords
// are always present, which are optional, and -- for container records like
// SCOL -- which repeat.

// Dumps one speaker's dialogue by walking DIAL topics and their child INFO
// records. Measurement first: the INFO layout, and above all HOW A LINE IS
// ATTRIBUTED TO A SPEAKER, are asserted nowhere in this codebase yet, and
// Fallout does not store a speaker field on INFO at all -- the link is a CTDA
// condition (GetIsID <actor>) inside the record. This prints what is actually
// there so the importer can be written against measurements instead of against
// a wiki page.
//
// CTDA is 28 bytes in FO3/FNV: type u8, 3 unused, comparison value f32,
// function index u32, param1 u32, param2 u32, runOn u32, reference formID u32.
int probeDialogue(const std::filesystem::path& pluginPath, const std::string& speakerSubstring,
                  std::size_t limit) {
    odai::importer::fnv::EsmReader reader;
    if (!reader.open(pluginPath)) {
        std::cout << "open failed: " << reader.lastError() << "\n";
        return 1;
    }
    const std::string wanted = toLowerAscii(speakerSubstring);

    // Pass 1: actor base records whose EDID matches, so we have formIDs to
    // match CTDA parameters against.
    std::map<std::uint32_t, std::string> speakers;
    {
        odai::importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
            if (record.type != "CREA" && record.type != "NPC_") {
                return;
            }
            for (const auto& sub : record.subrecords) {
                if (sub.type != "EDID" || sub.size == 0u) {
                    continue;
                }
                std::string edid(reinterpret_cast<const char*>(sub.data),
                                 static_cast<std::size_t>(sub.size));
                while (!edid.empty() && edid.back() == '\0') { edid.pop_back(); }
                if (toLowerAscii(edid).find(wanted) != std::string::npos) {
                    speakers[record.formId] = record.type + " " + edid;
                }
            }
        };
        reader.walk(visitor);
    }
    std::cout << "actor base records matching \"" << speakerSubstring << "\": " << speakers.size() << "\n";
    for (const auto& [formId, name] : speakers) {
        std::cout << "  " << std::hex << std::uppercase << formId << std::dec << "  " << name << "\n";
    }
    if (speakers.empty()) {
        return 0;
    }

    // Pass 2: DIAL topics, then the INFO records in each topic's child group.
    // Which CTDA function index means "the speaker is this actor" is DERIVED
    // here rather than assumed: every function index whose parameter matches a
    // matched actor formID is counted, and the histogram is printed.
    std::string currentTopicEdid;
    std::string currentTopicText;
    std::map<std::uint32_t, std::size_t> functionIndexHistogram;
    std::size_t printed = 0;
    std::size_t infosForSpeaker = 0;
    std::map<std::string, std::size_t> infoSubrecordCounts;

    odai::importer::fnv::EsmReader::Visitor visitor;
    visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
        if (record.type == "DIAL") {
            currentTopicEdid.clear();
            currentTopicText.clear();
            for (const auto& sub : record.subrecords) {
                if (sub.size == 0u) { continue; }
                std::string value(reinterpret_cast<const char*>(sub.data),
                                  static_cast<std::size_t>(sub.size));
                while (!value.empty() && value.back() == '\0') { value.pop_back(); }
                if (sub.type == "EDID") { currentTopicEdid = value; }
                else if (sub.type == "FULL") { currentTopicText = value; }
            }
            return;
        }
        if (record.type != "INFO") {
            return;
        }
        // Does any CTDA in this record name one of our speakers?
        bool mentionsSpeaker = false;
        std::uint32_t matchedFunction = 0;
        for (const auto& sub : record.subrecords) {
            if (sub.type != "CTDA" || sub.size < 28u) { continue; }
            std::uint32_t functionIndex = 0, param1 = 0;
            std::memcpy(&functionIndex, sub.data + 8, 4);
            std::memcpy(&param1, sub.data + 12, 4);
            if (speakers.find(param1) != speakers.end()) {
                mentionsSpeaker = true;
                matchedFunction = functionIndex;
                ++functionIndexHistogram[functionIndex];
            }
        }
        if (!mentionsSpeaker) {
            return;
        }
        ++infosForSpeaker;
        for (const auto& sub : record.subrecords) {
            ++infoSubrecordCounts[sub.type];
        }
        if (printed >= limit) {
            return;
        }
        ++printed;
        std::cout << "\n[" << printed << "] topic \"" << currentTopicEdid << "\""
                  << (currentTopicText.empty() ? "" : (" (" + currentTopicText + ")"))
                  << "  info=" << std::hex << std::uppercase << record.formId << std::dec
                  << "  ctdaFunc=" << matchedFunction << "\n";
        for (const auto& sub : record.subrecords) {
            if (sub.size == 0u) { continue; }
            if (sub.type == "NAM1" || sub.type == "NAM2") {
                std::string value(reinterpret_cast<const char*>(sub.data),
                                  static_cast<std::size_t>(sub.size));
                while (!value.empty() && value.back() == '\0') { value.pop_back(); }
                std::cout << "    " << sub.type << ": " << value << "\n";
            } else if (sub.type == "TCLT" && sub.size >= 4u) {
                std::uint32_t link = 0;
                std::memcpy(&link, sub.data, 4);
                std::cout << "    TCLT (choice -> topic): " << std::hex << std::uppercase
                          << link << std::dec << "\n";
            }
        }
    };
    reader.walk(visitor);

    std::cout << "\nINFO records naming a matched speaker: " << infosForSpeaker << "\n";
    std::cout << "CTDA function indices seen with a speaker formID as param1:";
    for (const auto& [functionIndex, count] : functionIndexHistogram) {
        std::cout << " " << functionIndex << "x" << count;
    }
    std::cout << "\nsubrecords present on those INFOs:";
    for (const auto& [type, count] : infoSubrecordCounts) {
        std::cout << " " << type << "=" << count;
    }
    std::cout << "\n";
    return 0;
}


// Builds the actual dialogue::DialogueTree the game would use and prints it as
// a conversation, so the graph can be read the way a player would walk it
// rather than as a record dump.
int probeDialogueTree(const std::filesystem::path& pluginPath, const std::string& speakerEditorId,
                      std::size_t depth) {
    odai::dialogue::DialogueTree tree;
    odai::importer::fnv::DialogueImportStats stats;
    std::string error;
    if (!odai::importer::fnv::buildSpeakerDialogueTree(pluginPath, speakerEditorId, tree, stats, error)) {
        std::cout << "failed: " << error << "\n";
        return 1;
    }
    std::cout << "speaker \"" << tree.id << "\": " << tree.nodes.size() << " nodes, start="
              << tree.startNode << "\n"
              << "  topics seen " << stats.topicsSeen << ", responses " << stats.responsesConcatenated
              << ", choice links " << stats.choiceLinks << " (" << stats.danglingLinks
              << " dropped as unreachable)\n"
              << "  NOTE: CTDA conditions are not evaluated -- lines the real game gates\n"
              << "        behind quest state are offered here.\n";

    // Walk from the start node, following the first choice each time, so the
    // output reads as one plausible run of the conversation.
    std::string current = tree.startNode;
    std::set<std::string> visited;
    for (std::size_t step = 0; step < depth && !current.empty(); ++step) {
        const auto it = tree.nodes.find(current);
        if (it == tree.nodes.end() || visited.count(current) != 0u) {
            break;
        }
        visited.insert(current);
        const odai::dialogue::DialogueNode& node = it->second;
        std::cout << "\n" << node.speaker << ": " << node.text << "\n";
        if (node.choices.empty()) {
            std::cout << "  [conversation ends]\n";
            break;
        }
        for (std::size_t c = 0; c < node.choices.size(); ++c) {
            std::cout << "  " << (c + 1) << ") " << node.choices[c].text
                      << (c == 0 ? "   <- following this one" : "") << "\n";
        }
        current = node.choices.front().targetNode;
    }
    return 0;
}

// Everything the plugin says about how one actor gets into the world and moves
// around it: the base record's AI package list and script, every ACRE/ACHR that
// places him (with its enable-parent, which is how Bethesda swaps one actor
// between locations), and the packages/script those name.
//
// Written because "how does Victor move around Goodsprings" cannot be answered
// from the reference alone -- a reference is a fixed position. Movement lives in
// PACK records and in script text, and the several places he appears are
// several references toggled by quest state, not one reference being driven.
// Every actor placed within `radius` of a world XY, with what its base record
// offers a renderer.
//
// The point is the last column. A CREA carries its own geometry (MODL is the
// skeleton, NIFZ the body parts) and can be loaded the way Victor is. An NPC_
// carries a skeleton and NOTHING else: its body comes from its RACE's part
// models plus whatever it is wearing, which is a different and much larger
// import path. "Populate the town" is cheap or expensive depending entirely on
// which of these the town is made of, and that is not guessable.
int probeActorsNear(
    const std::filesystem::path& pluginPath, float centreX, float centreY, float radius
) {
    odai::importer::fnv::FalloutActorScan scan;
    std::string error;
    if (!odai::importer::fnv::findActorsNear(pluginPath, centreX, centreY, radius, scan, error)) {
        std::cout << "scan failed: " << error << "\n";
        return 1;
    }
    const auto sourceName = [](odai::importer::fnv::ActorGeometrySource source) {
        switch (source) {
            case odai::importer::fnv::ActorGeometrySource::OwnBodyParts: return "own-parts";
            case odai::importer::fnv::ActorGeometrySource::Template:     return "TEMPLATE ";
            case odai::importer::fnv::ActorGeometrySource::Race:         return "race     ";
            default:                                                     return "NONE     ";
        }
    };
    std::cout << "scan tables: " << scan.bases.size() << " bases, " << scan.leveledLists.size()
              << " levelled actor lists, " << scan.leveledItems.size() << " levelled item lists, "
              << scan.races.size() << " races, " << scan.armors.size() << " armors, "
              << scan.voiceTypes.size() << " voice types\n";
    std::map<std::string, std::size_t> bySource;
    std::size_t disabled = 0;
    std::cout << scan.placements.size() << " placement(s) within " << radius << " units of ("
              << centreX << ", " << centreY << "), " << scan.bases.size() << " actor bases:\n";
    for (const auto& placement : scan.placements) {
        const auto resolved = scan.resolve(placement.baseFormId);
        ++bySource[sourceName(resolved.geometrySource)];
        disabled += placement.initiallyDisabled ? 1u : 0u;
        const float dx = placement.position[0] - centreX;
        const float dy = placement.position[1] - centreY;
        std::cout << "  " << sourceName(resolved.geometrySource) << " "
                  << (resolved.base != nullptr ? resolved.base->recordType : std::string("?"))
                  << " " << (resolved.base != nullptr ? resolved.base->editorId : std::string("?"))
                  << "  d=" << static_cast<int>(std::sqrt((dx * dx) + (dy * dy)))
                  << "  parts=" << resolved.bodyPartPaths.size();
        if (resolved.geometrySource == odai::importer::fnv::ActorGeometrySource::Template) {
            std::cout << "  via=" << std::hex << resolved.resolvedBaseFormId << std::dec;
        }
        if (resolved.base != nullptr && resolved.base->raceFormId != 0u &&
            resolved.geometrySource == odai::importer::fnv::ActorGeometrySource::Race) {
            std::cout << "  race=" << std::hex << resolved.base->raceFormId << std::dec
                      << (resolved.base->isFemale ? " female" : " male")
                      << "  carried=" << resolved.base->inventoryFormIds.size()
                      << "  worn=" << resolved.wornArmorFormIds.size();
        }
        if (resolved.base != nullptr && resolved.base->templateFlags != 0u) {
            std::cout << "  tplFlags=0x" << std::hex << resolved.base->templateFlags << std::dec
                      << " tplt=" << std::hex << resolved.base->templateFormId << std::dec
                      << " modl=\"" << resolved.base->skeletonPath << "\""
                      << " -> skeleton=\"" << resolved.skeletonPath << "\"";
        }
        if (resolved.base != nullptr) {
            const std::string voiceFolder = scan.voiceFolderFor(resolved.base->formId);
            std::cout << "  voice=" << (voiceFolder.empty() ? std::string("<none>") : voiceFolder);
        }
        if (resolved.geometrySource == odai::importer::fnv::ActorGeometrySource::None &&
            resolved.base != nullptr) {
            std::cout << "  tplt=" << std::hex << resolved.base->templateFormId
                      << " race=" << resolved.base->raceFormId << std::dec
                      << " modl=\"" << resolved.base->skeletonPath << "\""
                      << " -> skeleton=\"" << resolved.skeletonPath << "\"";
        }
        if (placement.initiallyDisabled) { std::cout << "  INITIALLY-DISABLED"; }
        std::cout << "\n";
        // The assembled part list, which is the whole answer for an NPC_ and
        // the only place a wrong slot (a hat where the head should be, an
        // outfit that never resolved) is visible before it reaches a screen.
        for (const std::string& part : resolved.bodyPartPaths) {
            std::cout << "        " << part << "\n";
        }
        // An NPC_ who resolved no armour at all is standing in the race's
        // underwear. Naming what he was carrying is the only way to tell
        // "carries nothing wearable" from "carries something this does not
        // follow yet".
        if (resolved.geometrySource == odai::importer::fnv::ActorGeometrySource::Race &&
            resolved.wornArmorFormIds.empty() && resolved.base != nullptr) {
            const auto* wardrobe = scan.inheritedFrom(
                resolved.base->formId, odai::importer::fnv::kActorTemplateUseInventory);
            std::cout << "        UNDRESSED, carrying:";
            if (wardrobe != nullptr) {
                for (const std::uint32_t item : wardrobe->inventoryFormIds) {
                    std::cout << " " << std::hex << item << std::dec
                              << (scan.leveledItems.count(item) != 0u ? "(list)" : "");
                }
            }
            std::cout << "\n";
        }
    }
    std::cout << "\nby geometry source:\n";
    for (const auto& [name, count] : bySource) {
        std::cout << "  " << name << " " << count << "\n";
    }
    std::cout << "  (" << disabled << " initially disabled)\n";
    return 0;
}

int probeActor(const std::filesystem::path& pluginPath, const std::string& wantedEditorId) {
    odai::importer::fnv::EsmReader reader;
    if (!reader.open(pluginPath)) {
        std::cout << "open failed: " << reader.lastError() << "\n";
        return 1;
    }
    const std::string wanted = toLowerAscii(wantedEditorId);

    const auto readU32 = [](const odai::importer::fnv::EsmSubrecordView& sub, std::size_t offset) {
        std::uint32_t value = 0;
        if (sub.size >= offset + 4u) {
            std::memcpy(&value, sub.data + offset, 4u);
        }
        return value;
    };
    const auto subString = [](const odai::importer::fnv::EsmSubrecordView& sub) {
        std::string out(reinterpret_cast<const char*>(sub.data), static_cast<std::size_t>(sub.size));
        while (!out.empty() && out.back() == '\0') { out.pop_back(); }
        return out;
    };

    // Pass 1: the base actor -- its script and its AI package list.
    std::uint32_t actorFormId = 0;
    std::uint32_t scriptFormId = 0;
    std::vector<std::uint32_t> packageFormIds;
    {
        odai::importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const odai::importer::fnv::EsmRecordHeaderView& header) {
            return actorFormId == 0u && (header.type == "CREA" || header.type == "NPC_");
        };
        visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
            if (actorFormId != 0u) { return; }
            bool match = false;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "EDID" && toLowerAscii(subString(sub)) == wanted) { match = true; }
            }
            if (!match) { return; }
            actorFormId = record.formId;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "SCRI") { scriptFormId = readU32(sub, 0); }
                else if (sub.type == "PKID") { packageFormIds.push_back(readU32(sub, 0)); }
            }
        };
        reader.walk(visitor);
    }
    if (actorFormId == 0u) {
        std::cout << "no CREA/NPC_ with EditorID \"" << wantedEditorId << "\"\n";
        return 1;
    }
    std::cout << wantedEditorId << " = " << std::hex << actorFormId << std::dec
              << "   script=" << std::hex << scriptFormId << std::dec
              << "   " << packageFormIds.size() << " AI packages\n";

    // Pass 2: every placement of him, and the cells they sit in.
    struct Placement {
        std::uint32_t refFormId = 0;
        std::uint32_t cellFormId = 0;
        float position[3] = {};
        std::uint32_t enableParent = 0;
        bool initiallyDisabled = false;
    };
    std::vector<Placement> placements;
    std::map<std::uint32_t, std::string> cellNames;
    {
        std::uint32_t currentCell = 0;
        odai::importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const odai::importer::fnv::EsmRecordHeaderView& header) {
            return header.type == "CELL" || header.type == "ACRE" || header.type == "ACHR";
        };
        visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
            if (record.type == "CELL") {
                currentCell = record.formId;
                for (const auto& sub : record.subrecords) {
                    if (sub.type == "EDID") { cellNames[record.formId] = subString(sub); }
                }
                return;
            }
            std::uint32_t base = 0;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "NAME") { base = readU32(sub, 0); }
            }
            if (base != actorFormId) { return; }
            Placement placement;
            placement.refFormId = record.formId;
            placement.cellFormId = currentCell;
            // Record header flag 0x800 is "Initially Disabled" -- the switch
            // that makes a placement dormant until something enables it.
            placement.initiallyDisabled = (record.flags & 0x00000800u) != 0u;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "DATA" && sub.size >= 12u) {
                    std::memcpy(placement.position, sub.data, sizeof(placement.position));
                } else if (sub.type == "XESP") {
                    placement.enableParent = readU32(sub, 0);
                }
            }
            placements.push_back(placement);
        };
        reader.walk(visitor);
    }
    std::cout << "\n" << placements.size() << " placement(s):\n";
    for (const Placement& placement : placements) {
        const auto named = cellNames.find(placement.cellFormId);
        std::cout << "  ref " << std::hex << placement.refFormId << std::dec
                  << "  cell " << std::hex << placement.cellFormId << std::dec
                  << " (" << (named == cellNames.end() ? std::string("<unnamed>") : named->second) << ")"
                  << "  pos (" << placement.position[0] << ", " << placement.position[1]
                  << ", " << placement.position[2] << ")";
        if (placement.initiallyDisabled) { std::cout << "  INITIALLY-DISABLED"; }
        if (placement.enableParent != 0u) {
            std::cout << "  enableParent=" << std::hex << placement.enableParent << std::dec;
        }
        std::cout << "\n";
    }

    // Pass 3: the AI packages he carries, and the script that drives him.
    std::vector<std::uint32_t> packageTargets;
    {
        std::set<std::uint32_t> wantedPackages(packageFormIds.begin(), packageFormIds.end());
        odai::importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const odai::importer::fnv::EsmRecordHeaderView& header) {
            return (header.type == "PACK" && wantedPackages.count(header.formId) != 0u) ||
                   (header.type == "SCPT" && header.formId == scriptFormId);
        };
        visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
            if (record.type == "PACK") {
                std::string edid;
                std::uint32_t packageType = 0xffffffffu;
                std::uint32_t locationType = 0xffffffffu;
                std::uint32_t locationTarget = 0;
                for (const auto& sub : record.subrecords) {
                    if (sub.type == "EDID") { edid = subString(sub); }
                    // PKDT: u32 flags, u8 type, ...
                    else if (sub.type == "PKDT" && sub.size >= 5u) { packageType = sub.data[4]; }
                    // PLDT: u32 type, u32 target, i32 radius.
                    else if (sub.type == "PLDT" && sub.size >= 8u) {
                        locationType = readU32(sub, 0);
                        locationTarget = readU32(sub, 4);
                    }
                }
                std::cout << "  PACK " << std::hex << record.formId << std::dec << " " << edid
                          << "  type=" << static_cast<int>(packageType);
                if (locationType != 0xffffffffu) {
                    std::cout << "  locationType=" << locationType
                              << " target=" << std::hex << locationTarget << std::dec;
                    if (locationTarget != 0u) { packageTargets.push_back(locationTarget); }
                }
                std::cout << "\n";
                return;
            }
            // SCPT: SCTX is the uncompiled source, which is where MoveTo /
            // Enable / Disable actually appear in readable form.
            for (const auto& sub : record.subrecords) {
                if (sub.type != "SCTX") { continue; }
                std::cout << "\n--- script source (" << sub.size << " bytes) ---\n"
                          << subString(sub) << "\n--- end script ---\n";
            }
        };
        std::cout << "\nAI packages:\n";
        reader.walk(visitor);
    }

    // Pass 4: resolve what the packages and enable-parents actually POINT AT.
    // A package target is a formID, and "target=16adc6" says nothing about
    // whether he is patrolling a marker, a door or another actor.
    // Patrol route. A Patrol package names ONE marker; the route is the chain
    // of markers reached from it by XLKR (linked reference), which is how
    // Bethesda expresses "walk this circuit". Every XMarker REFR is collected in
    // a single pass and the chain is then followed in memory -- following it by
    // re-walking the plugin per hop would be one full pass per marker.
    {
        struct Marker { float position[3] = {}; std::uint32_t linked = 0; };
        std::map<std::uint32_t, Marker> markers;
        odai::importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const odai::importer::fnv::EsmRecordHeaderView& header) {
            return header.type == "REFR";
        };
        visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
            Marker marker;
            bool linkedFound = false;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "XLKR") { marker.linked = readU32(sub, 0); linkedFound = true; }
                else if (sub.type == "DATA" && sub.size >= 12u) {
                    std::memcpy(marker.position, sub.data, sizeof(marker.position));
                }
            }
            if (linkedFound) { markers[record.formId] = marker; }
        };
        reader.walk(visitor);

        for (const std::uint32_t packageTarget : packageTargets) {
            if (markers.count(packageTarget) == 0u) { continue; }
            std::cout << "\npatrol route from " << std::hex << packageTarget << std::dec << ":\n";
            std::uint32_t current = packageTarget;
            std::set<std::uint32_t> visited;
            int hop = 0;
            while (current != 0u && visited.insert(current).second && hop < 32) {
                const auto found = markers.find(current);
                if (found == markers.end()) {
                    std::cout << "  " << hop << ": " << std::hex << current << std::dec
                              << " (not a linked marker)\n";
                    break;
                }
                std::cout << "  " << hop << ": " << std::hex << current << std::dec << "  ("
                          << found->second.position[0] << ", " << found->second.position[1]
                          << ", " << found->second.position[2] << ")\n";
                current = found->second.linked;
                ++hop;
            }
            if (current != 0u && visited.count(current) != 0u) {
                std::cout << "  loops back to " << std::hex << current << std::dec << "\n";
            }
        }
    }

    std::set<std::uint32_t> lookups;
    for (const Placement& placement : placements) {
        if (placement.enableParent != 0u) { lookups.insert(placement.enableParent); }
    }
    if (!lookups.empty()) {
        std::cout << "\nreferenced records:\n";
        odai::importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const odai::importer::fnv::EsmRecordHeaderView& header) {
            return lookups.count(header.formId) != 0u;
        };
        visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
            std::string edid;
            std::uint32_t base = 0;
            float position[3] = {};
            bool hasPosition = false;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "EDID") { edid = subString(sub); }
                else if (sub.type == "NAME") { base = readU32(sub, 0); }
                else if (sub.type == "DATA" && sub.size >= 12u) {
                    std::memcpy(position, sub.data, sizeof(position));
                    hasPosition = true;
                }
            }
            std::cout << "  " << std::hex << record.formId << std::dec << " " << record.type
                      << " " << edid;
            if (base != 0u) { std::cout << "  base=" << std::hex << base << std::dec; }
            if (hasPosition) {
                std::cout << "  pos (" << position[0] << ", " << position[1] << ", "
                          << position[2] << ")";
            }
            std::cout << "\n    subrecords:";
            for (const auto& sub : record.subrecords) {
                std::cout << " " << sub.type << "(" << sub.size << ")";
            }
            std::cout << "\n";
        };
        reader.walk(visitor);
    }
    return 0;
}

// Which CTDA function binds an INFO to a VOICE TYPE rather than to one actor.
//
// Derived the same way GetIsID was: histogram every CTDA function index whose
// param1 is the wanted VTYP's formID, across every INFO in the plugin. The one
// that dominates is the answer. Guessing from documentation is how you end up
// attributing a whole town's dialogue through the wrong field and finding out
// only when the lines are visibly wrong.
int probeVoiceTypeDialogue(const std::filesystem::path& pluginPath, const std::string& wantedEditorId) {
    odai::importer::fnv::EsmReader reader;
    if (!reader.open(pluginPath)) {
        std::cout << "open failed: " << reader.lastError() << "\n";
        return 1;
    }
    const std::string wanted = toLowerAscii(wantedEditorId);

    std::uint32_t voiceTypeFormId = 0;
    {
        odai::importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const odai::importer::fnv::EsmRecordHeaderView& header) {
            return voiceTypeFormId == 0u && header.type == "VTYP";
        };
        visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
            if (voiceTypeFormId != 0u) { return; }
            for (const auto& sub : record.subrecords) {
                if (sub.type != "EDID") { continue; }
                std::string edid(
                    reinterpret_cast<const char*>(sub.data), static_cast<std::size_t>(sub.size));
                while (!edid.empty() && edid.back() == '\0') { edid.pop_back(); }
                if (toLowerAscii(edid) == wanted) { voiceTypeFormId = record.formId; }
            }
        };
        reader.walk(visitor);
    }
    if (voiceTypeFormId == 0u) {
        std::cout << "no VTYP with EditorID \"" << wantedEditorId << "\"\n";
        return 1;
    }
    std::cout << wantedEditorId << " = " << std::hex << voiceTypeFormId << std::dec << "\n";

    std::map<std::uint32_t, std::size_t> functionHistogram;
    std::size_t infosReferencing = 0;
    std::size_t infosWithText = 0;
    {
        odai::importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecordHeader = [](const odai::importer::fnv::EsmRecordHeaderView& header) {
            return header.type == "INFO";
        };
        visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
            bool referenced = false;
            bool hasText = false;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "NAM1" && sub.size > 1u) { hasText = true; }
                if (sub.type != "CTDA" || sub.size < 28u) { continue; }
                std::uint32_t function = 0;
                std::uint32_t param1 = 0;
                std::memcpy(&function, sub.data + 8, 4u);
                std::memcpy(&param1, sub.data + 12, 4u);
                if (param1 == voiceTypeFormId) {
                    ++functionHistogram[function];
                    referenced = true;
                }
            }
            infosReferencing += referenced ? 1u : 0u;
            infosWithText += (referenced && hasText) ? 1u : 0u;
        };
        reader.walk(visitor);
    }

    std::cout << infosReferencing << " INFO(s) name this voice type in a condition ("
              << infosWithText << " of them carry spoken text)\n"
              << "CTDA function indices whose param1 is this voice type:\n";
    for (const auto& [function, count] : functionHistogram) {
        std::cout << "  function " << function << "  x" << count << "\n";
    }
    return 0;
}

// Answers "what IS 0x104f04". A formID is how every record in the format
// refers to every other one, so the usual question mid-investigation is what
// type a reference lands on -- an inventory entry that resolves to nothing
// wearable could be ammo, a note, or an outfit this code does not follow yet,
// and those need different fixes.
int probeFormId(const std::filesystem::path& pluginPath, std::uint32_t wantedFormId) {
    odai::importer::fnv::EsmReader reader;
    if (!reader.open(pluginPath)) {
        std::cout << "open failed: " << reader.lastError() << "\n";
        return 1;
    }
    bool found = false;
    odai::importer::fnv::EsmReader::Visitor visitor;
    visitor.onRecordHeader = [&](const odai::importer::fnv::EsmRecordHeaderView& header) {
        return !found && header.formId == wantedFormId;
    };
    visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
        if (found) { return; }
        found = true;
        std::cout << std::hex << record.formId << std::dec << " " << record.type << "\n";
        for (const auto& sub : record.subrecords) {
            std::cout << "  " << sub.type << "(" << sub.size << ")";
            const std::string text(
                reinterpret_cast<const char*>(sub.data), static_cast<std::size_t>(sub.size));
            const bool printable = !text.empty() &&
                std::all_of(text.begin(), text.end() - 1, [](char c) {
                    return static_cast<unsigned char>(c) >= 32u &&
                        static_cast<unsigned char>(c) < 127u;
                });
            if (printable && text.back() == '\0') {
                std::cout << " \"" << text.substr(0, text.size() - 1) << "\"";
            } else if (sub.size == 4u) {
                std::uint32_t value = 0;
                std::memcpy(&value, sub.data, 4u);
                std::cout << " = " << std::hex << value << std::dec;
            }
            std::cout << "\n";
            if (sub.type == "VMAD") {
                odai::bethesda::VmadAttachments attachments;
                std::string vmadError;
                if (!odai::bethesda::readVmadAttachments(
                        std::span<const std::uint8_t>(sub.data, sub.size),
                        attachments, vmadError)) {
                    std::cout << "    decode error: " << vmadError << "\n";
                    continue;
                }
                for (const odai::bethesda::VmadScriptAttachment& script : attachments.scripts) {
                    std::cout << "    script " << script.className << "\n";
                    for (const odai::bethesda::VmadProperty& property : script.properties) {
                        std::cout << "      " << property.name << " type="
                                  << static_cast<unsigned>(property.value.type);
                        switch (property.value.type) {
                            case odai::bethesda::VmadValueType::Object:
                                std::cout << " form=0x" << std::hex
                                          << property.value.object.formId << std::dec
                                          << " alias=" << property.value.object.alias;
                                break;
                            case odai::bethesda::VmadValueType::Integer:
                                std::cout << " value=" << property.value.integer; break;
                            case odai::bethesda::VmadValueType::Float:
                                std::cout << " value=" << property.value.real; break;
                            case odai::bethesda::VmadValueType::Boolean:
                                std::cout << " value=" << property.value.boolean; break;
                            case odai::bethesda::VmadValueType::String:
                                std::cout << " value=\"" << property.value.string << "\""; break;
                            default:
                                std::cout << " count=" << property.value.array.size(); break;
                        }
                        std::cout << "\n";
                    }
                }
            }
        }
    };
    reader.walk(visitor);
    if (!found) {
        std::cout << "no record with formID " << std::hex << wantedFormId << std::dec << "\n";
        return 1;
    }
    return 0;
}

int probeRecordType(const std::filesystem::path& pluginPath, const std::string& wantedType, std::size_t limit) {
    odai::importer::fnv::EsmReader reader;
    if (!reader.open(pluginPath)) {
        std::cout << "open failed: " << reader.lastError() << "\n";
        return 1;
    }
    std::size_t seen = 0;
    std::size_t total = 0;
    std::map<std::string, std::size_t> subrecordCounts;
    std::map<std::string, std::map<std::uint32_t, std::size_t>> subrecordSizes;

    odai::importer::fnv::EsmReader::Visitor visitor;
    visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
        if (record.type != wantedType) {
            return;
        }
        ++total;
        for (const auto& sub : record.subrecords) {
            ++subrecordCounts[sub.type];
            ++subrecordSizes[sub.type][sub.size];
        }
        if (seen >= limit) {
            return;
        }
        ++seen;

        // The editor ID up front: without it every dump looks alike and there is
        // no way to cross-check a value against what the record obviously is.
        std::string editorId;
        for (const auto& sub : record.subrecords) {
            if (sub.type == "EDID" && sub.size > 0u) {
                editorId.assign(reinterpret_cast<const char*>(sub.data), sub.size);
                while (!editorId.empty() && editorId.back() == '\0') {
                    editorId.pop_back();
                }
                break;
            }
        }
        std::cout << "\n" << wantedType << " formId=0x" << std::hex << record.formId << std::dec
                  << (editorId.empty() ? "" : "  \"" + editorId + "\"") << "\n";

        for (const auto& sub : record.subrecords) {
            std::cout << "  " << sub.type << "  size=" << sub.size << "\n";
            if (sub.type == "EDID" || sub.type == "MODL" || sub.type == "FULL") {
                std::string text(reinterpret_cast<const char*>(sub.data), sub.size);
                while (!text.empty() && text.back() == '\0') {
                    text.pop_back();
                }
                std::cout << "      \"" << text << "\"\n";
                continue;
            }
            // Cap the dump: a DATA holding an instance array can be thousands of
            // bytes, and the first few structs settle the stride.
            const std::size_t words = std::min<std::size_t>(sub.size / 4u, 24u);
            for (std::size_t w = 0; w < words; ++w) {
                std::uint32_t u = 0;
                std::int32_t i = 0;
                float f = 0.0f;
                std::memcpy(&u, sub.data + (w * 4u), 4u);
                std::memcpy(&i, sub.data + (w * 4u), 4u);
                std::memcpy(&f, sub.data + (w * 4u), 4u);
                std::cout << "      @" << std::setw(3) << (w * 4u) << "  0x" << std::hex << std::setw(8)
                          << std::setfill('0') << u << std::setfill(' ') << std::dec
                          << "  u=" << std::setw(11) << u << "  i=" << std::setw(11) << i
                          << "  f=" << f << "  |";
                for (std::size_t b = 0; b < 4u; ++b) {
                    const auto ch = static_cast<unsigned char>(sub.data[(w * 4u) + b]);
                    std::cout << ((ch >= 0x20u && ch <= 0x7eu) ? static_cast<char>(ch) : '.');
                }
                std::cout << "|\n";
            }
            if (sub.size / 4u > words) {
                std::cout << "      ... " << ((sub.size / 4u) - words) << " more word(s)\n";
            }
            if (sub.size % 4u != 0u) {
                std::cout << "      (+" << (sub.size % 4u) << " trailing byte(s) -- size is NOT a multiple of 4)\n";
            }
        }
    };
    if (!reader.walk(visitor)) {
        std::cout << "walk failed: " << reader.lastError() << "\n";
        return 1;
    }

    std::cout << "\n" << total << " " << wantedType << " record(s). Subrecord census:\n";
    for (const auto& [type, count] : subrecordCounts) {
        std::cout << "  " << count << "x " << type;
        // A single size means a fixed struct; several means either a variable
        // payload or an array whose element count differs per record. Both are
        // things a reader has to know before it walks anything.
        const auto& sizes = subrecordSizes[type];
        std::cout << "   size(s):";
        std::size_t shown = 0;
        for (const auto& [size, howMany] : sizes) {
            if (shown++ >= 6u) {
                std::cout << " ...(" << (sizes.size() - 6u) << " more)";
                break;
            }
            std::cout << " " << size << "(x" << howMany << ")";
        }
        std::cout << "\n";
    }
    return 0;
}

// Which cells actually place lights, and how many.
//
// "The importer emits no lights" and "this cell has no lights to emit" look
// identical from a single cell build, and the first Goodsprings cells tried
// happened to be the second case. This resolves the ambiguity and, as a side
// effect, says where to point a camera to see the feature working.
int probeRefsByBaseType(
    const std::filesystem::path& dataPath, const std::filesystem::path& pluginPath,
    const std::string& baseType, std::size_t limit) {
    using namespace odai::importer::fnv;
    (void)dataPath;

    std::string error;
    FalloutWorldTables tables;
    if (!buildFalloutWorldTables(pluginPath, tables, error)) {
        std::cout << "world tables FAILED: " << error << "\n";
        return 1;
    }
    // Which base formIDs count as this type. LIGH is answered from its own
    // table because a light is not required to have a model; everything else
    // comes from the record-type map the static tables already carry.
    std::unordered_set<std::uint32_t> wanted;
    if (baseType == "LIGH") {
        for (const auto& [formId, unused] : tables.lightsByFormId) {
            (void)unused;
            wanted.insert(formId);
        }
    } else {
        for (const auto& [formId, type] : tables.staticRecordTypes) {
            if (type == baseType) {
                wanted.insert(formId);
            }
        }
    }
    std::cout << baseType << " base records known: " << wanted.size() << "\n";

    FalloutSceneData data;
    FalloutExtractFilter filter{};  // everything: this is the expensive full pass, on purpose
    if (!extractFalloutScene(pluginPath, filter, data, error)) {
        std::cout << "extract FAILED: " << error << "\n";
        return 1;
    }

    struct CellLights {
        const FalloutCellRecord* cell = nullptr;
        std::size_t lightRefs = 0;
    };
    std::vector<CellLights> hits;
    std::size_t totalLightRefs = 0;
    std::size_t exteriorLightRefs = 0;
    for (const FalloutCellRecord& cell : data.cells) {
        std::size_t count = 0;
        for (const auto& ref : cell.references) {
            if (wanted.count(ref.baseFormId) != 0u) {
                ++count;
            }
        }
        if (count == 0u) {
            continue;
        }
        totalLightRefs += count;
        if (!cell.isInterior) {
            exteriorLightRefs += count;
        }
        hits.push_back({&cell, count});
    }
    std::sort(hits.begin(), hits.end(), [](const CellLights& a, const CellLights& b) {
        return a.lightRefs > b.lightRefs;
    });

    std::cout << totalLightRefs << " " << baseType << " reference(s) across " << hits.size() << " cell(s); "
              << exteriorLightRefs << " of them in exterior cells.\n";
    for (std::size_t i = 0; i < hits.size() && i < limit; ++i) {
        const FalloutCellRecord& cell = *hits[i].cell;
        std::cout << "  " << hits[i].lightRefs << "  " << (cell.isInterior ? "interior" : "exterior")
                  << "  \"" << cell.editorId << "\"";
        if (!cell.isInterior) {
            std::cout << "  grid (" << cell.gridX << "," << cell.gridZ << ")";
        }
        std::cout << "\n";
    }
    return 0;
}

int probePlacements(
    const std::filesystem::path& pluginPath, std::uint32_t wantedBaseFormId,
    std::size_t limit) {
    using namespace odai::importer::fnv;
    std::string error;
    FalloutSceneData data;
    FalloutExtractFilter filter{};
    if (!extractFalloutScene(pluginPath, filter, data, error)) {
        std::cout << "extract FAILED: " << error << "\n";
        return 1;
    }
    std::size_t matches = 0u;
    for (const FalloutCellRecord& cell : data.cells) {
        for (const FalloutPlacedReference& ref : cell.references) {
            if (ref.baseFormId != wantedBaseFormId) {
                continue;
            }
            ++matches;
            if (matches > limit) {
                continue;
            }
            std::cout << "  ref=0x" << std::hex << ref.formId << std::dec
                      << (cell.isInterior ? " interior" : " exterior")
                      << " cell=0x" << std::hex << cell.formId << std::dec;
            if (!cell.isInterior) {
                std::cout << " grid=(" << cell.gridX << "," << cell.gridZ << ")";
            }
            std::cout << " bethesda=(" << ref.position[0] << "," << ref.position[1]
                      << "," << ref.position[2] << ") engine=(" << ref.position[0]
                      << "," << ref.position[2] << "," << -ref.position[1] << ")"
                      << " rotation=(" << ref.rotationRadians[0] << "," << ref.rotationRadians[1]
                      << "," << ref.rotationRadians[2] << ") scale=" << ref.scale << "\n";
        }
    }
    std::cout << matches << " placement(s) of base 0x" << std::hex
              << wantedBaseFormId << std::dec << "\n";
    return matches == 0u ? 1 : 0;
}

int probeModelPlacements(
    const std::filesystem::path& pluginPath, const std::string& modelSubstring,
    std::size_t limit) {
    using namespace odai::importer::fnv;
    std::string error;
    EsmReader reader;
    if (!reader.open(pluginPath)) {
        std::cout << "open FAILED: " << reader.lastError() << "\n";
        return 1;
    }
    const std::string wanted = toLowerAscii(modelSubstring);
    std::unordered_set<std::uint32_t> baseFormIds;
    EsmReader::Visitor visitor;
    visitor.onRecord = [&](const EsmRecordView& record) {
        std::string editorId;
        std::string modelPath;
        std::string matchedSubrecord;
        for (const EsmSubrecordView& sub : record.subrecords) {
            std::string* destination = nullptr;
            if (sub.type == "EDID") destination = &editorId;
            if (sub.type == "MODL" || sub.type == "MOD2") destination = &modelPath;
            if (destination != nullptr && sub.size != 0u) {
                destination->assign(reinterpret_cast<const char*>(sub.data), sub.size);
                while (!destination->empty() && destination->back() == '\0') {
                    destination->pop_back();
                }
            }
            if (sub.size != 0u && matchedSubrecord.empty()) {
                const std::string payload(
                    reinterpret_cast<const char*>(sub.data), sub.size);
                if (toLowerAscii(payload).find(wanted) != std::string::npos) {
                    matchedSubrecord = sub.type;
                }
            }
        }
        if (matchedSubrecord.empty()) return;
        baseFormIds.insert(record.formId);
        std::cout << record.type << " base=0x" << std::hex << record.formId << std::dec
                  << " editor=\"" << editorId
                  << "\" matched=" << matchedSubrecord
                  << (modelPath.empty() ? std::string() : " model=\"" + modelPath + "\"")
                  << "\n";
    };
    if (!reader.walk(visitor)) {
        std::cout << "walk FAILED: " << reader.lastError() << "\n";
        return 1;
    }
    if (baseFormIds.empty()) {
        std::cout << "no base models contain \"" << modelSubstring << "\"\n";
        return 1;
    }

    FalloutSceneData data;
    FalloutExtractFilter filter{};
    if (!extractFalloutScene(pluginPath, filter, data, error)) {
        std::cout << "extract FAILED: " << error << "\n";
        return 1;
    }
    std::size_t matches = 0u;
    for (const FalloutCellRecord& cell : data.cells) {
        for (const FalloutPlacedReference& ref : cell.references) {
            if (!baseFormIds.contains(ref.baseFormId)) {
                continue;
            }
            ++matches;
            if (matches > limit) {
                continue;
            }
            std::cout << "  base=0x" << std::hex << ref.baseFormId
                      << " ref=0x" << ref.formId << std::dec
                      << (cell.isInterior ? " interior" : " exterior");
            if (cell.isInterior) {
                std::cout << " cell=\"" << cell.editorId << "\"";
            } else {
                std::cout << " grid=(" << cell.gridX << "," << cell.gridZ << ")";
            }
            std::cout << " engine=(" << ref.position[0] << "," << ref.position[2]
                      << "," << -ref.position[1] << ")"
                      << " rotation=(" << ref.rotationRadians[0] << "," << ref.rotationRadians[1]
                      << "," << ref.rotationRadians[2] << ") scale=" << ref.scale << "\n";
        }
    }
    std::cout << matches << " placement(s) matching \"" << modelSubstring << "\"\n";
    return matches == 0u ? 1 : 0;
}

int probePlugin(const std::filesystem::path& pluginPath, std::size_t typeLimit) {
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
    // The tail matters as much as the head here: the records worth importing
    // next (WATR, LIGH, REGN, WTHR) sit well below the top 20, buried under
    // REFR/CELL/LAND. Default to the same 20 as before, but let a caller ask
    // for the whole census.
    const std::size_t shown = std::min(sorted.size(), typeLimit);
    for (std::size_t i = 0; i < shown; ++i) {
        std::cout << "  " << sorted[i].second << "  " << sorted[i].first << "\n";
    }
    if (shown < sorted.size()) {
        std::cout << "  ... " << (sorted.size() - shown) << " more type(s); pass a count to list them\n";
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
    if (std::getenv("ODAI_PROBE_LIST_INSTANCES") != nullptr) {
        for (const auto& instance : scene.instances) {
            std::cout << "  instance ref=\"" << instance.sourceReferenceIdentity
                      << "\" model=\"" << instance.modelPath << "\" pos=("
                      << instance.transform[3] << ", " << instance.transform[7]
                      << ", " << instance.transform[11] << ") visible="
                      << (instance.initiallyVisible ? "yes" : "no") << "\n";
        }
    }

    // Per-texture alpha-test census over the packed vertices. loadImportedScene
    // just re-ran the content inference, so a texture the current classifier
    // calls opaque that still shows flagged vertices here proves the flag was
    // BAKED into the file by an older build -- the load-time pass only ORs
    // flags in, it cannot clear a stale one.
    {
        struct TexFlagCounts {
            std::size_t total = 0;
            std::size_t alphaTested = 0;
            std::size_t blended = 0;
        };
        std::map<std::uint32_t, TexFlagCounts> byTexture;
        for (const auto& vertex : scene.packedVertices) {
            TexFlagCounts& counts = byTexture[vertex.textureIndex];
            ++counts.total;
            if ((vertex.flags & odai::importer::kImportedSceneMaterialFlagAlphaTest) != 0u) {
                ++counts.alphaTested;
            }
            if ((vertex.flags & odai::importer::kImportedSceneMaterialFlagAlphaBlend) != 0u) {
                ++counts.blended;
            }
        }
        std::cout << "alpha-test census (textures with any flagged vertex):\n";
        for (const auto& [textureIndex, counts] : byTexture) {
            if (counts.alphaTested == 0u && counts.blended == 0u) {
                continue;
            }
            const std::string name = textureIndex < scene.textures.size()
                ? scene.textures[textureIndex].sourcePath
                : std::string("<out of range>");
            std::cout << "  tex[" << textureIndex << "] verts=" << counts.total
                      << " alphaTest=" << counts.alphaTested
                      << " blend=" << counts.blended
                      << " \"" << name << "\"\n";
        }
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

int scriptCheck(
    const std::filesystem::path& dataPath, std::string virtualPath, bool strict) {
    std::vector<std::uint8_t> bytes;
    std::string error;
    const std::filesystem::path directPath = virtualPath;
    if (std::filesystem::is_regular_file(directPath)) {
        std::ifstream input(directPath, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    } else {
        std::replace(virtualPath.begin(), virtualPath.end(), '/', '\\');
        if (virtualPath.find('\\') == std::string::npos) virtualPath = "scripts\\" + virtualPath;
        if (!toLowerAscii(virtualPath).ends_with(".pex")) virtualPath += ".pex";
        odai::importer::fnv::FalloutAssetSource assets;
        if (!assets.open(dataPath, odai::importer::fnv::kBsaContentMisc)) {
            std::cout << nlohmann::json({{"ok", false}, {"script", virtualPath},
                {"compatibility_errors", {"could not index the Data directory"}}}).dump(2) << '\n';
            return 1;
        }
        if (!assets.resolveAsset(virtualPath, bytes, error)) {
            std::cout << nlohmann::json({{"ok", false}, {"script", virtualPath},
                {"compatibility_errors", {error}}}).dump(2) << '\n';
            return 1;
        }
    }

    odai::bethesda::PexScript script;
    if (!odai::bethesda::readPexScript(bytes, script, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"script", virtualPath},
            {"size", bytes.size()}, {"compatibility_errors", {error}}}).dump(2) << '\n';
        return 1;
    }

    const odai::bethesda::PexCompatibilityReport report =
        odai::bethesda::inspectPexCompatibility(script);
    const bool compatible = report.compatibilityErrors.empty() && report.unresolvedCalls.empty();
    nlohmann::json objectProperties = nlohmann::json::array();
    for (const odai::bethesda::PexObjectInfo& object : script.objectInfo) {
        for (const odai::bethesda::PexPropertyInfo& property : object.properties) {
            objectProperties.push_back({{"object", object.name}, {"name", property.name},
                {"type", property.type}, {"auto_variable", property.autoVariable},
                {"flags", property.flags}});
        }
    }
    nlohmann::json functions = nlohmann::json::array();
    for (const odai::bethesda::PexFunctionInfo& function : script.functions) {
        nlohmann::json instructions = nlohmann::json::array();
        for (std::size_t index = 0u; index < function.instructions.size(); ++index) {
            const odai::bethesda::PexInstructionInfo& instruction =
                function.instructions[index];
            nlohmann::json arguments = nlohmann::json::array();
            for (const odai::bethesda::PexValue& value : instruction.arguments) {
                nlohmann::json argument{{"kind", static_cast<std::uint8_t>(value.kind)}};
                if (value.kind == odai::bethesda::PexValueKind::Integer) {
                    argument["integer"] = value.integer;
                } else if (value.kind == odai::bethesda::PexValueKind::Float) {
                    argument["float"] = value.real;
                } else if (value.kind == odai::bethesda::PexValueKind::Boolean) {
                    argument["boolean"] = value.boolean;
                } else if (value.kind != odai::bethesda::PexValueKind::None) {
                    argument["text"] = value.text;
                }
                arguments.push_back(std::move(argument));
            }
            instructions.push_back({{"index", index},
                {"opcode", odai::bethesda::pexOpcodeName(instruction.opcode)},
                {"arguments", std::move(arguments)}});
        }
        functions.push_back({{"qualified_name", function.qualifiedName()},
            {"state", function.stateName}, {"name", function.name},
            {"native", function.native()}, {"parameters", function.parameters},
            {"parameter_types", function.parameterTypes},
            {"instructions", std::move(instructions)}});
    }
    std::cout << nlohmann::json({
        {"ok", !strict || compatible}, {"strict", strict}, {"script", virtualPath},
        {"size", bytes.size()},
        {"pex", {{"major", script.module.majorVersion}, {"minor", script.module.minorVersion},
                 {"game_id", script.module.gameId},
                 {"byte_order", script.module.bigEndian ? "big" : "little"},
                 {"compilation_time", script.module.compilationTime},
                 {"source", script.module.sourceFile}, {"user", script.module.userName},
                 {"machine", script.module.machineName},
                 {"string_count", script.module.strings.size()},
                 {"object_count", script.objects.size()},
                 {"function_count", script.functions.size()}}},
        {"opcode_histogram", report.opcodeHistogram},
        {"declared_natives", report.declaredNatives},
        {"called_functions", report.calledFunctions},
        {"functions", std::move(functions)},
        {"properties", std::move(objectProperties)},
        {"unresolved_calls", report.unresolvedCalls},
        {"executable_compatible", compatible},
        {"compatibility_errors", report.compatibilityErrors}}).dump(2) << '\n';
    return strict && !compatible ? 1 : 0;
}

int questTrace(
    const std::filesystem::path& dataPath,
    const std::string& pluginName,
    const std::string& questEditorId) {
    odai::importer::fnv::EsmReader reader;
    if (!reader.open(dataPath / pluginName)) {
        std::cout << nlohmann::json({{"ok", false}, {"plugin", pluginName},
            {"quest", questEditorId}, {"compatibility_errors", {reader.lastError()}}}).dump(2) << '\n';
        return 1;
    }

    nlohmann::json foundRecord;
    bool found = false;
    std::vector<std::uint8_t> questVmad;
    odai::importer::fnv::EsmReader::Visitor visitor;
    visitor.onRecordHeader = [](const odai::importer::fnv::EsmRecordHeaderView& header) {
        return header.type == "QUST";
    };
    visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
        std::string editorId;
        for (const auto& subrecord : record.subrecords) {
            if (subrecord.type != "EDID") continue;
            editorId.assign(reinterpret_cast<const char*>(subrecord.data), subrecord.size);
            while (!editorId.empty() && editorId.back() == '\0') editorId.pop_back();
            break;
        }
        if (toLowerAscii(editorId) != toLowerAscii(questEditorId)) return;

        found = true;
        std::map<std::string, std::size_t> counts;
        std::map<std::string, std::uint64_t> bytesByType;
        for (const auto& subrecord : record.subrecords) {
            ++counts[subrecord.type];
            bytesByType[subrecord.type] += subrecord.size;
            if (subrecord.type == "VMAD" && questVmad.empty()) {
                questVmad.assign(subrecord.data, subrecord.data + subrecord.size);
            }
        }
        nlohmann::json subrecords = nlohmann::json::object();
        for (const auto& [type, count] : counts) {
            subrecords[type] = {{"count", count}, {"bytes", bytesByType[type]}};
        }
        foundRecord = {
            {"ok", true}, {"plugin", pluginName}, {"quest", editorId},
            {"form_id", record.formId}, {"record_flags", record.flags},
            {"vmad_bytes", bytesByType["VMAD"]}, {"condition_count", counts["CTDA"]},
            {"alias_markers", counts["ALST"] + counts["ALLS"]},
            {"subrecords", std::move(subrecords)},
            {"compatibility_errors", nlohmann::json::array()}
        };
        odai::bethesda::SkyrimQuestDefinition definition;
        std::string definitionError;
        if (!odai::bethesda::readSkyrimQuest(
                record,
                odai::bethesda::makeRecordKey(pluginName, record.formId & 0x00ffffffu),
                definition, definitionError)) {
            foundRecord["compatibility_errors"].push_back(definitionError);
        } else {
            nlohmann::json aliases = nlohmann::json::array();
            for (const odai::bethesda::SkyrimQuestAliasDefinition& alias : definition.aliases) {
                aliases.push_back({{"id", alias.id}, {"name", alias.name},
                    {"forced_reference_form_id", alias.forcedReferenceFormId},
                    {"unique_actor_form_id", alias.uniqueActorFormId},
                    {"created_object_form_id", alias.createdObjectFormId},
                    {"created_in_alias_id", alias.createdInAliasId},
                    {"created_level", alias.createdLevel}});
            }
            nlohmann::json fragments = nlohmann::json::array();
            for (const odai::bethesda::VmadQuestFragment& fragment :
                 definition.stageFragments) {
                fragments.push_back({{"stage", fragment.stage},
                    {"log_entry", fragment.logEntry},
                    {"script_class", fragment.scriptClass},
                    {"function", fragment.function}});
            }
            std::size_t aliasScriptCount = 0u;
            nlohmann::json aliasScriptAttachments = nlohmann::json::array();
            for (const odai::bethesda::VmadQuestAliasAttachment& alias :
                 definition.aliasScripts) {
                aliasScriptCount += alias.scripts.size();
                nlohmann::json scriptClasses = nlohmann::json::array();
                for (const odai::bethesda::VmadScriptAttachment& script : alias.scripts) {
                    nlohmann::json properties = nlohmann::json::array();
                    for (const odai::bethesda::VmadProperty& property : script.properties) {
                        nlohmann::json value{{"type",
                            static_cast<std::uint8_t>(property.value.type)}};
                        if (property.value.type == odai::bethesda::VmadValueType::Object) {
                            value["form_id"] = property.value.object.formId;
                            value["alias"] = property.value.object.alias;
                        } else if (property.value.type ==
                                odai::bethesda::VmadValueType::Integer) {
                            value["integer"] = property.value.integer;
                        } else if (property.value.type ==
                                odai::bethesda::VmadValueType::Float) {
                            value["float"] = property.value.real;
                        } else if (property.value.type ==
                                odai::bethesda::VmadValueType::Boolean) {
                            value["boolean"] = property.value.boolean;
                        } else if (property.value.type ==
                                odai::bethesda::VmadValueType::String) {
                            value["string"] = property.value.string;
                        }
                        properties.push_back({{"name", property.name},
                            {"value", std::move(value)}});
                    }
                    scriptClasses.push_back({{"class", script.className},
                        {"properties", std::move(properties)}});
                }
                aliasScriptAttachments.push_back({{"form_id", alias.object.formId},
                    {"alias", alias.object.alias},
                    {"version", alias.version},
                    {"object_format", alias.objectFormat},
                    {"scripts", std::move(scriptClasses)}});
            }
            nlohmann::json stages = nlohmann::json::array();
            for (const odai::bethesda::SkyrimQuestStageDefinition& stage : definition.stages) {
                nlohmann::json entries = nlohmann::json::array();
                for (const odai::bethesda::SkyrimQuestLogEntryDefinition& entry :
                     stage.logEntries) {
                    nlohmann::json conditions = nlohmann::json::array();
                    for (const odai::bethesda::Condition& condition : entry.conditions) {
                        conditions.push_back({{"function", condition.function},
                            {"comparison", static_cast<std::uint8_t>(condition.comparison)},
                            {"value", condition.comparisonValue},
                            {"parameter_1", condition.parameter1},
                            {"parameter_2", condition.parameter2},
                            {"string_parameter_1", condition.stringParameter1},
                            {"string_parameter_2", condition.stringParameter2},
                            {"run_on", condition.runOn},
                            {"reference", condition.reference},
                            {"or_with_next", condition.orWithNext}});
                    }
                    entries.push_back({{"flags", entry.flags},
                        {"conditions", std::move(conditions)}});
                }
                stages.push_back({{"stage", stage.index},
                    {"log_entries", std::move(entries)}});
            }
            foundRecord["runtime_definition"] = {
                {"record_key", definition.record.toString()},
                {"stage_count", definition.stages.size()},
                {"stages", std::move(stages)},
                {"objective_count", definition.objectives.size()},
                {"alias_count", definition.aliases.size()},
                {"stage_fragment_count", definition.stageFragments.size()},
                {"stage_fragments", std::move(fragments)},
                {"alias_attachment_count", definition.aliasScripts.size()},
                {"alias_script_count", aliasScriptCount},
                {"alias_script_attachments", std::move(aliasScriptAttachments)},
                {"referenced_form_count", definition.referencedFormIds.size()},
                {"script_count", definition.scripts.scripts.size()},
                {"aliases", std::move(aliases)}};
        }
    };
    if (!reader.walk(visitor)) {
        std::cout << nlohmann::json({{"ok", false}, {"plugin", pluginName},
            {"quest", questEditorId}, {"compatibility_errors", {reader.lastError()}}}).dump(2) << '\n';
        return 1;
    }
    if (!found) {
        std::cout << nlohmann::json({{"ok", false}, {"plugin", pluginName},
            {"quest", questEditorId},
            {"compatibility_errors", {"quest EditorID not found"}}}).dump(2) << '\n';
        return 1;
    }
    if (questVmad.empty()) {
        foundRecord["compatibility_errors"].push_back(
            "quest has no VMAD attachment in this record");
        foundRecord["attached_scripts"] = nlohmann::json::array();
    } else {
        odai::bethesda::VmadAttachments attachments;
        std::string vmadError;
        if (!odai::bethesda::readVmadAttachments(questVmad, attachments, vmadError)) {
            foundRecord["compatibility_errors"].push_back(vmadError);
        } else {
            odai::importer::fnv::FalloutAssetSource assets;
            const bool assetsOpen = assets.open(dataPath, odai::importer::fnv::kBsaContentMisc);
            nlohmann::json attachedScripts = nlohmann::json::array();
            for (const odai::bethesda::VmadScriptAttachment& attachment : attachments.scripts) {
                nlohmann::json scriptJson{{"class", attachment.className},
                    {"property_count", attachment.properties.size()}};
                nlohmann::json objectProperties = nlohmann::json::array();
                for (const odai::bethesda::VmadProperty& property : attachment.properties) {
                    if (property.value.type == odai::bethesda::VmadValueType::Object) {
                        objectProperties.push_back({{"name", property.name},
                            {"form_id", property.value.object.formId},
                            {"alias", property.value.object.alias}});
                    }
                }
                scriptJson["object_properties"] = std::move(objectProperties);
                std::vector<std::uint8_t> pexBytes;
                std::string pexError;
                const std::string scriptPath = "scripts\\" + attachment.className + ".pex";
                odai::bethesda::PexScript script;
                if (!assetsOpen || !assets.resolveAsset(scriptPath, pexBytes, pexError)) {
                    scriptJson["pex"] = "missing";
                    scriptJson["error"] = assetsOpen ? pexError : "could not index script archives";
                    foundRecord["compatibility_errors"].push_back(
                        "missing script " + attachment.className);
                } else if (!odai::bethesda::readPexScript(pexBytes, script, pexError)) {
                    scriptJson["pex"] = "malformed";
                    scriptJson["error"] = pexError;
                    foundRecord["compatibility_errors"].push_back(
                        attachment.className + ": " + pexError);
                } else {
                    const odai::bethesda::PexCompatibilityReport report =
                        odai::bethesda::inspectPexCompatibility(script);
                    scriptJson["pex"] = "decoded";
                    scriptJson["opcode_histogram"] = report.opcodeHistogram;
                    scriptJson["called_functions"] = report.calledFunctions;
                    scriptJson["unresolved_calls"] = report.unresolvedCalls;
                    scriptJson["compatibility_errors"] = report.compatibilityErrors;
                    for (const std::string& compatibilityError : report.compatibilityErrors) {
                        foundRecord["compatibility_errors"].push_back(
                            attachment.className + ": " + compatibilityError);
                    }
                    for (const std::string& unresolvedCall : report.unresolvedCalls) {
                        foundRecord["compatibility_errors"].push_back(
                            attachment.className + ": unresolved call binding " + unresolvedCall);
                    }
                }
                attachedScripts.push_back(std::move(scriptJson));
            }
            foundRecord["vmad"] = {{"version", attachments.version},
                {"object_format", attachments.objectFormat},
                {"trailing_bytes", questVmad.size() - attachments.trailingOffset}};
            foundRecord["attached_scripts"] = std::move(attachedScripts);
        }
    }
    std::cout << foundRecord.dump(2) << '\n';
    return 0;
}

// Dumps the authored Skyrim dialogue closure for one quest. Unlike the older
// Fallout speaker probe, this follows TES5's DIAL QNAM -> QUST ownership and
// topic-children GRUP labels, resolves INFO response IDs through ILSTRINGS,
// and preserves every CTDA/VMAD gate for runtime compatibility work.
int skyrimDialogueTrace(
    const std::filesystem::path& dataPath,
    const std::string& pluginName,
    const std::string& questEditorId) {
    using namespace odai::importer::fnv;
    EsmReader reader;
    if (!reader.open(dataPath / pluginName)) {
        std::cout << nlohmann::json({{"ok", false}, {"plugin", pluginName},
            {"quest", questEditorId}, {"compatibility_errors", {reader.lastError()}}}).dump(2)
                  << '\n';
        return 1;
    }

    std::uint32_t questFormId = 0u;
    EsmReader::Visitor questVisitor;
    questVisitor.onRecordHeader = [](const EsmRecordHeaderView& header) {
        return header.type == "QUST";
    };
    questVisitor.onRecord = [&](const EsmRecordView& record) {
        for (const EsmSubrecordView& subrecord : record.subrecords) {
            if (subrecord.type != "EDID") continue;
            std::string editorId(reinterpret_cast<const char*>(subrecord.data), subrecord.size);
            while (!editorId.empty() && editorId.back() == '\0') editorId.pop_back();
            if (toLowerAscii(editorId) == toLowerAscii(questEditorId)) questFormId = record.formId;
            break;
        }
    };
    if (!reader.walk(questVisitor) || questFormId == 0u) {
        std::cout << nlohmann::json({{"ok", false}, {"plugin", pluginName},
            {"quest", questEditorId}, {"compatibility_errors",
                {questFormId == 0u ? "quest EditorID not found" : reader.lastError()}}}).dump(2)
                  << '\n';
        return 1;
    }

    FalloutStringTable strings;
    FalloutStringTable ilStrings;
    FalloutAssetSource assets;
    std::string stringError;
    const bool assetsOpened = assets.open(dataPath, kBsaContentMisc);
    std::string promptStringError;
    const bool promptStringsLoaded = assetsOpened &&
        loadFalloutStringTable(assets, pluginName, falloutStringLanguage(),
            FalloutStringFileKind::Strings, strings, promptStringError);
    const bool stringsLoaded = assetsOpened &&
        loadFalloutStringTable(assets, pluginName, falloutStringLanguage(),
            FalloutStringFileKind::IlStrings, ilStrings, stringError);

    std::set<std::uint32_t> questTopics;
    nlohmann::json branches = nlohmann::json::array();
    nlohmann::json topics = nlohmann::json::array();
    nlohmann::json infos = nlohmann::json::array();
    std::uint32_t currentTopic = 0u;
    EsmReader::Visitor dialogueVisitor;
    dialogueVisitor.onGroupEnter = [&](const EsmGroupView& group) {
        if (group.groupType == 7 && group.rawLabel.size() == 4u) {
            std::memcpy(&currentTopic, group.rawLabel.data(), sizeof(currentTopic));
        }
        return true;
    };
    dialogueVisitor.onRecordHeader = [](const EsmRecordHeaderView& header) {
        return header.type == "DLBR" || header.type == "DIAL" ||
            header.type == "INFO";
    };
    dialogueVisitor.onRecord = [&](const EsmRecordView& record) {
        const auto readU32 = [](const EsmSubrecordView& subrecord) {
            std::uint32_t value = 0u;
            if (subrecord.size >= 4u) std::memcpy(&value, subrecord.data, sizeof(value));
            return value;
        };
        if (record.type == "DLBR") {
            std::uint32_t owner = 0u;
            std::uint32_t startTopic = 0u;
            std::uint32_t flags = 0u;
            std::string editorId;
            for (const EsmSubrecordView& subrecord : record.subrecords) {
                if (subrecord.type == "QNAM") owner = readU32(subrecord);
                else if (subrecord.type == "SNAM") startTopic = readU32(subrecord);
                else if (subrecord.type == "DNAM") flags = readU32(subrecord);
                else if (subrecord.type == "EDID") {
                    editorId.assign(reinterpret_cast<const char*>(subrecord.data), subrecord.size);
                    while (!editorId.empty() && editorId.back() == '\0') editorId.pop_back();
                }
            }
            if (owner == questFormId) {
                branches.push_back({{"form_id", record.formId}, {"editor_id", editorId},
                    {"start_topic", startTopic}, {"flags", flags}});
            }
            return;
        }
        if (record.type == "DIAL") {
            std::uint32_t owner = 0u;
            std::string editorId;
            std::uint32_t fullStringId = 0u;
            std::uint32_t branch = 0u;
            std::uint8_t type = 0xffu;
            for (const EsmSubrecordView& subrecord : record.subrecords) {
                if (subrecord.type == "QNAM") owner = readU32(subrecord);
                else if (subrecord.type == "FULL") fullStringId = readU32(subrecord);
                else if (subrecord.type == "BNAM") branch = readU32(subrecord);
                else if (subrecord.type == "DATA" && subrecord.size != 0u) type = subrecord.data[0];
                else if (subrecord.type == "EDID") {
                    editorId.assign(reinterpret_cast<const char*>(subrecord.data), subrecord.size);
                    while (!editorId.empty() && editorId.back() == '\0') editorId.pop_back();
                }
            }
            if (owner != questFormId) return;
            questTopics.insert(record.formId);
            const std::string* prompt = promptStringsLoaded
                ? strings.find(fullStringId) : nullptr;
            topics.push_back({{"form_id", record.formId}, {"editor_id", editorId},
                {"type", type}, {"full_string_id", fullStringId},
                {"branch_form_id", branch},
                {"prompt", prompt == nullptr ? std::string() : *prompt}});
            return;
        }
        if (!questTopics.contains(currentTopic)) return;

        nlohmann::json responses = nlohmann::json::array();
        nlohmann::json conditions = nlohmann::json::array();
        nlohmann::json links = nlohmann::json::array();
        nlohmann::json subrecords = nlohmann::json::array();
        std::uint32_t flags = 0u;
        std::uint32_t previous = 0u;
        std::uint32_t promptStringId = 0u;
        std::size_t vmadBytes = 0u;
        std::uint8_t vmadFlags = 0u;
        nlohmann::json vmadFragments = nlohmann::json::array();
        std::string vmadError;
        std::optional<std::size_t> currentConditionIndex;
        for (const EsmSubrecordView& subrecord : record.subrecords) {
            subrecords.push_back({{"type", subrecord.type}, {"size", subrecord.size}});
            if (subrecord.type == "ENAM") flags = readU32(subrecord);
            else if (subrecord.type == "RNAM") promptStringId = readU32(subrecord);
            else if (subrecord.type == "PNAM") previous = readU32(subrecord);
            else if (subrecord.type == "TCLT") links.push_back(readU32(subrecord));
            else if (subrecord.type == "NAM1") {
                const std::uint32_t stringId = readU32(subrecord);
                const std::string* text = stringsLoaded ? ilStrings.find(stringId) : nullptr;
                responses.push_back({{"string_id", stringId},
                    {"text", text == nullptr ? std::string() : *text}});
            } else if (subrecord.type == "CTDA") {
                odai::bethesda::Condition condition;
                std::string error;
                if (odai::bethesda::readCondition(
                        std::span<const std::uint8_t>(subrecord.data, subrecord.size),
                        condition, error)) {
                    conditions.push_back({{"function", condition.function},
                        {"comparison", static_cast<std::uint8_t>(condition.comparison)},
                        {"value", condition.comparisonValue},
                        {"parameter_1", condition.parameter1},
                        {"parameter_2", condition.parameter2},
                        {"string_parameter_1", std::string()},
                        {"string_parameter_2", std::string()},
                        {"run_on", condition.runOn}, {"reference", condition.reference},
                        {"or_with_next", condition.orWithNext}});
                    currentConditionIndex = conditions.size() - 1u;
                } else {
                    conditions.push_back({{"error", error}});
                    currentConditionIndex.reset();
                }
            } else if ((subrecord.type == "CIS1" || subrecord.type == "CIS2") &&
                       currentConditionIndex.has_value()) {
                std::string value(
                    reinterpret_cast<const char*>(subrecord.data), subrecord.size);
                while (!value.empty() && value.back() == '\0') value.pop_back();
                conditions[*currentConditionIndex][subrecord.type == "CIS1"
                        ? "string_parameter_1" : "string_parameter_2"] = std::move(value);
            } else if (subrecord.type == "VMAD") {
                vmadBytes += subrecord.size;
                odai::bethesda::VmadInfoAttachments attachments;
                if (odai::bethesda::readVmadInfoAttachments(
                        std::span<const std::uint8_t>(subrecord.data, subrecord.size),
                        attachments, vmadError)) {
                    vmadFlags = attachments.flags;
                    for (const odai::bethesda::VmadInfoFragment& fragment :
                         attachments.fragments) {
                        vmadFragments.push_back({{"script_class", fragment.scriptClass},
                            {"function", fragment.function}});
                    }
                }
            }
        }
        infos.push_back({{"form_id", record.formId}, {"topic_form_id", currentTopic},
            {"flags", flags}, {"previous_info", previous},
            {"rnam_string_id", promptStringId},
            {"prompt", promptStringsLoaded && strings.find(promptStringId) != nullptr
                ? *strings.find(promptStringId) : std::string()},
            {"responses", std::move(responses)}, {"conditions", std::move(conditions)},
            {"linked_topics", std::move(links)}, {"vmad_bytes", vmadBytes},
            {"vmad_flags", vmadFlags}, {"vmad_fragments", std::move(vmadFragments)},
            {"vmad_error", vmadError},
            {"subrecords", std::move(subrecords)}});
    };
    if (!reader.walk(dialogueVisitor)) {
        std::cout << nlohmann::json({{"ok", false}, {"plugin", pluginName},
            {"quest", questEditorId}, {"compatibility_errors", {reader.lastError()}}}).dump(2)
                  << '\n';
        return 1;
    }
    std::cout << nlohmann::json({{"ok", true}, {"plugin", pluginName},
        {"quest", questEditorId}, {"quest_form_id", questFormId},
        {"localized_prompts", promptStringsLoaded},
        {"localized_responses", stringsLoaded}, {"string_error", stringError},
        {"prompt_string_error", promptStringError},
        {"branches", std::move(branches)}, {"topics", std::move(topics)},
        {"infos", std::move(infos)}}).dump(2) << '\n';
    return 0;
}

int scenarioCheck(const std::filesystem::path& dataPath, const std::string& scenarioId) {
    const odai::bethesda::ScenarioDefinition* scenario =
        odai::bethesda::findScenario(scenarioId);
    if (scenario == nullptr) {
        std::cout << nlohmann::json({{"ok", false}, {"scenario", scenarioId},
            {"compatibility_errors", {"unknown scenario"}}}).dump(2) << '\n';
        return 1;
    }
    odai::importer::fnv::FalloutLoadOrder loadOrder;
    std::string error;
    if (!loadOrder.open(dataPath, {scenario->basePlugin, "Update.esm"}, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"scenario", scenarioId},
            {"compatibility_errors", {error}}}).dump(2) << '\n';
        return 1;
    }
    odai::importer::fnv::FalloutAssetSource assets;
    if (!assets.open(dataPath, odai::importer::fnv::kBsaContentMisc)) {
        std::cout << nlohmann::json({{"ok", false}, {"scenario", scenarioId},
            {"compatibility_errors", {"could not index Skyrim script archives"}}}).dump(2) << '\n';
        return 1;
    }
    odai::bethesda::BethesdaSession session;
    if (!session.configure({scenario->game, loadOrder.fingerprint(), scenario->id, 1u}, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"scenario", scenarioId},
            {"compatibility_errors", {error}}}).dump(2) << '\n';
        return 1;
    }
    odai::bethesda::SkyrimScenarioContentReport report;
    if (!odai::bethesda::loadSkyrimScenarioContent(
            *scenario, loadOrder, assets, session, report, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"scenario", scenarioId},
            {"compatibility_errors", {error}}, {"diagnostics", report.diagnostics}}).dump(2) << '\n';
        return 1;
    }
    std::vector<std::string> bootstrapDiagnostics;
    for (int tick = 0; tick < 4; ++tick) {
        odai::bethesda::BethesdaSessionStep step = session.advance(1.0 / 60.0);
        bootstrapDiagnostics.insert(bootstrapDiagnostics.end(),
            std::make_move_iterator(step.diagnostics.begin()),
            std::make_move_iterator(step.diagnostics.end()));
    }
    const odai::bethesda::QuestRuntimeState* mq102Bootstrap = session.findQuest("MQ102");
    const odai::bethesda::QuestRuntimeState* mq103BeforeRoute = session.findQuest("MQ103");
    bool mq102ObjectiveDisplayed = false;
    if (mq102Bootstrap != nullptr) {
        const auto objective = std::find_if(
            mq102Bootstrap->objectives.begin(), mq102Bootstrap->objectives.end(),
            [](const auto& value) { return value.index == 10; });
        mq102ObjectiveDisplayed = objective != mq102Bootstrap->objectives.end() &&
            objective->displayed;
    }
    const nlohmann::json beforeTheStormBootstrapCheck = {
        {"ok", bootstrapDiagnostics.empty() && mq102Bootstrap != nullptr &&
            mq102Bootstrap->stage == 10 && mq102Bootstrap->running &&
            mq102ObjectiveDisplayed && mq103BeforeRoute != nullptr &&
            mq103BeforeRoute->stage == 0},
        {"mq102_stage", mq102Bootstrap == nullptr ? 0 : mq102Bootstrap->stage},
        {"mq102_running", mq102Bootstrap != nullptr && mq102Bootstrap->running},
        {"objective_10_displayed", mq102ObjectiveDisplayed},
        {"mq103_stage", mq103BeforeRoute == nullptr ? 0 : mq103BeforeRoute->stage},
        {"diagnostics", bootstrapDiagnostics}};
    nlohmann::json goldenClawAliasCheck = {
        {"ok", false}, {"materialized_items", 0u}, {"transferred_items", 0u},
        {"ms13_stage", 0}, {"diagnostics", nlohmann::json::array()}};
    if (const odai::bethesda::QuestRuntimeState* ms13 = session.findQuest("MS13")) {
        const auto arvelAlias = std::find_if(
            ms13->aliases.begin(), ms13->aliases.end(), [](const auto& alias) {
                return toLowerAscii(alias.name) == "arvel";
            });
        if (arvelAlias != ms13->aliases.end() &&
            arvelAlias->target.kind == odai::bethesda::ObjectIdKind::PersistentReference) {
            odai::bethesda::RuntimeObject player;
            player.id = odai::bethesda::ObjectId::persistent(
                odai::bethesda::makeRecordKey("Skyrim.esm", 0x14u));
            player.base = odai::bethesda::makeRecordKey("Skyrim.esm", 0x7u);
            player.kind = odai::bethesda::RuntimeObjectKind::Actor;
            player.actorValues.emplace();
            odai::bethesda::RuntimeObject arvel;
            arvel.id = session.world().allocateRuntimeId();
            arvel.base = arvelAlias->target.reference;
            arvel.kind = odai::bethesda::RuntimeObjectKind::Actor;
            arvel.actorValues.emplace();
            arvel.actorValues->health = 0.0f;
            arvel.actorValues->dead = true;
            std::string checkError;
            if (session.world().addInitialObject(player, checkError) &&
                session.world().addInitialObject(arvel, checkError)) {
                // Exercise MQ103's authored dynamic alias fill: alias 52 finds
                // the sole Boss XLRT inside the forced Bleak Falls location;
                // alias 18 then creates the Dragonstone in that actor. The
                // harness injects residency, but kill/loot/stage progression
                // use the shared physical/runtime paths with no stage or item grant.
                nlohmann::json dragonstoneBossCheck = {
                    {"ok", false}, {"diagnostics", nlohmann::json::array()}};
                // MQ103 stage 10 enables/disables persistent quest references and
                // resolves Irileth's unique-actor alias. A headless probe has no
                // streamed Whiterun/Riverwood cells, so make that exact authored
                // dependency set resident before running the fragment.
                const auto addQuestReference = [&](std::uint32_t formId) {
                    odai::bethesda::RuntimeObject object;
                    object.id = odai::bethesda::ObjectId::persistent(
                        odai::bethesda::makeRecordKey("Skyrim.esm", formId));
                    object.base = object.id.reference;
                    object.kind = odai::bethesda::RuntimeObjectKind::Activator;
                    return session.world().addInitialObject(std::move(object), checkError);
                };
                const bool farengarLabResident = addQuestReference(0x000d50feu);
                const bool sleepingGiantResident = addQuestReference(0x000bcf06u);
                const bool riverwoodReferenceResident = addQuestReference(0x000fdb32u);
                odai::bethesda::RuntimeObject irileth;
                irileth.id = odai::bethesda::ObjectId::persistent(
                    odai::bethesda::makeRecordKey("Skyrim.esm", 0x0001a67fu));
                irileth.base =
                    odai::bethesda::makeRecordKey("Skyrim.esm", 0x00013bb8u);
                irileth.kind = odai::bethesda::RuntimeObjectKind::Actor;
                irileth.actorValues.emplace();
                const bool irilethResident =
                    session.world().addInitialObject(irileth, checkError);
                if (irilethResident) {
                    (void)session.bindQuestInventoryForActor(
                        irileth.id, irileth.base, checkError);
                }
                session.setQuestStage("MQ103", 10);
                odai::bethesda::RuntimeObject boss;
                boss.id = odai::bethesda::ObjectId::persistent(
                    odai::bethesda::makeRecordKey("Skyrim.esm", 0x0009bcd6u));
                boss.base = odai::bethesda::makeRecordKey("Skyrim.esm", 0x000b7989u);
                boss.kind = odai::bethesda::RuntimeObjectKind::Actor;
                boss.location =
                    odai::bethesda::makeRecordKey("Skyrim.esm", 0x00018ee9u);
                boss.referenceTypes = {
                    odai::bethesda::makeRecordKey("Skyrim.esm", 0x000130f7u)};
                boss.actorValues.emplace();
                if (session.world().addInitialObject(boss, checkError)) {
                    const std::size_t bossItems = session.bindQuestInventoryForActor(
                        boss.id, boss.base, checkError);
                    std::vector<std::string> bossDiagnostics;
                    for (int tick = 0; tick < 3; ++tick) {
                        odai::bethesda::BethesdaSessionStep step =
                            session.advance(1.0 / 60.0);
                        bossDiagnostics.insert(bossDiagnostics.end(),
                            std::make_move_iterator(step.diagnostics.begin()),
                            std::make_move_iterator(step.diagnostics.end()));
                    }
                    odai::bethesda::PhysicsCharacterConfig controller;
                    controller.position = {0.0f, 0.0f, 0.0f};
                    const bool playerPhysics = session.registerActorController(
                        player.id, controller, checkError);
                    controller.position = {100.0f, 0.0f, 0.0f};
                    const bool bossPhysics = session.registerActorController(
                        boss.id, controller, checkError);
                    odai::bethesda::MeleeAttackResult attack;
                    if (playerPhysics && bossPhysics) {
                        (void)session.advance(1.0 / 60.0,
                            [&](std::uint64_t, double) {
                                attack = session.performMeleeAttack(
                                    player.id, {1.0f, 0.0f, 0.0f}, 1000.0f);
                            });
                    }
                    const odai::bethesda::LootTransferResult bossLoot =
                        session.lootObject(player.id, boss.id);
                    for (int tick = 0; tick < 10; ++tick) {
                        odai::bethesda::BethesdaSessionStep step =
                            session.advance(1.0 / 60.0);
                        bossDiagnostics.insert(bossDiagnostics.end(),
                            std::make_move_iterator(step.diagnostics.begin()),
                            std::make_move_iterator(step.diagnostics.end()));
                    }
                    const auto* mq103AfterLoot = session.findQuest("MQ103");
                    bool bossAliasBound = false;
                    if (mq103AfterLoot != nullptr) {
                        const auto bossAlias = std::find_if(
                            mq103AfterLoot->aliases.begin(), mq103AfterLoot->aliases.end(),
                            [](const auto& alias) {
                                return toLowerAscii(alias.name) == "bleakfallsboss";
                            });
                        bossAliasBound = bossAlias != mq103AfterLoot->aliases.end() &&
                            bossAlias->target == boss.id;
                    }
                    const auto dragonstone = std::find_if(
                        bossLoot.transferred.begin(), bossLoot.transferred.end(),
                        [](const auto& item) {
                            return item.item == odai::bethesda::makeRecordKey(
                                "Skyrim.esm", 0x000df202u);
                        });
                    dragonstoneBossCheck = {
                        {"ok", farengarLabResident && sleepingGiantResident &&
                            riverwoodReferenceResident && irilethResident &&
                            checkError.empty() && bossDiagnostics.empty() &&
                            bossItems >= 1u && bossAliasBound && attack.accepted &&
                            attack.hit && attack.killed && bossLoot.accepted &&
                            dragonstone != bossLoot.transferred.end() &&
                            mq103AfterLoot != nullptr && mq103AfterLoot->stage >= 180},
                        {"boss_alias_bound", bossAliasBound},
                        {"materialized_items", bossItems},
                        {"combat_kill", attack.killed},
                        {"dragonstone_looted", dragonstone != bossLoot.transferred.end()},
                        {"mq103_stage", mq103AfterLoot == nullptr ? 0 : mq103AfterLoot->stage},
                        {"diagnostics", bossDiagnostics}};
                } else {
                    dragonstoneBossCheck["diagnostics"].push_back(checkError);
                }
                goldenClawAliasCheck["dragonstone_boss_alias_check"] =
                    std::move(dragonstoneBossCheck);
                // Start MS13 through Lucan's actual stage-0 player topic. The
                // old probe jumped straight to the dungeon and therefore never
                // completed stage 10; at hand-in time that made the correct
                // "I have the golden claw" INFO fail and accidentally selected
                // the unrelated volunteer prompt instead.
                nlohmann::json questStartCheck = {
                    {"ok", false}, {"diagnostics", nlohmann::json::array()}};
                odai::bethesda::ObjectId lucanRuntime;
                odai::bethesda::ObjectId camillaRuntime;
                const auto addResidentAliasActor = [&](std::string_view wanted,
                                                       odai::bethesda::ObjectId& outId) {
                    const auto alias = std::find_if(
                        ms13->aliases.begin(), ms13->aliases.end(), [&](const auto& candidate) {
                            return toLowerAscii(candidate.name) == wanted;
                        });
                    if (alias == ms13->aliases.end() ||
                        alias->target.kind !=
                            odai::bethesda::ObjectIdKind::PersistentReference) return false;
                    odai::bethesda::RuntimeObject actor;
                    actor.id = session.world().allocateRuntimeId();
                    actor.base = alias->target.reference;
                    actor.kind = odai::bethesda::RuntimeObjectKind::Actor;
                    actor.actorValues.emplace();
                    if (!session.world().addInitialObject(actor, checkError)) return false;
                    (void)session.bindQuestInventoryForActor(actor.id, actor.base, checkError);
                    outId = actor.id;
                    return true;
                };
                if (addResidentAliasActor("lucan", lucanRuntime) &&
                    addResidentAliasActor("camilla", camillaRuntime)) {
                    const auto startChoices = session.availableDialogueChoices(
                        lucanRuntime, player.id, true);
                    const auto volunteer = std::find_if(
                        startChoices.begin(), startChoices.end(), [](const auto& choice) {
                            return toLowerAscii(choice.prompt) ==
                                "i could help you get the claw back.";
                        });
                    if (volunteer != startChoices.end()) {
                        odai::bethesda::SkyrimDialogueSelectionResult selection =
                            session.selectDialogueInfo(
                                volunteer->info, lucanRuntime, player.id, 2u);
                        std::vector<std::string> diagnostics = selection.diagnostics;
                        for (int tick = 0; tick < 8; ++tick) {
                            odai::bethesda::BethesdaSessionStep step =
                                session.advance(1.0 / 60.0);
                            diagnostics.insert(diagnostics.end(),
                                std::make_move_iterator(step.diagnostics.begin()),
                                std::make_move_iterator(step.diagnostics.end()));
                        }
                        const auto* started = session.findQuest("MS13");
                        questStartCheck = {{"ok", selection.accepted && diagnostics.empty() &&
                                                started != nullptr && started->stage == 10},
                            {"prompt", volunteer->prompt},
                            {"info", volunteer->info.toString()},
                            {"stage", started == nullptr ? 0 : started->stage},
                            {"diagnostics", diagnostics}};
                    } else {
                        questStartCheck["diagnostics"].push_back(
                            "retail Lucan volunteer topic was not strictly available");
                    }
                } else {
                    questStartCheck["diagnostics"].push_back(
                        "Lucan/Camilla retail aliases could not be made resident");
                }
                goldenClawAliasCheck["quest_start_dialogue_check"] =
                    std::move(questStartCheck);
                const std::size_t materialized = session.bindQuestInventoryForActor(
                    arvel.id, arvel.base, checkError);
                goldenClawAliasCheck["materialized_items"] = materialized;
                std::vector<std::string> checkDiagnostics;
                for (int tick = 0; tick < 2; ++tick) {
                    odai::bethesda::BethesdaSessionStep step =
                        session.advance(1.0 / 60.0);
                    checkDiagnostics.insert(checkDiagnostics.end(),
                        std::make_move_iterator(step.diagnostics.begin()),
                        std::make_move_iterator(step.diagnostics.end()));
                }
                const odai::bethesda::LootTransferResult looted =
                    session.lootObject(player.id, arvel.id);
                goldenClawAliasCheck["transferred_items"] = looted.transferred.size();
                for (int tick = 0; tick < 4; ++tick) {
                    odai::bethesda::BethesdaSessionStep step =
                        session.advance(1.0 / 60.0);
                    checkDiagnostics.insert(checkDiagnostics.end(),
                        std::make_move_iterator(step.diagnostics.begin()),
                        std::make_move_iterator(step.diagnostics.end()));
                }
                const odai::bethesda::QuestRuntimeState* advanced =
                    session.findQuest("MS13");
                const std::int32_t stage = advanced == nullptr ? 0 : advanced->stage;
                goldenClawAliasCheck["ms13_stage"] = stage;
                nlohmann::json objectives = nlohmann::json::array();
                if (advanced != nullptr) {
                    for (const odai::bethesda::QuestObjectiveState& objective :
                         advanced->objectives) {
                        objectives.push_back({{"index", objective.index},
                            {"text", objective.displayText},
                            {"displayed", objective.displayed},
                            {"completed", objective.completed},
                            {"failed", objective.failed}});
                    }
                }
                goldenClawAliasCheck["objectives"] = std::move(objectives);
                session.setQuestStage("MS13", 50);
                std::vector<std::string> stage50Diagnostics;
                for (int tick = 0; tick < 2; ++tick) {
                    odai::bethesda::BethesdaSessionStep step =
                        session.advance(1.0 / 60.0);
                    stage50Diagnostics.insert(stage50Diagnostics.end(),
                        std::make_move_iterator(step.diagnostics.begin()),
                        std::make_move_iterator(step.diagnostics.end()));
                }
                const odai::bethesda::QuestRuntimeState* stage50 =
                    session.findQuest("MS13");
                goldenClawAliasCheck["stage_50_fragment_check"] = {
                    {"ok", stage50Diagnostics.empty() && stage50 != nullptr &&
                        stage50->stage == 50},
                    {"stage", stage50 == nullptr ? 0 : stage50->stage},
                    {"diagnostics", stage50Diagnostics}};
                nlohmann::json dialogueHandIns = {
                    {"ok", false}, {"golden_claw", nlohmann::json::object()},
                    {"dragonstone", nlohmann::json::object()}};
                session.setQuestStage("MS13", 60);
                for (int tick = 0; tick < 3; ++tick) {
                    (void)session.advance(1.0 / 60.0);
                }
                const odai::bethesda::QuestRuntimeState* ms13HandIn =
                    session.findQuest("MS13");
                if (ms13HandIn != nullptr && lucanRuntime.valid() &&
                    session.world().find(lucanRuntime) != nullptr) {
                        // Stage 100 queries Camilla's death state and enables
                        // the authored shop-display claw. Both are ordinarily
                        // resident in Riverwood while Lucan can be spoken to.
                        const auto displayAlias = std::find_if(
                            ms13HandIn->aliases.begin(), ms13HandIn->aliases.end(),
                            [](const auto& alias) {
                                return toLowerAscii(alias.name) == "lucanclaw";
                            });
                        if (displayAlias != ms13HandIn->aliases.end() &&
                            displayAlias->target.valid()) {
                            odai::bethesda::RuntimeObject displayClaw;
                            displayClaw.id = displayAlias->target;
                            displayClaw.base = displayAlias->target.reference;
                            displayClaw.kind = odai::bethesda::RuntimeObjectKind::Activator;
                            displayClaw.enabled = false;
                            (void)session.world().addInitialObject(
                                std::move(displayClaw), checkError);
                        }
                        const auto choices = session.availableDialogueChoices(
                            lucanRuntime, player.id, true);
                        const auto handIn = std::find_if(
                            choices.begin(), choices.end(), [](const auto& choice) {
                                return toLowerAscii(choice.prompt) ==
                                    "i have the golden claw.";
                            });
                        if (handIn != choices.end()) {
                            odai::bethesda::SkyrimDialogueSelectionResult selection =
                                session.selectDialogueInfo(
                                    handIn->info, lucanRuntime, player.id, 2u);
                            std::vector<std::string> diagnostics = selection.diagnostics;
                            for (int tick = 0; tick < 8; ++tick) {
                                odai::bethesda::BethesdaSessionStep step =
                                    session.advance(1.0 / 60.0);
                                diagnostics.insert(diagnostics.end(),
                                    std::make_move_iterator(step.diagnostics.begin()),
                                    std::make_move_iterator(step.diagnostics.end()));
                            }
                            const auto* completed = session.findQuest("MS13");
                            dialogueHandIns["golden_claw"] = {
                                {"ok", selection.accepted && diagnostics.empty() &&
                                    completed != nullptr && completed->stage == 100},
                                {"prompt", handIn->prompt}, {"responses", selection.responses},
                                {"info", handIn->info.toString()},
                                {"stage", completed == nullptr ? 0 : completed->stage},
                                {"diagnostics", diagnostics}};
                        } else {
                            dialogueHandIns["golden_claw"] = {
                                {"ok", false}, {"diagnostics",
                                    {"no strictly available retail 'I have the golden claw.' "
                                     "hand-in topic"}}};
                        }
                }

                const odai::bethesda::QuestRuntimeState* mq103 =
                    session.findQuest("MQ103");
                if (mq103 != nullptr) {
                    const auto dragonstoneAlias = std::find_if(
                        mq103->aliases.begin(), mq103->aliases.end(), [](const auto& alias) {
                            return toLowerAscii(alias.name) == "dragonstone";
                        });
                    const auto farengarAlias = std::find_if(
                        mq103->aliases.begin(), mq103->aliases.end(), [](const auto& alias) {
                            return toLowerAscii(alias.name) == "farengar";
                        });
                    if (dragonstoneAlias != mq103->aliases.end() &&
                        dragonstoneAlias->createdObject.valid() &&
                        farengarAlias != mq103->aliases.end() &&
                        farengarAlias->target.kind ==
                            odai::bethesda::ObjectIdKind::PersistentReference) {
                        odai::bethesda::RuntimeObject farengar;
                        farengar.id = session.world().allocateRuntimeId();
                        farengar.base = farengarAlias->target.reference;
                        farengar.kind = odai::bethesda::RuntimeObjectKind::Actor;
                        farengar.actorValues.emplace();
                        if (session.world().addInitialObject(farengar, checkError)) {
                            (void)session.bindQuestInventoryForActor(
                                farengar.id, farengar.base, checkError);
                            // MQ103 stage 190 immediately starts MQ104. At a
                            // real Farengar hand-in, Irileth, the messenger,
                            // and the captain marker are resident in the
                            // Dragonsreach scene, so mirror that residency in
                            // this headless route.
                            if (const auto* mq104 = session.findQuest("MQ104")) {
                                for (const std::string_view wanted :
                                     {std::string_view("irileth"),
                                      std::string_view("messenger")}) {
                                    const auto alias = std::find_if(
                                        mq104->aliases.begin(), mq104->aliases.end(),
                                        [&](const auto& candidate) {
                                            return toLowerAscii(candidate.name) == wanted;
                                        });
                                    if (alias == mq104->aliases.end() ||
                                        alias->target.kind != odai::bethesda::
                                            ObjectIdKind::PersistentReference) continue;
                                    odai::bethesda::RuntimeObject actor;
                                    actor.id = session.world().allocateRuntimeId();
                                    actor.base = alias->target.reference;
                                    actor.kind = odai::bethesda::RuntimeObjectKind::Actor;
                                    actor.actorValues.emplace();
                                    if (session.world().addInitialObject(actor, checkError)) {
                                        (void)session.bindQuestInventoryForActor(
                                            actor.id, actor.base, checkError);
                                    }
                                }
                                const odai::bethesda::ObjectId mq104Object =
                                    odai::bethesda::ObjectId::persistent(mq104->record);
                                const odai::bethesda::PapyrusValue* marker =
                                    session.papyrus().findProperty(
                                        mq104Object, "QF_MQ104B_0002610C",
                                        "CaptainStartMarker");
                                if (marker != nullptr && marker->type ==
                                        odai::bethesda::PapyrusValueType::Object &&
                                    marker->object.kind == odai::bethesda::
                                        ObjectIdKind::PersistentReference &&
                                    session.world().find(marker->object) == nullptr) {
                                    odai::bethesda::RuntimeObject markerObject;
                                    markerObject.id = marker->object;
                                    markerObject.base = marker->object.reference;
                                    markerObject.kind =
                                        odai::bethesda::RuntimeObjectKind::Activator;
                                    (void)session.world().addInitialObject(
                                        std::move(markerObject), checkError);
                                }
                            }
                            for (int tick = 0; tick < 4; ++tick) {
                                (void)session.advance(1.0 / 60.0);
                            }
                            const auto choices = session.availableDialogueChoices(
                                farengar.id, player.id, true);
                            const auto handIn = std::find_if(
                                choices.begin(), choices.end(), [](const auto& choice) {
                                    const std::string prompt =
                                        toLowerAscii(choice.prompt);
                                    return prompt.find("stone tablet you wanted") !=
                                            std::string::npos ||
                                        prompt.find("give dragonstone") !=
                                            std::string::npos;
                                });
                            if (handIn != choices.end()) {
                                const auto itemCount = [&](const odai::bethesda::RuntimeObject* object,
                                                           const odai::bethesda::RecordKey& item) {
                                    if (object == nullptr) return 0;
                                    const auto found = std::find_if(object->inventory.begin(),
                                        object->inventory.end(), [&](const auto& entry) {
                                            return entry.item == item;
                                        });
                                    return found == object->inventory.end() ? 0 : found->count;
                                };
                                const odai::bethesda::ObjectId aliasIdentityBefore =
                                    dragonstoneAlias->target;
                                const odai::bethesda::RecordKey dragonstoneItem =
                                    dragonstoneAlias->createdObject;
                                const int itemCountBefore = itemCount(
                                    session.world().find(player.id),
                                    dragonstoneItem);
                                odai::bethesda::SkyrimDialogueSelectionResult selection =
                                    session.selectDialogueInfo(
                                        handIn->info, farengar.id, player.id, 1u);
                                std::vector<std::string> diagnostics = selection.diagnostics;
                                for (int tick = 0; tick < 8; ++tick) {
                                    odai::bethesda::BethesdaSessionStep step =
                                        session.advance(1.0 / 60.0);
                                    diagnostics.insert(diagnostics.end(),
                                        std::make_move_iterator(step.diagnostics.begin()),
                                        std::make_move_iterator(step.diagnostics.end()));
                                }
                                const auto* completed = session.findQuest("MQ103");
                                odai::bethesda::ObjectId aliasIdentityAfter;
                                if (completed != nullptr) {
                                    const auto completedAlias = std::find_if(
                                        completed->aliases.begin(),
                                        completed->aliases.end(), [](const auto& alias) {
                                            return toLowerAscii(alias.name) == "dragonstone";
                                        });
                                    if (completedAlias != completed->aliases.end()) {
                                        aliasIdentityAfter = completedAlias->target;
                                    }
                                }
                                const int itemCountAfter = itemCount(
                                    session.world().find(player.id),
                                    dragonstoneItem);
                                const bool aliasIdentityStable =
                                    aliasIdentityAfter == aliasIdentityBefore;
                                bool saveReloadOk = false;
                                std::string saveError;
                                const std::filesystem::path savePath =
                                    std::filesystem::temp_directory_path() /
                                    "odai-skyrim-bleak-falls-probe-v7.json";
                                if (odai::bethesda::saveOdaiGameAtomic(
                                        savePath, session, saveError)) {
                                    odai::bethesda::SaveLoadReport saveReport;
                                    saveReloadOk = odai::bethesda::loadOdaiGame(
                                        savePath, session, {}, saveReport, saveError);
                                }
                                const auto* reloadedQuest = session.findQuest("MQ103");
                                odai::bethesda::ObjectId reloadedAliasIdentity;
                                if (reloadedQuest != nullptr) {
                                    const auto reloadedAlias = std::find_if(
                                        reloadedQuest->aliases.begin(),
                                        reloadedQuest->aliases.end(), [](const auto& alias) {
                                            return toLowerAscii(alias.name) == "dragonstone";
                                        });
                                    if (reloadedAlias != reloadedQuest->aliases.end()) {
                                        reloadedAliasIdentity = reloadedAlias->target;
                                    }
                                }
                                saveReloadOk = saveReloadOk && reloadedQuest != nullptr &&
                                    reloadedQuest->stage == 190 &&
                                    reloadedAliasIdentity == aliasIdentityBefore &&
                                    itemCount(session.world().find(player.id),
                                        dragonstoneItem) == 0;
                                if (!saveError.empty()) diagnostics.push_back(saveError);
                                dialogueHandIns["dragonstone"] = {
                                    {"ok", selection.accepted && diagnostics.empty() &&
                                        completed != nullptr && completed->stage == 190 &&
                                        itemCountBefore == 1 && itemCountAfter == 0 &&
                                        aliasIdentityStable && saveReloadOk},
                                    {"prompt", handIn->prompt},
                                    {"responses", selection.responses},
                                    {"info", handIn->info.toString()},
                                    {"stage", completed == nullptr ? 0 : completed->stage},
                                    {"item_count_before", itemCountBefore},
                                    {"item_count_after", itemCountAfter},
                                    {"alias_identity_stable", aliasIdentityStable},
                                    {"save_reload_ok", saveReloadOk},
                                    {"diagnostics", diagnostics}};
                            } else {
                                nlohmann::json prompts = nlohmann::json::array();
                                for (const auto& choice : choices) {
                                    prompts.push_back(choice.prompt);
                                }
                                dialogueHandIns["dragonstone"] = {
                                    {"ok", false}, {"diagnostics",
                                        {"no strictly available Dragonstone hand-in topic"}},
                                    {"available_prompts", std::move(prompts)}};
                            }
                        }
                    }
                }
                dialogueHandIns["ok"] =
                    dialogueHandIns["golden_claw"].value("ok", false) &&
                    dialogueHandIns["dragonstone"].value("ok", false);
                goldenClawAliasCheck["dialogue_handin_check"] =
                    std::move(dialogueHandIns);
                goldenClawAliasCheck["diagnostics"] = checkDiagnostics;
                goldenClawAliasCheck["ok"] = checkError.empty() &&
                    checkDiagnostics.empty() && materialized >= 1u &&
                    looted.accepted && !looted.transferred.empty() && stage >= 30 &&
                    stage50Diagnostics.empty() && stage50 != nullptr &&
                    stage50->stage >= 50 &&
                    goldenClawAliasCheck["quest_start_dialogue_check"].value("ok", false) &&
                    goldenClawAliasCheck["dragonstone_boss_alias_check"].value("ok", false) &&
                    goldenClawAliasCheck["dialogue_handin_check"].value("ok", false);
            } else {
                goldenClawAliasCheck["diagnostics"].push_back(checkError);
            }
        } else {
            goldenClawAliasCheck["diagnostics"].push_back(
                "retail MS13 Arvel alias did not resolve to a stable actor base");
        }
    } else {
        goldenClawAliasCheck["diagnostics"].push_back("MS13 runtime state is missing");
    }
    nlohmann::json quests = nlohmann::json::array();
    const std::size_t unresolved = report.unresolvedCallBindings.size();
    for (const odai::bethesda::ScenarioQuestLoadDetail& detail : report.quests) {
        quests.push_back({{"editor_id", detail.editorId}, {"stages", detail.stages},
            {"objectives", detail.objectives}, {"aliases", detail.aliases},
            {"stage_fragments", detail.stageFragments},
            {"alias_script_attachments", detail.aliasScriptAttachments},
            {"referenced_records", detail.referencedRecords}, {"scripts", detail.scripts},
            {"unresolved_calls", detail.unresolvedCalls}});
    }
    const bool fixtureAssertionsOk = beforeTheStormBootstrapCheck.value("ok", false) &&
        goldenClawAliasCheck.value("ok", false) && unresolved == 0u;
    const nlohmann::json fixtureSetup = {
        {"scenario_bootstrap", {"MQ101:900 (post-Helgen prerequisite)",
            "MQ102:10 (authored Riverwood startup fragment replayed after VMAD attachment)"}},
        {"injected_actor_state", {
            "Lucan and Camilla made resident in Riverwood Trader",
            "Arvel created already dead",
            "Installed Bleak Falls Boss reference made resident for physical alias/combat assertion",
            "MQ103 stage-10 persistent references and Irileth made resident",
            "Lucan display claw made resident",
            "Farengar, Irileth, messenger, and captain marker made resident"}},
        {"direct_stage_setup", {"MQ103:10", "MS13:50", "MS13:60"}},
        {"direct_inventory_setup", nlohmann::json::array()}};
    const nlohmann::json unverifiedRouteSegments = {
        "MQ102 Riverwood-friend conversation, travel, and Balgruuf/Farengar introduction",
        "Bleak Falls streaming traversal and authored triggers",
        "Arvel web activation, escape package, combat, and death",
        "Golden Claw door animation/audio and Hall of Stories stage progression",
        "authored boss encounter/package and streamed residency before the verified combat/loot path",
        "pre-acquired Dragonstone alternate Farengar dialogue branch",
        "save/reload during retail dialogue and dungeon traversal"};
    std::cout << nlohmann::json({{"ok", fixtureAssertionsOk}, {"scenario", scenarioId},
        {"content_loaded", true},
        {"gate_kind", "fixture-assisted retail content/fragment assertions"},
        {"fixture_assisted", true},
        {"fixture_assertions_ok", fixtureAssertionsOk},
        {"automated_playable_route_complete", false},
        {"release_gate_passed", false},
        {"setup", fixtureSetup},
        {"before_the_storm_bootstrap_check", beforeTheStormBootstrapCheck},
        {"unverified_route_segments", unverifiedRouteSegments},
        {"strict_ready", unresolved == 0u && report.runtimeBlockers.empty()},
        {"unresolved_calls", unresolved}, {"quests", std::move(quests)},
        {"transitive_script_classes", report.transitiveScriptClasses},
        {"transitive_script_instances", report.transitiveScriptInstances},
        {"locations_registered", report.locationsRegistered},
        {"global_variables_registered", report.globalVariablesRegistered},
        {"dialogue_topics_registered", report.dialogueTopicsRegistered},
        {"dialogue_branches_registered", report.dialogueBranchesRegistered},
        {"dialogue_infos_registered", report.dialogueInfosRegistered},
        {"dialogue_fragments_loaded", report.dialogueFragmentsLoaded},
        {"golden_claw_alias_event_check", std::move(goldenClawAliasCheck)},
        {"runtime_blockers", report.runtimeBlockers},
        {"unresolved_call_bindings", report.unresolvedCallBindings},
        {"diagnostics", report.diagnostics}}).dump(2) << '\n';
    return fixtureAssertionsOk ? 0 : 1;
}

void printUsage() {
    std::cout << "Usage:\n"
              << "  odai_bethesda_probe --profilecheck <profile> [--data <Data>] [--mods-root <dir>]\n"
              << "  odai_bethesda_probe --why <profile> <virtualPath|formID> [--data <Data>]\n"
              << "  odai_bethesda_probe --conflicts <profile> [--data <Data>]\n"
              << "  odai_bethesda_probe --export-profile <profile> <out.json> [--data <Data>]\n"
              << "  odai_bethesda_probe --tes3-scriptcheck <profile> [--strict] [--data <Data>]\n"
              << "  odai_bethesda_probe --tes3-script-source <profile> <script-id> [--data <Data>]\n"
              << "  odai_bethesda_probe --tes3-dialogue-trace <profile> <actor-or-topic> [--data <Data>]\n"
              << "  odai_bethesda_probe --tes3-quest-trace <profile> <journal-id> [--data <Data>]\n"
              << "  odai_bethesda_probe --tes3-quest-suite <profile> [--quest <journal-id>] [--data <Data>]\n"
              << "  odai_bethesda_probe --tes3-virtual-player <profile> [--strict] [--report <json>] [--data <Data>]\n"
              << "  odai_bethesda_probe <DataFilesPath> --archives\n"
              << "  odai_bethesda_probe <DataFilesPath> --nifs [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --nif <virtualPath>\n"
              << "  odai_bethesda_probe <DataFilesPath> --nifblocks <virtualPath>\n"
              << "  odai_bethesda_probe <DataFilesPath> --lodheight <lodBlock.nif> <worldX> <worldY>\n"
              << "  odai_bethesda_probe <DataFilesPath> --kf <virtualPath.kf>\n"
              << "  odai_bethesda_probe <DataFilesPath> --kfsweep <folderSubstring>\n"
              << "  odai_bethesda_probe <DataFilesPath> --actor <Plugin.esm> <ActorEditorID>\n"
              << "  odai_bethesda_probe <DataFilesPath> --footers [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --dialogue <Plugin.esm> <speakerEdid> [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --dialoguetree <Plugin.esm> <speakerEdid> [steps]\n"
              << "  odai_bethesda_probe <DataFilesPath> --regions <Plugin.esm> [topN]\n"
              << "  odai_bethesda_probe <DataFilesPath> --texture <texturePath>\n"
              << "  odai_bethesda_probe <DataFilesPath> --skeleton <virtualPath>\n"
              << "  odai_bethesda_probe <DataFilesPath> --skinned <virtualPath>\n"
              << "  odai_bethesda_probe <DataFilesPath> --character <skeleton.nif> <part.nif>...\n"
              << "  odai_bethesda_probe <DataFilesPath> --scriptcheck <script.pex> [--strict]\n"
              << "  odai_bethesda_probe <DataFilesPath> --animationcheck\n"
              << "  odai_bethesda_probe <DataFilesPath> --animation-strict\n"
              << "  odai_bethesda_probe <DataFilesPath> --quest-trace <Plugin.esm> <QuestEditorID>\n"
              << "  odai_bethesda_probe <DataFilesPath> --skyrim-dialogue-trace <Plugin.esm> <QuestEditorID>\n"
              << "  odai_bethesda_probe <DataFilesPath> --scenario-check <scenario>\n"
              // Modes that existed but were never listed here.
              << "  odai_bethesda_probe <DataFilesPath> --find <substring> [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --cellindex <Plugin.esm> [worldspace] [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --buildcell <Plugin.esm> <Worldspace> <x> <z>\n"
              << "  odai_bethesda_probe <DataFilesPath> --floaters <Plugin.esm> <Worldspace> <x> <z>\n"
              << "  odai_bethesda_probe <DataFilesPath> --plugin <Plugin.esm> [typeCount]\n"
              << "  odai_bethesda_probe <DataFilesPath> --loadorder [plugins.txt]\n"
              << "  odai_bethesda_probe <DataFilesPath> --doorcheck [plugins.txt]\n"
              << "  odai_bethesda_probe <DataFilesPath> --routecheck [plugins.txt]\n"
              << "  odai_bethesda_probe <DataFilesPath> --tourcheck <Worldspace> <tour.txt> [plugins.txt]\n"
              << "  odai_bethesda_probe <DataFilesPath> --record <Plugin.esm> <TYPE> [dumpCount]\n"
              << "  odai_bethesda_probe <DataFilesPath> --refs <Plugin.esm> <BASETYPE> [topN]\n"
              << "  odai_bethesda_probe <DataFilesPath> --placements <Plugin.esm> <baseFormID> [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --modelrefs <Plugin.esm> <modelSubstring> [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --cells <Plugin.esm> [filter]\n"
              << "  odai_bethesda_probe <DataFilesPath> --navm <Plugin.esm> [dumpCount]\n"
              << "  odai_bethesda_probe <DataFilesPath> --rotations <Plugin.esm> <CellEditorID>\n"
              << "  odai_bethesda_probe <anyDir> --scene <cooked.bin>\n";
}

int animationCheck(const std::filesystem::path& dataPath, bool strict) {
    using namespace odai::importer::fnv;
    FalloutAssetSource assets;
    if (!assets.open(dataPath)) {
        std::cout << nlohmann::json({{"ok", false},
            {"error", "could not open virtual Data asset source"}}).dump(2) << '\n';
        return 1;
    }
    // Preserve virtual-Data case insensitivity for directly extracted mods.
    (void)assets.addModDirectory(dataPath);
    if (const char* mods = std::getenv("ODAI_FNV_MODS")) {
        std::string root;
        for (const char* cursor = mods;; ++cursor) {
            if (*cursor == ':' || *cursor == '\0') {
                if (!root.empty()) {
                    (void)assets.addModDirectory(root);
                    root.clear();
                }
                if (*cursor == '\0') break;
            } else {
                root.push_back(*cursor);
            }
        }
    }
    SkyrimAnimationAssetReport report;
    std::string error;
    const bool inspected = inspectSkyrimAnimationBundle(assets, report, strict, error);
    nlohmann::json roots = nlohmann::json::array();
    for (const FalloutAssetSource::ResolvedAsset& asset : report.roots) {
        roots.push_back({{"path", asset.canonicalVirtualPath},
            {"provider_id", asset.providerId}, {"provider", asset.providerName},
            {"fingerprint", asset.contentFingerprint}, {"archive", asset.archiveName}});
    }
    odai::bethesda::BethesdaPhysicsWorld physics;
    std::string joltError;
    bool joltCharacter = physics.initialize(joltError);
    if (joltCharacter) {
        odai::bethesda::PhysicsCharacterConfig config;
        joltCharacter = physics.addCharacter(
            odai::bethesda::ObjectId::runtime(1u), config, joltError);
    }
    std::cout << nlohmann::json({{"ok", inspected && (!strict || report.strictCompatible)},
        {"strict", strict}, {"coherent", report.coherent},
        {"strict_ready", report.strictCompatible},
        {"generator", odai::anim::hkxGeneratorName(report.generator)},
        {"generator_provider", report.generatorProvider}, {"roots", std::move(roots)},
        {"missing_assets", report.missingAssets},
        {"unsupported_classes", report.unsupportedClasses},
        {"jolt_character_constructed", joltCharacter}, {"jolt_error", joltError},
        {"fallback", report.strictCompatible ? "none" : "per-actor procedural"},
        {"diagnostics", report.diagnostics}, {"error", error}}).dump(2) << '\n';
    return inspected && (!strict || report.strictCompatible) && joltCharacter ? 0 : 1;
}

}  // namespace

int checkSkyrimTraversalRoute(
    const std::filesystem::path& dataPath,
    const std::optional<std::filesystem::path>& explicitList) {
    using namespace odai::importer;
    using namespace odai::importer::fnv;

    std::vector<std::string> plugins;
    std::filesystem::path source;
    std::string error;
    if (!resolveInstalledSkyrimPluginList(
            dataPath, explicitList, plugins, source, error)) {
        std::cout << "resolve failed: " << error << "\n";
        return 1;
    }
    FalloutLoadOrder order;
    if (!order.open(dataPath, plugins, error)) {
        std::cout << "open failed: " << error << "\n";
        return 1;
    }
    FalloutCellIndex index;
    if (!buildFalloutCellIndex(order, index, error)) {
        std::cout << "index failed: " << error << "\n";
        return 1;
    }
    FalloutWorldTables tables;
    if (!buildFalloutWorldTables(order, tables, error)) {
        std::cout << "world tables failed: " << error << "\n";
        return 1;
    }

    const auto worldspaceId = [&](const std::string& editorId) -> std::uint32_t {
        const auto found = tables.worldspaceFormIdsByEditorId.find(toLowerAscii(editorId));
        return found == tables.worldspaceFormIdsByEditorId.end() ? 0u : found->second;
    };
    const std::uint32_t tamriel = worldspaceId("Tamriel");
    const std::uint32_t whiterun = worldspaceId("WhiterunWorld");
    if (tamriel == 0u || whiterun == 0u) {
        std::cout << "route failed: Tamriel or WhiterunWorld is absent\n";
        return 1;
    }

    const auto interiorIndex = std::find_if(
        index.cells.begin(), index.cells.end(), [](const FalloutCellIndexEntry& entry) {
            return entry.isInterior &&
                   toLowerAscii(entry.editorId) == "whiterunbanneredmare";
        });
    if (interiorIndex == index.cells.end()) {
        std::cout << "route failed: WhiterunBanneredMare is absent\n";
        return 1;
    }
    const std::size_t banneredMare =
        static_cast<std::size_t>(std::distance(index.cells.begin(), interiorIndex));

    struct RouteDoor {
        std::size_t sourceCell = 0u;
        std::size_t targetCell = 0u;
        FalloutPlacedReference reference;
    };
    const auto findDoor = [&](const auto& sourceMatches, const auto& targetMatches,
                              RouteDoor& out) {
        for (std::size_t sourceIndex = 0; sourceIndex < index.cells.size(); ++sourceIndex) {
            if (!sourceMatches(index.cells[sourceIndex])) {
                continue;
            }
            FalloutCellRecord cell;
            std::string cellError;
            if (!extractFalloutCellMerged(
                    index, order, index.cells[sourceIndex], cell, cellError)) {
                continue;
            }
            for (const FalloutPlacedReference& reference : cell.references) {
                if (!reference.hasTeleport || reference.isDeleted) {
                    continue;
                }
                const auto target = index.cellIndexByReferenceFormId.find(
                    reference.teleportTargetRefFormId);
                if (target == index.cellIndexByReferenceFormId.end() ||
                    !targetMatches(index.cells[target->second])) {
                    continue;
                }
                bool finite = true;
                for (float component : reference.teleportPosition) {
                    finite = finite && std::isfinite(component);
                }
                for (float component : reference.teleportRotationRadians) {
                    finite = finite && std::isfinite(component);
                }
                if (!finite) {
                    continue;
                }
                out = RouteDoor{sourceIndex, target->second, reference};
                return true;
            }
        }
        return false;
    };

    RouteDoor enterCity;
    RouteDoor enterInn;
    RouteDoor leaveInn;
    RouteDoor leaveCity;
    const auto exteriorIn = [](std::uint32_t wanted) {
        return [wanted](const FalloutCellIndexEntry& entry) {
            return !entry.isInterior && entry.worldspaceFormId == wanted;
        };
    };
    const auto exactCell = [](std::size_t wanted, const FalloutCellIndexEntry* base) {
        return [wanted, base](const FalloutCellIndexEntry& entry) {
            return static_cast<std::size_t>(&entry - base) == wanted;
        };
    };
    if (!findDoor(exteriorIn(tamriel), exteriorIn(whiterun), enterCity) ||
        !findDoor(exteriorIn(whiterun), exactCell(banneredMare, index.cells.data()), enterInn) ||
        !findDoor(exactCell(banneredMare, index.cells.data()), exteriorIn(whiterun), leaveInn) ||
        !findDoor(exteriorIn(whiterun), exteriorIn(tamriel), leaveCity)) {
        std::cout << "route failed: could not resolve Tamriel -> WhiterunWorld -> "
                     "WhiterunBanneredMare -> WhiterunWorld -> Tamriel\n";
        return 1;
    }

    FalloutAssetSource assets;
    if (!assets.open(dataPath)) {
        std::cout << "route failed: game archives could not be opened\n";
        return 1;
    }
    std::set<std::size_t> cellsToBuild{
        enterCity.sourceCell, enterCity.targetCell, banneredMare};
    std::size_t builtCells = 0u;
    std::size_t totalCollisionTriangles = 0u;
    for (const std::size_t cellIndex : cellsToBuild) {
        FalloutCellRecord cell;
        if (!extractFalloutCellMerged(index, order, index.cells[cellIndex], cell, error)) {
            std::cout << "route failed: cell extraction: " << error << "\n";
            return 1;
        }
        CellSceneBuilder builder(assets, tables);
        builder.setMaxTextureSize(64u);
        builder.addCell(cell);
        ImportedScene scene;
        builder.finish(scene);
        if (scene.packedVertices.empty() || scene.packedIndices.empty() ||
            scene.collisionTriangles.empty()) {
            std::cout << "route failed: "
                      << (index.cells[cellIndex].editorId.empty()
                              ? ("cell 0x" + toHex(index.cells[cellIndex].cellFormId))
                              : index.cells[cellIndex].editorId)
                      << " has incomplete render/collision geometry\n";
            return 1;
        }
        ++builtCells;
        totalCollisionTriangles += scene.collisionTriangles.size();
        std::cout << "built "
                  << (index.cells[cellIndex].editorId.empty()
                          ? ("cell 0x" + toHex(index.cells[cellIndex].cellFormId))
                          : index.cells[cellIndex].editorId)
                  << ": vertices=" << scene.packedVertices.size()
                  << " collision=" << scene.collisionTriangles.size() << "\n";
    }

    const auto printStep = [&](const char* label, const RouteDoor& door) {
        std::cout << label << " 0x" << std::hex << door.reference.formId << " -> 0x"
                  << door.reference.teleportTargetRefFormId << std::dec << " arrival=("
                  << door.reference.teleportPosition[0] << ","
                  << door.reference.teleportPosition[1] << ","
                  << door.reference.teleportPosition[2] << ")\n";
    };
    printStep("enter-city", enterCity);
    printStep("enter-inn", enterInn);
    printStep("leave-inn", leaveInn);
    printStep("leave-city", leaveCity);
    std::cout << "route=ok cellsBuilt=" << builtCells
              << " collisionTriangles=" << totalCollisionTriangles
              << " fingerprint=" << order.fingerprint() << "\n";
    return 0;
}

namespace {

struct TourAuditPoint {
    float position[3] = {};
};

std::string jsonString(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20u) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(c) << std::dec;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
    return out.str();
}

bool readTourPoints(const std::filesystem::path& path,
                    std::vector<TourAuditPoint>& out,
                    std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open tour file";
        return false;
    }
    std::string line;
    std::size_t lineNumber = 0u;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (const std::size_t comment = line.find('#'); comment != std::string::npos) {
            line.resize(comment);
        }
        std::istringstream row(line);
        float values[6]{};
        if (!(row >> values[0])) {
            continue;
        }
        bool valid = true;
        for (int component = 1; component < 6; ++component) {
            valid = valid && static_cast<bool>(row >> values[component]);
        }
        std::string trailing;
        if (!valid || (row >> trailing)) {
            error = "line " + std::to_string(lineNumber) +
                    " must contain exactly six numbers";
            return false;
        }
        TourAuditPoint point;
        std::copy_n(values, 3u, point.position);
        if (!std::all_of(std::begin(point.position), std::end(point.position),
                         [](float value) { return std::isfinite(value); })) {
            error = "line " + std::to_string(lineNumber) + " contains a non-finite position";
            return false;
        }
        out.push_back(point);
    }
    if (out.size() < 4u) {
        error = "tour needs at least four waypoints";
        return false;
    }
    return true;
}

void tourKnots(const float p0[3], const float p1[3], const float p2[3],
               const float p3[3], float out[4]) {
    const auto span = [](const float a[3], const float b[3]) {
        const float dx = b[0] - a[0];
        const float dy = b[1] - a[1];
        const float dz = b[2] - a[2];
        return std::max(std::sqrt(std::sqrt((dx * dx) + (dy * dy) + (dz * dz))), 1e-4f);
    };
    out[0] = 0.0f;
    out[1] = span(p0, p1);
    out[2] = out[1] + span(p1, p2);
    out[3] = out[2] + span(p2, p3);
}

void tourLerp(const float a[3], const float b[3], float ta, float tb, float t, float out[3]) {
    const float denominator = tb - ta;
    const float weight = std::abs(denominator) < 1e-6f ? 0.0f : (t - ta) / denominator;
    for (int axis = 0; axis < 3; ++axis) {
        out[axis] = a[axis] + ((b[axis] - a[axis]) * weight);
    }
}

TourAuditPoint sampleTourSegment(const TourAuditPoint& p0, const TourAuditPoint& p1,
                                 const TourAuditPoint& p2, const TourAuditPoint& p3,
                                 float parameter) {
    float knots[4];
    tourKnots(p0.position, p1.position, p2.position, p3.position, knots);
    const float t = knots[1] + ((knots[2] - knots[1]) * parameter);
    float a1[3], a2[3], a3[3], b1[3], b2[3];
    tourLerp(p0.position, p1.position, knots[0], knots[1], t, a1);
    tourLerp(p1.position, p2.position, knots[1], knots[2], t, a2);
    tourLerp(p2.position, p3.position, knots[2], knots[3], t, a3);
    tourLerp(a1, a2, knots[0], knots[2], t, b1);
    tourLerp(a2, a3, knots[1], knots[3], t, b2);
    TourAuditPoint out;
    tourLerp(b1, b2, knots[1], knots[2], t, out.position);
    return out;
}

std::vector<TourAuditPoint> sampleTour60Hz(const std::vector<TourAuditPoint>& points) {
    std::vector<TourAuditPoint> samples;
    for (std::size_t segment = 0; segment + 1u < points.size(); ++segment) {
        const TourAuditPoint& p0 = points[segment == 0u ? 0u : segment - 1u];
        const TourAuditPoint& p1 = points[segment];
        const TourAuditPoint& p2 = points[segment + 1u];
        const TourAuditPoint& p3 = points[std::min(segment + 2u, points.size() - 1u)];
        for (int frame = 0; frame < 60; ++frame) {
            samples.push_back(sampleTourSegment(p0, p1, p2, p3,
                                                static_cast<float>(frame) / 60.0f));
        }
    }
    samples.push_back(points.back());
    return samples;
}

float pointTriangleDistanceSquared(const float point[3], const float vertices[9]) {
    const auto dot = [](const float a[3], const float b[3]) {
        return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
    };
    const auto subtract = [](const float a[3], const float b[3], float out[3]) {
        for (int axis = 0; axis < 3; ++axis) out[axis] = a[axis] - b[axis];
    };
    const float* a = vertices;
    const float* b = vertices + 3;
    const float* c = vertices + 6;
    float ab[3], ac[3], ap[3];
    subtract(b, a, ab); subtract(c, a, ac); subtract(point, a, ap);
    const float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return dot(ap, ap);
    float bp[3]; subtract(point, b, bp);
    const float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return dot(bp, bp);
    const float vc = (d1 * d4) - (d3 * d2);
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        float nearest[3] = {a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]};
        float delta[3]; subtract(point, nearest, delta); return dot(delta, delta);
    }
    float cp[3]; subtract(point, c, cp);
    const float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return dot(cp, cp);
    const float vb = (d5 * d2) - (d1 * d6);
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        float nearest[3] = {a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]};
        float delta[3]; subtract(point, nearest, delta); return dot(delta, delta);
    }
    const float va = (d3 * d6) - (d5 * d4);
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        float bc[3]; subtract(c, b, bc);
        float nearest[3] = {b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]};
        float delta[3]; subtract(point, nearest, delta); return dot(delta, delta);
    }
    const float denominator = 1.0f / (va + vb + vc);
    const float v = vb * denominator, w = vc * denominator;
    float nearest[3] = {a[0] + ab[0] * v + ac[0] * w,
                        a[1] + ab[1] * v + ac[1] * w,
                        a[2] + ab[2] * v + ac[2] * w};
    float delta[3]; subtract(point, nearest, delta); return dot(delta, delta);
}

float sampleCellTerrain(const odai::importer::fnv::FalloutCellRecord& cell,
                        float engineX, float engineZ) {
    if (cell.land == nullptr || !cell.land->hasHeights || cell.land->gridSize < 2) {
        return -std::numeric_limits<float>::infinity();
    }
    const float worldSize = cell.land->cellWorldSize();
    const float localX = engineX - (static_cast<float>(cell.gridX) * worldSize);
    const float bethesdaY = -engineZ;
    const float localY = bethesdaY - (static_cast<float>(cell.gridZ) * worldSize);
    const float column = std::clamp(localX / odai::importer::fnv::kLandPostSpacing,
                                    0.0f, static_cast<float>(cell.land->gridSize - 1));
    const float row = std::clamp(localY / odai::importer::fnv::kLandPostSpacing,
                                 0.0f, static_cast<float>(cell.land->gridSize - 1));
    const int c0 = static_cast<int>(std::floor(column));
    const int r0 = static_cast<int>(std::floor(row));
    const int c1 = std::min(c0 + 1, cell.land->gridSize - 1);
    const int r1 = std::min(r0 + 1, cell.land->gridSize - 1);
    const float tx = column - static_cast<float>(c0);
    const float ty = row - static_cast<float>(r0);
    const auto height = [&](int r, int c) {
        return cell.land->heights[static_cast<std::size_t>((r * cell.land->gridSize) + c)];
    };
    const float h0 = height(r0, c0) + ((height(r0, c1) - height(r0, c0)) * tx);
    const float h1 = height(r1, c0) + ((height(r1, c1) - height(r1, c0)) * tx);
    return h0 + ((h1 - h0) * ty);
}

bool isIntentionalAuditModel(std::string_view modelPath) {
    const std::string lower = toLowerAscii(std::string(modelPath));
    const std::size_t slash = lower.find_last_of("\\/");
    const std::string_view name = slash == std::string::npos
        ? std::string_view(lower)
        : std::string_view(lower).substr(slash + 1u);
    return lower.rfind("markers\\", 0u) == 0u ||
           lower.find("\\markers\\") != std::string::npos ||
           name.rfind("marker", 0u) == 0u ||
           name.find("invisible") != std::string_view::npos ||
           lower.find("\\dummyitems\\") != std::string::npos ||
           lower.rfind("critters\\crittermarker", 0u) == 0u ||
           (lower.rfind("furniture\\", 0u) == 0u &&
            name.find("marker") != std::string_view::npos) ||
           odai::importer::fnv::isSkyOnlyModelPath(modelPath) ||
           odai::importer::fnv::isEffectOnlyModelPath(modelPath);
}

}  // namespace

int checkSkyrimTour(const std::filesystem::path& dataPath, std::string worldspace,
                    const std::filesystem::path& tourPath,
                    const std::optional<std::filesystem::path>& explicitList) {
    using namespace odai::importer;
    using namespace odai::importer::fnv;
    const auto failJson = [&](std::string_view message) {
        std::cout << "{\"ok\":false,\"error\":" << jsonString(message) << "}\n";
        return 1;
    };
    std::vector<TourAuditPoint> waypoints;
    std::string error;
    if (!readTourPoints(tourPath, waypoints, error)) return failJson(error);
    const std::vector<TourAuditPoint> samples = sampleTour60Hz(waypoints);

    std::vector<std::string> plugins;
    std::filesystem::path source;
    if (!resolveInstalledSkyrimPluginList(dataPath, explicitList, plugins, source, error)) {
        return failJson(error);
    }
    FalloutLoadOrder order;
    FalloutCellIndex index;
    FalloutWorldTables tables;
    if (!order.open(dataPath, plugins, error) || !buildFalloutCellIndex(order, index, error) ||
        !buildFalloutWorldTables(order, tables, error)) {
        return failJson(error);
    }
    const auto world = tables.worldspaceFormIdsByEditorId.find(toLowerAscii(worldspace));
    if (world == tables.worldspaceFormIdsByEditorId.end()) {
        return failJson("worldspace is absent from the resolved load order: " + worldspace);
    }
    const std::uint32_t worldspaceFormId = world->second;
    std::map<std::pair<int, int>, const FalloutCellIndexEntry*> cellsByGrid;
    for (const FalloutCellIndexEntry& entry : index.cells) {
        if (!entry.isInterior && entry.hasGridCoords &&
            entry.worldspaceFormId == worldspaceFormId) {
            cellsByGrid[{entry.gridX, entry.gridZ}] = &entry;
        }
    }

    std::set<std::pair<int, int>> requiredCells;
    for (const TourAuditPoint& sample : samples) {
        const int gridX = static_cast<int>(std::floor(sample.position[0] / kExteriorCellSize));
        const int gridZ = static_cast<int>(std::floor((-sample.position[2]) / kExteriorCellSize));
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dx = -1; dx <= 1; ++dx) requiredCells.insert({gridX + dx, gridZ + dz});
        }
    }
    FalloutAssetSource assets;
    if (!assets.open(dataPath)) return failJson("game archives could not be opened");

    struct BuiltCell {
        FalloutCellRecord record;
        ImportedScene scene;
    };
    std::map<std::pair<int, int>, BuiltCell> built;
    std::size_t missingCells = 0u;
    std::size_t intentionalDrops = 0u;
    std::size_t skippedShapes = 0u;
    std::size_t invalidTriangles = 0u;
    std::vector<std::string> visibleFailures;
    std::vector<std::string> modelDiagnostics;
    std::set<std::string> auditedModels;
    for (const auto& grid : requiredCells) {
        const auto entry = cellsByGrid.find(grid);
        if (entry == cellsByGrid.end()) {
            ++missingCells;
            continue;
        }
        BuiltCell cell;
        if (!extractFalloutCellMerged(index, order, *entry->second, cell.record, error)) {
            visibleFailures.push_back("cell " + std::to_string(grid.first) + "," +
                                      std::to_string(grid.second) + ": " + error);
            continue;
        }
        CellSceneBuilder builder(assets, tables);
        builder.setTextureBudget(1u);
        builder.setMaxTextureSize(64u);
        builder.addCell(cell.record);
        builder.finish(cell.scene);
        const CellBuildStats& stats = builder.stats();
        intentionalDrops += stats.editorMarkerModelsSkipped + stats.disabledReferencesSkipped +
                            stats.effectMeshesSkipped;
        for (const FalloutPlacedReference& reference : cell.record.references) {
            if (reference.isDeleted || (reference.recordFlags & 0x800u) != 0u) continue;
            const auto pathIt = tables.staticModelPaths.find(reference.baseFormId);
            if (pathIt == tables.staticModelPaths.end() || pathIt->second.empty()) continue;
            const std::string& modelPath = pathIt->second;
            if (isIntentionalAuditModel(modelPath)) {
                ++intentionalDrops;
                continue;
            }
            const std::string modelKey = toLowerAscii(modelPath);
            if (!auditedModels.insert(modelKey).second) continue;
            std::vector<std::uint8_t> bytes;
            std::string modelError;
            if (!assets.resolveMesh(modelPath, bytes, modelError)) {
                visibleFailures.push_back(modelPath + ": unresolved asset: " + modelError);
                continue;
            }
            NifModel model;
            if (!parseNifStaticMesh(bytes, model, modelError) || model.shapes.empty()) {
                visibleFailures.push_back(modelPath + ": no rendered geometry: " + modelError);
                continue;
            }
            std::size_t trianglesBefore = 0u;
            std::size_t trianglesAfter = 0u;
            std::array<std::size_t, 4> semanticCounts{};
            for (const NifShape& shape : model.shapes) {
                trianglesBefore += shape.sourceTriangleCount;
                trianglesAfter += shape.triangleIndices.size() / 3u;
                semanticCounts[static_cast<std::size_t>(shape.alphaSemantic)]++;
            }
            skippedShapes += model.skippedShapeCount + model.nodeParseFailedCount +
                             model.unhandledNodeTypeCount;
            intentionalDrops += model.hiddenShapeCount + model.editorMarkerShapeCount +
                                model.inactiveSwitchSubtreeCount;
            if (model.skippedShapeCount != 0u || model.nodeParseFailedCount != 0u ||
                model.unhandledNodeTypeCount != 0u) {
                visibleFailures.push_back(modelPath + ": skipped visible shape or subtree");
            }
            std::ostringstream diagnostic;
            diagnostic << "{\"model_path\":" << jsonString(modelPath)
                       << ",\"shape_count\":" << model.shapes.size()
                       << ",\"triangles_before_validation\":" << trianglesBefore
                       << ",\"triangles_after_validation\":" << trianglesAfter
                       << ",\"skipped_shape_count\":" << model.skippedShapeCount
                       << ",\"failed_node_count\":" << model.nodeParseFailedCount
                       << ",\"inactive_switch_subtree_count\":"
                       << model.inactiveSwitchSubtreeCount
                       << ",\"failed_node_types\":[";
            for (std::size_t i = 0; i < model.failedNodeTypes.size(); ++i) {
                if (i != 0u) diagnostic << ',';
                diagnostic << jsonString(model.failedNodeTypes[i]);
            }
            diagnostic << "],\"materials\":{\"opaque\":" << semanticCounts[0]
                       << ",\"cutout\":" << semanticCounts[1]
                       << ",\"explicit_transparency\":" << semanticCounts[2]
                       << ",\"vertex_fade\":" << semanticCounts[3] << "}}";
            modelDiagnostics.push_back(diagnostic.str());
        }
        for (const ImportedSceneCollisionTriangle& triangle : cell.scene.collisionTriangles) {
            for (float value : triangle.vertices) {
                if (!std::isfinite(value)) { ++invalidTriangles; break; }
            }
        }
        built.emplace(grid, std::move(cell));
    }

    std::size_t collisionSamples = 0u;
    std::size_t belowSurfaceSamples = 0u;
    float minimumClearance = std::numeric_limits<float>::infinity();
    std::vector<std::string> unsafeSamples;
    for (std::size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex) {
        const TourAuditPoint& sample = samples[sampleIndex];
        const int gridX = static_cast<int>(std::floor(sample.position[0] / kExteriorCellSize));
        const int gridZ = static_cast<int>(std::floor((-sample.position[2]) / kExteriorCellSize));
        const auto ownCell = built.find({gridX, gridZ});
        if (ownCell != built.end()) {
            float surface = sampleCellTerrain(ownCell->second.record,
                                              sample.position[0], sample.position[2]);
            const FalloutWorldspaceRecord* worldDefaults = tables.findWorldspace(worldspaceFormId);
            if (ownCell->second.record.hasWater) {
                surface = std::max(surface, ownCell->second.record.waterHeight);
            } else if (worldDefaults != nullptr && worldDefaults->hasDefaultHeights) {
                const float terrain = surface;
                if (!std::isfinite(terrain) || worldDefaults->defaultWaterHeight > terrain) {
                    surface = std::max(surface, worldDefaults->defaultWaterHeight);
                }
            }
            if (std::isfinite(surface)) {
                const float clearance = sample.position[1] - surface;
                minimumClearance = std::min(minimumClearance, clearance);
                if (clearance < 64.0f) {
                    ++belowSurfaceSamples;
                    if (unsafeSamples.size() < 64u) {
                        std::ostringstream issue;
                        issue << "{\"sample\":" << sampleIndex
                              << ",\"reason\":\"surface-clearance\",\"position\":["
                              << sample.position[0] << ',' << sample.position[1] << ','
                              << sample.position[2] << "],\"clearance\":" << clearance << '}';
                        unsafeSamples.push_back(issue.str());
                    }
                }
            }
        }
        bool intersects = false;
        float nearestDistanceSquared = std::numeric_limits<float>::infinity();
        float nearestTriangle[9]{};
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dx = -1; dx <= 1; ++dx) {
                const auto candidate = built.find({gridX + dx, gridZ + dz});
                if (candidate == built.end()) continue;
                for (const ImportedSceneCollisionTriangle& triangle :
                     candidate->second.scene.collisionTriangles) {
                    const float distanceSquared =
                        pointTriangleDistanceSquared(sample.position, triangle.vertices);
                    if (distanceSquared < nearestDistanceSquared) {
                        nearestDistanceSquared = distanceSquared;
                        std::copy_n(triangle.vertices, 9u, nearestTriangle);
                    }
                    if (distanceSquared < (48.0f * 48.0f)) {
                        intersects = true;
                    }
                }
            }
        }
        if (intersects) {
            ++collisionSamples;
            if (unsafeSamples.size() < 64u) {
                std::ostringstream issue;
                issue << "{\"sample\":" << sampleIndex
                      << ",\"reason\":\"collision-sphere\",\"position\":["
                      << sample.position[0] << ',' << sample.position[1] << ','
                      << sample.position[2] << "],\"distance\":"
                      << std::sqrt(nearestDistanceSquared) << ",\"nearest_triangle\":[";
                for (int component = 0; component < 9; ++component) {
                    if (component != 0) issue << ',';
                    issue << nearestTriangle[component];
                }
                issue << "]}";
                unsafeSamples.push_back(issue.str());
            }
        }
    }

    const bool ok = missingCells == 0u && visibleFailures.empty() && skippedShapes == 0u &&
                    invalidTriangles == 0u && collisionSamples == 0u && belowSurfaceSamples == 0u;
    std::cout << std::setprecision(7)
              << "{\"version\":1,\"ok\":" << (ok ? "true" : "false")
              << ",\"worldspace\":" << jsonString(worldspace)
              << ",\"tour\":" << jsonString(tourPath.string())
              << ",\"load_order_fingerprint\":" << jsonString(order.fingerprint())
              << ",\"sample_count\":" << samples.size()
              << ",\"required_cell_count\":" << requiredCells.size()
              << ",\"built_cell_count\":" << built.size()
              << ",\"missing_cell_count\":" << missingCells
              << ",\"collision_sample_count\":" << collisionSamples
              << ",\"below_surface_sample_count\":" << belowSurfaceSamples
              << ",\"minimum_surface_clearance\":"
              << (std::isfinite(minimumClearance) ? std::to_string(minimumClearance) : "null")
              << ",\"intentional_drop_count\":" << intentionalDrops
              << ",\"skipped_visible_shape_count\":" << skippedShapes
              << ",\"invalid_triangle_count\":" << invalidTriangles
              << ",\"visible_failures\":[";
    for (std::size_t i = 0; i < visibleFailures.size(); ++i) {
        if (i != 0u) std::cout << ',';
        std::cout << jsonString(visibleFailures[i]);
    }
    std::cout << "],\"model_diagnostics\":[";
    for (std::size_t i = 0; i < modelDiagnostics.size(); ++i) {
        if (i != 0u) std::cout << ',';
        std::cout << modelDiagnostics[i];
    }
    std::cout << "],\"unsafe_samples\":[";
    for (std::size_t i = 0; i < unsafeSamples.size(); ++i) {
        if (i != 0u) std::cout << ',';
        std::cout << unsafeSamples[i];
    }
    std::cout << "]}\n";
    return ok ? 0 : 1;
}

namespace {

bool resolveProbeContentProfile(
    const std::filesystem::path& source, int argc, char** argv, int optionStart,
    odai::importer::fnv::ResolvedContentProfile& profile, std::string& error) {
    odai::importer::fnv::ContentProfileResolveOptions options;
    for (int i = optionStart; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            options.dataRootOverride = std::filesystem::path(argv[++i]);
        } else if (std::strcmp(argv[i], "--mods-root") == 0 && i + 1 < argc) {
            options.modsRoot = std::filesystem::path(argv[++i]);
        } else if (std::strcmp(argv[i], "--reindex-content") == 0) {
            options.forceContentReindex = true;
        }
    }
    return odai::importer::fnv::resolveContentProfile(source, options, profile, error);
}

bool openProfileLoadOrder(
    const odai::importer::fnv::ResolvedContentProfile& profile,
    odai::importer::fnv::FalloutLoadOrder& order, std::string& error) {
    return order.open(profile, error);
}

struct Tes3ProbeContext {
    odai::importer::fnv::ResolvedContentProfile profile;
    odai::importer::fnv::FalloutLoadOrder order;
    std::shared_ptr<odai::bethesda::Tes3ContentStore> content;
    odai::bethesda::Tes3Runtime runtime;
};

bool loadTes3ProbeContext(
    const std::filesystem::path& source, int argc, char** argv, int optionStart,
    Tes3ProbeContext& out, std::string& error) {
    using namespace odai::bethesda;
    using namespace odai::importer::fnv;
    if (!resolveProbeContentProfile(source, argc, argv, optionStart, out.profile, error)) {
        return false;
    }
    if (out.profile.game != BethesdaGame::Morrowind) {
        error = "TES3 probe requires a Morrowind content profile";
        return false;
    }
    if (!openProfileLoadOrder(out.profile, out.order, error)) return false;
    out.content = std::make_shared<Tes3ContentStore>();
    if (!out.content->load(out.order, out.profile.encoding, error)) return false;
    return out.runtime.configure(out.content,
        ObjectId::persistent(makeTes3RecordKey("NPC_", "player")), error);
}

const char* tes3QuestStatusName(odai::bethesda::Tes3QuestStatus status) {
    using odai::bethesda::Tes3QuestStatus;
    switch (status) {
        case Tes3QuestStatus::None: return "none";
        case Tes3QuestStatus::Name: return "start";
        case Tes3QuestStatus::Finished: return "finish";
        case Tes3QuestStatus::Restart: return "restart";
    }
    return "none";
}

nlohmann::json tes3InfoTrace(const odai::bethesda::Tes3DialogueInfo& info) {
    nlohmann::json conditions = nlohmann::json::array();
    for (const odai::bethesda::Tes3DialogueCondition& condition : info.conditions) {
        conditions.push_back({{"rule", condition.rawRule},
            {"valid", condition.valid}, {"variable", condition.variable}});
    }
    return {{"id", info.id}, {"record", info.record.toString()},
        {"previous", info.previousId}, {"next", info.nextId},
        {"index", info.dispositionOrJournalIndex},
        {"status", tes3QuestStatusName(info.questStatus)},
        {"actor", info.actor}, {"faction", info.faction}, {"cell", info.cell},
        {"source", info.sourcePlugin}, {"conditions", std::move(conditions)},
        {"has_result_script", !info.resultScript.empty()}};
}

int tes3ScriptCheckCommand(
    const std::filesystem::path& source, int argc, char** argv, int optionStart) {
    using namespace odai::bethesda;
    Tes3ProbeContext context;
    std::string error;
    nlohmann::json output;
    if (!loadTes3ProbeContext(source, argc, argv, optionStart, context, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"error", error}}).dump(2) << '\n';
        return 1;
    }
    const Tes3ScriptCheckReport& report = context.runtime.scriptCheckReport();
    const bool strict = std::any_of(argv + optionStart, argv + argc,
        [](const char* value) { return std::strcmp(value, "--strict") == 0; });
    const bool ok = !strict || report.strictPass();
    output = {{"version", 1}, {"ok", ok}, {"strict", strict},
        {"strict_pass", report.strictPass()}, {"profile", context.profile.name},
        {"fingerprint", context.profile.fingerprint}, {"encoding", context.profile.encoding},
        {"scripts", report.scripts}, {"result_scripts", report.resultScripts},
        {"compiled", report.compiled}, {"command_use", report.commandUse},
        {"unsupported_commands", report.unsupportedCommands},
        {"diagnostics", report.diagnostics},
        {"content", {{"records", context.content->stats().recordsRead},
            {"dialogues", context.content->stats().dialogues},
            {"infos", context.content->stats().infos},
            {"references", context.content->stats().references}}}};
    std::cout << output.dump(2) << '\n';
    return ok ? 0 : 1;
}

int tes3ScriptSourceCommand(
    const std::filesystem::path& source, std::string_view scriptId,
    int argc, char** argv, int optionStart) {
    Tes3ProbeContext context;
    std::string error;
    if (!loadTes3ProbeContext(source, argc, argv, optionStart, context, error)) {
        std::cout << "TES3 profile failed: " << error << '\n';
        return 1;
    }
    const odai::bethesda::Tes3ScriptDefinition* script = context.content->findScript(scriptId);
    if (script == nullptr) {
        std::cout << "TES3 script not found: " << scriptId << '\n';
        return 1;
    }
    std::cout << script->source;
    if (script->source.empty() || script->source.back() != '\n') std::cout << '\n';
    return 0;
}

int tes3SpellTraceCommand(
    const std::filesystem::path& source, std::string_view spellId,
    int argc, char** argv, int optionStart) {
    Tes3ProbeContext context;
    std::string error;
    if (!loadTes3ProbeContext(source, argc, argv, optionStart, context, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"error", error}}).dump(2) << '\n';
        return 1;
    }
    const odai::bethesda::Tes3SpellDefinition* spell = context.content->findSpell(spellId);
    if (spell == nullptr) {
        std::cout << nlohmann::json({{"ok", false}, {"error", "spell id not found"},
            {"spell", spellId}}).dump(2) << '\n';
        return 1;
    }
    nlohmann::json effects = nlohmann::json::array();
    for (const odai::bethesda::Tes3SpellEffect& effect : spell->effects) {
        effects.push_back({{"effect_id", effect.effectId}, {"skill", effect.skill},
            {"attribute", effect.attribute}, {"range", effect.range}, {"area", effect.area},
            {"duration", effect.duration}, {"magnitude_min", effect.magnitudeMin},
            {"magnitude_max", effect.magnitudeMax}});
    }
    std::cout << nlohmann::json({{"version", 1}, {"ok", true},
        {"record", spell->record.toString()}, {"id", spell->id}, {"name", spell->name},
        {"type", spell->type}, {"cost", spell->cost}, {"flags", spell->flags},
        {"source", spell->sourcePlugin}, {"effects", std::move(effects)}}).dump(2) << '\n';
    return 0;
}

int tes3DialogueTraceCommand(
    const std::filesystem::path& source, std::string_view actorOrTopic,
    int argc, char** argv, int optionStart) {
    using namespace odai::bethesda;
    Tes3ProbeContext context;
    std::string error;
    if (!loadTes3ProbeContext(source, argc, argv, optionStart, context, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"error", error}}).dump(2) << '\n';
        return 1;
    }
    nlohmann::json output = {{"version", 1}, {"ok", true},
        {"query", actorOrTopic}, {"profile", context.profile.name}};
    if (const Tes3DialogueDefinition* topic = context.content->findDialogue(actorOrTopic)) {
        nlohmann::json infos = nlohmann::json::array();
        for (const Tes3DialogueInfo& info : topic->infos) infos.push_back(tes3InfoTrace(info));
        output["kind"] = "topic";
        output["topic"] = topic->record.toString();
        output["type"] = static_cast<std::int32_t>(topic->type);
        output["infos"] = std::move(infos);
    } else {
        Tes3DialogueActorState actor;
        actor.object = ObjectId::persistent(makeTes3ReferenceKey("probe", 1u));
        actor.id = std::string(actorOrTopic);
        Tes3DialoguePlayerState player;
        player.object = context.runtime.playerObject();
        const Tes3DialogueResponse greeting = context.runtime.startDialogue(actor, player, true);
        output["kind"] = "actor";
        output["accepted"] = greeting.accepted;
        output["greeting_topic"] = greeting.topic.toString();
        output["greeting_info"] = greeting.info.toString();
        output["discovered_topics"] = greeting.discoveredTopics;
        output["available_topics"] = context.runtime.availableTopics(true);
        output["diagnostics"] = greeting.diagnostics;
    }
    std::cout << output.dump(2) << '\n';
    return 0;
}

int tes3QuestTraceCommand(
    const std::filesystem::path& source, std::string_view questId,
    int argc, char** argv, int optionStart) {
    using namespace odai::bethesda;
    Tes3ProbeContext context;
    std::string error;
    if (!loadTes3ProbeContext(source, argc, argv, optionStart, context, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"error", error}}).dump(2) << '\n';
        return 1;
    }
    const Tes3DialogueDefinition* quest = context.content->findDialogue(questId);
    if (quest == nullptr || quest->type != Tes3DialogueType::Journal) {
        std::cout << nlohmann::json({{"ok", false},
            {"error", "journal id not found"}, {"quest", questId}}).dump(2) << '\n';
        return 1;
    }
    nlohmann::json entries = nlohmann::json::array();
    for (const Tes3DialogueInfo& info : quest->infos) entries.push_back(tes3InfoTrace(info));
    nlohmann::json resultEdges = nlohmann::json::array();
    const std::string normalizedQuest = normalizeTes3Symbol(questId);
    for (const auto& [programId, program] : context.runtime.scripts().programs()) {
        for (const Tes3Instruction& instruction : program.instructions) {
            if (instruction.op != Tes3OpCode::Call || instruction.command != "journal" ||
                instruction.arguments.empty()) continue;
            std::string target = instruction.arguments.front();
            target.erase(std::remove(target.begin(), target.end(), '"'), target.end());
            if (normalizeTes3Symbol(target) != normalizedQuest) continue;
            nlohmann::json nativeCalls = nlohmann::json::array();
            for (const Tes3Instruction& native : program.instructions) {
                if (native.op != Tes3OpCode::Call) continue;
                nativeCalls.push_back({{"line", native.sourceLine}, {"target", native.target},
                    {"command", native.command}, {"arguments", native.arguments}});
            }
            resultEdges.push_back({{"program", programId}, {"line", instruction.sourceLine},
                {"arguments", instruction.arguments}, {"program_commands", program.commands},
                {"native_calls", std::move(nativeCalls)}});
        }
    }
    std::cout << nlohmann::json({{"version", 1}, {"ok", true},
        {"profile", context.profile.name}, {"fingerprint", context.profile.fingerprint},
        {"quest", quest->record.toString()}, {"entries", std::move(entries)},
        {"result_script_transitions", std::move(resultEdges)}}).dump(2) << '\n';
    return 0;
}

int tes3QuestSuiteCommand(
    const std::filesystem::path& source, std::optional<std::string> onlyQuest,
    int argc, char** argv, int optionStart) {
    using namespace odai::bethesda;
    Tes3ProbeContext context;
    std::string error;
    if (!loadTes3ProbeContext(source, argc, argv, optionStart, context, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"error", error}}).dump(2) << '\n';
        return 1;
    }
    nlohmann::json quests = nlohmann::json::array();
    std::map<std::string, std::set<std::string>> journalCommands;
    for (const auto& [programId, program] : context.runtime.scripts().programs()) {
        (void)programId;
        for (const Tes3Instruction& instruction : program.instructions) {
            if (instruction.op != Tes3OpCode::Call || instruction.command != "journal" ||
                instruction.arguments.empty()) continue;
            std::string journal = instruction.arguments.front();
            journal.erase(std::remove(journal.begin(), journal.end(), '"'), journal.end());
            auto& commands = journalCommands[normalizeTes3Symbol(journal)];
            commands.insert(program.commands.begin(), program.commands.end());
        }
    }
    std::uint64_t entryCount = 0u;
    std::uint64_t starts = 0u;
    std::uint64_t terminals = 0u;
    for (const auto& [key, quest] : context.content->dialogues()) {
        (void)key;
        if (quest.type != Tes3DialogueType::Journal) continue;
        if (onlyQuest.has_value() &&
            normalizeTes3Symbol(quest.id) != normalizeTes3Symbol(*onlyQuest)) continue;
        std::uint64_t questStarts = 0u;
        std::uint64_t questTerminals = 0u;
        for (const Tes3DialogueInfo& info : quest.infos) {
            ++entryCount;
            if (info.questStatus == Tes3QuestStatus::Name) ++questStarts;
            if (info.questStatus == Tes3QuestStatus::Finished ||
                info.questStatus == Tes3QuestStatus::Restart) ++questTerminals;
        }
        starts += questStarts;
        terminals += questTerminals;
        quests.push_back({{"id", quest.id}, {"record", quest.record.toString()},
            {"entries", quest.infos.size()}, {"starts", questStarts},
            {"terminals", questTerminals},
            {"direct_transition_commands", journalCommands[normalizeTes3Symbol(quest.id)]}});
    }
    const bool selectionFound = !onlyQuest.has_value() || !quests.empty();
    const bool closurePass = context.runtime.scriptCheckReport().strictPass();
    // This report is deliberately not a release-gate pass: structural journal
    // enumeration and script closure are inputs to, not substitutes for, the
    // authored event-driven transition explorer described by the TR gate.
    std::cout << nlohmann::json({{"version", 1}, {"ok", selectionFound},
        {"profile", context.profile.name}, {"fingerprint", context.profile.fingerprint},
        {"selected_quest", onlyQuest.value_or("")}, {"quest_count", quests.size()},
        {"entry_count", entryCount}, {"start_count", starts},
        {"terminal_count", terminals}, {"script_closure_pass", closurePass},
        {"transition_explorer_complete", false}, {"release_gate_passed", false},
        {"quests", std::move(quests)},
        {"runtime_blockers", nlohmann::json::array({
            "authored event-driven transition explorer is not complete"})}}).dump(2) << '\n';
    return selectionFound ? 0 : 1;
}

struct Tes3VirtualDoor {
    std::size_t sourceCell = 0u;
    std::size_t targetCell = 0u;
    odai::importer::ImportedSceneDoor door;
};

std::string tes3VirtualCellName(
    const odai::importer::fnv::FalloutCellIndexEntry& cell) {
    if (!cell.editorId.empty()) return cell.editorId;
    if (cell.hasGridCoords) {
        return "#" + std::to_string(cell.gridX) + "," + std::to_string(cell.gridZ);
    }
    return "cell:" + std::to_string(cell.cellFormId);
}

nlohmann::json tes3VirtualPosition(const float position[3]) {
    return nlohmann::json::array({position[0], position[1], position[2]});
}

int tes3VirtualPlayerCommand(
    const std::filesystem::path& source, int argc, char** argv, int optionStart) {
    using namespace odai::bethesda;
    using namespace odai::importer;
    using namespace odai::importer::fnv;

    Tes3ProbeContext context;
    std::string error;
    if (!loadTes3ProbeContext(source, argc, argv, optionStart, context, error)) {
        std::cout << nlohmann::json({{"ok", false}, {"error", error}}).dump(2) << '\n';
        return 1;
    }
    const bool strict = std::any_of(argv + optionStart, argv + argc,
        [](const char* value) { return std::strcmp(value, "--strict") == 0; });
    std::optional<std::filesystem::path> reportPath;
    for (int i = optionStart; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--report") == 0) reportPath = argv[i + 1];
    }

    FalloutCellIndex index;
    if (!buildFalloutCellIndex(context.order, index, error)) {
        std::cout << nlohmann::json({{"ok", false},
            {"error", "TES3 cell index failed: " + error}}).dump(2) << '\n';
        return 1;
    }

    std::unordered_map<std::uint32_t, std::size_t> cellByFormId;
    std::unordered_map<std::string, std::size_t> cellByName;
    std::map<std::pair<std::int32_t, std::int32_t>, std::size_t> cellByGrid;
    for (std::size_t i = 0u; i < index.cells.size(); ++i) {
        const FalloutCellIndexEntry& cell = index.cells[i];
        cellByFormId[cell.cellFormId] = i;
        if (!cell.editorId.empty()) {
            cellByName.emplace(normalizeTes3Symbol(cell.editorId), i);
        }
        if (!cell.isInterior && cell.hasGridCoords) {
            cellByGrid.emplace(std::make_pair(cell.gridX, cell.gridZ), i);
        }
    }
    const auto cellForNameOrPosition = [&](std::string_view name, const float position[3])
        -> std::optional<std::size_t> {
        if (!name.empty()) {
            const auto named = cellByName.find(normalizeTes3Symbol(name));
            if (named != cellByName.end()) return named->second;
            if (name.front() == '#') {
                const std::size_t comma = name.find(',');
                if (comma != std::string_view::npos) {
                    try {
                        const std::int32_t x = std::stoi(std::string(name.substr(1u, comma - 1u)));
                        const std::int32_t y = std::stoi(std::string(name.substr(comma + 1u)));
                        const auto grid = cellByGrid.find({x, y});
                        if (grid != cellByGrid.end()) return grid->second;
                    } catch (const std::exception&) {
                    }
                }
            }
        }
        if (index.cellWorldSize > 0.0f) {
            const std::int32_t x = static_cast<std::int32_t>(
                std::floor(position[0] / index.cellWorldSize));
            const std::int32_t y = static_cast<std::int32_t>(
                std::floor(position[1] / index.cellWorldSize));
            const auto grid = cellByGrid.find({x, y});
            if (grid != cellByGrid.end()) return grid->second;
        }
        return std::nullopt;
    };

    std::vector<Tes3VirtualDoor> doors;
    nlohmann::json gaps = nlohmann::json::array();
    std::size_t extractionFailures = 0u;
    for (std::size_t cellIndex = 0u; cellIndex < index.cells.size(); ++cellIndex) {
        FalloutCellRecord cell;
        std::string cellError;
        if (!extractFalloutCellMerged(
                index, context.order, index.cells[cellIndex], cell, cellError)) {
            ++extractionFailures;
            gaps.push_back({{"kind", "cell_extraction"},
                {"cell", tes3VirtualCellName(index.cells[cellIndex])},
                {"detail", cellError}});
            continue;
        }
        ImportedScene scene;
        appendResolvedDoors(cell, index, scene);
        for (ImportedSceneDoor& door : scene.doors) {
            const auto target = cellByFormId.find(door.targetCellFormId);
            if (target == cellByFormId.end()) {
                gaps.push_back({{"kind", "unresolved_door_destination"},
                    {"source_cell", tes3VirtualCellName(index.cells[cellIndex])},
                    {"reference", door.sourceReferenceFormId},
                    {"target_cell_form_id", door.targetCellFormId}});
                continue;
            }
            doors.push_back(Tes3VirtualDoor{cellIndex, target->second, std::move(door)});
        }
    }

    nlohmann::json actions = nlohmann::json::array();
    nlohmann::json guilds = nlohmann::json::array();
    std::size_t guildsEntered = 0u;
    std::size_t guildsExited = 0u;
    std::vector<std::vector<std::size_t>> outgoingDoors(index.cells.size());
    for (std::size_t doorIndex = 0u; doorIndex < doors.size(); ++doorIndex) {
        outgoingDoors[doors[doorIndex].sourceCell].push_back(doorIndex);
    }
    const auto pathFromExterior = [&](std::size_t targetCell) {
        std::vector<std::int64_t> previousDoor(index.cells.size(), -1);
        std::vector<bool> visited(index.cells.size(), false);
        std::vector<std::size_t> queue;
        for (std::size_t cellIndex = 0u; cellIndex < index.cells.size(); ++cellIndex) {
            if (!index.cells[cellIndex].isInterior) {
                visited[cellIndex] = true;
                queue.push_back(cellIndex);
            }
        }
        for (std::size_t cursor = 0u; cursor < queue.size() && !visited[targetCell]; ++cursor) {
            const std::size_t current = queue[cursor];
            for (const std::size_t doorIndex : outgoingDoors[current]) {
                const Tes3VirtualDoor& door = doors[doorIndex];
                if (door.door.locked || visited[door.targetCell]) continue;
                visited[door.targetCell] = true;
                previousDoor[door.targetCell] = static_cast<std::int64_t>(doorIndex);
                queue.push_back(door.targetCell);
            }
        }
        std::vector<std::size_t> path;
        for (std::size_t current = targetCell;
             visited[targetCell] && previousDoor[current] >= 0;) {
            const std::size_t doorIndex = static_cast<std::size_t>(previousDoor[current]);
            path.push_back(doorIndex);
            current = doors[doorIndex].sourceCell;
        }
        std::reverse(path.begin(), path.end());
        return std::make_pair(visited[targetCell], path);
    };
    const auto pathToExterior = [&](std::size_t sourceCell) {
        std::vector<std::int64_t> previousDoor(index.cells.size(), -1);
        std::vector<bool> visited(index.cells.size(), false);
        std::vector<std::size_t> queue = {sourceCell};
        visited[sourceCell] = true;
        std::optional<std::size_t> target;
        for (std::size_t cursor = 0u; cursor < queue.size() && !target.has_value(); ++cursor) {
            const std::size_t current = queue[cursor];
            if (current != sourceCell && !index.cells[current].isInterior) {
                target = current;
                break;
            }
            for (const std::size_t doorIndex : outgoingDoors[current]) {
                const Tes3VirtualDoor& door = doors[doorIndex];
                if (door.door.locked || visited[door.targetCell]) continue;
                visited[door.targetCell] = true;
                previousDoor[door.targetCell] = static_cast<std::int64_t>(doorIndex);
                queue.push_back(door.targetCell);
            }
        }
        std::vector<std::size_t> path;
        if (target.has_value()) {
            for (std::size_t current = *target; previousDoor[current] >= 0;) {
                const std::size_t doorIndex = static_cast<std::size_t>(previousDoor[current]);
                path.push_back(doorIndex);
                current = doors[doorIndex].sourceCell;
            }
            std::reverse(path.begin(), path.end());
        }
        return std::make_pair(target.has_value(), path);
    };
    for (std::size_t guildCell = 0u; guildCell < index.cells.size(); ++guildCell) {
        const FalloutCellIndexEntry& cell = index.cells[guildCell];
        const std::string normalized = normalizeTes3Symbol(cell.editorId);
        const std::size_t fightersGuild = normalized.find("fighters guild");
        const std::size_t guildOfFighters = normalized.find("guild of fighters");
        const std::size_t guildPhrase = fightersGuild != std::string::npos
            ? fightersGuild : guildOfFighters;
        const std::size_t phraseLength = fightersGuild != std::string::npos
            ? std::string_view("fighters guild").size()
            : std::string_view("guild of fighters").size();
        if (!cell.isInterior || guildPhrase == std::string::npos) continue;
        // A basement or barracks belonging to a named guild is part of that
        // city's route, not a second Fighters Guild to count independently.
        if (normalized.find(':', guildPhrase + phraseLength) != std::string::npos) continue;

        std::vector<const Tes3VirtualDoor*> entrances;
        std::vector<const Tes3VirtualDoor*> exits;
        for (const Tes3VirtualDoor& door : doors) {
            if (door.targetCell == guildCell) entrances.push_back(&door);
            if (door.sourceCell == guildCell && !index.cells[door.targetCell].isInterior) {
                exits.push_back(&door);
            }
        }
        const auto [entered, entryPath] = pathFromExterior(guildCell);
        const auto [exited, exitPath] = pathToExterior(guildCell);
        if (entered) {
            ++guildsEntered;
            for (std::size_t step = 0u; step < entryPath.size(); ++step) {
                const Tes3VirtualDoor& routeDoor = doors[entryPath[step]];
                actions.push_back({{"sequence", actions.size()}, {"action", "activate_door"},
                    {"result", step + 1u == entryPath.size()
                        ? "entered_fighters_guild" : "continued_to_fighters_guild"},
                    {"from", tes3VirtualCellName(index.cells[routeDoor.sourceCell])},
                    {"to", tes3VirtualCellName(index.cells[routeDoor.targetCell])},
                    {"reference", routeDoor.door.sourceReferenceFormId},
                    {"at", tes3VirtualPosition(routeDoor.door.position)},
                    {"arrival", tes3VirtualPosition(routeDoor.door.arrivalPosition)}});
            }
        } else {
            gaps.push_back({{"kind", "unreachable_guild_entrance"},
                {"cell", cell.editorId},
                {"detail", "no unlocked authored door path from an exterior cell"}});
        }
        if (exited) {
            ++guildsExited;
            for (std::size_t step = 0u; step < exitPath.size(); ++step) {
                const Tes3VirtualDoor& routeDoor = doors[exitPath[step]];
                actions.push_back({{"sequence", actions.size()}, {"action", "activate_door"},
                    {"result", step + 1u == exitPath.size()
                        ? "exited_fighters_guild" : "continued_from_fighters_guild"},
                    {"from", tes3VirtualCellName(index.cells[routeDoor.sourceCell])},
                    {"to", tes3VirtualCellName(index.cells[routeDoor.targetCell])},
                    {"reference", routeDoor.door.sourceReferenceFormId},
                    {"at", tes3VirtualPosition(routeDoor.door.position)},
                    {"arrival", tes3VirtualPosition(routeDoor.door.arrivalPosition)}});
            }
        } else {
            gaps.push_back({{"kind", "unreachable_guild_exit"},
                {"cell", cell.editorId},
                {"detail", "no unlocked authored door path to an exterior cell"}});
        }
        std::string city = cell.editorId;
        const std::size_t separator = city.find_first_of(",:");
        if (separator != std::string::npos) city.resize(separator);
        guilds.push_back({{"city", city}, {"cell", cell.editorId},
            {"entrances", entrances.size()}, {"exterior_exits", exits.size()},
            {"entry_path_doors", entryPath.size()}, {"exit_path_doors", exitPath.size()},
            {"entered", entered}, {"exited", exited}});
    }

    nlohmann::json transport = nlohmann::json::array();
    std::size_t striderRoutes = 0u;
    std::size_t teleportRoutes = 0u;
    std::size_t resolvedTransportRoutes = 0u;
    std::size_t unresolvedRequiredRoutes = 0u;
    for (const auto& [object, reference] : context.content->references()) {
        (void)object;
        if (!reference.enabled || reference.deleted) continue;
        const auto actorIt = context.content->actors().find(reference.base);
        if (actorIt == context.content->actors().end() ||
            actorIt->second.travelDestinations.empty()) continue;
        const Tes3ActorDefinition& actor = actorIt->second;
        const auto sourceCell = cellForNameOrPosition(reference.cell.textId, reference.position);
        const std::string actorClass = normalizeTes3Symbol(actor.actorClass);
        const std::string sourceName = sourceCell.has_value()
            ? normalizeTes3Symbol(tes3VirtualCellName(index.cells[*sourceCell])) : std::string{};
        std::string kind = "other_transport";
        if (actorClass.find("guild guide") != std::string::npos ||
            sourceName.find("guild of mages") != std::string::npos ||
            sourceName.find("mages guild") != std::string::npos) {
            kind = "guild_guide_teleport";
        } else if (actorClass.find("caravaner") != std::string::npos ||
                   normalizeTes3Symbol(actor.id).find("caravaner") != std::string::npos) {
            kind = "silt_strider";
        }
        if (kind == "other_transport") continue;

        for (std::size_t destinationIndex = 0u;
             destinationIndex < actor.travelDestinations.size(); ++destinationIndex) {
            const Tes3ActorDefinition::TravelDestination& destination =
                actor.travelDestinations[destinationIndex];
            const auto targetCell = cellForNameOrPosition(destination.cell, destination.position);
            const bool resolved = sourceCell.has_value() && targetCell.has_value();
            if (kind == "silt_strider") ++striderRoutes;
            else ++teleportRoutes;
            if (resolved) {
                ++resolvedTransportRoutes;
                actions.push_back({{"sequence", actions.size()},
                    {"action", kind == "silt_strider" ? "ride_silt_strider" : "use_guild_guide"},
                    {"actor", actor.id},
                    {"from", tes3VirtualCellName(index.cells[*sourceCell])},
                    {"to", tes3VirtualCellName(index.cells[*targetCell])},
                    {"arrival", tes3VirtualPosition(destination.position)}});
            } else {
                ++unresolvedRequiredRoutes;
                gaps.push_back({{"kind", "unresolved_transport_route"},
                    {"transport", kind}, {"actor", actor.id},
                    {"source", reference.cell.textId},
                    {"destination", destination.cell},
                    {"detail", sourceCell.has_value()
                        ? "destination cell could not be resolved"
                        : "transport actor source cell could not be resolved"}});
            }
            transport.push_back({{"kind", kind}, {"actor", actor.id},
                {"actor_name", actor.name}, {"source_reference", reference.id.toString()},
                {"source", sourceCell.has_value()
                    ? tes3VirtualCellName(index.cells[*sourceCell]) : reference.cell.textId},
                {"destination", targetCell.has_value()
                    ? tes3VirtualCellName(index.cells[*targetCell]) : destination.cell},
                {"destination_index", destinationIndex}, {"resolved", resolved}});
        }
    }

    if (guilds.empty()) gaps.push_back({{"kind", "no_fighters_guilds"},
        {"detail", "load order exposed no Fighters Guild interiors"}});
    if (striderRoutes == 0u) gaps.push_back({{"kind", "no_silt_strider_network"},
        {"detail", "no placed caravaner with authored destinations was discovered"}});
    if (teleportRoutes == 0u) gaps.push_back({{"kind", "no_guild_guide_network"},
        {"detail", "no placed guild guide with authored destinations was discovered"}});

    const bool coveragePass = extractionFailures == 0u &&
        guildsEntered == guilds.size() && guildsExited == guilds.size() &&
        striderRoutes > 0u && teleportRoutes > 0u && unresolvedRequiredRoutes == 0u;
    nlohmann::json output = {{"version", 1}, {"ok", coveragePass}, {"strict", strict},
        {"profile", context.profile.name}, {"fingerprint", context.profile.fingerprint},
        {"virtual_player", {{"deterministic", true},
            {"uses_authored_doors_and_travel_destinations", true},
            {"actions_attempted", actions.size()}, {"actions", std::move(actions)}}},
        {"coverage", {{"cells", index.cells.size()}, {"doors", doors.size()},
            {"cell_extraction_failures", extractionFailures},
            {"fighters_guilds", guilds.size()}, {"fighters_guilds_entered", guildsEntered},
            {"fighters_guilds_exited", guildsExited}, {"silt_strider_routes", striderRoutes},
            {"guild_guide_routes", teleportRoutes},
            {"resolved_transport_routes", resolvedTransportRoutes},
            {"unresolved_required_routes", unresolvedRequiredRoutes}}},
        {"fighters_guilds", std::move(guilds)}, {"transport_routes", std::move(transport)},
        {"gaps", std::move(gaps)}};

    if (reportPath.has_value()) {
        std::ofstream report(*reportPath, std::ios::trunc);
        if (!report) {
            std::cout << nlohmann::json({{"ok", false},
                {"error", "could not write report " + reportPath->string()}}).dump(2) << '\n';
            return 1;
        }
        report << output.dump(2) << '\n';
        std::cout << nlohmann::json({{"ok", coveragePass},
            {"report", reportPath->string()}, {"coverage", output["coverage"]},
            {"gap_count", output["gaps"].size()}}).dump(2) << '\n';
    } else {
        std::cout << output.dump(2) << '\n';
    }
    return (!strict || coveragePass) ? 0 : 1;
}

int profileCheckCommand(
    const std::filesystem::path& source, int argc, char** argv, int optionStart) {
    using namespace odai::importer::fnv;
    ResolvedContentProfile profile;
    std::string error;
    nlohmann::json output;
    output["profile_source"] = source.string();
    if (!resolveProbeContentProfile(source, argc, argv, optionStart, profile, error)) {
        output["ok"] = false;
        output["error"] = error;
        std::cout << output.dump(2) << '\n';
        return 1;
    }
    FalloutLoadOrder order;
    if (!openProfileLoadOrder(profile, order, error)) {
        output["ok"] = false;
        output["error"] = error;
        output["fingerprint"] = profile.fingerprint;
        std::cout << output.dump(2) << '\n';
        return 1;
    }
    ContentRecordIndex recordIndex;
    if (!recordIndex.build(order, error)) {
        output["ok"] = false;
        output["error"] = "record provenance index failed: " + error;
        std::cout << output.dump(2) << '\n';
        return 1;
    }
    std::map<std::string, std::size_t> unsupported;
    const std::set<std::string> scriptExtensions = {
        ".dll", ".pex", ".psc", ".lua", ".omwscripts"};
    for (const ContentLayer& layer : profile.layers) {
        std::error_code scanError;
        for (std::filesystem::recursive_directory_iterator it(
                 layer.root, std::filesystem::directory_options::skip_permission_denied,
                 scanError), end;
             !scanError && it != end; it.increment(scanError)) {
            std::error_code typeError;
            if (!it->is_regular_file(typeError) || typeError) continue;
            const std::string extension = toLowerAscii(it->path().extension().string());
            if (scriptExtensions.contains(extension)) ++unsupported[extension];
        }
    }
    nlohmann::json diagnostics = nlohmann::json::array();
    for (const ContentDiagnostic& item : profile.diagnostics) {
        diagnostics.push_back({{"severity",
            item.severity == ContentDiagnosticSeverity::Error ? "error" :
            item.severity == ContentDiagnosticSeverity::Warning ? "warning" : "info"},
            {"code", item.code}, {"message", item.message}, {"source", item.source.string()}});
    }
    nlohmann::json placed = nlohmann::json::array();
    for (const FalloutLoadOrderEntry& entry : order.entries()) {
        placed.push_back({{"name", entry.header.fileName},
            {"kind", entry.slot.kind == FalloutPluginSlotKind::Light ? "light" : "regular"},
            {"slot", entry.slot.index}, {"path", entry.path.string()}});
    }
    output["ok"] = true;
    output["name"] = profile.name;
    output["game"] = bethesdaGameName(profile.game);
    output["data_root"] = profile.dataRoot.string();
    output["fingerprint"] = profile.fingerprint;
    output["layers"] = profile.layers.size();
    output["plugins"] = std::move(placed);
    output["archives"] = profile.archives.size();
    output["record_identities"] = recordIndex.recordCount();
    output["record_overrides"] = recordIndex.overrideCount();
    output["record_deletions"] = recordIndex.deletionCount();
    output["unsupported_executable_content"] = unsupported;
    output["diagnostics"] = std::move(diagnostics);
    std::cout << output.dump(2) << '\n';
    return 0;
}

std::string normalizedVirtualPath(std::string value) {
    for (char& c : value) if (c == '/') c = '\\';
    return toLowerAscii(std::move(value));
}

int whyCommand(
    const std::filesystem::path& source, const std::string& key,
    int argc, char** argv, int optionStart) {
    using namespace odai::importer::fnv;
    ResolvedContentProfile profile;
    std::string error;
    if (!resolveProbeContentProfile(source, argc, argv, optionStart, profile, error)) {
        std::cout << "profile failed: " << error << '\n'; return 1;
    }
    if (key.rfind("0x", 0u) == 0u) {
        FalloutLoadOrder order;
        if (!openProfileLoadOrder(profile, order, error)) {
            std::cout << "load order failed: " << error << '\n'; return 1;
        }
        const auto formId = static_cast<std::uint32_t>(std::strtoul(key.c_str() + 2, nullptr, 16));
        ContentRecordIndex records;
        if (!records.build(order, error)) {
            std::cout << "record index failed: " << error << '\n'; return 1;
        }
        const auto* versions = records.versions(formId);
        if (versions == nullptr || versions->empty()) {
            const FalloutLoadOrderEntry* owner = order.ownerOf(formId);
            if (owner == nullptr) { std::cout << "no active plugin owns " << key << '\n'; return 1; }
            std::cout << key << " is allocated to " << owner->header.fileName
                      << " but has no indexed record\n";
            return 1;
        }
        for (std::size_t i = 0; i < versions->size(); ++i) {
            const ContentRecordVersion& version = (*versions)[i];
            std::cout << (i + 1u == versions->size() ? "WINNER " : "overridden ")
                      << version.type << " in " << version.pluginName
                      << (version.deleted ? " [deleted]" : "") << " ("
                      << version.pluginPath.string() << ")\n";
        }
        return 0;
    }
    const std::string wanted = normalizedVirtualPath(key);
    std::vector<std::pair<std::string, std::filesystem::path>> providers;
    const auto scanRoot = [&](const std::string& name, const std::filesystem::path& root) {
        std::error_code scanError;
        for (std::filesystem::recursive_directory_iterator it(
                 root, std::filesystem::directory_options::skip_permission_denied, scanError), end;
             !scanError && it != end; it.increment(scanError)) {
            if (!it->is_regular_file()) continue;
            std::error_code relativeError;
            const auto relative = std::filesystem::relative(it->path(), root, relativeError);
            if (!relativeError && normalizedVirtualPath(relative.generic_string()) == wanted) {
                providers.emplace_back(name, it->path());
            }
        }
    };
    scanRoot("base Data", profile.dataRoot);
    for (const ContentLayer& layer : profile.layers) scanRoot(layer.name, layer.root);
    for (const ContentArchive& item : profile.archives) {
        BsaArchive archive;
        if (!archive.open(item.path)) continue;
        if (archive.find(wanted) != nullptr) providers.emplace_back("archive", item.path);
    }
    if (providers.empty()) { std::cout << key << " has no provider\n"; return 1; }
    for (std::size_t i = 0; i < providers.size(); ++i) {
        std::cout << (i + 1u == providers.size() ? "WINNER " : "overridden ")
                  << providers[i].first << ": " << providers[i].second.string() << '\n';
    }
    return 0;
}

int conflictsCommand(
    const std::filesystem::path& source, int argc, char** argv, int optionStart) {
    using namespace odai::importer::fnv;
    ResolvedContentProfile profile;
    std::string error;
    if (!resolveProbeContentProfile(source, argc, argv, optionStart, profile, error)) {
        std::cout << "profile failed: " << error << '\n'; return 1;
    }
    std::unordered_map<std::string, std::vector<std::string>> providers;
    for (const ContentLayer& layer : profile.layers) {
        std::error_code scanError;
        for (std::filesystem::recursive_directory_iterator it(
                 layer.root, std::filesystem::directory_options::skip_permission_denied,
                 scanError), end;
             !scanError && it != end; it.increment(scanError)) {
            if (!it->is_regular_file()) continue;
            std::error_code relativeError;
            const auto relative = std::filesystem::relative(it->path(), layer.root, relativeError);
            if (!relativeError) providers[normalizedVirtualPath(relative.generic_string())].push_back(layer.name);
        }
    }
    nlohmann::json conflicts = nlohmann::json::array();
    std::size_t total = 0u;
    for (const auto& [path, sources] : providers) {
        if (sources.size() < 2u) continue;
        ++total;
        if (conflicts.size() < 100u) conflicts.push_back({{"path", path}, {"providers", sources},
            {"winner", sources.back()}});
    }
    std::cout << nlohmann::json({{"profile", profile.name}, {"conflict_count", total},
        {"shown", conflicts.size()}, {"conflicts", std::move(conflicts)}}).dump(2) << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--profilecheck") == 0) {
        return profileCheckCommand(argv[2], argc, argv, 3);
    }
    if (argc >= 4 && std::strcmp(argv[1], "--why") == 0) {
        return whyCommand(argv[2], argv[3], argc, argv, 4);
    }
    if (argc >= 3 && std::strcmp(argv[1], "--conflicts") == 0) {
        return conflictsCommand(argv[2], argc, argv, 3);
    }
    if (argc >= 4 && std::strcmp(argv[1], "--export-profile") == 0) {
        odai::importer::fnv::ResolvedContentProfile profile;
        std::string error;
        if (!resolveProbeContentProfile(argv[2], argc, argv, 4, profile, error) ||
            !odai::importer::fnv::writeOdaiContentProfile(argv[3], profile, error)) {
            std::cout << "export failed: " << error << '\n'; return 1;
        }
        std::cout << "wrote " << argv[3] << '\n'; return 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "--tes3-scriptcheck") == 0) {
        return tes3ScriptCheckCommand(argv[2], argc, argv, 3);
    }
    if (argc >= 4 && std::strcmp(argv[1], "--tes3-script-source") == 0) {
        return tes3ScriptSourceCommand(argv[2], argv[3], argc, argv, 4);
    }
    if (argc >= 4 && std::strcmp(argv[1], "--tes3-spell-trace") == 0) {
        return tes3SpellTraceCommand(argv[2], argv[3], argc, argv, 4);
    }
    if (argc >= 4 && std::strcmp(argv[1], "--tes3-dialogue-trace") == 0) {
        return tes3DialogueTraceCommand(argv[2], argv[3], argc, argv, 4);
    }
    if (argc >= 4 && std::strcmp(argv[1], "--tes3-quest-trace") == 0) {
        return tes3QuestTraceCommand(argv[2], argv[3], argc, argv, 4);
    }
    if (argc >= 3 && std::strcmp(argv[1], "--tes3-quest-suite") == 0) {
        std::optional<std::string> quest;
        for (int i = 3; i + 1 < argc; ++i) {
            if (std::strcmp(argv[i], "--quest") == 0) quest = argv[i + 1];
        }
        return tes3QuestSuiteCommand(argv[2], std::move(quest), argc, argv, 3);
    }
    if (argc >= 3 && std::strcmp(argv[1], "--tes3-virtual-player") == 0) {
        return tes3VirtualPlayerCommand(argv[2], argc, argv, 3);
    }
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
    if (mode == "--animationcheck" || mode == "--animation-strict") {
        return animationCheck(dataPath, mode == "--animation-strict");
    }
    if (mode == "--scriptcheck" && argc >= 4) {
        return scriptCheck(
            dataPath, argv[3], argc >= 5 && std::strcmp(argv[4], "--strict") == 0);
    }
    if (mode == "--quest-trace" && argc >= 5) {
        return questTrace(dataPath, argv[3], argv[4]);
    }
    if (mode == "--skyrim-dialogue-trace" && argc >= 5) {
        return skyrimDialogueTrace(dataPath, argv[3], argv[4]);
    }
    if (mode == "--scenario-check" && argc >= 4) {
        return scenarioCheck(dataPath, argv[3]);
    }
    if (mode == "--nifs") {
        const std::size_t limit = argc >= 4 ? static_cast<std::size_t>(std::stoull(argv[3])) : 500u;
        return probeNifs(dataPath, limit);
    }
    if (mode == "--nif" && argc >= 4) {
        return probeSingleNif(dataPath, argv[3]);
    }
    if (mode == "--lodheight" && argc >= 6) {
        return lodHeightAt(dataPath, argv[3], static_cast<float>(std::atof(argv[4])),
                           static_cast<float>(std::atof(argv[5])));
    }
    if (mode == "--basemodel" && argc >= 5) {
        odai::importer::fnv::EsmReader reader;
        if (!reader.open(dataPath / argv[3])) {
            std::cout << "open failed: " << reader.lastError() << "\n";
            return 1;
        }
        const std::string wanted = toLowerAscii(argv[4]);
        odai::importer::fnv::EsmReader::Visitor visitor;
        visitor.onRecord = [&](const odai::importer::fnv::EsmRecordView& record) {
            if (record.type != "CREA" && record.type != "NPC_") return;
            std::string edid, modl;
            for (const auto& sub : record.subrecords) {
                if (sub.size == 0u) continue;
                std::string v(reinterpret_cast<const char*>(sub.data), sub.size);
                while (!v.empty() && v.back() == '\0') v.pop_back();
                if (sub.type == "EDID") edid = v;
                else if (sub.type == "MODL") modl = v;
            }
            if (!edid.empty() && toLowerAscii(edid) == wanted) {
                std::cout << record.type << " " << edid << " form=" << std::hex
                          << std::uppercase << record.formId << std::dec
                          << " MODL=\"" << modl << "\"\n";
                for (const auto& sub : record.subrecords) {
                    if (sub.type == "NIFZ" && sub.size != 0u) {
                        // Creature body-part meshes: NUL-separated model paths.
                        // MODL on a CREA names the SKELETON, so this is where
                        // the geometry actually lives.
                        std::string blob(reinterpret_cast<const char*>(sub.data), sub.size);
                        std::size_t begin = 0;
                        while (begin < blob.size()) {
                            const std::size_t end = blob.find('\0', begin);
                            const std::string part = blob.substr(
                                begin, end == std::string::npos ? std::string::npos : end - begin);
                            if (!part.empty()) {
                                std::cout << "   NIFZ part: \"" << part << "\"\n";
                            }
                            if (end == std::string::npos) break;
                            begin = end + 1;
                        }
                    }
                }
            }
        };
        reader.walk(visitor);
        return 0;
    }
    if (mode == "--speakerpos" && argc >= 5) {
        odai::importer::fnv::SpeakerPlacement placement;
        std::string error;
        if (!odai::importer::fnv::findSpeakerPlacement(dataPath / argv[3], argv[4], placement, error)) {
            std::cout << "failed: " << error << "\n";
            return 1;
        }
        std::cout << argv[4] << " ref=" << std::hex << std::uppercase << placement.referenceFormId
                  << " cell=" << placement.cellFormId << std::dec
                  << " pos=(" << placement.position[0] << ", " << placement.position[1] << ", "
                  << placement.position[2] << ")\n"
                  << "  skeleton=\"" << placement.skeletonPath << "\"\n";
        for (const std::string& part : placement.bodyPartPaths) {
            std::cout << "  part=\"" << part << "\"\n";
        }
        return 0;
    }
    if (mode == "--dialoguetree" && argc >= 5) {
        return probeDialogueTree(dataPath / argv[3], argv[4],
                                 argc >= 6 ? static_cast<std::size_t>(std::stoull(argv[5])) : 6u);
    }
    if (mode == "--dialogue" && argc >= 5) {
        return probeDialogue(dataPath / argv[3], argv[4],
                             argc >= 6 ? static_cast<std::size_t>(std::stoull(argv[5])) : 12u);
    }
    if (mode == "--footers") {
        const std::size_t limit = argc >= 4 ? static_cast<std::size_t>(std::stoull(argv[3])) : 5000u;
        return probeFooters(dataPath, limit);
    }
    if (mode == "--nifblocks" && argc >= 4) {
        return dumpNifBlocks(dataPath, argv[3]);
    }
    if (mode == "--kf" && argc >= 4) {
        return dumpKfAnimation(dataPath, argv[3]);
    }
    if (mode == "--actorsnear" && argc >= 7) {
        return probeActorsNear(dataPath / argv[3], static_cast<float>(std::atof(argv[4])),
                               static_cast<float>(std::atof(argv[5])),
                               static_cast<float>(std::atof(argv[6])));
    }
    if (mode == "--actor" && argc >= 5) {
        return probeActor(dataPath / argv[3], argv[4]);
    }
    if (mode == "--kfsweep" && argc >= 4) {
        return probeKfFolder(dataPath, argv[3]);
    }
    if (mode == "--regions" && argc >= 4) {
        return probeRegions(dataPath / argv[3], argc >= 5 ? static_cast<std::size_t>(std::atoi(argv[4])) : 15u);
    }
    if (mode == "--texture" && argc >= 4) {
        return probeTexture(dataPath, argv[3]);
    }
    if (mode == "--skeleton" && argc >= 4) {
        return probeSkeleton(dataPath, argv[3]);
    }
    if (mode == "--skinned" && argc >= 4) {
        return probeSkinnedNif(dataPath, argv[3]);
    }
    if (mode == "--character" && argc >= 5) {
        return probeCharacter(
            dataPath, argv[3], std::vector<std::string>(argv + 4, argv + argc));
    }
    if (mode == "--plugin" && argc >= 4) {
        return probePlugin(
            dataPath / argv[3],
            argc >= 5 ? static_cast<std::size_t>(std::atoi(argv[4])) : 20u);
    }
    if (mode == "--loadorder") {
        std::vector<std::string> plugins;
        std::filesystem::path source;
        std::string error;
        const std::optional<std::filesystem::path> explicitList =
            argc >= 4 ? std::optional<std::filesystem::path>(argv[3]) : std::nullopt;
        if (!odai::importer::fnv::resolveInstalledSkyrimPluginList(
                dataPath, explicitList, plugins, source, error)) {
            std::cout << "resolve failed: " << error << "\n";
            return 1;
        }
        odai::importer::fnv::FalloutLoadOrder order;
        if (!order.open(dataPath, plugins, error)) {
            std::cout << "open failed: " << error << "\n";
            return 1;
        }
        std::cout << "source: "
                  << (source.empty() ? std::string("installed official content")
                                     : source.string())
                  << "\n";
        for (const auto& entry : order.entries()) {
            std::cout << (entry.slot.kind ==
                                  odai::importer::fnv::FalloutPluginSlotKind::Light
                              ? "light   " : "regular ")
                      << entry.slot.index << "  " << entry.header.fileName << "\n";
        }
        std::cout << "fingerprint: " << order.fingerprint() << "\n";
        return 0;
    }
    if (mode == "--doorcheck") {
        std::vector<std::string> plugins;
        std::filesystem::path source;
        std::string error;
        const std::optional<std::filesystem::path> explicitList =
            argc >= 4 ? std::optional<std::filesystem::path>(argv[3]) : std::nullopt;
        if (!odai::importer::fnv::resolveInstalledSkyrimPluginList(
                dataPath, explicitList, plugins, source, error)) {
            std::cout << "resolve failed: " << error << "\n";
            return 1;
        }
        odai::importer::fnv::FalloutLoadOrder order;
        if (!order.open(dataPath, plugins, error)) {
            std::cout << "open failed: " << error << "\n";
            return 1;
        }
        odai::importer::fnv::FalloutCellIndex index;
        if (!odai::importer::fnv::buildFalloutCellIndex(order, index, error)) {
            std::cout << "index failed: " << error << "\n";
            return 1;
        }
        std::size_t doors = 0u;
        std::size_t unresolved = 0u;
        std::size_t failedCells = 0u;
        for (const auto& entry : index.cells) {
            odai::importer::fnv::FalloutCellRecord cell;
            if (!odai::importer::fnv::extractFalloutCellMerged(
                    index, order, entry, cell, error)) {
                ++failedCells;
                continue;
            }
            for (const auto& reference : cell.references) {
                if (!reference.hasTeleport || reference.isDeleted) {
                    continue;
                }
                ++doors;
                if (!index.cellIndexByReferenceFormId.contains(
                        reference.teleportTargetRefFormId)) {
                    ++unresolved;
                    if (unresolved <= 20u) {
                        std::cout << "unresolved door 0x" << std::hex
                                  << reference.formId << " -> 0x"
                                  << reference.teleportTargetRefFormId << std::dec << "\n";
                    }
                }
            }
        }
        std::cout << "doors=" << doors << " unresolved=" << unresolved
                  << " failedCells=" << failedCells << " markers="
                  << index.mapMarkers.size() << "\n";
        return (unresolved == 0u && failedCells == 0u) ? 0 : 1;
    }
    if (mode == "--routecheck") {
        const std::optional<std::filesystem::path> explicitList =
            argc >= 4 ? std::optional<std::filesystem::path>(argv[3]) : std::nullopt;
        return checkSkyrimTraversalRoute(dataPath, explicitList);
    }
    if (mode == "--tourcheck" && argc >= 5) {
        const std::optional<std::filesystem::path> explicitList =
            argc >= 6 ? std::optional<std::filesystem::path>(argv[5]) : std::nullopt;
        return checkSkyrimTour(dataPath, argv[3], argv[4], explicitList);
    }
    if (mode == "--refs" && argc >= 5) {
        return probeRefsByBaseType(
            dataPath, dataPath / argv[3], argv[4],
            argc >= 6 ? static_cast<std::size_t>(std::atoi(argv[5])) : 15u);
    }
    if (mode == "--placements" && argc >= 5) {
        return probePlacements(
            dataPath / argv[3], static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 16)),
            argc >= 6 ? static_cast<std::size_t>(std::atoi(argv[5])) : 25u);
    }
    if (mode == "--modelrefs" && argc >= 5) {
        return probeModelPlacements(
            dataPath / argv[3], argv[4],
            argc >= 6 ? static_cast<std::size_t>(std::atoi(argv[5])) : 25u);
    }
    if (mode == "--voicedialogue" && argc >= 5) {
        return probeVoiceTypeDialogue(dataPath / argv[3], argv[4]);
    }
    if (mode == "--formid" && argc >= 5) {
        return probeFormId(
            dataPath / argv[3], static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 16)));
    }
    if (mode == "--record" && argc >= 5) {
        return probeRecordType(
            dataPath / argv[3], argv[4], argc >= 6 ? static_cast<std::size_t>(std::atoi(argv[5])) : 3u);
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
