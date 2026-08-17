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

#include "import/dds.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/bsa_archive.h"
#include "import/fnv/cell_builder.h"
#include "import/fnv/character_builder.h"
#include "import/fnv/esm_reader.h"
#include "import/fnv/dialogue_records.h"
#include "import/fnv/actor_records.h"
#include "import/fnv/fallout_records.h"
#include "import/fnv/kf_animation.h"
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

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
    if (texture.format != TextureFormat::RGBA8 || texture.rgba8.empty()) {
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
                  << ", editor markers " << model.editorMarkerShapeCount << "\n";
        for (const odai::importer::fnv::KfAnimation& animation : model.embeddedAnimations) {
            std::cout << "  animation \"" << animation.name << "\": "
                      << animation.duration() << "s, "
                      << (animation.loops() ? "looping" : "one-shot") << ", "
                      << animation.tracks.size() << " tracks\n";
        }
        for (const odai::importer::fnv::NifShape& shape : model.shapes) {
            std::cout << "  \"" << shape.name << "\" verts " << (shape.positions.size() / 3u) << ", tris "
                      << (shape.triangleIndices.size() / 3u) << ", uvs " << (shape.uvs.size() / 2u)
                      << ", alphaTest=" << (shape.alphaTest ? "yes" : "no")
                      << (shape.alphaTest
                              ? (" thr=" + std::to_string(static_cast<int>(shape.alphaThreshold)))
                              : std::string())
                      << ", twoSided=" << (shape.twoSided ? "yes" : "no")
                      << ", alphaBlend=" << (shape.alphaBlend ? "yes" : "no")
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
              << " terrainParts=" << stats.terrainPartsEmitted << "\n";
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
    if (!scene.waterPatches.empty()) {
        std::cout << "  water patches (engine origin/extent):\n";
        for (const auto& water : scene.waterPatches) {
            std::cout << "    (" << water.originX << "," << water.waterLevel << ","
                      << water.originZ << ") +(" << water.sizeX << "," << water.sizeZ << ")\n";
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

void printUsage() {
    std::cout << "Usage:\n"
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
              // Modes that existed but were never listed here.
              << "  odai_bethesda_probe <DataFilesPath> --find <substring> [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --cellindex <Plugin.esm> [worldspace] [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --buildcell <Plugin.esm> <Worldspace> <x> <z>\n"
              << "  odai_bethesda_probe <DataFilesPath> --floaters <Plugin.esm> <Worldspace> <x> <z>\n"
              << "  odai_bethesda_probe <DataFilesPath> --plugin <Plugin.esm> [typeCount]\n"
              << "  odai_bethesda_probe <DataFilesPath> --record <Plugin.esm> <TYPE> [dumpCount]\n"
              << "  odai_bethesda_probe <DataFilesPath> --refs <Plugin.esm> <BASETYPE> [topN]\n"
              << "  odai_bethesda_probe <DataFilesPath> --placements <Plugin.esm> <baseFormID> [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --modelrefs <Plugin.esm> <modelSubstring> [limit]\n"
              << "  odai_bethesda_probe <DataFilesPath> --cells <Plugin.esm> [filter]\n"
              << "  odai_bethesda_probe <DataFilesPath> --navm <Plugin.esm> [dumpCount]\n"
              << "  odai_bethesda_probe <DataFilesPath> --rotations <Plugin.esm> <CellEditorID>\n"
              << "  odai_bethesda_probe <anyDir> --scene <cooked.bin>\n";
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
