// FNV texture-pack normalizer — prepares a downloaded texture mod for use as a
// --mod directory.
//
//   odai_fnv_texture_pack --in <dir> --out <dir> [--max-size 1024] [--dry-run]
//
// Two jobs, both of which the runtime would otherwise pay for on every cell
// load, forever:
//
//   * Mip-drop every .dds to the ceiling the game will actually ask for.
//     CellSceneBuilder drops the same levels at load time anyway, so keeping
//     them on disk buys nothing and costs the read. NMC's SMALL pack is 2.0 GB
//     with 2048-square art in it; at a 1024 ceiling the same pack is ~1.4 GB
//     and every texture above the ceiling is read once here instead of once per
//     cache miss. Because these files ship full mip chains, the drop is a byte
//     skip -- the surviving levels are the pack's own, bit for bit, not a
//     resample. Nothing is recompressed and no quality is lost beyond the
//     resolution you chose.
//
//   * Lowercase every path. FalloutAssetSource indexes a mod directory
//     case-insensitively so this is not load-bearing, but it makes the tree
//     match what the NIFs ask for, which is worth something when reading a
//     `find` listing at 2 a.m.
//
// Files that are not .dds (a pack often carries a few meshes) are copied
// through unchanged apart from the path lowercasing. A .dds that will not parse
// is copied through as-is rather than dropped: the runtime is the authority on
// what it can read, and silently losing a texture is worse than carrying one
// that never resolves.

#include "import/dds.h"
#include "import/imported_scene.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <system_error>

namespace {

std::string toLowerAsciiCopy(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

std::string formatMegabytes(std::uint64_t bytes) {
    char buffer[64];
    std::snprintf(
        buffer, sizeof(buffer), "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buffer;
}

struct Totals {
    std::uint64_t filesSeen = 0;
    std::uint64_t ddsRewritten = 0;
    std::uint64_t ddsUnchanged = 0;
    std::uint64_t ddsUnparsed = 0;
    std::uint64_t copied = 0;
    std::uint64_t failures = 0;
    std::uint64_t bytesIn = 0;
    std::uint64_t bytesOut = 0;
};

// Lowercased, with the host separator, relative to the pack root.
std::filesystem::path loweredRelativePath(
    const std::filesystem::path& file, const std::filesystem::path& root) {
    std::error_code relativeError;
    const std::filesystem::path relative = std::filesystem::relative(file, root, relativeError);
    if (relativeError) {
        return {};
    }
    std::filesystem::path lowered;
    for (const std::filesystem::path& component : relative) {
        lowered /= toLowerAsciiCopy(component.string());
    }
    return lowered;
}

bool copyThrough(
    const std::filesystem::path& from, const std::filesystem::path& to, bool dryRun) {
    if (dryRun) {
        return true;
    }
    std::error_code copyError;
    std::filesystem::copy_file(
        from, to, std::filesystem::copy_options::overwrite_existing, copyError);
    if (copyError) {
        std::cerr << "  copy failed: " << from.string() << ": " << copyError.message() << "\n";
        return false;
    }
    return true;
}

void processFile(
    const std::filesystem::path& file,
    const std::filesystem::path& inRoot,
    const std::filesystem::path& outRoot,
    std::uint32_t maxSize,
    bool dryRun,
    Totals& totals) {
    const std::filesystem::path relative = loweredRelativePath(file, inRoot);
    if (relative.empty()) {
        ++totals.failures;
        return;
    }
    const std::filesystem::path destination = outRoot / relative;

    std::error_code sizeError;
    const auto inputSize = std::filesystem::file_size(file, sizeError);
    if (!sizeError) {
        totals.bytesIn += static_cast<std::uint64_t>(inputSize);
    }

    if (!dryRun) {
        std::error_code createError;
        std::filesystem::create_directories(destination.parent_path(), createError);
        if (createError) {
            std::cerr << "  cannot create " << destination.parent_path().string() << ": "
                      << createError.message() << "\n";
            ++totals.failures;
            return;
        }
    }

    if (toLowerAsciiCopy(file.extension().string()) != ".dds") {
        if (copyThrough(file, destination, dryRun)) {
            ++totals.copied;
            totals.bytesOut += static_cast<std::uint64_t>(sizeError ? 0u : inputSize);
        } else {
            ++totals.failures;
        }
        return;
    }

    odai::importer::ImportedSceneTexture texture;
    if (!odai::importer::loadDds(file, texture)) {
        // Unreadable here does not mean unreadable at runtime -- pass it along.
        if (copyThrough(file, destination, dryRun)) {
            ++totals.ddsUnparsed;
            totals.bytesOut += static_cast<std::uint64_t>(sizeError ? 0u : inputSize);
        } else {
            ++totals.failures;
        }
        return;
    }

    const std::uint32_t originalWidth = texture.width;
    const std::uint32_t originalHeight = texture.height;
    odai::importer::dropDdsMipLevels(texture, maxSize);

    // Already at or under the ceiling, or a format dropDdsMipLevels leaves
    // alone (RGBA8). Copying preserves whatever the original header said rather
    // than round-tripping it through writeDds for no reason.
    if (texture.width == originalWidth && texture.height == originalHeight) {
        if (copyThrough(file, destination, dryRun)) {
            ++totals.ddsUnchanged;
            totals.bytesOut += static_cast<std::uint64_t>(sizeError ? 0u : inputSize);
        } else {
            ++totals.failures;
        }
        return;
    }

    if (!dryRun &&
        !odai::importer::writeDds(
            destination, texture.width, texture.height, texture.mipLevelCount, texture.format,
            texture.rgba8.data(), texture.rgba8.size())) {
        std::cerr << "  write failed: " << destination.string() << "\n";
        ++totals.failures;
        return;
    }
    ++totals.ddsRewritten;
    // 128-byte DDS header; close enough for a summary, exact for the part that
    // matters.
    totals.bytesOut += static_cast<std::uint64_t>(texture.rgba8.size()) + 128u;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path inRoot;
    std::filesystem::path outRoot;
    std::uint32_t maxSize = 1024u;
    bool dryRun = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--in") == 0 && i + 1 < argc) {
            inRoot = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            outRoot = argv[++i];
        } else if (std::strcmp(argv[i], "--max-size") == 0 && i + 1 < argc) {
            const int requested = std::atoi(argv[++i]);
            maxSize = requested > 0 ? static_cast<std::uint32_t>(requested) : 0u;
        } else if (std::strcmp(argv[i], "--dry-run") == 0) {
            dryRun = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout
                << "odai_fnv_texture_pack --in <dir> --out <dir> [--max-size 1024] [--dry-run]\n"
                << "  Mip-drops every .dds to the ceiling and lowercases every path,\n"
                << "  producing a directory ready to pass to odai --mod.\n"
                << "  --max-size 0 keeps every level (paths are still lowercased).\n"
                << "  Match --max-size to the ODAI_FNV_TEX_SIZE you intend to run;\n"
                << "  a pack cut to 512 cannot be turned back up later.\n";
            return 0;
        }
    }

    if (inRoot.empty() || outRoot.empty()) {
        std::cerr << "usage: odai_fnv_texture_pack --in <dir> --out <dir> [--max-size 1024]\n";
        return 2;
    }

    std::error_code iterError;
    std::filesystem::recursive_directory_iterator iterator(
        inRoot, std::filesystem::directory_options::skip_permission_denied, iterError);
    if (iterError) {
        std::cerr << "cannot read " << inRoot.string() << ": " << iterError.message() << "\n";
        return 1;
    }

    // Refuse to write into the directory being read: the walk is live, so the
    // output would feed back into the input.
    std::error_code inCanonError;
    std::error_code outCanonError;
    const std::filesystem::path inCanon = std::filesystem::weakly_canonical(inRoot, inCanonError);
    const std::filesystem::path outCanon =
        std::filesystem::weakly_canonical(outRoot, outCanonError);
    if (!inCanonError && !outCanonError && inCanon == outCanon) {
        std::cerr << "--in and --out must differ (this rewrites into a fresh tree)\n";
        return 2;
    }

    std::cout << "normalizing " << inRoot.string() << " -> " << outRoot.string()
              << " at ceiling " << (maxSize == 0u ? "none" : std::to_string(maxSize) + " px")
              << (dryRun ? " (dry run)" : "") << "\n";

    Totals totals;
    for (const auto& entry : iterator) {
        std::error_code typeError;
        if (!entry.is_regular_file(typeError) || typeError) {
            continue;
        }
        ++totals.filesSeen;
        processFile(entry.path(), inRoot, outRoot, maxSize, dryRun, totals);
        if (totals.filesSeen % 500u == 0u) {
            std::cout << "  " << totals.filesSeen << " files...\n";
        }
    }

    std::cout << "done: " << totals.filesSeen << " files\n"
              << "  dds mip-dropped : " << totals.ddsRewritten << "\n"
              << "  dds already ok  : " << totals.ddsUnchanged << "\n"
              << "  dds unparsed    : " << totals.ddsUnparsed << " (copied as-is)\n"
              << "  non-dds copied  : " << totals.copied << "\n"
              << "  failures        : " << totals.failures << "\n"
              << "  size            : " << formatMegabytes(totals.bytesIn) << " -> "
              << formatMegabytes(totals.bytesOut) << "\n";
    return totals.failures == 0u ? 0 : 1;
}
