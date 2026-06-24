// SVG bundler — tessellates .svg icons into .odaivec geometry caches. Run once on
// an asset directory; the runtime loader (VectorIconRegistry::registerFromFile)
// prefers a fresh .odaivec sidecar over re-parsing the SVG.
//
//   odai_svg_bundler <file1.svg> [file2.svg ...]
//   odai_svg_bundler --dir <directory>     (recursively bundles all .svg files)
//   odai_svg_bundler --dir icons --sizes 16,32,64
//   odai_svg_bundler --dir icons --out baked
//
// One cache is written per (file, size). For a single size the sidecar is
// "<name>.odaivec"; for multiple sizes it is "<name>.<size>.odaivec".

#include "ui/ui_draw_list.h"
#include "ui/vector/svg_document.h"
#include "ui/vector/vector_cache_io.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<float> parseSizes(const std::string& csv) {
    std::vector<float> sizes;
    std::string token;
    for (char c : csv) {
        if (c == ',') {
            if (!token.empty()) {
                sizes.push_back(static_cast<float>(std::atof(token.c_str())));
                token.clear();
            }
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) {
        sizes.push_back(static_cast<float>(std::atof(token.c_str())));
    }
    return sizes;
}

bool bundleOne(const std::filesystem::path& svgPath, float sizePx, bool singleSize,
               const std::filesystem::path& outDir) {
    odai::ui::SvgTessellateOptions opts;
    opts.targetSizePx = sizePx;

    odai::ui::UiGeometryBlock block;
    if (!odai::ui::tessellateSvgFile(svgPath, opts, block)) {
        std::cerr << "ERROR: cannot tessellate " << svgPath << " (is nanosvg available?)\n";
        return false;
    }

    std::filesystem::path cachePath;
    const std::filesystem::path dir = outDir.empty() ? svgPath.parent_path() : outDir;
    const std::string stem = svgPath.stem().string();
    if (singleSize) {
        cachePath = dir / (stem + ".odaivec");
    } else {
        cachePath = dir / (stem + "." + std::to_string(static_cast<int>(sizePx)) + ".odaivec");
    }

    if (!odai::ui::writeVectorCache(cachePath, block, sizePx)) {
        std::cerr << "ERROR: cannot write " << cachePath << "\n";
        return false;
    }

    std::cout << svgPath.filename().string() << " -> " << cachePath.filename().string() << "  ["
              << static_cast<int>(sizePx) << "px, " << block.vertices.size() << " verts, "
              << block.indices.size() / 3 << " tris]\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: odai_svg_bundler <file1.svg> [file2.svg ...]\n"
                  << "       odai_svg_bundler --dir <directory> [--sizes 16,32,64] [--out <dir>]\n";
        return 1;
    }

    std::vector<std::filesystem::path> inputs;
    std::vector<float> sizes{32.0f};
    std::filesystem::path outDir;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--dir" && i + 1 < argc) {
            ++i;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(argv[i])) {
                if (entry.is_regular_file() && entry.path().extension() == ".svg") {
                    inputs.push_back(entry.path());
                }
            }
        } else if (arg == "--sizes" && i + 1 < argc) {
            ++i;
            auto parsed = parseSizes(argv[i]);
            if (!parsed.empty()) {
                sizes = std::move(parsed);
            }
        } else if (arg == "--out" && i + 1 < argc) {
            ++i;
            outDir = argv[i];
        } else {
            inputs.emplace_back(arg);
        }
    }

    if (!outDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(outDir, ec);
    }

    const bool singleSize = sizes.size() == 1;
    int failures = 0;
    for (const auto& p : inputs) {
        for (float s : sizes) {
            if (!bundleOne(p, s, singleSize, outDir)) {
                ++failures;
            }
        }
    }

    if (failures > 0) {
        std::cerr << failures << " conversion(s) failed.\n";
    }
    return failures > 0 ? 1 : 0;
}
