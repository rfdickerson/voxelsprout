// DDS bundler — converts PNG textures to BC3-compressed DDS with a full mip chain.
// Run once on the assets directory before launching the game; the runtime loader
// checks for a .dds sidecar before falling through to the PNG path.
//
//   odai_dds_bundler <file1.png> [file2.png ...]
//   odai_dds_bundler --dir <directory>       (recursively converts all .png files)

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_DXT_IMPLEMENTATION
#include <stb_dxt.h>

#include "import/dds.h"
#include "import/imported_scene.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

// Box-filter 2× downsample of an RGBA8 image.
std::vector<std::uint8_t> downsample(const std::uint8_t* src,
                                     std::uint32_t w, std::uint32_t h) {
    const std::uint32_t dw = std::max(1u, w >> 1u);
    const std::uint32_t dh = std::max(1u, h >> 1u);
    std::vector<std::uint8_t> dst(static_cast<std::size_t>(dw) * dh * 4u);
    for (std::uint32_t y = 0; y < dh; ++y) {
        for (std::uint32_t x = 0; x < dw; ++x) {
            const std::uint32_t sx0 = x * 2u, sy0 = y * 2u;
            const std::uint32_t sx1 = std::min(sx0 + 1u, w - 1u);
            const std::uint32_t sy1 = std::min(sy0 + 1u, h - 1u);
            for (int c = 0; c < 4; ++c) {
                const int s00 = src[(sy0 * w + sx0) * 4u + c];
                const int s10 = src[(sy0 * w + sx1) * 4u + c];
                const int s01 = src[(sy1 * w + sx0) * 4u + c];
                const int s11 = src[(sy1 * w + sx1) * 4u + c];
                dst[(y * dw + x) * 4u + c] =
                    static_cast<std::uint8_t>((s00 + s10 + s01 + s11 + 2) / 4);
            }
        }
    }
    return dst;
}

// Compress one RGBA8 mip level to BC3 (DXT5). Each 4×4 block → 16 bytes.
std::vector<std::uint8_t> compressBC3(const std::uint8_t* src,
                                      std::uint32_t w, std::uint32_t h) {
    const std::uint32_t bw = std::max(1u, (w + 3u) / 4u);
    const std::uint32_t bh = std::max(1u, (h + 3u) / 4u);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(bw) * bh * 16u, 0u);
    std::uint8_t block[64]; // 4×4 × 4 RGBA bytes
    std::size_t outOff = 0;
    for (std::uint32_t by = 0; by < bh; ++by) {
        for (std::uint32_t bx = 0; bx < bw; ++bx) {
            for (std::uint32_t py = 0; py < 4u; ++py) {
                const std::uint32_t sy = std::min(by * 4u + py, h - 1u);
                for (std::uint32_t px = 0; px < 4u; ++px) {
                    const std::uint32_t sx = std::min(bx * 4u + px, w - 1u);
                    std::memcpy(block + (py * 4u + px) * 4u,
                                src   + (sy  * w  + sx) * 4u, 4u);
                }
            }
            stb_compress_dxt_block(out.data() + outOff, block,
                                   /*alpha=*/1, STB_DXT_HIGHQUAL);
            outOff += 16u;
        }
    }
    return out;
}

bool convertPng(const std::filesystem::path& pngPath) {
    int w = 0, h = 0;
    stbi_uc* pixels = stbi_load(pngPath.string().c_str(), &w, &h, nullptr, 4);
    if (!pixels || w <= 0 || h <= 0) {
        std::cerr << "ERROR: cannot load " << pngPath << "\n";
        if (pixels) stbi_image_free(pixels);
        return false;
    }

    std::vector<std::uint8_t> mipData;
    std::uint32_t mipCount = 0;
    std::vector<std::uint8_t> cur(pixels,
        pixels + static_cast<std::size_t>(w) * h * 4u);
    stbi_image_free(pixels);

    std::uint32_t mw = static_cast<std::uint32_t>(w);
    std::uint32_t mh = static_cast<std::uint32_t>(h);
    while (true) {
        const auto compressed = compressBC3(cur.data(), mw, mh);
        mipData.insert(mipData.end(), compressed.begin(), compressed.end());
        ++mipCount;
        if (mw == 1u && mh == 1u) break;
        cur = downsample(cur.data(), mw, mh);
        mw  = std::max(1u, mw >> 1u);
        mh  = std::max(1u, mh >> 1u);
    }

    auto ddsPath = pngPath;
    ddsPath.replace_extension(".dds");
    if (!odai::importer::writeDds(ddsPath,
                                   static_cast<std::uint32_t>(w),
                                   static_cast<std::uint32_t>(h),
                                   mipCount,
                                   odai::importer::TextureFormat::BC3,
                                   mipData.data(), mipData.size())) {
        std::cerr << "ERROR: cannot write " << ddsPath << "\n";
        return false;
    }

    const std::size_t rawBytes =
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4u;
    const float ratio = mipData.empty() ? 0.f
        : static_cast<float>(rawBytes) / static_cast<float>(mipData.size());
    std::cout << pngPath.filename().string()
              << " -> " << ddsPath.filename().string()
              << "  [" << w << "x" << h << ", " << mipCount << " mips"
              << ", BC3, " << static_cast<int>(ratio) << ":1 vs RGBA8]\n";
    return true;
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: odai_dds_bundler <file1.png> [file2.png ...]\n"
                  << "       odai_dds_bundler --dir <directory>\n";
        return 1;
    }

    std::vector<std::filesystem::path> inputs;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--dir" && i + 1 < argc) {
            ++i;
            for (const auto& entry :
                 std::filesystem::recursive_directory_iterator(argv[i])) {
                if (entry.is_regular_file() &&
                    entry.path().extension() == ".png")
                    inputs.push_back(entry.path());
            }
        } else {
            inputs.emplace_back(arg);
        }
    }

    int failures = 0;
    for (const auto& p : inputs)
        if (!convertPng(p)) ++failures;

    if (failures > 0)
        std::cerr << failures << " file(s) failed.\n";
    return failures > 0 ? 1 : 0;
}
