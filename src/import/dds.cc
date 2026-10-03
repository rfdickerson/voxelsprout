#include "import/dds.h"

#define BCDEC_IMPLEMENTATION
#include <bcdec.h>

#include <cstring>
#include <cmath>
#include <cctype>
#include <array>
#include <bit>
#include <algorithm>
#include <fstream>
#include <vector>

namespace odai::importer {

namespace {

constexpr std::uint32_t kDdsMagic      = 0x20534444u; // "DDS "
constexpr std::uint32_t kDdpfFourCC    = 0x4u;
constexpr std::uint32_t kDdpfRgb       = 0x40u;
constexpr std::uint32_t kDdsdCaps      = 0x1u;
constexpr std::uint32_t kDdsdHeight    = 0x2u;
constexpr std::uint32_t kDdsdWidth     = 0x4u;
constexpr std::uint32_t kDdsdPixelFmt  = 0x1000u;
constexpr std::uint32_t kDdsdMipmapCnt = 0x20000u;
constexpr std::uint32_t kDdsCaps1Tex   = 0x1000u;
constexpr std::uint32_t kDdsCaps1Mip   = 0x400000u;
constexpr std::uint32_t kDdsCaps1Cmplx = 0x8u;

constexpr std::uint32_t kFourCCDxt1    = 0x31545844u; // "DXT1"
constexpr std::uint32_t kFourCCDxt3    = 0x33545844u; // "DXT3"
constexpr std::uint32_t kFourCCDxt5    = 0x35545844u; // "DXT5"
constexpr std::uint32_t kFourCCAti1    = 0x31495441u; // "ATI1"
constexpr std::uint32_t kFourCCAti2    = 0x32495441u; // "ATI2"
constexpr std::uint32_t kFourCCDx10    = 0x30315844u; // "DX10"

constexpr std::uint32_t kDxgiBC1Unorm  = 71u;
constexpr std::uint32_t kDxgiBC2Unorm  = 74u;
constexpr std::uint32_t kDxgiBC3Unorm  = 77u;
constexpr std::uint32_t kDxgiBC4Unorm  = 80u;
constexpr std::uint32_t kDxgiBC5Unorm  = 83u;
constexpr std::uint32_t kDxgiBC7Unorm  = 98u;
constexpr std::uint32_t kDxgiBC7Srgb   = 99u;

#pragma pack(push, 1)
struct DdsPixelFormat {
    std::uint32_t size      = 32;
    std::uint32_t flags     = 0;
    std::uint32_t fourCC    = 0;
    std::uint32_t rgbBitCnt = 0;
    std::uint32_t rMask     = 0;
    std::uint32_t gMask     = 0;
    std::uint32_t bMask     = 0;
    std::uint32_t aMask     = 0;
};

struct DdsHeader {
    std::uint32_t  size            = 124;
    std::uint32_t  flags           = 0;
    std::uint32_t  height          = 0;
    std::uint32_t  width           = 0;
    std::uint32_t  pitchOrLinearSz = 0;
    std::uint32_t  depth           = 0;
    std::uint32_t  mipMapCount     = 0;
    std::uint32_t  reserved1[11]   = {};
    DdsPixelFormat ddspf;
    std::uint32_t  caps            = 0;
    std::uint32_t  caps2           = 0;
    std::uint32_t  caps3           = 0;
    std::uint32_t  caps4           = 0;
    std::uint32_t  reserved2       = 0;
};

struct DdsHeaderDxt10 {
    std::uint32_t dxgiFormat        = 0;
    std::uint32_t resourceDimension = 3; // D3D10_RESOURCE_DIMENSION_TEXTURE2D
    std::uint32_t miscFlag          = 0;
    std::uint32_t arraySize         = 1;
    std::uint32_t miscFlags2        = 0;
};
#pragma pack(pop)

static_assert(sizeof(DdsPixelFormat)  == 32);
static_assert(sizeof(DdsHeader)       == 124);
static_assert(sizeof(DdsHeaderDxt10)  == 20);

} // anonymous namespace

std::uint32_t ddsBlockBytes(TextureFormat format) {
    switch (format) {
        case TextureFormat::BC1:
        case TextureFormat::BC1Linear:
        case TextureFormat::BC4: return 8u;
        case TextureFormat::BC2:
        case TextureFormat::BC3:
        case TextureFormat::BC5:
        case TextureFormat::BC6HUfloat:
        case TextureFormat::BC6HSfloat:
        case TextureFormat::BC7: return 16u;
        default:                 return 0u;
    }
}

std::size_t ddsCubeByteCount(const ImportedSceneTexture& texture) {
    if (texture.arrayLayers != 6 || texture.width == 0 || texture.width > 32768 ||
        texture.width != texture.height || texture.mipLevelCount == 0 ||
        texture.mipLevelCount > std::bit_width(texture.width)) return 0;
    std::size_t bytes = 0;
    const auto block = ddsBlockBytes(texture.format);
    for (std::uint32_t mip = 0; mip < texture.mipLevelCount; ++mip) {
        const auto size = std::max(1u,texture.width >> mip);
        bytes += block ? std::size_t((size+3)/4)*((size+3)/4)*block : std::size_t(size)*size*4;
    }
    return bytes*6;
}

bool loadDdsFromMemory(const std::uint8_t* bytes, std::size_t byteCount, ImportedSceneTexture& out) {
    if (bytes == nullptr) return false;
    const std::size_t fileSize = byteCount;
    if (fileSize < 4u + sizeof(DdsHeader)) return false;
    // Aliased so the body below reads exactly as it did when it owned a vector.
    const std::uint8_t* const data = bytes;

    std::uint32_t magic = 0;
    std::memcpy(&magic, data, 4u);
    if (magic != kDdsMagic) return false;

    DdsHeader hdr{};
    std::memcpy(&hdr, data + 4u, sizeof(DdsHeader));
    if (hdr.size != 124u || hdr.ddspf.size != 32u || hdr.width == 0u || hdr.height == 0u) return false;
    if (hdr.width > 32768u || hdr.height > 32768u || hdr.mipMapCount > static_cast<unsigned>(std::bit_width(std::max(hdr.width,hdr.height))) ||
        (hdr.caps2 & 0x200000u) != 0u) return false;
    std::uint32_t layers = (hdr.caps2 & 0x200u) ? 6u : 1u;
    if (layers == 6 && hdr.ddspf.fourCC != kFourCCDx10 && ((hdr.caps2 & 0xfc00u) != 0xfc00u || hdr.width != hdr.height)) return false;
    TextureFormat  fmt        = TextureFormat::RGBA8;
    std::size_t    dataOffset = 4u + sizeof(DdsHeader);

    // Skyrim's per-cell water flow fields are legacy 32-bit ARGB DDS files,
    // not block-compressed textures. Decode their complete mip chain to RGBA8;
    // the masks are checked explicitly so this cannot reinterpret unrelated
    // legacy DDS layouts by accident.
    const bool legacyBgra =
        hdr.ddspf.rMask == 0x00ff0000u && hdr.ddspf.gMask == 0x0000ff00u &&
        hdr.ddspf.bMask == 0x000000ffu && hdr.ddspf.aMask == 0xff000000u;
    // Skyrim terrain LOD uses the same byte order without an alpha mask. It is
    // opaque display-colour data (Tamriel.<tier>.<x>.<y>.DDS), so synthesize
    // alpha=255 and retain sRGB sampling. Rejecting this exact XRGB layout made
    // every newly-loaded BTR surface a white fallback despite valid geometry.
    const bool legacyBgrx =
        hdr.ddspf.rMask == 0x00ff0000u && hdr.ddspf.gMask == 0x0000ff00u &&
        hdr.ddspf.bMask == 0x000000ffu && hdr.ddspf.aMask == 0u;
    // Skyrim's generated-object atlases use the other common 32-bit layout:
    // RGBA bytes with ABGR masks. Tamriel.Objects.DDS is the load-bearing case
    // (2048 square, eleven mips); rejecting it turns every BTO building into a
    // white silhouette, which is why distant Whiterun looked worse as soon as
    // its previously-unused generated geometry was enabled.
    const bool legacyRgba =
        hdr.ddspf.rMask == 0x000000ffu && hdr.ddspf.gMask == 0x0000ff00u &&
        hdr.ddspf.bMask == 0x00ff0000u && hdr.ddspf.aMask == 0xff000000u;
    if ((hdr.ddspf.flags & kDdpfRgb) && hdr.ddspf.rgbBitCnt == 24u &&
        (legacyBgrx || (hdr.ddspf.rMask == 0xffu && hdr.ddspf.gMask == 0xff00u &&
                       hdr.ddspf.bMask == 0xff0000u && hdr.ddspf.aMask == 0))) {
        const auto mips = std::max(1u, hdr.mipMapCount);
        std::size_t inputBytes = 0, outputBytes = 0;
        const bool padded = (hdr.flags & 8u) && hdr.pitchOrLinearSz == ((hdr.width * 3u + 3u) & ~3u);
        for (unsigned mip = 0; mip < mips; ++mip) {
            const auto w = std::max(1u, hdr.width >> mip), h = std::max(1u, hdr.height >> mip);
            inputBytes += std::size_t(padded ? ((w * 3u + 3u) & ~3u) : w * 3u) * h;
            outputBytes += std::size_t(w) * h * 4;
        }
        inputBytes *= layers; outputBytes *= layers;
        if (outputBytes > 512ull * 1024 * 1024 || inputBytes > fileSize - dataOffset) return false;
        ImportedSceneTexture decoded;
        decoded.width = hdr.width; decoded.height = hdr.height;
        decoded.mipLevelCount = mips; decoded.arrayLayers = layers;
        decoded.format = TextureFormat::RGBA8Srgb; decoded.rgba8.resize(outputBytes);
        std::size_t src = dataOffset, dst = 0;
        for (unsigned face = 0; face < layers; ++face) for (unsigned mip = 0; mip < mips; ++mip) {
            const auto w = std::max(1u, hdr.width >> mip), h = std::max(1u, hdr.height >> mip);
            const auto stride = padded ? ((w * 3u + 3u) & ~3u) : w * 3u;
            for (unsigned y = 0; y < h; ++y) for (unsigned x = 0; x < w; ++x) {
                const auto pixel = src + y * stride + x * 3;
                decoded.rgba8[dst++] = data[pixel + (legacyBgrx ? 2 : 0)];
                decoded.rgba8[dst++] = data[pixel + 1];
                decoded.rgba8[dst++] = data[pixel + (legacyBgrx ? 0 : 2)];
                decoded.rgba8[dst++] = 255;
            }
            src += std::size_t(stride) * h;
        }
        decoded.sourcePath = out.sourcePath;
        out = std::move(decoded); return true;
    }
    if ((hdr.ddspf.flags & kDdpfRgb) != 0u && hdr.ddspf.rgbBitCnt == 32u &&
        (legacyBgra || legacyBgrx || legacyRgba)) {
        const std::uint32_t mipCount = std::max(1u, hdr.mipMapCount);
        std::size_t chainBytes = 0u;
        std::uint32_t mw = hdr.width;
        std::uint32_t mh = hdr.height;
        for (std::uint32_t m = 0; m < mipCount; ++m) {
            chainBytes += static_cast<std::size_t>(mw) * mh * 4u;
            mw = std::max(1u, mw >> 1u);
            mh = std::max(1u, mh >> 1u);
        }
        chainBytes *= layers;
        if (chainBytes > 512ull * 1024 * 1024 || chainBytes > fileSize - dataOffset) return false;
        out.arrayLayers = layers;
        out.width = hdr.width;
        out.height = hdr.height;
        out.mipLevelCount = mipCount;
        out.format = (legacyRgba || legacyBgrx) ? TextureFormat::RGBA8Srgb : TextureFormat::RGBA8;
        out.rgba8.resize(chainBytes);
        if (legacyRgba) {
            std::memcpy(out.rgba8.data(), data + dataOffset, chainBytes);
        } else {
            for (std::size_t offset = 0u; offset < chainBytes; offset += 4u) {
                // Stored byte order is BGRA on little-endian hosts.
                out.rgba8[offset + 0u] = data[dataOffset + offset + 2u];
                out.rgba8[offset + 1u] = data[dataOffset + offset + 1u];
                out.rgba8[offset + 2u] = data[dataOffset + offset + 0u];
                out.rgba8[offset + 3u] = legacyBgrx ? 255u : data[dataOffset + offset + 3u];
            }
        }
        return true;
    }
    if (!(hdr.ddspf.flags & kDdpfFourCC)) return false;

    const std::uint32_t fcc = hdr.ddspf.fourCC;
    if      (fcc == kFourCCDxt1) { fmt = TextureFormat::BC1; }
    else if (fcc == kFourCCDxt3) { fmt = TextureFormat::BC2; }
    else if (fcc == kFourCCDxt5) { fmt = TextureFormat::BC3; }
    else if (fcc == kFourCCAti1) { fmt = TextureFormat::BC4; }
    else if (fcc == kFourCCAti2) { fmt = TextureFormat::BC5; }
    else if (fcc == kFourCCDx10) {
        if (fileSize < dataOffset + sizeof(DdsHeaderDxt10)) return false;
        DdsHeaderDxt10 dx10{};
        std::memcpy(&dx10, data + dataOffset, sizeof(DdsHeaderDxt10));
        dataOffset += sizeof(DdsHeaderDxt10);
        if (dx10.resourceDimension != 3u || dx10.arraySize != 1u) return false;
        layers = (dx10.miscFlag & 4u) ? 6u : 1u;
        if (layers == 6 && hdr.width != hdr.height) return false;
        switch (dx10.dxgiFormat) {
            case kDxgiBC1Unorm:                     fmt = TextureFormat::BC1; break;
            case kDxgiBC2Unorm:                     fmt = TextureFormat::BC2; break;
            case kDxgiBC3Unorm:                     fmt = TextureFormat::BC3; break;
            case kDxgiBC4Unorm:                     fmt = TextureFormat::BC4; break;
            case kDxgiBC5Unorm:                     fmt = TextureFormat::BC5; break;
            case 95u: fmt = TextureFormat::BC6HUfloat; break;
            case 96u: fmt = TextureFormat::BC6HSfloat; break;
            case kDxgiBC7Unorm: case kDxgiBC7Srgb:  fmt = TextureFormat::BC7; break;
            default: return false;
        }
    } else { return false; }

    const std::uint32_t bpb      = ddsBlockBytes(fmt);
    const std::uint32_t mipCount = std::max(1u, hdr.mipMapCount);

    std::size_t chainBytes = 0;
    {
        std::uint32_t mw = hdr.width, mh = hdr.height;
        for (std::uint32_t m = 0; m < mipCount; ++m) {
            chainBytes += static_cast<std::size_t>(std::max(1u, (mw + 3u) / 4u))
                        * std::max(1u, (mh + 3u) / 4u) * bpb;
            mw = std::max(1u, mw >> 1u);
            mh = std::max(1u, mh >> 1u);
        }
    }
    chainBytes *= layers;
    if (chainBytes > 512ull * 1024 * 1024 || chainBytes > fileSize - dataOffset) return false;

    out.arrayLayers = layers;
    out.width         = hdr.width;
    out.height        = hdr.height;
    out.mipLevelCount = mipCount;
    out.format        = fmt;
    out.rgba8.assign(data + dataOffset, data + dataOffset + chainBytes);
    return true;
}

bool loadDds(const std::filesystem::path& path, ImportedSceneTexture& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto fileSize = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::uint8_t> data(fileSize);
    if (fileSize != 0 &&
        !f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(fileSize))) {
        return false;
    }
    if (!loadDdsFromMemory(data.data(), data.size(), out)) return false;
    out.sourcePath = path.string();
    return true;
}

void dropDdsMipLevels(ImportedSceneTexture& tex, std::uint32_t maxDimension) {
    if (tex.arrayLayers != 1 && tex.arrayLayers != 6) return;
    const std::uint32_t bpb = ddsBlockBytes(tex.format);
    if (maxDimension == 0u || tex.mipLevelCount <= 1u) {
        return;
    }
    const auto faceBytes = tex.rgba8.size() / tex.arrayLayers;
    std::size_t dropBytes = 0;
    while (tex.mipLevelCount > 1u && (tex.width > maxDimension || tex.height > maxDimension)) {
        const std::size_t levelBytes = bpb ? static_cast<std::size_t>((tex.width + 3u) / 4u) *
            ((tex.height + 3u) / 4u) * bpb : std::size_t(tex.width) * tex.height * 4;
        if (dropBytes + levelBytes >= faceBytes) {
            break;  // never drop the whole chain
        }
        dropBytes += levelBytes;
        tex.width = std::max(1u, tex.width >> 1u);
        tex.height = std::max(1u, tex.height >> 1u);
        --tex.mipLevelCount;
    }
    if (dropBytes != 0u) {
        std::vector<std::uint8_t> retained;
        for (std::uint32_t face = 0; face < tex.arrayLayers; ++face)
            retained.insert(retained.end(),tex.rgba8.begin()+face*faceBytes+dropBytes,
                tex.rgba8.begin()+(face+1)*faceBytes);
        tex.rgba8 = std::move(retained);
    }
}

bool writeDds(const std::filesystem::path& path,
              std::uint32_t width, std::uint32_t height, std::uint32_t mipLevelCount,
              TextureFormat format,
              const std::uint8_t* mipData, std::size_t mipDataSize) {
    const std::uint32_t bpb = ddsBlockBytes(format);
    if (bpb == 0u || mipData == nullptr || mipDataSize == 0u) return false;

    const bool needDx10 = format == TextureFormat::BC7 || format == TextureFormat::BC6HUfloat || format == TextureFormat::BC6HSfloat;
    std::uint32_t fourCC = 0, dxgiFmt = 0;
    if (needDx10) {
        fourCC  = kFourCCDx10;
        dxgiFmt = format == TextureFormat::BC6HUfloat ? 95u : format == TextureFormat::BC6HSfloat ? 96u : kDxgiBC7Unorm;
    } else {
        switch (format) {
            case TextureFormat::BC1: fourCC = kFourCCDxt1; break;
            case TextureFormat::BC1Linear: fourCC = kFourCCDxt1; break;
            case TextureFormat::BC2: fourCC = kFourCCDxt3; break;
            case TextureFormat::BC3: fourCC = kFourCCDxt5; break;
            case TextureFormat::BC4: fourCC = kFourCCAti1; break;
            case TextureFormat::BC5: fourCC = kFourCCAti2; break;
            default: return false;
        }
    }

    DdsHeader hdr{};
    hdr.flags          = kDdsdCaps | kDdsdHeight | kDdsdWidth | kDdsdPixelFmt | kDdsdMipmapCnt;
    hdr.height         = height;
    hdr.width          = width;
    hdr.mipMapCount    = mipLevelCount;
    hdr.pitchOrLinearSz = std::max(1u, (width  + 3u) / 4u)
                        * std::max(1u, (height + 3u) / 4u) * bpb;
    hdr.ddspf.size     = 32u;
    hdr.ddspf.flags    = kDdpfFourCC;
    hdr.ddspf.fourCC   = fourCC;
    hdr.caps           = kDdsCaps1Tex
                       | (mipLevelCount > 1u ? kDdsCaps1Mip | kDdsCaps1Cmplx : 0u);

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const std::uint32_t magic = kDdsMagic;
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&hdr),   sizeof(DdsHeader));
    if (needDx10) {
        DdsHeaderDxt10 dx10{};
        dx10.dxgiFormat        = dxgiFmt;
        dx10.resourceDimension = 3u;
        dx10.arraySize         = 1u;
        f.write(reinterpret_cast<const char*>(&dx10), sizeof(DdsHeaderDxt10));
    }
    f.write(reinterpret_cast<const char*>(mipData),
            static_cast<std::streamsize>(mipDataSize));
    return f.good();
}

namespace {
std::uint32_t imageWord(const std::uint8_t* p, unsigned n) {
    std::uint32_t v = 0;
    for (unsigned i = 0; i < n; ++i) v |= std::uint32_t(p[i]) << (i * 8);
    return v;
}

bool decodeLegacyImage(const std::uint8_t* data, std::size_t size,
                       const std::string& extension, ImportedSceneTexture& tex) {
    if (!data) return false;
    std::size_t offset = 0, stride = 0;
    unsigned depth = 0;
    bool top = true, right = false, rle = false;
    if (extension == ".tga") {
        if (size < 18 || data[1] != 0 || (data[2] != 2 && data[2] != 10) ||
            (data[16] != 24 && data[16] != 32) || (data[17] & 0xc0)) return false;
        tex.width = imageWord(data + 12, 2); tex.height = imageWord(data + 14, 2);
        depth = data[16] / 8; offset = 18u + data[0];
        top = (data[17] & 0x20) != 0; right = (data[17] & 0x10) != 0;
        rle = data[2] == 10;
    } else if (extension == ".bmp") {
        if (size < 54 || data[0] != 'B' || data[1] != 'M' ||
            imageWord(data + 14, 4) < 40 || imageWord(data + 26, 2) != 1 ||
            imageWord(data + 30, 4) != 0 ||
            (imageWord(data + 28, 2) != 24 && imageWord(data + 28, 2) != 32)) return false;
        const auto headerSize = imageWord(data + 14, 4);
        const auto w = std::bit_cast<std::int32_t>(imageWord(data + 18, 4));
        const auto h = std::bit_cast<std::int32_t>(imageWord(data + 22, 4));
        if (w <= 0 || h == 0 || h == INT32_MIN) return false;
        tex.width = std::uint32_t(w); tex.height = std::uint32_t(h < 0 ? -h : h);
        depth = imageWord(data + 28, 2) / 8; offset = imageWord(data + 10, 4);
        if (headerSize > size - 14 || offset < 14u + headerSize) return false;
        top = h < 0;
        stride = (std::size_t(tex.width) * depth + 3) & ~std::size_t(3);
    } else return false;
    // Bound allocation before touching payload. 16K covers supported Vulkan
    // dimensions; image bytes are bounded too, including malicious RLE headers.
    if (!tex.width || !tex.height || tex.width > 16384 || tex.height > 16384 || offset > size) return false;
    const std::size_t pixels = std::size_t(tex.width) * tex.height;
    if (pixels > (512ull * 1024 * 1024) / 4) return false;
    if (!rle && ((stride ? stride * tex.height : pixels * depth) > size - offset)) return false;
    // Each RLE packet encodes at most 128 pixels with at least one header and
    // one pixel; this bounds output allocation against truncated tiny inputs.
    if (rle && (pixels + 127) / 128 > (size - offset) / (depth + 1)) return false;
    tex.rgba8.resize(pixels * 4);
    const auto put = [&](std::size_t index, const std::uint8_t* p) {
        const auto sx = index % tex.width, sy = index / tex.width;
        const auto x = right ? tex.width - 1 - sx : sx;
        const auto y = top ? sy : tex.height - 1 - sy;
        const auto dst = (y * tex.width + x) * 4;
        tex.rgba8[dst] = p[2]; tex.rgba8[dst + 1] = p[1]; tex.rgba8[dst + 2] = p[0];
        tex.rgba8[dst + 3] = depth == 4 ? p[3] : 255;
    };
    if (!rle) {
        for (std::size_t y = 0; y < tex.height; ++y)
            for (std::size_t x = 0; x < tex.width; ++x)
                put(y * tex.width + x, data + offset + y * (stride ? stride : tex.width * depth) + x * depth);
    } else {
        std::size_t written = 0;
        while (written < pixels) {
            if (offset >= size) return false;
            const auto packet = data[offset++];
            const auto count = std::size_t((packet & 127) + 1);
            if (count > pixels - written) return false;
            const auto bytes = (packet & 128) ? depth : count * depth;
            if (bytes > size - offset) return false;
            for (std::size_t i = 0; i < count; ++i) put(written++, data + offset + ((packet & 128) ? 0 : i * depth));
            offset += bytes;
        }
    }
    tex.format = TextureFormat::RGBA8Srgb;
    return true;
}

// Expand color BC images only when a missing chain needs generation or a
// requested ceiling cannot be reached. Authored compressed chains stay intact.
bool expandBcColor(ImportedSceneTexture& texture) {
    using Decode = void (*)(const void*, void*, int);
    Decode decode = nullptr;
    switch (texture.format) {
        case TextureFormat::BC1: case TextureFormat::BC1Linear: decode = bcdec_bc1; break;
        case TextureFormat::BC2: decode = bcdec_bc2; break;
        case TextureFormat::BC3: decode = bcdec_bc3; break;
        case TextureFormat::BC7: decode = bcdec_bc7; break;
        default: return false;
    }
    if (texture.arrayLayers != 1 || std::size_t(texture.width) * texture.height > (512ull * 1024 * 1024) / 4) return false;
    const auto blockBytes = ddsBlockBytes(texture.format);
    std::vector<std::uint8_t> rgba(std::size_t(texture.width) * texture.height * 4);
    for (unsigned by=0; by<(texture.height+3)/4; ++by) for (unsigned bx=0; bx<(texture.width+3)/4; ++bx) {
        alignas(8) std::array<std::uint8_t,64> block{};
        decode(texture.rgba8.data() + (std::size_t(by) * ((texture.width+3)/4) + bx) * blockBytes, block.data(), 16);
        for (unsigned y=0; y<4 && by*4+y<texture.height; ++y)
            for (unsigned x=0; x<4 && bx*4+x<texture.width; ++x)
                std::memcpy(rgba.data() + (std::size_t(by*4+y)*texture.width+bx*4+x)*4, block.data()+(y*4+x)*4,4);
    }
    texture.rgba8 = std::move(rgba); texture.mipLevelCount = 1;
    texture.format = TextureFormat::RGBA8Srgb;
    return true;
}

// Generate color mips in linear light; averaging encoded sRGB darkens distant
// textures. Alpha is averaged independently, preserving authored coverage.
void completeColorMips(ImportedSceneTexture& tex) {
    if (tex.arrayLayers != 1 || (tex.format != TextureFormat::RGBA8 && tex.format != TextureFormat::RGBA8Srgb)) return;
    std::uint32_t w = tex.width, h = tex.height;
    std::size_t offset = 0;
    for (unsigned mip = 1; mip < tex.mipLevelCount; ++mip) {
        offset += std::size_t(w) * h * 4; w = std::max(1u, w / 2); h = std::max(1u, h / 2);
    }
    while (w > 1 || h > 1) {
        const auto nw = std::max(1u, w / 2), nh = std::max(1u, h / 2);
        const auto next = tex.rgba8.size(); tex.rgba8.resize(next + std::size_t(nw) * nh * 4);
        for (unsigned y = 0; y < nh; ++y) for (unsigned x = 0; x < nw; ++x) {
            // Area bounds include the trailing row/column of odd-sized inputs.
            const auto x0 = x * w / nw, x1 = (x + 1) * w / nw;
            const auto y0 = y * h / nh, y1 = (y + 1) * h / nh;
            for (unsigned c = 0; c < 4; ++c) {
                float sum = 0;
                for (unsigned sy = y0; sy < y1; ++sy) for (unsigned sx = x0; sx < x1; ++sx) {
                    float value = tex.rgba8[offset + (std::size_t(sy) * w + sx) * 4 + c] / 255.f;
                    if (c < 3 && tex.format == TextureFormat::RGBA8Srgb)
                        value = value <= .04045f ? value / 12.92f : std::pow((value + .055f) / 1.055f, 2.4f);
                    sum += value;
                }
                float value = sum / float((x1 - x0) * (y1 - y0));
                if (c < 3 && tex.format == TextureFormat::RGBA8Srgb)
                    value = value <= .0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.f / 2.4f) - .055f;
                tex.rgba8[next + (std::size_t(y) * nw + x) * 4 + c] = std::uint8_t(std::clamp(std::lround(value * 255), 0l, 255l));
            }
        }
        offset = next; w = nw; h = nh; ++tex.mipLevelCount;
    }
}
} // namespace

bool loadTextureFromMemory(const std::uint8_t* bytes, std::size_t size,
                           const std::string& path, ImportedSceneTexture& out,
                           std::uint32_t maxDimension, std::string& error, bool linearData) {
    ImportedSceneTexture texture;
    const auto dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    const bool ok = ext == ".dds" ? loadDdsFromMemory(bytes, size, texture) : decodeLegacyImage(bytes, size, ext, texture);
    if (!ok) { error = "invalid or unsupported " + ext + " texture: " + path; return false; }
    if (!linearData && texture.mipLevelCount == 1 && (texture.width > 1 || texture.height > 1))
        expandBcColor(texture);
    if (texture.format == TextureFormat::RGBA8 || texture.format == TextureFormat::RGBA8Srgb)
        texture.format = linearData ? TextureFormat::RGBA8 : TextureFormat::RGBA8Srgb;
    completeColorMips(texture);
    dropDdsMipLevels(texture, maxDimension);
    if (maxDimension && (texture.width > maxDimension || texture.height > maxDimension) &&
        !linearData && expandBcColor(texture)) {
        completeColorMips(texture);
        dropDdsMipLevels(texture, maxDimension);
    }
    if (maxDimension && (texture.width > maxDimension || texture.height > maxDimension)) {
        error = "texture has no usable mip for requested ceiling " + std::to_string(maxDimension) + ": " + path;
        return false;
    }
    texture.sourcePath = path;
    out = std::move(texture); error.clear(); return true;
}

} // namespace odai::importer
