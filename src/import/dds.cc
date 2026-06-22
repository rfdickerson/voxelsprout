#include "import/dds.h"

#include <cstring>
#include <fstream>
#include <vector>

namespace odai::importer {

namespace {

constexpr std::uint32_t kDdsMagic      = 0x20534444u; // "DDS "
constexpr std::uint32_t kDdpfFourCC    = 0x4u;
constexpr std::uint32_t kDdsdCaps      = 0x1u;
constexpr std::uint32_t kDdsdHeight    = 0x2u;
constexpr std::uint32_t kDdsdWidth     = 0x4u;
constexpr std::uint32_t kDdsdPixelFmt  = 0x1000u;
constexpr std::uint32_t kDdsdMipmapCnt = 0x20000u;
constexpr std::uint32_t kDdsCaps1Tex   = 0x1000u;
constexpr std::uint32_t kDdsCaps1Mip   = 0x400000u;
constexpr std::uint32_t kDdsCaps1Cmplx = 0x8u;

constexpr std::uint32_t kFourCCDxt1    = 0x31545844u; // "DXT1"
constexpr std::uint32_t kFourCCDxt5    = 0x35545844u; // "DXT5"
constexpr std::uint32_t kFourCCAti1    = 0x31495441u; // "ATI1"
constexpr std::uint32_t kFourCCAti2    = 0x32495441u; // "ATI2"
constexpr std::uint32_t kFourCCDx10    = 0x30315844u; // "DX10"

constexpr std::uint32_t kDxgiBC1Unorm  = 71u;
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
        case TextureFormat::BC4: return 8u;
        case TextureFormat::BC3:
        case TextureFormat::BC5:
        case TextureFormat::BC7: return 16u;
        default:                 return 0u;
    }
}

bool loadDds(const std::filesystem::path& path, ImportedSceneTexture& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto fileSize = static_cast<std::size_t>(f.tellg());
    if (fileSize < 4u + sizeof(DdsHeader)) return false;
    f.seekg(0);

    std::vector<std::uint8_t> data(fileSize);
    if (!f.read(reinterpret_cast<char*>(data.data()),
                static_cast<std::streamsize>(fileSize))) return false;

    std::uint32_t magic = 0;
    std::memcpy(&magic, data.data(), 4u);
    if (magic != kDdsMagic) return false;

    DdsHeader hdr{};
    std::memcpy(&hdr, data.data() + 4u, sizeof(DdsHeader));
    if (hdr.size != 124u || hdr.width == 0u || hdr.height == 0u) return false;
    if (!(hdr.ddspf.flags & kDdpfFourCC)) return false; // uncompressed not supported

    TextureFormat  fmt        = TextureFormat::RGBA8;
    std::size_t    dataOffset = 4u + sizeof(DdsHeader);

    const std::uint32_t fcc = hdr.ddspf.fourCC;
    if      (fcc == kFourCCDxt1) { fmt = TextureFormat::BC1; }
    else if (fcc == kFourCCDxt5) { fmt = TextureFormat::BC3; }
    else if (fcc == kFourCCAti1) { fmt = TextureFormat::BC4; }
    else if (fcc == kFourCCAti2) { fmt = TextureFormat::BC5; }
    else if (fcc == kFourCCDx10) {
        if (fileSize < dataOffset + sizeof(DdsHeaderDxt10)) return false;
        DdsHeaderDxt10 dx10{};
        std::memcpy(&dx10, data.data() + dataOffset, sizeof(DdsHeaderDxt10));
        dataOffset += sizeof(DdsHeaderDxt10);
        switch (dx10.dxgiFormat) {
            case kDxgiBC1Unorm:                     fmt = TextureFormat::BC1; break;
            case kDxgiBC3Unorm:                     fmt = TextureFormat::BC3; break;
            case kDxgiBC4Unorm:                     fmt = TextureFormat::BC4; break;
            case kDxgiBC5Unorm:                     fmt = TextureFormat::BC5; break;
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
    if (fileSize < dataOffset + chainBytes) return false;

    out.sourcePath    = path.string();
    out.width         = hdr.width;
    out.height        = hdr.height;
    out.mipLevelCount = mipCount;
    out.format        = fmt;
    out.rgba8.assign(data.begin() + static_cast<std::ptrdiff_t>(dataOffset),
                     data.begin() + static_cast<std::ptrdiff_t>(dataOffset + chainBytes));
    return true;
}

bool writeDds(const std::filesystem::path& path,
              std::uint32_t width, std::uint32_t height, std::uint32_t mipLevelCount,
              TextureFormat format,
              const std::uint8_t* mipData, std::size_t mipDataSize) {
    const std::uint32_t bpb = ddsBlockBytes(format);
    if (bpb == 0u || mipData == nullptr || mipDataSize == 0u) return false;

    const bool needDx10 = (format == TextureFormat::BC7);
    std::uint32_t fourCC = 0, dxgiFmt = 0;
    if (needDx10) {
        fourCC  = kFourCCDx10;
        dxgiFmt = kDxgiBC7Unorm;
    } else {
        switch (format) {
            case TextureFormat::BC1: fourCC = kFourCCDxt1; break;
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

} // namespace odai::importer
