#include "import/imported_scene.h"
#include "import/morrowind_nif.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace odai::importer {

namespace {

constexpr std::uint32_t kImportedSceneMagic = 0x4E435356u;  // VSCN
constexpr std::uint32_t kImportedSceneVersion = 16u;
constexpr std::uint32_t kMinSupportedImportedSceneVersion = 15u;
constexpr float kExteriorWaterLevel = 0.0f;
constexpr int kLandSize = 65;
constexpr int kLandTextureSize = 16;
constexpr float kCellSizeUnits = 8192.0f;
constexpr float kHeightScale = 8.0f;
constexpr std::uint32_t kImportedSceneMaterialFlagAlphaTest = 1u;
constexpr float kRiverAuditMaxDistance = 1024.0f;
constexpr std::uint32_t kTerrainCompositeTextureSize = 1024u;
constexpr std::uint32_t kTes3LightFlagNegative = 0x004u;
constexpr std::uint32_t kTes3LightFlagOffDefault = 0x020u;

std::string g_lastImportedSceneError;

void setLastImportedSceneError(std::string message) {
    g_lastImportedSceneError = std::move(message);
}

std::optional<std::string> getEnvironmentVariable(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t length = 0u;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr || length == 0u) {
        if (value != nullptr) {
            std::free(value);
        }
        return std::nullopt;
    }
    std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        if (c == '\\') {
            return '/';
        }
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string trimNullTerminated(std::string_view bytes) {
    const std::size_t nullEnd = bytes.find('\0');
    std::string value(bytes.substr(0, nullEnd == std::string_view::npos ? bytes.size() : nullEnd));
    while (!value.empty()) {
        const unsigned char ch = static_cast<unsigned char>(value.back());
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
            value.pop_back();
            continue;
        }
        break;
    }
    return value;
}

std::vector<std::string> splitCommaSeparatedList(std::string value) {
    std::vector<std::string> result;
    std::size_t start = 0u;
    while (start < value.size()) {
        std::size_t end = value.find(',', start);
        if (end == std::string::npos) {
            end = value.size();
        }
        std::string item = value.substr(start, end - start);
        item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char c) { return !std::isspace(c); }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char c) { return !std::isspace(c); }).base(), item.end());
        if (!item.empty()) {
            result.push_back(lowerCopy(std::move(item)));
        }
        start = end + 1u;
    }
    return result;
}

struct DebugBounds {
    std::array<float, 3> min{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    std::array<float, 3> max{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    };
    bool valid = false;
};

void expandBounds(DebugBounds& bounds, const std::array<float, 3>& point) {
    bounds.valid = true;
    for (int axis = 0; axis < 3; ++axis) {
        bounds.min[axis] = std::min(bounds.min[axis], point[axis]);
        bounds.max[axis] = std::max(bounds.max[axis], point[axis]);
    }
}

DebugBounds computeMeshBounds(const ImportedSceneMesh& mesh) {
    DebugBounds bounds{};
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        expandBounds(bounds, {vertex.position[0], vertex.position[1], vertex.position[2]});
    }
    return bounds;
}

DebugBounds computeTransformedMeshBounds(
    const ImportedSceneMesh& mesh,
    const std::array<float, 16>& transform
) {
    DebugBounds bounds{};
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        const std::array<float, 3> point{
            (transform[0] * vertex.position[0]) + (transform[1] * vertex.position[1]) + (transform[2] * vertex.position[2]) + transform[3],
            (transform[4] * vertex.position[0]) + (transform[5] * vertex.position[1]) + (transform[6] * vertex.position[2]) + transform[7],
            (transform[8] * vertex.position[0]) + (transform[9] * vertex.position[1]) + (transform[10] * vertex.position[2]) + transform[11]
        };
        expandBounds(bounds, point);
    }
    return bounds;
}

float distanceBoundsToWaterPatchXz(const DebugBounds& bounds, const ImportedSceneWaterPatch& patch) {
    const float patchMaxX = patch.originX + patch.sizeX;
    const float patchMaxZ = patch.originZ + patch.sizeZ;
    float dx = 0.0f;
    if (bounds.max[0] < patch.originX) {
        dx = patch.originX - bounds.max[0];
    } else if (bounds.min[0] > patchMaxX) {
        dx = bounds.min[0] - patchMaxX;
    }
    float dz = 0.0f;
    if (bounds.max[2] < patch.originZ) {
        dz = patch.originZ - bounds.max[2];
    } else if (bounds.min[2] > patchMaxZ) {
        dz = bounds.min[2] - patchMaxZ;
    }
    return std::sqrt((dx * dx) + (dz * dz));
}

bool modelPathMatchesDebugFilters(
    std::string_view modelPath,
    const std::vector<std::string>& filters
) {
    if (filters.empty()) {
        return false;
    }
    const std::string normalizedPath = lowerCopy(std::string(modelPath));
    for (const std::string& filter : filters) {
        if (normalizedPath.find(filter) != std::string::npos) {
            return true;
        }
    }
    return false;
}

struct PackedRenderColor {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

PackedRenderColor packedRenderColorFromHash(std::string_view key) {
    std::uint32_t hash = 2166136261u;
    for (const char ch : key) {
        hash ^= static_cast<std::uint8_t>(ch);
        hash *= 16777619u;
    }
    PackedRenderColor color{};
    color.r = 0.30f + (static_cast<float>((hash >> 0) & 0xffu) / 255.0f) * 0.55f;
    color.g = 0.28f + (static_cast<float>((hash >> 8) & 0xffu) / 255.0f) * 0.52f;
    color.b = 0.25f + (static_cast<float>((hash >> 16) & 0xffu) / 255.0f) * 0.50f;
    return color;
}

PackedRenderColor packedTerrainColor(float height) {
    PackedRenderColor color{};
    const float normalized = std::clamp((height + 256.0f) / 1024.0f, 0.0f, 1.0f);
    color.r = 0.22f + (normalized * 0.32f);
    color.g = 0.24f + (normalized * 0.34f);
    color.b = 0.18f + (normalized * 0.14f);
    return color;
}

struct DdsPixelFormat {
    std::uint32_t size = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t fourCc = 0u;
    std::uint32_t rgbBitCount = 0u;
    std::uint32_t rMask = 0u;
    std::uint32_t gMask = 0u;
    std::uint32_t bMask = 0u;
    std::uint32_t aMask = 0u;
};

struct DdsHeader {
    std::uint32_t size = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t height = 0u;
    std::uint32_t width = 0u;
    std::uint32_t pitchOrLinearSize = 0u;
    std::uint32_t depth = 0u;
    std::uint32_t mipMapCount = 0u;
    std::uint32_t reserved1[11] = {};
    DdsPixelFormat pixelFormat{};
    std::uint32_t caps = 0u;
    std::uint32_t caps2 = 0u;
    std::uint32_t caps3 = 0u;
    std::uint32_t caps4 = 0u;
    std::uint32_t reserved2 = 0u;
};

bool readFileBytes(const std::filesystem::path& path, std::vector<std::uint8_t>& outBytes) {
    outBytes.clear();
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return false;
    }
    const std::streamsize fileSize = stream.tellg();
    if (fileSize <= 0) {
        return false;
    }
    stream.seekg(0, std::ios::beg);
    outBytes.resize(static_cast<std::size_t>(fileSize));
    return stream.read(reinterpret_cast<char*>(outBytes.data()), fileSize).good();
}

std::uint8_t decodeDdsChannel(std::uint32_t packedPixel, std::uint32_t mask, std::uint8_t defaultValue) {
    if (mask == 0u) {
        return defaultValue;
    }
    const std::uint32_t shift = std::countr_zero(mask);
    const std::uint32_t shiftedMask = mask >> shift;
    const std::uint32_t componentBits = std::popcount(shiftedMask);
    if (componentBits == 0u) {
        return defaultValue;
    }
    const std::uint32_t componentValue = (packedPixel & mask) >> shift;
    const std::uint32_t componentMax = (1u << componentBits) - 1u;
    if (componentMax == 0u) {
        return defaultValue;
    }
    return static_cast<std::uint8_t>((componentValue * 255u + (componentMax / 2u)) / componentMax);
}

std::uint8_t expandFiveBitColor(std::uint16_t value) {
    return static_cast<std::uint8_t>((static_cast<std::uint32_t>(value) * 255u + 15u) / 31u);
}

std::uint8_t expandSixBitColor(std::uint16_t value) {
    return static_cast<std::uint8_t>((static_cast<std::uint32_t>(value) * 255u + 31u) / 63u);
}

void decodeDxtColorBlock(
    const std::uint8_t* blockBytes,
    std::uint8_t* rgba,
    std::uint32_t imageWidth,
    std::uint32_t imageHeight,
    std::uint32_t blockX,
    std::uint32_t blockY,
    bool allowTransparentFourthEntry
) {
    std::uint16_t color0 = 0u;
    std::uint16_t color1 = 0u;
    std::memcpy(&color0, blockBytes + 0, sizeof(color0));
    std::memcpy(&color1, blockBytes + 2, sizeof(color1));
    const std::uint8_t colors[4][4] = {
        {expandFiveBitColor(static_cast<std::uint16_t>((color0 >> 11) & 0x1fu)),
         expandSixBitColor(static_cast<std::uint16_t>((color0 >> 5) & 0x3fu)),
         expandFiveBitColor(static_cast<std::uint16_t>(color0 & 0x1fu)),
         255u},
        {expandFiveBitColor(static_cast<std::uint16_t>((color1 >> 11) & 0x1fu)),
         expandSixBitColor(static_cast<std::uint16_t>((color1 >> 5) & 0x3fu)),
         expandFiveBitColor(static_cast<std::uint16_t>(color1 & 0x1fu)),
         255u},
        {0u, 0u, 0u, 255u},
        {0u, 0u, 0u, 255u},
    };
    std::array<std::array<std::uint8_t, 4>, 4> palette{};
    for (std::size_t i = 0; i < 4; ++i) {
        palette[0][i] = colors[0][i];
        palette[1][i] = colors[1][i];
    }
    if (!allowTransparentFourthEntry || color0 > color1) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            palette[2][channel] = static_cast<std::uint8_t>((2u * palette[0][channel] + palette[1][channel]) / 3u);
            palette[3][channel] = static_cast<std::uint8_t>((palette[0][channel] + 2u * palette[1][channel]) / 3u);
        }
        palette[2][3] = 255u;
        palette[3][3] = 255u;
    } else {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            palette[2][channel] = static_cast<std::uint8_t>((palette[0][channel] + palette[1][channel]) / 2u);
            palette[3][channel] = 0u;
        }
        palette[2][3] = 255u;
        palette[3][3] = 0u;
    }

    std::uint32_t indices = 0u;
    std::memcpy(&indices, blockBytes + 4, sizeof(indices));
    for (std::uint32_t pixelY = 0; pixelY < 4u; ++pixelY) {
        for (std::uint32_t pixelX = 0; pixelX < 4u; ++pixelX) {
            const std::uint32_t dstX = blockX * 4u + pixelX;
            const std::uint32_t dstY = blockY * 4u + pixelY;
            if (dstX >= imageWidth || dstY >= imageHeight) {
                indices >>= 2u;
                continue;
            }
            const std::uint32_t paletteIndex = indices & 0x3u;
            indices >>= 2u;
            const std::size_t dstOffset =
                (static_cast<std::size_t>(dstY) * static_cast<std::size_t>(imageWidth) + dstX) * 4u;
            rgba[dstOffset + 0u] = palette[paletteIndex][0];
            rgba[dstOffset + 1u] = palette[paletteIndex][1];
            rgba[dstOffset + 2u] = palette[paletteIndex][2];
            rgba[dstOffset + 3u] = palette[paletteIndex][3];
        }
    }
}

void decodeDxt5BlockAlpha(const std::uint8_t* blockBytes, std::uint8_t alphaValues[16]) {
    const std::uint8_t alpha0 = blockBytes[0];
    const std::uint8_t alpha1 = blockBytes[1];
    std::uint8_t alphaPalette[8]{};
    alphaPalette[0] = alpha0;
    alphaPalette[1] = alpha1;
    if (alpha0 > alpha1) {
        for (std::uint32_t i = 1u; i <= 6u; ++i) {
            alphaPalette[i + 1u] = static_cast<std::uint8_t>(
                (((7u - i) * static_cast<std::uint32_t>(alpha0)) + (i * static_cast<std::uint32_t>(alpha1))) / 7u);
        }
    } else {
        for (std::uint32_t i = 1u; i <= 4u; ++i) {
            alphaPalette[i + 1u] = static_cast<std::uint8_t>(
                (((5u - i) * static_cast<std::uint32_t>(alpha0)) + (i * static_cast<std::uint32_t>(alpha1))) / 5u);
        }
        alphaPalette[6] = 0u;
        alphaPalette[7] = 255u;
    }

    std::uint64_t alphaBits = 0u;
    for (std::uint32_t i = 0; i < 6u; ++i) {
        alphaBits |= static_cast<std::uint64_t>(blockBytes[2u + i]) << (8u * i);
    }
    for (std::uint32_t i = 0; i < 16u; ++i) {
        const std::uint32_t alphaIndex = static_cast<std::uint32_t>((alphaBits >> (3u * i)) & 0x7u);
        alphaValues[i] = alphaPalette[alphaIndex];
    }
}

bool decodeDdsRgba(
    const std::filesystem::path& path,
    std::uint32_t& outWidth,
    std::uint32_t& outHeight,
    std::vector<std::uint8_t>& outRgba8
) {
    outWidth = 0u;
    outHeight = 0u;
    outRgba8.clear();

    std::vector<std::uint8_t> bytes;
    if (!readFileBytes(path, bytes) || bytes.size() < (4u + sizeof(DdsHeader))) {
        return false;
    }
    if (std::memcmp(bytes.data(), "DDS ", 4) != 0) {
        return false;
    }

    DdsHeader header{};
    std::memcpy(&header, bytes.data() + 4u, sizeof(header));
    if (header.size != 124u || header.pixelFormat.size != 32u || header.width == 0u || header.height == 0u) {
        return false;
    }
    const std::size_t dataOffset = 4u + sizeof(DdsHeader);
    if (bytes.size() <= dataOffset) {
        return false;
    }

    outWidth = header.width;
    outHeight = header.height;
    outRgba8.assign(static_cast<std::size_t>(outWidth) * static_cast<std::size_t>(outHeight) * 4u, 255u);
    const std::uint32_t fourCc = header.pixelFormat.fourCc;
    const auto* pixelData = bytes.data() + dataOffset;
    const std::size_t payloadSize = bytes.size() - dataOffset;

    if (fourCc == 0u && header.pixelFormat.rgbBitCount == 32u) {
        const std::size_t pixelCount = static_cast<std::size_t>(outWidth) * static_cast<std::size_t>(outHeight);
        if (payloadSize < pixelCount * sizeof(std::uint32_t)) {
            return false;
        }
        for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
            std::uint32_t packedPixel = 0u;
            std::memcpy(&packedPixel, pixelData + pixelIndex * sizeof(std::uint32_t), sizeof(std::uint32_t));
            outRgba8[pixelIndex * 4u + 0u] =
                decodeDdsChannel(packedPixel, header.pixelFormat.rMask, 0u);
            outRgba8[pixelIndex * 4u + 1u] =
                decodeDdsChannel(packedPixel, header.pixelFormat.gMask, 0u);
            outRgba8[pixelIndex * 4u + 2u] =
                decodeDdsChannel(packedPixel, header.pixelFormat.bMask, 0u);
            outRgba8[pixelIndex * 4u + 3u] =
                decodeDdsChannel(packedPixel, header.pixelFormat.aMask, 255u);
        }
        return true;
    }

    const std::uint32_t blockWidth = std::max(1u, (outWidth + 3u) / 4u);
    const std::uint32_t blockHeight = std::max(1u, (outHeight + 3u) / 4u);
    if (fourCc == 0x31545844u) {  // DXT1
        const std::size_t requiredBytes = static_cast<std::size_t>(blockWidth) * static_cast<std::size_t>(blockHeight) * 8u;
        if (payloadSize < requiredBytes) {
            return false;
        }
        for (std::uint32_t y = 0; y < blockHeight; ++y) {
            for (std::uint32_t x = 0; x < blockWidth; ++x) {
                const std::size_t blockOffset = (static_cast<std::size_t>(y) * blockWidth + x) * 8u;
                decodeDxtColorBlock(pixelData + blockOffset, outRgba8.data(), outWidth, outHeight, x, y, true);
            }
        }
        return true;
    }
    if (fourCc == 0x35545844u) {  // DXT5
        const std::size_t requiredBytes = static_cast<std::size_t>(blockWidth) * static_cast<std::size_t>(blockHeight) * 16u;
        if (payloadSize < requiredBytes) {
            return false;
        }
        for (std::uint32_t y = 0; y < blockHeight; ++y) {
            for (std::uint32_t x = 0; x < blockWidth; ++x) {
                const std::size_t blockOffset = (static_cast<std::size_t>(y) * blockWidth + x) * 16u;
                std::uint8_t alphaValues[16]{};
                decodeDxt5BlockAlpha(pixelData + blockOffset, alphaValues);
                decodeDxtColorBlock(pixelData + blockOffset + 8u, outRgba8.data(), outWidth, outHeight, x, y, false);
                for (std::uint32_t pixelY = 0; pixelY < 4u; ++pixelY) {
                    for (std::uint32_t pixelX = 0; pixelX < 4u; ++pixelX) {
                        const std::uint32_t dstX = x * 4u + pixelX;
                        const std::uint32_t dstY = y * 4u + pixelY;
                        if (dstX >= outWidth || dstY >= outHeight) {
                            continue;
                        }
                        const std::size_t dstOffset =
                            (static_cast<std::size_t>(dstY) * static_cast<std::size_t>(outWidth) + dstX) * 4u;
                        outRgba8[dstOffset + 3u] = alphaValues[pixelY * 4u + pixelX];
                    }
                }
            }
        }
        return true;
    }

    return false;
}

bool decodeTgaRgba(
    const std::filesystem::path& path,
    std::uint32_t& outWidth,
    std::uint32_t& outHeight,
    std::vector<std::uint8_t>& outRgba8
) {
    outWidth = 0u;
    outHeight = 0u;
    outRgba8.clear();

    std::vector<std::uint8_t> bytes;
    if (!readFileBytes(path, bytes) || bytes.size() < 18u) {
        return false;
    }

    const std::uint8_t idLength = bytes[0];
    const std::uint8_t imageType = bytes[2];
    const std::uint16_t width = static_cast<std::uint16_t>(bytes[12] | (bytes[13] << 8u));
    const std::uint16_t height = static_cast<std::uint16_t>(bytes[14] | (bytes[15] << 8u));
    const std::uint8_t pixelDepth = bytes[16];
    const std::uint8_t imageDescriptor = bytes[17];
    const bool topOrigin = (imageDescriptor & 0x20u) != 0u;
    const bool leftOrigin = (imageDescriptor & 0x10u) == 0u;
    if (width == 0u || height == 0u) {
        return false;
    }
    if ((imageType != 2u && imageType != 10u) || (pixelDepth != 24u && pixelDepth != 32u)) {
        return false;
    }

    const std::size_t bytesPerPixel = static_cast<std::size_t>(pixelDepth / 8u);
    const std::size_t pixelCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t dataOffset = 18u + idLength;
    if (dataOffset > bytes.size()) {
        return false;
    }

    std::vector<std::uint8_t> decoded(pixelCount * bytesPerPixel);
    const std::uint8_t* src = bytes.data() + dataOffset;
    const std::uint8_t* const srcEnd = bytes.data() + bytes.size();
    if (imageType == 2u) {
        const std::size_t requiredBytes = pixelCount * bytesPerPixel;
        if (static_cast<std::size_t>(srcEnd - src) < requiredBytes) {
            return false;
        }
        std::memcpy(decoded.data(), src, requiredBytes);
    } else {
        std::size_t dstOffset = 0u;
        while (dstOffset < decoded.size() && src < srcEnd) {
            const std::uint8_t packetHeader = *src++;
            const std::size_t packetCount = static_cast<std::size_t>((packetHeader & 0x7fu) + 1u);
            if ((packetHeader & 0x80u) != 0u) {
                if (static_cast<std::size_t>(srcEnd - src) < bytesPerPixel) {
                    return false;
                }
                for (std::size_t i = 0; i < packetCount; ++i) {
                    if (dstOffset + bytesPerPixel > decoded.size()) {
                        return false;
                    }
                    std::memcpy(decoded.data() + dstOffset, src, bytesPerPixel);
                    dstOffset += bytesPerPixel;
                }
                src += bytesPerPixel;
            } else {
                const std::size_t packetBytes = packetCount * bytesPerPixel;
                if (static_cast<std::size_t>(srcEnd - src) < packetBytes || dstOffset + packetBytes > decoded.size()) {
                    return false;
                }
                std::memcpy(decoded.data() + dstOffset, src, packetBytes);
                dstOffset += packetBytes;
                src += packetBytes;
            }
        }
        if (dstOffset != decoded.size()) {
            return false;
        }
    }

    outWidth = width;
    outHeight = height;
    outRgba8.resize(pixelCount * 4u);
    for (std::uint32_t y = 0; y < outHeight; ++y) {
        const std::uint32_t srcY = topOrigin ? y : (outHeight - 1u - y);
        for (std::uint32_t x = 0; x < outWidth; ++x) {
            const std::uint32_t srcX = leftOrigin ? x : (outWidth - 1u - x);
            const std::size_t srcOffset =
                (static_cast<std::size_t>(srcY) * static_cast<std::size_t>(outWidth) + srcX) * bytesPerPixel;
            const std::size_t dstOffset =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(outWidth) + x) * 4u;
            outRgba8[dstOffset + 0u] = decoded[srcOffset + 2u];
            outRgba8[dstOffset + 1u] = decoded[srcOffset + 1u];
            outRgba8[dstOffset + 2u] = decoded[srcOffset + 0u];
            outRgba8[dstOffset + 3u] = (bytesPerPixel == 4u) ? decoded[srcOffset + 3u] : 255u;
        }
    }
    return true;
}

std::uint16_t readLe16(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1u] << 8u));
}

std::uint32_t readLe32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
        (static_cast<std::uint32_t>(bytes[offset + 1u]) << 8u) |
        (static_cast<std::uint32_t>(bytes[offset + 2u]) << 16u) |
        (static_cast<std::uint32_t>(bytes[offset + 3u]) << 24u);
}

std::int32_t readLeSigned32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    const std::uint32_t value = readLe32(bytes, offset);
    std::int32_t signedValue = 0;
    std::memcpy(&signedValue, &value, sizeof(signedValue));
    return signedValue;
}

bool decodeBmpRgba(
    const std::filesystem::path& path,
    std::uint32_t& outWidth,
    std::uint32_t& outHeight,
    std::vector<std::uint8_t>& outRgba8
) {
    outWidth = 0u;
    outHeight = 0u;
    outRgba8.clear();

    std::vector<std::uint8_t> bytes;
    if (!readFileBytes(path, bytes) || bytes.size() < 54u) {
        return false;
    }
    if (bytes[0] != 'B' || bytes[1] != 'M') {
        return false;
    }

    const std::uint32_t pixelDataOffset = readLe32(bytes, 10u);
    const std::uint32_t dibHeaderSize = readLe32(bytes, 14u);
    if (dibHeaderSize < 40u || pixelDataOffset >= bytes.size()) {
        return false;
    }

    const std::int32_t signedWidth = readLeSigned32(bytes, 18u);
    const std::int32_t signedHeight = readLeSigned32(bytes, 22u);
    const std::uint16_t planes = readLe16(bytes, 26u);
    const std::uint16_t bitsPerPixel = readLe16(bytes, 28u);
    const std::uint32_t compression = readLe32(bytes, 30u);
    if (signedWidth <= 0 || signedHeight == 0 ||
        signedHeight == std::numeric_limits<std::int32_t>::min() ||
        planes != 1u || compression != 0u) {
        return false;
    }
    if (bitsPerPixel != 24u && bitsPerPixel != 32u) {
        return false;
    }

    const std::uint32_t width = static_cast<std::uint32_t>(signedWidth);
    const std::uint32_t height = static_cast<std::uint32_t>(signedHeight < 0 ? -signedHeight : signedHeight);
    const bool topDown = signedHeight < 0;
    const std::size_t bytesPerPixel = static_cast<std::size_t>(bitsPerPixel / 8u);
    const std::size_t rowStride =
        ((static_cast<std::size_t>(width) * bitsPerPixel + 31u) / 32u) * 4u;
    const std::size_t requiredBytes = pixelDataOffset + rowStride * static_cast<std::size_t>(height);
    if (width == 0u || height == 0u || requiredBytes > bytes.size()) {
        return false;
    }

    outWidth = width;
    outHeight = height;
    outRgba8.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u);
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint32_t srcY = topDown ? y : (height - 1u - y);
        const std::size_t rowOffset = pixelDataOffset + static_cast<std::size_t>(srcY) * rowStride;
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t srcOffset = rowOffset + static_cast<std::size_t>(x) * bytesPerPixel;
            const std::size_t dstOffset =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + x) * 4u;
            outRgba8[dstOffset + 0u] = bytes[srcOffset + 2u];
            outRgba8[dstOffset + 1u] = bytes[srcOffset + 1u];
            outRgba8[dstOffset + 2u] = bytes[srcOffset + 0u];
            outRgba8[dstOffset + 3u] = (bytesPerPixel == 4u) ? bytes[srcOffset + 3u] : 255u;
        }
    }
    return true;
}

bool loadTextureRgba(
    const std::filesystem::path& path,
    std::uint32_t& outWidth,
    std::uint32_t& outHeight,
    std::vector<std::uint8_t>& outRgba8
) {
    const std::string extension = lowerCopy(path.extension().string());
    if (extension == ".dds") {
        return decodeDdsRgba(path, outWidth, outHeight, outRgba8);
    }
    if (extension == ".tga") {
        return decodeTgaRgba(path, outWidth, outHeight, outRgba8);
    }
    if (extension == ".bmp") {
        return decodeBmpRgba(path, outWidth, outHeight, outRgba8);
    }
    return false;
}

std::uint32_t computeTextureMipLevelCount(std::uint32_t width, std::uint32_t height) {
    std::uint32_t mipLevels = 1u;
    while (width > 1u || height > 1u) {
        width = std::max(1u, width >> 1u);
        height = std::max(1u, height >> 1u);
        ++mipLevels;
    }
    return mipLevels;
}

std::size_t computeTextureMipChainByteSize(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t mipLevelCount
) {
    std::size_t totalBytes = 0u;
    for (std::uint32_t mipLevel = 0; mipLevel < mipLevelCount; ++mipLevel) {
        totalBytes += static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
        width = std::max(1u, width >> 1u);
        height = std::max(1u, height >> 1u);
    }
    return totalBytes;
}

void generateTextureMipChain(
    std::uint32_t width,
    std::uint32_t height,
    std::vector<std::uint8_t>& rgba8,
    std::uint32_t& outMipLevelCount
) {
    const std::size_t baseLevelByteSize =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u;
    if (width == 0u || height == 0u || rgba8.size() != baseLevelByteSize) {
        outMipLevelCount = 0u;
        rgba8.clear();
        return;
    }

    outMipLevelCount = computeTextureMipLevelCount(width, height);
    if (outMipLevelCount <= 1u) {
        outMipLevelCount = 1u;
        return;
    }

    std::vector<std::uint8_t> mipChain;
    mipChain.reserve(computeTextureMipChainByteSize(width, height, outMipLevelCount));
    mipChain.insert(mipChain.end(), rgba8.begin(), rgba8.end());

    std::vector<std::uint8_t> currentLevel = rgba8;
    std::uint32_t currentWidth = width;
    std::uint32_t currentHeight = height;
    for (std::uint32_t mipLevel = 1u; mipLevel < outMipLevelCount; ++mipLevel) {
        const std::uint32_t nextWidth = std::max(1u, currentWidth >> 1u);
        const std::uint32_t nextHeight = std::max(1u, currentHeight >> 1u);
        std::vector<std::uint8_t> nextLevel(
            static_cast<std::size_t>(nextWidth) * static_cast<std::size_t>(nextHeight) * 4u,
            0u);
        for (std::uint32_t y = 0; y < nextHeight; ++y) {
            for (std::uint32_t x = 0; x < nextWidth; ++x) {
                std::uint32_t accum[4] = {0u, 0u, 0u, 0u};
                for (std::uint32_t sampleY = 0; sampleY < 2u; ++sampleY) {
                    for (std::uint32_t sampleX = 0; sampleX < 2u; ++sampleX) {
                        const std::uint32_t srcX = std::min((x * 2u) + sampleX, currentWidth - 1u);
                        const std::uint32_t srcY = std::min((y * 2u) + sampleY, currentHeight - 1u);
                        const std::size_t srcOffset =
                            (static_cast<std::size_t>(srcY) * static_cast<std::size_t>(currentWidth) + srcX) * 4u;
                        accum[0] += currentLevel[srcOffset + 0u];
                        accum[1] += currentLevel[srcOffset + 1u];
                        accum[2] += currentLevel[srcOffset + 2u];
                        accum[3] += currentLevel[srcOffset + 3u];
                    }
                }
                const std::size_t dstOffset =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(nextWidth) + x) * 4u;
                nextLevel[dstOffset + 0u] = static_cast<std::uint8_t>((accum[0] + 2u) / 4u);
                nextLevel[dstOffset + 1u] = static_cast<std::uint8_t>((accum[1] + 2u) / 4u);
                nextLevel[dstOffset + 2u] = static_cast<std::uint8_t>((accum[2] + 2u) / 4u);
                nextLevel[dstOffset + 3u] = static_cast<std::uint8_t>((accum[3] + 2u) / 4u);
            }
        }
        mipChain.insert(mipChain.end(), nextLevel.begin(), nextLevel.end());
        currentLevel = std::move(nextLevel);
        currentWidth = nextWidth;
        currentHeight = nextHeight;
    }

    rgba8 = std::move(mipChain);
}

bool textureUsesAlphaCutout(const ImportedSceneTexture& texture) {
    const std::size_t basePixelCount = static_cast<std::size_t>(texture.width) * texture.height;
    const std::size_t baseByteCount = basePixelCount * 4u;
    if (texture.width == 0u || texture.height == 0u || texture.rgba8.size() < baseByteCount) {
        return false;
    }

    bool sawTransparent = false;
    bool sawVisible = false;
    for (std::size_t pixelIndex = 0; pixelIndex < basePixelCount; ++pixelIndex) {
        const std::uint8_t alpha = texture.rgba8[(pixelIndex * 4u) + 3u];
        sawTransparent = sawTransparent || alpha < 250u;
        sawVisible = sawVisible || alpha > 8u;
        if (sawTransparent && sawVisible) {
            return true;
        }
    }
    return false;
}

std::vector<bool> buildTextureAlphaCutoutMask(const std::vector<ImportedSceneTexture>& textures) {
    std::vector<bool> mask(textures.size(), false);
    for (std::size_t textureIndex = 0; textureIndex < textures.size(); ++textureIndex) {
        mask[textureIndex] = textureUsesAlphaCutout(textures[textureIndex]);
    }
    return mask;
}

bool textureIndexUsesAlphaCutout(const std::vector<bool>& mask, std::uint32_t textureIndex) {
    return textureIndex < mask.size() && mask[textureIndex];
}

void applyTextureAlphaCutoutFlags(ImportedScene& scene) {
    const std::vector<bool> textureAlphaCutoutMask = buildTextureAlphaCutoutMask(scene.textures);
    for (ImportedSceneMesh& mesh : scene.meshes) {
        for (ImportedSceneMeshPart& part : mesh.parts) {
            if (textureIndexUsesAlphaCutout(textureAlphaCutoutMask, part.textureIndex)) {
                part.alphaTest = true;
            }
        }
    }
    for (ImportedScenePackedVertex& vertex : scene.packedVertices) {
        if (textureIndexUsesAlphaCutout(textureAlphaCutoutMask, vertex.textureIndex)) {
            vertex.flags |= kImportedSceneMaterialFlagAlphaTest;
        }
    }
}

bool readExact(std::istream& input, void* dst, std::size_t size) {
    input.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
    return input.good();
}

bool skipExact(std::istream& input, std::size_t size) {
    input.seekg(static_cast<std::streamoff>(size), std::ios::cur);
    return input.good();
}

template <typename T>
bool readValue(std::istream& input, T& out) {
    static_assert(std::is_trivially_copyable_v<T>);
    return readExact(input, &out, sizeof(T));
}

template <typename T>
void writeValue(std::ostream& output, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    output.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
}

void writeString(std::ostream& output, const std::string& value) {
    const std::uint32_t size = static_cast<std::uint32_t>(value.size());
    writeValue(output, size);
    if (!value.empty()) {
        output.write(value.data(), static_cast<std::streamsize>(value.size()));
    }
}

bool readString(std::istream& input, std::string& out) {
    std::uint32_t size = 0;
    if (!readValue(input, size)) {
        return false;
    }
    out.resize(size);
    return size == 0 || readExact(input, out.data(), size);
}

bool skipString(std::istream& input) {
    std::uint32_t size = 0;
    if (!readValue(input, size)) {
        return false;
    }
    return size == 0 || skipExact(input, size);
}

struct Tes3RecordHeader {
    char name[4] = {};
    std::uint32_t size = 0;
    std::uint32_t unknown = 0;
    std::uint32_t flags = 0;
};

struct Tes3SubRecordHeader {
    char name[4] = {};
    std::uint32_t size = 0;
};

std::string fourCcToString(const char name[4]) {
    return std::string(name, name + 4);
}

bool readRecordHeader(std::istream& input, Tes3RecordHeader& outHeader) {
    return readExact(input, &outHeader, sizeof(outHeader));
}

bool readSubRecordHeader(std::istream& input, Tes3SubRecordHeader& outHeader) {
    return readExact(input, &outHeader, sizeof(outHeader));
}

std::string readSizedString(std::istream& input, std::uint32_t size) {
    std::string buffer(size, '\0');
    if (size != 0) {
        input.read(buffer.data(), static_cast<std::streamsize>(size));
    }
    return trimNullTerminated(buffer);
}

struct ParsedCellRef {
    std::string refId;
    float position[3] = {};
    float rotation[3] = {};
    float scale = 1.0f;
    bool deleted = false;
};

struct ParsedCell {
    std::string name;
    int gridX = 0;
    int gridY = 0;
    std::vector<ParsedCellRef> refs;
};

struct ParsedLand {
    int gridX = 0;
    int gridY = 0;
    std::vector<float> heights;
    std::vector<std::uint16_t> textureIndices;
};

struct ParsedLightRecord {
    std::string id;
    std::string modelPath;
    float color[3] = {1.0f, 1.0f, 1.0f};
    std::int32_t radius = 0;
    std::int32_t time = 0;
    std::uint32_t flags = 0u;
};

struct CellSummary {
    std::string name;
    int gridX = 0;
    int gridY = 0;
    bool exterior = true;
};

bool parseCellSummary(std::istream& input, std::uint32_t recordSize, CellSummary& outCell) {
    const std::streampos recordStart = input.tellg();
    std::uint32_t bytesLeft = recordSize;
    bool reachedRefs = false;
    while (bytesLeft >= sizeof(Tes3SubRecordHeader)) {
        Tes3SubRecordHeader subHeader{};
        if (!readSubRecordHeader(input, subHeader)) {
            return false;
        }
        bytesLeft -= static_cast<std::uint32_t>(sizeof(subHeader));
        const std::string subName = fourCcToString(subHeader.name);
        if (subHeader.size > bytesLeft) {
            return false;
        }
        if (subName == "NAME" && !reachedRefs) {
            outCell.name = readSizedString(input, subHeader.size);
        } else if (subName == "DATA" && !reachedRefs && subHeader.size >= 12u) {
            std::int32_t flags = 0;
            std::int32_t x = 0;
            std::int32_t y = 0;
            readValue(input, flags);
            readValue(input, x);
            readValue(input, y);
            outCell.exterior = (flags & 0x01) == 0;
            outCell.gridX = x;
            outCell.gridY = y;
            if (subHeader.size > 12u) {
                input.seekg(static_cast<std::streamoff>(subHeader.size - 12u), std::ios::cur);
            }
        } else if (
            subName == "RGNN" || subName == "NAM5" || subName == "NAM0" ||
            subName == "INTV" || subName == "WHGT" || subName == "AMBI") {
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        } else {
            reachedRefs = true;
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        }
        bytesLeft -= subHeader.size;
    }
    input.seekg(recordStart + static_cast<std::streamoff>(recordSize), std::ios::beg);
    return true;
}

bool parseCellRefs(std::istream& input, std::uint32_t recordSize, ParsedCell& outCell) {
    const std::streampos recordStart = input.tellg();
    std::uint32_t bytesLeft = recordSize;
    ParsedCellRef currentRef{};
    bool inRef = false;
    auto flushRef = [&]() {
        if (inRef && !currentRef.refId.empty() && !currentRef.deleted) {
            outCell.refs.push_back(currentRef);
        }
        currentRef = ParsedCellRef{};
        inRef = false;
    };

    while (bytesLeft >= sizeof(Tes3SubRecordHeader)) {
        Tes3SubRecordHeader subHeader{};
        if (!readSubRecordHeader(input, subHeader)) {
            return false;
        }
        bytesLeft -= static_cast<std::uint32_t>(sizeof(subHeader));
        const std::string subName = fourCcToString(subHeader.name);
        if (subHeader.size > bytesLeft) {
            return false;
        }

        if (subName == "NAME" && !inRef) {
            outCell.name = readSizedString(input, subHeader.size);
        } else if (subName == "DATA" && !inRef && subHeader.size >= 12u) {
            std::int32_t flags = 0;
            readValue(input, flags);
            readValue(input, outCell.gridX);
            readValue(input, outCell.gridY);
            if (subHeader.size > 12u) {
                input.seekg(static_cast<std::streamoff>(subHeader.size - 12u), std::ios::cur);
            }
        } else if (subName == "FRMR") {
            flushRef();
            inRef = true;
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        } else if (subName == "MVRF") {
            flushRef();
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        } else if (subName == "CNDT") {
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        } else if (inRef && subName == "NAME") {
            currentRef.refId = readSizedString(input, subHeader.size);
        } else if (inRef && subName == "XSCL" && subHeader.size == sizeof(float)) {
            readValue(input, currentRef.scale);
        } else if (inRef && subName == "DATA" && subHeader.size >= sizeof(float) * 6u) {
            readExact(input, currentRef.position, sizeof(float) * 3u);
            readExact(input, currentRef.rotation, sizeof(float) * 3u);
            if (subHeader.size > sizeof(float) * 6u) {
                input.seekg(static_cast<std::streamoff>(subHeader.size - (sizeof(float) * 6u)), std::ios::cur);
            }
        } else if (inRef && subName == "DELE") {
            currentRef.deleted = true;
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        } else {
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        }
        bytesLeft -= subHeader.size;
    }

    flushRef();
    input.seekg(recordStart + static_cast<std::streamoff>(recordSize), std::ios::beg);
    return true;
}

std::vector<std::uint16_t> transposeLandTextureData(const std::vector<std::uint16_t>& source) {
    std::vector<std::uint16_t> transposed(source.size(), 0u);
    std::size_t readPos = 0;
    for (std::size_t y1 = 0; y1 < 4; ++y1) {
        for (std::size_t x1 = 0; x1 < 4; ++x1) {
            for (std::size_t y2 = 0; y2 < 4; ++y2) {
                for (std::size_t x2 = 0; x2 < 4; ++x2) {
                    const std::size_t writeIndex = ((y1 * 4u + y2) * 16u) + (x1 * 4u + x2);
                    transposed[writeIndex] = source[readPos++];
                }
            }
        }
    }
    return transposed;
}

bool parseLandRecord(std::istream& input, std::uint32_t recordSize, ParsedLand& outLand) {
    const std::streampos recordStart = input.tellg();
    std::uint32_t bytesLeft = recordSize;
    std::uint32_t flags = 0;
    std::vector<std::string> seenSubrecords;
    while (bytesLeft >= sizeof(Tes3SubRecordHeader)) {
        Tes3SubRecordHeader subHeader{};
        if (!readSubRecordHeader(input, subHeader)) {
            setLastImportedSceneError(
                "Failed to read LAND subrecord header after seeing " +
                std::to_string(seenSubrecords.size()) + " subrecords");
            return false;
        }
        bytesLeft -= static_cast<std::uint32_t>(sizeof(subHeader));
        const std::string subName = fourCcToString(subHeader.name);
        seenSubrecords.push_back(subName);
        if (subHeader.size > bytesLeft) {
            setLastImportedSceneError(
                "LAND parse overrun at subrecord " + subName +
                " with size " + std::to_string(subHeader.size) +
                " and only " + std::to_string(bytesLeft) + " bytes remaining");
            return false;
        }
        if (subName == "INTV" && subHeader.size >= 8u) {
            readValue(input, outLand.gridX);
            readValue(input, outLand.gridY);
            if (subHeader.size > 8u) {
                input.seekg(static_cast<std::streamoff>(subHeader.size - 8u), std::ios::cur);
            }
        } else if (subName == "DATA" && subHeader.size >= 4u) {
            readValue(input, flags);
            if (subHeader.size > 4u) {
                input.seekg(static_cast<std::streamoff>(subHeader.size - 4u), std::ios::cur);
            }
        } else if (subName == "VHGT" && subHeader.size >= 4u + (kLandSize * kLandSize) + 3u) {
            float heightOffset = 0.0f;
            readValue(input, heightOffset);
            std::vector<std::int8_t> deltas(kLandSize * kLandSize, 0);
            readExact(input, deltas.data(), deltas.size());
            char padding[3]{};
            readExact(input, padding, sizeof(padding));

            outLand.heights.assign(static_cast<std::size_t>(kLandSize * kLandSize), 0.0f);
            std::size_t index = 0;
            float previousRowStart = heightOffset * kHeightScale;
            for (int row = 0; row < kLandSize; ++row) {
                float previous = previousRowStart;
                for (int col = 0; col < kLandSize; ++col) {
                    if (row == 0 && col == 0) {
                        previous = heightOffset * kHeightScale;
                    } else if (col == 0) {
                        previous += static_cast<float>(deltas[index]) * kHeightScale;
                        previousRowStart = previous;
                    } else {
                        previous += static_cast<float>(deltas[index]) * kHeightScale;
                    }
                    outLand.heights[index] = previous;
                    ++index;
                }
            }
            if (subHeader.size > 4u + deltas.size() + 3u) {
                input.seekg(static_cast<std::streamoff>(subHeader.size - (4u + deltas.size() + 3u)), std::ios::cur);
            }
        } else if (subName == "VTEX" && subHeader.size >= (kLandTextureSize * kLandTextureSize * sizeof(std::uint16_t))) {
            std::vector<std::uint16_t> raw(static_cast<std::size_t>(kLandTextureSize * kLandTextureSize), 0u);
            readExact(input, raw.data(), raw.size() * sizeof(std::uint16_t));
            outLand.textureIndices = transposeLandTextureData(raw);
            if (subHeader.size > raw.size() * sizeof(std::uint16_t)) {
                input.seekg(static_cast<std::streamoff>(subHeader.size - (raw.size() * sizeof(std::uint16_t))), std::ios::cur);
            }
        } else if (subName == "DELE") {
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        } else {
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        }
        bytesLeft -= subHeader.size;
    }
    if (bytesLeft != 0u) {
        setLastImportedSceneError(
            "LAND record ended with " + std::to_string(bytesLeft) + " trailing bytes unparsed");
        return false;
    }
    input.seekg(recordStart + static_cast<std::streamoff>(recordSize), std::ios::beg);
    return true;
}

bool parseLandTextureRecord(std::istream& input, std::uint32_t recordSize, std::uint32_t& outIndex, std::string& outTexturePath) {
    const std::streampos recordStart = input.tellg();
    std::uint32_t bytesLeft = recordSize;
    while (bytesLeft >= sizeof(Tes3SubRecordHeader)) {
        Tes3SubRecordHeader subHeader{};
        if (!readSubRecordHeader(input, subHeader)) {
            return false;
        }
        bytesLeft -= static_cast<std::uint32_t>(sizeof(subHeader));
        const std::string subName = fourCcToString(subHeader.name);
        if (subHeader.size > bytesLeft) {
            return false;
        }
        if (subName == "INTV" && subHeader.size >= 4u) {
            readValue(input, outIndex);
            if (subHeader.size > 4u) {
                input.seekg(static_cast<std::streamoff>(subHeader.size - 4u), std::ios::cur);
            }
        } else if (subName == "DATA") {
            outTexturePath = readSizedString(input, subHeader.size);
        } else {
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        }
        bytesLeft -= subHeader.size;
    }
    input.seekg(recordStart + static_cast<std::streamoff>(recordSize), std::ios::beg);
    return true;
}

bool parseModelRecord(std::istream& input, std::uint32_t recordSize, std::string& outId, std::string& outModelPath) {
    const std::streampos recordStart = input.tellg();
    std::uint32_t bytesLeft = recordSize;
    while (bytesLeft >= sizeof(Tes3SubRecordHeader)) {
        Tes3SubRecordHeader subHeader{};
        if (!readSubRecordHeader(input, subHeader)) {
            return false;
        }
        bytesLeft -= static_cast<std::uint32_t>(sizeof(subHeader));
        const std::string subName = fourCcToString(subHeader.name);
        if (subHeader.size > bytesLeft) {
            return false;
        }
        if (subName == "NAME") {
            outId = lowerCopy(readSizedString(input, subHeader.size));
        } else if (subName == "MODL") {
            outModelPath = lowerCopy(readSizedString(input, subHeader.size));
        } else {
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        }
        bytesLeft -= subHeader.size;
    }
    input.seekg(recordStart + static_cast<std::streamoff>(recordSize), std::ios::beg);
    return true;
}

bool parseLightRecord(std::istream& input, std::uint32_t recordSize, ParsedLightRecord& outLight) {
    const std::streampos recordStart = input.tellg();
    std::uint32_t bytesLeft = recordSize;
    while (bytesLeft >= sizeof(Tes3SubRecordHeader)) {
        Tes3SubRecordHeader subHeader{};
        if (!readSubRecordHeader(input, subHeader)) {
            return false;
        }
        bytesLeft -= static_cast<std::uint32_t>(sizeof(subHeader));
        const std::string subName = fourCcToString(subHeader.name);
        if (subHeader.size > bytesLeft) {
            return false;
        }
        if (subName == "NAME") {
            outLight.id = lowerCopy(readSizedString(input, subHeader.size));
        } else if (subName == "MODL") {
            outLight.modelPath = lowerCopy(readSizedString(input, subHeader.size));
        } else if (subName == "LHDT" && subHeader.size >= 24u) {
            float weight = 0.0f;
            std::int32_t value = 0;
            std::int32_t time = 0;
            std::int32_t radius = 0;
            std::uint32_t color = 0xffffffffu;
            std::int32_t flags = 0;
            readValue(input, weight);
            readValue(input, value);
            readValue(input, time);
            readValue(input, radius);
            readValue(input, color);
            readValue(input, flags);
            (void)weight;
            (void)value;
            outLight.time = time;
            outLight.radius = radius;
            outLight.flags = static_cast<std::uint32_t>(flags);
            outLight.color[0] = static_cast<float>(color & 0xffu) * (1.0f / 255.0f);
            outLight.color[1] = static_cast<float>((color >> 8u) & 0xffu) * (1.0f / 255.0f);
            outLight.color[2] = static_cast<float>((color >> 16u) & 0xffu) * (1.0f / 255.0f);
            if (subHeader.size > 24u) {
                input.seekg(static_cast<std::streamoff>(subHeader.size - 24u), std::ios::cur);
            }
        } else {
            input.seekg(static_cast<std::streamoff>(subHeader.size), std::ios::cur);
        }
        bytesLeft -= subHeader.size;
    }
    input.seekg(recordStart + static_cast<std::streamoff>(recordSize), std::ios::beg);
    return true;
}

std::array<float, 3> computeCellNormal(const ParsedLand& land, int x, int y) {
    const auto sample = [&](int sx, int sy) -> float {
        sx = std::clamp(sx, 0, kLandSize - 1);
        sy = std::clamp(sy, 0, kLandSize - 1);
        return land.heights[static_cast<std::size_t>((sy * kLandSize) + sx)];
    };
    const float step = kCellSizeUnits / static_cast<float>(kLandSize - 1);
    const float hx = sample(x + 1, y) - sample(x - 1, y);
    const float hy = sample(x, y + 1) - sample(x, y - 1);
    std::array<float, 3> normal{
        -hx,
        2.0f * step,
        -hy
    };
    const float len = std::sqrt((normal[0] * normal[0]) + (normal[1] * normal[1]) + (normal[2] * normal[2]));
    if (len > 0.0001f) {
        normal[0] /= len;
        normal[1] /= len;
        normal[2] /= len;
    }
    return normal;
}

std::array<float, 16> identityMatrix() {
    return {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

std::array<float, 16> multiplyMatrices(
    const std::array<float, 16>& lhs,
    const std::array<float, 16>& rhs
) {
    std::array<float, 16> out{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                sum += lhs[row * 4 + i] * rhs[i * 4 + col];
            }
            out[row * 4 + col] = sum;
        }
    }
    return out;
}

std::array<float, 16> makeScaleMatrix(float scale) {
    return {
        scale, 0.0f, 0.0f, 0.0f,
        0.0f, scale, 0.0f, 0.0f,
        0.0f, 0.0f, scale, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

std::array<float, 16> makeTranslationMatrix(float x, float y, float z) {
    return {
        1.0f, 0.0f, 0.0f, x,
        0.0f, 1.0f, 0.0f, y,
        0.0f, 0.0f, 1.0f, z,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

struct Quaternion {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

Quaternion makeAxisAngleQuaternion(float axisX, float axisY, float axisZ, float radians) {
    const float halfAngle = radians * 0.5f;
    const float s = std::sin(halfAngle);
    const float c = std::cos(halfAngle);
    return {
        axisX * s,
        axisY * s,
        axisZ * s,
        c
    };
}

Quaternion multiplyQuaternions(const Quaternion& lhs, const Quaternion& rhs) {
    return {
        (lhs.w * rhs.x) + (lhs.x * rhs.w) + (lhs.y * rhs.z) - (lhs.z * rhs.y),
        (lhs.w * rhs.y) - (lhs.x * rhs.z) + (lhs.y * rhs.w) + (lhs.z * rhs.x),
        (lhs.w * rhs.z) + (lhs.x * rhs.y) - (lhs.y * rhs.x) + (lhs.z * rhs.w),
        (lhs.w * rhs.w) - (lhs.x * rhs.x) - (lhs.y * rhs.y) - (lhs.z * rhs.z)
    };
}

Quaternion normalizeQuaternion(Quaternion q) {
    const float length = std::sqrt((q.x * q.x) + (q.y * q.y) + (q.z * q.z) + (q.w * q.w));
    if (length > 1e-6f) {
        q.x /= length;
        q.y /= length;
        q.z /= length;
        q.w /= length;
    } else {
        q = {};
    }
    return q;
}

std::array<float, 16> makeRotationMatrix(const Quaternion& quaternion) {
    const Quaternion q = normalizeQuaternion(quaternion);
    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;
    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;
    const float wx = q.w * q.x;
    const float wy = q.w * q.y;
    const float wz = q.w * q.z;

    return {
        1.0f - (2.0f * (yy + zz)), 2.0f * (xy - wz), 2.0f * (xz + wy), 0.0f,
        2.0f * (xy + wz), 1.0f - (2.0f * (xx + zz)), 2.0f * (yz - wx), 0.0f,
        2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - (2.0f * (xx + yy)), 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
}

std::array<float, 3> transformPoint(
    const std::array<float, 16>& matrix,
    const std::array<float, 3>& point
) {
    return {
        (matrix[0] * point[0]) + (matrix[1] * point[1]) + (matrix[2] * point[2]) + matrix[3],
        (matrix[4] * point[0]) + (matrix[5] * point[1]) + (matrix[6] * point[2]) + matrix[7],
        (matrix[8] * point[0]) + (matrix[9] * point[1]) + (matrix[10] * point[2]) + matrix[11]
    };
}

std::array<float, 3> transformDirection(
    const std::array<float, 16>& matrix,
    const std::array<float, 3>& direction
) {
    return {
        (matrix[0] * direction[0]) + (matrix[1] * direction[1]) + (matrix[2] * direction[2]),
        (matrix[4] * direction[0]) + (matrix[5] * direction[1]) + (matrix[6] * direction[2]),
        (matrix[8] * direction[0]) + (matrix[9] * direction[1]) + (matrix[10] * direction[2])
    };
}

std::array<float, 3> normalizeVector(std::array<float, 3> value) {
    const float length = std::sqrt(
        (value[0] * value[0]) +
        (value[1] * value[1]) +
        (value[2] * value[2]));
    if (length > 1e-6f) {
        value[0] /= length;
        value[1] /= length;
        value[2] /= length;
    }
    return value;
}

std::array<float, 16> buildInstanceTransform(const ParsedCellRef& ref) {
    std::array<float, 16> transform = identityMatrix();
    // Morrowind object refs use X/Y ground-plane coordinates with Z as up.
    // Convert to the engine's X/Z ground plane with Y up before applying rotation.
    // Match OpenMW's direct non-actor rotation convention:
    // quat(z,-Z) * quat(y,-Y) * quat(x,-X), remapped through the engine's
    // handedness-changing Y/Z axis swap.
    transform = multiplyMatrices(transform, makeTranslationMatrix(ref.position[0], ref.position[2], ref.position[1]));
    const Quaternion rotation = multiplyQuaternions(
        multiplyQuaternions(
            makeAxisAngleQuaternion(0.0f, 1.0f, 0.0f, ref.rotation[2]),
            makeAxisAngleQuaternion(0.0f, 0.0f, 1.0f, ref.rotation[1])),
        makeAxisAngleQuaternion(1.0f, 0.0f, 0.0f, ref.rotation[0]));
    transform = multiplyMatrices(transform, makeRotationMatrix(rotation));
    transform = multiplyMatrices(transform, makeScaleMatrix(ref.scale));
    return transform;
}

std::array<float, 3> refPositionToEngineSpace(const float position[3]) {
    return {position[0], position[2], position[1]};
}

std::filesystem::path normalizeModelRelativePath(std::string path) {
    path = lowerCopy(std::move(path));
    while (path.rfind("./", 0) == 0) {
        path.erase(0, 2);
    }
    while (!path.empty() && path.front() == '/') {
        path.erase(path.begin());
    }
    if (path.rfind("data files/", 0) == 0) {
        path.erase(0, 11);
    }
    if (path.rfind("meshes/", 0) == 0) {
        path.erase(0, 7);
    }
    return std::filesystem::path(path);
}

std::string normalizeTextureRelativePath(std::string path) {
    path = lowerCopy(std::move(path));
    while (path.rfind("./", 0) == 0) {
        path.erase(0, 2);
    }
    while (!path.empty() && path.front() == '/') {
        path.erase(path.begin());
    }
    if (path.rfind("data files/", 0) == 0) {
        path.erase(0, 11);
    }
    return path;
}

std::optional<std::filesystem::path> resolveTextureFilePath(
    const std::unordered_map<std::string, std::filesystem::path>& dataFileIndex,
    const std::string& sourcePath
) {
    const std::string normalizedPath = normalizeTextureRelativePath(sourcePath);
    const auto direct = dataFileIndex.find(normalizedPath);
    if (direct != dataFileIndex.end()) {
        return direct->second;
    }
    if (normalizedPath.rfind("textures/", 0) != 0) {
        const auto prefixed = dataFileIndex.find("textures/" + normalizedPath);
        if (prefixed != dataFileIndex.end()) {
            return prefixed->second;
        }
    }
    return std::nullopt;
}

float sampleTextureChannelWrapped(
    const ImportedSceneTexture& texture,
    float u,
    float v,
    int channel
) {
    if (texture.width == 0u || texture.height == 0u ||
        texture.rgba8.size() < static_cast<std::size_t>(texture.width) * texture.height * 4u) {
        return channel == 3 ? 255.0f : 128.0f;
    }
    u -= std::floor(u);
    v -= std::floor(v);
    const float x = (u * static_cast<float>(texture.width)) - 0.5f;
    const float y = (v * static_cast<float>(texture.height)) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const auto texel = [&](int sx, int sy) -> float {
        sx %= static_cast<int>(texture.width);
        sy %= static_cast<int>(texture.height);
        if (sx < 0) {
            sx += static_cast<int>(texture.width);
        }
        if (sy < 0) {
            sy += static_cast<int>(texture.height);
        }
        const std::size_t offset =
            (static_cast<std::size_t>(sy) * texture.width + static_cast<std::size_t>(sx)) * 4u;
        return static_cast<float>(texture.rgba8[offset + static_cast<std::size_t>(channel)]);
    };
    const float c00 = texel(x0, y0);
    const float c10 = texel(x0 + 1, y0);
    const float c01 = texel(x0, y0 + 1);
    const float c11 = texel(x0 + 1, y0 + 1);
    return std::lerp(std::lerp(c00, c10, fx), std::lerp(c01, c11, fx), fy);
}

std::uint32_t buildTerrainCompositeTexture(
    ImportedScene& scene,
    const ParsedLand& land,
    const std::unordered_map<std::uint32_t, std::string>& landscapeTextureByIndex,
    const std::unordered_map<std::string, std::uint32_t>& terrainTextureSlotByPath
) {
    ImportedSceneTexture composite{};
    composite.sourcePath = "__generated/terrain_cell_" +
        std::to_string(land.gridX) + "_" + std::to_string(land.gridY) + "_splat";
    composite.width = kTerrainCompositeTextureSize;
    composite.height = kTerrainCompositeTextureSize;
    composite.mipLevelCount = 1u;
    composite.rgba8.resize(
        static_cast<std::size_t>(composite.width) * composite.height * 4u,
        255u);

    const auto textureSlotForVtex = [&](std::uint16_t vtex) -> std::uint32_t {
        const auto texIt = landscapeTextureByIndex.find(vtex);
        if (texIt == landscapeTextureByIndex.end()) {
            return std::numeric_limits<std::uint32_t>::max();
        }
        const auto slotIt = terrainTextureSlotByPath.find(texIt->second);
        if (slotIt == terrainTextureSlotByPath.end()) {
            return std::numeric_limits<std::uint32_t>::max();
        }
        return slotIt->second;
    };

    for (std::uint32_t y = 0; y < composite.height; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(composite.height);
        const float textureGridY = (v * static_cast<float>(kLandTextureSize)) - 0.5f;
        const int y0 = static_cast<int>(std::floor(textureGridY));
        const float fy = textureGridY - static_cast<float>(y0);
        const int tileY = std::clamp(static_cast<int>(v * static_cast<float>(kLandTextureSize)), 0, kLandTextureSize - 1);
        for (std::uint32_t x = 0; x < composite.width; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(composite.width);
            const float textureGridX = (u * static_cast<float>(kLandTextureSize)) - 0.5f;
            const int x0 = static_cast<int>(std::floor(textureGridX));
            const float fx = textureGridX - static_cast<float>(x0);
            const int tileX = std::clamp(static_cast<int>(u * static_cast<float>(kLandTextureSize)), 0, kLandTextureSize - 1);
            const float localU = (u * static_cast<float>(kLandTextureSize)) - static_cast<float>(tileX);
            const float localV = (v * static_cast<float>(kLandTextureSize)) - static_cast<float>(tileY);

            float accum[4] = {};
            float totalWeight = 0.0f;
            for (int oy = 0; oy < 2; ++oy) {
                const int sy = std::clamp(y0 + oy, 0, kLandTextureSize - 1);
                const float wy = oy == 0 ? (1.0f - fy) : fy;
                for (int ox = 0; ox < 2; ++ox) {
                    const int sx = std::clamp(x0 + ox, 0, kLandTextureSize - 1);
                    const float wx = ox == 0 ? (1.0f - fx) : fx;
                    const float weight = wx * wy;
                    if (weight <= 0.0f) {
                        continue;
                    }
                    const std::uint16_t vtex = land.textureIndices[static_cast<std::size_t>((sy * kLandTextureSize) + sx)];
                    const std::uint32_t slot = textureSlotForVtex(vtex);
                    if (slot >= scene.textures.size()) {
                        continue;
                    }
                    const ImportedSceneTexture& source = scene.textures[slot];
                    for (int channel = 0; channel < 4; ++channel) {
                        accum[channel] += sampleTextureChannelWrapped(source, localU, localV, channel) * weight;
                    }
                    totalWeight += weight;
                }
            }

            const std::size_t dstOffset = (static_cast<std::size_t>(y) * composite.width + x) * 4u;
            if (totalWeight <= 1e-5f) {
                composite.rgba8[dstOffset + 0u] = 128u;
                composite.rgba8[dstOffset + 1u] = 128u;
                composite.rgba8[dstOffset + 2u] = 128u;
                composite.rgba8[dstOffset + 3u] = 255u;
                continue;
            }
            for (int channel = 0; channel < 4; ++channel) {
                composite.rgba8[dstOffset + static_cast<std::size_t>(channel)] =
                    static_cast<std::uint8_t>(std::clamp((accum[channel] / totalWeight) + 0.5f, 0.0f, 255.0f));
            }
        }
    }

    const std::uint32_t textureIndex = static_cast<std::uint32_t>(scene.textures.size());
    scene.textures.push_back(std::move(composite));
    return textureIndex;
}

std::unordered_map<std::string, std::filesystem::path> buildCaseInsensitiveFileIndex(
    const std::filesystem::path& root
) {
    std::unordered_map<std::string, std::filesystem::path> index;
    std::error_code iterError;
    std::filesystem::recursive_directory_iterator it(
        root,
        std::filesystem::directory_options::skip_permission_denied,
        iterError);
    std::filesystem::recursive_directory_iterator end;
    for (; it != end; it.increment(iterError)) {
        if (iterError) {
            iterError.clear();
            continue;
        }
        if (!it->is_regular_file()) {
            continue;
        }
        const std::filesystem::path relativePath = std::filesystem::relative(it->path(), root, iterError);
        if (iterError) {
            iterError.clear();
            continue;
        }
        index.emplace(lowerCopy(relativePath.generic_string()), it->path());
    }
    return index;
}

ImportedScene buildSceneFromParsedData(
    const std::filesystem::path& morrowindDataFilesPath,
    const std::unordered_map<std::uint32_t, std::string>& landscapeTextureByIndex,
    const std::unordered_map<std::string, std::string>& modelPathById,
    const std::unordered_map<std::string, ParsedLightRecord>& lightById,
    const std::vector<ParsedCell>& cells,
    const std::vector<ParsedLand>& lands
) {
    ImportedScene scene{};
    scene.sourceTag = "morrowind_balmora";
    scene.landscapeCells.reserve(lands.size());
    for (const ParsedLand& land : lands) {
        ImportedSceneLandscapeCell cell{};
        cell.gridX = land.gridX;
        cell.gridY = land.gridY;
        cell.heights = land.heights;
        cell.textureIndices = land.textureIndices;
        scene.landscapeCells.push_back(std::move(cell));

        if (land.heights.size() != static_cast<std::size_t>(kLandSize * kLandSize)) {
            continue;
        }
        const float cellOriginX = static_cast<float>(land.gridX) * kCellSizeUnits;
        const float cellOriginZ = static_cast<float>(land.gridY) * kCellSizeUnits;
        const float quadSize = kCellSizeUnits / static_cast<float>(kLandSize - 1);
        for (int quadY = 0; quadY < (kLandSize - 1); ++quadY) {
            bool inWaterRun = false;
            int runStartQuadX = 0;
            auto flushWaterRun = [&](int endQuadX) {
                if (!inWaterRun || endQuadX <= runStartQuadX) {
                    return;
                }
                ImportedSceneWaterPatch patch{};
                patch.originX = cellOriginX + (static_cast<float>(runStartQuadX) * quadSize);
                patch.originZ = cellOriginZ + (static_cast<float>(quadY) * quadSize);
                patch.sizeX = static_cast<float>(endQuadX - runStartQuadX) * quadSize;
                patch.sizeZ = quadSize;
                patch.waterLevel = kExteriorWaterLevel;
                scene.waterPatches.push_back(patch);
                inWaterRun = false;
            };
            for (int quadX = 0; quadX < (kLandSize - 1); ++quadX) {
                const std::size_t i00 = static_cast<std::size_t>((quadY * kLandSize) + quadX);
                const std::size_t i10 = static_cast<std::size_t>((quadY * kLandSize) + quadX + 1);
                const std::size_t i01 = static_cast<std::size_t>(((quadY + 1) * kLandSize) + quadX);
                const std::size_t i11 = static_cast<std::size_t>(((quadY + 1) * kLandSize) + quadX + 1);
                const float centerHeight =
                    (land.heights[i00] + land.heights[i10] + land.heights[i01] + land.heights[i11]) * 0.25f;
                if (centerHeight <= kExteriorWaterLevel) {
                    if (!inWaterRun) {
                        inWaterRun = true;
                        runStartQuadX = quadX;
                    }
                } else {
                    flushWaterRun(quadX);
                }
            }
            flushWaterRun(kLandSize - 1);
        }
    }

    std::unordered_set<std::string> uniqueLandscapeTextures;
    for (const ParsedLand& land : lands) {
        for (const std::uint16_t textureIndex : land.textureIndices) {
            const auto it = landscapeTextureByIndex.find(textureIndex);
            if (it == landscapeTextureByIndex.end()) {
                continue;
            }
            uniqueLandscapeTextures.insert(it->second);
        }
    }
    for (const std::string& texturePath : uniqueLandscapeTextures) {
        ImportedSceneTexture texture{};
        texture.sourcePath = texturePath;
        scene.textures.push_back(std::move(texture));
    }

    std::unordered_map<std::string, std::uint32_t> terrainTextureSlotByPath;
    for (std::uint32_t i = 0; i < scene.textures.size(); ++i) {
        terrainTextureSlotByPath[scene.textures[i].sourcePath] = i;
    }
    const std::unordered_map<std::string, std::filesystem::path> dataFileIndex =
        buildCaseInsensitiveFileIndex(morrowindDataFilesPath);
    for (ImportedSceneTexture& texture : scene.textures) {
        const std::optional<std::filesystem::path> texturePath =
            resolveTextureFilePath(dataFileIndex, texture.sourcePath);
        if (!texturePath.has_value()) {
            continue;
        }
        loadTextureRgba(*texturePath, texture.width, texture.height, texture.rgba8);
    }

    auto addTextureSlot = [&](const std::string& sourcePath) {
        const std::string normalizedPath = lowerCopy(sourcePath);
        const auto existing = terrainTextureSlotByPath.find(normalizedPath);
        if (existing != terrainTextureSlotByPath.end()) {
            return existing->second;
        }
        ImportedSceneTexture texture{};
        texture.sourcePath = normalizedPath;
        const std::uint32_t textureIndex = static_cast<std::uint32_t>(scene.textures.size());
        scene.textures.push_back(std::move(texture));
        terrainTextureSlotByPath.emplace(normalizedPath, textureIndex);
        return textureIndex;
    };

    ImportedSceneMesh terrainMesh{};
    terrainMesh.name = "terrain";
    for (const ParsedLand& land : lands) {
        if (land.heights.size() != static_cast<std::size_t>(kLandSize * kLandSize) ||
            land.textureIndices.size() != static_cast<std::size_t>(kLandTextureSize * kLandTextureSize)) {
            continue;
        }

        const float cellOriginX = static_cast<float>(land.gridX) * kCellSizeUnits;
        const float cellOriginZ = static_cast<float>(land.gridY) * kCellSizeUnits;
        const float quadSize = kCellSizeUnits / static_cast<float>(kLandSize - 1);
        const std::uint32_t compositeTextureIndex = buildTerrainCompositeTexture(
            scene,
            land,
            landscapeTextureByIndex,
            terrainTextureSlotByPath);

        ImportedSceneMeshPart part{};
        part.firstIndex = static_cast<std::uint32_t>(terrainMesh.indices.size());
        part.textureIndex = compositeTextureIndex;
        part.alphaTest = false;
        const std::uint32_t baseVertex = static_cast<std::uint32_t>(terrainMesh.vertices.size());
        for (int y = 0; y < kLandSize; ++y) {
            for (int x = 0; x < kLandSize; ++x) {
                const std::size_t vertexIndex = static_cast<std::size_t>((y * kLandSize) + x);
                ImportedSceneVertex vertex{};
                vertex.position[0] = cellOriginX + (static_cast<float>(x) * quadSize);
                vertex.position[1] = land.heights[vertexIndex];
                vertex.position[2] = cellOriginZ + (static_cast<float>(y) * quadSize);
                const std::array<float, 3> normal = computeCellNormal(land, x, y);
                vertex.normal[0] = normal[0];
                vertex.normal[1] = normal[1];
                vertex.normal[2] = normal[2];
                vertex.uv[0] = static_cast<float>(x) / static_cast<float>(kLandSize - 1);
                vertex.uv[1] = static_cast<float>(y) / static_cast<float>(kLandSize - 1);
                terrainMesh.vertices.push_back(vertex);
            }
        }

        for (int y = 0; y < (kLandSize - 1); ++y) {
            for (int x = 0; x < (kLandSize - 1); ++x) {
                const std::uint32_t i0 = baseVertex + static_cast<std::uint32_t>((y * kLandSize) + x);
                const std::uint32_t i1 = i0 + 1u;
                const std::uint32_t i2 = i0 + static_cast<std::uint32_t>(kLandSize);
                const std::uint32_t i3 = i2 + 1u;
                terrainMesh.indices.push_back(i0);
                terrainMesh.indices.push_back(i2);
                terrainMesh.indices.push_back(i1);
                terrainMesh.indices.push_back(i1);
                terrainMesh.indices.push_back(i2);
                terrainMesh.indices.push_back(i3);
            }
        }
        part.indexCount = static_cast<std::uint32_t>(terrainMesh.indices.size()) - part.firstIndex;
        terrainMesh.parts.push_back(part);
    }
    if (!terrainMesh.indices.empty()) {
        scene.meshes.push_back(std::move(terrainMesh));
    }

    const std::filesystem::path meshesRoot = morrowindDataFilesPath / "Meshes";
    const std::unordered_map<std::string, std::filesystem::path> meshFileIndex = buildCaseInsensitiveFileIndex(meshesRoot);
    std::unordered_map<std::string, std::uint32_t> meshIndexByModelPath;
    std::unordered_map<std::string, std::string> failedMeshReasonByModelPath;
    std::unordered_map<std::string, std::size_t> failedMeshCountByReason;
    const std::vector<std::string> debugModelFilters = splitCommaSeparatedList(
        getEnvironmentVariable("ODAI_IMPORT_DEBUG_MODEL_FILTER").value_or(""));
    std::size_t attemptedModelImports = 0;
    std::size_t successfulModelImports = 0;

    for (const ParsedCell& cell : cells) {
        for (const ParsedCellRef& ref : cell.refs) {
            const auto lightIt = lightById.find(lowerCopy(ref.refId));
            if (lightIt != lightById.end()) {
                const ParsedLightRecord& lightRecord = lightIt->second;
                const bool isUsablePlacedLight =
                    lightRecord.radius > 0 &&
                    (lightRecord.flags & kTes3LightFlagOffDefault) == 0u &&
                    (lightRecord.flags & kTes3LightFlagNegative) == 0u;
                if (isUsablePlacedLight) {
                    ImportedSceneLight light{};
                    const std::array<float, 3> enginePosition = refPositionToEngineSpace(ref.position);
                    light.sourceId = ref.refId;
                    light.position[0] = enginePosition[0];
                    light.position[1] = enginePosition[1];
                    light.position[2] = enginePosition[2];
                    light.color[0] = lightRecord.color[0];
                    light.color[1] = lightRecord.color[1];
                    light.color[2] = lightRecord.color[2];
                    light.radius = static_cast<float>(lightRecord.radius) * std::max(ref.scale, 0.01f);
                    light.intensity = 1.0f;
                    light.flags = lightRecord.flags;
                    scene.lights.push_back(std::move(light));
                }
            }
            const auto modelIt = modelPathById.find(lowerCopy(ref.refId));
            if (modelIt != modelPathById.end()) {
                const std::string normalizedModelPath = lowerCopy(modelIt->second);
                std::uint32_t importedMeshIndex = std::numeric_limits<std::uint32_t>::max();
                bool haveImportedMesh = false;
                const auto existingMesh = meshIndexByModelPath.find(normalizedModelPath);
                if (existingMesh != meshIndexByModelPath.end()) {
                    importedMeshIndex = existingMesh->second;
                    haveImportedMesh = true;
                } else if (!failedMeshReasonByModelPath.contains(normalizedModelPath)) {
                    ++attemptedModelImports;
                    const auto fileIt = meshFileIndex.find(normalizeModelRelativePath(normalizedModelPath).generic_string());
                    if (fileIt != meshFileIndex.end()) {
                        ImportedNifResult nifResult{};
                        std::string nifError;
                        if (loadMorrowindStaticNif(fileIt->second, nifResult, nifError)) {
                            for (ImportedSceneMeshPart& part : nifResult.mesh.parts) {
                                if (!nifResult.diffuseTexturePath.empty()) {
                                    part.textureIndex = addTextureSlot(nifResult.diffuseTexturePath);
                                }
                            }
                            importedMeshIndex = static_cast<std::uint32_t>(scene.meshes.size());
                            scene.meshes.push_back(std::move(nifResult.mesh));
                            meshIndexByModelPath.emplace(normalizedModelPath, importedMeshIndex);
                            ++successfulModelImports;
                            haveImportedMesh = true;
                        } else {
                            failedMeshReasonByModelPath.emplace(normalizedModelPath, nifError);
                            ++failedMeshCountByReason[nifError];
                        }
                    } else {
                        failedMeshReasonByModelPath.emplace(normalizedModelPath, "Mesh file not found");
                        ++failedMeshCountByReason["Mesh file not found"];
                    }
                }

                if (haveImportedMesh) {
                    ImportedSceneInstance instance{};
                    instance.meshIndex = importedMeshIndex;
                    const std::array<float, 16> transform = buildInstanceTransform(ref);
                    std::copy(transform.begin(), transform.end(), instance.transform);
                    instance.sourceId = ref.refId;
                    instance.modelPath = normalizedModelPath;
                    if (modelPathMatchesDebugFilters(normalizedModelPath, debugModelFilters) &&
                        importedMeshIndex < scene.meshes.size()) {
                        const ImportedSceneMesh& debugMesh = scene.meshes[importedMeshIndex];
                        const DebugBounds localBounds = computeMeshBounds(debugMesh);
                        const DebugBounds worldBounds = computeTransformedMeshBounds(debugMesh, transform);
                        std::cerr << "[morrowind cooker][debug] model=" << normalizedModelPath
                                  << " refId=" << ref.refId
                                  << " cell=(" << cell.gridX << "," << cell.gridY << ")"
                                  << " pos=(" << ref.position[0] << "," << ref.position[1] << "," << ref.position[2] << ")"
                                  << " rot=(" << ref.rotation[0] << "," << ref.rotation[1] << "," << ref.rotation[2] << ")"
                                  << " scale=" << ref.scale;
                        if (localBounds.valid) {
                            std::cerr << " localBoundsMin=("
                                      << localBounds.min[0] << "," << localBounds.min[1] << "," << localBounds.min[2] << ")"
                                      << " localBoundsMax=("
                                      << localBounds.max[0] << "," << localBounds.max[1] << "," << localBounds.max[2] << ")";
                        }
                        if (worldBounds.valid) {
                            std::cerr << " worldBoundsMin=("
                                      << worldBounds.min[0] << "," << worldBounds.min[1] << "," << worldBounds.min[2] << ")"
                                      << " worldBoundsMax=("
                                      << worldBounds.max[0] << "," << worldBounds.max[1] << "," << worldBounds.max[2] << ")";
                        }
                        std::cerr << "\n";
                    }
                    scene.instances.push_back(std::move(instance));
                    continue;
                }
            }

            ImportedSceneCellRef unresolved{};
            unresolved.refId = ref.refId;
            if (modelIt != modelPathById.end()) {
                unresolved.modelPath = modelIt->second;
            }
            unresolved.position[0] = ref.position[0];
            unresolved.position[1] = ref.position[1];
            unresolved.position[2] = ref.position[2];
            unresolved.rotationRadians[0] = ref.rotation[0];
            unresolved.rotationRadians[1] = ref.rotation[1];
            unresolved.rotationRadians[2] = ref.rotation[2];
            unresolved.scale = ref.scale;
            scene.unresolvedRefs.push_back(std::move(unresolved));
        }
    }

    std::size_t loadedTextureCount = 0u;
    for (ImportedSceneTexture& texture : scene.textures) {
        if (texture.rgba8.empty()) {
            const std::optional<std::filesystem::path> texturePath =
                resolveTextureFilePath(dataFileIndex, texture.sourcePath);
            if (!texturePath.has_value()) {
                continue;
            }
            if (!loadTextureRgba(*texturePath, texture.width, texture.height, texture.rgba8)) {
                continue;
            }
        }
        generateTextureMipChain(texture.width, texture.height, texture.rgba8, texture.mipLevelCount);
        ++loadedTextureCount;
    }

    if (getEnvironmentVariable("ODAI_BAKE_RIVER_AUDIT").has_value()) {
        std::size_t printedRiverAuditRecords = 0u;
        for (const ImportedSceneInstance& instance : scene.instances) {
            if (instance.meshIndex >= scene.meshes.size()) {
                continue;
            }
            std::array<float, 16> transform{};
            std::copy(std::begin(instance.transform), std::end(instance.transform), transform.begin());
            const DebugBounds worldBounds = computeTransformedMeshBounds(scene.meshes[instance.meshIndex], transform);
            if (!worldBounds.valid) {
                continue;
            }
            float nearestWaterDistance = std::numeric_limits<float>::max();
            for (const ImportedSceneWaterPatch& patch : scene.waterPatches) {
                nearestWaterDistance = std::min(nearestWaterDistance, distanceBoundsToWaterPatchXz(worldBounds, patch));
            }
            if (nearestWaterDistance > kRiverAuditMaxDistance) {
                continue;
            }
            if (printedRiverAuditRecords == 0u) {
                std::cerr << "[morrowind cooker][river audit] statics within "
                          << kRiverAuditMaxDistance << " units of baked water:\n";
            }
            std::cerr << "[morrowind cooker][river audit] model=" << instance.modelPath
                      << " refId=" << instance.sourceId
                      << " nearestWaterDistance=" << nearestWaterDistance
                      << " boundsMin=(" << worldBounds.min[0] << "," << worldBounds.min[1] << "," << worldBounds.min[2] << ")"
                      << " boundsMax=(" << worldBounds.max[0] << "," << worldBounds.max[1] << "," << worldBounds.max[2] << ")\n";
            ++printedRiverAuditRecords;
            if (printedRiverAuditRecords >= 96u) {
                std::cerr << "[morrowind cooker][river audit] stopped after 96 nearby statics\n";
                break;
            }
        }
        if (printedRiverAuditRecords == 0u) {
            std::cerr << "[morrowind cooker][river audit] no statics found within "
                      << kRiverAuditMaxDistance << " units of baked water\n";
        }
    }

    std::cerr << "[morrowind cooker] Static mesh import: "
              << successfulModelImports << " succeeded out of "
              << attemptedModelImports << " attempted unique models, "
              << scene.instances.size() << " placed instances\n";
    std::cerr << "[morrowind cooker] Texture decode: "
              << loadedTextureCount << " succeeded out of "
              << scene.textures.size() << " referenced textures\n";
    if (!failedMeshReasonByModelPath.empty()) {
        for (const auto& [reason, count] : failedMeshCountByReason) {
            std::cerr << "[morrowind cooker] Static mesh failure bucket: "
                      << reason << " -> " << count << "\n";
        }
        int printed = 0;
        for (const auto& [modelPath, reason] : failedMeshReasonByModelPath) {
            std::cerr << "[morrowind cooker] Static mesh import failed: "
                      << modelPath << " -> " << reason << "\n";
            ++printed;
            if (printed >= 8) {
                break;
            }
        }
    }

    buildImportedScenePackedRenderData(scene);
    return scene;
}

}  // namespace

void buildImportedScenePackedRenderData(ImportedScene& scene) {
    applyTextureAlphaCutoutFlags(scene);
    scene.packedVertices.clear();
    scene.packedIndices.clear();
    scene.packedDraws.clear();
    scene.boundsMin[0] = std::numeric_limits<float>::max();
    scene.boundsMin[1] = std::numeric_limits<float>::max();
    scene.boundsMin[2] = std::numeric_limits<float>::max();
    scene.boundsMax[0] = std::numeric_limits<float>::lowest();
    scene.boundsMax[1] = std::numeric_limits<float>::lowest();
    scene.boundsMax[2] = std::numeric_limits<float>::lowest();

    auto expandBounds = [&](const ImportedScenePackedVertex& vertex) {
        scene.boundsMin[0] = std::min(scene.boundsMin[0], vertex.position[0]);
        scene.boundsMin[1] = std::min(scene.boundsMin[1], vertex.position[1]);
        scene.boundsMin[2] = std::min(scene.boundsMin[2], vertex.position[2]);
        scene.boundsMax[0] = std::max(scene.boundsMax[0], vertex.position[0]);
        scene.boundsMax[1] = std::max(scene.boundsMax[1], vertex.position[1]);
        scene.boundsMax[2] = std::max(scene.boundsMax[2], vertex.position[2]);
    };

    auto appendMesh = [&](const ImportedSceneMesh& mesh,
                          const std::array<float, 16>& transform,
                          const PackedRenderColor& color) {
        if (mesh.vertices.empty() || mesh.indices.empty()) {
            return;
        }
        const std::uint32_t firstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
        const auto appendVertex = [&](const ImportedSceneVertex& srcVertex,
                                      std::uint32_t textureIndex,
                                      std::uint32_t flags) {
            ImportedScenePackedVertex dstVertex{};
            const std::array<float, 3> localPosition{
                srcVertex.position[0],
                srcVertex.position[1],
                srcVertex.position[2]
            };
            const std::array<float, 3> localNormal{
                srcVertex.normal[0],
                srcVertex.normal[1],
                srcVertex.normal[2]
            };
            const std::array<float, 3> worldPosition = transformPoint(transform, localPosition);
            const std::array<float, 3> worldNormal = normalizeVector(transformDirection(transform, localNormal));
            dstVertex.position[0] = worldPosition[0];
            dstVertex.position[1] = worldPosition[1];
            dstVertex.position[2] = worldPosition[2];
            dstVertex.normal[0] = worldNormal[0];
            dstVertex.normal[1] = worldNormal[1];
            dstVertex.normal[2] = worldNormal[2];
            dstVertex.color[0] = color.r;
            dstVertex.color[1] = color.g;
            dstVertex.color[2] = color.b;
            dstVertex.uv[0] = srcVertex.uv[0];
            dstVertex.uv[1] = srcVertex.uv[1];
            dstVertex.textureIndex = textureIndex;
            dstVertex.flags = flags;
            const std::uint32_t packedVertexIndex = static_cast<std::uint32_t>(scene.packedVertices.size());
            scene.packedVertices.push_back(dstVertex);
            expandBounds(dstVertex);
            return packedVertexIndex;
        };

        if (mesh.parts.empty()) {
            const std::uint32_t invalidTextureIndex = std::numeric_limits<std::uint32_t>::max();
            std::vector<std::uint32_t> remappedVertexIndices(
                mesh.vertices.size(),
                std::numeric_limits<std::uint32_t>::max());
            for (const std::uint32_t index : mesh.indices) {
                if (index >= mesh.vertices.size()) {
                    continue;
                }
                std::uint32_t& remappedIndex = remappedVertexIndices[index];
                if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                    remappedIndex = appendVertex(mesh.vertices[index], invalidTextureIndex, 0u);
                }
                scene.packedIndices.push_back(remappedIndex);
            }
        } else {
            for (const ImportedSceneMeshPart& part : mesh.parts) {
                if (part.indexCount == 0u || part.firstIndex >= mesh.indices.size()) {
                    continue;
                }
                std::vector<std::uint32_t> remappedVertexIndices(
                    mesh.vertices.size(),
                    std::numeric_limits<std::uint32_t>::max());
                const std::size_t firstPartIndex = static_cast<std::size_t>(part.firstIndex);
                const std::size_t lastPartIndex = std::min(
                    firstPartIndex + static_cast<std::size_t>(part.indexCount),
                    mesh.indices.size());
                const std::uint32_t partFlags =
                    part.alphaTest ? kImportedSceneMaterialFlagAlphaTest : 0u;
                for (std::size_t indexOffset = firstPartIndex; indexOffset < lastPartIndex; ++indexOffset) {
                    const std::uint32_t index = mesh.indices[indexOffset];
                    if (index >= mesh.vertices.size()) {
                        continue;
                    }
                    std::uint32_t& remappedIndex = remappedVertexIndices[index];
                    if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                        remappedIndex = appendVertex(mesh.vertices[index], part.textureIndex, partFlags);
                    }
                    scene.packedIndices.push_back(remappedIndex);
                }
            }
        }
        const std::uint32_t indexCount = static_cast<std::uint32_t>(scene.packedIndices.size() - firstIndex);
        if (indexCount != 0u) {
            scene.packedDraws.push_back(ImportedScenePackedDraw{firstIndex, indexCount});
        }
    };

    if (!scene.meshes.empty()) {
        const ImportedSceneMesh& terrainMesh = scene.meshes.front();
        if (!terrainMesh.vertices.empty() && !terrainMesh.indices.empty()) {
            for (const ImportedSceneMeshPart& part : terrainMesh.parts) {
                if (part.indexCount == 0u || part.firstIndex >= terrainMesh.indices.size()) {
                    continue;
                }
                const std::uint32_t firstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
                std::vector<std::uint32_t> remappedVertexIndices(
                    terrainMesh.vertices.size(),
                    std::numeric_limits<std::uint32_t>::max());
                const std::size_t firstPartIndex = static_cast<std::size_t>(part.firstIndex);
                const std::size_t lastPartIndex = std::min(
                    firstPartIndex + static_cast<std::size_t>(part.indexCount),
                    terrainMesh.indices.size());
                for (std::size_t indexOffset = firstPartIndex; indexOffset < lastPartIndex; ++indexOffset) {
                    const std::uint32_t index = terrainMesh.indices[indexOffset];
                    if (index >= terrainMesh.vertices.size()) {
                        continue;
                    }
                    std::uint32_t& remappedIndex = remappedVertexIndices[index];
                    if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                        const ImportedSceneVertex& srcVertex = terrainMesh.vertices[index];
                        ImportedScenePackedVertex dstVertex{};
                        const std::array<float, 3> worldNormal = normalizeVector({
                            srcVertex.normal[0],
                            srcVertex.normal[1],
                            srcVertex.normal[2]
                        });
                        const PackedRenderColor color = packedTerrainColor(srcVertex.position[1]);
                        dstVertex.position[0] = srcVertex.position[0];
                        dstVertex.position[1] = srcVertex.position[1];
                        dstVertex.position[2] = srcVertex.position[2];
                        dstVertex.normal[0] = worldNormal[0];
                        dstVertex.normal[1] = worldNormal[1];
                        dstVertex.normal[2] = worldNormal[2];
                        dstVertex.color[0] = color.r;
                        dstVertex.color[1] = color.g;
                        dstVertex.color[2] = color.b;
                        dstVertex.uv[0] = srcVertex.uv[0];
                        dstVertex.uv[1] = srcVertex.uv[1];
                        dstVertex.textureIndex = part.textureIndex;
                        dstVertex.flags = 0u;
                        remappedIndex = static_cast<std::uint32_t>(scene.packedVertices.size());
                        scene.packedVertices.push_back(dstVertex);
                        expandBounds(dstVertex);
                    }
                    scene.packedIndices.push_back(remappedIndex);
                }
                const std::uint32_t indexCount =
                    static_cast<std::uint32_t>(scene.packedIndices.size() - firstIndex);
                if (indexCount != 0u) {
                    scene.packedDraws.push_back(ImportedScenePackedDraw{firstIndex, indexCount});
                }
            }
        }

        for (const ImportedSceneInstance& instance : scene.instances) {
            if (instance.meshIndex == 0u || instance.meshIndex >= scene.meshes.size()) {
                continue;
            }
            std::array<float, 16> transform{};
            std::copy(std::begin(instance.transform), std::end(instance.transform), transform.begin());
            appendMesh(
                scene.meshes[instance.meshIndex],
                transform,
                packedRenderColorFromHash(instance.modelPath));
        }
    }

    if (scene.packedVertices.empty()) {
        scene.boundsMin[0] = 0.0f;
        scene.boundsMin[1] = 0.0f;
        scene.boundsMin[2] = 0.0f;
        scene.boundsMax[0] = 0.0f;
        scene.boundsMax[1] = 0.0f;
        scene.boundsMax[2] = 0.0f;
    }
}

void computeImportedSceneBoundsFromPackedData(ImportedScene& scene) {
    if (scene.packedVertices.empty()) {
        scene.boundsMin[0] = 0.0f;
        scene.boundsMin[1] = 0.0f;
        scene.boundsMin[2] = 0.0f;
        scene.boundsMax[0] = 0.0f;
        scene.boundsMax[1] = 0.0f;
        scene.boundsMax[2] = 0.0f;
        return;
    }
    scene.boundsMin[0] = std::numeric_limits<float>::max();
    scene.boundsMin[1] = std::numeric_limits<float>::max();
    scene.boundsMin[2] = std::numeric_limits<float>::max();
    scene.boundsMax[0] = std::numeric_limits<float>::lowest();
    scene.boundsMax[1] = std::numeric_limits<float>::lowest();
    scene.boundsMax[2] = std::numeric_limits<float>::lowest();
    for (const ImportedScenePackedVertex& vertex : scene.packedVertices) {
        scene.boundsMin[0] = std::min(scene.boundsMin[0], vertex.position[0]);
        scene.boundsMin[1] = std::min(scene.boundsMin[1], vertex.position[1]);
        scene.boundsMin[2] = std::min(scene.boundsMin[2], vertex.position[2]);
        scene.boundsMax[0] = std::max(scene.boundsMax[0], vertex.position[0]);
        scene.boundsMax[1] = std::max(scene.boundsMax[1], vertex.position[1]);
        scene.boundsMax[2] = std::max(scene.boundsMax[2], vertex.position[2]);
    }
}

const std::string& getImportedSceneLastError() {
    return g_lastImportedSceneError;
}

bool loadMorrowindTexture(
    const std::filesystem::path& morrowindDataFilesPath,
    const std::string& sourcePath,
    ImportedSceneTexture& outTexture
) {
    g_lastImportedSceneError.clear();
    outTexture = ImportedSceneTexture{};
    outTexture.sourcePath = normalizeTextureRelativePath(sourcePath);

    static std::filesystem::path cachedDataFilesPath;
    static std::unordered_map<std::string, std::filesystem::path> cachedDataFileIndex;
    if (cachedDataFileIndex.empty() || cachedDataFilesPath != morrowindDataFilesPath) {
        cachedDataFilesPath = morrowindDataFilesPath;
        cachedDataFileIndex = buildCaseInsensitiveFileIndex(morrowindDataFilesPath);
    }
    const std::optional<std::filesystem::path> texturePath =
        resolveTextureFilePath(cachedDataFileIndex, outTexture.sourcePath);
    if (!texturePath.has_value()) {
        setLastImportedSceneError("Morrowind texture file not found: " + sourcePath);
        return false;
    }
    if (!loadTextureRgba(*texturePath, outTexture.width, outTexture.height, outTexture.rgba8)) {
        setLastImportedSceneError("Failed to decode Morrowind texture: " + texturePath->string());
        return false;
    }
    generateTextureMipChain(outTexture.width, outTexture.height, outTexture.rgba8, outTexture.mipLevelCount);
    if (outTexture.rgba8.empty() || outTexture.mipLevelCount == 0u) {
        setLastImportedSceneError("Failed to generate mip chain for Morrowind texture: " + texturePath->string());
        return false;
    }
    return true;
}

bool saveImportedScene(const ImportedScene& scene, const std::filesystem::path& outputPath) {
    g_lastImportedSceneError.clear();
    const std::filesystem::path parentPath = outputPath.parent_path();
    if (!parentPath.empty()) {
        std::error_code mkdirError;
        std::filesystem::create_directories(parentPath, mkdirError);
        if (mkdirError) {
            setLastImportedSceneError(
                "Failed to create output directory " + parentPath.string() + ": " + mkdirError.message());
            return false;
        }
    }

    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        setLastImportedSceneError("Failed to open output file for writing: " + outputPath.string());
        return false;
    }

    writeValue(output, kImportedSceneMagic);
    writeValue(output, kImportedSceneVersion);
    writeString(output, scene.sourceTag);

    const std::uint32_t textureCount = static_cast<std::uint32_t>(scene.textures.size());
    const std::uint32_t meshCount = static_cast<std::uint32_t>(scene.meshes.size());
    const std::uint32_t instanceCount = static_cast<std::uint32_t>(scene.instances.size());
    const std::uint32_t landscapeCellCount = static_cast<std::uint32_t>(scene.landscapeCells.size());
    const std::uint32_t waterPatchCount = static_cast<std::uint32_t>(scene.waterPatches.size());
    const std::uint32_t lightCount = static_cast<std::uint32_t>(scene.lights.size());
    const std::uint32_t unresolvedRefCount = static_cast<std::uint32_t>(scene.unresolvedRefs.size());
    const std::uint32_t packedVertexCount = static_cast<std::uint32_t>(scene.packedVertices.size());
    const std::uint32_t packedIndexCount = static_cast<std::uint32_t>(scene.packedIndices.size());
    const std::uint32_t packedDrawCount = static_cast<std::uint32_t>(scene.packedDraws.size());
    writeValue(output, textureCount);
    writeValue(output, meshCount);
    writeValue(output, instanceCount);
    writeValue(output, landscapeCellCount);
    writeValue(output, waterPatchCount);
    writeValue(output, lightCount);
    writeValue(output, unresolvedRefCount);
    writeValue(output, packedVertexCount);
    writeValue(output, packedIndexCount);
    writeValue(output, packedDrawCount);
    output.write(reinterpret_cast<const char*>(scene.boundsMin), static_cast<std::streamsize>(sizeof(scene.boundsMin)));
    output.write(reinterpret_cast<const char*>(scene.boundsMax), static_cast<std::streamsize>(sizeof(scene.boundsMax)));

    for (const ImportedSceneTexture& texture : scene.textures) {
        writeString(output, texture.sourcePath);
        writeValue(output, texture.width);
        writeValue(output, texture.height);
        writeValue(output, texture.mipLevelCount);
        const std::uint32_t rgbaSize = static_cast<std::uint32_t>(texture.rgba8.size());
        writeValue(output, rgbaSize);
        if (!texture.rgba8.empty()) {
            output.write(reinterpret_cast<const char*>(texture.rgba8.data()), static_cast<std::streamsize>(texture.rgba8.size()));
        }
    }

    for (const ImportedSceneMesh& mesh : scene.meshes) {
        writeString(output, mesh.name);
        const std::uint32_t vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
        const std::uint32_t indexCount = static_cast<std::uint32_t>(mesh.indices.size());
        const std::uint32_t partCount = static_cast<std::uint32_t>(mesh.parts.size());
        writeValue(output, vertexCount);
        writeValue(output, indexCount);
        writeValue(output, partCount);
        if (!mesh.vertices.empty()) {
            output.write(reinterpret_cast<const char*>(mesh.vertices.data()),
                         static_cast<std::streamsize>(mesh.vertices.size() * sizeof(ImportedSceneVertex)));
        }
        if (!mesh.indices.empty()) {
            output.write(reinterpret_cast<const char*>(mesh.indices.data()),
                         static_cast<std::streamsize>(mesh.indices.size() * sizeof(std::uint32_t)));
        }
        if (!mesh.parts.empty()) {
            output.write(reinterpret_cast<const char*>(mesh.parts.data()),
                         static_cast<std::streamsize>(mesh.parts.size() * sizeof(ImportedSceneMeshPart)));
        }
    }

    for (const ImportedSceneInstance& instance : scene.instances) {
        writeValue(output, instance.meshIndex);
        output.write(reinterpret_cast<const char*>(instance.transform), static_cast<std::streamsize>(sizeof(instance.transform)));
        writeString(output, instance.sourceId);
        writeString(output, instance.modelPath);
    }

    for (const ImportedSceneLandscapeCell& cell : scene.landscapeCells) {
        writeValue(output, cell.gridX);
        writeValue(output, cell.gridY);
        const std::uint32_t heightCount = static_cast<std::uint32_t>(cell.heights.size());
        const std::uint32_t textureIndexCount = static_cast<std::uint32_t>(cell.textureIndices.size());
        writeValue(output, heightCount);
        writeValue(output, textureIndexCount);
        if (!cell.heights.empty()) {
            output.write(reinterpret_cast<const char*>(cell.heights.data()),
                         static_cast<std::streamsize>(cell.heights.size() * sizeof(float)));
        }
        if (!cell.textureIndices.empty()) {
            output.write(reinterpret_cast<const char*>(cell.textureIndices.data()),
                         static_cast<std::streamsize>(cell.textureIndices.size() * sizeof(std::uint16_t)));
        }
    }

    if (!scene.waterPatches.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.waterPatches.data()),
            static_cast<std::streamsize>(scene.waterPatches.size() * sizeof(ImportedSceneWaterPatch)));
    }

    for (const ImportedSceneLight& light : scene.lights) {
        writeString(output, light.sourceId);
        output.write(reinterpret_cast<const char*>(light.position), static_cast<std::streamsize>(sizeof(light.position)));
        output.write(reinterpret_cast<const char*>(light.color), static_cast<std::streamsize>(sizeof(light.color)));
        writeValue(output, light.radius);
        writeValue(output, light.intensity);
        writeValue(output, light.flags);
    }

    for (const ImportedSceneCellRef& ref : scene.unresolvedRefs) {
        writeString(output, ref.refId);
        writeString(output, ref.modelPath);
        output.write(reinterpret_cast<const char*>(ref.position), static_cast<std::streamsize>(sizeof(ref.position)));
        output.write(reinterpret_cast<const char*>(ref.rotationRadians), static_cast<std::streamsize>(sizeof(ref.rotationRadians)));
        writeValue(output, ref.scale);
    }

    if (!scene.packedVertices.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.packedVertices.data()),
            static_cast<std::streamsize>(scene.packedVertices.size() * sizeof(ImportedScenePackedVertex)));
    }
    if (!scene.packedIndices.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.packedIndices.data()),
            static_cast<std::streamsize>(scene.packedIndices.size() * sizeof(std::uint32_t)));
    }
    if (!scene.packedDraws.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.packedDraws.data()),
            static_cast<std::streamsize>(scene.packedDraws.size() * sizeof(ImportedScenePackedDraw)));
    }

    if (!output.good()) {
        setLastImportedSceneError("Failed while writing output file: " + outputPath.string());
        return false;
    }
    return true;
}

bool loadImportedScene(const std::filesystem::path& inputPath, ImportedScene& outScene) {
    g_lastImportedSceneError.clear();
    std::ifstream input(inputPath, std::ios::binary);
    if (!input) {
        setLastImportedSceneError("Failed to open imported scene file: " + inputPath.string());
        return false;
    }

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!readValue(input, magic) || !readValue(input, version) || magic != kImportedSceneMagic) {
        setLastImportedSceneError("Invalid imported scene file header: " + inputPath.string());
        return false;
    }
    if (version < kMinSupportedImportedSceneVersion || version > kImportedSceneVersion) {
        setLastImportedSceneError(
            "Imported scene file version " + std::to_string(version) +
            " is unsupported; recook with the current odai_balmora_cooker (supported versions " +
            std::to_string(kMinSupportedImportedSceneVersion) + "-" +
            std::to_string(kImportedSceneVersion) + ")");
        return false;
    }

    ImportedScene scene{};
    scene.sourceFileVersion = version;
    if (!readString(input, scene.sourceTag)) {
        setLastImportedSceneError("Failed to read imported scene source tag: " + inputPath.string());
        return false;
    }

    std::uint32_t textureCount = 0;
    std::uint32_t meshCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t landscapeCellCount = 0;
    std::uint32_t waterPatchCount = 0;
    std::uint32_t lightCount = 0;
    std::uint32_t unresolvedRefCount = 0;
    std::uint32_t packedVertexCount = 0;
    std::uint32_t packedIndexCount = 0;
    std::uint32_t packedDrawCount = 0;
    if (!readValue(input, textureCount) ||
        !readValue(input, meshCount) ||
        !readValue(input, instanceCount) ||
        !readValue(input, landscapeCellCount) ||
        !readValue(input, waterPatchCount) ||
        !readValue(input, lightCount) ||
        !readValue(input, unresolvedRefCount)) {
        return false;
    }
    if (version >= 2u &&
        (!readValue(input, packedVertexCount) ||
         !readValue(input, packedIndexCount) ||
         !readValue(input, packedDrawCount))) {
        return false;
    }
    if (version >= 3u &&
        (!readExact(input, scene.boundsMin, sizeof(scene.boundsMin)) ||
         !readExact(input, scene.boundsMax, sizeof(scene.boundsMax)))) {
        return false;
    }
    scene.sourceTextureCount = textureCount;
    scene.sourceMeshCount = meshCount;
    scene.sourceInstanceCount = instanceCount;
    scene.sourceLandscapeCellCount = landscapeCellCount;
    scene.sourceWaterPatchCount = waterPatchCount;
    scene.sourceLightCount = lightCount;
    scene.sourceUnresolvedRefCount = unresolvedRefCount;

    scene.textures.resize(textureCount);
    for (ImportedSceneTexture& texture : scene.textures) {
        if (!readString(input, texture.sourcePath) ||
            !readValue(input, texture.width) ||
            !readValue(input, texture.height) ||
            !readValue(input, texture.mipLevelCount)) {
            return false;
        }
        std::uint32_t rgbaSize = 0;
        if (!readValue(input, rgbaSize)) {
            return false;
        }
        texture.rgba8.resize(rgbaSize);
        if (rgbaSize != 0 && !readExact(input, texture.rgba8.data(), rgbaSize)) {
            return false;
        }
    }

    scene.meshes.resize(meshCount);
    for (ImportedSceneMesh& mesh : scene.meshes) {
        std::uint32_t vertexCount = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t partCount = 0;
        if (!readString(input, mesh.name) ||
            !readValue(input, vertexCount) ||
            !readValue(input, indexCount) ||
            !readValue(input, partCount)) {
            return false;
        }
        mesh.vertices.resize(vertexCount);
        mesh.indices.resize(indexCount);
        mesh.parts.resize(partCount);
        if (vertexCount != 0 &&
            !readExact(input, mesh.vertices.data(), mesh.vertices.size() * sizeof(ImportedSceneVertex))) {
            return false;
        }
        if (indexCount != 0 &&
            !readExact(input, mesh.indices.data(), mesh.indices.size() * sizeof(std::uint32_t))) {
            return false;
        }
        if (partCount != 0 &&
            !readExact(input, mesh.parts.data(), mesh.parts.size() * sizeof(ImportedSceneMeshPart))) {
            return false;
        }
    }

    scene.instances.resize(instanceCount);
    for (ImportedSceneInstance& instance : scene.instances) {
        if (!readValue(input, instance.meshIndex) ||
            !readExact(input, instance.transform, sizeof(instance.transform)) ||
            !readString(input, instance.sourceId) ||
            !readString(input, instance.modelPath)) {
            return false;
        }
    }

    scene.landscapeCells.resize(landscapeCellCount);
    for (ImportedSceneLandscapeCell& cell : scene.landscapeCells) {
        std::uint32_t heightCount = 0;
        std::uint32_t textureIndexCount = 0;
        if (!readValue(input, cell.gridX) ||
            !readValue(input, cell.gridY) ||
            !readValue(input, heightCount) ||
            !readValue(input, textureIndexCount)) {
            return false;
        }
        cell.heights.resize(heightCount);
        cell.textureIndices.resize(textureIndexCount);
        if (heightCount != 0 &&
            !readExact(input, cell.heights.data(), cell.heights.size() * sizeof(float))) {
            return false;
        }
        if (textureIndexCount != 0 &&
            !readExact(input, cell.textureIndices.data(), cell.textureIndices.size() * sizeof(std::uint16_t))) {
            return false;
        }
    }

    scene.waterPatches.resize(waterPatchCount);
    if (waterPatchCount != 0 &&
        !readExact(input, scene.waterPatches.data(), scene.waterPatches.size() * sizeof(ImportedSceneWaterPatch))) {
        return false;
    }

    scene.lights.resize(lightCount);
    for (ImportedSceneLight& light : scene.lights) {
        if (!readString(input, light.sourceId) ||
            !readExact(input, light.position, sizeof(light.position)) ||
            !readExact(input, light.color, sizeof(light.color)) ||
            !readValue(input, light.radius) ||
            !readValue(input, light.intensity) ||
            !readValue(input, light.flags)) {
            return false;
        }
        if (version < 16u) {
            const float morrowindY = light.position[1];
            light.position[1] = light.position[2];
            light.position[2] = morrowindY;
        }
    }

    scene.unresolvedRefs.resize(unresolvedRefCount);
    for (ImportedSceneCellRef& ref : scene.unresolvedRefs) {
        if (!readString(input, ref.refId) ||
            !readString(input, ref.modelPath) ||
            !readExact(input, ref.position, sizeof(ref.position)) ||
            !readExact(input, ref.rotationRadians, sizeof(ref.rotationRadians)) ||
            !readValue(input, ref.scale)) {
            return false;
        }
    }

    if (version >= 2u) {
        scene.packedVertices.resize(packedVertexCount);
        scene.packedIndices.resize(packedIndexCount);
        scene.packedDraws.resize(packedDrawCount);
        if (packedVertexCount != 0 &&
            !readExact(input, scene.packedVertices.data(), scene.packedVertices.size() * sizeof(ImportedScenePackedVertex))) {
            return false;
        }
        if (packedIndexCount != 0 &&
            !readExact(input, scene.packedIndices.data(), scene.packedIndices.size() * sizeof(std::uint32_t))) {
            return false;
        }
        if (packedDrawCount != 0 &&
            !readExact(input, scene.packedDraws.data(), scene.packedDraws.size() * sizeof(ImportedScenePackedDraw))) {
            return false;
        }
    } else {
        buildImportedScenePackedRenderData(scene);
    }

    applyTextureAlphaCutoutFlags(scene);
    outScene = std::move(scene);
    return true;
}

bool loadImportedSceneRuntime(const std::filesystem::path& inputPath, ImportedScene& outScene) {
    g_lastImportedSceneError.clear();
    std::ifstream input(inputPath, std::ios::binary);
    if (!input) {
        setLastImportedSceneError("Failed to open imported scene file: " + inputPath.string());
        return false;
    }

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!readValue(input, magic) || !readValue(input, version) || magic != kImportedSceneMagic) {
        setLastImportedSceneError("Invalid imported scene file header: " + inputPath.string());
        return false;
    }
    if (version < kMinSupportedImportedSceneVersion || version > kImportedSceneVersion) {
        setLastImportedSceneError(
            "Imported scene file version " + std::to_string(version) +
            " is unsupported; recook with the current odai_balmora_cooker (supported versions " +
            std::to_string(kMinSupportedImportedSceneVersion) + "-" +
            std::to_string(kImportedSceneVersion) + ")");
        return false;
    }

    ImportedScene scene{};
    scene.sourceFileVersion = version;
    if (!readString(input, scene.sourceTag)) {
        return false;
    }

    std::uint32_t textureCount = 0;
    std::uint32_t meshCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t landscapeCellCount = 0;
    std::uint32_t waterPatchCount = 0;
    std::uint32_t lightCount = 0;
    std::uint32_t unresolvedRefCount = 0;
    std::uint32_t packedVertexCount = 0;
    std::uint32_t packedIndexCount = 0;
    std::uint32_t packedDrawCount = 0;
    if (!readValue(input, textureCount) ||
        !readValue(input, meshCount) ||
        !readValue(input, instanceCount) ||
        !readValue(input, landscapeCellCount) ||
        !readValue(input, waterPatchCount) ||
        !readValue(input, lightCount) ||
        !readValue(input, unresolvedRefCount) ||
        !readValue(input, packedVertexCount) ||
        !readValue(input, packedIndexCount) ||
        !readValue(input, packedDrawCount)) {
        return false;
    }
    if (version >= 3u &&
        (!readExact(input, scene.boundsMin, sizeof(scene.boundsMin)) ||
         !readExact(input, scene.boundsMax, sizeof(scene.boundsMax)))) {
        return false;
    }
    scene.sourceTextureCount = textureCount;
    scene.sourceMeshCount = meshCount;
    scene.sourceInstanceCount = instanceCount;
    scene.sourceLandscapeCellCount = landscapeCellCount;
    scene.sourceWaterPatchCount = waterPatchCount;
    scene.sourceLightCount = lightCount;
    scene.sourceUnresolvedRefCount = unresolvedRefCount;

    scene.textures.resize(textureCount);
    for (ImportedSceneTexture& texture : scene.textures) {
        std::uint32_t rgbaSize = 0;
        if (!readString(input, texture.sourcePath) ||
            !readValue(input, texture.width) ||
            !readValue(input, texture.height) ||
            !readValue(input, texture.mipLevelCount) ||
            !readValue(input, rgbaSize)) {
            return false;
        }
        texture.rgba8.resize(rgbaSize);
        if (rgbaSize != 0 &&
            !readExact(input, texture.rgba8.data(), texture.rgba8.size())) {
            return false;
        }
    }

    for (std::uint32_t i = 0; i < meshCount; ++i) {
        std::uint32_t vertexCount = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t partCount = 0;
        if (!skipString(input) ||
            !readValue(input, vertexCount) ||
            !readValue(input, indexCount) ||
            !readValue(input, partCount)) {
            return false;
        }
        const std::size_t vertexBytes = static_cast<std::size_t>(vertexCount) * sizeof(ImportedSceneVertex);
        const std::size_t indexBytes = static_cast<std::size_t>(indexCount) * sizeof(std::uint32_t);
        const std::size_t partBytes = static_cast<std::size_t>(partCount) * sizeof(ImportedSceneMeshPart);
        if ((vertexBytes != 0 && !skipExact(input, vertexBytes)) ||
            (indexBytes != 0 && !skipExact(input, indexBytes)) ||
            (partBytes != 0 && !skipExact(input, partBytes))) {
            return false;
        }
    }

    for (std::uint32_t i = 0; i < instanceCount; ++i) {
        std::uint32_t meshIndex = 0;
        float transform[16] = {};
        if (!readValue(input, meshIndex) ||
            !readExact(input, transform, sizeof(transform)) ||
            !skipString(input) ||
            !skipString(input)) {
            return false;
        }
    }

    for (std::uint32_t i = 0; i < landscapeCellCount; ++i) {
        int gridX = 0;
        int gridY = 0;
        std::uint32_t heightCount = 0;
        std::uint32_t textureIndexCount = 0;
        if (!readValue(input, gridX) ||
            !readValue(input, gridY) ||
            !readValue(input, heightCount) ||
            !readValue(input, textureIndexCount)) {
            return false;
        }
        const std::size_t heightBytes = static_cast<std::size_t>(heightCount) * sizeof(float);
        const std::size_t textureIndexBytes = static_cast<std::size_t>(textureIndexCount) * sizeof(std::uint16_t);
        if ((heightBytes != 0 && !skipExact(input, heightBytes)) ||
            (textureIndexBytes != 0 && !skipExact(input, textureIndexBytes))) {
            return false;
        }
    }

    scene.waterPatches.resize(waterPatchCount);
    if (waterPatchCount != 0 &&
        !readExact(input, scene.waterPatches.data(), scene.waterPatches.size() * sizeof(ImportedSceneWaterPatch))) {
        return false;
    }

    scene.lights.resize(lightCount);
    for (ImportedSceneLight& light : scene.lights) {
        if (!readString(input, light.sourceId) ||
            !readExact(input, light.position, sizeof(light.position)) ||
            !readExact(input, light.color, sizeof(light.color)) ||
            !readValue(input, light.radius) ||
            !readValue(input, light.intensity) ||
            !readValue(input, light.flags)) {
            return false;
        }
        if (version < 16u) {
            const float morrowindY = light.position[1];
            light.position[1] = light.position[2];
            light.position[2] = morrowindY;
        }
    }

    for (std::uint32_t i = 0; i < unresolvedRefCount; ++i) {
        float position[3] = {};
        float rotation[3] = {};
        float scale = 1.0f;
        if (!skipString(input) ||
            !skipString(input) ||
            !readExact(input, position, sizeof(position)) ||
            !readExact(input, rotation, sizeof(rotation)) ||
            !readValue(input, scale)) {
            return false;
        }
    }

    scene.packedVertices.resize(packedVertexCount);
    scene.packedIndices.resize(packedIndexCount);
    scene.packedDraws.resize(packedDrawCount);
    if ((packedVertexCount != 0 &&
         !readExact(input, scene.packedVertices.data(), scene.packedVertices.size() * sizeof(ImportedScenePackedVertex))) ||
        (packedIndexCount != 0 &&
         !readExact(input, scene.packedIndices.data(), scene.packedIndices.size() * sizeof(std::uint32_t))) ||
        (packedDrawCount != 0 &&
         !readExact(input, scene.packedDraws.data(), scene.packedDraws.size() * sizeof(ImportedScenePackedDraw)))) {
        return false;
    }
    if (version < 3u) {
        computeImportedSceneBoundsFromPackedData(scene);
    }

    applyTextureAlphaCutoutFlags(scene);
    outScene = std::move(scene);
    return true;
}

bool exportImportedSceneTerrainObj(const ImportedScene& scene, const std::filesystem::path& outputObjPath) {
    g_lastImportedSceneError.clear();
    if (scene.meshes.empty()) {
        setLastImportedSceneError("Scene does not contain any meshes to export");
        return false;
    }
    const ImportedSceneMesh& mesh = scene.meshes.front();
    const std::filesystem::path parentPath = outputObjPath.parent_path();
    if (!parentPath.empty()) {
        std::error_code mkdirError;
        std::filesystem::create_directories(parentPath, mkdirError);
        if (mkdirError) {
            setLastImportedSceneError(
                "Failed to create OBJ output directory " + parentPath.string() + ": " + mkdirError.message());
            return false;
        }
    }
    std::ofstream output(outputObjPath, std::ios::trunc);
    if (!output) {
        setLastImportedSceneError("Failed to open OBJ output file for writing: " + outputObjPath.string());
        return false;
    }
    output << "o terrain\n";
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        output << "v " << vertex.position[0] << " " << vertex.position[1] << " " << vertex.position[2] << "\n";
    }
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        output << "vn " << vertex.normal[0] << " " << vertex.normal[1] << " " << vertex.normal[2] << "\n";
    }
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        output << "vt " << vertex.uv[0] << " " << (1.0f - vertex.uv[1]) << "\n";
    }
    for (const ImportedSceneMeshPart& part : mesh.parts) {
        output << "g terrain_part_" << part.textureIndex << "_" << part.firstIndex << "\n";
        for (std::uint32_t i = 0; i + 2u < part.indexCount; i += 3u) {
            const std::uint32_t i0 = mesh.indices[part.firstIndex + i] + 1u;
            const std::uint32_t i1 = mesh.indices[part.firstIndex + i + 1u] + 1u;
            const std::uint32_t i2 = mesh.indices[part.firstIndex + i + 2u] + 1u;
            output << "f "
                   << i0 << "/" << i0 << "/" << i0 << " "
                   << i1 << "/" << i1 << "/" << i1 << " "
                   << i2 << "/" << i2 << "/" << i2 << "\n";
        }
    }
    if (!output.good()) {
        setLastImportedSceneError("Failed while writing OBJ output file: " + outputObjPath.string());
        return false;
    }
    return true;
}

bool cookMorrowindBalmoraScene(
    const std::filesystem::path& morrowindDataFilesPath,
    MorrowindBalmoraCookResult& outResult
) {
    g_lastImportedSceneError.clear();
    const std::filesystem::path esmPath = morrowindDataFilesPath / "Morrowind.esm";
    std::cerr << "[morrowind cooker] Data Files: " << morrowindDataFilesPath << "\n";
    std::cerr << "[morrowind cooker] ESM: " << esmPath << "\n";
    std::ifstream pass1(esmPath, std::ios::binary);
    if (!pass1) {
        setLastImportedSceneError("Failed to open Morrowind.esm");
        return false;
    }
    std::cerr << "[morrowind cooker] Pass 1: scanning CELL/LTEX/object records\n";

    std::unordered_map<std::string, std::string> modelPathById;
    std::unordered_map<std::string, ParsedLightRecord> lightById;
    std::unordered_map<std::uint32_t, std::string> landscapeTextureByIndex;
    std::vector<std::pair<int, int>> balmoraCells;

    Tes3RecordHeader header{};
    if (!readRecordHeader(pass1, header) || fourCcToString(header.name) != "TES3") {
        setLastImportedSceneError("File does not begin with a TES3 record");
        return false;
    }
    pass1.seekg(static_cast<std::streamoff>(header.size), std::ios::cur);

    std::size_t pass1RecordIndex = 0;
    while (readRecordHeader(pass1, header)) {
        ++pass1RecordIndex;
        const std::string recordName = fourCcToString(header.name);
        if ((pass1RecordIndex % 10000u) == 0u) {
            std::cerr << "[morrowind cooker] Pass 1 progress: record " << pass1RecordIndex
                      << " (" << recordName << ")\n";
        }
        if (recordName == "CELL") {
            CellSummary cell{};
            if (!parseCellSummary(pass1, header.size, cell)) {
                setLastImportedSceneError(
                    "Failed to parse CELL record during pass 1 at record " + std::to_string(pass1RecordIndex));
                return false;
            }
            if (cell.exterior && lowerCopy(cell.name) == "balmora") {
                balmoraCells.emplace_back(cell.gridX, cell.gridY);
                std::cerr << "[morrowind cooker] Found Balmora exterior cell at ("
                          << cell.gridX << ", " << cell.gridY << ")\n";
            }
        } else if (recordName == "LTEX") {
            std::uint32_t landscapeIndex = 0;
            std::string texturePath;
            if (!parseLandTextureRecord(pass1, header.size, landscapeIndex, texturePath)) {
                setLastImportedSceneError(
                    "Failed to parse LTEX record during pass 1 at record " + std::to_string(pass1RecordIndex));
                return false;
            }
            if (!texturePath.empty()) {
                landscapeTextureByIndex[landscapeIndex] = lowerCopy(texturePath);
            }
        } else if (recordName == "LIGH") {
            ParsedLightRecord light{};
            if (!parseLightRecord(pass1, header.size, light)) {
                setLastImportedSceneError(
                    "Failed to parse LIGH record during pass 1 at record " + std::to_string(pass1RecordIndex));
                return false;
            }
            if (!light.id.empty()) {
                lightById[light.id] = light;
                if (!light.modelPath.empty()) {
                    modelPathById[light.id] = light.modelPath;
                }
            }
        } else if (
            recordName == "STAT" || recordName == "DOOR" || recordName == "CONT" || recordName == "FLOR" ||
            recordName == "ACTI" || recordName == "FURN" || recordName == "MISC" ||
            recordName == "APPA" || recordName == "WEAP" || recordName == "ARMO" || recordName == "CLOT" ||
            recordName == "BOOK" || recordName == "INGR" || recordName == "ALCH" || recordName == "LOCK" ||
            recordName == "PROB" || recordName == "REPA") {
            std::string id;
            std::string modelPath;
            if (!parseModelRecord(pass1, header.size, id, modelPath)) {
                setLastImportedSceneError(
                    "Failed to parse " + recordName + " record during pass 1 at record " + std::to_string(pass1RecordIndex));
                return false;
            }
            if (!id.empty() && !modelPath.empty()) {
                modelPathById[id] = modelPath;
            }
        } else {
            pass1.seekg(static_cast<std::streamoff>(header.size), std::ios::cur);
        }
    }

    if (balmoraCells.empty()) {
        setLastImportedSceneError("Did not find any exterior CELL records named Balmora");
        return false;
    }

    std::cerr << "[morrowind cooker] Pass 1 complete: "
              << balmoraCells.size() << " Balmora cells, "
              << landscapeTextureByIndex.size() << " landscape textures, "
              << modelPathById.size() << " model records, "
              << lightById.size() << " light records\n";

    std::set<std::pair<int, int>> includedCellsSet;
    for (const std::pair<int, int>& cell : balmoraCells) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                includedCellsSet.emplace(cell.first + dx, cell.second + dy);
            }
        }
    }

    std::ifstream pass2(esmPath, std::ios::binary);
    if (!pass2 || !readRecordHeader(pass2, header) || fourCcToString(header.name) != "TES3") {
        setLastImportedSceneError("Failed to reopen or validate Morrowind.esm for pass 2");
        return false;
    }
    pass2.seekg(static_cast<std::streamoff>(header.size), std::ios::cur);
    std::cerr << "[morrowind cooker] Pass 2: collecting CELL refs and LAND records for "
              << includedCellsSet.size() << " included cells\n";

    std::vector<ParsedCell> parsedCells;
    std::vector<ParsedLand> parsedLands;
    std::size_t pass2RecordIndex = 0;
    while (readRecordHeader(pass2, header)) {
        ++pass2RecordIndex;
        const std::string recordName = fourCcToString(header.name);
        if ((pass2RecordIndex % 10000u) == 0u) {
            std::cerr << "[morrowind cooker] Pass 2 progress: record " << pass2RecordIndex
                      << " (" << recordName << "), collected "
                      << parsedCells.size() << " cells and "
                      << parsedLands.size() << " lands\n";
        }
        if (recordName == "CELL") {
            ParsedCell cell{};
            if (!parseCellRefs(pass2, header.size, cell)) {
                setLastImportedSceneError(
                    "Failed to parse CELL record during pass 2 at record " + std::to_string(pass2RecordIndex));
                return false;
            }
            if (includedCellsSet.contains({cell.gridX, cell.gridY})) {
                parsedCells.push_back(std::move(cell));
            }
        } else if (recordName == "LAND") {
            ParsedLand land{};
            if (!parseLandRecord(pass2, header.size, land)) {
                if (g_lastImportedSceneError.empty()) {
                    setLastImportedSceneError(
                        "Failed to parse LAND record during pass 2 at record " + std::to_string(pass2RecordIndex));
                } else {
                    setLastImportedSceneError(
                        g_lastImportedSceneError + " (pass 2 record " + std::to_string(pass2RecordIndex) + ")");
                }
                return false;
            }
            if (includedCellsSet.contains({land.gridX, land.gridY})) {
                parsedLands.push_back(std::move(land));
            }
        } else {
            pass2.seekg(static_cast<std::streamoff>(header.size), std::ios::cur);
        }
    }

    std::cerr << "[morrowind cooker] Pass 2 complete: "
              << parsedCells.size() << " parsed cells, "
              << parsedLands.size() << " land records\n";

    outResult.scene = buildSceneFromParsedData(
        morrowindDataFilesPath,
        landscapeTextureByIndex,
        modelPathById,
        lightById,
        parsedCells,
        parsedLands
    );
    std::cerr << "[morrowind cooker] Scene build complete: "
              << outResult.scene.meshes.size() << " meshes, "
              << outResult.scene.landscapeCells.size() << " landscape cells, "
              << outResult.scene.lights.size() << " lights, "
              << outResult.scene.unresolvedRefs.size() << " unresolved refs\n";
    outResult.balmoraCells = balmoraCells;
    outResult.includedCells.assign(includedCellsSet.begin(), includedCellsSet.end());
    outResult.modelPathById = std::move(modelPathById);
    outResult.texturePathByLandscapeIndex = std::move(landscapeTextureByIndex);
    return true;
}

}  // namespace odai::importer
