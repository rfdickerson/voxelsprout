#include "ui/vector/vector_cache_io.h"

#include <cstdint>
#include <fstream>
#include <istream>
#include <ostream>
#include <type_traits>

namespace odai::ui {

namespace {

constexpr std::uint32_t kVectorCacheMagic = 0x43455641u;  // 'AVEC' little-endian
constexpr std::uint32_t kVectorCacheVersion = 1u;

template <typename T>
void writeValue(std::ostream& output, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    output.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
}

template <typename T>
bool readValue(std::istream& input, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    return input.good();
}

}  // namespace

bool writeVectorCache(const std::filesystem::path& path, const UiGeometryBlock& block,
                      float sizePx) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    writeValue(out, kVectorCacheMagic);
    writeValue(out, kVectorCacheVersion);
    writeValue(out, sizePx);

    const auto vertexCount = static_cast<std::uint32_t>(block.vertices.size());
    const auto indexCount = static_cast<std::uint32_t>(block.indices.size());
    writeValue(out, vertexCount);
    writeValue(out, indexCount);
    if (vertexCount > 0) {
        out.write(reinterpret_cast<const char*>(block.vertices.data()),
                  static_cast<std::streamsize>(vertexCount * sizeof(UiVertex)));
    }
    if (indexCount > 0) {
        out.write(reinterpret_cast<const char*>(block.indices.data()),
                  static_cast<std::streamsize>(indexCount * sizeof(std::uint32_t)));
    }
    return out.good();
}

bool loadVectorCache(const std::filesystem::path& path, UiGeometryBlock& outBlock,
                     float& outSizePx) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!readValue(in, magic) || magic != kVectorCacheMagic) {
        return false;
    }
    if (!readValue(in, version) || version != kVectorCacheVersion) {
        return false;
    }
    if (!readValue(in, outSizePx)) {
        return false;
    }
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    if (!readValue(in, vertexCount) || !readValue(in, indexCount)) {
        return false;
    }

    outBlock.vertices.clear();
    outBlock.indices.clear();
    outBlock.commands.clear();
    outBlock.vertices.resize(vertexCount);
    outBlock.indices.resize(indexCount);
    if (vertexCount > 0) {
        in.read(reinterpret_cast<char*>(outBlock.vertices.data()),
                static_cast<std::streamsize>(vertexCount * sizeof(UiVertex)));
    }
    if (indexCount > 0) {
        in.read(reinterpret_cast<char*>(outBlock.indices.data()),
                static_cast<std::streamsize>(indexCount * sizeof(std::uint32_t)));
    }
    if (!in.good() && !in.eof()) {
        return false;
    }

    // Synthesize the single SolidColor command covering all indices.
    UiDrawCmd cmd{};
    cmd.indexOffset = 0;
    cmd.indexCount = indexCount;
    cmd.textureId = kUiNoTexture;
    outBlock.commands.push_back(cmd);
    return true;
}

}  // namespace odai::ui
