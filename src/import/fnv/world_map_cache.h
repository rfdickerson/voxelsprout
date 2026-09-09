#pragma once

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace odai::importer::fnv {

// A derived map raster, independent of streamed/cooked scene serialization.
struct WorldMapRaster {
    int minX = 0, minY = 0, cellsX = 0, cellsY = 0;
    int width = 0, height = 0;
    std::vector<float> heights;
    std::vector<std::uint8_t> water;
};

inline bool validWorldMapRaster(const WorldMapRaster& raster) {
    if (raster.cellsX <= 0 || raster.cellsY <= 0 || raster.cellsX > 256 || raster.cellsY > 256 ||
        raster.width != raster.cellsX * 8 || raster.height != raster.cellsY * 8 ||
        raster.minX < -100000 || raster.minX > 100000 || raster.minY < -100000 || raster.minY > 100000) return false;
    const auto size = std::size_t(raster.width) * raster.height;
    if (raster.heights.size() != size || raster.water.size() != size) return false;
    for (std::size_t i = 0; i < size; ++i)
        if (!std::isfinite(raster.heights[i]) || raster.water[i] > 1) return false;
    return true;
}

inline std::uint32_t worldMapChecksum(const std::vector<std::uint8_t>& bytes, std::size_t size) {
    std::uint32_t hash = 2166136261u;
    for (std::size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 16777619u;
    return hash;
}

inline bool loadWorldMapRaster(const std::filesystem::path& path, WorldMapRaster& output) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto length = file.tellg();
    if (length < 36 || length > 36 + 2048 * 2048 * 5) return false;
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), length)) return false;
    const auto word = [&](std::size_t at) {
        return std::uint32_t(bytes[at]) | (std::uint32_t(bytes[at+1]) << 8) |
            (std::uint32_t(bytes[at+2]) << 16) | (std::uint32_t(bytes[at+3]) << 24);
    };
    if (word(0) != 0x50414d4fu || word(4) != 1 || word(bytes.size()-4) != worldMapChecksum(bytes, bytes.size()-4)) return false;
    WorldMapRaster raster;
    raster.minX = std::bit_cast<std::int32_t>(word(8)); raster.minY = std::bit_cast<std::int32_t>(word(12));
    const auto cx = word(16), cy = word(20), width = word(24), height = word(28);
    if (!cx || !cy || cx > 256 || cy > 256 || width != cx * 8 || height != cy * 8) return false;
    const auto count = std::size_t(width) * height;
    if (bytes.size() != 36 + count * 5) return false;
    raster.cellsX = int(cx); raster.cellsY = int(cy); raster.width = int(width); raster.height = int(height);
    raster.heights.resize(count); raster.water.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        raster.heights[i] = std::bit_cast<float>(word(32 + i * 5));
        raster.water[i] = bytes[36 + i * 5];
    }
    if (!validWorldMapRaster(raster)) return false;
    output = std::move(raster);
    return true;
}

inline bool saveWorldMapRaster(const std::filesystem::path& path, const WorldMapRaster& raster) {
    if (!validWorldMapRaster(raster)) return false;
    std::vector<std::uint8_t> bytes;
    bytes.reserve(36 + raster.heights.size() * 5);
    const auto append = [&](std::uint32_t value) {
        for (int i = 0; i < 4; ++i) bytes.push_back(std::uint8_t(value >> (i * 8)));
    };
    append(0x50414d4fu); append(1);
    append(std::bit_cast<std::uint32_t>(std::int32_t(raster.minX)));
    append(std::bit_cast<std::uint32_t>(std::int32_t(raster.minY)));
    append(raster.cellsX); append(raster.cellsY); append(raster.width); append(raster.height);
    for (std::size_t i = 0; i < raster.heights.size(); ++i) {
        append(std::bit_cast<std::uint32_t>(raster.heights[i])); bytes.push_back(raster.water[i]);
    }
    append(worldMapChecksum(bytes, bytes.size()));
    auto temporary = path;
    temporary += ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    file.close();
    std::error_code error;
    if (file) std::filesystem::rename(temporary, path, error);
    if (!file || error) { std::filesystem::remove(temporary, error); return false; }
    return true;
}

} // namespace odai::importer::fnv
