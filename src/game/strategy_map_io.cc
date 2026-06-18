#include "game/strategy_map_io.h"

#include <cstdint>
#include <fstream>
#include <istream>
#include <ostream>
#include <type_traits>

namespace odai::game {

namespace {

constexpr std::uint32_t kStrategyMapMagic = 0x50414D53u;  // 'SMAP'
constexpr std::uint32_t kStrategyMapVersion = 1u;

std::string g_lastError;

void setLastError(std::string message) {
    g_lastError = std::move(message);
}

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

void writeString(std::ostream& output, const std::string& value) {
    writeValue(output, static_cast<std::uint32_t>(value.size()));
    if (!value.empty()) {
        output.write(value.data(), static_cast<std::streamsize>(value.size()));
    }
}

bool readString(std::istream& input, std::string& value) {
    std::uint32_t size = 0;
    if (!readValue(input, size)) {
        return false;
    }
    value.resize(size);
    if (size == 0) {
        return true;
    }
    input.read(value.data(), static_cast<std::streamsize>(size));
    return input.good();
}

}  // namespace

bool saveStrategyMap(const StrategyMap& map, const std::filesystem::path& outputPath) {
    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        setLastError("failed to open strategy map for writing: " + outputPath.string());
        return false;
    }

    writeValue(output, kStrategyMapMagic);
    writeValue(output, kStrategyMapVersion);
    writeValue(output, map.width);
    writeValue(output, map.height);
    writeValue(output, map.hexSize);
    writeValue(output, map.elevationStep);

    writeValue(output, static_cast<std::uint32_t>(map.tiles.size()));
    for (const MapTile& tile : map.tiles) {
        writeValue(output, static_cast<std::uint8_t>(tile.terrain));
        writeValue(output, tile.elevation);
        writeValue(output, tile.flags);
        writeValue(output, tile.owner);
        writeValue(output, static_cast<std::uint8_t>(tile.visibility));
    }

    writeValue(output, static_cast<std::uint32_t>(map.settlements.size()));
    for (const Settlement& settlement : map.settlements) {
        writeString(output, settlement.name);
        writeValue(output, settlement.col);
        writeValue(output, settlement.row);
        writeValue(output, settlement.tier);
        writeValue(output, settlement.owner);
    }

    if (!output.good()) {
        setLastError("write error while saving strategy map: " + outputPath.string());
        return false;
    }
    return true;
}

bool loadStrategyMap(const std::filesystem::path& inputPath, StrategyMap& outMap) {
    std::ifstream input(inputPath, std::ios::binary);
    if (!input) {
        setLastError("failed to open strategy map for reading: " + inputPath.string());
        return false;
    }

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!readValue(input, magic) || magic != kStrategyMapMagic) {
        setLastError("strategy map has bad magic: " + inputPath.string());
        return false;
    }
    if (!readValue(input, version) || version != kStrategyMapVersion) {
        setLastError("unsupported strategy map version in: " + inputPath.string());
        return false;
    }

    StrategyMap map{};
    if (!readValue(input, map.width) ||
        !readValue(input, map.height) ||
        !readValue(input, map.hexSize) ||
        !readValue(input, map.elevationStep)) {
        setLastError("truncated strategy map header: " + inputPath.string());
        return false;
    }

    std::uint32_t tileCount = 0;
    if (!readValue(input, tileCount)) {
        setLastError("truncated strategy map tile count: " + inputPath.string());
        return false;
    }
    if (tileCount != static_cast<std::uint64_t>(map.width) * static_cast<std::uint64_t>(map.height)) {
        setLastError("strategy map tile count does not match dimensions: " + inputPath.string());
        return false;
    }
    map.tiles.resize(tileCount);
    for (MapTile& tile : map.tiles) {
        std::uint8_t terrain = 0;
        std::uint8_t visibility = 0;
        if (!readValue(input, terrain) ||
            !readValue(input, tile.elevation) ||
            !readValue(input, tile.flags) ||
            !readValue(input, tile.owner) ||
            !readValue(input, visibility)) {
            setLastError("truncated strategy map tile record: " + inputPath.string());
            return false;
        }
        tile.terrain = static_cast<TerrainType>(terrain);
        tile.visibility = static_cast<TileVisibility>(visibility);
    }

    std::uint32_t settlementCount = 0;
    if (!readValue(input, settlementCount)) {
        setLastError("truncated strategy map settlement count: " + inputPath.string());
        return false;
    }
    map.settlements.resize(settlementCount);
    for (Settlement& settlement : map.settlements) {
        if (!readString(input, settlement.name) ||
            !readValue(input, settlement.col) ||
            !readValue(input, settlement.row) ||
            !readValue(input, settlement.tier) ||
            !readValue(input, settlement.owner)) {
            setLastError("truncated strategy map settlement record: " + inputPath.string());
            return false;
        }
    }

    outMap = std::move(map);
    return true;
}

const std::string& getStrategyMapLastError() {
    return g_lastError;
}

}  // namespace odai::game
