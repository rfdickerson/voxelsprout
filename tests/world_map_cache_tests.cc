#include "import/fnv/world_map_cache.h"
#include <cassert>
#include <iostream>
#include <limits>

using namespace odai::importer::fnv;
int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("odai-map-cache-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    WorldMapRaster map;
    map.minX = -12; map.minY = 7; map.cellsX = 2; map.cellsY = 1; map.width = 16; map.height = 8;
    map.heights.resize(128); map.water.resize(128);
    for (int i = 0; i < 128; ++i) { map.heights[i] = float(i) * 0.25f - 16; map.water[i] = i % 2; }
    assert(saveWorldMapRaster(path, map));
    WorldMapRaster restored;
    assert(loadWorldMapRaster(path, restored));
    assert(restored.minX == -12 && restored.minY == 7 && restored.width == 16 && restored.height == 8);
    assert(restored.heights == map.heights && restored.water == map.water);
    // Corrupt data cannot partially replace a live raster.
    { std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary); file.seekp(45); file.put('\xff'); }
    assert(!loadWorldMapRaster(path, restored));
    assert(restored.heights == map.heights);
    assert(saveWorldMapRaster(path, map));
    std::filesystem::resize_file(path, 38);
    assert(!loadWorldMapRaster(path, restored));
    map.heights[0] = std::numeric_limits<float>::quiet_NaN();
    assert(!saveWorldMapRaster(path, map));
    map.heights[0] = 0; map.cellsX = 257;
    assert(!saveWorldMapRaster(path, map));
    std::filesystem::remove(path);
    assert(!loadWorldMapRaster(path, restored));
    std::cout << "World map cache tests passed\n";
}
