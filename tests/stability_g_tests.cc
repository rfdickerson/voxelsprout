#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "core/grid3.h"
#include "world/chunk.h"
#include "world/chunk_grid.h"
#include "world/chunk_mesher.h"
#include "world/clipmap_index.h"
#include "world/csg.h"

namespace {

std::filesystem::path makeTempWorldPath(const char* tag) {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / ("voxel_" + std::string(tag) + "_" + std::to_string(stamp) + ".vxw");
}

void removeFileIfExists(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

class TempFileGuard {
public:
    explicit TempFileGuard(std::filesystem::path filePath) : m_path(std::move(filePath)) {}
    ~TempFileGuard() {
        removeFileIfExists(m_path);
    }

    const std::filesystem::path& path() const { return m_path; }

private:
    std::filesystem::path m_path;
};

std::size_t countSolidCells(const odai::world::CsgVolume& volume) {
    std::size_t count = 0;
    for (const odai::world::CsgCell& cell : volume.cells()) {
        if (cell.voxel.type != odai::world::VoxelType::Empty) {
            ++count;
        }
    }
    return count;
}

odai::world::Chunk makePatternChunk() {
    odai::world::Chunk chunk(0, 0, 0);
    for (int y = 0; y < odai::world::Chunk::kSizeY; ++y) {
        for (int z = 0; z < odai::world::Chunk::kSizeZ; ++z) {
            for (int x = 0; x < odai::world::Chunk::kSizeX; ++x) {
                const std::uint32_t hx = static_cast<std::uint32_t>(x) * 73856093u;
                const std::uint32_t hy = static_cast<std::uint32_t>(y) * 19349663u;
                const std::uint32_t hz = static_cast<std::uint32_t>(z) * 83492791u;
                const std::uint32_t hash = hx ^ hy ^ hz;
                if ((hash % 7u) == 0u) {
                    continue;
                }

                odai::world::VoxelType type = odai::world::VoxelType::Stone;
                switch (hash % 6u) {
                case 0u: type = odai::world::VoxelType::Stone; break;
                case 1u: type = odai::world::VoxelType::Dirt; break;
                case 2u: type = odai::world::VoxelType::Grass; break;
                case 3u: type = odai::world::VoxelType::Wood; break;
                case 4u: type = odai::world::VoxelType::SolidRed; break;
                default: type = odai::world::VoxelType::Stone; break;
                }
                chunk.setVoxel(x, y, z, odai::world::Voxel{type});
            }
        }
    }
    return chunk;
}

void expectMeshEqual(const odai::world::ChunkMeshData& lhs, const odai::world::ChunkMeshData& rhs) {
    ASSERT_EQ(lhs.vertices.size(), rhs.vertices.size());
    ASSERT_EQ(lhs.indices.size(), rhs.indices.size());
    for (std::size_t i = 0; i < lhs.vertices.size(); ++i) {
        EXPECT_EQ(lhs.vertices[i].bits, rhs.vertices[i].bits) << "vertex mismatch at " << i;
    }
    for (std::size_t i = 0; i < lhs.indices.size(); ++i) {
        EXPECT_EQ(lhs.indices[i], rhs.indices[i]) << "index mismatch at " << i;
    }
}

} // namespace

TEST(ChunkMesherStability, DeterministicOutputAcrossRuns) {
    const odai::world::Chunk chunk = makePatternChunk();

    const odai::world::MeshingOptions naive{odai::world::MeshingMode::Naive};
    const odai::world::MeshingOptions greedy{odai::world::MeshingMode::Greedy};

    const odai::world::ChunkMeshData naiveA = odai::world::buildChunkMesh(chunk, naive);
    const odai::world::ChunkMeshData naiveB = odai::world::buildChunkMesh(chunk, naive);
    expectMeshEqual(naiveA, naiveB);

    const odai::world::ChunkMeshData greedyA = odai::world::buildChunkMesh(chunk, greedy);
    const odai::world::ChunkMeshData greedyB = odai::world::buildChunkMesh(chunk, greedy);
    expectMeshEqual(greedyA, greedyB);
}

TEST(ChunkMesherStability, GreedyMeshIsNotLargerThanNaive) {
    const odai::world::Chunk chunk = makePatternChunk();

    const odai::world::ChunkMeshData naive = odai::world::buildChunkMesh(chunk, odai::world::MeshingOptions{odai::world::MeshingMode::Naive});
    const odai::world::ChunkMeshData greedy = odai::world::buildChunkMesh(chunk, odai::world::MeshingOptions{odai::world::MeshingMode::Greedy});

    EXPECT_LE(greedy.vertices.size(), naive.vertices.size());
    EXPECT_LE(greedy.indices.size(), naive.indices.size());
}

TEST(ClipmapIndexStability, StableCameraUpdatesDoNotDirtyBricks) {
    odai::world::ChunkGrid grid;
    grid.initializeEmptyWorld();

    odai::world::ChunkClipmapIndex clipmapIndex;
    clipmapIndex.rebuild(grid);
    ASSERT_TRUE(clipmapIndex.valid());

    odai::world::SpatialQueryStats firstUpdate{};
    clipmapIndex.updateCamera(0.0f, 0.0f, 0.0f, &firstUpdate);
    EXPECT_GT(firstUpdate.clipmapUpdatedBrickCount, 0u);

    for (int i = 0; i < 5; ++i) {
        odai::world::SpatialQueryStats stableUpdate{};
        clipmapIndex.updateCamera(0.25f, 0.25f, 0.25f, &stableUpdate);
        EXPECT_EQ(stableUpdate.clipmapUpdatedLevelCount, 0u);
        EXPECT_EQ(stableUpdate.clipmapUpdatedBrickCount, 0u);
    }
}

TEST(ClipmapIndexStability, QueryOutsideActiveBoundsReturnsNoChunks) {
    odai::world::ChunkGrid grid;
    grid.initializeEmptyWorld();

    odai::world::ChunkClipmapIndex clipmapIndex;
    clipmapIndex.rebuild(grid);
    ASSERT_TRUE(clipmapIndex.valid());
    clipmapIndex.updateCamera(0.0f, 0.0f, 0.0f, nullptr);

    odai::core::CellAabb farBounds{};
    farBounds.valid = true;
    farBounds.minInclusive = odai::core::Cell3i{100000, 100000, 100000};
    farBounds.maxExclusive = odai::core::Cell3i{100032, 100032, 100032};

    odai::world::SpatialQueryStats stats{};
    const std::vector<std::size_t> visible = clipmapIndex.queryChunksIntersecting(farBounds, &stats);
    EXPECT_TRUE(visible.empty());
    EXPECT_EQ(stats.visibleChunkCount, 0u);
}

TEST(CsgStability, OutOfBoundsCommandIsNoOp) {
    odai::world::CsgVolume volume(8, 8, 8);
    const std::vector<odai::world::CsgCell> before = volume.cells();

    odai::world::CsgCommand command{};
    command.op = odai::world::CsgOp::AddSolid;
    command.materialId = 3;
    command.brush.kind = odai::world::BrushKind::Box;
    command.brush.minCell = odai::core::Cell3i{100, 100, 100};
    command.brush.maxCell = odai::core::Cell3i{104, 104, 104};

    const odai::core::CellAabb touched = odai::world::applyCsgCommand(volume, command);
    EXPECT_FALSE(touched.valid);
    EXPECT_EQ(countSolidCells(volume), 0u);
    EXPECT_EQ(volume.cells(), before);
}

TEST(CsgStability, CommandReplayIsDeterministic) {
    odai::world::CsgVolume a(16, 16, 16);
    odai::world::CsgVolume b(16, 16, 16);

    odai::world::CsgCommand add{};
    add.op = odai::world::CsgOp::AddSolid;
    add.materialId = 9;
    add.brush.kind = odai::world::BrushKind::Box;
    add.brush.minCell = odai::core::Cell3i{2, 1, 2};
    add.brush.maxCell = odai::core::Cell3i{13, 6, 13};

    odai::world::CsgCommand carve{};
    carve.op = odai::world::CsgOp::SubtractSolid;
    carve.brush.kind = odai::world::BrushKind::PrismPipe;
    carve.brush.axis = odai::core::Dir6::PosX;
    carve.brush.minCell = odai::core::Cell3i{2, 2, 2};
    carve.brush.maxCell = odai::core::Cell3i{14, 5, 5};
    carve.brush.radiusQ8 = 160;

    odai::world::CsgCommand paint{};
    paint.op = odai::world::CsgOp::PaintMaterial;
    paint.materialId = 12;
    paint.affectMask = odai::world::kCsgAffectSolid;
    paint.brush.kind = odai::world::BrushKind::Ramp;
    paint.brush.axis = odai::core::Dir6::PosZ;
    paint.brush.minCell = odai::core::Cell3i{1, 0, 1};
    paint.brush.maxCell = odai::core::Cell3i{15, 10, 15};

    const std::vector<odai::world::CsgCommand> commands = {add, carve, paint};
    odai::world::applyCsgCommands(a, commands);
    odai::world::applyCsgCommands(b, commands);

    EXPECT_EQ(a.cells(), b.cells());
}

TEST(WorldBinaryStability, SaveLoadRoundTripPreservesVoxelTypes) {
    TempFileGuard tempFile(makeTempWorldPath("roundtrip"));
    removeFileIfExists(tempFile.path());

    odai::world::ChunkGrid source;
    source.chunks().clear();
    source.chunks().emplace_back(2, 0, -3);
    odai::world::Chunk& srcChunk = source.chunks().front();
    srcChunk.setVoxel(1, 1, 1, odai::world::Voxel{odai::world::VoxelType::Stone});
    srcChunk.setVoxel(2, 1, 1, odai::world::Voxel{odai::world::VoxelType::Dirt});
    srcChunk.setVoxel(3, 1, 1, odai::world::Voxel{odai::world::VoxelType::Grass});
    srcChunk.setVoxel(4, 1, 1, odai::world::Voxel{odai::world::VoxelType::Wood});
    srcChunk.setVoxel(5, 1, 1, odai::world::Voxel{odai::world::VoxelType::SolidRed});

    ASSERT_TRUE(source.saveToBinaryFile(tempFile.path()));

    odai::world::ChunkGrid loaded;
    ASSERT_TRUE(loaded.loadFromBinaryFile(tempFile.path()));
    ASSERT_EQ(loaded.chunkCount(), 1u);
    ASSERT_EQ(loaded.chunks().front().chunkX(), 2);
    ASSERT_EQ(loaded.chunks().front().chunkY(), 0);
    ASSERT_EQ(loaded.chunks().front().chunkZ(), -3);

    const odai::world::Chunk& dstChunk = loaded.chunks().front();
    EXPECT_EQ(dstChunk.voxelAt(1, 1, 1).type, odai::world::VoxelType::Stone);
    EXPECT_EQ(dstChunk.voxelAt(2, 1, 1).type, odai::world::VoxelType::Dirt);
    EXPECT_EQ(dstChunk.voxelAt(3, 1, 1).type, odai::world::VoxelType::Grass);
    EXPECT_EQ(dstChunk.voxelAt(4, 1, 1).type, odai::world::VoxelType::Wood);
    EXPECT_EQ(dstChunk.voxelAt(5, 1, 1).type, odai::world::VoxelType::SolidRed);
}

TEST(WorldBinaryStability, LoadRejectsInvalidMagicHeader) {
    TempFileGuard tempFile(makeTempWorldPath("bad_magic"));
    removeFileIfExists(tempFile.path());

    std::ofstream out(tempFile.path(), std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    const char badMagic[4] = {'N', 'O', 'P', 'E'};
    const std::uint32_t version = 2u;
    const std::uint32_t chunkCount = 1u;
    out.write(badMagic, sizeof(badMagic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&chunkCount), sizeof(chunkCount));
    out.close();

    odai::world::ChunkGrid loaded;
    EXPECT_FALSE(loaded.loadFromBinaryFile(tempFile.path()));
}
