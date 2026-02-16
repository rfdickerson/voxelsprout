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

std::size_t countSolidCells(const voxelsprout::world::CsgVolume& volume) {
    std::size_t count = 0;
    for (const voxelsprout::world::CsgCell& cell : volume.cells()) {
        if (cell.voxel.type != voxelsprout::world::VoxelType::Empty) {
            ++count;
        }
    }
    return count;
}

voxelsprout::world::Chunk makePatternChunk() {
    voxelsprout::world::Chunk chunk(0, 0, 0);
    for (int y = 0; y < voxelsprout::world::Chunk::kSizeY; ++y) {
        for (int z = 0; z < voxelsprout::world::Chunk::kSizeZ; ++z) {
            for (int x = 0; x < voxelsprout::world::Chunk::kSizeX; ++x) {
                const std::uint32_t hx = static_cast<std::uint32_t>(x) * 73856093u;
                const std::uint32_t hy = static_cast<std::uint32_t>(y) * 19349663u;
                const std::uint32_t hz = static_cast<std::uint32_t>(z) * 83492791u;
                const std::uint32_t hash = hx ^ hy ^ hz;
                if ((hash % 7u) == 0u) {
                    continue;
                }

                voxelsprout::world::VoxelType type = voxelsprout::world::VoxelType::Stone;
                switch (hash % 6u) {
                case 0u: type = voxelsprout::world::VoxelType::Stone; break;
                case 1u: type = voxelsprout::world::VoxelType::Dirt; break;
                case 2u: type = voxelsprout::world::VoxelType::Grass; break;
                case 3u: type = voxelsprout::world::VoxelType::Wood; break;
                case 4u: type = voxelsprout::world::VoxelType::SolidRed; break;
                default: type = voxelsprout::world::VoxelType::Stone; break;
                }
                chunk.setVoxel(x, y, z, voxelsprout::world::Voxel{type});
            }
        }
    }
    return chunk;
}

void expectMeshEqual(const voxelsprout::world::ChunkMeshData& lhs, const voxelsprout::world::ChunkMeshData& rhs) {
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
    const voxelsprout::world::Chunk chunk = makePatternChunk();

    const voxelsprout::world::MeshingOptions naive{voxelsprout::world::MeshingMode::Naive};
    const voxelsprout::world::MeshingOptions greedy{voxelsprout::world::MeshingMode::Greedy};

    const voxelsprout::world::ChunkMeshData naiveA = voxelsprout::world::buildChunkMesh(chunk, naive);
    const voxelsprout::world::ChunkMeshData naiveB = voxelsprout::world::buildChunkMesh(chunk, naive);
    expectMeshEqual(naiveA, naiveB);

    const voxelsprout::world::ChunkMeshData greedyA = voxelsprout::world::buildChunkMesh(chunk, greedy);
    const voxelsprout::world::ChunkMeshData greedyB = voxelsprout::world::buildChunkMesh(chunk, greedy);
    expectMeshEqual(greedyA, greedyB);
}

TEST(ChunkMesherStability, GreedyMeshIsNotLargerThanNaive) {
    const voxelsprout::world::Chunk chunk = makePatternChunk();

    const voxelsprout::world::ChunkMeshData naive = voxelsprout::world::buildChunkMesh(chunk, voxelsprout::world::MeshingOptions{voxelsprout::world::MeshingMode::Naive});
    const voxelsprout::world::ChunkMeshData greedy = voxelsprout::world::buildChunkMesh(chunk, voxelsprout::world::MeshingOptions{voxelsprout::world::MeshingMode::Greedy});

    EXPECT_LE(greedy.vertices.size(), naive.vertices.size());
    EXPECT_LE(greedy.indices.size(), naive.indices.size());
}

TEST(ClipmapIndexStability, StableCameraUpdatesDoNotDirtyBricks) {
    voxelsprout::world::ChunkGrid grid;
    grid.initializeEmptyWorld();

    voxelsprout::world::ChunkClipmapIndex clipmapIndex;
    clipmapIndex.rebuild(grid);
    ASSERT_TRUE(clipmapIndex.valid());

    voxelsprout::world::SpatialQueryStats firstUpdate{};
    clipmapIndex.updateCamera(0.0f, 0.0f, 0.0f, &firstUpdate);
    EXPECT_GT(firstUpdate.clipmapUpdatedBrickCount, 0u);

    for (int i = 0; i < 5; ++i) {
        voxelsprout::world::SpatialQueryStats stableUpdate{};
        clipmapIndex.updateCamera(0.25f, 0.25f, 0.25f, &stableUpdate);
        EXPECT_EQ(stableUpdate.clipmapUpdatedLevelCount, 0u);
        EXPECT_EQ(stableUpdate.clipmapUpdatedBrickCount, 0u);
    }
}

TEST(ClipmapIndexStability, QueryOutsideActiveBoundsReturnsNoChunks) {
    voxelsprout::world::ChunkGrid grid;
    grid.initializeEmptyWorld();

    voxelsprout::world::ChunkClipmapIndex clipmapIndex;
    clipmapIndex.rebuild(grid);
    ASSERT_TRUE(clipmapIndex.valid());
    clipmapIndex.updateCamera(0.0f, 0.0f, 0.0f, nullptr);

    voxelsprout::core::CellAabb farBounds{};
    farBounds.valid = true;
    farBounds.minInclusive = voxelsprout::core::Cell3i{100000, 100000, 100000};
    farBounds.maxExclusive = voxelsprout::core::Cell3i{100032, 100032, 100032};

    voxelsprout::world::SpatialQueryStats stats{};
    const std::vector<std::size_t> visible = clipmapIndex.queryChunksIntersecting(farBounds, &stats);
    EXPECT_TRUE(visible.empty());
    EXPECT_EQ(stats.visibleChunkCount, 0u);
}

TEST(CsgStability, OutOfBoundsCommandIsNoOp) {
    voxelsprout::world::CsgVolume volume(8, 8, 8);
    const std::vector<voxelsprout::world::CsgCell> before = volume.cells();

    voxelsprout::world::CsgCommand command{};
    command.op = voxelsprout::world::CsgOp::AddSolid;
    command.materialId = 3;
    command.brush.kind = voxelsprout::world::BrushKind::Box;
    command.brush.minCell = voxelsprout::core::Cell3i{100, 100, 100};
    command.brush.maxCell = voxelsprout::core::Cell3i{104, 104, 104};

    const voxelsprout::core::CellAabb touched = voxelsprout::world::applyCsgCommand(volume, command);
    EXPECT_FALSE(touched.valid);
    EXPECT_EQ(countSolidCells(volume), 0u);
    EXPECT_EQ(volume.cells(), before);
}

TEST(CsgStability, CommandReplayIsDeterministic) {
    voxelsprout::world::CsgVolume a(16, 16, 16);
    voxelsprout::world::CsgVolume b(16, 16, 16);

    voxelsprout::world::CsgCommand add{};
    add.op = voxelsprout::world::CsgOp::AddSolid;
    add.materialId = 9;
    add.brush.kind = voxelsprout::world::BrushKind::Box;
    add.brush.minCell = voxelsprout::core::Cell3i{2, 1, 2};
    add.brush.maxCell = voxelsprout::core::Cell3i{13, 6, 13};

    voxelsprout::world::CsgCommand carve{};
    carve.op = voxelsprout::world::CsgOp::SubtractSolid;
    carve.brush.kind = voxelsprout::world::BrushKind::PrismPipe;
    carve.brush.axis = voxelsprout::core::Dir6::PosX;
    carve.brush.minCell = voxelsprout::core::Cell3i{2, 2, 2};
    carve.brush.maxCell = voxelsprout::core::Cell3i{14, 5, 5};
    carve.brush.radiusQ8 = 160;

    voxelsprout::world::CsgCommand paint{};
    paint.op = voxelsprout::world::CsgOp::PaintMaterial;
    paint.materialId = 12;
    paint.affectMask = voxelsprout::world::kCsgAffectSolid;
    paint.brush.kind = voxelsprout::world::BrushKind::Ramp;
    paint.brush.axis = voxelsprout::core::Dir6::PosZ;
    paint.brush.minCell = voxelsprout::core::Cell3i{1, 0, 1};
    paint.brush.maxCell = voxelsprout::core::Cell3i{15, 10, 15};

    const std::vector<voxelsprout::world::CsgCommand> commands = {add, carve, paint};
    voxelsprout::world::applyCsgCommands(a, commands);
    voxelsprout::world::applyCsgCommands(b, commands);

    EXPECT_EQ(a.cells(), b.cells());
}

TEST(WorldBinaryStability, SaveLoadRoundTripPreservesVoxelTypes) {
    TempFileGuard tempFile(makeTempWorldPath("roundtrip"));
    removeFileIfExists(tempFile.path());

    voxelsprout::world::ChunkGrid source;
    source.chunks().clear();
    source.chunks().emplace_back(2, 0, -3);
    voxelsprout::world::Chunk& srcChunk = source.chunks().front();
    srcChunk.setVoxel(1, 1, 1, voxelsprout::world::Voxel{voxelsprout::world::VoxelType::Stone});
    srcChunk.setVoxel(2, 1, 1, voxelsprout::world::Voxel{voxelsprout::world::VoxelType::Dirt});
    srcChunk.setVoxel(3, 1, 1, voxelsprout::world::Voxel{voxelsprout::world::VoxelType::Grass});
    srcChunk.setVoxel(4, 1, 1, voxelsprout::world::Voxel{voxelsprout::world::VoxelType::Wood});
    srcChunk.setVoxel(5, 1, 1, voxelsprout::world::Voxel{voxelsprout::world::VoxelType::SolidRed});

    ASSERT_TRUE(source.saveToBinaryFile(tempFile.path()));

    voxelsprout::world::ChunkGrid loaded;
    ASSERT_TRUE(loaded.loadFromBinaryFile(tempFile.path()));
    ASSERT_EQ(loaded.chunkCount(), 1u);
    ASSERT_EQ(loaded.chunks().front().chunkX(), 2);
    ASSERT_EQ(loaded.chunks().front().chunkY(), 0);
    ASSERT_EQ(loaded.chunks().front().chunkZ(), -3);

    const voxelsprout::world::Chunk& dstChunk = loaded.chunks().front();
    EXPECT_EQ(dstChunk.voxelAt(1, 1, 1).type, voxelsprout::world::VoxelType::Stone);
    EXPECT_EQ(dstChunk.voxelAt(2, 1, 1).type, voxelsprout::world::VoxelType::Dirt);
    EXPECT_EQ(dstChunk.voxelAt(3, 1, 1).type, voxelsprout::world::VoxelType::Grass);
    EXPECT_EQ(dstChunk.voxelAt(4, 1, 1).type, voxelsprout::world::VoxelType::Wood);
    EXPECT_EQ(dstChunk.voxelAt(5, 1, 1).type, voxelsprout::world::VoxelType::SolidRed);
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

    voxelsprout::world::ChunkGrid loaded;
    EXPECT_FALSE(loaded.loadFromBinaryFile(tempFile.path()));
}
