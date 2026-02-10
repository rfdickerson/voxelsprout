#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "core/Grid3.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkGrid.hpp"
#include "world/ChunkMesher.hpp"
#include "world/ClipmapIndex.hpp"
#include "world/Csg.hpp"

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

std::size_t countSolidCells(const world::CsgVolume& volume) {
    std::size_t count = 0;
    for (const world::CsgCell& cell : volume.cells()) {
        if (cell.voxel.type != world::VoxelType::Empty) {
            ++count;
        }
    }
    return count;
}

world::Chunk makePatternChunk() {
    world::Chunk chunk(0, 0, 0);
    for (int y = 0; y < world::Chunk::kSizeY; ++y) {
        for (int z = 0; z < world::Chunk::kSizeZ; ++z) {
            for (int x = 0; x < world::Chunk::kSizeX; ++x) {
                const std::uint32_t hx = static_cast<std::uint32_t>(x) * 73856093u;
                const std::uint32_t hy = static_cast<std::uint32_t>(y) * 19349663u;
                const std::uint32_t hz = static_cast<std::uint32_t>(z) * 83492791u;
                const std::uint32_t hash = hx ^ hy ^ hz;
                if ((hash % 7u) == 0u) {
                    continue;
                }

                world::VoxelType type = world::VoxelType::Stone;
                switch (hash % 6u) {
                case 0u: type = world::VoxelType::Stone; break;
                case 1u: type = world::VoxelType::Dirt; break;
                case 2u: type = world::VoxelType::Grass; break;
                case 3u: type = world::VoxelType::Wood; break;
                case 4u: type = world::VoxelType::SolidRed; break;
                default: type = world::VoxelType::Stone; break;
                }
                chunk.setVoxel(x, y, z, world::Voxel{type});
            }
        }
    }
    return chunk;
}

void expectMeshEqual(const world::ChunkMeshData& lhs, const world::ChunkMeshData& rhs) {
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
    const world::Chunk chunk = makePatternChunk();

    const world::MeshingOptions naive{world::MeshingMode::Naive};
    const world::MeshingOptions greedy{world::MeshingMode::Greedy};

    const world::ChunkMeshData naiveA = world::buildChunkMesh(chunk, naive);
    const world::ChunkMeshData naiveB = world::buildChunkMesh(chunk, naive);
    expectMeshEqual(naiveA, naiveB);

    const world::ChunkMeshData greedyA = world::buildChunkMesh(chunk, greedy);
    const world::ChunkMeshData greedyB = world::buildChunkMesh(chunk, greedy);
    expectMeshEqual(greedyA, greedyB);
}

TEST(ChunkMesherStability, GreedyMeshIsNotLargerThanNaive) {
    const world::Chunk chunk = makePatternChunk();

    const world::ChunkMeshData naive = world::buildChunkMesh(chunk, world::MeshingOptions{world::MeshingMode::Naive});
    const world::ChunkMeshData greedy = world::buildChunkMesh(chunk, world::MeshingOptions{world::MeshingMode::Greedy});

    EXPECT_LE(greedy.vertices.size(), naive.vertices.size());
    EXPECT_LE(greedy.indices.size(), naive.indices.size());
}

TEST(ClipmapIndexStability, StableCameraUpdatesDoNotDirtyBricks) {
    world::ChunkGrid grid;
    grid.initializeEmptyWorld();

    world::ChunkClipmapIndex clipmapIndex;
    clipmapIndex.rebuild(grid);
    ASSERT_TRUE(clipmapIndex.valid());

    world::SpatialQueryStats firstUpdate{};
    clipmapIndex.updateCamera(0.0f, 0.0f, 0.0f, &firstUpdate);
    EXPECT_GT(firstUpdate.clipmapUpdatedBrickCount, 0u);

    for (int i = 0; i < 5; ++i) {
        world::SpatialQueryStats stableUpdate{};
        clipmapIndex.updateCamera(0.25f, 0.25f, 0.25f, &stableUpdate);
        EXPECT_EQ(stableUpdate.clipmapUpdatedLevelCount, 0u);
        EXPECT_EQ(stableUpdate.clipmapUpdatedBrickCount, 0u);
    }
}

TEST(ClipmapIndexStability, QueryOutsideActiveBoundsReturnsNoChunks) {
    world::ChunkGrid grid;
    grid.initializeEmptyWorld();

    world::ChunkClipmapIndex clipmapIndex;
    clipmapIndex.rebuild(grid);
    ASSERT_TRUE(clipmapIndex.valid());
    clipmapIndex.updateCamera(0.0f, 0.0f, 0.0f, nullptr);

    core::CellAabb farBounds{};
    farBounds.valid = true;
    farBounds.minInclusive = core::Cell3i{100000, 100000, 100000};
    farBounds.maxExclusive = core::Cell3i{100032, 100032, 100032};

    world::SpatialQueryStats stats{};
    const std::vector<std::size_t> visible = clipmapIndex.queryChunksIntersecting(farBounds, &stats);
    EXPECT_TRUE(visible.empty());
    EXPECT_EQ(stats.visibleChunkCount, 0u);
}

TEST(CsgStability, OutOfBoundsCommandIsNoOp) {
    world::CsgVolume volume(8, 8, 8);
    const std::vector<world::CsgCell> before = volume.cells();

    world::CsgCommand command{};
    command.op = world::CsgOp::AddSolid;
    command.materialId = 3;
    command.brush.kind = world::BrushKind::Box;
    command.brush.minCell = core::Cell3i{100, 100, 100};
    command.brush.maxCell = core::Cell3i{104, 104, 104};

    const core::CellAabb touched = world::applyCsgCommand(volume, command);
    EXPECT_FALSE(touched.valid);
    EXPECT_EQ(countSolidCells(volume), 0u);
    EXPECT_EQ(volume.cells(), before);
}

TEST(CsgStability, CommandReplayIsDeterministic) {
    world::CsgVolume a(16, 16, 16);
    world::CsgVolume b(16, 16, 16);

    world::CsgCommand add{};
    add.op = world::CsgOp::AddSolid;
    add.materialId = 9;
    add.brush.kind = world::BrushKind::Box;
    add.brush.minCell = core::Cell3i{2, 1, 2};
    add.brush.maxCell = core::Cell3i{13, 6, 13};

    world::CsgCommand carve{};
    carve.op = world::CsgOp::SubtractSolid;
    carve.brush.kind = world::BrushKind::PrismPipe;
    carve.brush.axis = core::Dir6::PosX;
    carve.brush.minCell = core::Cell3i{2, 2, 2};
    carve.brush.maxCell = core::Cell3i{14, 5, 5};
    carve.brush.radiusQ8 = 160;

    world::CsgCommand paint{};
    paint.op = world::CsgOp::PaintMaterial;
    paint.materialId = 12;
    paint.affectMask = world::kCsgAffectSolid;
    paint.brush.kind = world::BrushKind::Ramp;
    paint.brush.axis = core::Dir6::PosZ;
    paint.brush.minCell = core::Cell3i{1, 0, 1};
    paint.brush.maxCell = core::Cell3i{15, 10, 15};

    const std::vector<world::CsgCommand> commands = {add, carve, paint};
    world::applyCsgCommands(a, commands);
    world::applyCsgCommands(b, commands);

    EXPECT_EQ(a.cells(), b.cells());
}

TEST(WorldBinaryStability, SaveLoadRoundTripPreservesVoxelTypes) {
    TempFileGuard tempFile(makeTempWorldPath("roundtrip"));
    removeFileIfExists(tempFile.path());

    world::ChunkGrid source;
    source.chunks().clear();
    source.chunks().emplace_back(2, 0, -3);
    world::Chunk& srcChunk = source.chunks().front();
    srcChunk.setVoxel(1, 1, 1, world::Voxel{world::VoxelType::Stone});
    srcChunk.setVoxel(2, 1, 1, world::Voxel{world::VoxelType::Dirt});
    srcChunk.setVoxel(3, 1, 1, world::Voxel{world::VoxelType::Grass});
    srcChunk.setVoxel(4, 1, 1, world::Voxel{world::VoxelType::Wood});
    srcChunk.setVoxel(5, 1, 1, world::Voxel{world::VoxelType::SolidRed});

    ASSERT_TRUE(source.saveToBinaryFile(tempFile.path()));

    world::ChunkGrid loaded;
    ASSERT_TRUE(loaded.loadFromBinaryFile(tempFile.path()));
    ASSERT_EQ(loaded.chunkCount(), 1u);
    ASSERT_EQ(loaded.chunks().front().chunkX(), 2);
    ASSERT_EQ(loaded.chunks().front().chunkY(), 0);
    ASSERT_EQ(loaded.chunks().front().chunkZ(), -3);

    const world::Chunk& dstChunk = loaded.chunks().front();
    EXPECT_EQ(dstChunk.voxelAt(1, 1, 1).type, world::VoxelType::Stone);
    EXPECT_EQ(dstChunk.voxelAt(2, 1, 1).type, world::VoxelType::Dirt);
    EXPECT_EQ(dstChunk.voxelAt(3, 1, 1).type, world::VoxelType::Grass);
    EXPECT_EQ(dstChunk.voxelAt(4, 1, 1).type, world::VoxelType::Wood);
    EXPECT_EQ(dstChunk.voxelAt(5, 1, 1).type, world::VoxelType::SolidRed);
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

    world::ChunkGrid loaded;
    EXPECT_FALSE(loaded.loadFromBinaryFile(tempFile.path()));
}
