#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "core/Grid3.hpp"
#include "math/Math.hpp"
#include "sim/NetworkGraph.hpp"
#include "sim/NetworkProcedural.hpp"
#include "sim/Simulation.hpp"
#include "render/FrameArenaAlias.hpp"
#include "world/ClipmapIndex.hpp"
#include "world/Chunk.hpp"
#include "world/ChunkMesher.hpp"
#include "world/MagicaVoxel.hpp"
#include "world/Csg.hpp"

namespace {

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[foundation test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectNear(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) > epsilon) {
        std::cerr << "[foundation test] FAIL: " << message
                  << " (expected " << expected
                  << ", got " << actual << ")\n";
        ++g_failures;
    }
}

std::size_t countSolidCells(const world::CsgVolume& volume) {
    std::size_t count = 0;
    for (const world::CsgCell& cell : volume.cells()) {
        if (cell.voxel.type != world::VoxelType::Empty) {
            ++count;
        }
    }
    return count;
}

void testGridPrimitives() {
    const core::Cell3i start{10, 5, -2};
    expectTrue(core::neighborCell(start, core::Dir6::PosX) == core::Cell3i{11, 5, -2}, "PosX neighbor offset");
    expectTrue(core::neighborCell(start, core::Dir6::NegZ) == core::Cell3i{10, 5, -3}, "NegZ neighbor offset");

    for (const core::Dir6 dir : core::kAllDir6) {
        expectTrue(core::oppositeDir(core::oppositeDir(dir)) == dir, "Opposite direction involution");
    }

    core::CellAabb dirty{};
    dirty.includeCell(core::Cell3i{0, 0, 0});
    dirty.includeCell(core::Cell3i{2, 1, 0});
    expectTrue(dirty.valid, "AABB valid after include");
    expectTrue(dirty.contains(core::Cell3i{1, 0, 0}), "AABB contains interior");
    expectTrue(!dirty.contains(core::Cell3i{3, 1, 0}), "AABB excludes max edge");
}

void testNetworkGraphAndProceduralUtilities() {
    sim::NetworkGraph graph;
    const sim::NodeId n0 = graph.addNode(sim::Socket{core::Cell3i{0, 1, 0}, core::Dir6::PosX, 0});
    const sim::NodeId n1 = graph.addNode(sim::Socket{core::Cell3i{3, 1, 0}, core::Dir6::NegX, 0});
    const sim::EdgeSpan span{core::Cell3i{1, 1, 0}, core::Dir6::PosX, 3};
    const sim::EdgeId e0 = graph.addEdge(n0, n1, span, sim::NetworkKind::Pipe, 2);

    expectTrue(n0 == 0u && n1 == 1u, "Node ids assigned deterministically");
    expectTrue(e0 == 0u, "Edge id assigned deterministically");
    expectTrue(graph.nodeCount() == 2u, "Graph node count");
    expectTrue(graph.edgeCount() == 1u, "Graph edge count");
    expectTrue(graph.edgesForNode(n0).size() == 1u, "Node adjacency contains edge");
    expectTrue(sim::spanEndCell(span) == core::Cell3i{3, 1, 0}, "Span end cell");

    const auto cells = sim::rasterizeSpan(span);
    expectTrue(cells.size() == 3u, "Span rasterization count");
    expectTrue(cells[0] == core::Cell3i{1, 1, 0}, "Span rasterization start");
    expectTrue(cells[2] == core::Cell3i{3, 1, 0}, "Span rasterization end");

    const core::Cell3i pivot{4, 4, 4};
    const std::uint8_t mask = sim::neighborMask6(pivot, [&](const core::Cell3i& cell) {
        return cell == core::neighborCell(pivot, core::Dir6::PosX) ||
               cell == core::neighborCell(pivot, core::Dir6::NegY);
    });
    expectTrue((mask & core::dirBit(core::Dir6::PosX)) != 0u, "Neighbor mask includes PosX");
    expectTrue((mask & core::dirBit(core::Dir6::NegY)) != 0u, "Neighbor mask includes NegY");
    expectTrue(sim::connectionCount(mask) == 2u, "Neighbor degree count");

    expectTrue(sim::classifyJoinPiece(0u) == sim::JoinPiece::Isolated, "Join classification isolated");
    expectTrue(sim::classifyJoinPiece(core::dirBit(core::Dir6::PosX)) == sim::JoinPiece::EndCap, "Join classification endcap");
    expectTrue(
        sim::classifyJoinPiece(static_cast<std::uint8_t>(core::dirBit(core::Dir6::PosX) | core::dirBit(core::Dir6::NegX))) ==
            sim::JoinPiece::Straight,
        "Join classification straight"
    );
    expectTrue(
        sim::classifyJoinPiece(static_cast<std::uint8_t>(core::dirBit(core::Dir6::PosX) | core::dirBit(core::Dir6::PosZ))) ==
            sim::JoinPiece::Elbow,
        "Join classification elbow"
    );
    expectTrue(
        sim::classifyJoinPiece(
            static_cast<std::uint8_t>(
                core::dirBit(core::Dir6::PosX) |
                core::dirBit(core::Dir6::NegX) |
                core::dirBit(core::Dir6::PosZ)
            )
        ) == sim::JoinPiece::Tee,
        "Join classification tee"
    );

    const sim::QuantizedTransform q =
        sim::quantizeTransform(math::Vector3{1.125f, -2.5f, 0.03125f}, math::Vector3{-181.0f, 450.0f, 0.0f});
    const math::Vector3 dqPos = sim::dequantizePosition(q);
    const math::Vector3 dqRot = sim::dequantizeEulerDegrees(q);
    expectNear(dqPos.x, 1.125f, 1.0f / 4096.0f, "Quantized position x");
    expectNear(dqPos.y, -2.5f, 1.0f / 4096.0f, "Quantized position y");
    expectNear(dqPos.z, 0.03125f, 1.0f / 4096.0f, "Quantized position z");
    expectNear(dqRot.x, 179.0f, 0.25f, "Quantized pitch wrap");
    expectNear(dqRot.y, 90.0f, 0.25f, "Quantized yaw wrap");
}

void testCsgCommands() {
    world::CsgVolume volume(8, 8, 8);
    world::CsgCommand addBox{};
    addBox.op = world::CsgOp::AddSolid;
    addBox.materialId = 3;
    addBox.brush.kind = world::BrushKind::Box;
    addBox.brush.minCell = core::Cell3i{1, 1, 1};
    addBox.brush.maxCell = core::Cell3i{4, 4, 4};
    const core::CellAabb boxTouched = world::applyCsgCommand(volume, addBox);
    expectTrue(boxTouched.valid && !boxTouched.empty(), "Add box touched bounds");
    expectTrue(countSolidCells(volume) == 27u, "Add box solid count");
    expectTrue(volume.cellAtWorld(core::Cell3i{1, 1, 1}).materialId == 3u, "Add box material write");

    world::CsgCommand subtractCenter{};
    subtractCenter.op = world::CsgOp::SubtractSolid;
    subtractCenter.brush.kind = world::BrushKind::Box;
    subtractCenter.brush.minCell = core::Cell3i{2, 2, 2};
    subtractCenter.brush.maxCell = core::Cell3i{3, 3, 3};
    world::applyCsgCommand(volume, subtractCenter);
    expectTrue(countSolidCells(volume) == 26u, "Subtract box solid count");
    expectTrue(
        volume.cellAtWorld(core::Cell3i{2, 2, 2}).voxel.type == world::VoxelType::Empty,
        "Subtract clears center cell"
    );

    world::CsgCommand paint{};
    paint.op = world::CsgOp::PaintMaterial;
    paint.materialId = 7;
    paint.brush.kind = world::BrushKind::Box;
    paint.brush.minCell = core::Cell3i{1, 1, 1};
    paint.brush.maxCell = core::Cell3i{4, 4, 4};
    paint.affectMask = world::kCsgAffectSolid;
    world::applyCsgCommand(volume, paint);
    expectTrue(volume.cellAtWorld(core::Cell3i{1, 1, 1}).materialId == 7u, "Paint updates solid material");
    expectTrue(volume.cellAtWorld(core::Cell3i{2, 2, 2}).materialId == 0u, "Paint does not recolor empty cells");

    world::CsgVolume pipeVolume(6, 6, 6);
    world::CsgCommand addPipe{};
    addPipe.op = world::CsgOp::AddSolid;
    addPipe.materialId = 11;
    addPipe.brush.kind = world::BrushKind::PrismPipe;
    addPipe.brush.axis = core::Dir6::PosY;
    addPipe.brush.minCell = core::Cell3i{2, 0, 2};
    addPipe.brush.maxCell = core::Cell3i{4, 6, 4};
    addPipe.brush.radiusQ8 = 128;
    world::applyCsgCommand(pipeVolume, addPipe);
    expectTrue(countSolidCells(pipeVolume) == 24u, "Prism pipe solid count");

    world::CsgVolume rampVolume(4, 4, 1);
    world::CsgCommand addRamp{};
    addRamp.op = world::CsgOp::AddSolid;
    addRamp.brush.kind = world::BrushKind::Ramp;
    addRamp.brush.axis = core::Dir6::PosX;
    addRamp.brush.minCell = core::Cell3i{0, 0, 0};
    addRamp.brush.maxCell = core::Cell3i{4, 4, 1};
    world::applyCsgCommand(rampVolume, addRamp);
    expectTrue(countSolidCells(rampVolume) == 10u, "Ramp solid count");

    world::CsgVolume deterministicA(8, 8, 8);
    world::CsgVolume deterministicB(8, 8, 8);
    const std::vector<world::CsgCommand> commands = {addBox, subtractCenter, paint};
    world::applyCsgCommands(deterministicA, commands);
    world::applyCsgCommands(deterministicB, commands);
    expectTrue(deterministicA.cells() == deterministicB.cells(), "CSG deterministic replay");

    world::Chunk chunk(0, 0, 0);
    const core::CellAabb copyTouched = world::copyVolumeSolidsToChunk(deterministicA, chunk);
    expectTrue(copyTouched.valid, "Copy to chunk touched bounds");
    expectTrue(chunk.isSolid(1, 1, 1), "Chunk copied solid");
    expectTrue(!chunk.isSolid(2, 2, 2), "Chunk copied carved cell");
}

void testFrameArenaAliasUtilities() {
    using render::FrameArenaPass;
    using render::FrameArenaPassRange;

    const FrameArenaPassRange ssao{FrameArenaPass::Ssao, FrameArenaPass::Ssao};
    const FrameArenaPassRange mainToPost{FrameArenaPass::Main, FrameArenaPass::Post};
    const FrameArenaPassRange ui{FrameArenaPass::Ui, FrameArenaPass::Ui};
    const FrameArenaPassRange invalid{FrameArenaPass::Unknown, FrameArenaPass::Post};

    expectTrue(render::isValidFrameArenaPassRange(ssao), "Pass range valid (SSAO)");
    expectTrue(render::isValidFrameArenaPassRange(mainToPost), "Pass range valid (Main->Post)");
    expectTrue(!render::isValidFrameArenaPassRange(invalid), "Pass range invalid (Unknown)");

    expectTrue(render::frameArenaPassRangesOverlap(ssao, ssao), "Overlap on identical range");
    expectTrue(!render::frameArenaPassRangesOverlap(ssao, mainToPost), "No overlap across disjoint ranges");
    expectTrue(!render::frameArenaPassRangesOverlap(mainToPost, ui), "No overlap (post vs ui)");

    std::vector<FrameArenaPassRange> reservedRanges;
    render::addAliasPassRange(reservedRanges, ssao);
    expectTrue(!render::canAliasWithPassRanges(reservedRanges, ssao), "Cannot alias with overlapping range");
    expectTrue(render::canAliasWithPassRanges(reservedRanges, mainToPost), "Can alias with disjoint range");
    render::addAliasPassRange(reservedRanges, mainToPost);
    expectTrue(!render::canAliasWithPassRanges(reservedRanges, mainToPost), "Cannot alias when reserved already");
    expectTrue(render::canAliasWithPassRanges(reservedRanges, ui), "Can alias with later disjoint range");

    uint32_t refCount = 0;
    render::acquireAliasBlockRef(refCount);
    render::acquireAliasBlockRef(refCount);
    expectTrue(refCount == 2u, "Alias ref count increments");
    expectTrue(!render::releaseAliasBlockRef(refCount), "Alias release not zero on first release");
    expectTrue(refCount == 1u, "Alias ref count decrements");
    expectTrue(render::releaseAliasBlockRef(refCount), "Alias release returns zero when last ref released");
    expectTrue(refCount == 0u, "Alias ref count reaches zero");
}

void testChunkMeshingModes() {
    using world::Chunk;
    using world::MeshingMode;
    using world::MeshingOptions;
    using world::Voxel;
    using world::VoxelType;

    const MeshingOptions naiveOptions{MeshingMode::Naive};
    const MeshingOptions greedyOptions{MeshingMode::Greedy};

    {
        Chunk chunk(0, 0, 0);
        const world::ChunkMeshData naive = world::buildChunkMesh(chunk, naiveOptions);
        const world::ChunkMeshData greedy = world::buildChunkMesh(chunk, greedyOptions);
        expectTrue(naive.vertices.empty() && naive.indices.empty(), "Naive empty chunk has no geometry");
        expectTrue(greedy.vertices.empty() && greedy.indices.empty(), "Greedy empty chunk has no geometry");
    }

    {
        Chunk chunk(0, 0, 0);
        chunk.setVoxel(3, 3, 3, Voxel{VoxelType::Solid});
        const world::ChunkMeshData naive = world::buildChunkMesh(chunk, naiveOptions);
        const world::ChunkMeshData greedy = world::buildChunkMesh(chunk, greedyOptions);
        expectTrue(naive.vertices.size() == 24u, "Naive single voxel vertex count");
        expectTrue(naive.indices.size() == 36u, "Naive single voxel index count");
        expectTrue(greedy.vertices.size() == naive.vertices.size(), "Greedy single voxel matches naive vertices");
        expectTrue(greedy.indices.size() == naive.indices.size(), "Greedy single voxel matches naive indices");
    }

    {
        Chunk chunk(0, 0, 0);
        for (int y = 0; y < Chunk::kSizeY; ++y) {
            for (int z = 0; z < Chunk::kSizeZ; ++z) {
                for (int x = 0; x < Chunk::kSizeX; ++x) {
                    chunk.setVoxel(x, y, z, Voxel{VoxelType::Solid});
                }
            }
        }
        const world::ChunkMeshData naive = world::buildChunkMesh(chunk, naiveOptions);
        const world::ChunkMeshData greedy = world::buildChunkMesh(chunk, greedyOptions);

        const std::size_t expectedNaiveVisibleQuads = static_cast<std::size_t>(6 * Chunk::kSizeX * Chunk::kSizeY);
        expectTrue(naive.vertices.size() == expectedNaiveVisibleQuads * 4u, "Naive full chunk visible vertex count");
        expectTrue(naive.indices.size() == expectedNaiveVisibleQuads * 6u, "Naive full chunk visible index count");
        expectTrue(greedy.vertices.size() == 24u, "Greedy full chunk collapses to 6 quads");
        expectTrue(greedy.indices.size() == 36u, "Greedy full chunk collapses to 6 quads indices");
    }

    {
        Chunk chunk(0, 0, 0);
        const int y = 8;
        for (int z = 0; z < Chunk::kSizeZ; ++z) {
            for (int x = 0; x < Chunk::kSizeX; ++x) {
                chunk.setVoxel(x, y, z, Voxel{VoxelType::Solid});
            }
        }

        const world::ChunkMeshData naive = world::buildChunkMesh(chunk, naiveOptions);
        const world::ChunkMeshData greedy = world::buildChunkMesh(chunk, greedyOptions);
        expectTrue(!naive.vertices.empty(), "Naive slab produces geometry");
        expectTrue(!greedy.vertices.empty(), "Greedy slab produces geometry");
        expectTrue(greedy.vertices.size() < naive.vertices.size(), "Greedy slab reduces vertex count");
        expectTrue(greedy.indices.size() < naive.indices.size(), "Greedy slab reduces index count");
    }
}

void testClipmapIndex() {
    world::ChunkGrid grid;
    grid.initializeEmptyWorld();

    world::ChunkClipmapIndex clipmapIndex;
    clipmapIndex.rebuild(grid);
    expectTrue(clipmapIndex.valid(), "Clipmap index valid after rebuild");
    expectTrue(clipmapIndex.chunkCount() == grid.chunkCount(), "Clipmap chunk count matches grid");

    world::SpatialQueryStats updateStats{};
    clipmapIndex.updateCamera(0.0f, 0.0f, 0.0f, &updateStats);
    expectTrue(updateStats.clipmapActiveLevelCount > 0u, "Clipmap level count populated");
    expectTrue(updateStats.clipmapUpdatedLevelCount > 0u, "Clipmap updates levels on first camera update");
    expectTrue(updateStats.clipmapUpdatedBrickCount > 0u, "Clipmap updates bricks on first camera update");
    expectTrue(updateStats.clipmapResidentBrickCount > 0u, "Clipmap resident brick count populated");

    core::CellAabb broadPhase{};
    broadPhase.valid = true;
    broadPhase.minInclusive = core::Cell3i{-96, -96, -96};
    broadPhase.maxExclusive = core::Cell3i{96, 96, 96};
    world::SpatialQueryStats queryStats{};
    const std::vector<std::size_t> visibleChunks = clipmapIndex.queryChunksIntersecting(broadPhase, &queryStats);
    expectTrue(!visibleChunks.empty(), "Clipmap query returns visible chunks near camera");
    expectTrue(queryStats.candidateChunkCount >= queryStats.visibleChunkCount, "Clipmap candidates >= visible");

    world::SpatialQueryStats stableUpdateStats{};
    clipmapIndex.updateCamera(0.2f, 0.2f, 0.2f, &stableUpdateStats);
    expectTrue(stableUpdateStats.clipmapUpdatedLevelCount == 0u, "Clipmap stays stable within same snapped cell");

    world::SpatialQueryStats movedUpdateStats{};
    clipmapIndex.updateCamera(33.0f, 0.0f, 0.0f, &movedUpdateStats);
    expectTrue(movedUpdateStats.clipmapUpdatedLevelCount > 0u, "Clipmap updates when camera crosses snapped boundary");
    expectTrue(movedUpdateStats.clipmapUpdatedBrickCount > 0u, "Clipmap updates bricks when crossing snapped boundary");
}

void testSimulationBeltCargoDeterminism() {
    sim::Simulation simA;
    sim::Simulation simB;
    simA.initializeSingleBelt();
    simB.initializeSingleBelt();

    // Extend the seed with a second belt to exercise cross-segment handoff.
    simA.belts().emplace_back(1, 1, 0, sim::BeltDirection::East);
    simB.belts().emplace_back(1, 1, 0, sim::BeltDirection::East);

    constexpr float kFixedDt = 1.0f / 60.0f;
    for (int tick = 0; tick < 300; ++tick) {
        simA.update(kFixedDt);
        simB.update(kFixedDt);
    }

    const std::vector<sim::BeltCargo>& cargoA = simA.beltCargoes();
    const std::vector<sim::BeltCargo>& cargoB = simB.beltCargoes();
    expectTrue(!cargoA.empty(), "Simulation spawns belt cargo");
    expectTrue(cargoA.size() == cargoB.size(), "Simulation cargo count deterministic");

    if (!cargoA.empty() && cargoA.size() == cargoB.size()) {
        for (std::size_t i = 0; i < cargoA.size(); ++i) {
            expectTrue(cargoA[i].itemId == cargoB[i].itemId, "Cargo id deterministic");
            expectTrue(cargoA[i].beltIndex == cargoB[i].beltIndex, "Cargo belt assignment deterministic");
            expectTrue(cargoA[i].alongQ16 == cargoB[i].alongQ16, "Cargo fixed-step progress deterministic");
            expectNear(cargoA[i].currWorldPos[0], cargoB[i].currWorldPos[0], 0.0001f, "Cargo world X deterministic");
            expectNear(cargoA[i].currWorldPos[1], cargoB[i].currWorldPos[1], 0.0001f, "Cargo world Y deterministic");
            expectNear(cargoA[i].currWorldPos[2], cargoB[i].currWorldPos[2], 0.0001f, "Cargo world Z deterministic");
        }
    }
}

void testMagicaVoxelMeshing() {
    world::MagicaVoxelModel model{};
    model.sizeX = 4;
    model.sizeY = 4;
    model.sizeZ = 4;
    model.hasPalette = true;
    model.paletteRgba.fill(0u);
    model.paletteRgba[1] = 0xFF4444FFu;
    model.paletteRgba[2] = 0xFF44FF44u;
    model.voxels = {
        world::MagicaVoxel{1u, 1u, 1u, 1u},
        world::MagicaVoxel{2u, 1u, 1u, 2u}
    };

    const world::ChunkMeshData meshA = world::buildMagicaVoxelMesh(model);
    const world::ChunkMeshData meshB = world::buildMagicaVoxelMesh(model);
    expectTrue(meshA.vertices.size() == 40u, "Magica mesher adjacent voxel vertex count");
    expectTrue(meshA.indices.size() == 60u, "Magica mesher adjacent voxel index count");
    expectTrue(meshA.indices == meshB.indices, "Magica mesher deterministic indices");
    bool vertexArraysEqual = (meshA.vertices.size() == meshB.vertices.size());
    if (vertexArraysEqual) {
        for (std::size_t i = 0; i < meshA.vertices.size(); ++i) {
            if (meshA.vertices[i].bits != meshB.vertices[i].bits) {
                vertexArraysEqual = false;
                break;
            }
        }
    }
    expectTrue(vertexArraysEqual, "Magica mesher deterministic vertices");

    bool foundRedMaterial = false;
    bool foundNonRedMaterial = false;
    for (const world::PackedVoxelVertex& vertex : meshA.vertices) {
        const std::uint32_t material = (vertex.bits >> world::PackedVoxelVertex::kShiftMaterial) & world::PackedVoxelVertex::kMask8;
        if (material == 251u) {
            foundRedMaterial = true;
        } else if (material > 0u) {
            foundNonRedMaterial = true;
        }
    }
    expectTrue(foundRedMaterial, "Magica mesher maps bright red to red material");
    expectTrue(foundNonRedMaterial, "Magica mesher maps non-red palette colors");

    world::MagicaVoxelModel greedyModel{};
    greedyModel.sizeX = 4;
    greedyModel.sizeY = 4;
    greedyModel.sizeZ = 4;
    greedyModel.hasPalette = true;
    greedyModel.paletteRgba.fill(0u);
    greedyModel.paletteRgba[1] = 0xFF808080u;
    greedyModel.voxels = {
        world::MagicaVoxel{1u, 1u, 1u, 1u},
        world::MagicaVoxel{2u, 1u, 1u, 1u}
    };
    const world::ChunkMeshData greedyMesh = world::buildMagicaVoxelMesh(greedyModel);
    expectTrue(greedyMesh.vertices.size() == 24u, "Magica greedy mesher merges coplanar same-material faces");
    expectTrue(greedyMesh.indices.size() == 36u, "Magica greedy mesher index count");
}

void testMagicaVoxelChunkedMeshing() {
    world::MagicaVoxelModel model{};
    model.sizeX = 40;
    model.sizeY = 8;
    model.sizeZ = 8;
    model.hasPalette = true;
    model.paletteRgba.fill(0u);
    model.paletteRgba[1] = 0xFF808080u;
    model.voxels = {
        world::MagicaVoxel{31u, 1u, 1u, 1u},
        world::MagicaVoxel{32u, 1u, 1u, 1u}
    };

    const std::vector<world::MagicaVoxelMeshChunk> chunksA = world::buildMagicaVoxelMeshChunks(model);
    const std::vector<world::MagicaVoxelMeshChunk> chunksB = world::buildMagicaVoxelMeshChunks(model);
    expectTrue(chunksA.size() == 2u, "Magica chunk mesher splits large model along X");
    expectTrue(chunksA.size() == chunksB.size(), "Magica chunk mesher deterministic chunk count");
    if (chunksA.size() == 2u) {
        expectTrue(chunksA[0].originX == 0 && chunksA[1].originX == 32, "Magica chunk mesher chunk origins");
    }

    std::size_t totalVertices = 0;
    std::size_t totalIndices = 0;
    bool chunkMeshesEqual = (chunksA.size() == chunksB.size());
    for (std::size_t i = 0; i < chunksA.size(); ++i) {
        const world::MagicaVoxelMeshChunk& chunkA = chunksA[i];
        const world::MagicaVoxelMeshChunk& chunkB = chunksB[i];
        totalVertices += chunkA.mesh.vertices.size();
        totalIndices += chunkA.mesh.indices.size();
        if (chunkA.originX != chunkB.originX || chunkA.originY != chunkB.originY || chunkA.originZ != chunkB.originZ) {
            chunkMeshesEqual = false;
            break;
        }
        if (chunkA.mesh.indices != chunkB.mesh.indices ||
            chunkA.mesh.vertices.size() != chunkB.mesh.vertices.size()) {
            chunkMeshesEqual = false;
            break;
        }
        for (std::size_t vertexIndex = 0; vertexIndex < chunkA.mesh.vertices.size(); ++vertexIndex) {
            if (chunkA.mesh.vertices[vertexIndex].bits != chunkB.mesh.vertices[vertexIndex].bits) {
                chunkMeshesEqual = false;
                break;
            }
        }
        if (!chunkMeshesEqual) {
            break;
        }
    }

    expectTrue(totalVertices == 40u, "Magica chunk mesher hides shared faces across chunk boundaries (vertices)");
    expectTrue(totalIndices == 60u, "Magica chunk mesher hides shared faces across chunk boundaries (indices)");
    expectTrue(chunkMeshesEqual, "Magica chunk mesher deterministic output");
}

} // namespace

int main() {
    testGridPrimitives();
    testNetworkGraphAndProceduralUtilities();
    testCsgCommands();
    testFrameArenaAliasUtilities();
    testChunkMeshingModes();
    testClipmapIndex();
    testSimulationBeltCargoDeterminism();
    testMagicaVoxelMeshing();
    testMagicaVoxelChunkedMeshing();

    if (g_failures != 0) {
        std::cerr << "[foundation test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[foundation test] all checks passed\n";
    return 0;
}
