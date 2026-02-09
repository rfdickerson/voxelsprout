#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "core/Grid3.hpp"
#include "math/Math.hpp"
#include "sim/NetworkGraph.hpp"
#include "sim/NetworkProcedural.hpp"
#include "world/Chunk.hpp"
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

} // namespace

int main() {
    testGridPrimitives();
    testNetworkGraphAndProceduralUtilities();
    testCsgCommands();

    if (g_failures != 0) {
        std::cerr << "[foundation test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[foundation test] all checks passed\n";
    return 0;
}
