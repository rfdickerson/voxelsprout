#include <gtest/gtest.h>

#include "core/grid3.h"
#include "sim/network_procedural.h"
#include "sim/pipe.h"

namespace {

void ExpectCell(const voxelsprout::core::Cell3i& cell, int x, int y, int z) {
    EXPECT_EQ(cell.x, x);
    EXPECT_EQ(cell.y, y);
    EXPECT_EQ(cell.z, z);
}

} // namespace

TEST(SimNetworkProceduralTest, NeighborMask6FlagsAdjacentCells) {
    using namespace voxelsprout;

    constexpr int kPipeX = 0;
    constexpr int kPipeY = 0;
    constexpr int kPipeZ = 0;
    const sim::Pipe pipe{ kPipeX, kPipeY, kPipeZ, math::Vector3{1.0f, 0.0f, 0.0f}, 1.0f, 0.45f, math::Vector3{1.0f, 1.0f, 1.0f}};

    const std::vector<core::Cell3i> occupied{
        core::Cell3i{kPipeX + 1, kPipeY, kPipeZ},
        core::Cell3i{kPipeX - 1, kPipeY, kPipeZ},
        core::Cell3i{kPipeX, kPipeY, kPipeZ + 1}
    };

    const uint8_t mask = sim::neighborMask6(
        core::Cell3i{pipe.x, pipe.y, pipe.z},
        [&occupied](const core::Cell3i& cell) -> bool {
            for (const core::Cell3i& occupiedCell : occupied) {
                if (occupiedCell == cell) {
                    return true;
                }
            }
            return false;
        }
    );

    const uint8_t expected = static_cast<uint8_t>(core::dirBit(core::Dir6::PosX) |
                                                  core::dirBit(core::Dir6::NegX) |
                                                  core::dirBit(core::Dir6::PosZ));
    EXPECT_EQ(mask, expected);
}

TEST(SimNetworkProceduralTest, RasterizeSpanMatchesDirection) {
    using namespace voxelsprout;

    const sim::EdgeSpan span{
        .start = core::Cell3i{1, 2, 3},
        .dir = core::Dir6::PosX,
        .lengthVoxels = 4
    };
    const std::vector<core::Cell3i> cells = sim::rasterizeSpan(span);

    ASSERT_EQ(cells.size(), 4u);
    ExpectCell(cells[0], 1, 2, 3);
    ExpectCell(cells[1], 2, 2, 3);
    ExpectCell(cells[2], 3, 2, 3);
    ExpectCell(cells[3], 4, 2, 3);
}

TEST(SimNetworkProceduralTest, ClassifyJoinPieceCategories) {
    using namespace voxelsprout;

    EXPECT_EQ(sim::classifyJoinPiece(0u), sim::JoinPiece::Isolated);
    EXPECT_EQ(sim::classifyJoinPiece(core::dirBit(core::Dir6::PosY)), sim::JoinPiece::EndCap);
    EXPECT_EQ(
        sim::classifyJoinPiece(
            static_cast<uint8_t>(core::dirBit(core::Dir6::NegX) | core::dirBit(core::Dir6::PosX))
        ),
        sim::JoinPiece::Straight
    );
    EXPECT_EQ(
        sim::classifyJoinPiece(
            static_cast<uint8_t>(core::dirBit(core::Dir6::PosY) | core::dirBit(core::Dir6::PosX))
        ),
        sim::JoinPiece::Elbow
    );
    EXPECT_EQ(
        sim::classifyJoinPiece(
            static_cast<uint8_t>(
                core::dirBit(core::Dir6::PosY) |
                core::dirBit(core::Dir6::NegY) |
                core::dirBit(core::Dir6::PosX)
            )
        ),
        sim::JoinPiece::Tee
    );
    EXPECT_EQ(
        sim::classifyJoinPiece(0b111111u),
        sim::JoinPiece::Cross
    );
}
