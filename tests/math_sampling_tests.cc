#include "core/math.h"
#include "core/rng.h"

#include <cassert>
#include <cmath>

int main() {
    const voxelsprout::core::Vec3 v{0.0f, 3.0f, 4.0f};
    const voxelsprout::core::Vec3 n = voxelsprout::core::normalize(v);
    const float len = voxelsprout::core::length(n);
    assert(std::abs(len - 1.0f) < 1e-4f);

    voxelsprout::core::Pcg32 rngA(12345u);
    voxelsprout::core::Pcg32 rngB(12345u);
    for (int i = 0; i < 1024; ++i) {
        assert(rngA.nextU32() == rngB.nextU32());
    }

    voxelsprout::core::Pcg32 rng(42u);
    for (int i = 0; i < 2048; ++i) {
        const float value = rng.nextFloat01();
        assert(value >= 0.0f && value < 1.0f);
    }

    return 0;
}
