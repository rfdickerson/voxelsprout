#pragma once

#include <array>

namespace odai::bethesda {

struct RuntimeTransform {
    std::array<double, 3> position{};
    std::array<float, 3> rotationRadians{};
    float scale = 1.0f;
    friend bool operator==(const RuntimeTransform&, const RuntimeTransform&) = default;
};

}  // namespace odai::bethesda
