#pragma once

#include "core/math.h"

namespace voxelsprout::scene {

struct SunLight {
    voxelsprout::core::Vec3 direction = voxelsprout::core::normalize({0.4f, 0.8f, 0.2f});
    float intensity = 8.0f;
};

struct VolumeSettings {
    float densityScale = 0.2f;
    float anisotropyG = 0.6f;
    float albedo = 0.99f;
    float macroScale = 0.34f;
    float detailScale = 0.56f;
    float densityCutoff = 0.08f;
    float chunkiness = 0.8f;
    float coverage = 0.62f;
    float weatherScale = 0.08f;
    float cloudBase = 1.4f;
    float cloudTop = 8.8f;
    float warpStrength = 0.9f;
    float erosionStrength = 0.55f;
    float brightnessBoost = 2.2f;
    float ambientLift = 0.55f;
    int maxBounces = 3;
};

struct SceneState {
    SunLight sun{};
    VolumeSettings volume{};
};

} // namespace voxelsprout::scene
