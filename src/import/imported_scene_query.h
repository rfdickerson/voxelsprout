#pragma once

// Ray queries against a cooked ImportedScene's packed geometry.
//
// Extracted from App::raycastImportedSceneFromCamera (src/app/app.cc), where it
// was a private method whose only output was a VOX_LOGI line behind the `I` key.
// It resolves everything a material inspector needs — draw, triangle, texture
// and the packed material flags — so pulling it out here turns a debug print
// into the selection mechanism for the material editor, and makes it testable
// with no window and no Vulkan.
//
// Brute force over every triangle in the scene. That is fine at click rate and
// deliberately not optimized: an acceleration structure here would be a second
// copy of the geometry to keep in sync for no measurable gain.
//
// Note it reads vertex i0 of each triangle for the flags. That is correct
// rather than sloppy: inFlags is declared `nointerpolation` in the shaders, so
// the provoking vertex is exactly what the GPU shades the triangle with.

#include <cstdint>

#include "import/imported_scene.h"
#include "math/geometry.h"

namespace odai::importer {

struct ImportedSceneRayHit {
    bool hit = false;
    float distance = 0.0f;
    math::Vector3 position{};
    std::uint32_t drawIndex = 0;
    std::uint32_t triangleIndex = 0;   // triangle ordinal within the whole scene
    std::uint32_t textureIndex = 0;    // 0xffffffff = untextured, uses vertex colour
    std::uint32_t flags = 0;           // packed material flags of the provoking vertex
    std::uint32_t materialIndex = 0;   // library slot, 0 = none (decoded from flags)
};

// Nearest front-or-back facing triangle along the ray, within maxDistance.
// Returns hit = false for an empty scene or a degenerate ray.
[[nodiscard]] ImportedSceneRayHit raycastImportedScene(const ImportedScene& scene,
                                                       const math::Ray& ray,
                                                       float maxDistance);

}  // namespace odai::importer
