#pragma once

#include "anim/skeleton.h"
#include "math/math.h"
#include "render/renderer_types.h"

#include <cstdint>
#include <vector>

// Procedural low-poly humanoid body -- cylinder-and-box limb segments rigged
// (rigid, single-bone-weight-per-vertex) against anim::makeBipedSkeleton()'s
// joint layout. No traditional character-modeling/rigging pipeline exists in
// this engine (see docs/ROADMAP.md's NIF-import notes for why), so this is
// authored directly as flat vertex/index data at build time, matching how
// procgen/primitives.h generates the rest of this project's geometry.
// Built for the Dragon Age: Origins touchstone's small-party (4-8 character)
// combat -- not a mass-battle crowd system.
namespace odai::procgen {

struct HumanoidMeshColors {
    odai::math::Vector3 torso{0.6f, 0.1f, 0.1f};    // tunic / vest
    odai::math::Vector3 limbs{0.82f, 0.65f, 0.5f};  // bare skin / sleeves
    odai::math::Vector3 accent{0.35f, 0.25f, 0.12f}; // boots / helmet / trim
};

struct HumanoidSkinnedMesh {
    std::vector<odai::render::ImportedSkinnedMeshVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::uint32_t boneCount = 0;
};

// Builds one humanoid body rigged against `skeleton` (must be shaped like
// anim::makeBipedSkeleton() -- looks up joints by the kBipedBone* name
// constants). Every vertex carries a single dominant bone weight (rigid
// per-segment skinning, no joint blending) -- a deliberate simplification
// given there is no way to visually verify smoother blending in this build
// environment; segments read as distinct rigid capsules, in keeping with
// this project's stylized low-poly look (docs/stylized_low_poly.md).
[[nodiscard]] HumanoidSkinnedMesh buildHumanoidSkinnedMesh(
    const odai::anim::Skeleton& skeleton, const HumanoidMeshColors& colors
);

}  // namespace odai::procgen
