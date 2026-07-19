#pragma once

#include <cstdint>

#include "procgen/mesh_emit.h"

// Small ambient props (trees, vehicles, seasonal decorations) assembled from
// the same CSG primitives as the era buildings. Like generateBuilding, results
// are deterministic per seed and returned as flat-shaded local-space TriMeshes
// for caching.
namespace odai::procgen {

enum class Season : std::uint8_t { Spring, Summer, Autumn, Winter };

// Low-poly tree standing at the local origin, ground at y = 0. Even variants
// are broadleaf (trunk + bulbous canopy), odd variants are conifer (stacked
// tapered tiers). Height/width jitter comes from the seed. The season drives
// the foliage: spring is fresh green with occasional blossom, autumn turns
// orange/amber, winter strips broadleaf trees to bare branches and dusts
// conifers with snow.
TriMesh generateTree(std::uint32_t variant, std::uint32_t seed, Season season = Season::Summer);

// Low-poly car centered at the local origin, ground at y = 0, nose facing +X
// (about 0.15 world units long — sized for kTileWorldSize = 1 roads). The seed
// picks the body color and slight proportions.
TriMesh generateVehicle(std::uint32_t seed);

// Squat ribbed pumpkin with a stem, sitting at the local origin on y = 0
// (about 0.05 world units tall). Autumn yard decoration.
TriMesh generatePumpkin(std::uint32_t seed);

// Wooden utility pole — post, cross-arm, three insulators — standing at the
// local origin, ground at y = 0 (about 0.32 world units tall, cross-arm along
// local X). Placed along powered road tiles; wire spans between poles are
// plain boxes emitted directly into the scene, not part of this mesh.
TriMesh generatePowerPole(std::uint32_t seed);

// Cast-iron streetlamp with a warm glowing head, standing at the local
// origin, ground at y = 0 (about 0.24 world units tall). Purely decorative —
// present along roads regardless of power state.
TriMesh generateStreetlamp(std::uint32_t seed);

}  // namespace odai::procgen
