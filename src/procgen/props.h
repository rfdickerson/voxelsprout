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

// Park/sidewalk bench: two end frames, slat seat and back, facing -Z. About
// 0.09 world units wide, sitting at the local origin on y = 0.
TriMesh generateBench(std::uint32_t seed);

// Squat fire hydrant at the local origin, ground at y = 0 (about 0.045 world
// units tall). The seed picks the paint (red/yellow families).
TriMesh generateHydrant(std::uint32_t seed);

// Roadside billboard: post + tilted panel with a seeded two-tone "ad" stripe
// pattern, panel facing -Z. About 0.30 world units tall.
TriMesh generateBillboard(std::uint32_t seed);

// Bus stop shelter: back wall, roof, bench slat, and a signpost, open side
// facing -Z (the street). About 0.16 world units tall.
TriMesh generateBusStop(std::uint32_t seed);

// Small watercraft floating with its waterline at y = 0, nose facing +X.
// Variant 0 is a rowboat (~0.14 long), variant 1 a working barge (~0.22).
// The seed picks hull color and proportions.
TriMesh generateBoat(std::uint32_t variant, std::uint32_t seed);

// Low-poly pedestrian standing at the local origin, ground at y = 0 (about
// 0.075 world units tall — reads against 0.15-long cars). The seed picks the
// shirt/trousers palette and build.
TriMesh generatePedestrian(std::uint32_t seed);

// School bus: long yellow body, white roof, window band, nose facing +X
// (about 0.17 world units long). Drives the weekday pickup loop.
TriMesh generateSchoolBus(std::uint32_t seed);

// Garbage truck: dark cab + boxy hopper, nose facing +X (about 0.15 long).
// Trundles the residential loop on collection days.
TriMesh generateGarbageTruck(std::uint32_t seed);

// Curbside trash can: small lidded cylinder at the local origin, ground at
// y = 0 (about 0.03 tall). Appears at residential curbs on collection day.
TriMesh generateTrashCan(std::uint32_t seed);

}  // namespace odai::procgen
