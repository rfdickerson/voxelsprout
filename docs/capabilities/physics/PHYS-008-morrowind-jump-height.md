# PHYS-008: Morrowind Player Jump Height

Status: Implemented

Verification: `odai_bethesda_runtime_tests` includes PHYS-JUMP-001. The
Seyda Neen walking capture settled on resident ground with the changed build.
`ctest --test-dir build-linux --output-on-failure -j 4` passed 56/56 tests.

## Goal

The walking Morrowind player performs a modest, repeatable jump. A jump from
level ground rises about 70 Bethesda units (approximately one metre) before
gravity returns the player to the same surface.

## Dependencies

- PHYS-001: Static World Collision
- PHYS-002: Terrain Collision
- The Bethesda player character controller and fixed physics tick

## Required Behavior

- The Morrowind player's Jolt controller derives its upward launch speed from
  the target rise, Jolt gravity, and the existing Bethesda-unit conversion.
  The camera and gameplay world continue to follow the controller pose.
- A grounded jump reaches a peak 60–80 Bethesda units above its starting feet
  position on a flat static floor, then lands on that floor.
- An airborne jump request does not add a second upward impulse.
- This capability changes neither Skyrim's movement policy nor the separate
  camera-only movement fallback. Acrobatics scaling and variable-height jumps
  are deferred.
- Verification runs headlessly and requires no renderer or retail game data.

## Verification

PHYS-JUMP-001: Build a deterministic static floor and a Morrowind-style
character controller. From a grounded pose, apply the same launch speed used
by the walking game path for one fixed tick. At 60 Hz, record the maximum feet
height and assert it is 60–80 units above the start. Assert the controller
subsequently lands and remains grounded. Request another jump while airborne
and assert the arc does not acquire a second impulse.

Run the relevant physics test executable and the complete CTest suite. Smoke
test the Seyda Neen walking launch when a display and local game data are
available.

## Definition of Done

PHYS-008 is Implemented when the Morrowind walking player uses the height-based
impulse, PHYS-JUMP-001 passes headlessly, the full test suite passes, and
`docs/PARITY.md` marks this capability Implemented.
