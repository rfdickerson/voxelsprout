# Weather presentation: authored precipitation coverage

The weather reader now resolves WTHR MNAM to SPGD records through the active
load order. SPGD decoding retains gravity, rotation, particle size, center offset,
rotation range, atlas dimensions, type, box size, density, and texture path.
The 40-, 44-, and 48-byte DATA layouts are supported; omitted density defaults
to one. Nonfinite values and incomplete fields are rejected. Deleted or malformed
winning SPGD records suppress older definitions, with malformed-record diagnostics.
Zero atlas dimensions are preserved: original SlowTimeParticles uses them.

Skyrim's existing screen-space rain fallback now uses SPGD type and density,
instead of weather-name guesses. Density maps to coverage as `1 - exp(-density)`:
ordinary RainParticles (1) produces 0.632 coverage and RainStormParticles (2)
produces 0.865. This bounded mapping is an approximation for the existing renderer,
not Skyrim's authored particle simulation. Snow, unknown types, null and missing
references produce no rain. Earlier games retain their prior rain heuristic.

Fog distance now interpolates over two hours centered on dawn and dusk rather
than abruptly switching day/night distances. The transition width is an engine
policy; distances and climate times come from records.

WTHR now also preserves RFCT references, sky-static references, the aurora model,
precipitation transition thresholds, wind direction/range, and maximum fog values.
The local coverage report lists these as preserved but not consumed. It resolves
precipitation texture dependencies separately from runtime support.

## Remaining presentation work

The original precipitation textures, particle geometry, velocities and rotations
still need connection to a particle renderer. Snow, ash-like effects, aurora/sky
statics, RFCT effects, precipitation transition thresholds, authored wind direction,
and maximum fog are not yet rendered by this change. Existing rain remains a
screen-space fallback, including its existing lack of world-space roof occlusion.
This is a weather coverage increment, not complete weather or retail parity.

No renderer pass, attachment, descriptor, or serialized scene layout is added.
Policy records are reconstructed from the active plugins; no cell-cache change
is needed. Capture helper `--weather` selects any authored EditorID while retaining
the chosen look's hour, fixed view, resolution and warmup.

Synthetic tests cover full and optional layouts, truncation, nonfinite density,
load-order overrides, malformed/deleted winners, weather references, density
differences, snow/missing-reference fallback and fog continuity/midnight wrapping.
Local audit and captures live under `captures/asset-coverage/weather.*` and
`captures/weather-*`; original assets are not redistributed.

The installed Skyrim.esm + Update.esm audit decodes all 16 SPGD records with no
diagnostics. Fourteen name resolvable textures; two have empty texture paths.
All nonzero weather precipitation references resolve. Debug CTest passes 36/36;
the import tests also pass in the optimized build. Optimized rain, storm, snow
and clear captures at native 1080p complete with synchronization validation
enabled, zero validation errors/warnings, and clean shutdown. Snow capture checks
the dry fallback, not rendered snow. Storm visual review confirms the existing
rain overlay remains visible across foreground roofs, a known fallback limitation.

Layout reference: [xEdit Skyrim SPGD and WTHR definitions](https://raw.githubusercontent.com/TES5Edit/TES5Edit/dev-4.1.6/Core/wbDefinitionsTES5.pas).

## Authored color policy and rain showcase (2026-09-09)

Skyrim now defaults to neutral sky saturation and no additional sky/fog
desaturation before IMGS grading. Existing HDR magnitude/gain mapping remains;
explicit sky saturation overrides and other-game defaults remain supported.
This restores authored chromatic direction, not a matched retail tone curve.

Local JK + SMIM storm showcase: captures/authored-rain/run.py. Uses
SkyrimStormRain at hour 13 over Riverwood, a 768x432 logical window with native
1536x864 rendering. The 360-frame capture completed with Vulkan synchronization
validation, no validation errors/warnings and clean shutdown. Optimized build
and all 40 CTest tests passed. Validation-run rolling GPU p50/p95 ended at
26.92/31.32 ms; this is not a 60fps or non-validation performance acceptance.
Rain remains the screen-space fallback; roof occlusion and authored precipitation
texture/particle simulation are still outstanding.

## World-space rain (2026-09-09)

The former two UV-scrolling rain layers are replaced by a world-anchored falling
particle lattice. The existing HDR post pass traces camera rays through bounded
cells and evaluates drop segments; perspective and camera translation now affect
rain distance/size and parallax. Scene depth clips drops behind visible opaque
geometry. A short fixed shutter, pixel-footprint filtering, near fade and distance
fade limit giant foreground streaks and subpixel shimmer.

SPGD supplies width, length, fall speed and range through a typed weather interface.
Density retains the existing bounded coverage mapping. Camera uniform grows by
16 bytes; descriptor bindings, attachments and pass dependencies are unchanged.
The implementation remains procedural: authored atlas appearance, wind drift,
roof shelter, water/splash interactions and transparent-surface occlusion are not
implemented. This is world-space rendering within the post pass, not persistent
simulated particle entities. The depth read already belongs to this pass.

Local Whiterun JK+SMIM captures and the translation route are under
captures/jk-skyrim-showcase (run_rain.py, run_rain_walk.py). The first validated
native-DPI capture reported rolling GPU p50/p95 23.13/28.52 ms, post 3.10 ms;
this includes validation and is not a 60fps acceptance or an isolated rain delta.

Optimized build and all 40 CTests passed. The corrected 600-unit Whiterun
market-street camera route recorded 540 samples and completed with zero Vulkan
validation errors/warnings and clean shutdown. Final shader adjustment distributes
drops throughout the safe cell interior rather than clustering them centrally.
Interactive launch remains small/native DPI with JK+SMIM. No retail rain parity
or roof-shelter acceptance is claimed.
