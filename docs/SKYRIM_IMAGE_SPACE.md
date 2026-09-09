# Skyrim image-space records

IMGS and IMAD are now decoded from the active plugin load order. Winning overrides,
deleted records, remapped references, malformed fields, and missing references are
handled explicitly. The local Skyrim.esm + Update.esm audit found 275 IMGS and
172 IMAD records with no malformed records. Coverage JSON reports source values
and separates runtime support from preserved but unused channels.

## Runtime support

- Weather IMSP references select and interpolate sunrise/day/sunset/night IMGS.
- Interior CELL XCIM references select the interior policy; missing references
  use neutral settings instead of retaining the previous interior.
- IMGS bloom threshold, scale and radius, adaptation speed, cinematic saturation,
  brightness, contrast, tint, and no-sky depth of field reach existing passes.
- Papyrus ImageSpaceModifier.Apply and Remove queue presentation changes. IMAD
  scalar and RGBA curves interpolate with duration and strength, including tint
  and fade. Configuration resets clear transient state.
- Explicit color-look, contrast, saturation and exposure environment overrides
  retain precedence. Existing explicit DOF controls also take precedence.
  ODAI_SKYRIM_NO_IMAGE_SPACE=1 disables this policy for comparisons.

Source values are preserved independently of renderer normalization. Adaptation
speed is divided by 30 and bloom scale multiplied by 0.01 at the renderer boundary.
These are engine mappings, not verified retail transfer functions. Cinematic
grading operates in display space after the existing tone mapper; tint blends
toward tinted luminance. Fade is applied after output gamma conversion. The
existing tone curve is retained; cinematic contrast now uses a continuous toe
and shoulder to avoid clipping night shadows (see [night correction](SKYRIM_NIGHT_LIGHTING.md)).
This does not establish retail pixel parity.

No-sky DOF uses authored distance/range and the decoded radius with strength.
Target-dependent and sky-blurring DOF remain unsupported. Other preserved but
unused data include HDR receive threshold, white point, sunlight/sky scales and
eye strength, IMAD target luminance, radial blur, motion blur and double vision.
ApplyCrossFade, target tracking, room and underwater overrides, and automatic
effect-record triggers remain future work.

## Compatibility and verification

There is one imported-scene rendering path, with no new attachments or passes.
The camera uniform grows by 48 bytes. Existing explicit pass synchronization is
retained. Captures wait up to one second for a requested frame instead of using
the short interactive frame-pacing budget before readback.

Cooked scene version 37 and cell-cache version 96 are unchanged: image-space
tables are reconstructed from plugins rather than added to serialized scenes.
Cooked-only scenes without the plugin policy retain neutral defaults. Active
modifiers are transient and are not restored from saves.

Synthetic tests cover record sizes, nonfinite values, curve counts and ordering,
interpolation, strength, expiry, tint/fade, weather transitions, interior bindings,
load-order overrides and deletion. Runtime tests cover Apply/Remove command
delivery. Debug CTest passes all 36 tests. Local capture evidence is under
captures/image-space-*. These captures do not establish other-game visual
regressions, full cell-eviction behavior, or retail parity.

The optimized suite passes 32/36 tests, including image-space tests. Its four
pre-existing failures remain inventory, Bethesda runtime, save and world-map-cache
tests; the Debug suite passes those tests. Day, overcast, dusk, night, disabled
comparison and 3840x2160 captures completed with synchronization validation
enabled and zero validation errors or warnings. Visual review still shows dark
night shadows and bright daytime haze; matching retail exposure and tone response
remains unfinished.

The final moving-camera route produced 30 frames with synchronization validation
enabled, zero validation errors or warnings, and clean shutdown. Its 144 cells
were pinned, so it does not test eviction. Route logs still report skipped
effects and dropped landscape layers outside this image-space work.

Native 1080p optimized runs without validation, after 300 warmup frames, report
final rolling GPU p50/p95 of 45.549/47.704 ms enabled and 45.615/47.178 ms disabled.
The difference is within run-to-run variation, not evidence of a speedup. Both
runs held 81 resident cells. No new texture or attachment residency is introduced;
the GPU uniform increases by 48 bytes. CPU table allocation and total process/GPU
memory deltas were not instrumented. Commands, executable hashes and timing logs
are retained locally in captures/image-space-timing-on.json and the off variant.

Layout references: [xEdit Skyrim definitions](https://raw.githubusercontent.com/TES5Edit/TES5Edit/dev-4.1.6/Core/wbDefinitionsTES5.pas)
and [shared definitions](https://raw.githubusercontent.com/TES5Edit/TES5Edit/dev-4.1.6/Core/wbDefinitionsCommon.pas).
