# Skyrim night lighting — Whiterun market

The night readability correction uses installed Skyrim SE WTHR/IMGS records,
original market geometry, materials and placed lights. It does not establish
pixel parity with a matched retail screenshot.

## Corrections

- Skyrim night fill now uses the sampled, linear NAM0 ambient color directly as
  constant SH irradiance. Previously a procedural dark-blue night palette was
  multiplied by a second weather tint. A smooth transition over sun elevation
  0–6 degrees avoids the old hard night switch in ambient illumination. Daytime
  SH, other games, and the explicit weather-light weight override are retained.
- IMGS cinematic contrast now has a continuous toe and shoulder, preserving
  black/white and its middle-grey slope. The old affine contrast clipped dark
  values to zero, particularly with the authored night contrast around 1.45.
  The replacement is an engine transfer mapping, not a recovered retail shader.
  Authored brightness, saturation and tint still apply. Explicit grading
  overrides still take precedence.
- Whiterun's extra linear contrast offset is neutral; it no longer clips the
  recovered shadows after IMGS grading.
- Whiterun's empty parent-LOD chunk exposed a separate image lifetime error:
  slot rollback could destroy textures before asynchronous copies completed.
  Image retirement now waits for both the graphics and pending upload timeline
  values. No new pass, attachment, layout or serialized scene version is needed.

The local `SkyrimClear` record has night ambient RGB 70/104/134 and links to
`ISSkyrimClearNIGHT` (brightness 1.15, contrast 1.45). These are source values,
not invented night colors. Record interpretation follows the
[xEdit Skyrim definitions](https://raw.githubusercontent.com/TES5Edit/TES5Edit/dev-4.1.6/Core/wbDefinitionsTES5.pas).
Resource retirement follows the existing timeline mechanism and
[Khronos synchronization guidance](https://docs.vulkan.org/guide/latest/synchronization.html).

## Local evidence

`captures/whiterun-night/` retains command/environment JSON, native PNG captures,
logs, source-record inspection and `comparison.json`. The market camera is the
existing authored gate-relative showcase camera. Clear weather, hour 0.5 and
240 warmup frames are identical for before/after. Fire/cloud animation phase is
not locked, so this is not a pixel-exact temporal comparison.

- Window: 768×432 logical; framebuffer and rendering: 1536×864; native scale 1.
- Night pixels at display luma ≤5/255: 62.7% before, 2.6% after. This measures
  shadow clipping, not retail similarity.
- Night GPU rolling p50/p95: 11.83/14.13 ms after (11.58/13.29 before).
  These short stationary captures do not certify sustained frame pacing.
- Daylight (14:00) and dusk (19:15) preserve visible market materials; dusk
  remains warm. Night retains dark sky and warm authored braziers.
- Optimized shader/runtime build passes. Focused render-policy and image-space
  tests pass, including monotonic contrast, endpoints, neutral identity, night
  policy isolation, override and twilight cases.
- Final Whiterun synchronization validation: zero errors/warnings and clean
  image shutdown (`final-validation.log`). Riverwood night regression also
  retains road/building detail, though unlit areas remain deliberately dark.
- Full optimized CTest: 35/39 pass. The same four prior failures remain:
  Skyrim inventory, Bethesda runtime, save and world-map-cache.

Calendar-driven lunar phases/orbits, directional moonlight, exact retail tone transfer,
matched retail captures and the wider weather/moving-camera acceptance remain
separate work. The corrected night is a visibility fix, not full night-sky parity.

## Authored stars, moons and source lights

The sky-mesh pass now consumes the four surfaces in `Sky/Stars.nif`, including
its original UVs, vertex coverage and sky material UV transforms. The SSE
`BSSkyShaderProperty` decoder preserves source texture, flags and object type;
truncated properties are not published. The assets are `SkyStars.dds`,
`SkyrimGalaxy.dds` and `SkyrimConstellations01/02.dds`. DDS alpha controls additive
star/galaxy radiance; ignoring it makes the galaxy an opaque bright backdrop.
These fields follow the [NifTools sky property definition](https://raw.githubusercontent.com/niftools/nifxml/develop/nif.xml).

Masser and Secunda use shipped full-phase DDS textures. `iMasserSize` and
`iSecundaSize` are read from active plugin GMST records (installed values 90/40).
Weather's Stars channel and a smooth twilight visibility gate control the sky.
The fixed 00:30 composition is a full-moon presentation, with an approximate
clock-driven rotation and size-to-angle conversion; lunar calendar, accurate
orbits and dedicated directional moonlight remain unfinished. No point light
is attached to either moon.

Stars, moon discs and clouds use the same sky geometry pass and bindless texture
residency. Premultiplied blending preserves ordinary cloud compositing while
allowing additive stars. Clouds cover the celestial geometry; opaque world
geometry occludes the sky through the existing depth test. There is no new pass
or render target. Camera uniforms add 48 bytes; cooked layouts are unchanged.

Skyrim now uses only placed LIGH records for local illumination. The generated
fire-light fallback remains available to the other games. Rebuilt Whiterun
reports 10 authored lights, compared with 18 including prior fallbacks; 14 fire
particle emitters remain. Fire particle presentation is separate from LIGH
illumination and this work does not complete every NIF particle modifier.
Generated cell cache version 104 invalidates the old synthetic light results.

Local captures: `captures/whiterun-night/celestial-final.png` (Masser and market),
`celestial-showcase.png` (higher framing for both moons), and `celestial-day.png`
(daylight visibility regression). `celestial-final.log` has synchronization
validation enabled, zero validation errors/warnings and clean image shutdown.
Synthetic tests cover sky material preservation/truncation, moon size import,
winning/malformed/deleted overrides, daylight visibility and the source-only
Skyrim light policy. Optimized CTest remains 35/39, with the same four existing
failures listed above. Matched retail pixel parity and weather-change/eviction
stress coverage remain unverified.

## Sunset regression (2026-09-09)

Forced WTHR selection used to bypass the worldspace CLMT clock. Climate timing
now resolves independently, while the requested weather remains selected.
Whiterun inherits SkyrimClimate (dawn midpoint 7.75, dusk midpoint 18.25).

The renderer also applied its procedural 0.14 sky exposure as soon as solar
elevation crossed zero, including authored WTHR skies whose night colors were
already dark. The night exposure now blends only over the procedural portion;
fully authored weather retains the configured sky exposure. Sun disk/halo remain
suppressed below the horizon. No pass/resource/synchronization changes.

Local before/after captures and launch scripts live in
captures/jk-skyrim-showcase. The sunset showcase uses 18:15 and native DPI.
