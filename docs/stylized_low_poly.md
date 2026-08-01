# Stylized low-poly meshes (SimCity 2013 look)

Goal: make the procgen city read like SimCity 2013 / Monument Valley-style dioramas —
flat-shaded facets, disciplined flat color palettes, soft ambient gradients, baked
corner darkening, subtle bevels that catch light, and a tilt-shift "miniature" frame.

The engine is already most of the way there. All procgen geometry is genuinely
flat-shaded by construction (`triangulate()` in `src/procgen/mesh_emit.cc` duplicates
vertices per face and copies the polygon plane normal), colors are per-face flat hex
palettes, ambient comes from SH irradiance (soft gradient across facets for free), the
citybuilder already runs a 20° telephoto diorama camera with depth of field, and the
tone mapper has a full grading chain with a "StylizedVivid" preset.

This doc lists the concrete gaps, in priority order, with the exact seams to cut at.

## 1. Stop the slope-blend from desaturating building walls  (small, highest impact)

`imported_static.frag.slang` (~line 420) blends steep faces toward grey rock keyed
purely off `normal.y`:

```slang
const float slope = saturate(1.0 - normal.y);
albedo = lerp(albedo, float3(0.40, 0.39, 0.36), smoothstep(0.35, 0.66, slope) * 0.6);
albedo = lerp(albedo, float3(0.52, 0.51, 0.47), smoothstep(0.66, 0.92, slope) * 0.5);
```

Every vertical wall has slope ≈ 1.0, so every building facade is being pulled ~60%
then ~50% toward grey. This is the single biggest obstacle to clean SC2013 color —
the curated palettes in `building_generator.cc` / `civic_generator.cc` never reach the
screen at full saturation.

Fix: gate the blend on a per-vertex flag bit. `ImportedScenePackedVertex::flags`
(`src/import/imported_scene.h`) uses only bit 0 (`kImportedMaterialFlagAlphaTest`);
claim bit 1 as `kImportedMaterialFlagTerrainSlopeBlend`, set it only in the terrain
meshers (`strategy_map_mesh.cc` terrain prefix, citybuilder ground quads), and skip
the blend otherwise. Vertex shader passes the flag through already (it forwards
`flags` for alpha-test).

## 2. Baked per-vertex AO on procgen meshes  (medium, the signature SC2013 cue)

Outdoors, the imported-static shader now multiplies the interior sky-visibility term by
screen-space SSAO (`sampleSsaoAmbientFactor`, folded into `ambientOcclusion` — see
`imported_static.frag.slang`), scoped to the ambient term only so direct sun stays
clean. That's still screen-space and blur-radius-limited, and it can't give the stable,
drawn-on corner shading that sells the toy look.

The voxel path already proves the pattern: `src/world/chunk_mesher.cc` bakes 4-bit
per-vertex AO (`cornerAoLevel`, weights 0.36/0.36/0.20), and
`voxel_packed.frag.slang` multiplies SH ambient by it
(`ShHemisphereVertexAoAmbientPolicy`). Port the idea to the CSG path:

- **Storage:** untextured geometry (`textureIndex == 0xFFFFFFFF`) never reads `uv`,
  so pack AO into `uv[0]` for the vertex-color path — zero layout change, no
  serialization break. (Alternative: an 8-bit range in `flags`.)
- **Bake:** compute in or just after `triangulate()` (`mesh_emit.cc`), where the full
  `CsgMesh` polygon soup is still available for adjacency queries. A cheap heuristic
  works at this scale: darken vertices at concave edges and where a face meets the
  ground plane (y ≈ lot height), plus optional short ambient ray casts against the
  building's own CsgMesh for inner corners (setbacks, art-deco tiers).
- **Consume:** in `imported_static.frag.slang`, fold the vertex AO into the same
  `ambientOcclusion` term the SSAO factor already multiplies (ambient only, like the
  voxel policy — direct sun should stay clean or corners look dirty instead of soft).
- **Test:** headless via `odai_procgen_tests` — assert AO ∈ [0,1], ground-adjacent
  vertices darker than roof vertices, determinism.

## 3. Bevel/chamfer primitive  (medium)

No fillet/bevel exists — the only chamfer is the 45° art-deco corner cutter in
`building_generator.cc` (`cornerChamferCutter`). SC2013 massing reads "toy-like"
largely because every exterior edge is chamfered ~2–5 cm so it catches a distinct
lit facet.

Add `chamferConvex(CsgMesh&, float width)` (or a `makeBeveledBox`) in
`src/procgen/primitives.{h,cc}`: for a convex solid, offset each face plane inward
and re-intersect, emitting one new quad per original edge and one polygon per original
vertex — no BSP needed for convex input. Each bevel face gets its own plane normal, so
the flat-shading pipeline lights it as a distinct facet automatically, and it inherits
the face color (or a slightly lightened tint of it, which is exactly the SC2013 edge
highlight).

Cautions:
- Beveled solids that overlap must go through real `csgUnion`, not the cheap
  `merge()` (`csg.h`), or seam faces interpenetrate.
- Keep the width small and screen-space aware — at citybuilder zoom, 0.03–0.06 units.
- Validate with the existing watertightness harness in `tests/procgen_tests.cc`
  (`signedVolume`, `expectClosed`, `eulerCharacteristic`).

Apply first to building massings and civic silhouettes; props are small enough on
screen that bevels there are wasted triangles.

## 4. True tilt-shift  (small, shader-only)

`depthOfFieldRadiusPixels` (`tone_map.frag.slang` ~line 134) is strictly one-sided:
`max(viewDepth - sharpDistance, 0.0)` — nothing in front of focus ever blurs, so the
"miniature" illusion is half-missing. Bonus finding: `dofConfig2.x` is already
uploaded every frame from `depthOfFieldNearBlurScale` (`frame_run.cc:873`, with an
existing debug slider in `ui_panels.cc`) **but no shader reads it** — a ready-made
slot.

- Make the radius two-sided: `abs(viewDepth - sharpDistance)`, scaling the near side
  by `dofConfig2.x`.
- Optionally add a screen-space Y-band term (blur ∝ distance from horizontal center
  line) — the classic tilt-shift approximation, and cheaper/stabler than pushing the
  depth-based blur hard.
- The 12-tap fixed hexagonal kernel bands beyond ~6 px radius; if a strong blur is
  wanted, bump taps with radius or do the blur at half-res.

The citybuilder camera (20° FOV telephoto, `citybuilder_app.cc`) is already correct
for the look; only the near-field blur is missing.

## 5. Unify and grade the palettes  (small, content-side)

Three independent palettes exist: `kBrickPool`/`kTrimPool`/`kDecoBodyPool`/... in
`building_generator.cc`, the accent-color block in `civic_generator.cc`, and
citybuilder's own ground/road colors. SC2013's cohesion comes from one constrained
ramp set (limited hues, aligned value steps, slightly desaturated mids, warm light /
cool shadow). Pull the pools into a shared `src/procgen/palette.h` so the whole city
can be re-graded in one place, and keep saturation headroom for the tone mapper's
StylizedVivid preset instead of double-saturating.

## 6. Optional polish (cheap, ordered by payoff)

- **Roof/wall two-tone**: SC2013 tops most massings with a distinctly lighter or
  accent-colored roof slab; the generators already emit roofs as separate faces, so
  this is palette-only.
- **Vertical gradient per face**: darken vertex color slightly toward the ground
  (fake bounce/dirt) at emit time in `MeshBuilder`/`triangulate` — pairs with baked
  AO, costs nothing at runtime.
- **Outline pass**: nothing exists today, but the normal-depth prepass
  (`imported_static_normaldepth.frag.slang`) is already bound to the tone-map pass —
  a Sobel over it gives clean silhouette/crease lines with no new render target.
  SC2013 itself doesn't outline, so this is for pushing further toward a
  cartoon/board-game read; keep it toggleable.
- **Vignette**: none exists; a two-line addition to the tone-map grading chain, and
  it strengthens the diorama framing.

## What NOT to do

- Don't smooth-shade anything. Flat shading is already exact by construction
  (duplicated verts per face); the only smooth normals in the pipeline are the
  deliberate terrain-fan fix in `strategy_map_mesh.cc` (keep it — smooth terrain
  under faceted buildings is also what SC2013 does).
- Don't add textures. The `textureIndex == 0xFFFFFFFF` vertex-color path is the
  style; window detail as painted-on quads (`building_generator.cc`) already scales
  better than any texture would at diorama zoom.
- Don't reach for a toon/cel step-lighting ramp. SC2013 is soft-lit flat-poly, not
  cel-shaded; the wrapped diffuse (`ndotl * 0.65 + 0.35`) plus SH ambient is already
  the right lighting model.

## Suggested order

1. Slope-blend gating (unlocks the palettes; a flag bit + 3-line shader change)
2. Baked vertex AO (biggest visual step; fully headless-testable)
3. Bevels on building/civic massings
4. Two-sided tilt-shift DOF
5. Palette unification + roof two-tone
6. Vignette / outlines as taste dictates

Steps 1–3 and 5 are CPU/mesh-side and covered by `odai_procgen_tests` on Linux;
steps 4 and 6 are shader-only and need a Vulkan machine to eyeball.
