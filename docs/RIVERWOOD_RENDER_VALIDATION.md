# Riverwood render validation

Riverwood targets the unmodded Skyrim Special Edition presentation using the
installed retail assets. The renderer uses authored weather colors, restrained
sky fill, directional sunlight, XeGTAO contact occlusion, and screen-space
diffuse bounce. AO affects ambient light and diffuse wrap; it does not multiply
the finished image or direct sunlight.

## Native presentation and startup policy correction (2026-09-08)

Interactive launches now follow the native framebuffer instead of silently
injecting `ODAI_RENDER_SIZE=1920x1080`. Explicit size/scale and upscaler-quality
requests still take precedence. Fixed-size benchmark windows intentionally disable
HiDPI framebuffer scaling; do not use `ODAI_WINDOW_SIZE` for ordinary interactive
launches on a scaled desktop. The checked window is 1920×1024 logical with a
3840×2048 framebuffer; render and swapchain extents both equal 3840×2048.
The existing surface/swapchain path is retained (Khronos
[WSI guide](https://docs.vulkan.org/guide/latest/wsi.html)).

Skyrim exterior lighting was previously selected before `initStreaming()` detected
the game. Standard launches inherited generic wrap=0.35, ambient=1, and disabled
SSGI. Selecting the policy after detection now applies wrap=0.10, ambient=0.62,
sun=1.08, and exterior SSGI. Exposure uses key 0.10 with range 0.18–1.45 to retain
headroom for authored IMGS brightness and contrast. Explicit environment overrides
remain authoritative. Image-space grading and texture colors are unchanged.

Local diagnostic captures compare the original presentation, lower exposure floor,
image-space-disabled output, corrected startup policy, and a restrained exposure
target (`captures/presentation-*.png`). The final native capture and log are
`captures/presentation-native.*`; it rendered SSGI and completed synchronization
validation without errors or warnings, and reported no renderer-owned image leaks.
Both runtime builds succeed and all 38 Debug CTests pass, including image-space,
upscaler and imported-lighting-policy coverage. Haze and material-response fidelity
still need matched retail comparisons; this is not a retail-parity claim.

## Reproducible stills

Build the optimized runtime, then capture the four acceptance views:

```bash
cmake --build --preset linux-vcpkg-relwithdebinfo -j 6
python3 scripts/capture_riverwood.py --view street --look day
python3 scripts/capture_riverwood.py --view mill --look day
python3 scripts/capture_riverwood.py --view shade --look overcast
python3 scripts/capture_riverwood.py --view forest --look dusk
```

The script fixes native resolution, camera, weather, time, streaming radius,
texture ceiling, anisotropy, TAA, AO, and LOD distance. Each PNG has a JSON
sidecar containing the complete command, environment, executable hash, pixel
hash, and sampled GPU timings. Captures and installed game data remain ignored.

Use `--size 3840x2160` for material and foliage inspection. Use `--variant ao`,
`--variant gi`, and `--variant effects-off` to isolate contact occlusion,
indirect diffuse, and the combined baseline. `--validation` enables Vulkan and
synchronization validation for the same deterministic view.

The existing moving-camera acceptance route exercises vegetation residency,
LOD transitions, water, and temporal history:

```bash
ODAI_WINDOW_SIZE=1920x1080 ODAI_RENDER_SIZE=1920x1080 \
ODAI_FNV_HOUR=10.5 ODAI_FNV_NOHUD=1 ODAI_GPU_TIMINGS=1 \
build-linux-relwithdebinfo/odai --stream "$SKYRIM_DATA" \
  --plugin Skyrim.esm --worldspace Tamriel --no-resume \
  --tour-file assets/tours/riverwood_to_whiterun.txt --flythrough 45 \
  --capture-video captures/riverwood-to-whiterun.mp4 60 45
```

## Acceptance checks

- Pale logs and roofs retain texture and highlight detail in clear daylight.
- Building shade remains readable without flattening surface orientation.
- Foundations, timber joints, rocks, and tree roots have contact shading without
  silhouettes or broad halos.
- GI adds restrained local color bounce and does not brighten disocclusions or
  leave history trails during the moving route.
- Pine and shrub cards remain two-sided, alpha-tested, shadowed, and softly
  backlit; LOD changes do not expose opaque rectangles or emissive foliage.
- Terrain blends remain continuous across cell boundaries and keep their
  authored world-space scale.

The comparison establishes behavior in the listed Riverwood views. It should
not be described as pixel parity with Skyrim SE without matching retail captures
from the same cameras, weather, time, and display transform.

## Grass pipeline validation (2026-09-07)

Skyrim LTEX GNAM associations now resolve GRAS records and original NIF/DDS
assets through the existing cell builder. Load-order remapping and deleted
records are retained. A deterministic, half-open 64-unit candidate grid samples
the LAND paint stack and triangle heights. GRAS density, slope limits, all eight
water modes, height variation, uniform scaling, and slope alignment control
placement. Grassless paint suppresses underlying grass; nearby collision
geometry excludes obstructed roots. Generated roots have no gameplay identity
or collision and stream with the owning cell. Cache build version 90 invalidates
previous grassless cell results without changing scene/chunk struct layouts.

Grass uses the existing two-sided, alpha-tested foliage material and shares
root-height collapse from 2,800 to 4,000 units in color/depth/shadow vertices.
No additional GPU resources, passes, or synchronization dependencies were added.

Validation evidence is local under `captures/grass-pipeline-*` (PNG, JSON, log):

- Optimized runtime/shader build succeeded; all 35 Debug CTest tests passed.
  Optimized CTest passed 31/35 with the previously observed inventory, runtime,
  save, and world-map-cache failures.
- Synthetic coverage includes malformed/deleted records, negative and adjacent
  cell ownership, repeatability, grassless roads, zero density, water modes,
  worldspace/interior exclusion, and slope alignment at different rotations.
- `ODAI_GRASS_TEST_DATA="$SKYRIM_DATA" build-linux-relwithdebinfo/odai_fnv_import_tests`
  also verifies original assets and scene serialization. Four Riverwood cells
  produced 99 eligible clumps (73 after collision exclusion), with no missing
  grass textures or grass physics triangles.
- The optimized forest capture reached 81 resident cells with none loading.
  Its final sampled GPU window was p50 42.95 ms / p95 45.15 ms at native
  1920×1080 on Intel LNL. This is whole-scene timing, not a measured grass cost;
  no before/after memory delta has been established.
- The Debug forest capture logged validation enabled, no validation errors, and
  no renderer-owned image leaks on shutdown.

The candidate distribution and distance collapse are engine approximations,
not verified replicas of Bethesda's scatter/fade algorithm. GRAS position range,
color range, wave period, and vertex-lighting flags are parsed but do not yet
drive dedicated grass variation/animation. Retail density parity and moving
camera shimmer still need matched reference validation. Current captures also
show broader lighting and tree-silhouette differences unrelated to grass import.
