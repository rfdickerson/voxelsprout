# RENDER-TEX-001 verification

Verified on 2026-10-03 in RelWithDebInfo on Intel Graphics (LNL), Vulkan 1.4,
with the SDK validation layer enabled. Retail files, replacement images,
JSON reports, frame CSVs, and captures remain local under `/tmp/render-tex-*`.
The replacement profile contains generated images; this is a texture pipeline
check, not certification of a particular commercial or community texture pack.

## Runtime policy

- TES3 cell streaming defaults to full source resolution. `ODAI_FNV_TEX_SIZE=N`
  explicitly selects a maximum dimension; `0` means full resolution. Other
  games retain their existing default ceilings.
- Provider priority wins before extension priority: later enabled profile
  layers, then loose files over archives within a layer, then archive order.
  Equal-provider candidates prefer DDS, TGA, BMP in that order. Legacy NIF and
  LTEX `.tga`/`.bmp` names can select shipped DDS or real replacement images.
- DDS color support includes 24-bit RGB/BGR, 32-bit RGBA/BGRA/BGRX, BC1, BC2,
  BC3, and DX10 BC7. TGA supports 24/32-bit true-color raw/RLE images and all
  horizontal/vertical origins. BMP supports 24/32-bit BI_RGB with padded rows,
  including top-down images. Paletted images, JPEG/PNG, BMP compression, and
  other TGA encodings are rejected.
- Color uses sRGB sampling. RGBA mip generation averages color in linear light
  and alpha independently; linear material data remains linear. Single-level
  BC1/2/3/7 color images expand through pinned `bcdec`, generate a complete RGBA
  chain, and obey explicit ceilings. Authored compressed chains remain intact.
  Rectangular mip dimensions use `max(1, dimension / 2)`.
- Legacy decodes are bounded to 16K dimensions and 512 MiB of top-level RGBA;
  DDS payload/decompression allocations are also bounded. The Vulkan upload
  validates actual device dimensions, format features, and chain byte counts.
  Unsupported device formats and failed required uploads produce controlled
  scene-load failures with paths, rather than successful gray materials.
- The decoded cache retains at most its configured byte budget (512 MiB by
  default). Larger requests stay caller-owned. Scene texture-slot budgets
  report failure explicitly. GPU residency logs include allocated bytes.
- Fresh asset sources have separate decoded-cache identities. Cooked identity
  includes ordered resource manifests (paths, sizes, timestamps) and quality;
  GPU identity includes bytes, dimensions, mips, color role, and sampler state.
  Cooked build version 115 invalidates older imports without changing serialized
  scene/chunk layouts. Live file watching is outside this capability.
- TES3 base descriptors preserve selected UV sets and S/T clamp modes; classic
  material and vertex colors reach the packed shader input. Version 4.0.0.2
  base descriptors do not contain the later NIF texture-transform structure.
  Instance translation/rotation/scale uses the existing model transform.
- The normal sky/cloud background now precedes world transparency, so glass
  over sky survives; reverse-Z, the depth prepass, and the existing material
  and explicit Vulkan synchronization paths remain in use.

## Portable acceptance

`tests/tes3_texture_fixture.h` supplies generated NIF, DDS, TGA, and BMP assets.
No retail bytes are committed. The verification matrix is covered by:

| Contract | Verification |
| --- | --- |
| TEX-TEST-001 | Import tests: profile/archive precedence, mixed case and aliases, higher-layer TGA over base DDS/BMP, deterministic same-layer DDS preference, plugin-local LTEX ownership. |
| TEX-TEST-002 | Scene tests: DDS RGB/RGBA and BC1/2/3/7, raw/RLE TGA, BMP padding/origins, asymmetric pixels, alpha, rectangular/sub-block mips, truncated inputs and RLE overruns. Vulkan checks render BC2/3/7 with authored and generated mips. |
| TEX-TEST-003 | NIF UV-set selection, exact coordinates, clamp, material tint, transformed instances, and shared-image sampler identities; framebuffer repeat selects red while clamp selects green at the same independently chosen point. |
| TEX-TEST-004 | Existing TES3 LAND/VTEX/LTEX assignment, elevation, weights, default and cell-edge tests; top-down Vulkan samples three painted regions, default ground, and a blend boundary. |
| TEX-TEST-005 | CPU full/reduced rectangular 2K/4K/8K and single-level checks; square BC1 2K/4K/8K resident-dimension and red/green/blue/yellow framebuffer checks; oversized headers/upload rejection. |
| TEX-TEST-006 | Fresh resource/cache reopen, byte changes, quality/color/sampler identity, shared material bindings, bounded retention, existing residency/bindless tests, GPU eviction/re-entry, and optional harness cached-build byte equality. |
| TEX-TEST-007 | Normal NIF/cell import, Vulkan upload/draw/capture: object quadrant pixels, normal-path tint, cutout, zero and half opacity, plus low/high terrain region equality. Existing water depth and GI smoke remain passing. |
| TEX-TEST-008 | Missing object/ground assets make the strict texture harness fail; malformed/unsupported encodings, cache/scene budgets and impossible Vulkan texture layout produce explicit diagnostics. |

The object quadrant samples are red, green, blue, yellow at each resident
resolution. BC7 red is `(255,6,6)` after display encoding, within the expected
endpoint tolerance. Normal-path alpha pixels are opaque `(229,0,0)`, cutout/
zero opacity `(112,103,139)` matching sky, half opacity `(200,33,75)`, and
dark material tint `(30,0,0)`.

Low/high terrain samples match exactly in the recorded run:
green `(76,152,70)`, boundary `(123,153,113)`, neutral rock `(160,154,149)`,
default ground `(106,126,151)`, blue `(66,59,222)`. Tests allow three channel
values of variation between resolutions. Neutral rock allows the small
atmospheric color shift across the controlled 9,000-unit sightline.

Required commands:

```sh
cmake --preset linux-vcpkg-relwithdebinfo
cmake --build --preset linux-vcpkg-relwithdebinfo -j
ctest --test-dir build-linux-relwithdebinfo --output-on-failure
python3 tools/ai/validate.py
```

The full optimized suite passed 66/66, including importer, serialization,
render policy, terrain, residency, bindless, GPU arena, and probe record checks.
Navigation validation passed 8/8. Vulkan smoke was run separately because this
host has a local X display but no `xvfb-run`; it is not counted as a CTest pass.
Run it from the build directory with a working display, SDK layer path and
loader library path:

```sh
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d" \
LD_LIBRARY_PATH="$VULKAN_SDK/lib:$VULKAN_SDK/lib/VulkanLoader/lib" \
./odai_imported_scene_vulkan_smoke
```

The native smoke exits successfully with zero Vulkan validation errors. Its
deliberately invalid upload logs one required-texture failure, as expected;
this is an assertion of controlled rejection, not an accepted missing fixture.
The 8K BC1 image retains 8192x8192, 14 mips, and 50,331,648 allocated GPU bytes.

## Strict real-content harness

Use `tests/fixtures/harness/tes3_textures_balmora.json` with a local version-1
Morrowind profile. Plugins and archives are ordered Morrowind, Tribunal,
Bloodmoon. Archive paths must be absolute or relative to the profile file.

```sh
build-linux-relwithdebinfo/odai_bethesda_probe --tes3-texturecheck \
  /tmp/render-tex-vanilla-profile.json \
  tests/fixtures/harness/tes3_textures_balmora.json > /tmp/vanilla-textures.json
build-linux-relwithdebinfo/odai_bethesda_probe --tes3-texturecheck \
  /tmp/render-tex-highres-profile.json \
  tests/fixtures/harness/tes3_textures_balmora.json > /tmp/highres-textures.json
```

All six checks pass: zero missing textures, zero mesh failures, valid bindings.
This certifies the selected scenes, not every possible cell or installed mod.
An intentionally hidden, fully parsed skeleton with no image requests is
reported as an intentional reference omission, rather than a missing texture.
The audit includes requested/resolved paths, provider, archive, content
fingerprint, source/resident dimensions, mips, clamp, format and bytes; enum
formats are RGBA8=0, BC1=1, BC3=2, BC7=5, BC2=6, RGBA8-sRGB=8.
Missing LTEX namespace references identify their LAND grid and index.

For repeatable replacement evidence, generate constant-color BC1 files under
an added profile layer, without using retail source pixels:

```python
import json, struct
from pathlib import Path
root = Path('/tmp/odai-tes3-highres')
for name, size in [('tx_wood_oldwood_design_01', 2048),
                   ('tx_hlaalu_doorplank_01', 4096), ('tx_scrubplain_01', 8192)]:
    levels = size.bit_length()
    words = [124, 0x81007, size, size, size*size//2, 0, levels] + [0]*11
    words += [32, 4, int.from_bytes(b'DXT1', 'little'), 0, 0, 0, 0, 0]
    words += [0x401008, 0, 0, 0, 0]
    data = bytearray(b'DDS ' + struct.pack('<31I', *words))
    for level in range(levels):
        side = max(1, size >> level)
        data += struct.pack('<HHI', 0x7bef, 0, 0) * (((side+3)//4)**2)
    path = root/'textures'/(name+'.dds')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
profile = json.loads(Path('/tmp/render-tex-vanilla-profile.json').read_text())
profile['layers'] = [{'id':'highres', 'name':'Generated high resolution fixtures',
                      'path':str(root)}]
Path('/tmp/render-tex-highres-profile.json').write_text(json.dumps(profile))
```

## Optimized measurements and capture poses

Setting `measure_cached: true` on each scene-textures check records
`scene_build_ms` with an empty decoded cache and `cached_build_ms` on a second
build. The second build must have identical texture bytes/paths and draw counts.
These are CPU scene assembly/decode timings; OS file pages were warm. Separate
`build_ms` includes inventory verification and is not a decode benchmark.

| Scene | Textures / packed draws, both profiles | Vanilla decoded bytes | Replacement decoded bytes | Vanilla cold / cached ms | Replacement cold / cached ms |
| --- | --- | --- | --- | --- | --- |
| Balmora, Guild of Mages | 125 / 1071 | 2,035,908 | 16,012,876 | 18.14 / 17.55 | 63.58 / 32.83 |
| Balmora, Guild of Fighters | 91 / 895 | 1,500,440 | 12,682,576 | 11.24 / 9.89 | 35.51 / 24.45 |
| Balmora exterior (-2,-2) | 39 / 562 | 1,434,080 | 57,311,792 | 9.12 / 8.55 | 176.19 / 72.53 |

Native captures use 768x432 logical windows, 1152x648 native-DPI framebuffers,
render scale 1, fly mode, no HUD/resume/cache, stationary benchmark motion,
and the same poses in each profile. Interior: Mages guild, source position
`-35.032,-575.067,828.033`, yaw 0, pitch -8, exposure range 8..8, frame 120.
Exterior: Vvardenfell, source position `-19920,300,12960`, yaw -90, pitch -8,
hour 14, clear weather, exposure 1..1, fixed dt, zero warmup, frame 300.
Inspection found stable object seams/orientation, terrain scale/assignment,
architecture and vegetation alpha, with no unresolved required textures.

Reproduce captures using `--profile PROFILE --stream DATA --no-resume --no-cache`
plus `--interior 'Balmora, Guild of Mages'` or `--worldspace Vvardenfell`, and
`--screenshot OUTPUT.ppm FRAME`. Supply the poses through
`ODAI_FNV_SPAWN_POS`, `ODAI_FNV_YAW`, `ODAI_FNV_PITCH`; use
`ODAI_WINDOW_SIZE=768x432`, `ODAI_RENDER_SCALE=1`, `ODAI_FNV_FLY=1`,
`ODAI_FNV_NOHUD=1`, `ODAI_FNV_BENCH=1`, `ODAI_FNV_BENCH_SPEED=0`,
`ODAI_FNV_BENCH_TURN=0`, `ODAI_FNV_EXPOSURE_RANGE`, and exterior
`ODAI_FNV_BENCH_FIXED_DT=1`, `ODAI_FNV_BENCH_WARMUP_FRAMES=0`,
`ODAI_FNV_HOUR=14`, `ODAI_FNV_WEATHER=clear`.
`ODAI_FRAME_STATS_CSV=OUTPUT.csv ODAI_BENCHMARK_CSV=1` records timings.

| Native scene | Vanilla / replacement allocated texture bytes | Vanilla / replacement median GPU ms | Visible draws / triangles, both profiles |
| --- | --- | --- | --- |
| Mages guild (last 60 frames) | 5,881,856 / 21,585,920 | 4.156 / 4.131 | 43 / 203,471 |
| Exterior, 25 resident cells (last 120 frames) | 58,900,480 / 121,741,312 | 7.497 / 7.211 | 64 / 2,481,442 |

Allocated bytes sum actual resident texture allocation log entries, including
actors and neighboring streamed cells. Exterior median chunk-add CPU time is
9.68 / 47.21 ms; this includes upload/geometry publication, not GPU copy-only
time. Frame CPU medians are 16.608 / 16.514 ms outside and 16.641 / 16.602 ms
inside. These short local runs demonstrate stable geometry and bounded
residency; they do not establish a frame-rate target or a performance win.
