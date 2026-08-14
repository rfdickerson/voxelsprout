# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Reference Touchstones

Four games define what this project is actually building toward. Each maps to an existing area of the codebase — none are being cloned literally, but each names the specific feel/systems to chase:

- **Morrowind** — open-world exploration, hand-placed regional identity, readable terrain and settlements (`src/world/`, `src/import/`).
- **Civilization VI** — turn-based hex-grid 4X strategy: yields, tech tree, borders, map modes/lenses, diplomacy (`src/game/`).
- **SimCity (2013)** — agent-driven city simulation: RCI zoning, traffic congestion, land value, service coverage, data-layer overlays (`src/games/citybuilder/`, already explicitly modeled on it).
- **Dragon Age: Origins** — party-based real-time-with-pause tactical combat, branching dialogue with consequences, companion relationships/approval. The foundations exist (`src/dialogue/`, `src/anim/`, the GPU skinning pass) but no party/combat layer is built — see `docs/ROADMAP.md`'s Party RPG / Narrative section.

These are creative direction, not a mandate to merge four genres into one game — each pillar can keep developing largely on its own track. `docs/ROADMAP.md` tracks concrete status and priority per touchstone.

## Non-goals

This project is not:
- an ECS experiment
- an enterprise architecture exercise
- a dynamic-loading plugin/mod distribution platform (no `dlopen`/DLL loading, no Steam Workshop-style discovery)

It prioritizes explicit, hand-written control flow over implicit machinery (e.g. render passes never assume implicit barriers — see Renderer internals below). These are style constraints on *how* things get built, not a cap on *which* genres get pursued — see Reference Touchstones above. Note the "Engine plugins" pattern documented below is a statically linked, compile-time composition seam — a different thing than the distribution platform this list rules out.

## Build Commands

**Dependencies come from vcpkg** (`vcpkg.json` manifest). `nlohmann_json`, `ZLIB`, `Lua`, and `sol2` are `REQUIRED` at the top of `CMakeLists.txt`, so **even a tools/tests-only build needs the vcpkg toolchain** — there is no dependency-free configuration. `stb` and `nanosvg` are header-only and degrade gracefully (`ODAI_STB_INCLUDE_DIR` / `ODAI_NANOSVG_INCLUDE_DIR_OVERRIDE` override them for headless checks; without nanosvg the SVG importer compiles to a no-op).

**Windows (app + games + tools):**
```powershell
cmake -S . -B cmake-build-release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build cmake-build-release --target odai -j 4
```

**Linux / WSL2 — full build, matching CI** (Vulkan targets build and run against mesa/lavapipe):
```bash
cmake -S . -B cmake-build-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DODAI_BUILD_EXAMPLES=ON \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=x64-linux
cmake --build cmake-build-linux -j
```

**Linux / WSL2 — tools + tests only** (skips everything Vulkan/GLFW):
```bash
cmake -S . -B cmake-build-linux \
  -DODAI_BUILD_APP=OFF -DODAI_BUILD_TOOLS=ON -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build cmake-build-linux -j 4
```

CMake options: `ODAI_BUILD_APP` (ON), `ODAI_BUILD_TOOLS` (ON), `ODAI_BUILD_EXAMPLES` (OFF), `ODAI_RENDER_BACKEND` (`VULKAN`; `DX12`/`METAL` are declared but unimplemented). Every app/game/demo target sits behind `if(ODAI_BUILD_APP)` *and* the `VULKAN` backend guard.

### Optimized builds — required before believing any perf number

**Never profile a Debug build.** Measured on this tree (64 procedural chunks, gcc 13, best of 3):

| | worldgen | chunk meshing |
|---|---|---|
| `Debug` (`-O0 -g`) | 16255 ms | 617 ms |
| `RelWithDebInfo` (`-O2 -g`) | 6551 ms | 85 ms |
| `Release` (`-O3`) | 2041 ms | 75 ms |

That's **8x** on worldgen and **8x** on meshing between Debug and Release — a Debug measurement is not a measurement. Note worldgen is a further 3.2x from `-O2` to `-O3` (the noise loops vectorize), so use `Release` when the number is the point and `RelWithDebInfo` when you also need readable stacks.

Presets (`CMakePresets.json`) — the three original presets are Debug and unchanged:

| Preset | Build type |
|---|---|
| `default`, `vcpkg`, `linux-vcpkg` | `Debug` |
| `vcpkg-relwithdebinfo`, `linux-vcpkg-relwithdebinfo` | `RelWithDebInfo` — **use this to profile** |
| `vcpkg-release`, `linux-vcpkg-release` | `Release` |

```bash
cmake --preset linux-vcpkg-relwithdebinfo && cmake --build --preset linux-vcpkg-relwithdebinfo
```

Optimization level comes from `CMAKE_BUILD_TYPE` — no `-O` flags are hardcoded. Two opt-in switches a plain build type does not turn on:

- **`ODAI_ENABLE_LTO`** (OFF) — interprocedural optimization for optimized configs only; costs link time. Falls back with a warning if the toolchain can't do it.
- **`ODAI_ENABLE_NATIVE_ARCH`** (OFF) — `-march=native` / `/arch:AVX2`. **Local profiling only.** Wider SIMD and FMA contraction change floating-point results, and this project pins content-affecting hashes and RNG with golden vectors because worldgen output must reproduce; the binary may also fault outright on an older CPU. `-ffast-math` is never enabled anywhere, deliberately.

If `CMAKE_BUILD_TYPE` is unset on a single-config generator it now defaults to `RelWithDebInfo` (an empty build type means no `-O` *and* no `-g`). Passing a build type explicitly always wins — the Debug presets above are unaffected.

**Build a single target:**
```powershell
cmake --build cmake-build-release --target odai_ui_tests -j 4
```

**Run tests:**
```powershell
ctest --test-dir cmake-build-release --output-on-failure   # all
ctest --test-dir cmake-build-release -R odai_ui_tests -V   # one suite
ctest --test-dir cmake-build-release -N                    # list registered tests
```

### Test targets

All are plain executables with a hand-rolled `int main()` and inline assertions — **no test framework** — except `odai_stability_gtests`, the one GTest suite (skipped with a `message(STATUS)` if GTest isn't found for your triplet). No test executable links Vulkan; the only headless-Vulkan check is the offscreen-capture smoke test under `examples/ui_stress_test/`, built when `ODAI_BUILD_EXAMPLES=ON`. `ctest -N` lists the full set; the sources in `tests/` map onto the target names.

Three suites carry contracts the source alone won't tell you: `odai_core_types_tests` pins content-affecting hashes and RNG with **golden vectors** (worldgen output must reproduce), `odai_fnv_import_tests` runs on **synthetic fixtures only** — never real game data — and `odai_content_tests` covers `mods/base` load order, not just parsing.

CI (`.github/workflows/ci.yml`) runs Linux only (full build including Vulkan on lavapipe, `slangc` installed, examples ON). Windows is a supported local dev target (see Build Commands above) but is not built in CI.

### Content generation tools

The offline generators and headless sim harnesses (`ODAI_BUILD_TOOLS=ON`, all pure CPU) are
documented in the `.claude/skills/content-tools` skill — including `--sweep N`, the CPU
regression harness for the turn loop.

**Stale README warning:** `README.md` still documents a Morrowind `odai_balmora_cooker` plus ESM/LAND/LTEX extraction. **Neither the target nor any Morrowind import code exists in the tree** — `src/import/` holds only the shared `ImportedScene`/DDS/GPU-scene code plus `import/fnv/`. Trust the source and this file over that part of the README.

**Run the app / games:**
```powershell
$env:ODAI_STRATEGY_MAP = "strategy_map.smap"
cmake-build-release\odai.exe
cmake-build-release\odai_game_citybuilder.exe   # also _snake, _minesweeper, _stellaris, _swtor, _voxelcraft
```

**Fallout 3 works through the same path.** Both games are the same engine and the same
formats, so `--stream <FO3 Data> --plugin Fallout3.esm --worldspace Wasteland` streams the
Capital Wasteland with no Fallout-3-specific code: every BSA opens, 38443 exterior cells index,
and actors build, animate and talk. Two things were New-Vegas-shaped and are no longer: a voice
path's first component is the **plugin's own file name** (`sound\voice\fallout3.esm\...`), which
was hardcoded and silently voiced none of Fallout 3's 46 actors; and the default spawn interior
`GSDocMitchellHouse` is a New Vegas default rather than a constant.

**A TEMPLATE ACTOR'S SKELETON IS NOT ON THE RECORD THAT NAMES IT.** An actor whose ACBS
template flags borrow the model (`0x0040`, Use Model/Animation) stores `marker_creature.nif` as
its own MODL -- a real, parseable NIF carrying none of the bones a body is weighted to. Taking
it does not fail: it binds a character whose every bone is unresolved. And the hop is NESTED --
TPLT lands on a levelled actor list (LVLC for creatures, **LVLN** for NPCs) whose entries are
routinely more lists, so following one level finds no actor and quietly hands the marker back.
Measured on Fallout 3: 91 unresolved bones on `SpringvaleElemMiniboss` and 71 on
`LvlRaiderMelee`, both standing in bind pose. Following the chain properly takes Fallout 3 from
43 to 46 of 46 actors animated and 211 to 193 placements with no resolvable geometry.

**Fallout: New Vegas viewer** — streams the worldspace straight out of the game's own
`FalloutNV.esm` and `.bsa` archives. No cooking step and no cooked assets on disk:

```bash
cmake-build-linux/odai_game_newvegas          # finds an installed copy on its own
cmake-build-linux/odai_game_newvegas --stream "<.../Fallout New Vegas/Data>"   # or point at it
```

`--worldspace <EditorID>` (default `WastelandNV`) and `--plugin <Plugin.esm>` (default
`FalloutNV.esm`) select what to stream; `ODAI_FNV_LOAD_RADIUS` sets the cell load radius.
`--scene <path.bin>` still loads a single cooked scene instead, which is what
`odai_newvegas_cooker` produces — that path is for baking a fixed region or a distant-LOD
tier (`--lod`), not for normal play.

**Asset mods (texture packs).** `docs/FNV_MODS.md` has the install + launch commands; this
section is the mechanism behind them. `--mod <dir>` (repeatable, later wins; or
`ODAI_FNV_MODS`, `:`-separated) adds a root laid out like `Data` itself — `textures\...`, `meshes\...` —
searched ahead of the game's loose files and archives. Mod roots are indexed recursively
and looked up **case-insensitively**, which is load-bearing on Linux: packs ship
`Textures/` while NIFs ask for `textures/`, and the game's own loose-file path is a
case-sensitive `exists()` that would miss every one of them.

Two traps, both silent:

- **`ODAI_FNV_TEX_SIZE` gates whether a pack is visible at all.** The mip-drop ceiling
  defaults to **512 px** (`CellSceneBuilder::m_maxTextureSize`), so a 1K/2K pack is
  dropped straight back to vanilla resolution — full I/O cost, zero visual change. Set it
  to `1024` to actually see the pack. Memory goes as the square (BC1 1024² with mips is
  683 KB against 171 KB at 512), so this is the first knob to turn down on an iGPU.

  **What raising it actually buys is less ALIASING, not more detail — and it is not a
  vanilla no-op.** Measured on a pinned unmodded Goodsprings frame, 512 → 1024 moves 39.6%
  of pixels (mean |Δ| 15.8), and local texel contrast *falls*: 11.23 → 8.50 on the
  Prospector Saloon, 13.48 → 9.77 on a shack and road. Smoother, not sharper. The reading
  that fits: mip-dropping removes the TOP of the chain, so for a given screen footprint the
  sampler lands on a relatively more-detailed level and shimmers more; restoring the full
  chain settles it. Expect the balance to favour 1024 up close, where the extra texels
  actually resolve.

  Two corrections to the intuition this section used to invite. Vanilla art is a **mix**,
  not all ≤512 — `NV_WaterTank.dds` is 512 but `NV_ProspectorSaloon.dds` is 1024 — so the
  ceiling changes an unmodded scene too. And a full-frame sharpness proxy is the wrong
  instrument here: edge energy reads *higher* at 512 (3106 vs 1908), which is the aliasing
  being counted as detail. Compare crops on a known 1024-source asset instead.
- **Cached cells bake their textures in.** The mod set and the ceiling are folded into the
  cell-cache directory name, so installing a pack lands on a fresh cache rather than
  serving vanilla art forever. Nothing is deleted — the old cache directory just stops
  being consulted, and costs disk until removed.

Run a downloaded pack through `odai_fnv_texture_pack` first (see Content generation tools):
it mip-drops to the ceiling you intend to run and lowercases every path. NMC's SMALL pack
goes 2.0 GB → 1.5 GB at a 1024 ceiling in about a second, and the levels above the ceiling
are then read once here instead of on every cache miss.

Only the streaming path honours mods — a cooked `--scene` already has its textures baked
in, and `odai_newvegas_cooker` has no `--mod` flag yet.

**Plugin mods and weather.** `--plugin-add <Plugin.esp>` (repeatable; or `ODAI_FNV_PLUGINS`,
comma-separated) loads an additional plugin after `--plugin`. Plugins resolve from the
`--mod` directories first and the `--stream` directory second (`FalloutLoadOrder::addSearchRoot`),
so a mod's `.esp` lives beside the `.bsa` it ships with and nothing is copied into the game
install. Masters resolve automatically, so naming only the mod is enough — `--plugin-add NevadaSkies.esp` pulls in FalloutNV.esm and
the four DLC masters, in order.

`src/import/fnv/plugin_load_order.{h,cc}` is what makes more than one plugin loadable. A
formID's high byte is a mod index that is **local to the file it is stored in** — it indexes
that plugin's own TES4 master list, and one past the last master means "a record I define".
Inside `NevadaSkies.esp`, `0x05` is its own 375 new weather records; load the same mod next
to a different plugin set and that byte has to mean something else. Getting this wrong does
not fail loudly, it resolves references to the wrong records. `remapFormId()` rewrites local
→ global; every formID that crosses into engine data structures must go through it.

`src/import/fnv/weather_records.{h,cc}` reads WTHR/CLMT across the load order (473 weathers
with Nevada Skies installed). `--weather <EditorID>` forces one by name; otherwise the
worldspace's climate picks its highest-chance entry. WTHR's NAM0 is **10 color channels × 6
time-of-day slots × RGBA** — New Vegas added noon and midnight to Fallout 3's four, which is
why the record carries six `\x00IAD`–`\x05IAD` adapters and NAM0 is 240 bytes, not 160.

**Which meant Fallout 3 had NO SKY AT ALL and nothing said so.** The reader guarded on
`sub.size >= 240` (and PNAM on `>= 96`), so every one of Fallout 3's weathers failed the
guard and kept its default all-zero colour table: `sky linear rgb: upper(0,0,0)
horizon(0,0,0) fog(0,0,0)` at noon, rendering a black-to-grey void over correctly lit
terrain. That is indistinguishable on screen from a genuinely dark authored weather, which
is why it survived — the log line is the tell, not the frame. `weatherSlotsInSubrecord`
now picks the layout from the byte count and widens four slots to six (Noon ← Day,
Midnight ← Night; the first four channels are identical in both games).

Colors reach the sky through `Renderer::setWeatherSky(WeatherSkyParams)`. They **blend over**
the procedural Rayleigh/Mie sky rather than replacing it: `weight` 0 is the default and
renders byte-identically to before, which is what keeps every other game unaffected. WTHR
authors three vertical stops (Horizon at the skyline, SkyLower just above it, SkyUpper at the
zenith); collapsing that into a two-end gradient loses the narrow near-ground band that gives
these skies their look.

**`fogFarDistance` is the one field that is honoured at weight 0.** Both atmospheres — the
aerial perspective in `imported_static.frag.slang` and the volumetric height fog whose
density `frame_run.cc` derives — take the distance from the record without consulting the
weight, while the *colours* still lerp by it. That split exists so a caller can say "leave
my sky procedural, but this is how far you can see". Every Oblivion worldspace needs it:
they name no climate, so nothing publishes colours, and the 15000-unit fallback (~214 m, a
dust-storm value) put Anvil behind a wall of milk. `ODAI_FNV_FOGFAR` (default 160000) sets
what the no-weather path publishes. Measured on the Anvil tour: frame contrast (stddev of
luma) 14.0 → 22.6. Getting the two gates to *disagree* is the trap — the volumetric one
still checked `weight > 0`, so the two atmospheres calibrated themselves against distances
2.7x apart.

**Cloud layers** render from the same records. A weather's four layers (DNAM/CNAM/ANAM/BNAM,
tinted per time of day by PNAM) are uploaded through the imported-texture slot table and
sampled in the fullscreen sky pass. Two things there are non-obvious and were both bugs
first: the cloud SHAPE lives in the texture's **alpha**, not its luminance (rgb is nearly
white, so a luminance key makes the layer opaque), and the layer colour is PNAM's tint —
multiplying it by the texture's rgb squares the darkness and renders a heavy overcast pure
black.

**These textures are SKY DOME MAPS, not tiling planes.** Each is a fisheye of the whole sky:
zenith at the centre of the image, horizon on the rim of an inscribed circle. Nevada Skies'
`WesternSky1.dds` is unmistakable about it — a cloud disc with sun rays radiating into the
dark corners. So the mapping is angle-from-zenith → radius, compass bearing → angle, landing
every sample inside one copy of the texture. Two earlier attempts got this wrong: a true
plane projection `dir.xz / dir.y` diverges at the horizon into radial streaks, and the
softened `dir.xz / (dir.y + k)` that replaced it maps the sky into roughly [-0.76, 0.76] of
UV space, crosses zero, and lets the sampler WRAP — tiling the entire fisheye and putting a
visible seam down the sky with the image repeating from it. Tiling a fisheye is meaningless;
no amount of scale tuning hides it. Scrolling is likewise a ROTATION about the zenith, not a
UV translation, which would slide the fisheye off its own centre.

**`ODAI_FNV_SKY_GAIN` / `ODAI_FNV_SKY_CONTRAST` exist because WTHR colours are
display-referred.** They were authored as final sRGB for a renderer that showed them
directly; this one is HDR with an ACES curve and auto-exposure keyed to a sunlit desert.
Decoding to linear and stopping there renders an overcast sky (sRGB 23,27,30) as pure
black while the terrain looks correct. A flat gain cannot fix it — the value that makes
overcast readable washes a clear zenith to pale haze — so the decode applies
`pow(linear, contrast) * gain`, which lifts darks more than brights. This is a fudge; the
principled fix is inverting the tonemap on the GPU, where the exposure scale is known.
`ODAI_FNV_NOCLOUDS=1` separates "authored-dark gradient" from "total cloud cover", which
look identical on screen.

The decode shapes **magnitude and hue separately** — `pow(length(rgb), contrast)` on the
magnitude, `pow(rgb/length(rgb), saturation)` on the direction. Doing it per channel (which
it did first) pulls the channels toward each other, so lifting the darks also desaturated a
clear zenith from deep blue to pale haze. The split is lifted from ENB's tonemap
(Enhanced Shaders' `enbeffect.fx`), which applies contrast to `color/normalize(color)` and
saturation to `normalize(color)` for exactly this reason. `ODAI_FNV_SKY_SATURATION` tunes
it; ENB itself runs 1.25 by day against these same records and 0.9 at night.

**`ODAI_FNV_TONEMAP=enb`** swaps the post pass's ACES fit for ENB's extended Reinhard
`x(1+x/L)/(x+C)`, with Enhanced Shaders' tuned Fallout values. Two details are load-bearing:
ENB *divides* by an adaptation term (~0.1) to put scene values in the 1-10 range its `C=8`
knee expects — multiplying by this engine's exposure scale instead renders pure black — and
`grayadaptation` is adapted scene *luminance*, recovered as `0.18 / exposureScale`, not the
reciprocal of the exposure multiplier. Both mistakes were made and measured on the way in.
Default is ACES, so every other game is unaffected.

**Render scale defaults to native.** It used to default to 0.6, drawing the 3D scene at 36%
of the pixels and upscaling into a native-resolution UI composite — visibly soft. Measured
on the LNL iGPU at a 1920x1080 swapchain, 0.6 -> 1.0 costs ~3.5 ms (8.8 -> 12.3 ms/frame,
~81 fps), and only two passes move: main (3.6 -> 5.7) and SSAO (0.46 -> 1.5). Shadow (~1.3)
and prepass (~2.3) are **vertex-bound and do not change with resolution at all**, which is
why dropping resolution bought far less than it appeared to. `ODAI_RENDER_SCALE` still dials
it back, and is worth reaching for on a 4K swapchain.

**Weather audio** comes from the installed game. Fallout ships music as ~199 loose `.mp3`
under `Data/Music` (playable directly), but every ambient loop is `.ogg` inside a BSA, and
**miniaudio has no Vorbis decoder** — only WAV/FLAC/MP3. `newvegas_ogg.cc` compiles
`stb_vorbis` in its own TU and converts to `.wav` in the cell cache on first use, which is
a smaller change than threading a custom decoding backend through the audio PIMPL. Rain
plays only when the record's classification bits say it is raining. `ODAI_FNV_MUSIC`
overrides the track.

Not wired up yet: WTHR's **Ambient/Sunlight** channels, so terrain lighting does not yet
respond to the weather, and there are no rain **particles** — the mood is sky, fog and
sound only.

**Victor** (`src/games/newvegas/newvegas_victor.{h,cc}`) is the one live NPC: a GPU-skinned
Securitron standing beside the spawn, animated from the game's own `.kf` clips, who answers
**E** with his real DIAL/INFO dialogue and his real recorded voice. He is the worked example
of the whole actor path, and four things about it were each a bug first:

- **A skinned template's vertices reach the GPU verbatim.** There is no
  scene-index-to-bindless-slot remap like `addImportedSceneChunk` does for world geometry, so
  `ImportedSkinnedMeshVertex::textureIndex` must already hold a bindless slot.
  `Renderer::uploadSkinnedActorTextures` is how a caller gets one.
- **`AnimationSampler::bindSkeleton(skeleton)` is the wrong overload for Fallout.** It derives
  inverse binds from the skeleton; NiSkinData stores them explicitly and they differ (see
  `FalloutCharacter::inverseBindMatrices`). Pass them in.
- **An unweighted vertex is not merely un-animated.** The skinning shader passes it through at
  its authored position, so it misses the actor's world placement too and draws at the world
  origin — rigid props parented to a bone (Victor's face screen) must be weighted 1.0 to it.
- **A creature's NIFZ list carries every part the model *can* wear**, including alternate face
  screens the game swaps at runtime, plus alpha-blended glare quads the opaque skinned path
  renders as solid slabs. Drawing the list literally gives Victor colour bars for a face.

**A STRAY `NiAlphaProperty` IS NOT TRANSPARENCY.** Fallout marks surfaces blend=1/test=0
that have nothing to blend: Goodsprings' water tank (`nv_watertank.nif`) is three shapes and
two of them — the tank body and the concrete pad under it — are authored blended while their
textures are **97% fully opaque**. Drawn through the blended pipeline that reads as a
see-through tank, and the look is the smaller half of it: a blended draw writes no depth, is
skipped by the shadow pass and by the normal-depth prepass, so the tank also casts no shadow
and contributes nothing to AO. `demoteFalseAlphaBlendFlags` (`imported_scene.cc`) re-reads the
same alpha histogram the cutout classifier builds and demotes three ways — no transparent
texels at all → **opaque**; bimodal (transparent + opaque, thin rim) → **alpha test**; a real
mid-range gradient → **left alone**, which is what keeps glass and dust sheets working. It only
ever takes work away from the blended pass, which is what makes it safe where the old
content-based guess was not (that one *forced* alpha test onto opaque geometry sharing a
cutout's texture). Measured on Goodsprings: blended pixels fall 76%. It runs on **load** as
well as on build, so cached cells are corrected without a `kCellBuildVersion` bump.

**GAMEBRYO PROPERTIES INHERIT DOWN THE SCENE GRAPH.** A `NiAlphaProperty` or
`NiStencilProperty` on a parent `NiNode` applies to every shape beneath it, so a reader
that walks only a shape's own property list imports those shapes with no alpha mode at
all — and an unflagged shape renders fully opaque, showing the black that sits under a
Fallout texture's transparent texels. `nif_scene.cc` resolves a shape's **own** properties
first and its ancestors' second: the diffuse texture is first-wins so own-first cannot
change any texture that already resolved, while alpha/stencil/unlit accumulate with `||`
so an inherited property can only ever turn one **on**. The static path carries the
accumulated ref list down its DFS; the skinned path is a flat block scan with no parent on
hand, so it reconstructs a child→parent map and walks up. Pinned by
`testNifParserInheritsPropertiesFromParentNodes`.

This was masked until recently: `applyTextureAlphaCutoutFlags` inferred a cutout from
texture CONTENT and happened to catch these. That inference is off for this importer now
(`ImportedScene::alphaFlagsAuthored`), so nothing covers the gap any more. Note the fix is
**baked into cached cells** — alpha/blend/two-sided live in the packed vertex flags — which
is why `kCellBuildVersion` (`cell_streamer.cc`) had to go to 12. Any change to what the NIF
reader decides about a material needs that bump, or every existing install keeps serving
pre-fix geometry forever and the fix appears to do nothing.

**Any new scan of a plugin MUST set `EsmReader::Visitor::onRecordHeader`.** Leaving it null
materializes every record in the file to hand to `onRecord`, which means inflating
FalloutNV.esm's 29363 compressed LAND records to go looking for a creature. Measured: two
scans that forgot it (Victor's placement and his dialogue) cost **3.4 s each** and took launch
from 2.2 s to 8.8 s. Filtering by record type on the header brought them to 73 ms and 82 ms
with byte-identical output. The failure is invisible in the result and shows up only as a slow
start, so it will not be caught by anything except looking.

Related startup costs, for scale: mod-directory indexing is ~0.2 s (5431 loose files for NMC),
and `BsaArchive::open` takes an optional **folder-prefix filter** — unfiltered it indexes all
105517 entries of `Fallout - Voices1.bsa` to keep the 487 an actor needs. Pass Victor's asset
source in rather than opening a second one: a private one carries no `--mod` directories, so
he silently ignores texture packs the world around him is using.

`.kf` reading is `src/import/fnv/kf_animation.{h,cc}` (parse, Bethesda space) plus
`buildFalloutAnimationClip` (bone-name resolution + the basis change, which must match
`buildFalloutSkeleton`'s exactly). 5009 of the game's 5014 `.kf` files parse; the 5 failures
are older NIF versions.

**B-spline-compressed interpolators are decoded** (`NiBSplineCompTransformInterpolator`):
cubic curves with 16-bit quantized control points, sampled into ordinary keys at parse time so
nothing downstream knows they existed. This is not an optional refinement — Bethesda stores
the bones that carry a *human* animation this way (both arms, both legs, the head), and keeps
the near-static ones (pelvis, spine, toes) as plain keyframes. A reader that decodes only the
second kind reports a healthy 42-of-58 tracks bound on a townsperson's idle **and still renders
a T-pose**, which is what the track count hides. Two conventions in the decode are load-bearing
and both are pinned by `testKfBSplineDecoding` (four control points = one span = a Bézier, so
the curve's endpoints are exactly its outer control points): the knot vector is **clamped**
(an unclamped spline never reaches either end of the pose), and control points dequantize as
`bias + multiplier * short / 32767`.

## Goodsprings' other actors

`src/games/newvegas/newvegas_actors.{h,cc}` + `src/import/fnv/actor_records.{h,cc}` populate
the town around Victor: every ACRE/ACHR within `kActorLoadRadius`, discovered from the plugin
rather than from a hardcoded list. `--actorsnear <plugin> <x> <y> <radius>` on the probe prints
the whole resolution, part paths included, which is where to look first.

There are **three kinds of actor** and only the first is a creature-shaped problem:

- A **CREA with a NIFZ list** carries its own geometry — MODL is its skeleton, NIFZ names the
  parts beside it. Victor is one.
- A **CREA with no NIFZ** is a spawn variant whose TPLT usually lands on a **levelled creature
  list (LVLC)**, not on another actor. Following TPLT only through CREA/NPC_ resolves none of
  the VSpawnTier1 coyotes.
- An **NPC_ carries a skeleton and nothing else.** Its body is assembled from RACE part slots —
  upper body, left hand, right hand, head — and then individual slots are replaced by what it
  is wearing. RACE is a **positional** format (NAM0 opens the head section, NAM1 the body,
  MNAM/FNAM switch sex, each INDX names the slot the next MODL fills), and the same subrecord
  types reappear past HNAM meaning something else entirely.

Two traps in the wardrobe, both of which render as an undressed town rather than as an error:
an outfit is usually **not carried directly** but as an **LVLI** the inventory walk has to
expand (a settler carries "CondOutfitRepublican02", not a shirt), and an actor's race, sex and
inventory are each **individually inheritable from its TPLT** via ACBS's template-use flags.
`resolve()` hands back **full mesh paths in every case**, because NIFZ stores names relative to
the skeleton's directory and RACE/ARMO store full ones.

Every human body NIF also ships its own **dismemberment caps** — "bodycaps", "meatneck01",
"meathead01" — ordinary skinned shapes in the same file as the skin. Drawn literally they hang
slabs of raw meat off an otherwise fine settler. They are filtered on the one thing they share,
the `textures\gore\` folder.

`ODAI_FNV_ACTORS_PARADE=<distance>` lines every built actor up in front of the camera (along
its own right vector, so `ODAI_FNV_YAW` alone picks what they stand in front of). The town's
people are spread over 12000 units and a `--screenshot` run cannot walk to them, so without it
a change to how a body is assembled cannot be looked at at all.

**A skinned shape's vertices are not necessarily in the character's space.** NiSkinData's
overall `skinTransform` maps the character's space into the *shape's own* geometry space, and
`appendFalloutCharacterMesh` composes it into the inverse bind — which normalizes the BINDING
and leaves the GEOMETRY where it was. Fallout's human parts are where that becomes visible: a
hand NIF is authored around the hand, a head NIF around the head, a body NIF around the whole
character, so a settler renders as a clothed torso with his head and both hands piled at his
feet, each the right shape in the wrong place. The vertices need the inverse of that transform
applied at append time. Every *creature* part in the game has an identity `skinTransform`,
which is why Victor never showed it. Note the bind-pose round-trip check in
`odai_newvegas_probe --character` is only a valid test for shapes whose skin space IS model
space — for a head or a hand, the correct bind pose deliberately MOVES the geometry, and
chasing a zero round-trip there is what introduced this bug in the first place.

A conversation is **modal, Skyrim-style**: movement keys are not read at all while it is up (so
a held W does not accumulate and release when the card closes), mouselook is suppressed, and
the camera eases onto Victor and holds. Two details there are not obvious. The aim targets a
point ~150 units up his body, because his placement is his FEET and aiming at the origin points
the camera at his wheel. And it deliberately aims *below* his face by a pitch offset derived
from the projection (`atan(f · tan(fovY/2))`, using the card's measured top edge) — aiming AT
the face centres it, which is exactly where the card is, and a conversation must not hide the
person talking. Gravity, the terrain pin and collision keep running, so opening a conversation
mid-step still settles the player on the ground.

It also **dollies in** (75° → 55°) and pulls **depth of field** onto the speaker, both eased.
Two things were fixed to make that possible, and both are general:

- **`camera.fovDegrees` is now honoured every frame.** It used to be latched on the first frame
  and ignored forever after, so any per-frame FOV an app set was a silent no-op. The ImGui
  slider now claims FOV only once someone actually drags it and hands it back via "Follow app"
  (`m_debugCameraFovOverride`) — without that handshake the slider and an animating game would
  fight and the value would flicker.
- **DoF grew a near field.** `tone_map.frag.slang` blurred only *beyond* the focal plane, which
  on a portrait framing separates the background but leaves whatever the camera stands behind
  razor sharp. `dofConfig2.x` was already being uploaded for this and simply never read.
  `setDepthOfField`'s new `nearBlurScale` **defaults to 0**, so no existing caller's look
  changed; ~1.25 is a tilt-shift miniature, and *below 1* stretches the near ramp for a
  portrait — needed here because Victor is ~100 units deep standing ON the focal plane, and a
  near ramp as short as the far one measurably softens his own front along with the ground.

`maxRadiusPixels` is in pixels, so a fixed value is a different-sized blur per display; the
conversation scales it by framebuffer height so the look is resolution-independent.
`ODAI_FNV_DIALOGUE_NODOF=1` keeps the framing and drops only the blur, which is the control a
measurement needs — with the camera pointed anywhere else the same crop is not the same content.

The conversation itself draws as a **centred modal card** sized for a TV: its own 48 px / 40 px
type steps (baked in the game, not through `GameApp::loadFonts`, whose four slots are a shared
contract), word-wrapped against the font's own metrics, with the highlighted reply stated three
ways at once — fill, border, and caret — because on a green-on-green palette any one of them
alone fails for someone. Replies are driven by `UiNavInput`, so a gamepad works identically to
the arrow keys, with the number keys kept as the keyboard fast path. Anything else that draws
large centred type (the location-discovery banner) must be held while it is up.

## The flythrough camera, and `--tour-file`

`--flythrough <seconds> --capture-seq <dir> <fps> <seconds>` flies a spline path and writes
numbered PPMs. The path is a **centripetal** Catmull-Rom (Barry–Goldman, α=0.5) with an
arc-length table, so speed is even along it rather than surging through tight corners.
`--tour-file <path>` replaces the built-in Goodsprings path with a text file of
`px py pz  lx ly lz` rows (`#` comments; y is height; at least four rows, because
Catmull-Rom needs four control points). That is what makes a location reachable without a
rebuild, which matters because every framing decision here is one round trip through a
render.

**An authored tour suppresses the actor hand-off, and it must.** Past 70% of the built-in
tour the camera latches onto the nearest wandering townsperson and eases its aim onto them —
a deliberate flourish, written for Goodsprings, where the tour ends among the residents.
Applied to a tour file it silently overrides the authored `lookAt`: in Megaton a pan across
the shanties became a top-down stare at one settler on the crater floor, and the symptom
reads as "my waypoints are being ignored" rather than as anything to do with actors.
`tourIsAuthored()` gates it.

Two more traps a capture run hits, neither of which fails loudly:

- **`ODAI_WINDOW_SIZE` is a request, not a contract.** The captures for the three-game
  showcase came out 2133x1200 with `1280x720` asked for. Check the PPM header before
  assuming a frame's size — an overlay composited at the wrong scale lands in a corner and
  looks like a working pipeline at first glance.
- **The first second or two of any capture is still streaming.** Goodsprings' frame 0 is an
  empty grey plane and the Capital Wasteland needed ~300 frames to fill in. Render more
  than needed and drop the head.

Diagnostics: `ODAI_FNV_VICTOR_TALK=1` opens the conversation on the first tick so a
`--screenshot` run can exercise it, `ODAI_FNV_DIALOGUE_SELECT=<n>` starts on the nth reply
(a screenshot run cannot press a key, so this is the only way to photograph the highlight
anywhere but row 1), `ODAI_FNV_VICTOR_NOANIM=1` freezes him at bind pose (the control that
makes a screenshot diff of his own pixels attributable — frozen is byte-identical across
frames, animated is not), and `ODAI_FNV_VICTOR_HOME=1` puts him back at his authored ACRE
position instead of beside the spawn. `odai_newvegas_probe --kf <path>` dumps one clip (and
names any node left undecoded, which is the difference between "a few finger joints are stiff"
and "both arms never move"); `--kfsweep <folderSubstring>` parses every `.kf` under a folder
and reports the failures. `--formid <hex>` answers "what IS 0x104f04" — the usual question
when a reference resolves to nothing.

**`ODAI_ARENA_POISON=<bytes>`** takes a FrameArena slice before any pass allocates. It is the
regression probe for a camera-UBO bug worth knowing by its symptom: the descriptor-buffer path
published the camera at ring offset **0** rather than at its real slice offset, which worked
only while the camera was the frame's first allocation. One earlier allocation (the skinned
pose) made every pass read a garbage view-projection, and that renders as **a single flat
colour with the UI still correct on top of it**. Flat frame + intact UI means the camera, not
the geometry. Any value here must now render identically to unset.

Runtime env vars: `ODAI_STRATEGY_MAP`, `ODAI_IMPORTED_SCENE` (view any cooked `.bin` with no strategy-map support compiled in), `ODAI_LOG_LEVEL`, `ODAI_PRESENT_MODE`, `ODAI_PERF_OVERLAY` (start every `GameApp` game with the CPU timing overlay up; **F3** toggles it at runtime), `ODAI_FNV_MODS` / `ODAI_FNV_TEX_SIZE` (see Asset mods above), and `ODAI_CITY_DEMO` / `ODAI_CITY_SEED` / `ODAI_CITY_STORM` / `ODAI_CITY_STORY` for citybuilder.

**AO resolution and sun shafts are the two biggest costs after the main pass.** Measured on
the LNL iGPU at a 2560x1440 swapchain, native render scale: frame 37.9 ms, of which the AO
estimator is 6.0 and sun shafts 3.0 by the GPU timer but **10.2 and 4.7 by toggling them**.
Two defaults changed as a result:

- `ODAI_AO_DOWNSCALE` (default **2**) runs the AO estimator at a quarter of the render
  extent rather than the half it used to (`m_aoExtent` was already half). Only the *raw*
  target shrinks; `ssao_blur.comp.slang` is now a **joint bilateral upsample** back to
  `m_aoExtent`, so no consumer knows. Taps snap to raw texel centres and are weighted by
  full-res normal/depth — sampling the raw at an arbitrary UV instead lets the hardware
  blend across a depth discontinuity *before* the bilateral weights can reject it, which
  haloes every silhouette at 2x. Measured: AO 6.04 → 1.66 ms, for a 0.4% mean brightness
  change (run-to-run noise with `ODAI_FNV_NOWANDER=1` is 0.05%). `=1` restores the old
  resolution. Read once at swapchain build, since it sizes the targets.
- **Sun shafts now default OFF** for the FNV viewer. `skyConfig4`'s density/falloff/scatter
  are near zero for this game, so the pass was invisible and cost 4.7 ms. `ODAI_FNV_SHAFTS=1`
  turns it back on — do that *first* when tuning those densities, and flip the default back
  the moment they are non-trivial.

Together: 37.9 → 28.5 ms (26.4 → 35.1 fps) at native 1440p with no visible change.

**F4 opens the renderer's own ImGui panels** in any `GameApp` game — frame stats,
shadows/AO, sun/sky/post (the whole exposure/bloom/grading chain), and the render
debug views below. Distinct from F3, which is this engine's CPU timing overlay.
`buildShadowDebugUi`/`buildSunDebugUi` were written but had **no call site** in the
frame path, and no `GameApp` game ever called `setDebugUiVisible`, so that entire
tuning surface existed and was unreachable.

### Render debug views

**`shadow` and `directratio` answer "why does this look flat", and the answer is usually
not the renderer.** `shadow` is the cascaded visibility term; `directratio` is
`direct / (direct + ambient)`, i.e. how much of a surface's lighting a shadow is even able
to remove. Measured on Goodsprings that ratio is **0.81** -- the sun/ambient balance is
fine, and a flat-looking frame is almost always the VIEW: the default spawn faces roughly
down-sun, where every cast shadow hides behind its caster. Swinging `ODAI_FNV_YAW` across
the sun takes the shadowed fraction of the frame from 3% to 51%. Check that before
touching shadow code -- the sun's azimuth is `90 + (hour/24)*360` and its elevation is
`cos(hour/24 * 2pi) * 75`, negative being above the horizon.

Both are computed mid-shading rather than in `debugViewColor()`, because their inputs do
not exist until the sun and ambient terms have been evaluated -- so they are exempted from
the early-out at the top of `main()`. A new view that forgets that exemption falls into
`debugViewColor()`, which does not know it, and renders as some other view instead. Also
worth knowing: `ODAI_SHADOW_DUMP=<path.pgm>` writes the atlas, and it is **16-bit** --
reading it as 8-bit collapses the depth range and makes a perfectly good atlas look empty.

`render::DebugView` (`renderer_types.h`) replaces the frame with one channel of
what the main pass shaded with: albedo, normal, **alpha**, **material flags**,
roughness, metallic, mip level, cascade index, texture ID, linear depth. Set from
the F4 panel, or `ODAI_FNV_DEBUGVIEW=<albedo|normal|alpha|flags|roughness|metallic|mip|cascade|texid|depth>`
— a `--screenshot` run cannot operate ImGui, so the env var is the only way to
photograph one from a script.

It rides in `CameraUniform::tonemapConfig2.y` (a documented spare channel) and
`tone_map.frag.slang` bypasses exposure/tonemap/grading whenever one is active,
so what the shader returns reaches the screen through nothing but the sRGB encode.
Coverage is `imported_static.frag.slang` — which for a Fallout scene is the
terrain, every static and every actor, but not sky or water.

Two are load-bearing for diagnosing transparency, and neither works alone:

- **Alpha** deliberately **bypasses the alpha-test discard**. Discarding first
  throws away exactly the texels the view exists to inspect.
- **Material flags** false-colours what the importer decided: red alphaTest,
  green alphaBlend, blue twoSided, yellow unlit, **dark grey = no flags at all**.
  This is the one that answers the question, because a low alpha value is not by
  itself a defect — Fallout uses diffuse alpha as a **specular mask** on opaque
  surfaces, so black alpha on an unflagged surface is correct and expected. The
  defect signature is unflagged **and** near-zero alpha over a large area; that
  renders as a solid slab of the RGB sitting under transparent texels, which for
  these textures is black. Measured across the whole Goodsprings flythrough that
  signature peaks at 0.01% of unflagged samples, i.e. it is not currently present.

`ODAI_CITY_SCREENSHOT=<path>` (plus `ODAI_CITY_SCREENSHOT_FRAMES`) renders N frames, writes a
PPM and quits — the same headless-verification hook `odai_game_newvegas --screenshot` has, and
the only way to check a citybuilder visual change from a script on Wayland. Note the capture
trips `VUID-vkCmdCopyImageToBuffer-srcImage-00186` (the swapchain image lacks
`TRANSFER_SRC_BIT`); the images come out correct on this driver, but that is luck, not
contract.

**Shaders** compile automatically when `slangc` is on PATH; if it isn't, shader targets are skipped rather than failing the configure. Outputs are `.slang.spv` next to the source. Manual compile:
```bash
slangc -target spirv -entry main -stage fragment -matrix-layout-column-major \
  -I src/render/shaders src/render/shaders/ui.frag.slang -o src/render/shaders/ui.frag.slang.spv
```
Use `add_slang_shader_variant(..., -DODAI_RT_SHADOWS=1)` for define-based shader variants.

## Architecture

### Module boundaries

```
core/     — time, logging (VOX_LOGE/W/I/D/T macros), input state, job system, grid utils,
            shared containers/primitives: hash.h (spatial + coordinate hashes), lcg.h
            (the project's one deterministic RNG), ring_buffer.h, frame_profiler.h
            (Stopwatch/ScopedTimerMs/TimingChannel — CPU timing primitives)
math/     — header-only vector/matrix/quaternion + noise + geometry.h (Aabb3f, Ray,
            ray-triangle/ray-AABB intersection)
world/    — terrain, chunk grids, voxels, meshing/scheduling, clipmap, grass scatter
            (ChunkMeshScheduler::stats() reports wasted-work counters: meshes built
             then discarded because the chunk was edited or evicted mid-flight)
import/   — ImportedScene (de)serialization, DDS, GPU scene upload + import/fnv/ (Fallout: NV BSA/ESM/NIF)
game/     — Civ-style 4X model: hex strategy map, economy, advisors, religion, great people, units, AI
sim/      — header-only factory simulation (pipes, belts, tracks, network graph)
procgen/  — building/civic generators, CSG, primitives, props, city terrain
content/  — JSON content database + mod discovery/load order (mods/base)
script/   — Lua (sol2) scripting: sandboxed ScriptHost implementing game::IModHost
dialogue/ — branching dialogue graph: types, runtime, save-state serialization
anim/     — skeleton/bone hierarchy, animation clips, samplers (feeds the GPU skinning pass)
audio/    — PIMPL audio facade over miniaudio, with a silent null backend fallback
ui/       — Vulkan-free UI framework: draw list, font, rich text, widget tree, theming (see below)
render/   — public Renderer facade + everything Vulkan (only place that includes Vulkan headers)
engine/   — GameApp lifecycle base + PluginRegistry; what src/games/* build on (docs/GAME_API.md)
games/    — self-contained games on GameApp: citybuilder, minesweeper, snake, stellaris, swtor, voxelcraft
app/      — lifecycle, input routing, per-frame coordination for the older Civ-style `odai` app
tools/    — offline generators, headless sim harnesses, asset bundlers, UI editor + demos
tests/    — correctness tests; no Vulkan in test executables
```

**Hard rule:** only `src/render/` may include Vulkan headers. No Vulkan types cross this boundary — never let a `Vk*` type reach `src/games/`.

**Two app lineages, no shared code:** `odai::app::App` (`src/app/`) is the older, larger Civ-style application. `odai::engine::GameApp` (`src/engine/game_app.h`) is the thin GLFW + Renderer + UiContext loop that every `src/games/*` game subclasses. New games use `GameApp`; `src/app/app.cc` is not a template for anything.

### Data flow

World state never flows back from the renderer. The direction is:

```
World/Game data  →  Meshing / ImportedScene  →  Renderer::upload*()
                                                       ↓
                                              FrameArena (per-frame GPU upload)
                                                       ↓
                                 Passes (Shadow → Prepass → Skinning → Main → SSAO → Post → UI)
                                                       ↓
                                              Swapchain present
```

The seam between `src/ui/` and the renderer is `Renderer::setUiDrawData(const ui::UiDrawData&)` + `setUiFontAtlas(...)`. All UI logic and geometry assembly happens Vulkan-free in `src/ui/`; `UiRenderer` only streams the resulting vertex/index data to the GPU.

### Renderer internals

- **`render/renderer.h`** — narrow public facade. Call `upload*` to push world data, `setUiDrawData` for UI, `renderFrame` to record and submit.
- **`render/backend/vulkan/renderer_backend.h`** — the actual Vulkan state machine. Owns instance, device, swapchain, command pools, pipelines, descriptor sets, and `FrameArena`. All per-pass recording happens in files named `frame_*.cc`.
- **`render/frame_graph.{h,cc}`** + **`frame_graph_runtime.cc`** — declarative pass dependency graph. It resolves execution order and validates it; **barriers are hand-written per pass by design**, never inferred.

**THE FRAME LAYS DEPTH ONCE, AND THE PREPASS IS WHERE.** It used to lay it twice: the
normal-depth prepass rendered the visible set at *half* resolution into its own depth image
and threw it away (`storeOp DONT_CARE`), so the main pass — whose forward shader alpha-tests,
and therefore cannot get early-Z out of its own depth writes — rasterized the same set a
second time, depth-only, to lay depth it could then test against. Measured on Goodsprings:
2.7 ms for the prepass and 2.7 ms for the prewrite inside main's 8.5.

`useMergedDepthPrepass()` (`swapchain_targets.cc`) collapses them. The prepass now runs at
**full render extent** and writes the real `m_depthImages`, main does `loadOp = LOAD` with
`depthWriteEnable = FALSE`, and the prewrite is gone. Measured **18.2 → 15.8 ms** and
**24.8 → 21.2 ms** on two interleaved passes of the same walk. Four things about it are
load-bearing:

- **It is nearly free to make the prepass bigger** because it is vertex-bound, not
  fill-bound — it measured the same at native and at 44% of the pixels. That is the fact the
  whole merge rests on.
- **`m_normalDepthExtent` follows the prepass**, so normal-depth is now full resolution
  (a render pass cannot have a colour attachment smaller than its render area). Every
  consumer samples it by normalized UV, so nothing downstream changed — but the AO estimator
  now gets a finer input, which is why a before/after diff is a low-amplitude wash over lit
  surfaces rather than byte-identical.
- **`ao.depth` is no longer created**, and `m_colorSampleCount != 1` keeps the old
  two-rasterization structure: the normal-depth pipelines are `VK_SAMPLE_COUNT_1_BIT` and
  cannot write a multisampled depth buffer. `ODAI_MERGED_PREPASS=0` forces the old path for
  A/B; both are valid Vulkan.
- **The debug views cannot A/B this.** `imported_static.frag` returns `debugViewColor` *before*
  its alpha-test discard while the prepass still applies one, so an albedo/normal/depth view
  shows cutout vegetation as solid billboard quads under the old path and correctly cut out
  under the merged one. That is the instrument, not the geometry. Compare lit frames.

Related: at one sample there is no resolve, and the main pass now renders straight into
`hdrResolve`. It used to point at a 1-sample `m_msaaColorImages` and set
`resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT` regardless — a full-resolution copy into an
identical image every frame, and a spec violation
(`VUID-VkRenderingAttachmentInfo-imageView-06861`). Worth only ~0.2–0.7 ms on this iGPU
(lossless colour compression makes the copy cheap) but it returns three swapchain images'
worth of R16G16B16A16 VRAM.

**`ImportedMeshVertex` is 48 bytes, and vertex WIDTH is not what these passes cost.**
Position and UV stay full float on purpose — position feeds the clip transform the merged
prepass and main must agree on bit-exactly, and UV drives an alpha test whose cutout both
must reproduce. Normal is octahedral snorm16x2, colour is **sRGB-encoded** unorm8x4 (linear
quantization spends all 256 steps on highlights and bands the darks; all 256 sRGB source
bytes round-trip exactly), and the four terrain-layer slots are u16 read as one
`R16G16B16A16_UINT` fetch. Encoders are in `src/import/imported_scene.h` — deliberately
outside `src/render/` so a Vulkan-free test can reach them — mirrored by
`shaders/imported_vertex_pack.slang` and pinned by `testImportedVertexPacking`.

Measure before extending this. `ODAI_FAT_SHADOW_STREAM=1` puts the shadow pass on the main
stream instead of its 28-byte compact one, over identical geometry with both pipelines
already built, which makes it the cheapest vertex-width experiment in the tree. Cutting
72 → 28 moved it 0.36/0.13 ms; cutting 72 → 48 moved the same gap only 0.36 → 0.32 and
0.13 → 0.11, i.e. **the 24 bytes are worth 0.02–0.04 ms.** These passes are bound by
geometry submission and primitive throughput, not attribute fetch. The 48-byte format is a
**memory** win (a third of geometry vertex storage) that happens to cost nothing.

Two traps this exposed. `skinning.comp.slang` writes this struct through a StructuredBuffer,
so its `SkinnedVertexOut` must match to the last byte — verify with
`spirv-dis src/render/shaders/skinning.comp.slang.spv | grep OpMemberDecorate` and check
ArrayStride is 48; sub-32-bit members there would need `VK_KHR_8bit_storage`, which is why
the narrowed fields are `uint` and packed by hand. And the octahedral worst case is **0.034
degrees at the fold diagonals just below the equator**, not at the poles — a random-sample
estimate understates it tenfold, which is how the test's first threshold came out wrong.

GPU timestamps now cover the **depth prewrite, the skinned-velocity pass and TAA/upscale**
(`kGpuTimestampQuery*`). Before that, ~2 ms of every frame ran between main's end timestamp
and post's start with no query over it, which is where the temporal chain was hiding.
- **`descriptors.cc`** — bindless texture table, plus a classic-descriptor-set fallback path so headless CI renders on lavapipe.
- **`docs/FrameArena.md`** — per-frame transient GPU memory (host-visible upload arena + device-local scratch arena; reclaimed after the timeline fence).

Swapchain format is `B8G8R8A8_UNORM` (driver presents raw bytes, display interprets as sRGB). The UI fragment shader works in linear space and applies a manual `linearToSrgb` encode before output — matching the `pow(1/2.2)` the tonemapper applies for the 3-D pass. Vertex colors authored as sRGB hex are decoded to linear on entry so hex values are WYSIWYG. Color textures use `VK_FORMAT_R8G8B8A8_SRGB` image views so the sampler returns linear values.

### UI framework (`src/ui/`)

Fully headless and Vulkan-free: a retained widget tree over an immediate draw list. Packaged for vendoring into other engines (`odai_ui` + `odai_ui_vulkan` install/export targets; see `examples/vulkan_ui_integration/` for a from-scratch embedding and `docs/UI_LIBRARY.md` for the full guide). ImGui exists only as a separate dev/debug overlay with no code in common.

- `ui_draw_list` emits `UiDrawData` (quads, 9-slice, glyph-alpha quads, per-command texture + clip rect)
- `font` bakes an R8 atlas via `stb_truetype` + `stb_rect_pack`; `icon_atlas`, `resource_style`, `tooltip`, `ui_cursor`, and `animation.h` (easings, `Sequence`, `RectTween`) round out the primitives
- `rich_text` parses `<b>/<i>/<color=#rrggbb>/<br>` markup, wraps, and aligns; `cached_rich_text` memoizes layout
- `widgets/` — ~35 widgets from `Button`/`Label`/`Panel` up through domain panels (`research_panel`, `minimap_panel`, `advisors_panel`, `dialogue_panel`, charts). `signal.h` is the callback/slot registry; `UiContext` owns the root and dispatches input
- `theme/ui_theme` — JSON-driven color/font/size tokens with hot reload (`odai_theme_viewer`)
- `document/` — `.ui.json` documents instantiate live widget trees with `{binding}` expressions and hot reload (`assets/ui/docs/city_panel.ui.json`)
- `kits/` — prebuilt genre panel sets (`strategy_4x_kit`, `city_builder_kit`, `colony_sim_kit`)
- `vector/` — nanosvg-backed SVG import, tessellation, and an `.odaivec` geometry cache
- `render/backend/vulkan/ui_renderer.cc` is the only UI file touching Vulkan: alpha-blend pipeline, per-texture descriptor sets, per-frame geometry streaming

### Content, mods, and scripting

Three existing seams — prefer composing through one of them over inventing a fourth:

- **`content/`** — `ContentDatabase` built from `mods/base/data/*.json` (buildables, techs, units, leaders, religions, great people, balance, civpedia). `mod_loader` resolves the base game today; user-mod discovery and override order are planned, not built.
- **`game/mod_host.h`** — `IModHost`, the simulation's gameplay-event hook interface. `script::ScriptHost` (`src/script/`) is the Lua implementation: a sandboxed sol2 state dispatching to `Events.on` / `Effects.register` callbacks from `mods/*/scripts/*.lua`. `script_engine.h` is Lua-free (PIMPL) so callers don't pull in Lua; script errors are recorded, never thrown into the sim.
- **`engine/plugin.h`** — see below.

### Engine plugins (proposed pattern, still unproven — read before citing this as settled)

`GameApp` owns a `PluginRegistry`. The intent is an extension point: a game registers `IEnginePlugin` implementations from its own `onInit()` to compose in optional behavior (a debug overlay, a stats collector, a `mods/`-driven hook) without `GameApp` or the base game needing to know about it. `docs/GAME_API.md` §11 has the lifecycle contract and the one ordering caveat (`onRender` must be invoked manually from the game's own `onRender()`, after `beginFrameDraw()` and before `submitFrame()`; `onTick` is fanned out automatically).

**`IEnginePlugin` still has zero implementations anywhere in this codebase** and has never been built against a real Vulkan+GLFW target. Do not treat it as a settled architectural decision or cite it as having superseded anything — that requires a real plugin, in a real game, on a real build, first. Treat the interface as a draft: verify the shape still fits before building on it, and expect `GameApp`'s `protected` members (`m_renderer`, `m_uiContext`, …) may need new accessors before a plugin holding only `GameApp&` can do anything useful.

No dynamic loading (`dlopen`/DLL) is involved anywhere — a "plugin" here is a statically linked C++ class registered at startup, the same shape as the `IModHost` and `mod_loader` seams above.

### Hex strategy map

`src/game/` is pure CPU (no Vulkan, no imgui). Pointy-top hex grid in odd-r offset coordinates. `strategy_map_mesh.cc` converts a `StrategyMap` into an `ImportedScene` using the packed vertex-color render path (`textureIndex=0xFFFFFFFF` → per-vertex color), so no new renderer code is required to view a map. `strategy_map_io.cc` is versioned binary `.smap` serialization.

### Shader system

Shaders use **Slang** (`.slang` → `.slang.spv` SPIR-V). Shared includes live in `src/render/shaders/`. One of them has a contract worth stating up front:

- `pbr.slang` — metallic-roughness BRDF (GGX/Smith/Schlick + analytic env BRDF). Specular only: it layers onto the existing baked-GI diffuse chain rather than replacing it. Materials are opt-in per vertex through the packed flag bits defined in `src/import/imported_scene.h` — geometry without the PBR bit shades exactly as it did before

Ray-traced shadow/reflection variants compile the same `.slang` source with `-DODAI_RT_SHADOWS=1` or `-DODAI_RT_REFLECTIONS=1`.

### Naming conventions

- Namespaces: `odai::app`, `odai::render`, `odai::world`, `odai::ui`, `odai::anim`, `odai::content`, etc.
- Types: PascalCase — `StrategyMap`, `UiDrawList`, `RendererBackend`
- Functions: camelCase — `buildStrategyMapScene`, `setUiDrawData`
- Private members: `m_camelCase` prefix
- Module-scoped constants: `k` prefix — `kMaxFramesInFlight`, `kUiNoTexture`
- Source files: `.cc` (not `.cpp`)

## Docs index

| Doc | What's in it |
|---|---|
| `docs/GAME_API.md` | **Read first when adding a game under `src/games/`** — `GameApp` contract, renderer/UI API surface, the CMake block to copy |
| `docs/UI_LIBRARY.md` | UI architecture, widget catalog, theming, container reflow contract, integration walkthrough |
| `docs/FNV_MODS.md` | **Read first to launch the FNV viewer with mods** — verified install + launch commands, env-var reference, and the silent traps |
| `docs/FrameArena.md` | Per-frame GPU memory model |
| `docs/ROADMAP.md` | Feature-by-feature status against the four touchstones, with explicit out-of-scope calls |
| `docs/EARLY_ACCESS_PLAN.md`, `docs/devlog.md` | Planning and history |
| `docs/bloom.md`, `voxel_gi.md`, `shadow_occluder.md`, `spatial_partitioning_plan.md`, `stylized_low_poly.md`, `minecraft_clone_modernization.md` | Per-feature rendering/design notes |

The `.claude/skills/new-game` skill scaffolds a new `src/games/<name>/` target; `.claude/skills/vulkan-docs` checks current Vulkan practice before touching `render/backend/vulkan/`.

## Review agents (`.claude/agents/`)

Nine subagents, each carrying a distinct lens grounded in this codebase rather than generic advice. They are review/advice specialists — pick the one whose lens matches the question rather than asking the same question of several. Most can edit; `jonas` is read-only by design. Each agent's own `description:` states its lens and when to reach for it.

## Local Paths

| Resource | Path |
|---|---|
| Morrowind Data Files (Windows) | `C:\GOG Games\Morrowind\Data Files` |
| Morrowind Data Files (WSL) | `/mnt/c/GOG Games/Morrowind/Data Files` |
| Fallout: New Vegas Data Files (Linux/Steam) | `~/.steam/steam/steamapps/common/Fallout New Vegas/Data` — `odai_game_newvegas` finds this (and the usual GOG/WSL/Windows locations) on its own; the cooker and probe take it as their first argument |
| OpenMW source (Windows) | `C:\Users\rfdic\OneDrive\Documents\GitHub\openmw` |
| Build dir (Windows) | `cmake-build-release` |
| Build dir (Linux) | `cmake-build-linux` |

### Capturing a recording

`--capture-video <out.mp4> [fps] [seconds]` encodes as it renders, piping raw rgb24 into an
ffmpeg child process. Prefer it over `--capture-seq`: the swapchain follows the window and
routinely opens at 4K here, where one PPM is 24 MB — three 60 fps legs as stills is over
100 GB and the earlier 30 fps attempt exhausted the disk quota mid-run, which then took the
shell down with it. The encoder is **openh264**, which ships with every ffmpeg build here and needs no
licensing dance -- libx264 is absent from a lot of distribution ffmpegs (Fedora's included).
It is bitrate-controlled and silently ignores `-crf`, so pass `-b:v`; a CRF here is not an
error, it is a soft-looking capture with nothing to point at. `$ODAI_CAPTURE_ENCODER`
overrides.

**READING A HOST_COHERENT MAPPING BYTE BY BYTE COST 40x THE FRAME IT WAS CAPTURING.** The
readback picked plain `HOST_VISIBLE | HOST_COHERENT` memory, which is typically
write-combined — fine to write, dreadful to read — and the BGRA→RGB swizzle then pulled three
strided bytes per pixel straight out of it. Measured at 2560x1440: **1.13 s per frame**, against
a scene that renders in tens of milliseconds. It is invisible to every instrument you would
reach for: not GPU time, not I/O, just userspace CPU, which is why it read as the encoder's
fault. The stills path measured *identically* (1.13 s/frame), which is what finally located it.
Two fixes together took it to **2.4 s for 60 frames, a 28x speedup**: prefer `HOST_CACHED`
memory when the driver offers it, and `memcpy` the mapping into a staging vector once before
swizzling from the copy. The readback buffer, command pool and command buffer are also built
once and reused now rather than per frame.

**A frame-count warm-up is not a streaming warm-up.** `m_captureWarmupFrames` (60) is there for
auto-exposure and TAA, and it silently doubled as "wait for cells to arrive" — which held only
because capture was slow. The moment it got 28x faster those same 60 frames went from over a
minute of wall time to about a second, and captures began opening on half-built towns.
`captureWarmupComplete()` now requires the frame count **and** `CellStreamer::isStreamingIdle()`,
with a hard ceiling so a worldspace that never settles cannot stall a capture forever.

`assets/tours/` holds the authored paths for the three-game showcase (Megaton, Anvil);
Goodsprings uses the built-in tour. See `--tour-file` above.
