---
name: content-tools
description: Run this project's offline content-generation tools and headless sim harnesses (ODAI_BUILD_TOOLS=ON, all pure CPU) — odai_strategy_map_gen, odai_civ_sim, odai_stellaris_sim, odai_dds_bundler, odai_svg_bundler, odai_theme_viewer, odai_newvegas_cooker, odai_fnv_texture_pack. Use when asked to generate a strategy map, cook a Fallout: New Vegas cell or worldspace, prepare/install a Fallout NV texture mod, bundle PNGs to BC3 .dds or SVGs to .odaivec, dump theme tokens, or run a headless 4X playtest — and especially for CPU regression testing of the turn loop via --sweep, which is the harness for "did the turn loop get slower" as well as "did the game get less fun".
---

# Content generation tools

All of these build under `ODAI_BUILD_TOOLS=ON` (the default) and are pure CPU — no Vulkan,
no GLFW. They run from the build directory.

```powershell
odai_strategy_map_gen [smap] [bin] [w] [h] [seed]   # writes strategy_map.smap + strategy_map_scene.bin
odai_civ_sim        [turns] [seed] [empires] [--quiet|--sweep N]   # headless 4X playtest metrics + CPU benchmark
odai_stellaris_sim  [turns] [seed] [empires] [--quiet|--sweep N]   # headless space-4X playtest metrics + CPU benchmark
odai_dds_bundler    <file.png>... | --dir <dir>     # offline PNG -> BC3 .dds sidecars
odai_svg_bundler    <file.svg>... | --dir <dir> [--sizes 16,32,64]  # SVG -> .odaivec cache
odai_theme_viewer                                   # terminal theme-token dump with hot reload
odai_newvegas_cooker <DataFiles> <Plugin.esm> <out.bin> --cell <EditorID>
odai_newvegas_cooker <DataFiles> <Plugin.esm> <out.bin> --worldspace <EditorID> <x0> <z0> <x1> <z1>
odai_fnv_texture_pack --in <dir> --out <dir> [--max-size 1024]  # prep a texture mod for --mod
```

## `odai_fnv_texture_pack` — prepare a downloaded texture mod

Extract the pack, point this at it, and hand the output to `odai_game_newvegas --mod <dir>`.
It does two things, both of which the runtime would otherwise pay for on every cell load:

- **Mip-drops every `.dds` to `--max-size`.** `CellSceneBuilder` drops the same levels at
  load time anyway, so keeping them on disk buys nothing and costs the read. These packs
  ship full mip chains, so the drop is a byte skip — the surviving levels are the pack's
  own, bit for bit. Nothing is recompressed.
- **Lowercases every path**, so the tree matches what the NIFs ask for.

Non-`.dds` files (packs often carry a few meshes) and any `.dds` that will not parse are
copied through unchanged — losing a texture silently is worse than carrying a dead one.

Match `--max-size` to the `ODAI_FNV_TEX_SIZE` you intend to run; a pack cut to 512 cannot
be turned back up later. Measured on NMC's SMALL pack: 5431 files, 2.0 GB → 1.5 GB at a
1024 ceiling, 182 files mip-dropped, ~1 s.

**`ODAI_FNV_TEX_SIZE` is what makes a pack visible at all** — the ceiling defaults to 512,
so without raising it the pack is mip-dropped back to vanilla resolution and nothing
changes on screen. See CLAUDE.md's Asset mods section for the cache-key consequence.

## `--sweep N` is the CPU regression harness

Both sim tools replay N deterministic seeded matches headlessly and report wall clock alongside
the balance metrics — mean/median/p95 per match, µs per turn, and turns/sec
(`src/tools/sim_bench.h`). Balance output is unchanged and stays bit-identical across build
types, so one command answers both "did the game get less fun" and "did the turn loop get
slower".

Measured with `--sweep 8`: **28203 turns/sec** at `-O3 -DNDEBUG` vs **4879** at `-O0` (5.8x).
The report prints a warning when `NDEBUG` is absent for exactly that reason — **build optimized
before believing any number it prints.** See the optimized-build section of `CLAUDE.md` for the
presets.

To view a generated map, point the app at it:

```powershell
$env:ODAI_STRATEGY_MAP = "strategy_map.smap"
cmake-build-release\odai.exe
```

`ODAI_IMPORTED_SCENE` views any cooked `.bin` with no strategy-map support compiled in — useful
for checking `odai_newvegas_cooker` output.
