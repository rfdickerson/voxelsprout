# Running the Fallout: New Vegas viewer with mods

How to install real Nexus mods and launch `odai_game_newvegas` with them. The *why* behind
each mechanism — formID remapping, WTHR colour layout, the cloud projection — lives in
CLAUDE.md's Fallout: New Vegas section; this file is the operational guide.

Every command here has been run end to end on Linux against a Steam install.

---

## 1. Prerequisites

A build with the app targets on:

```bash
cmake --preset linux-vcpkg-relwithdebinfo && cmake --build --preset linux-vcpkg-relwithdebinfo
```

**Run from the build directory.** Shader paths are resolved relative to the working
directory, so launching from the repo root fails with a wall of
`missing required runtime asset: ... .slang.spv`:

```bash
cd cmake-build-relwithdebinfo   # or cmake-build-app, whatever you configured
./odai_game_newvegas --help
```

You also need the game installed. On Linux/Steam the data directory is
`~/.steam/steam/steamapps/common/Fallout New Vegas/Data`; the viewer finds that (and the
usual GOG/WSL/Windows locations) on its own if you omit `--stream`.

---

## 2. The two kinds of mod

They install differently, and confusing them is the most common way to get a mod that
silently does nothing.

| | What it contains | How it loads |
|---|---|---|
| **Asset mod** | Loose `textures\`/`meshes\` trees, and/or a `.bsa` | `--mod <dir>` |
| **Plugin mod** | An `.esp`/`.esm` of records | `--plugin-add <Name.esp>` |
| **ENB preset** | `.fx`/`.ini` for the injected ENB runtime | not loadable — see §4b |

A mod can be both. Nevada Skies is: `NevadaSkies.esp` (the weather records) plus
`NevadaSkies.bsa` (330 MB of sky art). It needs **both** flags.

A plugin is resolved by name against the `--mod` directories first, then the `--stream`
directory. So a mod's `.esp` lives in its own mod directory beside the `.bsa` it ships with,
and **nothing is copied into the game install**. Masters still resolve from wherever they
are, so a mod directory holding nothing but the `.esp` works.

---

## 3. Installing a texture pack

Normalize the pack first. `odai_fnv_texture_pack` mip-drops it to the ceiling you intend to
run and lowercases every path:

```bash
7z x -o/tmp/nmc "NMCs Textures NV SMALL Pack SINGLE FILE FOR NMM-43135-1-0.7z"

./odai_fnv_texture_pack \
  --in /tmp/nmc/NMCsTexPack_SMALL \
  --out ~/.local/share/odai/fnv-mods/nmc \
  --max-size 1024
```

Measured on NMC's SMALL pack: 5431 files, 2.0 GB → 1.5 GB, 182 files mip-dropped, ~1 second.
Because these packs ship full mip chains the drop is a byte skip — the surviving levels are
the pack's own, bit for bit. Nothing is recompressed.

Match `--max-size` to the `ODAI_FNV_TEX_SIZE` you intend to run. **A pack cut to 512 cannot
be turned back up later.**

---

## 4. Installing a plugin mod (Nevada Skies)

Everything the mod ships goes into one directory — plugin, archive and loose files together,
exactly as the download lays them out:

```bash
7z x -o/tmp/ns "Nevada Skies 2281 Rework-35998-Final-Rework-1622851359.7z"

mkdir -p ~/.local/share/odai/fnv-mods/nevadaskies
cp /tmp/ns/NevadaSkies.esp ~/.local/share/odai/fnv-mods/nevadaskies/
cp /tmp/ns/NevadaSkies.bsa ~/.local/share/odai/fnv-mods/nevadaskies/
cp -r /tmp/ns/meshes       ~/.local/share/odai/fnv-mods/nevadaskies/
```

That is the whole install. `--stream` keeps pointing at the real game directory, which is
never written to — no copying into the game, and no staging directory mirroring it.

---

## 4b. Enhanced Shaders (and why it installs differently)

Enhanced Shaders is an **ENB preset**: `.fx` shaders and `.ini` parameters for the ENB
runtime, a DLL injected into the retail `FalloutNV.exe` that wraps its D3D9 calls. There is
no ENB runtime here and nothing to inject into, so it cannot be *loaded* the way the other
two are. There is nothing to put in a `--mod` directory.

What it is good for is its **numbers**. Its tone curve and the values it was tuned to — against
these same weather records — are ported into the post pass and selected with:

```bash
ODAI_FNV_TONEMAP=enb
```

That switches from the engine's ACES fit to ENB's extended Reinhard,
`x(1 + x/L)/(x + C)`, with Enhanced Shaders' own `enbeffect.fx.ini` values: curve 8.0 day /
10.0 night, contrast 1.35 / 1.25, saturation 1.25 / 0.9, overbright dampening 75 / 50.
Contrast is applied to colour *magnitude* and saturation to *hue direction*, which is the
structural trick that keeps it from desaturating as it adds contrast.

It is a different look, not a strict upgrade — try both. Everything outside this game keeps
the ACES fit regardless.

---

## 5. Launching

```bash
cd cmake-build-relwithdebinfo

ODAI_FNV_TEX_SIZE=1024 \
ODAI_FNV_MODS="$HOME/.local/share/odai/fnv-mods/nmc:$HOME/.local/share/odai/fnv-mods/nevadaskies" \
./odai_game_newvegas \
  --stream "$HOME/.steam/steam/steamapps/common/Fallout New Vegas/Data" \
  --plugin-add NevadaSkies.esp \
  --weather WEAVarHRainHeavy
```

`--mod <dir>` works too and is equivalent to an `ODAI_FNV_MODS` entry; both are repeatable
and **later wins**, matching a mod manager's load order.

### Confirming it actually loaded

The startup log answers this directly:

```
[streamer] mods: 2 directories, 5431 loose files, 1 archives override the base game
[newvegas] load order: FalloutNV.esm -> DeadMoney.esm -> HonestHearts.esm ->
                       OldWorldBlues.esm -> LonesomeRoad.esm -> NevadaSkies.esp
[newvegas] weather: 473 WTHR, 58 CLMT
[newvegas] weather forced to WEAVarHRainHeavy (0x509d339)
[newvegas] cloud layers: 1 of 4 in use
[newvegas] rain: sound\fx\weather\amb_weather_rain_heavy_lp.wav
[newvegas] wind: sound\fx\weather\amb_windheavy_lp.wav
[newvegas] radio: MUS_Lone_Star
```

Note the masters resolve on their own — asking for `NevadaSkies.esp` pulled in the base game
and all four DLC, in the right order. **473 weathers** means the mod's 387 merged over the
base game's 86.

`0 archives` next to a mod that ships only a `.bsa` means the mod directory is wrong. A
`rain:`/`wind:` path under `sound\fx\weather\` means the mod's own audio won; anything
else means it did not resolve and the fallback list was walked.

---

## 6. A moody rainy scene

```bash
ODAI_FNV_TEX_SIZE=1024 ODAI_FNV_HOUR=16.0 ODAI_FNV_TONEMAP=enb \
ODAI_FNV_MODS="$HOME/.local/share/odai/fnv-mods/nmc:$HOME/.local/share/odai/fnv-mods/nevadaskies" \
./odai_game_newvegas \
  --stream "$HOME/.steam/steam/steamapps/common/Fallout New Vegas/Data" \
  --plugin-add NevadaSkies.esp \
  --weather WEAVarHRainHeavy
```

**Rain and wind come from the loaded mods first.** Nevada Skies ships its own weather set
inside its BSA (`amb_weather_rain_light/medium/heavy_lp.wav`, `amb_windlight/windheavy_lp.wav`,
thunder), and because a mod archive outranks the base game those are what play. Intensity is
picked from the weather's editor ID — a heuristic, because WTHR records carry no rain-intensity
field, only a "rainy" classification bit. Rain plays *only* when that bit is set, so
`--weather WEAVarNV01` is silent by design.

**Music is the radio, not the score.** Fallout ships two separate loose music sets:
orchestral exploration beds under `Data/Music`, and the 48 licensed radio songs under
`Data/Sound/songs/radionv` — Big Iron, Blue Moon, Johnny Guitar, Jingle Jangle Jingle. The
radio station is what plays, and a track is picked from it at random on each launch.
`ODAI_FNV_MUSIC` pins one, by song name or full path:

```bash
ODAI_FNV_MUSIC=Big_Iron        # name, case-insensitive, MUS_ prefix and .mp3 optional
ODAI_FNV_MUSIC=Johnny_Guitar
ODAI_FNV_MUSIC=/path/to/anything.mp3
```

Nevada Skies weather editor IDs are `WEA01v…`–`WEA22v…`, `WEAVar…` (e.g. `WEAVarNV01`,
`WEAVarHRain`, `WEAVarHRainHeavy`), and a set of named ones — `NSFoggy1`, `NSDeepPurple1`,
`NSNuclearMemories4`. Omit `--weather` entirely to let the worldspace's climate choose.

---

## 7. Environment variables

| Variable | Default | What it does |
|---|---|---|
| `ODAI_FNV_MODS` | — | `:`-separated asset mod roots, later wins |
| `ODAI_FNV_PLUGINS` | — | `,`-separated extra plugins, in load order |
| `ODAI_FNV_TEX_SIZE` | `512` | Texture mip-drop ceiling in px. **A pack is invisible without raising this** |
| `ODAI_FNV_WEATHER` | — | Weather editor ID, same as `--weather` |
| `ODAI_FNV_HOUR` | `9.5` | Time of day, 0–24 |
| `ODAI_FNV_MUSIC` | random radio song | Song name (`Big_Iron`) or full path to an `.mp3` |
| `ODAI_FNV_NOCLOUDS` | off | Disable cloud layers — separates "dark sky" from "total cloud cover" |
| `ODAI_FNV_SKY_GAIN` / `_CONTRAST` / `_SATURATION` | `1.6` / `0.6` / `1.15` | Sky colour transfer, see below |
| `ODAI_FNV_TONEMAP` | `aces` | `enb` selects Enhanced Shaders' tone curve and its tuned values |
| `ODAI_RENDER_SCALE` | `1.0` | Internal 3D resolution. `0.6` costs ~3.5 ms less on an iGPU |
| `ODAI_WINDOW_SIZE` | `1600x900` | Logical window size (physical = this × desktop scale) |
| `ODAI_FNV_CACHE_DIR` | `~/.cache/odai/fnv` | Built-cell cache |

---

## 8. Traps

Every one of these fails **silently** — the game runs and looks almost right.

**A texture pack does nothing without `ODAI_FNV_TEX_SIZE`.** The ceiling defaults to 512, so
a 1K/2K pack is mip-dropped straight back to vanilla resolution: full I/O cost, zero visual
change.

**Cached cells bake their textures in.** The mod set and the ceiling are folded into the
cache directory name, so installing a pack lands on a *fresh* cache rather than serving
vanilla art forever. Nothing is deleted — the old directory just stops being consulted and
keeps costing disk. Prune `~/.cache/odai/fnv/` when it grows; each distinct
(plugin set, mod set, ceiling) combination gets its own few hundred MB.

**A plugin in a `--mod` directory is not loaded.** Plugins resolve against `--stream` only.
The failure is `plugin not found: NevadaSkies.esp`, which at least names itself.

**Music is loose, not archived.** The radio songs live in `Data/Sound/songs/radionv` and the
score in `Data/Music`; neither is in a BSA. Point `--stream` at anything that is not the real
game directory and the radio silently does not play while rain still does — the log says
`no radio songs found under ...`.

**Sky colours are display-referred.** Fallout authored them as final sRGB for a renderer that
drew them directly; this one is HDR with a tone curve and auto exposure. The decode applies
`pow(magnitude, contrast) * gain` with saturation on the hue direction to compensate. If a
weather looks washed out or too dark, those three knobs are the ones to turn — this is a
fudge, not physics.

---

## 9. Known gaps

- `odai_newvegas_cooker` has no `--mod`; only the streaming path honours mods.
- Only **weather** records are read from extra plugins. Cell contents still come from the
  main plugin alone, so a mod's placed objects do not appear.
- No rain **particles** — a rainy scene is sky, fog and sound.
- WTHR's Ambient/Sunlight channels are unused, so terrain lighting does not respond to the
  weather; the ground stays desert-bright under an overcast sky.
