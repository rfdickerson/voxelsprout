# odai

`odai` is an open-source Vulkan runtime for exploring Bethesda Game Studios worlds from
Morrowind, Oblivion, Fallout 3, Fallout: New Vegas, and Skyrim.

The engine streams original game archives and plugins into one retained `ImportedScene`
rendering path. It supports terrain, statics, water, fire, authored weather and clouds,
local lights, GPU-skinned actors, dialogue records, temporal rendering, and optional
ray-traced Bethesda scene variants. No original game assets are distributed here.

## Build

Dependencies are supplied by the vcpkg manifest.

```bash
cmake --preset linux-vcpkg
cmake --build --preset linux-vcpkg -j
ctest --test-dir build-linux --output-on-failure
```

The useful switches are `ODAI_BUILD_RUNTIME`, `ODAI_BUILD_TOOLS`, `BUILD_TESTING`,
`ODAI_ENABLE_CCACHE`, `ODAI_ENABLE_LTO`, `ODAI_ENABLE_NATIVE_ARCH`, and
`ODAI_ENABLE_XESS`.

## Run

```bash
./build-linux/odai --help
./build-linux/odai --stream "/path/to/Oblivion/Data" \
  --plugin Oblivion.esm --worldspace Tamriel
```

The same command accepts `Morrowind.esm`, `Fallout3.esm`, `FalloutNV.esm`, and
`Skyrim.esm`. Authored camera tours live in `assets/tours/`.

For Skyrim Special Edition, `odai` resolves the active `plugins.txt` from the
native or Proton profile and recursively includes its masters. An explicit file
wins over discovery:

```bash
./build-linux/odai --stream "/path/to/Skyrim Special Edition/Data" \
  --plugin Skyrim.esm --load-order "/path/to/plugins.txt"
```

The official fallback preserves Skyrim, Update, installed DLC, and locally
present `Skyrim.ccc` order; it never enables arbitrary plugins by scanning the
Data directory. `ODAI_FNV_LOAD_ORDER` is the environment equivalent.

The Skyrim-first gameplay session has an explicit native-save entry point:

```bash
./build-linux/odai --scenario skyrim-bleak-falls \
  --stream "/path/to/Skyrim Special Edition/Data"
```

It starts at Riverwood, seeds the completed MQ101/Helgen prerequisite, then
replays MQ102's authored Riverwood startup stage after its retail VMAD is
attached, and uses checksummed ODAI saves (`F5`/`F9`, or
`--save-game`/`--load-game`). It does
not read or write Skyrim `.ess` files. Press `I` (controller Back) to open the
player inventory. Its Skyrim-style category and item columns use Left/Right
to change column and Up/Down to browse. Enter/A equips supported melee weapons,
uses immediate healing potions, or reads books. Escape/B closes the view.
Inventory shows saved item counts and pauses gameplay. Search defeated actors
with `E`, then Enter/A takes one item or R/X takes all. Item names, book text,
weapon damage, and supported healing effects come from installed records.
The Skyrim inventory reads Futura Condensed outlines from the installed
`Interface/fonts_en.swf` and previews the selected item's installed NIF model.
Drag the preview or use the right stick to orbit/tilt; Q/E also rotates it.
The preview uses orthographic projection and simple studio lighting; BC1/BC2/BC3
and RGBA8 diffuse textures are supported, with a shaded mesh fallback for other
formats. It does not reproduce retail enchantment effects or material shaders.
No game models, textures, or fonts are bundled with the repository.
Screenshot and video capture runs suppress traversal and gameplay saves.
Skyrim streaming retains textures up to 2048 pixels by default; set
`ODAI_FNV_TEX_SIZE=512` for the previous lower-memory ceiling. Higher-resolution
textures increase GPU memory use. Imported normal maps retain their authored
surface strength. The default Skyrim grade preserves dark values instead of
clipping them with extra global contrast.

The deterministic session, world registry,
VMAD/PEX readers, strict script diagnostics, and save lifecycle are implemented;
the Golden Claw/Dragonstone route is not yet release-gate complete. See
[`docs/SKYRIM_FIRST_RUNTIME.md`](docs/SKYRIM_FIRST_RUNTIME.md) for exact gate status.

Large existing MO2, OpenMW, and ODAI JSON setups can be loaded read-only as one
resolved content graph:

```bash
./build-linux/odai --profile "/path/to/MO2/profiles/Default" \
  --stream "/path/to/Skyrim Special Edition/Data" --worldspace Tamriel
./build-linux/odai --profile "$HOME/.config/openmw/openmw.cfg" \
  --worldspace Vvardenfell
```

Use `--mods-root` for a nonstandard MO2 layout, `--compat-report <json>` for an
atomic launch report, and `--reindex-content` after manually changing indexed
files. `--mod` and `--plugin-add` still append at highest priority. See
[`docs/MOD_PROFILES.md`](docs/MOD_PROFILES.md) for profile formats, precedence,
diagnostics, and the deliberately unsupported script-runtime boundary.

WASD and the mouse explore, `E` activates actors and real XTEL doors, `F`
toggles walking, and Escape opens the pause menu and discovered-location list.
Exterior cells stream continuously; doors fade between interiors and child
worldspaces such as WhiterunWorld. The runtime saves a native traversal state
every five seconds while grounded and resumes it on the next launch. Use
`--state <path>` to relocate that file or `--no-resume` for a fresh session;
explicit `--worldspace`, `--interior`, or `--spawn` selections take precedence.

Retained inspection and content commands:

```text
odai_bethesda_probe
odai_newvegas_cooker
odai_fnv_texture_pack
```

For Skyrim compatibility work, `odai_bethesda_probe <Data> --scriptcheck
<script.pex> --strict` reports decoded opcodes/calls, while `--quest-trace
<Plugin.esm> <QuestEditorID>` follows QUST → VMAD → attached PEX scripts without
redistributing installed data. `--skyrim-dialogue-trace <Plugin.esm>
<QuestEditorID>` reports localized DIAL/INFO, CTDA, links, and INFO fragment
metadata. `--scenario-check skyrim-bleak-falls` loads the same retail
QUST/DLBR/DIAL/INFO/VMAD/PEX closure as the runtime and runs fixture-assisted
Golden Claw alias/event and hand-in assertions without starting Vulkan. Its boss
fixture uses the exact installed ACHR identity plus authored location/XLRT data,
matches MQ103's forced-location/reference-type alias, kills that runtime actor
through the physical combat path, loots the authored Dragonstone, and proves
Farengar's fragment changes its player count from one to zero across save/reload.
This does not yet prove natural streamed boss residency. Its JSON
separates injected setup from verified behavior, lists unverified route segments,
and leaves `release_gate_passed` false until the continuous route exists. In the
scenario UI, nearby actors expose localized retail branch roots whose INFO
conditions pass; `INFO.RNAM` overrides the topic prompt, `TCLT` gates linked
choices, and begin/end fragments are separated by response completion.

See `docs/index.md` for profile, import, and mod-root usage.

The Skyrim quest journal opens with **J** and pauses gameplay. Use Up/Down or
D-pad to select a quest, Left/Right to switch active/completed quests, and
Page Up/Page Down or the right stick to scroll the entry. J, Escape, or B closes
it. The journal uses the installed Futura Condensed font, localized quest titles
and stage journal text, and live objective completion/failure state. It shows
only reached journal stages and player-visible objectives; opening it does not
advance quests. `ODAI_FNV_UI_DEMO=quests` opens it for isolated UI captures.

The Skyrim world map opens with **M** and pauses gameplay. Pan with arrow keys,
left stick, or right mouse drag; zoom with the wheel, +/- or controller triggers.
C/Y centers the last known exterior player position. Select a visible location
or place a session destination with Enter/A or a click; Delete clears it.
The map draws shaded terrain from installed LAND heights and water levels through
the resolved load order, with authored visible and discovered location markers.
It loads in bounded batches without making map cells gameplay-resident. This is
a topographic world map with simplified marker symbols; retail 3D cloud effects,
local interior maps, fast travel, and saved custom destinations are not implemented.
`ODAI_FNV_UI_DEMO=map` opens the map for isolated captures, waiting for terrain.

Riverwood and Bleak Falls Barrow remain visible and labeled on the world map
before discovery, using their installed marker positions. This does not mark
them discovered or grant fast travel.

World-map rasters are now cached atomically as `world-map-v1.bin` in the existing
content/load-order/worldspace cache directory. The first open reads only LAND
records (skipping placed objects and navigation); later launches load the bounded,
checksummed raster directly. Content changes select a new cache directory, and
corrupt or truncated cache files rebuild automatically. No player save is changed.

Imported terrain uses its original mesh by default. The experimental smoothing
and procedural displacement path is available with `ODAI_TERRAIN_TESS=1`, but
subdivision runs in both the depth and main passes and is expensive in Riverwood.
Leave this variable unset (or set it to `0`) for normal play. This setting does not
change imported assets, collision meshes, or cache formats.
