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
not read or write Skyrim `.ess` files. The deterministic session, world registry,
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
