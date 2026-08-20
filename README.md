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

See `docs/index.md` for profile, import, and mod-root usage.
