# Scalable Bethesda mod profiles

`odai` can consume an existing mod-manager profile without copying, editing,
downloading, or reordering its content. The resolved result is one immutable
content graph shared by streaming, asset lookup, plugin loading, caches, and
the Bethesda probe.

Supported profile sources are:

- an MO2 profile directory containing `modlist.txt` and `plugins.txt`;
- an OpenMW `openmw.cfg`, including chained `config=` files;
- an ODAI JSON profile version 1.

## Launching a profile

MO2 does not store the game installation's Data path in the profile directory,
so provide it with `--stream`. The normal sibling `mods` directory is inferred;
use `--mods-root` for a portable or nonstandard instance.

```bash
odai --profile "/path/to/MO2/profiles/Default" \
  --stream "/path/to/Skyrim Special Edition/Data" \
  --worldspace Tamriel

odai --profile "$HOME/.config/openmw/openmw.cfg" --worldspace Vvardenfell
```

`--stream` replaces the profile's base Data root. Repeatable `--mod` roots and
`--plugin-add` plugins append at the highest priority. `--profile` and
`--load-order` cannot be combined because both claim to be authoritative.

Use `--list-profiles` to show conventional profiles and paths listed in the
colon-separated `ODAI_PROFILE_ROOTS` environment variable. `--profile-picker`
selects the only discovered profile; when several exist it lists them and asks
for an explicit `--profile`, avoiding a silent choice.

## Precedence

MO2's reverse-written `modlist.txt` is converted to ascending asset priority,
with the instance `overwrite` directory last. Plugin activation from
`plugins.txt` remains independent from plugin ordering in `loadorder.txt`.
When `archives.txt` exists it is authoritative. Otherwise ODAI activates
official BSAs and BSAs associated with active plugins, and warns about other
archives instead of loading them alphabetically.

OpenMW parsing supports quoting, comments, environment and OpenMW path tokens,
`config=`, `replace=`, `data=`, `data-local=`, `fallback-archive=`, and
`content=`. Later data roots and archives win; `data-local` is last. Include
cycles and missing required archives are launch errors.

ODAI JSON v1 contains `version`, `name`, `game`, `data_root`, and ordered
`layers`, `plugins`, and `archives`. Relative paths are resolved beside the
manifest. Export a deterministic manifest from either external format with:

```bash
odai_bethesda_probe --export-profile <profile> odai-profile.json \
  --data "/path/to/game/Data"
```

## Diagnostics and cache behavior

```bash
odai_bethesda_probe --profilecheck <profile> [--data <Data>]
odai_bethesda_probe --why <profile> <virtual-path-or-0xFormID> [--data <Data>]
odai_bethesda_probe --conflicts <profile> [--data <Data>]
```

`--profilecheck` reports the resolved regular/light plugin slots, diagnostics,
record overrides/deletions, and executable script assets. `--why` shows the
winning loose/archive provider or complete record override chain. The pause
menu exposes the active profile and its launch diagnostics; `--compat-report
<file.json>` writes the same launch information atomically.

The familiar cooker also accepts the resolved graph and bakes merged winning
records rather than a single plugin in isolation:

```bash
odai_newvegas_cooker --profile <profile> region.bin --data <Data> \
  --worldspace Tamriel -2 -2 2 2
odai_newvegas_cooker --profile <profile> interior.bin --data <Data> \
  --cell MyInteriorEditorID
```

Each content layer receives a persistent case-insensitive loose-file index
under the XDG cache directory. Changed layer metadata causes a rebuild;
`--reindex-content` forces one after manual edits. The ordered manifest digest
covers the profile source, game, layer order, plugin metadata, and archives, so
content changes select a different cell-cache namespace.

## Compatibility boundary

The current target includes Bethesda gameplay runtimes as well as large-world
exploration. ODAI executes clean-room TES3 MWScript and Skyrim Papyrus/VMAD
behavior through generation-specific deterministic VMs. Unsupported commands
remain visible in strict profile reports; they are not silently accepted.
OpenMW Lua, MWSE, SKSE, OBSE, and native extender DLLs remain disabled. It is
not a LOOT replacement and never downloads content.

That boundary supports rendering and traversing Tamriel Rebuilt-, Morroblivion-,
and Skywind-scale worlds without claiming quest or gameplay parity.
