# Skyrim SE asset mods — what works, and what installing one buys

The mechanism is the same one `docs/FNV_MODS.md` documents: `--mod <dir>` adds a
root laid out like `Data` itself, searched ahead of the game's loose files and
archives, indexed recursively and matched case-insensitively. This file records
only what is Skyrim-specific and what was measured.

## The filter that decides whether a mod is worth installing

**Assets replace on `--mod` alone. Records need `--plugin-add` as well.**

| Mod ships | `--mod` only | `--mod` + `--plugin-add` |
|---|---|---|
| a mesh/texture at a path the game already uses | replaces it | replaces it |
| a NEW mesh placed by REFR records | nothing references it | **placed** |
| base-record edits (models, land textures, trees) | ignored | **applied**, later plugin wins |
| grass (GRAS assigned to a land texture) | — | **still nothing** — procedural GRAS scattering is outside the retained renderer |

Prefer Skyrim's active profile through auto-discovery or `--load-order
<plugins.txt>`. Repeatable `--plugin-add` entries are appended after that
authoritative order. Base records and cell contents come from every plugin in
order, regular and ESL form IDs are remapped into distinct slots, and the
ordered fingerprint joins the cell-cache and traversal-state identities.

## Installed here

Both live under `~/.local/share/odai/skyrim-mods/`, self-contained, nothing
copied into the Steam install.

```bash
SSE="$HOME/.steam/steam/steamapps/common/Skyrim Special Edition/Data"
MODS="$HOME/.local/share/odai/skyrim-mods"

ODAI_FNV_TEX_SIZE=1024 cmake-build-app/odai \
  --stream "$SSE" --plugin Skyrim.esm --worldspace Tamriel \
  --mod "$MODS/smim" --mod "$MODS/sfo"
```

`ODAI_FNV_TEX_SIZE=1024` is not optional decoration: the default ceiling is
**512**, which mip-drops a high-resolution pack straight back to vanilla
resolution — full I/O cost, zero visible change.

### SMIM (Static Mesh Improvement Mod) — lands in full

The archive is a FOMOD; `00 Core` is the base install and is what is unpacked
here (the rest are style options: barrel textures, rope variants, chain sets).

**417 of its 418 meshes replace a vanilla path**, plus 17 textures. The one
new-only file is `meshes\betterdynamicsnow\shader\snowshader.nif`, which
belongs to a different mod's shader hook and does nothing here.

### Skyrim Flora Overhaul 2.74a — lands in part, and the part matters

**27 of 92 meshes and 86 of 236 textures replace vanilla paths.** Those 27 are
the vanilla tree and grass meshes (`treepineforest*`, `ferngrass01`,
`fieldgrass01`, `tundragrass01`…), and they carry most of what the mod is for.

The other 65 meshes are `vurt_*` grass billboards, and they stay unreachable
even with `--plugin-add`, because **Skyrim does not place grass as
references**. The plugin is 66 GRAS, 16 TREE, 14 LTEX and exactly ONE REFR:
grass is scattered procedurally from the land texture painted on the terrain,
and that procedural renderer is intentionally outside the Bethesda-only slice.

The 16 TREE records DO apply with `--plugin-add`, which is worth having on its
own.

## Measured

Same pinned camera, same 81 streamed cells, Tamriel at the default spawn:

| | vertices uploaded | pixels changed |
|---|---|---|
| vanilla | 4,114,930 | — |
| SMIM + SFO | **4,642,620** (+12.8%) | 4.2% (Whiterun framing) |

The vertex count is the honest instrument here — a screenshot diff cannot tell
"a denser mesh loaded" from "the wind moved a branch". Visually the change is
unmistakable on conifers: vanilla's sparse spindles become full crowns.

Whiterun's own cells gain only +1.6%, which is the expected shape: the city is
architecture, and SMIM's architecture work there is subtler than its clutter.

## Cache

Every distinct (plugin set, mod set, texture ceiling) gets its own cell cache
under `~/.cache/odai/fnv/`, several hundred MB each. Installing a mod does not
corrupt the old cache; it stops consulting it. Prune when disk matters.
