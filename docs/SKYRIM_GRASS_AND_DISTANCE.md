# Mod placements, procedural grass, and why Whiterun looks broken from a hill

Three findings that arrived together while installing SMIM and Skyrim Flora
Overhaul. The first corrects something this project believed; the second is the
feature SFO actually needs; the third is why the horizon is empty.

---

## 1. Reading placements from mods ALREADY WORKS

This was recorded as missing. It is not. `--plugin-add <Plugin.esp>` builds a
`FalloutLoadOrder`, and from there:

| stage | what already merges across the order |
|---|---|
| `buildFalloutWorldTables(order, …)` | base records — STAT/TREE/FLOR/LIGH models, land textures, worldspace defaults — later plugin wins |
| `buildFalloutCellIndex(order, …)` | per cell, a `contributions` list: every plugin with something to say about it |
| `extractFalloutCellMerged(…)` | walks those contributions in order and merges references |
| `remapCellFormIds(…)` | rewrites every formID from the plugin's local mod-index space into the order's global one |

`CellStreamer::setLoadOrder` switches the whole pipeline onto that path, the
cell cache key includes `m_loadOrder.fingerprint()`, and mod `.esp` files
resolve out of `--mod` directories first. Verified on Skyrim:

```
[newvegas] streaming across 2 plugins (record overrides active)
[newvegas] load order: Skyrim.esm -> Skyrim Flora Overhaul.esp
```

**The gate is that it only engages when `--plugin-add` is passed.** A mod
installed with `--mod` alone contributes assets and nothing else, which is
correct — but it means "I installed the mod and its content is missing" has two
very different causes, and only one of them is a missing feature.

**So the honest correction:** the earlier claim that a mod's records are inert
came from a stale note rather than from reading `cell_streamer.cc`. Placement
merging is built, wired and working.

---

## 2. What SFO actually needs is a GRASS SYSTEM, not placements

Skyrim Flora Overhaul's plugin is 100 records:

```
  66  GRAS      14  LTEX       1  CELL
  16  TREE       1  WRLD       1  REFR
```

**One REFR.** Merging placements perfectly buys this mod nothing, because
Skyrim does not place grass as references at all. Grass is procedural:

```
  LTEX (a land texture)  --GNAM-->  GRAS (a grass type)  -->  MODL (the billboard mesh)
        |                                  |
   painted per-quadrant                DATA: density, slope range, height/position
   by LAND's BTXT/ATXT                 jitter, wave period
```

The engine scatters instances of that GRAS over every part of the terrain
painted with that LTEX. Nothing is stored per instance; the placement is
regenerated from the texture map and a seed.

That is why 65 of SFO's 92 meshes sit installed and unreachable: they are
`vurt_*` grass billboards that only a grass scatter can ever ask for.

### The GRAS DATA layout, read off SFO's own records

32 bytes, verified against `vurt_fallgrass` and `FernGrass01`:

| offset | field | example |
|---|---|---|
| 0 | density (u8) | 20 |
| 1 | min slope, degrees (u8) | 0 |
| 2 | max slope, degrees (u8) | 40 |
| 4 | units from water (u16) | — |
| 8 | water distance mode (u32) | — |
| 12 | position range (f32) | 32.0 |
| 16 | height range (f32) | 0.6 |
| 20 | colour range (f32) | 0.0 |
| 24 | wave period (f32) | 180.0 |
| 28 | flags (u8) | 6 |

The `0xcd` filler bytes in the unused slots are the giveaway that the layout is
right — that is uninitialised Creation Kit memory, and it only lands in the
fields the record does not use.

### Implementation sketch

The scatter itself is not new work: `src/world/grass_scatter.{h,cc}` already
does deterministic billboard scatter for the voxel world, and the FNV importer
already knows which LTEX paints each terrain quadrant (that is where the
four-slot layer blend comes from). What is missing is the join.

1. **Read GRAS** in `fallout_records.cc` — an EDID + MODL + the DATA above,
   into `FalloutGrassRecord`, keyed by formID like every other base record.
2. **Read LTEX's GNAM** — one formID, alongside the TNAM/MNAM already parsed.
   `FalloutWorldTables` gains `grassByLandTexture`.
3. **Scatter at cell build**, in `CellSceneBuilder::addCellTerrain`, where the
   per-post layer weights already exist. For each terrain post, for each layer
   with weight above a threshold, take that LTEX's GRAS list and place
   `density` instances per unit area, jittered by `position range`, rejected
   against `min/max slope` using the post normal already decoded, and seeded by
   `(cellX, cellZ, postIndex, grassFormId)` so the result is identical every
   time the cell is built or loaded from cache.
4. **Emit as instances**, not as unique geometry — `ImportedSceneInstance`
   already carries a transform per placement, and a cell's grass is thousands
   of copies of a handful of meshes.

Two traps worth stating before anyone starts:

- **It must be deterministic and cache-stable.** Grass lands in the packed
  vertex stream, so a scatter that depends on anything but the cell's own data
  makes a cached cell differ from a rebuilt one, and the difference appears as
  grass that moves when you walk away and come back.
- **Density is per unit AREA, not per post.** Skyrim's posts are 128 units
  apart; treating density as per-post gives 1/16th the grass on the same
  terrain and reads as "the mod barely did anything".

---

## 3. Whiterun from a distance: no LOD, and a 234 m horizon

Distinct from the above and worth not confusing with it. Seen from a hill in
Tamriel, Whiterun is a few disconnected fragments of wall on a bare ridge. Two
causes, neither a mesh bug:

- **The load radius is 4 cells** — 16384 units, about 234 m. Past it there is
  no geometry at all. The fragments are the last loaded ring, which is why they
  look cut off rather than absent.
- **Skyrim's own distant assets are never loaded.** Tamriel alone ships **717
  `.bto` object LOD, 3060 `.btr` terrain LOD, 329 `.btt` tree LOD** and 6123 LOD
  textures. `loadDistantLandLod()` handles FNV/FO3 terrain tiers only, is off by
  default, and `docs`' own note records why a whole-world tier does not work:
  a coarse tier drowns the detailed terrain, a fine one exhausts the 1024-slot
  bindless texture table.

The design that fits both is the one already written down in
`newvegas_app.cc`: **per-tile chunks with tier RINGS** — fine tiles just outside
the loaded cells, coarser further out, tiles overlapping the loaded square
excluded. That bounds triangle count and, more importantly, texture count.
Skyrim's `.bto` files make this more attractive than it was for FNV, because
object LOD is exactly the missing content: the city silhouette on the ridge.
