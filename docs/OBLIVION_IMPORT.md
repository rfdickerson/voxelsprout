# The Elder Scrolls IV: Oblivion — import status and plan

Oblivion is one Gamebryo generation earlier than Fallout 3 / New Vegas. This document
records what was **measured** against a retail install, what is **inferred** and still
needs proving, and the staged plan for getting one Tamriel cell's geometry into an
`ImportedScene`.

Everything below was measured on
`/home/rfdickerson/.local/share/Steam/steamapps/common/Oblivion` (Steam, Linux) using
`odai_bethesda_probe` plus byte-level dumps of the retail files. Where a claim is an
inference from layout rather than something a tool printed, it says so.

**Nothing in `src/import/fnv/` was renamed or re-namespaced.** Fallout 3 already runs
through the same code with no Fallout-3-specific branches (see CLAUDE.md), and Oblivion
should follow the same rule: additive version branches, never a fork.

---

## 0. Tamriel's TERRAIN already renders (2026-08-14)

No Oblivion-specific code, no cooking step:

```bash
odai_game_newvegas --stream "<.../Oblivion/Data>" --plugin Oblivion.esm --worldspace Tamriel
```

streams the Great Forest out of the retail install — 14686 exterior cells indexed, 13708
statics resolved, 11 archives opened, LTEX ground textures, region discovery naming the
region on screen. Everything below the NIF layer is already version-agnostic, which is the
real finding: **the only thing standing between this and Oblivion proper is the mesh reader.**
Every static logs `Not a Gamebryo NIF 20.2.0.7 file`, so the world is landscape and nothing
else — no trees, no rocks, no architecture, no actors.

## 1. Verified working as-is

### BSA archives — v103 now reads (implemented, one additive branch)

Oblivion ships **BSA v103**; FO3/FNV ship v104. Measured header of every retail archive:

| Archive | version | archiveFlags | folders | files | fileFlags |
|---|---|---|---|---|---|
| `Oblivion - Meshes.bsa` | 103 | `0x787` | 524 | 20182 | `0x41` |
| `Oblivion - Textures - Compressed.bsa` | 103 | `0x707` | 1090 | 18040 | `0x2` |
| `Oblivion - Misc.bsa` | 103 | `0x703` | 10 | 115 | `0x1a4` |
| `Knights.bsa` | 103 | `0x703` | 66 | 4810 | `0x1b` |

Every structure the reader touches is **byte-identical to v104**: the 36-byte header, the
16-byte folder records, the 16-byte file records, the `totalFileNameLength` bias on folder
offsets, the NUL-terminated file-name block, and the `uint32` original-size prefix ahead of
each zlib payload. Confirmed by parsing the whole folder/file table with the v104 layout and
getting clean paths (`trees\treecottonwoodsu.spt`, `meshes\architecture\bruma\brumaupperinnint.nif`).

**The one real difference is `kEmbedFileNames` (0x100).** Oblivion sets the bit — `0x787`
and `0x707` both have it — but writes **no embedded name in any data block**. Honouring it
eats the first `1 + N` payload bytes, so the zlib stream starts mid-header. Measured before
the fix: every sampled extraction failed with `incorrect header check` / `incomplete or
truncated stream`. Clearing the bit for v103 makes all 40 of 40 sampled files inflate to
their declared size.

Implemented at `src/import/fnv/bsa_archive.cc:25` (`kBsaVersionOblivion`), `:140`
(`peekBsaContentFlags`), `:165` (`open`) and `:177` (the flag mask). The mask is applied
once at open time rather than branching in `extract()`, so `extract()` runs the exact code
path it always did. Pinned by `testBsaArchiveReadsOblivionV103` in
`tests/fnv_import_tests.cc` on a synthetic fixture that reproduces the contradiction
(v103 header + `0x100` set + no embedded names).

Result, measured:

```
odai_bethesda_probe <Oblivion/Data> --archives
  Archives: 147629 files indexed, 0 failure(s).     # was: 0 indexed, 17 failures
odai_bethesda_probe <FNV/Data> --archives
  Archives: 182177 files indexed, 0 failure(s).     # unchanged
```

### DDS textures — no change needed

`--texture textures\architecture\castle\kvatch\KvatchCastleTowerLOD01.dds` decodes
`128x256 format=BC1 mips=8 bytes=21984` and runs the alpha-cutout classifier to a verdict.
Over 300 randomly sampled textures: DXT1 100, DXT3 84, DXT5 107, and **8 with a zero
fourCC** (uncompressed RGB/RGBA), which `src/import/dds.cc:107` rejects — the same
limitation FNV already has, not a new blocker. ~2.7% of Oblivion's textures are affected.

### LAND terrain records — byte-identical

`src/import/fnv/fallout_records.cc:444-468` reads Oblivion LAND unchanged. Measured
subrecord census over 200 Tamriel LANDs:

| Subrecord | Size | Same as FNV? |
|---|---|---|
| `VNML` | 3267 (33×33×3) | yes |
| `VHGT` | 1096 (4 + 1089 + 3) | yes |
| `VCLR` | 3267 | yes |
| `BTXT` | 8 | yes |
| `ATXT` | 8 | yes |
| `VTXT` | 8×N | yes |

LAND and PGRD are the only compressed record types in the WRLD tree (31083 of 31823 LANDs,
6584 of 6586 PGRDs). CELL, REFR, ACHR, ACRE are all uncompressed.

### Group / subrecord conventions

Group types in Oblivion's WRLD tree are `{1, 4, 5, 6, 8, 9, 10}` — the same worldspace →
exterior block → sub-block → cell → persistent/temporary/distant scheme FNV uses. The
`XXXX` oversized-subrecord convention is present and behaves identically (Tamriel's WRLD
carries `XXXX` + a 684-byte `OFST`), which `esm_reader.cc:60-67` already handles.

### Worldspace addressing

`Tamriel` is **formID `0x0000003C`**, subrecords `EDID NAM2 ICON MNAM DATA NAM0 NAM9 XXXX
OFST`. 84 worldspaces total. Cells are addressed exactly as in FNV: a `CELL` record whose
`XCLC` holds the grid `(x, y)`, nested under `GRUP` type 1 labelled with the worldspace
formID.

`XCLC` is **8 bytes** in Oblivion (two `int32`) against FNV's 12 (it gained a flags word).
`fallout_records.cc:185` already tests `sub.size >= 8u`, so this needs no change.

---

## 2. Verified to be different — the blockers

### Blocker 1: the ESM record header is 20 bytes, not 24 — SOLVED (Stage 1)

`esm_reader.cc` used to hardcode `kGroupHeaderSize = kRecordHeaderSize = 24u`.
FO3/FNV inserted a 4-byte formVersion/unknown field after the record's versionControlInfo
(and a versionControlInfo on the GRUP); Oblivion has neither. Measured on the first bytes
of both files:

```
Oblivion.esm   TES4 dataSize=744  ... bytes[16..20] = 00000000  bytes[20..24] = "HEDR"
FalloutNV.esm  TES4 dataSize=30   ... bytes[16..20] = 00000000  bytes[20..24] = 02000000
                                                                bytes[24..28] = "HEDR"
```

So the discriminator is cheap and unambiguous: **`HEDR` sits at file offset 20 for a
TES4-era plugin and at 24 for a FO3-era one.** That is a one-record sniff at `open()`,
before any walk. (Note the HEDR *version float* does **not** discriminate — Oblivion is
1.0, Fallout 3 is 0.94, New Vegas is 1.34, so 1.0 is not a clean boundary.)

Before Stage 1, this is what every probe subcommand died on:

```
--plugin Oblivion.esm     ->  walk failed: Malformed subrecords in record: TES4
--buildcell ...           ->  world tables FAILED: Malformed subrecords in record: TES4
--cellindex ...           ->  Index build FAILED: Truncated record data
```

Implemented as `EsmPluginFormat` (`esm_reader.h`) + `detectEsmPluginFormat()` /
`esmRecordHeaderSize()` / `esmGroupHeaderSize()` (`esm_reader.cc`), sniffed once by
`EsmReader::open()` via `finishOpen()` and hoisted out of the walk loop in `walkRange()`.
`readFalloutPluginHeader()` (`plugin_load_order.cc`) parses the TES4 record itself rather
than through `EsmReader`, so it got the same treatment — it now over-reads and seeks back
to the sniffed header size.

The subrecord header stays 6 bytes and the compressed-record flag stays `0x00040000`;
nothing else in the container format changed.

**After, on retail data:**

```
--plugin Oblivion.esm     ->  Walked 1167017 records of 64 types, 0 tolerated checksum failures
--plugin Knights.esp      ->  Walked 9563 records of 38 types
--cellindex Oblivion.esm Tamriel
                          ->  35494 cells (33560 exterior with grid coords), 84 worldspaces,
                              1025617 references mapped, built in 141 ms
                              Tamriel: 14687 exterior cells, SEWorld: 4187, CamoranParadise: 417
--buildcell Oblivion.esm Tamriel 1 7
                          ->  land=yes, packedVerts=1156, packedIndices=6144, terrainParts=4,
                              textures=10
```

1156 = 34×34, the four terrain quadrants meshed. **Tamriel's terrain builds.**

Note Tamriel (0,0) is open water and genuinely carries no LAND — `land=no` there is correct,
not a failure. (1,7) is the first cell with terrain.

**No FNV regression.** A baseline probe built from this same worktree at HEAD (Stage 1
stashed) produces byte-identical output to the Stage 1 binary across `--plugin`,
`--cellindex`, `--cells`, `--buildcell`, `--record LTEX`, `--archives` and `--regions` on
`FalloutNV.esm`. Pinned going forward by `testEsmReaderWalksBothHeaderGenerations` and
`testPluginHeaderReadsBothGenerations`, which build the same logical tree in *both* layouts
and assert the walks are indistinguishable — a one-sided Oblivion assertion would pass just
as well with the 24-byte path broken.

### Blocker 1b: LTEX names its texture in ICON, not through a TXST — SOLVED (Stage 1)

Found while verifying Stage 1: `--buildcell` reported `0 land textures` even with the
record layer working. `fallout_records.cc`'s `parseLandTextureRecord` read only `TNAM`
(the formID of a TXST whose TX00 holds the path), which is a Fallout 3-era indirection.
Oblivion's LTEX carries the path directly in `ICON`, and it is relative to
`textures\landscape\`, not `textures\` — so `normalizeTexturePath()` alone produces
`textures\Dementia\DementiaMoss01.dds`, which resolves to nothing.

Measured over all 229 LTEX records in `Oblivion.esm`: **229 carry ICON, 0 carry TNAM;
226 resolve under `textures\landscape\`, 0 under `textures\` directly**, and the 3 that
resolve nowhere name assets the game does not ship (`CHRock01.dds`,
`Oblivion\TerrainHDOblivionEvilSymbol01.dds`, `TerrainAnvilGrass01.dds`).

Fixed additively: an `ICON` branch that prepends `landscape\`, plus a guard so the
post-walk TXST resolution only fills a path that is still empty. Fallout plugins carry no
ICON on an LTEX, so the branch never fires for them. `--buildcell Tamriel 1 7` went from
`0 land textures` / `textures=0` to `229 land textures` / `textures=10`. Pinned by
`testOblivionLandTextureIconPath`.

### Blocker 2 (now the top blocker): NIF is 20.0.0.4, and there is no block-size table

`src/import/fnv/nif_scene.cc:15-16` pins one version:

```cpp
constexpr std::uint32_t kSupportedNifVersion = 0x14020007u;  // 20.2.0.7
constexpr std::string_view kHeaderMagicPrefix = "Gamebryo File Format, Version 20.2.0.7";
```

Rejected at `nif_scene.cc:208-211`. Measured over all 9612 NIFs the probe now finds:

```
odai_bethesda_probe <Oblivion/Data> --nifs 60000
  Found 9612 .nif entries across 9 archive(s).
  extract failures  0
  parse failures    9612
  [9612x] Not a Gamebryo NIF 20.2.0.7 file (unrecognized header line)
```

Version census over the 8032 NIFs in `Oblivion - Meshes.bsa`:

| Header string | Count |
|---|---|
| `Gamebryo File Format, Version 20.0.0.4` | 7282 |
| `Gamebryo File Format, Version 20.0.0.5` | 100 |
| `Gamebryo File Format, Version 10.2.0.0` | 490 |
| `Gamebryo File Format, Version 10.1.0.106` | 82 |
| `Gamebryo File Format, Version 10.1.0.101` | 8 |
| `NetImmerse File Format, Version 10.0.1.x / 4.x / 3.3` | 70 |

**Oblivion is 20.0.0.4, not 20.0.0.5.** (20.0.0.5 exists but is 1.2% of the archive.)
`userVersion` and `userVersion2` are both **11**, against FNV's 34 — which matters, because
`nif_scene.cc:794` widens `NiAVObject::flags` from `ushort` to `uint` when
`userVersion2 > 26`. That existing branch is already correct for Oblivion and takes the
16-bit path; it just never runs today.

Three header-layout differences, all measured on
`meshes\dungeons\fortruins\exterior\clutter\farraguttreeentrance.nif`:

1. **No block-size table.** `nif_scene.cc:253-259` reads `numBlocks` uint32s of block size.
   That table arrived at 20.2.0.5. At 20.0.0.4 the bytes at that position are the first
   block's data — reading them as sizes yields `[0, 12, 1819045736, 1684371311, …]`,
   which is the ASCII of the block's own name.
2. **No header string table.** `nif_scene.cc:261-273` reads `numStrings` / `maxStringLength`
   and the global string table. That arrived at 20.1.0.x. At 20.0.0.4 every name — node
   names, `NiSourceTexture` file names, shader names — is stored **inline in its block** as
   a `uint32`-length-prefixed string.
3. **`numGroups` is still there**, immediately after the block-type index table, and is 0 in
   practice.

So the header runs: magic line, `version`, `endian`, `userVersion`, `numBlocks`,
`userVersion2`, three `uint8`-length export strings, `numBlockTypes`, the block-type name
table, the per-block `uint16` type index, `numGroups` (+ groups), then block 0.

**The missing block-size table is the real cost, and it is structural.** `nif_scene.cc:1201-1207`
derives every block's start offset by accumulating `header.blockSize[i]`, which is what lets
the current reader parse only the block types it cares about and *skip* the rest byte-exactly.
At 20.0.0.4 there are no sizes: the block table can only be walked **strictly sequentially**,
so a reader must consume every block type in the file exactly, or it desyncs from the first
unknown block onward and every subsequent block is garbage.

**Measured cost of that.** Greedy coverage over the 5137 static-family NIFs
(`architecture/ landscape/ dungeons/ clutter/ furniture/ plants/ trees/`) at 20.0.0.x,
counting a file as usable only when *every* block type in it has a reader:

| Readers implemented | Files fully walkable |
|---|---|
| 14 (all the geometry + material + property types) | 478 (9.3%) |
| 18 (+ `bhkCollisionObject`, `bhkNiTriStripsShape`, `bhkMoppBvTreeShape`, `bhkRigidBody`) | 2683 (52.2%) |
| 19 (+ `bhkRigidBodyT`) | 3770 (73.4%) |
| 21 (+ `bhkConvexVerticesShape`, `bhkBoxShape`) | 4604 (89.6%) |
| 28 | 4870 (94.8%) |
| 59 | 4920 (95.8%) — long tail |

**Read that table before estimating this work.** The geometry types alone buy 9.3%. What
takes it to ~90% is being able to *skip* the Havok collision blocks, which this engine will
never use for anything — `bhkCollisionObject` appears 7484 times across the archive, roughly
one per static mesh. There is no way around them: they sit between the geometry blocks and
carry no length of their own.

For scale on the narrower target: the meshes placed by the 3×3 Tamriel cells around the
origin (200 distinct NIFs resolved from 310 REFR base records) span **83 distinct block
types**.

**Verified feasible.** A hand-written sequential reader for a small subset was run against
`meshes\architecture\castle\kvatch\kvatchcastletower01_far.nif` and consumed blocks
back-to-back with no drift:

```
  numBlocks=7  numGroups=0  first block at 260
  block 0 NiNode             260..369  (109 bytes)
  block 1 NiTriShape         369..489  (120 bytes)
  block 2 NiBinaryExtraData  489..7763 (7274 bytes)
```

The layouts that fell out of that, byte-checked against the hex dump:

- `NiObjectNET` = `name` **SizedString(u32)** + `numExtraData` u32 + refs + `controller` ref.
  Compare `nif_scene.cc:768`, which reads `out.nameRef` as a **u32 index into the header
  string table**. This is the single most invasive difference in the block layer: it changes
  the first field of nearly every block.
- `NiAVObject` = `NiObjectNET` + `flags` **u16** + translation(3f) + rotation(9f) + scale(f) +
  `numProperties` u32 + refs + `collisionObject` ref. No `velocity`, no `hasBoundingVolume`.
- `NiGeometry` tail = `data` ref + `skinInstance` ref + `hasShader` **u8** (+ `shaderName`
  SizedString + unknown i32 when set). The current FNV reader
  (`readNiTriBasedGeom`, `nif_scene.cc:887`) does not have the `hasShader` field.
- `NiExtraData` derives from `NiObject`, **not** `NiObjectNET`: `name` only, no extra-data
  list and no controller ref. Getting this wrong desyncs immediately.
- `NiTexturingProperty` at 20.0.0.4 = `NiObjectNET` + `applyMode` u32 + `textureCount` u32 +
  per-slot `has` u8 (+ source ref, clamp, filter, uvSet when set). The measured block is 48
  bytes with `applyMode=2`, `textureCount=7`; **5 trailing bytes are not yet attributed** and
  need pinning before this reader is written for real.
- `NiSourceTexture` = `NiObjectNET` + `useExternal` u8 + `fileName` **SizedString**, e.g.
  `textures\architecture\castle\kvatch\KvatchCastleTowerLOD01.dds` read inline.

### Measured again with a working sequential walker (2026-08-14)

`src/tools/oblivion_nif_lab/` is that walker: a standalone Python BSA v103 reader plus a
20.0.0.4 block walker with per-type readers, and `coverage.py` reports the state of play over
the whole archive. It exists because iterating on binary layouts against 7382 real files is
minutes in Python and hours in C++; the layouts it settles get ported afterwards.

It self-checks two ways, and both matter. Per block, the next block must open with a
plausible SizedString length — which localizes a desync to the block that CAUSED it instead
of to wherever the file finally falls apart. Per file, after `numBlocks` blocks the remainder
must be exactly the footer, `4 + 4*numRoots` bytes and nothing else.

With 17 readers written (the geometry, property and extra-data types), **640 of 7382 files —
8.7% — have a reader for every block type they contain.** That lines up with the 9.3% in the
greedy table above and confirms it from a second direction.

**THE OBVIOUS ESCAPE HATCH DOES NOT EXIST, and this is the most useful thing measured here.**
A desync only corrupts blocks AFTER the unknown one, so "walk until the first block type you
do not know, keep the geometry you already have, and stop" looks like it should salvage most
of the archive for free. It salvages almost nothing:

| | files | share |
|---|---|---|
| NO geometry before the first unknown block | 5987 | 81.1% |
| all geometry before the first unknown block | 1023 | 13.9% |
| some geometry before the first unknown block | 330 | 4.5% |
| no geometry at all | 42 | 0.6% |

The reason is structural: `collisionObject` is a field on `NiAVObject`, so the root node
names its `bhkCollisionObject` before it names its children, and the Havok blocks are emitted
ahead of the geometry they belong to. Four files in five hit Havok before they hit a single
`NiTriShape`. **The Havok layouts are not optional and cannot be deferred.**

Which four to write first is also now measured. The single most common set of missing types,
in **2631 files**, is exactly:

```
bhkCollisionObject, bhkMoppBvTreeShape, bhkNiTriStripsShape, bhkRigidBody
```

with `bhkRigidBodyT` swapped for `bhkRigidBody` in another 1433, and
`{bhkCollisionObject, bhkConvexVerticesShape, bhkRigidBody}` in 644. So seven Havok layouts
(`bhkCollisionObject`, `bhkRigidBody`, `bhkRigidBodyT`, `bhkMoppBvTreeShape`,
`bhkNiTriStripsShape`, `bhkConvexVerticesShape`, `bhkBoxShape`) unlock the bulk of the
archive, and none of them needs to be *understood* — only sized exactly.

Layout corrections the walker turned up, beyond what section 2 records:

- **Every `*ExtraData` type takes the name-only base**, not `NiObjectNET` — `NiStringExtraData`,
  `NiBinaryExtraData`, `NiIntegerExtraData`, `BSXFlags` and `BSBound` all desync immediately
  if given the extra-data list and controller ref. `BSXFlags` alone appears in 384 of the
  first 500 sampled files, so this is the first wall anyone hits.
- **`NiTexturingProperty` has no `flags` u16 at 20.0.0.4.** Its per-slot `TexDesc` is source
  ref + clamp u32 + filter u32 + uvSet u32 + `hasTextureTransform` u8, and that trailing bool
  (plus the transform it guards) is what the "5 trailing bytes are not yet attributed" note
  in section 2 was seeing.
- **`NiGeometryData`'s UV field is stock Gamebryo `NiVectorFlags` here**, bits 0-5 a COUNT --
  not Bethesda's `BSVectorFlags` where bit 0 is a boolean. `nif_scene.cc` deliberately uses
  the boolean reading because that is correct for FO3/FNV; the version branch has to pick.

### Stage 2 SOLVED: the layouts, and how (2026-08-14)

**A sequential 20.0.0.4 walker now parses 82.1% of the archive, and 91.7% of
Anvil's own architecture.**

| set | walked |
|---|---|
| all 20.0.0.x meshes | 6058 / 7382 (82.1%) |
| `architecture/` | 1702 / 1808 (94.1%) |
| `architecture/anvil/` | **66 / 72 (91.7%)** |
| `dungeons/` | 1957 / 2083 (94.0%) |
| `clutter/` | 754 / 826 (91.3%) |

**The layouts came from nif.xml, not from the data.** Three search strategies were
tried first and all three are dead ends, which is worth recording so nobody
repeats them: per-type brute force is exponential because the Havok blocks arrive
as one consecutive chain; only **3 files** in the archive have exactly one unknown
block instance to bootstrap from; and only **2 files** have the unknown run as a
suffix. Resolving `nif.xml` for version 20.0.0.4 / BSVER 11 gives every field
outright, and says WHY each one is or is not present.
`src/tools/oblivion_nif_lab/from_nifxml.py` does that resolution.

Sizes, resolved and then confirmed against 7382 retail files:

| block | layout at 20.0.0.4 |
|---|---|
| `bhkCollisionObject` | target Ptr + `bhkCOFlags` u16 + body Ref = **10 B** |
| `bhkRigidBody` / `bhkRigidBodyT` | 228 B body + numConstraints u32 + 4·n + bodyFlags u32 = **236 + 4n** |
| `bhkMoppBvTreeShape` | shape Ref + unused[12] + scale f + dataSize u32 + offset Vector4 = **40 + dataSize** |
| `bhkNiTriStripsShape` | material + radius + unused[20] + growBy + scale Vector4 + refs + filters = **56 + 4n + 4m** |
| `bhkConvexVerticesShape` | material + radius + two 12-B properties + verts + normals = **40 + 16v + 16n** |
| `bhkBoxShape` | **32 B** · `bhkCapsuleShape` **48 B** · `bhkSphereShape` **8 B** |
| `bhkTransformShape` / `bhkConvexTransformShape` | **84 B** |
| `bhkListShape` | numSubShapes + refs + material + two 12-B properties + filters = **36 + 4n + 4m** |

`bhkRigidBodyT` is byte-identical to `bhkRigidBody` -- the T changes whether
translation/rotation are HONOURED, not whether they are stored. And
`hkMoppCode`'s Build Type is gated on `#BS_GT_FO3#`, so Oblivion does not have it.

Two non-Havok bugs mattered more than any Havok layout, and both were mine:

- **`NiTexturingProperty` ends with `Num Shader Textures`** (present from 10.0.1.0),
  which a count-driven slot loop misses. Omitting it under-consumes every
  texturing property in the archive and desyncs the `NiSourceTexture` after it.
  Fixing this alone took no-Havok files from **96 to 630** of 1280.
- **THE VALIDATOR ITSELF WAS WRONG.** The per-block plausibility check assumed
  every block opens with a SizedString name. Havok blocks open with a Ref and
  `NiGeometryData` opens with Group ID, so the check was aborting *correct* walks
  and blaming the Havok layouts, which had been right for some time. Fixing it
  took Havok files from **0 to 5361** of 6102 in one change. When an oracle and a
  parser disagree, suspect the oracle too.

Remaining failures are dominated by `NiSkinInstance` / `NiSkinData` /
`NiSkinPartition` (698 files) -- skinned characters, not buildings -- plus
`bhkLimitedHingeConstraint`, `bhkRagdollConstraint` and `NiParticleSystem`.

### What is NOT a blocker in the NIF layer

Two things that look like they should be, and measurably are not:

- **`NiTriStrips` is already supported.** The FNV reader handles both
  `NiTriShapeData` and `NiTriStripsData`, expanding strips into a triangle list
  (`nif_scene.cc:1099-1142`). Oblivion's heavy strip use (38372 `NiTriStripsData` against
  1956 `NiTriShapeData`) costs nothing new.
- **The `NiTexturingProperty` → `NiSourceTexture` texture chain already exists**
  (`nif_scene.cc:566-601`, `605+`), written because retail FNV still uses it on cliffs and
  rock formations. Oblivion has no `BSShaderPPLightingProperty` / `BSShaderTextureSet` at
  all — it is 28292 `NiTexturingProperty` + 29203 `NiSourceTexture` — so the path this
  engine needs is the one already there. The scanning heuristic in
  `findSourceTextureFileName` even accepts the inline-SizedString spelling already.

---

## 3. Not yet investigated

Named explicitly so nobody reads silence as "fine":

- **NPC_/CREA/RACE actor assembly.** Oblivion RACE and body-part slots are a different shape
  from FO3's. Out of scope for first geometry.
- **`.kf` animation.** Oblivion's controller blocks (`NiControllerSequence`,
  `NiTransformInterpolator`, `NiStringPalette`) differ from FNV's, and there are no
  B-spline-compressed interpolators at 20.0.0.4.
- **WTHR/CLMT weather.** Oblivion's NAM0 has fewer time-of-day slots than New Vegas's six.
  `weather_records.cc` will need a size branch.
- **DIAL/INFO dialogue** and the voice-path convention (`sound\voice\oblivion.esm\...` —
  the probe already prints this, and CLAUDE.md notes the first component is the plugin's own
  file name, which is already generalized).
- **`plugin_load_order.cc` remapping.** `readFalloutPluginHeader` now reads an Oblivion
  TES4 header correctly (pinned by `testPluginHeaderReadsBothGenerations`), but
  `remapFormId` and multi-plugin load order have not been exercised against a real Oblivion
  mod set.
- **The 580 pre-20.0.0.x NIFs** (10.1/10.2/NetImmerse). These are a further version tier and
  should stay unsupported; they are 7% of the archive and mostly interiors and props.

---

## 4. Staged plan

Each stage is independently landable and independently verifiable with an existing probe
subcommand. Stages 1 and 2 touch shared parsers and must be **additive version branches**,
gated on a sniffed plugin/NIF version, never on a game identity.

### Stage 0 — BSA v103 ✅ done

`src/import/fnv/bsa_archive.{h,cc}` + `testBsaArchiveReadsOblivionV103`. Verified:
`--archives` indexes 147629 Oblivion files with 0 failures, `--find` and `--texture` work,
FNV still indexes 182177 with 0 failures, `ctest` 49/49 green.

### Stage 1 — 20-byte ESM record/group headers + LTEX ICON ✅ done

See Blockers 1 and 1b above for the implementation and the measured results. End state:
the whole record layer works on Oblivion, and Tamriel's terrain builds textured. `ctest`
49/49, and FNV probe output is byte-identical to a same-source baseline.

One thing deliberately NOT done: no `EsmPluginFormat` value is exposed to or consulted by
any caller above `src/import/fnv/`. The format is a property of the file, sniffed at open,
and nothing downstream should ever branch on which game it is.

### Stage 2 — NIF 20.0.0.4 header + sequential block walk  ← **next**

Split the version-specific parts of `nif_scene.cc` behind a `NifVersion` carried on
`NifHeader`, so the 20.2.0.7 path is untouched:

1. `parseHeader` accepts `20.0.0.4` and `20.0.0.5`; skips the block-size and string-table
   reads for them; keeps `numGroups`.
2. Add a **sequential block walker** used only when there is no size table. Block starts are
   discovered as the previous block's reader finishes, so every reader must consume its
   block exactly. Track a "desynced" flag: the moment a block type has no reader, abandon the
   file and report it, rather than emitting geometry from a garbage offset.
3. Name resolution becomes `SizedString` inline instead of a string-table index. Give
   `AvObjectFields` a `name` string alongside `nameRef` so the two paths converge downstream.
4. Implement the 21-reader set from the coverage table (geometry + properties +
   `bhkCollisionObject`, `bhkNiTriStripsShape`, `bhkMoppBvTreeShape`, `bhkRigidBody`,
   `bhkRigidBodyT`, `bhkConvexVerticesShape`, `bhkBoxShape`). The Havok readers only need
   correct *lengths*, not meaning — but the MOPP blob and the convex-vertices arrays are
   variable-length, so they must be genuinely parsed, not offset-skipped.

Verify with `--nifs 60000` and `--nifblocks <path>`: target ≥89% of static-family NIFs
parsed with non-zero shapes, and 0 files that produce geometry after a desync.

Pin the byte layouts with a synthetic-NIF fixture in `fnv_import_tests.cc` — a hand-built
20.0.0.4 file with a `NiNode` → `NiTriStrips` → `NiTriStripsData` chain plus one Havok block
between them, so the test fails if a reader's length is wrong rather than only if geometry
is wrong.

### Stage 3 — one Tamriel cell into an `ImportedScene`

With Stage 2 landed, `CellSceneBuilder` should already work the rest of the way: it consumes
records and `ImportedScene` shapes, neither of which is version-specific, and it already
produces terrain for an Oblivion cell today.

- `--buildcell Oblivion.esm Tamriel 1 7` for the smoke test (**not** `0 0` — that cell is
  open water and legitimately has no LAND).
- **`kCellBuildVersion` (`cell_streamer.cc:64`, currently 11) must be bumped** if any change
  in Stages 1–2 alters what the importer decides about a material or a vertex flag, or every
  existing FNV install keeps serving pre-change geometry. A pure ESM-header change does not
  need it; a NIF material change does.
- Only then consider a `src/games/oblivion/` viewer target. Do not scaffold one earlier —
  there is nothing to look at until Stage 1, and nothing worth looking at until Stage 2.

### Stage 4 and beyond — out of scope for now

Actors, animation, weather, dialogue. Each is its own version-branch problem and none blocks
seeing Tamriel.

---

## 5. Reproducing the measurements

```bash
cmake -S . -B cmake-build-tools -G Ninja \
  -DODAI_BUILD_APP=OFF -DODAI_BUILD_TOOLS=ON -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=x64-linux
cmake --build cmake-build-tools -j

OB="$HOME/.local/share/Steam/steamapps/common/Oblivion/Data"
cmake-build-tools/odai_bethesda_probe "$OB" --archives
cmake-build-tools/odai_bethesda_probe "$OB" --nifs 60000                 # fails: Stage 2
cmake-build-tools/odai_bethesda_probe "$OB" --plugin Oblivion.esm
cmake-build-tools/odai_bethesda_probe "$OB" --cellindex Oblivion.esm Tamriel 5
cmake-build-tools/odai_bethesda_probe "$OB" --buildcell Oblivion.esm Tamriel 1 7
ctest --test-dir cmake-build-tools --output-on-failure
```

No Vulkan is required for any of it.
