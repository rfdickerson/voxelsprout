# Skyrim Special Edition and Morrowind — import status and plan

Two more Bethesda generations, one on each side of the three this engine already
reads. Everything below was **measured** against retail installs on 2026-08-14
using `odai_newvegas_probe` plus byte-level dumps; where a claim is an inference
rather than something a tool printed, it says so.

The rule from `docs/OBLIVION_IMPORT.md` still holds and matters more with every
generation added: **additive version branches, never a fork.** Nothing in
`src/import/fnv/` was renamed. The directory name is now wrong for four of the
five games it reads, and that is deliberate — renaming it would be a large
diff that changes no behaviour and breaks every existing reference.

**The headline is that the two games are nothing like each other in difficulty.**
Skyrim SE is one block reader away from static geometry. Morrowind is a
different container format, a different record format, and a different mesh
format, and shares almost nothing below the texture layer.

---

## 0. Where they are

| Game | Data directory | Master |
|---|---|---|
| Morrowind | `~/.steam/steam/steamapps/common/Morrowind/Data Files` | `Morrowind.esm` |
| Skyrim Special Edition | `~/.steam/steam/steamapps/common/Skyrim Special Edition/Data` | `Skyrim.esm` |

Note Morrowind's is `Data Files`, with a space, not `Data`.

---

## 1. Skyrim SE — what already works, unchanged

### The plugin needs nothing

`Skyrim.esm` walks today, with no code written for it:

```
odai_newvegas_probe <SSE Data> --plugin Skyrim.esm
  Walked 869688 records of 119 types.
    693333  REFR      15966  NAVM
     31465  INFO      15564  LAND
     17568  CELL      15037  DIAL
  Tolerated checksum failures: 0
```

TES4 magic with `HEDR` at offset 24, so `detectEsmPluginFormat` sniffs it as
`kFallout3` and the 24-byte container headers are correct. The subrecord layer,
the `XXXX` oversize convention, compressed records and the GRUP tree are all
shared. **The whole record layer is already generation-correct for Skyrim.**

### The NIF header needs nothing either

Skyrim SE meshes are `Gamebryo File Format, Version 20.2.0.7`, `userVersion 12`,
**`BSVersion 100`** — censused over 3000 meshes from `Skyrim - Meshes0.bsa`,
which returned exactly one distinct header tuple. The version word is the same
one FO3/FNV use, so `parseHeader` accepts it, the block-size table is present,
and the global string table is present.

`BSVersion` (`userVersion2`) is 100 against New Vegas's 34. The one place that
already branches on it — `NiAVObject::flags` widening from `ushort` to `uint`
past 26 (`nif_scene.cc:829`) — is a `>` comparison and is therefore already
right for Skyrim.

Measured after the archive work in section 2 landed:

```
odai_newvegas_probe <SSE Data> --nifs 4000
  Found 22091 .nif entries across 3 archive(s).
  extract failures  0
  parse failures    0
  with geometry     0
  shapes            0
```

**Zero parse failures and zero shapes** is the whole situation in two numbers:
every file opens, and none of them contains a block type this reader knows how
to take geometry out of.

---

## 2. Skyrim SE — BSA v105 ✅ done

Two structural differences from v104, not one, and only the second is widely
documented.

**The folder record grew from 16 bytes to 24.** A 4-byte pad and a 64-bit data
offset replace the 32-bit one. A reader that assumes the v104 stride does not
fail — it reads the folder table as noise and produces binary garbage where
folder names should be, which reads as a corrupt archive rather than as a
layout mismatch. Confirmed both ways against `Skyrim - Meshes0.bsa` (978
folders, 19443 files): the 24-byte stride yields
`meshes\dungeons\nordic\levers\pullchainanim` as the first folder, and the
16-byte stride yields nothing printable.

**Compressed entries are LZ4 frames, not zlib.** The payload behind the
`uint32` original-size prefix opens `04 22 4d 18`, which is the LZ4 frame magic.

Everything else is v104: the 36-byte header, 16-byte **file** records, the
`totalFileNameLength` bias on folder offsets, the NUL-terminated name block, and
the original-size prefix itself.

Implemented at `src/import/fnv/bsa_archive.cc` — `kBsaVersionSkyrimSe`,
`bsaFolderRecordSize()`, `isSupportedBsaVersion()`, the per-version folder-record
read, and an LZ4 branch in `extract()` that sniffs the **payload magic** rather
than the version, so a v105 archive holding a zlib entry still decodes.

Result, measured:

```
odai_newvegas_probe <SSE Data> --archives
  Archives: 172918 files indexed, 0 failure(s).     # was: 0 indexed, 23 failures
odai_newvegas_probe <FNV Data> --archives
  Archives: 182177 files indexed, 0 failure(s).     # unchanged
```

### The LZ4 decoder is vendored, and why

`src/import/fnv/lz4_frame.{h,cc}` is decompression only, about 120 lines,
bounds-checked throughout. A vcpkg dependency was the alternative and is one
manifest line plus link lines — the deciding argument was not size but
verification: the exact algorithm was run over 3000 retail SSE meshes in a
scratch script *before* it was written in C++, and every one produced a
well-formed NIF header. What is vendored has already been checked against the
only data it will ever read. Take the dependency the moment anything needs to
**write** LZ4, or needs dictionaries or the streaming API.

Pinned by `testLz4FrameDecoding` on a hand-built frame rather than a round trip
— a decoder checked only against its own encoder agrees with itself and can
still be wrong about the format. The fixture deliberately includes a match that
**overlaps its own destination** (offset 3, length 9, against 3 bytes of
output), because that is the case a `memcpy` gets wrong and a byte-at-a-time
copy gets right, and it is how the format encodes any repeating run.

---

## 3. Skyrim SE — the one real blocker: `BSTriShape`

Block-type census over 3000 meshes (18862 `.nif` entries in `Skyrim - Meshes0.bsa`):

| block | instances | files containing it |
|---|---|---|
| `NiNode` | 15478 | — (handled) |
| **`BSTriShape`** | **9595** | **2696 / 3000** |
| **`BSLightingShaderProperty`** | **9074** | **2909 / 3000** |
| `BSShaderTextureSet` | 7635 | — (handled) |
| `bhkCollisionObject` | 2199 | 1865 |
| `bhkCompressedMeshShape(Data)` | 1575 each | 1531 |
| `NiSkinData` / `NiSkinPartition` | 1404 each | 836 |
| `BSDismemberSkinInstance` | 1281 | 769 |

**Skyrim keeps the block-size table, which changes everything about the shape of
this work.** Oblivion's coverage table in `docs/OBLIVION_IMPORT.md` is dominated
by Havok blocks that had to be *sized exactly* because there was no other way to
find the next block; ~21 readers were needed before geometry was reachable at
all. Skyrim has sizes, so **every unknown block is free to skip**. `bhk*`,
`Ni*Interpolator`, `Ni*Controller` and the rest never need a reader.

What is actually required for static geometry is two block types:

- **`BSTriShape`** — Skyrim merged `NiTriShape` and `NiTriShapeData` into one
  block, and packed the vertex data: half-float positions and UVs, byte-packed
  normals and tangents, in a per-file vertex descriptor that says which fields
  are present and how wide a vertex is. This is not a rename; the vertex format
  is genuinely different from `ImportedMeshVertex`'s inputs and has to be
  unpacked. `BSDynamicTriShape` (284 files) and `BSSubIndexTriShape` are the
  same block with a tail.
- **`BSLightingShaderProperty`** — replaces `BSShaderPPLightingProperty` as the
  thing that points at a `BSShaderTextureSet`. The texture set itself is
  **already handled** (`nif_scene.cc`), so this is the pointer, not the payload.

Everything downstream — `CellSceneBuilder`, `ImportedScene`, the render path,
DDS — is already generation-independent and needs nothing.

### Skinned meshes and animation

`NiSkinPartition` + `BSDismemberSkinInstance` cover skinned geometry, and are a
second, smaller step after `BSTriShape`.

**Animation is the wall, and it is a hard one.** `Skyrim - Animations.bsa` holds
**7699 `.hkx` files and no `.kf` at all**. Skyrim replaced Gamebryo's `.kf`
animation with Havok's binary `.hkx`, so `src/import/fnv/kf_animation.{h,cc}` —
including all the B-spline decoding work — does not apply. An `.hkx` reader is a
research project of its own, not a version branch.

So the reachable goal is **Skyrim creatures standing in Cyrodiil in bind pose**,
not moving. That is worth saying out loud before anyone plans around the
cheerful version.

---

## 4. Morrowind — a different engine, not an older version

Morrowind predates the TES4 container entirely. Three separate formats differ.

### The BSA is not a BSA

```
Morrowind.bsa first 16 bytes:
  00 01 00 00  b8 7c 07 00  52 2b 00 00  84 18 00 00
  version=0x00000100  hashOffset=490680  fileCount=11090
```

There is no `BSA\0` magic — the file opens with the literal value `0x100`. The
whole container is different: a file-size/offset table, a name-offset table, a
name block and a hash table, then data. No folders, no per-file compression, no
archive flags. `BsaArchive` cannot read it with a version branch; it needs a
separate reader behind the same interface.

Current behaviour is at least honest: `Not a BSA archive (bad magic)`, 3 of 3
archives.

### The plugin is TES3, and has no formIDs

```
Morrowind.esm first 32 bytes:
  54 45 53 33  34 01 00 00  00 00 00 00  00 00 00 00   |TES34...........|
  48 45 44 52  2c 01 00 00  9a 99 99 3f  01 00 00 00   |HEDR,......?....|
  type='TES3' dataSize=308 unknown=0 flags=0x0  -> HEDR at offset 16
```

- The record header is **16 bytes**: type, dataSize, unknown, flags. No formID.
- Records are keyed by **string NAME**, not by a 32-bit formID. The entire
  `remapFormId` / mod-index machinery in `plugin_load_order.cc` has no meaning
  here.
- There is **no GRUP tree**. Records are a flat list, so the whole group stack in
  `walkRange` is unused.

`detectEsmPluginFormat` currently tests for `"TES4"` and falls through to
`kFallout3` for anything else, so Morrowind is silently routed to the 24-byte
path and dies as `Malformed subrecords in record: TES3`. A third enum value is
the smallest correct change, but the sizes are not the only difference — a
`kMorrowind` format also has to suppress group handling and give the record
layer a different identity key.

### The meshes are NetImmerse 4.0.0.2

Not measured here (the archives do not open yet), but Morrowind ships
NetImmerse-era NIFs, a generation below the 10.x files
`docs/OBLIVION_IMPORT.md` already recommends staying away from: "The 580
pre-20.0.0.x NIFs ... should stay unsupported." Morrowind is entirely that tier.

---

## 4b. Skyrim SE — sky, sun, lighting, clouds and fog ✅ done

The record layer needed no work; the WTHR/CLMT reader in
`src/import/fnv/weather_records.{h,cc}` did, in five places. The whole of it is
written up in `CLAUDE.md` under "Cloud layers"; the short version, because the
first four are silent and the fifth is not:

| | Fallout / Oblivion | Skyrim |
|---|---|---|
| cloud textures | DNAM/CNAM/ANAM/BNAM | `chr('0'+layer) + "0TX"`, up to 29 |
| which are live | `sky\alpha.dds` placeholder | **NAM1**, a disabled bitfield |
| tints | PNAM, 4 layers | PNAM, **32** layers |
| opacity | none | **JNAM**, 128 floats |
| drift | 2 bytes in DATA | **RNAM/QNAM**, per layer |
| NAM0 | 10 rows | **17** rows (row 12 is Fog Far) |
| FNAM | 6 floats (4 in Oblivion) | **8** floats |
| projection | fisheye dome map | **tiling sheet**, plus a cylindrical horizon band |

**The one that costs an afternoon is the first row crossed with the third.**
Skyrim authors a black daytime tint on exactly the layers NAM1 disables, so
picking textures and tints independently pairs a live texture with a dead
tint and paints the entire sky opaque black over correctly lit ground. It does
not look like a channel mix-up; it looks like a broken shader.
`testSkyrimWeatherCloudLayers` pins the pairing.

Also landed here and not Skyrim-specific: WTHR's **Ambient and Sunlight**
channels now light the ground (hue from the record, bounded gain, the renderer
keeps its own intensity), DATA's **Sun Glare** byte scales the sun's halo, and
the **climate's TNAM** supplies the sunrise and sunset hours the colour slots
interpolate against instead of a hardcoded 6 and 19 — SkyrimClimate's are 7.75
and 18.25, so the defaults sampled the wrong slots for over an hour either side
of both.

---

## 5. Recommended order

1. **Skyrim BSA v105 + LZ4** ✅ done. Unblocks every other Skyrim step and is
   independently verifiable (`--archives`, `--find`, `--texture`).
2. **`BSTriShape` + `BSLightingShaderProperty`** ✅ done. Two readers, and Skyrim
   static meshes render. This was the single highest-value piece of work in this
   document: it takes `--nifs` from 0 shapes to most of 22091 files.
3. **Skyrim textures.** Unmeasured. SSE uses BC7 widely, and `src/import/dds.cc`
   rejects a zero fourCC today (the DX10 header extension), so expect this to
   need work — check it before assuming step 2 produces a textured mesh.
4. **Skyrim skinned meshes** (`NiSkinPartition`, `BSDismemberSkinInstance`),
   which gets creatures in bind pose. Stop here unless `.hkx` is worth its own
   project.
5. **Morrowind** — only if it is wanted for its own sake. It is three new
   readers (archive, record container, mesh) sharing almost nothing with the
   other four games, and it is the one generation where "additive version
   branch" is the wrong shape.

### About "Skyrim monsters in Cyrodiil"

Steps 1, 2 and 4 get a Skyrim creature's geometry into an `ImportedScene`, and
`ImportedScene` has no idea which game anything came from — so placing one in a
streamed Oblivion cell is not a further format problem, it is a placement API.
The honest caveats are that it will stand in bind pose until `.hkx` is read, and
that its scale and skeleton conventions are Skyrim's, so it will not match
Oblivion's ground or animations without a deliberate mapping.

---

## 6. Reproducing the measurements

```bash
SSE="$HOME/.steam/steam/steamapps/common/Skyrim Special Edition/Data"
MW="$HOME/.steam/steam/steamapps/common/Morrowind/Data Files"

cmake-build-app/odai_newvegas_probe "$SSE" --archives          # 172918 files, 0 failures
cmake-build-app/odai_newvegas_probe "$SSE" --plugin Skyrim.esm # 869688 records
cmake-build-app/odai_newvegas_probe "$SSE" --nifs 4000         # 0 failures, 0 shapes
cmake-build-app/odai_newvegas_probe "$MW"  --archives          # 3 failures, bad magic
cmake-build-app/odai_newvegas_probe "$MW"  --plugin Morrowind.esm  # Malformed TES3
ctest --test-dir cmake-build-linux --output-on-failure
```

No Vulkan is required for any of it.
