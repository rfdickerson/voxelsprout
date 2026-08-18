# Skyrim Special Edition and Morrowind import status

Both games run through the same `odai` executable and retained
`ImportedScene` Vulkan path. The runtime reads original archives and plugins;
the repository distributes no Bethesda assets.

## Skyrim Special Edition

The streaming path supports SSE BSA v105/LZ4 archives, TES5 records, localized
strings, BSTriShape materials, BC textures, terrain and object LOD, weather and
multi-layer clouds, static and skinned geometry, GPU skinning, TES5 NAVM
prefixes, and authored static Havok collision. Collision extraction follows
fixed/keyframed compressed meshes, MOPP and transform/list wrappers, packed
triangles, boxes, and convex shapes; unsupported or absent authored shapes fall
back per NIF instance to opaque visible geometry.

Skyrim load order selection is, in precedence order:

1. `--load-order <plugins.txt>`;
2. `ODAI_FNV_LOAD_ORDER`;
3. an auto-discovered native or Proton Skyrim profile;
4. the official installed content, preserving `Skyrim.ccc` order.

Active `*` entries are authoritative, masters resolve recursively, and
repeatable `--plugin-add` entries are appended last. Missing active plugins are
fatal. Regular plugins and ESL light plugins occupy separate form-ID slot
spaces, and their ordered names, kinds, sizes, and timestamps form the cache
and traversal-state fingerprint.

```bash
SSE="/path/to/Skyrim Special Edition/Data"
./build-linux/odai --stream "$SSE" --plugin Skyrim.esm
./build-linux/odai_bethesda_probe "$SSE" --loadorder
./build-linux/odai_bethesda_probe "$SSE" --doorcheck
./build-linux/odai_bethesda_probe "$SSE" --routecheck
```

`--routecheck` resolves the real Tamriel → WhiterunWorld → Bannered Mare →
WhiterunWorld → Tamriel door chain, builds each destination cell, and verifies
that render geometry and authored/fallback collision are present. It is the
non-interactive installed-data regression check for the playable route.

Real XTEL doors connect streamed exterior cells, walled child worldspaces, and
interiors with a fade-and-swap transition. Locked doors are labelled and may be
bypassed in exploration mode. Named XMRK markers are discovered within 4096
Bethesda units and appear on the compass and pause-menu list. A native JSON
state resumes the last exterior/interior identity and camera pose; it is not a
Skyrim `.ess` file.

Quests, Papyrus, combat, lockpicking, HKX animation playback, dynamic physics,
fast travel, scripted transports, and cross-cell actor pathfinding remain out
of scope.

## Morrowind

TES3 cells, terrain, VTEX terrain layers, statics, water, actors, and authored
tours use the same runtime. Morrowind has neither TES4/5 XTEL worldspace
semantics nor a Skyrim plugin profile, so launch it with an explicit master and
worldspace:

```bash
./build-linux/odai --stream "/path/to/Morrowind/Data Files" \
  --plugin Morrowind.esm --worldspace Vvardenfell
```

Morrowind terrain has one VTEX texture sample per 512-unit block rather than
TES4/5 per-vertex ATXT/VTXT weights. The importer synthesizes the retained
four-layer blend from neighbouring block-centre samples to avoid hard square
boundaries.

## Cache compatibility

Imported scenes use format 29 and streamed cells use build version 46. The
door destination and collision payloads changed their serialized layout, so
older cooked scenes are rejected clearly and old cell caches rebuild; there is
no legacy expansion path.
