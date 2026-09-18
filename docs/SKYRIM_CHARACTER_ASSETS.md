# Skyrim character asset discovery

The profile-wide character probe inventories winning actor, race, head-part,
armor, armor-addon and outfit records. Head-part bindings retain race/expression/
chargen purpose and armor-addon bindings retain first/third-person perspective,
following the [xEdit TES5 record definitions](https://github.com/TES5Edit/TES5Edit/blob/dev-4.1.6/Core/wbDefinitionsTES5.pas). It excludes deleted record winners and
retains remapped record references and direct model/morph dependencies. FaceGen
files found in the inventory are associated with their defining actor where
possible; missing optional FaceGen is not counted as a required-file failure.

```sh
build-linux-relwithdebinfo/odai_bethesda_probe --character-coverage \
  captures/jk-skyrim-showcase/profile.json \
  captures/character-coverage/combined-profile.json
```

The command writes JSON and adjacent Markdown. Profile overrides, archives and
loose files use the runtime's virtual Data resolver. The inventory includes all
HKX/TRI files, actor meshes, armor/clothing meshes, native animation packs and
additional record dependencies. This includes creature and first-person bundles;
animated-object HKX files are also inventoried, without implying character use.

Dependency discovery follows character catalogs, graph references and native
pack variants, blend samples, layers and pose-graph clips. It runs independently
of graph execution admission, deduplicates canonical paths, detects cycles and
limits expansion to 100,000 files and depth 64. Nested bundles and parent-relative
references resolve within virtual Data; escaping its root is rejected. Catalog
aliases are not treated as virtual filenames.

Each asset reports winning provider, physical/archive provenance, fingerprint,
references, decoding status and separate binding/runtime evidence. NIF reporting
includes static/skinned shape counts and resolved texture dependencies. Texture
bytes are resolved but texture codecs are not assessed by this probe. TRI uses
its typed unsupported/malformed/resource-limit diagnostics. Boolean NIF/HKX
failures remain unclassified rather than being mislabeled unsupported or corrupt.
Clip binding is measured against the character definition's animation skeleton.
Inventory-only clips may be checked against a unique bundle skeleton, labeled as
an inferred candidate. Neither check proves binding to assembled actor geometry.
Unreferenced files and unused dependency edges are not visible failures.

## Runtime integration

The probe and native/recovery runtime loaders share source clip resolution and
binding. Source tracks and annotations remain untouched by playback policy;
looping and controller jump adjustments operate on independent playback copies.
Native rules sharing a file may use different looping and root policies without
contaminating each other. Ordinary gameplay still loads only required clips and
keeps existing gameplay state mappings. Native animation remains the default.

The native loader fingerprint is revised so incompatible old playback selections
are discarded by the existing restore checks. No save, cooked-scene, streamed
chunk or GPU layout changes are introduced.

HKX class inventory now incorporates the exact classes identified by virtual
fixups. This prevents binary signature bytes preceding a class name from hiding
valid animation/skeleton classes in the heuristic text scan.

## Limits

The report does not establish GPU consumption, retail visual parity, facial
animation, creature behavior or FNIS graph execution. Unsupported graph classes
remain listed; successful graph decoding does not imply execution admission.
Full transitive gameplay record semantics and actual render-rig binding for every
actor remain separate work. Proprietary assets and reports stay local.

## Verification

Synthetic tests cover overrides/tombstones, nested and first-person paths,
bounded parent references, missing/malformed assets, TRI format classification,
non-executable graph catalogs, cycle/resource limits, fixup-backed class names,
and independent source/playback state. Existing animation tests retain
no-index-substitution, save/restore and stream-eviction coverage.

### Local combined-profile result (2026-09-17)

Debug runtime/tools build and all **51 CTest tests pass**. RelWithDebInfo runtime
and probe builds pass. The completed report is local at
`captures/character-coverage/combined-profile.{json,md}` and includes 13,018 winning
records, one excluded deleted winner, and 15,645 unique asset paths:

| Outcome | Assets |
|---|---:|
| Decoded by the measured reader/binding path | 14,570 |
| Inspected, detailed decode/binding unassessed | 501 |
| Missing dependency | 72 |
| Clip decode or source-binding failure | 347 |
| Other decoder failure, cause unclassified | 155 |

The 6,171 animation clips comprise 5,787 decoded/bound to at least one source
skeleton, 347 decode/binding failures and 37 without a unique source-rig context.
Some clips have multiple source-rig checks; inspect each binding result rather
than interpreting one successful binding as universal compatibility. All 569
resolved TRI files decode; three additional authored TRI references are missing.
There are 3,038 inventory-only files without observed incoming references.

Representative findings include missing `dlc01trailerwalkforward.hkx` and
`idlevalericawalkforward.hkx` catalog entries, missing hair TRI references,
first-person clip scalar-bound decoding failures, and undecoded LOD behavior
variants. Their appearance in a catalog does not establish gameplay use.

Optimized Ralof and Lydia (`HousecarlWhiterun`) studio runs used the unmodified
combined profile, 768×432 logical windows, 1536×864 native-DPI framebuffers and
render scale 1. Both built visible skinned actors and exercised run/jump requests;
logs and inspected captures are under `captures/character-coverage/validation/`.
Both shut down without renderer-owned live images. These are baseline playback
smoke checks, not full movement, retargeting or retail visual acceptance.

The installed DovaJump files decode against source skeletons, but their native
runtime import reports `retarget hierarchy mismatch: NPC Root [Root]` against
the assembled actor rigs. Recovery clips remain in use. This retargeting gap is
explicitly unresolved; the shared loader retains the existing strict hierarchy
policy rather than substituting bone indices or silently claiming mod playback.
