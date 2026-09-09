# Facial morphs — first implementation slice

Updated 2026-09-09. This starts the deferred facial-animation milestone with a
TRI reader, source-order CPU expression evaluation, and an asset probe. **Actors
do not yet blink, emote or lip-sync through this code.** No rendering path or
cooked-scene layout changed, and the running JK/SMIM city preview is unaffected.

## Implemented

`src/import/fnv/tri_morph.{h,cc}` reads FRTRI003 using the
[NifTools format definition](https://raw.githubusercontent.com/niftools/pyffi/develop/pyffi/formats/tri/tri.xml).
It preserves source vertices, triangle/quad topology, UVs, named signed-short
relative targets and scales, absolute modifier replacements, and reserved header
values. Unsupported signatures, malformed data and allocation limits have separate
statuses. Bounds, indices, finite values, string termination and payload lengths
are checked before consumption; failure clears partial output.

Retail files can contain duplicate names. These are retained by ordinal and
flagged as ambiguous, rather than discarded or silently resolved by name.

`evaluateTriMorphs` evaluates explicit indexed weights against the caller's neutral
head in original vertex order. It does not replace that head with the TRI reference
shape. Re-evaluation starts from neutral, avoiding accumulated deformation drift.
Absolute character-creation modifiers are preserved separately and are not applied
as relative expressions. Evaluation validates weights and catches output overflow.

The probe uses the normal asset resolver, including active mod overrides:

```sh
build-linux-relwithdebinfo/odai_bethesda_probe --tri-profile \
  captures/jk-skyrim-showcase/profile.json --all \
  > captures/facial-morphs/tri-audit.json

build-linux-relwithdebinfo/odai_bethesda_probe '<Skyrim Data>' \
  --tri 'meshes/actors/character/character assets/malehead.tri'
```

Its JSON reports provider identity, content fingerprint, source topology counts,
names/scales, maximum displacement and ambiguity. Runtime consumption is explicitly
false; decoding alone is not proof of visible facial animation.

## Validation and local evidence

- Optimized runtime and probe build passed; **40/40 CTest tests pass**.
- Synthetic coverage includes signed deltas, multiple expressions, a distinct
  actor neutral shape, neutral reset, separate absolute replacements, UV topology,
  every truncation boundary, invalid indices/counts/names/weights, duplicate names,
  missing assets and a winning case-insensitive loose-mod override.
- The current JK/SMIM profile resolves **569 unique TRI assets; all 569 decode**,
  preserving **15,785 relative targets**. This count includes chargen/race assets,
  not just expression targets or visible actors. Four assets have ambiguous names.
- The ordinary male head has 898 vertices and 44 targets; the female head has 996
  vertices and 44 targets. Original files and the detailed audit remain local at
  `captures/facial-morphs/tri-audit.json`.

## Next implementation slice

1. Resolve authored head-part expression/chargen links through winning actor and
   head-part records. Distinguish expression data from chargen/race morph families.
2. Bind TRI topology to the correct NIF shape and original vertex ordering. A
   matching count alone is insufficient. Preserve the shape-to-merged-character
   vertex mapping and the per-shape FaceGen bind-space conversion.
3. Apply expression deltas before skeletal deformation through the existing
   skinned-character path; update normals/tangents and previous-frame geometry
   consistently. Add serialization/residency/reset coverage when that interface
   becomes persistent or GPU-owned.
4. Add an actor close-up with controlled blink/expression weights, then wire
   dialogue timing and phonemes. FUZ/LIP decoding and speech synchronization remain
   separate, unfinished work. TRIP/BodySlide body morph formats remain unsupported.

No retail parity claim, GPU facial-animation validation, or speech playback claim
is made for this reader-only milestone.
