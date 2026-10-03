# HARNESS-007: Bethesda Probe Artifact and Record Validation

Status: Implemented

## Goal

Use `odai_bethesda_probe` to assert that Bethesda content is resolved and parsed
correctly through the production importers, starting with TES3. A successful
file open or a printed inspection is insufficient: expected typed values,
identities, provenance, and decoded artifact properties produce an automated
verdict. Parsing evidence and gameplay evidence remain separately identifiable.

## Dependencies

- TES3 ESM/ESP reader, load-order resolution, text decoding, and `Tes3ContentStore`
- Profile-aware asset resolution, Morrowind BSA extraction, NIF, DDS, and animation importers
- `BethesdaSession` and the dialogue/script/journal runtime for behavioral traces
- Existing synthetic import, content, script, runtime, and save/load tests

## Required Behavior

### HARNESS-007-A: Record validation

The probe loads the same immutable content profile and importers as `odai`.
Versioned expectations assert typed GLOB, NPC_/CREA, FACT, DIAL/INFO, SCPT,
SPEL, item, generic named-record, and CELL/FRMR reference properties.
Assertions cover integer conversion, text encoding, actor stats and inventory,
faction requirements and reactions, dialogue links and condition operands,
script declarations and bytecode lengths, spell effects, and item metadata.

Load-order checks cover case-insensitive identity, later overrides, deletion,
master-relative FRMR ownership, and winning-record source. Missing records may
be explicitly asserted absent; an unresolved required record fails.
Truncated record/subrecord streams fail with a diagnostic.

### HARNESS-007-B: Artifact validation

Checks resolve required assets through the profile's real archive/loose-file
precedence, then invoke production decoders. Morrowind BSA checks validate
archive tables and entry bounds. Asset checks expose the winning provider and
archive and assert NIF version/block types/geometry counts, DDS dimensions,
mips/layers/decoded bytes, or animation clip names/durations/track counts.
Resolution and decode failures always fail, even if an expectation omits
`parsed`. Unsupported interpolation counts remain visible and can be asserted.

Synthetic fixtures contain known triangle geometry, a compressed texture,
direct TES3 keyframes, a linked KF animation stream, and a Morrowind archive.
Malformed fixtures must fail.
No window or GPU is required to execute these checks.

### HARNESS-007-C: Machine-readable assertions

`--tes3-recordcheck <profile> <expectations.json>` accepts version 1 with
1..10000 checks. Every check names `kind`, `id`, and a nonempty `expected`
object. Actor checks may supply `type`; item/generic record and asset checks
require `type`. Reference checks require owner `plugin` and local `frmr`.

Objects match the requested fields recursively; arrays match length and order.
A misspelled expected field fails rather than being ignored. The JSON report
contains profile fingerprint, encoding, counts, expected and actual values,
per-check verdict, mismatch field path, and decode diagnostics. Identical
content and expectations produce identical reports. Invalid contracts,
unsupported check kinds, parsing failures, and mismatches exit nonzero.

Example (synthetic identifiers):

```json
{
  "version": 1,
  "checks": [
    {"kind": "global", "id": "test_global", "expected": {"type": "s", "value": 427}},
    {"kind": "item", "type": "CLOT", "id": "test_robe", "expected": {"worn_value": 200}},
    {"kind": "archive", "id": "test.bsa", "expected": {"parsed": true, "files": 3}},
    {"kind": "asset", "type": "nif", "id": "meshes/test.nif", "expected": {"triangles": 1}}
  ]
}
```

Supported kinds: `global`, `actor`, `faction`, `dialogue`, `script`, `spell`,
`reference`, `record`, `item`, `archive`, `asset`. Asset decoders: `nif`, `dds`,
`animation`. Artifact reports summarize content; they do not embed game bytes.

### HARNESS-007-D: Scoped behavioral evidence

Existing script, dialogue, journal, route-script, and route-checkpoint probe
commands remain available. `--tes3-dialogue-replay <profile> <intents.json>`
shares one session across authored interactions, verifies expected INFO/journal
checkpoints, and supports save/reload checks. Its report explicitly identifies
headless placement at authored locations; it does not certify navigation or UI
input. It offers no externally supplied quest stage, global, INFO selection,
or item-grant action.

A parsing pass, script compilation pass, fresh greeting, or synthetic quest
transition must never be presented as a complete retail gameplay route.
Mechanics capabilities retain their own acceptance criteria and status.

## Verification

1. CTest runs generated, game-data-free master/patch and BSA/NIF/DDS/animation
   fixtures through `odai_bethesda_probe`, asserting the fields above.
2. Repeated runs produce identical reports. Wrong expectations for every check
   category fail with a field path. Invalid contracts and malformed artifacts fail.
3. Existing import, TES3 content/script/runtime/save tests and the full suite pass.
4. Optional local base-game checks assert representative retail records and
   artifacts using a recorded profile fingerprint. Profiles, retail bytes,
   captures, and extracted data stay local and are not required by CI.
5. Behavioral traces report accepted interactions and checkpoint failures with
   their actual scope, without upgrading unrelated capability status.

## Definition of Done

All required behavior and verification pass for the TES3 vertical slice.
The probe invokes production importers, assertions can detect incorrect typed
values and malformed artifacts, and local retail checks are reproducible.
Only then update `docs/PARITY.md` to Implemented. Extending this harness to
other Bethesda games does not require a separate runner.

## Implementation

`--tes3-recordcheck` implements record and artifact expectations with JSON
verdicts. `odai_tes3_probe_records` generates a master, overriding patch,
Morrowind BSA, triangle NIF, BC1 DDS, direct-keyframe NIF, and linked KF stream,
then exercises positive, negative, malformed, and deterministic cases, including
loose-file precedence and rejection of non-finite float globals. The parser checks
Morrowind archive tables, filenames, and entry bounds before publishing entries.
TES3 `NiSequenceStreamHelper` binds linked bone-name extras to keyframe
controllers; tests assert frequency-scaled times and decoded quaternion poses
and reject cyclic/mismatched chains, invalid links, and invalid timing.

### Verification (2026-10-03)

The complete Debug build and all 64 registered CTest targets pass, including
`odai_tes3_probe_records`. Repeated synthetic reports are identical.

Sixteen local base-game record/artifact assertions pass with profile fingerprint
`5c585c66599e8cf14fa4956f5c056307`. Independent raw-header expectations cover
five NPCs and Vivec's CREA record, short globals, Blades faction requirements,
Wraithguard's armor value, and Caius's script declarations. Artifact checks
cover Morrowind.bsa's 11090 entries, Wraithguard's DDS dimensions/mips and ground
mesh, and the Redoran flag's KF stream. Repeated retail reports are identical.
The KF decodes three tracks with a four-second duration.

The 35-intent opening dialogue replay also passes its expected INFO/journal
checkpoints and save/reload checks. Its fixture contains interaction identities
and expectations, with no game bytes or dialogue response text. It stops at
Caius's authored level-3 prerequisite; completing the rest of MECH-TES3-002
requires normal leveling and further route verification. That mechanics
capability remains Partial.

```bash
cmake --build build-linux -j 4
ctest --test-dir build-linux --output-on-failure
./build-linux/odai_bethesda_probe --tes3-recordcheck <local-profile.json> <expectations.json>
./build-linux/odai_bethesda_probe --tes3-dialogue-replay <local-profile.json> tests/fixtures/harness/tes3_mainquest_dialogue_opening.json
```

Retail profiles, expectation manifests, reports, and extracted data remain
local under `/tmp`; CI requires no installed game data. These parsing and
headless interaction results do not certify rendering, navigation, UI input,
or complete quest playthroughs.
