# MOD-TES3-001 local certification: Partial

The installed release is Tamriel Rebuilt 25.08.12 with Tamriel Data 25.05.
No installed-content cell or quest is certified. Profile resolution and static
script/journal checks have been executed; ordinary-input traversal, quest
resolution, causal gameplay traces, and process-restart persistence have not.
The acceptance contract and Definition of Done are unchanged.

## Content identity

Profile A loads, in order, the first five plugins below. Profile B loads all six.
The engine revision is `6401ce1d84f16943c6ef40618e3d64f378668593` with the
existing dirty working tree. Each local `content-identity.json` records the
working-tree diff digest, audit script digest, executing probe SHA-256, absolute
plugin paths, plugin SHA-256, resource roots, profile hash, and archive hashes.
`harness-identity.json` pins all three audit/controller/test source files.
These executable and source hashes distinguish this run from an unmodified HEAD.

| Filename | SHA-256 |
| --- | --- |
| Morrowind.esm | 5c3c8c2cbd20e25901b59b3ece33d36b7ef0e3d60ad8d11828bcc61a5ead1647 |
| Tribunal.esm | 2ace511f23cc2a9ddd5f3aa59c7919789b9378cf4b17c8ae3375dd6b782f3f2b |
| Bloodmoon.esm | bd27090d0e6ad4c1bf1abc83f1a2dac56fcc82cae7bfe8263c413fb301801357 |
| Tamriel_Data.esm | 10ff97449f55db61b065979c57ba0de8adc3ebde852ae80c462dfcee917d9df0 |
| TR_Mainland.esm | 7ef0d15aedd854e4a9451a985b54d6a7e19e98d6d7e0d0f91072b52e75aeb89d |
| TR_Factions.esp | 33b95a68a0943a35d5ef21239ee920812eff16ca38c2b9a4899b01d2a093f4bc |

Resource layers, ascending priority:

1. `/home/rfdickerson/.local/share/Steam/steamapps/common/Morrowind/Data Files`
2. `/home/rfdickerson/.local/share/odai/morrowind/Tamriel_Data_25.05/00 Data Files`
3. `/home/rfdickerson/.local/share/odai/morrowind/Tamriel_Rebuilt_25.08.12/00 Core`
4. Profile B only: `/home/rfdickerson/.local/share/odai/morrowind/Tamriel_Rebuilt_25.08.12/01 Faction Integration`

Required archives are `Morrowind.bsa`, `Tribunal.bsa`, and `Bloodmoon.bsa` in
layer 1. The installed mod resource roots use loose files. Required-asset
resolution has not been verified by traversing every cell. Unrelated local mods,
shader packs, and preview plugins are excluded from these explicitly requested
profiles. Each probe creates a fresh context; no Profile B progression state is
used for Profile A.

## Inventory and verification

The independent binary scanner uses active records and load order, not a list of
TR content IDs. Exterior CELL identity uses coordinates even when NAME changes;
interiors use names; INFO identity includes its topic; LTEX identity retains
plugin-local palette provenance. Winning records, deletions, record hashes,
source offsets, earlier contributions, and malformed/unclassified records remain
visible. The embedded reference ledger includes earlier CELL contributions;
it does **not** claim to resolve winning moved-reference semantics.

| Coverage | Profile A | Profile B |
| --- | ---: | ---: |
| Exterior CELL records contributed/modified by scoped plugins | 2,572 | 2,572 |
| Interior CELL records contributed/modified by scoped plugins | 3,398 | 3,400 |
| Scoped journal records | 1,410 | 1,429 |
| Additional unresolved lexical quest candidates | 3 | 3 |
| All active journal records, including earlier masters | 2,083 | 2,090 |
| VERIFIED cells / quests | 0 / 0 | 0 / 0 |
| BLOCKED cells / quests from executed gameplay scenarios | 0 / 0 | 0 / 0 |
| UNVERIFIED cells / journal-and-candidate entries | 5,970 / 1,413 | 5,972 / 1,432 |

Both independent journal sets exactly match the engine's journal enumeration.
This does not establish a complete authored quest/branch inventory: quests
without journals, script attachments, transitive dependencies, branches,
completion routes, and winning reference/transition linkage need classification.
Every scoped script, INFO, actor, and CELL remains in the unclassified
quest-bearing ledger. Unresolved lexical targets `check`, `once`, and `state`
remain visible as candidates rather than being silently discarded or counted as
confirmed authored quests. The counts above are record inventory counts, not an
assertion that every journal corresponds to a separately playable quest.

Profile B adds 171 named records and changes 75 winning records; all those
changes have `TR_Factions.esp` provenance. It adds 19 scoped journal entries and
2 scoped interior cells. No unexpected record winner changes were detected.
Behavioral equivalence/integration remains UNVERIFIED.

## Exact local machine-readable evidence

Generated evidence stays in ignored `captures/tr-certification/`:

- `A/certification.json` and `B/certification.json`: every non-verified cell and
  quest/candidate ID, subsystem blockers, affected IDs, reproduction commands,
  observed/expected behavior, and probe results.
- `A/inventory.json` and `B/inventory.json`: complete binary-record occurrence
  audit, winning named records, cell/quest candidates, embedded reference
  contribution ledger, actor travel destinations, dialogue conditions/ordering
  links, script command candidates, tombstones, unclassified entries, and
  explicit classifier limitations. `inventory_complete` is false.
- Per-profile `content-identity.json`, `winning-record-manifest.json`,
  `journal-reconciliation.json`, `probe-results.json`, probe stdout/stderr.
- `optional-profile-delta.json`: every added/changed/removed record ID and quest
  candidate delta; `behavior_verified` is false.
- `regression-results.json`, `ctest.log`, `vulkan-smoke.log`,
  `harness-identity.json`, and `audit-quote-fix-before-after.json`.

These files contain local metadata, not redistributed mod/plugin assets or
captures. The scanner never exports authored dialogue responses or script source.
The raw occurrence audit retains even records whose identity cannot be classified,
using plugin and byte offset as their reproducible identifier.

## Remaining blockers

1. **MWScript:** both strict checks exit 1, with 43 compiler diagnostics and 171
   unsupported operation/effect labels. The exact scripts, lines, diagnostics,
   unsupported operations, and command-use counts are in the probe output.
   These are profile-wide static failures; their individual quest impact is
   unestablished, and no gameplay reproduction or runtime fix is claimed.
2. **Inventory/validator:** classify authored quest branches, entry/prerequisite
   chains, actor/item/script/reference dependencies, non-journal quests, and
   winning moved references. Resolve every unclassified record/candidate. Static
   lexical candidates are not a complete MWScript behavioral classifier.
3. **Cell playability:** generate and execute every authored ordinary-input route,
   including movement, terrain/water, collision, required assets, actors, doors,
   exits, scripted transitions, travel/return, streaming state, and save/reload.
   No cell has executed scenario evidence for these checks.
4. **Quest gameplay:** generate and execute ordinary-input successful/authored
   failure routes with significant gates and distinct legal branches, inspecting
   dialogue/INFO, journal, scripts, inventory, factions, actors, references,
   combat, travel, and one-time rewards. No quest route has executed evidence.
5. **Process persistence:** compare uninterrupted execution with save, process
   termination, reload, and identical subsequent inputs at irreversible points.
   No installed-profile restart comparison exists.
6. **Validator faults:** the new evidence-schema tests reject omissions,
   prerequisite flags indicating bypass, duplicate reward counts, lost persistent
   variables, wrong references, missing transitions, skipped/crashed/timed-out
   scenarios, direct-mutation inputs, and in-memory-only reloads. They are schema
   guards. Actual runtime fault injection through a complete installed-content
   gameplay runner remains required; trusted Boolean claims cannot prove gameplay.
7. **Optional profile:** record deltas are verified structurally; execute
   `TR_Factions.esp` gates/effects and compare gameplay against isolated Profile A.

All affected IDs are expanded in the machine reports. The conservative
profile-wide script-impact lists are explicitly labeled; they do not claim that
every quest executes every unsupported command. Skipped or unavailable execution
remains UNVERIFIED, with no expected-failure workaround.

## Engine work and regression results

No runtime engine defect was fixed or waived in this work. A generic audit defect
that interpreted Journal keywords inside quoted message strings as quest
operations was reproduced and fixed. A game-data-free scanner regression also
covers semicolons inside strings, comments, quoted journal operands, and journal
queries. The override fixture covers named exterior cells sharing coordinates,
interiors, earlier-master quest overrides, optional winning INFO records,
reference deletion scope, unresolved journal targets, and unknown record types.

- `python3 tests/tes3_release_audit_tests.py`: 13 tests pass.
- `cmake --build build-linux -j 4`: passes. CMake regeneration required access to
  the installed vcpkg toolchain outside the sandbox; no dependencies changed.
- `ctest --test-dir build-linux --output-on-failure -j 4`: 63/63 pass, including
  TES3 content/script/runtime/navigation, import/profile, save, integration,
  headless, and harness tests.
- `./build-linux/odai_imported_scene_vulkan_smoke`: exits 0. This executes the
  synthetic Vulkan integration fixtures and gives no TR playability credit.
- Both `--profilecheck` runs exit 0; both strict `--tes3-scriptcheck` runs exit 1;
  both `--tes3-quest-suite` runs exit 0 while explicitly reporting
  `transition_explorer_complete=false` and `release_gate_passed=false`.

## Reproduction

From the repository root:

```bash
cmake --build build-linux -j 4
python3 scripts/tr_certify.py \
  --base-root '/home/rfdickerson/.local/share/Steam/steamapps/common/Morrowind/Data Files' \
  --data-root '/home/rfdickerson/.local/share/odai/morrowind/Tamriel_Data_25.05/00 Data Files' \
  --mainland-root '/home/rfdickerson/.local/share/odai/morrowind/Tamriel_Rebuilt_25.08.12/00 Core' \
  --factions-root '/home/rfdickerson/.local/share/odai/morrowind/Tamriel_Rebuilt_25.08.12/01 Faction Integration'
python3 tests/tes3_release_audit_tests.py
ctest --test-dir build-linux --output-on-failure -j 4
./build-linux/odai_imported_scene_vulkan_smoke
```

`tr_certify.py` exits **1** when it produces the expected Partial certification;
**2** means a fatal/incompatible audit or missing required file. No zero exit is
provided for incomplete content certification. Per-profile probe commands and
stdout/stderr are recorded in each `probe-results.json`. To rerun one profile:

```bash
python3 scripts/tes3_release_audit.py \
  --profile captures/tr-certification/profile-A.json \
  --scope-plugin TR_Mainland.esm --output captures/tr-certification/A
python3 scripts/tes3_release_audit.py \
  --profile captures/tr-certification/profile-B.json \
  --scope-plugin TR_Mainland.esm --scope-plugin TR_Factions.esp \
  --output captures/tr-certification/B
```

These regenerate structural inventories and blocker evidence. Commands to execute
complete normal-gameplay coverage do not exist yet; that absence is a blocker,
not an invitation to substitute teleportation or state injection.
