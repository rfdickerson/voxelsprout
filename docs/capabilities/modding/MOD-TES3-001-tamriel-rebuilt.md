# MOD-TES3-001: Tamriel Rebuilt Mainland and Quests

Status: Partial

## Goal

With a locally installed Tamriel Rebuilt release, `odai` can load, explore, and
play **every released region and quest** in `TR_Mainland.esm` through normal
gameplay. The active plugin records, rather than an engine-maintained list of
place names or quest IDs, define the coverage set. New released content enters
scope when the supported content version is updated.

Tamriel Rebuilt describes `TR_Mainland.esm` as its finished, inhabited and
quested landmass; work-in-progress preview content is a separate optional
plugin. Its installation instructions require `Tamriel_Data.esm` and
`TR_Mainland.esm`, with `TR_Factions.esp` as optional faction integration.
The project's [installation guide](https://www.tamriel-rebuilt.org/content/how-install-tamriel-rebuilt),
[completion FAQ](https://www.tamriel-rebuilt.org/about/frequently-asked-questions/completion),
and [quest FAQ](https://www.tamriel-rebuilt.org/about/frequently-asked-questions/quests)
describe that scope. The quest FAQ reports 1,019 quests in release 26.08a;
this count is context, not a hard-coded acceptance target.

## Dependencies

- TES3 content profiles, master dependency resolution, archive and loose asset
  lookup, plugin overrides, and stable record identity across save/load.
- WORLD-001, WORLD-002, and WORLD-003 for mainland exterior and interior travel
  and persistent cell state.
- RENDER-TERRAIN-001/002, RENDER-WATER-001, PHYS-001/002, and the imported-scene
  renderer for playable terrain, objects, water, and collision.
- TES3 dialogue, journal, MWScript, actor, faction, inventory, combat, travel,
  and save/load behavior, including the relevant MECH-TES3-002/003/004/005
  acceptance criteria. MECH-002's base-game main-quest completion is not a
  prerequisite for independent mainland quests.
- HARNESS-001/002/003/004 for deterministic scenario and state verification.

## Required Behavior

### MOD-TES3-001-A: Load the released content

- Resolve the installed release's master chain in declared order, including
  `Morrowind.esm`, `Tribunal.esm`, `Bloodmoon.esm`, `Tamriel_Data.esm`, and
  `TR_Mainland.esm`, and support `TR_Factions.esp` when selected. Missing or
  incompatible required data produces a specific error before play.
- Apply winning TES3 record and reference overrides in load order, including
  changes to base-game content. Resolve Tamriel_Data assets, archives, terrain
  textures, scripts, dialogue, and cell references without mod-specific paths
  or ID exceptions. Keep the one imported-scene rendering path.
- Bind saves to the content profile and preserve persistent references, quest
  progress, and script state across cell transitions and reloads. Detect a
  changed or incompatible plugin set rather than silently applying a save to
  different records.

### MOD-TES3-001-B: Explore every released region

- Enumerate every exterior and interior cell contributed or modified by the
  active `TR_Mainland.esm`, including cells shared with earlier plugins.
  Every reachable released region supports terrain, assets, weather, water,
  collision, actors, doors, and authored travel links as its records require.
- The player can enter the mainland through authored connections, cross its
  exterior cell boundaries, enter and leave its interiors, and return to
  Vvardenfell. Streaming and save/load preserve modified local state.
- No region is accepted solely because a cell renders in a debug capture;
  ordinary movement and interactions must work there.

### MOD-TES3-001-C: Play every released quest

- Build a release-specific inventory from all quest-bearing `DIAL`/`INFO`,
  journal, script, actor, item, and reference records in the active content.
  Account for quests added by `TR_Mainland.esm` and quests it changes in earlier
  plugins. Track each quest's entry, legal branches, completion or failure state,
  and dependencies without assuming that every branch can coexist in one save.
- Every quest has at least one authored route that can start, progress, and
  finish or fail as designed using ordinary player input. Dialogue conditions,
  journal updates, script commands and effects, items, faction ranks, AI,
  encounters, travel, and world changes follow the winning plugin records.
  Unsupported operations fail visibly; they never silently skip a gate or
  grant progress.
- Quest state and one-time rewards survive cell streaming and save/load.
  Repeated interactions do not duplicate rewards, bypass conditions, or make
  another legal route impossible.
- With `TR_Factions.esp` selected, its additional requirements and mainland
  integration follow that plugin's winning records. Its absence must not
  change the `TR_Mainland.esm` only route behavior.

## Verification

1. Add game-data-free fixtures for multi-master TES3 overrides, mainland
   exterior/interior streaming, asset lookup, authored travel, dialogue and
   journal gates, script effects, quest branches, and save/load. Extend the
   existing TR-shaped load-order and content-profile tests rather than adding
   special cases to the runtime.
2. On a locally installed, legally obtained release, record the exact plugin
   filenames, versions or hashes, load order, optional plugins, and generated
   region/quest inventory. Compare the inventory to the winning records and
   report every missing, unsupported, or unverified entry by ID. Do not commit
   or redistribute mod or game files or local captures.
3. Run deterministic traversal checks for **every** released exterior and
   interior cell, including their required transitions and return travel.
   Exercise each distinct asset/record path, and visually review representative
   regions and any automated visual failures.
4. Run a normal-input completion or authored-failure scenario for **every**
   inventoried quest. Use separate saves for mutually exclusive branches; test
   alternate branches where they change required engine behavior. Include
   save/load and repeated-interaction checkpoints. Run both the required-only
   profile and a `TR_Factions.esp` profile.
5. Run relevant CTest targets and the full suite. Record unresolved content
   IDs and blockers. Only update this status and `docs/PARITY.md` after the
   inventory has zero unverified released cells or quests and all required
   checks pass.

## Out of Scope

- Unreleased or preview-only cells and quests in `TR_Preview.esp`.
- Other Project Tamriel mods or unrelated third-party patches.
- Creating substitute quest content, rewriting mod dialogue, or redistributing
  Bethesda or Tamriel Rebuilt assets.

## Definition of Done

MOD-TES3-001 is Implemented when a pinned released content profile loads with
its required assets, every released `TR_Mainland.esm` region is playable, every
inventoried quest has a verified ordinary-input resolution, optional
`TR_Factions.esp` integration behaves as authored, persistence and content
compatibility checks pass, and the complete automated suite passes. A content
update requires a new inventory and verification before claiming support for
the newer release. If any released cell or quest remains blocked, retain
Partial and list the missing IDs and behavior.

## Current implementation

Synthetic tests exercise a TR-shaped TES3 master chain, cell/terrain overrides,
and OpenMW content-profile parsing. Installed TR 25.08.12 / Tamriel Data 25.05
profiles with and without TR_Factions.esp have now been pinned and structurally
audited. Profile A includes 2,572 exterior cells, 3,398 interior cells, and 1,410
scoped journal records; Profile B includes 2,572 exterior cells, 3,400 interior
cells, and 1,429 scoped journal records. Both retain three unresolved lexical
quest candidates and all unclassified quest-bearing records. The independent
active-journal inventory matches the runtime in both profiles.

Every cell and quest remains UNVERIFIED. Both strict script checks fail (43
compiler diagnostics and 171 unsupported operation/effect labels). Complete
quest/branch classification, winning moved-reference linkage, ordinary-input
cell/quest scenarios, causal gameplay evidence, process-restart persistence,
actual gameplay-validator fault injection, and optional-profile behavioral
verification remain required. Structural probe success is not gameplay evidence.

The full registered CTest suite passes 63/63, including the new game-data-free
audit/evidence-schema tests; the synthetic Vulkan integration smoke also passes.
No runtime engine defect was fixed or waived. A generic quoted-message keyword
bug in the new audit scanner has a synthetic regression. The acceptance contract
above is unchanged.

See [the certification report](../../validation/MOD-TES3-001.md) for content
hashes, inventory limitations, test results, and exact reproduction commands.
Local ignored `captures/tr-certification/certification.json` expands every
non-verified ID and remaining blocker for both profiles. Plugin assets, authored
script/dialogue source, and captures are not committed or redistributed.
