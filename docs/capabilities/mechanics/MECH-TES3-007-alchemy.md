# MECH-TES3-007: Morrowind Alchemy

Status: Planned

## Goal

The player can use owned alchemy apparatus and ingredients to create named
potions through normal Morrowind inventory input, eat ingredients, and drink
the resulting potions. Outcomes follow TES3 rules and remain consistent across
inventory, active effects, skill advancement, cell transitions, and save/load.

## Behavioral References

- [UESP: Morrowind Alchemy](https://en.uesp.net/wiki/Morrowind:Alchemy)
- Read-only local OpenMW reference: `references/openmw/apps/openmw/mwmechanics/alchemy.cpp`,
  `references/openmw/apps/openmw/mwclass/ingredient.cpp`, and ingredient/potion
  handling in `references/openmw/apps/openmw/mwmechanics/spellcasting.cpp`.

Use UESP for the player-facing contract and OpenMW to investigate calculation
and state-transition details. Resolve disagreements with reproducible local
retail TES3 checks, especially apparatus rounding, random-roll boundaries,
effect visibility, and ingredient consumption. Record the verified expectations;
do not silently adopt OpenMW extensions or copy its architecture.

## Dependencies

- MECH-001: authoritative player inventory, quantities, and world-item lifecycle
- MECH-TES3-001: persistent player attributes, skills, and character class
- MECH-TES3-006: earned Alchemy skill-use progress and leveling integration
- TES3 INGR, APPA, ALCH, MGEF, SKIL, and GMST content with load-order overrides
- Ingredient and potion use, active magic effects, and gameplay time advancement
- Versioned gameplay saves and generated-item identity
- HARNESS-002, HARNESS-003, and HARNESS-006 for deterministic state and UI checks

Missing active-effect or generated-item support is implementation work required
by this capability. Inventory display or progression arithmetic alone does not
establish alchemy parity.

## Required Behavior

### Ingredients and effect knowledge

- Import each ingredient's ordered four effect slots, including absent slots,
  target attribute or skill, weight, value, name, model, and icon. Import
  apparatus type and quality and the effect metadata needed for calculations.
  Authored overrides and expansion/mod ingredients use the same rules.
- Ingredient effect visibility follows modified Alchemy skill and
  `fWortChanceValue`: vanilla thresholds are 15, 30, 45, and 60 for the first
  through fourth slots. Below 15 no effects are identified. Hidden effects remain
  unknown in the UI; selecting or eating an ingredient does not introduce a
  Skyrim-style permanent discovery ledger.
- Brewing accepts two to four distinct ingredient records. A stack cannot
  occupy multiple slots to match with itself. Effects shared by at least two
  selected ingredients are candidates, including hidden effects; visibility
  does not filter the underlying recipe. Match the effect and its attribute or
  skill argument, not merely the effect ID. More than two matching ingredients
  do not create duplicate copies of an effect.
- Preserve TES3 effect ordering and the complete combined effect list, including
  harmful effects. Verify slot-order consequences and potion-preview visibility
  separately from ingredient tooltips. Do not limit a potion to four effects
  merely because an ingredient has four slots.

### Apparatus and brewing calculations

- A mortar and pestle is mandatory; alembic, calcinator, and retort are optional.
  All selected tools must still be owned when brewing commits. Tools are reusable
  and are not consumed. The player can inspect and change the selected apparatus.
- Mortar quality influences potion strength; a retort improves beneficial
  effects, an alembic reduces harmful effects, and a calcinator modifies both.
  Follow verified TES3 combinations rather than treating tool qualities as one
  interchangeable multiplier. Include effects without magnitude or duration.
- Success and strength use modified Alchemy, Intelligence, and Luck. Verify the
  brewing factor `Alchemy + Intelligence / 10 + Luck / 10`, exact random-roll
  comparison, and fatigue treatment against TES3. Do not reuse spellcasting or
  ingredient-eating chance rules for brewing.
- Read applicable settings, including `fPotionStrengthMult`, `fPotionT1MagMult`,
  `fPotionT1DurMult`, and `iAlchemyMod`, plus MGEF base costs and flags. Match
  magnitude/duration rounding, omission of ineffective results, potion value,
  and average selected-ingredient weight. Do not substitute a recipe price list
  or balance caps for imported rules. Invalid content must produce a useful
  diagnostic without partially mutating inventory.

### Brewing transaction and inventory UI

- Normal inventory use of owned apparatus opens the alchemy panel. The player
  can select/remove ingredients, select tools, enter a potion name, inspect the
  permitted effect preview, create a potion, and close the panel. Visible
  ingredient quantities and results reflect authoritative session state.
- A successful attempt consumes one unit of every selected ingredient and adds
  exactly one generated potion. A random failure consumes those ingredients
  without adding a potion. Report success, failure, and unmet prerequisites
  distinctly. Cancelling or editing a selection consumes nothing.
- Verify no-common-effect attempts separately: TES3 can waste ingredients on an
  unsuccessful combination. Do not make every unsuccessful attempt a harmless
  validation rejection. Missing mortar, insufficient ingredients, missing name,
  depleted stacks, and stale selections have explicit tested outcomes.
- Revalidate ownership and quantities at commit. Inventory changes, generated
  potion creation, and eligible skill progress form one coherent operation;
  rejected stale input cannot consume only part of a recipe. One input edge
  submits one attempt; held input or reopening the panel cannot replay it.
  Repeated intentional attempts stop safely as ingredients run out. Batch
  brewing, if offered, applies the same per-attempt rules and quantity bounds.
- Generated potions retain name, ordered effects and arguments, magnitude,
  duration, weight, value, presentation assets, and stable identity. Stack only
  equivalent potion definitions; equal names alone do not establish equality.
  Never alias a generated potion to an authored quest item because its name or
  effects happen to match. Inspection, dropping, pickup, and existing inventory
  transfers preserve the generated definition.

### Ingredient eating, potion use, and progression

- Normal inventory use can eat an ingredient without apparatus. Consume one
  unit and attempt its first effect with TES3 eating chance, magnitude, and
  duration rules. Eating does not apply all four effects or permanently reveal
  them. Test no-effect and failed-effect outcomes and relevant fatigue/settings
  independently from brewing.
- Drinking a crafted potion consumes one unit and applies its complete ordered
  effects to the player through shared active-effect state. Beneficial and
  harmful effects, cures, effects with no magnitude/duration, repeated doses,
  resistance interactions, and expiry follow TES3 behavior. Do not add an
  Oblivion/Skyrim weapon-poison application action.
- Fortify Intelligence, Luck, or Alchemy affects later eligible calculations
  while active and expires without permanently rewriting base stats. Brewing
  stores the computed result; later stat changes do not recalculate an existing
  potion. Temporary modifiers do not count as earned skill increases.
- Successful creation and qualifying ingredient eating submit their respective
  TES3 SKIL use events to MECH-TES3-006 once. Failed brewing, failed eating,
  preview changes, and drinking potions do not invent skill progress. Verify
  precise event eligibility and use indices, including fractional advancement,
  skill caps, class membership, and Intelligence level-up counters.

### Persistence

- Save/load preserves generated definitions and identities, quantities, skill
  progress, active potion/ingredient effects, and remaining durations. Effects
  also survive cell transitions and expire through normal gameplay time.
- Saving with the panel open either restores an uncommitted valid selection or
  closes it safely. Loading never retries a brew, duplicates a potion, consumes
  ingredients again, or reapplies skill credit or an active effect.
- Older saves initialize absent alchemy state without inventing items or gains.
  Generated definitions belong to versioned gameplay state; preserve cooked
  scenes and streamed-chunk serialization layouts.

## Implementation Boundaries

Extend BethesdaSession, existing TES3 content adapters, gameplay inventory,
progression, and save state. Keep recipe matching, effect calculation, and
attempt resolution deterministic and testable with the session's replayable
randomness. The application and retained UI submit intents and show session
results; they do not own another inventory, generated-item registry, or skill
ledger. Keep dependencies in the existing application/runtime/importer
direction. No generic crafting framework, dynamic plugin, or Lua system is
required.

Other games' alchemy rules, merchant services, ingredient harvesting/respawning,
training services, and an automatic recipe optimizer are outside this capability.
Existing item acquisition and transfer paths must still accept alchemy items.

## Verification

- Add synthetic hand-rolled CTest fixtures for INGR/APPA/ALCH/MGEF parsing and
  load-order overrides, missing slots, distinct-record selection, argument-aware
  matching, hidden effects, duplicate suppression, effect order, and recipes
  combining more than four effects. No retail assets are required.
- Test visibility just below/at each 15/30/45/60 threshold and modified/custom
  `fWortChanceValue` values. Test potion-preview knowledge independently.
- Pin expected calculations for mortar alone and every optional-tool
  combination across qualities, mixed beneficial/harmful effects, no-magnitude
  and no-duration effects, rounding boundaries, low strength, custom settings,
  temporary stats, and fatigue. Assert values against verified expectations,
  not another call to the implementation under test.
- Seed rolls for success/failure boundaries. Assert ingredient counts, unchanged
  apparatus, potion identity/definition, and exactly-once skill credit for each
  outcome, including no shared effects, stale selections, final units, repeated
  input, same-name different potions, and invalid content.
- HARNESS-006 exercises production mapped input and the shared alchemy UI flow:
  opening from inventory, selecting/removing ingredients and tools, naming,
  brewing, failure feedback, exhaustion, cancellation, reopening, eating, and
  drinking. Incorrect result/count expectations must fail usefully.
- Test eating independently from brewing; drinking mixed effects, cures,
  repeated doses, temporary-stat feedback into later brews, and expiry. Verify
  progression events and excluded actions through MECH-TES3-006.
- Round-trip saves before and after success/failure, with generated potions
  dropped into the world, while effects are active, and with an open panel.
  Test cell transitions, process restart, older saves, and replay determinism.
- With local base `Morrowind.esm`, obtain apparatus and ingredients through
  ordinary gameplay, brew a named potion, eat an ingredient, drink a crafted
  potion, and verify inventory, skill progress, effect expiry, drop/pickup, and
  save/load through normal input. Compare tool combinations, a mixed-effect
  recipe, hidden-effect behavior, and a failed attempt with retail TES3. Record
  reproducible actions and assertions; no console grants or test-only brew calls
  establish retail acceptance. Keep all game data and captures local.
- Run the relevant content, runtime, inventory, progression, active-effect,
  save, and headless UI targets, then the full CTest suite before changing status
  in `docs/PARITY.md`. Review the final implementation diff.

## Definition of Done

All required behavior passes deterministic synthetic tests and normal-input
local TES3 verification. A player-created character can create, retain, drop,
recover, and drink correctly calculated potions, eat ingredients, earn the
appropriate skill progress, and continue across transitions and save/load
without item or effect duplication. A calculation-only or UI-only slice remains
Partial with its missing behavior documented; mark Implemented only after the
complete acceptance and regression checks pass.
