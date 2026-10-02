# MECH-002: Play and Complete Morrowind's Main Quest

Status: Partial

## Goal

The player can begin a new Morrowind game, follow the standard main-quest route through ordinary gameplay, defeat Dagoth Ur, and continue playing afterward. Quest progression uses the imported `Morrowind.esm` content and its authored conditions, dialogue, scripts, journal entries, items, actors, and world references. A debug command or preseeded journal state is not a substitute for playable progression.

Reference: [UESP's Morrowind Main Quest guide](https://en.uesp.net/wiki/Morrowind:Main_Quest). The imported game data is authoritative for record IDs, conditions, ordering, and outcomes.

## Dependencies

- MECH-TES3-001: Morrowind New Game and Character Generation
- MECH-TES3-002: Main Quest Dialogue and Journal Conditions
- MECH-TES3-003: Main Quest MWScript Commands and Effects
- MECH-TES3-004: Main Quest Items and Equipment
- MECH-TES3-005: Main Quest Encounters and Finale
- TES3 plugin and asset loading, including quest dialogue, journal entries, and scripts
- Playable Morrowind exterior and interior travel, interaction, and combat
- Player inventory, item pickup/use, and quest-item transfer
- TES3 dialogue, journal, and MWScript runtime behavior
- Save/load of the player, world, journal, dialogue, and script state
- Deterministic headless verification infrastructure

MECH-002 remains Partial until the full route works, including travel to all required quest locations and end-to-end verification.

## Required Behavior

### MECH-002-A: Start and investigate

From normal character creation and release in Seyda Neen, the player can deliver the package to Caius Cosades. Caius's instructions and the informant investigations progress through their authored dialogue, item, and journal conditions. The player can reach the Urshilaku, Sixth House, and Corprus Cure portions of the story without setting quest stages externally.

### MECH-002-B: Fulfill the prophecies

The player can pursue the lost prophecies and the Path of the Incarnate. The three Great House Hortator recognitions and four Ashlander Nerevarine recognitions are all obtainable through their authored encounters. These seven recognitions can progress in the orders allowed by the game content; completing one does not erase or incorrectly block another.

### MECH-002-C: Defeat Dagoth Ur

After the standard recognition route, the player can meet the Archcanon and Vivec, obtain and use the required artifacts, enter the Sixth House citadels, and complete the final confrontation and Heart of Lorkhan sequence. The resulting journal and world state reflect completion, and the player can continue traveling and interacting afterward.

### MECH-002-D: Preserve and present progression

The journal and available dialogue reflect completed and pending steps using the imported content. Quest conditions, item ownership, actor state, and world changes remain consistent across cell transitions and save/load. Repeating an interaction or reloading a save must not grant duplicate one-time rewards or leave the route in an impossible state.

## Verification

- Add focused, game-data-free tests for representative TES3 journal and dialogue conditions, scripted item and artifact gates, independent recognition progress, final encounter state changes, and save/load at quest checkpoints.
- Verify the standard route end to end in `odai` using locally installed base-game data, from a fresh start through the post-finale playable state. Record the build, content profile, checkpoints, and any remaining blockers; do not commit or redistribute game data or local captures.
- Run the relevant CTest targets and the full suite before marking the capability Implemented. Keep failures and incomplete acceptance criteria visible as Partial in `docs/PARITY.md`.

## Out of Scope

- The level-gated short route and Yagrum Bagarn back path
- Tribunal and Bloodmoon main quests
- Optional side quests, including independently resolving Sleepers Awake
- New quest content or rewritten dialogue

## Definition of Done

MECH-002 is Implemented only when all required behavior is playable through normal input, the focused automated tests and existing suite pass, a local base-game end-to-end playthrough reaches and verifies the post-finale state, and `docs/PARITY.md` records the verified status.

## Current implementation

TES3 dialogue item conditions now read the live player inventory. Scripted `AddItem` and `RemoveItem` calls update that dialogue view immediately and are reconciled with world state after the queued commands apply. A synthetic test covers pickup, delivery, scripted grant, and subsequent dialogue eligibility.

The standard route remains unverified and incomplete. The runtime does not yet provide the full character-generation opening, all quest-required gameplay commands and effects, or a complete event-driven transition and end-to-end playthrough for the base-game quest.
