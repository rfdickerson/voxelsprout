# MECH-TES3-005: Main Quest Encounters and Finale

Status: Planned

## Goal

The required actors and encounters on MECH-002's standard route can be resolved through gameplay, culminating in Dagoth Ur and the Heart of Lorkhan sequence.

## Dependencies

- Playable world/cell interaction, actor placement, movement, collision, and combat
- MECH-TES3-002 dialogue and journal conditions
- MECH-TES3-003 quest scripts and effects
- MECH-TES3-004 quest items and equipment

## Required Behavior

- Required NPCs and creatures are reachable in their authored locations and can be spoken to, activated, followed, fought, or otherwise interacted with as the quest requires. Cell changes preserve their relevant state.
- Combat, health, death, and disease outcomes are visible to TES3 scripts, dialogue conditions, and journal transitions. Required kills or cures are recorded once; defeated actors do not reappear in a state that blocks progression.
- The player can survive and resolve the Sixth House and final citadel encounters using ordinary controls and acquired equipment. The Heart sequence produces its authored world and quest-state changes, followed by continued playable interaction.
- Save/load before and after required encounters preserves actor, combat, disease, artifact, and finale state without repeating one-time events.

## Verification

- Add synthetic tests for actor death counts, scripted encounter transitions, disease/cure state, Heart interaction gates, and save/load before and after an encounter.
- With local base-game data, verify representative Sixth House and Corprus encounters and the complete final sequence through normal play. Record any inaccessible location or failed transition; do not commit game data or captures.
- Run the relevant CTest targets and existing suite before changing status in `docs/PARITY.md`.

## Definition of Done

Every required encounter on the standard route has a playable resolution, and the finale reaches the authored post-quest state without debug commands or seeded journal progress.
