# MECH-TES3-004: Main Quest Items and Equipment

Status: Planned

## Goal

The player can obtain, inspect, transfer, equip, and use the items needed to complete MECH-002 through normal Morrowind gameplay input.

## Dependencies

- MECH-001 player inventory and world-item representation
- TES3 item/content records, object activation, containers, dialogue transfers, and save/load
- MECH-TES3-003 for item-related scripts and effects

## Required Behavior

- Picking up and receiving a quest item creates one owned instance and removes or updates its world source. Dropping, consuming, or handing it over updates both world and inventory state without duplication.
- The inventory UI supports normal item actions needed by the route, including reading quest books and notes, selecting equipment, and using applicable items. Actions unavailable for an item are clearly rejected rather than silently treated as dropping it.
- Quest-required weapons and artifacts, including Wraithguard, Sunder, and Keening, retain identity and relevant instance state. Their equip/use rules and consequences follow imported content and are visible to scripts and dialogue.
- An item transferred by dialogue or script is immediately visible to later conditions and is preserved across cell transitions and save/load.

## Verification

- Add synthetic pickup, reading, transfer, equip, use, duplicate-prevention, and save/load tests, including item gates used by dialogue and scripts.
- With local base-game data, verify package delivery, required evidence and artifact acquisition, and the equipped/use state needed for the final encounter through ordinary UI input.
- Run the relevant CTest targets and existing suite before changing status in `docs/PARITY.md`.

## Definition of Done

The standard route's required item interactions work through normal input, and the final artifacts can be acquired and used with consistent world, inventory, script, and save state.
