# MECH-TES3-002: Main Quest Dialogue and Journal Conditions

Status: Planned

## Goal

The TES3 runtime selects main-quest dialogue and advances journal entries from the imported game's conditions and result scripts, so the standard MECH-002 route can progress through conversation.

## Dependencies

- TES3 DIAL/INFO import, dialogue UI, journal state, and MWScript result execution
- Player and actor state, inventory, factions, and save/load

## Required Behavior

- Dialogue filters used by the standard main quest evaluate their authored actor, player, cell, faction, disposition, global, local, inventory, death-count, and journal inputs. An unavailable input does not silently make a guarded response eligible.
- Topics, choices, greetings, and result scripts are presented and executed in the order allowed by imported content. A conversation observes changes made by an earlier response, including the item-state behavior already added for MECH-002.
- Journal entries are recorded once when their authored transition occurs. The journal shows current and historical progress, including the informants, lost prophecies, and seven House and Ashlander recognitions. Independent recognitions can progress in any allowed order.
- Dialogue and journal state survives cell transitions and save/load. Reopening a conversation does not duplicate one-time quest rewards or advance a quest without its conditions.

## Verification

- Add synthetic dialogue fixtures covering each condition type needed by the route, both eligible and ineligible responses, choice/result ordering, independent recognition progress, and save/load.
- Trace the local `Morrowind.esm` route at representative Caius, Urshilaku, Great House, Ashlander, Archcanon, and Vivec conversations. Assert that the expected INFO and journal index are reached by normal interactions.
- Run the relevant CTest targets and existing suite before changing status in `docs/PARITY.md`.

## Definition of Done

Every dialogue and journal gate on MECH-002's standard route is reachable through its authored conditions, with no permissive fallback or externally set quest stage required.
