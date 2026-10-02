# MECH-TES3-003: Main Quest MWScript Commands and Effects

Status: Planned

## Goal

MWScript commands and magic effects required by MECH-002 produce their authored gameplay results, from character generation through the Corprus Cure and Dagoth Ur finale.

## Dependencies

- TES3 script import/compiler/VM, reference identity, world commands, and save/load
- Actor, item, weather, disease, and magic state where the authored commands require them

## Required Behavior

- Identify the scripts and result scripts reachable on the standard route using local base-game data. Implement the commands and expression queries they actually use; registering a command or returning a placeholder value does not count as support.
- Scripted object activation, enable/disable, item and spell changes, actor state, travel, message choices, weather/region changes, and control changes affect the same runtime state observed by subsequent scripts and dialogue.
- Required spell and disease effects, including Corprus-related progression and artifact-related effects, apply with their authored conditions and durations. Save/load retains active effects and running script state.
- Script failures identify the program, source line, target, and unsupported operation. A failure cannot silently advance a quest or leave a partially applied one-time transition.

## Verification

- Add synthetic fixtures for each newly supported route command and effect, including same-tick reads after mutations, failure cases, and save/load.
- Run a route-scoped base-game script closure audit and local checkpoints for character generation, Corprus Cure, artifact acquisition, and the final sequence. Track remaining unsupported route operations explicitly.
- Run the relevant CTest targets and existing suite before changing status in `docs/PARITY.md`.

## Definition of Done

Every MWScript operation and effect exercised by MECH-002's standard route has tested gameplay behavior, and the route's script checkpoints complete without unsupported-command or VM diagnostics.
