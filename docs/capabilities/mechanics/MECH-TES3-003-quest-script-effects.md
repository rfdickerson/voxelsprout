# MECH-TES3-003: Main Quest MWScript Commands and Effects

Status: Partial

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

## Current implementation and remaining work

The VM preflights unsupported native operations and scripted cast effects before starting a program. Blocked dialogue results leave journal, inventory, and one-time response state untouched. Immediate failures and failures after a suspended MessageBox choice roll back those changes and queued world commands; saves reject an in-flight result transaction. World-script MessageBox text and choices are presented to the player and survive save/load with the suspended thread. Diagnostics identify program, line, target, and operation. Active spell and repeating local-script state survives save/load. Scripted reference activation, item pickup, equipment events, Corprus disease/cure, weather mutation, player controls and character menus, voice gating, and several actor queries now affect runtime state; synthetic fixtures cover these changes.

The local base-game route closure audit includes journal conditions/results, related actor and item scripts, character generation, the Heart of Lorkhan, and literal `StartScript` dependencies. Its 1,609 candidate programs have no blocked native/effect operation or unresolved start. Eight isolated local checkpoints pass: opening script, name voice/menu gate, Corprus Cure, finale `EndGame`, Wraithguard equip, Sunder and Keening with Wraithguard, and the Heart's Sunder/Keening hit sequence with Dagoth Ur's shield changes. Synthetic fixtures also cover `HitOnMe`, cross-reference locals, shield and Restore Attribute effects, scripted doors, pickup, and same-tick non-player spell queries. The 60-test CTest suite passes.

Remaining: this static closure is deliberately broad and cannot prove which scripts are reached during ordinary play. The checkpoints seed late-game inventory and execute selected scripts directly. Verify an event-driven standard route, including the dialogue, item, and encounter gates, and exercise every reached operation's gameplay behavior before marking Implemented.
