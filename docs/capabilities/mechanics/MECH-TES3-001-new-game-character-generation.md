# MECH-TES3-001: Morrowind New Game and Character Generation

Status: Partial

## Goal

A new `Morrowind.esm` game begins aboard the Imperial Prison Ship and reaches Seyda Neen through the authored character-generation sequence. The player can then pursue MECH-002 without a spawn override, seeded journal entry, or test-only command.

## Dependencies

- TES3 content loading, scripted references, interior transitions, and save/load
- Player creation, movement, interaction, and basic UI input
- MECH-TES3-003 for the character-generation MWScript commands used by this sequence

## Required Behavior

- A fresh-game action initializes the player and runs the base game's opening scripts in their authored order. Loading a save does not restart character generation.
- The player can choose and review the required character identity, race, class, birthsign, and appearance through ordinary UI input. Those choices become gameplay state used by dialogue and scripts.
- The prison ship and Census and Excise Office interactions advance only when their authored conditions are met. The package is acquired through the intended interaction, and the exit gate enforces its authored requirement.
- The player exits into Seyda Neen with the expected inventory, enabled controls, journal state, and position. Saving and loading during and after the opening preserves those states without repeating grants or menus.

## Verification

- Use synthetic content to test the opening script and menu state transitions, required choices, package gate, restart behavior, and save/load.
- With local base-game data, complete the opening through normal input and verify the package, initial main-quest journal entry, controls, and Seyda Neen arrival. Do not commit game data or captures.
- Run the relevant CTest targets and existing suite before changing status in `docs/PARITY.md`.

## Definition of Done

The fresh-game path reaches Seyda Neen with a persistent, player-created character and the authored package and quest state. Both synthetic verification and the local base-game opening pass without debug setup.

## Implementation progress

- A fresh Morrowind launch defaults to Vvardenfell, loads the Imperial Prison Ship, and executes the base game's `CharGen` script. An explicit start or gameplay save load skips that opening.
- `PositionCell`, `GetPos`/`SetPos`, and angle operations map TES3 coordinates to the engine's coordinate system. The opening position, interior, state gate, and disabled-control flags survive a synthetic save/load test.
- The direct-start interior binds its placed gameplay references and local scripts after the session is initialized. A local base-game launch rendered the ship and its three nearby actors.

Still required: working name, race, appearance, class, birthsign, and review menus; authored `SayDone` and related scene presentation; enforcement of scripted control locks; ordinary-input ship and Census Office progression; package and exit gate verification; and a base-game save/load playthrough to Seyda Neen. The current camera view is obstructed by ship geometry and needs correction. The Definition of Done has not passed.
