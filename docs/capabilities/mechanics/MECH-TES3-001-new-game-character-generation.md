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

The runtime now exposes name, race, class, birthsign, and review menus and persists the selected state. Head and hair choices come from playable BODY records matching the chosen race and gender. The opening actor's `SayDone` script reaches the name menu in a local base-game checkpoint. The app honors authored player-control, jump, and view-switch locks while the opening scripts hold them.

- Scripted `AITravel` destinations now enter the actor movement loop in engine coordinates and publish arrival for `GetAIPackageDone`. The opening guard's coordinate `AIEscort` leads toward the hatch instead of following the player. A new package resets the previous route and completion; an arrived actor stays put. Saved movement state restores whether the actor should move.
- Tutorial MessageBox choices count as `MenuMode`, so other opening scripts wait until the prompt is acknowledged. Synthetic verification includes save/load while that gate is open.
- Fresh player placement snaps down onto resident physics before capsule creation. This prevents generic overlap recovery from lifting the player through the ship ceiling onto its deck. A local base-game `odai` Debug capture, with `Morrowind.esm`, `--no-resume`, no start override, a 768x432 logical window, and render scale 1, reached the name-entry menu below deck at frame 600. Evidence remains local at `/tmp/mech-002-ship.log` and `/tmp/mech-002-ship.ppm`. The complete build and all 66 CTest targets passed.

Still required: ordinary-input ship and Census Office progression, package and exit-gate verification, opening camera facing, and a base-game save/load playthrough to Seyda Neen. Escort waiting/duration and cell-qualified packages remain incomplete; the coordinate destination bridge does not establish full AI-package parity. The Definition of Done has not passed.
