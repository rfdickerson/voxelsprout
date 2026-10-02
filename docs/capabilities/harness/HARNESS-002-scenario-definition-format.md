# HARNESS-002: Scenario definition format

Status: Implemented

## Goal

Give synthetic headless scenarios a documented, versioned JSON contract with
clear input errors and no retail game-data dependency.

## Contract

`odai_headless <scenario.json>` accepts a JSON object with `version: 1`, a
nonempty `name`, and `kind`. The `version` and `kind` keys may be omitted for
older actor fixtures; omission means version 1 and `actor`. New fixtures should
write both keys. Unknown fields, unsupported versions and kinds, missing
required fields, and invalid primitive types fail with a JSON error and a
nonzero exit status before simulation begins.

Kinds:

- `actor`: `seed`, `step_seconds`, `steps`, and an `actor` with stable `id`,
  `base`, three component `position`, and `desired_velocity`. Optional
  `expect`, `snapshots`, `inventory_actions`, and `inventory_expect` provide
  assertions and checkpoints. See `one_actor.json` and `player_inventory.json`.
- `static_physics`: triangle mesh `objects` and named ray or sphere `queries`.
  See `static_world_physics.json`.
- `terrain_physics`: synthetic LAND `cells` and downward ray or sphere
  `queries`. See `morrowind_terrain_physics.json`.

All four examples live in `tests/fixtures/harness/`. The runtime uses the
production session and physics code. Its one-line JSON output contains the
scenario result; an incorrect expectation exits nonzero.

## Verification and Definition of Done

1. All four example scenarios pass through CTest without game data.
2. The runner accepts explicit version 1 and the legacy actor form.
3. Unsupported version/kind, unknown field, malformed vector, and wrong type
   fail with a path in the diagnostic.
4. The full CTest suite passes before status is marked Implemented.
