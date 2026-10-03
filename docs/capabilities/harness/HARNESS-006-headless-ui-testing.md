# HARNESS-006: Headless UI testing

Status: Implemented

## Goal

Run deterministic tests of player-facing UI flows without a window, GPU,
Vulkan device, or installed game data. The first vertical slice exercises the
inventory through the same input and UI/gameplay code used by `odai`.

The harness must make UI behavior observable to CTest and engineering agents,
including which screen is open, what the player can see and select, and what
gameplay action an input actually causes.

## Dependencies

- HARNESS-001 headless runner, HARNESS-002 scenario format, and HARNESS-003
  state snapshots provide deterministic execution and gameplay checkpoints.
- The existing CPU `odai_ui` input, widget, and draw-list interfaces provide
  reusable UI behavior. UI flows currently owned by `BethesdaApp` may need a
  focused separation from its GLFW/Vulkan presentation host.
- MECH-001 defines the inventory behavior and first acceptance scenario.
  HARNESS-006 uses synthetic inventory content for that scenario; completing
  HARNESS-006 does not require all MECH-001 acceptance criteria to pass.

## Required behavior

### HARNESS-006-A: Production UI flow without presentation

The harness initializes the real UI state and gameplay session used by the
runtime. It can open, navigate, act within, close, and reopen a UI screen
without creating a window, swapchain, or renderer. UI behavior must not be
reimplemented as a separate test-only model. A direct gameplay API call alone
does not count as a UI test.

The headless build configuration must be able to compile and run the UI test
target without GLFW, Vulkan, Slang, or retail game assets. CPU draw-list
generation may be used for inspection, but GPU image comparison is outside
this capability.

### HARNESS-006-B: Deterministic input replay

A versioned synthetic scenario can describe ordered key, navigation, pointer,
and button inputs needed by its UI flow, with a fixed viewport and timestep.
Input uses the runtime's normal action mapping and UI event dispatch, including
press, hold, and release transitions. The runner advances UI and gameplay in a
defined order so repeated runs produce the same observable result.

Tests must not set private `BethesdaApp` flags, directly invoke widget
callbacks, or bypass the normal input path to open inventory or activate an
item.

### HARNESS-006-C: Observable UI and gameplay checkpoints

At requested steps the runner exposes a stable, machine-readable UI snapshot
containing the active screen, visible item or control identity and label,
displayed quantity where applicable, selection/focus, and enabled state. It
also exposes the corresponding gameplay snapshot from the session. Assertions
must distinguish a displayed value from the underlying world value, so a stale
inventory view can fail even when the gameplay mutation succeeds.

Snapshots and diagnostics must identify the scenario, input step, UI control
or item, expected value, and actual value. Snapshot ordering and identifiers
must be stable across identical runs; tests must not depend on pixel colors or
platform font metrics.

### HARNESS-006-D: Automated verdict

The scenario runner exits zero when every UI and gameplay assertion passes and
nonzero for invalid scenarios, failed assertions, or an unavailable required UI
path. CTest runs the synthetic scenario with no display server and no game
data. A failure returns enough evidence to reproduce the input sequence.

## Initial inventory scenario

Use a synthetic player with Iron Dagger x1 and Restore Health x3, or equivalent
distinct fixture items. Through normal mapped input:

1. Open inventory and assert that both items and their quantities are visible.
2. Navigate to Restore Health, invoke the inventory drop action, and advance
   the session until the command is applied.
3. Assert that the player has two Restore Health items and that one matching
   world item exists near the player.
4. Reopen or refresh inventory through input and assert that the displayed
   quantity is two; close the screen through input.

This scenario must exercise the UI action path, rather than using the existing
headless runner's direct `inventory_actions` fixture field.

## Verification

1. The inventory scenario passes in the headless CTest configuration without
   a display, GPU, or retail game data.
2. Two identical runs produce equivalent UI and gameplay snapshots.
3. Incorrect expected screen, item, quantity, focus, and world state each
   cause a nonzero exit with an actionable diagnostic.
4. Invalid or unsupported input steps fail during scenario validation.
5. Existing headless, retained UI, and full CTest suites pass.

## Definition of Done

- All required behavior and verification checks pass.
- The inventory vertical slice proves normal input, visible UI state, and the
  resulting gameplay mutation in one deterministic run.
- The same harness can host another UI flow without adding a second runner or
  duplicating UI logic.
- `docs/PARITY.md` is updated to Implemented only after verification succeeds.

## Implementation and verification

`InventoryUiFlow` is shared by `odai` and `odai_headless_ui`. The runtime maps
GLFW keys and gamepad navigation into the flow; the headless runner replays
versioned key-level frames against the same flow and `BethesdaSession`. The
Morrowind inventory draw path reads the flow's visible entries, labels,
quantities, and selection. The JSON runner records each UI frame alongside
owned inventory and spawned world items, and fails on mismatched expectations.
The fixture covers opening, navigation, stack and final-item drops, reopening,
and closing. The replay test checks deterministic output, negative UI/gameplay
expectations, and invalid input diagnostics.

```bash
cmake --build build-linux-headless -j 4
ctest --test-dir build-linux-headless --output-on-failure
```

The headless configuration passes 9/9 tests without presentation or game data.
The full build passes 62/62 CTest tests. The optional local
`odai_headless_ui --probe-item-scene <Data Files> <model path>` command checks
that a retail item NIF becomes imported-scene geometry; its data and output
remain local.
