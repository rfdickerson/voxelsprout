# HARNESS-003: State snapshot/dump

Status: Implemented

## Goal

Expose deterministic, machine-readable checkpoints from synthetic headless
scenarios so tests and developers can inspect how runtime state changes.

## Contract

An actor fixture may include `"snapshots": {"ticks": [0, 5, 30]}`. The `ticks`
array is optional; `"snapshots": {}` requests only the final state. Tick zero
means the configured session before its first simulation advance. Requested
ticks may include advances caused by inventory actions. Ticks must be unique
integers in the inclusive range from zero to the fixture's total number of
advances. The runner sorts checkpoint requests into tick order and rejects
invalid requests with a nonzero exit and a JSON error.

When requested, the existing one-line JSON result gains `snapshots`:

```json
{
  "version": 1,
  "checkpoints": [{"tick": 0, "state": {"tick": 0}}],
  "final": {"tick": 30, "state": {"tick": 30}}
}
```

The actual `state` includes the tick, random state, deterministic hash, player
object ID, all resident objects, registered physics characters, and quest
summaries. Objects expose stable identity, base, kind, enabled flag, transform,
current space, inventory, and actor values when present. Characters expose
position, velocity, and grounded state. Quests expose stage, running/completed/
failed flags, and objective flags. Collections use stable identity order.
The final snapshot is taken after all fixture actions and is also present when
the fixture's expectation fails. Existing fixtures without a snapshot request
keep their output shape.

This is an inspection format, not a restorable save or a full serialization of
AI, animation, Papyrus, rendering, or other subsystem internals. The state hash
may cover state that is not expanded in this JSON view. The format does not
change streamed chunks or cooked scenes and requires no retail game data.

## Verification and Definition of Done

1. CTest checks initial, intermediate, action, and final checkpoints against
   synthetic actor and inventory fixtures.
2. Three identical runs yield identical JSON, while the state shows movement
   and inventory changes at the expected ticks.
3. Duplicate, negative, noninteger, and out-of-range ticks fail clearly.
4. A wrong fixture expectation still returns the final snapshot with a nonzero
   exit code.
5. The headless build and full CTest suite pass; only then mark HARNESS-003
   Implemented in `docs/PARITY.md`.
