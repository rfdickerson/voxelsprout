# Bleak Falls Barrow route

## Milestone 1: reproducible start and observable checkpoints

Contract version 1 starts in Riverwood after Helgen on the Imperial/Hadvar
branch. The only allowed stage bootstrap, in order, is:

| Quest | Stages applied | Completed |
| --- | --- | --- |
| MQ101 | 900 | yes |
| MQ102 | 10 | no |
| MQ102A | 0, 1, 5, 10, 20 | no |

MQ102 objective 10 must be displayed and active. MQ102B, MS13 and MQ103 remain
at stage 0. No Golden Claw or Dragonstone is granted to the player. Startup
fragments run after their installed VMAD attachments are loaded. This contract
is independently frozen in `src/bethesda/route_checkpoint.cc`; changing the
scenario seeds requires reviewing and versioning the contract.

Use JK's Skyrim 1.7 and SMIM SE 2.08 with the exact DLC, Creation Club and
ResourcePack baseline in [the release manifest](../packaging/skyrim-slice.json).
The report records resolved plugin order, layer versions, required archive
filenames and content fingerprints. It omits installation roots. It records
selected content; it does not certify mod provenance or arbitrary compatibility.

Headless baseline check (does not create a player or inject encounter residency):

```bash
build-linux-relwithdebinfo/odai_bethesda_probe "$SKYRIM_DATA" \
  --scenario-start-check skyrim-bleak-falls --profile "$ODAI_PROFILE" > start.json
```

Run twice with the same content and compare `checkpoint` and `session_hash`.
The repeatable helper also compares the selected mod versions, plugin order
and required archives against the release manifest:

```bash
python3 scripts/check_bleak_falls_start.py \
  --probe build-linux-relwithdebinfo/odai_bethesda_probe \
  --data "$SKYRIM_DATA" --profile "$ODAI_PROFILE" --out captures/route-start
```

`ok` covers the bootstrap only. `player_inventory_verified: false` is expected
for this headless check. The older `--scenario-check` remains a fixture-assisted
integration probe and is not acceptance evidence for ordinary gameplay.

Start the runtime with a new report path and isolated save slot:

```bash
ODAI_WINDOW_SIZE=768x432 ODAI_WINDOW_HIDPI=1 ODAI_NATIVE_LOGICAL_PRESENT=1 \
ODAI_RENDER_SCALE=1 build-linux-relwithdebinfo/odai \
  --profile "$ODAI_PROFILE" --scenario skyrim-bleak-falls --no-resume \
  --scenario-report route.jsonl --save-game route-save.json
```

Reporting begins after four simulation ticks, allowing queued startup fragments
to run before the initial assessment (the same minimum as the headless check).
F8 appends a manual checkpoint and shows confirmation. Changes sampled once per
second also append checkpoints; output is flushed after each record. Use an
existing writable parent directory. A write failure shows an error and disables
reporting for that session. Existing reports are appended to, with a new session
header on each launch. F5 saves; F9 loads. Save files retain their current schema.

Each checkpoint includes route quest stages, objective flags, alias bindings,
player inventory/cell/death state, dialogue speaker and choice count, scene
playing flags, activator counts and puzzle ring states, and story-event sequence.
The runtime additionally records visual chunk and collision-cell counts.
Registered alias targets are **not proof of streamed actor residency**. Scene
playing flags are **not proof that a scene executed**. Sampled observations can
miss intermediate events and are not a complete dungeon-trigger trace. Reports
never certify ordinary-input completion automatically. Capture runs are demos,
not gameplay acceptance. Keep reports and game captures local.

## Following milestones and acceptance

1. **Start contract and reporting:** repeat a fresh baseline; demonstrate Riverwood
   using the selected content and record the first player checkpoint.
2. **Riverwood conversations:** Hadvar/Alvor arrival and help sequence, Lucan's
   Golden Claw offer, and the Whiterun handoff through normal dialogue. Resolve
   authored scene sequencing and dialogue gates before marking this complete.
3. **Entrance and Arvel:** ordinary travel and doors, entrance pillar puzzle,
   spider encounter, Arvel release/escape/death and claw recovery. Wrong puzzle
   attempts and alternate Arvel outcomes must remain recoverable.
4. **Dungeon and sanctum:** authored triggers, traps and enemies, inspectable claw,
   claw door, word wall and naturally resident boss. No seeded boss or quest item.
5. **Dragonstone and returns:** loot Dragonstone once, exit, return claw to Lucan,
   and complete Farengar's hand-in through authored conditions. Repeat with an
   early-acquired Dragonstone and alternate hand-in order.
6. **Persistence and endurance:** reload during dialogue, both puzzle states,
   transitions, combat, looting and each hand-in; revisit evicted cells without
   duplicate loot or lost progress. Complete the two-hour soak and optimized
   hardware checks in [the release checklist](RELEASE_CHECKLIST.md).

For each route beat keep a before/after checkpoint, normal-input capture and
save/reload result. Record failures explicitly. Milestone 1 does not clear any
later milestone or the public playable-release gate.
