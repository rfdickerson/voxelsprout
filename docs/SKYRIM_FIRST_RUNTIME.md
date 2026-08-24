# Skyrim-first runtime status

ODAI's first gameplay contract is `--scenario skyrim-bleak-falls`. The scenario
starts at Riverwood, bootstraps MQ101 stage 900, and registers MQ102 before
replaying its authored stage-10 Riverwood startup fragment after VMAD attachment.
MS13, the later MQ102/MQ103 stages, their objectives, aliases, and effects must
come from installed Skyrim content.
ODAI never reads or writes retail `.ess` saves.

## Implemented foundation

- `RecordKey` stores normalized plugin name plus local form ID. Conversion to
  and from resolved regular/light-plugin IDs goes through the active load order.
- `ObjectId` distinguishes persistent placed references from save-owned spawned
  IDs. The streamed Skyrim actor population is registered by placed reference.
- `BethesdaSession` owns a fixed 60 Hz clock, bounded catch-up, seeded RNG,
  deterministic command ordering, runtime objects, quests, and Papyrus state.
- `WorldCommand` is the mutation seam for transforms, visibility, actor values,
  inventory, outfits, persistent navigation requests, spawning, and destruction. Applied commands produce
  `RuntimeRenderDelta` values without modifying `ImportedScene`.
- Skyrim VMAD script/property attachments and complete big-endian PEX object,
  property, state, function, value, and instruction tables are bounds checked.
  Supported PEX instructions translate atomically into the deterministic VM;
  unsupported opcodes and native declarations are compatibility errors.
- The VM executes arithmetic, comparisons, branches, calls, properties, arrays,
  state handlers, event posting, update timers, and deterministic nested stacks.
  Instances, active states, timers, and latent stacks are snapshotted. It has no
  filesystem, network, process, DLL, or SKSE path.
- Retail QUST/VMAD loading registers MS13, MQ103, their stages, localized
  objective text, QSDT/CTDA log-entry groups, stage fragments, aliases, and
  alias-script blocks, then follows the reachable game-authored PEX closure.
  Stage changes dispatch only authored log entries whose conditions match.
  PEX auto-property backing variables resolve to the attached VMAD value and
  native-posted events are deferred safely when the VM is already advancing.
  Strict output is reachability-based, so unrelated functions in a large class
  do not inflate the scenario report.
- TES5 dialogue is a separate typed path from Fallout dialogue: DLBR records
  define branch roots, DIAL.QNAM owns a topic, DIAL.BNAM resolves its branch,
  and type-7 child groups own authored-order INFO variants. `.STRINGS` supplies
  DIAL prompts and INFO.RNAM player-line overrides; `.ILSTRINGS` supplies spoken
  responses. INFO VMAD begin/end
  fragments are fully parsed and attached to stable INFO `RecordKey`s. Strict
  CTDA selection currently covers the route's quest stage/done, inventory,
  speaker ID, alias reference, death/dead-count, relationship-rank, and
  `GetVMQuestVariable` gates. TES5 CIS1/CIS2 string operands resolve attached
  quest properties instead of treating their raw CTDA slots as form IDs.
  Selecting a response dispatches its authored phase fragment through
  `TopicInfo` in the same deterministic Papyrus VM; it does not synthesize
  quest outcomes.
- Skyrim CELL `XLCN` ownership remains runtime metadata outside immutable
  `ImportedScene`; actors receive stable location keys in exterior and interior
  cells. Papyrus movement requests resolve authored reference positions and use
  deterministic resident-NAVM point-to-point paths, including coincident border
  stitching across records without depending on streaming completion order.
  Exterior doors whose destination NAVM is resident become typed off-mesh
  actions keyed by their source reference; NPC controllers walk to the authored
  portal and relocate to the resolved XTEL arrival. Interior NAVM now enters and
  leaves residency with its room instead of falling back to collision wandering.
- Skyrim actor/quest residency now follows the same main-thread cell lifecycle
  as geometry, collision, and NAVM. Exterior ACHR population is filtered by the
  planner's resident set (not the renderer chunk set, because an empty visual
  cell can still own gameplay references), and one population refresh runs only
  after the asynchronous ring settles. `XLCN` plus ACHR `XLRT` reach
  `BethesdaSession` before dynamic aliases are evaluated. Eviction hides actor
  presentation and removes controllers but does not mutate the authored
  enabled bit or discard the persistent `ObjectId`, quest binding, inventory,
  death state, VM state, or AI cursor; a revisit reuses and re-enables residency
  for the same runtime object. Interior actors use the synchronous room
  transition and the identical registration path. The current settled-ring
  refresh still rescans nearby actor records; replacing that with a cooked
  per-cell gameplay-reference payload is required before the streaming
  performance gate can pass.
- Reachable `LCTN`/`GLOB` records are registered by stable key. Location keyword
  data, loaded state, globals, sandboxed debug-log names, and ordered story
  events implement the native state required by the installed route closure.
- CTDA parsing/evaluation is shared and supports strict or diagnostic-permissive
  handling, including OR chains and all six comparison operators.
- `OdaiSaveV7` is checksummed, staged/atomic, and contains the content
  fingerprint, scenario, fixed clock, RNG, objects/inventories, quest state,
  actor AI path cursors, and VM threads/globals plus Jolt character and
  behavior-graph snapshots. Changed profiles reconcile through `RecordKey` or
  refuse before mutating the live session. Typed walk/door path actions store
  stable door `RecordKey`s, melee target/cooldown/counter state, and activator
  puzzle rings/solution/open state and quest-created inventory provenance;
  regression fixtures migrate v1-v6 explicitly.
- The focused Jolt runtime provides authored triangle collision, deterministic
  character controllers, support identity, and strict snapshot restore. The HKX
  boundary validates Skyrim SE x64 packfiles and drives deterministic locomotion,
  root motion, sockets, animation events, and graph snapshots for registered rigs.
- Player and AI melee share a fixed-tick Jolt query: stable nearest-target
  ordering, facing cone, static-collision occlusion, stamina/cooldown, damage,
  death state, `Actor.StartCombat`/`StopCombat`, and saveable combat targets.
  Player attacks are edge-latched from primary input and resolved on the next
  simulation tick; dead actors stop producing locomotion intent and an earlier
  live save restores it.
- Quest aliases decode `ALCO`/`ALCA`/`ALCL` created-object ownership. MS13
  therefore materializes the retail Golden Claw and Arvel's Journal in the
  placed Arvel actor exactly once instead of granting either at bootstrap.
  Dead actors are searchable with `E`; transfers use paired deterministic
  remove/add commands, zero-count entries are erased, and created-item
  provenance round-trips in V7 so reload cannot duplicate loot. Materialization,
  transfer, and lethal damage post alias `OnContainerChanged`/`OnDeath` events
  only after their deterministic world-command batch has applied.
- TES5 dynamic aliases also decode forced-location `ALFL`, reference-alias
  `ALFA`, reference-type `ALRT`, and actor placement `XLRT`. The MQ103
  BleakFallsBoss alias therefore binds the installed boss reference only when
  its location and Boss reference type match. Dragonstone materialization owns
  a persistent spawned-item identity whose base `RecordKey` survives save/load
  and is accepted by Papyrus `RemoveItem` as a Form.
- The installed-data scenario probe runs fixture-assisted Golden Claw assertions:
  it injects an already-dead Arvel from the retail alias, materializes both authored items,
  transfers the corpse inventory to the player, executes
  `MS13GoldenClawScript.OnContainerChanged`, and dispatches the retail stage
  fragment. Current installed base data reaches MS13 stage 40, completes
  “Retrieve the Golden Claw,” displays “Find the secret of Bleak Falls Barrow,”
  and emits no VM diagnostic.
- The same fixture-assisted probe verifies both retail dialogue hand-ins. It selects
  Lucan's localized, condition-gated INFO, executes its authored end fragment,
  reaches MS13 stage 100, and follows the retail reward-quest dependency. It
  selects Farengar's “I have the stone tablet you wanted” INFO, executes its
  authored begin fragment, removes the Dragonstone, reaches MQ103 stage 190,
  and initializes MQ104 stage 10 with the resident Dragonsreach participants.
  The probe establishes MQ103 stage 10 and the exact non-streamed quest-reference
  residency, feeds the exact installed boss ACHR identity/location/XLRT metadata
  into the registry, then matches it via ALFL/ALFA/ALRT, runs a
  real Jolt-backed melee kill, transfers the authored ALCO Dragonstone through
  the loot path, and reaches stage 180 through the attached alias script. It
  still directly establishes MS13 stages 50/60 for the door/hand-in fragment
  assertions. The stage-190 check additionally proves the player item count
  changes from one to zero, the spawned alias identity remains stable, and an
  immediate save/reload preserves both. These inputs are reported under `setup`,
  not counted as traversal or proof of natural boss streaming.
  Current installed base data registers 25 DLBR branches, 127 route DIAL topics,
  239 INFO variants, and 49 INFO fragments; both hand-ins finish with no VM or
  command diagnostic.
- The interactive Skyrim scenario uses that same typed dialogue registry. A
  nearby actor is talkable only when a retail branch-root INFO passes strict
  CTDA; the existing keyboard/gamepad modal displays DIAL/INFO.RNAM player
  prompts and INFO responses, exposes only INFO.TCLT-linked successors, and
  queues the authored begin fragment on selection and the end fragment once on
  response completion or conversation exit through the
  fixed-step VM. TES5 actors never fall through to the TES4/Fallout dialogue
  importer, avoiding synthetic or blank fallback conversations. An installed-
  data Vulkan smoke confirms Lucan exposes “I could help you get the claw
  back” from a clean stage-0 session. Retail scene startup, voice-duration
  synchronization, dialogue save boundaries, and verified goodbye/say-once
  flag semantics remain incomplete.
- Winning placed-reference VMAD bytes remain gameplay metadata outside
  `ImportedScene`. The clean-room activator adapter persists ring states and
  solutions, rotates rings deterministically, requires the retail key item,
  and advances the retail quest/stage identity only on the correct combination.
  Installed Skyrim diagnostics confirm the Hall of Stories keyhole supplies
  its Golden Claw, MS13 stage 50, and three ring references, whose own VMAD
  supplies initial/solve states. While that interior is resident, `1`/`2`/`3`
  rotate the large/medium/small ring and `E` uses the claw. Successful state is
  saved, advances retail MS13 stage 50, filters only the mechanism's attributed
  Jolt triangles, and atomically replaces the interior with a presentation copy
  that hides the opened mechanism. Authored HKX door motion/audio is still open.
- Exterior and interior `ImportedScene` collision is copied into session-owned
  Jolt residency bodies in engine Y-up space and removed with its streamed cell.
  Authored triangles carry their resolved placed-reference identity; terrain is
  unowned and never filtered. Session setup replays cells that arrived before
  scenario initialization.
- `odai_bethesda_probe --scriptcheck` emits PEX opcode/call/native closure data
  plus per-function instruction operands; `--quest-trace` reports QUST/VMAD
  stages, log-entry conditions, fragment names, alias properties, and attached
  installed PEX. `--skyrim-dialogue-trace` reports localized DIAL prompts,
  DLBR roots, DIAL branch membership, INFO/RNAM prompts, responses, CTDA gates,
  links, and begin/end fragment metadata for one
  quest. `--scenario-check` loads the exact shared runtime closure, runs the
  fixture-assisted Golden Claw alias-event and hand-in assertions, and reports
  injected setup, unverified route segments, unresolved calls, and physical/runtime
  blockers without launching Vulkan. It does not claim the playable gate.
- Procedural world generation, clipmap implementation, and MagicaVoxel sources
  are no longer linked into the renderer. A packed legacy chunk helper remains
  entangled with the Vulkan backend and is tracked below.

The PEX/VMAD readers follow the byte ordering and table structure independently
implemented by [Champollion](https://github.com/Orvid/Champollion/blob/main/Pex/FileReader.cpp)
and the [open Papyrus compiler](https://github.com/russo-2025/papyrus-compiler/blob/master/modules/pex/reader.v),
with QUST fragment and alias-tail layout cross-checked against
[OpenSkyrim's parser](https://github.com/realfakenerd/OpenSkyrim/blob/main/crates/converter/src/esm/records/record_type/vmad.rs).

## Commands

```bash
odai --scenario skyrim-bleak-falls --stream "/games/Skyrim Special Edition/Data"
odai --scenario skyrim-bleak-falls --load-game save.odai.json --stream <Data>
odai_bethesda_probe <Data> --scriptcheck QF_MS13_00039645.pex --strict
odai_bethesda_probe <Data> --quest-trace Skyrim.esm MS13
odai_bethesda_probe <Data> --quest-trace Skyrim.esm MQ103
odai_bethesda_probe <Data> --skyrim-dialogue-trace Skyrim.esm MS13
odai_bethesda_probe <Data> --scenario-check skyrim-bleak-falls
```

## Gate status

| Gate | Status | Evidence / blocker |
|---|---|---|
| Foundation | Partial | All retained tests pass and unreachable generators are unlinked. Vulkan backend still exposes a legacy chunk helper/API. |
| Content/VM | Partial | Synthetic malformed-input, opcode, call-stack, event-order, timer, latent-call, full QUST/INFO VMAD tails, typed DLBR/DIAL/INFO parsing including RNAM, branch-root/TCLT selection, dialogue phase dispatch, auto-property, CTDA log/dialogue selection, LCTN, GLOB, and strict-failure fixtures pass. The installed-data scenario probe loads 4 route quests, 236 stage fragments, 65 alias-script blocks, 25 DLBR branches, 127 DIAL topics, 239 INFO variants, and 49 INFO fragments; follows the reachable PEX/cross-quest closure; registers 81 locations and 91 globals; replays MQ102 stage 10 with objective 10 visible while MQ103 remains at 0; and reports zero unresolved reachable native calls. Physical route behavior remains outside this gate. |
| Simulation | Partial | Fixed-step replay hashes, actor values, inventory, quest-created actor loot, outfits, activation commands, VMAD-backed puzzle state, location/global/story state, persistent movement requests, authored reference goals, cross-record NAVM stitching, resident teleport actions, per-reference streamed/interior Jolt collision, fixed-tick AI intent, occluded player/AI melee, damage/death state, and deterministic animation graph state exist. Retail faction hostility, death/ragdoll, general containers/leveled loot/equipment, authored claw-door HKX motion/audio, cross-space packages, Skyrim preferred links, and full retail HKX binding remain. |
| Save | Partial | V7 exactly round-trips nested VM, objects, navigation, typed AI walk/door actions, combat targets/cooldowns, activator puzzle state, quest-created spawned-item identity/provenance, outfits, physical characters, and graph events; checksum rejection, interrupted-commit recovery, fingerprint reconciliation, and mandatory v1-v6 migration fixtures pass. The installed-data assertion now save/reloads immediately after MQ103 stage 190 and rechecks the removed Dragonstone plus alias identity. Mid-response dialogue flow and dungeon streaming boundaries remain. |
| Playable | Blocked | The authored MQ102 Riverwood startup stage now runs, keeps MQ103 at 0, and displays objective 10. Retail alias inventory/event assertions reach MS13 stage 40, and the installed keyhole/ring VMAD drives puzzle requirements, input, persistent collision/visibility, and stage 50. The fixture-assisted MQ103 assertion dynamically fills the retail boss alias, kills it through the physical combat path, loots the authored Dragonstone, reaches stage 180, and then completes the stage-190 Farengar fragment without a direct item grant. The headless hand-ins still depend on explicitly reported stage and residency injection; they are fragment assertions, not a completed route. MQ102’s Riverwood-to-Whiterun conversations, Arvel web/escape/combat, authored stage-50/60 triggers, natural boss streaming/encounter packages, the pre-acquired-Dragonstone Farengar branch, hostility/faction assignment, death/ragdoll presentation, general loot/equipment, authored claw-door motion/audio, dialogue save boundaries, and continuous traversal remain incomplete. `release_gate_passed` is false. |
| Rendering | Partial | Runtime deltas are consumed separately from immutable `ImportedScene`; validation/device-loss/cell-churn soak gates have not run here. |
| Performance | Not measured | Requires an installed-data release build and RX 6600/RTX 3060-class reference machine. |
| Release | Not reached | Depends on all preceding route, soak, and performance gates. |

This table is intentionally a release checklist, not a claim of gameplay parity
with OpenMW or Skyrim.
