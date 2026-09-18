# Skyrim humanoid actor animation coverage

This is an incremental implementation of the humanoid NPC parity plan, **not
Skyrim SE parity**. Retail graph decoding and executable graph semantics are
reported separately. The installed master graph still fails runtime admission;
ordinary NPCs use an explicitly reported retail-clip recovery catalog.

## Implemented

- Fixed-tick session animation instances feed NPC skinning in gameplay scenarios
  and plain Skyrim previews. Preview sessions do not create player controllers or
  apply quest scenarios. Physics proximity does not determine animation residency.
- Immutable graph assets and matching rig/skin bindings are shared within an actor
  population build. Stream eviction preserves actor animation snapshots.
- Referenced HKX graphs are linked through virtual Data provider resolution, with
  graph-count, node-count, recursion and cycle limits. Event indices are remapped
  by name across graph references.
- HKX event/variable names, clip playback parameters, and timeline triggers are
  decoded. The clip playback-speed offset is corrected from 0x60 (start time) to
  0x64. Relative-to-end and acyclic triggers are retained.
- Conservative execution supports single-generator graph/state wrappers, nested
  state machines, named event transitions, local-before-wildcard priority ties,
  and the admitted default blend-effect settings. Unsupported dependencies,
  settings, unimplemented bindings/conditions/flags, windows, and generators fail admission.
- Scalar variable initial values and variable binding records are decoded. Named
  binding indices are remapped across referenced graphs. Manual selectors support
  `selectedGeneratorIndex` bindings and retain their selected child in snapshots.
- Expression conditions compile once into bounded immutable expression trees.
  Arithmetic, comparisons, Boolean operators and short-circuit evaluation use
  actor-local saved variables. Invalid arithmetic produces actor-local diagnostics.
  Unsupported functions/condition classes remain explicit admission gaps.
- Disabled rules, disabled conditions, local wildcard metadata and explicitly
  allowed wildcard self-transitions are handled. Nested-target, delayed,
  interruption and interval flags remain unsupported.
- Retail recovery clips cover directional walk/run, sprint, turns, jump/fall/land,
  sneak, swim, a dialogue gesture, unarmed/one-handed/two-handed melee families,
  stagger and death, when assets decode against the actor rig. Catalog membership
  does not mean gameplay drives every state yet.
- Action request pulses no longer truncate attack/equip/landing/reaction clips on
  the next tick. Death interrupts them. Dialogue gesture playback is tied to
  conversation entry rather than restarting every tick.
- HKX additive blend hints are retained; local-TRS layering supports bone masks
  and identity defaults for absent additive channels. Bone-mask utilities are
  tested; authored graph bone-weight bindings remain unsupported.
- Root translation deltas account for loop crossings and actor yaw. Ordinary NPC
  catalog movement removes horizontal root drift and uses navigation/physics for
  translation. General authored root-rotation extraction is not implemented.
- Melee clips with an authored `HitFrame`/`hitFrame` marker defer contact until that
  marker. Contact is consumed once, including across save/load; interruption
  cancels it. Script request events cannot impersonate clip timeline crossings.
  Clips without a supported contact marker retain legacy immediate contact and
  are not considered timing parity.
- Save version 11 adds nested graph state, variables, provider fingerprint,
  conversation-edge state and pending melee contact. Earlier saves use defaults.
  Actor animation snapshots can exist without active physics controllers.
  Cooked-scene and streamed-chunk formats are unchanged by this work.

## Remaining parity work

| Area | Status |
| --- | --- |
| Full retail graph execution | Partial infrastructure only: manual selectors and scalar conditions execute; blenders, modifier execution, remaining bindings/conditions, transition flags/windows/nested overrides, graph notifications and non-default playback/effects still prevent retail admission |
| NPC locomotion fidelity | Partial: clips and controller path exist; authored start/stop, gait synchronization, equipment layering and all gameplay stance drivers remain |
| Contextual idles and furniture | Pending: IDLE conditions, furniture markers/reservations, package-driven enter/loop/exit actions and interruption ownership |
| Combat fidelity | Partial melee timing and family selection; power attacks, dual-wield combinations, projectile/casting/shout timing remain; draw/sheath clip markers now drive equipment attachments |
| Physical reactions | Clip playback only; humanoid ragdoll construction, pose handoff, recovery and physical snapshots remain |
| Animation interpolation | State transitions blend local TRS; render-time interpolation still uses the existing matrix interpolation path |
| Visual/reference parity | Not established; asset decoding and synthetic tests do not substitute for Skyrim SE reference comparisons |

## Validation

Synthetic CTest coverage exercises graph admission/execution, event priority,
invalid saved state/provider identity, one-shot lifetime, root-motion loop/yaw,
additive masks, delayed melee contact, duplicate contact markers, request-event
isolation, save continuation, and animation without physics residency.

The optional actor movement probe (`ODAI_SKYRIM_DATA`) resolves male and female
retail clip sets, checks locomotion and equipment-family states and samples an
attack pose. It does not redistribute assets or establish visual parity.

`odai_bethesda_probe <Data> --animationcheck` includes linked graph admission gaps
and available clip paths. `--animation-strict` succeeds only when both the bundle
and linked runtime graph are ready; a structurally coherent bundle is insufficient.

Local preview validation uses `captures/jk-skyrim-showcase/profile.json`, a
768x432 logical window, native-DPI framebuffer and render scale 1. Performance
must be measured using RelWithDebInfo or Release.

Layout references: [CommonLibSSE-NG clip generator](https://github.com/CharmedBaryon/CommonLibSSE-NG/blob/main/include/RE/H/hkbClipGenerator.h),
[behavior string data](https://github.com/CharmedBaryon/CommonLibSSE-NG/blob/main/include/RE/H/hkbBehaviorGraphStringData.h),
and [state machine/effect layouts](https://github.com/adamhynek/activeragdoll/blob/master/include/RE/havok_behavior.h).

### Local validation result (2026-09-10)

- Debug build: all 45 CTest cases passed.
- Installed-data rig probe: 39 clips resolved for each male/female variant; the
  admitted graph flag remains false for retail master behavior.
- RelWithDebInfo runtime captures completed with the combined JK/SMIM profile at
  768x432 logical / 1536x864 framebuffer, render scale 1. The selected captures
  contained no successfully built humanoid NPCs, so they establish runtime smoke
  coverage only, not humanoid visual or actor-heavy performance parity.
- Logs, graph admission output and videos are local under
  `captures/npc-animation-validation/`; no game assets are committed.

### Retail graph execution follow-up (2026-09-10)

The follow-up adds synthetic binary fixtures for scalar initial values, expression
conditions and binding arrays, including malformed counts and non-finite values.
Runtime tests cover expression precedence, short-circuiting, bounded parsing,
condition errors, disabled conditions/rules, variable-driven manual selectors,
invalid selection retention, and selector/variable snapshot continuation.

The animation probe now exposes condition expressions, compiled-condition counts,
scalar-default counts and decoded binding member counts. These are decoding and
compilation measurements, not proof of execution of the retail master graph.

Additional binary-layout references: [serde-hkx expression condition schema](https://github.com/SARDONYX-sard/serde-hkx/blob/main/assets/classes/hkbExpressionCondition.json),
[variable binding schema](https://github.com/SARDONYX-sard/serde-hkx/blob/main/assets/classes/hkbVariableBindingSetBinding.json),
[variable value set schema](https://github.com/SARDONYX-sard/serde-hkx/blob/main/assets/classes/hkbVariableValueSet.json),
and [transition flag schema](https://github.com/SARDONYX-sard/serde-hkx/blob/main/assets/classes/hkbStateMachineTransitionInfo.json).

Follow-up validation: all 45 Debug CTest cases passed. Both retail rig probes
resolved 39 clips. The linked retail graph compiled 191 conditions and retained
297 scalar defaults; 3,079 admission diagnostics remain and runtime execution is
still disabled. Evidence is local in
`captures/npc-animation-validation/retail-graph-followup/`.

### NPC locomotion follow-up

Recovery playback now selects directional weapon-family walk/run clips while
moving with a drawn weapon, with neutral movement fallback for unavailable
families. Sneak and swim support backward/left/right clips. Loop-to-loop gait
changes preserve normalized phase, including changes in clip duration. This is
phase continuity, not authored synchronization-marker execution.

Session movement speed and direction use physical velocity relative to the
supporting surface when a controller is present. Streamed actors without a
controller use navigation velocity; their input speed is no longer forced to a
constant walk speed. Synthetic coverage verifies direction/gait changes, phase
continuity, armed movement and unavailable-family fallback. All 45 CTest cases
pass. Authored start/stop, full gait blends and visual/reference parity remain
unverified and incomplete.

### Ralof follow integration

The `skyrim-helgen-ralof` scenario now routes its companion toward the player
using resident navigation and the existing movement/controller path. Follow
movement continues off camera, stops within 160 units, resumes beyond 230 units,
and enters a 2.5× catch-up gait beyond 650 units, returning to walking below
450 units. Routes refresh at half-second
intervals; unavailable navigation, conversation and death stop movement. Scenario
follow intent supersedes package movement routing for this companion. The follow
route is reconstructed after actor rebuilding rather than storing a second
persistent route format. Shared character-asset resolution and animation poses
remain in use; this does not enable unsupported FNIS graph semantics.

Synthetic tests cover routed movement off camera, walk/run intent, stopping
hysteresis, unavailable navigation and death. All 45 CTest cases pass. In-game
Helgen path-follow captures remain unverified.

Ralof jitter follow-up: ordinary reached waypoints now advance within the same
movement tick, preserving velocity intent across navigation edges. Catch-up gait
uses hysteresis and remains stable during turn alignment. Synthetic regressions
cover nonzero waypoint velocity and both gait thresholds; all 45 CTest cases pass.
Visual smoothness still needs confirmation in a restarted game.

### Equipment follow-up

[Equipment handling](SKYRIM_EQUIPMENT.md) now has persistent slot ownership,
script equip/unequip policies, geometry refresh, skin coverage, weight morphs,
headgear hair hiding and timeline-driven weapon handoffs. Save version 12 stores
these fields with defaults for earlier versions. Synthetic tests cover weapon/
shield contention without undressing the actor, script policies, and attachment
continuation across save/load and animation eviction.

Local Ralof captures show his retail war axe in the drawn hand attachment. The
retail equip clip emits authored timing, but `graphExecuted=0` remains explicit.
This verifies equipment integration with recovery playback, not FNIS graph parity.
The configured JK/SMIM profile still lacks local FNIS-generated output.
