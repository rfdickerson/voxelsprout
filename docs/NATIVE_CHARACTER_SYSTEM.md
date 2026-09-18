# Native character system

This is an implementation status document, not a claim that the full character
roadmap is complete.

## Implemented

- Native animation is the default. `ODAI_SKYRIM_ANIMATION_MODE=havok` opts into
  the existing admitted Havok interpreter; it does not enable unsupported graph
  classes or execute FNIS/Pandora generators.
- Version 1 and 2 JSON animation packs load from virtual `odai/animations/*.json`.
  Distinct pack paths compose; the existing virtual Data resolver selects the
  winning file for an overridden path. Priority, profile-layer precedence, and
  file-qualified rule ID determine the winner, in that order.
- Weighted variants are chosen on state/rule entry and retained through that
  selection. Missing replacements use the built-in state clip. The output exposes
  execution mode, rule, provider, active clip, and fallback reason.
- JSON and HKX clip assets load through virtual Data. HKX needs the imported
  animation skeleton; native JSON tracks bind by bone name. Native HKX root
  translation stays available for physics extraction while its pose is sampled
  in place. Existing built-in catalogs retain their import behavior.
- Humanoid conversion validates required Skyrim bone names and ancestry, orders
  bones deterministically, and remaps skin indices, inverse binds, and existing
  clip tracks together. Authored bind transforms, proportions, and extra bones
  remain intact. This is canonical indexing and semantic mapping of compatible
  rigs, **not** general retargeting between different skeleton hierarchies.
- Registered first-person animation instances follow the third-person state,
  action identity, and clock, while selecting their own clips. Only the
  third-person instance emits gameplay clip markers. The player avatar now uses
  profile assets and the common animation loader, but dedicated first-person
  avatar geometry/presentation is still outstanding.
- Skyrim session controllers use explicit jump requests, 100 ms coyote time,
  120 ms jump buffering, 4.572 m/s takeoff speed, and bounded 8 m/s² air control.
  These defaults are configurable through `PhysicsCharacterConfig::movement`.
  Other games retain the existing movement path. Jump phases and relative
  surface-impact severity are exposed to animation. Landing boundaries default
  to 2, 5, and 8 m/s. Severity does not yet activate a ragdoll.
- Native rules may include two-dimensional speed/direction blend samples and
  ordered masked layers. Base poses blend in local TRS; absolute layers precede
  additive layers. Additive layers require an explicit reference clip sampled
  at time zero. Only the selected primary clip supplies gameplay events and
  requested root motion. Physics remains the movement authority.
- Humanoid packets now support ground-probed foot plants, pelvis correction,
  two-bone limb IK and bounded aiming. Legacy pose fallback retains its bounded
  foot corrections. See [version 2 status](NATIVE_ANIMATION_GRAPH_V2.md) for the
  GPU path, authoring schema and outstanding acceptance work.
- Save version 15 persists selected clips/providers, deterministic random state,
  action identities, transition clips, layer clocks, and movement timers.
  Older fields default safely. Content/mode changes discard incompatible
  animation state with diagnostics. Cooked scene/chunk layouts are unchanged.

## Pack example

Place this file in a profile layer at `odai/animations/guard.json`. Clips may name
an existing catalog clip or a virtual `meshes/...` HKX/native JSON asset. No game
or mod asset is included with this example.

```json
{
  "version": 1,
  "rules": [
    {
      "id": "wounded-walk",
      "state": "walk_forward",
      "priority": 50,
      "blend_time": 0.18,
      "conditions": {
        "sex": "female",
        "injury": { "min": 0.3 },
        "tags": ["guard"]
      },
      "variants": [
        { "clip": "meshes/actors/character/animations/local/wounded.hkx", "weight": 3, "loop": true },
        { "clip": "meshes/actors/character/animations/local/tired.hkx", "weight": 1, "loop": true }
      ]
    }
  ]
}
```

Conditions are ANDed. Scalars compare typed strings, booleans, or numbers;
numeric ranges use inclusive `min`/`max`; `tags` requires every listed tag.
Unavailable context does not match. Gameplay automatically supplies movement,
speed, vertical speed, weapon style/drawn state, stance, grounded state, jump
phase, landing severity/impact speed, sex defaults, combat, interior, injury,
and fatigue where available. `AnimationInputState::selectorContext` accepts
additional race/body, equipment, class/personality, or environment facts and
tags from callers; not all of those currently have automatic runtime producers.

`blend_samples` is an optional rule array of `{ "clip": "...", "x": 0,
"z": -120 }`, in actor-local Bethesda units/second. Samples use normalized
inverse-distance weights and shared normalized clip phase. The rule's selected
variant remains its event source. `layers` is an optional array of objects with
`id`, `clip`, `order`, `weight`, `bones`, `additive`, and `reference_clip`.
Bone masks name exact bones or `role:left_hand`-style semantic roles. Masks list
individual bones, not implicit subtrees. Invalid layers produce diagnostics and
leave the base pose usable. Layer clocks restart on selection transitions.

## Body morphs, hair simulation, and ragdolls

Native body morph packs are versioned JSON resolved by the active Data/profile
stack. Set `ODAI_SKYRIM_BODY_MORPH_PACK` to its virtual path and provide slider
values as `ODAI_SKYRIM_BODY_SLIDERS=weight=0.6,muscular=0.2`. Version 1 requires
`vertex_count`, `topology_fingerprint`, sparse `targets`, and an explicit
`outfit_mappings` entry for the active outfit. Invalid topology, non-finite or
out-of-range sliders, duplicate deltas, and missing outfit mappings reject the
whole update. Morph positions replace the rest positions before the existing
Vulkan compute skinning pass; no second rendering path is introduced. The CPU
validates the pack and converts its sparse targets once to vertex-major CSR
buffers. Those offsets and deltas stay device-local, while the current slider
weights travel with the per-frame bone palette. The compute shader accumulates
each vertex's weighted morph deltas and immediately skins the resulting rest
position. The skinned velocity pass repeats the same accumulation with current
and previous weights so temporal reprojection follows the deformed surface.
Templates without morphs bind zero-filled sentinel buffers and retain their
existing output. Slider state is carried by animation snapshots and save
version 14.

Skyrim player rigs are inspected for authored hair, ponytail, or braid chains.
Discovered chains use damped fixed-step Verlet motion, length limits, and
head/spine collision spheres after authored animation. The chain resets from
the authored pose after teleport-sized root changes and holds the authored pose
when `ODAI_SKYRIM_NO_HAIR_SIM=1` is set. Rigs without a suitable bone chain keep
their authored hair pose.

Canonical pelvis, spine, head, arm, and leg roles now build an articulated Jolt
ragdoll. Activation transfers the live pose and velocity, removes the player
`CharacterVirtual` from capsule collision, and drives the skin palette from the
Jolt bodies. Press `K` in the third-person Skyrim showcase to toggle knockdown
and recovery. Recovery only restores capsule authority after a walkable static
support surface is found. The camera follows the pelvis while down. Active
ragdoll transforms and velocities are saved and reconstructed by save version
14; older saves initialize ragdolls as inactive.

The synthetic pack schema is in
`tests/fixtures/native_body_morph_v1.json`. `odai_character_dynamics_tests`
covers topology rejection, outfit mapping, deformation, hair constraints,
authored fallback, and teleport reset. `odai_skyrim_animation_tests` covers Jolt
activation, capsule suspension, supported recovery, and active-ragdoll
save/load continuation.

The Vulkan implementation follows the Khronos Vulkan Guide's synchronization
guidance: the established compute-to-vertex `VK_KHR_synchronization2` barrier
continues to make the fused morph-and-skin output visible to vertex input.

## Remaining roadmap work
Rules may set `hold_until_landing: true` to retain their selected clip and clock
through ascent and descent, including changes in selector context while airborne.
Ground contact resumes normal landing selection. Non-looping clips hold their
final pose if their duration ends before contact; walking off a ledge still uses
the ordinary fall state.


- General bind-space retargeting, dedicated first-person geometry and content,
  complete automatic selector context producers, and first-person visual QA.
- Ground probes/planting and limb IK; complete movement-driven recovery policy.
- In-game slider UI and authored production morph packs for locally installed
  body/outfit topologies.
- Multi-chain hair authoring and distance budgets for crowd NPCs.
- Angular joint limits, get-up animation matching, death policy, and first-person
  body-camera presentation during knockdown.
- End-to-end mod-pack visuals and performance measurements for those systems.

## Checks

`odai_native_animation_tests` covers selector precedence, conditions, malformed
packs, weighted continuation, missing assets, profile layering, shared action
clocks, event ownership, additive reference/mask math, movement buffering,
coyote time, air-control limits, and landing thresholds.
`odai_humanoid_rig_tests` checks skin/clip remapping, preserved sampled matrices,
optional bones, idempotence, and atomic rejection of incompatible input.
The existing Skyrim animation/Jolt and save suites cover integration and saved
variant continuation. The full CTest suite remains required.

For a synthetic optimized selector/sampler measurement:

```bash
cmake --build --preset linux-vcpkg-relwithdebinfo --target odai_native_animation_tests -j
./build-linux-relwithdebinfo/odai_native_animation_tests --benchmark
```

This measures 200 synthetic actors with 64 bones and 128 rules over 300 ticks.
It is not a measurement of imported scene rendering, hair, or ragdolls.

## Local validation evidence

- Debug runtime build and all 48 CTest tests passed.
- RelWithDebInfo synthetic hair-chain measurement (200 actors, 300 fixed ticks):
  5.21 ms total, 0.087 microseconds per actor-tick on the local host. This is
  solver-only CPU cost and excludes rendering and Jolt.
- The RelWithDebInfo synthetic benchmark measured approximately 7.3 µs per
  actor/tick on this machine; this is a fixture measurement, not a scene budget.
- The locally installed XPMSSE male/female NIFs passed canonical admission with
  543/649 bones and 16 semantic roles each. The probe is available as
  `odai_humanoid_rig_tests --rig-file <local-skeleton.nif>`; no fixture assets
  are copied into the repository.
- JK's Skyrim + SMIM third-person captures ran at 768×432 logical size and
  1536×864 native framebuffer size, render scale 1. The converted avatar was
  visible in the market capture, but settlement placed it below the visible
  street. **Visual acceptance failed**: the profile's spawn/grounding issue
  remains unresolved. First-person and motion captures are also outstanding.
- Local evidence is under `captures/native-character-check/` and
  `captures/native-character-check-market/`, outside committed source assets.
