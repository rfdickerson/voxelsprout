# Roadmap

## Runtime

- Continue unifying TES3/TES4/Fallout/TES5 record behavior behind the existing
  archive, plugin-load-order, NIF, cell-streaming, actor, dialogue, and weather code.
- Keep streamed cells and cooked `ImportedScene` files serialization-compatible.
- Improve conditional real-data smoke coverage without redistributing game data.

## Skyrim NPC animation and FNIS

Updated 2026-09-10. Humanoid NPC animation remains **partial**: neither Skyrim SE
animation parity nor FNIS-generated graph compatibility is established. See the
[actor animation coverage matrix](SKYRIM_ACTOR_ANIMATION_COVERAGE.md) and
[FNIS compatibility status](FNIS_COMPATIBILITY.md) for implementation details.

### Implemented foundations

- Streamed NPCs use fixed-tick session animation poses, with shared immutable
  assets and animation snapshot retention across eviction.
- The admitted graph subset supports nested state machines, named events,
  scalar defaults, compiled expression conditions, bound manual selectors,
  selected transition flags and local-transform transition blending.
- Recovery playback includes directional walk/run, weapon-family movement,
  directional sneak/swim and phase continuity across gait changes. Physical
  movement inputs are relative to the supporting surface. Recovery playback
  does not count as authored locomotion fidelity.
- One-shot action persistence and exactly-once melee contact timing are covered
  by tests, including save continuation and interruption.
- Character definitions supply animation skeleton and behavior references.
  Layered providers are validated by resolved dependencies; custom-rig mapping
  no longer substitutes unrelated bone indices after a failed name match.
  Rig/skin-bind identity and loaded animation assets contribute to fingerprints.
- Skyrim armor assembly rejects conflicting biped slots and duplicate worn
  items under the importer's existing first-claim policy. Missing/empty or
  race-incompatible addons do not claim slots or report an invisible worn item.
  This is a static assembly safeguard, not runtime equipment-selection parity.

### Ralof equipment integration

The [equipment implementation and remaining acceptance](SKYRIM_EQUIPMENT.md)
now cover persistent slots and script policies, inventory-driven geometry,
profile-resolved weapon attachments, authored clip-event handoffs, weight morphs,
headgear hair coverage and save/stream continuation. Equipping a weapon no longer
unequips clothing. Ralof's duplicate naked torso beneath his cuirass is removed.

Retail `1hm_equip.hkx` playback moves the war axe from `WeaponAxe` to `WEAPON`.
The trace still reports `graphExecuted=0`; recovery playback is not retail/FNIS
behavior-graph parity. Full graph execution, broader weapon/rig validation and
local FNIS-generated assets remain required.

### Remaining milestones

1. **Authored graph execution and locomotion:** implement per-generator clocks
   and distinct playback settings for shared clips, remaining bindings,
   blenders/masks/modifiers, transition windows/interruption/nested overrides,
   notifications and synchronization. Complete authored start/stop, gait blends,
   root-motion rotation and collision handling. Retail graph admission still fails.
2. **Contextual behavior:** import IDLE conditions and package/scene cues; add
   furniture reservation, approach/alignment, enter/loop/exit and cancellation;
   complete authored gesture layers, look-at and limb IK.
3. **Combat and reactions:** extend action/event integration to power attacks,
   ranged weapons, casting, shouts and equipment attachments; implement authored
   reactions and humanoid ragdoll handoff/recovery with persistent state.
4. **FNIS single-actor compatibility:** consume existing generated output through
   the shared graph runtime; complete generated sequences, script event routing,
   dependency manifests, required/optional bone classification and generated-action
   save/stream continuation. This stage excludes running FNIS, player first-person,
   creatures, paired actions, furniture coordination and killmoves; contextual
   furniture work above remains part of the broader Skyrim parity effort.

### Acceptance gates

- Latest Debug validation: all 45 CTest cases pass. Retail male/female probes
  each resolve 61 clips; this verifies regression coverage, not graph execution.
- Require authored event traces, save/load and eviction tests, and reference
  captures before marking a behavior family verified. Ralof equipment captures
  now contain a built humanoid and weapon, but do not establish full visual parity.
- Validate with JK's Skyrim + SMIM, a 768×432 logical window, native-DPI
  framebuffer and render scale 1. Measure actor-heavy scenes in RelWithDebInfo
  or Release.
- FNIS acceptance awaits local generated output, a single-actor animation pack
  and a compatible skeleton replacement in a separate validation profile.
  Keep proprietary assets and captures local; distribute synthetic fixtures only.
- Version changed save layouts with backward-compatible defaults. Preserve
  cooked-scene and streamed-chunk formats unless their stored layout changes.

## Rendering

The current Skyrim visual work is tracked in the
[Skyrim visual parity roadmap](SKYRIM_VISUAL_ROADMAP.md), updated 2026-09-08.
Coverage reporting, standard lighting/terrain materials, supported cubemaps,
image-space records, grass placement and the mill-mist fixture are implemented
within their documented subsets. Native HiDPI presentation and Skyrim lighting
startup order are corrected. Next priorities are animated waterfall/creek
materials, broader authored particles and world-space precipitation. Retail
matching, runtime-consumption instrumentation and the final validation matrix
remain open.


- Preserve explicit Vulkan pass/barrier control.
- Improve terrain tessellation, water/fire, authored skies and clouds, local lights,
  GPU skinning, velocity/TAA, AO/XeGTAO, SSGI, contact shadows, post-processing,
  capture/video, and temporal/XeSS upscaling.
- Continue deleting renderer state that cannot be reached by a Bethesda imported scene.

## RPG surface

- Expand dialogue, inventory/grid picking, minimap, factions/reputation, resources,
  quest/event tracking, entity inspection, navigation, tooltips, and notifications.
- Complete the authored Skyrim slice and its save/reload acceptance gates.
