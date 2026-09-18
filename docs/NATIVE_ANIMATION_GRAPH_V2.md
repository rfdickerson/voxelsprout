# Native animation graph version 2

Version 2 extends the virtual `odai/animations/*.json` pack format. Version 1
selection priority, weighted choices, layers, gameplay-marker ownership and
fallbacks remain supported. Legacy character evaluation is retained until the
replacement passes real-scene acceptance; GPU parity alone is insufficient.

## Authoring

A version 2 rule may contain a `graph`. Rule conditions and catalog/asset clip
resolution use the existing native pack interface. Graph parameters are typed by
their JSON defaults (boolean, number or string). Runtime values must keep that
type. Missing values use the declared default.

```json
{
  "version": 2,
  "rules": [{
    "id": "walk-graph",
    "state": "walk_forward",
    "variants": [{"clip": "walk", "weight": 1}],
    "graph": {
      "root": "locomotion",
      "parameters": {"speed": 0},
      "nodes": [
        {"id": "idle", "type": "clip", "clip": "idle"},
        {"id": "walk", "type": "clip", "clip": "walk"},
        {"id": "locomotion", "type": "blend1d", "parameter": "speed",
         "samples": [{"node": "idle", "x": 0}, {"node": "walk", "x": 120}]}
      ]
    }
  }]
}
```

Clip names above are placeholders for the installed catalog, not bundled assets.

| Node | Fields |
| --- | --- |
| `clip` | `clip`, optional `speed` (0–4) |
| `blend` | Two `inputs`, `weight` or numeric `parameter` |
| `blend1d` | `samples` of `{node,x}`, numeric `parameter` |
| `blend2d` | `samples` of `{node,x,y}`, `parameter`, `parameter_y` |
| `layer` | Two `inputs`, `bones`, `weight`; additive requires `additive:true` and `reference_clip` |
| `cache` | One input, reused within the evaluation packet |
| `state_machine` | States in `inputs`, `initial`, ordered `transitions` |
| `limb_ik` | One input, limb `role`, three numeric names in `target_parameters` |
| `aim` | One input, bone `role`, `target_parameters`, bounded `max_radians` |
| `translate` | One input, bone `role`, `target_parameters` |

Transitions specify `from`, `to`, `parameter`, typed `equals`, and `duration`
(seconds). State machines may contain other state machines. Bone masks name exact
bones or `role:left_hand`; they do not imply a subtree. Procedural graph targets
are in skeleton model space. Callers can also supply world-space foot contacts,
explicit limb/aim targets, and named motion-warp targets through animation input.
Missing named warp targets disable that window. Each window has its own bounded
translation budget; requested movement is still accepted or clipped by physics.

New 2D spaces use barycentric interpolation on empty-circumcircle triangles and
nearest-hull clamping; version 1 retains its existing inverse-distance weights.
Clip samples synchronize on paired `sync:` annotations with normalized-phase
fallback. The first authored sample leads a synchronization group, including at
zero weight. Layers do not take gameplay-event or root-motion ownership.

Admission is atomic. Limits include 4 MiB JSON, 64 parameters, 256 nodes and
expanded instructions, 32 dependency levels, 32 blend samples and 128 transitions
per machine. Cycles, bad references, duplicate IDs/coordinates, invalid types and
nonfinite configuration values are rejected. Missing rig roles or clip resources
fall back at evaluation without partially advancing the graph.

## Rig and execution contract

The active profile supplies the skeleton; no XPMSSE assets are bundled. Semantic
mapping preserves authored rest transforms, proportions and extra bones. Stable
fingerprints, depth levels and explicit limb chains do not assume a fixed bone
count. Clip bindings retain source/target rest transforms, apply rest rotation
correction and local limb-length translation scaling, and leave unmapped target
bones at rest. This binding supports matching named parent relationships; it is
not arbitrary hierarchy or creature retargeting. HKX reference poses are used
when present; older imports without them retain name binding.

Fixed ticks decide states, clocks, events, root motion and procedural contacts on
CPU. Immutable evaluation packets feed both a CPU reference evaluator and Vulkan
compute. Compute samples clips, blends local TRS, applies retained inertial
residuals and procedural operations, composes hierarchy by depth, and produces
skin palettes before the existing morph/skinning pass. Resources and packets
contain no Vulkan types outside the renderer. Actor work is batched on the
existing queue with explicit Synchronization2 dependencies.

Current and previous rendered palettes have persistent GPU history. Spawn,
teleport, template replacement and rig replacement invalidate history. Frame
uploads/scratch are frame-owned; retired immutable resources use the existing
queue lifetime. Four-weight linear skinning and morph-before-skin ordering remain.

Save format 15 adds graph clocks, synchronization clocks, transition state,
inertial pose/velocity history, foot plants and warp budgets. Older saves default
these fields; incompatible content invalidates restored animation state. Cooked
scene and chunk layouts are unchanged.

Foot plants retain one world-space position and surface normal through stance.
They acquire only near the probed floor, fade as the authored foot lifts, and
release on lost contact or when the leg would exceed its authored reach. Ground
probes moving beneath a travelling actor do not relocate an existing plant.

## Acceptance status and remaining work

The implementation is still under validation. Full CPU poses remain available for
sockets/physics and renderer admission compares packet output against the
submitted palette. Unsupported external equipment, hair or ragdoll corrections
therefore use whole-actor CPU pose fallback. CPU evaluation has **not** yet been
reduced to gameplay-critical dependency chains, and external overrides are not
uploaded as a separate GPU override stage. Immutable GPU resources currently
reside per actor slot, rather than sharing one allocation across identical rigs.

Synthetic coverage includes 650-bone rigs, rest binding/proportions, atomic
rejection, graph validation, save continuation, interrupted inertial transitions,
marker synchronization, unreachable IK and foot alignment. The headless Vulkan
test compares local TRS at 0.0001 absolute component tolerance (quaternion signs
normalized for comparison), procedural palettes at 0.002 absolute element
tolerance, and synthetic four-influence deformed vertices at 0.002 position
distance. It also checks previous-palette history. These tolerances do not
establish a deformed-vertex error bound for every real rig.

The 1/16/48 workload in that executable repeats actor-equivalent work and includes
submission/wait cost. It is not a batched scene benchmark or a claimed speedup.
Real-scene frame cost, upload volumes, GPU timestamps, deformed vertices and
ragdoll/attack/slope visual acceptance must be recorded before removing legacy
character evaluation. The initial local Whiterun capture was obstructed by gate
geometry. Full-capsule placement checks corrected the arrival; subsequent male
and female captures show visible equipped avatars on the gate-apron platform
at 768×432 logical / 1536×864 framebuffer size. This is spawn/idle evidence,
not attack, slope, locomotion or recovery acceptance. Game assets and captures
remain local.

### Local measurement, 2026-09-13

RelWithDebInfo, Intel Graphics (LNL), 650 bones, sampling three clips plus
masked/additive composition, inertia, translation, limb IK and aiming. No
validation layer for this timing run. One cold measurement per workload; GPU
numbers are timestamp spans and CPU numbers evaluate the same packet plus
palettes. This is an initial measurement, not a stable performance budget.

| Repeated actor work | GPU timestamp ms | CPU reference ms | Submit/wait ms |
| ---: | ---: | ---: | ---: |
| 1 | 0.160 | 0.323 | 1.805 |
| 16 | 4.023 | 3.358 | 5.788 |
| 48 | 16.862 | 10.202 | 17.956 |

Immutable storage was 255,040 bytes; one frame packet with retained inertial
residuals was 65,684 bytes (1,050,944 / 3,152,832 bytes for 16 / 48 copies).
The renderer batches disjoint actor work; the sequential test does not measure
that benefit. Maximum observed local / palette / vertex errors were
4.77e-7 / 3.81e-6 / 1.11e-4. A separate run with Khronos synchronization
validation passed without errors. These numbers justify retaining fallback and
do not justify removing the legacy path or claiming a frame-time speedup.

Compatibility references: [XPMSSE upstream](https://github.com/acepleiades/XP32-Maximum-Skeleton-Special-Extended)
and the [Vulkan synchronization guide](https://docs.vulkan.org/guide/latest/synchronization.html).
