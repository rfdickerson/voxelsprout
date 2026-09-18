# Helgen departure checks

## Tilted rocks along the Riverwood route

Skyrim compound REFR rotations now match `NiMatrix3::SetEulerAnglesXYZ`:
clockwise local Z, then Y, then X, composed as `Rx(-x)*Ry(-y)*Rz(-z)`
for column vectors. The previous reversed composition exposed open rock
undersides and moved slabs across the road. Render instances and authored
collision triangles share the corrected transform. Other games retain their
existing convention. Cell build version 113 invalidates old placements.
Procedural grass slope angles use the same Skyrim convention.

Reference: [CommonLibSSE-NG NiMatrix3](https://github.com/CharmedBaryon/CommonLibSSE-NG/blob/main/src/RE/N/NiMatrix3.cpp).
The regression checks the compound basis of Guardian Stones slab REFR 0xA0086.
Local same-camera evidence is `captures/helgen-fixes/rock-placement-before.png`
and `rock-placement-after.png`: the floating upper-right rock and overhanging
left slab return to the landscape. This verifies placement, not completion of
the physics-controlled Helgen–Riverwood walk. The existing guided MP4 uses a
collision-free camera and remains unchanged.

## Authored departure

The `skyrim-helgen-ralof` start uses the Helgen Keep exit door's authored XTEL
arrival. It completes the skipped MQ101 prerequisite, selects the MQ102B branch
with stage 5, starts its stage 10, and runs the MQ102 stage 15 exit script.
It does not replay MQ102B stage 0 (a test teleport into the keep), or seed
stage 20 ahead of its dialogue. Ralof is placed once beside the fresh-start
player. His travel speech runs through the authored SCEN records and PEX
fragments, with spatial voice playback and subtitles that leave movement active.
Existing `skyrim-bleak-falls` Hadvar starts and saved games retain their branch.

Run with the locally installed content profile:

```sh
ODAI_WINDOW_SIZE=768x432 build-linux-relwithdebinfo/odai \
  --profile captures/jk-skyrim-showcase/profile.json \
  --scenario skyrim-helgen-ralof --no-resume
```

Explicit scenario spawn coordinates now retain on-foot movement. Diagnostic
flight remains available through the explicit flight setting. Missing terrain
in the fallback movement path no longer turns Space into free flight.

Object LOD waits for all sixteen replacement cells before removing a nearby
regular tile. Partial-cell handoff retains connected rock geometry until all
of its vertices have resident coverage, avoiding triangle-centroid cutaways.

LAND interpolation preserves authored quadrant boundary posts. Overfull paint
stacks split into smaller patches with consistent texture IDs per triangle.
The material still has a four-overlay limit when more than four layers overlap
within one authored terrain quad. Cell-build version 112 invalidates old terrain
caches; the serialized scene layout is unchanged.

Skyrim voice playback indexes FUZ responses by INFO and response number, validates
the container bounds, and decodes the embedded xWMA through the existing ffmpeg
path. Ambient scene responses advance after their decoded audio duration;
interactive conversations retain Continue. Promptless linked topics are retained,
and actor names resolve through their source plugin's localized string table.
The FUZ layout was checked against the [AutoMod audio format documentation](https://github.com/SpookyPirate/spookys-automod-toolkit/blob/main/docs/audio.md).

Local evidence is under `captures/helgen-fixes/`; it contains game-derived content
and must remain local. The bootstrap probe and runtime captures distinguish
quest state, scene phases, ambient speech, player grounding, and modal dialogue.

The SCEN adapter executes phase conditions, timers, dialogue actions and
begin/end fragments. Travel actions use the existing NAVM movement bridge;
authored alias packages resume between scene overrides. Distant destinations
use partial routes confined to reachable resident NAVM and replan as cells load.
Escort distances come
from the package data. Quest-linked trigger boxes merge base VMAD properties
with reference overrides and dispatch OnTriggerEnter to the retail PEX scripts.
Scene cursors, action timers and response progress are saved with the session.

Flying quest actors use coarse motion along authored linked patrol markers and
dispatch package-end PEX fragments. This allows the Alduin exit gate to complete;
it is not a flight animation implementation. The adapter is not a complete
Creation Engine package-procedure or SCEN implementation. Full Riverwood route
completion, standing-stone magic effects, and interrupted-scene recovery still
need live validation.

Local departure evidence: `ambient-verified.log` records four authored responses
through phases 6–9 and MQ102B stage 20. Its 76 observed checkpoints kept modal
dialogue closed and the player grounded. Synthetic tests cover phase/timer
parsing, audio acknowledgement and response advancement, flight patrols, save
round trips, and partial navigation without crossing disconnected geometry.

Guardian Stones MSAA regression: opaque texture/vertex alpha was reaching
alpha-to-coverage, making terrain and mountain surfaces disappear. The main
shader now exports alpha one for materials without authored alpha test/blend.
The depth prewrite explicitly exports coverage alpha, including the cutout
ramp, despite its disabled colour write mask. Blended materials retain their
ordinary alpha test because their pipeline does not use alpha-to-coverage.
Vulkan coverage rules: https://docs.vulkan.org/spec/latest/chapters/fragops.html#fragops-covg

Validation: shader compilation and all 45 CTests pass. Local same-camera 4x MSAA
captures at engine position (3513.76, 1586.53, 58809.16), yaw 90, pitch 5 show
the previous opaque-alpha behavior exposing sky through the right mountainside
and ground, and the corrected shader restoring those surfaces. Evidence is in
`captures/helgen-fixes/guardian-{before,after}.png` (local game data only).

For grounded gameplay recordings, `ODAI_CAPTURE_FOLLOW_ACTOR=Ralof` with
`--capture-video <path> 30 <seconds> --capture-audio` follows that actor through
the normal player physics controller. It replans across resident navigation,
stops near the actor, and eases camera heading toward him. It does not advance
quest stages or replace the actor's authored packages. Missing paths stop the
player rather than teleporting through unloaded cells. This is an opt-in capture
control; ordinary play and fixed camera tours keep their existing controls.
With `--tour-file` but without `--flythrough`, this capture control changes to
grounded waypoint replay after 50 seconds of departure dialogue. Recorded
waypoints steer the player controller directly; the final destination uses
resident navigation. This allows traversal evidence even where an NPC navmesh
does not cover the player's previous walk. The capture does not repair Ralof's
observed stalled escort movement after the departure dialogue.

Recording evidence: `helgen-riverwood-guided.mp4` is a 150-second 1536x864
30 fps camera walkthrough with AAC game audio, ending at Riverwood's entrance.
Its route combines local player samples with authored road placements. It is
not evidence that the player controller can complete the route: automated
physics runs stopped at hillside obstacles. `helgen-walk-progress.mp4` preserves
95 seconds of actual player movement and Ralof's departure speech for diagnosis.
