# Guardian Stones authored effect audit

Inspected local Skyrim SE WarriorStone.nif, ThiefStone.nif, and WizardStone.nif
under Clutter/PowerShrines/PowerShrine01. No matching loose mesh overrides were
found in the configured SMIM and JK Skyrim directories. Game assets and raw
probe outputs remain local.

## Original import gaps

- All three assets expose AnimIdle (0.166667 seconds) and AnimPlay (14.8
  seconds), but the importer reports zero supported animation tracks.
- WarriorStone AnimIdle has 43 controlled blocks, all unsupported by the
  transform animation reader. Six NiVisController bindings set BlueBeam,
  topGlow2, topGlow, ambientBeam, OuterGlowCircle, and ambientBeam2 invisible.
  AnimPlay binds keyed NiBoolInterpolators that reveal those groups and hide
  them near the end. Ignoring these bindings loses the authored effect state.
- The linked material controllers use NiBlendFloatInterpolator and
  NiBlendPoint3Interpolator endpoints driven by the controller manager.
  nif_material_animation.h supports direct float/point interpolators, so the
  ray opacity, UV and color tracks remain unresolved. Playing every sequence
  globally would also be wrong: activation must select the instance's sequence.
- readBsEffectShaderProperty skips the packed clamp/lighting fields and all
  four falloff values. The ambient ray properties enable falloff (flag 0x40)
  and author angle values approximately 0.99705 and 0.06105, with opacity
  endpoints 1 and 0. Those values do not reach the renderer.
- NiBillboardNode is recognized as a node but its facing behavior is not
  retained in the flattened static mesh path. The stone has five such nodes.

## What is already correct

The WarriorStone probe loads all 18 shapes without rejected triangles. Its
14 NiAlphaProperty blocks request source-alpha/one additive blending. Ray
materials resolve their source and gradient textures and are marked blended
and two-sided. This is not evidence of missing NIF triangles or a winding bug.

## Implemented sequence support

`nif_effect_sequence.h` resolves each Skyrim NiControllerSequence's concrete
interpolator/controller binding. Named tracks carry sequence timing and cycle
mode; autonomous controllers remain separate. Parent NiVisController tracks
gate descendant shapes without permanently removing their geometry.

The Warrior Stone resolves 98 scalar tracks across its two sequences (color
channels are individual tracks). This includes all six idle/activation visibility
bindings. Material slots are owned by placed reference, so the three stones do
not share playback state. Unknown reference/sequence playback returns false.
Repeated playback can explicitly restart; an ordinary repeated script request
keeps its current time. One-shot tracks clamp to their authored final keys.

Effect falloff angles and opacity endpoints now reach the animated effect
shader, and gradient color lookup includes the authored vertex red channel.
Updates use the existing fenced per-frame material buffer regions; no shared
in-flight GPU material memory is mutated.

The script entry point is `ObjectReference.PlayGamebryoAnimation`, which selects
embedded NIF sequences. Optional start-over is supported; nonzero ease-in is
explicitly unsupported. E provides nearby visual activation of `AnimPlay`.
This visual interaction does not grant Standing Stone blessings.

Cooked scene version 41 appends reference ownership, sequence names, and
effect falloff. Older supported versions retain their previous defaults.
Cell build version 114 invalidates stale cells.

Local captures in `captures/helgen-fixes/`:

- `guardian-effect-idle.png`: no activation beam.
- `guardian-effect-active.png`: Warrior Stone at 3 seconds; neighboring stone
  remains idle.
- `guardian-effect-finished.png`: 15 seconds; beam and activation glow gone.

`ODAI_EFFECT_PREVIEW=e7bdd,3` selects a single reference and fixed activation
time for deterministic captures. It is not set for normal play.

Regression tests cover manager-bound interpolation, named sequence selection,
step visibility, end clamping, repeated playback sampling, serialization,
malformed/truncated data, and the Papyrus callback.

## Remaining separate work

Camera-facing NiBillboardNode behavior is still not retained by the flattened
static path. General HKX behavior-event routing (`PlayAnimation`, distinct from
`PlayGamebryoAnimation`) and the stones' blessing/message scripts remain outside
this sequence playback implementation. The implementation does not hardcode
stone model names or replace the authored timing with a synthetic effect.
