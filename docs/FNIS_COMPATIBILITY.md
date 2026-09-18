# FNIS-generated humanoid compatibility

Native animation is now the default; the existing interpreter is an explicit
`ODAI_SKYRIM_ANIMATION_MODE=havok` compatibility option. See
[NATIVE_CHARACTER_SYSTEM.md](NATIVE_CHARACTER_SYSTEM.md) for implemented native
features and remaining character-system work.

Status: asset and rig integration is partial; FNIS runtime compatibility is not
established. The retail behavior graph still fails execution admission. Generator
execution, creatures, first-person, paired actions, furniture coordination and
killmoves remain outside this stage.

## Implemented

- Decode x64 `hkbCharacterStringData` skeleton, behavior and animation-catalog
  references with bounded, fixup-backed reads. Preserve these references on the
  immutable animation view.
- Resolve actor animation skeletons and behavior roots from character definitions
  through virtual Data. Empty references use the retail male/female defaults;
  malformed definitions and missing explicit references produce diagnostics.
- Permit roots from different providers. Bundle inspection validates referenced
  skeletons and linked behavior dependencies instead of inferring compatibility
  from provider names.
- Use source skeleton binding indices to identify animation bones. Annotation
  labels cannot override that mapping, and an unmatched source bone cannot fall
  through to the same numeric index in the render skeleton.
- Reject duplicate/empty source bone names and invalid parent order. Report
  omitted source tracks that have no render bone; reject clips with no bound
  tracks. Unanimated render bones retain their bind transforms.
- Include character-definition, animation skeleton, loaded clip content and exact
  render-rig/skin-bind identity in the view fingerprint. This is not yet a complete
  manifest of all reachable generated dependencies.

## Outstanding acceptance work

Per-generator clocks/settings, remaining bindings, blenders, modifiers, transition
semantics and generated sequences still require implementation. Script request
integration, complete dependency manifests, required-versus-optional rig track
classification and generated-action persistence are also incomplete. Existing
save/streamed-scene layouts have not been changed by this increment.

Synthetic tests cover character references, malformed catalogs, layered roots,
missing referenced skeletons, and source-bone mapping without index substitution.
Retail male/female probes remain regression checks, not FNIS acceptance evidence.
No installed FNIS output/animation pack was found in the inspected profile. A
real-data validation profile and captures remain pending those local assets.

Field-layout reference: [serde-hkx character string schema](https://github.com/SARDONYX-sard/serde-hkx/blob/main/assets/classes/hkbCharacterStringData.json).
