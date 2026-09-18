# Public Skyrim release checklist

Status: **not release-qualified**. Packaging does not change this status.

- [ ] Fresh-profile Riverwood → Bleak Falls → Whiterun route, normal player input only.
- [ ] Opening conversations, entrance puzzle, Arvel encounter, authored triggers,
      natural boss residency, Golden Claw and Dragonstone hand-ins.
- [ ] Early-acquired Dragonstone dialogue branch.
- [ ] Save/reload across dialogue, doors, puzzles, combat, loot and both hand-ins.
- [ ] Two-hour traversal/eviction/revisit soak without crashes, validation errors,
      duplicate loot, lost state or continuing memory growth.
- [ ] Optimized AMD and NVIDIA measurements: actual framebuffer size, p95/p99,
      worst stall, peak CPU/GPU memory; stable 16.67 ms frame delivery target.
- [ ] Fresh-user installation on a machine without the checkout or SDK.
- [ ] Dependency notices, asset provenance and reachable-history audit resolved.
- [ ] Packaged archive checked for local game data and accompanied by SHA-256.

Use JK’s Skyrim + SMIM for local scene runs. Retain the 768×432 logical window,
native-DPI framebuffer and render scale 1. Keep game/mod data and capture evidence
local. Report exact content fingerprints and hardware with every acceptance run.
No injected quest stage, actor, inventory or residency can count as route evidence.

See SKYRIM_FIRST_RUNTIME.md for implementation blockers. Headless scenario probes
remain fixture-assisted and must keep `release_gate_passed` false.

The fixture-assisted probe can use the same mod profile as the runtime:

```sh
odai_bethesda_probe /path/to/Skyrim/Data --scenario-check skyrim-bleak-falls \
  --profile /path/to/profile.json
```

Its report identifies the content fingerprint. This remains a fragment test, not
a continuous-route driver.

## Validation of release infrastructure

Local Debug and RelWithDebInfo builds pass all 41 registered CTest targets.
The installed package passes a synthetic imported-scene Vulkan render/capture
from a temporary working directory on Intel LNL/Mesa 26.1.8. The archive contains
75 compiled shaders and passes the content-format exclusion check.
The setup window was visually checked using temporary Tk dependencies.
A JK’s Skyrim + SMIM Riverwood scenario capture starts successfully at a 768×432
logical/native framebuffer on this display. Its profile-aware fragment probe
passes with zero unresolved calls, but still reports runtime blockers and
`release_gate_passed: false`. These checks do not satisfy the unchecked route,
soak, AMD/NVIDIA, or fresh-machine gates above. CI configuration is updated;
remote CI results have not been observed in this session.
