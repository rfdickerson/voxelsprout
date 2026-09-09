# Roadmap

## Runtime

- Continue unifying TES3/TES4/Fallout/TES5 record behavior behind the existing
  archive, plugin-load-order, NIF, cell-streaming, actor, dialogue, and weather code.
- Keep streamed cells and cooked `ImportedScene` files serialization-compatible.
- Improve conditional real-data smoke coverage without redistributing game data.

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
- Build party and tactical-combat systems directly on the retained animation,
  dialogue, actor-import, and GPU-skinning foundations.
