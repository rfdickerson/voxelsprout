# Roadmap

## Runtime

- Continue unifying TES3/TES4/Fallout/TES5 record behavior behind the existing
  archive, plugin-load-order, NIF, cell-streaming, actor, dialogue, and weather code.
- Keep streamed cells and cooked `ImportedScene` files serialization-compatible.
- Improve conditional real-data smoke coverage without redistributing game data.

## Rendering

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
