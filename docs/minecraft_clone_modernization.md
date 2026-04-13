# Minecraft-Clone Modernization Roadmap

This project already has a strong rendering foundation. The next modernization wave should serve a Minecraft-like game first, not a renderer-first sandbox.

## Priorities

1. Large-world chunk streaming
2. Persistent region/chunk save format
3. Cave, ore, water, and biome-rich terrain generation
4. Broader block/item taxonomy and inventory persistence
5. Crafting, drops, and survival-oriented interaction rules
6. Basic mob/entity simulation in streamed chunks
7. Renderer work that scales with streamed worlds

## Streaming World Foundation

- Replace the current bounded world assumption with load/generate/mesh/unload around the player.
- Keep simulation deterministic inside loaded chunks.
- Prioritize explicit chunk lifecycle states over a generic job system.
- Add region-oriented persistence before adding deep gameplay loops.

## Gameplay Systems

- Expand beyond the current creative-only item set into terrain, crafted, foliage, ore, and utility blocks.
- Persist player inventory, hotbar, and world modifications.
- Add block drops and simple crafting before more advanced automation or mob systems.
- Keep controls and interaction rules aligned with Minecraft-style expectations.

## Renderer Alignment

- Optimize for chunk upload budgets, visibility stability, and streaming-scale world movement.
- Prioritize Minecraft-relevant visuals next:
  - water
  - weather and clouds
  - cave lighting readability
  - foliage/tree polish
- Keep RT shadows, GI, and ReSTIR optional and measurable rather than mandatory for core gameplay.

## Current Default

- GI surface mode should default to `ReSTIR Surface` with automatic fallback to `RT Surface`, then `Legacy`.
