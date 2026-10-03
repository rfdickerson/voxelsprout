# Rendering

| Capability | Description | Status |
|---|---|---|
| RENDER-TERRAIN-001 | Render Morrowind exterior LAND cell | Implemented |
| RENDER-TERRAIN-002 | Render Morrowind LAND terrain textures | Implemented |
| RENDER-TEX-001 | TES3 object and terrain textures, including high-resolution replacements | Implemented — DDS/TGA/BMP, authored UV/clamp/tint/alpha, full/reduced resolution, 2K/4K/8K pixels, cache/residency and strict zero-missing audits pass; see [verification](validation/RENDER-TEX-001.md). |
| RENDER-WATER-001 | Basic world water rendering | Partial — exterior/interior water, renderer toggle, deterministic animation, packaged normal resource, terrain depth checks, validation smoke, and performance baseline pass; a portable visual golden with clear normal and specular detail remains. |

# Performance

| Capability | Description | Status |
|---|---|---|
| PERF-001 | 120 Hz frame pacing | Partial — strict Balmora p95/p99 and long-frame limits fail; streamed cell publication and presentation jitter remain. |
| PERF-002 | Multithreaded game updates | Partial — cell collision, generated navigation, and vertex normal/color encoding prepare on workers. Optimized Balmora tick p99 is 7.20 ms (2 ms gate); cell upload still reaches 10.91 ms and gameplay upsert 4.86 ms (2 ms step gate). Arena growth, bounded publication/cleanup, steady-state job overlap, and cross-game deterministic verification remain. |

# Physics

| Capability | Description | Status |
|---|---|---|
| PHYS-001 | Static world collision and headless ray/shape queries | Implemented |
| PHYS-002 | Morrowind exterior LAND terrain collision | Implemented |
| PHYS-008 | Morrowind player jump height | Implemented |

# Game Mechanics

| Capability | Description | Status |
|---|---|---|
| MECH-001 | Player inventory: view and drop items | Partial — gameplay drop and inventory UI are wired; spawned items still need imported-scene rendering and automated UI input verification. |
| MECH-002 | Play and complete Morrowind's main quest | Partial — live player inventory now drives TES3 dialogue item conditions, including scripted transfers; character generation, quest-required gameplay behavior, and an end-to-end base-game playthrough remain. |
| MECH-TES3-001 | Morrowind new game and character generation | Partial — fresh launch runs `CharGen` aboard the prison ship and opening state survives save/load; character menus, authored progression, camera placement, package gate, and Seyda Neen completion remain. |
| MECH-TES3-002 | Main quest dialogue and journal conditions | Planned |
| MECH-TES3-003 | Main quest MWScript commands and effects | Planned |
| MECH-TES3-004 | Main quest items and equipment | Planned |
| MECH-TES3-005 | Main quest encounters and finale | Planned |

## Engineering Harness

| Capability | Description | Status |
|---|---|---|
| HARNESS-001 | Headless engine test runner | Implemented |
| HARNESS-002 | Scenario definition format | Implemented |
| HARNESS-003 | State snapshot/dump | Implemented |
| HARNESS-004 | Visual regression testing | Implemented |
| HARNESS-005 | Performance scenarios | Implemented |
