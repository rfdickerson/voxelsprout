# Rendering

| Capability | Description | Status |
|---|---|---|
| RENDER-TERRAIN-001 | Render Morrowind exterior LAND cell | Implemented |
| RENDER-TERRAIN-002 | Render Morrowind LAND terrain textures | Implemented |
| RENDER-TEX-001 | TES3 object and terrain textures, including high-resolution replacements | Implemented — DDS/TGA/BMP, authored UV/clamp/tint/alpha, full/reduced resolution, 2K/4K/8K pixels, cache/residency and strict zero-missing audits pass; see [verification](validation/RENDER-TEX-001.md). |
| RENDER-WATER-001 | Basic world water rendering | Partial — exterior/interior water, renderer toggle, deterministic animation, packaged normal resource, terrain depth checks, validation smoke, and performance baseline pass; a portable visual golden with clear normal and specular detail remains. |
| RENDER-GI-001 | Global illumination | Partial — deterministic linear red/blue transfer, blocker, off-screen retention, light response, eviction and grid movement pass Vulkan validation; full optimized CTest 60/60. TES3 AMBI and compressed diffuse color feed the existing volume path. Comparison probes use pinned fly cameras at native DPI. Real-scene source/receiver visual acceptance, traversal/transition history evidence, and the full PERF-001 pacing gate remain; see the capability for local measurements. |

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
| MECH-001 | Player inventory: view and drop items | Implemented |
| MECH-002 | Play and complete Morrowind's main quest | Partial — live inventory drives TES3 dialogue item conditions and scripted transfers; opening guard movement and tutorial gates pass focused tests, and a base-game ship start reaches name entry. Character-generation release, quest-required gameplay behavior, and an end-to-end playthrough remain. |
| MECH-TES3-001 | Morrowind new game and character generation | Partial — fresh launch runs `CharGen`, character menus and persistence are wired, BODY appearance choices and authored input locks are available. Guard coordinate movement/arrival and tutorial gates have synthetic coverage; corrected ship floor placement reaches retail name entry; 66 CTest targets pass. Ordinary-input progression through the Census Office, package gate, camera facing, and Seyda Neen completion remain. |
| MECH-TES3-002 | Main quest dialogue and journal conditions | Partial — authored filters, faction/clothing/stat inputs, result ordering, journal persistence, and a 35-step retail opening replay pass; normal leveling and the remaining retail main-quest gate trace remain. |
| MECH-TES3-003 | Main quest MWScript commands and effects | Partial — the 1,609-program local route closure has no blocked operations or unresolved starts; eight isolated base-game opening, Corprus, artifact, Heart, and finale checkpoints and 60 CTest tests pass. Dialogue-result rollback spans MessageBox suspension. Event-driven standard-route verification remains. |
| MECH-TES3-004 | Main quest items and equipment | Partial — scripted world pickup, duplicate prevention, inventory equip, artifact script events, and save/load fixtures pass; normal-input route acquisition, reading, and final use remain. |
| MECH-TES3-005 | Main quest encounters and finale | Planned |
| MECH-TES3-006 | Morrowind player skill advancement and leveling | Partial — progression rules, live run/jump/melee uses, training/books, rest/attribute UI, and save/load pass synthetic verification; 65 CTest targets and local retail record initialization pass. Remaining skill-use producers, complete rest/service parity, and normal-input level-3/Caius verification remain. |
| MECH-TES3-007 | Morrowind alchemy: brewing, ingredient eating, and potion use | Planned |

# Mod Content

| Capability | Description | Status |
|---|---|---|
| MOD-TES3-001 | Tamriel Rebuilt released mainland regions and quests | Partial — installed 25.08.12 A/B profiles pinned and structurally inventoried; all cells/quests remain UNVERIFIED, strict scripts fail, and gameplay/classification/process-restart validation is incomplete. [Certification](validation/MOD-TES3-001.md). |

## Engineering Harness

| Capability | Description | Status |
|---|---|---|
| HARNESS-001 | Headless engine test runner | Implemented |
| HARNESS-002 | Scenario definition format | Implemented |
| HARNESS-003 | State snapshot/dump | Implemented |
| HARNESS-004 | Visual regression testing | Implemented |
| HARNESS-005 | Performance scenarios | Implemented |
| HARNESS-006 | Headless UI testing | Implemented |
| HARNESS-007 | Bethesda probe artifact and record validation | Implemented — typed record and BSA/NIF/DDS/animation assertions, malformed-input checks, deterministic reports, and local retail validation pass. |
