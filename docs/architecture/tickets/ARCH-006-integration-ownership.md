# ARCH-006: Give streaming and navigation integration explicit build ownership

Status: Proposed
Priority: P2
Rank: 6

## Current structure

Renderer-aware streaming lives under import but compiles into odai; navigation/actor integration sources are compiled separately into application and test targets.

## Evidence

- [cell_streamer.h](../../../src/import/bethesda/cell_streamer.h) accepts Renderer& in update and load application routines; its implementation is application-owned.
- [CMakeLists.txt](../../../CMakeLists.txt) compiles navigation_world.cc, navigation_simulation.cc, bethesda_collision.cc and actor integration sources directly into selected tests.
- Generated source ownership maps record these directory/target exceptions.

## Why it is problematic

Directory ownership is misleading; tests duplicate production source lists instead of consuming an explicit integration boundary.

## Impact on human development

Developers must inspect build declarations to establish ownership and whether test compilation matches production compilation.

## Impact on AI development and context usage

Agents encounter multiple compilation sites; testing streaming naturally pulls toward the concrete rendering interface.

## Proposed architectural direction

Establish focused integration targets, separating simulation/navigation from presentation-dependent streaming. Introduce a small chunk-residency sink only where it enables deterministic streaming tests; physical moves are optional.

## Expected blast radius

CMake targets, streaming interfaces/adapters, navigation integration and existing tests. Keep the residency planner independent.

## Risk of changing it

Moderate: preserve compile definitions, main-thread renderer calls, worker isolation and shutdown ordering.

## Validation and acceptance criteria

- [ ] Application and affected tests consume the same production integration targets rather than duplicating source lists.
- [ ] Use deterministic sink tests for failed uploads, eviction and completion ordering.
- [ ] Retain navigation/actor movement tests and appropriate runtime streaming evidence.
- [ ] Regenerate maps and verify ownership/dependency direction, without introducing generic plugin or render-graph machinery.

## Priority rationale

P2; ranked 6 in the [architecture improvement backlog](README.md).
This is a proposed bounded change, not authorization for a broad rewrite.
