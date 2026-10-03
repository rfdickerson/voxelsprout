# ARCH-005: Separate shared geometry contracts from renderer interfaces

Status: Proposed
Priority: P2
Rank: 5

## Current structure

Character import produces GPU-compatible vertices using a type owned by a broad renderer header.

## Evidence

- [character_builder.h](../../../src/import/bethesda/character_builder.h) includes render/renderer_types.h and stores render::ImportedSkinnedMeshVertex in FalloutCharacter.
- [renderer_types.h](../../../src/render/renderer_types.h) combines that layout with presentation/pacing/diagnostic types and includes ImportedScene; it measured 817 lines at audit time.
- This is reverse header coupling between conceptual layers, not a configured linker-target cycle.

## Why it is problematic

Importer compilation inherits unrelated renderer interface changes; ownership of the shared byte layout is ambiguous.

## Impact on human development

Presentation-type edits can affect asset-import consumers and headless compilation.

## Impact on AI development and context usage

A character-import task leads into a large renderer type collection to locate a single data contract.

## Proposed architectural direction

Extract the exact shared geometry/layout contract into a small dependency-neutral header, preserving representation and compatibility aliases initially. Add targeted include-boundary checks.

## Expected blast radius

Character builder, renderer upload/skinning consumers, layout tests and public includes. No format or layout change is part of this ticket.

## Risk of changing it

Low to moderate for header ownership alone; high if GPU/serialized layout changes are mixed into it.

## Validation and acceptance criteria

- [ ] Verify size/offset/layout assertions for the extracted types.
- [ ] Run character assembly and applicable CPU/GPU skinning tests.
- [ ] Verify cooked-scene compatibility where the affected representation participates in serialization.
- [ ] Confirm importer interfaces no longer include renderer presentation headers and no new dependency cycle is introduced.

## Priority rationale

P2; ranked 5 in the [architecture improvement backlog](README.md).
This is a proposed bounded change, not authorization for a broad rewrite.
