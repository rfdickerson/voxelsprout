# ARCH-002: Extract testable application transition orchestration

Status: Proposed
Priority: P1
Rank: 2

## Current structure

BethesdaApp coordinates scenes, configuration, conversations, puzzles, actors, transitions, physics, saves, UI and rendering.

## Evidence

- [bethesda_app.cc](../../../src/games/bethesda/bethesda_app.cc) measured 16,552 lines / 857,546 bytes at audit time; bethesda_app.h measured 1,245 lines.
- beginConversation, completeDoorTransition and the BethesdaSession::advance call demonstrate distinct orchestration responsibilities.
- completeDoorTransition coordinates destination construction, renderer upload, residency and collision changes, explicitly preserving the old space when preparation fails.

## Why it is problematic

Failure and ordering rules are embedded in the application rather than a bounded transaction that can be tested independently.

## Impact on human development

A transition fix requires reasoning across loading, allocation, actors and physics; unrelated edits share one large file.

## Impact on AI development and context usage

Complete file context would be roughly 214,000 tokens. Symbol lookup alone cannot establish all state mutations relevant to a transition.

## Proposed architectural direction

Extract a door-transition coordinator with explicit prepare, commit and failure outcomes. Keep BethesdaApp as the composition point; extracting member functions into files alone is insufficient.

## Expected blast radius

Transition methods, streamer/renderer adapters, collision/residency integration and focused tests. Start with one transition transaction.

## Risk of changing it

Moderate for a bounded extraction, high for a wholesale split. Preserve the existing old-space safety guarantees and threading rules.

## Validation and acceptance criteria

- [ ] Inject destination-build and renderer-upload failures; assert that the current space remains usable with unchanged residency/collision state.
- [ ] Exercise successful interior/exterior transitions, actor and collision synchronization, and cleanup.
- [ ] Use the same fixtures through coordinator tests and application integration where practical.
- [ ] Record files/context needed for a representative transition fix before and after extraction.

## Priority rationale

P1; ranked 2 in the [architecture improvement backlog](README.md).
This is a proposed bounded change, not authorization for a broad rewrite.
