# ARCH-003: Narrow session mutation and presentation callback contracts

Status: Proposed
Priority: P1
Rank: 3

## Current structure

BethesdaSession provides domain operations alongside mutable world, physics, clock and VM access, mutable restore containers and presentation callbacks.

## Evidence

- [bethesda_session.h](../../../src/bethesda/bethesda_session.h) exposes mutable world(), physics(), clock(), papyrus() and tes3(), plus *ForRestore containers.
- [bethesda_app.cc](../../../src/games/bethesda/bethesda_app.cc) queues and immediately applies world commands during puzzle setup and installs an effect callback capturing load-order and renderer state.
- [bethesda_session.cc](../../../src/bethesda/bethesda_session.cc) invokes BeforeSimulationTick before simulateTick. Twenty indexed files directly included the session header at audit time.

## Why it is problematic

Callers can bypass domain timing/validation rules. Some runtime behavior depends on callbacks installed during presentation initialization.

## Impact on human development

Developers must inspect callers to determine mutation visibility and the services an operation can invoke.

## Impact on AI development and context usage

An inventory/dialogue change can require physics, VM, application and persistence context to assess side effects.

## Proposed architectural direction

Introduce narrow commands and read views, starting with inventory/dialogue. Constrain restore mutation to an explicit restore operation. Document presentation requests/events and retain synchronous callbacks where their results are required.

## Expected blast radius

Selected session callers, save restoration, native bindings and headless/application adapters; migrate one domain at a time.

## Risk of changing it

High around tick timing, script return values and save compatibility. Do not indiscriminately replace synchronous callbacks with asynchronous events.

## Validation and acceptance criteria

- [ ] Compare replay hashes, command visibility by tick and save round trips before/after migration.
- [ ] Verify affected native return values and failure behavior.
- [ ] Test the chosen domain without installing renderer callbacks.
- [ ] Inventory mutable callers and show that selected domain callers use the narrow contract; explicitly document justified remaining escape hatches.

## Priority rationale

P1; ranked 3 in the [architecture improvement backlog](README.md).
This is a proposed bounded change, not authorization for a broad rewrite.
