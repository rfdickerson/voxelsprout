# ARCH-008: Partition Bethesda probe commands and publish dispatch metadata

Status: Proposed
Priority: P2
Rank: 8

## Current structure

Asset inspection, record checks, dialogue/quest traces and scenario commands share one main translation unit.

## Evidence

- [bethesda_probe_main.cc](../../../src/tools/bethesda_probe_main.cc) measured 9,029 lines / 470,644 bytes at audit time.
- TES3 command dispatch and numerous mode branches coexist with implementations; separate character/asset coverage helpers demonstrate existing partial decomposition.

## Why it is problematic

The tool intended to explain runtime behavior requires substantial exploration to locate command ownership and shared setup.

## Impact on human development

Command edits increase compilation/merge costs and risk unrelated CLI behavior.

## Impact on AI development and context usage

Finding the right evidence command can lead into a source file of approximately 118,000 tokens.

## Proposed architectural direction

Extract bounded command families, starting with TES3 records/scripts/dialogue. Keep explicit static dispatch and shared argument/profile parsing. Publish command-to-handler and output-schema metadata.

## Expected blast radius

Probe sources, CMake and CLI contract tests; runtime libraries should remain unchanged. Do not add dynamic engine plugins or a generic service registry.

## Risk of changing it

Moderate: preserve argument interpretation, output schemas, exit codes and profile resolution.

## Validation and acceptance criteria

- [ ] Run synthetic command and existing probe-record tests.
- [ ] Check malformed inputs, dispatch/help consistency, exit codes and output schema compatibility.
- [ ] Ensure command metadata identifies concrete handlers and validation fixtures.
- [ ] Measure files/context required to locate and modify a representative command; real-data probes remain optional and local.

## Priority rationale

P2; ranked 8 in the [architecture improvement backlog](README.md).
This is a proposed bounded change, not authorization for a broad rewrite.
