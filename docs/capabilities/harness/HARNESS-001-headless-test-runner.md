# HARNESS-001: Headless Engine Test Runner

Status: Implemented

## Goal

Provide a deterministic, non-interactive way to run Iridius engine
scenarios for automated testing, CI, and agent-driven verification.

The runner must allow tests and engineering agents to start the engine,
load controlled content, simulate engine execution, inspect resulting
state, and terminate without human interaction.

This capability is foundational infrastructure for later gameplay,
world, rendering, physics, and compatibility tests.

## Motivation

Currently, validating many engine behaviors requires launching Iridius
interactively and manually observing the result.

Codex and CI need a way to execute engine behavior and obtain
machine-readable evidence about the result.

The intended workflow should eventually support operations conceptually
similar to:

    iridius-test <scenario>

The exact CLI and internal architecture are implementation decisions.

## Required Behavior

### HARNESS-001-A: Non-interactive execution

The test runner can start Iridius without requiring user interaction.

It must be possible to:

1. initialize the required engine subsystems
2. load a controlled test fixture or scene
3. advance simulation
4. terminate automatically

Tests must not require keyboard/mouse input.

### HARNESS-001-B: Deterministic simulation

The runner must support deterministic execution suitable for regression
tests.

At minimum, tests must be able to control sources of nondeterminism
relevant to the scenario, such as:

- simulation timestep
- random seed
- number of simulation steps or frames

Running the same deterministic scenario repeatedly should produce
equivalent observable state.

### HARNESS-001-C: Test fixture loading

The runner can load small synthetic Iridius test fixtures without
requiring the full game.

Fixtures should be suitable for testing isolated engine behavior.

Examples include:

- empty world
- single static object
- simple physics scene
- player + door
- two connected cells
- NPC navigation scene

The runner should not require production Morrowind data for ordinary
engine regression tests.

### HARNESS-001-D: Machine-readable state

A scenario can expose relevant resulting engine state in a
machine-readable form.

The mechanism should support assertions against values such as:

- entity transforms
- active world/cell
- loaded resources
- physics state
- animation state
- gameplay state

The representation and serialization format are implementation
decisions.

### HARNESS-001-E: Automated pass/fail

The runner must return a process exit status suitable for CTest/CI:

    0     scenario passed
    != 0  scenario failed

Failures should provide enough diagnostic information for a developer
or coding agent to investigate the cause.

### HARNESS-001-F: CTest integration

At least one headless engine scenario must execute through the project's
normal automated test infrastructure.

For example:

    ctest --test-dir build -R Headless

must be capable of exercising the runner without manual interaction.

## Initial Scope

HARNESS-001 does NOT need to provide:

- visual regression testing
- GPU screenshot comparison
- OpenMW differential testing
- complete gameplay scripting
- performance benchmarking
- arbitrary input recording/playback
- network testing

Those should be separate capabilities built on top of the runner.

HARNESS-001 should establish the smallest useful foundation for future
scenario-driven engine testing.

## Example Scenario

The initial vertical slice should demonstrate something equivalent to:

1. start the engine
2. load a synthetic scene containing one entity
3. advance simulation by a known amount
4. inspect the entity's resulting state
5. verify the expected result
6. exit automatically

The exact scene and behavior should be chosen based on existing Iridius
architecture.

## Architectural Constraints

- Reuse the production engine implementation rather than creating a
  separate mock engine.
- Avoid duplicating subsystem initialization logic solely for tests.
- Test-only infrastructure must not leak into normal runtime behavior.
- Headless execution must not require presentation to a window.
- Preserve existing interactive engine behavior.
- Prefer exposing testable engine interfaces over adding test-specific
  hacks to production systems.

## Verification

The capability is complete when:

1. Iridius can execute at least one engine scenario non-interactively.
2. The scenario loads a controlled synthetic fixture.
3. Simulation advances deterministically.
4. Resulting state can be inspected programmatically.
5. A deliberately incorrect expectation causes the test to fail.
6. The correct expectation causes the test to pass.
7. The test runs through CTest.
8. Existing tests continue to pass.

## Definition of Done

- headless runner exists
- at least one synthetic fixture exists
- deterministic execution is demonstrated
- machine-readable or programmatically inspectable state is demonstrated
- CTest integration exists
- failure diagnostics are useful
- existing interactive Iridius executable continues to function
- relevant documentation is updated

## Implementation

`odai_headless` runs the production `BethesdaSession` and its Jolt physics
world directly. The interactive `GameApp` remains the GLFW/Vulkan presentation
host; it is not needed to initialize or advance the simulation session.

The initial fixture format is a deliberately small JSON actor scene. It specifies
an actor's stable identity, starting position, desired velocity, random seed,
fixed timestep, and step count. An optional `expect` block checks the final X
range, falling Y position, applied world commands, and physics/world agreement.
Without `expect`, the runner completes and reports state for inspection. It uses one
`BethesdaSession::advance()` call per fixed step and fails if a step is dropped
or produces a diagnostic. It writes one JSON result to stdout with the final
transform, physics position, active space/cell, tick count, random state, applied
world-command count, and deterministic state
hash. Failures also write a diagnostic to stderr and exit nonzero.

Build and run the fixture without retail game data:

```bash
cmake --build --preset linux-vcpkg --target odai_headless
./build-linux/odai_headless tests/fixtures/harness/one_actor.json
ctest --test-dir build-linux --output-on-failure -R odai_headless
```

For CI that needs only the simulation runner:

```bash
cmake --preset linux-vcpkg-headless
cmake --build --preset linux-vcpkg-headless --target odai_headless
ctest --test-dir build-linux-headless --output-on-failure
```

The headless preset omits the presentation feature in `vcpkg.json`, so it does
not install GLFW, ImGui, Vulkan, Miniaudio, or Stb, and it does not require
Slang or configure the renderer. The replay test runs the fixture
three times and compares the complete JSON result. It verifies that seed,
timestep, and step count control the outcome, that zero movement intent fails
the movement expectation, and that an intentionally wrong expectation exits
nonzero with a useful error. It also checks inspection without expectations.
This foundation does not initialize rendering, audio, or
content archives; scenarios that require those systems need their own focused
extensions.
