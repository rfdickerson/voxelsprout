🎯 Project Vision

Voxel Factory Toy is:

A playable systems sandbox

A graphics R&D playground

A deterministic simulation testbed

A modern Vulkan renderer built for clarity

It is not:

A generic engine framework

A tech demo without gameplay

A content-heavy authored experience

An enterprise architecture experiment

🧠 Core Principles
1️⃣ Systems Over Features

Add small composable rules.
Avoid large feature drops.

2️⃣ Determinism First

Simulation must:

Run at fixed timestep

Be order-stable

Be independent of rendering

Produce identical results across runs

Rendering is allowed to be non-deterministic. Simulation is not.

3️⃣ Renderer Isolation

Only render/ may include Vulkan headers.

No Vulkan types may leak into:

sim/

world/

core/

Rendering consumes data. It does not control simulation.

4️⃣ Performance Is a Feature

Maintain:

Smooth frame pacing

Stable GPU timing

Predictable memory behavior

Clear synchronization boundaries

Avoid hidden work or implicit allocations in hot paths.

5️⃣ Readability Over Abstraction

Prefer:

Clear structs

Explicit ownership

Flat data

Small focused files

Avoid:

Deep inheritance

Over-generalization

Framework layering

Indirection for its own sake

🧩 Architecture Overview
app/      – Main loop, lifecycle
core/     – Math, time, input, grid primitives
world/    – Voxels, chunk storage, CSG
sim/      – Deterministic simulation + networks
render/   – Vulkan renderer + shaders
assets/   – Textures, SPIR-V, materials
tests/    – Deterministic unit tests

Data Flow (Strictly One Direction)
Input
↓
Simulation (fixed tick)
↓
World mutation
↓
Meshing
↓
Renderer
↓
Post-processing


No upward callbacks.
No render → sim dependencies.

🧱 Rendering Architecture Guidelines

The renderer currently includes:

Vulkan 1.3 dynamic rendering

Synchronization2

Timeline semaphores

Reverse-Z projection

Cascaded shadow maps (atlas)

SH-based ambient

Voxel GI (surface → inject → propagate)

Froxel volumetrics

SSAO

HDR + bloom + ACES

GPU timestamp profiling

Rendering Rules
1️⃣ Render passes must be explicit.

No hidden side effects between passes.

2️⃣ Storage image reads/writes require explicit barriers.

Agents must not assume implicit synchronization.

3️⃣ GPU work must be measurable.

When adding passes:

Integrate into GPU timestamp system

Expose tuning in debug UI

4️⃣ Post-processing must occur after tone mapping.

Color grading operates in LDR unless explicitly justified.

🌍 World & Simulation Rules
Voxels

Grid-aligned

Deterministic

Small (~0.25m scale)

No floating geometry

No physics-driven placement

Simulation

Fixed timestep

Graph-based transport

No dependency on rendering

No variable-rate tick logic

World Editing

Use CSG commands

Propagate AABB of changes

Minimize chunk remeshing

⚙️ Voxel GI Guidelines

Voxel GI uses:

Surface cache (RGB per face)

Inject pass (bounce albedo weighted)

Transport-aware propagation

Shared memory tiled compute

Openness-based bleed control

When modifying GI:

Preserve energy stability

Avoid increasing light bleed

Keep memory access coherent

Maintain fixed iteration count

If adding directional basis (e.g. SH), document memory impact.

🧪 Testing Expectations

Add tests when:

Introducing deterministic math utilities

Modifying grid/graph logic

Adding transport behavior

Rendering features do not require unit tests,
but must be:

Measurable

Toggleable

Stable under GPU timing panel

🚫 Non-Goals (Unless Explicitly Requested)

Do not introduce:

ECS frameworks

Job schedulers

Plugin systems

Script engines

Asset pipelines

Networking

Serialization layers

Editor frameworks

This project is intentionally single-binary and self-contained.

🤖 AI Agent Guidelines

When generating code:

Prefer

Clear C++20

Google C++ style

Small functions

Explicit lifetime

Value semantics

Stack allocation where reasonable

Avoid

Refactoring unrelated files

Introducing new abstraction layers

Changing architecture without request

Over-optimizing prematurely

Adding new subsystems

Rendering Changes

Respect existing pass structure

Do not merge passes unless requested

Maintain explicit synchronization

Keep debug UI hooks intact

Simulation Changes

Preserve determinism

Avoid floating-point drift when possible

Keep tick logic separate from rendering

If uncertain: generate the simplest correct implementation.

📊 Definition of Progress

A change is successful if:

The toy is more interactive

Systems are more emergent

Rendering is more stable or expressive

Frame pacing is unaffected

The codebase remains readable

🧭 Final Principle

This is a small, serious engine.

Add power carefully.

If complexity increases, clarity must increase with it.

Why This Version Is Better

It:

Matches your renderer’s maturity

Establishes synchronization rules

Protects deterministic simulation

Prevents AI from “engine-ifying” the project

Keeps it scalable without losing identity

Linux builds should use
cmake-build-linux

Windows builds use:
cmake-build-release