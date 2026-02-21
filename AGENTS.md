# 🎯 Project Vision

Cloud Path Tracer is:

A physically-based volumetric rendering lab
A Vulkan compute path tracing research platform
A multiple-scattering experimentation sandbox
A progressive offline renderer with real-time UI control

It is not:

A generic engine framework
A gameplay sandbox
A voxel simulation project
An over-abstracted graphics architecture experiment
A production content pipeline

---

# 🧠 Core Principles

## 1️⃣ Physics Over Hacks

Prioritize physically-correct light transport.

Prefer:

Radiative transfer equation fidelity
Explicit phase functions
Proper transmittance evaluation
Energy conservation

Avoid:

Ad-hoc lighting models
Baked lighting shortcuts
Unjustified approximations

Approximation is allowed only if documented and measurable.

---

## 2️⃣ Progressive Monte Carlo First

Rendering must:

Be unbiased by default
Improve monotonically with more samples
Use explicit accumulation
Reset cleanly when parameters change

Denoising is optional and post-process only.

No hidden caching that invalidates correctness.

---

## 3️⃣ Renderer Isolation

Only `render/` may include Vulkan headers.

No Vulkan types may leak into:

core/
math/
scene/
medium/

Rendering consumes scene data.
It does not mutate it.

---

## 4️⃣ Explicit Synchronization

All GPU work must be:

Explicitly dispatched
Explicitly synchronized
Explicitly measurable

No implicit layout transitions.
No hidden barriers.
No implicit descriptor updates.

Storage image and buffer usage requires correct memory barriers.

Timeline semaphores must remain predictable.

---

## 5️⃣ Readability Over Cleverness

Prefer:

Small self-contained files
Explicit integrator structure
Clear math
Flat data
Deterministic RNG per pixel

Avoid:

Metaprogramming
Template-heavy abstractions
Inheritance hierarchies
Over-generalized renderer layers
Indirection without benefit

The integrator must remain understandable.

---

# 🧩 Architecture Overview

```
app/        – Window, lifecycle, frame loop
core/       – Math, RNG, camera, utilities
scene/      – Density volumes, bounds, lights
medium/     – Phase functions, transmittance logic
render/     – Vulkan pipelines + compute passes
shaders/    – Slang compute shaders
assets/     – Density textures, reference data
tests/      – Math & sampling validation
```

---

# 📡 Data Flow (Strictly One Direction)

User Input
↓
Parameter Update
↓
Scene / Medium State
↓
Path Tracer Dispatch
↓
Accumulation Buffer
↓
Tone Mapping
↓
UI Overlay
↓
Present

No upward dependencies.
No render → scene mutation.

---

# ☁️ Cloud Rendering Guidelines

## Medium Model

Clouds must use:

Henyey–Greenstein phase function
High albedo (~0.99)
Delta tracking for heterogeneous density
Next-event estimation for sun

Multiple scattering is required.

Single-scatter-only modes must be toggleable for debugging.

---

## Density Volumes

Density may be:

Procedural (Worley + FBM + gradient shaping)
Precomputed 3D textures
OpenVDB (optional future)

Density must:

Be bounded in AABB
Be normalized and scaled explicitly
Avoid uniform mid-density volumes

Silhouette shaping is preferred over raw noise.

---

## Integrator Structure

The integrator must clearly separate:

Ray/AABB intersection
Free-flight sampling
Real vs null collisions
Direct lighting estimation
Phase sampling
Throughput update
Russian roulette
Accumulation

No monolithic shader functions.

---

# ⚙️ Rendering Architecture Guidelines

The renderer currently includes:

Vulkan 1.3
Dynamic rendering
Synchronization2
Timeline semaphores
Compute-based path tracing
HDR accumulation buffer
Filmic tone mapping
GPU timestamp profiling

---

## Rendering Rules

### 1️⃣ All passes must be explicit.

CloudPathTrace.compute
ToneMap.compute
UI composite
Present

No hidden side effects between passes.

---

### 2️⃣ Progressive accumulation must be correct.

Accumulation must:

Average samples correctly
Reset when parameters change
Avoid floating precision drift

Frame index must be explicit.

---

### 3️⃣ GPU work must be measurable.

All compute passes must:

Integrate into timestamp profiling
Expose iteration count and depth in UI
Allow toggling for debugging

---

### 4️⃣ No silent performance regressions.

Maintain:

Stable memory usage
Predictable register pressure
Reasonable occupancy
Clear resource lifetimes

Avoid hidden allocations in frame loop.

---

# 🧪 Testing Expectations

Add tests when:

Implementing RNG
Implementing phase sampling
Implementing free-flight sampling
Implementing transmittance estimation
Adding mathematical utilities

Rendering output does not require golden-image tests,
but numerical components must be validated.

---

# 🚫 Non-Goals (Unless Explicitly Requested)

Do not introduce:

ECS frameworks
Scene graphs
Asset pipelines
Plugin systems
Job schedulers
Scripting systems
Networking
Editor tooling
Serialization layers

This is a focused rendering lab, not a general engine.

---

# 🤖 AI Agent Guidelines

When generating code:

Prefer:

Clear C++20
Google C++ style
Small functions
Explicit lifetime
Stack allocation where reasonable
Deterministic RNG design
Value semantics

Avoid:

Refactoring unrelated files
Introducing new architectural layers
Changing synchronization model
Over-optimizing prematurely
Merging compute passes without request

---

## Rendering Changes

Respect:

Explicit pass structure
Explicit barriers
Existing descriptor layout philosophy
GPU timing integration

Do not collapse stages unless asked.

---

## Integrator Changes

Preserve:

Energy conservation
Correct PDF handling
Unbiased Monte Carlo estimates
Correct Russian roulette logic

If adding variance reduction techniques, document bias properties.

---

# 📊 Definition of Progress

A change is successful if:

Noise decreases per sample
Physical correctness increases
Image quality improves
Frame pacing remains stable
Code readability improves

---

# 🧭 Final Principle

This is a small, serious renderer.

Add capability carefully.

If complexity increases, clarity must increase with it.

---

Linux builds use:

```
cmake-build-linux
```

Windows builds use:

```
cmake-build-release
```
