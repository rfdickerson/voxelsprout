# AGENTS.md

This project is an experimental **voxel-based factory toy game** focused on *emergent discovery*, *mechanical intuition*, and *kid-friendly play*.
It is intentionally small, readable, and system-driven.

This document defines how **humans and AI agents (Codex / ChatGPT)** should contribute to the codebase.

---

## 🎯 Project Goals

* Build a **playable toy** quickly, not a generic engine
* Favor **systems and rules** over authored content
* Encourage **emergent discovery** (Maxis-style)
* Keep the game **non-violent** (traps, hazards, environmental challenges only)
* Maintain **smooth frame pacing** and deterministic simulation
* Keep architecture understandable by a single person in one sitting

If a change does not support these goals, it likely does not belong (yet).

---

## 🧠 Core Design Philosophy

1. **Systems > Features**
   Add simple rules that interact, not complex one-off mechanics.

2. **Data-first, not framework-first**
   Plain structs, vectors, and clear ownership beat abstraction layers.

3. **Discovery over instruction**
   The game should explain itself through behavior and visuals.

4. **No punishment loops**
   Failure should be interesting, reversible, or funny—never harsh.

5. **Elegance beats realism**
   This is a toy world, not a simulator.

---

## 🧩 High-Level Architecture

The codebase is intentionally divided into a few clear subsystems:

```
app/     – Application bootstrap & main loop
core/    – Time, input, math, and grid primitives (Cell3i / Dir6 / AABB)
world/   – Voxels, chunks, meshing, and CSG command application
sim/     – Deterministic simulation + transport network graph utilities
render/  – Rendering only (Vulkan); shading/composition only
assets/  – Textures and shaders
tests/   – CTests for deterministic utility and foundation behavior
```

### Key Rule

**Only `render/` knows about Vulkan.**
All other systems must be renderer-agnostic.

---

## 🧱 Current Baseline (Latest)

This section tracks current foundations so contributors build on the same assumptions.

* Ambient shading currently uses **vertex AO only**. SSAO passes are intentionally removed for performance.
* Pipes are currently **voxel-style cuboids** (thicker endcaps, narrower transfer section).
* Pipe endpoints show endcaps only when attaching to voxels; pipe-to-pipe continuation should not add extra endcaps.
* Pipe meshes avoid coplanar seam faces to prevent z-fighting/flickering between connected segments.
* The transfer section is visually emphasized as bright fluid (pink tint) to improve readability.
* Core transport/world modeling now has shared foundation headers:
  * `core/Grid3.hpp`
  * `sim/NetworkGraph.hpp`
  * `sim/NetworkProcedural.hpp`
  * `world/Csg.hpp`

---

## 🔁 Data Flow (One Direction Only)

```
Input
  ↓
Game rules
  ↓
Simulation (fixed tick)
  ↓
World (voxel changes)
  ↓
Meshing
  ↓
Renderer
```

* No circular dependencies
* No callbacks upward
* Systems communicate via data, not control flow

---

## 🧱 Voxels & Scale

* Voxels are **smaller than the character** (~0.25m per voxel)
* Voxels are the **construction unit**, not the body unit
* Early game uses **full voxels only** (no slabs initially)
* Shape variants (ramps/stairs) may be added later as block metadata

Structural logic must remain grid-aligned and deterministic.

---

## ⚙️ Simulation Rules

* Simulation runs at a **fixed timestep** (e.g. 30 Hz)
* Rendering may run faster or slower
* Simulation must be:

    * Deterministic
    * Order-independent where possible
    * Independent of rendering

Belts, machines, and mechanical systems are modeled as **graphs**, not physics.

---

## 🧰 Procedural + CSG Foundations

When adding belts, pipes, rails, and trains, prefer extending these foundations instead of creating parallel systems.

* Use `Cell3i` + `Dir6` for all grid-aligned placement/connectivity logic.
* Use `NetworkGraph` (`Socket`, `EdgeSpan`, nodes/edges) as the base structure for transport networks.
* Use deterministic helper utilities (`neighborMask6`, span rasterization, join classification, transform quantization).
* Use CSG commands (`AddSolid`, `SubtractSolid`, `PaintMaterial`) to stamp world edits in bounded regions.
* Preserve and propagate touched AABBs so chunk remeshing remains minimal.

---

## 🏭 Factories & Discovery

* Machines should be **built from parts**, not dropped as prefabs
* Prefer **properties** (hot, heavy, color, rotation) over hard recipes
* Machines do not “fail” — they *behave*
* Visual feedback replaces error messages

If a child can predict what will happen by watching, the design is correct.

---

## 🚫 Explicit Non-Goals (for now)

Do **not** add unless there is a clear, immediate need:

* ECS frameworks
* Job systems
* Plugin architectures
* Render graphs
* Asset pipelines
* Save/load systems
* Networking
* Combat systems

These can be revisited later if the toy proves fun.

---

## 🤖 Guidelines for AI Agents (Codex / ChatGPT)

When generating or modifying code:

* Prefer **simple, readable C++** over clever abstractions
* Keep files **small and focused** (<500 lines)
* Avoid introducing new subsystems without explicit instruction
* Do not refactor unrelated code
* Add comments explaining *intent*, not implementation trivia
* Preserve deterministic behavior
* Add/update CTests when introducing new deterministic utilities

If unsure, ask or generate the **simplest possible version**.

---

## 🧪 Definition of Progress

A change is successful if:

* The game is more *playable* than before
* A new interaction can be *discovered*, not explained
* Frame pacing remains smooth
* The codebase is still easy to reason about

If the project feels boring internally but delightful to play, that’s ideal.

---

## 🧭 Final Principle

> **If the code feels like a toy box instead of a toolbox, we’re doing it right.**
