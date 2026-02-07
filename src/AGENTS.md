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
core/    – Time, input, logging, small math helpers
world/   – Voxels, chunks, spatial data
sim/     – Deterministic simulation (belts, items, machines)
render/  – Rendering only (Vulkan later)
game/    – Game rules, block/item/machine definitions
assets/  – Textures and shaders
```

### Key Rule

**Only `render/` knows about Vulkan.**
All other systems must be renderer-agnostic.

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
