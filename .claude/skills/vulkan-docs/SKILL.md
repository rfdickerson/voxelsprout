---
name: vulkan-docs
description: Look up the latest Vulkan documentation to find the best modern Vulkan feature, extension, or API pattern for a rendering task in src/render/. Use this whenever the user asks how to implement something in Vulkan, wants to know "what's the modern way" to do a GPU task (synchronization, descriptor management, dynamic rendering, ray tracing, memory allocation, pipelines), is debugging a Vulkan validation error, or is deciding between an older and newer Vulkan API path. Trigger even if the user doesn't say "Vulkan" explicitly but describes a rendering/GPU problem inside render/backend/vulkan/ — e.g. "how should I set up shadow pass barriers", "what's the best way to manage descriptor sets now", "is there a newer way to do this than VkRenderPass". Always check the live docs at docs.vulkan.org rather than relying on trained-in knowledge, since Vulkan's recommended patterns (dynamic rendering, synchronization2, descriptor buffers, etc.) have shifted significantly and stale knowledge leads to recommending deprecated approaches.
---

# Vulkan Docs Lookup

Vulkan's "modern" API surface has moved fast (dynamic rendering, synchronization2,
timeline semaphores, descriptor buffers, buffer device address, maintenance
extensions). Trained-in knowledge skews toward older patterns (VkRenderPass,
old-style barriers, descriptor pools as the only option). Before recommending
or implementing a Vulkan approach, check the live guide instead of guessing.

## Workflow

1. **Identify the topic.** Narrow the task to a Vulkan subsystem: synchronization,
   rendering/render passes, descriptors, memory/allocation, pipelines, ray
   tracing, multithreading, etc.

2. **Fetch the relevant guide page(s).** Start from the index and drill into
   the matching chapter — don't fetch the whole site.
   - Index: `https://docs.vulkan.org/guide/latest/index.html`
   - Use WebFetch on the index first if you don't already know which chapter
     covers the topic, then WebFetch the specific chapter page(s) it links to.
   - Common chapters worth knowing by name: `synchronization.html`,
     `extensions/dynamic_rendering.html`, `descriptor_pool.html` /
     `extensions/descriptor_buffer.html`, `memory_allocation.html`,
     `ray_tracing.html`, `wsi.html`, `shader_memory_layout.html`.
   - If the index structure has changed, just follow the links present on the
     page rather than assuming the old URL still exists.

3. **Prefer the extension/"modern" path when the guide recommends one.**
   The guide generally calls out where an older pattern has been superseded
   (e.g. `VK_KHR_dynamic_rendering` over `VkRenderPass`+`VkFramebuffer`,
   `VK_KHR_synchronization2` over the original barrier API). Recommend the
   modern path unless there's a concrete reason in this codebase to avoid it
   (e.g. driver/platform support, or an existing pattern already used
   consistently across `src/render/backend/vulkan/`).

4. **Check what this codebase already does before proposing a change.**
   Grep `src/render/backend/vulkan/` for the existing pattern (e.g. is
   dynamic rendering already in use? what sync approach do the `frame_*.cc`
   files use?) so the suggestion is consistent with existing code rather than
   introducing a second competing style. Only propose a switch when there's a
   real benefit (correctness, simplicity, perf) — don't churn working code
   just because the docs mention a newer option.

5. **Respect the project's Vulkan boundary.** Per `AGENTS.md` / `CLAUDE.md`,
   only `src/render/` may include Vulkan headers — any implementation lands
   there, never in `world/`, `game/`, `sim/`, or `ui/`.

6. **Cite what you used.** When you base a recommendation on the docs, name
   the chapter/extension you pulled it from so the user can verify — Vulkan
   guidance is easy to get subtly wrong and worth being checkable.
