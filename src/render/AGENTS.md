# src/render

- Responsibility: One explicit Vulkan imported-scene renderer, GPU residency, lighting and presentation.
- Public interfaces: `renderer.h`, `renderer_types.h`, `renderer_shared.h`; backend internals in `backend/vulkan/renderer_backend.h`.
- Invariants: Use explicit pass ordering/barriers. Keep CPU/Slang packing and shared depth policies consistent.
- Dependencies: Renderer links importer, core, UI, upscale, Vulkan, VMA, ImGui and GLFW; slang_shaders is a build dependency.
- Extension points: Explicit frame/pass routines, resource initialization, matching Slang shader and policy headers.
- Tests: `odai_imported_render_policy_tests`, `odai_frame_graph_tests`, bindless/GPU arena/TAA tests, Vulkan smoke; `src/render/tests/pose_compute_tests.cc`.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: Historical voxel shader names implement GI, not voxel gameplay. frame_graph names do not authorize generalized machinery. Smoke needs Vulkan/Xvfb; policy tests do not prove rendered output.
