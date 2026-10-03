# src/anim

- Responsibility: CPU animation sampling, skeletons, Skyrim programs, pose graphs and GPU pose data.
- Public interfaces: `animation_sampler.h`, `skeleton.h`, `pose_graph.h`, `skyrim_animation.h`, `gpu_pose.h`.
- Invariants: Preserve skeleton/clip interpretation and CPU/GPU pose layout agreement.
- Dependencies: odai_anim uses JSON; importer consumes it; GPU execution lives in renderer.
- Extension points: Samplers, pose modifiers, native/behavior programs and rig adapters.
- Tests: animation, pose_graph, humanoid_rig, native_animation, character_dynamics and pose_compute tests.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: biped_rig.cc is not in the current build. GPU smoke can skip with code 77; a skip is not GPU verification.
