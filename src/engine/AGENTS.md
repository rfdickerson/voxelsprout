# src/engine

- Responsibility: Application loop and frame/tick statistics integration.
- Public interfaces: `game_app.h`, `game_frame_stats.h`, `frame_timing_csv.h`.
- Invariants: Keep timing evidence useful; use optimized builds for performance.
- Dependencies: game_app.cc is part of odai; stats utilities are header-only.
- Extension points: Loop integration and focused frame accounting.
- Tests: `odai_engine_stats_tests`; runtime smoke for app integration.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: This directory is not a separately linked engine library.
