# src/games/bethesda

- Responsibility: Interactive application assembly: window/input, actors, collision, audio decoding and streaming.
- Public interfaces: `bethesda_app.h`, `bethesda_actors.h`, `bethesda_collision.h`, `traversal_state.h`.
- Invariants: Maintain the single imported-scene path and native-DPI render extent.
- Dependencies: odai consumes renderer, audio, traversal and Bethesda runtime. engine/game_app drives the loop.
- Extension points: App event/update integration and game-specific presentation adapters.
- Tests: actor_movement, navigation_simulation, traversal_state and mod_check_headless tests.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: Historical newvegas types/env names are compatibility APIs. Streaming is compiled here from src/import; traversal_state is its own library.
