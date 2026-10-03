# src/dialogue

- Responsibility: Generic dialogue definitions, evaluation/runtime and state IO.
- Public interfaces: `dialogue_types.h`, `dialogue_context.h`, `dialogue_runtime.h`, `dialogue_state_io.h`.
- Invariants: Preserve dialogue state round trips and context-driven evaluation.
- Dependencies: odai_dialogue uses JSON; importer/runtime adapt game-specific dialogue.
- Extension points: Evaluation and serialization; game adapters belong to import/bethesda or bethesda.
- Tests: `odai_dialogue_tests`; TES3/Skyrim runtime tests for adapters.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: Generic dialogue ownership does not include every game-specific dialogue script/native.
