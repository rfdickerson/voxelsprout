# src/bethesda

- Responsibility: Game state and deterministic sessions, inventories, quests/scripts, persistence and physics.
- Public interfaces: `bethesda_session.h`, `runtime_world.h`, `record_resolver.h`, `tes3_runtime.h`, `papyrus_vm.h`, `save_game.h`.
- Invariants: Keep stable record/object identities and deterministic harness replay; preserve save compatibility.
- Dependencies: Runtime links importer and core; headless runners consume runtime without presentation.
- Extension points: Session native bindings, game-specific record adapters, commands and state transitions.
- Tests: `odai_bethesda_runtime_tests`, TES3/Papyrus/save/inventory tests and headless fixtures.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: `condition.cc` is importer-owned. Navigation sources and simulation integration are also compiled separately by application/tests; query exact file ownership. UI flow here is semantic state, not retained widget rendering.
