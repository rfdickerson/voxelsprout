# tests

- Responsibility: Hand-rolled CTest executables, Python integration checks and synthetic fixtures.
- Public interfaces: Test names/commands in tools/ai/test-map.json; registrations in CMakeLists.txt.
- Invariants: C++ tests use assert for setup and verification; keep assertions enabled in optimized builds.
- Dependencies: Test links indicate build dependencies, not measured coverage.
- Extension points: Add cases in the smallest owning test; fixture contracts in fixtures/harness.
- Tests: Use exact CTest regex or nav run-tests for a file.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: Some navigation tests compile production sources directly. Optional GPU/real-data evidence cannot be replaced by CPU policy tests.
