# src/tools

- Responsibility: Headless simulation/UI replay, content inspection/cooking, texture packing and benchmark helpers.
- Public interfaces: Executable main files; coverage helpers have headers in this directory.
- Invariants: Use synthetic fixtures for portable validation; never distribute real game data.
- Dependencies: Probe/headless link runtime/importer; cooker/texture pack link importer.
- Extension points: Command dispatch, structured JSON reports and harness schema consumers.
- Tests: headless replay/snapshot/UI tests, probe record tests, coverage tests.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: Large bethesda_probe_main.cc: locate command dispatch first. oblivion_nif_lab is outside active targets. Heap interposer is Linux-specific.
