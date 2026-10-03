# src/core

- Responsibility: Logging, jobs, resource paths and small shared utilities.
- Public interfaces: `job_system.h`, `log.h`, `resource_path.h`; other utilities may be header-only.
- Invariants: Zero workers runs jobs inline; waitIdle drains active jobs as well as queued work.
- Dependencies: odai_core links Threads; keep game and renderer dependencies out.
- Extension points: Small shared utilities, resource lookup and diagnostics.
- Tests: `odai_job_system_tests`, `odai_core_types_tests`, engine stats tests.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: Legacy comments mention removed systems. A header in this directory need not be compiled into odai_core.
