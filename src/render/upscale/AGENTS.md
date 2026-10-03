# src/render/upscale

- Responsibility: Resolution policy/contracts and temporal/vendor upscaling implementations.
- Public interfaces: `upscale_policy.h`, `upscale_contract.h`, `upscaler_backend.h`, `temporal_upscaler.h`.
- Invariants: Maintain contract validation and temporal fallback when vendor SDK is unavailable.
- Dependencies: odai_upscale owns policy/contract; odai_renderer owns temporal_upscaler.cc and vendor_backends.cc.
- Extension points: Policy selection, resolution calculations and backend lifecycle.
- Tests: `odai_upscaler_tests`.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: Directory ownership is split across two targets; XeSS is optional and platform-dependent.
