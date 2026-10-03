# Architecture improvement backlog

These tickets capture the architecture improvement audit of the working checkout.
Each ticket records its implementation status; product capability status is separate.
Rank reflects expected development/validation payoff, scope and risk. No P0 issue
was established by the bounded audit. Tickets are engineering improvements rather
than product capability definitions, so they do not add entries to PARITY.md.

| Rank | Ticket | Priority | Direction | Status |
| --- | --- | --- | --- | --- |
| 1 | [ARCH-001](ARCH-001-portable-validation.md) | P1 | Decouple portable validation from presentation builds | Implemented |
| 2 | [ARCH-002](ARCH-002-application-transitions.md) | P1 | Extract testable application transition orchestration | Proposed |
| 3 | [ARCH-003](ARCH-003-session-contracts.md) | P1 | Narrow session mutation and presentation callback contracts | Proposed |
| 4 | [ARCH-004](ARCH-004-explicit-configuration.md) | P1 | Resolve renderer configuration explicitly and report effective values | Proposed |
| 5 | [ARCH-005](ARCH-005-geometry-contracts.md) | P2 | Separate shared geometry contracts from renderer interfaces | Proposed |
| 6 | [ARCH-006](ARCH-006-integration-ownership.md) | P2 | Give streaming and navigation integration explicit build ownership | Proposed |
| 7 | [ARCH-007](ARCH-007-vulkan-resource-locality.md) | P2 | Localize Vulkan effect resource state and lifecycle | Proposed |
| 8 | [ARCH-008](ARCH-008-probe-command-locality.md) | P2 | Partition Bethesda probe commands and publish dispatch metadata | Proposed |

## Working a ticket

Read the owning local AGENTS.md, revalidate the cited evidence in the current
checkout, and choose the smallest coherent slice. Evidence counts/file sizes are
audit snapshots, not acceptance thresholds. Use the navigation tools to identify
owners, direct dependencies and focused tests; regenerate maps after build/include
changes. Record the tests and before/after locality or dependency measurements
before marking a ticket complete. If a slice also implements a product capability,
follow that capability's Definition of Done and the root status policy separately.

ARCH-001 improves verification for subsequent runtime work. ARCH-003 and ARCH-004
can make later ARCH-002 slices easier to isolate, but are not prerequisites for a
bounded door-transition extraction. ARCH-005 can support ARCH-006; neither requires
a whole-repository include restructuring. ARCH-007 must retain explicit renderer
control flow. ARCH-008 preserves static executable dispatch.

## Audit limits and findings not promoted to tickets

The audit used configured CMake/CTest metadata, include relationships and targeted
source windows; it did not crawl every source body. The configured target graph
had no cycles. Header/runtime/disabled-configuration cycles are not ruled out.

High fan-in for math, logging and basic widget contracts did not establish harmful
ownership. Large parsers were not flagged by size alone. Jolt's call_once-guarded
global initialization did not demonstrate a lifecycle failure. No pervasive service
locator, problematic inheritance hierarchy, generated/source mixing or confirmed
duplicated domain logic was established. Headless UI state verification and pixel
verification serve different purposes; that distinction alone is not a defect.

For the current-state inventory and navigation experiments, see
[agent-navigation.md](../agent-navigation.md).
