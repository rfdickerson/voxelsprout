# HKX behavior semantics progress

The x64 Skyrim SE reader now retains state-local and wildcard transition arrays:
event and target IDs, nested-state IDs, priorities, raw flags, trigger/initiation
windows, decoded effect node references, and condition presence/class. Shared
arrays preserve their ordered rules on each owner. Import limits bound total
expanded rules. Failed graph reads publish an empty result.

The animation probe exposes transition rules and explicitly reports
runtime_consumed=false. This is an import milestone, not authored graph execution.
Existing actor fallback behavior and save layouts are unchanged.

Synthetic coverage checks both owners, effect linkage, unsupported condition
presence, flags/priority, timing, oversized arrays, nonfinite timing, and failure
output. No proprietary fixtures are committed.

Next: graph event-name/variable tables, condition decoding and a conservative
runtime compiler; then deterministic event/priority selection and authored blend
timing in BehaviorGraphInstance. Nested transitions, bindings, modifiers and
unsupported flags require explicit admission before retail execution.

Layout reference:
https://github.com/adamhynek/activeragdoll/blob/master/include/RE/havok_behavior.h
