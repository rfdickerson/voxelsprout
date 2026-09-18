# HKX behavior semantics progress

The x64 Skyrim SE reader now retains state-local and wildcard transition arrays:
event and target IDs, nested-state IDs, priorities, raw flags, trigger/initiation
windows, decoded effect node references, and condition presence/class. Shared
arrays preserve their ordered rules on each owner. Import limits bound total
expanded rules. Failed graph reads publish an empty result.

The animation probe exposes transition rules and explicitly reports
runtime_consumed=false. This is an import milestone, not authored graph execution.
The subsequent NPC integration is documented in [actor animation coverage](SKYRIM_ACTOR_ANIMATION_COVERAGE.md), including the conservative executable subset and version-11 save state.

Synthetic coverage checks both owners, effect linkage, unsupported condition
presence, flags/priority, timing, oversized arrays, nonfinite timing, and failure
output. No proprietary fixtures are committed.

Event/variable name tables, clip playback parameters, timeline triggers and a
conservative runtime compiler now exist. Event priority and admitted blend timing
are consumed by BehaviorGraphInstance. Scalar defaults, expression conditions and
manual-selector variable bindings now execute, with saved per-actor variables and
selector state. Disabled rules/conditions and local wildcard self-transitions are
also handled. Other bindings, modifier/blender execution, transition windows and
remaining flags/nested overrides still prevent retail master-graph execution.

Layout reference:
https://github.com/adamhynek/activeragdoll/blob/master/include/RE/havok_behavior.h
