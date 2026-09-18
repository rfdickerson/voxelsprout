# Contributing

Start with README.md, docs/COMPATIBILITY.md and AGENTS.md. Describe the player-visible
problem, affected game/profile, and reproduction before proposing a change.

Use one explicit imported-scene rendering path. Preserve CLI/environment compatibility
and serialized scenes unless their layout changes. Do not add voxel worlds, engine
plugins, Lua systems, mini-games or generalized render-graph infrastructure.

Build with the documented vcpkg preset and run CTest. Use optimized builds for
performance evidence. Add focused synthetic regressions for bugs; installed-game
probes are optional and must not require assets in public CI.

PRs should state behavior changed, validation performed, and remaining limitations.
Never commit game/mod assets, extracted data, saves, personal paths or capture evidence.
Document new third-party code and its license in THIRD_PARTY_NOTICES.md.
Contributions are provided under this repository’s MIT license.
