# Navigation tooling

- Responsibility: generate/query configured target, include and CTest evidence.
- Public interfaces: nav.py CLI and versioned JSON maps; README.md describes use.
- Invariants: do not maintain a second CMake parser/build definition. Keep output
  scoped; distinguish literal references, build dependencies and runtime coverage.
- Dependencies: Python standard library, CMake File API, CTest and rg.
- Extension points: filtered query subcommands and evidence-based associations.
- Tests/validation: `python3 tools/ai/validate.py`; regenerate twice and compare
  output when changing generation. No engine tests needed for prose-only edits.
- Traps: CTest regex is not Python regex; require --no-tests=error. File API edges
  can be transitive. Configured source ownership is not directory ownership.
  Regenerate following configuration/include changes; exclude vendor source trees.
