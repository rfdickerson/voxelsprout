# Asset provenance review

Publication remains blocked on unresolved history and texture provenance.
The binary package uses an explicit allowlist: compiled engine shaders, Inter
fonts plus their license, engine documentation and the content filename manifest.
It does not package textures, cooked scenes, game archives or local captures.

| Item | Evidence / disposition |
|---|---|
| Inter fonts | Bundled SIL Open Font License; included with license text |
| Engine shaders | Compiled from tracked Slang sources; no game shader bytecode |
| `docmitchell.bin` | Added by `7996ca3`, whose description identifies a cooked GSDocMitchellHouse interior; removed from tracked working tree and retained only in ignored local quarantine |
| `assets/water.dds` | Added in `e106be0` / `54de76d` on reachable history; authorship/license not established, excluded from packages |
| `assets/textures/morrowind_water_normal.png` | Added in `c3b84f6`; authorship/license not established, excluded from packages; runtime uses its generated fallback unless explicitly overridden |
| Legacy executable and strategy scenes | Removed from current tree; no longer part of the project |

Deleting a file from the current tree does not remove its Git history. A public
repository/source release requires resolving these findings and reviewing all
reachable objects. Do not rewrite history automatically. The audit script records
candidate asset paths/object IDs for local review; extension scanning cannot certify
copyright provenance or absence of embedded assets/secrets.

The local reachable-object scan found 85 asset candidates. Both unresolved water
textures are excluded from Git export archives as well as binary packages. Their
reachable Git objects still require a separate publication review.
