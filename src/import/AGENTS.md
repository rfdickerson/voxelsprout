# src/import

- Responsibility: Decode Bethesda assets and records into ImportedScene and streaming inputs.
- Public interfaces: `imported_scene.h`, `dds.h`, `cell_residency_planner.h`; `bethesda/{asset_source,cell_builder,nif_scene,esm_reader,content_profile}.h`.
- Invariants: Preserve cooked-scene/chunk formats. Residency planning is Vulkan-, IO-, and thread-free; callers perform loads and eviction.
- Dependencies: Importer links core, anim, dialogue, ZLIB, Jolt and JSON.
- Extension points: Format decoding, record extraction, material conversion and cell assembly.
- Tests: `odai_imported_scene_tests`, `odai_bethesda_import_tests`, `odai_cell_residency_planner_tests`, format-specific tests.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: `bethesda/cell_streamer.cc` is compiled by odai, not the importer. `src/bethesda/condition.cc` belongs to the importer. Broad importer linkage is not test coverage.
