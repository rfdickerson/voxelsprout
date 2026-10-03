# src/ui

- Responsibility: Retained RPG widgets, draw lists, text/font handling and input context.
- Public interfaces: `ui_context.h`, `ui_draw_list.h`, `font.h`, `rich_text.h`, `widgets/` headers.
- Invariants: Preserve retained widget/event behavior and text/font resource handling.
- Dependencies: odai_ui links core, JSON and ZLIB; uses Stb. Vulkan UI backend belongs to renderer.
- Extension points: Widget state/layout/event routines and draw-list production.
- Tests: `odai_ui_tests`, `odai_swf_font_tests`.
- Validation: from repository root, `python3 tools/ai/nav.py tests <changed-file>`;
  `python3 tools/ai/nav.py run-tests <changed-file>` builds and runs associated tests.
  Use `check <target>` for a compile check; read parent guidance for full validation.
- Traps: Headless UI harness validates runtime UI intents; it does not validate retained widget pixels.
