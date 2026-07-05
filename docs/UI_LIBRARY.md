# odai_ui: a UI library for strategy and builder games

`odai_ui` is a first-party, retained-mode UI toolkit for Vulkan-based games,
built specifically for strategy, 4X, city-builder, and colony-sim interfaces —
persistent panels, resource bars, tech trees, minimaps, build queues — rather
than debug tooling. It ships as two libraries: `odai_ui` (pure C++20, no
Vulkan) and `odai_ui_vulkan` (the thin Vulkan bridge), both of which can be
vendored into a project that has nothing else in common with this repo. See
[`examples/vulkan_ui_integration/`](../examples/vulkan_ui_integration/) for a
from-scratch integration.

## What this is, and isn't

**odai_ui is not ImGui**, and isn't trying to be a drop-in replacement for it.
The two solve different problems:

| | ImGui | odai_ui |
|---|---|---|
| Model | Immediate mode — you re-declare the whole UI every frame | Retained mode — you build a widget tree once, mutate it as state changes |
| Intended use | Debug tooling, editor panels, dev overlays | Shippable, styled game UI |
| Styling | Push/pop style vars in code | JSON theme files (`UiTheme`), hot-reloadable, artist-editable |
| Text | Plain strings, one font | Rich markup (`<b>`, `<i>`, `<color=#rrggbb>`, `[icon=name]`), OpenType shaping (ligatures, kerning), multi-face `FontSet` |
| Icons | App supplies texture handles | CPU-tessellated vector icons from SVG, cached to disk (`.odaivec`) |
| Callbacks | Poll return values (`if (ImGui::Button(...))`) | `Signal<>` connections (`button->activated.connect(...)`) |
| Genre presets | None | `kits/strategy_4x_kit.h`, `city_builder_kit.h`, `colony_sim_kit.h` — curated widget sets per genre |

This repo keeps ImGui around purely as a dev/debug overlay
(`RendererBackend::buildFrameStatsUi`, etc.) — it is a separate, unrelated
system from `odai_ui` and shares no code with it.

## Architecture

```
src/ui/                          — pure C++20, zero Vulkan (library: odai_ui)
  ui_draw_list.{h,cc}             immediate-mode geometry emitter -> UiDrawData
  widget.h, ui_context.{h,cc}     retained widget tree + input dispatch
  font.{h,cc}                     stb_truetype R8 atlas + OpenType shaping
  rich_text.{h,cc}                markup parsing, wrap, alignment
  theme/ui_theme.{h,cc}           JSON design tokens (fonts/colors/frames/icons/sizes)
  vector/                         SVG -> CPU tessellation -> cached icon meshes
  widgets/                        ~44 widgets (see catalog below)
  kits/                           genre-curated widget presets

src/render/backend/vulkan/       — the ONLY files in the UI stack that touch Vulkan
  ui_renderer.{h,cc}               bindless-texture pipeline, atlas/texture upload,
                                    per-frame geometry streaming (library: odai_ui_vulkan)
  buffer_helpers.{h,cc}            BufferAllocator + FrameArena (also part of
                                    odai_ui_vulkan — see "Dependencies" below)
```

Data flows one way: widget tree → `UiDrawList` → `UiDrawData` (vertices,
indices, per-command texture + clip rect) → `UiRenderer::uploadGeometry()` /
`record()`. Nothing about a host renderer's world state, camera, or game
logic crosses into `src/ui/`.

### Vertex format and draw modes

`UiVertex` is a fixed 40-byte layout: pixel-space position, UV/SDF params,
packed vertex color, and a mode word whose low 8 bits select a `UiDrawMode`
(`SolidColor`, `Textured`, `GlyphAlpha`, `Shadow`, `RoundRect`,
`RoundRectGlow`) with the remaining bits holding a bindless texture slot. One
draw command can therefore mix glyphs, icons, and solid fills without
splitting a batch per texture — texture selection lives per-vertex, not per
descriptor-set bind.

### Dependencies

`odai_ui` depends on nothing beyond `nlohmann_json` (plus optional `stb` and
`nanosvg`, both PRIVATE — no public header leaks them). `odai_ui_vulkan`
depends PUBLICLY on `Vulkan::Vulkan` and `GPUOpen::VulkanMemoryAllocator`
because its installed public header (`ui_renderer.h`) exposes `VkDevice` /
`VmaAllocator` types directly in `UiRenderer::InitInfo`. It also compiles
`buffer_helpers.cc` (`BufferAllocator`, `FrameArena`) itself, since
`UiRenderer`'s public API depends on those types — a consumer linking only
`odai_ui_vulkan` gets a complete, self-sufficient library with no missing
symbols. The one thing it does *not* provide is a VMA implementation
translation unit (VMA is header-only and requires exactly one `#define
VMA_IMPLEMENTATION` TU somewhere in your binary) — see
`examples/vulkan_ui_integration/vma_impl.cc`.

## Integration guide

1. **Init the Vulkan bridge**: `UiRenderer::init(InitInfo)` — device, physical
   device, `VmaAllocator`, a `BufferAllocator*`, an upload queue + family, the
   swapchain color format, a max bindless texture count, and the directory
   holding `ui.vert.slang.spv` / `ui.frag.slang.spv`.
2. **Bake and upload a font**: `Font::loadFromFile(path, pixelHeight)`, then
   `UiRenderer::setFontAtlasR8(font.atlasPixels().data(), font.atlasWidth(),
   font.atlasHeight())`. Additional faces (bold/italic) go through
   `registerFontAtlasR8`; icon/frame textures go through
   `registerTextureRgba8` / `registerTextureRgba8Mipmapped`.
3. **Build a widget tree** — pure `odai_ui`, no Vulkan: compose `Panel`,
   `Label`, `Button`, etc., wire callbacks via `widget->activated.connect(...)`
   or a per-widget `std::function` member (e.g. `Toggle::onChange`).
4. **Per frame**: feed a `UiInput` snapshot (mouse position in framebuffer
   pixels, per-button edges, scroll, text codepoints) into
   `UiContext::update()`, then `UiContext::build()` into a `UiDrawList`.
5. **Record**: `UiRenderer::uploadGeometry(cmd, frameArena, drawData)` outside
   any open render pass (it's a `HOST → VERTEX_INPUT` barrier, and
   `VERTEX_INPUT` isn't a framebuffer-space stage), then, inside an open
   dynamic-rendering pass targeting your swapchain image,
   `UiRenderer::record(cmd, frameIndex, drawData, extent)`.

Full working code for all five steps, including the Vulkan bootstrap most
engines already have in some form, is in
[`examples/vulkan_ui_integration/main.cc`](../examples/vulkan_ui_integration/main.cc).

### Vendoring via CMake

```cmake
find_package(odai_ui CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE odai::odai_ui odai::odai_ui_vulkan)
```

`odai_ui`'s exported config only requires `nlohmann_json`; it pulls in Vulkan
and VMA `find_dependency()` calls only when `odai_ui_vulkan` was actually
built (so a headless consumer of `odai_ui` alone never needs a Vulkan SDK
installed).

## Theming

`UiTheme::loadFromFile(path, uploadCallback)` loads a JSON file of design
tokens — fonts, colors (dotted keys like `text.dim`, `panel.bg`), 9-slice
frame definitions, SVG icon references, and numeric sizes — and hot-reloads
on file change via `UiHotReload`. The full JSON schema is documented inline in
`src/ui/theme/ui_theme.h`. Beyond token-based theming, several widgets
(`Panel`, `Button`) bake whole visual identities into C++ preset methods —
`styleCard()`, `styleOrnate()`, `styleSoft()`, `styleGradientCard()`, and
period-accurate retro looks (`styleWin95()`, `styleMotif()`,
`styleClassicMac()`).

## Widget catalog

**Primitives**: `Button`, `Label`, `Panel`, `Window`, `Image`, `Spacer`,
`IconButton`, `RadioButton` / `ButtonGroup`, `Spinner`, `Modal`,
`ContextMenu`.

**Forms**: `TextBox` (single/multi-line), `Slider`, `Toggle`, `Dropdown`,
`TabBar`.

**Containers & layout**: `ScrollView` (virtualized), `HorizontalStack` /
`VerticalStack`, `Repeater` (list/grid templating).

**Charts**: `DonutChart`, `LineChart`.

**Domain panels**: `AdvisorsPanel`, `BuildQueuePanel`, `EventTrackerPanel`,
`FactionPanel`, `GridPickerPanel`, `MinimapPanel`, `NotableEntityPanel`,
`ResearchPanel`, `ResourceBarPanel`, `SelectionInspectorPanel`,
`SimControlsPanel`, `StatBadge`, `Toast`, `Tooltip`.

**Genre kits**: `kits/strategy_4x_kit.h`, `kits/city_builder_kit.h`,
`kits/colony_sim_kit.h` — pre-curated widget selections per genre, for
projects that want a starting point rather than assembling from primitives.

## Tests

```powershell
cmake --build cmake-build-release --target odai_ui_tests
ctest --test-dir cmake-build-release -R odai_ui_tests -V
```

`tests/ui_tests.cc` covers the draw list, font metrics, rich text, theming,
and per-widget hit-test/callback behavior headlessly (no Vulkan, no GPU). The
Vulkan bridge itself has no automated test — it's verified by building and
running `examples/vulkan_ui_integration/` or the main `odai` app.
