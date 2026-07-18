# Building a new game — API reference

Fast-reference index for adding a new mini-game under `src/games/<name>/`. Read this
instead of re-exploring `src/engine/`, `src/render/`, `src/ui/` from scratch. If something
here looks stale (renamed function, moved file), trust the source over this doc and fix
this file in the same change.

See also: [`AGENTS.md`](../AGENTS.md) (project-wide rules), [`docs/UI_LIBRARY.md`](UI_LIBRARY.md)
(deep dive on the UI stack), [`docs/FrameArena.md`](FrameArena.md) (renderer internals, rarely
needed for a game).

## 1. Which base class?

Use `odai::engine::GameApp` (`src/engine/game_app.h`). Every existing mini-game
(`citybuilder`, `minesweeper`, `snake`, `stellaris`, `swtor`) subclasses it — it's a thin
GLFW + Vulkan-renderer + UI-context loop. Do **not** look at `src/app/app.cc`/`app.h`
(`odai::app::App`) — that's the large, older Civ-style "odai" strategy game and shares no
code with `GameApp`; it's not a template for anything new.

## 2. `GameApp` contract (`src/engine/game_app.h`)

```cpp
class GameApp {
public:
    bool init(const char* title = "odai");
    void run();
    void shutdown();
protected:
    virtual bool onInit() = 0;             // create game state, load fonts, build UI tree
    virtual void onTick(float dt) = 0;      // input + simulation step
    virtual void onRender(float dt) = 0;    // beginFrameDraw() ... draw ... submitFrame()
    virtual void onShutdown() {}            // optional

    virtual bool wantsMinimalRendering() const { return false; }
    // Return true for any game that is pure 2D/UI (no voxel/imported-scene content).
    // Skips building the 3D pipelines (pipe/imported/sky-cloud/water/grass, SSAO, hex
    // terrain). Must be decided before init() runs — it's a virtual called from init().
    // Minesweeper returns true; CityBuilder does not override it (defaults false, so it
    // pays for pipelines it never uses — prefer true for a fresh 2D game).

    bool loadFonts(regularPath, boldPath, italicPath, numericPath,
                   bodySize = 18.0f, numericSize = 16.0f);
    static std::string resolveAssetPath(const std::string& relativePath);
    void  framebufferSize(int& outW, int& outH) const;
    float contentScale() const;             // multiply hardcoded px constants by this for HiDPI
    void  beginFrameDraw();                 // call first in onRender()
    void  submitFrame(const render::CameraPose& camera, float simulationAlpha = 0.0f);
                                             // call last in onRender()

    // Available to subclasses:
    GLFWwindow*       m_window;
    render::Renderer  m_renderer;
    ui::Font          m_uiFont, m_uiFontBold, m_uiFontItalic, m_uiFontNumeric;
    ui::FontSet       m_uiFonts;
    ui::UiContext     m_uiContext;
    ui::UiDrawList    m_uiDrawList;
    ui::UiInput       m_uiInput;
};
```

`run()` drives the loop: poll GLFW events → sample mouse/scroll/text into `m_uiInput` →
`m_uiContext.update(m_uiInput)` + `.tick(dt)` → `onTick(dt)` → `onRender(dt)`.

`submitFrame()` appends the retained widget tree's geometry onto whatever you already drew
into `m_uiDrawList`, draws the custom OS-cursor replacement on top (automatic — never call
this yourself), uploads to the renderer, and calls `Renderer::renderFrame(...)`.

### `main.cc` template (identical across all games)

```cpp
#include "games/<name>/<name>_app.h"

int main() {
    odai::games::<name>::<Name>App app;
    if (!app.init("<Window Title>")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
```

### `onInit()` template

```cpp
bool MyGameApp::onInit() {
    if (!loadFonts(resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
                    resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
                    resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
                    resolveAssetPath("assets/fonts/Inter-Regular.ttf"))) {
        return false;
    }
    resetGame();
    // Optional: build a retained widget tree here (see §4).
    return true;
}
```

## 3. Renderer facade — what a game actually calls (`src/render/renderer.h`)

The renderer is a pImpl facade; **no Vulkan type ever appears here** (`AGENTS.md`: only
`src/render/` may include Vulkan headers). A `GameApp` subclass only ever touches this
subset directly (the rest is called by `GameApp` internals):

- `m_renderer.setUiDrawData(...)`, `setUiFontAtlas(...)`, `registerUiFontAtlas(...)`,
  `registerUiTextureRgba8(...)`/`registerUiTextureRgba8Mipmapped(...)` — UI bridge; mostly
  called for you via `loadFonts()`/`submitFrame()`.
- `render::CameraPose` — plain struct (`x,y,z,yawDegrees,pitchDegrees,fovDegrees,
  orthographic,orthoHalfHeight`) passed to `submitFrame()`. A 2D game can leave it
  default-constructed.
- Debug/tuning setters (`setDebugUiVisible`, `setFrameStatsVisible`,
  `setFramePacingSettings`, `setSsaoEnabled`, `setShadowSettings`, ...) — optional, rarely
  needed for a self-contained mini-game.

Everything else on `Renderer` (`uploadMagicaVoxelMesh`, `uploadImportedScene`,
`uploadHexTerrain`, `updateChunkMesh`, ...) is 3D/voxel/world content upload — irrelevant
unless your game renders actual world geometry, in which case read `world/` and
`import/` first.

**Never** include anything under `src/render/backend/vulkan/` from game code.

## 4. UI framework (`src/ui/*`)

Two ways to put pixels on screen, freely mixable:

**Immediate mode** — draw directly onto `m_uiDrawList` every frame from `onRender()`:
`addRectFilled`, `addRoundRectFilled`, `addBevel`, `addDropShadow`, `addRoundRectGlow`,
`addText(font, utf8, pos, color)`, `addImage`, `add9Slice`, `pushClip/popClip`,
`pushOpacity/popOpacity`. This is what CityBuilder uses for *everything* (hand-rolled
`uiButton()`/`textLeft/Center/Right()` helpers) and what both CityBuilder and Minesweeper
use for their game boards/HUD. Good default for board/grid-style game content that
changes every frame anyway.

**Retained widget tree** — build once in `onInit()`, read events via signals:

```cpp
auto root = std::make_unique<ui::Widget>();
root->mousePassthrough = true;           // let unhandled clicks fall through to the board
ui::Widget* rootRaw = m_uiContext.setRoot(std::move(root));

auto b = std::make_unique<ui::Button>(&m_uiFontBold, "New Game", nullptr);
b->slotName = "new_game";
b->setRect(ui::UiRect::fromXYWH(x, y, w, h));
rootRaw->addChild(std::move(b));

ui::SlotRegistry reg;
reg.on("new_game", [this]{ resetGame(); });
reg.wire(*rootRaw);
```

(Or skip `SlotRegistry` and connect directly: `button->activated.connect([this]{ ... });`
— every `Widget` has an `activated` `Signal<>`.) This is what Minesweeper uses for its
toolbar buttons. Use this for chrome/menus/dialogs — anything with hover/press states,
where hand-rolling immediate-mode buttons is wasted effort.

`UiContext::wantsMouse()` returns true when the cursor is over any widget — **gate your
board/world hit-testing on `!m_uiContext.wantsMouse()`** so clicks on UI buttons don't
also register as game-board clicks (see Minesweeper's `onTick`).

Widget catalog (`src/ui/widgets/*`, 43 files): `Panel` (styles: `styleCard`, `styleWin95`,
`styleCiv6`, `styleClassicMac`, `styleRetroOS`, ...), `Button`, `Label` (rich-text markup),
`Toggle`, `Slider`, `ProgressBar`, `TextBox`, `Dropdown`, `RadioButton`, `TabBar`,
`Toolbar`, `ScrollView`, `Modal`, `Toast`, `ContextMenu`, `Window`, `Spinner`,
`StackLayout`, `Repeater`, `Image`, `IconButton`, `DonutChart`, `LineChart`, `StatBadge`,
plus genre compound panels (`AdvisorsPanel`, `BuildQueuePanel`, `MinimapPanel`,
`ResourceBarPanel`, `SimControlsPanel`, ...).

**Check `src/ui/kits/` before hand-assembling a panel set** — `city_builder_kit.h`,
`colony_sim_kit.h`, `strategy_4x_kit.h` bundle the widgets a given genre typically needs.

Animation: `ui::Easing` (`Linear/EaseIn/EaseOut/EaseInOut/CubicIn/CubicOut/BackOut/Spring`),
`Tween`/`ColorTween`/`Vec2Tween`/`RectTween`/`Sequence` in `src/ui/animation.h`. Driven
automatically by `UiContext::tick(dt)` — no manual per-widget update needed. See
`src/tools/tween_demo/` for a live gallery of every easing curve.

Rich text markup (`<b>/<i>/<color=#rrggbb>/<br>`) via `Label`/`rich_text.h` for anything
beyond a single styled run of text.

### Live reference tools — run these before inventing a widget pattern

- `odai_design_system_demo` (`src/tools/design_system_demo/`) — every widget in one place,
  tabbed (Buttons/Inputs/Panels/Effects/Animation/DataFeedback/Layout).
- `odai_retro_theme_demo` (`src/tools/retro_theme_demo/`) — 5 OS-chrome themes (Win95,
  Motif, Classic Mac, Flat Retro, RetroOS) skinning real interactive widgets.
- `odai_tween_demo` (`src/tools/tween_demo/`) — animation/easing gallery.

## 5. Input

There's no input-mapping abstraction (deliberately, per `AGENTS.md`'s "avoid new
abstraction layers"). Two channels:

- **Mouse** — via `m_uiInput` (`ui::UiInput`), populated each frame by `GameApp::run()`:
  `m_uiInput.mousePx`, `.mouseDeltaPx`, `.scrollDelta`, `.button(ui::UiMouseButton::Left)`
  → `{down, pressed, released}`. Widgets consume this automatically through
  `UiContext::update()`; for direct board hit-testing read `m_uiInput` yourself (don't call
  raw GLFW mouse APIs).
- **Keyboard** — raw `glfwGetKey(m_window, GLFW_KEY_X)`, hand-rolled rising-edge latches.
  No shared helper exists; both reference games just keep `bool m_prevX` members (see
  Minesweeper's `m_prevR/m_prev1/m_prev2/m_prev3`) or a generic `edgeDown(int key)` +
  `std::unordered_map<int,bool>` (see CityBuilder's `edgeDown`). Copy whichever shape fits.

Ignore `core::InputState` (`src/core/input.h`) — it's explicitly-commented placeholder
scaffolding used only by the legacy `app::App`, not wired into `GameApp` at all.

## 6. Reusable simulation (`src/sim/simulation.h`) — optional

Renderer/UI-agnostic factory-sim primitives: conveyor `Belt`s, `Pipe`s, rail `Track`s,
and `BeltCargo` (fixed-point position + flood-style advance). `GameApp` owns an always-empty
`sim::Simulation m_emptySimulation` just to satisfy `Renderer::renderFrame`'s parameter.
If your game wants belts/pipes/rails (a Factorio-like), replace that empty member with a
real `sim::Simulation`, call `.update(dt)` from `onTick()`, and pass it through instead.
Mutate belts via `addBelt()`/`removeBeltAt()`, not the raw `belts()` vector directly — direct
mutation skips topology-dirty invalidation.

## 7. Reference implementations — which one to copy

| Game | Rendering model | wantsMinimalRendering | Best example of |
|---|---|---|---|
| `src/games/minesweeper/` | Immediate-mode board/HUD + retained widget-tree toolbar | `true` | **Start here for a new 2D game.** Smallest, most idiomatic mix of both UI models, `SlotRegistry` wiring, mouse-vs-UI gating. |
| `src/games/snake/` | Immediate-mode | not overridden (`false`) | Minimal real-time game loop, ~370 lines. |
| `src/games/citybuilder/` | Fully immediate-mode, no widget tree | not overridden | Large from-scratch UI surface: grid/zoning economy, charts, minimap, hand-rolled buttons — good reference for heavy custom `UiDrawList` usage without any widget tree. |
| `src/games/stellaris/`, `src/games/swtor/` | Immediate-mode | — | Larger genre-specific examples (4X, RPG-ish) if citybuilder/minesweeper don't cover your case. |

## 8. CMake registration

Add a new `add_executable` block next to the other `odai_game_*` targets in
`CMakeLists.txt` (roughly lines 630-800, inside the
`if(ODAI_RENDER_BACKEND_UPPER STREQUAL "VULKAN")` guard). Copy the Minesweeper block
verbatim and rename:

```cmake
add_executable(odai_game_<name>
    src/games/<name>/<name>_main.cc
    src/games/<name>/<name>_app.cc
    src/engine/game_app.cc
    src/import/dds.cc
    src/import/gpu_scene.cc
    src/import/imported_scene.cc
    src/render/frame_graph.cc
    src/render/renderer.cc
    src/world/world.cc
    src/world/chunk_grid_worldgen.cc
    src/world/chunk_mesher.cc
    src/world/clipmap_index.cc
    src/world/magica_voxel.cc
    ${RENDER_BACKEND_SOURCES}
)
target_include_directories(odai_game_<name> PRIVATE src ${ODAI_STB_INCLUDE_DIRS})
target_compile_definitions(odai_game_<name> PRIVATE
    ODAI_RENDER_BACKEND_VULKAN=1
    ODAI_HAS_VULKAN=1
    ODAI_HAS_GLFW=1
    ODAI_PROJECT_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
)
target_link_libraries(odai_game_<name> PRIVATE
    odai_ui odai_ui_vulkan odai_audio
    Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator imgui::imgui
)
if(TARGET glfw)
    target_link_libraries(odai_game_<name> PRIVATE glfw)
elseif(TARGET glfw3::glfw)
    target_link_libraries(odai_game_<name> PRIVATE glfw3::glfw)
endif()
if(TARGET slang_shaders)
    add_dependencies(odai_game_<name> slang_shaders)
endif()
if(MSVC)
    target_compile_options(odai_game_<name> PRIVATE ${ODAI_WARN_FLAGS_MSVC})
else()
    target_compile_options(odai_game_<name> PRIVATE -Wall -Wextra -Wpedantic)
endif()
```

The `world/`/`render/`/`import/` sources are required for every `GameApp` game even if it
never uploads 3D content — `Renderer::init` always constructs those systems unless
`setMinimalRenderMode` skips *pipeline* creation (it still needs the code compiled in).

Build just the new target: `cmake --build cmake-build-release --target odai_game_<name> -j 4`.

## 9. `AGENTS.md` constraints that bind new games

- Only `src/render/` may include Vulkan headers; never let a `Vk*` type leak into
  `src/games/`.
- Small focused functions, explicit ownership, flat structs — avoid inheritance beyond the
  one `GameApp` subclass, avoid inventing new abstraction layers (input mapping, ECS, etc.)
  unless asked.
- Performance: no hidden allocations or unbounded growth in `onTick`/`onRender` hot paths.
- World-feel/water/world-building rules in `AGENTS.md` apply only if the game touches
  `world/` terrain content — irrelevant to a self-contained 2D mini-game.
