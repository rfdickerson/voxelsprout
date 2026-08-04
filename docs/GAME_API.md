# Building a new game — API reference

Fast-reference index for adding a new mini-game under `src/games/<name>/`. Read this
instead of re-exploring `src/engine/`, `src/render/`, `src/ui/` from scratch. If something
here looks stale (renamed function, moved file), trust the source over this doc and fix
this file in the same change.

See also: [`CLAUDE.md`](../CLAUDE.md) (project-wide rules), [`docs/UI_LIBRARY.md`](UI_LIBRARY.md)
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

    virtual audio::AudioConfig audioConfig() const { return audio::AudioConfig{}; }
    // Initial volumes/mute state passed to Audio::init(); defaults to AudioConfig{}'s
    // built-in defaults. See §4.

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
    audio::Audio      m_audio;              // init/update/shutdown already wired; see §4
    ui::Font          m_uiFont, m_uiFontBold, m_uiFontItalic, m_uiFontNumeric;
    ui::FontSet       m_uiFonts;
    ui::UiContext     m_uiContext;
    ui::UiDrawList    m_uiDrawList;
    ui::UiInput       m_uiInput;
};
```

`run()` drives the loop: poll GLFW events → sample mouse/scroll/text into `m_uiInput` →
`m_uiContext.update(m_uiInput)` + `.tick(dt)` → `onTick(dt)` → `m_audio.update(dt)` →
`onRender(dt)`.

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
    // Optional: build a retained widget tree here (see §5).
    return true;
}
```

## 3. Renderer facade — what a game actually calls (`src/render/renderer.h`)

The renderer is a pImpl facade; **no Vulkan type ever appears here** (`CLAUDE.md`: only
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

## 4. Audio facade — what a game actually calls (`src/audio/audio.h`)

`GameApp` owns `m_audio` (`audio::Audio`) directly — init/update/shutdown are already
wired into the base class lifecycle (`init()` brings it up right after the renderer and
before `onInit()` runs; `run()` pumps it once per frame after `onTick()`; `shutdown()`
tears it down before the renderer). A subclass never calls those three itself — only the
content-facing API below, typically from `onInit()` (load + start ambient/music) and
`onTick()`/`onRender()` (one-shots, listener updates):

- `m_audio.loadSound(path, category)` / `loadMusic(path)` — decode a short SFX/ambient
  clip or register a streamed music track; both return an invalid handle (never a crash)
  if the file is missing, so it's safe to reference assets that don't exist yet.
- `m_audio.playSound(clip)` — fire-and-forget one-shot (UI clicks, non-positional cues).
- `m_audio.playSoundAt(clip, position, attenuation = {})` — spatialized one-shot (footsteps,
  impacts), distance-attenuated against the current listener transform.
- `m_audio.startAmbient(loop, fadeSeconds = 1.0f)` / `startAmbientAt(loop, position,
  attenuation = {}, fadeSeconds = 1.0f)` — start a global (unpositioned, e.g. wind) or
  positional (e.g. torch, river) ambient loop; both return an `AmbientHandle` for a slot in
  a fixed pool of `kMaxAmbientSlots` (`audio/audio_types.h`) — several can run concurrently.
  `stopAmbient(handle, fadeSeconds)` crossfades one out; `setAmbientPosition(handle,
  position)` moves a positional loop (e.g. it's attached to a moving object).
- `m_audio.setListenerTransform(listener)` — call once per frame with the player/camera
  position (+ optional forward/up) so positional SFX/ambient loops attenuate correctly; see
  VoxelCraft's `onRender()` for a worked example building this from an interpolated camera pose.
- `m_audio.playMusic(track, fadeSeconds = 2.0f, loop = true)` / `stopMusic(fadeSeconds)` —
  crossfades to/from a single current music track.
- `m_audio.setMasterVolume`/`setCategoryVolume(category, v)`/`categoryVolume(category)`/
  `setMuted`/`muted()` — the four buses are `SoundCategory::{Master,Music,Ambient,Ui}`.

Override `audioConfig()` (defaults to `AudioConfig{}`) if a game wants different starting
volumes — there's no persisted GameApp-level settings file yet (unlike `src/app/App`'s
`odai.cfg`), so this is the seam until one exists. On a machine with no audio device, or a
build without the miniaudio backend, every call above becomes a safe no-op — always call
`loadSound`/`startAmbient*` unconditionally rather than gating on `deviceActive()`.

## 5. UI framework (`src/ui/*`)

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

## 6. Input

There's no input-mapping abstraction (deliberately — this project avoids new
abstraction layers unless clearly justified). Two channels:

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

### Reserved key: F3

`GameApp::run()` consumes **F3** for the built-in CPU timing overlay (see §6.5). Don't
bind it in a game.

## 6.5. CPU frame profiling — free for every game

`GameApp::run()` times its own loop every frame, so every game gets a CPU breakdown with
no per-game code. Zones are defined in `src/engine/game_frame_stats.h`:

| Zone | Covers |
|---|---|
| `poll` | `glfwPollEvents` + per-frame input sampling |
| `ui update` | `UiContext::setViewport/update/tick` |
| `tick` | your `onTick(dt)` |
| `plugins` | `PluginRegistry::tickAll` |
| `audio` | `Audio::update` |
| `render` | your `onRender(dt)`, in full |
| `  ui build` | `UiContext::buildAppend` + cursor — **nested inside `render`** |
| `  submit` | `Renderer::renderFrame` — **nested inside `render`** |
| `FRAME` | the whole loop iteration |

> **Build optimized before reading any of this.** A Debug build is ~8x slower on
> real work in this tree, so its numbers describe the compiler, not your code.
> Use the `vcpkg-relwithdebinfo` / `linux-vcpkg-relwithdebinfo` preset — see
> "Optimized builds" in `CLAUDE.md`.

**Overlay:** press **F3** in any game, or set `ODAI_PERF_OVERLAY=1` to start with it up.
It shows last/avg/p99 per zone, a proportional bar, an `other` row for unattributed time,
and the renderer's own CPU wait buckets so a CPU spike can be told apart from a frame
merely spent blocked on the GPU or the presenter.

**Reading it in code** — for a game's own HUD, or from a plugin:

```cpp
const auto& prof = frameProfiler();                      // GameApp accessor
const float tickMs = prof.channel(GameZone::Tick).ewmaMs();
const float worst  = prof.channel(GameZone::Tick).p99Ms();
```

`ui build` and `submit` are counted inside `render` — skip them when summing, or use
`gameZoneIsNested(zone)`. `unattributedMs()` already does this and is what the `other`
row reports; if `other` is large, the cost is somewhere `run()` doesn't yet measure.

Percentiles are computed **on demand**, not cached per frame — calling `p99Ms()` copies
and partially sorts the 240-sample window, so don't call it in a hot loop. `lastMs()` and
`ewmaMs()` are free.

To time something *inside* your own `onTick`/`onRender`, use the same primitives directly
(`src/core/frame_profiler.h`): `core::ScopedTimerMs zone(myFloat);` accumulates a scope's
duration into a float you own, and `core::TimingChannel<N>` gives it the same
last/avg/percentile window. Don't add zones to `GameZone` — that set is deliberately
closed to the shared loop.

## 7. Reusable simulation (`src/sim/simulation.h`) — optional

Renderer/UI-agnostic factory-sim primitives: conveyor `Belt`s, `Pipe`s, rail `Track`s,
and `BeltCargo` (fixed-point position + flood-style advance). `GameApp` owns an always-empty
`sim::Simulation m_emptySimulation` just to satisfy `Renderer::renderFrame`'s parameter.
If your game wants belts/pipes/rails (a Factorio-like), replace that empty member with a
real `sim::Simulation`, call `.update(dt)` from `onTick()`, and pass it through instead.
Mutate belts via `addBelt()`/`removeBeltAt()`, not the raw `belts()` vector directly — direct
mutation skips topology-dirty invalidation.

## 8. Reference implementations — which one to copy

| Game | Rendering model | wantsMinimalRendering | Best example of |
|---|---|---|---|
| `src/games/minesweeper/` | Immediate-mode board/HUD + retained widget-tree toolbar | `true` | **Start here for a new 2D game.** Smallest, most idiomatic mix of both UI models, `SlotRegistry` wiring, mouse-vs-UI gating. |
| `src/games/snake/` | Immediate-mode | not overridden (`false`) | Minimal real-time game loop, ~370 lines. |
| `src/games/citybuilder/` | Fully immediate-mode, no widget tree | not overridden | Large from-scratch UI surface: grid/zoning economy, charts, minimap, hand-rolled buttons — good reference for heavy custom `UiDrawList` usage without any widget tree. |
| `src/games/stellaris/`, `src/games/swtor/` | Immediate-mode | — | Larger genre-specific examples (4X, RPG-ish) if citybuilder/minesweeper don't cover your case. |
| `src/games/legion/` | Immediate-mode HUD + `ImportedScene` ground + GPU-skinned actors | not overridden | Only example calling `Renderer::uploadSkinnedMeshTemplate`/`setSkinnedActorPose` (multi-instance skinning, up to `kMaxSkinnedInstances`) — start here for any game with skeletally animated characters. Procedural rig/mesh authoring (`anim/biped_rig.h`, `procgen/humanoid_generator.h`) since there's no character-import pipeline yet. |

## 9. CMake registration

Add a new `add_executable` block next to the other `odai_game_*` targets in
`CMakeLists.txt` (roughly lines 630-800, inside the
`if(ODAI_RENDER_BACKEND_UPPER STREQUAL "VULKAN")` guard). Copy the Minesweeper block
verbatim and rename:

```cmake
add_executable(odai_game_<name>
    src/games/<name>/<name>_main.cc
    src/games/<name>/<name>_app.cc
    src/engine/game_app.cc
    src/engine/plugin.cc
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

## 10. Project constraints that bind new games

- Only `src/render/` may include Vulkan headers; never let a `Vk*` type leak into
  `src/games/`.
- Small focused functions, explicit ownership, flat structs — avoid inheritance beyond the
  one `GameApp` subclass, avoid inventing new abstraction layers (input mapping, ECS, etc.)
  unless asked.
- Performance: no hidden allocations or unbounded growth in `onTick`/`onRender` hot paths.
- World-feel/water/world-building conventions apply only if the game touches `world/`
  terrain content — irrelevant to a self-contained 2D mini-game.

## 11. Plugins — optional lifecycle extension points (`src/engine/plugin.h`)

Most games need nothing here — a single `GameApp` subclass is still the default and
correct shape for a self-contained mini-game. `PluginRegistry`/`IEnginePlugin` exist for
the case where a piece of behavior should be *composed onto* a `GameApp` rather than
written into its subclass — e.g. an optional debug overlay, a stats/telemetry collector,
or (the case this was built for) something a `mods/` package wants to hook into a running
game without the base game's `GameApp` subclass knowing about it. This is the engine's
sanctioned extension seam; don't invent a second one (a signal bus, an event queue) for
the same problem.

Every `GameApp` owns a `PluginRegistry m_plugins` for free — nothing to add to your
subclass. Register plugins during `onInit()`, before returning `true`:

```cpp
bool MyGameApp::onInit() {
    // ... existing setup ...
    m_plugins.add(std::make_unique<MyPlugin>());
    return true;
}
```

`IEnginePlugin` mirrors `GameApp`'s own hook shape (`onAttach`/`onTick`/`onRender`/
`onDetach`), all defaulted to no-ops. `GameApp::init()` calls `attachAll()` right after
your `onInit()` succeeds; `run()` calls `tickAll()` right after your own `onTick(dt)`
every frame; `shutdown()` calls `detachAll()` right after your own `onShutdown()`.

**`onRender` is the one hook that is *not* wired in automatically.** A game's `onRender()`
ends by calling `submitFrame()`, which flushes the UI draw list and submits the frame —
anything drawn after that point never reaches the GPU. If a plugin needs to draw, call
`m_plugins.renderAll(*this, dt)` yourself from `onRender()`, after `beginFrameDraw()` and
before `submitFrame()`.

No dynamic loading (`dlopen`/DLL) is involved — a "plugin" here is a statically linked
C++ class registered at startup, matching how this codebase already treats `IModHost`
(`src/game/mod_host.h`) and `mods/base` content loading (`src/content/mod_loader.h`): compose
behavior via a small registered interface, not via a build-time or runtime plugin loader.
