# odai_ui standalone integration sample

Proves `odai_ui` + `odai_ui_vulkan` can be embedded in a Vulkan renderer with
**no dependency** on this repo's `app/`, `game/`, or `world/` modules — only
the two SDK libraries, GLFW, and the Vulkan SDK.

## Build

From the repo root:

```powershell
cmake -S . -B cmake-build-release -DODAI_BUILD_EXAMPLES=ON
cmake --build cmake-build-release --target odai_ui_integration_sample -j4
```

Run it with the repo root as the working directory (it resolves shader/font
assets relative to `ODAI_PROJECT_SOURCE_DIR`, falling back to the current
working directory — see `resolveAssetPath` in `main.cc`):

```powershell
cmake-build-release\examples\vulkan_ui_integration\Debug\odai_ui_integration_sample.exe
```

It opens a window with a themed card, a label, and a button that increments a
click counter — proof the widget tree, input dispatch, and Vulkan draw path
all work end to end.

## The contract, step by step

`main.cc` is split into two halves. Everything before the `--- odai_ui
contract ---` comments is generic Vulkan bootstrap (instance, device,
swapchain, command buffers, sync objects) that any engine already has in some
form. Everything after is the actual SDK integration surface:

1. **Init the Vulkan bridge** — `UiRenderer::init(InitInfo)`. Needs a device,
   physical device, `VmaAllocator`, a `BufferAllocator*` (see below), an
   upload queue + family, the swapchain's color format, a max bindless
   texture count, and the directory holding `ui.vert.slang.spv` /
   `ui.frag.slang.spv`.
2. **Bake and upload a font atlas** — `Font::loadFromFile(...)` then
   `UiRenderer::setFontAtlasR8(pixels, width, height)`.
3. **Build a widget tree** — pure `odai_ui`, zero Vulkan: `Panel`, `Label`,
   `Button`, wired up with `widget->activated.connect(...)`.
4. **Per frame**: feed a `UiInput` snapshot (mouse position in framebuffer
   pixels, button edges) into `UiContext::update()`, then `UiContext::build()`
   into a `UiDrawList`.
5. **Record**: `UiRenderer::uploadGeometry(cmd, frameArena, drawData)` — must
   run *outside* any open `vkCmdBeginRendering` block — then, inside an
   open dynamic-rendering pass targeting the swapchain image,
   `UiRenderer::record(cmd, frameIndex, drawData, extent)`.

## Two things a real integration must also provide

These aren't part of `odai_ui`'s job — they're generic Vulkan/VMA plumbing
every engine has already, but a from-scratch sample has to spell out:

- **A `BufferAllocator` + `FrameArena`** (`render/backend/vulkan/buffer_helpers.h`,
  compiled into `odai_ui_vulkan`). `UiRenderer` uses these for texture and
  vertex/index uploads rather than owning its own allocator, so a host engine
  that already has an allocation strategy can plug it in instead of carrying
  two.
- **Exactly one VMA implementation translation unit** — VMA is a header-only
  library requiring `#define VMA_IMPLEMENTATION` before including
  `<vk_mem_alloc.h>` in one `.cc` file. This sample provides it in
  `vma_impl.cc`. If your engine already has a VMA implementation TU anywhere
  in the final binary, you don't need a second one.

## Device requirements

The device selection in `main.cc` is deliberately narrower than what a full
3D renderer would require — a UI-only consumer only needs:

- Vulkan 1.3 `dynamicRendering` (the UI pipeline has no render-pass object)
- Vulkan 1.2 `descriptorBindingPartiallyBound`,
  `descriptorBindingSampledImageUpdateAfterBind`, and
  `shaderSampledImageArrayNonUniformIndexing` (the bindless UI texture array)
- `VK_KHR_swapchain`

No MSAA, tessellation, ray tracing, or timeline semaphores — those are this
repo's main renderer's requirements, not the UI SDK's.

## Not production-hardened

This sample skips swapchain-resize handling, depth buffers, and multi-queue
transfer — it exists to prove the contract, not to be a renderer starting
point. See `docs/UI_LIBRARY.md` for the full integration guide and widget
catalog.
