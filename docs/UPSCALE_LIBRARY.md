# `odai_upscale` — pluggable temporal upscaling

Two libraries, mirroring the `odai_ui` / `odai_ui_vulkan` split:

| Target | Needs Vulkan | What it is |
|---|---|---|
| `odai_upscale` | no | Backend selection, the fallback contract, and the reconstruction rules (jitter, phase count, mip bias) any host must honour. |
| `odai_upscale_vulkan` | yes | The `IUpscaler` plug-in seam and this engine's built-in temporal backend. |

The Vulkan-free half is separate so it can be unit-tested without standing up a
device — `odai_upscaler_tests` links it alone — and so tools and config parsers
can reason about backends without a renderer.

## The two halves of an integration

Every upscaler SDK splits the same way, and this module keeps the split explicit
because it is what makes a backend swappable.

**What the host owns:** the frame's images, their layouts and barriers,
descriptor provisioning, debug labels, timestamps. All of that reaches a backend
through `HostServices` (`upscaler_backend.h`), which is modelled on FSR2's
`FfxFsr2Interface` — an application-supplied interface the technique calls back
into. It is why this engine's `VK_EXT_descriptor_buffer` sets and its frame arena
never leak into a backend that has no idea what either is.

**What the backend owns:** its pipelines, its history, its algorithm, and the
resource states it needs. `IUpscaler::dispatch()` takes images, not descriptors,
because that is what XeSS, FSR and DLSS take.

## Adding a backend

One class and one case:

```cpp
class MyUpscaler final : public upscale::IUpscaler {
    UpscalerBackend id() const override { return UpscalerBackend::Xess; }
    Capabilities capabilities() const override;   // available + why not
    bool setup(const SetupInfo&) override;        // extents, inverted depth, layout
    DispatchResult dispatch(const DispatchInfo&) override;
    void shutdown() override;
};
```

then return it from `createUpscaler()` in `vendor_backends.cc`. Nothing in the
frame code changes. `DispatchResult` carries the layout the colour input was left
in, because the upscaling and same-resolution paths genuinely differ there and a
bool cannot express three outcomes.

XeSS, FSR and DLSS are already present as enum entries that resolve, parse and
report — they simply return `nullptr` with a specific reason. That is deliberate:
it means `--upscaler dlss` exercises the whole selection path today and prints

```
upscaler: requested dlss but running temporal -- the DLSS backend is declared but not implemented
```

rather than the selection path being first exercised on the day an SDK lands.

## The contract a host must honour

`upscale_contract.h`. These are properties of temporal reconstruction, not of any
implementation, and every vendor states the same four:

- **`renderExtentFor(display, quality)`** — the preset's internal resolution.
  Rounds rather than truncates; the lost half-pixel at UltraQuality shows up as a
  permanent shimmer along one screen edge.
- **`jitterPhaseCount(render, display)`** — **scales with the upscale ratio.**
  Eight phases is right only when input and output share a grid. Upscaling, each
  input pixel covers ratio² output pixels and a length-8 sequence leaves most
  output pixels with no sample near them. Derived from the extents, not the
  preset, so a host that sets render scale by hand still gets a long enough
  sequence.
- **`jitterOffsetNdc(phase, render)`** — Halton(2,3), 1-based (Halton(0) is 0 in
  every base, so a 0-based sequence wastes its first frame), centred on the pixel.
  Apply it to the projection matrix, not to UVs, so depth, normals and colour are
  all rasterized on the same jittered grid and stay reprojectable together.
- **`recommendedMipLodBias(render, display)`** — `log2(render/display) - 1`, the
  value FSR2, XeSS and DLSS all publish. **This engine does not apply it yet**;
  its material samplers are created once with no bias and the value would have to
  reach them at sampler creation. Stated here because a vendoring host has to
  satisfy it, and because "the upscaled image is soft" is otherwise very hard to
  attribute.

## Fallback

`resolveUpscaler()` always returns something that runs. Asking for a backend that
was not compiled in, or whose runtime rejected the device, is not an error — it
reports the reason and hands back Temporal. That is what lets one command line
and one CI invocation work across machines with different GPUs and different SDKs
installed, and it is why the built-in temporal backend has no way to be
unavailable: it is what everything else falls back *to*.

Selection and construction are deliberately separate. `resolveUpscaler()` decides
once, at init, and logs why; `createUpscaler()` only builds what was decided. A
backend that fails `setup()` afterwards logs and drops to no upscaler rather than
silently rendering at the wrong grid.

## Shaders

`taa.comp.slang` (same-resolution resolve) and `temporal_upscale.comp.slang`
(reconstructing) share one descriptor layout, so the pass picks a pipeline rather
than duplicating its plumbing. Both install to
`share/odai_upscale/shaders`. The upscale shader is **optional**: without it the
backend runs as a same-resolution resolve — softer than a real upscale, but not a
broken frame.

## Status

The temporal backend is the only implementation, and it is the one the engine
runs every frame — this is not an interface waiting for its first user. XeGTAO
ships its shaders here but is **not yet behind an equivalent `IAmbientOcclusion`
seam**; it is still dispatched directly from `frame_pass_ssao.cc`.
