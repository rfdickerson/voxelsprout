# Dragonsreach light-shaft blending (2026-09-09)

Dragonsreach's `Effects/Ambient/FXAmbBeamXbDustBig02.nif` declares
NiAlphaProperty flags `0x100d`: source alpha / one additive blending.
The importer previously retained only the blend enable bit, so the renderer
attenuated the background with one-minus-source-alpha. Black texture regions
became dark polygon sheets across the hall.

The static imported path now carries a typed additive blend mode from the NIF
property through mesh parts, packed draws, upload merge keys and pipeline
selection. It uses the original texture, lighting, vertex alpha and animation.
Depth testing remains enabled and transparent depth writes remain disabled.
Spatial page building now copies the complete draw metadata; reconstructing only
indices and threshold discarded blend, animation and vegetation state.

One spare mesh-part byte and one spare packed-draw bit preserve the existing
24-byte / 16-byte serialized strides. Legacy zero values retain ordinary alpha
blending. Generated cell cache semantics advance to 105. Other NIF blend-factor
pairs retain the existing alpha behavior; this is not complete blend-mode support.

Validation:

- Optimized runtime/shader build passed.
- Synthetic inherited-NiAlphaProperty fixtures distinguish additive beams,
  ordinary alpha glass and existing Fallout blend-plus-test cutouts.
- Packing, page rebuilding, full cooked loading and runtime loading retain the
  blend distinction for surfaces sharing a texture; vegetation metadata remains
  intact and serialized strides are checked.
- CTest: 35/39 passed. Existing failures remain inventory, Bethesda runtime,
  save and world-map cache tests; no new test failures.
- Local native-DPI captures: `captures/dragonsreach-rays/before.png` and
  `fixed.png`, 768x432 logical / 1536x864 framebuffer, 240 warmup frames.
  The dark shaft polygons disappear in the matched fixed capture.
- Vulkan validation and synchronization validation enabled for the fixed capture;
  no validation errors or synchronization hazards were reported. A second
  240-frame moving-camera run (`moving.png` / `moving.log`) also passed.

This fixes the reproduced compositing defect, not a claim of matched retail
brightness or complete NIF effect falloff/soft-depth support. Original assets and
capture evidence remain local.

Source definitions: [NifTools nif.xml](https://raw.githubusercontent.com/niftools/nifxml/develop/nif.xml).
Depth-state reference: [Vulkan depth guide](https://docs.vulkan.org/guide/latest/depth.html).
