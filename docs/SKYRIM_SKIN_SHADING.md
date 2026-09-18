# Skyrim skin shading

The actor importer now retains BSShaderTextureSet slot 1 and the authored
SLSF1_Model_Space_Normals flag. Ralof's retail `MaleHeadNord` shape has 898
vertices, no vertex normals, and references `MaleHead_msn.dds`. Dropping that
texture forced the renderer to approximate skin lighting from geometry alone.

Model-space maps are sampled as linear RGB, preserving negative Z. Each vertex
carries the three source-model basis directions through the same coordinate
conversion, FaceGen bind bake, and weighted animated palette as its surface.
The rasterizer interpolates those directions and the fragment shader transforms
the sampled normal into world space. Tangent-space maps on other actor parts use
the existing derivative-based frame. Missing textures retain geometric normals.
Normal debug views now show the mapped shading normal. The shadow diagnostic
also evaluates back-facing surfaces instead of returning white for a skipped
NdotL query; earlier mostly-white actor captures did not prove that those
surfaces were unoccluded.

Skin materials also retain their soft-light rolloff, specular color, strength,
and glossiness. Texture slot 2 supplies the authored soft-light mask and slot 7
supplies the external skin specular mask. Sun and local lights evaluate a
masked smooth diffuse extension and the authored Blinn specular lobe, including
existing light attenuation and shadow visibility. Soft lighting also evaluates
occlusion on back-facing surfaces; local-light shadow bias uses geometric
normals rather than the sampled normal-map detail. This is Skyrim's inexpensive
soft-light approximation, not a screen-space subsurface scattering pass.
Ralof's head uses rolloff 0.4, glossiness 33, and specular strength 2.69. The retail
female body probe resolves its own MSN/skin/specular maps through the same path.

The transient GPU skinning input is 148 bytes; the output remains the existing
76-byte imported vertex. Three unused actor surface words carry octahedrally
encoded basis directions. The material values reuse the actor-only normal/layer
payload; this path is explicitly gated off for packed PBR materials. The velocity binding uses the new input stride with
unchanged position/influence offsets. Cooked scenes, streamed chunks, and saves
have no layout changes.

Validation: all 45 CTest tests pass; Debug and RelWithDebInfo builds succeed.
SPIR-V validation covers skinning, imported vertex/fragment, and terrain domain
shaders. Disassembly verifies input/output strides of 148/76. Synthetic coverage
checks soft-rolloff decoding, material/texture retention, and the signed Z-up/Y-up basis conversion alongside
existing weighted bind and missing-normal tests. Local retail probes and captures
are retained in `captures/ralof-skin-shading/`, using JK's Skyrim + SMIM at
768×432 logical / 1536×864 framebuffer and render scale 1.

This fixes the missing normal and skin lighting paths. It does not claim complete Skyrim shader
parity: FaceGen tint/detail layering, authored rim/back-light branches,
and eye-specific reflection still need their own
material implementations and reference comparisons. Real cast shadows and the
existing scene light levels are preserved. The corrected shadow capture confirms that the front of the face is occluded
in the saved view. This work preserves that darkness and does not establish
pixel parity with a matching Skyrim SE framebuffer.

References:
- [Niftools Skyrim shader flags](https://github.com/niftools/nifxml/blob/develop/nif.xml): bit 12 selects model-space normals and an external specular map.
- [Vulkan shader memory layout](https://docs.vulkan.org/guide/latest/shader_memory_layout.html): scalar member layout and SPIR-V stride validation.

- [Skyrim SE shader reconstruction](https://github.com/aers/Skyrim-SE-Shader-Tools/blob/master/old/shaders/Lighting/BSLightingShader.ps.hlsl): soft-light lobe, external specular mask, and authored gloss interpretation. The implementation here adapts the equations to the existing renderer.

Lydia/CBBE follow-up: the MSN sample must decode image axes into NIF axes as
`(r, b, -g)` after signed expansion, before multiplying by the imported animated
basis. Treating RGB as NIF XYZ pointed broad back and leg surfaces downward.
A local comparison of geometric and mapped-normal captures verifies the corrected
orientation; shadow-only captures showed those surfaces were already unoccluded.
The shader keeps the authored normal detail and existing light/shadow strengths.
Local evidence: `/tmp/lydia-geometric.png`, `/tmp/lydia-basis-probe2.png`, and
`/tmp/lydia-normal-fixed-lit.ppm` (temporary, not redistributed game assets).
The earlier BC7_UNORM-as-linear color experiment was reverted: DDS UNORM storage
alone does not establish a Bethesda diffuse texture's transfer function. Keep
color textures on the existing sRGB path and normal/other data maps linear.
