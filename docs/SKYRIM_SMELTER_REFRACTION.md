# Smelter / forge purple distortion correction

SMIM's furniture/smeltermarker.nif contains a HeatBlur surface with
textures/effects/VaporTileNormal_n.dds and SLSF1_Refraction. The Whiterun forge
replacement contains the same normal-map distortion surface. The source texture
was being rendered as albedo in the ordinary static path, producing a purple sheet.
This is an unsupported shader-family fallback, not a missing original texture.

The Skyrim cell builder now omits alpha-blended surfaces flagged Refraction or Fire_Refraction
from ordinary color rendering, logs the model/shape, and reports
refractionShapesSkipped in coverage output. Other shapes and authored light records
remain intact. Opaque refractive materials retain their existing base-surface fallback. Source material flags remain preserved by the NIF reader.

Heat-haze refraction is still unsupported. A future implementation needs a safe
scene-color input; sampling the currently written color target is not a valid fix.
No replacement fire or arbitrary light tint is added. Generated caches use version
111; cooked field layouts are unchanged. Previously cooked scenes must be rebuilt
to remove already-baked invalid distortion geometry.

Synthetic parser coverage checks both refraction flags, zero current animated
strength and an ordinary-material control. Local source probe output is retained
outside committed fixtures; no proprietary assets are redistributed.

Optimized build and all 40 CTests pass. A local Whiterun validation capture
reported no Vulkan validation errors/warnings and logged the smelter HeatBlur and
forge distortion surfaces being omitted. The default market view does not prove
close-up visual acceptance at the furnace; the interactive scene is available for
that check. Format reference: https://raw.githubusercontent.com/niftools/nifxml/develop/nif.xml
