#pragma once

#include "import/imported_scene.h"
#include "math/math.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace odai::render {

enum class FramePacingMode : std::uint8_t {
    Off = 0,
    Passive = 1,
    Scheduled = 2,
};

struct FramePacingSettings {
    FramePacingMode mode = FramePacingMode::Passive;
    std::uint32_t cadenceDivisor = 1;
    std::uint32_t maxQueuedFrames = 2;
};

struct FramePacingStats {
    bool displayTimingSupported = false;
    bool displayTimingEnabled = false;
    bool schedulingActive = false;
    std::uint32_t cadenceDivisor = 1;
    std::uint32_t maxQueuedFrames = 2;
    std::uint32_t queuedFrames = 0;
    std::uint32_t latePresentCount = 0;
    std::uint32_t gpuTimestampSkippedFrames = 0;
    float refreshMs = 0.0f;
    float targetPresentIntervalMs = 0.0f;
    float desiredLeadTimeMs = 0.0f;
    float presentMarginMs = 0.0f;
    float actualPresentDeltaMs = 0.0f;
    float presentScheduleErrorMs = 0.0f;
    float cpuWaitFrameSlotMs = 0.0f;
    float cpuWaitAcquireMs = 0.0f;
    float cpuWaitPresentMs = 0.0f;
    float cpuWaitTransferMs = 0.0f;
    bool gpuTimestampsPending = false;
    std::uint64_t desiredPresentTimeNs = 0;
};

struct UiRenderStats {
    std::uint32_t textureSlots = 0;
    std::uint32_t commandCount = 0;
    std::uint32_t drawCallCount = 0;
    std::uint64_t dynamicUploadBytes = 0;
    std::uint64_t skippedDrawCalls = 0;
};

enum class ShadowMode : std::uint8_t {
    ShadowMaps = 0,
    RayTraced = 1,
    Auto = 2,
};

// Temporal upscaling backend.
//
// The renderer renders the 3D scene at a fraction of the swapchain extent and an
// upscaler reconstructs it. Which one is a runtime choice, but availability is a
// COMPILE-time fact: XeSS is Intel's closed-source SDK and DLSS is NVIDIA's, so
// neither can be implemented here -- they are linked, and only when the SDK is
// present. Requesting an unavailable backend is not an error; it reports
// unavailable and falls back, so the same command line works on every machine.
enum class UpscalerBackend : std::uint8_t {
    // Render at native resolution. No reconstruction, no jitter requirement.
    Off = 0,
    // This engine's own temporal upscaler, built on the existing jittered TAA
    // and the skinned motion vectors. Always available -- it is the default and
    // the fallback for every other backend.
    Temporal = 1,
    // Intel XeSS-SR. Requires ODAI_ENABLE_XESS and the SDK at build time.
    Xess = 2,
    // AMD FidelityFX Super Resolution. DECLARED, NOT IMPLEMENTED.
    Fsr = 3,
    // NVIDIA DLSS. DECLARED, NOT IMPLEMENTED.
    Dlss = 4,
};

// Quality preset, which is really a render-scale choice. The ratios are XeSS's
// published ones so a preset means the same internal resolution whichever
// backend renders it -- otherwise "Quality" would be a different measurement on
// each and no comparison between them would mean anything.
enum class UpscalerQuality : std::uint8_t {
    UltraQuality = 0,   // 1.3x
    Quality = 1,        // 1.5x
    Balanced = 2,       // 1.7x
    Performance = 3,    // 2.0x
    UltraPerformance = 4,  // 3.0x
    // Reconstruct at the SAME resolution: no upscale, just the temporal
    // resolve. This exists because TAA and upscaling are separate things that
    // this API had welded together -- recordTaaPass returns early without an
    // upscaler object, so setTaaEnabled(true) with the default backend of Off
    // was a silent no-op, and the only way to get TAA was to accept a 1/1.5
    // render scale along with it.
    Native = 5,  // 1.0x
};

inline constexpr float upscalerQualityScale(UpscalerQuality quality) {
    switch (quality) {
    case UpscalerQuality::UltraQuality: return 1.0f / 1.3f;
    case UpscalerQuality::Quality: return 1.0f / 1.5f;
    case UpscalerQuality::Balanced: return 1.0f / 1.7f;
    case UpscalerQuality::Performance: return 1.0f / 2.0f;
    case UpscalerQuality::UltraPerformance: return 1.0f / 3.0f;
    case UpscalerQuality::Native: return 1.0f;
    }
    return 1.0f;
}

struct UpscalerSettings {
    UpscalerBackend backend = UpscalerBackend::Off;
    UpscalerQuality quality = UpscalerQuality::Quality;
    // 0 = none. Applied by backends that expose it; ignored by those that do not.
    float sharpness = 0.0f;
};

// What actually happened, as opposed to what was asked for. `reason` is only
// meaningful when requested != active, and exists because "I asked for XeSS and
// got something else" has several distinct causes that look identical on screen.
struct UpscalerStatus {
    UpscalerBackend requested = UpscalerBackend::Off;
    UpscalerBackend active = UpscalerBackend::Off;
    bool compiledIn = false;      // the backend was built into this binary
    bool runtimeAvailable = false;  // and its runtime/device support is present
    float renderScale = 1.0f;
    const char* reason = "";
};

// Screen-space ambient-occlusion estimator. Each maps to its own compute pipeline
// built from ssao.comp.slang with a different ODAI_AO_MODE, so the sample loop
// carries no uniform branch. Off dispatches neither the AO nor the blur pass and
// leaves the world shaders' ambient factor at 1.
//
// Ssao is the original normal-oriented hemisphere point sampler and the cheapest.
// Hbao (Bavoil 2008) marches horizons and catches contact darkening a point
// sampler misses. Gtao (Jimenez 2016) does the same horizon search through the
// correct cosine-weighted visibility integral, so it converges on ray-traced AO;
// it is the default and the right pick unless you are budget-bound.
enum class AoMode : std::uint8_t {
    Off = 0,
    Ssao = 1,
    Hbao = 2,
    Gtao = 3,
    // Intel's XeGTAO: the same ground-truth integral as Gtao, but marched
    // against a prefiltered depth pyramid with blue noise and adaptive sample
    // counts, and producing bent normals alongside the AO term.
    Xegtao = 4,
};

// Single-channel visualizations of what the main pass actually shaded with,
// for answering "is this surface wrong because of the geometry, the texture,
// or the material flags" without guessing. One global mode rather than a set
// of booleans: they are mutually exclusive by construction (each one replaces
// the frame's colour entirely), and one uniform channel is cheaper to plumb
// than one per view.
//
// MaterialFlags is the one that earns its keep. Alpha handling in this engine
// is decided entirely by flags the importer sets -- the shader cannot tell an
// opacity mask from a specular mask by looking at a texture -- so a surface
// that renders as a black slab where it should be transparent is either
// "the flag never got set" or "the flag is set and the alpha is wrong", and
// nothing on screen distinguishes those two until you false-colour the flags.
//
// Alpha deliberately bypasses the alpha-test discard: discarding first would
// throw away exactly the texels this view exists to look at.
//
// These are read by imported_static.frag.slang, which for a Fallout scene
// covers the terrain and every static and actor -- but not the sky or water,
// which have their own shaders and pass through unchanged.
enum class DebugView : std::uint8_t {
    Off = 0,
    Albedo = 1,        // sampled base colour, unlit
    Normal = 2,        // shading normal, remapped to 0..1
    Alpha = 3,         // sampled alpha as greyscale, discard bypassed
    MaterialFlags = 4, // red=alphaTest green=alphaBlend blue=twoSided yellow=unlit
    Roughness = 5,
    Metallic = 6,
    MipLevel = 7,      // false-coloured texture LOD
    CascadeIndex = 8,  // false-coloured shadow cascade selection
    TextureId = 9,     // bindless slot hashed to a colour
    LinearDepth = 10,  // post-pass; every other view is main-pass
    // The two lighting-balance views. Both are computed mid-shading rather than
    // in debugViewColor(), because their inputs do not exist until the sun and
    // ambient terms have been evaluated.
    Shadow = 11,       // cascaded shadow visibility: white lit, black occluded
    DirectRatio = 12,  // direct / (direct + ambient): how much of the lighting
                       // a shadow is even able to remove. Black means a surface
                       // is lit entirely by unshadowed ambient, which is what
                       // "the shadows do nothing" looks like as a measurement.
    // Terrain layer blend, false-coloured: red/green/blue are ATXT layers 0/1/2
    // and grey is the quadrant's BTXT base showing through. Geometry carrying no
    // terrain layers at all is left near-black.
    //
    // It shows the weights AS AUTHORED -- before the noise perturbation and
    // smoothstep the lit shader applies -- which is the whole point: a wedge-
    // shaped path boundary is either in the DATA or in that stylization, and
    // those have completely different fixes. Comparing this against the lit
    // frame is what separates them.
    TerrainLayers = 13,
    // Ambient occlusion alone, as a clay render: white unoccluded, dark
    // occluded, no albedo and no lighting. The view to reach for when AO is
    // suspected, because every other way of looking at it measures AO through
    // the albedo-weighted ambient term and so returns a picture of the diffuse
    // texture rather than of the occlusion.
    AmbientOcclusion = 14,
    ScreenSpaceGi = 15,
};

// True for views whose value is already a display-ready colour and must not be
// run through exposure, tonemapping or grading on the way to the screen.
// LinearDepth is produced by the post pass itself and so is not in this set.
inline constexpr bool debugViewBypassesTonemap(DebugView view) {
    return view != DebugView::Off && view != DebugView::LinearDepth;
}

enum class VoxelGiSurfaceMode : std::uint8_t {
    Legacy = 0,
    RtSurface = 1,
    RestirSurface = 2,
};

enum class ShadowFallbackReason : std::uint8_t {
    None = 0,
    RayTracingUnsupported = 1,
    RayTracingDisabled = 2,
    MainPassNotImplemented = 3,
    RayTracingSceneUnavailable = 4,
};

struct ShadowSettings {
    ShadowMode mode = ShadowMode::Auto;
};

struct ShadowStats {
    ShadowMode requestedMode = ShadowMode::ShadowMaps;
    ShadowMode activeMode = ShadowMode::ShadowMaps;
    bool rayTracingSupported = false;
    bool rayQuerySupported = false;
    bool accelerationStructureSupported = false;
    bool rayTracingRuntimeEnabled = false;
    bool mainPassRayTracingReady = false;
    bool mainPassRayTracingActive = false;
    bool fallbackActive = false;
    ShadowFallbackReason fallbackReason = ShadowFallbackReason::None;
};

// An authored sky, as a Fallout WTHR record describes one. Colors are LINEAR
// rgb in 0..1 -- WTHR stores them as sRGB bytes, so the caller decodes before
// getting here, the same rule the rest of the renderer follows.
//
// `weight` blends over the procedural Rayleigh/Mie sky rather than replacing
// it: 0 is the default and renders exactly as before, which is what keeps every
// other game unaffected by this existing.

// Fallout authors four cloud layers per weather. The ones a weather does not
// use point at "sky\alpha.dds", a fully transparent 1520-byte placeholder, so
// "four layers" in the record usually means one or two on screen.
inline constexpr int kWeatherCloudLayerCount = 4;

struct WeatherSkyParams {
    float weight = 0.0f;
    float skyUpper[3] = {0.0f, 0.0f, 0.0f};   // zenith
    float skyLower[3] = {0.0f, 0.0f, 0.0f};   // the band just above the horizon
    float horizon[3] = {0.0f, 0.0f, 0.0f};    // the skyline itself
    float fogColor[3] = {0.0f, 0.0f, 0.0f};
    float fogFarDistance = 0.0f;              // world units; 0 leaves fog alone

    // Per-layer tint (linear, from PNAM) and coverage, updated per frame
    // because both track time of day. A layer with opacity 0 costs nothing:
    // the shader skips it.
    float cloudTint[kWeatherCloudLayerCount][3] = {};
    float cloudOpacity[kWeatherCloudLayerCount] = {};

    // What the weather says the GROUND should be lit by: NAM0's Sunlight (the
    // direct sun) and Ambient (the sky's fill), linear.
    //
    // These are display-referred sRGB in the record, authored for a renderer
    // that used them directly, so only their HUE is taken at face value. The
    // renderer keeps its own physically-derived intensity and applies a bounded
    // gain from the record's own luminance -- enough that an overcast reads
    // dimmer and a sunset reads warm, without a weather being able to blow out
    // or black out an exposure the rest of the frame is calibrated against.
    //
    // lightingWeight 0 is the default and leaves the procedural sun and sky
    // ambient EXACTLY as they were, which is the state every game that does not
    // read WTHR records is in.
    float sunlightColor[3] = {0.0f, 0.0f, 0.0f};
    float ambientColor[3] = {0.0f, 0.0f, 0.0f};
    float lightingWeight = 0.0f;

    // How much of the sun's HALO and haze bloom this weather lets through, from
    // WTHR's Sun Glare byte. 1 is the default and the unmodified look; the sun
    // DISC is not scaled, because a weather that hides the sun does it with
    // cloud cover and this would only make a visible sun look wrong.
    float sunGlare = 1.0f;
};

// How a cloud texture covers the sky. These are different projections, not
// different scales of one -- drawing a layer with the wrong one does not look
// mistuned, it looks broken, and no amount of scale tuning recovers it. Which
// one is right is a property of how the ART WAS DRAWN, so it is per layer.
enum class WeatherCloudMapping : std::uint32_t {
    // Fallout and Oblivion: one fisheye image of the whole sky, zenith at the
    // centre and horizon on the rim of the inscribed circle. Sampled exactly
    // once, so it must never wrap, and it drifts by ROTATING about the zenith.
    DomeFisheye = 0,
    // Skyrim's overhead decks (SkyrimCloudsUpper*, SkyrimCloudsLower*): a
    // seamlessly tiling cloud field with no radial structure at all, sampled
    // many times across a dome with wrap addressing, drifting by translation.
    TilingPlane = 1,
    // Skyrim's horizon banks (SkyrimCloudsHorizon*): the texture is four
    // HORIZONTAL STRIPES of cloud tops, drawn to be seen edge-on around the
    // skyline. Compass bearing is u and elevation is v, so the stripes stack
    // upward and wrap around the sky. Under the plane projection above the same
    // art becomes stripes running across the sky in one compass direction,
    // which is the giveaway that the mapping, not the tuning, is wrong.
    Cylindrical = 2,
};

// One uploaded cloud layer and how to draw it.
struct WeatherCloudLayer {
    // Decoded DDS exactly as it comes out of the game's or a mod's archive. An
    // empty texture disables the layer.
    odai::importer::ImportedSceneTexture texture;
    WeatherCloudMapping mapping = WeatherCloudMapping::DomeFisheye;
    // Drift. What a unit means depends on the mapping: radians per second about
    // the zenith for a fisheye (which uses scrollU alone), texture units per
    // second otherwise.
    float scrollU = 0.0f;
    float scrollV = 0.0f;
    // For a fisheye this is a DOME scale and never a tiling count: 1.0 puts the
    // horizon on the texture's inscribed circle, and above 1 pushes the rim
    // past the horizon and out of the texture. For the other two it is exactly
    // a tiling count, and above 1 is the normal case.
    float scale = 1.0f;
    // The elevation window this layer occupies, as dir.y, feathered at both
    // ends. Skyrim stacks an overhead deck and a horizon bank in the same sky
    // and they are not the same size of thing -- drawing both across the whole
    // hemisphere is a whiteout. 0..1 is the whole sky and is what every
    // Fallout and Oblivion layer uses, so their look is unchanged.
    float bandLow = 0.0f;
    float bandHigh = 1.0f;
};

// The layers behind a weather. Separate from WeatherSkyParams because uploading
// is expensive and only happens when the weather changes, while the tints there
// move every frame.
struct WeatherCloudTextures {
    WeatherCloudLayer layers[kWeatherCloudLayerCount];
};

// Lighting authored on an imported interior CELL. Colours are scene-linear;
// importers must decode the record's sRGB bytes before submitting them.
//
// `hasAuthoredLighting` deliberately differs from `enabled`: the old
// setImportedSceneInteriorMode(bool) API only knew that a scene was indoors.
// Keeping that state distinct preserves its fixed-lighting compatibility path,
// while streamed cells with XCLL can opt into the source-faithful policy.
struct ImportedInteriorLighting {
    enum class LocalShadowMode : std::uint8_t {
        Off,
        ShadowMaps,
        ShadowMapsWithContact,
        RayTraced,
    };

    enum class IndirectLightingMode : std::uint8_t {
        Off,
        ScreenSpaceDiffuse,
    };

    bool enabled = false;
    bool hasAuthoredLighting = false;
    float ambientColor[3] = {};
    float directionalColor[3] = {};
    float fogColor[3] = {};
    float fogNear = 0.0f;
    float fogFar = 0.0f;
    bool showSky = false;
    bool useSkyLighting = false;
    // Source interiors are primarily lit by point lights. ShadowMaps is the
    // production path; RayTraced is retained as a measurable A/B reference.
    LocalShadowMode localShadowMode = LocalShadowMode::Off;
    // Separate from voxel GI: Bethesda rooms are much larger than that
    // camera-following 64^3 volume and instead reuse screen depth + TAA history.
    IndirectLightingMode indirectLightingMode = IndirectLightingMode::Off;
};

inline constexpr bool useAuthoredImportedInteriorLighting(
    const ImportedInteriorLighting& lighting) {
    return lighting.enabled && lighting.hasAuthoredLighting;
}

inline constexpr bool shouldRenderImportedDirectionalShadows(
    const ImportedInteriorLighting& lighting) {
    return !useAuthoredImportedInteriorLighting(lighting);
}

inline constexpr bool shouldRenderImportedSky(const ImportedInteriorLighting& lighting) {
    return !useAuthoredImportedInteriorLighting(lighting) || lighting.showSky;
}

inline constexpr bool shouldUseImportedSkyLighting(const ImportedInteriorLighting& lighting) {
    return !useAuthoredImportedInteriorLighting(lighting) || lighting.useSkyLighting;
}

inline constexpr bool shouldUseImportedPointShadowMaps(
    const ImportedInteriorLighting& lighting) {
    return useAuthoredImportedInteriorLighting(lighting) &&
        (lighting.localShadowMode == ImportedInteriorLighting::LocalShadowMode::ShadowMaps ||
         lighting.localShadowMode ==
             ImportedInteriorLighting::LocalShadowMode::ShadowMapsWithContact);
}

inline constexpr bool shouldUseImportedContactShadows(
    const ImportedInteriorLighting& lighting) {
    return useAuthoredImportedInteriorLighting(lighting) &&
        lighting.localShadowMode ==
            ImportedInteriorLighting::LocalShadowMode::ShadowMapsWithContact;
}

inline constexpr bool shouldUseImportedRayTracedLocalShadows(
    const ImportedInteriorLighting& lighting) {
    return useAuthoredImportedInteriorLighting(lighting) &&
        lighting.localShadowMode == ImportedInteriorLighting::LocalShadowMode::RayTraced;
}

inline constexpr bool shouldUseImportedScreenSpaceGi(
    const ImportedInteriorLighting& lighting) {
    return useAuthoredImportedInteriorLighting(lighting) &&
        lighting.indirectLightingMode ==
            ImportedInteriorLighting::IndirectLightingMode::ScreenSpaceDiffuse;
}

// Which tone curve the post pass runs, and its parameters.
//
// The ENB values are Enhanced Shaders' own defaults for Fallout: New Vegas,
// read out of its enbeffect.fx.ini. They were tuned against the same weather
// records this engine now reads, which is why they are a better starting point
// for that game than numbers picked here would be.
enum class TonemapMode : std::uint32_t {
    Aces = 0,  // the fixed rational fit; what every game rendered with before
    Enb = 1,   // extended Reinhard, contrast on magnitude, saturation on hue
};

struct TonemapSettings {
    TonemapMode mode = TonemapMode::Aces;
    float contrast = 1.35f;             // ENB "Contrast Day"
    float saturation = 1.25f;           // ENB "Saturation Day"
    float curve = 8.0f;                 // ENB "ToneMapping Curve Day"
    float overbrightDampening = 75.0f;  // ENB "Overbright Dampening Day"
    // ACES-path highlight shaping. The scene-linear value (post-exposure) that
    // should reach display white, and how much of that normalization to apply.
    //
    // 0 is the DEFAULT and is exactly the plain fit, so no existing game's look
    // changes. Above 0 the curve is divided by its own value at this point --
    // the standard curve(x)/curve(W) construction -- which is the only knob in
    // this chain that moves the top of the histogram. Without it, auto-exposure
    // holding the scene near middle grey leaves the frame's 99th percentile
    // around 0.65 and the brightest third of the display range unused.
    float whitePoint = 0.0f;
    float highlightShoulder = 0.7f;
};

// The post colour grade, as one value instead of fourteen scattered fields on
// the debug-settings blob.
//
// The chain in tone_map.frag.slang runs UNCONDITIONALLY -- there is no enable
// bit -- so "no grading" is this struct's defaults, every term at identity, and
// that is what Renderer::setNeutralColorGrading() writes. Anything else is a
// look, and a look is worth naming and measuring rather than leaving as whatever
// the debug panel happened to be initialised with.
struct ColorGradingSettings {
    float whiteBalance[3] = {1.0f, 1.0f, 1.0f};
    float contrast = 1.0f;         // clamped to [0.70, 1.40] in the shader
    float midtoneContrast = 1.0f;  // clamped to [0.80, 1.40]
    float saturation = 1.0f;
    // Vibrance pushes the LEAST saturated pixels hardest, so it compounds with
    // saturation in a way that is easy to over-apply: a dusty or foggy scene is
    // mostly low-saturation pixels, which is exactly what this term targets.
    float vibrance = 0.0f;
    // Below 1 DARKENS shadows. Worth being careful with on any scene carrying
    // real aerial perspective -- the fog IS the depth cue, and crushing it to
    // black removes the thing that made the distance readable.
    float shadowDensity = 1.0f;
    float shadowTint[3] = {0.0f, 0.0f, 0.0f};
    float highlightTint[3] = {0.0f, 0.0f, 0.0f};
};

struct CameraPose {
    float x;
    float y;
    float z;
    float yawDegrees;
    float pitchDegrees;
    float fovDegrees;
    bool orthographic = false;
    float orthoHalfHeight = 1000.0f;
};

struct VoxelPreview {
    enum class Mode {
        Add,
        Remove
    };

    bool visible = false;
    int x = 0;
    int y = 0;
    int z = 0;
    int brushSize = 1;
    Mode mode = Mode::Add;
    bool faceVisible = false;
    int faceX = 0;
    int faceY = 0;
    int faceZ = 0;
    std::uint32_t faceId = 0;
    bool pipeStyle = false;
    float pipeAxisX = 0.0f;
    float pipeAxisY = 1.0f;
    float pipeAxisZ = 0.0f;
    float pipeRadius = 0.45f;
    float pipeStyleId = 0.0f;
};

struct ImportedActorFrameData {
    std::span<const odai::importer::ImportedScenePackedVertex> vertices;
    std::span<const std::uint32_t> indices;
    std::span<const odai::importer::ImportedScenePackedDraw> draws;
};

// A rest-pose (bind-pose) vertex for a GPU-skinned mesh (see docs/ROADMAP.md's
// Party RPG / Narrative section). position/normal feed the skinning compute
// pass; color/uv/textureIndex/flags pass through unchanged into the skinned
// output, matching ImportedMeshVertex's layout exactly so the existing
// imported_static.vert/frag.slang consume it with no changes. Up to 4 bone
// influences per vertex.
struct ImportedSkinnedMeshVertex {
    float position[3] = {};
    float normal[3] = {};
    float color[3] = {};
    float uv[2] = {};
    std::uint32_t textureIndex = 0xffffffffu;
    std::uint32_t flags = 0u;
    std::uint16_t boneIndices[4] = {};
    float boneWeights[4] = {};
};

// A one-time, device-local GPU-skinned mesh template: rest-pose geometry for
// a skinned actor, uploaded once via Renderer::uploadSkinnedMeshTemplate. Posed
// per-frame through ImportedSkinnedActorFrameData without re-uploading
// geometry every frame.
struct ImportedSkinnedMeshTemplate {
    std::span<const ImportedSkinnedMeshVertex> vertices;
    std::span<const std::uint32_t> indices;
    std::span<const odai::importer::ImportedScenePackedDraw> draws;
    std::uint32_t boneCount = 0;
};

// One frame's pose for a previously uploaded ImportedSkinnedMeshTemplate.
// boneMatrices.size() must equal that instance slot's bound template's
// boneCount.
struct ImportedSkinnedActorFrameData {
    std::span<const odai::math::Matrix4> boneMatrices;
};

// Skinning supports a fixed number of independent instance slots. Each has its
// own rest-pose template, pose, and draws via
// Renderer::uploadSkinnedMeshTemplate(instanceIndex, ...) /
// Renderer::setSkinnedActorPose(instanceIndex, ...).
//
// This is still NOT a mass-battle crowd system (see docs/ROADMAP.md's
// out-of-scope note on thousands of units): every slot is fully independent,
// so an ACTIVE one costs its own device-local rest-pose/index/output buffers,
// its own descriptor-buffer set, and one compute dispatch plus one FrameArena
// pose upload per frame. The ceiling is what a populated Fallout settlement
// needs -- Goodsprings alone places 37 actors -- not what a battle does.
//
// An UNUSED slot costs only an empty struct, so the array being larger than any
// one game needs is close to free; raising it from the original 8 was measured
// at no startup or frame cost with nothing extra uploaded.
inline constexpr std::uint32_t kMaxSkinnedInstances = 48;

enum class InventoryItemId : std::uint8_t {
    Empty = 0,
    Stone = 1,
    Dirt = 2,
    Grass = 3,
    Wood = 4,
    Red = 5,
};

static constexpr std::size_t kGameplayHotbarSlotCount = 9;
static constexpr std::size_t kCreativeInventoryItemCount = 5;

struct GameplayUiRect {
    float minX = 0.0f;
    float minY = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;

    [[nodiscard]] bool contains(float x, float y) const {
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
    }
};

struct GameplayUiLayout {
    GameplayUiRect hotbarPanel{};
    std::array<GameplayUiRect, kGameplayHotbarSlotCount> hotbarSlots{};
    GameplayUiRect inventoryPanel{};
    std::array<GameplayUiRect, kCreativeInventoryItemCount> inventorySlots{};
};

// How much of the renderer's ImGui surface is built.
//
// The full panel is a tuning console -- FOV, shadow bias, AO mode, the whole
// sky and exposure chain -- which is the right thing when you are dialling a
// look in and the wrong thing when you are watching numbers during play. Stats
// keeps the readouts (frame timings, GPU stages, memory, draw calls, whatever
// the game pushed through setDebugStatGroups) and drops every control.
enum class DebugUiMode : std::uint8_t {
    Stats = 0,
    Full,
};

// One labelled readout in the debug stats window.
//
// Deliberately just two strings. The alternative -- the renderer knowing what a
// "cell streamer" is so it can format one -- would drag importer types across
// the src/render boundary for the sake of a debug panel. The game formats its
// own numbers and the renderer only prints them.
struct DebugStatRow {
    std::string label;
    std::string value;
};

struct DebugStatGroup {
    std::string title;
    std::vector<DebugStatRow> rows;
};

struct GameplayUiState {
    bool inventoryVisible = false;
    std::uint32_t selectedHotbarSlot = 0;
    std::array<InventoryItemId, kGameplayHotbarSlotCount> hotbarItems{};
    std::array<InventoryItemId, kCreativeInventoryItemCount> creativeInventoryItems = {
        InventoryItemId::Stone,
        InventoryItemId::Dirt,
        InventoryItemId::Grass,
        InventoryItemId::Wood,
        InventoryItemId::Red,
    };
};

inline GameplayUiLayout buildGameplayUiLayout(float displayWidth, float displayHeight) {
    GameplayUiLayout layout{};
    const float hotbarSlotSize = 52.0f;
    const float hotbarGap = 8.0f;
    const float hotbarWidth =
        (hotbarSlotSize * static_cast<float>(kGameplayHotbarSlotCount)) +
        (hotbarGap * static_cast<float>(kGameplayHotbarSlotCount - 1));
    const float hotbarMinX = (displayWidth - hotbarWidth) * 0.5f;
    const float hotbarMinY = displayHeight - 84.0f;
    layout.hotbarPanel = {
        hotbarMinX - 14.0f,
        hotbarMinY - 14.0f,
        hotbarMinX + hotbarWidth + 14.0f,
        hotbarMinY + hotbarSlotSize + 14.0f
    };
    for (std::size_t slotIndex = 0; slotIndex < kGameplayHotbarSlotCount; ++slotIndex) {
        const float slotMinX = hotbarMinX + (static_cast<float>(slotIndex) * (hotbarSlotSize + hotbarGap));
        layout.hotbarSlots[slotIndex] = {
            slotMinX,
            hotbarMinY,
            slotMinX + hotbarSlotSize,
            hotbarMinY + hotbarSlotSize
        };
    }

    const float inventorySlotSize = 76.0f;
    const float inventoryGap = 18.0f;
    const float inventoryWidth =
        (inventorySlotSize * static_cast<float>(kCreativeInventoryItemCount)) +
        (inventoryGap * static_cast<float>(kCreativeInventoryItemCount - 1));
    const float inventoryPanelMinX = (displayWidth - (inventoryWidth + 56.0f)) * 0.5f;
    const float inventoryPanelMinY = (displayHeight - 214.0f) * 0.5f;
    layout.inventoryPanel = {
        inventoryPanelMinX,
        inventoryPanelMinY,
        inventoryPanelMinX + inventoryWidth + 56.0f,
        inventoryPanelMinY + 214.0f
    };
    const float inventorySlotMinX = layout.inventoryPanel.minX + 28.0f;
    const float inventorySlotMinY = layout.inventoryPanel.minY + 84.0f;
    for (std::size_t itemIndex = 0; itemIndex < kCreativeInventoryItemCount; ++itemIndex) {
        const float slotMinX = inventorySlotMinX + (static_cast<float>(itemIndex) * (inventorySlotSize + inventoryGap));
        layout.inventorySlots[itemIndex] = {
            slotMinX,
            inventorySlotMinY,
            slotMinX + inventorySlotSize,
            inventorySlotMinY + inventorySlotSize
        };
    }
    return layout;
}

} // namespace odai::render
