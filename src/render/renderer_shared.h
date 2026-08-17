#include <cstdlib>
#include <cstring>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

namespace {

constexpr std::array<const char*, 1> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};
// Only extensions not yet promoted to Vulkan 1.4 core are listed here.
// Promoted-to-core features (timelineSemaphore/1.2, synchronization2/1.3,
// dynamicRendering/1.3, maintenance4/1.3) are enabled via VkPhysicalDeviceVulkan1xFeatures
// chains in init.cc and do not need extension strings at 1.4+.
constexpr std::array<const char*, 2> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
};
// VK_EXT_memory_priority is a residency *hint* to the allocator, not something
// any pass depends on: VMA drops the priority plumbing and allocates normally
// without it. Requiring it disqualified otherwise-capable hardware (Mesa's Intel
// driver does not expose it), so it is probed per candidate and enabled only
// where present -- same treatment as descriptor buffer and ray tracing.
constexpr const char* kOptionalMemoryPriorityExtension = VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME;
constexpr uint32_t kBindlessTargetTextureCapacity = 1024;
constexpr uint32_t kBindlessMinTextureCapacity = 64;
constexpr uint32_t kBindlessReservedSampledDescriptors = 16;
constexpr uint32_t kBindlessTextureIndexDiffuse = 0;
constexpr uint32_t kBindlessTextureIndexHdrResolved = 1;
constexpr uint32_t kBindlessTextureIndexShadowAtlas = 2;
constexpr uint32_t kBindlessTextureIndexNormalDepth = 3;
constexpr uint32_t kBindlessTextureIndexSsaoBlur = 4;
constexpr uint32_t kBindlessTextureIndexSsaoRaw = 5;
constexpr uint32_t kBindlessTextureIndexPlantDiffuse = 6;
constexpr uint32_t kBindlessTextureIndexSkyDaylight = 7;
constexpr uint32_t kBindlessTextureIndexWaterNormal = 8;
constexpr uint32_t kBindlessTextureIndexTerrainDetail = 9;
constexpr uint32_t kBindlessTextureIndexFogMap = 10;
// Number of fixed singleton slots at the head of the bindless table; imported
// scene textures are assigned from here upward (chunk_upload.cc).
//
// MIRRORED, and not enforced by the compiler — this header is included inside
// `namespace odai::render` by some TUs and not at all by others, so each copy is
// a distinct entity and a mismatch is silent. Keep these three in sync:
//   * this block,
//   * the private copy in backend/vulkan/descriptors.cc (which writes the
//     descriptors these indices name),
//   * kBindlessIndexFogMap in shaders/imported_static.frag.slang.
// They were out of step until now: this header lacked FogMap and stopped at 10
// while descriptors.cc wrote at 11+i, so imported texture 0 sampled the fog map.
constexpr uint32_t kBindlessTextureStaticCount = 11;
// "No bindless slot" -- returned when a texture is unusable or the bindless
// table is full. Matches kInvalidImportedTextureIndex on the shader side, which
// falls back to the vertex-colour palette rather than sampling slot 0.
constexpr uint32_t kInvalidImportedTextureSlot = 0xFFFFFFFFu;
constexpr uint32_t kShadowCascadeCount = 4;
constexpr uint32_t kImportedLocalLightCapacity = 64;
// Clustered (Forward+) light culling grid. MIRRORED in
// src/render/shaders/light_clusters.slang -- change both together. A mismatch
// does not fail: the compute pass and the fragment shader simply disagree about
// which cluster a pixel is in, and lights switch off in bands across the screen.
constexpr uint32_t kLightClusterTileSize = 64;
constexpr uint32_t kLightClusterSliceCount = 24;
// The three shadow-atlas constants below are one layout expressed three ways and
// must be edited together: kShadowCascadeResolution feeds the texel-snapping math
// (frame_run.cc), kShadowAtlasRects places each cascade in the atlas and derives
// the sampling UV rects, and kShadowAtlasSize is the atlas dimension the rects are
// normalized against. A fourth mirror lives outside this header:
// renderer_backend.h's private kShadowAtlasSize, which is what the image
// allocation actually uses (class scope wins inside member functions). If these
// disagree, cascades sample the wrong atlas region instead of failing loudly.
// Halved from 4096/2048/2048/1024 in an 8192 atlas after GPU timestamps put
// the shadow pass at ~5 ms of a ~20 ms frame on the Intel LNL iGPU -- a
// quarter of the whole GPU budget rasterizing shadow maps. Halving the linear
// resolution quarters the rasterized texels. Cascade 0's texel goes from
// ~0.5 to ~1.0 world units (~1.5 cm at Fallout scale), and TAA now averages
// the shadow edges temporally, which is what makes the coarser maps
// acceptable where they were not before it existed.
// AND THE 4096 ATLAS WAS ONLY 39% ALLOCATED. 2048/1024/1024/512 occupies
// 6.55M of 16.78M texels; the image was already paying for all of it. Worse,
// the cascade covering the LARGEST area got the SMALLEST tile, so the far field
// -- which is most of the screen on any camera looking out across a landscape
// -- was resolved four times more coarsely than the ground under your feet.
//
// Four equal 2048 tiles fill the atlas exactly (a 2x2 grid) and take cascade 3
// from 512 to 2048 texels. What that is worth, measured on a Vvardenfell flight
// at 24000 units of shadow distance: cascade 3's texel goes from 122.8 world
// units to 30.7, i.e. the far cascade now resolves what the NEAR one used to.
constexpr std::array<uint32_t, kShadowCascadeCount> kShadowCascadeResolution = {2048u, 2048u, 2048u, 2048u};
struct ShadowAtlasRect {
    uint32_t x;
    uint32_t y;
    uint32_t size;
};
constexpr std::array<ShadowAtlasRect, kShadowCascadeCount> kShadowAtlasRects = {
    ShadowAtlasRect{0u, 0u, 2048u},
    ShadowAtlasRect{2048u, 0u, 2048u},
    ShadowAtlasRect{0u, 2048u, 2048u},
    ShadowAtlasRect{2048u, 2048u, 2048u}
};
constexpr uint32_t kShadowAtlasSize = 4096u;
// Authored interiors reuse the directional atlas for up to 35 local lights.
// Every cubemap uses six 256x256 faces arranged as a 3x2 tile. Five tiles per
// row by seven rows occupy 3840x3584 of the 4096 atlas. Dragonsreach has 34
// active lights (28 authored plus the fire emitters), so every light that can
// fill a table/contact shadow now has a map. This is also less raster work than
// the old 8x512 + 8x256 split (13.76M vs 15.73M depth texels).
// The selection stays frozen until geometry residency changes, so camera
// motion cannot trigger an atlas rebuild.
constexpr uint32_t kInteriorPointShadowLightCount = 35u;
constexpr uint32_t kInteriorPointShadowFaceCount = 6u;
constexpr uint32_t kInteriorPointShadowFaceSize = 256u;
constexpr uint32_t kInteriorPointShadowCubesPerRow = 5u;
constexpr uint32_t kInteriorPointShadowMatrixCount =
    kInteriorPointShadowLightCount * kInteriorPointShadowFaceCount;
static_assert(
    kInteriorPointShadowCubesPerRow * 3u * kInteriorPointShadowFaceSize <= kShadowAtlasSize);
static_assert(
    ((kInteriorPointShadowLightCount + kInteriorPointShadowCubesPerRow - 1u) /
     kInteriorPointShadowCubesPerRow) * 2u * kInteriorPointShadowFaceSize <= kShadowAtlasSize);
constexpr uint32_t kVoxelGiGridResolution = 64u;
constexpr uint32_t kVoxelGiWorkgroupSize = 4u;
constexpr uint32_t kVoxelGiPropagationIterations = 8u;
constexpr uint32_t kHdrResolveBloomMipCount = 6u;
constexpr uint32_t kAutoExposureHistogramBins = 64u;
constexpr uint32_t kAutoExposureWorkgroupSize = 16u;
constexpr uint32_t kSunShaftWorkgroupSize = 8u;
constexpr uint32_t kSsaoComputeWorkgroupSize = 8u;
constexpr float kVoxelGiCellSize = 1.0f;
constexpr float kPipeTransferHalfExtent = 0.58f;
constexpr float kPipeMinRadius = 0.02f;
constexpr float kPipeMaxRadius = 0.5f;
constexpr float kPipeBranchRadiusBoost = 0.05f;
constexpr float kPipeMaxEndExtension = 0.49f;
constexpr float kBeltRadius = 0.49f;
constexpr float kTrackRadius = 0.38f;
constexpr odai::math::Vector3 kBeltTint{0.78f, 0.62f, 0.18f};
constexpr odai::math::Vector3 kTrackTint{0.52f, 0.54f, 0.58f};
constexpr float kBeltCargoLength = 0.30f;
constexpr float kBeltCargoRadius = 0.30f;
constexpr std::array<odai::math::Vector3, 5> kBeltCargoTints = {
    odai::math::Vector3{0.92f, 0.31f, 0.31f},
    odai::math::Vector3{0.31f, 0.71f, 0.96f},
    odai::math::Vector3{0.95f, 0.84f, 0.32f},
    odai::math::Vector3{0.56f, 0.88f, 0.48f},
    odai::math::Vector3{0.84f, 0.54f, 0.92f},
};
constexpr uint64_t kAcquireNextImageTimeoutNs = 2000000ull; // 2 ms
constexpr uint64_t kFrameTimelineWarnLagThreshold = 6u;
constexpr double kFrameTimelineWarnCooldownSeconds = 2.0;
constexpr float kCpuFrameEwmaAlpha = 0.08f;

void imguiCheckVkResult(VkResult result) {
    if (result != VK_SUCCESS) {
        VOX_LOGE("imgui") << "Vulkan backend error: " << static_cast<int>(result);
    }
}

struct alignas(16) CameraUniform {
    float mvp[16];
    float view[16];
    float proj[16];
    float lightViewProj[kShadowCascadeCount][16];
    float invLightViewProj[kShadowCascadeCount][16];
    float shadowCascadeSplits[4];
    float shadowAtlasUvRects[kShadowCascadeCount][4];
    float sunDirectionIntensity[4];
    float sunColorShadow[4];
    float shIrradiance[9][4];
    float shadowConfig0[4];
    float shadowConfig1[4];
    float shadowConfig2[4];
    float shadowConfig3[4];
    float shadowConfig4[4];
    float shadowVoxelGridOrigin[4];
    float shadowVoxelGridSize[4];
    float skyConfig0[4];
    float skyConfig1[4];
    float skyConfig2[4];
    float skyConfig3[4];
    float skyConfig4[4];
    float skyConfig5[4];
    float voxelGiRestirConfig0[4];
    float voxelGiRestirConfig1[4];
    float colorGrading0[4];
    float colorGrading1[4];
    float colorGrading2[4];
    float colorGrading3[4];
    float dofConfig[4];
    float dofConfig2[4];
    float waterConfig[4];
    float importedLightPositionRadius[kImportedLocalLightCapacity][4];
    float importedLightColorIntensity[kImportedLocalLightCapacity][4];
    float importedLightConfig[4];
    float voxelBaseColorPalette[16][4];
    float voxelGiGridOriginCellSize[4];
    float voxelGiGridExtentStrength[4];
    float fogMapConfig[4]; // [0]=invExtentX, [1]=invExtentZ, [2]=unused, [3]=enabled
    // Authored sky, from a Fallout WTHR record. The procedural Rayleigh/Mie sky
    // stays the default and stays the only path when weight is 0 -- these blend
    // over it rather than replacing it, so every other game renders exactly as
    // before. Appended at the end of the block so no existing field's offset
    // moves. Mirrored in src/render/shaders/camera_uniform.slang.
    float weatherSkyUpper[4];  // [0..2]=linear rgb at zenith, [3]=blend weight 0..1
    float weatherSkyLower[4];  // [0..2]=linear rgb above the horizon band, [3]=unused
    float weatherHorizon[4];   // [0..2]=linear rgb at the horizon line, [3]=sun-glare scale
    float weatherFog[4];       // [0..2]=linear fog rgb, [3]=fog far distance in world units
    // Four cloud layers. [0..2]=linear tint, [3]=opacity (0 = layer off).
    float weatherCloudTint[4][4];
    // [0]=bindless texture slot as a float, [1]=drift u, [2]=scale (dome scale
    // for a fisheye, tiling count otherwise), [3]=drift v. Slot is carried as a
    // float because the whole block is floats; the shader rounds it to an index.
    float weatherCloudParams[4][4];
    // [0..1]=the dir.y window this layer covers, [2]=WeatherCloudMapping as a
    // float, [3]=unused. Skyrim stacks an overhead deck and a horizon bank in
    // one sky; they need different projections and different slices of it.
    float weatherCloudBand[4][4];
    // Tonemap selection and parameters.
    // [0] = mode: 0 = the ACES fit, 1 = the ENB/Enhanced Shaders curve
    // [1] = contrast, [2] = saturation, [3] = curve knee (ENB's "ToneMapping Curve")
    float tonemapConfig[4];
    // [0] = overbright dampening (ENB's "Overbright Dampening")
    // [1] = DebugView (renderer_types.h), 0 = off. Carried as a float because
    //       the whole block is floats; the shader rounds it back to an index.
    // [2..3] = this frame's TAA sub-pixel jitter in NDC units, 0 when TAA or
    //       jitter is off. Anything mapping a UV back to a view position must
    //       subtract it: a jittered projection moves the NDC of a texel centre
    //       off the usual uv*2-1.
    float tonemapConfig2[4];
    // Terrain layer-blend shaping, for the ATXT/VTXT weights in
    // imported_static.frag.slang.
    //
    // Tunable rather than baked because these numbers decide whether a painted
    // road reads as an organic edge or as a set of hard triangular wedges, and
    // that judgement needs one round trip through a render per value. The
    // defaults live in the renderer, not here.
    //
    // [0] = sharpness in 0..1. 0 is a PLAIN LERP of the authored weight and is
    //       the control any comparison needs; 1 is the full smoothstep.
    // [1] = world units per coarse noise cell, [2] = per fine noise cell.
    //       Bigger than the 128-unit LAND post spacing or the noise cannot move
    //       a boundary far enough to break the lattice it sits on.
    // [3] = how far the noise may displace a boundary, in weight units.
    float terrainBlendConfig[4];
    // HDR highlight shaping for the ACES path.
    //
    // [0] = white point, in post-exposure scene-linear units. The scene value
    //       that should map to display white. 0 DISABLES the normalization and
    //       renders the plain fit byte-identically, which is what keeps every
    //       other game unaffected.
    // [1] = shoulder strength in 0..1, how much of the normalization to apply.
    // [2..3] = unused.
    //
    // Why this exists: the Narkowicz ACES fit reaches 1.0 only asymptotically,
    // so with auto-exposure holding the scene near middle grey NOTHING in the
    // frame reaches display white. Measured across a Morrowind flight, the 99th
    // percentile of luma sat at 0.64-0.70 in every single frame and moved by
    // less than 0.02 under every existing knob -- fog distance, ENB curve, the
    // stylized colour look. The top third of the display range was simply never
    // addressed, which is what "flat" looked like.
    float hdrHighlightConfig[4];
    // Clustered (Forward+) local-light culling. See
    // src/render/shaders/light_clusters.slang for the scheme.
    //
    // config0 = (grid x, grid y, grid z, tile size in pixels)
    // config1 = (slice scale, slice bias, unused, unused)
    //
    // A ZERO GRID MEANS THE PASS DID NOT RUN and every consumer falls back to
    // walking the full light array. That is the safe direction: a stale mask
    // would silently unlight geometry, and "no lights this frame" is not
    // distinguishable from "the cull pass was skipped" at the shader.
    float lightClusterConfig0[4];
    float lightClusterConfig1[4];
    // Shadow atlas geometry the fragment shaders need per PCF tap.
    //
    // [0] = one atlas texel in UV, i.e. 1/kShadowAtlasSize. [1..3] unused.
    //
    // THIS EXISTS BECAUSE `shadowMap.GetDimensions()` IS NOT FREE. Every PCF
    // site asked the sampler for the atlas size, and the cascade blend asks
    // twice; on the LNL iGPU that lowers to a real resinfo message the compiler
    // will not hoist out of the fragment. Measured on Whiterun at 1080p:
    // replacing the two queries with this constant took the main pass from
    // 17.15 ms to 15.36 ms -- 1.79 ms, for byte-identical output.
    //
    // Appended at the end of the block so no existing field's offset moves.
    // Mirrored in src/render/shaders/camera_uniform.slang.
    float shadowAtlasConfig[4];
    // Authored imported-interior CELL lighting. Appended to preserve every
    // existing offset; mirrored in camera_uniform.slang.
    // ambient.w = has authored XCLL
    float interiorAmbient[4];
    // directional.w = use sky lighting
    float interiorDirectional[4];
    // fog.w = show sky
    float interiorFog[4];
    // x/y = fog near/far, z = directional shadows enabled, w = interior enabled
    float interiorFogRange[4];
    // Interior point lights x six cubemap faces. These matrices are used both to
    // rasterize the atlas and to project a receiver into the matching face.
    float interiorPointShadowViewProj[kInteriorPointShadowMatrixCount][16];
    // Selected imported-light index for atlas slots 0..3; -1 means unused.
    // Indexed by uploaded/clustered light, value is the stable atlas slot.
    float interiorPointShadowLightIndices[kImportedLocalLightCapacity / 4u][4];
    // x = active slot count, y = face UV scale, z = atlas texel size,
    // w = enabled. Appended to keep every pre-existing uniform offset stable.
    float interiorPointShadowParams[4];
    // Contact-shadow reconstruction and main-pass lookup. invView maps the
    // normal/depth prepass's reconstructed view position back to world space.
    // config = (full width, full height, enabled, four-frame phase).
    float contactShadowInvView[16];
    float contactShadowConfig[4];
    // Quarter-resolution diffuse GI. x/y are record extent, z is enabled,
    // w is the receiver bounce scale. Appended to preserve existing offsets.
    float screenSpaceGiConfig[4];
    // Legacy imported-material PBR override. x/y = object/terrain roughness,
    // z = metallic, w = enabled. Appended; mirrored in camera_uniform.slang.
    float importedPbrConfig[4];
    // Dominant planar-water reflection for this frame.
    // x = horizontal water plane in world Y, y = pass valid, z/w unused.
    // Appended so all existing camera-uniform offsets remain stable.
    float waterReflectionConfig[4];
};

struct alignas(16) ChunkPushConstants {
    float chunkOffset[4];
    float cascadeData[4];
    // [0] = alpha-test threshold in 0..1 for the draw being recorded, [1..3]
    // spare. Mirrored in src/render/shaders/chunk_push_constants.slang and in
    // the private copy of this struct in pass_pipelines.cc -- all three must
    // agree or the pipeline layout's push range stops covering the block.
    float materialParams[4];
};

struct alignas(16) ChunkInstanceData {
    float chunkOffset[4];
};

struct alignas(16) AutoExposureHistogramPushConstants {
    uint32_t width = 1u;
    uint32_t height = 1u;
    uint32_t totalPixels = 1u;
    uint32_t binCount = kAutoExposureHistogramBins;
    float minLogLuminance = -10.0f;
    float maxLogLuminance = 4.0f;
    float sourceMipLevel = 0.0f;
    float _pad1 = 0.0f;
};

struct alignas(16) AutoExposureUpdatePushConstants {
    uint32_t totalPixels = 1u;
    uint32_t binCount = kAutoExposureHistogramBins;
    uint32_t resetHistory = 1u;
    uint32_t _pad0 = 0u;
    float minLogLuminance = -10.0f;
    float maxLogLuminance = 4.0f;
    float lowPercentile = 0.5f;
    float highPercentile = 0.98f;
    float keyValue = 0.18f;
    float minExposure = 0.25f;
    float maxExposure = 2.2f;
    float adaptUpRate = 3.0f;
    float adaptDownRate = 1.4f;
    float deltaTimeSeconds = 0.016f;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
};

struct alignas(16) SunShaftPushConstants {
    uint32_t width = 1u;
    uint32_t height = 1u;
    uint32_t sampleCount = 10u;
    uint32_t _pad0 = 0u;
};

struct PipeMeshData {
    struct Vertex {
        float position[3];
        float normal[3];
    };
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

odai::world::ChunkMeshData buildSingleVoxelPreviewMesh(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t ao,
    std::uint32_t material
) {
    odai::world::ChunkMeshData mesh{};
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    for (std::uint32_t faceId = 0; faceId < 6; ++faceId) {
        const std::uint32_t baseVertex = static_cast<std::uint32_t>(mesh.vertices.size());
        for (std::uint32_t corner = 0; corner < 4; ++corner) {
            odai::world::PackedVoxelVertex vertex{};
            vertex.bits = odai::world::PackedVoxelVertex::pack(x, y, z, faceId, corner, ao, material, 0u, 2u);
            mesh.vertices.push_back(vertex);
        }

        mesh.indices.push_back(baseVertex + 0);
        mesh.indices.push_back(baseVertex + 1);
        mesh.indices.push_back(baseVertex + 2);
        mesh.indices.push_back(baseVertex + 0);
        mesh.indices.push_back(baseVertex + 2);
        mesh.indices.push_back(baseVertex + 3);
    }

    return mesh;
}

void appendBoxMesh(
    PipeMeshData& mesh,
    float minX,
    float minY,
    float minZ,
    float maxX,
    float maxY,
    float maxZ
) {
    auto appendFace = [&mesh](
                          const std::array<std::array<float, 3>, 4>& corners,
                          const std::array<float, 3>& normal
                      ) {
        const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
        for (const std::array<float, 3>& corner : corners) {
            PipeMeshData::Vertex vertex{};
            vertex.position[0] = corner[0];
            vertex.position[1] = corner[1];
            vertex.position[2] = corner[2];
            vertex.normal[0] = normal[0];
            vertex.normal[1] = normal[1];
            vertex.normal[2] = normal[2];
            mesh.vertices.push_back(vertex);
        }
        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 1u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 0u);
        mesh.indices.push_back(base + 2u);
        mesh.indices.push_back(base + 3u);
    };

    appendFace(
        {{
            {{maxX, minY, minZ}},
            {{maxX, maxY, minZ}},
            {{maxX, maxY, maxZ}},
            {{maxX, minY, maxZ}},
        }},
        {{1.0f, 0.0f, 0.0f}}
    );
    appendFace(
        {{
            {{minX, minY, maxZ}},
            {{minX, maxY, maxZ}},
            {{minX, maxY, minZ}},
            {{minX, minY, minZ}},
        }},
        {{-1.0f, 0.0f, 0.0f}}
    );
    appendFace(
        {{
            {{minX, maxY, minZ}},
            {{minX, maxY, maxZ}},
            {{maxX, maxY, maxZ}},
            {{maxX, maxY, minZ}},
        }},
        {{0.0f, 1.0f, 0.0f}}
    );
    appendFace(
        {{
            {{minX, minY, maxZ}},
            {{minX, minY, minZ}},
            {{maxX, minY, minZ}},
            {{maxX, minY, maxZ}},
        }},
        {{0.0f, -1.0f, 0.0f}}
    );
    appendFace(
        {{
            {{minX, minY, maxZ}},
            {{maxX, minY, maxZ}},
            {{maxX, maxY, maxZ}},
            {{minX, maxY, maxZ}},
        }},
        {{0.0f, 0.0f, 1.0f}}
    );
    appendFace(
        {{
            {{maxX, minY, minZ}},
            {{minX, minY, minZ}},
            {{minX, maxY, minZ}},
            {{maxX, maxY, minZ}},
        }},
        {{0.0f, 0.0f, -1.0f}}
    );
}

PipeMeshData buildTransportBoxMesh() {
    PipeMeshData mesh{};
    mesh.vertices.reserve(24u);
    mesh.indices.reserve(36u);
    appendBoxMesh(
        mesh,
        -kPipeTransferHalfExtent,
        0.0f,
        -kPipeTransferHalfExtent,
        kPipeTransferHalfExtent,
        1.0f,
        kPipeTransferHalfExtent
    );
    return mesh;
}

PipeMeshData buildPipeCylinderMesh() {
    PipeMeshData mesh{};
    constexpr std::uint32_t kSegments = 16u;
    mesh.vertices.reserve(static_cast<std::size_t>(kSegments * 4u + 2u));
    mesh.indices.reserve(static_cast<std::size_t>(kSegments * 12u));

    const float radius = kPipeTransferHalfExtent;
    const float twoPi = 6.28318530718f;

    for (std::uint32_t i = 0; i < kSegments; ++i) {
        const float t0 = (static_cast<float>(i) / static_cast<float>(kSegments)) * twoPi;
        const float t1 = (static_cast<float>(i + 1u) / static_cast<float>(kSegments)) * twoPi;
        const float x0 = std::cos(t0) * radius;
        const float z0 = std::sin(t0) * radius;
        const float x1 = std::cos(t1) * radius;
        const float z1 = std::sin(t1) * radius;

        // Side quad
        const std::uint32_t sideBase = static_cast<std::uint32_t>(mesh.vertices.size());
        PipeMeshData::Vertex v0{{x0, 0.0f, z0}, {std::cos(t0), 0.0f, std::sin(t0)}};
        PipeMeshData::Vertex v1{{x0, 1.0f, z0}, {std::cos(t0), 0.0f, std::sin(t0)}};
        PipeMeshData::Vertex v2{{x1, 1.0f, z1}, {std::cos(t1), 0.0f, std::sin(t1)}};
        PipeMeshData::Vertex v3{{x1, 0.0f, z1}, {std::cos(t1), 0.0f, std::sin(t1)}};
        mesh.vertices.push_back(v0);
        mesh.vertices.push_back(v1);
        mesh.vertices.push_back(v2);
        mesh.vertices.push_back(v3);
        mesh.indices.push_back(sideBase + 0u);
        mesh.indices.push_back(sideBase + 1u);
        mesh.indices.push_back(sideBase + 2u);
        mesh.indices.push_back(sideBase + 0u);
        mesh.indices.push_back(sideBase + 2u);
        mesh.indices.push_back(sideBase + 3u);
    }

    const std::uint32_t bottomCenter = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(PipeMeshData::Vertex{{0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}});
    const std::uint32_t topCenter = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(PipeMeshData::Vertex{{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}});

    for (std::uint32_t i = 0; i < kSegments; ++i) {
        const float t0 = (static_cast<float>(i) / static_cast<float>(kSegments)) * twoPi;
        const float t1 = (static_cast<float>(i + 1u) / static_cast<float>(kSegments)) * twoPi;
        const float x0 = std::cos(t0) * radius;
        const float z0 = std::sin(t0) * radius;
        const float x1 = std::cos(t1) * radius;
        const float z1 = std::sin(t1) * radius;

        const std::uint32_t bottomBase = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(PipeMeshData::Vertex{{x0, 0.0f, z0}, {0.0f, -1.0f, 0.0f}});
        mesh.vertices.push_back(PipeMeshData::Vertex{{x1, 0.0f, z1}, {0.0f, -1.0f, 0.0f}});
        mesh.indices.push_back(bottomCenter);
        mesh.indices.push_back(bottomBase + 1u);
        mesh.indices.push_back(bottomBase + 0u);

        const std::uint32_t topBase = static_cast<std::uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(PipeMeshData::Vertex{{x0, 1.0f, z0}, {0.0f, 1.0f, 0.0f}});
        mesh.vertices.push_back(PipeMeshData::Vertex{{x1, 1.0f, z1}, {0.0f, 1.0f, 0.0f}});
        mesh.indices.push_back(topCenter);
        mesh.indices.push_back(topBase + 0u);
        mesh.indices.push_back(topBase + 1u);
    }

    return mesh;
}

struct PipeEndpointState {
    odai::math::Vector3 axis{0.0f, 1.0f, 0.0f};
    float renderedRadius = 0.45f;
    float startExtension = 0.0f;
    float endExtension = 0.0f;
};

odai::core::Dir6 dominantAxisDir6(const odai::math::Vector3& direction) {
    if (odai::math::lengthSquared(direction) <= 0.000001f) {
        return odai::core::Dir6::PosY;
    }
    const odai::math::Vector3 normalized = odai::math::normalize(direction);
    const float absX = std::abs(normalized.x);
    const float absY = std::abs(normalized.y);
    const float absZ = std::abs(normalized.z);
    if (absX >= absY && absX >= absZ) {
        return normalized.x >= 0.0f ? odai::core::Dir6::PosX : odai::core::Dir6::NegX;
    }
    if (absY >= absX && absY >= absZ) {
        return normalized.y >= 0.0f ? odai::core::Dir6::PosY : odai::core::Dir6::NegY;
    }
    return normalized.z >= 0.0f ? odai::core::Dir6::PosZ : odai::core::Dir6::NegZ;
}

odai::math::Vector3 beltDirectionAxis(odai::sim::BeltDirection direction) {
    switch (direction) {
    case odai::sim::BeltDirection::East:
        return odai::math::Vector3{1.0f, 0.0f, 0.0f};
    case odai::sim::BeltDirection::West:
        return odai::math::Vector3{-1.0f, 0.0f, 0.0f};
    case odai::sim::BeltDirection::South:
        return odai::math::Vector3{0.0f, 0.0f, 1.0f};
    case odai::sim::BeltDirection::North:
    default:
        return odai::math::Vector3{0.0f, 0.0f, -1.0f};
    }
}

odai::math::Vector3 trackDirectionAxis(odai::sim::TrackDirection direction) {
    switch (direction) {
    case odai::sim::TrackDirection::East:
        return odai::math::Vector3{1.0f, 0.0f, 0.0f};
    case odai::sim::TrackDirection::West:
        return odai::math::Vector3{-1.0f, 0.0f, 0.0f};
    case odai::sim::TrackDirection::South:
        return odai::math::Vector3{0.0f, 0.0f, 1.0f};
    case odai::sim::TrackDirection::North:
    default:
        return odai::math::Vector3{0.0f, 0.0f, -1.0f};
    }
}

bool dirSharesAxis(odai::core::Dir6 lhs, odai::core::Dir6 rhs) {
    return lhs == rhs || odai::core::areOpposite(lhs, rhs);
}

float computeRenderedPipeRadius(float baseRadius, bool hasBranchConnection) {
    float renderedRadius = std::clamp(baseRadius, kPipeMinRadius, kPipeMaxRadius);
    if (hasBranchConnection) {
        renderedRadius = std::min(kPipeMaxRadius, renderedRadius + kPipeBranchRadiusBoost);
    }
    return renderedRadius;
}

std::uint64_t pipeCellKey(const odai::core::Cell3i& cell) {
    return odai::core::packCell21(cell);
}

std::vector<PipeEndpointState> buildPipeEndpointStates(
    const std::vector<odai::sim::Pipe>& pipes
) {
    std::unordered_map<std::uint64_t, std::size_t> pipeCellToIndex;
    pipeCellToIndex.reserve(pipes.size() * 2u);
    for (std::size_t i = 0; i < pipes.size(); ++i) {
        const odai::core::Cell3i cell{
            pipes[i].x,
            pipes[i].y,
            pipes[i].z
        };
        pipeCellToIndex.emplace(pipeCellKey(cell), i);
    }

    auto hasPipeAtCell = [&pipeCellToIndex](const odai::core::Cell3i& cell) -> bool {
        return pipeCellToIndex.find(pipeCellKey(cell)) != pipeCellToIndex.end();
    };

    std::vector<odai::core::Dir6> axisDirections(pipes.size(), odai::core::Dir6::PosY);
    std::vector<float> renderedRadii(pipes.size(), 0.45f);
    std::vector<bool> hasBranchConnections(pipes.size(), false);
    for (std::size_t i = 0; i < pipes.size(); ++i) {
        const odai::sim::Pipe& pipe = pipes[i];
        const odai::core::Cell3i cell{
            pipe.x,
            pipe.y,
            pipe.z
        };
        const odai::core::Dir6 axisDir = dominantAxisDir6(pipe.axis);
        const odai::core::Dir6 startDir = odai::core::oppositeDir(axisDir);
        const odai::core::Dir6 endDir = axisDir;
        const std::uint8_t neighborMask = odai::sim::neighborMask6(cell, hasPipeAtCell);
        const std::uint8_t axialMask = static_cast<std::uint8_t>(odai::core::dirBit(startDir) | odai::core::dirBit(endDir));
        const bool hasBranchConnection = (neighborMask & static_cast<std::uint8_t>(~axialMask & 0x3Fu)) != 0u;

        axisDirections[i] = axisDir;
        hasBranchConnections[i] = hasBranchConnection;
        renderedRadii[i] = computeRenderedPipeRadius(pipe.radius, hasBranchConnection);
    }

    auto endExtensionForDirection = [&](
                                        std::size_t pipeIndex,
                                        const odai::core::Cell3i& cell,
                                        odai::core::Dir6 endDirection
                                    ) -> float {
        const odai::core::Cell3i neighborCell = odai::core::neighborCell(cell, endDirection);
        const auto neighborIt = pipeCellToIndex.find(pipeCellKey(neighborCell));
        if (neighborIt == pipeCellToIndex.end()) {
            return 0.0f;
        }

        const std::size_t neighborIndex = neighborIt->second;
        if (neighborIndex >= pipes.size()) {
            return 0.0f;
        }

        if (dirSharesAxis(axisDirections[pipeIndex], axisDirections[neighborIndex])) {
            return 0.0f;
        }

        const float neighborHalfExtent = kPipeTransferHalfExtent * renderedRadii[neighborIndex];
        return std::clamp(0.5f - neighborHalfExtent, 0.0f, kPipeMaxEndExtension);
    };

    std::vector<PipeEndpointState> states(pipes.size());
    for (std::size_t i = 0; i < pipes.size(); ++i) {
        const odai::sim::Pipe& pipe = pipes[i];
        const odai::core::Cell3i cell{
            pipe.x,
            pipe.y,
            pipe.z
        };
        const odai::core::Dir6 axisDir = axisDirections[i];
        const odai::core::Dir6 startDir = odai::core::oppositeDir(axisDir);
        const odai::core::Dir6 endDir = axisDir;
        states[i].axis = odai::core::dirToUnitVector(axisDir);
        states[i].renderedRadius = renderedRadii[i];
        states[i].startExtension = endExtensionForDirection(i, cell, startDir);
        states[i].endExtension = endExtensionForDirection(i, cell, endDir);
    }

    return states;
}

odai::math::Matrix4 transpose(const odai::math::Matrix4& matrix) {
    odai::math::Matrix4 result{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            result(row, col) = matrix(col, row);
        }
    }
    return result;
}

odai::math::Matrix4 perspectiveVulkan(float fovYRadians, float aspectRatio, float nearPlane, float farPlane) {
    return odai::math::perspectiveVulkanReverseZ(fovYRadians, aspectRatio, nearPlane, farPlane);
}

odai::math::Matrix4 orthographicVulkan(
    float left,
    float right,
    float bottom,
    float top,
    float nearPlane,
    float farPlane
) {
    return odai::math::orthographicVulkanReverseZ(left, right, bottom, top, nearPlane, farPlane);
}

bool chunkIntersectsShadowCascadeClip(
    const odai::world::Chunk& chunk,
    const odai::math::Matrix4& lightViewProj,
    float clipMargin
) {
    const float chunkMinX = static_cast<float>(chunk.chunkX() * odai::world::Chunk::kSizeX);
    const float chunkMinY = static_cast<float>(chunk.chunkY() * odai::world::Chunk::kSizeY);
    const float chunkMinZ = static_cast<float>(chunk.chunkZ() * odai::world::Chunk::kSizeZ);
    const float chunkMaxX = chunkMinX + static_cast<float>(odai::world::Chunk::kSizeX);
    const float chunkMaxY = chunkMinY + static_cast<float>(odai::world::Chunk::kSizeY);
    const float chunkMaxZ = chunkMinZ + static_cast<float>(odai::world::Chunk::kSizeZ);

    std::array<odai::math::Vector3, 8> corners = {
        odai::math::Vector3{chunkMinX, chunkMinY, chunkMinZ},
        odai::math::Vector3{chunkMaxX, chunkMinY, chunkMinZ},
        odai::math::Vector3{chunkMinX, chunkMaxY, chunkMinZ},
        odai::math::Vector3{chunkMaxX, chunkMaxY, chunkMinZ},
        odai::math::Vector3{chunkMinX, chunkMinY, chunkMaxZ},
        odai::math::Vector3{chunkMaxX, chunkMinY, chunkMaxZ},
        odai::math::Vector3{chunkMinX, chunkMaxY, chunkMaxZ},
        odai::math::Vector3{chunkMaxX, chunkMaxY, chunkMaxZ},
    };

    float ndcMinX = std::numeric_limits<float>::max();
    float ndcMinY = std::numeric_limits<float>::max();
    float ndcMinZ = std::numeric_limits<float>::max();
    float ndcMaxX = std::numeric_limits<float>::lowest();
    float ndcMaxY = std::numeric_limits<float>::lowest();
    float ndcMaxZ = std::numeric_limits<float>::lowest();
    for (const odai::math::Vector3& corner : corners) {
        const odai::math::Vector3 clip = odai::math::transformPoint(lightViewProj, corner);
        ndcMinX = std::min(ndcMinX, clip.x);
        ndcMinY = std::min(ndcMinY, clip.y);
        ndcMinZ = std::min(ndcMinZ, clip.z);
        ndcMaxX = std::max(ndcMaxX, clip.x);
        ndcMaxY = std::max(ndcMaxY, clip.y);
        ndcMaxZ = std::max(ndcMaxZ, clip.z);
    }

    return !(ndcMaxX < (-1.0f - clipMargin) ||
             ndcMinX > (1.0f + clipMargin) ||
             ndcMaxY < (-1.0f - clipMargin) ||
             ndcMinY > (1.0f + clipMargin) ||
             ndcMaxZ < (0.0f - clipMargin) ||
             ndcMinZ > (1.0f + clipMargin));
}

float saturate(float value) {
    return odai::math::saturate(value);
}

float smoothStep(float edge0, float edge1, float x) {
    const float t = saturate((x - edge0) / std::max(edge1 - edge0, 1e-6f));
    return t * t * (3.0f - (2.0f * t));
}

odai::math::Vector3 lerpVec3(const odai::math::Vector3& a, const odai::math::Vector3& b, float t) {
    return (a * (1.0f - t)) + (b * t);
}

struct SkyTuningSample {
    float rayleighStrength = 1.0f;
    float mieStrength = 1.0f;
    float mieAnisotropy = 0.55f;
    float skyExposure = 1.0f;
    float sunDiskIntensity = 1150.0f;
    float sunHaloIntensity = 22.0f;
    float sunDiskSize = 2.0f;
    float sunHazeFalloff = 0.35f;
};

SkyTuningSample evaluateSunriseSkyTuning(float sunElevationDegrees) {
    const float h = saturate((sunElevationDegrees + 12.0f) / 32.0f);
    const float day = smoothStep(0.15f, 0.85f, h);

    SkyTuningSample sample{};
    sample.rayleighStrength = std::lerp(1.10f, 0.96f, day);
    sample.mieStrength = std::lerp(1.45f, 0.72f, day);
    sample.mieAnisotropy = std::lerp(0.82f, 0.76f, day);
    sample.skyExposure = std::lerp(1.12f, 0.96f, day);
    sample.sunDiskIntensity = std::lerp(1180.0f, 1040.0f, day);
    sample.sunHaloIntensity = std::lerp(24.0f, 18.0f, day);
    sample.sunDiskSize = std::lerp(2.45f, 1.70f, day);
    sample.sunHazeFalloff = std::lerp(0.46f, 0.30f, day);
    return sample;
}

SkyTuningSample blendSkyTuningSample(const SkyTuningSample& base, const SkyTuningSample& target, float blend) {
    const float t = std::clamp(blend, 0.0f, 1.0f);
    SkyTuningSample result{};
    result.rayleighStrength = std::lerp(base.rayleighStrength, target.rayleighStrength, t);
    result.mieStrength = std::lerp(base.mieStrength, target.mieStrength, t);
    result.mieAnisotropy = std::lerp(base.mieAnisotropy, target.mieAnisotropy, t);
    result.skyExposure = std::lerp(base.skyExposure, target.skyExposure, t);
    result.sunDiskIntensity = std::lerp(base.sunDiskIntensity, target.sunDiskIntensity, t);
    result.sunHaloIntensity = std::lerp(base.sunHaloIntensity, target.sunHaloIntensity, t);
    result.sunDiskSize = std::lerp(base.sunDiskSize, target.sunDiskSize, t);
    result.sunHazeFalloff = std::lerp(base.sunHazeFalloff, target.sunHazeFalloff, t);
    return result;
}

odai::math::Vector3 computeSunColor(
    const RendererBackend::SkyDebugSettings& settings,
    const odai::math::Vector3& sunDirection
) {
    const odai::math::Vector3 toSun = -odai::math::normalize(sunDirection);
    const float sunAltitude = std::clamp(toSun.y, -1.0f, 1.0f);
    const float dayFactor = smoothStep(0.05f, 0.65f, sunAltitude);
    const float twilightFactor = 1.0f - dayFactor;
    const float horizonBand = saturate(1.0f - (std::abs(sunAltitude) / 0.35f));
    const float warmAmount = twilightFactor * std::pow(horizonBand, 1.2f);
    const float pinkAmount = warmAmount * saturate((0.10f - sunAltitude) / 0.30f);

    const float rayleigh = std::max(settings.rayleighStrength, 0.01f);
    const float mie = std::max(settings.mieStrength, 0.01f);
    const odai::math::Vector3 dayTint{1.00f, 0.98f, 0.94f};
    const odai::math::Vector3 goldenTint{1.18f, 0.72f, 0.34f};
    const odai::math::Vector3 pinkTint{1.08f, 0.56f, 0.74f};

    odai::math::Vector3 sunTint = lerpVec3(dayTint, goldenTint, warmAmount);
    sunTint = lerpVec3(sunTint, pinkTint, pinkAmount * 0.45f);

    const float scatteringScale = (rayleigh * 0.55f) + (mie * 0.80f);
    const float twilightBoost = 0.85f + (warmAmount * 0.45f);
    return sunTint * (scatteringScale * twilightBoost);
}

// includeSunDirect=false drops the sun disk and its glow, leaving the SKY only.
// See computeIrradianceShCoefficients for why that distinction exists.
odai::math::Vector3 proceduralSkyRadiance(
    const odai::math::Vector3& direction,
    const odai::math::Vector3& sunDirection,
    const odai::math::Vector3& sunColor,
    const RendererBackend::SkyDebugSettings& settings,
    bool includeSunDirect = true
) {
    const odai::math::Vector3 dir = odai::math::normalize(direction);
    const odai::math::Vector3 toSun = -odai::math::normalize(sunDirection);
    const float horizonT = saturate((dir.y * 0.5f) + 0.5f);
    const float skyT = std::pow(horizonT, 0.35f);
    const float sunAltitude = std::clamp(toSun.y, -1.0f, 1.0f);
    const float dayFactor = smoothStep(0.05f, 0.65f, sunAltitude);
    const float twilightFactor = 1.0f - dayFactor;
    const float horizonBand = saturate(1.0f - (std::abs(sunAltitude) / 0.35f));
    const float warmAmount = twilightFactor * std::pow(horizonBand, 1.2f);
    const float pinkAmount = warmAmount * saturate((0.10f - sunAltitude) / 0.30f);

    const float rayleigh = std::max(settings.rayleighStrength, 0.01f);
    const float mie = std::max(settings.mieStrength, 0.01f);

    const odai::math::Vector3 dayHorizonRayleigh{0.54f, 0.70f, 1.00f};
    const odai::math::Vector3 dayHorizonMie{1.00f, 0.74f, 0.42f};
    const odai::math::Vector3 sunsetHorizonRayleigh{0.74f, 0.44f, 0.52f};
    const odai::math::Vector3 sunsetHorizonMie{1.18f, 0.54f, 0.30f};
    const odai::math::Vector3 pinkHorizonRayleigh{0.70f, 0.36f, 0.68f};
    const odai::math::Vector3 pinkHorizonMie{1.08f, 0.46f, 0.72f};

    const float zenithWarm = twilightFactor * 0.58f;
    const odai::math::Vector3 dayZenithRayleigh{0.06f, 0.24f, 0.54f};
    const odai::math::Vector3 dayZenithMie{0.22f, 0.20f, 0.15f};
    const odai::math::Vector3 duskZenithRayleigh{0.16f, 0.12f, 0.30f};
    const odai::math::Vector3 duskZenithMie{0.30f, 0.18f, 0.24f};

    odai::math::Vector3 horizonRayleigh = lerpVec3(dayHorizonRayleigh, sunsetHorizonRayleigh, warmAmount);
    odai::math::Vector3 horizonMie = lerpVec3(dayHorizonMie, sunsetHorizonMie, warmAmount);
    horizonRayleigh = lerpVec3(horizonRayleigh, pinkHorizonRayleigh, pinkAmount * 0.70f);
    horizonMie = lerpVec3(horizonMie, pinkHorizonMie, pinkAmount * 0.85f);

    const odai::math::Vector3 zenithRayleigh = lerpVec3(dayZenithRayleigh, duskZenithRayleigh, zenithWarm);
    const odai::math::Vector3 zenithMie = lerpVec3(dayZenithMie, duskZenithMie, zenithWarm);

    const odai::math::Vector3 horizonColor =
        (horizonRayleigh * rayleigh) +
        (horizonMie * (mie * 0.58f));
    const odai::math::Vector3 zenithColor =
        (zenithRayleigh * rayleigh) +
        (zenithMie * (mie * 0.25f));
    const odai::math::Vector3 baseSky = (horizonColor * (1.0f - skyT)) + (zenithColor * skyT);

    const float sunDot = std::max(odai::math::dot(dir, toSun), 0.0f);
    const float sunDisk = std::pow(sunDot, 1100.0f);
    const float sunGlow = std::pow(sunDot, 24.0f);
    const float g = std::clamp(settings.mieAnisotropy, 0.0f, 0.98f);
    constexpr float kInv4Pi = 0.0795774715f;
    const float phaseRayleigh = kInv4Pi * 0.75f * (1.0f + (sunDot * sunDot));
    const float phaseMie = kInv4Pi * (1.0f - (g * g)) /
        std::max(0.001f, std::pow(1.0f + (g * g) - (2.0f * g * sunDot), 1.5f));
    const float phaseBoost = (phaseRayleigh * rayleigh) + (phaseMie * mie * 1.4f);

    const float aboveHorizon = saturate(dir.y * 4.0f + 0.2f);
    const odai::math::Vector3 sunTerm = includeSunDirect
        ? (sunColor * (((sunDisk * 5.0f) + (sunGlow * 1.2f)) * (1.0f + phaseBoost)))
        : odai::math::Vector3{};
    const odai::math::Vector3 sky = (baseSky * aboveHorizon) + sunTerm;

    const odai::math::Vector3 groundColor{0.05f, 0.06f, 0.07f};
    const float belowHorizon = saturate(-dir.y);
    const odai::math::Vector3 horizonGroundColor = horizonColor * 0.32f;
    const float groundWeight = std::pow(belowHorizon, 0.55f);
    const odai::math::Vector3 ground = (horizonGroundColor * (1.0f - groundWeight)) + (groundColor * groundWeight);

    const float skyWeight = saturate((dir.y + 0.18f) / 0.20f);
    const float skyExposure = std::max(settings.skyExposure, 0.01f);
    return ((ground * (1.0f - skyWeight)) + (sky * skyWeight)) * skyExposure;
}

float shBasis(int index, const odai::math::Vector3& direction) {
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    switch (index) {
    case 0: return 0.282095f;
    case 1: return 0.488603f * y;
    case 2: return 0.488603f * z;
    case 3: return 0.488603f * x;
    case 4: return 1.092548f * x * y;
    case 5: return 1.092548f * y * z;
    case 6: return 0.315392f * ((3.0f * z * z) - 1.0f);
    case 7: return 1.092548f * x * z;
    case 8: return 0.546274f * ((x * x) - (y * y));
    default: return 0.0f;
    }
}

std::array<odai::math::Vector3, 9> computeIrradianceShCoefficients(
    const odai::math::Vector3& sunDirection,
    const odai::math::Vector3& sunColor,
    const RendererBackend::SkyDebugSettings& settings
) {
    constexpr uint32_t kThetaSamples = 16;
    constexpr uint32_t kPhiSamples = 32;
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTwoPi = 2.0f * kPi;

    std::array<odai::math::Vector3, 9> coefficients{};
    for (odai::math::Vector3& coefficient : coefficients) {
        coefficient = odai::math::Vector3{};
    }

    float weightSum = 0.0f;
    for (uint32_t thetaIdx = 0; thetaIdx < kThetaSamples; ++thetaIdx) {
        const float v = (static_cast<float>(thetaIdx) + 0.5f) / static_cast<float>(kThetaSamples);
        const float theta = v * kPi;
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);

        for (uint32_t phiIdx = 0; phiIdx < kPhiSamples; ++phiIdx) {
            const float u = (static_cast<float>(phiIdx) + 0.5f) / static_cast<float>(kPhiSamples);
            const float phi = u * kTwoPi;
            const odai::math::Vector3 dir{
                std::cos(phi) * sinTheta,
                cosTheta,
                std::sin(phi) * sinTheta
            };

            // SKY ONLY -- THE SUN IS DELIBERATELY EXCLUDED, and this is what
            // makes shadows visible at all.
            //
            // These coefficients become `ambient` in imported_static.frag,
            // which is UNSHADOWED by construction. Integrating a sky that still
            // contains the sun disk and its pow(sunDot, 24) glow therefore
            // delivers most of the sun's energy a second time, omnidirectionally
            // and with no occlusion -- so putting a surface in shadow removed
            // only the small remainder. Measured on Goodsprings at an hour with
            // a low sun: disabling the shadow pass entirely moved the frame by
            // 0.61/255, i.e. the shadows were already doing nothing.
            //
            // The sun's contribution is the `direct` term, which IS shadowed.
            // Counting it here as well was double-counting it, and the copy
            // that won was the one no occluder could touch.
            const odai::math::Vector3 radiance = proceduralSkyRadiance(
                dir, sunDirection, sunColor, settings, /*includeSunDirect=*/false);
            const float sampleWeight = sinTheta;
            for (int basisIndex = 0; basisIndex < 9; ++basisIndex) {
                const float basisValue = shBasis(basisIndex, dir);
                coefficients[basisIndex] += radiance * (basisValue * sampleWeight);
            }
            weightSum += sampleWeight;
        }
    }

    if (weightSum <= 0.0f) {
        return coefficients;
    }

    const float normalization = (4.0f * kPi) / weightSum;
    for (odai::math::Vector3& coefficient : coefficients) {
        coefficient *= normalization;
    }

    // Convolve SH radiance with Lambert kernel for diffuse irradiance.
    coefficients[0] *= kPi;
    coefficients[1] *= (2.0f * kPi / 3.0f);
    coefficients[2] *= (2.0f * kPi / 3.0f);
    coefficients[3] *= (2.0f * kPi / 3.0f);
    coefficients[4] *= (kPi * 0.25f);
    coefficients[5] *= (kPi * 0.25f);
    coefficients[6] *= (kPi * 0.25f);
    coefficients[7] *= (kPi * 0.25f);
    coefficients[8] *= (kPi * 0.25f);

    return coefficients;
}

uint32_t findMemoryTypeIndex(
    VkPhysicalDevice physicalDevice,
    uint32_t typeBits,
    VkMemoryPropertyFlags requiredProperties
) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool typeMatches = (typeBits & (1u << i)) != 0;
        const bool propertiesMatch =
            (memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties;
        if (typeMatches && propertiesMatch) {
            return i;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

void transitionImageLayout(
    VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask,
    VkImageAspectFlags aspectMask,
    uint32_t baseArrayLayer = 0,
    uint32_t layerCount = 1,
    uint32_t baseMipLevel = 0,
    uint32_t levelCount = 1
) {
    VkImageMemoryBarrier2 imageBarrier{};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imageBarrier.srcStageMask = srcStageMask;
    imageBarrier.srcAccessMask = srcAccessMask;
    imageBarrier.dstStageMask = dstStageMask;
    imageBarrier.dstAccessMask = dstAccessMask;
    imageBarrier.oldLayout = oldLayout;
    imageBarrier.newLayout = newLayout;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image;
    imageBarrier.subresourceRange.aspectMask = aspectMask;
    imageBarrier.subresourceRange.baseMipLevel = baseMipLevel;
    imageBarrier.subresourceRange.levelCount = levelCount;
    imageBarrier.subresourceRange.baseArrayLayer = baseArrayLayer;
    imageBarrier.subresourceRange.layerCount = layerCount;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

void transitionBufferAccess(
    VkCommandBuffer commandBuffer,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask
) {
    VkBufferMemoryBarrier2 bufferBarrier{};
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    bufferBarrier.srcStageMask = srcStageMask;
    bufferBarrier.srcAccessMask = srcAccessMask;
    bufferBarrier.dstStageMask = dstStageMask;
    bufferBarrier.dstAccessMask = dstAccessMask;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &bufferBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

// synchronization2 one-shot submit for the blocking init/upload paths: a single
// command buffer, no wait/signal semaphores. Keeps these off the deprecated
// VkSubmitInfo path so the whole backend speaks submit2.
VkResult submitCommandBufferOneShot(VkQueue queue, VkCommandBuffer commandBuffer, VkFence fence) {
    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = commandBuffer;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    return vkQueueSubmit2(queue, 1, &submitInfo, fence);
}

VkFormat findSupportedDepthFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 3> kDepthCandidates = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT
    };

    for (VkFormat format : kDepthCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedShadowDepthFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kShadowDepthCandidates = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D16_UNORM
    };

    for (VkFormat format : kShadowDepthCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedHdrColorFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kHdrCandidates = {
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_B10G11R11_UFLOAT_PACK32
    };

    for (VkFormat format : kHdrCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedNormalDepthFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kNormalDepthCandidates = {
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R32G32B32A32_SFLOAT
    };

    for (VkFormat format : kNormalDepthCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedSsaoFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kSsaoCandidates = {
        VK_FORMAT_R16_SFLOAT,
        VK_FORMAT_R8_UNORM
    };

    for (VkFormat format : kSsaoCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedVoxelGiFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 2> kVoxelGiCandidates = {
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R32G32B32A32_SFLOAT
    };

    for (VkFormat format : kVoxelGiCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }

    return VK_FORMAT_UNDEFINED;
}

VkFormat findSupportedVoxelGiOccupancyFormat(VkPhysicalDevice physicalDevice) {
    constexpr std::array<VkFormat, 1> kOccupancyCandidates = {
        VK_FORMAT_R8G8B8A8_UNORM
    };

    for (VkFormat format : kOccupancyCandidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        const VkFormatFeatureFlags requiredFeatures =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures) {
            return format;
        }
    }

    return VK_FORMAT_UNDEFINED;
}

std::array<std::uint8_t, 3> voxelTypeAlbedoRgb(odai::world::VoxelType type) {
    switch (type) {
    case odai::world::VoxelType::Stone:
        return {150u, 154u, 160u};
    case odai::world::VoxelType::Dirt:
        return {122u, 93u, 58u};
    case odai::world::VoxelType::Grass:
        return {80u, 142u, 63u};
    case odai::world::VoxelType::Wood:
        return {141u, 106u, 64u};
    case odai::world::VoxelType::Leaves:
        return {92u, 148u, 78u};
    case odai::world::VoxelType::SolidRed:
        return {255u, 71u, 56u};
    case odai::world::VoxelType::Empty:
    default:
        return {0u, 0u, 0u};
    }
}

std::array<std::uint8_t, 3> voxelGiAlbedoRgb(
    const odai::world::Voxel& voxel,
    const std::array<std::uint32_t, 16>& palette
) {
    if (voxel.baseColorIndex <= 0x0Fu) {
        const std::uint32_t rgba = palette[voxel.baseColorIndex & 0x0Fu];
        return {
            static_cast<std::uint8_t>(rgba & 0xFFu),
            static_cast<std::uint8_t>((rgba >> 8u) & 0xFFu),
            static_cast<std::uint8_t>((rgba >> 16u) & 0xFFu)
        };
    }
    return voxelTypeAlbedoRgb(voxel.type);
}

struct QueueFamilyChoice {
    std::optional<uint32_t> graphicsAndPresent;
    std::optional<uint32_t> transfer;
    uint32_t graphicsQueueIndex = 0;
    uint32_t transferQueueIndex = 0;

    [[nodiscard]] bool valid() const {
        return graphicsAndPresent.has_value() && transfer.has_value();
    }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

const char* vkResultName(VkResult result) {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    default: return "VK_RESULT_UNKNOWN";
    }
}

void logVkFailure(const char* context, VkResult result) {
    VOX_LOGE("render") << context << " failed: "
                       << vkResultName(result) << " (" << static_cast<int>(result) << ")";
}

int floorDiv(int value, int divisor) {
    const int q = value / divisor;
    const int r = value % divisor;
    if (r != 0 && ((r < 0) != (divisor < 0))) {
        return q - 1;
    }
    return q;
}

template <typename VkHandleT>
uint64_t vkHandleToUint64(VkHandleT handle) {
    if constexpr (std::is_pointer_v<VkHandleT>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

bool isLayerAvailable(const char* layerName) {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

    for (const VkLayerProperties& layer : layers) {
        if (std::strcmp(layer.layerName, layerName) == 0) {
            return true;
        }
    }
    return false;
}

bool isInstanceExtensionAvailable(const char* extensionName) {
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    for (const VkExtensionProperties& extension : extensions) {
        if (std::strcmp(extension.extensionName, extensionName) == 0) {
            return true;
        }
    }
    return false;
}

void appendInstanceExtensionIfMissing(std::vector<const char*>& extensions, const char* extensionName) {
    const auto found = std::find_if(
        extensions.begin(),
        extensions.end(),
        [extensionName](const char* existing) {
            return std::strcmp(existing, extensionName) == 0;
        }
    );
    if (found == extensions.end()) {
        extensions.push_back(extensionName);
    }
}

QueueFamilyChoice findQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    QueueFamilyChoice choice;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());

    std::optional<uint32_t> dedicatedTransferFamily;
    std::optional<uint32_t> anyTransferFamily;

    for (uint32_t familyIndex = 0; familyIndex < familyCount; ++familyIndex) {
        const VkQueueFlags queueFlags = families[familyIndex].queueFlags;
        const bool hasGraphics = (queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool hasTransfer = (queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;

        if (hasGraphics && !choice.graphicsAndPresent.has_value()) {
            VkBool32 hasPresent = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex, surface, &hasPresent);
            if (hasPresent == VK_TRUE) {
                choice.graphicsAndPresent = familyIndex;
            }
        }

        if (hasTransfer) {
            if (!anyTransferFamily.has_value()) {
                anyTransferFamily = familyIndex;
            }
            if (!dedicatedTransferFamily.has_value() && !hasGraphics) {
                dedicatedTransferFamily = familyIndex;
            }
        }
    }

    if (!choice.graphicsAndPresent.has_value()) {
        return choice;
    }

    if (dedicatedTransferFamily.has_value()) {
        choice.transfer = dedicatedTransferFamily.value();
    } else if (anyTransferFamily.has_value()) {
        choice.transfer = anyTransferFamily.value();
    } else {
        choice.transfer = choice.graphicsAndPresent.value();
    }

    if (choice.transfer.value() == choice.graphicsAndPresent.value()) {
        const uint32_t queueCount = families[choice.graphicsAndPresent.value()].queueCount;
        if (queueCount > 1) {
            choice.transferQueueIndex = 1;
        }
    }

    return choice;
}

bool hasRequiredDeviceExtensions(VkPhysicalDevice physicalDevice) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data());

    for (const char* required : kDeviceExtensions) {
        bool found = false;
        for (const VkExtensionProperties& available : extensions) {
            if (std::strcmp(required, available.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    return true;
}

SwapchainSupport querySwapchainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    SwapchainSupport support;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &support.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    support.formats.resize(formatCount);
    if (formatCount > 0) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, support.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    support.presentModes.resize(presentModeCount);
    if (presentModeCount > 0) {
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, support.presentModes.data());
    }

    return support;
}

VkSurfaceFormatKHR chooseSwapchainFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return formats.front();
}

inline const char* presentModeName(VkPresentModeKHR mode) {
    switch (mode) {
        case VK_PRESENT_MODE_MAILBOX_KHR:      return "MAILBOX";
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED";
        case VK_PRESENT_MODE_IMMEDIATE_KHR:    return "IMMEDIATE";
        case VK_PRESENT_MODE_FIFO_KHR:         return "FIFO";
        default:                               return "UNKNOWN";
    }
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes) {
    const auto has = [&](VkPresentModeKHR mode) {
        return std::find(presentModes.begin(), presentModes.end(), mode) != presentModes.end();
    };
    // Escape hatch for A/B-testing latency vs. pacing (e.g. "is FIFO's queuing
    // delay the reason a UI-rendered cursor feels laggier than the OS cursor")
    // without changing the shipped default below. Unset or unsupported/invalid
    // values fall through to the normal preference order untouched.
    if (const char* override = std::getenv("ODAI_PRESENT_MODE")) {
        VkPresentModeKHR requested = VK_PRESENT_MODE_MAX_ENUM_KHR;
        if (std::strcmp(override, "immediate") == 0) requested = VK_PRESENT_MODE_IMMEDIATE_KHR;
        else if (std::strcmp(override, "mailbox") == 0) requested = VK_PRESENT_MODE_MAILBOX_KHR;
        else if (std::strcmp(override, "fifo") == 0) requested = VK_PRESENT_MODE_FIFO_KHR;
        else if (std::strcmp(override, "fifo_relaxed") == 0) requested = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        if (requested != VK_PRESENT_MODE_MAX_ENUM_KHR && has(requested)) {
            return requested;
        }
    }
    // Prefer clean vsync (dual buffering) for steady pacing and no tearing:
    //   FIFO_RELAXED  - vsynced; if a frame is late it presents immediately rather
    //                   than waiting a full extra vblank (best of both worlds).
    //   FIFO          - strict vsync; guaranteed by spec.
    //   MAILBOX       - vsynced but requires a third image and wastes GPU on
    //                   frames that get discarded.
    //   IMMEDIATE     - no vsync fallback (tears; uncapped frame rate).
    if (has(VK_PRESENT_MODE_FIFO_RELAXED_KHR)) {
        return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    }
    if (has(VK_PRESENT_MODE_FIFO_KHR)) {
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    if (has(VK_PRESENT_MODE_MAILBOX_KHR)) {
        return VK_PRESENT_MODE_MAILBOX_KHR;
    }
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
}

VkExtent2D chooseExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D extent{};
    extent.width = std::clamp(
        static_cast<uint32_t>(std::max(width, 1)),
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width
    );
    extent.height = std::clamp(
        static_cast<uint32_t>(std::max(height, 1)),
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height
    );
    return extent;
}

std::optional<std::vector<std::uint8_t>> readBinaryFile(const char* filePath) {
    if (filePath == nullptr) {
        return std::nullopt;
    }

    const std::filesystem::path path(filePath);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::nullopt;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return std::nullopt;
    }
    file.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> data(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        return std::nullopt;
    }
    return data;
}

bool createShaderModuleFromFile(
    VkDevice device,
    const char* filePath,
    const char* debugName,
    VkShaderModule& outShaderModule
) {
    outShaderModule = VK_NULL_HANDLE;

    const std::optional<std::vector<std::uint8_t>> shaderFileData = readBinaryFile(filePath);
    if (!shaderFileData.has_value()) {
        VOX_LOGE("render") << "missing shader file for " << debugName << ": " << (filePath != nullptr ? filePath : "<null>") << "\n";
        return false;
    }
    if ((shaderFileData->size() % sizeof(std::uint32_t)) != 0) {
        VOX_LOGE("render") << "invalid SPIR-V byte size for " << debugName << ": " << filePath << "\n";
        return false;
    }
    const std::uint32_t* code = reinterpret_cast<const std::uint32_t*>(shaderFileData->data());
    const size_t codeSize = shaderFileData->size();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = code;
    const VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &outShaderModule);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateShaderModule(fileOrFallback)", result);
        return false;
    }
    return true;
}

void destroyShaderModules(VkDevice device, std::span<const VkShaderModule> shaderModules) {
    for (const VkShaderModule shaderModule : shaderModules) {
        if (shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }
}

} // namespace

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
