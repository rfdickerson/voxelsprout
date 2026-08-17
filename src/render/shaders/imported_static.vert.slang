import camera_uniform;
import chunk_push_constants;
import imported_vertex_pack;

struct VertexInput
{
    [[vk::location(0)]] float3 inPosition : TEXCOORD0;
    // Octahedral snorm16x2; the fetch hands it over as float2 in [-1,1].
    [[vk::location(1)]] float2 inNormalOct : TEXCOORD1;
    // sRGB-encoded unorm8x3 plus a LINEAR alpha byte. The alpha is authored
    // vertex alpha (see ImportedSceneVertex::colorAlpha) and is not sRGB: it is
    // a coverage fraction, not a colour, so it passes through undecoded.
    [[vk::location(2)]] float4 inColorSrgb : TEXCOORD2;
    [[vk::location(3)]] float2 inUv : TEXCOORD3;
    [[vk::location(4)]] uint inTextureIndex : TEXCOORD4;
    [[vk::location(5)]] uint inFlags : TEXCOORD5;
    // Terrain layer blend (Fallout ATXT/VTXT). The four slots are bindless
    // texture indices, constant across a draw, and arrive as ONE
    // R16G16B16A16_UINT fetch; the weights arrive as one packed word and are
    // unpacked here so they can interpolate across the triangle.
    //
    // 0xffff, not 0xffffffff, is "no layer" now that the slots are 16 bits --
    // widened back below so the fragment shader keeps comparing against the
    // 32-bit invalid sentinel it always did.
    [[vk::location(6)]] uint4 inLayerTextures : TEXCOORD6;
    [[vk::location(7)]] uint inLayerWeights : TEXCOORD7;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 outWorldPosition : TEXCOORD0;
    float3 outWorldNormal : TEXCOORD1;
    // .rgb the decoded linear tint, .a the authored vertex alpha.
    float4 outColor : TEXCOORD2;
    float2 outUv : TEXCOORD3;
    nointerpolation uint outTextureIndex : TEXCOORD4;
    nointerpolation uint outFlags : TEXCOORD5;
    nointerpolation uint4 outLayerTextures : TEXCOORD6;
    // Interpolated, unlike the slots above: the per-post opacity is exactly what
    // has to vary across the surface for a layer to fade in rather than appear
    // in hard per-vertex blocks.
    float4 outLayerWeights : TEXCOORD7;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    float3 worldPosition = input.inPosition;
    float3 worldNormal = decodeImportedVertexNormal(input.inNormalOct);
    if (chunkPc.rigidAnimationParams.x > 0.5)
    {
        const float4 p = float4(worldPosition, 1.0);
        worldPosition = float3(
            dot(chunkPc.rigidAnimationTransform[0], p),
            dot(chunkPc.rigidAnimationTransform[1], p),
            dot(chunkPc.rigidAnimationTransform[2], p));
        worldNormal = normalize(float3(
            dot(chunkPc.rigidAnimationTransform[0].xyz, worldNormal),
            dot(chunkPc.rigidAnimationTransform[1].xyz, worldNormal),
            dot(chunkPc.rigidAnimationTransform[2].xyz, worldNormal)));
    }
    output.outWorldPosition = worldPosition;
    output.outWorldNormal = worldNormal;
    output.outColor = float4(decodeImportedVertexColor(input.inColorSrgb), input.inColorSrgb.a);
    output.outUv = input.inUv;
    output.outTextureIndex = input.inTextureIndex;
    output.outFlags = input.inFlags;
    output.outLayerTextures = select(
        input.inLayerTextures == uint4(0xffffu, 0xffffu, 0xffffu, 0xffffu),
        uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu),
        input.inLayerTextures);
    // Byte n is layer n's weight over [0,1]; mirrors
    // packImportedSceneTerrainLayerWeights in src/import/imported_scene.h.
    output.outLayerWeights = float4(
        float((input.inLayerWeights >> 0) & 0xffu) / 255.0,
        float((input.inLayerWeights >> 8) & 0xffu) / 255.0,
        float((input.inLayerWeights >> 16) & 0xffu) / 255.0,
        float((input.inLayerWeights >> 24) & 0xffu) / 255.0);
    float3 rasterPosition = worldPosition;
    // A camera mirrored across y=h is equivalent to reflecting every world
    // point across that plane and rendering it through the original camera.
    // Keeping outWorldPosition unreflected preserves authored lighting and
    // shadow lookups; only rasterization sees the mirrored point.
    if (chunkPc.cascadeData.x > 0.5)
    {
        rasterPosition.y = (2.0 * chunkPc.materialParams.w) - rasterPosition.y;
    }
    output.position = mul(camera.mvp, float4(rasterPosition, 1.0));
    return output;
}
