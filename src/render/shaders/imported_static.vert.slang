import camera_uniform;

struct VertexInput
{
    [[vk::location(0)]] float3 inPosition : TEXCOORD0;
    [[vk::location(1)]] float3 inNormal : TEXCOORD1;
    [[vk::location(2)]] float3 inColor : TEXCOORD2;
    [[vk::location(3)]] float2 inUv : TEXCOORD3;
    [[vk::location(4)]] uint inTextureIndex : TEXCOORD4;
    [[vk::location(5)]] uint inFlags : TEXCOORD5;
    // Terrain layer blend (Fallout ATXT/VTXT). The three slots are bindless
    // texture indices, constant across a draw; the weights arrive as one packed
    // word and are unpacked here so they can interpolate across the triangle.
    [[vk::location(6)]] uint inLayerTexture0 : TEXCOORD6;
    [[vk::location(7)]] uint inLayerTexture1 : TEXCOORD7;
    [[vk::location(8)]] uint inLayerTexture2 : TEXCOORD8;
    [[vk::location(9)]] uint inLayerWeights : TEXCOORD9;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 outWorldPosition : TEXCOORD0;
    float3 outWorldNormal : TEXCOORD1;
    float3 outColor : TEXCOORD2;
    float2 outUv : TEXCOORD3;
    nointerpolation uint outTextureIndex : TEXCOORD4;
    nointerpolation uint outFlags : TEXCOORD5;
    nointerpolation uint3 outLayerTextures : TEXCOORD6;
    // Interpolated, unlike the slots above: the per-post opacity is exactly what
    // has to vary across the surface for a layer to fade in rather than appear
    // in hard per-vertex blocks.
    float3 outLayerWeights : TEXCOORD7;
};

VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.outWorldPosition = input.inPosition;
    output.outWorldNormal = normalize(input.inNormal);
    output.outColor = input.inColor;
    output.outUv = input.inUv;
    output.outTextureIndex = input.inTextureIndex;
    output.outFlags = input.inFlags;
    output.outLayerTextures = uint3(input.inLayerTexture0, input.inLayerTexture1, input.inLayerTexture2);
    // Byte n is layer n's weight over [0,1]; mirrors
    // packImportedSceneTerrainLayerWeights in src/import/imported_scene.h.
    output.outLayerWeights = float3(
        float((input.inLayerWeights >> 0) & 0xffu) / 255.0,
        float((input.inLayerWeights >> 8) & 0xffu) / 255.0,
        float((input.inLayerWeights >> 16) & 0xffu) / 255.0);
    output.position = mul(camera.mvp, float4(input.inPosition, 1.0));
    return output;
}
