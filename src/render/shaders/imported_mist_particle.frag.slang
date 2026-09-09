import camera_uniform;
struct MistPush { float4 positionRadius; float4 color; float4 material; };
[[vk::push_constant]] ConstantBuffer<MistPush> mist;
[[vk::binding(0,1)]] Sampler2D<float4> textures[];
[[vk::binding(6,0)]] Sampler2D<float4> normalDepth;
struct Input {float2 uv:TEXCOORD0;float depth:TEXCOORD1;float4 position:SV_Position;};
float4 main(Input i):SV_Target0 {
    float4 tex=textures[NonUniformResourceIndex(uint(mist.material.x))].Sample(i.uv);
    float soft=1;
    if(mist.material.z>0) {
        float2 uv=i.position.xy/max(camera.contactShadowConfig.xy,float2(1));
        float depth=normalDepth.SampleLevel(uv,0).w;
        if(depth>0)soft=saturate((depth-i.depth)/mist.material.z);
    }
    return float4(tex.rgb*mist.color.rgb,saturate(tex.a*mist.color.a)*soft);
}
