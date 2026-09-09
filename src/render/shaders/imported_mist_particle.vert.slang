import camera_uniform;
struct MistPush { float4 positionRadius; float4 color; float4 material; };
[[vk::push_constant]] ConstantBuffer<MistPush> mist;
struct Output { float4 position:SV_Position; float2 uv:TEXCOORD0; float depth:TEXCOORD1; };
Output main(uint id:SV_VertexID) {
    const float2 corners[6]={float2(-1,-1),float2(1,-1),float2(1,1),float2(-1,-1),float2(1,1),float2(-1,1)};
    float2 p=corners[id];float c=cos(mist.material.y),s=sin(mist.material.y);
    float4 view=mul(camera.view,float4(mist.positionRadius.xyz,1));
    view.xy+=float2(c*p.x-s*p.y,s*p.x+c*p.y)*mist.positionRadius.w;
    Output o;o.position=mul(camera.proj,view);o.uv=p*.5+.5;o.depth=-view.z;return o;
}
