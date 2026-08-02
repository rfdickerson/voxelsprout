// Compile for Vulkan SPIR-V with column-major matrices:
// slangc -target spirv -entry main -stage compute -matrix-layout-column-major src/render/shaders/skinning.comp.slang -o src/render/shaders/skinning.comp.slang.spv
//
// GPU skeletal skinning pre-pass (Dragon Age: Origins touchstone, see
// docs/ROADMAP.md's Party RPG / Narrative section). Resolves rest-pose
// vertices against this frame's bone matrices into a buffer laid out exactly
// like ImportedMeshVertex (renderer_backend.h), so the existing
// imported_static.vert/frag.slang and all of their pipeline variants consume
// the result as a plain vertex buffer with zero shader changes.
//
// VERIFY BEFORE SHIPPING: boneMatrices must be uploaded in the same
// row-major/column-major convention the camera MVP matrix already uses for
// this -matrix-layout-column-major compile flag (odai::math::Matrix4 is
// row-major on the CPU side per math/math.h) -- transpose on upload if the
// camera-matrix upload path does, mirror it exactly if it doesn't.

struct SkinnedVertexIn
{
    float3 position;
    float3 normal;
    float3 color;
    float2 uv;
    uint textureIndex;
    uint flags;
    uint4 boneIndices;
    float4 boneWeights;
};

struct SkinnedVertexOut
{
    float3 position;
    float3 normal;
    float3 color;
    float2 uv;
    uint textureIndex;
    uint flags;
};

struct SkinningPushConstants
{
    uint vertexCount;
    uint boneCount;
    float _pad0;
    float _pad1;
};

[[vk::push_constant]]
ConstantBuffer<SkinningPushConstants> skinningPc;

[[vk::binding(0, 0)]]
StructuredBuffer<SkinnedVertexIn> restPoseVertices;

[[vk::binding(1, 0)]]
StructuredBuffer<float4x4> boneMatrices;

[[vk::binding(2, 0)]]
RWStructuredBuffer<SkinnedVertexOut> skinnedVertices;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint vertexIndex = dispatchThreadId.x;
    if (vertexIndex >= skinningPc.vertexCount)
    {
        return;
    }

    const SkinnedVertexIn v = restPoseVertices[vertexIndex];
    const uint maxBoneIndex = max(skinningPc.boneCount, 1u) - 1u;

    float4 skinnedPosition = float4(0.0, 0.0, 0.0, 0.0);
    float4 skinnedNormal = float4(0.0, 0.0, 0.0, 0.0);
    [unroll]
    for (uint i = 0; i < 4u; ++i)
    {
        const float weight = v.boneWeights[i];
        if (weight <= 0.0)
        {
            continue;
        }
        const uint boneIndex = min(v.boneIndices[i], maxBoneIndex);
        const float4x4 boneMatrix = boneMatrices[boneIndex];
        skinnedPosition += weight * mul(boneMatrix, float4(v.position, 1.0));
        skinnedNormal += weight * mul(boneMatrix, float4(v.normal, 0.0));
    }

    SkinnedVertexOut result;
    result.position = skinnedPosition.xyz;
    result.normal = normalize(skinnedNormal.xyz);
    result.color = v.color;
    result.uv = v.uv;
    result.textureIndex = v.textureIndex;
    result.flags = v.flags;
    skinnedVertices[vertexIndex] = result;
}
