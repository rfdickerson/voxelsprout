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
// boneMatrices are uploaded transposed (skinning_resources.cc), matching the
// camera-MVP path's convention for this -matrix-layout-column-major compile.
//
// STRUCT LAYOUT -- read this before touching either struct below.
//
// Both are declared with SCALAR members rather than float3/float2/uint4, and
// that is load-bearing, not a style choice. A StructuredBuffer here lays out
// under std430, where a float3 is 16-byte ALIGNED. Written the natural way:
//
//   struct { float3 position; float3 normal; float3 color; float2 uv;
//            uint textureIndex; uint flags; uint4 boneIndices;
//            float4 boneWeights; };
//
// SPIR-V puts normal at offset 16 (not 12), color at 32 (not 24), and gives the
// struct a stride of 96 -- while the CPU side writes a tightly packed 84 bytes.
// Every field past position is then read from the wrong offset, and every
// vertex after the first from the wrong base. The result on screen is not a
// subtly wrong character, it is a cloud of shards.
//
// Scalars are 4-byte aligned in std430, so a scalar-by-scalar declaration packs
// exactly like the C struct. Confirmed by disassembly, which is the only way to
// confirm it -- run:
//
//   spirv-dis src/render/shaders/skinning.comp.slang.spv | grep OpMemberDecorate
//
// and check the offsets against GpuSkinnedVertexIn / ImportedMeshVertex in
// skinning_resources.cc and renderer_backend.h. Input stride must be 84,
// output stride 48.
//
// The output struct additionally carries the terrain-layer fields. It is bound
// as a plain vertex buffer for the main pass, which reads it as
// ImportedMeshVertex -- so it has to match that struct to its last byte, not
// just up to the fields skinning happens to care about.

import imported_vertex_pack;

struct SkinnedVertexIn
{
    float positionX, positionY, positionZ;
    float normalX, normalY, normalZ;
    float colorR, colorG, colorB;
    float u, v;
    uint textureIndex;
    uint flags;
    uint boneIndex0, boneIndex1, boneIndex2, boneIndex3;
    float boneWeight0, boneWeight1, boneWeight2, boneWeight3;
};

// 48 bytes, matching ImportedMeshVertex exactly -- see renderer_backend.h.
//
// The narrowed fields are declared as uint and packed by hand rather than as
// int16_t/uint8_t: this is a StructuredBuffer, and sub-32-bit members in one
// would require VK_KHR_8bit_storage / 16bit_storage, which this device path
// does not enable. The graphics pipelines get the same bytes unpacked for free
// by their vertex-attribute formats.
struct SkinnedVertexOut
{
    float positionX, positionY, positionZ;
    uint packedNormal;
    uint packedColor;
    float u, v;
    uint textureIndex;
    uint flags;
    // ImportedMeshVertex's terrain-layer tail. Skinned geometry never uses
    // terrain layers, but the fields must exist and be initialized: the main
    // pass reads them off this same buffer, and leaving them out shortens the
    // stride so every vertex after the first is fetched from the wrong offset.
    uint packedLayerTexture01;
    uint packedLayerTexture23;
    uint layerWeights;
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

    // Reassembled into vectors here rather than declared as vectors above --
    // see the struct-layout note at the top of this file for why the storage
    // has to stay scalar.
    const float3 restPosition = float3(v.positionX, v.positionY, v.positionZ);
    const float3 restNormal = float3(v.normalX, v.normalY, v.normalZ);
    const uint4 boneIndices = uint4(v.boneIndex0, v.boneIndex1, v.boneIndex2, v.boneIndex3);
    const float4 boneWeights = float4(v.boneWeight0, v.boneWeight1, v.boneWeight2, v.boneWeight3);

    float4 skinnedPosition = float4(0.0, 0.0, 0.0, 0.0);
    float4 skinnedNormal = float4(0.0, 0.0, 0.0, 0.0);
    [unroll]
    for (uint i = 0; i < 4u; ++i)
    {
        const float weight = boneWeights[i];
        if (weight <= 0.0)
        {
            continue;
        }
        const uint boneIndex = min(boneIndices[i], maxBoneIndex);
        const float4x4 boneMatrix = boneMatrices[boneIndex];
        skinnedPosition += weight * mul(boneMatrix, float4(restPosition, 1.0));
        skinnedNormal += weight * mul(boneMatrix, float4(restNormal, 0.0));
    }

    // A vertex with no influences at all would otherwise collapse to the
    // origin. Falling back to the rest pose keeps it where the artist put it,
    // which is both more useful and much easier to recognize than a spike.
    const float weightTotal = boneWeights.x + boneWeights.y + boneWeights.z + boneWeights.w;
    const float3 finalPosition = (weightTotal > 0.0) ? skinnedPosition.xyz : restPosition;
    const float3 finalNormal =
        (weightTotal > 0.0 && dot(skinnedNormal.xyz, skinnedNormal.xyz) > 1e-12)
            ? normalize(skinnedNormal.xyz)
            : restNormal;

    SkinnedVertexOut result;
    result.positionX = finalPosition.x;
    result.positionY = finalPosition.y;
    result.positionZ = finalPosition.z;
    result.packedNormal = encodeImportedVertexNormal(finalNormal);
    result.packedColor = encodeImportedVertexColor(float3(v.colorR, v.colorG, v.colorB));
    result.u = v.u;
    result.v = v.v;
    result.textureIndex = v.textureIndex;
    result.flags = v.flags;
    // 0xffff per 16-bit slot, so both words are all-ones.
    result.packedLayerTexture01 = 0xffffffffu;
    result.packedLayerTexture23 = 0xffffffffu;
    result.layerWeights = 0u;
    skinnedVertices[vertexIndex] = result;
}
