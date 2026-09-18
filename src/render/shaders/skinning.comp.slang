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
// skinning_resources.cc and renderer_backend.h. Input stride must be 148,
// output stride 76.
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
    uint normalTextureIndex;
    float basisXX, basisXY, basisXZ, basisYX, basisYY, basisYZ, basisZX, basisZY, basisZZ;
    uint skinSoftTexture, skinSpecularTexture;
    float skinSpecularStrength, skinGlossiness, skinSoftRolloff;
    uint skinSpecularColor;
};

// 76 bytes, matching ImportedMeshVertex exactly -- see renderer_backend.h.
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
    uint packedTerrainNormal01;
    uint packedTerrainNormal23;
    uint packedTerrainNormal45;
    uint lightingMaterialIndex;
    uint terrainSurface01;
    uint terrainSurface23;
    uint terrainSurface45;
};

struct SkinningPushConstants
{
    uint vertexCount;
    uint boneCount;
    uint morphTargetCount;
    uint _pad0;
};

[[vk::push_constant]]
ConstantBuffer<SkinningPushConstants> skinningPc;

[[vk::binding(0, 0)]]
StructuredBuffer<SkinnedVertexIn> restPoseVertices;

[[vk::binding(1, 0)]]
StructuredBuffer<float4x4> boneMatrices;

[[vk::binding(2, 0)]]
RWStructuredBuffer<SkinnedVertexOut> skinnedVertices;

// Sparse morphs are stored in vertex-major CSR form, so one invocation reads
// only the deltas affecting its vertex. These scalar members give MorphDelta a
// 16-byte stride matching ImportedSkinnedMeshTemplate::MorphDelta.
struct MorphDelta
{
    uint targetIndex;
    float deltaX, deltaY, deltaZ;
};

[[vk::binding(3, 0)]] StructuredBuffer<uint> morphVertexOffsets;
[[vk::binding(4, 0)]] StructuredBuffer<MorphDelta> morphDeltas;
[[vk::binding(5, 0)]] StructuredBuffer<float> morphWeights;

// Linear blend skinning combines several rigid bone transforms into a matrix
// that can contain scale and shear. Positions use that blended matrix, while
// normals must use its inverse transpose. The three arguments are the columns
// of the blended 3x3 position transform; the common 1/determinant magnitude
// cancels when the result is normalized, but its sign must be retained.
float3 transformSkinnedNormal(
    float3 deformX, float3 deformY, float3 deformZ,
    float3 normal, float3 fallback)
{
    const float3 cofactorX = cross(deformY, deformZ);
    const float3 cofactorY = cross(deformZ, deformX);
    const float3 cofactorZ = cross(deformX, deformY);
    const float determinant = dot(deformX, cofactorX);
    float3 transformed = cofactorX * normal.x +
        cofactorY * normal.y + cofactorZ * normal.z;
    if (determinant < 0.0)
        transformed = -transformed;
    return abs(determinant) > 1e-8 && dot(transformed, transformed) > 1e-12
        ? normalize(transformed) : fallback;
}

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
    float3 restPosition = float3(v.positionX, v.positionY, v.positionZ);
    if (skinningPc.morphTargetCount != 0u)
    {
        const uint begin = morphVertexOffsets[vertexIndex];
        const uint end = morphVertexOffsets[vertexIndex + 1u];
        for (uint deltaIndex = begin; deltaIndex < end; ++deltaIndex)
        {
            const MorphDelta delta = morphDeltas[deltaIndex];
            const uint target = min(delta.targetIndex, skinningPc.morphTargetCount - 1u);
            restPosition += morphWeights[target] *
                float3(delta.deltaX, delta.deltaY, delta.deltaZ);
        }
    }
    const float3 restNormal = float3(v.normalX, v.normalY, v.normalZ);
    const uint4 boneIndices = uint4(v.boneIndex0, v.boneIndex1, v.boneIndex2, v.boneIndex3);
    const float4 boneWeights = float4(v.boneWeight0, v.boneWeight1, v.boneWeight2, v.boneWeight3);

    float4 skinnedPosition = float4(0.0, 0.0, 0.0, 0.0);
    float4 skinnedNormal = float4(0.0, 0.0, 0.0, 0.0);
    float3 deformX = 0.0, deformY = 0.0, deformZ = 0.0;
    float weightTotal = 0.0;
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
        deformX += weight * mul(boneMatrix, float4(1.0, 0.0, 0.0, 0.0)).xyz;
        deformY += weight * mul(boneMatrix, float4(0.0, 1.0, 0.0, 0.0)).xyz;
        deformZ += weight * mul(boneMatrix, float4(0.0, 0.0, 1.0, 0.0)).xyz;
        weightTotal += weight;
    }

    // A vertex with no influences at all would otherwise collapse to the
    // origin. Falling back to the rest pose keeps it where the artist put it,
    // which is both more useful and much easier to recognize than a spike.
    // World translation is in the palette. Even small weight-sum errors
    // otherwise displace vertices by worldPosition * (weightTotal - 1),
    // corrupting geometry and disagreeing with the velocity pass.
    const float3 finalPosition = (weightTotal > 1e-5) ? skinnedPosition.xyz / weightTotal : restPosition;
    const float3 weightedNormalFallback =
        dot(skinnedNormal.xyz, skinnedNormal.xyz) > 1e-12
            ? normalize(skinnedNormal.xyz) : restNormal;
    const float3 finalNormal = weightTotal > 1e-5
        ? transformSkinnedNormal(deformX, deformY, deformZ,
            restNormal, weightedNormalFallback)
        : restNormal;
    const float3 restBasisX = float3(v.basisXX, v.basisXY, v.basisXZ);
    const float3 restBasisY = float3(v.basisYX, v.basisYY, v.basisYZ);
    const float3 restBasisZ = float3(v.basisZX, v.basisZY, v.basisZZ);
    const float3 basisX = transformSkinnedNormal(
        deformX, deformY, deformZ, restBasisX, restBasisX);
    const float3 basisY = transformSkinnedNormal(
        deformX, deformY, deformZ, restBasisY, restBasisY);
    const float3 basisZ = transformSkinnedNormal(
        deformX, deformY, deformZ, restBasisZ, restBasisZ);

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
    result.packedLayerTexture01 = 0xffff0000u | min(v.normalTextureIndex, 0xffffu);
    result.packedLayerTexture23 = 0xffffffffu;
    result.packedTerrainNormal01 = 0xffffffffu;
    result.packedTerrainNormal23 = 0xffffffffu;
    result.packedTerrainNormal45 = 0xffffffffu;
    result.lightingMaterialIndex = 0xffffffffu;
    result.terrainSurface01 = encodeImportedVertexNormal(basisX);
    result.terrainSurface23 = encodeImportedVertexNormal(basisY);
    result.terrainSurface45 = encodeImportedVertexNormal(basisZ);
    result.layerWeights = 0u;
    if ((v.flags & ((1u << 14u) | 4u)) == (1u << 14u)) {
        result.packedLayerTexture01 = min(v.normalTextureIndex, 0xffffu) | (min(v.skinSoftTexture, 0xffffu) << 16);
        result.packedLayerTexture23 = 0xffff0000u | min(v.skinSpecularTexture, 0xffffu);
        result.packedTerrainNormal01 = asuint(v.skinSpecularStrength);
        result.packedTerrainNormal23 = asuint(v.skinGlossiness);
        result.packedTerrainNormal45 = asuint(v.skinSoftRolloff);
        result.layerWeights = v.skinSpecularColor;
    }
    skinnedVertices[vertexIndex] = result;
}
