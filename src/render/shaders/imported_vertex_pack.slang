module imported_vertex_pack;

// GPU half of ImportedMeshVertex's packing. Mirrors the pack* helpers next to
// that struct in render/backend/vulkan/renderer_backend.h -- change both
// together, because nothing checks that they agree.
//
// Only skinning.comp.slang needs the ENCODE side: it writes ImportedMeshVertex
// through a StructuredBuffer, where sub-32-bit members would require
// VK_KHR_8bit_storage / 16bit_storage. Graphics pipelines get the decode for
// free from the vertex-attribute formats (R16G16_SNORM, R8G8B8A8_UNORM,
// R16G16B16A16_UINT), so the only decode here is the sRGB one, which no vertex
// format can express.

// Byte triple authored as sRGB -> linear. Inverse of packImportedVertexColor.
public float3 decodeImportedVertexColor(float4 srgb)
{
    const float3 c = srgb.rgb;
    const float3 low = c / 12.92;
    const float3 high = pow((c + 0.055) / 1.055, float3(2.4, 2.4, 2.4));
    return float3(
        c.x <= 0.04045 ? low.x : high.x,
        c.y <= 0.04045 ? low.y : high.y,
        c.z <= 0.04045 ? low.z : high.z);
}

public float3 decodeImportedVertexNormal(float2 oct)
{
    float3 n = float3(oct.x, 1.0 - abs(oct.x) - abs(oct.y), oct.y);
    if (n.y < 0.0)
    {
        const float foldedX = (1.0 - abs(n.z)) * (n.x >= 0.0 ? 1.0 : -1.0);
        const float foldedZ = (1.0 - abs(n.x)) * (n.z >= 0.0 ? 1.0 : -1.0);
        n.x = foldedX;
        n.z = foldedZ;
    }
    return normalize(n);
}

public uint encodeImportedVertexNormal(float3 normal)
{
    const float3 n = normalize(normal);
    const float sum = abs(n.x) + abs(n.y) + abs(n.z);
    float u = n.x / sum;
    float v = n.z / sum;
    if (n.y < 0.0)
    {
        const float foldedU = (1.0 - abs(v)) * (u >= 0.0 ? 1.0 : -1.0);
        const float foldedV = (1.0 - abs(u)) * (v >= 0.0 ? 1.0 : -1.0);
        u = foldedU;
        v = foldedV;
    }
    const int qu = int(round(clamp(u, -1.0, 1.0) * 32767.0));
    const int qv = int(round(clamp(v, -1.0, 1.0) * 32767.0));
    return (uint(qu) & 0xffffu) | ((uint(qv) & 0xffffu) << 16);
}

public uint encodeImportedVertexColor(float3 linearColor)
{
    const float3 c = clamp(linearColor, 0.0, 1.0);
    const float3 low = c * 12.92;
    const float3 high = (1.055 * pow(c, float3(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4))) - 0.055;
    const float3 srgb = float3(
        c.x <= 0.0031308 ? low.x : high.x,
        c.y <= 0.0031308 ? low.y : high.y,
        c.z <= 0.0031308 ? low.z : high.z);
    const uint r = uint(round(srgb.x * 255.0)) & 0xffu;
    const uint g = uint(round(srgb.y * 255.0)) & 0xffu;
    const uint b = uint(round(srgb.z * 255.0)) & 0xffu;
    return r | (g << 8) | (b << 16) | (0xffu << 24);
}
