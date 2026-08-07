#pragma once

// Material library: the canonical layout for named, indexed PBR materials.
//
// This header is the ONE place the material-index bit layout and the GPU struct
// are defined. It deliberately has no dependencies — no JSON, and above all no
// Vulkan — because three very different consumers need it:
//
//   * src/import/         — packing indices into vertex flags, (de)serialization
//   * src/render/         — building the GPU table and the descriptor binding
//   * tests/              — no test executable links Vulkan (a hard project
//                           rule), so the static_assert below could not live in
//                           renderer_shared.h, which includes Vulkan headers
//
// ---------------------------------------------------------------------------
// Why a table at all
//
// Before this, a material WAS its vertices: metallic and roughness were 8-bit
// values quantized into ImportedScenePackedVertex::flags bits 8-23, and there
// was no way to name a material, reuse it, or change one without re-quantizing
// every vertex that used it and re-uploading the whole scene. An index plus a
// small GPU table means editing a coefficient is a 32-byte write.
//
// ---------------------------------------------------------------------------
// Slot 0 is a reserved sentinel and is never read.
//
// It exists so that the index stored in vertex flags and the index into the
// table are the SAME NUMBER everywhere — no -1 anywhere. That off-by-one would
// otherwise have to be reproduced identically in the packer, the loader and the
// fragment shader, which are three hand-mirrored sites with no compiler
// checking them against each other. 32 wasted bytes buys that away.
//
// ---------------------------------------------------------------------------
// The fallback contract
//
//   index = (flags >> 24) & 0xff
//   index != 0 && index < materialCount  ->  table[index]                (library)
//   otherwise                            ->  unpackImportedSceneMaterialFlags(flags)
//
// Every scene cooked before this feature has bits 24-31 zero, so it takes the
// legacy branch unconditionally and shades exactly as it did. The packer also
// keeps writing quantized metallic/roughness into bits 8-23 even when an index
// is set, so a scene whose material table fails to load degrades to an
// approximation rather than going flat.

#include <cstdint>

namespace odai::importer {

// Bits 24-31 of ImportedScenePackedVertex::flags. Bits 3-7 are RESERVED for
// widening this field to 13 bits (8192 materials) if a large Morrowind/New
// Vegas cook ever exceeds 255 distinct archetypes — do not consume them for
// anything else, or that escape hatch closes.
inline constexpr int kImportedSceneMaterialIndexShift = 24;
inline constexpr std::uint32_t kImportedSceneMaterialIndexMask = 0xffu;

// Including the reserved slot 0, so 255 authorable materials.
inline constexpr std::uint32_t kImportedSceneMaterialTableCapacity = 256u;

// GPU-side material record. Two float4s: the layout is then identical under
// std140 and std430, which sidesteps the alignment question entirely and
// matches the house style (every CameraUniform field is a float[4]). Metallic
// and roughness ride in the .w lanes that would otherwise be padding.
struct alignas(16) GpuImportedMaterial {
    float baseColorMetallic[4] = {1.0f, 1.0f, 1.0f, 0.0f};  // rgb = tint, w = metallic
    float emissiveRoughness[4] = {0.0f, 0.0f, 0.0f, 1.0f};  // rgb = emissive, w = roughness
};
static_assert(sizeof(GpuImportedMaterial) == 32u,
              "GpuImportedMaterial is mirrored in imported_static.frag.slang and sized into the "
              "descriptor range; changing it silently reinterprets the whole table");
static_assert(alignof(GpuImportedMaterial) == 16u);

inline constexpr std::uint32_t importedSceneMaterialIndex(std::uint32_t flags) {
    return (flags >> kImportedSceneMaterialIndexShift) & kImportedSceneMaterialIndexMask;
}

// NOTE on nointerpolation: inFlags is declared `nointerpolation` in the vertex
// and fragment shaders, so the material index is taken from the provoking
// vertex — it is effectively per-triangle. Any future tool that re-welds
// vertices across a material boundary would produce wrong indices. Nothing in
// the tree does that today.

}  // namespace odai::importer
