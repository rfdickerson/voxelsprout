#pragma once

// Minimal reader for Gamebryo NIF files as produced for Fallout 3 / Fallout:
// New Vegas static meshes (header version 20.2.0.7). Extracts only what a
// static-mesh cooker needs: the NiNode transform hierarchy and NiTriShape /
// NiTriShapeData triangle geometry (positions, normals, indices). It does
// NOT extract materials/textures (BSShaderPPLightingProperty's exact field
// layout across FO3/FNV patch versions is not something this port could
// verify against a real sample in this environment — see nif_scene.cc's
// file comment) or skinned/animated meshes (skin instance refs are read but
// not resolved).
//
// Design note on robustness: NIF 20.x block headers give an explicit byte
// size per block, so a block whose internal fields this parser doesn't fully
// understand can always be *skipped* correctly by jumping to its declared
// end offset — this parser leans on that heavily. Every block type reads
// only the leading fields it actually needs, then unconditionally resumes
// at the header-declared next-block offset, so a wrong guess about a
// trailing field's layout cannot desynchronize the rest of the file. Where a
// block's own internal optional-field layout is genuinely uncertain (see
// NiTriShapeData in the .cc), the parser self-checks that its byte
// consumption lands within the block's declared size and drops that single
// shape's geometry (not the whole file) if it doesn't.
//
// This is a best-effort implementation against public Gamebryo/NIF format
// documentation. It has not been validated against a real Fallout: New
// Vegas .nif file in this environment (no Data Files available here) — test
// it against real assets before trusting cooked output.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace odai::importer::fnv {

struct NifShape {
    std::string name;               // from the NIF string table; may be empty
    std::vector<float> positions;   // xyz per vertex, world space (parent transforms applied)
    std::vector<float> normals;     // xyz per vertex, world space; empty if the source had none
    std::vector<float> uvs;         // uv per vertex (set 0); empty if the source had none
    std::vector<std::uint32_t> triangleIndices;  // 3 per triangle, indexes into positions/normals
    // Diffuse texture path as stored in the NIF, relative to Data\textures
    // and backslash-separated. Empty when the shape has no resolvable
    // BSShaderTextureSet.
    std::string diffuseTexturePath;
    // NiAlphaProperty declared alpha testing (flag 0x200) for this shape.
    bool alphaTest = false;
    // NiStencilProperty draw mode DRAW_BOTH. Fallout marks window glass,
    // foliage cards and awnings this way; drawn single-sided they lose
    // whichever face points away from the camera.
    bool twoSided = false;
    // NiAlphaProperty's blend bit. The imported static path draws opaque only,
    // so a blended shape (glass, an additive effect billboard) rendered through
    // it appears as a solid slab -- Goodsprings' window panes and dust effects
    // were floating white rectangles until this was read.
    bool alphaBlend = false;
};

struct NifModel {
    std::vector<NifShape> shapes;
    // Count of geometry blocks that were dropped rather than emitted as
    // possibly-corrupt geometry: either the NiTriShapeData/NiTriStripsData
    // field layout did not parse cleanly, or a NiTriShape/NiTriStrips pointed
    // at a data block that could not be read. A nonzero count here means
    // geometry is missing from the model — it is not decorative.
    std::uint32_t skippedShapeCount = 0;
    // Blocks whose type name ends in "Node" but which this parser does not
    // know how to walk. Nonzero means geometry may be missing or, worse,
    // reparented to the origin — see isNodeTypeName in nif_scene.cc.
    std::uint32_t unhandledNodeTypeCount = 0;
    // Type names of blocks referenced as a shape's property that this parser did
    // not turn into either a texture set or an alpha setting. A shape can hold a
    // shader property of a type the parser has no branch for, in which case it
    // silently ends up with no diffuse path and shades from the per-model hashed
    // colour -- grey patches on an otherwise textured model. Populated for
    // diagnostics (odai_newvegas_probe --nif) so those types can be identified
    // rather than guessed at.
    std::vector<std::string> unresolvedPropertyTypes;
};

// Raw block/string inventory of a NIF, for diagnostics only (odai_newvegas_probe
// --nifblocks). Nothing in the import path consumes this; it exists so questions
// like "what type is this shape's property, and where does its texture path
// actually live" get answered by reading the file instead of guessing at field
// offsets -- the failure mode this parser has hit repeatedly.
struct NifBlockSummary {
    std::vector<std::string> blockTypeNames;  // per block, indexed by block index
    std::vector<std::uint32_t> blockSizes;    // per block, bytes
    std::vector<std::string> strings;         // header string table
    std::vector<std::size_t> blockStarts;     // byte offset of each block
};

bool parseNifBlockSummary(
    const std::vector<std::uint8_t>& bytes, NifBlockSummary& outSummary, std::string& outError);

bool parseNifStaticMesh(const std::vector<std::uint8_t>& bytes, NifModel& outModel, std::string& outError);
bool loadNifStaticMesh(const std::filesystem::path& path, NifModel& outModel, std::string& outError);

}  // namespace odai::importer::fnv
