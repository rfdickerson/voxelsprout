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
    std::string name;
    std::vector<float> positions;   // xyz per vertex, world space (parent transforms applied)
    std::vector<float> normals;     // xyz per vertex, world space; empty if the source had none
    std::vector<std::uint32_t> triangleIndices;  // 3 per triangle, indexes into positions/normals
};

struct NifModel {
    std::vector<NifShape> shapes;
    // Count of NiTriShape blocks encountered whose geometry could not be
    // safely extracted (unrecognized NiTriShapeData field layout) and were
    // therefore dropped rather than emitted as possibly-corrupt geometry.
    std::uint32_t skippedShapeCount = 0;
};

bool parseNifStaticMesh(const std::vector<std::uint8_t>& bytes, NifModel& outModel, std::string& outError);
bool loadNifStaticMesh(const std::filesystem::path& path, NifModel& outModel, std::string& outError);

}  // namespace odai::importer::fnv
