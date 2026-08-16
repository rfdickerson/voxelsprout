#include "import/imported_scene.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace odai::importer {

namespace {

constexpr std::uint32_t kImportedSceneMagic = 0x4E435356u;  // VSCN
// v16 -> v17: per-texture TextureFormat byte (BC data no longer reloads as
// RGBA8) and a trailing pageRanges section (per-page frustum culling survives
// the save/load round trip).
// v17 -> v18: a trailing named material library. Appended and version-gated, so
// v15-v17 files load with an empty table; their vertices have material index 0
// in flag bits 24-31 and take the legacy per-vertex path, rendering unchanged.
// v18 -> v19: ImportedSceneVertex gained a colour. This one is NOT appended, it
// widens a struct that is read back as a raw array blit, so v15-v18 files need
// their vertices read in the old 8-float layout and expanded (readSceneMeshes
// does this). Their colour defaults to white and no vertex carries the tint
// flag, so they shade exactly as before.
// v19 -> v20: terrain layer blending (Fallout ATXT/VTXT). BOTH vertex structs
// widened again -- ImportedSceneVertex by 3 layer indices and 3 weights,
// ImportedScenePackedVertex by 3 indices and a packed weight word -- so both
// raw-blit arrays need their own legacy expansion, in three places: the full
// loader reads meshes and packed vertices, the runtime loader skips meshes and
// reads packed vertices.
// v20 -> v21: the terrain layer budget went from 3 slots to 4, widening BOTH
// vertex structs again. Same treatment as before -- every raw-blit read site is
// version-gated, and the v20 expansion below reads the narrower layout.
// v21 -> v22: a trailing doors section. Appended and version-gated, so v15-v21
// files load with no doors and behave exactly as before -- unlike the vertex
// widenings above, this one touches no existing bytes.
// v23 -> v24: a trailing alphaFlagsAuthored byte (see ImportedScene). Appended
// and version-gated like the doors section; older files read false and keep
// running the content inference they were cooked under.
// v24 -> v25: both vertex structs gained a `colorAlpha` float (authored vertex
// alpha; see ImportedSceneVertex::colorAlpha). Unlike the doors and
// alphaFlagsAuthored sections this widens the raw-blit arrays AGAIN, so it gets
// the same treatment every previous widening did: a legacy stride and a
// field-by-field expansion, which leaves older files reading alpha 1.0 and
// therefore shading exactly as they did.
constexpr std::uint32_t kImportedSceneVersion = 25u;
constexpr std::uint32_t kMinSupportedImportedSceneVersion = 15u;
// The pre-v19 ImportedSceneVertex: position[3], normal[3], uv[2].
constexpr std::size_t kImportedSceneVertexFloatsV18 = 8;
// The v19 ImportedSceneVertex: the above plus color[3].
constexpr std::size_t kImportedSceneVertexFloatsV19 = 11;
// The pre-v20 ImportedScenePackedVertex: position[3], normal[3], color[3],
// uv[2], then textureIndex and flags as two more 4-byte words.
constexpr std::size_t kImportedScenePackedVertexWordsV19 = 13;
// The v20 layouts, before the fourth layer slot: ImportedSceneVertex was 11
// floats plus 3 layer indices plus 3 weights; ImportedScenePackedVertex was the
// v19 13 words plus 3 indices and one packed weight word.
constexpr std::size_t kImportedSceneVertexFloatsV20 = 17;
constexpr std::size_t kImportedScenePackedVertexWordsV20 = 17;
constexpr int kImportedSceneMaxTerrainLayersV20 = 3;
// The v21-v24 layouts, before colorAlpha: ImportedSceneVertex was position[3],
// normal[3], uv[2], color[3] then 4 layer indices and 4 weights;
// ImportedScenePackedVertex was position[3], normal[3], color[3], uv[2],
// textureIndex, flags, 4 layer indices and one packed weight word.
constexpr std::size_t kImportedSceneVertexFloatsV21 = 19;
constexpr std::size_t kImportedScenePackedVertexWordsV21 = 18;
constexpr std::uint8_t kImportedSceneMaxTextureFormat =
    static_cast<std::uint8_t>(TextureFormat::BC2);

// pageRanges are serialized as a raw array, so the layout must stay packed.
static_assert(sizeof(ImportedScenePageRange) == 36u);
// packedVertices is blitted to disk field-for-field, so a size change silently
// reinterprets every existing file. There IS a version gate now
// (readPackedVertexArray), but this assert stays: it is what forces whoever
// widens the struct to go and add the next legacy branch, rather than shipping
// a reader that quietly mis-strides. Was 52 through v19; v20 added three layer
// texture indices and a packed weight word.
static_assert(sizeof(ImportedScenePackedVertex) == 76u);
static_assert(sizeof(ImportedSceneVertex) == 80u);
// The pre-v20 width readPackedVertexArray expands from must stay in step with
// the layout it decodes field by field.
static_assert(kImportedScenePackedVertexWordsV19 * sizeof(std::uint32_t) == 52u);
static_assert(kImportedScenePackedVertexWordsV20 * sizeof(std::uint32_t) == 68u);
static_assert(kImportedSceneVertexFloatsV20 * sizeof(float) == 68u);
static_assert(kImportedScenePackedVertexWordsV21 * sizeof(std::uint32_t) == 72u);
static_assert(kImportedSceneVertexFloatsV21 * sizeof(float) == 76u);

// Materials are NOT raw-blitted -- ImportedSceneMaterial holds a std::string --
// so they are written field by field. Both loaders must stay in step with this.
bool readString(std::istream& input, std::string& out);
void writeSceneMaterials(std::ostream& output, const std::vector<ImportedSceneMaterial>& materials);
void writeSceneDoors(std::ostream& output, const std::vector<ImportedSceneDoor>& doors);
bool readSceneDoors(std::istream& input, std::vector<ImportedSceneDoor>& out);
bool readSceneMaterials(std::istream& input, std::vector<ImportedSceneMaterial>& out);

std::string g_lastImportedSceneError;

void setLastImportedSceneError(std::string message) {
    g_lastImportedSceneError = std::move(message);
}

struct DebugBounds {
    std::array<float, 3> min{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    std::array<float, 3> max{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    };
    bool valid = false;
};

[[maybe_unused]] void expandBounds(DebugBounds& bounds, const std::array<float, 3>& point) {
    bounds.valid = true;
    for (int axis = 0; axis < 3; ++axis) {
        bounds.min[axis] = std::min(bounds.min[axis], point[axis]);
        bounds.max[axis] = std::max(bounds.max[axis], point[axis]);
    }
}

struct PackedRenderColor {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

PackedRenderColor packedRenderColorFromHash(std::string_view key) {
    std::uint32_t hash = 2166136261u;
    for (const char ch : key) {
        hash ^= static_cast<std::uint8_t>(ch);
        hash *= 16777619u;
    }
    PackedRenderColor color{};
    color.r = 0.30f + (static_cast<float>((hash >> 0) & 0xffu) / 255.0f) * 0.55f;
    color.g = 0.28f + (static_cast<float>((hash >> 8) & 0xffu) / 255.0f) * 0.52f;
    color.b = 0.25f + (static_cast<float>((hash >> 16) & 0xffu) / 255.0f) * 0.50f;
    return color;
}

PackedRenderColor packedTerrainColor(float height) {
    PackedRenderColor color{};
    const float normalized = std::clamp((height + 256.0f) / 1024.0f, 0.0f, 1.0f);
    color.r = 0.22f + (normalized * 0.32f);
    color.g = 0.24f + (normalized * 0.34f);
    color.b = 0.18f + (normalized * 0.14f);
    return color;
}


// Decodes the 8-byte BC3/BC5-style alpha block palette and folds each texel's
// alpha into the transparent/visible presence flags.
// Verdict for the bimodality test below: a cutout mask is mostly fully
// transparent or fully opaque, with few texels in between. A specular mask --
// which is what Bethesda puts in the diffuse alpha of ordinary building
// surfaces -- sits in the middle and is nearly all "in between".
struct AlphaBandCounts {
    std::size_t low = 0;   // < 32, effectively transparent
    std::size_t mid = 0;   // the rest
    std::size_t high = 0;  // > 224, effectively opaque

    void add(std::uint8_t alpha) {
        if (alpha < 32u) {
            ++low;
        } else if (alpha > 224u) {
            ++high;
        } else {
            ++mid;
        }
    }

    // A cutout needs BOTH a real transparent region and a mid band small
    // enough to be just the antialiased rim between the two.
    //
    // The old test was "any texel below 250 and any above 8", which a flat
    // 0.5 alpha satisfies trivially -- and that is exactly the value a
    // specular mask holds. Every ordinary wall and roof therefore had alpha
    // test forced on at the default 0.5 threshold, and tore into ragged holes
    // wherever the mask crossed it. That is the "missing parts of the
    // building" symptom, and it is this heuristic, not the shader.
    [[nodiscard]] bool looksLikeCutout() const {
        const std::size_t total = low + mid + high;
        if (total == 0u) {
            return false;
        }
        constexpr double kMinTransparentFraction = 0.01;  // at least 1% cut away
        constexpr double kMaxMidFraction = 0.20;          // rim, not a gradient
        const double lowFraction = static_cast<double>(low) / static_cast<double>(total);
        const double midFraction = static_cast<double>(mid) / static_cast<double>(total);
        return lowFraction >= kMinTransparentFraction && midFraction <= kMaxMidFraction;
    }
};

// Folds one BC1 block's 16 texels into the transparent/opaque bands.
//
// BC1 alpha is one bit and it is a SIDE EFFECT OF THE ENCODING MODE, not an
// authored channel: a block signals 3-colour mode with color0 <= color1, and
// index 3 in that mode is transparent black. Encoders choose that mode whenever
// a block needs only three colours, and Fallout's encoder emits index-3 texels
// for near-black pixels in textures that are entirely opaque.
//
// So "this block has a transparent texel" says almost nothing on its own, and
// the counts are what matter. See the caller.
void bc1BlockBands(const std::uint8_t* block, AlphaBandCounts& bands) {
    const std::uint16_t color0 = static_cast<std::uint16_t>(block[0] | (block[1] << 8));
    const std::uint16_t color1 = static_cast<std::uint16_t>(block[2] | (block[3] << 8));
    const bool punchThrough = color0 <= color1;
    for (int byteIndex = 4; byteIndex < 8; ++byteIndex) {
        std::uint8_t bits = block[byteIndex];
        for (int texel = 0; texel < 4; ++texel) {
            const bool transparent = punchThrough && ((bits & 0x3u) == 0x3u);
            // One bit of alpha: there is no mid band to land in.
            bands.add(transparent ? 0u : 255u);
            bits >>= 2;
        }
    }
}

void bc3AlphaBlockBands(const std::uint8_t* block, AlphaBandCounts& bands) {
    const std::uint8_t alpha0 = block[0];
    const std::uint8_t alpha1 = block[1];
    std::uint8_t palette[8] = {alpha0, alpha1};
    if (alpha0 > alpha1) {
        for (int step = 1; step <= 6; ++step) {
            palette[1 + step] = static_cast<std::uint8_t>(((7 - step) * alpha0 + step * alpha1) / 7);
        }
    } else {
        for (int step = 1; step <= 4; ++step) {
            palette[1 + step] = static_cast<std::uint8_t>(((5 - step) * alpha0 + step * alpha1) / 5);
        }
        palette[6] = 0u;
        palette[7] = 255u;
    }
    std::uint64_t indexBits = 0;
    for (int byteIndex = 0; byteIndex < 6; ++byteIndex) {
        indexBits |= static_cast<std::uint64_t>(block[2 + byteIndex]) << (8 * byteIndex);
    }
    for (int texel = 0; texel < 16; ++texel) {
        bands.add(palette[(indexBits >> (3 * texel)) & 0x7u]);
    }
}

// Fills `bands` with the texture's alpha histogram. False means the format
// carries no readable colour alpha (BC4/BC5) or the blob is short -- callers
// must then fall back to whatever the source authored rather than guess.
//
// Split out of textureUsesAlphaCutout so demoteFalseAlphaBlendFlags can ask a
// different question of the same histogram instead of re-deriving it.
bool collectTextureAlphaBands(const ImportedSceneTexture& texture, AlphaBandCounts& bands) {
    if (texture.width == 0u || texture.height == 0u) {
        return false;
    }
    const std::size_t baseBlockCount =
        (static_cast<std::size_t>(texture.width) + 3u) / 4u *
        ((static_cast<std::size_t>(texture.height) + 3u) / 4u);
    switch (texture.format) {
        case TextureFormat::RGBA8: {
            const std::size_t basePixelCount = static_cast<std::size_t>(texture.width) * texture.height;
            const std::size_t baseByteCount = basePixelCount * 4u;
            if (texture.rgba8.size() < baseByteCount) {
                return false;
            }
            for (std::size_t pixelIndex = 0; pixelIndex < basePixelCount; ++pixelIndex) {
                bands.add(texture.rgba8[(pixelIndex * 4u) + 3u]);
            }
            return true;
        }
        case TextureFormat::BC1: {
            if (texture.rgba8.size() < baseBlockCount * 8u) {
                return false;
            }
            // This used to return true on the FIRST block holding a
            // transparent texel, which is the same indefensible "any texel"
            // test the bimodal rewrite removed from the other formats -- BC1
            // was simply missed. Because BC1's transparency is an artifact of
            // the block encoding mode, a handful of dark blocks in an opaque
            // wall texture were enough to classify the whole thing as a cutout,
            // and every one of those texels was then discarded. That is the
            // ragged holes in Goodsprings' buildings: they cluster in the dark
            // regions because that is where the encoder chose 3-colour mode.
            for (std::size_t blockIndex = 0; blockIndex < baseBlockCount; ++blockIndex) {
                bc1BlockBands(texture.rgba8.data() + (blockIndex * 8u), bands);
            }
            return true;
        }
        case TextureFormat::BC2: {
            // BC2 stores 16 explicit 4-bit alpha values in the first 8 bytes
            // of each 16-byte block, so this needs no decode at all.
            if (texture.rgba8.size() < baseBlockCount * 16u) {
                return false;
            }
            for (std::size_t blockIndex = 0; blockIndex < baseBlockCount; ++blockIndex) {
                const std::uint8_t* alphaBytes = texture.rgba8.data() + (blockIndex * 16u);
                for (std::size_t byteIndex = 0; byteIndex < 8u; ++byteIndex) {
                    const std::uint8_t packed = alphaBytes[byteIndex];
                    for (const std::uint8_t nibble : {static_cast<std::uint8_t>(packed & 0x0Fu),
                                                      static_cast<std::uint8_t>(packed >> 4u)}) {
                        // 4-bit alpha widened to 8 so one band test serves
                        // every format.
                        bands.add(static_cast<std::uint8_t>(nibble * 17u));
                    }
                }
            }
            return true;
        }
        case TextureFormat::BC3: {
            if (texture.rgba8.size() < baseBlockCount * 16u) {
                return false;
            }
            for (std::size_t blockIndex = 0; blockIndex < baseBlockCount; ++blockIndex) {
                bc3AlphaBlockBands(texture.rgba8.data() + (blockIndex * 16u), bands);
            }
            return true;
        }
        // BC4/BC5 carry no color alpha; BC7 alpha needs a full per-mode decode,
        // so a BC7 cook that wants cutout must set the part flag explicitly.
        default:
            return false;
    }
}

bool textureUsesAlphaCutout(const ImportedSceneTexture& texture) {
    AlphaBandCounts bands;
    return collectTextureAlphaBands(texture, bands) && bands.looksLikeCutout();
}

std::vector<bool> buildTextureAlphaCutoutMask(const std::vector<ImportedSceneTexture>& textures) {
    std::vector<bool> mask(textures.size(), false);
    for (std::size_t textureIndex = 0; textureIndex < textures.size(); ++textureIndex) {
        mask[textureIndex] = textureUsesAlphaCutout(textures[textureIndex]);
    }
    return mask;
}

bool textureIndexUsesAlphaCutout(const std::vector<bool>& mask, std::uint32_t textureIndex) {
    return textureIndex < mask.size() && mask[textureIndex];
}

// Infers alpha test from texture CONTENT, for importers that do not tell us.
// It is a guess, and a deliberately eager one: any texture holding both a
// transparent and a visible texel counts.
//
// It must not fire on a surface whose source format stated its own blend mode.
// A blended surface's alpha is a coverage ramp, not a cutout mask, and running
// it through a 0.5 discard throws away exactly the part that was supposed to
// blend -- glass at a flat alpha of 0.3 discards entirely and disappears, and a
// dust sheet keeps only its opaque half with a hard edge where the gradient
// was. On Fallout's Goodsprings this hit 554 of 557 blended draws, so the
// blended pass had almost nothing left to draw by the time it ran.
//
// So alphaBlend vetoes the guess. It does NOT veto alphaTest set by the
// importer itself: NiAlphaProperty can legitimately set both bits, and that
// surface still wants its own threshold applied.
// The other direction: a surface the SOURCE marked alpha-blended whose texture
// holds no gradient to blend. Only ever demotes, never promotes.
//
// Fallout ships stray NiAlphaProperties. Goodsprings' water tank
// (nv_watertank.nif) is three shapes, and two of them -- the tank body and the
// concrete pad it stands on -- are authored blend=1/test=0 while their textures
// are 97% fully opaque. Drawn through the blended pipeline that reads as a
// faintly see-through tank, and worse than the look: the blended pass writes no
// depth and is skipped by the shadow and normal-depth passes, so the tank also
// casts no shadow and contributes nothing to AO.
//
// Three-way on the same alpha histogram the cutout classifier already builds:
//
//   no transparent texels at all  -> the blend is a no-op, make it OPAQUE
//   bimodal (transparent + opaque, thin rim) -> it is a cutout, make it TESTED
//   anything with a real mid-range gradient  -> genuine transparency, keep it
//
// The last branch is what keeps glass and dust sheets working, and it is why
// this is safe where the old content guess was not: that one FORCED alpha test
// onto opaque geometry sharing a cutout's texture (554 of 557 blended draws on
// this same map). This only ever takes work away from the blended pass, and a
// flat-alpha pane -- low=0, mid=100% -- matches neither demotion branch.
enum class BlendDemotion { Keep, ToAlphaTest, ToOpaque };

BlendDemotion classifyAuthoredBlend(const ImportedSceneTexture& texture) {
    AlphaBandCounts bands;
    if (!collectTextureAlphaBands(texture, bands)) {
        return BlendDemotion::Keep;  // format we cannot read -- trust the author
    }
    const std::size_t total = bands.low + bands.mid + bands.high;
    if (total == 0u) {
        return BlendDemotion::Keep;
    }
    const double lowFraction = static_cast<double>(bands.low) / static_cast<double>(total);
    const double midFraction = static_cast<double>(bands.mid) / static_cast<double>(total);
    // Nothing meaningfully transparent and no gradient: blending this composites
    // the surface over the background at ~1.0 and is pure cost.
    if (lowFraction < 0.001 && midFraction < 0.02) {
        return BlendDemotion::ToOpaque;
    }
    if (bands.looksLikeCutout()) {
        return BlendDemotion::ToAlphaTest;
    }
    return BlendDemotion::Keep;
}

void demoteFalseAlphaBlendFlags(ImportedScene& scene) {
    // Only for importers that state the mode per shape. Where the mode was
    // itself inferred there is nothing authored to disagree with.
    if (!scene.alphaFlagsAuthored || scene.textures.empty()) {
        return;
    }
    std::vector<BlendDemotion> perTexture(scene.textures.size(), BlendDemotion::Keep);
    for (std::size_t i = 0; i < scene.textures.size(); ++i) {
        perTexture[i] = classifyAuthoredBlend(scene.textures[i]);
    }
    const auto verdictFor = [&](std::uint32_t textureIndex) {
        return textureIndex < perTexture.size() ? perTexture[textureIndex] : BlendDemotion::Keep;
    };
    for (ImportedSceneMesh& mesh : scene.meshes) {
        for (ImportedSceneMeshPart& part : mesh.parts) {
            if (!part.alphaBlend) {
                continue;
            }
            switch (verdictFor(part.textureIndex)) {
                case BlendDemotion::ToOpaque:
                    part.alphaBlend = false;
                    break;
                case BlendDemotion::ToAlphaTest:
                    part.alphaBlend = false;
                    part.alphaTest = true;
                    break;
                case BlendDemotion::Keep:
                    break;
            }
        }
    }
    // Runs on load as well as on build, so a cell coming back from the disk
    // cache is corrected the same way -- the flags are packed into the vertex
    // there, which is the only copy that survives.
    for (ImportedScenePackedVertex& vertex : scene.packedVertices) {
        if ((vertex.flags & kImportedSceneMaterialFlagAlphaBlend) == 0u) {
            continue;
        }
        switch (verdictFor(vertex.textureIndex)) {
            case BlendDemotion::ToOpaque:
                vertex.flags &= ~kImportedSceneMaterialFlagAlphaBlend;
                break;
            case BlendDemotion::ToAlphaTest:
                vertex.flags &= ~kImportedSceneMaterialFlagAlphaBlend;
                vertex.flags |= kImportedSceneMaterialFlagAlphaTest;
                break;
            case BlendDemotion::Keep:
                break;
        }
    }
}

void applyTextureAlphaCutoutFlags(ImportedScene& scene) {
    // A scene whose importer authored the alpha mode per shape (FNV NIFs)
    // never gets the content guess: a texture can be a real cutout for one
    // shape and plain opaque decoration for another, and only the source
    // format knows which is which. Overriding it here forced alpha test onto
    // authored-opaque geometry that shared a cutout's texture -- see the
    // field's comment in imported_scene.h.
    if (scene.alphaFlagsAuthored) {
        return;
    }
    const std::vector<bool> textureAlphaCutoutMask = buildTextureAlphaCutoutMask(scene.textures);
    for (ImportedSceneMesh& mesh : scene.meshes) {
        for (ImportedSceneMeshPart& part : mesh.parts) {
            if (!part.alphaBlend &&
                textureIndexUsesAlphaCutout(textureAlphaCutoutMask, part.textureIndex)) {
                part.alphaTest = true;
            }
        }
    }
    // Runs on load as well as on build, so a cell coming back from the disk
    // cache has to be filtered the same way -- the flag is already packed into
    // the vertex there, which is what makes the veto checkable at all.
    for (ImportedScenePackedVertex& vertex : scene.packedVertices) {
        if ((vertex.flags & kImportedSceneMaterialFlagAlphaBlend) != 0u) {
            continue;
        }
        if (textureIndexUsesAlphaCutout(textureAlphaCutoutMask, vertex.textureIndex)) {
            vertex.flags |= kImportedSceneMaterialFlagAlphaTest;
        }
    }
}

bool readExact(std::istream& input, void* dst, std::size_t size) {
    input.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
    return input.good();
}

bool skipExact(std::istream& input, std::size_t size) {
    input.seekg(static_cast<std::streamoff>(size), std::ios::cur);
    return input.good();
}

template <typename T>
bool readValue(std::istream& input, T& out) {
    static_assert(std::is_trivially_copyable_v<T>);
    return readExact(input, &out, sizeof(T));
}

template <typename T>
void writeValue(std::ostream& output, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    output.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
}

void writeString(std::ostream& output, const std::string& value) {
    const std::uint32_t size = static_cast<std::uint32_t>(value.size());
    writeValue(output, size);
    if (!value.empty()) {
        output.write(value.data(), static_cast<std::streamsize>(value.size()));
    }
}

void writeSceneMaterials(std::ostream& output,
                         const std::vector<ImportedSceneMaterial>& materials) {
    writeValue(output, static_cast<std::uint32_t>(materials.size()));
    for (const ImportedSceneMaterial& m : materials) {
        writeString(output, m.name);
        writeValue(output, m.baseColorTint[0]);
        writeValue(output, m.baseColorTint[1]);
        writeValue(output, m.baseColorTint[2]);
        writeValue(output, m.metallic);
        writeValue(output, m.roughness);
        writeValue(output, m.emissive[0]);
        writeValue(output, m.emissive[1]);
        writeValue(output, m.emissive[2]);
        writeValue(output, m.emissiveStrength);
    }
}

bool readSceneMaterials(std::istream& input, std::vector<ImportedSceneMaterial>& out) {
    std::uint32_t count = 0;
    if (!readValue(input, count)) {
        return false;
    }
    if (count > kImportedSceneMaterialTableCapacity) {
        // Refuse rather than truncate: the vertex indices in this same file
        // reference these slots, so silently dropping entries would repaint
        // whatever geometry pointed past the cut.
        setLastImportedSceneError("Material table has " + std::to_string(count) +
                                  " entries, more than the " +
                                  std::to_string(kImportedSceneMaterialTableCapacity) +
                                  "-slot capacity");
        return false;
    }
    out.resize(count);
    for (ImportedSceneMaterial& m : out) {
        if (!readString(input, m.name) || !readValue(input, m.baseColorTint[0]) ||
            !readValue(input, m.baseColorTint[1]) || !readValue(input, m.baseColorTint[2]) ||
            !readValue(input, m.metallic) || !readValue(input, m.roughness) ||
            !readValue(input, m.emissive[0]) || !readValue(input, m.emissive[1]) ||
            !readValue(input, m.emissive[2]) || !readValue(input, m.emissiveStrength)) {
            return false;
        }
    }
    // Slot 0 is the reserved sentinel; a file claiming otherwise is malformed.
    // Normalise rather than reject -- the geometry is still good, and every
    // consumer already treats index 0 as "no library material".
    if (!out.empty() && !out[0].name.empty()) {
        out[0] = ImportedSceneMaterial{};
    }
    return true;
}

// Doors are written field by field, not blitted: ImportedSceneDoor holds a
// std::string. Same treatment materials already get.
void writeSceneDoors(std::ostream& output, const std::vector<ImportedSceneDoor>& doors) {
    writeValue(output, static_cast<std::uint32_t>(doors.size()));
    for (const ImportedSceneDoor& door : doors) {
        output.write(reinterpret_cast<const char*>(door.position), sizeof(door.position));
        output.write(reinterpret_cast<const char*>(door.arrivalPosition), sizeof(door.arrivalPosition));
        writeValue(output, door.arrivalYawDegrees);
        writeString(output, door.targetCellEditorId);
    }
}

bool readSceneDoors(std::istream& input, std::vector<ImportedSceneDoor>& out) {
    std::uint32_t count = 0;
    if (!readValue(input, count)) {
        return false;
    }
    out.resize(count);
    for (ImportedSceneDoor& door : out) {
        if (!readExact(input, door.position, sizeof(door.position)) ||
            !readExact(input, door.arrivalPosition, sizeof(door.arrivalPosition)) ||
            !readValue(input, door.arrivalYawDegrees) ||
            !readString(input, door.targetCellEditorId)) {
            return false;
        }
    }
    return true;
}

bool readString(std::istream& input, std::string& out) {
    std::uint32_t size = 0;
    if (!readValue(input, size)) {
        return false;
    }
    out.resize(size);
    return size == 0 || readExact(input, out.data(), size);
}

// --- Legacy vertex layouts -------------------------------------------------
//
// Both vertex structs are stored as raw array blits, so every widening of one
// leaves older files with a narrower stride on disk. These keep the "how wide
// was it in version N" answer in one place: the readers below expand, and
// legacyPackedVertexStride is also what the runtime loader uses to SKIP the
// mesh block. Getting a stride wrong does not error, it desyncs every section
// after the array, so all three sites go through here.

std::size_t legacyMeshVertexStride(std::uint32_t version) {
    if (version >= 25u) {
        return sizeof(ImportedSceneVertex);
    }
    if (version >= 21u) {
        return kImportedSceneVertexFloatsV21 * sizeof(float);
    }
    if (version >= 20u) {
        return kImportedSceneVertexFloatsV20 * sizeof(float);
    }
    if (version >= 19u) {
        return kImportedSceneVertexFloatsV19 * sizeof(float);
    }
    return kImportedSceneVertexFloatsV18 * sizeof(float);
}

std::size_t legacyPackedVertexStride(std::uint32_t version) {
    if (version >= 25u) {
        return sizeof(ImportedScenePackedVertex);
    }
    if (version >= 21u) {
        return kImportedScenePackedVertexWordsV21 * sizeof(std::uint32_t);
    }
    if (version >= 20u) {
        return kImportedScenePackedVertexWordsV20 * sizeof(std::uint32_t);
    }
    return kImportedScenePackedVertexWordsV19 * sizeof(std::uint32_t);
}


// A count read out of a scene file is UNTRUSTED. A cache file truncated by a
// killed process or a full disk still parses a valid header, and every count
// after the truncation point is then whatever bytes happened to be on disk --
// which is how a 15 MB cached cell came to claim 608,890,047 vertices and kill
// the streaming worker with std::bad_alloc instead of simply rebuilding.
//
// Bounding by the bytes the stream can still supply, BEFORE allocating, turns
// that into a clean parse failure that the caller already handles. The element
// size passed is the SMALLEST the on-disk record can be across every supported
// version, so the check never rejects a file it should accept.
[[nodiscard]] bool countFitsInStream(
    std::istream& input, std::uint32_t count, std::size_t minBytesPerElement) {
    if (count == 0u) {
        return true;
    }
    const std::streampos here = input.tellg();
    if (here < 0) {
        return true;  // not seekable; nothing to check against
    }
    input.seekg(0, std::ios::end);
    const std::streampos end = input.tellg();
    input.seekg(here, std::ios::beg);
    if (!input || end < here) {
        return false;
    }
    const auto remaining = static_cast<std::uintmax_t>(end - here);
    return (static_cast<std::uintmax_t>(count) * minBytesPerElement) <= remaining;
}

bool readMeshVertexArray(
    std::istream& input,
    std::uint32_t version,
    std::vector<ImportedSceneVertex>& out
) {
    if (out.empty()) {
        return true;
    }
    if (version >= 25u) {
        return readExact(input, out.data(), out.size() * sizeof(ImportedSceneVertex));
    }
    const std::size_t floatsPerVertex = legacyMeshVertexStride(version) / sizeof(float);
    std::vector<float> legacy(out.size() * floatsPerVertex);
    if (!readExact(input, legacy.data(), legacy.size() * sizeof(float))) {
        return false;
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float* src = legacy.data() + (i * floatsPerVertex);
        ImportedSceneVertex& dst = out[i];
        dst.position[0] = src[0];
        dst.position[1] = src[1];
        dst.position[2] = src[2];
        dst.normal[0] = src[3];
        dst.normal[1] = src[4];
        dst.normal[2] = src[5];
        dst.uv[0] = src[6];
        dst.uv[1] = src[7];
        if (floatsPerVertex >= kImportedSceneVertexFloatsV19) {
            dst.color[0] = src[8];
            dst.color[1] = src[9];
            dst.color[2] = src[10];
        }
        if (floatsPerVertex >= kImportedSceneVertexFloatsV20) {
            // v20 stored 3 layer indices (as uint32 bit patterns in the float
            // array) followed by 3 weights. Pre-v20 has none, and those keep the
            // constructor's "no layer" defaults -- no such vertex carries
            // kImportedSceneMaterialFlagTerrainLayers, so nothing reads them.
            for (int layer = 0; layer < kImportedSceneMaxTerrainLayersV20; ++layer) {
                std::memcpy(&dst.layerTextureIndex[layer], &src[11 + layer], sizeof(std::uint32_t));
                dst.layerWeight[layer] = src[14 + layer];
            }
        }
        if (floatsPerVertex >= kImportedSceneVertexFloatsV21) {
            // v21 widened the layer budget to four and moved the weights along
            // with it. colorAlpha does not exist in this layout and keeps its
            // 1.0 default, which is what makes an older file shade unchanged.
            for (int layer = 0; layer < kImportedSceneMaxTerrainLayers; ++layer) {
                std::memcpy(&dst.layerTextureIndex[layer], &src[11 + layer], sizeof(std::uint32_t));
                dst.layerWeight[layer] = src[15 + layer];
            }
        }
    }
    return true;
}

// ImportedScenePackedDraw grew from two uints to two uints plus the alpha
// threshold and its padding at version 23. Anything older is read as the old
// pair and keeps the neutral 128 default, which is exactly the behaviour those
// files were produced under.
bool readPackedDrawArray(
    std::istream& input,
    std::uint32_t version,
    std::vector<ImportedScenePackedDraw>& out
) {
    if (out.empty()) {
        return true;
    }
    if (version >= 23u) {
        return readExact(input, out.data(), out.size() * sizeof(ImportedScenePackedDraw));
    }
    std::vector<std::uint32_t> legacy(out.size() * 2u);
    if (!readExact(input, legacy.data(), legacy.size() * sizeof(std::uint32_t))) {
        return false;
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = ImportedScenePackedDraw{};
        out[i].firstIndex = legacy[i * 2u];
        out[i].indexCount = legacy[(i * 2u) + 1u];
    }
    return true;
}

bool readPackedVertexArray(
    std::istream& input,
    std::uint32_t version,
    std::vector<ImportedScenePackedVertex>& out
) {
    if (out.empty()) {
        return true;
    }
    if (version >= 25u) {
        return readExact(input, out.data(), out.size() * sizeof(ImportedScenePackedVertex));
    }
    const std::size_t wordsPerVertex = legacyPackedVertexStride(version) / sizeof(std::uint32_t);
    // 11 floats then 2 uints, plus (v20) 3 layer indices and a weight word.
    std::vector<std::uint32_t> legacy(out.size() * wordsPerVertex);
    if (!readExact(input, legacy.data(), legacy.size() * sizeof(std::uint32_t))) {
        return false;
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::uint32_t* src = legacy.data() + (i * wordsPerVertex);
        float floats[11];
        std::memcpy(floats, src, sizeof(floats));
        ImportedScenePackedVertex& dst = out[i];
        dst.position[0] = floats[0];
        dst.position[1] = floats[1];
        dst.position[2] = floats[2];
        dst.normal[0] = floats[3];
        dst.normal[1] = floats[4];
        dst.normal[2] = floats[5];
        dst.color[0] = floats[6];
        dst.color[1] = floats[7];
        dst.color[2] = floats[8];
        dst.uv[0] = floats[9];
        dst.uv[1] = floats[10];
        dst.textureIndex = src[11];
        dst.flags = src[12];
        if (wordsPerVertex >= kImportedScenePackedVertexWordsV20) {
            for (int layer = 0; layer < kImportedSceneMaxTerrainLayersV20; ++layer) {
                dst.layerTextureIndex[layer] = src[13 + layer];
            }
            dst.layerWeights = src[16];
        }
        if (wordsPerVertex >= kImportedScenePackedVertexWordsV21) {
            // v21's fourth layer slot pushed the weight word out by one.
            // colorAlpha is absent here and keeps its 1.0 default.
            for (int layer = 0; layer < kImportedSceneMaxTerrainLayers; ++layer) {
                dst.layerTextureIndex[layer] = src[13 + layer];
            }
            dst.layerWeights = src[17];
        }
    }
    return true;
}

bool skipString(std::istream& input) {
    std::uint32_t size = 0;
    if (!readValue(input, size)) {
        return false;
    }
    return size == 0 || skipExact(input, size);
}

std::array<float, 3> transformPoint(
    const std::array<float, 16>& matrix,
    const std::array<float, 3>& point
) {
    return {
        (matrix[0] * point[0]) + (matrix[1] * point[1]) + (matrix[2] * point[2]) + matrix[3],
        (matrix[4] * point[0]) + (matrix[5] * point[1]) + (matrix[6] * point[2]) + matrix[7],
        (matrix[8] * point[0]) + (matrix[9] * point[1]) + (matrix[10] * point[2]) + matrix[11]
    };
}

std::array<float, 3> transformDirection(
    const std::array<float, 16>& matrix,
    const std::array<float, 3>& direction
) {
    return {
        (matrix[0] * direction[0]) + (matrix[1] * direction[1]) + (matrix[2] * direction[2]),
        (matrix[4] * direction[0]) + (matrix[5] * direction[1]) + (matrix[6] * direction[2]),
        (matrix[8] * direction[0]) + (matrix[9] * direction[1]) + (matrix[10] * direction[2])
    };
}

std::array<float, 3> normalizeVector(std::array<float, 3> value) {
    const float length = std::sqrt(
        (value[0] * value[0]) +
        (value[1] * value[1]) +
        (value[2] * value[2]));
    if (length > 1e-6f) {
        value[0] /= length;
        value[1] /= length;
        value[2] /= length;
    }
    return value;
}

}  // namespace

void buildImportedScenePackedRenderData(ImportedScene& scene) {
    applyTextureAlphaCutoutFlags(scene);
    // Paired with the call above: that one infers a mode where none was
    // authored, this one corrects one that was authored wrong. Exactly one
    // of the two does anything for any given scene (they test
    // alphaFlagsAuthored in opposite senses).
    demoteFalseAlphaBlendFlags(scene);
    scene.packedVertices.clear();
    scene.packedIndices.clear();
    scene.packedDraws.clear();
    scene.boundsMin[0] = std::numeric_limits<float>::max();
    scene.boundsMin[1] = std::numeric_limits<float>::max();
    scene.boundsMin[2] = std::numeric_limits<float>::max();
    scene.boundsMax[0] = std::numeric_limits<float>::lowest();
    scene.boundsMax[1] = std::numeric_limits<float>::lowest();
    scene.boundsMax[2] = std::numeric_limits<float>::lowest();

    auto expandBounds = [&](const ImportedScenePackedVertex& vertex) {
        scene.boundsMin[0] = std::min(scene.boundsMin[0], vertex.position[0]);
        scene.boundsMin[1] = std::min(scene.boundsMin[1], vertex.position[1]);
        scene.boundsMin[2] = std::min(scene.boundsMin[2], vertex.position[2]);
        scene.boundsMax[0] = std::max(scene.boundsMax[0], vertex.position[0]);
        scene.boundsMax[1] = std::max(scene.boundsMax[1], vertex.position[1]);
        scene.boundsMax[2] = std::max(scene.boundsMax[2], vertex.position[2]);
    };

    auto appendMesh = [&](const ImportedSceneMesh& mesh,
                          const std::array<float, 16>& transform,
                          const PackedRenderColor& color) {
        if (mesh.vertices.empty() || mesh.indices.empty()) {
            return;
        }
        const std::uint32_t firstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
        const auto appendVertex = [&](const ImportedSceneVertex& srcVertex,
                                      std::uint32_t textureIndex,
                                      std::uint32_t flags) {
            ImportedScenePackedVertex dstVertex{};
            const std::array<float, 3> localPosition{
                srcVertex.position[0],
                srcVertex.position[1],
                srcVertex.position[2]
            };
            const std::array<float, 3> localNormal{
                srcVertex.normal[0],
                srcVertex.normal[1],
                srcVertex.normal[2]
            };
            const std::array<float, 3> worldPosition = transformPoint(transform, localPosition);
            const std::array<float, 3> worldNormal = normalizeVector(transformDirection(transform, localNormal));
            dstVertex.position[0] = worldPosition[0];
            dstVertex.position[1] = worldPosition[1];
            dstVertex.position[2] = worldPosition[2];
            dstVertex.normal[0] = worldNormal[0];
            dstVertex.normal[1] = worldNormal[1];
            dstVertex.normal[2] = worldNormal[2];
            dstVertex.color[0] = color.r;
            dstVertex.color[1] = color.g;
            dstVertex.color[2] = color.b;
            // Carried through even though the RGB above is a per-model
            // stand-in: alpha is authored data whether or not the colour beside
            // it is. See ImportedSceneVertex::colorAlpha.
            dstVertex.colorAlpha = srcVertex.colorAlpha;
            dstVertex.uv[0] = srcVertex.uv[0];
            dstVertex.uv[1] = srcVertex.uv[1];
            dstVertex.textureIndex = textureIndex;
            dstVertex.flags = flags;
            const std::uint32_t packedVertexIndex = static_cast<std::uint32_t>(scene.packedVertices.size());
            scene.packedVertices.push_back(dstVertex);
            expandBounds(dstVertex);
            return packedVertexIndex;
        };

        // One draw per PART, not per mesh.
        //
        // A part is one source shape, and material state that cannot live on
        // the vertex -- the alpha-test threshold -- is per part. An importer
        // routinely builds one mesh out of many shapes (a Fallout house is
        // walls plus glass plus trim in a single NIF), so collapsing them into
        // one draw would force all of them to share one threshold.
        //
        // This does not inflate what the GPU sees: the renderer's upload merges
        // adjacent draws that agree on every piece of state it cares about, so
        // parts that genuinely match are recombined there, and only parts that
        // actually differ stay apart.
        const auto emitDraw = [&](std::uint32_t drawFirstIndex, std::uint8_t alphaThreshold) {
            const std::uint32_t indexCount =
                static_cast<std::uint32_t>(scene.packedIndices.size() - drawFirstIndex);
            if (indexCount == 0u) {
                return;
            }
            ImportedScenePackedDraw draw{};
            draw.firstIndex = drawFirstIndex;
            draw.indexCount = indexCount;
            draw.alphaThreshold = alphaThreshold;
            scene.packedDraws.push_back(draw);
        };

        if (mesh.parts.empty()) {
            const std::uint32_t invalidTextureIndex = std::numeric_limits<std::uint32_t>::max();
            std::vector<std::uint32_t> remappedVertexIndices(
                mesh.vertices.size(),
                std::numeric_limits<std::uint32_t>::max());
            for (const std::uint32_t index : mesh.indices) {
                if (index >= mesh.vertices.size()) {
                    continue;
                }
                std::uint32_t& remappedIndex = remappedVertexIndices[index];
                if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                    remappedIndex = appendVertex(mesh.vertices[index], invalidTextureIndex, 0u);
                }
                scene.packedIndices.push_back(remappedIndex);
            }
            // No part, so no authored alpha mode; the neutral default applies
            // and this draw never alpha-tests anyway.
            emitDraw(firstIndex, 128u);
        } else {
            for (const ImportedSceneMeshPart& part : mesh.parts) {
                if (part.indexCount == 0u || part.firstIndex >= mesh.indices.size()) {
                    continue;
                }
                const std::uint32_t partDrawFirstIndex =
                    static_cast<std::uint32_t>(scene.packedIndices.size());
                std::vector<std::uint32_t> remappedVertexIndices(
                    mesh.vertices.size(),
                    std::numeric_limits<std::uint32_t>::max());
                const std::size_t firstPartIndex = static_cast<std::size_t>(part.firstIndex);
                const std::size_t lastPartIndex = std::min(
                    firstPartIndex + static_cast<std::size_t>(part.indexCount),
                    mesh.indices.size());
                const std::uint32_t partFlags =
                    (part.alphaTest ? kImportedSceneMaterialFlagAlphaTest : 0u) |
                    (part.alphaBlend ? kImportedSceneMaterialFlagAlphaBlend : 0u) |
                    (part.twoSided ? kImportedSceneMaterialFlagTwoSided : 0u) |
                    (part.unlit ? kImportedSceneMaterialFlagUnlit : 0u);
                for (std::size_t indexOffset = firstPartIndex; indexOffset < lastPartIndex; ++indexOffset) {
                    const std::uint32_t index = mesh.indices[indexOffset];
                    if (index >= mesh.vertices.size()) {
                        continue;
                    }
                    std::uint32_t& remappedIndex = remappedVertexIndices[index];
                    if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                        remappedIndex = appendVertex(mesh.vertices[index], part.textureIndex, partFlags);
                    }
                    scene.packedIndices.push_back(remappedIndex);
                }
                emitDraw(partDrawFirstIndex, part.alphaThreshold);
            }
        }
    };

    const bool hasTerrainMesh = !scene.meshes.empty() && scene.meshes.front().name == "terrain";
    if (hasTerrainMesh) {
        const ImportedSceneMesh& terrainMesh = scene.meshes.front();
        if (!terrainMesh.vertices.empty() && !terrainMesh.indices.empty()) {
            for (const ImportedSceneMeshPart& part : terrainMesh.parts) {
                if (part.indexCount == 0u || part.firstIndex >= terrainMesh.indices.size()) {
                    continue;
                }
                const std::uint32_t firstIndex = static_cast<std::uint32_t>(scene.packedIndices.size());
                std::vector<std::uint32_t> remappedVertexIndices(
                    terrainMesh.vertices.size(),
                    std::numeric_limits<std::uint32_t>::max());
                const std::size_t firstPartIndex = static_cast<std::size_t>(part.firstIndex);
                const std::size_t lastPartIndex = std::min(
                    firstPartIndex + static_cast<std::size_t>(part.indexCount),
                    terrainMesh.indices.size());
                for (std::size_t indexOffset = firstPartIndex; indexOffset < lastPartIndex; ++indexOffset) {
                    const std::uint32_t index = terrainMesh.indices[indexOffset];
                    if (index >= terrainMesh.vertices.size()) {
                        continue;
                    }
                    std::uint32_t& remappedIndex = remappedVertexIndices[index];
                    if (remappedIndex == std::numeric_limits<std::uint32_t>::max()) {
                        const ImportedSceneVertex& srcVertex = terrainMesh.vertices[index];
                        ImportedScenePackedVertex dstVertex{};
                        const std::array<float, 3> worldNormal = normalizeVector({
                            srcVertex.normal[0],
                            srcVertex.normal[1],
                            srcVertex.normal[2]
                        });
                        // An authored vertex colour tints the texture; the
                        // height ramp is only a stand-in for geometry that has
                        // none, and it is not something to multiply a real
                        // texture by -- it would stripe the terrain by altitude.
                        const bool hasAuthoredColor =
                            srcVertex.color[0] != 1.0f ||
                            srcVertex.color[1] != 1.0f ||
                            srcVertex.color[2] != 1.0f;
                        const PackedRenderColor color = hasAuthoredColor
                            ? PackedRenderColor{srcVertex.color[0], srcVertex.color[1], srcVertex.color[2]}
                            : packedTerrainColor(srcVertex.position[1]);
                        dstVertex.position[0] = srcVertex.position[0];
                        dstVertex.position[1] = srcVertex.position[1];
                        dstVertex.position[2] = srcVertex.position[2];
                        dstVertex.normal[0] = worldNormal[0];
                        dstVertex.normal[1] = worldNormal[1];
                        dstVertex.normal[2] = worldNormal[2];
                        dstVertex.color[0] = color.r;
                        dstVertex.color[1] = color.g;
                        dstVertex.color[2] = color.b;
                        dstVertex.colorAlpha = srcVertex.colorAlpha;
                        dstVertex.uv[0] = srcVertex.uv[0];
                        dstVertex.uv[1] = srcVertex.uv[1];
                        dstVertex.textureIndex = part.textureIndex;
                        dstVertex.flags = hasAuthoredColor
                            ? kImportedSceneMaterialFlagVertexColorTint
                            : 0u;
                        // Terrain layers ride through untouched; the flag opts
                        // in only when a layer is actually present, so a scene
                        // whose cooker never filled these reads exactly as
                        // before.
                        bool hasTerrainLayer = false;
                        float layerWeights[kImportedSceneMaxTerrainLayers] = {};
                        for (int layer = 0; layer < kImportedSceneMaxTerrainLayers; ++layer) {
                            dstVertex.layerTextureIndex[layer] = srcVertex.layerTextureIndex[layer];
                            layerWeights[layer] = srcVertex.layerWeight[layer];
                            if (srcVertex.layerTextureIndex[layer] != kImportedSceneNoTerrainLayer &&
                                srcVertex.layerWeight[layer] > 0.0f) {
                                hasTerrainLayer = true;
                            }
                        }
                        dstVertex.layerWeights = packImportedSceneTerrainLayerWeights(layerWeights);
                        if (hasTerrainLayer) {
                            dstVertex.flags |= kImportedSceneMaterialFlagTerrainLayers;
                        }
                        remappedIndex = static_cast<std::uint32_t>(scene.packedVertices.size());
                        scene.packedVertices.push_back(dstVertex);
                        expandBounds(dstVertex);
                    }
                    scene.packedIndices.push_back(remappedIndex);
                }
                const std::uint32_t indexCount =
                    static_cast<std::uint32_t>(scene.packedIndices.size() - firstIndex);
                if (indexCount != 0u) {
                    ImportedScenePackedDraw draw{};
                    draw.firstIndex = firstIndex;
                    draw.indexCount = indexCount;
                    draw.alphaThreshold = part.alphaThreshold;
                    scene.packedDraws.push_back(draw);
                }
            }
        }

    }

    for (const ImportedSceneInstance& instance : scene.instances) {
        if ((hasTerrainMesh && instance.meshIndex == 0u) || instance.meshIndex >= scene.meshes.size()) {
            continue;
        }
        std::array<float, 16> transform{};
        std::copy(std::begin(instance.transform), std::end(instance.transform), transform.begin());
        appendMesh(
            scene.meshes[instance.meshIndex],
            transform,
            packedRenderColorFromHash(instance.modelPath));
    }

    if (scene.packedVertices.empty()) {
        scene.boundsMin[0] = 0.0f;
        scene.boundsMin[1] = 0.0f;
        scene.boundsMin[2] = 0.0f;
        scene.boundsMax[0] = 0.0f;
        scene.boundsMax[1] = 0.0f;
        scene.boundsMax[2] = 0.0f;
    }
}

void computeImportedSceneBoundsFromPackedData(ImportedScene& scene) {
    if (scene.packedVertices.empty()) {
        scene.boundsMin[0] = 0.0f;
        scene.boundsMin[1] = 0.0f;
        scene.boundsMin[2] = 0.0f;
        scene.boundsMax[0] = 0.0f;
        scene.boundsMax[1] = 0.0f;
        scene.boundsMax[2] = 0.0f;
        return;
    }
    scene.boundsMin[0] = std::numeric_limits<float>::max();
    scene.boundsMin[1] = std::numeric_limits<float>::max();
    scene.boundsMin[2] = std::numeric_limits<float>::max();
    scene.boundsMax[0] = std::numeric_limits<float>::lowest();
    scene.boundsMax[1] = std::numeric_limits<float>::lowest();
    scene.boundsMax[2] = std::numeric_limits<float>::lowest();
    for (const ImportedScenePackedVertex& vertex : scene.packedVertices) {
        scene.boundsMin[0] = std::min(scene.boundsMin[0], vertex.position[0]);
        scene.boundsMin[1] = std::min(scene.boundsMin[1], vertex.position[1]);
        scene.boundsMin[2] = std::min(scene.boundsMin[2], vertex.position[2]);
        scene.boundsMax[0] = std::max(scene.boundsMax[0], vertex.position[0]);
        scene.boundsMax[1] = std::max(scene.boundsMax[1], vertex.position[1]);
        scene.boundsMax[2] = std::max(scene.boundsMax[2], vertex.position[2]);
    }
}

void buildImportedScenePageRanges(ImportedScene& scene, float pageSize) {
    scene.pageRanges.clear();
    const std::size_t drawCount = scene.packedDraws.size();
    if (drawCount == 0u || pageSize <= 0.0f || scene.packedIndices.empty()) {
        return;
    }

    // Mirror the renderer's terrain classification: exterior scenes treat the
    // leading one-draw-per-landscape-cell range as terrain, interiors have none.
    const bool isInterior = importedSceneSourceTagIsInterior(scene.sourceTag);
    const std::uint32_t landscapeCellCount = !scene.landscapeCells.empty()
        ? static_cast<std::uint32_t>(scene.landscapeCells.size())
        : scene.sourceLandscapeCellCount;
    const std::uint32_t terrainDrawCount = isInterior
        ? 0u
        : std::min<std::uint32_t>(landscapeCellCount, static_cast<std::uint32_t>(drawCount));

    struct DrawPageInfo {
        std::uint32_t drawIndex = 0;
        std::int32_t cellX = 0;
        std::int32_t cellZ = 0;
        bool terrain = false;
        bool hasBounds = false;
        std::array<float, 3> boundsMin{
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()};
        std::array<float, 3> boundsMax{
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::lowest()};
    };

    std::vector<DrawPageInfo> infos(drawCount);
    for (std::size_t drawIndex = 0; drawIndex < drawCount; ++drawIndex) {
        DrawPageInfo& info = infos[drawIndex];
        info.drawIndex = static_cast<std::uint32_t>(drawIndex);
        info.terrain = drawIndex < terrainDrawCount;
        const ImportedScenePackedDraw& draw = scene.packedDraws[drawIndex];
        const std::size_t firstIndex = std::min(
            static_cast<std::size_t>(draw.firstIndex), scene.packedIndices.size());
        const std::size_t lastIndex = std::min(
            firstIndex + static_cast<std::size_t>(draw.indexCount), scene.packedIndices.size());
        for (std::size_t indexOffset = firstIndex; indexOffset < lastIndex; ++indexOffset) {
            const std::uint32_t vertexIndex = scene.packedIndices[indexOffset];
            if (vertexIndex >= scene.packedVertices.size()) {
                continue;
            }
            const ImportedScenePackedVertex& vertex = scene.packedVertices[vertexIndex];
            for (int axis = 0; axis < 3; ++axis) {
                info.boundsMin[axis] = std::min(info.boundsMin[axis], vertex.position[axis]);
                info.boundsMax[axis] = std::max(info.boundsMax[axis], vertex.position[axis]);
            }
            info.hasBounds = true;
        }
        if (info.hasBounds) {
            const float centerX = (info.boundsMin[0] + info.boundsMax[0]) * 0.5f;
            const float centerZ = (info.boundsMin[2] + info.boundsMax[2]) * 0.5f;
            info.cellX = static_cast<std::int32_t>(std::floor(centerX / pageSize));
            info.cellZ = static_cast<std::int32_t>(std::floor(centerZ / pageSize));
        }
    }

    // Terrain first (so the [0, terrainDrawCount) invariant survives), then by
    // XZ tile so page members become contiguous. Terrain and statics never mix
    // in one page. Stable ordering keeps output deterministic.
    std::stable_sort(
        infos.begin(),
        infos.end(),
        [](const DrawPageInfo& a, const DrawPageInfo& b) {
            if (a.terrain != b.terrain) {
                return a.terrain;
            }
            if (a.cellZ != b.cellZ) {
                return a.cellZ < b.cellZ;
            }
            return a.cellX < b.cellX;
        });

    std::vector<std::uint32_t> newIndices;
    newIndices.reserve(scene.packedIndices.size());
    std::vector<ImportedScenePackedDraw> newDraws;
    newDraws.reserve(drawCount);
    std::vector<ImportedScenePageRange> pages;
    bool pageHasBounds = false;
    bool lastTerrain = false;
    std::int32_t lastCellX = 0;
    std::int32_t lastCellZ = 0;
    for (const DrawPageInfo& info : infos) {
        const ImportedScenePackedDraw& srcDraw = scene.packedDraws[info.drawIndex];
        const std::size_t firstIndex = std::min(
            static_cast<std::size_t>(srcDraw.firstIndex), scene.packedIndices.size());
        const std::size_t lastIndex = std::min(
            firstIndex + static_cast<std::size_t>(srcDraw.indexCount), scene.packedIndices.size());

        ImportedScenePackedDraw dstDraw{};
        dstDraw.firstIndex = static_cast<std::uint32_t>(newIndices.size());
        dstDraw.indexCount = static_cast<std::uint32_t>(lastIndex - firstIndex);
        dstDraw.alphaThreshold = srcDraw.alphaThreshold;
        newIndices.insert(
            newIndices.end(),
            scene.packedIndices.begin() + static_cast<std::ptrdiff_t>(firstIndex),
            scene.packedIndices.begin() + static_cast<std::ptrdiff_t>(lastIndex));

        const bool startNewPage = pages.empty() ||
            lastTerrain != info.terrain ||
            lastCellX != info.cellX ||
            lastCellZ != info.cellZ;
        lastTerrain = info.terrain;
        lastCellX = info.cellX;
        lastCellZ = info.cellZ;
        if (startNewPage) {
            ImportedScenePageRange page{};
            page.firstDraw = static_cast<std::uint32_t>(newDraws.size());
            pages.push_back(page);
            pageHasBounds = false;
        }
        ImportedScenePageRange& page = pages.back();
        ++page.drawCount;
        if (info.terrain) {
            ++page.terrainDrawCount;
        }
        if (info.hasBounds) {
            if (!pageHasBounds) {
                for (int axis = 0; axis < 3; ++axis) {
                    page.boundsMin[axis] = info.boundsMin[axis];
                    page.boundsMax[axis] = info.boundsMax[axis];
                }
                pageHasBounds = true;
            } else {
                for (int axis = 0; axis < 3; ++axis) {
                    page.boundsMin[axis] = std::min(page.boundsMin[axis], info.boundsMin[axis]);
                    page.boundsMax[axis] = std::max(page.boundsMax[axis], info.boundsMax[axis]);
                }
            }
        }
        newDraws.push_back(dstDraw);
    }

    scene.packedIndices = std::move(newIndices);
    scene.packedDraws = std::move(newDraws);
    scene.pageRanges = std::move(pages);
}

bool importedSceneSourceTagIsInterior(std::string_view sourceTag) {
    return sourceTag == "morrowind_interior" || sourceTag == "fnv_interior";
}

std::string importedSceneInteriorFileName(
    const std::string& exteriorStem, const std::string& cellEditorId
) {
    return exteriorStem + "_" + cellEditorId + ".bin";
}

const std::string& getImportedSceneLastError() {
    return g_lastImportedSceneError;
}

bool saveImportedScene(const ImportedScene& scene, const std::filesystem::path& outputPath) {
    g_lastImportedSceneError.clear();
    const std::filesystem::path parentPath = outputPath.parent_path();
    if (!parentPath.empty()) {
        std::error_code mkdirError;
        std::filesystem::create_directories(parentPath, mkdirError);
        if (mkdirError) {
            setLastImportedSceneError(
                "Failed to create output directory " + parentPath.string() + ": " + mkdirError.message());
            return false;
        }
    }

    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        setLastImportedSceneError("Failed to open output file for writing: " + outputPath.string());
        return false;
    }

    writeValue(output, kImportedSceneMagic);
    writeValue(output, kImportedSceneVersion);
    writeString(output, scene.sourceTag);

    const std::uint32_t textureCount = static_cast<std::uint32_t>(scene.textures.size());
    const std::uint32_t meshCount = static_cast<std::uint32_t>(scene.meshes.size());
    const std::uint32_t instanceCount = static_cast<std::uint32_t>(scene.instances.size());
    const std::uint32_t landscapeCellCount = static_cast<std::uint32_t>(scene.landscapeCells.size());
    const std::uint32_t waterPatchCount = static_cast<std::uint32_t>(scene.waterPatches.size());
    const std::uint32_t lightCount = static_cast<std::uint32_t>(scene.lights.size());
    const std::uint32_t unresolvedRefCount = static_cast<std::uint32_t>(scene.unresolvedRefs.size());
    const std::uint32_t packedVertexCount = static_cast<std::uint32_t>(scene.packedVertices.size());
    const std::uint32_t packedIndexCount = static_cast<std::uint32_t>(scene.packedIndices.size());
    const std::uint32_t packedDrawCount = static_cast<std::uint32_t>(scene.packedDraws.size());
    writeValue(output, textureCount);
    writeValue(output, meshCount);
    writeValue(output, instanceCount);
    writeValue(output, landscapeCellCount);
    writeValue(output, waterPatchCount);
    writeValue(output, lightCount);
    writeValue(output, unresolvedRefCount);
    writeValue(output, packedVertexCount);
    writeValue(output, packedIndexCount);
    writeValue(output, packedDrawCount);
    output.write(reinterpret_cast<const char*>(scene.boundsMin), static_cast<std::streamsize>(sizeof(scene.boundsMin)));
    output.write(reinterpret_cast<const char*>(scene.boundsMax), static_cast<std::streamsize>(sizeof(scene.boundsMax)));

    for (const ImportedSceneTexture& texture : scene.textures) {
        writeString(output, texture.sourcePath);
        writeValue(output, texture.width);
        writeValue(output, texture.height);
        writeValue(output, texture.mipLevelCount);
        writeValue(output, static_cast<std::uint8_t>(texture.format));
        const std::uint32_t rgbaSize = static_cast<std::uint32_t>(texture.rgba8.size());
        writeValue(output, rgbaSize);
        if (!texture.rgba8.empty()) {
            output.write(reinterpret_cast<const char*>(texture.rgba8.data()), static_cast<std::streamsize>(texture.rgba8.size()));
        }
    }

    for (const ImportedSceneMesh& mesh : scene.meshes) {
        writeString(output, mesh.name);
        const std::uint32_t vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
        const std::uint32_t indexCount = static_cast<std::uint32_t>(mesh.indices.size());
        const std::uint32_t partCount = static_cast<std::uint32_t>(mesh.parts.size());
        writeValue(output, vertexCount);
        writeValue(output, indexCount);
        writeValue(output, partCount);
        if (!mesh.vertices.empty()) {
            output.write(reinterpret_cast<const char*>(mesh.vertices.data()),
                         static_cast<std::streamsize>(mesh.vertices.size() * sizeof(ImportedSceneVertex)));
        }
        if (!mesh.indices.empty()) {
            output.write(reinterpret_cast<const char*>(mesh.indices.data()),
                         static_cast<std::streamsize>(mesh.indices.size() * sizeof(std::uint32_t)));
        }
        if (!mesh.parts.empty()) {
            output.write(reinterpret_cast<const char*>(mesh.parts.data()),
                         static_cast<std::streamsize>(mesh.parts.size() * sizeof(ImportedSceneMeshPart)));
        }
    }

    for (const ImportedSceneInstance& instance : scene.instances) {
        writeValue(output, instance.meshIndex);
        output.write(reinterpret_cast<const char*>(instance.transform), static_cast<std::streamsize>(sizeof(instance.transform)));
        writeString(output, instance.sourceId);
        writeString(output, instance.modelPath);
    }

    for (const ImportedSceneLandscapeCell& cell : scene.landscapeCells) {
        writeValue(output, cell.gridX);
        writeValue(output, cell.gridY);
        const std::uint32_t heightCount = static_cast<std::uint32_t>(cell.heights.size());
        const std::uint32_t textureIndexCount = static_cast<std::uint32_t>(cell.textureIndices.size());
        writeValue(output, heightCount);
        writeValue(output, textureIndexCount);
        if (!cell.heights.empty()) {
            output.write(reinterpret_cast<const char*>(cell.heights.data()),
                         static_cast<std::streamsize>(cell.heights.size() * sizeof(float)));
        }
        if (!cell.textureIndices.empty()) {
            output.write(reinterpret_cast<const char*>(cell.textureIndices.data()),
                         static_cast<std::streamsize>(cell.textureIndices.size() * sizeof(std::uint16_t)));
        }
    }

    if (!scene.waterPatches.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.waterPatches.data()),
            static_cast<std::streamsize>(scene.waterPatches.size() * sizeof(ImportedSceneWaterPatch)));
    }

    for (const ImportedSceneLight& light : scene.lights) {
        writeString(output, light.sourceId);
        output.write(reinterpret_cast<const char*>(light.position), static_cast<std::streamsize>(sizeof(light.position)));
        output.write(reinterpret_cast<const char*>(light.color), static_cast<std::streamsize>(sizeof(light.color)));
        writeValue(output, light.radius);
        writeValue(output, light.intensity);
        writeValue(output, light.flags);
    }

    for (const ImportedSceneCellRef& ref : scene.unresolvedRefs) {
        writeString(output, ref.refId);
        writeString(output, ref.modelPath);
        output.write(reinterpret_cast<const char*>(ref.position), static_cast<std::streamsize>(sizeof(ref.position)));
        output.write(reinterpret_cast<const char*>(ref.rotationRadians), static_cast<std::streamsize>(sizeof(ref.rotationRadians)));
        writeValue(output, ref.scale);
    }

    if (!scene.packedVertices.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.packedVertices.data()),
            static_cast<std::streamsize>(scene.packedVertices.size() * sizeof(ImportedScenePackedVertex)));
    }
    if (!scene.packedIndices.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.packedIndices.data()),
            static_cast<std::streamsize>(scene.packedIndices.size() * sizeof(std::uint32_t)));
    }
    if (!scene.packedDraws.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.packedDraws.data()),
            static_cast<std::streamsize>(scene.packedDraws.size() * sizeof(ImportedScenePackedDraw)));
    }

    const std::uint32_t pageRangeCount = static_cast<std::uint32_t>(scene.pageRanges.size());
    writeValue(output, pageRangeCount);
    if (!scene.pageRanges.empty()) {
        output.write(
            reinterpret_cast<const char*>(scene.pageRanges.data()),
            static_cast<std::streamsize>(scene.pageRanges.size() * sizeof(ImportedScenePageRange)));
    }

    // v18: named material library. Last section, so older readers that stop
    // after pageRanges are unaffected by its presence.
    writeSceneMaterials(output, scene.materials);
    writeSceneDoors(output, scene.doors);
    // v24: whether alpha flags were authored by the importer (see header).
    writeValue(output, static_cast<std::uint8_t>(scene.alphaFlagsAuthored ? 1u : 0u));

    if (!output.good()) {
        setLastImportedSceneError("Failed while writing output file: " + outputPath.string());
        return false;
    }
    return true;
}

bool loadImportedScene(const std::filesystem::path& inputPath, ImportedScene& outScene) {
    g_lastImportedSceneError.clear();
    std::ifstream input(inputPath, std::ios::binary);
    if (!input) {
        setLastImportedSceneError("Failed to open imported scene file: " + inputPath.string());
        return false;
    }

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!readValue(input, magic) || !readValue(input, version) || magic != kImportedSceneMagic) {
        setLastImportedSceneError("Invalid imported scene file header: " + inputPath.string());
        return false;
    }
    if (version < kMinSupportedImportedSceneVersion || version > kImportedSceneVersion) {
        setLastImportedSceneError(
            "Imported scene file version " + std::to_string(version) +
            " is unsupported; recook with the current odai_balmora_cooker (supported versions " +
            std::to_string(kMinSupportedImportedSceneVersion) + "-" +
            std::to_string(kImportedSceneVersion) + ")");
        return false;
    }

    ImportedScene scene{};
    scene.sourceFileVersion = version;
    if (!readString(input, scene.sourceTag)) {
        setLastImportedSceneError("Failed to read imported scene source tag: " + inputPath.string());
        return false;
    }

    std::uint32_t textureCount = 0;
    std::uint32_t meshCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t landscapeCellCount = 0;
    std::uint32_t waterPatchCount = 0;
    std::uint32_t lightCount = 0;
    std::uint32_t unresolvedRefCount = 0;
    std::uint32_t packedVertexCount = 0;
    std::uint32_t packedIndexCount = 0;
    std::uint32_t packedDrawCount = 0;
    if (!readValue(input, textureCount) ||
        !readValue(input, meshCount) ||
        !readValue(input, instanceCount) ||
        !readValue(input, landscapeCellCount) ||
        !readValue(input, waterPatchCount) ||
        !readValue(input, lightCount) ||
        !readValue(input, unresolvedRefCount)) {
        return false;
    }
    if (version >= 2u &&
        (!readValue(input, packedVertexCount) ||
         !readValue(input, packedIndexCount) ||
         !readValue(input, packedDrawCount))) {
        return false;
    }
    if (version >= 3u &&
        (!readExact(input, scene.boundsMin, sizeof(scene.boundsMin)) ||
         !readExact(input, scene.boundsMax, sizeof(scene.boundsMax)))) {
        return false;
    }
    scene.sourceTextureCount = textureCount;
    scene.sourceMeshCount = meshCount;
    scene.sourceInstanceCount = instanceCount;
    scene.sourceLandscapeCellCount = landscapeCellCount;
    scene.sourceWaterPatchCount = waterPatchCount;
    scene.sourceLightCount = lightCount;
    scene.sourceUnresolvedRefCount = unresolvedRefCount;

    if (!countFitsInStream(input, textureCount, 12u)) {
        return false;
    }
    scene.textures.resize(textureCount);
    for (ImportedSceneTexture& texture : scene.textures) {
        if (!readString(input, texture.sourcePath) ||
            !readValue(input, texture.width) ||
            !readValue(input, texture.height) ||
            !readValue(input, texture.mipLevelCount)) {
            return false;
        }
        if (version >= 17u) {
            std::uint8_t formatValue = 0;
            if (!readValue(input, formatValue) || formatValue > kImportedSceneMaxTextureFormat) {
                setLastImportedSceneError(
                    "Invalid texture format in imported scene file: " + inputPath.string());
                return false;
            }
            texture.format = static_cast<TextureFormat>(formatValue);
        }
        std::uint32_t rgbaSize = 0;
        if (!readValue(input, rgbaSize)) {
            return false;
        }
        if (!countFitsInStream(input, static_cast<std::uint32_t>(rgbaSize), 1u)) {
            return false;
        }
        texture.rgba8.resize(rgbaSize);
        if (rgbaSize != 0 && !readExact(input, texture.rgba8.data(), rgbaSize)) {
            return false;
        }
    }

    if (!countFitsInStream(input, meshCount, 16u)) {
        return false;
    }
    scene.meshes.resize(meshCount);
    for (ImportedSceneMesh& mesh : scene.meshes) {
        std::uint32_t vertexCount = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t partCount = 0;
        if (!readString(input, mesh.name) ||
            !readValue(input, vertexCount) ||
            !readValue(input, indexCount) ||
            !readValue(input, partCount)) {
            return false;
        }
        // 32 bytes is the pre-v19 vertex (8 floats); v21+ records are larger.
        if (!countFitsInStream(input, vertexCount, 32u) ||
            !countFitsInStream(input, indexCount, 4u) ||
            !countFitsInStream(input, partCount, 12u)) {
            return false;
        }
        mesh.vertices.resize(vertexCount);
        mesh.indices.resize(indexCount);
        mesh.parts.resize(partCount);
        if (!readMeshVertexArray(input, version, mesh.vertices)) {
            return false;
        }
        if (indexCount != 0 &&
            !readExact(input, mesh.indices.data(), mesh.indices.size() * sizeof(std::uint32_t))) {
            return false;
        }
        if (partCount != 0 &&
            !readExact(input, mesh.parts.data(), mesh.parts.size() * sizeof(ImportedSceneMeshPart))) {
            return false;
        }
    }

    scene.instances.resize(instanceCount);
    for (ImportedSceneInstance& instance : scene.instances) {
        if (!readValue(input, instance.meshIndex) ||
            !readExact(input, instance.transform, sizeof(instance.transform)) ||
            !readString(input, instance.sourceId) ||
            !readString(input, instance.modelPath)) {
            return false;
        }
    }

    scene.landscapeCells.resize(landscapeCellCount);
    for (ImportedSceneLandscapeCell& cell : scene.landscapeCells) {
        std::uint32_t heightCount = 0;
        std::uint32_t textureIndexCount = 0;
        if (!readValue(input, cell.gridX) ||
            !readValue(input, cell.gridY) ||
            !readValue(input, heightCount) ||
            !readValue(input, textureIndexCount)) {
            return false;
        }
        cell.heights.resize(heightCount);
        cell.textureIndices.resize(textureIndexCount);
        if (heightCount != 0 &&
            !readExact(input, cell.heights.data(), cell.heights.size() * sizeof(float))) {
            return false;
        }
        if (textureIndexCount != 0 &&
            !readExact(input, cell.textureIndices.data(), cell.textureIndices.size() * sizeof(std::uint16_t))) {
            return false;
        }
    }

    scene.waterPatches.resize(waterPatchCount);
    if (waterPatchCount != 0 &&
        !readExact(input, scene.waterPatches.data(), scene.waterPatches.size() * sizeof(ImportedSceneWaterPatch))) {
        return false;
    }

    scene.lights.resize(lightCount);
    for (ImportedSceneLight& light : scene.lights) {
        if (!readString(input, light.sourceId) ||
            !readExact(input, light.position, sizeof(light.position)) ||
            !readExact(input, light.color, sizeof(light.color)) ||
            !readValue(input, light.radius) ||
            !readValue(input, light.intensity) ||
            !readValue(input, light.flags)) {
            return false;
        }
        if (version < 16u) {
            const float morrowindY = light.position[1];
            light.position[1] = light.position[2];
            light.position[2] = morrowindY;
        }
    }

    scene.unresolvedRefs.resize(unresolvedRefCount);
    for (ImportedSceneCellRef& ref : scene.unresolvedRefs) {
        if (!readString(input, ref.refId) ||
            !readString(input, ref.modelPath) ||
            !readExact(input, ref.position, sizeof(ref.position)) ||
            !readExact(input, ref.rotationRadians, sizeof(ref.rotationRadians)) ||
            !readValue(input, ref.scale)) {
            return false;
        }
    }

    if (version >= 2u) {
        if (!countFitsInStream(input, packedVertexCount, 32u) ||
            !countFitsInStream(input, packedIndexCount, 4u) ||
            !countFitsInStream(input, packedDrawCount, 12u)) {
            return false;
        }
        scene.packedVertices.resize(packedVertexCount);
        scene.packedIndices.resize(packedIndexCount);
        scene.packedDraws.resize(packedDrawCount);
        if (!readPackedVertexArray(input, version, scene.packedVertices)) {
            return false;
        }
        if (packedIndexCount != 0 &&
            !readExact(input, scene.packedIndices.data(), scene.packedIndices.size() * sizeof(std::uint32_t))) {
            return false;
        }
        if (packedDrawCount != 0 && !readPackedDrawArray(input, version, scene.packedDraws)) {
            return false;
        }
    } else {
        buildImportedScenePackedRenderData(scene);
    }

    if (version >= 17u) {
        std::uint32_t pageRangeCount = 0;
        if (!readValue(input, pageRangeCount)) {
            return false;
        }
        scene.pageRanges.resize(pageRangeCount);
        if (pageRangeCount != 0 &&
            !readExact(input, scene.pageRanges.data(), scene.pageRanges.size() * sizeof(ImportedScenePageRange))) {
            return false;
        }
    }
    if (scene.pageRanges.empty() && scene.packedDraws.size() > 1u) {
        // Pre-v17 cooks (and cooks that skipped paging) draw the whole scene
        // every frame. Rebuild culling pages here so old files get per-page
        // frustum culling without a recook.
        buildImportedScenePageRanges(scene);
    }

    // v18 material library. Absent in older files, which leaves the table empty
    // -- every vertex then carries material index 0 and shades through the
    // legacy per-vertex path exactly as it always did.
    if (version >= 18u && !readSceneMaterials(input, scene.materials)) {
        return false;
    }
    // After materials, matching the write order in saveImportedScene.
    if (version >= 22u && !readSceneDoors(input, scene.doors)) {
        setLastImportedSceneError("Failed to read imported scene doors: " + inputPath.string());
        return false;
    }
    if (version >= 24u) {
        std::uint8_t authored = 0;
        if (!readValue(input, authored)) {
            setLastImportedSceneError(
                "Failed to read imported scene alpha-authored flag: " + inputPath.string());
            return false;
        }
        scene.alphaFlagsAuthored = authored != 0u;
    }

    applyTextureAlphaCutoutFlags(scene);
    // Paired with the call above: that one infers a mode where none was
    // authored, this one corrects one that was authored wrong. Exactly one
    // of the two does anything for any given scene (they test
    // alphaFlagsAuthored in opposite senses).
    demoteFalseAlphaBlendFlags(scene);
    outScene = std::move(scene);
    return true;
}

bool loadImportedSceneRuntime(const std::filesystem::path& inputPath, ImportedScene& outScene) {
    g_lastImportedSceneError.clear();
    std::ifstream input(inputPath, std::ios::binary);
    if (!input) {
        setLastImportedSceneError("Failed to open imported scene file: " + inputPath.string());
        return false;
    }

    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    if (!readValue(input, magic) || !readValue(input, version) || magic != kImportedSceneMagic) {
        setLastImportedSceneError("Invalid imported scene file header: " + inputPath.string());
        return false;
    }
    if (version < kMinSupportedImportedSceneVersion || version > kImportedSceneVersion) {
        setLastImportedSceneError(
            "Imported scene file version " + std::to_string(version) +
            " is unsupported; recook with the current odai_balmora_cooker (supported versions " +
            std::to_string(kMinSupportedImportedSceneVersion) + "-" +
            std::to_string(kImportedSceneVersion) + ")");
        return false;
    }

    ImportedScene scene{};
    scene.sourceFileVersion = version;
    if (!readString(input, scene.sourceTag)) {
        return false;
    }

    std::uint32_t textureCount = 0;
    std::uint32_t meshCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t landscapeCellCount = 0;
    std::uint32_t waterPatchCount = 0;
    std::uint32_t lightCount = 0;
    std::uint32_t unresolvedRefCount = 0;
    std::uint32_t packedVertexCount = 0;
    std::uint32_t packedIndexCount = 0;
    std::uint32_t packedDrawCount = 0;
    if (!readValue(input, textureCount) ||
        !readValue(input, meshCount) ||
        !readValue(input, instanceCount) ||
        !readValue(input, landscapeCellCount) ||
        !readValue(input, waterPatchCount) ||
        !readValue(input, lightCount) ||
        !readValue(input, unresolvedRefCount) ||
        !readValue(input, packedVertexCount) ||
        !readValue(input, packedIndexCount) ||
        !readValue(input, packedDrawCount)) {
        return false;
    }
    if (version >= 3u &&
        (!readExact(input, scene.boundsMin, sizeof(scene.boundsMin)) ||
         !readExact(input, scene.boundsMax, sizeof(scene.boundsMax)))) {
        return false;
    }
    scene.sourceTextureCount = textureCount;
    scene.sourceMeshCount = meshCount;
    scene.sourceInstanceCount = instanceCount;
    scene.sourceLandscapeCellCount = landscapeCellCount;
    scene.sourceWaterPatchCount = waterPatchCount;
    scene.sourceLightCount = lightCount;
    scene.sourceUnresolvedRefCount = unresolvedRefCount;

    if (!countFitsInStream(input, textureCount, 12u)) {
        return false;
    }
    scene.textures.resize(textureCount);
    for (ImportedSceneTexture& texture : scene.textures) {
        if (!readString(input, texture.sourcePath) ||
            !readValue(input, texture.width) ||
            !readValue(input, texture.height) ||
            !readValue(input, texture.mipLevelCount)) {
            return false;
        }
        if (version >= 17u) {
            std::uint8_t formatValue = 0;
            if (!readValue(input, formatValue) || formatValue > kImportedSceneMaxTextureFormat) {
                setLastImportedSceneError(
                    "Invalid texture format in imported scene file: " + inputPath.string());
                return false;
            }
            texture.format = static_cast<TextureFormat>(formatValue);
        }
        std::uint32_t rgbaSize = 0;
        if (!readValue(input, rgbaSize)) {
            return false;
        }
        if (!countFitsInStream(input, static_cast<std::uint32_t>(rgbaSize), 1u)) {
            return false;
        }
        texture.rgba8.resize(rgbaSize);
        if (rgbaSize != 0 &&
            !readExact(input, texture.rgba8.data(), texture.rgba8.size())) {
            return false;
        }
    }

    for (std::uint32_t i = 0; i < meshCount; ++i) {
        std::uint32_t vertexCount = 0;
        std::uint32_t indexCount = 0;
        std::uint32_t partCount = 0;
        if (!skipString(input) ||
            !readValue(input, vertexCount) ||
            !readValue(input, indexCount) ||
            !readValue(input, partCount)) {
            return false;
        }
        // Must track the on-disk width, not sizeof(ImportedSceneVertex): v19
        // added a colour and v20 the terrain layers, so using the in-memory size
        // here would over-skip on older files and desync every section after.
        const std::size_t vertexBytes =
            static_cast<std::size_t>(vertexCount) * legacyMeshVertexStride(version);
        const std::size_t indexBytes = static_cast<std::size_t>(indexCount) * sizeof(std::uint32_t);
        const std::size_t partBytes = static_cast<std::size_t>(partCount) * sizeof(ImportedSceneMeshPart);
        if ((vertexBytes != 0 && !skipExact(input, vertexBytes)) ||
            (indexBytes != 0 && !skipExact(input, indexBytes)) ||
            (partBytes != 0 && !skipExact(input, partBytes))) {
            return false;
        }
    }

    for (std::uint32_t i = 0; i < instanceCount; ++i) {
        std::uint32_t meshIndex = 0;
        float transform[16] = {};
        if (!readValue(input, meshIndex) ||
            !readExact(input, transform, sizeof(transform)) ||
            !skipString(input) ||
            !skipString(input)) {
            return false;
        }
    }

    for (std::uint32_t i = 0; i < landscapeCellCount; ++i) {
        int gridX = 0;
        int gridY = 0;
        std::uint32_t heightCount = 0;
        std::uint32_t textureIndexCount = 0;
        if (!readValue(input, gridX) ||
            !readValue(input, gridY) ||
            !readValue(input, heightCount) ||
            !readValue(input, textureIndexCount)) {
            return false;
        }
        const std::size_t heightBytes = static_cast<std::size_t>(heightCount) * sizeof(float);
        const std::size_t textureIndexBytes = static_cast<std::size_t>(textureIndexCount) * sizeof(std::uint16_t);
        if ((heightBytes != 0 && !skipExact(input, heightBytes)) ||
            (textureIndexBytes != 0 && !skipExact(input, textureIndexBytes))) {
            return false;
        }
    }

    scene.waterPatches.resize(waterPatchCount);
    if (waterPatchCount != 0 &&
        !readExact(input, scene.waterPatches.data(), scene.waterPatches.size() * sizeof(ImportedSceneWaterPatch))) {
        return false;
    }

    scene.lights.resize(lightCount);
    for (ImportedSceneLight& light : scene.lights) {
        if (!readString(input, light.sourceId) ||
            !readExact(input, light.position, sizeof(light.position)) ||
            !readExact(input, light.color, sizeof(light.color)) ||
            !readValue(input, light.radius) ||
            !readValue(input, light.intensity) ||
            !readValue(input, light.flags)) {
            return false;
        }
        if (version < 16u) {
            const float morrowindY = light.position[1];
            light.position[1] = light.position[2];
            light.position[2] = morrowindY;
        }
    }

    for (std::uint32_t i = 0; i < unresolvedRefCount; ++i) {
        float position[3] = {};
        float rotation[3] = {};
        float scale = 1.0f;
        if (!skipString(input) ||
            !skipString(input) ||
            !readExact(input, position, sizeof(position)) ||
            !readExact(input, rotation, sizeof(rotation)) ||
            !readValue(input, scale)) {
            return false;
        }
    }

    if (!countFitsInStream(input, packedVertexCount, 32u) ||
        !countFitsInStream(input, packedIndexCount, 4u) ||
        !countFitsInStream(input, packedDrawCount, 12u)) {
        return false;
    }
    scene.packedVertices.resize(packedVertexCount);
    scene.packedIndices.resize(packedIndexCount);
    scene.packedDraws.resize(packedDrawCount);
    if (!readPackedVertexArray(input, version, scene.packedVertices)) {
        return false;
    }
    if ((packedIndexCount != 0 &&
         !readExact(input, scene.packedIndices.data(), scene.packedIndices.size() * sizeof(std::uint32_t))) ||
        (packedDrawCount != 0 &&
         !readExact(input, scene.packedDraws.data(), scene.packedDraws.size() * sizeof(ImportedScenePackedDraw)))) {
        return false;
    }
    if (version >= 17u) {
        std::uint32_t pageRangeCount = 0;
        if (!readValue(input, pageRangeCount)) {
            return false;
        }
        scene.pageRanges.resize(pageRangeCount);
        if (pageRangeCount != 0 &&
            !readExact(input, scene.pageRanges.data(), scene.pageRanges.size() * sizeof(ImportedScenePageRange))) {
            return false;
        }
    }
    if (scene.pageRanges.empty() && scene.packedDraws.size() > 1u) {
        buildImportedScenePageRanges(scene);
    }
    // v18 material library -- see the note in loadImportedScene().
    if (version >= 18u && !readSceneMaterials(input, scene.materials)) {
        return false;
    }
    // After materials, matching the write order in saveImportedScene.
    if (version >= 22u && !readSceneDoors(input, scene.doors)) {
        setLastImportedSceneError("Failed to read imported scene doors: " + inputPath.string());
        return false;
    }
    if (version >= 24u) {
        std::uint8_t authored = 0;
        if (!readValue(input, authored)) {
            setLastImportedSceneError(
                "Failed to read imported scene alpha-authored flag: " + inputPath.string());
            return false;
        }
        scene.alphaFlagsAuthored = authored != 0u;
    }
    if (version < 3u) {
        computeImportedSceneBoundsFromPackedData(scene);
    }

    applyTextureAlphaCutoutFlags(scene);
    // Paired with the call above: that one infers a mode where none was
    // authored, this one corrects one that was authored wrong. Exactly one
    // of the two does anything for any given scene (they test
    // alphaFlagsAuthored in opposite senses).
    demoteFalseAlphaBlendFlags(scene);
    outScene = std::move(scene);
    return true;
}

bool exportImportedSceneTerrainObj(const ImportedScene& scene, const std::filesystem::path& outputObjPath) {
    g_lastImportedSceneError.clear();
    if (scene.meshes.empty()) {
        setLastImportedSceneError("Scene does not contain any meshes to export");
        return false;
    }
    const ImportedSceneMesh& mesh = scene.meshes.front();
    const std::filesystem::path parentPath = outputObjPath.parent_path();
    if (!parentPath.empty()) {
        std::error_code mkdirError;
        std::filesystem::create_directories(parentPath, mkdirError);
        if (mkdirError) {
            setLastImportedSceneError(
                "Failed to create OBJ output directory " + parentPath.string() + ": " + mkdirError.message());
            return false;
        }
    }
    std::ofstream output(outputObjPath, std::ios::trunc);
    if (!output) {
        setLastImportedSceneError("Failed to open OBJ output file for writing: " + outputObjPath.string());
        return false;
    }
    output << "o terrain\n";
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        output << "v " << vertex.position[0] << " " << vertex.position[1] << " " << vertex.position[2] << "\n";
    }
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        output << "vn " << vertex.normal[0] << " " << vertex.normal[1] << " " << vertex.normal[2] << "\n";
    }
    for (const ImportedSceneVertex& vertex : mesh.vertices) {
        output << "vt " << vertex.uv[0] << " " << (1.0f - vertex.uv[1]) << "\n";
    }
    for (const ImportedSceneMeshPart& part : mesh.parts) {
        output << "g terrain_part_" << part.textureIndex << "_" << part.firstIndex << "\n";
        for (std::uint32_t i = 0; i + 2u < part.indexCount; i += 3u) {
            const std::uint32_t i0 = mesh.indices[part.firstIndex + i] + 1u;
            const std::uint32_t i1 = mesh.indices[part.firstIndex + i + 1u] + 1u;
            const std::uint32_t i2 = mesh.indices[part.firstIndex + i + 2u] + 1u;
            output << "f "
                   << i0 << "/" << i0 << "/" << i0 << " "
                   << i1 << "/" << i1 << "/" << i1 << " "
                   << i2 << "/" << i2 << "/" << i2 << "\n";
        }
    }
    if (!output.good()) {
        setLastImportedSceneError("Failed while writing OBJ output file: " + outputObjPath.string());
        return false;
    }
    return true;
}


}  // namespace odai::importer
