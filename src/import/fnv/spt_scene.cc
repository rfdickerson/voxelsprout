#include "import/fnv/spt_scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace odai::importer::fnv {
namespace {

constexpr float kPi = 3.14159265358979323846f;

std::string lowerAscii(std::string value) {
    for (char& c : value) {
        if (c == '/') c = '\\';
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return value;
}

std::string fileStem(std::string_view path) {
    const std::size_t slash = path.find_last_of("\\/");
    std::string name(path.substr(slash == std::string_view::npos ? 0u : slash + 1u));
    const std::size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name.resize(dot);
    return lowerAscii(std::move(name));
}

std::vector<std::string> asciiStrings(const std::vector<std::uint8_t>& bytes) {
    std::vector<std::string> result;
    std::size_t begin = 0u;
    while (begin < bytes.size()) {
        while (begin < bytes.size() && (bytes[begin] < 0x20u || bytes[begin] > 0x7eu)) ++begin;
        std::size_t end = begin;
        while (end < bytes.size() && bytes[end] >= 0x20u && bytes[end] <= 0x7eu) ++end;
        if (end - begin >= 4u) {
            result.emplace_back(reinterpret_cast<const char*>(bytes.data() + begin), end - begin);
        }
        begin = end + 1u;
    }
    return result;
}

struct Rng {
    std::uint32_t state;
    std::uint32_t next() {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    }
    float unit() { return static_cast<float>(next() & 0x00ffffffu) / 16777215.0f; }
};

void appendCylinder(
    NifShape& shape, const std::array<float, 3>& a, const std::array<float, 3>& b,
    float radiusA, float radiusB, int sides, float windA, float windB, float phase) {
    const std::array<float, 3> axis{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const float length = std::sqrt(
        axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (length <= 1e-4f) return;
    const std::array<float, 3> n{
        axis[0] / length, axis[1] / length, axis[2] / length};
    const std::array<float, 3> helper = std::abs(n[2]) < 0.9f
        ? std::array<float, 3>{0.0f, 0.0f, 1.0f}
        : std::array<float, 3>{1.0f, 0.0f, 0.0f};
    std::array<float, 3> u{
        n[1] * helper[2] - n[2] * helper[1],
        n[2] * helper[0] - n[0] * helper[2],
        n[0] * helper[1] - n[1] * helper[0]};
    const float uLength = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    for (float& component : u) component /= std::max(uLength, 1e-5f);
    const std::array<float, 3> v{
        n[1] * u[2] - n[2] * u[1],
        n[2] * u[0] - n[0] * u[2],
        n[0] * u[1] - n[1] * u[0]};
    const std::uint32_t base = static_cast<std::uint32_t>(shape.positions.size() / 3u);
    for (int ring = 0; ring < 2; ++ring) {
        const auto& center = ring == 0 ? a : b;
        const float radius = ring == 0 ? radiusA : radiusB;
        for (int side = 0; side < sides; ++side) {
            const float angle = 2.0f * kPi * static_cast<float>(side) /
                static_cast<float>(sides);
            const float ca = std::cos(angle);
            const float sa = std::sin(angle);
            const std::array<float, 3> radial{
                u[0] * ca + v[0] * sa,
                u[1] * ca + v[1] * sa,
                u[2] * ca + v[2] * sa};
            for (int component = 0; component < 3; ++component) {
                shape.positions.push_back(center[component] + radial[component] * radius);
                shape.normals.push_back(radial[component]);
            }
            shape.uvs.push_back(static_cast<float>(side) / static_cast<float>(sides));
            shape.uvs.push_back(static_cast<float>(ring));
            shape.windWeights.push_back(ring == 0 ? windA : windB);
            shape.windPhases.push_back(phase);
        }
    }
    for (int side = 0; side < sides; ++side) {
        const std::uint32_t next = static_cast<std::uint32_t>((side + 1) % sides);
        const std::uint32_t lower = base + static_cast<std::uint32_t>(side);
        const std::uint32_t lowerNext = base + next;
        const std::uint32_t upperNext = base + static_cast<std::uint32_t>(sides) + next;
        const std::uint32_t upper =
            base + static_cast<std::uint32_t>(sides) + static_cast<std::uint32_t>(side);
        shape.triangleIndices.insert(
            shape.triangleIndices.end(),
            {lower, lowerNext, upperNext, lower, upperNext, upper});
    }
}

void appendLeafCard(
    NifShape& shape, const std::array<float, 3>& center, float width, float height,
    float angle, float phase,
    const std::array<float, 4>& uvRect = {0.0f, 0.0f, 1.0f, 1.0f},
    float tilt = 0.0f) {
    const float ca = std::cos(angle);
    const float sa = std::sin(angle);
    const std::array<float, 3> right{ca * width * 0.5f, sa * width * 0.5f, 0.0f};
    const float st = std::sin(tilt);
    const float ct = std::cos(tilt);
    const std::array<float, 3> up{
        -sa * st * height * 0.5f,
        ca * st * height * 0.5f,
        ct * height * 0.5f};
    const std::uint32_t base = static_cast<std::uint32_t>(shape.positions.size() / 3u);
    for (const auto& signs : std::array<std::array<float, 2>, 4>{
             std::array<float, 2>{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}) {
        shape.positions.insert(shape.positions.end(), {
            center[0] + right[0] * signs[0] + up[0] * signs[1],
            center[1] + right[1] * signs[0] + up[1] * signs[1],
            center[2] + right[2] * signs[0] + up[2] * signs[1]});
        shape.normals.insert(shape.normals.end(), {-sa * ct, ca * ct, st});
        shape.uvs.insert(shape.uvs.end(), {
            signs[0] < 0.0f ? uvRect[0] : uvRect[2],
            signs[1] < 0.0f ? uvRect[3] : uvRect[1]});
        shape.windWeights.push_back(1.0f);
        shape.windPhases.push_back(phase);
    }
    shape.triangleIndices.insert(shape.triangleIndices.end(), {
        base, base + 1u, base + 2u, base, base + 2u, base + 3u});
}

} // namespace

bool parseOblivionSpt(
    const std::vector<std::uint8_t>& bytes, std::string_view sourcePath,
    std::string_view leafTexturePath, std::uint32_t seed,
    float billboardWidth, float billboardHeight, const float wind[8],
    OblivionSptTree& outTree, std::string& outError) {
    outTree = {};
    outError.clear();
    constexpr std::string_view magic = "__IdvSpt_02_";
    // Retail Oblivion wraps every tagged value as [type/version u32][length
    // u32][payload].  The first payload is the 12-byte SpeedTree signature;
    // keeping offset-zero support makes standalone clean-room fixtures useful.
    const bool bareHeader = bytes.size() >= magic.size() &&
        std::memcmp(bytes.data(), magic.data(), magic.size()) == 0;
    std::uint32_t firstTagLength = 0u;
    if (bytes.size() >= 8u) {
        std::memcpy(&firstTagLength, bytes.data() + 4u, sizeof(firstTagLength));
    }
    const bool taggedHeader = bytes.size() >= 8u + magic.size() &&
        firstTagLength == magic.size() &&
        std::memcmp(bytes.data() + 8u, magic.data(), magic.size()) == 0;
    if (!bareHeader && !taggedHeader) {
        char prefix[3u * 12u + 1u] = {};
        const std::size_t prefixBytes = std::min<std::size_t>(12u, bytes.size());
        for (std::size_t i = 0; i < prefixBytes; ++i) {
            std::snprintf(prefix + (i * 3u), sizeof(prefix) - (i * 3u), "%02x ", bytes[i]);
        }
        outError = "SPT is missing __IdvSpt_02_ header (first bytes: " +
            std::string(prefix) + ")";
        return false;
    }
    if (bytes.size() > 16u * 1024u * 1024u) {
        outError = "SPT exceeds the bounded 16 MiB source limit";
        return false;
    }
    outTree.sourcePath = std::string(sourcePath);
    outTree.leafTexturePath = std::string(leafTexturePath);
    outTree.seed = seed;
    outTree.billboardWidth = billboardWidth;
    outTree.billboardHeight = billboardHeight;
    std::copy_n(wind, 8u, outTree.wind);

    const std::vector<std::string> strings = asciiStrings(bytes);
    for (const std::string& raw : strings) {
        const std::string value = lowerAscii(raw);
        if (value.find("bezierspline") != std::string::npos) ++outTree.splineTokenCount;
        if (value.find("begin") != std::string::npos || value.find("end") != std::string::npos) {
            ++outTree.taggedSectionCount;
        }
        const std::size_t dds = value.find(".dds");
        if (dds == std::string::npos) continue;
        std::string texture = value.substr(0u, dds + 4u);
        const std::size_t nonPath = texture.find_last_of(" \t\r\n=\"");
        if (nonPath != std::string::npos) texture = texture.substr(nonPath + 1u);
        if (texture.find("bark") != std::string::npos) {
            const bool candidateNormal = texture.ends_with("_n.dds");
            const bool currentNormal = outTree.barkTexturePath.ends_with("_n.dds");
            if (outTree.barkTexturePath.empty() || (currentNormal && !candidateNormal)) {
                outTree.barkTexturePath = texture;
            }
        } else if (outTree.leafTexturePath.empty()) {
            outTree.leafTexturePath = texture;
        }
    }
    if (outTree.leafTexturePath.empty()) {
        outError = "SPT has no leaf texture and TREE ICON is empty";
        return false;
    }
    const std::string stem = fileStem(sourcePath);
    outTree.billboardTexturePath = "trees\\billboards\\" + stem + ".dds";
    if (outTree.billboardHeight <= 0.0f || !std::isfinite(outTree.billboardHeight)) {
        outTree.billboardHeight = 300.0f;
    }
    if (outTree.billboardWidth <= 0.0f || !std::isfinite(outTree.billboardWidth)) {
        outTree.billboardWidth = outTree.billboardHeight;
    }
    if (outTree.barkTexturePath.empty()) {
        // Retail SPT variants occasionally carry the bark filename in a binary
        // string block the conservative scanner cannot identify. A missing bark
        // is diagnosable, but leaves still produce a complete silhouette.
        outTree.barkTexturePath = outTree.leafTexturePath;
    }
    return true;
}

bool buildOblivionSptModel(
    const OblivionSptTree& tree, NifModel& outModel, std::string& outError) {
    outModel = {};
    outError.clear();
    if (tree.billboardHeight <= 0.0f || tree.billboardWidth <= 0.0f) {
        outError = "SPT billboard bounds are not positive";
        return false;
    }
    Rng rng{tree.seed != 0u ? tree.seed : 0x6d2b79f5u};
    const float height = tree.billboardHeight;
    const std::array<float, 3> billboardCenter{0.0f, 0.0f, height * 0.5f};
    const auto makeRetailBillboard = [&](std::uint8_t lod, int cardCount) {
        NifShape shape;
        shape.name = lod == 1u ? "SpeedTree retail high silhouette" :
            (lod == 2u ? "SpeedTree retail reduced silhouette" :
                         "SpeedTree retail billboard");
        shape.sourceBlockType = "SPTBillboardGeometry";
        shape.diffuseTexturePath = tree.billboardTexturePath;
        shape.alphaTest = true;
        // Retail billboards reserve most of their BC3 page as transparent
        // padding. A half-coverage cutout keeps the antialiased crown while
        // preventing that padding from becoming a dark crossed-card slab.
        shape.alphaThreshold = 128u;
        shape.twoSided = true;
        shape.alphaSemantic = NifAlphaSemantic::Cutout;
        shape.vegetationLod = lod;
        const float baseAngle = rng.unit() * kPi;
        for (int card = 0; card < cardCount; ++card) {
            appendLeafCard(
                shape, billboardCenter, tree.billboardWidth, height,
                baseAngle + kPi * static_cast<float>(card) /
                    static_cast<float>(cardCount),
                rng.unit());
        }
        shape.sourceTriangleCount =
            static_cast<std::uint32_t>(shape.triangleIndices.size() / 3u);
        return shape;
    };

    NifShape highWood;
    highWood.name = "SpeedTree volumetric high wood";
    highWood.sourceBlockType = "SPTBranchGeometry";
    highWood.diffuseTexturePath = tree.barkTexturePath;
    highWood.vegetationLod = 1u;
    const float crownRadius = std::clamp(
        tree.billboardWidth * 0.46f, height * 0.18f, height * 0.48f);
    const float trunkPhase = rng.unit();
    appendCylinder(
        highWood, {0.0f, 0.0f, 0.0f},
        {height * 0.012f, -height * 0.008f, height * 0.68f},
        height * 0.052f, height * 0.014f, 10, 0.0f, 0.35f, trunkPhase);

    struct CrownLobe {
        std::array<float, 3> center;
        float angle;
        float phase;
        int atlasRegion;
    };
    std::vector<CrownLobe> lobes;
    lobes.reserve(64u);
    const std::string lowerSource = lowerAscii(tree.sourcePath);
    const bool conifer = lowerSource.find("pine") != std::string::npos ||
        lowerSource.find("hemlock") != std::string::npos ||
        lowerSource.find("juniper") != std::string::npos ||
        lowerSource.find("deodar") != std::string::npos ||
        lowerSource.find("redwood") != std::string::npos ||
        lowerSource.find("cypress") != std::string::npos;
    constexpr int lobeCount = 64;
    for (int lobe = 0; lobe < lobeCount; ++lobe) {
        const float zUnit = rng.unit();
        const float z = height * (conifer ? (0.24f + zUnit * 0.72f)
                                         : (0.38f + zUnit * 0.55f));
        const float verticalProfile = conifer
            ? std::clamp(1.15f - zUnit, 0.18f, 1.0f)
            : std::sqrt(std::max(
                  0.0f, 1.0f - std::pow((zUnit - 0.52f) / 0.58f, 2.0f)));
        const float radial = std::sqrt(rng.unit()) * crownRadius * verticalProfile;
        const float polar = rng.unit() * 2.0f * kPi;
        lobes.push_back({
            {std::cos(polar) * radial, std::sin(polar) * radial, z},
            rng.unit() * kPi, rng.unit(), lobe % 3});
    }
    highWood.sourceTriangleCount =
        static_cast<std::uint32_t>(highWood.triangleIndices.size() / 3u);

    NifShape highLeaves;
    highLeaves.name = "SpeedTree volumetric high leaves";
    highLeaves.sourceBlockType = "SPTLeafGeometry";
    highLeaves.diffuseTexturePath = tree.leafTexturePath;
    highLeaves.alphaTest = true;
    highLeaves.alphaThreshold = 96u;
    highLeaves.twoSided = true;
    highLeaves.alphaSemantic = NifAlphaSemantic::Cutout;
    highLeaves.vegetationLod = 1u;
    constexpr std::array<std::array<float, 4>, 3> leafAtlasRegions{{
        {0.0f, 0.0f, 0.5f, 0.5f},
        {0.5f, 0.0f, 1.0f, 0.5f},
        {0.0f, 0.5f, 1.0f, 1.0f}}};
    for (const CrownLobe& lobe : lobes) {
        const bool largeCluster = lobe.atlasRegion == 2;
        const float cardWidth = crownRadius * (largeCluster ? 0.43f : 0.30f);
        const float cardHeight = height * (largeCluster ? 0.14f : 0.10f);
        const auto& uv = leafAtlasRegions[static_cast<std::size_t>(lobe.atlasRegion)];
        for (int plane = 0; plane < 3; ++plane) {
            appendLeafCard(
                highLeaves, lobe.center, cardWidth, cardHeight,
                lobe.angle + kPi * static_cast<float>(plane) / 3.0f,
                lobe.phase, uv,
                (rng.unit() - 0.5f) * 1.15f);
        }
    }
    highLeaves.sourceTriangleCount =
        static_cast<std::uint32_t>(highLeaves.triangleIndices.size() / 3u);

    // Only the near LOD is reconstructed as volume. At medium and far ranges,
    // Oblivion's complete per-species billboard remains both more faithful and
    // substantially cheaper.
    outModel.shapes.push_back(std::move(highWood));
    outModel.shapes.push_back(std::move(highLeaves));
    outModel.shapes.push_back(makeRetailBillboard(2u, 2));
    outModel.shapes.push_back(makeRetailBillboard(3u, 2));
    return true;
}

} // namespace odai::importer::fnv
