#pragma once

#include "import/imported_scene.h"
#include "math/math.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace odai::render {

// Sample the same linear base colour as the imported material shader. Only
// classic DDS colour blocks are needed by TES3; unsupported formats use tint.
inline std::array<float, 3> sampleImportedGiAlbedo(
    const std::vector<importer::ImportedSceneTexture>& textures,
    const importer::ImportedScenePackedVertex& vertex) {
    using importer::TextureFormat;
    std::array<float, 3> color{1, 1, 1};
    if (vertex.textureIndex < textures.size()) {
        const auto& t = textures[vertex.textureIndex];
        if (t.width && t.height && std::isfinite(vertex.uv[0]) && std::isfinite(vertex.uv[1])) {
            const auto x = std::min(static_cast<unsigned>((vertex.uv[0] - std::floor(vertex.uv[0])) * t.width), t.width - 1);
            const auto y = std::min(static_cast<unsigned>((vertex.uv[1] - std::floor(vertex.uv[1])) * t.height), t.height - 1);
            bool sampled = false;
            if (t.format == TextureFormat::RGBA8 || t.format == TextureFormat::RGBA8Srgb) {
                const auto offset = (std::size_t(y) * t.width + x) * 4;
                if (offset + 4 <= t.rgba8.size()) {
                    for (unsigned c = 0; c < 3; ++c) color[c] = t.rgba8[offset + c] / 255.0f;
                    sampled = true;
                }
            } else if (t.format == TextureFormat::BC1 || t.format == TextureFormat::BC1Linear ||
                       t.format == TextureFormat::BC2 || t.format == TextureFormat::BC3) {
                const bool bc1 = t.format == TextureFormat::BC1 || t.format == TextureFormat::BC1Linear;
                const unsigned blockBytes = bc1 ? 8 : 16;
                const auto offset = (std::size_t(y / 4) * ((t.width + 3) / 4) + x / 4) * blockBytes;
                if (offset + blockBytes <= t.rgba8.size()) {
                    const auto* b = t.rgba8.data() + offset + (bc1 ? 0 : 8);
                    const unsigned e0 = b[0] | (unsigned(b[1]) << 8);
                    const unsigned e1 = b[2] | (unsigned(b[3]) << 8);
                    const auto rgb565 = [](unsigned e) {
                        return std::array<float, 3>{float((e >> 11) & 31) / 31,
                            float((e >> 5) & 63) / 63, float(e & 31) / 31};
                    };
                    const auto a = rgb565(e0), z = rgb565(e1);
                    const unsigned index = (b[4 + y % 4] >> (2 * (x % 4))) & 3;
                    for (unsigned c = 0; c < 3; ++c) {
                        color[c] = index == 0 ? a[c] : index == 1 ? z[c] :
                            (bc1 && e0 <= e1) ? (index == 2 ? (a[c] + z[c]) * 0.5f : 0.0f) :
                            (index == 2 ? (2 * a[c] + z[c]) / 3 : (a[c] + 2 * z[c]) / 3);
                    }
                    sampled = true;
                }
            }
            if (sampled && t.format != TextureFormat::RGBA8 && t.format != TextureFormat::BC1Linear) {
                for (float& c : color) c = c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
            }
        }
    }
    if (vertex.flags & importer::kImportedSceneMaterialFlagVertexColorTint) {
        for (unsigned c = 0; c < 3; ++c) color[c] *= vertex.color[c];
    }
    return color;
}

// Conservative triangle/box SAT. Unlike filling a triangle's AABB this keeps
// diagonal walls thin, and unlike sparse point sampling it keeps large walls solid.
inline bool importedGiTriangleOverlapsCell(const float p0[3], const float p1[3],
    const float p2[3], math::Vector3 center, float halfSize) {
    const math::Vector3 v[3] = {
        {p0[0] - center.x, p0[1] - center.y, p0[2] - center.z},
        {p1[0] - center.x, p1[1] - center.y, p1[2] - center.z},
        {p2[0] - center.x, p2[1] - center.y, p2[2] - center.z}};
    const auto separated = [&](math::Vector3 axis) {
        const float a = math::dot(v[0], axis), b = math::dot(v[1], axis), c = math::dot(v[2], axis);
        const float radius = halfSize * (std::abs(axis.x) + std::abs(axis.y) + std::abs(axis.z));
        return std::min({a, b, c}) > radius || std::max({a, b, c}) < -radius;
    };
    const math::Vector3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const math::Vector3 edges[3] = {v[1] - v[0], v[2] - v[1], v[0] - v[2]};
    for (auto axis : axes) {
        if (separated(axis)) return false;
        for (auto edge : edges) if (separated(math::cross(edge, axis))) return false;
    }
    return !separated(math::cross(edges[0], edges[1]));
}

} // namespace odai::render
