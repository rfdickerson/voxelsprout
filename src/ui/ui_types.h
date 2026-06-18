#pragma once

#include <cstdint>

// Core UI primitives shared across the UI module. Pure CPU, no Vulkan or
// renderer types: panels, text, widgets, and the draw list all build on these.
// Pixel space has its origin at the top-left with +Y pointing down.
namespace odai::ui {

struct UiVec2 {
    float x = 0.0f;
    float y = 0.0f;

    constexpr UiVec2() = default;
    constexpr UiVec2(float xIn, float yIn) : x(xIn), y(yIn) {}

    constexpr UiVec2 operator+(const UiVec2& rhs) const { return {x + rhs.x, y + rhs.y}; }
    constexpr UiVec2 operator-(const UiVec2& rhs) const { return {x - rhs.x, y - rhs.y}; }
};

struct UiRect {
    float minX = 0.0f;
    float minY = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;

    [[nodiscard]] constexpr float width() const { return maxX - minX; }
    [[nodiscard]] constexpr float height() const { return maxY - minY; }

    [[nodiscard]] constexpr bool contains(float x, float y) const {
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
    }
    [[nodiscard]] constexpr bool contains(const UiVec2& p) const { return contains(p.x, p.y); }

    [[nodiscard]] constexpr bool valid() const { return maxX > minX && maxY > minY; }

    // Largest rect contained in both inputs (empty/invalid if they don't overlap).
    [[nodiscard]] static constexpr UiRect intersect(const UiRect& a, const UiRect& b) {
        return UiRect{
            a.minX > b.minX ? a.minX : b.minX,
            a.minY > b.minY ? a.minY : b.minY,
            a.maxX < b.maxX ? a.maxX : b.maxX,
            a.maxY < b.maxY ? a.maxY : b.maxY,
        };
    }

    [[nodiscard]] static constexpr UiRect fromXYWH(float x, float y, float w, float h) {
        return UiRect{x, y, x + w, y + h};
    }
};

// Straight-alpha RGBA color, 0..1 per channel. Packed for the GPU as ABGR8 to
// match the conventional UI vertex-color layout.
struct UiColor {
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;

    constexpr UiColor() = default;
    constexpr UiColor(float rIn, float gIn, float bIn, float aIn = 1.0f) : r(rIn), g(gIn), b(bIn), a(aIn) {}

    [[nodiscard]] std::uint32_t packAbgr8() const {
        const auto channel = [](float value) -> std::uint32_t {
            const float clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
            return static_cast<std::uint32_t>((clamped * 255.0f) + 0.5f);
        };
        return channel(r) | (channel(g) << 8) | (channel(b) << 16) | (channel(a) << 24);
    }

    // Build from a 0xRRGGBB hex literal (designer-friendly, full opacity).
    [[nodiscard]] static constexpr UiColor fromRgbHex(std::uint32_t rgb, float alpha = 1.0f) {
        return UiColor{
            static_cast<float>((rgb >> 16) & 0xFFu) / 255.0f,
            static_cast<float>((rgb >> 8) & 0xFFu) / 255.0f,
            static_cast<float>(rgb & 0xFFu) / 255.0f,
            alpha,
        };
    }
};

// Identifies a texture/atlas to the renderer without exposing any GPU handle.
// 0 is reserved for the renderer's built-in opaque white 1x1 texture, so solid
// colored quads can share the textured path.
using UiTextureId = std::uint32_t;
constexpr UiTextureId kUiNoTexture = 0;
constexpr UiTextureId kUiFontAtlas = 1;

}  // namespace odai::ui
