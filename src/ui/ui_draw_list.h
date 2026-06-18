#pragma once

#include "ui/ui_types.h"

#include <cstdint>
#include <string_view>
#include <vector>

// Immediate-mode draw list: the renderer-agnostic core of the UI module. Widgets
// and the HUD append primitives here; the result (UiDrawData) is handed to the
// Vulkan UiRenderer. No Vulkan, no GPU handles -- textures are referenced by id.
namespace odai::ui {

class Font;

// Per-vertex draw mode. Texture binding is per-command (UiDrawCmd::textureId),
// but the shading mode is per-vertex so one command may mix solid fills and
// glyph quads that share a texture binding and clip rect.
enum class UiDrawMode : std::uint32_t {
    SolidColor = 0,  // Use vertex color, ignore texture.
    Textured = 1,    // Sample rgba texture, multiply by vertex color.
    GlyphAlpha = 2,  // Sample R8 atlas as coverage; rgb from vertex color.
};

struct UiVertex {            // 24 bytes; matches the renderer's vertex input layout.
    float posPx[2] = {};      // Pixel space, top-left origin, +Y down. offset 0
    float uv[2] = {};         // 0..1 atlas coordinates.                offset 8
    std::uint32_t rgba8 = 0;  // Packed ABGR8 vertex color.             offset 16
    std::uint32_t mode = 0;   // UiDrawMode.                            offset 20
};

struct UiDrawCmd {
    std::uint32_t indexOffset = 0;       // First index into UiDrawData::indices.
    std::uint32_t indexCount = 0;        // Number of indices to draw.
    UiTextureId textureId = kUiNoTexture;
    UiRect clipRect{};                   // Pixel-space scissor.
};

struct UiDrawData {
    std::vector<UiVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<UiDrawCmd> commands;
    UiVec2 framebufferSizePx{};

    [[nodiscard]] bool empty() const { return commands.empty(); }
};

// A 9-slice (scalable border) source description. Corner insets are given both
// in destination pixels (how big the corners are drawn) and in UV fractions of
// the source sub-rect (which texels map to the corners).
struct UiNineSlice {
    UiTextureId textureId = kUiNoTexture;
    UiRect uv{0.0f, 0.0f, 1.0f, 1.0f};
    float borderLeftPx = 0.0f;
    float borderRightPx = 0.0f;
    float borderTopPx = 0.0f;
    float borderBottomPx = 0.0f;
    float uvBorderLeft = 0.0f;
    float uvBorderRight = 0.0f;
    float uvBorderTop = 0.0f;
    float uvBorderBottom = 0.0f;

    // Equal borders sourced from a square texture: corner is borderPx wide, the
    // texture is texSizePx on a side, full UV range used.
    [[nodiscard]] static UiNineSlice uniform(UiTextureId id, float borderPx, float texSizePx) {
        const float uvBorder = texSizePx > 0.0f ? (borderPx / texSizePx) : 0.0f;
        return UiNineSlice{id, UiRect{0.0f, 0.0f, 1.0f, 1.0f},
                           borderPx, borderPx, borderPx, borderPx,
                           uvBorder, uvBorder, uvBorder, uvBorder};
    }
};

enum class UiTextAlign : std::uint8_t { Left = 0, Center = 1, Right = 2 };

class UiDrawList {
public:
    // Start a new frame; clears geometry and resets the clip stack to the full
    // framebuffer.
    void reset(const UiVec2& framebufferSizePx);

    void addRectFilled(const UiRect& rect, const UiColor& color);
    void addRect(const UiRect& rect, const UiColor& color, float thicknessPx = 1.0f);
    void addImage(const UiRect& rect, UiTextureId textureId, const UiColor& tint = {},
                  const UiRect& uv = UiRect{0.0f, 0.0f, 1.0f, 1.0f});
    void add9Slice(const UiRect& rect, const UiNineSlice& slice, const UiColor& color = {});

    // Draw a single line of text with the top-left baseline box at posPx; returns
    // the pen x advance. Newlines are not interpreted (use RichText for layout).
    float addText(const Font& font, std::string_view utf8, const UiVec2& posPx, const UiColor& color);

    void pushClip(const UiRect& rect);
    void popClip();
    [[nodiscard]] const UiRect& currentClip() const { return m_clipStack.back(); }

    [[nodiscard]] const UiDrawData& data() const { return m_data; }
    [[nodiscard]] UiDrawData& data() { return m_data; }

    // Low-level quad emission, used by addText/RichText. Vertices are p00 (min),
    // p10, p11, p01 with matching UVs; emitted as two triangles.
    void addQuad(const UiRect& dst, const UiRect& uv, std::uint32_t rgba8, UiDrawMode mode,
                 UiTextureId textureId);

private:
    // Ensure the current command matches (textureId, current clip); start a new
    // one otherwise. Returns the index base for appended geometry.
    UiDrawCmd& currentCommand(UiTextureId textureId);

    UiDrawData m_data;
    std::vector<UiRect> m_clipStack;
};

}  // namespace odai::ui
