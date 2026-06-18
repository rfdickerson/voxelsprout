#include "ui/ui_draw_list.h"

#include "ui/font.h"
#include "ui/ui_text_util.h"

#include <cmath>

namespace odai::ui {

namespace {

bool rectsEqual(const UiRect& a, const UiRect& b) {
    return a.minX == b.minX && a.minY == b.minY && a.maxX == b.maxX && a.maxY == b.maxY;
}

}  // namespace

void UiDrawList::reset(const UiVec2& framebufferSizePx) {
    m_data.vertices.clear();
    m_data.indices.clear();
    m_data.commands.clear();
    m_data.framebufferSizePx = framebufferSizePx;
    m_clipStack.clear();
    m_clipStack.push_back(UiRect{0.0f, 0.0f, framebufferSizePx.x, framebufferSizePx.y});
    m_opacityStack.clear();
}

UiDrawCmd& UiDrawList::currentCommand(UiTextureId textureId) {
    // reset() always seeds the clip stack; fall back to the framebuffer rect if a
    // caller emits geometry before reset().
    const UiRect fallback{0.0f, 0.0f, m_data.framebufferSizePx.x, m_data.framebufferSizePx.y};
    const UiRect& clip = m_clipStack.empty() ? fallback : m_clipStack.back();
    if (!m_data.commands.empty()) {
        UiDrawCmd& last = m_data.commands.back();
        if (last.textureId == textureId && rectsEqual(last.clipRect, clip)) {
            return last;
        }
    }
    UiDrawCmd cmd{};
    cmd.indexOffset = static_cast<std::uint32_t>(m_data.indices.size());
    cmd.indexCount = 0;
    cmd.textureId = textureId;
    cmd.clipRect = clip;
    m_data.commands.push_back(cmd);
    return m_data.commands.back();
}

namespace {

// Scale the packed-ABGR8 alpha byte by `opacity` (1.0 = unchanged).
std::uint32_t scaleAlpha(std::uint32_t rgba8, float opacity) {
    if (opacity >= 1.0f) {
        return rgba8;
    }
    const std::uint32_t a = (rgba8 >> 24) & 0xFFu;
    const float clamped = opacity < 0.0f ? 0.0f : opacity;
    const auto na = static_cast<std::uint32_t>(static_cast<float>(a) * clamped + 0.5f);
    return (rgba8 & 0x00FFFFFFu) | (na << 24);
}

}  // namespace

void UiDrawList::addQuad(const UiRect& dst, const UiRect& uv, std::uint32_t rgba8, UiDrawMode mode,
                         UiTextureId textureId) {
    rgba8 = scaleAlpha(rgba8, currentOpacity());
    UiDrawCmd& cmd = currentCommand(textureId);
    const auto base = static_cast<std::uint32_t>(m_data.vertices.size());
    const auto modeBits = static_cast<std::uint32_t>(mode);

    const UiVertex v00{{dst.minX, dst.minY}, {uv.minX, uv.minY}, rgba8, modeBits};
    const UiVertex v10{{dst.maxX, dst.minY}, {uv.maxX, uv.minY}, rgba8, modeBits};
    const UiVertex v11{{dst.maxX, dst.maxY}, {uv.maxX, uv.maxY}, rgba8, modeBits};
    const UiVertex v01{{dst.minX, dst.maxY}, {uv.minX, uv.maxY}, rgba8, modeBits};
    m_data.vertices.push_back(v00);
    m_data.vertices.push_back(v10);
    m_data.vertices.push_back(v11);
    m_data.vertices.push_back(v01);

    m_data.indices.push_back(base + 0);
    m_data.indices.push_back(base + 1);
    m_data.indices.push_back(base + 2);
    m_data.indices.push_back(base + 0);
    m_data.indices.push_back(base + 2);
    m_data.indices.push_back(base + 3);
    cmd.indexCount += 6;
}

void UiDrawList::appendCached(const UiGeometryBlock& block, const UiVec2& translate) {
    if (block.vertices.empty() || block.commands.empty()) {
        return;
    }
    const auto base = static_cast<std::uint32_t>(m_data.vertices.size());
    const float opacity = currentOpacity();
    m_data.vertices.reserve(m_data.vertices.size() + block.vertices.size());
    for (const UiVertex& v : block.vertices) {
        UiVertex out = v;
        out.posPx[0] += translate.x;
        out.posPx[1] += translate.y;
        out.rgba8 = scaleAlpha(out.rgba8, opacity);
        m_data.vertices.push_back(out);
    }
    // Re-emit each block command under the live clip. currentCommand only ever
    // grows the last command, so appending indices immediately keeps each
    // command's index range contiguous (same invariant as addQuad).
    for (const UiDrawCmd& blockCmd : block.commands) {
        if (blockCmd.indexCount == 0) {
            continue;
        }
        UiDrawCmd& dst = currentCommand(blockCmd.textureId);
        m_data.indices.reserve(m_data.indices.size() + blockCmd.indexCount);
        for (std::uint32_t k = 0; k < blockCmd.indexCount; ++k) {
            m_data.indices.push_back(base + block.indices[blockCmd.indexOffset + k]);
        }
        dst.indexCount += blockCmd.indexCount;
    }
}

void UiDrawList::addRectFilled(const UiRect& rect, const UiColor& color) {
    if (!rect.valid()) {
        return;
    }
    addQuad(rect, UiRect{0.0f, 0.0f, 0.0f, 0.0f}, color.packAbgr8(), UiDrawMode::SolidColor, kUiNoTexture);
}

void UiDrawList::addRect(const UiRect& rect, const UiColor& color, float thicknessPx) {
    if (thicknessPx <= 0.0f) {
        return;
    }
    const float t = thicknessPx;
    // Four edge bars: top, bottom, left, right (corners covered by top/bottom).
    addRectFilled(UiRect{rect.minX, rect.minY, rect.maxX, rect.minY + t}, color);
    addRectFilled(UiRect{rect.minX, rect.maxY - t, rect.maxX, rect.maxY}, color);
    addRectFilled(UiRect{rect.minX, rect.minY + t, rect.minX + t, rect.maxY - t}, color);
    addRectFilled(UiRect{rect.maxX - t, rect.minY + t, rect.maxX, rect.maxY - t}, color);
}

void UiDrawList::addImage(const UiRect& rect, UiTextureId textureId, const UiColor& tint, const UiRect& uv) {
    if (!rect.valid()) {
        return;
    }
    addQuad(rect, uv, tint.packAbgr8(), UiDrawMode::Textured, textureId);
}

void UiDrawList::add9Slice(const UiRect& rect, const UiNineSlice& slice, const UiColor& color) {
    if (!rect.valid()) {
        return;
    }
    const std::uint32_t rgba = color.packAbgr8();
    const auto mode = UiDrawMode::Textured;

    // Destination column / row boundaries.
    const float dx0 = rect.minX;
    const float dx1 = rect.minX + slice.borderLeftPx;
    const float dx2 = rect.maxX - slice.borderRightPx;
    const float dx3 = rect.maxX;
    const float dy0 = rect.minY;
    const float dy1 = rect.minY + slice.borderTopPx;
    const float dy2 = rect.maxY - slice.borderBottomPx;
    const float dy3 = rect.maxY;

    // Source UV boundaries within the slice's sub-rect.
    const float uvW = slice.uv.width();
    const float uvH = slice.uv.height();
    const float ux0 = slice.uv.minX;
    const float ux1 = slice.uv.minX + (slice.uvBorderLeft * uvW);
    const float ux2 = slice.uv.maxX - (slice.uvBorderRight * uvW);
    const float ux3 = slice.uv.maxX;
    const float uy0 = slice.uv.minY;
    const float uy1 = slice.uv.minY + (slice.uvBorderTop * uvH);
    const float uy2 = slice.uv.maxY - (slice.uvBorderBottom * uvH);
    const float uy3 = slice.uv.maxY;

    const float dxs[4] = {dx0, dx1, dx2, dx3};
    const float dys[4] = {dy0, dy1, dy2, dy3};
    const float uxs[4] = {ux0, ux1, ux2, ux3};
    const float uys[4] = {uy0, uy1, uy2, uy3};

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const UiRect dst{dxs[col], dys[row], dxs[col + 1], dys[row + 1]};
            if (dst.maxX <= dst.minX || dst.maxY <= dst.minY) {
                continue;
            }
            const UiRect uv{uxs[col], uys[row], uxs[col + 1], uys[row + 1]};
            addQuad(dst, uv, rgba, mode, slice.textureId);
        }
    }
}

float UiDrawList::addText(const Font& font, std::string_view utf8, const UiVec2& posPx, const UiColor& color) {
    const std::uint32_t rgba = color.packAbgr8();
    float penX = posPx.x;
    // Snap the baseline to an integer pixel row so horizontal strokes align to
    // the grid (crisper text). X stays fractional so the atlas's horizontal
    // oversampling still reconstructs sub-pixel glyph placement.
    const float baselineY = std::round(posPx.y + font.ascentPx());
    for (const ShapedGlyph& sg : font.shape(utf8)) {
        const Glyph& g = font.glyph(sg.codepoint);
        if (g.size.x > 0.0f && g.size.y > 0.0f) {
            const UiRect dst{
                penX + g.bearing.x,
                baselineY - g.bearing.y,
                penX + g.bearing.x + g.size.x,
                baselineY - g.bearing.y + g.size.y,
            };
            addQuad(dst, g.uv, rgba, UiDrawMode::GlyphAlpha, font.textureId());
        }
        penX += g.advance + sg.kern;
    }
    return penX - posPx.x;
}

void UiDrawList::pushClip(const UiRect& rect) {
    const UiRect parent = m_clipStack.empty() ? rect : m_clipStack.back();
    m_clipStack.push_back(UiRect::intersect(parent, rect));
}

void UiDrawList::popClip() {
    if (m_clipStack.size() > 1) {
        m_clipStack.pop_back();
    }
}

void UiDrawList::pushOpacity(float opacity) {
    m_opacityStack.push_back(currentOpacity() * opacity);
}

void UiDrawList::popOpacity() {
    if (!m_opacityStack.empty()) {
        m_opacityStack.pop_back();
    }
}

float UiDrawList::currentOpacity() const {
    return m_opacityStack.empty() ? 1.0f : m_opacityStack.back();
}

}  // namespace odai::ui
