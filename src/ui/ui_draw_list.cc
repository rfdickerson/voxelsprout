#include "ui/ui_draw_list.h"

#include "ui/font.h"
#include "ui/ui_text_util.h"

#include <algorithm>
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

UiDrawCmd& UiDrawList::currentCommand() {
    // reset() always seeds the clip stack; fall back to the framebuffer rect if a
    // caller emits geometry before reset().
    const UiRect fallback{0.0f, 0.0f, m_data.framebufferSizePx.x, m_data.framebufferSizePx.y};
    const UiRect& clip = m_clipStack.empty() ? fallback : m_clipStack.back();
    if (!m_data.commands.empty()) {
        UiDrawCmd& last = m_data.commands.back();
        if (rectsEqual(last.clipRect, clip)) {
            return last;
        }
    }
    UiDrawCmd cmd{};
    cmd.indexOffset = static_cast<std::uint32_t>(m_data.indices.size());
    cmd.indexCount = 0;
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
                         UiTextureId textureId, const float sdf[4]) {
    rgba8 = scaleAlpha(rgba8, currentOpacity());
    UiDrawCmd& cmd = currentCommand();
    const auto base = static_cast<std::uint32_t>(m_data.vertices.size());
    const auto modeBits = static_cast<std::uint32_t>(mode) | (textureId << 8u);
    const float s0 = sdf ? sdf[0] : 0.0f;
    const float s1 = sdf ? sdf[1] : 0.0f;
    const float s2 = sdf ? sdf[2] : 0.0f;
    const float s3 = sdf ? sdf[3] : 0.0f;

    const UiVertex v00{{dst.minX, dst.minY}, {uv.minX, uv.minY}, rgba8, modeBits, {s0, s1, s2, s3}};
    const UiVertex v10{{dst.maxX, dst.minY}, {uv.maxX, uv.minY}, rgba8, modeBits, {s0, s1, s2, s3}};
    const UiVertex v11{{dst.maxX, dst.maxY}, {uv.maxX, uv.maxY}, rgba8, modeBits, {s0, s1, s2, s3}};
    const UiVertex v01{{dst.minX, dst.maxY}, {uv.minX, uv.maxY}, rgba8, modeBits, {s0, s1, s2, s3}};
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
        UiDrawCmd& dst = currentCommand();
        m_data.indices.reserve(m_data.indices.size() + blockCmd.indexCount);
        for (std::uint32_t k = 0; k < blockCmd.indexCount; ++k) {
            m_data.indices.push_back(base + block.indices[blockCmd.indexOffset + k]);
        }
        dst.indexCount += blockCmd.indexCount;
    }
}

void UiDrawList::appendCachedClipped(const UiGeometryBlock& block, const UiVec2& translate,
                                      float yLocalMin, float yLocalMax) {
    if (block.vertices.empty() || block.commands.empty()) {
        return;
    }
    const auto base = static_cast<std::uint32_t>(m_data.vertices.size());
    const float opacity = currentOpacity();

    // Translate every vertex. Even culled quads must live in the array so that
    // surviving quads' rebased indices continue to point at the right slots.
    m_data.vertices.reserve(m_data.vertices.size() + block.vertices.size());
    for (const UiVertex& v : block.vertices) {
        UiVertex out = v;
        out.posPx[0] += translate.x;
        out.posPx[1] += translate.y;
        out.rgba8 = scaleAlpha(out.rgba8, opacity);
        m_data.vertices.push_back(out);
    }

    for (const UiDrawCmd& blockCmd : block.commands) {
        if (blockCmd.indexCount == 0) {
            continue;
        }
        UiDrawCmd& dst = currentCommand();

        // Process in groups of 6 indices (one quad = two triangles = 4 vertices).
        // For each group check whether the quad's local Y range overlaps the window.
        const std::uint32_t numQuads = blockCmd.indexCount / 6;
        const std::uint32_t remainder = blockCmd.indexCount % 6;

        for (std::uint32_t q = 0; q < numQuads; ++q) {
            const std::uint32_t iBase = blockCmd.indexOffset + q * 6;
            // Standard quad pattern: {v0, v1, v2, v0, v2, v3}.
            // v0 = top-left, v2 = bottom-right for axis-aligned text quads.
            // Reading just v0 and v2 suffices for all convex axis-aligned rects.
            const std::uint32_t v0idx = block.indices[iBase];
            const std::uint32_t v2idx = block.indices[iBase + 2];
            const float y0 = block.vertices[v0idx].posPx[1];
            const float y2 = block.vertices[v2idx].posPx[1];
            const float qMinY = y0 < y2 ? y0 : y2;
            const float qMaxY = y0 > y2 ? y0 : y2;

            if (qMaxY < yLocalMin || qMinY > yLocalMax) {
                continue;  // Entirely outside the visible window — skip.
            }

            for (std::uint32_t k = 0; k < 6; ++k) {
                m_data.indices.push_back(base + block.indices[iBase + k]);
            }
            dst.indexCount += 6;
        }

        // Pass any remainder (triangle-based sector geometry, etc.) through unculled.
        if (remainder > 0) {
            const std::uint32_t remBase = blockCmd.indexOffset + numQuads * 6;
            for (std::uint32_t k = 0; k < remainder; ++k) {
                m_data.indices.push_back(base + block.indices[remBase + k]);
            }
            dst.indexCount += remainder;
        }
    }
}

void UiDrawList::addRectFilled(const UiRect& rect, const UiColor& color) {
    if (!rect.valid()) {
        return;
    }
    addQuad(rect, UiRect{0.0f, 0.0f, 0.0f, 0.0f}, color.packAbgr8(), UiDrawMode::SolidColor, kUiNoTexture);
}

void UiDrawList::addRectFilledVGradient(const UiRect& rect, const UiColor& top,
                                        const UiColor& bottom) {
    if (!rect.valid()) {
        return;
    }
    const float opacity = currentOpacity();
    const std::uint32_t topRgba = scaleAlpha(top.packAbgr8(), opacity);
    const std::uint32_t botRgba = scaleAlpha(bottom.packAbgr8(), opacity);
    UiDrawCmd& cmd = currentCommand();
    const auto base = static_cast<std::uint32_t>(m_data.vertices.size());
    const auto mode = static_cast<std::uint32_t>(UiDrawMode::SolidColor);
    // v00 / v10 top edge -> top color; v11 / v01 bottom edge -> bottom color.
    m_data.vertices.push_back(UiVertex{{rect.minX, rect.minY}, {0, 0}, topRgba, mode, {}});
    m_data.vertices.push_back(UiVertex{{rect.maxX, rect.minY}, {0, 0}, topRgba, mode, {}});
    m_data.vertices.push_back(UiVertex{{rect.maxX, rect.maxY}, {0, 0}, botRgba, mode, {}});
    m_data.vertices.push_back(UiVertex{{rect.minX, rect.maxY}, {0, 0}, botRgba, mode, {}});
    m_data.indices.push_back(base + 0);
    m_data.indices.push_back(base + 1);
    m_data.indices.push_back(base + 2);
    m_data.indices.push_back(base + 0);
    m_data.indices.push_back(base + 2);
    m_data.indices.push_back(base + 3);
    cmd.indexCount += 6;
}

void UiDrawList::addRectFilledHGradient(const UiRect& rect, const UiColor& left,
                                        const UiColor& right) {
    if (!rect.valid()) return;
    const float opacity = currentOpacity();
    const std::uint32_t lRgba = scaleAlpha(left.packAbgr8(), opacity);
    const std::uint32_t rRgba = scaleAlpha(right.packAbgr8(), opacity);
    UiDrawCmd& cmd = currentCommand();
    const auto base = static_cast<std::uint32_t>(m_data.vertices.size());
    const auto mode = static_cast<std::uint32_t>(UiDrawMode::SolidColor);
    // Left vertices → left color; right vertices → right color.
    m_data.vertices.push_back(UiVertex{{rect.minX, rect.minY}, {0, 0}, lRgba, mode, {}});
    m_data.vertices.push_back(UiVertex{{rect.maxX, rect.minY}, {0, 0}, rRgba, mode, {}});
    m_data.vertices.push_back(UiVertex{{rect.maxX, rect.maxY}, {0, 0}, rRgba, mode, {}});
    m_data.vertices.push_back(UiVertex{{rect.minX, rect.maxY}, {0, 0}, lRgba, mode, {}});
    m_data.indices.push_back(base + 0);
    m_data.indices.push_back(base + 1);
    m_data.indices.push_back(base + 2);
    m_data.indices.push_back(base + 0);
    m_data.indices.push_back(base + 2);
    m_data.indices.push_back(base + 3);
    cmd.indexCount += 6;
}

void UiDrawList::addRoundRectFilledHGradient(const UiRect& rect, const UiColor& left,
                                              const UiColor& right, float radiusPx) {
    if (!rect.valid() || (left.a <= 0.0f && right.a <= 0.0f)) return;
    const float halfW  = rect.width()  * 0.5f;
    const float halfH  = rect.height() * 0.5f;
    const float r      = std::min(std::max(radiusPx, 0.0f), std::min(halfW, halfH));
    const float feather = 1.5f;
    const UiRect dst{rect.minX - feather, rect.minY - feather,
                     rect.maxX + feather, rect.maxY + feather};
    // sdf: {halfW, halfH, cornerRadius, borderPx=0 (fill)}
    const float sdf[4] = {halfW, halfH, r, 0.0f};

    const float opacity = currentOpacity();
    const std::uint32_t lRgba = scaleAlpha(left.packAbgr8(),  opacity);
    const std::uint32_t rRgba = scaleAlpha(right.packAbgr8(), opacity);

    UiDrawCmd& cmd = currentCommand();
    const auto base     = static_cast<std::uint32_t>(m_data.vertices.size());
    const auto modeBits = static_cast<std::uint32_t>(UiDrawMode::RoundRect);

    // UV encodes the pixel offset from the rect center (same convention as emitRoundRect).
    // The GPU interpolates vertex colors across the quad; the SDF mask clips to the
    // rounded shape. Left edge → left color, right edge → right color.
    m_data.vertices.push_back(UiVertex{{dst.minX, dst.minY}, {-halfW - feather, -halfH - feather}, lRgba, modeBits, {sdf[0], sdf[1], sdf[2], sdf[3]}});
    m_data.vertices.push_back(UiVertex{{dst.maxX, dst.minY}, { halfW + feather, -halfH - feather}, rRgba, modeBits, {sdf[0], sdf[1], sdf[2], sdf[3]}});
    m_data.vertices.push_back(UiVertex{{dst.maxX, dst.maxY}, { halfW + feather,  halfH + feather}, rRgba, modeBits, {sdf[0], sdf[1], sdf[2], sdf[3]}});
    m_data.vertices.push_back(UiVertex{{dst.minX, dst.maxY}, {-halfW - feather,  halfH + feather}, lRgba, modeBits, {sdf[0], sdf[1], sdf[2], sdf[3]}});
    m_data.indices.push_back(base + 0);
    m_data.indices.push_back(base + 1);
    m_data.indices.push_back(base + 2);
    m_data.indices.push_back(base + 0);
    m_data.indices.push_back(base + 2);
    m_data.indices.push_back(base + 3);
    cmd.indexCount += 6;
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

namespace {

// Emit a single rounded-box SDF quad. The quad is grown by `feather` pixels on
// every side so the anti-aliased outer edge (and the outer half of a centered
// stroke) is never clipped by the geometry itself. UV carries the pixel offset
// from the rect centre; the shader evaluates the distance field per fragment.
void emitRoundRect(UiDrawList& dl, const UiRect& rect, std::uint32_t rgba,
                   float radiusPx, float borderPx, float feather) {
    const float halfW = rect.width() * 0.5f;
    const float halfH = rect.height() * 0.5f;
    const float r = std::min(std::max(radiusPx, 0.0f), std::min(halfW, halfH));
    const UiRect dst{rect.minX - feather, rect.minY - feather,
                     rect.maxX + feather, rect.maxY + feather};
    const UiRect local{-halfW - feather, -halfH - feather,
                       halfW + feather, halfH + feather};
    const float sdf[4] = {halfW, halfH, r, borderPx};
    dl.addQuad(dst, local, rgba, UiDrawMode::RoundRect, kUiNoTexture, sdf);
}

}  // namespace

void UiDrawList::addRoundRectFilled(const UiRect& rect, const UiColor& color, float radiusPx) {
    if (!rect.valid() || color.a <= 0.0f) {
        return;
    }
    emitRoundRect(*this, rect, color.packAbgr8(), radiusPx, 0.0f, 1.5f);
}

void UiDrawList::addRoundRect(const UiRect& rect, const UiColor& color, float radiusPx,
                             float thicknessPx) {
    if (!rect.valid() || color.a <= 0.0f || thicknessPx <= 0.0f) {
        return;
    }
    // The stroke is centred on the edge, so it reaches thickness/2 outside the rect.
    emitRoundRect(*this, rect, color.packAbgr8(), radiusPx, thicknessPx, thicknessPx * 0.5f + 1.5f);
}

void UiDrawList::addCircleFilled(const UiVec2& center, float radiusPx, const UiColor& color) {
    if (radiusPx <= 0.0f || color.a <= 0.0f) {
        return;
    }
    const UiRect rect{center.x - radiusPx, center.y - radiusPx,
                      center.x + radiusPx, center.y + radiusPx};
    emitRoundRect(*this, rect, color.packAbgr8(), radiusPx, 0.0f, 1.5f);
}

void UiDrawList::addCircle(const UiVec2& center, float radiusPx, const UiColor& color,
                          float thicknessPx) {
    if (radiusPx <= 0.0f || color.a <= 0.0f || thicknessPx <= 0.0f) {
        return;
    }
    const UiRect rect{center.x - radiusPx, center.y - radiusPx,
                      center.x + radiusPx, center.y + radiusPx};
    emitRoundRect(*this, rect, color.packAbgr8(), radiusPx, thicknessPx, thicknessPx * 0.5f + 1.5f);
}

void UiDrawList::addRoundRectGlow(const UiRect& rect, const UiColor& color, float radiusPx,
                                 float glowSizePx) {
    if (!rect.valid() || color.a <= 0.0f || glowSizePx <= 0.0f) {
        return;
    }
    const float halfW = rect.width() * 0.5f;
    const float halfH = rect.height() * 0.5f;
    const float r = std::min(std::max(radiusPx, 0.0f), std::min(halfW, halfH));
    const float feather = glowSizePx + 2.0f;  // quad must cover the full falloff
    const UiRect dst{rect.minX - feather, rect.minY - feather,
                     rect.maxX + feather, rect.maxY + feather};
    const UiRect local{-halfW - feather, -halfH - feather,
                       halfW + feather, halfH + feather};
    const float sdf[4] = {halfW, halfH, r, glowSizePx};
    addQuad(dst, local, color.packAbgr8(), UiDrawMode::RoundRectGlow, kUiNoTexture, sdf);
}

void UiDrawList::addImage(const UiRect& rect, UiTextureId textureId, const UiColor& tint, const UiRect& uv) {
    if (!rect.valid()) {
        return;
    }
    addQuad(rect, uv, tint.packAbgr8(), UiDrawMode::Textured, textureId);
}

void UiDrawList::addDropShadow(const UiRect& rect, const UiColor& color, float blurSigma,
                               float offsetX, float offsetY) {
    if (!rect.valid() || blurSigma <= 0.0f || color.a <= 0.0f) {
        return;
    }
    const float pad = blurSigma * 3.0f;
    const float k   = 3.0f;  // pad / blurSigma — normalized distance at outer edge
    const std::uint32_t rgba = color.packAbgr8();

    const float cx[4] = {rect.minX + offsetX - pad, rect.minX + offsetX,
                          rect.maxX + offsetX,       rect.maxX + offsetX + pad};
    const float cy[4] = {rect.minY + offsetY - pad, rect.minY + offsetY,
                          rect.maxY + offsetY,       rect.maxY + offsetY + pad};
    // UV encodes normalized distance from the inner rect edge: 0 on/inside, k at outer edge.
    const float ux[4] = {k, 0.0f, 0.0f, k};
    const float uy[4] = {k, 0.0f, 0.0f, k};

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const UiRect dst{cx[col], cy[row], cx[col + 1], cy[row + 1]};
            if (!dst.valid()) {
                continue;
            }
            addQuad(dst, {ux[col], uy[row], ux[col + 1], uy[row + 1]},
                    rgba, UiDrawMode::Shadow, kUiNoTexture);
        }
    }
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

void UiDrawList::addBevel(const UiRect& rect, const UiColor& highlightColor,
                          const UiColor& shadowColor, float radiusPx,
                          float thicknessPx, bool inward) {
    if (!rect.valid() || thicknessPx <= 0.0f) return;

    const UiColor& hlCol = inward ? shadowColor    : highlightColor;
    const UiColor& shCol = inward ? highlightColor : shadowColor;

    // The SDF quad for addRoundRect is grown by (thicknessPx*0.5 + 1.5) beyond
    // `rect` so the anti-aliased outer edge isn't clipped by its own geometry.
    // Expand the clip regions by the same feather so the outer anti-aliased fringe
    // isn't scissored away before it gets to the screen.
    const float feather = thicknessPx * 0.5f + 1.5f;
    const float midX    = (rect.minX + rect.maxX) * 0.5f;
    const float midY    = (rect.minY + rect.maxY) * 0.5f;

    // Diagonal key light from the upper-left. Highlight first along the top and left
    // edge bands, then shadow over the bottom and right edge bands. Where the shadow
    // bands overlap a lit edge (the top-right of the top edge, the bottom of the left
    // edge) they win, so the lit edges fade toward the bottom-right corner — selling a
    // diagonal light rather than a flat overhead one.

    // Highlight — top edge band.
    pushClip(UiRect{rect.minX - feather, rect.minY - feather, rect.maxX + feather, midY});
    addRoundRect(rect, hlCol, radiusPx, thicknessPx);
    popClip();
    // Highlight — left edge band.
    pushClip(UiRect{rect.minX - feather, rect.minY - feather, midX, rect.maxY + feather});
    addRoundRect(rect, hlCol, radiusPx, thicknessPx);
    popClip();

    // Shadow — bottom edge band.
    pushClip(UiRect{rect.minX - feather, midY, rect.maxX + feather, rect.maxY + feather});
    addRoundRect(rect, shCol, radiusPx, thicknessPx);
    popClip();
    // Shadow — right edge band.
    pushClip(UiRect{midX, rect.minY - feather, rect.maxX + feather, rect.maxY + feather});
    addRoundRect(rect, shCol, radiusPx, thicknessPx);
    popClip();
}

void UiDrawList::addSectorFilled(const UiVec2& center, float innerRadiusPx, float outerRadiusPx,
                                  float startAngleRad, float endAngleRad, const UiColor& color,
                                  int numSteps) {
    if (numSteps < 1 || outerRadiusPx <= 0.0f || color.a <= 0.0f
        || startAngleRad == endAngleRad) {
        return;
    }
    const std::uint32_t rgba = scaleAlpha(color.packAbgr8(), currentOpacity());
    UiDrawCmd& cmd = currentCommand();
    const auto solid = static_cast<std::uint32_t>(UiDrawMode::SolidColor);
    const float da = (endAngleRad - startAngleRad) / static_cast<float>(numSteps);
    const bool ring = innerRadiusPx > 0.0f;

    for (int i = 0; i < numSteps; ++i) {
        const float a0 = startAngleRad + static_cast<float>(i) * da;
        const float a1 = a0 + da;
        const float c0 = std::cos(a0), s0 = std::sin(a0);
        const float c1 = std::cos(a1), s1 = std::sin(a1);
        const auto base = static_cast<std::uint32_t>(m_data.vertices.size());

        if (!ring) {
            // Solid wedge: three vertices (center, outer0, outer1).
            m_data.vertices.push_back(UiVertex{
                {center.x, center.y}, {0, 0}, rgba, solid, {}});
            m_data.vertices.push_back(UiVertex{
                {center.x + outerRadiusPx * c0, center.y + outerRadiusPx * s0}, {0, 0}, rgba, solid, {}});
            m_data.vertices.push_back(UiVertex{
                {center.x + outerRadiusPx * c1, center.y + outerRadiusPx * s1}, {0, 0}, rgba, solid, {}});
            m_data.indices.push_back(base);
            m_data.indices.push_back(base + 1);
            m_data.indices.push_back(base + 2);
            cmd.indexCount += 3;
        } else {
            // Ring segment: quad (inner0, outer0, outer1, inner1).
            m_data.vertices.push_back(UiVertex{
                {center.x + innerRadiusPx * c0, center.y + innerRadiusPx * s0}, {0, 0}, rgba, solid, {}});
            m_data.vertices.push_back(UiVertex{
                {center.x + outerRadiusPx * c0, center.y + outerRadiusPx * s0}, {0, 0}, rgba, solid, {}});
            m_data.vertices.push_back(UiVertex{
                {center.x + outerRadiusPx * c1, center.y + outerRadiusPx * s1}, {0, 0}, rgba, solid, {}});
            m_data.vertices.push_back(UiVertex{
                {center.x + innerRadiusPx * c1, center.y + innerRadiusPx * s1}, {0, 0}, rgba, solid, {}});
            m_data.indices.push_back(base);
            m_data.indices.push_back(base + 1);
            m_data.indices.push_back(base + 2);
            m_data.indices.push_back(base);
            m_data.indices.push_back(base + 2);
            m_data.indices.push_back(base + 3);
            cmd.indexCount += 6;
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
