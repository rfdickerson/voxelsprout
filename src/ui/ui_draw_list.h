#pragma once

#include "ui/ui_types.h"
#include "ui/vector/vector_path.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

// Immediate-mode draw list: the renderer-agnostic core of the UI module. Widgets
// and the HUD append primitives here; the result (UiDrawData) is handed to the
// Vulkan UiRenderer. No Vulkan, no GPU handles -- textures are referenced by id.
namespace odai::ui {

class Font;
struct StrokeOptions;  // ui/vector/vector_tessellator.h

// Per-vertex draw mode. The low 8 bits of UiVertex::mode select this mode; the
// remaining bits hold the bindless UI texture slot. One command can therefore
// mix icons, font atlases, and solid geometry under one shared clip rect.
enum class UiDrawMode : std::uint32_t {
    SolidColor = 0,  // Use vertex color, ignore texture.
    Textured = 1,    // Sample rgba texture, multiply by vertex color.
    GlyphAlpha = 2,  // Sample R8 atlas as coverage; rgb from vertex color.
    Shadow = 3,      // Gaussian drop shadow; uv = normalized distance from inner rect edge.
    RoundRect = 4,   // Analytic rounded-box SDF; uv = pixel pos from center, sdf = params.
    RoundRectGlow = 5,  // Soft outer glow around a rounded box; sdf.w = glow falloff px.
};

struct UiVertex {            // 40 bytes; matches the renderer's vertex input layout.
    float posPx[2] = {};      // Pixel space, top-left origin, +Y down. offset 0
    float uv[2] = {};         // Atlas coords, or (RoundRect) px from center. offset 8
    std::uint32_t rgba8 = 0;  // Packed ABGR8 vertex color.             offset 16
    std::uint32_t mode = 0;   // UiDrawMode in low 8 bits; texture slot above. offset 20
    // Mode-specific params. RoundRect: {halfWidthPx, halfHeightPx, cornerRadiusPx,
    // borderPx} where borderPx <= 0 fills the shape and > 0 strokes it (centered
    // on the edge). Unused (zero) for all other modes.
    float sdf[4] = {};        //                                        offset 24
};

struct UiDrawCmd {
    std::uint32_t indexOffset = 0;       // First index into UiDrawData::indices.
    std::uint32_t indexCount = 0;        // Number of indices to draw.
    // Legacy source texture for cached geometry. Runtime batching is clip-only;
    // texture selection is encoded per vertex.
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

// A reusable, pre-generated chunk of UI geometry in local coordinates (origin at
// 0,0). Cached by a widget so an unchanged frame can re-emit it via
// UiDrawList::appendCached without re-running text layout or quad generation.
// Indices are 0-based into `vertices`; command clip rects are ignored on replay
// (the live clip from the draw list is applied instead).
struct UiGeometryBlock {
    std::vector<UiVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<UiDrawCmd> commands;

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
    // Solid quad with a vertical gradient: the top edge uses `top`, the bottom
    // edge `bottom`, interpolated per fragment. Used for parchment/gilt panel fills.
    void addRectFilledVGradient(const UiRect& rect, const UiColor& top, const UiColor& bottom);
    // Solid quad with a horizontal gradient: left edge uses `left`, right edge uses `right`.
    void addRectFilledHGradient(const UiRect& rect, const UiColor& left, const UiColor& right);
    // Rounded-rect fill with a horizontal gradient (left→right). The SDF mask is
    // applied over the per-vertex color gradient so the rounded corners clip the
    // gradient cleanly. Use for progress bar fills with cornerRadiusPx > 0.
    void addRoundRectFilledHGradient(const UiRect& rect, const UiColor& left,
                                     const UiColor& right, float radiusPx);
    // Rounded-rect fill with a vertical gradient (top→bottom). Same SDF-masked
    // path as the horizontal variant; use for soft duotone "gradient cards".
    void addRoundRectFilledVGradient(const UiRect& rect, const UiColor& top,
                                     const UiColor& bottom, float radiusPx);
    void addRect(const UiRect& rect, const UiColor& color, float thicknessPx = 1.0f);

    // --- Vector primitives (resolution-independent, anti-aliased SDF) ---
    // All radii/thicknesses are in pixels; callers scale by the DPI factor so the
    // shapes stay crisp at any framebuffer scale. `radius` is clamped to half the
    // shorter side, so passing a large radius yields a pill/stadium or circle.
    void addRoundRectFilled(const UiRect& rect, const UiColor& color, float radiusPx);
    // Stroke a rounded rect; the line is centered on the rounded edge path.
    void addRoundRect(const UiRect& rect, const UiColor& color, float radiusPx, float thicknessPx);
    void addCircleFilled(const UiVec2& center, float radiusPx, const UiColor& color);
    void addCircle(const UiVec2& center, float radiusPx, const UiColor& color, float thicknessPx);
    // Soft outer glow that hugs a rounded rect: solid under `rect`, fading to zero
    // `glowSizePx` outside the rounded edge. Draw it behind the shape it lights.
    void addRoundRectGlow(const UiRect& rect, const UiColor& color, float radiusPx, float glowSizePx);
    // Gaussian drop shadow behind `rect`, offset by (offsetX, offsetY) pixels.
    // blurSigma controls softness; shadow fades to ~1% at 3*blurSigma from the edge.
    void addDropShadow(const UiRect& rect, const UiColor& color, float blurSigma,
                       float offsetX = 0.0f, float offsetY = 4.0f);

    // Draw a bevel border around `rect`, simulating a diagonal key light from the
    // upper-left. The top and left edges catch `highlightColor`; the bottom and right
    // edges fall into `shadowColor`. Because the shadow bands overpaint the lower-right,
    // the top and left edges fade from lit (near the top-left corner) to dark (near the
    // bottom-right) — a believable diagonal light, not a flat overhead one. radiusPx
    // matches the Panel/Button cornerRadiusPx so the bevel follows the same rounded-
    // corner arc. Pass inward=true for a pressed/recessed look (swaps highlight and
    // shadow). Each edge band is an anti-aliased SDF stroke via addRoundRect, scissored
    // to that edge's region.
    void addBevel(const UiRect& rect, const UiColor& highlightColor,
                  const UiColor& shadowColor, float radiusPx,
                  float thicknessPx, bool inward = false);
    void addImage(const UiRect& rect, UiTextureId textureId, const UiColor& tint = {},
                  const UiRect& uv = UiRect{0.0f, 0.0f, 1.0f, 1.0f});
    void add9Slice(const UiRect& rect, const UiNineSlice& slice, const UiColor& color = {});

    // Draw a filled annular sector (pie or donut wedge). Angles in radians; 0 = +X,
    // increasing clockwise (+Y down). innerRadiusPx = 0 → solid pie wedge; > 0 → ring.
    // numSteps controls arc smoothness (32 is good for most UI sizes).
    void addSectorFilled(const UiVec2& center, float innerRadiusPx, float outerRadiusPx,
                         float startAngleRad, float endAngleRad, const UiColor& color,
                         int numSteps = 32);

    // --- Procedural vector paths (CPU-tessellated, anti-aliased) ---
    // Fill an arbitrary path (curves/arcs/polygons). Subpaths are treated as
    // closed; holes and concave shapes are handled per the fill rule. The result
    // is a triangle mesh with a feathered AA fringe, emitted as SolidColor.
    void addPathFilled(const VectorPath& path, const UiColor& color,
                       FillRule fillRule = FillRule::NonZero);
    // Stroke an arbitrary path with width/join/cap from `opts`.
    void addPathStroked(const VectorPath& path, const UiColor& color, const StrokeOptions& opts);
    // Convenience: stroke a polyline of `count` points with a round-join/cap line.
    void addPolylineAA(const UiVec2* points, std::size_t count, const UiColor& color,
                       float widthPx, bool closed = false);

    // Draw a single line of text with the top-left baseline box at posPx; returns
    // the pen x advance. Newlines are not interpreted (use RichText for layout).
    float addText(const Font& font, std::string_view utf8, const UiVec2& posPx, const UiColor& color);

    void pushClip(const UiRect& rect);
    void popClip();
    [[nodiscard]] const UiRect& currentClip() const { return m_clipStack.back(); }

    // Opacity stack: multiplies the alpha of all subsequently emitted geometry
    // (nested multiplicatively). Used to fade widgets/overlays in and out. Applied
    // by addQuad and appendCached, so cached geometry fades correctly too.
    void pushOpacity(float opacity);
    void popOpacity();
    [[nodiscard]] float currentOpacity() const;

    [[nodiscard]] const UiDrawData& data() const { return m_data; }
    [[nodiscard]] UiDrawData& data() { return m_data; }

    // Low-level quad emission, used by addText/RichText. Vertices are p00 (min),
    // p10, p11, p01 with matching UVs; emitted as two triangles.
    void addQuad(const UiRect& dst, const UiRect& uv, std::uint32_t rgba8, UiDrawMode mode,
                 UiTextureId textureId, const float sdf[4] = nullptr);

    // Low-level triangle-mesh emission. Vertices are taken as-is (pixel space);
    // the current opacity scales their alpha and indices are rebased into the
    // live buffer. Used by the vector tessellator (addPath*) and cached icons.
    void addTriangleMesh(const UiVertex* vertices, std::size_t vertexCount,
                         const std::uint32_t* indices, std::size_t indexCount);

    // Append a cached geometry block, translating its vertices by `translate` and
    // re-emitting its commands under the current clip (rebasing indices). Lets a
    // widget skip layout + quad generation when nothing but position changed.
    void appendCached(const UiGeometryBlock& block, const UiVec2& translate);

    // Like appendCached, but performs CPU-side quad culling: quads whose local-space
    // Y extent falls entirely outside [yLocalMin, yLocalMax] are skipped (indices not
    // emitted). Vertices are still translated and pushed so rebased indices stay valid.
    // "Local" means pre-translate coordinates — the same space the block was built in.
    // Remainder indices (non-multiple of 6, e.g. sector triangles) are passed through.
    void appendCachedClipped(const UiGeometryBlock& block, const UiVec2& translate,
                             float yLocalMin, float yLocalMax);

    // Like appendCached, but multiplies every vertex color (channel-wise, incl.
    // alpha) by `tintMul` before emission. Recolors a monochrome cached glyph
    // without re-tessellating. Pass white (1,1,1,1) for no tint.
    void appendCachedTinted(const UiGeometryBlock& block, const UiVec2& translate,
                            const UiColor& tintMul);

    // Resolve a vector icon by name from VectorIconRegistry::global() and draw it
    // fitted into `dst` (uniform aspect-preserving scale). `tint` multiplies the
    // baked colors; pass default white to draw the icon's own colors.
    void addVectorIcon(std::string_view name, const UiRect& dst, const UiColor& tint = UiColor{});

private:
    // Ensure the current command matches the current clip; texture selection is
    // per-vertex through the bindless UI texture table.
    UiDrawCmd& currentCommand();

    UiDrawData m_data;
    std::vector<UiRect> m_clipStack;
    std::vector<float> m_opacityStack;
};

}  // namespace odai::ui
