#include "ui/vector/svg_document.h"

// nanosvg is an optional dependency. When its header isn't on the include path
// (e.g. a headless build without the vcpkg port) the importer compiles as a
// no-op so odai_ui still builds; everything else in the vector module works.
// The vcpkg port installs it under <nanosvg/nanosvg.h>; other distributions ship
// a bare <nanosvg.h>, so handle both.
#if __has_include(<nanosvg.h>) || __has_include(<nanosvg/nanosvg.h>)

#include "ui/vector/vector_mesh_sink.h"
#include "ui/vector/vector_path.h"
#include "ui/vector/vector_tessellator.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// nanosvg is header-only; define the implementation in exactly this translation
// unit (mirrors STB_IMAGE_IMPLEMENTATION in ui/theme/ui_theme.cc).
#define NANOSVG_IMPLEMENTATION
#if __has_include(<nanosvg.h>)
#include <nanosvg.h>
#else
#include <nanosvg/nanosvg.h>
#endif

namespace odai::ui {

namespace {

// nanosvg packs colors as 0xAABBGGRR — identical byte order to our ABGR8
// (UiColor::packAbgr8). This is asserted by a unit test. We only need to fold in
// the shape opacity (a separate 0..1 multiplier on the alpha byte).
std::uint32_t nsvgColorWithOpacity(unsigned int nsvgColor, float opacity) {
    const std::uint32_t a = (nsvgColor >> 24) & 0xFFu;
    const float scaled = static_cast<float>(a) * std::clamp(opacity, 0.0f, 1.0f);
    const auto na = static_cast<std::uint32_t>(scaled + 0.5f);
    return (nsvgColor & 0x00FFFFFFu) | ((na > 255u ? 255u : na) << 24);
}

// Average a gradient's stop colors (Phase 1 approximation for non-solid paint).
std::uint32_t averageGradientColor(const NSVGgradient* grad, float opacity) {
    if (grad == nullptr || grad->nstops <= 0) {
        return nsvgColorWithOpacity(0xFF000000u, opacity);
    }
    std::uint32_t rs = 0, gs = 0, bs = 0, as = 0;
    for (int i = 0; i < grad->nstops; ++i) {
        const unsigned int c = grad->stops[i].color;
        rs += c & 0xFFu;
        gs += (c >> 8) & 0xFFu;
        bs += (c >> 16) & 0xFFu;
        as += (c >> 24) & 0xFFu;
    }
    const auto n = static_cast<std::uint32_t>(grad->nstops);
    const std::uint32_t avg = (rs / n) | ((gs / n) << 8) | ((bs / n) << 16) | ((as / n) << 24);
    return nsvgColorWithOpacity(avg, opacity);
}

std::uint32_t resolvePaintColor(const NSVGpaint& paint, float opacity) {
    switch (paint.type) {
        case NSVG_PAINT_COLOR:
            return nsvgColorWithOpacity(paint.color, opacity);
        case NSVG_PAINT_LINEAR_GRADIENT:
        case NSVG_PAINT_RADIAL_GRADIENT:
            return averageGradientColor(paint.gradient, opacity);
        case NSVG_PAINT_NONE:
        default:
            return 0u;  // Fully transparent → skipped by the caller.
    }
}

LineJoin mapJoin(int nsvgJoin) {
    switch (nsvgJoin) {
        case NSVG_JOIN_ROUND: return LineJoin::Round;
        case NSVG_JOIN_BEVEL: return LineJoin::Bevel;
        case NSVG_JOIN_MITER:
        default: return LineJoin::Miter;
    }
}

LineCap mapCap(int nsvgCap) {
    switch (nsvgCap) {
        case NSVG_CAP_ROUND: return LineCap::Round;
        case NSVG_CAP_SQUARE: return LineCap::Square;
        case NSVG_CAP_BUTT:
        default: return LineCap::Butt;
    }
}

// Append one nanosvg path (a flattened cubic-bezier polyline) into a VectorPath,
// scaling coordinates by `s`. nanosvg stores npts points as a start point
// followed by groups of three (cp1, cp2, end) per cubic segment.
void appendNsvgPath(VectorPath& out, const NSVGpath* path, float s) {
    if (path == nullptr || path->npts < 1) {
        return;
    }
    const float* p = path->pts;
    out.moveTo(p[0] * s, p[1] * s);
    const int segCount = (path->npts - 1) / 3;
    for (int k = 0; k < segCount; ++k) {
        const int i = k * 3;
        out.cubicBezierTo(p[(i + 1) * 2] * s, p[(i + 1) * 2 + 1] * s,
                          p[(i + 2) * 2] * s, p[(i + 2) * 2 + 1] * s,
                          p[(i + 3) * 2] * s, p[(i + 3) * 2 + 1] * s);
    }
    if (path->closed) {
        out.close();
    }
}

bool tessellateImage(NSVGimage* image, const SvgTessellateOptions& opts, UiGeometryBlock& outBlock) {
    if (image == nullptr) {
        return false;
    }
    const float docMax = std::max(image->width, image->height);
    const float s = (docMax > 0.0f) ? (opts.targetSizePx / docMax) : 1.0f;

    GeometryBlockMeshSink sink(outBlock);

    for (NSVGshape* shape = image->shapes; shape != nullptr; shape = shape->next) {
        if ((shape->flags & NSVG_FLAGS_VISIBLE) == 0) {
            continue;
        }

        // Accumulate all subpaths of the shape so holes tessellate correctly.
        VectorPath path;
        path.setTessellationTolerancePx(opts.tolerancePx);
        for (NSVGpath* sp = shape->paths; sp != nullptr; sp = sp->next) {
            appendNsvgPath(path, sp, s);
        }
        if (path.empty()) {
            continue;
        }

        // Fill.
        if (shape->fill.type != NSVG_PAINT_NONE) {
            std::uint32_t fillColor = opts.useTint
                                          ? opts.tintOverride.packAbgr8()
                                          : resolvePaintColor(shape->fill, shape->opacity);
            if ((fillColor >> 24) != 0u) {
                TessOptions tess;
                tess.dpiScale = opts.dpiScale;
                tess.fillRule = (shape->fillRule == NSVG_FILLRULE_EVENODD) ? FillRule::EvenOdd
                                                                          : FillRule::NonZero;
                tessellateFill(path, fillColor, tess, sink);
            }
        }

        // Stroke.
        if (shape->stroke.type != NSVG_PAINT_NONE && shape->strokeWidth > 0.0f) {
            std::uint32_t strokeColor = opts.useTint
                                            ? opts.tintOverride.packAbgr8()
                                            : resolvePaintColor(shape->stroke, shape->opacity);
            if ((strokeColor >> 24) != 0u) {
                StrokeOptions stroke;
                stroke.widthPx = shape->strokeWidth * s;
                stroke.join = mapJoin(shape->strokeLineJoin);
                stroke.cap = mapCap(shape->strokeLineCap);
                stroke.miterLimit = shape->miterLimit > 0.0f ? shape->miterLimit : 4.0f;
                stroke.dpiScale = opts.dpiScale;
                tessellateStroke(path, strokeColor, stroke, sink);
            }
        }
    }
    return true;
}

}  // namespace

bool tessellateSvgFile(const std::filesystem::path& path, const SvgTessellateOptions& opts,
                       UiGeometryBlock& outBlock) {
    NSVGimage* image = nsvgParseFromFile(path.string().c_str(), "px", 96.0f);
    if (image == nullptr) {
        return false;
    }
    const bool ok = tessellateImage(image, opts, outBlock);
    nsvgDelete(image);
    return ok;
}

bool tessellateSvgString(const char* svgText, const SvgTessellateOptions& opts,
                         UiGeometryBlock& outBlock) {
    if (svgText == nullptr) {
        return false;
    }
    // nanosvg mutates its input buffer, so parse a private copy.
    std::string buffer(svgText);
    NSVGimage* image = nsvgParse(buffer.data(), "px", 96.0f);
    if (image == nullptr) {
        return false;
    }
    const bool ok = tessellateImage(image, opts, outBlock);
    nsvgDelete(image);
    return ok;
}

}  // namespace odai::ui

#else  // nanosvg not available — importer degrades to a no-op.

namespace odai::ui {

bool tessellateSvgFile(const std::filesystem::path&, const SvgTessellateOptions&,
                       UiGeometryBlock&) {
    return false;
}

bool tessellateSvgString(const char*, const SvgTessellateOptions&, UiGeometryBlock&) {
    return false;
}

}  // namespace odai::ui

#endif  // __has_include(<nanosvg.h>) || __has_include(<nanosvg/nanosvg.h>)
