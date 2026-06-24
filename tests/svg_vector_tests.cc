// Tests for the procedural vector module + SVG icon importer (odai_ui). Custom
// harness matching tests/ui_tests.cc (no gtest): geometry is validated by vertex/
// triangle counts, coverage probes, and AA-fringe alpha checks.

#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"
#include "ui/vector/svg_document.h"
#include "ui/vector/vector_cache_io.h"
#include "ui/vector/vector_icon_registry.h"
#include "ui/vector/vector_mesh_sink.h"
#include "ui/vector/vector_path.h"
#include "ui/vector/vector_tessellator.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>

namespace {

using namespace odai::ui;

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[svg test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectNear(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) > epsilon) {
        std::cerr << "[svg test] FAIL: " << message << " (expected " << expected << ", got "
                  << actual << ")\n";
        ++g_failures;
    }
}

bool isFullAlpha(std::uint32_t rgba8) { return ((rgba8 >> 24) & 0xFFu) == 0xFFu; }

float triArea(const UiVec2& a, const UiVec2& b, const UiVec2& c) {
    return 0.5f * std::fabs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

bool pointInTri(const UiVec2& a, const UiVec2& b, const UiVec2& c, const UiVec2& p) {
    const float d1 = (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
    const float d2 = (p.x - c.x) * (b.y - c.y) - (b.x - c.x) * (p.y - c.y);
    const float d3 = (p.x - a.x) * (c.y - a.y) - (c.x - a.x) * (p.y - a.y);
    const bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    const bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(hasNeg && hasPos);
}

UiVec2 vpos(const UiVertex& v) { return UiVec2{v.posPx[0], v.posPx[1]}; }

// Sum the area of "solid" triangles (all three vertices at full alpha) — i.e. the
// interior fill, excluding the AA fringe.
float solidArea(const UiGeometryBlock& b) {
    float area = 0.0f;
    for (std::size_t i = 0; i + 2 < b.indices.size(); i += 3) {
        const UiVertex& a = b.vertices[b.indices[i]];
        const UiVertex& v1 = b.vertices[b.indices[i + 1]];
        const UiVertex& v2 = b.vertices[b.indices[i + 2]];
        if (isFullAlpha(a.rgba8) && isFullAlpha(v1.rgba8) && isFullAlpha(v2.rgba8)) {
            area += triArea(vpos(a), vpos(v1), vpos(v2));
        }
    }
    return area;
}

// Is point p covered by any solid (full-alpha) triangle?
bool solidCovers(const UiGeometryBlock& b, const UiVec2& p) {
    for (std::size_t i = 0; i + 2 < b.indices.size(); i += 3) {
        const UiVertex& a = b.vertices[b.indices[i]];
        const UiVertex& v1 = b.vertices[b.indices[i + 1]];
        const UiVertex& v2 = b.vertices[b.indices[i + 2]];
        if (isFullAlpha(a.rgba8) && isFullAlpha(v1.rgba8) && isFullAlpha(v2.rgba8)) {
            if (pointInTri(vpos(a), vpos(v1), vpos(v2), p)) {
                return true;
            }
        }
    }
    return false;
}

bool hasZeroAlphaVertex(const UiGeometryBlock& b) {
    for (const UiVertex& v : b.vertices) {
        if (((v.rgba8 >> 24) & 0xFFu) == 0u) {
            return true;
        }
    }
    return false;
}

UiGeometryBlock fill(const VectorPath& p, const UiColor& c, FillRule fr = FillRule::NonZero) {
    UiGeometryBlock b;
    GeometryBlockMeshSink sink(b);
    TessOptions opts;
    opts.fillRule = fr;
    tessellateFill(p, c.packAbgr8(), opts, sink);
    return b;
}

// --- Tests -------------------------------------------------------------------

void testFlattenToleranceMonotonic() {
    VectorPath coarse;
    coarse.setTessellationTolerancePx(2.0f);
    coarse.circle(0.0f, 0.0f, 50.0f);
    VectorPath fine;
    fine.setTessellationTolerancePx(0.1f);
    fine.circle(0.0f, 0.0f, 50.0f);
    const std::size_t coarseN = coarse.subPaths().front().points.size();
    const std::size_t fineN = fine.subPaths().front().points.size();
    expectTrue(fineN > coarseN, "finer tolerance yields more flattened points");
    expectTrue(coarseN >= 4, "coarse circle still has several segments");
}

void testCircleVerticesOnRadius() {
    const float r = 40.0f;
    VectorPath p;
    p.setTessellationTolerancePx(0.25f);
    p.circle(0.0f, 0.0f, r);
    float maxErr = 0.0f;
    for (const UiVec2& pt : p.subPaths().front().points) {
        const float d = std::sqrt(pt.x * pt.x + pt.y * pt.y);
        maxErr = std::max(maxErr, std::fabs(d - r));
    }
    expectTrue(maxErr <= 0.26f, "flattened circle vertices lie within tolerance of the radius");
}

void testSquareFillAndFringe() {
    VectorPath p;
    p.rect(0.0f, 0.0f, 10.0f, 10.0f);
    UiGeometryBlock b = fill(p, UiColor{1, 1, 1, 1});
    expectTrue(!b.vertices.empty() && !b.indices.empty(), "square fill emits geometry");
    expectTrue(b.commands.size() == 1u, "fill uses a single SolidColor command");
    expectNear(solidArea(b), 100.0f, 0.5f, "square solid area equals 10x10");
    expectTrue(solidCovers(b, UiVec2{5.0f, 5.0f}), "square center is covered");
    expectTrue(!solidCovers(b, UiVec2{20.0f, 20.0f}), "point outside square is not covered");
    expectTrue(hasZeroAlphaVertex(b), "fill has a transparent AA-fringe vertex");
}

void testConcaveFillArea() {
    // L-shape: 10x10 minus the 6x6 top-right corner = area 64.
    VectorPath p;
    p.moveTo(0, 0).lineTo(10, 0).lineTo(10, 4).lineTo(4, 4).lineTo(4, 10).lineTo(0, 10).close();
    UiGeometryBlock b = fill(p, UiColor{1, 1, 1, 1});
    expectNear(solidArea(b), 64.0f, 0.5f, "concave L solid area equals 64");
    expectTrue(solidCovers(b, UiVec2{2.0f, 8.0f}), "L lower-left arm covered");
    expectTrue(!solidCovers(b, UiVec2{8.0f, 8.0f}), "L notch (top-right) not covered");
}

void testEvenOddDonutHole() {
    VectorPath p;
    p.setTessellationTolerancePx(0.25f);
    p.circle(20.0f, 20.0f, 10.0f);
    p.circle(20.0f, 20.0f, 5.0f);
    UiGeometryBlock b = fill(p, UiColor{1, 1, 1, 1}, FillRule::EvenOdd);
    expectTrue(solidCovers(b, UiVec2{20.0f, 12.5f}), "donut band (radius ~7.5) is covered");
    expectTrue(!solidCovers(b, UiVec2{20.0f, 20.0f}), "donut hole center is NOT covered");
}

void testStrokeGeometryAndFringe() {
    VectorPath p;
    p.moveTo(0, 0).lineTo(10, 0);
    StrokeOptions opts;
    opts.widthPx = 2.0f;
    opts.cap = LineCap::Butt;
    UiGeometryBlock b;
    GeometryBlockMeshSink sink(b);
    tessellateStroke(p, UiColor{1, 1, 1, 1}.packAbgr8(), opts, sink);
    expectTrue(!b.indices.empty(), "stroke emits triangles");
    expectTrue(hasZeroAlphaVertex(b), "stroke has a transparent AA-fringe vertex");
    expectTrue(solidCovers(b, UiVec2{5.0f, 0.0f}), "stroke core covers the line midpoint");
}

void testCacheRoundTrip() {
    VectorPath p;
    p.rect(0.0f, 0.0f, 16.0f, 16.0f);
    UiGeometryBlock src = fill(p, UiColor{0.2f, 0.4f, 0.6f, 1.0f});

    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / "odai_vec_roundtrip.odaivec";
    expectTrue(writeVectorCache(tmp, src, 16.0f), "writeVectorCache succeeds");

    UiGeometryBlock loaded;
    float size = 0.0f;
    expectTrue(loadVectorCache(tmp, loaded, size), "loadVectorCache succeeds");
    expectNear(size, 16.0f, 0.001f, "cache preserves bake size");
    expectTrue(loaded.vertices.size() == src.vertices.size(), "cache preserves vertex count");
    expectTrue(loaded.indices.size() == src.indices.size(), "cache preserves index count");
    expectTrue(loaded.commands.size() == 1u, "cache synthesizes one command");
    if (!loaded.vertices.empty()) {
        expectTrue(loaded.vertices[0].rgba8 == src.vertices[0].rgba8,
                   "cache preserves packed vertex color");
    }
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
}

void testRegistryAndDraw() {
    VectorPath p;
    p.rect(0.0f, 0.0f, 32.0f, 32.0f);
    UiGeometryBlock block = fill(p, UiColor{1, 1, 1, 1});

    VectorIconRegistry& reg = VectorIconRegistry::global();
    reg.registerBaked("test_square", block, 32.0f);
    const VectorIcon* icon = reg.resolve("test_square");
    expectTrue(icon != nullptr, "registered icon resolves");
    expectTrue(reg.resolve("missing") == nullptr, "unknown icon resolves to null");

    if (icon != nullptr) {
        UiDrawList dl;
        dl.reset(UiVec2{200.0f, 200.0f});
        const std::size_t before = dl.data().vertices.size();
        dl.addVectorIcon("test_square", UiRect::fromXYWH(10.0f, 10.0f, 32.0f, 32.0f));
        expectTrue(dl.data().vertices.size() > before, "addVectorIcon emits geometry");
    }
}

void testTintedReplay() {
    // A single fully-white triangle block.
    UiGeometryBlock block;
    GeometryBlockMeshSink sink(block);
    const std::uint32_t white = UiColor{1, 1, 1, 1}.packAbgr8();
    const std::uint32_t i0 = sink.pushVertex(0, 0, white);
    const std::uint32_t i1 = sink.pushVertex(10, 0, white);
    const std::uint32_t i2 = sink.pushVertex(0, 10, white);
    sink.pushTriangle(i0, i1, i2);

    UiDrawList dl;
    dl.reset(UiVec2{100.0f, 100.0f});
    dl.appendCachedTinted(block, UiVec2{0.0f, 0.0f}, UiColor{1.0f, 0.0f, 0.0f, 1.0f});
    expectTrue(!dl.data().vertices.empty(), "tinted replay emits vertices");
    if (!dl.data().vertices.empty()) {
        // White multiplied by red → red (0xAABBGGRR == 0xFF0000FF).
        expectTrue(dl.data().vertices[0].rgba8 == 0xFF0000FFu,
                   "white tinted by red yields packed red");
    }
}

void testInlineSvgParse() {
    const char* svg =
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\" width=\"10\" "
        "height=\"10\"><rect x=\"0\" y=\"0\" width=\"10\" height=\"10\" fill=\"#FF0000\"/></svg>";
    SvgTessellateOptions opts;
    opts.targetSizePx = 20.0f;
    UiGeometryBlock block;
    const bool ok = tessellateSvgString(svg, opts, block);
    if (!ok) {
        std::cout << "[svg test] note: nanosvg unavailable, skipping SVG parse assertions\n";
        return;
    }
    expectTrue(!block.vertices.empty(), "inline SVG rect produces geometry");
    expectTrue(block.commands.size() == 1u, "SVG geometry uses one SolidColor command");
    // Find a full-alpha vertex and verify the color decoded as red (byte-order pin).
    bool foundRed = false;
    for (const UiVertex& v : block.vertices) {
        if (isFullAlpha(v.rgba8)) {
            foundRed = v.rgba8 == 0xFF0000FFu;
            break;
        }
    }
    expectTrue(foundRed, "SVG #FF0000 fill decodes to packed red (nanosvg byte order)");
    // The 10x10 doc scaled to 20px should roughly fill a 20x20 box.
    expectNear(solidArea(block), 400.0f, 4.0f, "scaled SVG rect solid area ~ 20x20");
}

}  // namespace

int main() {
    testFlattenToleranceMonotonic();
    testCircleVerticesOnRadius();
    testSquareFillAndFringe();
    testConcaveFillArea();
    testEvenOddDonutHole();
    testStrokeGeometryAndFringe();
    testCacheRoundTrip();
    testRegistryAndDraw();
    testTintedReplay();
    testInlineSvgParse();

    if (g_failures != 0) {
        std::cerr << "[svg test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[svg test] all checks passed\n";
    return 0;
}
