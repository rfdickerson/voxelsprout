// Headless tests for the odai_ui_editor document core: real `.ui.json` schema
// authoring, layout resolution, undo/redo, snapping and the editor operations.
//
// The load-bearing test here is testLoaderRoundTrip(): whatever the editor
// saves must instantiate through the *real* ui::UiDocumentLoader with the same
// tree shape and the same rects the canvas showed. That is what stops the
// editor from drifting into a parallel format again.

#include "tools/ui_editor/editor_color.h"
#include "tools/ui_editor/editor_document.h"
#include "tools/ui_editor/editor_history.h"
#include "tools/ui_editor/editor_ops.h"
#include "tools/ui_editor/editor_snap.h"

#include "ui/document/ui_document.h"
#include "ui/theme/ui_theme.h"
#include "ui/widget.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace {

using namespace odai;
using namespace odai::tools::ui_editor;
using json = nlohmann::json;

int g_failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[ui editor test] FAIL: " << message << '\n';
        ++g_failures;
    }
}

void expectNear(float actual, float expected, const char* message, float tol = 0.01f) {
    if (std::fabs(actual - expected) > tol) {
        std::cerr << "[ui editor test] FAIL: " << message << " (got " << actual << ", expected "
                  << expected << ")\n";
        ++g_failures;
    }
}

void expectEq(const std::string& actual, const std::string& expected, const char* message) {
    if (actual != expected) {
        std::cerr << "[ui editor test] FAIL: " << message << " (got \"" << actual
                  << "\", expected \"" << expected << "\")\n";
        ++g_failures;
    }
}

// --------------------------------------------------------------------------
// Document basics
// --------------------------------------------------------------------------

void testAddAndAddressNodes() {
    EditorDocument doc;
    doc.setViewport(ui::UiRect::fromXYWH(0, 0, 1280, 720));

    expectEq(doc.typeOf({}), "Panel", "root defaults to a Panel");
    expectTrue(doc.childCount({}) == 0, "new document has no children");

    NodePath panel;
    expectTrue(doc.addNode({}, "Panel", 40, 30, panel), "add Panel under root");
    expectTrue(panel == NodePath{0}, "first child sits at index 0");

    NodePath label;
    expectTrue(doc.addNode(panel, "Label", 12, 8, label), "add Label under the Panel");
    expectTrue((label == NodePath{0, 0}), "nested child path is {0,0}");
    expectEq(doc.typeOf(label), "Label", "nested node keeps its type");
    expectTrue(doc.childCount(panel) == 1, "panel reports one child");

    // Spacer declares acceptsChildren=false, matching the loader's factory.
    NodePath spacer;
    expectTrue(doc.addNode({}, "Spacer", 0, 0, spacer), "add Spacer");
    NodePath rejected;
    expectTrue(!doc.addNode(spacer, "Label", 0, 0, rejected),
               "Spacer refuses children, matching the loader");

    expectTrue(doc.allPaths().size() == 4, "allPaths walks root + 3 nodes");
    expectTrue(!doc.isValid(NodePath{9}), "out-of-range path is invalid");
}

void testUniqueIds() {
    EditorDocument doc;
    NodePath a, b, c;
    doc.addNode({}, "Panel", 0, 0, a);
    doc.addNode({}, "Panel", 0, 0, b);
    doc.addNode({}, "Panel", 0, 0, c);
    expectEq(doc.idOf(a), "panel", "first Panel gets the bare type id");
    expectEq(doc.idOf(b), "panel2", "second Panel is suffixed");
    expectEq(doc.idOf(c), "panel3", "third Panel keeps counting");
}

// --------------------------------------------------------------------------
// Layout
// --------------------------------------------------------------------------

void testAbsoluteRectAndPercentages() {
    EditorDocument doc;
    doc.setViewport(ui::UiRect::fromXYWH(0, 0, 1000, 800));
    doc.setProperty({}, "width", 1000);
    doc.setProperty({}, "height", 800);

    NodePath panel;
    doc.addNode({}, "Panel", 100, 50, panel);
    doc.setProperty(panel, "width", 400);
    doc.setProperty(panel, "height", 200);

    const ui::UiRect pr = doc.absoluteRect(panel);
    expectNear(pr.minX, 100.0f, "panel x");
    expectNear(pr.minY, 50.0f, "panel y");
    expectNear(pr.width(), 400.0f, "panel width");

    // A percentage child resolves against the parent's span, like the loader.
    NodePath child;
    doc.addNode(panel, "Label", 0, 0, child);
    doc.setProperty(child, "x", "50%");
    doc.setProperty(child, "width", "25%");
    const ui::UiRect cr = doc.absoluteRect(child);
    expectNear(cr.minX, 100.0f + 200.0f, "50% of a 400px parent, offset by parent x");
    expectNear(cr.width(), 100.0f, "25% of a 400px parent");

    // A node with no width/height fills its parent, matching resolveRect().
    NodePath filler;
    doc.addNode(panel, "Panel", 0, 0, filler);
    doc.eraseProperty(filler, "width");
    doc.eraseProperty(filler, "height");
    const ui::UiRect fr = doc.absoluteRect(filler);
    expectNear(fr.width(), 400.0f, "omitted width fills the parent");
    expectNear(fr.height(), 200.0f, "omitted height fills the parent");
}

void testSetAbsoluteRectPreservesUnits() {
    EditorDocument doc;
    doc.setViewport(ui::UiRect::fromXYWH(0, 0, 1000, 800));
    doc.setProperty({}, "width", 1000);
    doc.setProperty({}, "height", 800);

    NodePath pct;
    doc.addNode({}, "Panel", 0, 0, pct);
    doc.setProperty(pct, "x", "10%");
    doc.setProperty(pct, "width", "50%");

    doc.translate(pct, 100.0f, 0.0f);
    const json* n = doc.nodeAt(pct);
    expectTrue(n != nullptr && n->at("x").is_string(),
               "a percentage-authored x stays a percentage after a nudge");
    expectNear(doc.absoluteRect(pct).minX, 200.0f, "10% of 1000 plus a 100px nudge");
    expectTrue(n != nullptr && n->at("width").is_string(), "width stays a percentage too");

    // A pixel-authored field stays pixels.
    NodePath px;
    doc.addNode({}, "Panel", 20, 20, px);
    doc.translate(px, 5.0f, 5.0f);
    const json* pn = doc.nodeAt(px);
    expectTrue(pn != nullptr && pn->at("x").is_number(), "a pixel-authored x stays a number");
    expectNear(doc.absoluteRect(px).minX, 25.0f, "pixel nudge applies directly");
}

void testHitTestPrefersTopmostDeepest() {
    EditorDocument doc;
    doc.setViewport(ui::UiRect::fromXYWH(0, 0, 500, 500));
    doc.setProperty({}, "width", 500);
    doc.setProperty({}, "height", 500);

    NodePath back, front, nested;
    doc.addNode({}, "Panel", 0, 0, back);
    doc.setProperty(back, "width", 300);
    doc.setProperty(back, "height", 300);
    doc.addNode({}, "Panel", 100, 100, front);
    doc.setProperty(front, "width", 300);
    doc.setProperty(front, "height", 300);
    doc.addNode(front, "Label", 10, 10, nested);
    doc.setProperty(nested, "width", 50);
    doc.setProperty(nested, "height", 20);

    NodePath hit;
    expectTrue(doc.hitTest(150, 150, hit), "point inside overlapping panels hits something");
    expectTrue(hit == front, "later sibling wins the overlap");

    expectTrue(doc.hitTest(120, 115, hit), "point inside the nested label hits");
    expectTrue(hit == nested, "deepest node wins over its parent");

    expectTrue(doc.hitTest(450, 450, hit), "empty canvas area still hits the root");
    expectTrue(hit.empty(), "root path is empty");

    expectTrue(!doc.hitTest(-10, -10, hit), "outside the root hits nothing");
}

void testStackChildrenAreMarked() {
    EditorDocument doc;
    NodePath stack, child;
    doc.addNode({}, "HorizontalStack", 0, 0, stack);
    doc.addNode(stack, "Label", 0, 0, child);
    expectTrue(doc.parentControlsLayout(child),
               "a stack child is flagged as runtime-laid-out");
    expectTrue(!doc.parentControlsLayout(stack), "the stack itself is not");
}

// --------------------------------------------------------------------------
// Reparenting
// --------------------------------------------------------------------------

void testReparentPreservesScreenPosition() {
    EditorDocument doc;
    doc.setViewport(ui::UiRect::fromXYWH(0, 0, 1000, 1000));
    doc.setProperty({}, "width", 1000);
    doc.setProperty({}, "height", 1000);

    NodePath host, loose;
    doc.addNode({}, "Panel", 200, 200, host);
    doc.setProperty(host, "width", 400);
    doc.setProperty(host, "height", 400);
    doc.addNode({}, "Button", 250, 260, loose);

    const ui::UiRect before = doc.absoluteRect(loose);
    NodePath moved;
    expectTrue(doc.reparent(loose, host, -1, moved), "reparent into the host panel");
    expectTrue((moved == NodePath{0, 0}), "button is now the host's first child");
    const ui::UiRect after = doc.absoluteRect(moved);
    expectNear(after.minX, before.minX, "reparent keeps the on-screen x");
    expectNear(after.minY, before.minY, "reparent keeps the on-screen y");
    const json* n = doc.nodeAt(moved);
    expectNear(n->at("x").get<float>(), 50.0f, "x is rewritten relative to the new parent");

    NodePath bad;
    expectTrue(!doc.reparent(moved, moved, -1, bad), "a node cannot be reparented into itself");
    expectTrue(!doc.reparent(host, moved, -1, bad),
               "a node cannot be reparented into its own descendant");
    expectTrue(!doc.reparent({}, host, -1, bad), "the root cannot be reparented");
}

// --------------------------------------------------------------------------
// Serialization + the real loader
// --------------------------------------------------------------------------

void testUnknownFieldsSurviveRoundTrip() {
    const std::string src = R"({
      "version": 1, "type": "Panel", "id": "root", "width": 800, "height": 600,
      "children": [
        { "type": "Label", "id": "hp", "x": 10, "y": 10, "width": 100, "height": 20,
          "text": "{unit.hp}", "customAppField": {"nested": [1, 2, 3]} }
      ]
    })";
    EditorDocument doc;
    std::string err;
    expectTrue(doc.loadFromString(src, &err), "load a document with app-specific fields");

    doc.translate(NodePath{0}, 5.0f, 0.0f);

    EditorDocument reloaded;
    expectTrue(reloaded.loadFromString(doc.toString(), &err), "reload what we saved");
    const json* n = reloaded.nodeAt(NodePath{0});
    expectTrue(n != nullptr && n->contains("customAppField"),
               "a field the editor has no inspector for still survives the round-trip");
    expectTrue(n != nullptr && n->at("customAppField").at("nested").size() == 3,
               "the unknown field's value is preserved verbatim");
    expectEq(n->value("text", ""), "{unit.hp}", "binding expressions are not mangled");
}

void testRejectsMalformedJson() {
    EditorDocument doc;
    NodePath p;
    doc.addNode({}, "Panel", 10, 10, p);
    const std::string before = doc.toString();

    std::string err;
    expectTrue(!doc.loadFromString("{ not json", &err), "malformed JSON is rejected");
    expectTrue(!err.empty(), "the parse error is reported");
    expectTrue(!doc.loadFromString("[1,2,3]", &err), "a non-object root is rejected");
    expectEq(doc.toString(), before, "a failed load leaves the document untouched");
}

// Author a document in the editor, save it, and instantiate the saved text
// through the real ui::UiDocumentLoader — the tree shape, ids and rects the
// loader produces must match what the editor canvas showed.
void testLoaderRoundTrip() {
    EditorDocument doc;
    const ui::UiRect viewport = ui::UiRect::fromXYWH(0, 0, 1280, 720);
    doc.setViewport(viewport);
    doc.setProperty({}, "width", 1280);
    doc.setProperty({}, "height", 720);

    NodePath panel;
    doc.addNode({}, "Panel", 60, 40, panel);
    doc.setProperty(panel, "width", 420);
    doc.setProperty(panel, "height", 300);
    doc.setProperty(panel, "id", "city_panel");

    NodePath title;
    doc.addNode(panel, "Label", 12, 12, title);
    doc.setProperty(title, "width", "50%");
    doc.setProperty(title, "height", 28);
    doc.setProperty(title, "id", "title");
    doc.setProperty(title, "text", "<b>Balmora</b>");

    NodePath confirm;
    doc.addNode(panel, "Button", 12, 240, confirm);
    doc.setProperty(confirm, "id", "confirm");
    doc.setProperty(confirm, "label", "Confirm");
    doc.setProperty(confirm, "on_click", "city.confirm");

    // ── Save, then load through the production loader ────────────────────────
    const std::string saved = doc.toString();
    json parsed;
    try {
        parsed = json::parse(saved);
    } catch (...) {
        expectTrue(false, "editor output is valid JSON");
        return;
    }

    ui::UiTheme theme;  // headless: no fonts baked, only structure is under test
    const ui::UiDocumentLoader loader(theme);
    std::unique_ptr<ui::Widget> rootWidget = loader.instantiate(parsed, {}, viewport);

    expectTrue(rootWidget != nullptr, "the loader instantiates the editor's output");
    if (!rootWidget) return;

    expectEq(rootWidget->id, "root", "root id survives");
    expectTrue(rootWidget->children().size() == 1, "root has the authored panel");
    if (rootWidget->children().empty()) return;

    const ui::Widget& panelW = *rootWidget->children()[0];
    expectEq(panelW.id, "city_panel", "panel id survives");
    expectTrue(panelW.children().size() == 2, "panel has both authored children");

    const ui::UiRect editorPanelRect = doc.absoluteRect(panel);
    expectNear(panelW.rect().minX, editorPanelRect.minX, "panel x matches the editor canvas");
    expectNear(panelW.rect().minY, editorPanelRect.minY, "panel y matches the editor canvas");
    expectNear(panelW.rect().width(), editorPanelRect.width(), "panel width matches");
    expectNear(panelW.rect().height(), editorPanelRect.height(), "panel height matches");

    if (panelW.children().size() < 2) return;
    const ui::Widget& titleW = *panelW.children()[0];
    const ui::UiRect editorTitleRect = doc.absoluteRect(title);
    expectEq(titleW.id, "title", "label id survives");
    expectNear(titleW.rect().minX, editorTitleRect.minX, "percentage-sized label x matches");
    expectNear(titleW.rect().width(), editorTitleRect.width(),
               "percentage-sized label width matches (50% of 420)");
    expectNear(titleW.rect().width(), 210.0f, "50% of the 420px panel resolves to 210px");

    const ui::Widget& confirmW = *panelW.children()[1];
    expectEq(confirmW.id, "confirm", "button id survives");
    expectEq(confirmW.slotName, "city.confirm", "on_click becomes the widget's slot name");
    expectNear(confirmW.rect().minY, doc.absoluteRect(confirm).minY, "button y matches");
}

// The editor must be able to open the engine's own shipped document, keep its
// structure, and hand it back byte-identical when nothing was edited. Skipped
// when the assets dir isn't wired in (a bare compiler invocation).
void testOpensShippedSampleDocument() {
#ifdef ODAI_TEST_ASSETS_DIR
    const std::string path = std::string(ODAI_TEST_ASSETS_DIR) + "/ui/docs/city_panel.ui.json";
    EditorDocument doc;
    std::string err;
    if (!doc.loadFile(path, &err)) {
        expectTrue(false, "load assets/ui/docs/city_panel.ui.json");
        return;
    }
    doc.setViewport(ui::UiRect::fromXYWH(0, 0, 420, 600));

    expectEq(doc.idOf({}), "city_panel", "sample root id");
    expectTrue(doc.childCount({}) == 7, "sample has seven top-level children");

    // The stack's children are authored without x/y, so they fill it — and the
    // editor flags them as runtime-positioned rather than claiming otherwise.
    NodePath yieldRow{2};
    expectEq(doc.typeOf(yieldRow), "HorizontalStack", "third child is the yield row");
    expectTrue(doc.childCount(yieldRow) == 4, "yield row has four labels");
    expectTrue(doc.parentControlsLayout(NodePath{2, 0}), "yield row children are stack-managed");

    const ui::UiRect health = doc.absoluteRect(NodePath{3});
    expectNear(health.minX, 12.0f, "health bar x from the real document");
    expectNear(health.width(), 396.0f, "health bar width from the real document");

    // An untouched load->save must not perturb the file's content.
    EditorDocument reloaded;
    expectTrue(reloaded.loadFromString(doc.toString(), &err), "editor output reloads");
    expectEq(reloaded.toString(), doc.toString(), "load->save->load is stable");

    // And it still instantiates through the production loader.
    ui::UiTheme theme;
    const ui::UiDocumentLoader loader(theme);
    std::unique_ptr<ui::Widget> w = loader.instantiate(json::parse(doc.toString()), {},
                                                      ui::UiRect{0.0f, 0.0f, 420.0f, 600.0f});
    expectTrue(w != nullptr && w->children().size() == 7,
               "the saved sample still loads into a widget tree");
#endif
}

// --------------------------------------------------------------------------
// Undo / redo
// --------------------------------------------------------------------------

void testUndoRedo() {
    EditorDocument doc;
    EditorHistory history(doc);

    expectTrue(!history.canUndo(), "a fresh history has nothing to undo");
    expectTrue(!history.canRedo(), "a fresh history has nothing to redo");

    NodePath a;
    doc.addNode({}, "Panel", 10, 10, a);
    history.commit(doc, "add Panel");
    NodePath b;
    doc.addNode({}, "Button", 20, 20, b);
    history.commit(doc, "add Button");
    expectTrue(doc.childCount({}) == 2, "two nodes authored");

    expectTrue(history.canUndo(), "undo is available");
    expectEq(history.undoLabel(), "add Button", "undo label names the last edit");

    expectTrue(history.undo(doc), "undo the second add");
    expectTrue(doc.childCount({}) == 1, "one node left after undo");
    expectTrue(history.undo(doc), "undo the first add");
    expectTrue(doc.childCount({}) == 0, "back to the empty document");
    expectTrue(!history.canUndo(), "nothing left to undo");

    expectTrue(history.redo(doc), "redo the first add");
    expectTrue(doc.childCount({}) == 1, "redo restores the node");
    expectEq(doc.typeOf(NodePath{0}), "Panel", "redo restores the right node");
    expectTrue(history.redo(doc), "redo the second add");
    expectTrue(doc.childCount({}) == 2, "both nodes back");
    expectTrue(!history.canRedo(), "redo stack exhausted");
}

void testUndoBranchTruncatesRedo() {
    EditorDocument doc;
    EditorHistory history(doc);
    NodePath a, b, c;
    doc.addNode({}, "Panel", 0, 0, a);
    history.commit(doc, "add Panel");
    doc.addNode({}, "Button", 0, 0, b);
    history.commit(doc, "add Button");

    history.undo(doc);
    expectTrue(history.canRedo(), "redo is available after an undo");
    doc.addNode({}, "Label", 0, 0, c);
    history.commit(doc, "add Label");
    expectTrue(!history.canRedo(), "a new edit after an undo discards the redo branch");
    expectEq(doc.typeOf(NodePath{1}), "Label", "the new branch is what stuck");
}

void testNoOpCommitIsDropped() {
    EditorDocument doc;
    EditorHistory history(doc);
    NodePath a;
    doc.addNode({}, "Panel", 0, 0, a);
    history.commit(doc, "add Panel");
    const std::size_t size = history.size();
    history.commit(doc, "drag that moved nothing");
    expectTrue(history.size() == size, "a commit with no state change adds no undo step");
}

// --------------------------------------------------------------------------
// Snapping
// --------------------------------------------------------------------------

void testSnapToGridAndEdges() {
    EditorDocument doc;
    doc.setViewport(ui::UiRect::fromXYWH(0, 0, 1000, 1000));
    doc.setProperty({}, "width", 1000);
    doc.setProperty({}, "height", 1000);

    NodePath anchor, mover;
    doc.addNode({}, "Panel", 300, 200, anchor);
    doc.setProperty(anchor, "width", 200);
    doc.setProperty(anchor, "height", 100);
    doc.addNode({}, "Panel", 0, 0, mover);
    doc.setProperty(mover, "width", 80);
    doc.setProperty(mover, "height", 40);

    SnapSettings s;
    s.grid = true;
    s.gridSize = 10.0f;
    s.edges = true;
    s.threshold = 8.0f;

    SnapEngine snap;
    snap.collectTargets(doc, {mover});

    // Left edge 3px away from the anchor's left edge: edge snap wins.
    const ui::UiVec2 snapped = snap.snapPosition(ui::UiRect::fromXYWH(303, 500, 80, 40), s);
    expectNear(snapped.x, 300.0f, "x snaps to the anchor's left edge");
    expectTrue(!snap.lines().empty(), "an edge snap records a guide line");

    // Grid-only fallback: nothing to align to near y=497 but the 10px grid.
    const ui::UiVec2 gridSnapped = snap.snapPosition(ui::UiRect::fromXYWH(697, 803, 80, 40), s);
    expectNear(gridSnapped.y, 800.0f, "y falls back to the pixel grid");

    // Far from everything and off the grid by more than the threshold.
    s.grid = false;
    const ui::UiVec2 free = snap.snapPosition(ui::UiRect::fromXYWH(654, 733, 80, 40), s);
    expectNear(free.x, 654.0f, "no snap target leaves x untouched");
    expectTrue(snap.lines().empty(), "no guide lines when nothing snapped");
}

void testSnapExcludesSelfAndDescendants() {
    EditorDocument doc;
    doc.setViewport(ui::UiRect::fromXYWH(0, 0, 1000, 1000));
    doc.setProperty({}, "width", 1000);
    doc.setProperty({}, "height", 1000);

    NodePath parent, child;
    doc.addNode({}, "Panel", 400, 400, parent);
    doc.setProperty(parent, "width", 200);
    doc.setProperty(parent, "height", 200);
    doc.addNode(parent, "Label", 0, 0, child);
    doc.setProperty(child, "width", 60);
    doc.setProperty(child, "height", 30);

    SnapSettings s;
    s.grid = false;
    s.edges = true;
    s.threshold = 8.0f;

    SnapEngine snap;
    snap.collectTargets(doc, {parent});
    // 402 is 2px from the excluded node's own left/top edge. The moving rect is
    // small and sits well away from the root canvas's own edges and centre, so
    // the excluded node's edges are the only candidates in range — and they must
    // not fire.
    const ui::UiVec2 out = snap.snapPosition(ui::UiRect::fromXYWH(402, 402, 40, 40), s);
    expectNear(out.x, 402.0f, "a node does not snap to itself");
    expectNear(out.y, 402.0f, "nor to its own descendants");
    expectTrue(snap.lines().empty(), "and records no guide line for the excluded node");
}

// --------------------------------------------------------------------------
// Editor operations
// --------------------------------------------------------------------------

void testAlignAndDistribute() {
    EditorDocument doc;
    doc.setViewport(ui::UiRect::fromXYWH(0, 0, 1000, 1000));
    doc.setProperty({}, "width", 1000);
    doc.setProperty({}, "height", 1000);

    Selection sel;
    for (int i = 0; i < 3; ++i) {
        NodePath p;
        doc.addNode({}, "Panel", 10.0f + static_cast<float>(i) * 37.0f,
                    20.0f + static_cast<float>(i) * 11.0f, p);
        doc.setProperty(p, "width", 40);
        doc.setProperty(p, "height", 40);
        sel.add(p);
    }

    expectTrue(alignSelection(doc, sel, AlignEdge::Left), "align left moves something");
    for (const NodePath& p : sel.paths()) {
        expectNear(doc.absoluteRect(p).minX, 10.0f, "all left edges match the bounds' left");
    }

    expectTrue(alignSelection(doc, sel, AlignEdge::Top), "align top moves something");
    for (const NodePath& p : sel.paths()) {
        expectNear(doc.absoluteRect(p).minY, 20.0f, "all top edges match the bounds' top");
    }

    // Spread them back out, then distribute to equal gaps.
    doc.setAbsoluteRect(NodePath{0}, ui::UiRect::fromXYWH(0, 0, 40, 40));
    doc.setAbsoluteRect(NodePath{1}, ui::UiRect::fromXYWH(50, 0, 40, 40));
    doc.setAbsoluteRect(NodePath{2}, ui::UiRect::fromXYWH(340, 0, 40, 40));
    expectTrue(distributeSelection(doc, sel, /*horizontal=*/true), "distribute moves the middle");
    const float gapA = doc.absoluteRect(NodePath{1}).minX - doc.absoluteRect(NodePath{0}).maxX;
    const float gapB = doc.absoluteRect(NodePath{2}).minX - doc.absoluteRect(NodePath{1}).maxX;
    expectNear(gapA, gapB, "gaps between adjacent nodes are equal");
    expectNear(doc.absoluteRect(NodePath{0}).minX, 0.0f, "the first node stays put");
    expectNear(doc.absoluteRect(NodePath{2}).minX, 340.0f, "the last node stays put");

    Selection two;
    two.add(NodePath{0});
    two.add(NodePath{1});
    expectTrue(!distributeSelection(doc, two, true), "distribute needs at least three nodes");

    Selection one;
    one.add(NodePath{0});
    expectTrue(!alignSelection(doc, one, AlignEdge::Left), "align needs at least two nodes");
}

void testNudgeAndDelete() {
    EditorDocument doc;
    Selection sel;
    NodePath a, b;
    doc.addNode({}, "Panel", 10, 10, a);
    doc.addNode({}, "Panel", 100, 10, b);
    sel.add(a);
    sel.add(b);

    expectTrue(nudgeSelection(doc, sel, 1.0f, -1.0f), "nudge reports a move");
    expectNear(doc.absoluteRect(a).minX, 11.0f, "first node nudged");
    expectNear(doc.absoluteRect(b).minY, 9.0f, "second node nudged");
    expectTrue(!nudgeSelection(doc, sel, 0.0f, 0.0f), "a zero nudge is a no-op");

    expectTrue(deleteSelection(doc, sel), "delete removes the selection");
    expectTrue(doc.childCount({}) == 0, "both nodes gone");
    expectTrue(sel.empty(), "selection is cleared after a delete");
}

void testDeleteHandlesNestedAndSiblingPaths() {
    EditorDocument doc;
    NodePath p0, p1, p2, nested;
    doc.addNode({}, "Panel", 0, 0, p0);
    doc.addNode({}, "Panel", 0, 0, p1);
    doc.addNode({}, "Panel", 0, 0, p2);
    doc.addNode(p1, "Label", 0, 0, nested);
    doc.setProperty(p2, "id", "survivor");

    // Deleting an earlier sibling would shift a later one; the sweep must order
    // its removals so every path stays valid.
    Selection sel;
    sel.add(p0);
    sel.add(nested);
    sel.add(p1);
    expectTrue(deleteSelection(doc, sel), "mixed nested/sibling delete succeeds");
    expectTrue(doc.childCount({}) == 1, "exactly one node survives");
    expectEq(doc.idOf(NodePath{0}), "survivor", "the right node survived");
}

void testCopyPasteAndDuplicateFreshenIds() {
    EditorDocument doc;
    NodePath panel, label;
    doc.addNode({}, "Panel", 10, 10, panel);
    doc.setProperty(panel, "id", "hud");
    doc.addNode(panel, "Label", 4, 4, label);
    doc.setProperty(label, "id", "hud_title");

    Selection sel;
    sel.set(panel);
    const std::vector<json> clip = copySelection(doc, sel);
    expectTrue(clip.size() == 1, "one subtree copied");

    Selection pasted;
    expectTrue(pasteInto(doc, {}, clip, 20.0f, 20.0f, pasted), "paste under the root");
    expectTrue(doc.childCount({}) == 2, "the paste added a sibling");
    expectTrue(pasted.size() == 1, "the pasted node is selected");

    const NodePath copyPath = pasted.paths().front();
    expectTrue(doc.idOf(copyPath) != "hud", "the pasted root got a fresh id");
    expectTrue(doc.idOf(NodePath{copyPath[0], 0}) != "hud_title",
               "the pasted child got a fresh id too");
    expectNear(doc.absoluteRect(copyPath).minX, 30.0f, "paste applies the offset");

    Selection dup;
    dup.set(NodePath{0});
    expectTrue(duplicateSelection(doc, dup, 8.0f, 8.0f), "duplicate the original");
    expectTrue(doc.childCount({}) == 3, "duplicate added a third node");

    // Every id in the document must still be unique.
    std::vector<std::string> ids;
    for (const NodePath& p : doc.allPaths()) {
        const std::string id = doc.idOf(p);
        if (id.empty()) continue;
        expectTrue(std::find(ids.begin(), ids.end(), id) == ids.end(),
                   "no duplicate ids after paste + duplicate");
        ids.push_back(id);
    }

    Selection rootSel;
    rootSel.set(NodePath{});
    expectTrue(copySelection(doc, rootSel).empty(), "the root is not copyable");
}

void testReorderSiblings() {
    EditorDocument doc;
    NodePath a, b, c;
    doc.addNode({}, "Panel", 0, 0, a);
    doc.setProperty(a, "id", "a");
    doc.addNode({}, "Panel", 0, 0, b);
    doc.setProperty(b, "id", "b");
    doc.addNode({}, "Panel", 0, 0, c);
    doc.setProperty(c, "id", "c");

    NodePath moved;
    expectTrue(reorderSibling(doc, NodePath{0}, +1, moved), "move 'a' one step later");
    expectEq(doc.idOf(NodePath{0}), "b", "'b' took the front slot");
    expectEq(doc.idOf(NodePath{1}), "a", "'a' moved back one");
    expectTrue(!reorderSibling(doc, NodePath{2}, +1, moved), "cannot move past the end");
    expectTrue(!reorderSibling(doc, NodePath{0}, -1, moved), "cannot move before the start");
}

void testSelectionPrune() {
    EditorDocument doc;
    Selection sel;
    NodePath a, b;
    doc.addNode({}, "Panel", 0, 0, a);
    doc.addNode({}, "Panel", 0, 0, b);
    sel.add(a);
    sel.add(b);
    sel.toggle(a);
    expectTrue(sel.size() == 1, "toggle removes an already-selected path");
    sel.toggle(a);
    expectTrue(sel.size() == 2, "toggle re-adds it");

    doc.removeNode(b);
    sel.prune(doc);
    expectTrue(sel.size() == 1, "prune drops paths that no longer resolve");
}

// --------------------------------------------------------------------------
// Color theory
// --------------------------------------------------------------------------

void expectColorNear(const ui::UiColor& actual, const ui::UiColor& expected, const char* message,
                     float tol = 0.005f) {
    if (std::fabs(actual.r - expected.r) > tol || std::fabs(actual.g - expected.g) > tol ||
        std::fabs(actual.b - expected.b) > tol || std::fabs(actual.a - expected.a) > tol) {
        std::cerr << "[ui editor test] FAIL: " << message << " (got " << actual.r << "," << actual.g
                  << "," << actual.b << "," << actual.a << " expected " << expected.r << ","
                  << expected.g << "," << expected.b << "," << expected.a << ")\n";
        ++g_failures;
    }
}

void testHsvKnownValues() {
    const Hsv red = rgbToHsv(ui::UiColor{1, 0, 0, 1});
    expectNear(red.h, 0.0f, "pure red hue");
    expectNear(red.s, 1.0f, "pure red saturation");
    expectNear(red.v, 1.0f, "pure red value");

    const Hsv green = rgbToHsv(ui::UiColor{0, 1, 0, 1});
    expectNear(green.h, 120.0f, "pure green hue");
    const Hsv blue = rgbToHsv(ui::UiColor{0, 0, 1, 1});
    expectNear(blue.h, 240.0f, "pure blue hue");

    // Achromatic: saturation zero, hue reported as 0 rather than NaN.
    const Hsv grey = rgbToHsv(ui::UiColor{0.5f, 0.5f, 0.5f, 1});
    expectNear(grey.s, 0.0f, "grey has no saturation");
    expectNear(grey.h, 0.0f, "grey hue is reported as zero, not NaN");
    expectNear(grey.v, 0.5f, "grey value is its channel level");

    const Hsv black = rgbToHsv(ui::UiColor{0, 0, 0, 1});
    expectNear(black.s, 0.0f, "black has no saturation (no divide by zero)");
    expectNear(black.v, 0.0f, "black value is zero");

    expectColorNear(hsvToRgb(Hsv{120.0f, 1.0f, 1.0f}), ui::UiColor{0, 1, 0, 1}, "hsv->rgb green");
    expectColorNear(hsvToRgb(Hsv{60.0f, 1.0f, 1.0f}), ui::UiColor{1, 1, 0, 1}, "hsv->rgb yellow");
    expectColorNear(hsvToRgb(Hsv{0.0f, 0.0f, 0.75f}), ui::UiColor{0.75f, 0.75f, 0.75f, 1},
                    "hsv->rgb grey");
}

void testHsvRoundTrip() {
    const ui::UiColor samples[] = {
        {0.85f, 0.72f, 0.38f, 1.00f}, {0.05f, 0.10f, 0.15f, 0.88f},
        {0.20f, 0.60f, 1.00f, 0.50f}, {1.00f, 1.00f, 1.00f, 1.00f},
        {0.00f, 0.00f, 0.00f, 1.00f}, {0.33f, 0.11f, 0.77f, 0.25f},
    };
    for (const ui::UiColor& c : samples) {
        const Hsv hsv = rgbToHsv(c);
        expectColorNear(hsvToRgb(hsv, c.a), c, "rgb -> hsv -> rgb round-trip");
    }
}

void testWrapHue() {
    expectNear(wrapHue(0.0f), 0.0f, "wrapHue(0)");
    expectNear(wrapHue(360.0f), 0.0f, "wrapHue(360) wraps to 0");
    expectNear(wrapHue(390.0f), 30.0f, "wrapHue past a full turn");
    expectNear(wrapHue(-30.0f), 330.0f, "wrapHue of a negative angle");
    expectNear(wrapHue(-390.0f), 330.0f, "wrapHue of more than a negative turn");
}

void testHarmonySchemes() {
    // A saturated mid-value base so hue rotation is unambiguous.
    const ui::UiColor base = hsvToRgb(Hsv{30.0f, 0.8f, 0.9f}, 0.75f);

    auto hueOf = [](const ui::UiColor& c) { return rgbToHsv(c).h; };

    const std::vector<ui::UiColor> comp = harmonyColors(base, Harmony::Complementary);
    expectTrue(comp.size() == 2, "complementary yields two colors");
    expectColorNear(comp[0], base, "index 0 is always the base color");
    expectNear(hueOf(comp[1]), 210.0f, "complement sits opposite the base hue");

    const std::vector<ui::UiColor> triad = harmonyColors(base, Harmony::Triad);
    expectTrue(triad.size() == 3, "triad yields three colors");
    expectNear(hueOf(triad[1]), 150.0f, "triad second hue is +120");
    expectNear(hueOf(triad[2]), 270.0f, "triad third hue is +240");

    const std::vector<ui::UiColor> split = harmonyColors(base, Harmony::SplitComplementary);
    expectNear(hueOf(split[1]), 180.0f, "split-complement flanks the opposite hue below");
    expectNear(hueOf(split[2]), 240.0f, "split-complement flanks the opposite hue above");

    const std::vector<ui::UiColor> analogous = harmonyColors(base, Harmony::Analogous);
    expectNear(hueOf(analogous[1]), 0.0f, "analogous neighbour 30 deg below");
    expectNear(hueOf(analogous[2]), 60.0f, "analogous neighbour 30 deg above");

    const std::vector<ui::UiColor> square = harmonyColors(base, Harmony::Square);
    expectTrue(square.size() == 4, "square yields four colors");
    expectNear(hueOf(square[2]), 210.0f, "square third hue is the complement");

    // Hue rotation must preserve saturation, value and alpha, or the partners
    // stop reading as the same palette.
    const Hsv baseHsv = rgbToHsv(base);
    for (const ui::UiColor& c : triad) {
        const Hsv h = rgbToHsv(c);
        expectNear(h.s, baseHsv.s, "rotated hue keeps the base saturation");
        expectNear(h.v, baseHsv.v, "rotated hue keeps the base value");
        expectNear(c.a, base.a, "rotated hue keeps the base alpha");
    }

    // The two ramps hold hue and sweep one axis instead.
    const std::vector<ui::UiColor> mono = harmonyColors(base, Harmony::Monochromatic);
    expectTrue(mono.size() == 6, "monochromatic yields the base plus a five-step ramp");
    for (std::size_t i = 2; i < mono.size(); ++i) {
        expectTrue(rgbToHsv(mono[i]).v > rgbToHsv(mono[i - 1]).v,
                   "monochromatic ramp increases in value");
    }
    expectNear(rgbToHsv(mono[3]).h, baseHsv.h, "monochromatic ramp holds the hue");

    const std::vector<ui::UiColor> shades = harmonyColors(base, Harmony::Shades);
    for (std::size_t i = 2; i < shades.size(); ++i) {
        expectTrue(rgbToHsv(shades[i]).s > rgbToHsv(shades[i - 1]).s,
                   "shades ramp increases in saturation");
    }
    expectNear(rgbToHsv(shades[3]).v, baseHsv.v, "shades ramp holds the value");

    // Every scheme must be named and non-empty, so the inspector can list them.
    for (const Harmony h : allHarmonies()) {
        expectTrue(!harmonyColors(base, h).empty(), "every scheme produces colors");
        expectTrue(!harmonyName(h).empty(), "every scheme has a name");
    }
    expectTrue(allHarmonies().size() == 8, "all eight schemes are listed");
}

void testContrastRatio() {
    const ui::UiColor white{1, 1, 1, 1};
    const ui::UiColor black{0, 0, 0, 1};

    expectNear(contrastRatio(white, black), 21.0f, "black on white is the 21:1 maximum", 0.02f);
    expectNear(contrastRatio(white, white), 1.0f, "a color against itself is 1:1");
    expectNear(contrastRatio(black, white), 21.0f, "contrast is symmetric", 0.02f);

    expectNear(relativeLuminance(white), 1.0f, "white luminance is 1");
    expectNear(relativeLuminance(black), 0.0f, "black luminance is 0");
    // Mid grey #777777 sits near 0.18 relative luminance — the check that the
    // piecewise sRGB decode is actually being applied rather than a raw value.
    const float mid = relativeLuminance(ui::UiColor{0.4667f, 0.4667f, 0.4667f, 1.0f});
    expectNear(mid, 0.1845f, "mid grey luminance uses the sRGB decode, not the raw channel", 0.01f);

    expectEq(std::string(contrastRating(21.0f)), "AAA", "21:1 rates AAA");
    expectEq(std::string(contrastRating(4.6f)), "AA", "4.6:1 rates AA");
    expectEq(std::string(contrastRating(3.2f)), "AA Large", "3.2:1 rates AA Large");
    expectEq(std::string(contrastRating(2.0f)), "fail", "2:1 fails");
}

void testCompositeOver() {
    const ui::UiColor opaqueRed{1, 0, 0, 1};
    const ui::UiColor white{1, 1, 1, 1};
    expectColorNear(compositeOver(opaqueRed, white), opaqueRed, "an opaque source replaces the dst");

    const ui::UiColor halfBlack{0, 0, 0, 0.5f};
    expectColorNear(compositeOver(halfBlack, white), ui::UiColor{0.5f, 0.5f, 0.5f, 1.0f},
                    "50% black over white is mid grey");

    const ui::UiColor clear{0, 1, 0, 0};
    expectColorNear(compositeOver(clear, white), white, "a fully transparent source is a no-op");

    // A translucent light-grey label would score as high-contrast on its raw
    // values, but composited onto its real backdrop it does not.
    const ui::UiColor faintText{0.9f, 0.9f, 0.9f, 0.25f};
    const ui::UiColor panel{0.85f, 0.85f, 0.85f, 1.0f};
    const float raw = contrastRatio(faintText, panel);
    const float composited = contrastRatio(compositeOver(faintText, panel), panel);
    expectTrue(composited < raw + 0.001f,
               "compositing never overstates a translucent color's contrast");
    expectTrue(composited < 1.2f, "a faint label on a light panel scores as unreadable");
}

}  // namespace

int main() {
    testAddAndAddressNodes();
    testUniqueIds();
    testAbsoluteRectAndPercentages();
    testSetAbsoluteRectPreservesUnits();
    testHitTestPrefersTopmostDeepest();
    testStackChildrenAreMarked();
    testReparentPreservesScreenPosition();
    testUnknownFieldsSurviveRoundTrip();
    testRejectsMalformedJson();
    testLoaderRoundTrip();
    testOpensShippedSampleDocument();
    testUndoRedo();
    testUndoBranchTruncatesRedo();
    testNoOpCommitIsDropped();
    testSnapToGridAndEdges();
    testSnapExcludesSelfAndDescendants();
    testAlignAndDistribute();
    testNudgeAndDelete();
    testDeleteHandlesNestedAndSiblingPaths();
    testCopyPasteAndDuplicateFreshenIds();
    testReorderSiblings();
    testSelectionPrune();
    testHsvKnownValues();
    testHsvRoundTrip();
    testWrapHue();
    testHarmonySchemes();
    testContrastRatio();
    testCompositeOver();

    if (g_failures != 0) {
        std::cerr << "[ui editor test] " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "[ui editor test] all checks passed\n";
    return 0;
}
