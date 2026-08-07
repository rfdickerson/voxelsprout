#pragma once
#include "engine/game_app.h"
#include "render/renderer_types.h"
#include "tools/ui_editor/editor_color.h"
#include "tools/ui_editor/editor_document.h"
#include "tools/ui_editor/editor_history.h"
#include "tools/ui_editor/editor_ops.h"
#include "tools/ui_editor/editor_snap.h"
#include "ui/ui_types.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace odai::tools::ui_editor {

// A click target the properties panel / outliner draws by hand into the draw
// list. The panels are immediate-mode (no widget tree), so each frame they emit
// geometry plus the hit regions that go with it, and onTick() resolves clicks
// against the regions the *previous* frame emitted.
struct EditorHit {
    enum class Action {
        StepRectX,      // move the node by `delta` px on X
        StepRectY,
        StepRectW,      // resize by `delta` px, keeping the top-left anchored
        StepRectH,
        StepFloatProp,  // numeric schema property, by `delta`
        SetColorProp,
        SetBoolProp,    // delta != 0 -> true
        ClearProp,      // remove the property from the node entirely
        BeginTextEdit,  // focus this string-valued property for typing
        TogglePicker,   // open/close the HSV + harmony picker for this property
        SetHarmony,     // `delta` indexes allHarmonies()
        SelectNode,     // outliner row
        ReorderNode,    // move among siblings by `delta`
        DeleteNode,
    };

    ui::UiRect rect{};
    Action action = Action::SelectNode;
    std::string key;              // JSON property key, for the property actions
    float delta = 0.0f;
    ui::UiColor colorValue{};
    NodePath path;                // for the node actions
};

class UiEditorApp : public engine::GameApp {
protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;
    bool wantsMinimalRendering() const override { return true; }

private:
    // ── DPI ───────────────────────────────────────────────────────────────────
    // Every chrome dimension in ui_editor_app.cc is authored in logical pixels
    // at 100 % DPI and passed through scaled() before it reaches the draw list,
    // which works in framebuffer pixels. m_scale is resolved once in onInit()
    // because the font atlases are baked at that size and are not re-baked.
    [[nodiscard]] float scaled(float logicalPx) const { return logicalPx * m_scale; }

    // Pane dimensions, in framebuffer pixels.
    [[nodiscard]] float paletteW() const;
    [[nodiscard]] float propsW() const;
    [[nodiscard]] float toolbarH() const;
    [[nodiscard]] float statusH() const;
    [[nodiscard]] float paletteRowH() const;
    [[nodiscard]] float outlinerRowH() const;
    // Bottom edge of the palette section of the left column.
    [[nodiscard]] float paletteBottom() const;

    // ── Coordinate spaces ─────────────────────────────────────────────────────
    // Document space is the design surface (the root node's own width/height).
    // Canvas space is where that surface sits inside the centre pane on screen.
    // One document unit is m_viewScale framebuffer pixels.
    [[nodiscard]] ui::UiVec2 docToScreen(float x, float y) const;
    [[nodiscard]] ui::UiVec2 screenToDoc(float x, float y) const;
    [[nodiscard]] ui::UiRect docRectToScreen(const ui::UiRect& r) const;
    void layoutPanes(int fbW, int fbH);

    // ── Canvas ────────────────────────────────────────────────────────────────
    void drawCanvas();
    void drawNode(const NodePath& path);
    void drawSelectionOverlay();
    void drawPlacementGhost();

    // Handle index 0-7 under a document-space point, for the primary selection.
    // Order: TL=0, TC=1, TR=2, MR=3, BR=4, BC=5, BL=6, ML=7. -1 when none.
    [[nodiscard]] int hitTestHandle(float docX, float docY) const;

    // ── Panels ────────────────────────────────────────────────────────────────
    void buildWidgetTree(int fbW, int fbH);
    void drawOutliner();
    void drawOutlinerRow(const NodePath& path, int depth, float& y);
    void drawProperties();
    void drawStatusBar(int fbW, int fbH);

    // Property-panel row helpers. Each appends geometry and its hit regions.
    void drawSectionLabel(const char* label, float x, float width, float& y);
    void drawStepRow(const char* label, float value, EditorHit::Action action,
                     const std::string& key, const float (&deltas)[4], float x, float width,
                     float& y);
    void drawColorRow(const char* label, const std::string& key, const ui::UiColor& current,
                      bool present, float x, float width, float& y);
    // The expanded HSV square + hue/alpha bars, harmony schemes and WCAG readout
    // for one color property. Only one is open at a time.
    void drawColorPicker(const std::string& key, const ui::UiColor& current, float x, float width,
                         float& y);
    void drawBoolRow(const char* label, const std::string& key, bool value, float x, float width,
                     float& y);
    void drawTextRow(const char* label, const std::string& key, const std::string& value, float x,
                     float width, float& y);

    // ── Input ─────────────────────────────────────────────────────────────────
    void handleCanvasInput();
    void handlePanelClicks();
    void handleColorDrag();
    void handleKeyboard();
    void applyHit(const EditorHit& hit);

    // Records a hit region, dropping it when it falls outside the pane's content
    // clip — otherwise a row scrolled under the pinned buttons would still be
    // clickable where it isn't drawn.
    void pushHit(const EditorHit& hit);

    // ── Color ─────────────────────────────────────────────────────────────────
    void setColorProperty(const std::string& key, const ui::UiColor& color, bool commitToHistory);

    // What the given node's color sits on: the design surface with every
    // ancestor's background composited down onto it. `includeOwnBackground`
    // adds the node's own background too, which is what a foreground color
    // (text, border, fill) is actually read against.
    [[nodiscard]] ui::UiColor backdropFor(const NodePath& path, bool includeOwnBackground) const;

    // ── Editing operations ────────────────────────────────────────────────────
    void placeNode(float docX, float docY);
    void commitTextEdit();
    void cancelTextEdit();
    void doSave();
    void doLoad();
    void doUndo();
    void doRedo();

    // Edge-triggered key helper: true only on the frame the key goes down.
    bool keyPressed(int glfwKey);

    // ── State ─────────────────────────────────────────────────────────────────
    EditorDocument m_doc;
    Selection m_selection;
    // Declared after m_doc so its default member initializer can seed from it.
    EditorHistory m_history{m_doc};
    SnapEngine m_snap;
    SnapSettings m_snapSettings{};
    std::vector<nlohmann::json> m_clipboard;

    std::string m_documentPath;
    std::string m_statusMessage;

    // Framebuffer pixels per logical pixel (the OS content scale: 1.5 at 150 %).
    float m_scale = 1.0f;
    // Framebuffer pixels per document unit. Equal to m_scale so the design
    // surface is previewed at the size the running game would draw it, but
    // shrunk below that when the surface would otherwise overflow the canvas
    // pane — this editor has no pan or zoom, so an overflowing document would
    // put its right/bottom edge permanently out of reach.
    float m_viewScale = 1.0f;

    // Pane rects, recomputed on resize.
    float m_canvasX = 0, m_canvasY = 0, m_canvasW = 0, m_canvasH = 0;
    float m_docOriginX = 0, m_docOriginY = 0;  // design surface offset inside the canvas
    float m_docW = 1280, m_docH = 720;

    // Placement
    bool m_placing = false;
    std::string m_placeType;

    // Drag / resize gestures
    bool m_dragging = false;
    bool m_resizing = false;
    int m_resizeHandle = -1;
    ui::UiVec2 m_gestureStartDoc{};
    std::vector<ui::UiRect> m_gestureStartRects;  // parallel to m_selection.paths()

    // Inline text editing in the properties panel.
    std::string m_editingKey;
    std::string m_editingBuffer;
    NodePath m_editingPath;
    bool m_editing = false;

    // Expanded color picker. Open for at most one property of one node.
    enum class ColorDrag { None, SaturationValue, Hue, Alpha };
    std::string m_colorKey;   // empty when no picker is open
    NodePath m_colorPath;     // the node the open picker belongs to
    Harmony m_harmony = Harmony::Complementary;
    ColorDrag m_colorDrag = ColorDrag::None;
    // Sticky HSV for the open picker: dragging value to 0 (or saturation to 0)
    // destroys hue in RGB, so the picker keeps its own copy and writes RGB from
    // it. Re-seeded whenever the stored property stops matching it.
    Hsv m_pickerHsv{};
    ui::UiRect m_svRect{};
    ui::UiRect m_hueRect{};
    ui::UiRect m_alphaRect{};

    std::vector<EditorHit> m_hits;
    ui::UiRect m_hitClip{};      // active content clip for pushHit()
    bool m_hitClipActive = false;

    bool m_uiDirty = true;  // rebuild the toolbar/palette widget tree next tick
    int m_lastFbW = 0, m_lastFbH = 0;

    std::vector<int> m_prevKeyDown;  // edge detection, indexed by GLFW key code

    render::CameraPose m_camera{};
};

}  // namespace odai::tools::ui_editor
