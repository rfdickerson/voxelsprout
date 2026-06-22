#pragma once
#include "engine/game_app.h"
#include "render/renderer_types.h"
#include "ui/ui_types.h"

#include <string>
#include <vector>

namespace odai::tools::ui_editor {

struct DesignWidget {
    std::string type = "Panel";
    std::string name;
    float x = 80, y = 60, w = 200, h = 80;
    ui::UiColor bg{0.05f, 0.10f, 0.15f, 0.88f};
    ui::UiColor border{0.85f, 0.72f, 0.44f, 0.35f};
    float borderW   = 1.0f;
    float cornerR   = 0.0f;
    bool  shadow    = false;
    float shadowBlur= 8.0f;
    std::string label;
};

// Click target in the properties panel (drawn via draw list).
struct PropButton {
    ui::UiRect rect;
    enum class Action { StepX, StepY, StepW, StepH,
                        SetCorner, ToggleShadow, SetColor, Delete
    } action{Action::StepX};
    float delta = 0.0f;
    ui::UiColor colorValue{};
};

class UiEditorApp : public engine::GameApp {
protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;

private:
    // ── Snap ──────────────────────────────────────────────────────────────────
    struct SnapLine { bool horiz; float pos; };  // pos in canvas space

    // Returns the snapped (x,y) for a widget and fills m_snapLines.
    ui::UiVec2 applySnap(float wx, float wy, float ww, float wh, int excludeIdx);

    // Snap a single value against 1-D targets; records a snap-line at the snapped
    // feature position when a match is found within the threshold.
    float snap1D(float val, float widgetSpan,
                 const std::vector<float>& targets, float threshold,
                 bool isHoriz, float gridSize, bool doGrid, bool doEdges);

    bool  m_snapGrid   = true;
    float m_gridSnap   = 10.0f;
    bool  m_snapEdges  = true;
    float m_snapThresh = 8.0f;

    std::vector<SnapLine> m_snapLines;

    // ── Resize ────────────────────────────────────────────────────────────────
    // Returns handle index 0-7 under (cx,cy) in canvas space, or -1.
    // Handle order: TL=0, TC=1, TR=2, MR=3, BR=4, BC=5, BL=6, ML=7
    int hitTestHandle(float cx, float cy) const;

    bool  m_resizing      = false;
    int   m_resizeHandle  = -1;
    float m_resizeMX      = 0, m_resizeMY      = 0;  // mouse at resize start
    float m_resizeOrigX   = 0, m_resizeOrigY   = 0;  // widget rect at resize start
    float m_resizeOrigW   = 0, m_resizeOrigH   = 0;

    // ── Drawing ───────────────────────────────────────────────────────────────
    void setupWidgetTree(int fbW, int fbH);
    void drawCanvas(int fbW, int fbH);
    void drawDesignWidget(const DesignWidget& w, bool selected);
    void drawPropertiesPanel(int fbW, int fbH);
    void deleteSelected();

    // ── JSON round-trip ───────────────────────────────────────────────────────
    void saveJson(const std::string& path);
    void loadJson(const std::string& path);

    int  hitTestCanvas(float cx, float cy) const;
    static float defaultW(const std::string& type);
    static float defaultH(const std::string& type);
    static ui::UiColor defaultBg(const std::string& type);

    // ── State ─────────────────────────────────────────────────────────────────
    std::vector<DesignWidget> m_widgets;
    int  m_selected  = -1;
    bool m_placing   = false;
    std::string m_placeType;

    bool  m_dragging = false;
    float m_dragMX   = 0, m_dragMY = 0;
    float m_dragWX   = 0, m_dragWY = 0;

    float m_canvasX = 0, m_canvasY = 0, m_canvasW = 0, m_canvasH = 0;

    std::vector<PropButton> m_propBtns;

    std::string m_loadedPath;

    int  m_lastFbW       = 0, m_lastFbH = 0;
    bool m_prevDeleteKey = false;
    bool m_prevGKey      = false, m_prevSKey = false;

    render::CameraPose m_camera{};
};

} // namespace odai::tools::ui_editor
