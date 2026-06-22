#pragma once

#include "engine/game_app.h"
#include "render/renderer_types.h"
#include "ui/widgets/resource_bar_panel.h"
#include "ui/widgets/event_tracker_panel.h"
#include "ui/widgets/research_panel.h"
#include "ui/widgets/sim_controls_panel.h"

#include <cstdint>
#include <string>
#include <vector>

namespace odai::games::stellaris {

struct StarSystem {
    float x = 0.0f;       // 0..1 normalized galaxy-map position
    float y = 0.0f;
    int   empireId = -1;  // -1 = unclaimed
    std::string name;
    bool  isCapital = false;
};

struct StelEmpire {
    int         id   = 0;
    std::string name;
    std::string leaderName;
    ui::UiColor color{};

    int energy    = 200;
    int minerals  = 200;
    int alloys    = 50;
    int influence = 50;

    int physProg = 0, socProg = 0, engProg = 0;
    int physIncome = 5, socIncome = 5, engIncome = 5;
    std::string physTarget, socTarget, engTarget;
    int techsCompleted = 0;

    int systemCount = 6;
    bool isPlayer   = false;
};

struct StelGalaxy {
    int                  turn    = 2200;  // game year
    bool                 paused  = false;
    float                tickAccum = 0.0f;  // seconds until next auto-advance
    std::vector<StelEmpire>  empires;
    std::vector<StarSystem>  systems;
    std::vector<std::string> log;           // most recent at back
};

class StellarisApp : public engine::GameApp {
protected:
    bool onInit()   override;
    void onTick(float dt) override;
    void onRender(float dt) override;
    void onShutdown() override {}

private:
    void initGalaxy();
    void advanceTurn();
    std::string pickNextTech(const StelEmpire& emp, int track) const;

    void rebuildPanels(int fbW, int fbH);
    void drawGalaxyMap(int fbW, int fbH);
    void drawHyperlanes(float mapL, float mapT, float mapW, float mapH);
    void drawSystems(float mapL, float mapT, float mapW, float mapH);

    StelGalaxy m_galaxy;
    bool       m_panelsDirty = true;
    int        m_lastFbW = 0, m_lastFbH = 0;
    float      m_uiScale  = 1.0f;
    float      m_animTime = 0.0f;

    // Auto-advance timing: one turn every kTurnIntervalSec seconds in real time.
    static constexpr float kTurnIntervalSec = 2.0f;

    // Panel raw pointers (owned by UiContext widget tree).
    ui::ResourceBarPanel*      m_resourceBar  = nullptr;
    ui::EventTrackerPanel*     m_eventTracker = nullptr;
    ui::ResearchPanel*         m_researchPanel = nullptr;
    ui::SimControlsPanel*      m_simControls  = nullptr;

    render::CameraPose m_camera{};
};

} // namespace odai::games::stellaris
