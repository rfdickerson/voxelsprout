#pragma once
#include "engine/game_app.h"
#include "render/renderer_types.h"
#include <string>
#include <vector>

namespace odai::games::swtor {

struct ChatMsg { std::string channel, sender, text; };

// A single inventory item slot.
struct InvItem {
    int type;    // 0–7 icon type (blaster, helm, chest, legs, belt, boots, accessory, implant)
    int quality; // 0=grey 1=white 2=green 3=blue 4=purple 5=gold
    int count;   // stack count shown in corner; 0 = no label
};

class SwtorApp : public engine::GameApp {
protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;

private:
    // ── Shared helpers ────────────────────────────────────────────────────────
    void drawPanelBg(float x, float y, float w, float h, float s, float radius = 4.0f);
    void drawBar(float x, float y, float w, float h, float frac,
                 const ui::UiColor& fill, float s, float radius = 2.0f);
    void drawCooldown(float cx, float cy, float radius, float frac);

    // ── HUD elements ──────────────────────────────────────────────────────────
    void drawUnitFrame(float x, float y, float w, float h, const char* name,
                       float hpFrac, float resFrac, bool isPlayer, int level, float s);
    void drawBuffRow(float x, float y, int count, float size, float s);
    void drawMinimap(float cx, float cy, float radius, float s);
    void drawActionBar(float x, float y, int slots, float s);
    void drawChatWindow(float x, float y, float w, float h, float s,
                        const std::string& inputText, float caretPhase);
    void drawMissionTracker(float x, float y, float w, float h, float s);
    void drawXpBar(float x, float y, float w, float h, float frac, int level, float s);

    // ── Character / inventory window ──────────────────────────────────────────
    void drawCharWindow(float x, float y, float w, float h, float s);
    void drawGearIcon(float cx, float cy, float sz, int type, float s);
    void drawGearSlot(float x, float y, float sz, int iconType, int quality, float s);

    // ── Combat / HUD state ────────────────────────────────────────────────────
    float m_playerHp    = 0.82f;
    float m_playerForce = 0.54f;
    float m_targetHp    = 0.75f;
    float m_combatTimer = 0.0f;
    float m_lastDmg     = -3.0f;
    float m_cooldowns[12] = {};
    float m_xpFrac      = 0.67f;
    float m_mapPhase    = 0.0f;

    std::vector<ChatMsg> m_chatLog;
    float m_chatTimer   = 0.0f;
    int   m_nextChat    = 0;

    std::string m_chatInput;
    bool  m_prevBackspace = false;
    bool  m_prevEnter     = false;

    // ── Character window state ────────────────────────────────────────────────
    static constexpr int kInvCapacity = 80;
    bool    m_showCharWindow = true;
    bool    m_prevCKey       = false;
    InvItem m_inventory[kInvCapacity]{};
    int     m_invCount       = 67;

    render::CameraPose m_camera{};
};

} // namespace odai::games::swtor
