#include "games/legion/legion_app.h"

#include "anim/biped_rig.h"
#include "import/imported_scene.h"
#include "procgen/humanoid_generator.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace odai::games::legion {

namespace {

using odai::importer::ImportedScene;
using odai::importer::ImportedScenePackedDraw;
using odai::importer::ImportedScenePackedVertex;
using odai::math::Matrix4;
using odai::math::Vector3;
using odai::render::ImportedSkinnedMeshTemplate;

constexpr float kArenaHalfExtent = 10.0f;

// Minimal flat-shaded packed-vertex-color mesh builder for the arena ground,
// mirroring CityMeshBuilder's winding convention (citybuilder_app.cc) --
// quad a-b-c-d wound CCW viewed from outside.
struct GroundMeshBuilder {
    ImportedScene& scene;

    std::uint32_t addVertex(const Vector3& p, const Vector3& n, const Vector3& color) {
        ImportedScenePackedVertex v{};
        v.position[0] = p.x;
        v.position[1] = p.y;
        v.position[2] = p.z;
        v.normal[0] = n.x;
        v.normal[1] = n.y;
        v.normal[2] = n.z;
        v.color[0] = color.x;
        v.color[1] = color.y;
        v.color[2] = color.z;
        const auto index = static_cast<std::uint32_t>(scene.packedVertices.size());
        scene.packedVertices.push_back(v);
        return index;
    }
    void addTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        scene.packedIndices.push_back(a);
        scene.packedIndices.push_back(b);
        scene.packedIndices.push_back(c);
    }
    void addFlatTriangle(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& color) {
        const Vector3 n = odai::math::normalize(odai::math::cross(b - a, c - a));
        addTriangle(addVertex(a, n, color), addVertex(b, n, color), addVertex(c, n, color));
    }
    void addQuad(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d, const Vector3& color) {
        addFlatTriangle(a, d, c, color);
        addFlatTriangle(a, c, b, color);
    }
};

ImportedScene buildArenaScene() {
    ImportedScene scene{};
    scene.sourceTag = "legion_arena";
    GroundMeshBuilder builder{scene};
    const float e = kArenaHalfExtent;
    const Vector3 groundColor{0.55f, 0.47f, 0.33f};  // packed dirt
    builder.addQuad(
        Vector3{-e, 0.0f, -e}, Vector3{e, 0.0f, -e}, Vector3{e, 0.0f, e}, Vector3{-e, 0.0f, e}, groundColor);

    scene.packedDraws.push_back(
        ImportedScenePackedDraw{0u, static_cast<std::uint32_t>(scene.packedIndices.size())});
    scene.boundsMin[0] = scene.boundsMin[2] = -e;
    scene.boundsMin[1] = 0.0f;
    scene.boundsMax[0] = scene.boundsMax[2] = e;
    scene.boundsMax[1] = 2.0f;
    return scene;
}

Character makeCharacter(std::uint32_t instanceSlot, bool isEnemy, const Vector3& position) {
    Character c;
    c.instanceSlot = instanceSlot;
    c.isEnemy = isEnemy;
    c.position = position;
    return c;
}

}  // namespace

bool LegionApp::onInit() {
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"))) {
        return false;
    }

    m_skeleton = odai::anim::makeBipedSkeleton();
    m_idleClip = odai::anim::makeBipedIdleClip(m_skeleton);
    m_walkClip = odai::anim::makeBipedWalkClip(m_skeleton);
    m_attackClip = odai::anim::makeBipedAttackClip(m_skeleton);
    m_sampler.bindSkeleton(m_skeleton);

    uploadCharacterMeshes();
    m_renderer.uploadImportedScene(buildArenaScene());

    resetBattle();
    return true;
}

void LegionApp::uploadCharacterMeshes() {
    procgen::HumanoidMeshColors legionaryColors{};
    legionaryColors.torso = Vector3{0.55f, 0.08f, 0.08f};    // red tunic
    legionaryColors.limbs = Vector3{0.85f, 0.68f, 0.52f};    // skin
    legionaryColors.accent = Vector3{0.72f, 0.55f, 0.22f};   // bronze trim

    procgen::HumanoidMeshColors raiderColors{};
    raiderColors.torso = Vector3{0.30f, 0.32f, 0.34f};   // grey furs
    raiderColors.limbs = Vector3{0.75f, 0.60f, 0.46f};   // skin
    raiderColors.accent = Vector3{0.18f, 0.14f, 0.10f};  // dark leather

    const auto uploadVariant = [&](const procgen::HumanoidMeshColors& colors,
                                    std::uint32_t firstSlot, std::uint32_t count) {
        const procgen::HumanoidSkinnedMesh mesh = procgen::buildHumanoidSkinnedMesh(m_skeleton, colors);
        const std::vector<ImportedScenePackedDraw> draws = {
            ImportedScenePackedDraw{0u, static_cast<std::uint32_t>(mesh.indices.size())}
        };
        ImportedSkinnedMeshTemplate meshTemplate{};
        meshTemplate.vertices = mesh.vertices;
        meshTemplate.indices = mesh.indices;
        meshTemplate.draws = draws;
        meshTemplate.boneCount = mesh.boneCount;
        for (std::uint32_t i = 0; i < count; ++i) {
            m_renderer.uploadSkinnedMeshTemplate(firstSlot + i, meshTemplate);
        }
    };

    uploadVariant(legionaryColors, 0u, static_cast<std::uint32_t>(kPartySize));
    uploadVariant(raiderColors, static_cast<std::uint32_t>(kPartySize), static_cast<std::uint32_t>(kEnemySize));
}

void LegionApp::resetBattle() {
    static constexpr float kSpacing = 1.1f;
    for (std::size_t i = 0; i < kPartySize; ++i) {
        const float x = (static_cast<float>(i) - (static_cast<float>(kPartySize - 1) * 0.5f)) * kSpacing;
        m_party[i] = makeCharacter(static_cast<std::uint32_t>(i), false, Vector3{x, 0.0f, -4.0f});
    }
    for (std::size_t i = 0; i < kEnemySize; ++i) {
        const float x = (static_cast<float>(i) - (static_cast<float>(kEnemySize - 1) * 0.5f)) * kSpacing;
        m_enemies[i] = makeCharacter(static_cast<std::uint32_t>(kPartySize + i), true, Vector3{x, 0.0f, 4.0f});
    }
    m_paused = false;
}

void LegionApp::onTick(float dt) {
    const bool spaceDown = glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS;
    if (spaceDown && !m_prevSpace) {
        m_paused = !m_paused;
    }
    m_prevSpace = spaceDown;

    const bool rDown = glfwGetKey(m_window, GLFW_KEY_R) == GLFW_PRESS;
    if (rDown && !m_prevR) {
        resetBattle();
    }
    m_prevR = rDown;

    const bool leftDown = glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(m_window, GLFW_KEY_LEFT) == GLFW_PRESS;
    const bool rightDown = glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(m_window, GLFW_KEY_RIGHT) == GLFW_PRESS;
    constexpr float kOrbitDegPerSecond = 45.0f;
    if (leftDown) m_camYawDeg -= kOrbitDegPerSecond * dt;
    if (rightDown) m_camYawDeg += kOrbitDegPerSecond * dt;
    m_camZoom = std::clamp(m_camZoom - (m_uiInput.scrollDelta * 0.5f), 3.0f, 14.0f);

    if (!m_paused) {
        tickOneSide(std::span<Character>(m_party), std::span<Character>(m_enemies), dt);
        tickOneSide(std::span<Character>(m_enemies), std::span<Character>(m_party), dt);
    }
}

void LegionApp::tickOneSide(std::span<Character> mine, std::span<Character> theirs, float dt) {
    for (Character& self : mine) {
        if (!self.alive) {
            self.deathSinkY = std::min(kDeathSinkMax, self.deathSinkY + (kDeathSinkSpeed * dt));
            self.state = CombatState::Dead;
            continue;
        }

        // Re-pick a target whenever the current one is gone.
        if (self.targetIndex < 0 || self.targetIndex >= static_cast<int>(theirs.size()) ||
            !theirs[static_cast<std::size_t>(self.targetIndex)].alive) {
            float bestDistSq = std::numeric_limits<float>::max();
            int bestIndex = -1;
            for (std::size_t i = 0; i < theirs.size(); ++i) {
                if (!theirs[i].alive) continue;
                const Vector3 delta = theirs[i].position - self.position;
                const float distSq = (delta.x * delta.x) + (delta.z * delta.z);
                if (distSq < bestDistSq) {
                    bestDistSq = distSq;
                    bestIndex = static_cast<int>(i);
                }
            }
            self.targetIndex = bestIndex;
        }

        if (self.targetIndex < 0) {
            self.state = CombatState::Idle;
            self.animClock += dt;
            continue;
        }

        Character& target = theirs[static_cast<std::size_t>(self.targetIndex)];
        Vector3 toTarget = target.position - self.position;
        toTarget.y = 0.0f;
        const float distance = odai::math::length(toTarget);
        if (distance > 1e-4f) {
            self.facingYawRadians = std::atan2(toTarget.x, toTarget.z);
        }

        if (distance > kEngageRange) {
            self.state = CombatState::Walk;
            const Vector3 direction = toTarget * (1.0f / distance);
            const float step = std::min(kWalkSpeed * dt, distance - (kEngageRange * 0.9f));
            if (step > 0.0f) {
                self.position = self.position + (direction * step);
            }
            self.animClock += dt;
        } else {
            self.state = CombatState::Attack;
            self.animClock += dt;
            if (self.animClock >= m_attackClip.duration) {
                self.animClock -= m_attackClip.duration;
                target.hp = std::max(0.0f, target.hp - kAttackDamage);
                if (target.hp <= 0.0f) {
                    target.alive = false;
                    target.state = CombatState::Dead;
                }
            }
        }
    }
}

void LegionApp::updateCharacterPose(const Character& character) {
    const odai::anim::AnimationClip* clip = &m_idleClip;
    float clockForPose = character.animClock;
    switch (character.state) {
        case CombatState::Walk:   clip = &m_walkClip; break;
        case CombatState::Attack: clip = &m_attackClip; break;
        case CombatState::Idle:   clip = &m_idleClip; break;
        case CombatState::Dead:   clip = &m_idleClip; clockForPose = 0.0f; break;
    }

    m_sampler.sample(m_skeleton, *clip, clockForPose, m_localPoseScratch);

    const Vector3 worldPosition = character.position - Vector3{0.0f, character.deathSinkY, 0.0f};
    const Matrix4 actorWorldMatrix = Matrix4::translation(worldPosition) * Matrix4::rotationY(character.facingYawRadians);

    m_finalPoseScratch.resize(m_localPoseScratch.size());
    for (std::size_t i = 0; i < m_localPoseScratch.size(); ++i) {
        m_finalPoseScratch[i] = actorWorldMatrix * m_localPoseScratch[i];
    }

    render::ImportedSkinnedActorFrameData pose{};
    pose.boneMatrices = m_finalPoseScratch;
    m_renderer.setSkinnedActorPose(character.instanceSlot, pose);
}

render::CameraPose LegionApp::computeCameraPose() const {
    render::CameraPose cam{};
    cam.fovDegrees = 45.0f;
    cam.yawDegrees = m_camYawDeg;
    cam.pitchDegrees = -28.0f;
    const float yawR = odai::math::radians(cam.yawDegrees);
    const float pitchR = odai::math::radians(cam.pitchDegrees);
    const float cp = std::cos(pitchR);
    const Vector3 fwd{std::cos(yawR) * cp, std::sin(pitchR), std::sin(yawR) * cp};
    cam.x = -fwd.x * m_camZoom;
    cam.y = 1.6f - (fwd.y * m_camZoom);
    cam.z = -fwd.z * m_camZoom;
    return cam;
}

void LegionApp::onRender(float /*dt*/) {
    beginFrameDraw();

    for (const Character& c : m_party) updateCharacterPose(c);
    for (const Character& c : m_enemies) updateCharacterPose(c);

    m_camera = computeCameraPose();
    drawHud();
    submitFrame(m_camera, 0.0f);
}

void LegionApp::drawHud() {
    int fw = 0, fh = 0;
    framebufferSize(fw, fh);
    const float s = contentScale();

    const auto drawRoster = [&](const auto& roster, float x0, const char* label) {
        m_uiDrawList.addText(m_uiFontBold, label, ui::UiVec2{x0, 12.0f * s}, ui::UiColor{0.92f, 0.86f, 0.7f, 1.0f});
        for (std::size_t i = 0; i < roster.size(); ++i) {
            const float y = (34.0f + (static_cast<float>(i) * 22.0f)) * s;
            const float barW = 120.0f * s;
            const float barH = 14.0f * s;
            const ui::UiRect back = ui::UiRect::fromXYWH(x0, y, barW, barH);
            m_uiDrawList.addRectFilled(back, ui::UiColor{0.15f, 0.13f, 0.11f, 0.85f});
            const float pct = roster[i].alive ? std::clamp(roster[i].hp / roster[i].maxHp, 0.0f, 1.0f) : 0.0f;
            const ui::UiRect fill = ui::UiRect::fromXYWH(x0, y, barW * pct, barH);
            const ui::UiColor fillColor = roster[i].alive
                ? ui::UiColor{0.75f, 0.15f, 0.12f, 1.0f}
                : ui::UiColor{0.3f, 0.3f, 0.3f, 1.0f};
            m_uiDrawList.addRectFilled(fill, fillColor);
        }
    };
    drawRoster(m_party, 16.0f * s, "LEGION");
    drawRoster(m_enemies, static_cast<float>(fw) - (136.0f * s), "RAIDERS");

    const std::string status = m_paused ? "PAUSED (space to resume)" : "space: pause  |  r: reset  |  a/d: orbit";
    m_uiDrawList.addText(
        m_uiFont, status,
        ui::UiVec2{(static_cast<float>(fw) * 0.5f) - (110.0f * s), static_cast<float>(fh) - (28.0f * s)},
        ui::UiColor{0.85f, 0.85f, 0.85f, 1.0f});

    const bool partyWiped = std::none_of(m_party.begin(), m_party.end(), [](const Character& c) { return c.alive; });
    const bool enemiesWiped = std::none_of(m_enemies.begin(), m_enemies.end(), [](const Character& c) { return c.alive; });
    if (partyWiped || enemiesWiped) {
        const std::string banner = enemiesWiped ? "THE LEGION HOLDS" : "THE LEGION HAS FALLEN";
        m_uiDrawList.addText(
            m_uiFontBold, banner,
            ui::UiVec2{(static_cast<float>(fw) * 0.5f) - (100.0f * s), (static_cast<float>(fh) * 0.5f) - (60.0f * s)},
            ui::UiColor{0.95f, 0.85f, 0.3f, 1.0f});
    }
}

}  // namespace odai::games::legion
