#include "games/newvegas/newvegas_app.h"

#include "core/log.h"
#include "ui/ui_types.h"

#include <cstdio>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace odai::games::newvegas {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Bethesda world units: 1 unit is about 1.43 cm, so ~70 units to the metre.
constexpr float kWalkUnitsPerSecond = 400.0f;
constexpr float kSprintMultiplier = 4.0f;
constexpr float kMouseSensitivity = 0.12f;
constexpr float kPitchLimitDegrees = 89.0f;
constexpr float kEyeHeightUnits = 120.0f;
// Fallout's own jump is about 1 metre at ~70 units/metre. v = sqrt(2*g*h) with a
// gravity that keeps the arc short enough to feel like a jump rather than a
// hop on the moon.
constexpr float kGravityUnitsPerSecondSq = 2600.0f;
constexpr float kJumpUnitsPerSecond = 620.0f;

// LAND posts sit on a regular 128-unit lattice (kLandPostSpacing), and the
// cooked terrain mesh preserves that, so the ground can be sampled from a plain
// 2D grid instead of a ray cast. raycastImportedScene is the obvious tool and
// the wrong one here: it is brute force over every triangle by design (see its
// header), which is fine at click rate but this scene has ~3.7M of them and the
// camera needs a ground height every frame.
constexpr float kGroundGridSpacing = 128.0f;

bool keyDown(GLFWwindow* window, int key) {
    return glfwGetKey(window, key) == GLFW_PRESS;
}

}  // namespace

void NewVegasApp::buildGroundHeightField(const importer::ImportedScene& scene) {
    m_groundHeights.clear();
    if (scene.meshes.empty() || scene.meshes.front().name != "terrain") {
        VOX_LOGW("newvegas") << "no terrain mesh; camera will not be ground-clamped";
        return;
    }
    const importer::ImportedSceneMesh& terrain = scene.meshes.front();
    if (terrain.vertices.empty()) {
        return;
    }

    float minX = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();
    for (const importer::ImportedSceneVertex& vertex : terrain.vertices) {
        minX = std::min(minX, vertex.position[0]);
        maxX = std::max(maxX, vertex.position[0]);
        minZ = std::min(minZ, vertex.position[2]);
        maxZ = std::max(maxZ, vertex.position[2]);
    }

    m_groundOriginX = minX;
    m_groundOriginZ = minZ;
    m_groundCols = static_cast<int>(std::lround((maxX - minX) / kGroundGridSpacing)) + 1;
    m_groundRows = static_cast<int>(std::lround((maxZ - minZ) / kGroundGridSpacing)) + 1;
    if (m_groundCols <= 1 || m_groundRows <= 1) {
        m_groundCols = 0;
        m_groundRows = 0;
        return;
    }

    // Cells overlap at their shared edge posts and adjacent cells can disagree
    // slightly there, so keep the highest sample per lattice point: standing a
    // few units high reads as a step, sinking reads as falling through the world.
    m_groundHeights.assign(
        static_cast<std::size_t>(m_groundCols) * static_cast<std::size_t>(m_groundRows),
        -std::numeric_limits<float>::max());
    for (const importer::ImportedSceneVertex& vertex : terrain.vertices) {
        const int col = static_cast<int>(std::lround((vertex.position[0] - minX) / kGroundGridSpacing));
        const int row = static_cast<int>(std::lround((vertex.position[2] - minZ) / kGroundGridSpacing));
        if (col < 0 || col >= m_groundCols || row < 0 || row >= m_groundRows) {
            continue;
        }
        float& slot = m_groundHeights[(static_cast<std::size_t>(row) * m_groundCols) + col];
        slot = std::max(slot, vertex.position[1]);
    }
    VOX_LOGI("newvegas") << "ground height field: " << m_groundCols << "x" << m_groundRows
                         << " posts at " << kGroundGridSpacing << " units";
}

bool NewVegasApp::groundHeightAt(float x, float z, float& outHeight) const {
    if (m_groundHeights.empty()) {
        return false;
    }
    const float gridX = (x - m_groundOriginX) / kGroundGridSpacing;
    const float gridZ = (z - m_groundOriginZ) / kGroundGridSpacing;
    const int col = static_cast<int>(std::floor(gridX));
    const int row = static_cast<int>(std::floor(gridZ));
    if (col < 0 || row < 0 || col + 1 >= m_groundCols || row + 1 >= m_groundRows) {
        return false;
    }
    const float tx = gridX - static_cast<float>(col);
    const float tz = gridZ - static_cast<float>(row);
    const auto sample = [this](int c, int r) {
        return m_groundHeights[(static_cast<std::size_t>(r) * m_groundCols) + c];
    };
    const float h00 = sample(col, row);
    const float h10 = sample(col + 1, row);
    const float h01 = sample(col, row + 1);
    const float h11 = sample(col + 1, row + 1);
    // A lattice point no terrain vertex landed on is a hole, not a height of
    // -FLT_MAX; bilinear-blending one would yank the camera through the floor.
    const float unset = -std::numeric_limits<float>::max();
    if (h00 == unset || h10 == unset || h01 == unset || h11 == unset) {
        return false;
    }
    const float bottom = h00 + ((h10 - h00) * tx);
    const float top = h01 + ((h11 - h01) * tx);
    outHeight = bottom + ((top - bottom) * tz);
    return true;
}

bool NewVegasApp::onInit() {
    // Without this the font atlas is empty, so every addText() emits zero
    // quads and GameApp::drawPerfOverlay bails outright — the HUD and F3 both
    // render nothing, silently, with no error anywhere.
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"))) {
        VOX_LOGE("newvegas") << "failed to load UI fonts";
        return false;
    }

    if (m_scenePath.empty()) {
        if (const char* fromEnv = std::getenv("ODAI_FNV_SCENE")) {
            m_scenePath = fromEnv;
        }
    }
    if (m_scenePath.empty()) {
        VOX_LOGE("newvegas") << "no scene given; pass --scene <path.bin> or set ODAI_FNV_SCENE";
        return false;
    }
    // Local, not a member: uploadImportedScene deep-copies the whole scene, so
    // keeping a second copy alive for the process lifetime costs ~100 MB of
    // resident memory that nothing ever reads again.
    //
    // The full loader, NOT loadImportedSceneRuntime. The runtime one keeps only
    // the packed stream: it skips the mesh block outright and reads instances
    // just to discard them. Both are needed here and neither failure is visible
    // -- the containers come back empty rather than erroring -- so with the
    // runtime loader the ground height field is never built (camera stays in fly
    // mode) and the town centroid finds nothing (spawn falls back to the middle
    // of the map). This costs the mesh + instance arrays for the duration of
    // onInit, which is the price of knowing where the ground and the town are.
    importer::ImportedScene scene;
    if (!importer::loadImportedScene(std::filesystem::path(m_scenePath), scene)) {
        VOX_LOGE("newvegas") << "failed to load scene '" << m_scenePath
                             << "': " << importer::getImportedSceneLastError();
        return false;
    }
    VOX_LOGI("newvegas") << "loaded " << m_scenePath << " (" << scene.packedVertices.size() << " vertices, "
                         << scene.textures.size() << " textures, " << scene.lights.size() << " lights)";

    if (!m_renderer.uploadImportedScene(scene)) {
        VOX_LOGE("newvegas") << "failed to upload scene to the renderer";
        return false;
    }
    const bool interior = importer::importedSceneSourceTagIsInterior(scene.sourceTag);
    m_renderer.setImportedSceneInteriorMode(interior);

    // Sun plus cascaded shadow maps, and nothing the original game didn't have.
    // Fallout: New Vegas lit its world with a directional sun, shadow maps and
    // baked ambient — no global illumination, no ray tracing, no screen-space AO,
    // no sun shafts. Matching that is both the look we want and, on an integrated
    // GPU, the difference between a playable frame rate and a driver hang-check
    // reset (VK_ERROR_DEVICE_LOST) on the very first frame.
    m_renderer.setShadowSettings(render::ShadowSettings{render::ShadowMode::Auto});

    // Voxel GI contributes nothing here anyway: the grid is 64 world units wide
    // and camera-following, which at Bethesda scale (~70 units/metre) is under a
    // metre across inside a scene spanning tens of thousands of units, so
    // sampleImportedVoxelGi lands outside the volume and returns black for
    // essentially every pixel. Without this the whole ReSTIR sequence — candidate,
    // temporal, spatial, resolve, all traced against the TLAS — ran every frame
    // for a contribution that was already invisible.
    m_renderer.setVoxelGiEnabled(false);
    // No TLAS to trace against once GI is off, so stop building acceleration
    // structures on every uploadImportedScene too.
    m_renderer.setRayTracingEnabled(false);
    m_renderer.setSunShaftsEnabled(false);

    // AoMode::Off skips both AO dispatches, so setSsaoEnabled is moot — but set it
    // anyway so the two agree and neither reads as a leftover.
    //
    // If AO is ever restored here it needs re-tuning for Bethesda scale, and the
    // reasoning is worth keeping: GameApp::init calls setStrategyMapMode, which
    // pins the AO radius to 7 world units — sensible for a strategy map, but 10 cm
    // at ~70 units/metre. The GTAO march takes six steps across a screen-space
    // radius of roughly `radius * 9297 / depth` pixels, so a 7-unit radius
    // collapses to sub-pixel steps beyond ~1500 units and the estimator early-outs
    // to "unoccluded" for the whole frame. The working value was
    // setAmbientOcclusionTuning(128.0f, 40.0f, 0.85f): 128 is not a taste call, it
    // is the shader's own clamp ceiling in ssao.comp.slang and happens to be ~1.8 m.
    m_renderer.setSsaoEnabled(false);
    m_renderer.setAmbientOcclusionMode(render::AoMode::Off);

    // Eye adaptation. Without it the renderer holds a fixed exposure, and the
    // Mojave at noon came out around 46/255 -- textured, detailed, and far too
    // dark to read, with shadowed ground going to black because AO, GI and sun
    // shafts are all deliberately off here and nothing fills them. It also gives
    // the day/night cycle somewhere to go: at a fixed exposure midnight rendered
    // pure black rather than moonlit.
    m_renderer.setAutoExposureEnabled(true);

    // Diagnostic A/B: ODAI_FNV_NOTEX forces every imported surface to shade from
    // its vertex colour instead of its texture. Comparing a capture with and
    // without it answers a question that is otherwise guesswork -- whether a
    // washed-out surface is showing a pale TEXTURE or is falling back to vertex
    // colour and being blown out by lighting. The two look identical on screen.
    if (const char* noTextures = std::getenv("ODAI_FNV_NOTEX")) {
        if (noTextures[0] != '\0' && noTextures[0] != '0') {
            m_renderer.setImportedSceneDebugState(true, true, false, false, false);
            VOX_LOGI("newvegas") << "ODAI_FNV_NOTEX: imported textures disabled (vertex colour only)";
        }
    }

    // Start hour override. Lighting bugs and "it's just a dim hour" look the
    // same in a single capture; being able to shoot the same view at several
    // times of day separates them.
    if (const char* hourEnv = std::getenv("ODAI_FNV_HOUR")) {
        const float hour = static_cast<float>(std::atof(hourEnv));
        if (hour >= 0.0f && hour < 24.0f) {
            m_timeOfDayHours = hour;
        }
    }
    applyTimeOfDay();

    buildGroundHeightField(scene);

    // Spawn standing in Goodsprings rather than hovering over the map.
    //
    // The previous spawn put the camera at boundsMax[1] — above the highest peak
    // in the cooked region — pitched 35 degrees down. That framed the terrain but
    // left no horizon on screen at all, which is why the sky appeared to be
    // missing: the skybox draws with VK_COMPARE_OP_EQUAL against a reversed-Z
    // depth buffer, so it fills exactly the pixels no geometry covered, and from
    // up there geometry covered all of them.
    //
    // The anchor is the centroid of the town's own architecture rather than a
    // hand-entered coordinate, so it stays right if the cooked grid moves. Note
    // this lands you in the middle of Goodsprings by the houses, not on Doc
    // Mitchell's doorstep specifically -- picking out that one building needs its
    // formID from the GECK, which is not something the cooked scene records.
    float spawnX = (scene.boundsMin[0] + scene.boundsMax[0]) * 0.5f;
    float spawnZ = (scene.boundsMin[2] + scene.boundsMax[2]) * 0.5f;
    double townX = 0.0;
    double townZ = 0.0;
    std::size_t townCount = 0;
    for (const importer::ImportedSceneInstance& instance : scene.instances) {
        std::string path = instance.modelPath;
        std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (path.find("goodsprings") == std::string::npos) {
            continue;
        }
        // Row-major 4x4 with translation in the last COLUMN: the cooker's
        // writeTransform puts it at 3/7/11, not the 12/13/14 a column-major
        // layout would use. Reading 12/14 gets the bottom row, which is all
        // zeroes here, so the centroid silently collapses to the origin.
        townX += static_cast<double>(instance.transform[3]);
        townZ += static_cast<double>(instance.transform[11]);
        ++townCount;
    }
    if (townCount > 0) {
        spawnX = static_cast<float>(townX / static_cast<double>(townCount));
        spawnZ = static_cast<float>(townZ / static_cast<double>(townCount));
        VOX_LOGI("newvegas") << "spawning at the centroid of " << townCount
                             << " Goodsprings placements";
    } else {
        VOX_LOGW("newvegas") << "no Goodsprings placements in this scene; spawning at scene centre";
    }
    m_cameraX = spawnX;
    m_cameraZ = spawnZ;
    float groundHeight = 0.0f;
    if (groundHeightAt(m_cameraX, m_cameraZ, groundHeight)) {
        m_cameraY = groundHeight + kEyeHeightUnits;
    } else {
        m_cameraY = scene.boundsMax[1] + kEyeHeightUnits;
        VOX_LOGW("newvegas") << "spawn point is off the terrain grid; starting in fly mode";
        m_walkMode = false;
    }
    // Level, so the horizon -- and therefore the sky -- is on screen. The
    // override has to come after this, not before: the spawn pitch is assigned
    // here, so an earlier override would be silently discarded.
    m_pitchDegrees = 0.0f;
    if (const char* pitchEnv = std::getenv("ODAI_FNV_PITCH")) {
        m_pitchDegrees = static_cast<float>(std::atof(pitchEnv));
    }

    setMouseCaptured(true);
    return true;
}

void NewVegasApp::applyTimeOfDay() {
    // Map 0..24h onto a sun that rises in the east and sets in the west.
    //
    // Sign convention, which is easy to get backwards: setSunAngles takes the
    // direction the light TRAVELS, not the direction to the sun. frame_run.cc
    // computes toSun = -sunDirection, so the sun is above the horizon only
    // while pitch is NEGATIVE — hence the debug slider's -89..+5 range and
    // citybuilder's -38 for ordinary daylight. Getting this backwards puts the
    // sun under the map at every hour and the whole world renders in ambient
    // only, which reads as "everything is super dark".
    //
    //   midnight -> +75 (below horizon)   dawn/dusk -> 0 (on the horizon)
    //   noon     -> -75 (high overhead)
    const float dayFraction = m_timeOfDayHours / 24.0f;
    const float pitchDegrees = std::cos(dayFraction * 2.0f * kPi) * 75.0f;
    const float yawDegrees = 90.0f + (dayFraction * 360.0f);
    m_renderer.setSunAngles(yawDegrees, pitchDegrees);
}

void NewVegasApp::updateCamera(float deltaSeconds) {
    // Mouselook from raw cursor deltas; GameApp has put the cursor in
    // GLFW_CURSOR_DISABLED mode so it reports unbounded relative motion.
    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(m_window, &cursorX, &cursorY);
    if (m_mouseCaptured) {
        if (m_hasCursorSample) {
            m_yawDegrees += static_cast<float>(cursorX - m_lastCursorX) * kMouseSensitivity;
            m_pitchDegrees -= static_cast<float>(cursorY - m_lastCursorY) * kMouseSensitivity;
            m_pitchDegrees = std::clamp(m_pitchDegrees, -kPitchLimitDegrees, kPitchLimitDegrees);
        }
        m_hasCursorSample = true;
    } else {
        m_hasCursorSample = false;
    }
    m_lastCursorX = cursorX;
    m_lastCursorY = cursorY;

    // Must match the renderer's own camera basis exactly, or WASD walks off at
    // an angle to where you are looking. computeCameraForward
    // (render/backend/vulkan/frame_math.h) is:
    //     forward = (cos(yaw)*cos(pitch), sin(pitch), sin(yaw)*cos(pitch))
    // so in the XZ plane forward is (cos(yaw), sin(yaw)) — NOT (sin, -cos),
    // which is that basis rotated 90 degrees and was what this used.
    // Right is forward advanced a quarter turn: (cos(yaw+90), sin(yaw+90)).
    const float yawRadians = m_yawDegrees * (kPi / 180.0f);
    const float forwardX = std::cos(yawRadians);
    const float forwardZ = std::sin(yawRadians);
    const float rightX = -std::sin(yawRadians);
    const float rightZ = std::cos(yawRadians);

    float moveX = 0.0f;
    float moveY = 0.0f;
    float moveZ = 0.0f;
    if (keyDown(m_window, GLFW_KEY_W)) { moveX += forwardX; moveZ += forwardZ; }
    if (keyDown(m_window, GLFW_KEY_S)) { moveX -= forwardX; moveZ -= forwardZ; }
    if (keyDown(m_window, GLFW_KEY_D)) { moveX += rightX;   moveZ += rightZ; }
    if (keyDown(m_window, GLFW_KEY_A)) { moveX -= rightX;   moveZ -= rightZ; }
    if (keyDown(m_window, GLFW_KEY_SPACE)) { moveY += 1.0f; }
    if (keyDown(m_window, GLFW_KEY_LEFT_CONTROL)) { moveY -= 1.0f; }

    const float lengthSquared = (moveX * moveX) + (moveZ * moveZ);
    if (lengthSquared > 1e-6f) {
        const float inverseLength = 1.0f / std::sqrt(lengthSquared);
        moveX *= inverseLength;
        moveZ *= inverseLength;
    }

    float speed = kWalkUnitsPerSecond;
    if (keyDown(m_window, GLFW_KEY_LEFT_SHIFT)) {
        speed *= kSprintMultiplier;
    }
    m_cameraX += moveX * speed * deltaSeconds;
    m_cameraZ += moveZ * speed * deltaSeconds;

    // Walk mode pins the eye to the terrain; fly mode (F) keeps the old free
    // movement, which is still the only way to inspect the scene from above or
    // to get back if you walk off the edge of the cooked grid.
    float groundHeight = 0.0f;
    if (m_walkMode && groundHeightAt(m_cameraX, m_cameraZ, groundHeight)) {
        const float standingHeight = groundHeight + kEyeHeightUnits;
        if (m_airborne) {
            m_verticalVelocity -= kGravityUnitsPerSecondSq * deltaSeconds;
            m_cameraY += m_verticalVelocity * deltaSeconds;
            // Land when the ground catches up. Tested against the ground under
            // the CURRENT position, so walking off a ledge mid-jump lands on
            // whatever is actually below rather than the height jumped from.
            if (m_cameraY <= standingHeight) {
                m_cameraY = standingHeight;
                m_verticalVelocity = 0.0f;
                m_airborne = false;
            }
        } else {
            m_cameraY = standingHeight;
            if (keyDown(m_window, GLFW_KEY_SPACE)) {
                m_verticalVelocity = kJumpUnitsPerSecond;
                m_airborne = true;
            }
        }
    } else {
        // Fly mode (or off the terrain grid): Space/Ctrl move straight up and
        // down, and there is nothing to fall onto.
        m_cameraY += moveY * speed * deltaSeconds;
        m_airborne = false;
        m_verticalVelocity = 0.0f;
    }
}

void NewVegasApp::onTick(float deltaSeconds) {
    if (keyDown(m_window, GLFW_KEY_ESCAPE)) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        return;
    }

    updateCamera(deltaSeconds);

    // Time-of-day controls. Edge-latched so a held key steps once.
    const bool bracketLeft = keyDown(m_window, GLFW_KEY_LEFT_BRACKET);
    if (bracketLeft && !m_bracketLeftLatch) {
        m_timeOfDayHours = std::fmod(m_timeOfDayHours - 1.0f + 24.0f, 24.0f);
    }
    m_bracketLeftLatch = bracketLeft;

    const bool bracketRight = keyDown(m_window, GLFW_KEY_RIGHT_BRACKET);
    if (bracketRight && !m_bracketRightLatch) {
        m_timeOfDayHours = std::fmod(m_timeOfDayHours + 1.0f, 24.0f);
    }
    m_bracketRightLatch = bracketRight;

    const bool pausePressed = keyDown(m_window, GLFW_KEY_P);
    if (pausePressed && !m_pauseLatch) {
        m_dayCyclePaused = !m_dayCyclePaused;
    }
    m_pauseLatch = pausePressed;

    const bool walkPressed = keyDown(m_window, GLFW_KEY_F);
    if (walkPressed && !m_walkModeLatch) {
        m_walkMode = !m_walkMode;
    }
    m_walkModeLatch = walkPressed;

    const bool tabPressed = keyDown(m_window, GLFW_KEY_TAB);
    if (tabPressed && !m_tabLatch) {
        m_mouseCaptured = !m_mouseCaptured;
        setMouseCaptured(m_mouseCaptured);
    }
    m_tabLatch = tabPressed;

    if (!m_dayCyclePaused) {
        m_timeOfDayHours = std::fmod(m_timeOfDayHours + (m_dayCycleHoursPerSecond * deltaSeconds), 24.0f);
    }
    applyTimeOfDay();
}

void NewVegasApp::drawHud() {
    const int hours = static_cast<int>(m_timeOfDayHours);
    const int minutes = static_cast<int>((m_timeOfDayHours - static_cast<float>(hours)) * 60.0f);
    char text[192];
    std::snprintf(
        text, sizeof(text),
        "New Vegas  |  %02d:%02d %s  |  pos %.0f %.0f %.0f  |  [ ] time   P cycle   Tab cursor",
        hours, minutes, m_dayCyclePaused ? "(paused)" : "(running)", m_cameraX, m_cameraY, m_cameraZ);
    const float margin = 16.0f * contentScale();
    m_uiDrawList.addText(m_uiFont, text, ui::UiVec2{margin, margin}, ui::UiColor{0.91f, 0.85f, 0.69f, 1.0f});
}

void NewVegasApp::onRender(float /*deltaSeconds*/) {
    beginFrameDraw();
    drawHud();

    render::CameraPose camera{};
    camera.x = m_cameraX;
    camera.y = m_cameraY;
    camera.z = m_cameraZ;
    camera.yawDegrees = m_yawDegrees;
    camera.pitchDegrees = m_pitchDegrees;
    camera.fovDegrees = 75.0f;
    submitFrame(camera);

    // Capture AFTER submitFrame: the capture reads the last presented image, so
    // running it before there is one gets nothing. The warm-up frames matter
    // too -- auto-exposure adapts over several frames, so a capture on frame 0
    // shows a mid-adaptation image rather than what the scene settles at.
    if (!m_screenshotPath.empty()) {
        ++m_framesRendered;
        if (m_framesRendered >= m_screenshotWarmupFrames) {
            if (!m_renderer.captureFrameToFile(m_screenshotPath)) {
                VOX_LOGE("newvegas") << "screenshot capture failed";
            }
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        }
    }
}

}  // namespace odai::games::newvegas
