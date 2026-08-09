#include "games/newvegas/newvegas_app.h"

#include "core/log.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/nif_scene.h"
#include "ui/ui_types.h"

#include <cstdio>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cctype>
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
    // Streaming owns its own terrain: the whole-scene height field below is
    // built once from a loaded .bin and has nothing in it when cells arrive and
    // leave continuously.
    if (m_streamer) {
        // referenceY is the player's foot height, so ceilings and upper
        // storeys above them are not mistaken for the ground.
        return m_collision.groundHeight(x, z, m_cameraY - kEyeHeightUnits, outHeight);
    }
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

bool NewVegasApp::loadScene(
    const std::filesystem::path& path, const float* arrivalPosition, const float* arrivalYawDegrees
) {
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
    // of the map).
    importer::ImportedScene scene;
    if (!importer::loadImportedScene(path, scene)) {
        VOX_LOGE("newvegas") << "failed to load scene '" << path.string()
                             << "': " << importer::getImportedSceneLastError();
        return false;
    }
    VOX_LOGI("newvegas") << "loaded " << path.string() << " (" << scene.packedVertices.size()
                         << " vertices, " << scene.textures.size() << " textures, "
                         << scene.doors.size() << " doors)";

    // Diagnostic A/B: ODAI_FNV_AS_CHUNK routes the same scene through the
    // streaming chunk path instead of the whole-scene upload, with everything
    // else identical. Isolates "the geometry is wrong" from "the chunk path is
    // wrong" -- they look the same on screen.
    if (std::getenv("ODAI_FNV_AS_CHUNK") != nullptr) {
        if (m_renderer.addImportedSceneChunk(scene) ==
            render::Renderer::kInvalidImportedChunkIndex) {
            VOX_LOGE("newvegas") << "failed to add scene as a chunk";
            return false;
        }
        VOX_LOGI("newvegas") << "ODAI_FNV_AS_CHUNK: uploaded via addImportedSceneChunk";
    } else if (!m_renderer.uploadImportedScene(scene)) {
        VOX_LOGE("newvegas") << "failed to upload scene to the renderer";
        return false;
    }
    // ODAI_FNV_CHUNK_TEST exercises the streaming add/remove path before a real
    // cell streamer exists to drive it. It re-adds the scene just loaded as a
    // second resident chunk and then evicts it, which is the only way today to
    // check the three invariants that matter and all fail silently:
    //   * every texture is shared, so the second add must upload zero of them
    //     (the refcount table returns the resident slot instead);
    //   * the geometry arena must grow and copy, leaving chunk 0 renderable;
    //   * eviction must return the arena ranges and drop the texture refcounts
    //     back, restoring exactly the pre-test state;
    //   * eviction must release the chunk's punctual lights. These used to be
    //     appended straight into one flat list with no record of which chunk
    //     owned them, so they were never released -- invisible until the
    //     64-light budget filled up with lights from cells long since left.
    // Loading the same scene twice means the two chunks occupy the same space,
    // so the screen should look unchanged throughout -- which is the point.
    if (std::getenv("ODAI_FNV_CHUNK_TEST") != nullptr) {
        VOX_LOGI("newvegas") << "chunk test: live chunks before add = "
                             << m_renderer.liveImportedSceneChunkCount()
                             << ", lights = " << m_renderer.importedLocalLightCount();
        const std::size_t testChunk = m_renderer.addImportedSceneChunk(scene);
        if (testChunk == render::Renderer::kInvalidImportedChunkIndex) {
            VOX_LOGE("newvegas") << "chunk test: addImportedSceneChunk failed";
        } else {
            VOX_LOGI("newvegas") << "chunk test: added chunk " << testChunk
                                 << ", live chunks = " << m_renderer.liveImportedSceneChunkCount()
                                 << ", lights = " << m_renderer.importedLocalLightCount();
            m_renderer.removeImportedSceneChunk(testChunk);
            VOX_LOGI("newvegas") << "chunk test: removed chunk " << testChunk
                                 << ", live chunks = " << m_renderer.liveImportedSceneChunkCount()
                                 << ", lights = " << m_renderer.importedLocalLightCount();
        }
    }

    const bool interior = importer::importedSceneSourceTagIsInterior(scene.sourceTag);
    m_renderer.setImportedSceneInteriorMode(interior);
    m_doors = scene.doors;

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


    // An arrival transform from a door wins over the spawn heuristics above.
    if (arrivalPosition != nullptr) {
        m_cameraX = arrivalPosition[0];
        m_cameraZ = arrivalPosition[2];
        float groundHeight = 0.0f;
        // Fallout's arrival Y is the floor the player stands on, so lift it to
        // eye height. Prefer the terrain lattice where there is one -- an
        // interior has no LAND, and there the authored height is all we have.
        if (groundHeightAt(m_cameraX, m_cameraZ, groundHeight)) {
            m_cameraY = groundHeight + kEyeHeightUnits;
        } else {
            m_cameraY = arrivalPosition[1] + kEyeHeightUnits;
            m_walkMode = false;
        }
        if (arrivalYawDegrees != nullptr) {
            m_yawDegrees = *arrivalYawDegrees;
        }
        m_pitchDegrees = 0.0f;
    }
    return true;
}

int NewVegasApp::findUsableDoor() const {
    // Near, and roughly in front. Both matter: a doorway you have walked past
    // should not keep offering itself, and Fallout's doors come in pairs close
    // enough that distance alone picks the wrong one.
    constexpr float kMaxDoorDistance = 260.0f;   // ~3.7 m at Bethesda scale
    constexpr float kMinFacingDot = 0.35f;
    const float yawRadians = m_yawDegrees * (kPi / 180.0f);
    const float forwardX = std::cos(yawRadians);
    const float forwardZ = std::sin(yawRadians);
    int best = -1;
    float bestDistanceSquared = kMaxDoorDistance * kMaxDoorDistance;
    for (std::size_t i = 0; i < m_doors.size(); ++i) {
        const float dx = m_doors[i].position[0] - m_cameraX;
        const float dz = m_doors[i].position[2] - m_cameraZ;
        const float dy = m_doors[i].position[1] - m_cameraY;
        const float distanceSquared = (dx * dx) + (dy * dy) + (dz * dz);
        if (distanceSquared > bestDistanceSquared) {
            continue;
        }
        const float horizontal = std::sqrt((dx * dx) + (dz * dz));
        if (horizontal > 1e-3f && (((dx / horizontal) * forwardX) + ((dz / horizontal) * forwardZ)) < kMinFacingDot) {
            continue;
        }
        best = static_cast<int>(i);
        bestDistanceSquared = distanceSquared;
    }
    return best;
}

void NewVegasApp::useDoor(const importer::ImportedSceneDoor& door) {
    // An empty target cell means the exterior this interior was cooked beside;
    // both spellings go through importedSceneInteriorFileName so the cooker's
    // naming convention lives in exactly one place.
    const std::filesystem::path target = door.targetCellEditorId.empty()
        ? (m_sceneDirectory / (m_exteriorStem + ".bin"))
        : (m_sceneDirectory /
           importer::importedSceneInteriorFileName(m_exteriorStem, door.targetCellEditorId));
    if (!std::filesystem::exists(target)) {
        VOX_LOGW("newvegas") << "door leads to " << target.filename().string()
                             << ", which is not cooked; re-run the cooker with --with-interiors";
        return;
    }
    const float arrivalYaw = door.arrivalYawDegrees;
    if (!loadScene(target, door.arrivalPosition, &arrivalYaw)) {
        VOX_LOGE("newvegas") << "failed to walk through the door into " << target.filename().string();
    }
}

namespace {

// Common install locations for Fallout: New Vegas, in the order they are tried.
// A directory only counts when it actually holds the master plugin -- an empty
// or partial directory would otherwise be "found" and then fail later with a
// much less obvious message.
std::string findFalloutDataDirectory() {
    std::vector<std::filesystem::path> candidates;

    const char* home = std::getenv("HOME");
    if (home != nullptr) {
        const std::filesystem::path homePath(home);
        candidates.push_back(homePath / ".steam/steam/steamapps/common/Fallout New Vegas/Data");
        candidates.push_back(homePath / ".local/share/Steam/steamapps/common/Fallout New Vegas/Data");
        candidates.push_back(homePath / "GOG Games/Fallout New Vegas/Data");
    }
    // WSL and dual-boot mounts of a Windows install.
    candidates.emplace_back("/mnt/c/Program Files (x86)/Steam/steamapps/common/Fallout New Vegas/Data");
    candidates.emplace_back("/mnt/c/GOG Games/Fallout New Vegas/Data");
    // Native Windows.
    candidates.emplace_back("C:/Program Files (x86)/Steam/steamapps/common/Fallout New Vegas/Data");
    candidates.emplace_back("C:/GOG Games/Fallout New Vegas/Data");

    for (const std::filesystem::path& candidate : candidates) {
        std::error_code existsError;
        if (std::filesystem::exists(candidate / "FalloutNV.esm", existsError) && !existsError) {
            return candidate.string();
        }
    }
    return {};
}

}  // namespace

bool NewVegasApp::onInit() {
    // Without this the font atlas is empty, so every addText() emits zero
    // quads and GameApp::drawPerfOverlay bails outright — the HUD and F3 both
    // render nothing, silently, with no error anywhere.
    // TV-sized type. The defaults (18 px body) are a desk-monitor scale: at a
    // couch viewing distance that is roughly half the angular size a console UI
    // needs, and it is why the first pass of this HUD was unreadable on a TV.
    // 28 px body / 76 px display is the 10-foot scale, and contentScale() still
    // multiplies on top for high-DPI panels.
    //
    // Inter is the Helvetica-like grotesque already vendored here, so the
    // display face costs no new asset.
    constexpr float kTvBodySize = 28.0f;
    constexpr float kTvNumericSize = 26.0f;
    constexpr float kTvCaptionSize = 22.0f;
    constexpr float kTvDisplaySize = 76.0f;  // the discovery banner
    if (!loadFonts(
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
            resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
            resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
            kTvBodySize, kTvNumericSize, kTvCaptionSize, kTvDisplaySize)) {
        VOX_LOGE("newvegas") << "failed to load UI fonts";
        return false;
    }

    if (m_streamDirectory.empty()) {
        if (const char* fromEnv = std::getenv("ODAI_FNV_STREAM_DIR")) {
            m_streamDirectory = fromEnv;
        }
    }
    if (m_streamDirectory.empty() && m_scenePath.empty() &&
        std::getenv("ODAI_FNV_SCENE") == nullptr) {
        // Nothing specified at all: look for an installed copy of the game and
        // stream from it. Streaming needs no cooked assets, so a bare launch now
        // has a sensible thing to do -- which it did not when a cooked scene was
        // the only possible source.
        m_streamDirectory = findFalloutDataDirectory();
        if (!m_streamDirectory.empty()) {
            VOX_LOGI("newvegas") << "found Fallout: New Vegas data at " << m_streamDirectory;
        }
    }
    // NOTE: streaming init happens further down, AFTER the renderer pass-stack
    // configuration. Returning here instead left streaming running with ray
    // tracing, voxel GI and sun shafts all still enabled -- which showed up as a
    // BLAS/TLAS rebuild on every single streamed cell.
    // Character mode STREAMS THE WORLD TOO, and that turned out to be
    // load-bearing rather than cosmetic. With streaming off nothing calls
    // uploadImportedScene, and a frame with no imported geometry renders no
    // sky, no ground and no skinned actor -- a flat clear-colour screen that
    // looks exactly like a failed character upload. Standing the character in
    // Goodsprings costs a few seconds of streaming and makes the view both
    // correct and legible: a body at Fallout's own scale, next to Fallout's own
    // buildings, is the only way to see that the scale is right.
    const bool streamingMode = !m_streamDirectory.empty();

    if (!streamingMode && !m_characterMode && m_scenePath.empty()) {
        if (const char* fromEnv = std::getenv("ODAI_FNV_SCENE")) {
            m_scenePath = fromEnv;
        }
    }
    if (m_characterMode && m_streamDirectory.empty()) {
        VOX_LOGE("newvegas")
            << "--character needs the game's Data directory (for the skeleton and "
               "body meshes); none was found. Pass --stream \"<.../Fallout New Vegas/Data>\".";
        return false;
    }
    if (!streamingMode && !m_characterMode && m_scenePath.empty()) {
        VOX_LOGE("newvegas")
            << "no Fallout: New Vegas install found, and no scene given.\n"
               "  Stream from the game (no cooking): --stream \"<.../Fallout New Vegas/Data>\"\n"
               "  Load a cooked scene:               --scene <path.bin>\n"
               "  Or set ODAI_FNV_STREAM_DIR / ODAI_FNV_SCENE.";
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
    if (!streamingMode && !m_characterMode) {
        m_sceneDirectory = std::filesystem::path(m_scenePath).parent_path();
        m_exteriorStem = std::filesystem::path(m_scenePath).stem().string();
        if (!loadScene(std::filesystem::path(m_scenePath), nullptr, nullptr)) {
            return false;
        }
    }
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

    // Ambient occlusion, tuned for Bethesda scale.
    //
    // The radius is NOT a taste call. GameApp::init calls setStrategyMapMode,
    // which pins the AO radius to 7 world units -- sensible for a strategy map,
    // but 10 cm at Fallout's ~70 units/metre. The GTAO march takes six steps
    // across a screen-space radius of roughly `radius * 9297 / depth` pixels, so
    // a 7-unit radius collapses to sub-pixel steps beyond ~1500 units and the
    // estimator early-outs to "unoccluded" for the entire frame -- AO that costs
    // its full dispatch and produces nothing. 128 is the shader's own clamp
    // ceiling in ssao.comp.slang and lands at ~1.8 m, which is the scale of the
    // contact darkening this world wants.
    //
    // ODAI_FNV_AO overrides the mode (off/ssao/hbao/gtao) for A/B comparison.
    render::AoMode aoMode = render::AoMode::Gtao;
    if (const char* aoEnv = std::getenv("ODAI_FNV_AO")) {
        const std::string requested = aoEnv;
        if (requested == "off") {
            aoMode = render::AoMode::Off;
        } else if (requested == "ssao") {
            aoMode = render::AoMode::Ssao;
        } else if (requested == "hbao") {
            aoMode = render::AoMode::Hbao;
        }
    }
    m_renderer.setSsaoEnabled(aoMode != render::AoMode::Off);
    m_renderer.setAmbientOcclusionMode(aoMode);
    // Sweepable, because "too subtle" is a measurable claim: the A/B against
    // AO-off below is what says whether a value actually changed the image.
    //
    // NOTE the intensity is an EXPONENT: sampleSsaoAmbientFactor computes
    // pow(ssaoRaw, intensity) on a value in [0,1]. Anything below 1 pushes the
    // result toward 1, i.e. actively weakens the occlusion -- which is what the
    // inherited 0.85 was doing.
    float aoRadius = 300.0f;
    float aoBias = 40.0f;
    float aoIntensity = 1.7f;
    if (const char* env = std::getenv("ODAI_FNV_AO_RADIUS")) {
        aoRadius = static_cast<float>(std::atof(env));
    }
    if (const char* env = std::getenv("ODAI_FNV_AO_BIAS")) {
        aoBias = static_cast<float>(std::atof(env));
    }
    if (const char* env = std::getenv("ODAI_FNV_AO_INTENSITY")) {
        aoIntensity = static_cast<float>(std::atof(env));
    }
    m_renderer.setAmbientOcclusionTuning(aoRadius, aoBias, aoIntensity);

    // Multi-scale: the coarse march reaches well past contact range, the fine
    // one at ~22% of it catches where objects meet the ground. One radius
    // cannot do both -- the march has a fixed step count, so widening it just
    // spreads the same samples further apart.
    float aoFineScale = 0.22f;
    if (const char* env = std::getenv("ODAI_FNV_AO_FINE")) {
        aoFineScale = static_cast<float>(std::atof(env));
    }
    m_renderer.setAmbientOcclusionFineScale(aoFineScale);

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

    // Last, so the streamer inherits the pass-stack configuration above rather
    // than paying for ray tracing and voxel GI on every streamed cell.
    if (streamingMode && !initStreaming()) {
        return false;
    }
    if (m_characterMode && !initCharacter(m_streamDirectory)) {
        return false;
    }

    // Pip-Boy palette for notifications, matching the HUD chrome.
    ui::ToastStyle toastStyle{};
    toastStyle.widthPx = 300.0f;
    m_toasts.setStyle(toastStyle);
    ui::ToastTiming toastTiming{};
    toastTiming.holdSeconds = 4.5f;
    m_toasts.setTiming(toastTiming);
    m_toasts.setMaxVisible(3);

    // Discovery banner: centred, chrome-free, slow fade.
    m_banner.setStyle(ui::makeBannerStyle());
    m_banner.setTiming(ui::makeBannerTiming());
    m_banner.setMaxVisible(1);

    // ODAI_FNV_UI_DEMO=1 opens the menu and stacks sample toasts at startup.
    // The screenshot path cannot press buttons, so without this the menu and a
    // multi-toast stack are the two things that can only be checked by a human
    // with a controller in hand -- which is to say, not checked.
    if (std::getenv("ODAI_FNV_UI_DEMO") != nullptr) {
        // ODAI_FNV_UI_DEMO=menu opens the pause menu; anything else shows the
        // discovery banner. They are mutually exclusive at runtime (the banner
        // holds while the menu is up), so the demo cannot show both either.
        const std::string demoMode = std::getenv("ODAI_FNV_UI_DEMO");
        m_navDriving = true;  // show the focus highlight and the controller labels
        m_menuOpen = (demoMode == "menu");
        if (!m_menuOpen) {
            m_banner.push("Goodsprings", "Location discovered", "region:Goodsprings");
        }
        m_toasts.push("Stimpak", "Added to inventory");
        m_toasts.push("Quest updated", "Back in the Saddle");
    }

    setMouseCaptured(true);
    return true;
}

bool NewVegasApp::initCharacter(const std::filesystem::path& dataFilesPath) {
    importer::fnv::FalloutAssetSource assets;
    if (!assets.open(dataFilesPath)) {
        VOX_LOGE("newvegas") << "could not index archives under " << dataFilesPath;
        return false;
    }

    std::string error;
    std::vector<std::uint8_t> bytes;
    if (!assets.resolveMesh(m_characterSkeletonPath, bytes, error)) {
        VOX_LOGE("newvegas") << "skeleton not found: " << m_characterSkeletonPath << " (" << error << ")";
        return false;
    }
    importer::fnv::NifSkeleton nifSkeleton;
    if (!importer::fnv::parseNifSkeleton(bytes, nifSkeleton, error)) {
        VOX_LOGE("newvegas") << "skeleton parse failed: " << error;
        return false;
    }
    if (!importer::fnv::buildFalloutSkeleton(nifSkeleton, m_character.skeleton)) {
        VOX_LOGE("newvegas") << "skeleton conversion failed";
        return false;
    }

    for (const std::string& partPath : m_characterPartPaths) {
        if (!assets.resolveMesh(partPath, bytes, error)) {
            VOX_LOGW("newvegas") << "body part not found: " << partPath << " (" << error << ")";
            continue;
        }
        importer::fnv::NifSkinnedModel model;
        if (!importer::fnv::parseNifSkinnedMesh(bytes, model, error)) {
            VOX_LOGW("newvegas") << "body part parse failed: " << partPath << " (" << error << ")";
            continue;
        }
        if (!importer::fnv::appendFalloutCharacterMesh(model, m_character, error)) {
            VOX_LOGW("newvegas") << "body part bind failed: " << partPath << " (" << error << ")";
        }
    }
    if (m_character.vertices.empty()) {
        VOX_LOGE("newvegas") << "no skinned geometry loaded";
        return false;
    }

    // One draw per part -- minus the gore caps. The draws index the merged
    // buffer, which is why appendFalloutCharacterMesh records
    // firstIndex/indexCount rather than leaving each part with its own arrays.
    //
    // A body NIF ships dismemberment geometry alongside the body: on
    // characters\_male\upperbody.nif, 3 of the 6 shapes ("limbcaps",
    // "meatneck01", "meathead01") are meat caps the game reveals only when a
    // limb comes off. They are skinned and they bind correctly -- their
    // measured bind-pose bounds are simply not on the standing body
    // ("limbcaps" sits at y -102..-18, well below the feet at 0.78) because
    // nothing positions them until a limb is severed.
    //
    // Drawing them makes an otherwise correct character look broken, and it was
    // the reason the first framed capture of this view looked like it had
    // failed. Excluded by texture because the proper discriminator --
    // BSDismemberSkinInstance's per-partition body-part IDs -- is the one part
    // of that block this importer deliberately does not read. All three use
    // textures\gore\MeatCapGore01.dds and no non-gore part does.
    // ODAI_FNV_CHAR_ALL=1 keeps the caps, for diagnosing which parts reach the
    // screen at all.
    const bool keepAllParts = std::getenv("ODAI_FNV_CHAR_ALL") != nullptr;
    const auto isGoreCap = [keepAllParts](const std::string& texturePath) {
        if (keepAllParts) {
            return false;
        }
        std::string lowered = texturePath;
        for (char& ch : lowered) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        return lowered.find("\\gore\\") != std::string::npos;
    };
    m_characterDraws.clear();
    m_characterDraws.reserve(m_character.parts.size());
    std::vector<const importer::fnv::FalloutCharacterPart*> drawnParts;
    for (const auto& part : m_character.parts) {
        if (isGoreCap(part.diffuseTexturePath)) {
            VOX_LOGI("newvegas") << "  skipping gore cap \"" << part.name << "\"";
            continue;
        }
        importer::ImportedScenePackedDraw draw{};
        draw.firstIndex = part.firstIndex;
        draw.indexCount = part.indexCount;
        draw.alphaThreshold = part.alphaThreshold;
        m_characterDraws.push_back(draw);
        drawnParts.push_back(&part);
    }
    // ODAI_FNV_CHAR_NODRAW=1 uploads the template (so the skinning dispatch
    // still runs) but issues no draws. Paired with ODAI_FNV_SKIN_BYPASS, which
    // does the opposite, it separates "the compute pass corrupts the frame"
    // from "the geometry it produces does".
    if (std::getenv("ODAI_FNV_CHAR_NODRAW") != nullptr) {
        VOX_LOGW("newvegas") << "character draws SUPPRESSED: dispatch only";
        m_characterDraws.clear();
    } else if (m_characterDraws.empty()) {
        VOX_LOGE("newvegas") << "every part was filtered out; nothing to draw";
        return false;
    }

    importer::fnv::computeFalloutBindPose(m_character, m_characterBindPose);

    render::ImportedSkinnedMeshTemplate meshTemplate{};
    meshTemplate.vertices = m_character.vertices;
    meshTemplate.indices = m_character.indices;
    meshTemplate.draws = m_characterDraws;
    meshTemplate.boneCount = static_cast<std::uint32_t>(m_character.skeleton.bones.size());
    // ODAI_FNV_SKIN_BYPASS=1 skips the skinning dispatch, leaving the output
    // buffer at the rest pose the upload seeded it with. It is the one
    // diagnostic that separates "the vertex data I handed over is wrong" from
    // "the compute pass is not doing what I think": a clean figure under bypass
    // and an exploded one without it puts the fault squarely in the dispatch.
    if (std::getenv("ODAI_FNV_SKIN_BYPASS") != nullptr) {
        VOX_LOGW("newvegas") << "skinning dispatch BYPASSED: showing the rest pose";
        m_renderer.setSkinningDebugBypass(true);
    }
    // ODAI_FNV_CHAR_NOUPLOAD=1 does everything except hand the mesh to the GPU.
    // It answers the one question the other toggles cannot: whether the frame
    // breaks because of the skinned instance at all, or because of something
    // else this mode does.
    if (std::getenv("ODAI_FNV_CHAR_NOUPLOAD") != nullptr) {
        VOX_LOGW("newvegas") << "character GPU upload SKIPPED";
        m_characterBindPose.clear();
        return true;
    }
    if (!m_renderer.uploadSkinnedMeshTemplate(0u, meshTemplate)) {
        VOX_LOGE("newvegas") << "uploadSkinnedMeshTemplate failed";
        return false;
    }

    VOX_LOGI("newvegas") << "character: " << m_character.skeleton.bones.size() << " bones, "
                         << m_character.vertices.size() << " vertices, "
                         << (m_character.indices.size() / 3u) << " triangles, "
                         << m_character.parts.size() << " parts, "
                         << m_character.unresolvedBoneCount << " unresolved bones";

    // Frame the camera on the skinned bind-pose bounds rather than on a guessed
    // height. The character's extent is the only thing in the scene, and
    // guessing it wrong means an empty screen that looks exactly like a failed
    // upload -- which is the outcome this whole mode exists to rule out.
    float boundsMin[3] = {1e30f, 1e30f, 1e30f};
    float boundsMax[3] = {-1e30f, -1e30f, -1e30f};
    // Over the DRAWN parts only. Including the filtered gore caps here would
    // frame the camera on a body twice its real height and push the character
    // itself into the top half of the screen.
    for (const importer::fnv::FalloutCharacterPart* part : drawnParts) {
    for (std::uint32_t idx = part->firstIndex; idx < part->firstIndex + part->indexCount; ++idx) {
        const auto& vertex = m_character.vertices[m_character.indices[idx]];
        odai::math::Vector3 skinned{0.0f, 0.0f, 0.0f};
        const odai::math::Vector3 rest{vertex.position[0], vertex.position[1], vertex.position[2]};
        for (int k = 0; k < importer::fnv::kNifMaxBoneInfluences; ++k) {
            const float weight = vertex.boneWeights[k];
            if (weight <= 0.0f) {
                continue;
            }
            const std::size_t bone = vertex.boneIndices[k];
            if (bone >= m_characterBindPose.size()) {
                continue;
            }
            const odai::math::Vector3 contribution =
                odai::math::transformPoint(m_characterBindPose[bone], rest);
            skinned.x += contribution.x * weight;
            skinned.y += contribution.y * weight;
            skinned.z += contribution.z * weight;
        }
        const float values[3] = {skinned.x, skinned.y, skinned.z};
        for (int a = 0; a < 3; ++a) {
            boundsMin[a] = std::min(boundsMin[a], values[a]);
            boundsMax[a] = std::max(boundsMax[a], values[a]);
        }
    }
    }
    const float centreX = (boundsMin[0] + boundsMax[0]) * 0.5f;
    const float centreZ = (boundsMin[2] + boundsMax[2]) * 0.5f;
    const float height = std::max(1.0f, boundsMax[1] - boundsMin[1]);
    VOX_LOGI("newvegas") << "character bind-pose bounds"
                         << " x " << boundsMin[0] << ".." << boundsMax[0]
                         << " y " << boundsMin[1] << ".." << boundsMax[1]
                         << " z " << boundsMin[2] << ".." << boundsMax[2]
                         << " (" << height << " units tall)";

    // Stand the character in front of wherever streaming spawned the camera,
    // rather than moving the camera to the character. The spawn is on the
    // ground in Goodsprings and the camera is at eye height there; dragging it
    // to a bare bounding box would give up the one thing this view is for,
    // which is seeing the body at the same scale as the world around it.
    //
    // The offsets fold into the bone matrices in updateCharacterPose: a skinned
    // actor has no separate instance transform.
    const float yawRadians = m_yawDegrees * (kPi / 180.0f);
    const float forwardX = std::cos(yawRadians);
    const float forwardZ = std::sin(yawRadians);
    // Far enough that the whole figure fits a 75-degree vertical FOV with
    // margin (tan(37.5 deg) ~= 0.767), and no closer -- near clip aside, a body
    // filling the frame hides exactly the scale comparison being made.
    const float standoff = height * 1.1f / 0.767f;
    m_characterWorldX = m_cameraX + (forwardX * standoff) - centreX;
    m_characterWorldZ = m_cameraZ + (forwardZ * standoff) - centreZ;
    // The bind pose already stands on y = 0 (measured: feet at 0.78), so the
    // ground height goes in unmodified.
    float groundY = m_cameraY;
    if (groundHeightAt(m_characterWorldX, m_characterWorldZ, groundY)) {
        m_characterWorldY = groundY;
    } else {
        // No collision data yet (the cell may still be streaming): drop the
        // character to the camera's own foot height, which the spawn put on the
        // ground.
        m_characterWorldY = m_cameraY - kEyeHeightUnits;
    }
    VOX_LOGI("newvegas") << "character placed at " << m_characterWorldX << ", " << m_characterWorldY
                         << ", " << m_characterWorldZ << " (camera at " << m_cameraX << ", "
                         << m_cameraY << ", " << m_cameraZ << ")";
    return true;
}

void NewVegasApp::updateCharacterPose() {
    if (m_characterBindPose.empty()) {
        return;
    }
    // ODAI_FNV_CHAR_NOPOSE=1 never submits a pose. With the output buffer now
    // seeded at upload time, the actor should still draw -- in rest pose, at
    // the origin -- which isolates the per-frame pose upload from everything
    // else the skinned path does.
    if (std::getenv("ODAI_FNV_CHAR_NOPOSE") != nullptr) {
        return;
    }
    // World placement rides on the bone matrices, pre-multiplied: the skinning
    // pass consumes bone matrices and nothing else, so there is no separate
    // instance transform to put it in.
    const odai::math::Matrix4 actorWorld = odai::math::Matrix4::translation(
        odai::math::Vector3{m_characterWorldX, m_characterWorldY, m_characterWorldZ});
    m_characterPoseScratch.resize(m_characterBindPose.size());
    for (std::size_t i = 0; i < m_characterBindPose.size(); ++i) {
        m_characterPoseScratch[i] = actorWorld * m_characterBindPose[i];
    }
    render::ImportedSkinnedActorFrameData pose{};
    pose.boneMatrices = m_characterPoseScratch;
    m_renderer.setSkinnedActorPose(0u, pose);
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
    // ODAI_FNV_BENCH=1 walks the camera forward on a slow turn instead of
    // reading input. "It is jittery when I move" is not reproducible from a
    // standing start, and a hand-driven walk is not comparable between runs --
    // this makes the motion identical every time so a frame-time change is
    // attributable to the code rather than to how the tester moved.
    static const bool s_bench = std::getenv("ODAI_FNV_BENCH") != nullptr;
    if (s_bench) {
        m_yawDegrees += 6.0f * deltaSeconds;
        const float yawRadians = m_yawDegrees * (kPi / 180.0f);
        constexpr float kBenchSpeed = 400.0f;  // ~5.7 m/s, a fast jog
        m_cameraX += std::cos(yawRadians) * kBenchSpeed * deltaSeconds;
        m_cameraZ += std::sin(yawRadians) * kBenchSpeed * deltaSeconds;
        float groundHeight = 0.0f;
        if (groundHeightAt(m_cameraX, m_cameraZ, groundHeight)) {
            m_cameraY = groundHeight + kEyeHeightUnits;
        }
        return;
    }

    // Mouselook from raw cursor deltas; GameApp has put the cursor in
    // GLFW_CURSOR_DISABLED mode so it reports unbounded relative motion.
    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(m_window, &cursorX, &cursorY);
    // A screenshot run must not mouselook. The cursor sits wherever the desktop
    // left it, so 700 warm-up frames of deltas rotate the camera by an
    // arbitrary amount -- which silently defeats ODAI_FNV_YAW/PITCH and makes
    // two captures of "the same view" incomparable. That cost a bogus A/B.
    const bool suppressMouseLook = !m_screenshotPath.empty();
    if (m_mouseCaptured && !suppressMouseLook) {
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

    // Push back out of anything solid the move ended inside. Walk mode only:
    // fly mode is the diagnostic camera and deliberately passes through
    // everything, which is what makes it useful for looking at geometry.
    if (m_walkMode && m_streamer) {
        m_collision.resolveHorizontal(m_cameraX, m_cameraY, m_cameraZ);
    }

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

bool NewVegasApp::initStreaming() {
    // One worker per core minus the main thread and a little headroom, floored
    // at 2. Streaming is latency-sensitive rather than throughput-bound, so
    // oversubscribing here would just contend with the render thread.
    const unsigned hardwareThreads = std::max(4u, std::thread::hardware_concurrency());
    unsigned streamThreads = std::max(2u, hardwareThreads - 2u);
    if (const char* env = std::getenv("ODAI_FNV_STREAM_THREADS")) {
        streamThreads = std::max(1u, static_cast<unsigned>(std::atoi(env)));
    }
    VOX_LOGI("newvegas") << "streaming workers: " << streamThreads;
    m_streamJobs = std::make_unique<core::JobSystem>(streamThreads);
    m_streamer = std::make_unique<importer::fnv::CellStreamer>();

    if (m_streamCacheEnabled) {
        if (m_streamCacheDirectory.empty()) {
            if (const char* fromEnv = std::getenv("ODAI_FNV_CACHE_DIR")) {
                m_streamCacheDirectory = fromEnv;
            }
        }
        if (m_streamCacheDirectory.empty()) {
            // XDG cache location, falling back to the home directory. Built
            // cells are derived data: safe to lose, expensive to recompute.
            if (const char* xdgCache = std::getenv("XDG_CACHE_HOME")) {
                m_streamCacheDirectory = (std::filesystem::path(xdgCache) / "odai" / "fnv").string();
            } else if (const char* home = std::getenv("HOME")) {
                m_streamCacheDirectory =
                    (std::filesystem::path(home) / ".cache" / "odai" / "fnv").string();
            }
        }
        if (!m_streamCacheDirectory.empty()) {
            m_streamer->setCacheDirectory(std::filesystem::path(m_streamCacheDirectory));
        }
    }

    std::string error;
    if (!m_streamer->open(
            std::filesystem::path(m_streamDirectory), std::filesystem::path(m_streamPlugin),
            m_streamWorldspace, *m_streamJobs, error)) {
        VOX_LOGE("newvegas") << "streaming init failed: " << error;
        return false;
    }
    if (m_streamer->availableCellCount() == 0u) {
        VOX_LOGE("newvegas") << "worldspace " << m_streamWorldspace << " has no streamable cells in "
                             << m_streamDirectory;
        return false;
    }

    // Mirror the resident set into collision. Registered before the first
    // update() so no cell can become resident without collision knowing.
    m_collision.clear();
    m_streamer->setCellCallbacks(
        [this](const importer::CellCoord& cell, const importer::ImportedScene& scene) {
            m_collision.addCell(cell, scene);
        },
        [this](const importer::CellCoord& cell) { m_collision.removeCell(cell); });

    importer::CellResidencyConfig config;
    // Fallout exterior cells are 4096 world units square (33 height posts at
    // 128-unit spacing); see fnv::kExteriorCellSize.
    config.cellSize = 4096.0f;
    if (const char* radiusEnv = std::getenv("ODAI_FNV_LOAD_RADIUS")) {
        config.loadRadius = std::max(0, std::atoi(radiusEnv));
        config.unloadRadius = config.loadRadius + 2;
    }
    m_streamer->setConfig(config);

    // Spawn at the centre of the available cells so the first ring has content
    // on every side; the world origin is often outside a cooked region entirely.
    // ENGINE space (Y-up), not Fallout space (Z-up). Assigning a Fallout grid
    // coordinate to m_cameraY put the camera tens of thousands of units below
    // the terrain -- the streamed world rendered correctly and the player was
    // simply underneath it.
    float spawn[3] = {0.0f, 0.0f, 0.0f};
    // Doc Mitchell's doorstep first -- that is where New Vegas actually begins.
    // Fall back to the middle of the worldspace if the cell is missing, so a
    // different plugin or a trimmed install still starts somewhere sensible.
    const bool spawnedAtDoorstep =
        !m_streamSpawnInterior.empty() &&
        m_streamer->spawnAtInteriorDoorEngineSpace(m_streamSpawnInterior, spawn);
    const bool haveSpawn =
        spawnedAtDoorstep || m_streamer->suggestedSpawnEngineSpace(spawn);
    if (haveSpawn) {
        m_cameraX = spawn[0];
        m_cameraY = spawn[1];  // height
        m_cameraZ = spawn[2];
        // A doorstep spawn is at eye height on the ground, so look at the
        // horizon; the worldspace-centre fallback is well above the terrain and
        // wants to look down at it.
        m_pitchDegrees = spawnedAtDoorstep ? 0.0f : -20.0f;
        // Same diagnostic override the cooked-scene path has: being able to
        // look straight down separates "above the ground" from "inside it".
        if (const char* pitchEnv = std::getenv("ODAI_FNV_PITCH")) {
            m_pitchDegrees = static_cast<float>(std::atof(pitchEnv));
        }
        // Pinning yaw as well is what makes two captures comparable: the mouse
        // position at startup otherwise rotates the camera differently per run,
        // so an A/B of a rendering change compares two different views.
        if (const char* yawEnv = std::getenv("ODAI_FNV_YAW")) {
            m_yawDegrees = static_cast<float>(std::atof(yawEnv));
        }
        if (const char* heightEnv = std::getenv("ODAI_FNV_SPAWN_HEIGHT")) {
            m_cameraY += static_cast<float>(std::atof(heightEnv));
        }
        // Full engine-space spawn override, for pinning the camera somewhere
        // known-good while diagnosing.
        if (const char* posEnv = std::getenv("ODAI_FNV_SPAWN_POS")) {
            float px = 0.0f;
            float py = 0.0f;
            float pz = 0.0f;
            if (std::sscanf(posEnv, "%f,%f,%f", &px, &py, &pz) == 3) {
                m_cameraX = px;
                m_cameraY = py;
                m_cameraZ = pz;
            }
        }
        VOX_LOGI("newvegas") << "spawn (engine space): x=" << m_cameraX
                             << " y=" << m_cameraY << " (height) z=" << m_cameraZ;
        // Walk on arrival at a doorstep: collision now supplies terrain height
        // from the streamed cells, so there is ground to stand on. The
        // worldspace-centre fallback still starts in fly mode, because it aims
        // the camera from well above the terrain.
        m_walkMode = spawnedAtDoorstep;
    }
    VOX_LOGI("newvegas") << "streaming " << m_streamer->availableCellCount()
                         << " cells from " << m_streamDirectory
                         << " (load radius " << config.loadRadius
                         << ", unload " << config.unloadRadius << ")";
    return true;
}

// Headless check that collision is actually doing its job, because walking
// around by hand is not a repeatable test and "it felt solid" is not a result.
//
// Two properties, both of which fail silently: terrain has to be sampleable
// everywhere the player can stand (otherwise they fall through), and a point
// placed inside an obstacle has to come back out (otherwise buildings are
// scenery).
void NewVegasApp::runCollisionSelfTest() {
    const float step = 256.0f;
    int sampled = 0;
    int grounded = 0;
    float minClearance = 1e30f;
    float maxClearance = -1e30f;
    for (int dz = -6; dz <= 6; ++dz) {
        for (int dx = -6; dx <= 6; ++dx) {
            const float x = m_cameraX + (static_cast<float>(dx) * step);
            const float z = m_cameraZ + (static_cast<float>(dz) * step);
            float height = 0.0f;
            ++sampled;
            // Terrain only here: this samples coverage, and mixing in geometry
            // would report a rooftop as "the ground".
            if (!m_collision.terrainHeight(x, z, height)) {
                continue;
            }
            ++grounded;
            minClearance = std::min(minClearance, height);
            maxClearance = std::max(maxClearance, height);
        }
    }

    // Walk a straight line and confirm the player actually travels. The failure
    // this catches is collision pinning them in place a few steps in, which is
    // exactly what a single box per mesh did and what "it looked solid" cannot
    // distinguish from working.
    const float walkStep = 40.0f;
    float px = m_cameraX;
    float pz = m_cameraZ;
    float py = m_cameraY;
    float travelled = 0.0f;
    int blockedSteps = 0;
    for (int step = 0; step < 200; ++step) {
        const float beforeX = px;
        const float beforeZ = pz;
        pz -= walkStep;  // due north in engine space
        m_collision.resolveHorizontal(px, py, pz);
        float ground = 0.0f;
        if (m_collision.groundHeight(px, pz, py - m_collision.tuning().eyeHeight, ground)) {
            py = ground + m_collision.tuning().eyeHeight;
        }
        const float moved =
            std::sqrt(((px - beforeX) * (px - beforeX)) + ((pz - beforeZ) * (pz - beforeZ)));
        travelled += moved;
        if (moved < walkStep * 0.25f) {
            ++blockedSteps;
        }
    }

    // The opposite failure: collision so permissive that nothing blocks. Probe
    // each wall triangle's centroid and require the player to be pushed out of
    // it. Walking freely is only good news if walls still stop you.
    int wallProbes = 0;
    int wallBlocks = 0;
    m_collision.forEachNearbyTriangle(
        m_cameraX, m_cameraZ, [&](const CollisionWorld::Triangle& triangle) {
            if (triangle.normalY >= m_collision.tuning().minWalkableNormalY) {
                return;
            }
            const float minY = std::min({triangle.v[1], triangle.v[4], triangle.v[7]});
            const float maxY = std::max({triangle.v[1], triangle.v[4], triangle.v[7]});
            if ((maxY - minY) < m_collision.tuning().eyeHeight) {
                return;  // too short to be a wall the player walks into
            }
            const float cx = (triangle.v[0] + triangle.v[3] + triangle.v[6]) / 3.0f;
            const float cy = (triangle.v[1] + triangle.v[4] + triangle.v[7]) / 3.0f;
            const float cz = (triangle.v[2] + triangle.v[5] + triangle.v[8]) / 3.0f;
            float wx = cx;
            float wz = cz;
            ++wallProbes;
            m_collision.resolveHorizontal(wx, cy + m_collision.tuning().eyeHeight * 0.5f, wz);
            if (std::sqrt(((wx - cx) * (wx - cx)) + ((wz - cz) * (wz - cz))) > 1.0f) {
                ++wallBlocks;
            }
        });

    VOX_LOGI("newvegas") << "collision self-test: walls " << wallBlocks << "/" << wallProbes
                         << " pushed a probe off their surface";
    VOX_LOGI("newvegas") << "collision self-test: terrain " << grounded << "/" << sampled
                         << " sample points grounded (heights " << minClearance << ".."
                         << maxClearance << "); walked " << travelled << " of "
                         << (walkStep * 200.0f) << " units due north, " << blockedSteps
                         << "/200 steps blocked; " << m_collision.triangleCount()
                         << " collision triangles across " << m_collision.residentCellCount()
                         << " cells";
}

void NewVegasApp::updateStreaming(float deltaSeconds) {
    if (!m_streamer) {
        return;
    }

    // Velocity by differencing the camera rather than reading the movement
    // code's own: that one is zeroed by collision and jumping, which would make
    // the planner think a walking player had stopped.
    float velocity[3] = {0.0f, 0.0f, 0.0f};
    if (m_hasPreviousCameraPosition && deltaSeconds > 0.0f) {
        velocity[0] = (m_cameraX - m_previousCameraX) / deltaSeconds;
        velocity[1] = (m_cameraY - m_previousCameraY) / deltaSeconds;
        velocity[2] = (m_cameraZ - m_previousCameraZ) / deltaSeconds;
    }
    m_previousCameraX = m_cameraX;
    m_previousCameraY = m_cameraY;
    m_previousCameraZ = m_cameraZ;
    m_hasPreviousCameraPosition = true;

    // The planner ranks cells in FALLOUT space; the camera moves in engine
    // space. Converting is not optional -- feeding engine coordinates straight
    // in makes the grid's second axis the player's altitude, so streaming
    // follows how high they are rather than where they are.
    const float enginePosition[3] = {m_cameraX, m_cameraY, m_cameraZ};
    float falloutPosition[3] = {0.0f, 0.0f, 0.0f};
    float falloutVelocity[3] = {0.0f, 0.0f, 0.0f};
    importer::fnv::CellStreamer::engineToFallout(enginePosition, falloutPosition);
    importer::fnv::CellStreamer::engineToFallout(velocity, falloutVelocity);
    m_streamer->update(m_renderer, falloutPosition, falloutVelocity);

    // Once, after the first ring has settled.
    if (!m_collisionSelfTestDone && std::getenv("ODAI_FNV_COLLISION_TEST") != nullptr &&
        m_streamer->stats().residency.loadingCount == 0u &&
        m_streamer->stats().residentChunks > 0u) {
        m_collisionSelfTestDone = true;
        runCollisionSelfTest();
    }

    m_streamStatsLogTimer += deltaSeconds;
    if (m_streamStatsLogTimer >= 2.0f) {
        m_streamStatsLogTimer = 0.0f;
        const importer::fnv::CellStreamerStats stats = m_streamer->stats();
        VOX_LOGI("streamer") << "resident=" << stats.residentChunks
                             << " loading=" << stats.residency.loadingCount
                             << " loaded=" << stats.scenesLoaded
                             << " evicted=" << stats.residency.evictions
                             << " wasted=" << stats.residency.wastedLoads
                             << " missing=" << stats.residency.unavailableCells
                             << " applyMs(last/worst)=" << stats.lastApplyMs
                             << "/" << stats.worstApplyMs
                             << " buildMs(last/worst)=" << stats.lastBuildMs
                             << "/" << stats.worstBuildMs
                             << " cache(hit/miss)=" << stats.cacheHits << "/" << stats.cacheMisses
                             << " cacheLoadMs=" << stats.lastCacheLoadMs
                             << " fxSkipped=" << stats.effectMeshesSkipped
                             << " blendedDraws=" << stats.blendedPartsLoaded;
    }
}

void NewVegasApp::onTick(float deltaSeconds) {
    // Before anything reads input: the menu toggle decided here gates whether
    // camera movement runs at all this frame.
    pollNavInput(deltaSeconds);
    m_toasts.update(deltaSeconds);
    // The banner is a WORLD event, so it pauses with the world. Letting it run
    // under an open menu means a discovery fades in and out behind a modal
    // panel and the player never sees the one thing it existed to tell them --
    // and while it lasted, two pieces of large centred type sat on top of each
    // other. Held here, it plays the moment the menu closes.
    if (!m_menuOpen) {
        m_banner.update(deltaSeconds);
    }
    // Region lookup walks the cell index, so it is polled a few times a second
    // rather than every frame. A player cannot cross a 4096-unit cell in less
    // than that even sprinting, so nothing is missed.
    m_regionPollSeconds += deltaSeconds;
    if (m_regionPollSeconds >= 0.25f) {
        m_regionPollSeconds = 0.0f;
        updateRegionDiscovery();
    }

    if (keyDown(m_window, GLFW_KEY_ESCAPE)) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        return;
    }

    updateCamera(deltaSeconds);
    updateStreaming(deltaSeconds);

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

    // Edge-latched: holding E must not re-trigger on the door you arrive next
    // to, which is always within range of the one you just came through.
    const bool doorPressed = keyDown(m_window, GLFW_KEY_E);
    if (doorPressed && !m_doorKeyLatch) {
        const int doorIndex = findUsableDoor();
        if (doorIndex >= 0) {
            useDoor(m_doors[static_cast<std::size_t>(doorIndex)]);
        }
    }
    m_doorKeyLatch = doorPressed;

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


// ---------------------------------------------------------------------------
// Console-friendly UI: nav input, region-discovery toasts, Pip-Boy HUD.

namespace {

// Pip-Boy phosphor. One palette, used by every piece of chrome below, so the
// HUD reads as one instrument rather than a pile of independently styled boxes.
constexpr ui::UiColor kPipGreen{0.42f, 1.00f, 0.52f, 1.00f};
constexpr ui::UiColor kPipGreenDim{0.26f, 0.66f, 0.32f, 1.00f};
constexpr ui::UiColor kPipPanel{0.02f, 0.07f, 0.03f, 0.82f};
constexpr ui::UiColor kPipPanelSolid{0.02f, 0.07f, 0.03f, 0.95f};

float deadzone(float value, float threshold) {
    if (value > -threshold && value < threshold) {
        return 0.0f;
    }
    return value;
}

}  // namespace

void NewVegasApp::pollNavInput(float deltaSeconds) {
    m_nav.beginFrame();

    const bool keyUp = keyDown(m_window, GLFW_KEY_UP);
    const bool keyDownArrow = keyDown(m_window, GLFW_KEY_DOWN);
    const bool keyLeft = keyDown(m_window, GLFW_KEY_LEFT);
    const bool keyRight = keyDown(m_window, GLFW_KEY_RIGHT);
    bool accept = keyDown(m_window, GLFW_KEY_ENTER);
    bool cancel = keyDown(m_window, GLFW_KEY_ESCAPE);
    bool menu = cancel;

    // Gamepad, when one is present. GLFW's gamepad mapping gives the same
    // button indices for every recognized pad, so this needs no per-controller
    // handling -- an unmapped joystick simply reports false here rather than
    // producing garbage input.
    float stickX = 0.0f;
    float stickY = 0.0f;
    GLFWgamepadstate pad{};
    const bool hasPad = glfwJoystickIsGamepad(GLFW_JOYSTICK_1) == GLFW_TRUE &&
        glfwGetGamepadState(GLFW_JOYSTICK_1, &pad) == GLFW_TRUE;
    if (hasPad) {
        constexpr float kStickDeadzone = 0.25f;
        stickX = deadzone(pad.axes[GLFW_GAMEPAD_AXIS_LEFT_X], kStickDeadzone);
        stickY = deadzone(pad.axes[GLFW_GAMEPAD_AXIS_LEFT_Y], kStickDeadzone);
        accept = accept || pad.buttons[GLFW_GAMEPAD_BUTTON_A] == GLFW_PRESS;
        cancel = cancel || pad.buttons[GLFW_GAMEPAD_BUTTON_B] == GLFW_PRESS;
        menu = menu || pad.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS;
        m_nav.setAction(ui::UiNavAction::PrevTab,
                        pad.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] == GLFW_PRESS);
        m_nav.setAction(ui::UiNavAction::NextTab,
                        pad.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS);
    }

    // Stick first, then OR in the d-pad and arrow keys. The stick mapper owns
    // the four directions' latched state, so digital sources are folded in
    // after it rather than fighting it for the same flags.
    m_navStick.apply(m_nav, stickX, stickY);
    const bool padUp = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP] == GLFW_PRESS;
    const bool padDown = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN] == GLFW_PRESS;
    const bool padLeft = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT] == GLFW_PRESS;
    const bool padRight = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] == GLFW_PRESS;
    if (keyUp || padUp) { m_nav.setAction(ui::UiNavAction::Up, true); }
    if (keyDownArrow || padDown) { m_nav.setAction(ui::UiNavAction::Down, true); }
    if (keyLeft || padLeft) { m_nav.setAction(ui::UiNavAction::Left, true); }
    if (keyRight || padRight) { m_nav.setAction(ui::UiNavAction::Right, true); }
    m_nav.setAction(ui::UiNavAction::Accept, accept);
    m_nav.setAction(ui::UiNavAction::Cancel, cancel);
    m_nav.setAction(ui::UiNavAction::Menu, menu);

    m_navRepeat.update(m_nav, deltaSeconds);

    // Any directional or accept input means a controller is driving. The mouse
    // takes it back in updateCamera, which is where mouse motion is already
    // being read.
    m_navDriving = m_navDriving || m_nav.active;
    m_nav.active = false;

    if (m_nav.pressed(ui::UiNavAction::Menu)) {
        m_menuOpen = !m_menuOpen;
        // Releasing the mouse with the menu up is what makes it usable on PC;
        // on a controller it costs nothing.
        setMouseCaptured(!m_menuOpen);
    }
}

void NewVegasApp::updateRegionDiscovery() {
    if (!m_streamer) {
        return;
    }
    const float position[3] = {m_cameraX, m_cameraY, m_cameraZ};
    for (const std::string& name : m_streamer->regionNamesAtEngineSpace(position)) {
        // insert() reports whether it was new, so the "have I seen this?" check
        // and the record of having seen it are one operation -- there is no
        // window where a second call in the same frame announces it twice.
        if (!m_discoveredRegions.insert(name).second) {
            continue;
        }
        VOX_LOGI("newvegas") << "discovered region: " << name;
        // Keyed on the region so a player standing on a cell boundary, where
        // the streamer flips between two cells, refreshes one announcement
        // instead of queueing a run of identical ones.
        m_banner.push(name, "Location discovered", "region:" + name);
    }
}

void NewVegasApp::drawPipBoyHud() {
    const float scale = contentScale();
    int screenWidth = 0;
    int screenHeight = 0;
    framebufferSize(screenWidth, screenHeight);
    const float margin = 16.0f * scale;

    // Status strip, bottom-left: the readouts that belong on screen all the
    // time. Kept to one line so it never competes with the world.
    const int hours = static_cast<int>(m_timeOfDayHours);
    const int minutes = static_cast<int>((m_timeOfDayHours - static_cast<float>(hours)) * 60.0f);
    char status[192];
    const std::size_t regionCount = m_discoveredRegions.size();
    std::snprintf(
        status, sizeof(status), "%02d:%02d%s   %s   %zu region%s",
        hours, minutes, m_dayCyclePaused ? " (paused)" : "",
        m_walkMode ? "ON FOOT" : "FLY", regionCount, regionCount == 1u ? "" : "s");

    const float statusWidth = m_uiFont.measureText(status) + (margin * 1.5f);
    const float statusHeight = m_uiFont.lineHeightPx() + (10.0f * scale);
    ui::UiRect statusRect{};
    statusRect.minX = margin;
    statusRect.maxX = margin + statusWidth;
    statusRect.maxY = static_cast<float>(screenHeight) - margin;
    statusRect.minY = statusRect.maxY - statusHeight;
    m_uiDrawList.addRoundRectFilled(statusRect, kPipPanel, 3.0f * scale);
    m_uiDrawList.addRoundRect(statusRect, kPipGreenDim, 3.0f * scale, 1.0f * scale);
    m_uiDrawList.addText(
        m_uiFont, status,
        ui::UiVec2{statusRect.minX + (margin * 0.75f), statusRect.minY + (5.0f * scale)},
        kPipGreen);

    // Interaction prompt, centred low -- where an action prompt belongs, and
    // labelled for whichever device is driving.
    const int usableDoor = findUsableDoor();
    if (usableDoor >= 0 && !m_menuOpen) {
        const importer::ImportedSceneDoor& door = m_doors[static_cast<std::size_t>(usableDoor)];
        char prompt[192];
        std::snprintf(prompt, sizeof(prompt), "%s  %s", m_navDriving ? "(A)" : "[E]",
                      door.targetCellEditorId.empty() ? "Exit" : door.targetCellEditorId.c_str());
        const float promptWidth = m_uiFont.measureText(prompt);
        ui::UiVec2 promptPosition{};
        promptPosition.x = (static_cast<float>(screenWidth) - promptWidth) * 0.5f;
        promptPosition.y = static_cast<float>(screenHeight) - (96.0f * scale);
        m_uiDrawList.addText(m_uiFont, prompt, promptPosition, kPipGreen);
    }

    // Hint line, top-left. Names the buttons of whichever device is in use --
    // showing "Tab" to someone holding a controller is worse than showing
    // nothing.
    const char* hint = m_navDriving
        ? "(Start) menu   (LS) move   (A) use"
        : "Esc menu   [ ] time   P cycle   Tab cursor";
    m_uiDrawList.addText(m_uiFont, hint, ui::UiVec2{margin, margin}, kPipGreenDim);
}

void NewVegasApp::drawPauseMenu() {
    if (!m_menuOpen) {
        // Keep the ring empty so a stale focus index cannot survive a close and
        // reopen and act on the wrong entry.
        m_menuFocus.beginFrame();
        return;
    }
    const float scale = contentScale();
    int screenWidth = 0;
    int screenHeight = 0;
    framebufferSize(screenWidth, screenHeight);

    // Dim the world so the menu is unambiguously modal.
    ui::UiRect full{0.0f, 0.0f, static_cast<float>(screenWidth), static_cast<float>(screenHeight)};
    m_uiDrawList.addRectFilled(full, ui::UiColor{0.0f, 0.02f, 0.0f, 0.55f});

    struct Entry {
        const char* label;
        const char* value;
    };
    char timeValue[32];
    std::snprintf(timeValue, sizeof(timeValue), "%s", m_dayCyclePaused ? "Paused" : "Running");
    char regionValue[32];
    std::snprintf(regionValue, sizeof(regionValue), "%zu", m_discoveredRegions.size());
    const Entry entries[] = {
        {m_walkMode ? "Movement: On Foot" : "Movement: Fly", ""},
        {"Day cycle", timeValue},
        {"Regions discovered", regionValue},
        {"Resume", ""},
    };
    constexpr std::size_t kEntryCount = sizeof(entries) / sizeof(entries[0]);

    // Panel metrics as three explicit bands -- header, rows, footer -- rather
    // than one fudged total. The first version folded the footer into a single
    // padding constant and the footer text landed on top of the last row: the
    // arithmetic has to close, and it only closes if each band is named.
    // Every band is derived from the font's line height rather than being a
    // fixed pixel count. The type scale moved once already (to a TV size) and
    // the fixed values silently stopped fitting -- the header ran into the
    // first row. Derived bands cannot drift out of step with the type.
    const float lineHeight = m_uiFont.lineHeightPx();
    const float rowHeight = lineHeight + (16.0f * scale);
    const float headerBand = lineHeight + (28.0f * scale);
    const float footerBand = lineHeight + (22.0f * scale);
    // Wide enough for the widest row, so a longer label cannot overrun the
    // panel it is drawn inside.
    float contentWidth = m_uiFont.measureText("PIP-BOY 3000");
    for (const Entry& entry : entries) {
        const float rowWidth = m_uiFont.measureText(entry.label) +
            (entry.value[0] != '\0' ? m_uiFont.measureText(entry.value) + (48.0f * scale) : 0.0f);
        contentWidth = std::max(contentWidth, rowWidth);
    }
    const float panelWidth = std::max(460.0f * scale, contentWidth + (64.0f * scale));
    const float panelHeight =
        headerBand + (rowHeight * static_cast<float>(kEntryCount)) + footerBand;
    ui::UiRect panel{};
    panel.minX = (static_cast<float>(screenWidth) - panelWidth) * 0.5f;
    panel.maxX = panel.minX + panelWidth;
    panel.minY = (static_cast<float>(screenHeight) - panelHeight) * 0.5f;
    panel.maxY = panel.minY + panelHeight;
    m_uiDrawList.addRoundRectFilled(panel, kPipPanelSolid, 4.0f * scale);
    m_uiDrawList.addRoundRect(panel, kPipGreen, 4.0f * scale, 1.5f * scale);
    m_uiDrawList.addText(
        m_uiFont, "PIP-BOY 3000",
        ui::UiVec2{panel.minX + (24.0f * scale), panel.minY + (14.0f * scale)}, kPipGreen);

    // Register every row with the focus ring, THEN navigate. Navigating with a
    // partial list would let the first row absorb every move.
    m_menuFocus.beginFrame();
    ui::UiRect rows[kEntryCount];
    for (std::size_t i = 0; i < kEntryCount; ++i) {
        ui::UiRect row{};
        row.minX = panel.minX + (16.0f * scale);
        row.maxX = panel.maxX - (16.0f * scale);
        row.minY = panel.minY + headerBand + (static_cast<float>(i) * rowHeight);
        row.maxY = row.minY + rowHeight - (4.0f * scale);
        rows[i] = row;
        m_menuFocus.addItem(row);
    }
    if (!m_navDriving) {
        double cursorX = 0.0;
        double cursorY = 0.0;
        glfwGetCursorPos(m_window, &cursorX, &cursorY);
        m_menuFocus.focusHovered(
            ui::UiVec2{static_cast<float>(cursorX), static_cast<float>(cursorY)});
    }
    m_menuFocus.applyNavigation(m_nav);

    for (std::size_t i = 0; i < kEntryCount; ++i) {
        const bool focused = m_menuFocus.isFocused(static_cast<int>(i));
        if (focused) {
            m_uiDrawList.addRoundRectFilled(
                rows[i], ui::UiColor{0.16f, 0.42f, 0.20f, 0.85f}, 3.0f * scale);
        }
        m_uiDrawList.addText(
            m_uiFont, entries[i].label,
            ui::UiVec2{rows[i].minX + (16.0f * scale), rows[i].minY + (8.0f * scale)},
            focused ? kPipGreen : kPipGreenDim);
        if (entries[i].value[0] != '\0') {
            const float valueWidth = m_uiFont.measureText(entries[i].value);
            m_uiDrawList.addText(
                m_uiFont, entries[i].value,
                ui::UiVec2{rows[i].maxX - valueWidth - (16.0f * scale), rows[i].minY + (8.0f * scale)},
                focused ? kPipGreen : kPipGreenDim);
        }
    }

    if (m_nav.pressed(ui::UiNavAction::Accept)) {
        switch (m_menuFocus.focused()) {
            case 0: m_walkMode = !m_walkMode; break;
            case 1: m_dayCyclePaused = !m_dayCyclePaused; break;
            case 2: break;  // a readout, not an action
            case 3: m_menuOpen = false; setMouseCaptured(true); break;
            default: break;
        }
    }

    const char* footer = m_navDriving ? "(A) select    (B) back" : "Enter select    Esc back";
    m_uiDrawList.addText(
        m_uiFont, footer,
        ui::UiVec2{panel.minX + (24.0f * scale), panel.maxY - footerBand + (12.0f * scale)},
        kPipGreenDim);
}

void NewVegasApp::drawHud() {
    drawPipBoyHud();
    drawPauseMenu();

    // Toasts last so they sit above the menu: a discovery that fires while the
    // menu is open must not be hidden behind it.
    int screenWidth = 0;
    int screenHeight = 0;
    framebufferSize(screenWidth, screenHeight);
    const ui::UiRect screen{
        0.0f, 0.0f, static_cast<float>(screenWidth), static_cast<float>(screenHeight)};
    m_toasts.draw(m_uiDrawList, m_uiFont, screen, contentScale());
    // The banner draws its title in the display face and its subtitle in the
    // body face -- the size jump between them is what makes the location name
    // read as the headline rather than as another line of HUD text. Falls back
    // to the body face if loadFonts was not given a display size.
    if (!m_menuOpen) {
        const ui::Font& bannerFont = m_uiFontDisplay.valid() ? m_uiFontDisplay : m_uiFont;
        m_banner.draw(m_uiDrawList, bannerFont, m_uiFont, screen, contentScale());
    }
}

void NewVegasApp::onRender(float /*deltaSeconds*/) {
    // Before beginFrameDraw: the backend consumes the pending pose while
    // recording this frame, so setting it afterwards would always be a frame
    // late -- invisible on a still bind pose, and a lag on an animated one.
    if (m_characterMode) {
        updateCharacterPose();
    }
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
