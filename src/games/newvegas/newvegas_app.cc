#include "games/newvegas/newvegas_app.h"

#include "import/fnv/dialogue_records.h"

#include "import/dds.h"
#include "games/newvegas/newvegas_ogg.h"
#include "import/fnv/bsa_archive.h"

#include <fstream>
#include <random>
#include <chrono>

#include "core/log.h"
#include "import/fnv/asset_source.h"
#include "import/fnv/nif_scene.h"
#include "ui/ui_types.h"

#include <cstdio>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <limits>
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

    // Conversation type. Baked here rather than through loadFonts because
    // GameApp's four slots (body/numeric/caption/display) are a shared contract
    // every game uses, and widening it for one game's dialogue would change all
    // of them. registerUiFontAtlas is public for exactly this.
    //
    // 48/40 px against a 28 px body: a reply the player has to READ and CHOOSE
    // from across a room is not the same reading task as a status strip, and
    // the previous pass drew both at body size in a corner.
    constexpr float kDialogueLineSize = 48.0f;
    constexpr float kDialogueChoiceSize = 40.0f;
    const std::string regularFontPath = resolveAssetPath("assets/fonts/Inter-Regular.ttf");
    if (m_dialogueFont.loadFromFile(regularFontPath, kDialogueLineSize)) {
        m_dialogueFont.setTextureId(m_renderer.registerUiFontAtlas(
            m_dialogueFont.atlasPixels().data(), m_dialogueFont.atlasWidth(),
            m_dialogueFont.atlasHeight()));
    }
    if (m_dialogueChoiceFont.loadFromFile(regularFontPath, kDialogueChoiceSize)) {
        m_dialogueChoiceFont.setTextureId(m_renderer.registerUiFontAtlas(
            m_dialogueChoiceFont.atlasPixels().data(), m_dialogueChoiceFont.atlasWidth(),
            m_dialogueChoiceFont.atlasHeight()));
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
    // Temporal AA. This is what stops textured surfaces shimmering in motion
    // -- measured at 13x the frame-to-frame instability of flat-shaded
    // geometry before TAA existed. ODAI_TAA=0 turns it off for A/B.
    {
        const char* taaEnv = std::getenv("ODAI_TAA");
        const bool taaEnabled = taaEnv == nullptr || taaEnv[0] != '0';
        m_renderer.setTaaEnabled(taaEnabled);
    }

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
    // Neutral colour grade. The post chain's defaults are a stylized look and
    // are applied with no enable gate, so this viewer inherited +8% saturation,
    // +12% vibrance, +10% contrast and an 8% blue cut on top of the tonemap.
    // Measured on a Goodsprings frame that put mean pixel saturation at 0.43
    // with a p90 of 0.80 -- a vivid image of a landscape that is meant to read
    // as dust and sun-bleached tan.
    //
    // ODAI_FNV_COLOR_LOOK=stylized restores the defaults. There was no runtime
    // knob for any of this, which is why the report of "oversaturated" had to
    // be answered by reading the shader instead of an A/B.
    if (const char* look = std::getenv("ODAI_FNV_COLOR_LOOK");
        look == nullptr || std::string(look) != "stylized") {
        m_renderer.setNeutralColorGrading();
    }

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

    // After streaming init, so a failed stream never leaves weather half-set,
    // and after applyTimeOfDay so the first push uses the real hour.
    if (const char* pluginsEnv = std::getenv("ODAI_FNV_PLUGINS")) {
        const std::string plugins = pluginsEnv;
        std::size_t start = 0;
        while (start <= plugins.size()) {
            const std::size_t end = plugins.find(',', start);
            const std::string entry =
                plugins.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!entry.empty()) {
                m_extraPlugins.push_back(entry);
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
    }
    if (const char* weatherEnv = std::getenv("ODAI_FNV_WEATHER")) {
        m_requestedWeatherEditorId = weatherEnv;
    }
    initWeather();

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
    // ODAI_FNV_CHAR_IDENTITY=1 submits a pose of the right SHAPE but with no
    // data in it: as many identity matrices as the slot expects. It separates
    // the two ways this path can fail, which look identical on screen -- a
    // mechanical fault in the per-frame upload (wrong buffer, missing barrier)
    // still corrupts the frame with identity matrices, whereas a fault in the
    // bind-pose MATRICES themselves cannot, and the actor merely collapses to
    // the origin.
    if (std::getenv("ODAI_FNV_CHAR_IDENTITY") != nullptr) {
        m_characterPoseScratch.assign(m_characterBindPose.size(), odai::math::Matrix4::identity());
        render::ImportedSkinnedActorFrameData identityPose{};
        identityPose.boneMatrices = m_characterPoseScratch;
        m_renderer.setSkinnedActorPose(0u, identityPose);
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
    static bool loggedPose = false;
    if (!loggedPose && !m_characterPoseScratch.empty()) {
        loggedPose = true;
        const odai::math::Matrix4& b = m_characterBindPose[0];
        const odai::math::Matrix4& f = m_characterPoseScratch[0];
        VOX_LOGI("newvegas") << "pose[0] bind translation (" << b(0, 3) << "," << b(1, 3) << ","
                             << b(2, 3) << ") final (" << f(0, 3) << "," << f(1, 3) << ","
                             << f(2, 3) << ") bones=" << m_characterPoseScratch.size();
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
    applyWeather();
}

void NewVegasApp::initWeather() {
    if (m_streamDirectory.empty()) {
        return;  // a cooked scene has no plugin to read weather from
    }
    // Nothing to gain from reading 473 weather records when none of them can be
    // selected and the procedural sky is what will render anyway.
    if (m_extraPlugins.empty() && m_requestedWeatherEditorId.empty()) {
        return;
    }

    std::vector<std::string> requested;
    requested.push_back(m_streamPlugin);
    requested.insert(requested.end(), m_extraPlugins.begin(), m_extraPlugins.end());

    importer::fnv::FalloutLoadOrder order;
    // Mod directories are searched for the plugin too, so a mod that ships an
    // .esp beside its .bsa needs nothing copied into the game install.
    for (const std::string& modDirectory : m_modDirectories) {
        order.addSearchRoot(std::filesystem::path(modDirectory));
    }
    std::string error;
    if (!order.open(std::filesystem::path(m_streamDirectory), requested, error)) {
        VOX_LOGW("newvegas") << "weather disabled: " << error;
        return;
    }
    if (!buildFalloutWeatherTables(order, m_weatherTables, error)) {
        VOX_LOGW("newvegas") << "weather disabled: " << error;
        return;
    }

    std::string loadOrderText;
    for (const auto& entry : order.entries()) {
        if (!loadOrderText.empty()) {
            loadOrderText += " -> ";
        }
        loadOrderText += entry.header.fileName;
    }
    VOX_LOGI("newvegas") << "load order: " << loadOrderText;
    VOX_LOGI("newvegas") << "weather: " << m_weatherTables.weathers.size() << " WTHR, "
                         << m_weatherTables.climates.size() << " CLMT";

    if (!m_requestedWeatherEditorId.empty()) {
        const importer::fnv::FalloutWeatherRecord* weather =
            m_weatherTables.findWeatherByEditorId(m_requestedWeatherEditorId);
        if (weather == nullptr) {
            VOX_LOGW("newvegas") << "no weather named \"" << m_requestedWeatherEditorId
                                 << "\"; falling back to the climate";
        } else {
            m_activeWeatherFormId = weather->formId;
            VOX_LOGI("newvegas") << "weather forced to " << weather->editorId << " (0x"
                                 << std::hex << weather->formId << std::dec << ")";
        }
    }

    if (m_activeWeatherFormId == 0u) {
        // Fall back to the worldspace's own climate: whichever of its weathers
        // has the highest chance is the closest thing to "what you would
        // normally see here" without running the mod's selection scripts.
        const importer::fnv::FalloutClimateRecord* bestClimate = nullptr;
        for (const auto& [worldspaceFormId, climateFormId] :
             m_weatherTables.climateByWorldspaceFormId) {
            (void)worldspaceFormId;
            const auto found = m_weatherTables.climates.find(climateFormId);
            if (found != m_weatherTables.climates.end() && !found->second.weathers.empty()) {
                bestClimate = &found->second;
                break;
            }
        }
        if (bestClimate != nullptr) {
            const auto best = std::max_element(
                bestClimate->weathers.begin(), bestClimate->weathers.end(),
                [](const auto& a, const auto& b) { return a.chance < b.chance; });
            m_activeWeatherFormId = best->weatherFormId;
            const importer::fnv::FalloutWeatherRecord* weather =
                m_weatherTables.findWeather(m_activeWeatherFormId);
            VOX_LOGI("newvegas") << "weather from climate " << bestClimate->editorId << ": "
                                 << (weather != nullptr ? weather->editorId : "<unresolved>");
        }
    }

    // Cloud layers. These come out of the mod's own BSA, which the streamer's
    // asset source already indexes -- reusing it means no second search path and
    // no second copy of the loose-beats-archive precedence rules.
    const importer::fnv::FalloutWeatherRecord* active =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    if (active != nullptr && m_streamer != nullptr) {
        const importer::fnv::FalloutAssetSource& assets = m_streamer->assets();
        render::WeatherCloudTextures clouds;
        int loadedLayers = 0;
        for (std::size_t layer = 0; layer < importer::fnv::FalloutWeatherRecord::kCloudLayerCount;
             ++layer) {
            m_cloudLayerEnabled[layer] = false;
            const std::string& path = active->cloudTextures[layer];
            if (importer::fnv::isEmptyCloudLayer(path)) {
                continue;
            }
            std::vector<std::uint8_t> bytes;
            std::string assetError;
            if (!assets.resolveTexture(path, bytes, assetError)) {
                VOX_LOGW("newvegas") << "cloud layer " << layer << " (" << path
                                     << ") unresolved: " << assetError;
                continue;
            }
            if (!importer::loadDdsFromMemory(bytes.data(), bytes.size(), clouds.layers[layer])) {
                VOX_LOGW("newvegas") << "cloud layer " << layer << " (" << path << ") failed to decode";
                clouds.layers[layer] = importer::ImportedSceneTexture{};
                continue;
            }
            clouds.layers[layer].sourcePath = path;
            // Layers 0/1 are the lower pair and 2/3 the upper, which is what the
            // record's two cloud speeds refer to. Speeds are stored 0..255 over
            // a range the game treats as roughly +/- one texture width a minute.
            const std::uint8_t speedByte =
                (layer < 2) ? active->cloudSpeedLower : active->cloudSpeedUpper;
            // 128 is "still"; either side of it scrolls in opposite directions.
            // Radians per second about the zenith -- a dome map rotates, it does
            // not translate. 128 is "still"; either side turns the other way.
            clouds.scrollSpeed[layer] =
                (static_cast<float>(speedByte) - 128.0f) / 128.0f * 0.0035f;
            // Dome scale: 1.0 puts the horizon exactly on the texture's
            // inscribed circle, which is how these fisheye sky maps are drawn.
            // Slightly under 1 for the upper layers pulls their rim inside the
            // horizon so they read as higher and further away.
            clouds.domeScale[layer] = (layer < 2) ? 1.0f : 0.92f;
            m_cloudLayerEnabled[layer] = true;
            ++loadedLayers;
        }
        VOX_LOGI("newvegas") << "cloud layers: " << loadedLayers << " of 4 in use";
        m_renderer.setWeatherClouds(clouds);
    }

    applyWeather();
    initWeatherAudio();

    // ODAI_FNV_TONEMAP=enb switches the post pass to the curve Enhanced Shaders
    // uses, with its own tuned Fallout values. Off by default: it is a distinct
    // look, not a strict improvement, and every other game keeps the ACES fit
    // regardless because the setting is per-renderer and only this game sets it.
    if (const char* tonemapEnv = std::getenv("ODAI_FNV_TONEMAP")) {
        std::string mode = tonemapEnv;
        for (char& c : mode) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (mode == "enb" || mode == "1") {
            render::TonemapSettings tonemap;
            tonemap.mode = render::TonemapMode::Enb;
            // Enhanced Shaders retunes these by time of day (contrast 1.35 day /
            // 1.25 night, saturation 1.25 / 0.9, curve 8.0 / 10.0). Interpolate
            // on the same day/night axis rather than picking one and calling it
            // done -- the night values exist because the day ones look wrong
            // after dark.
            const bool night = m_timeOfDayHours < 6.0f || m_timeOfDayHours >= 19.0f;
            tonemap.contrast = night ? 1.25f : 1.35f;
            tonemap.saturation = night ? 0.90f : 1.25f;
            tonemap.curve = night ? 10.0f : 8.0f;
            tonemap.overbrightDampening = night ? 50.0f : 75.0f;
            m_renderer.setTonemapSettings(tonemap);
            VOX_LOGI("newvegas") << "tonemap: ENB (Enhanced Shaders values, "
                                 << (night ? "night" : "day") << ")";
        }
    }
}

namespace {

// Writes already-resolved sound bytes to a playable file and returns its path.
//
// Two reasons this exists rather than handing bytes straight to the audio
// facade. It loads by std::filesystem::path only, and Fallout's ambient loops
// are Ogg Vorbis, which miniaudio cannot decode at all -- so the .ogg is
// converted to .wav here (see newvegas_ogg.cc). Cached by name, so a sound
// costs one conversion per install rather than one per run.
std::filesystem::path cacheWeatherSound(
    const std::string& virtualPath,
    const std::vector<std::uint8_t>& bytes,
    const std::filesystem::path& cacheDirectory) {
    if (cacheDirectory.empty() || bytes.empty()) {
        return {};
    }
    std::string leaf = virtualPath;
    const std::size_t lastSeparator = leaf.find_last_of("\\/");
    if (lastSeparator != std::string::npos) {
        leaf = leaf.substr(lastSeparator + 1u);
    }
    const std::filesystem::path raw = cacheDirectory / leaf;
    std::filesystem::path playable = raw;
    const bool needsConversion = raw.extension() == ".ogg";
    if (needsConversion) {
        playable.replace_extension(".wav");
    }

    std::error_code existsError;
    if (std::filesystem::exists(playable, existsError) && !existsError) {
        return playable;
    }

    std::error_code createError;
    std::filesystem::create_directories(cacheDirectory, createError);
    {
        std::ofstream out(raw, std::ios::binary | std::ios::trunc);
        if (!out) {
            return {};
        }
        out.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (needsConversion && !decodeOggToWav(raw, playable)) {
        return {};
    }
    return playable;
}

}  // namespace

void NewVegasApp::initWeatherAudio() {
    const importer::fnv::FalloutWeatherRecord* weather =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    if (weather == nullptr || m_streamDirectory.empty() || m_streamer == nullptr) {
        return;
    }

    const importer::fnv::FalloutAssetSource& assets = m_streamer->assets();
    const std::filesystem::path dataFilesPath(m_streamDirectory);
    const std::filesystem::path audioCache = m_streamCacheDirectory.empty()
        ? std::filesystem::path{}
        : std::filesystem::path(m_streamCacheDirectory) / "audio";

    // Resolves the first candidate that exists, through the ordinary asset
    // precedence, and caches it as a playable .wav.
    //
    // Order matters: a weather mod ships sounds authored for its own weathers
    // inside its BSA, which is indexed as a mod archive, so it wins. Reaching
    // past that into the base game's archives is how this first shipped, and it
    // picked "emt_raintoggle_lp" -- a MONO 6-second object-emitter loop from Old
    // World Blues, meant to play from a dripping pipe, not a global rain bed.
    const auto loadFirst = [&](std::initializer_list<const char*> candidates,
                               const char* what) -> audio::SoundHandle {
        for (const char* candidate : candidates) {
            std::vector<std::uint8_t> bytes;
            std::string assetError;
            if (!assets.resolveAsset(candidate, bytes, assetError) || bytes.empty()) {
                continue;
            }
            const std::filesystem::path cached =
                cacheWeatherSound(candidate, bytes, audioCache);
            if (cached.empty()) {
                continue;
            }
            const audio::SoundHandle handle =
                m_audio.loadSound(cached, audio::SoundCategory::Ambient);
            if (handle.id != 0u) {
                VOX_LOGI("newvegas") << what << ": " << candidate;
                return handle;
            }
        }
        VOX_LOGW("newvegas") << "no " << what << " found in the loaded archives";
        return {};
    };

    if (weather->hasPrecipitation()) {
        // WTHR has no rain-intensity field -- classification only says "rainy" --
        // so intensity comes from the editor ID, which is a heuristic and named
        // as one. The fallbacks walk down to whatever exists.
        std::string lowered = weather->editorId;
        for (char& c : lowered) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        const bool heavy = lowered.find("heavy") != std::string::npos ||
            lowered.find("storm") != std::string::npos;
        m_rainLoop = heavy
            ? loadFirst({"sound\\fx\\weather\\amb_weather_rain_heavy_lp.wav",
                         "sound\\fx\\weather\\amb_rainstorm_lp.wav",
                         "sound\\fx\\weather\\amb_weather_rain_medium_lp.wav",
                         "sound\\fx\\weather\\nvdlc02_rain-amb.wav"},
                        "rain")
            : loadFirst({"sound\\fx\\weather\\amb_weather_rain_medium_lp.wav",
                         "sound\\fx\\weather\\amb_weather_rain_light_lp.wav",
                         "sound\\fx\\weather\\amb_rain_lp.wav",
                         "sound\\fx\\weather\\nvdlc02_rain-amb.wav"},
                        "rain");
        if (m_rainLoop.id != 0u) {
            m_rainAmbient = m_audio.startAmbient(m_rainLoop, 2.5f);
        }
    }

    if (weather->windSpeed > 40u) {
        const bool strongWind = weather->windSpeed > 80u;
        m_windLoop = loadFirst(
            strongWind
                ? std::initializer_list<const char*>{
                      "sound\\fx\\weather\\amb_windheavy_lp.wav",
                      "sound\\fx\\weather\\amb_windlight_lp.wav"}
                : std::initializer_list<const char*>{
                      "sound\\fx\\weather\\amb_windlight_lp.wav",
                      "sound\\fx\\weather\\amb_windheavy_lp.wav"},
            "wind");
        if (m_windLoop.id != 0u) {
            m_windAmbient = m_audio.startAmbient(m_windLoop, 3.0f);
        }
    }

    // Radio, not score. Fallout keeps two separate sets of loose music: the
    // orchestral exploration beds under Data\Music, and the 48 licensed radio
    // songs under Data\Sound\songs\radionv -- Big Iron, Blue Moon, Johnny
    // Guitar. The radio station is the one that sounds like Fallout, and it is
    // what this plays.
    //
    // ODAI_FNV_MUSIC takes either a full path or a song name ("Big_Iron",
    // "MUS_Big_Iron", "MUS_Big_Iron.mp3"); with nothing set, a track is picked
    // from the station at random, like tuning in.
    std::filesystem::path musicPath;
    const std::filesystem::path stationDir = dataFilesPath / "Sound" / "songs" / "radionv";
    if (const char* musicEnv = std::getenv("ODAI_FNV_MUSIC")) {
        const std::string request = musicEnv;
        std::error_code existsError;
        if (std::filesystem::exists(request, existsError) && !existsError) {
            musicPath = request;
        } else {
            // Match by name, case-insensitively, with or without the MUS_ prefix
            // and the extension.
            std::string wanted = request;
            for (char& c : wanted) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            std::error_code iterError;
            std::filesystem::directory_iterator iterator(stationDir, iterError);
            if (!iterError) {
                for (const auto& entry : iterator) {
                    std::string name = entry.path().filename().string();
                    for (char& c : name) {
                        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    }
                    if (name.find(wanted) != std::string::npos) {
                        musicPath = entry.path();
                        break;
                    }
                }
            }
            if (musicPath.empty()) {
                VOX_LOGW("newvegas") << "no song matching \"" << request << "\" in "
                                     << stationDir.string();
            }
        }
    }
    if (musicPath.empty()) {
        std::vector<std::filesystem::path> station;
        std::error_code iterError;
        std::filesystem::directory_iterator iterator(stationDir, iterError);
        if (!iterError) {
            for (const auto& entry : iterator) {
                if (entry.path().extension() == ".mp3") {
                    station.push_back(entry.path());
                }
            }
        }
        if (!station.empty()) {
            // Sorted first so the pick depends only on the seed, not on readdir
            // order, which differs between machines.
            std::sort(station.begin(), station.end());
            std::mt19937 rng(static_cast<std::uint32_t>(
                std::chrono::steady_clock::now().time_since_epoch().count()));
            musicPath = station[rng() % station.size()];
        }
    }

    std::error_code musicError;
    if (!musicPath.empty() && std::filesystem::exists(musicPath, musicError) && !musicError) {
        m_musicTrack = m_audio.loadMusic(musicPath);
        if (m_musicTrack.id != 0u) {
            m_audio.playMusic(m_musicTrack, 4.0f, true);
            VOX_LOGI("newvegas") << "radio: " << musicPath.stem().string();
        }
    } else {
        VOX_LOGW("newvegas") << "no radio songs found under " << stationDir.string();
    }
}

void NewVegasApp::applyWeather() {
    const importer::fnv::FalloutWeatherRecord* weather =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    if (weather == nullptr) {
        return;  // leave the procedural sky alone
    }

    // WTHR colours are authored as sRGB bytes for a renderer that displayed them
    // directly. This one is HDR: the frame goes through an ACES curve and auto
    // exposure keyed to a sunlit desert sitting around 0.3 linear.
    //
    // Decoding to linear and stopping there is not enough, and the failure is
    // silent. A heavy-overcast sky is authored sRGB 23,27,30 -- linear 0.0086 --
    // which ACES maps to ~0.002 and the exposure scale then buries. The sky
    // rendered PURE BLACK while the terrain looked correctly lit, because the
    // values are display-referred and were being read as radiance.
    //
    // A flat gain cannot fix this. Measured on two real weathers: the gain that
    // makes a heavy overcast readable (~10) washes a clear zenith from deep blue
    // to pale haze, and the gain that keeps the blue (~3) puts the overcast back
    // at pure black. The pipeline's response below ~0.05 linear is far steeper
    // than the rest of its range, so darks need lifting MORE than brights.
    //
    // pow(linear, contrast) does exactly that, and one exponent covers both
    // cases where no single multiplier does. This is a display-referred fudge,
    // not physics; the principled fix is to invert the tonemap on the GPU, where
    // the auto-exposure scale is actually known. Both knobs are env-tunable.
    static const float s_skyContrast = []() {
        const char* env = std::getenv("ODAI_FNV_SKY_CONTRAST");
        const float value = env != nullptr ? static_cast<float>(std::atof(env)) : 0.60f;
        return value > 0.0f ? value : 0.60f;
    }();
    static const float s_skyGain = []() {
        const char* env = std::getenv("ODAI_FNV_SKY_GAIN");
        const float value = env != nullptr ? static_cast<float>(std::atof(env)) : 1.6f;
        return value > 0.0f ? value : 1.6f;
    }();
    // Enhanced Shaders runs 1.25 by day and 0.9 at night against these same
    // weather records; 1.15 is a compromise for a single global value.
    static const float s_skySaturation = []() {
        const char* env = std::getenv("ODAI_FNV_SKY_SATURATION");
        const float value = env != nullptr ? static_cast<float>(std::atof(env)) : 1.15f;
        return value > 0.0f ? value : 1.15f;
    }();
    const auto decode = [&](const importer::fnv::FalloutColorRgb& color, float* out) {
        const auto channel = [](std::uint8_t value) {
            const float srgb = static_cast<float>(value) / 255.0f;
            return srgb <= 0.04045f ? (srgb / 12.92f)
                                    : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
        };
        // Shape MAGNITUDE, keep HUE. Applying pow() per channel (which this
        // did first) pulls the channels toward each other, so an exponent
        // below 1 desaturates: it lifted the overcast greys correctly and
        // simultaneously washed a clear zenith from deep blue to pale haze.
        //
        // Splitting magnitude from direction is how ENB's tonemap does it
        // (Enhanced Shaders' enbeffect.fx: contrast on `color/normalize(color)`,
        // saturation on `normalize(color)`), and it fixes the hue shift for the
        // same reason -- the direction vector is left alone unless saturation
        // explicitly touches it.
        const float linear[3] = {channel(color.r), channel(color.g), channel(color.b)};
        const float magnitude =
            std::sqrt((linear[0] * linear[0]) + (linear[1] * linear[1]) + (linear[2] * linear[2]));
        if (magnitude <= 1e-6f) {
            out[0] = out[1] = out[2] = 0.0f;
            return;
        }
        const float shaped = std::pow(magnitude, s_skyContrast) * s_skyGain;
        for (int i = 0; i < 3; ++i) {
            // pow on the unit direction is ENB's saturation control; above 1
            // pushes the dominant channel further ahead of the others.
            out[i] = std::pow(linear[i] / magnitude, s_skySaturation) * shaped;
        }
    };

    using importer::fnv::FalloutWeatherColor;
    const float hour = m_timeOfDayHours;
    render::WeatherSkyParams params;
    params.weight = 1.0f;
    decode(sampleFalloutWeatherColor(*weather, FalloutWeatherColor::SkyUpper, hour), params.skyUpper);
    decode(sampleFalloutWeatherColor(*weather, FalloutWeatherColor::SkyLower, hour), params.skyLower);
    decode(sampleFalloutWeatherColor(*weather, FalloutWeatherColor::Horizon, hour), params.horizon);
    decode(sampleFalloutWeatherColor(*weather, FalloutWeatherColor::Fog, hour), params.fogColor);
    // Day fog until dusk, night fog after; the record authors the two
    // separately and there is no third value to interpolate toward.
    const bool daytime = hour >= 6.0f && hour < 19.0f;
    params.fogFarDistance = daytime ? weather->fogDayFar : weather->fogNightFar;

    // Cloud tints come from PNAM, one colour per layer per time slot, sampled
    // the same way the sky colours are. Layers with no texture were switched
    // off at upload time and their opacity is ignored.
    for (int layer = 0; layer < render::kWeatherCloudLayerCount; ++layer) {
        const importer::fnv::FalloutColorRgb tint =
            sampleFalloutWeatherCloudTint(*weather, layer, hour);
        decode(tint, params.cloudTint[layer]);
        // ODAI_FNV_NOCLOUDS isolates the sky gradient from the cloud layers.
        // Worth keeping: "the sky is black" has two very different causes
        // (an authored-dark gradient vs. total cloud cover) and they are
        // indistinguishable on screen.
        static const bool s_noClouds = std::getenv("ODAI_FNV_NOCLOUDS") != nullptr;
        params.cloudOpacity[layer] =
            (m_cloudLayerEnabled[layer] && !s_noClouds) ? 1.0f : 0.0f;
    }
    // One line per weather change, not per frame: "the sky is black" is
    // otherwise indistinguishable from "the sky is not being set at all".
    static std::uint32_t s_loggedWeather = 0;
    if (s_loggedWeather != m_activeWeatherFormId) {
        s_loggedWeather = m_activeWeatherFormId;
        VOX_LOGI("newvegas") << "sky linear rgb: upper(" << params.skyUpper[0] << ","
                             << params.skyUpper[1] << "," << params.skyUpper[2] << ") horizon("
                             << params.horizon[0] << "," << params.horizon[1] << ","
                             << params.horizon[2] << ") weight=" << params.weight;
    }
    m_renderer.setWeatherSky(params);
}

void NewVegasApp::updateCamera(float deltaSeconds) {
    // ODAI_FNV_BENCH=1 walks the camera forward on a slow turn instead of
    // reading input. "It is jittery when I move" is not reproducible from a
    // standing start, and a hand-driven walk is not comparable between runs --
    // this makes the motion identical every time so a frame-time change is
    // attributable to the code rather than to how the tester moved.
    static const bool s_bench = std::getenv("ODAI_FNV_BENCH") != nullptr;
    if (s_bench) {
        // ODAI_FNV_BENCH_FIXED_DT=1 advances by a FIXED step instead of real
        // elapsed time, which makes frame N land at exactly the same camera
        // position on every run. That is what lets two captures taken one frame
        // apart be compared: without it the walk depends on how fast the
        // machine happened to render, and any diff is dominated by the camera
        // having moved a different distance.
        static const bool s_fixedDt = std::getenv("ODAI_FNV_BENCH_FIXED_DT") != nullptr;
        // ODAI_FNV_BENCH_SPEED overrides the walk speed. A very low value is
        // what isolates temporal shimmer: with the camera barely moving between
        // two frames, anything that still differs is the renderer being
        // unstable rather than the world going past. Turn rate scales with it
        // so a slow walk is also a slow turn.
        static const float s_benchSpeed = []() {
            const char* env = std::getenv("ODAI_FNV_BENCH_SPEED");
            return env != nullptr ? static_cast<float>(std::atof(env)) : 400.0f;
        }();
        // ODAI_FNV_BENCH_TURN is the turn rate in degrees/second at the
        // default speed; 0 walks a straight line.
        //
        // The default 6 deg/s is a CIRCLE of radius speed/turn -- about 3820
        // units at 400 u/s, which is smaller than one 4096-unit exterior cell.
        // That is fine for measuring steady-state rendering (the point it was
        // written for) and useless for measuring streaming: the walk never
        // leaves the cells it started resident in, so no cell is ever loaded or
        // evicted mid-run. Testing traversal means setting this to 0.
        static const float s_benchTurn = []() {
            const char* env = std::getenv("ODAI_FNV_BENCH_TURN");
            return env != nullptr ? static_cast<float>(std::atof(env)) : 6.0f;
        }();
        // ODAI_FNV_BENCH_HEADING picks the initial compass direction, applied
        // once, so a straight-line run can be aimed at a specific neighbour
        // region instead of wherever spawn happened to face.
        static const float s_benchHeading = []() {
            const char* env = std::getenv("ODAI_FNV_BENCH_HEADING");
            return env != nullptr ? static_cast<float>(std::atof(env))
                                  : std::numeric_limits<float>::quiet_NaN();
        }();
        if (!std::isnan(s_benchHeading) && !m_benchHeadingApplied) {
            m_yawDegrees = s_benchHeading;
            m_benchHeadingApplied = true;
        }
        const float step = s_fixedDt ? (1.0f / 60.0f) : deltaSeconds;
        m_yawDegrees += (s_benchSpeed / 400.0f) * s_benchTurn * step;
        const float yawRadians = m_yawDegrees * (kPi / 180.0f);
        const float kBenchSpeed = s_benchSpeed;  // default ~5.7 m/s, a fast jog
        m_cameraX += std::cos(yawRadians) * kBenchSpeed * step;
        m_cameraZ += std::sin(yawRadians) * kBenchSpeed * step;
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
    // A conversation is MODAL, the way Skyrim's is: while it is up the player
    // neither walks nor looks around, and the camera turns onto the speaker.
    //
    // Mouselook is suppressed rather than the cursor being re-captured, and
    // m_lastCursorX/Y keep being written below either way. That is what makes
    // leaving a conversation seamless: the else-branch clears m_hasCursorSample,
    // so the first frame after the card closes applies no delta at all instead
    // of one worth however far the mouse travelled while it was open.
    const bool inConversation = m_victor.talking;
    if (m_mouseCaptured && !suppressMouseLook && !inConversation) {
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

    // Turn to the speaker's face and hold there.
    //
    // Aiming at his ORIGIN would point the camera at his wheel: he stands ~187
    // units tall and his feet are the placement. The face screen is the thing a
    // conversation is about, so that is what gets centred.
    //
    // Eased rather than snapped, and re-aimed every frame rather than once on
    // open: a hard cut to a new orientation is disorienting, and he is animated,
    // so a one-shot aim would drift off him as the idle moves him.
    // The dolly. Eased on the way in AND on the way out, so leaving a
    // conversation widens back rather than snapping.
    {
        constexpr float kFovTauSeconds = 0.22f;
        const float targetFov = inConversation ? kConversationFovDegrees : kDefaultFovDegrees;
        const float blend = 1.0f - std::exp(-deltaSeconds / kFovTauSeconds);
        m_cameraFovDegrees += (targetFov - m_cameraFovDegrees) * blend;
    }

    if (inConversation && m_victor.placed) {
        constexpr float kVictorFaceHeightUnits = 150.0f;
        // Time constant, not a per-frame fraction: a fixed fraction converges
        // at whatever rate the machine happens to render at, so the turn would
        // be visibly faster on a fast GPU.
        constexpr float kAimTauSeconds = 0.12f;
        const float dx = m_victor.position[0] - m_cameraX;
        const float dy = (m_victor.position[1] + kVictorFaceHeightUnits) - m_cameraY;
        const float dz = m_victor.position[2] - m_cameraZ;
        const float horizontal = std::sqrt((dx * dx) + (dz * dz));
        if (horizontal > 1e-3f) {
            const float desiredYaw = std::atan2(dz, dx) * (180.0f / kPi);

            // Aiming AT his face centres it -- directly behind the card, which
            // is the one thing a conversation must not hide. Skyrim keeps the
            // speaker in frame and puts the words under them, so the camera
            // aims low enough that his face rises to just above the card's top
            // edge.
            //
            // The offset is derived from the projection rather than dialled in
            // by eye: a point f half-heights above centre subtends
            // atan(f * tan(fovY/2)), so the pitch has to come DOWN by that much
            // to lift the face there. A hardcoded degree count would drift the
            // moment the FOV changed.
            int framebufferWidth = 0;
            int framebufferHeight = 0;
            framebufferSize(framebufferWidth, framebufferHeight);
            float pitchOffsetDegrees = 0.0f;
            if (framebufferHeight > 0) {
                const auto heightPx = static_cast<float>(framebufferHeight);
                // Before the card has ever been drawn there is no measured top
                // edge; 0.30 is where a typical four-reply card starts.
                const float panelTopPx =
                    m_dialoguePanelTopPx > 1.0f ? m_dialoguePanelTopPx : (heightPx * 0.30f);
                const float faceTargetPx =
                    std::max(heightPx * 0.10f, panelTopPx - (heightPx * 0.07f));
                const float halfHeights =
                    std::clamp(((heightPx * 0.5f) - faceTargetPx) / (heightPx * 0.5f), 0.0f, 0.9f);
                // The LIVE fov, not the default: it is easing while this runs,
                // and the offset that lands his face above the card is a
                // function of it. Using the constant here would slide the
                // framing down over the length of the zoom.
                const float halfFovTangent =
                    std::tan((m_cameraFovDegrees * 0.5f) * (kPi / 180.0f));
                pitchOffsetDegrees =
                    std::atan(halfHeights * halfFovTangent) * (180.0f / kPi);
            }
            const float desiredPitch = std::clamp(
                (std::atan2(dy, horizontal) * (180.0f / kPi)) - pitchOffsetDegrees,
                -kPitchLimitDegrees, kPitchLimitDegrees);
            // Shortest way round: without the wrap, turning from 350 to 10
            // degrees takes the camera the long way, a full spin past the
            // world, which reads as the view being thrown rather than turned.
            float yawDelta = std::fmod((desiredYaw - m_yawDegrees) + 540.0f, 360.0f) - 180.0f;
            const float blend = 1.0f - std::exp(-deltaSeconds / kAimTauSeconds);
            m_yawDegrees += yawDelta * blend;
            m_pitchDegrees += (desiredPitch - m_pitchDegrees) * blend;
        }
    }

    // Shallow focus on the speaker, arriving with the dolly.
    //
    // What an 80 mm portrait lens actually does is throw everything off the
    // subject plane out, and that is the half a narrower FOV cannot fake: at
    // 55 degrees the background is merely smaller, not separated. Focus rides
    // the measured distance to Victor's face -- the same point the aim uses --
    // so it stays locked on him rather than on a fixed distance he happens to
    // stand at.
    //
    // The focus RANGE is not the physical depth of field. A real 80 mm at f/2.8
    // on a subject 4.4 m away holds about 24 cm sharp, which here is ~17 units
    // and would blur Victor's own body along with the town. This is the
    // distance over which blur ramps to full BEYOND him, so it is set to keep
    // the robot sharp and take everything past him: ~3 m.
    {
        constexpr float kDofTauSeconds = 0.28f;
        constexpr float kFocusRangeUnits = 220.0f;
        constexpr float kMaxBlurRadiusPixels = 12.0f;
        constexpr float kVictorFaceHeightUnits = 150.0f;
        // Well below the 1.25 diorama default, which stretches the near ramp to
        // ~400 units. Victor is a solid object roughly 100 units deep standing
        // ON the focal plane, so a near ramp as short as the far one blurs his
        // own front along with the ground -- measured: his edge detail dropped
        // 25% at 1.25 and holds at this. Long enough to still take the fence
        // and the dirt the camera is standing over.
        constexpr float kNearBlurScale = 0.55f;

        // ODAI_FNV_DIALOGUE_NODOF=1 keeps the conversation framing -- the aim,
        // the dolly, the modal lock -- and only drops the lens blur. It is the
        // control for measuring the DoF: with the camera pointed anywhere else
        // the same screen crop is not the same content, so a no-conversation
        // capture cannot be the baseline.
        static const bool s_noDialogueDof = std::getenv("ODAI_FNV_DIALOGUE_NODOF") != nullptr;
        const bool wantDof = inConversation && m_victor.placed && !s_noDialogueDof;
        const float easeBlend = 1.0f - std::exp(-deltaSeconds / kDofTauSeconds);
        m_dialogueDofBlend += ((wantDof ? 1.0f : 0.0f) - m_dialogueDofBlend) * easeBlend;

        const float dx = m_victor.position[0] - m_cameraX;
        const float dy = (m_victor.position[1] + kVictorFaceHeightUnits) - m_cameraY;
        const float dz = m_victor.position[2] - m_cameraZ;
        const float focusDistance =
            std::max(std::sqrt((dx * dx) + (dy * dy) + (dz * dz)), 1.0f);

        // The renderer's radius is in PIXELS, which is the honest contract for a
        // post-process kernel but means a fixed number is a different-sized
        // blur on every display -- the same shot reads as a strong lens at
        // 1080p and a mild one at 4K. Scale it so the effect is a constant
        // fraction of the image instead, which is what "an 80 mm lens" means
        // to anyone looking at it.
        int dofWidth = 0;
        int dofHeight = 0;
        framebufferSize(dofWidth, dofHeight);
        const float resolutionScale =
            dofHeight > 0 ? (static_cast<float>(dofHeight) / 1080.0f) : 1.0f;

        if (m_dialogueDofBlend > 0.002f) {
            m_renderer.setDepthOfField(
                true, focusDistance, kFocusRangeUnits,
                kMaxBlurRadiusPixels * resolutionScale * m_dialogueDofBlend, kNearBlurScale);
            m_dialogueDofActive = true;
        } else if (m_dialogueDofActive) {
            // Hand it back once, flipping only the enable so anything dialled
            // into the debug sliders survives.
            m_renderer.setDepthOfField(false, focusDistance, kFocusRangeUnits, 0.0f, kNearBlurScale);
            m_dialogueDofActive = false;
        }
    }

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
    // Rooted for the conversation. The keys are not merely ignored further
    // down -- they are never read -- so a held W does not accumulate anywhere
    // and release the moment the card closes. Everything below this point
    // still runs: gravity, the terrain pin and the collision push-out all keep
    // working, so a conversation opened while stepping off a kerb still settles
    // the player onto the ground rather than freezing them mid-air.
    if (!inConversation) {
        if (keyDown(m_window, GLFW_KEY_W)) { moveX += forwardX; moveZ += forwardZ; }
        if (keyDown(m_window, GLFW_KEY_S)) { moveX -= forwardX; moveZ -= forwardZ; }
        if (keyDown(m_window, GLFW_KEY_D)) { moveX += rightX;   moveZ += rightZ; }
        if (keyDown(m_window, GLFW_KEY_A)) { moveX -= rightX;   moveZ -= rightZ; }
        if (keyDown(m_window, GLFW_KEY_SPACE)) { moveY += 1.0f; }
        if (keyDown(m_window, GLFW_KEY_LEFT_CONTROL)) { moveY -= 1.0f; }
    }

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

    // ODAI_FNV_MODS is ':'-separated, appended after any --mod so the flag
    // keeps the lower priority position it was given on the command line and
    // the env can layer on top.
    if (const char* modsEnv = std::getenv("ODAI_FNV_MODS")) {
        const std::string mods = modsEnv;
        std::size_t start = 0;
        while (start <= mods.size()) {
            const std::size_t end = mods.find(':', start);
            const std::string entry =
                mods.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!entry.empty()) {
                m_modDirectories.push_back(entry);
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
    }
    for (const std::string& modDirectory : m_modDirectories) {
        VOX_LOGI("newvegas") << "mod directory: " << modDirectory;
        m_streamer->addModDirectory(std::filesystem::path(modDirectory));
    }

    // ODAI_FNV_TEX_SIZE is the mip-drop ceiling. The 512 default is what makes
    // the base game fit; a high-resolution texture pack is invisible without
    // raising it, because its art gets dropped straight back down. Memory goes
    // as the square, so this is the knob to reach for first when the GPU starts
    // complaining.
    if (const char* texSizeEnv = std::getenv("ODAI_FNV_TEX_SIZE")) {
        const int requested = std::atoi(texSizeEnv);
        if (requested >= 0) {
            m_streamer->setMaxTextureSize(static_cast<std::uint32_t>(requested));
            VOX_LOGI("newvegas") << "texture ceiling: "
                                 << (requested == 0 ? "unclamped" : std::to_string(requested) + " px");
        }
    }

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
        // Stand Victor beside wherever the player starts, rather than at his
        // ACRE reference ~7400 units away. Talking to him is the thing being
        // built; a hike across Goodsprings before every test is friction with
        // no upside. ODAI_FNV_VICTOR_HOME=1 puts him back at his real spot.
        if (std::getenv("ODAI_FNV_VICTOR_HOME") == nullptr) {
            const float offsetX = 220.0f;
            const float offsetZ = 220.0f;
            m_victorSpawnPosition[0] = m_cameraX + offsetX;
            m_victorSpawnPosition[2] = m_cameraZ + offsetZ;
            float ground = 0.0f;
            m_victorSpawnPosition[1] =
                groundHeightAt(m_victorSpawnPosition[0], m_victorSpawnPosition[2], ground)
                    ? ground
                    : (m_cameraY - kEyeHeightUnits);
        }
        {
            const std::filesystem::path dataPath(m_streamDirectory);
            if (loadVictor(dataPath, dataPath / m_streamPlugin, m_streamer->assets(), m_victor,
                           m_victorSpawnPosition[1] != 0.0f ? m_victorSpawnPosition : nullptr)) {
                m_victorUploadPending = true;
                // Turn him to face wherever the player starts. His authored
                // ACRE rotation is not used: standing him beside the spawn
                // already overrode his authored POSITION, and a robot facing
                // the direction he faces in a different part of town reads as
                // broken rather than as fidelity.
                m_victor.yawRadians = std::atan2(
                    m_cameraZ - m_victor.position[2], m_cameraX - m_victor.position[0]);
            }
            VOX_LOGI("newvegas") << "Victor: " << m_victor.status;
            // The rest of the town, discovered from the plugin around wherever
            // the player actually is rather than from a hardcoded list.
            {
                const float engineCentre[3] = {m_cameraX, m_cameraY, m_cameraZ};
                float bethesdaCentre[3] = {};
                importer::fnv::CellStreamer::engineToFallout(engineCentre, bethesdaCentre);
                const float centreXY[2] = {bethesdaCentre[0], bethesdaCentre[1]};
                ActorPopulationStats actorStats;
                loadGoodspringsActors(
                    dataPath / m_streamPlugin, m_streamer->assets(), centreXY, kActorLoadRadius,
                    kFirstCrowdSkinnedInstance,
                    render::kMaxSkinnedInstances - kFirstCrowdSkinnedInstance,
                    {m_victor.baseFormId}, m_actors, actorStats);
                m_actorsUploadPending = !m_actors.empty();
                // ODAI_FNV_ACTORS_PARADE lines every built actor up in front of
                // the spawn, for the same reason Victor stands beside it: the
                // town's people are spread over 12000 units and a screenshot
                // run cannot walk to them, so without this a change to how a
                // body is assembled cannot be looked at at all.
                if (const char* parade = std::getenv("ODAI_FNV_ACTORS_PARADE")) {
                    // Laid out along the camera's own right vector at a fixed
                    // distance ahead of it, so ODAI_FNV_YAW alone chooses what
                    // the parade stands in front of -- a fixed compass
                    // direction puts them behind Doc Mitchell's house.
                    const float spacing = 130.0f;
                    const float distance = std::max(200.0f, static_cast<float>(std::atof(parade)));
                    const float yaw = m_yawDegrees * (kPi / 180.0f);
                    const float forwardX = std::cos(yaw);
                    const float forwardZ = std::sin(yaw);
                    const float centreX = m_cameraX + (forwardX * distance);
                    const float centreZ = m_cameraZ + (forwardZ * distance);
                    for (std::size_t i = 0; i < m_actors.size(); ++i) {
                        SkinnedActor& actor = m_actors[i];
                        const float offset =
                            (static_cast<float>(i) -
                             (static_cast<float>(m_actors.size() - 1) * 0.5f)) * spacing;
                        // Right vector is forward rotated -90 degrees in XZ.
                        actor.position[0] = centreX + (forwardZ * offset);
                        actor.position[2] = centreZ - (forwardX * offset);
                        float ground = 0.0f;
                        actor.position[1] =
                            groundHeightAt(actor.position[0], actor.position[2], ground)
                                ? ground
                                : (m_cameraY - kEyeHeightUnits);
                        // Facing the camera, so a face that failed to build is
                        // visible as a face and not as the back of a head.
                        actor.yawRadians = std::atan2(forwardX, forwardZ);
                    }
                }
                VOX_LOGI("newvegas") << "Goodsprings actors: " << actorStats.detail;
                for (const SkinnedActor& actor : m_actors) {
                    VOX_LOGI("newvegas")
                        << "  actor " << actor.name << " slot=" << actor.instanceSlot << " at ("
                        << actor.position[0] << ", " << actor.position[1] << ", "
                        << actor.position[2] << ") verts=" << actor.character.vertices.size()
                        << " parts=" << actor.character.parts.size()
                        << " unresolvedBones=" << actor.character.unresolvedBoneCount
                        << " bindConflicts=" << actor.character.conflictingInverseBindCount
                        << " clip=" << (actor.idleClip.tracks.empty() ? "none" : "idle");
                }
            }
            if (m_victor.placed) {
                VOX_LOGI("newvegas") << "Victor animation: " << m_victor.animationStatus;
                VOX_LOGI("newvegas") << "Victor load: " << m_victor.timing;
                VOX_LOGI("newvegas") << "Victor voice: " << m_victor.voice.status;
            }
        }

        // AFTER the doorstep decision, which would otherwise overwrite it ten
        // lines later -- the first attempt at this set walk mode above and was
        // silently undone here, so every headless fly capture stayed ON FOOT.
        // F toggles this interactively; these are the same switch for a
        // headless run, which cannot press a key.
        if (std::getenv("ODAI_FNV_SPAWN_POS") != nullptr) {
            // Placing the camera explicitly implies flying it: walk mode
            // re-snaps Y to the ground every frame, so an authored height
            // survived exactly one frame.
            m_walkMode = false;
        }
        if (const char* flyEnv = std::getenv("ODAI_FNV_FLY")) {
            if (flyEnv[0] != '\0' && flyEnv[0] != '0') {
                m_walkMode = false;
            }
        }
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
                             << " nodeParseFails=" << stats.nodeParseFailures
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
    //
    // A conversation counts for the same reason, and now more literally: the
    // dialogue card is centred large type, and "Goodsprings / Location
    // discovered" landed straight across Victor's first two replies.
    if (!m_menuOpen && !m_victor.talking) {
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

    // Uploaded from the frame loop, not onInit: an init-time GPU upload lands
    // as zeros (see chunk_upload.cc's add-time note) and draws nothing at all.
    //
    // Textures first, because their bindless slots have to be written into the
    // vertices before the template carrying those vertices goes to the GPU --
    // a skinned template is uploaded verbatim, with none of the index remapping
    // a scene chunk gets.
    if (m_victorUploadPending) {
        m_victorUploadPending = false;
        const std::vector<std::uint32_t> textureSlots =
            m_renderer.uploadSkinnedActorTextures(kVictorSkinnedInstance, m_victor.textures);
        remapVictorTextureSlots(m_victor, textureSlots);

        render::ImportedSkinnedMeshTemplate meshTemplate{};
        meshTemplate.vertices = m_victor.character.vertices;
        meshTemplate.indices = m_victor.character.indices;
        meshTemplate.draws = m_victor.draws;
        meshTemplate.boneCount =
            static_cast<std::uint32_t>(m_victor.character.skeleton.bones.size());
        m_victor.uploaded =
            m_renderer.uploadSkinnedMeshTemplate(kVictorSkinnedInstance, meshTemplate);
        std::size_t texturedSlots = 0;
        for (const std::uint32_t slot : textureSlots) {
            texturedSlots += (slot != 0xffffffffu) ? 1u : 0u;
        }
        VOX_LOGI("newvegas") << "Victor upload: "
                             << (m_victor.uploaded ? "ok" : "FAILED") << ", " << texturedSlots
                             << "/" << textureSlots.size() << " textures bound";
    }

    if (m_actorsUploadPending) {
        m_actorsUploadPending = false;
        std::size_t uploaded = 0;
        for (SkinnedActor& actor : m_actors) {
            const std::vector<std::uint32_t> slots =
                m_renderer.uploadSkinnedActorTextures(actor.instanceSlot, actor.textures);
            for (odai::render::ImportedSkinnedMeshVertex& vertex : actor.character.vertices) {
                vertex.textureIndex = (vertex.textureIndex < slots.size())
                    ? slots[vertex.textureIndex]
                    : 0xffffffffu;
            }
            render::ImportedSkinnedMeshTemplate meshTemplate{};
            meshTemplate.vertices = actor.character.vertices;
            meshTemplate.indices = actor.character.indices;
            meshTemplate.draws = actor.draws;
            meshTemplate.boneCount =
                static_cast<std::uint32_t>(actor.character.skeleton.bones.size());
            actor.uploaded = m_renderer.uploadSkinnedMeshTemplate(actor.instanceSlot, meshTemplate);
            uploaded += actor.uploaded ? 1u : 0u;
        }
        VOX_LOGI("newvegas") << "Goodsprings actors uploaded: " << uploaded << "/"
                             << m_actors.size();
    }
    if (!m_actors.empty()) {
        updateActorPoses(m_actors, deltaSeconds);
        for (const SkinnedActor& actor : m_actors) {
            if (!actor.uploaded) {
                continue;
            }
            render::ImportedSkinnedActorFrameData pose{};
            pose.boneMatrices = actor.poseScratch;
            m_renderer.setSkinnedActorPose(actor.instanceSlot, pose);
        }
    }

    // Pose him every frame, whether or not he is being talked to -- the idle
    // clip is what makes him read as a machine that is running rather than a
    // statue of one.
    if (m_victor.uploaded) {
        updateVictorPose(m_victor, deltaSeconds);
        render::ImportedSkinnedActorFrameData pose{};
        pose.boneMatrices = m_victor.poseScratch;
        m_renderer.setSkinnedActorPose(kVictorSkinnedInstance, pose);
    }

    // ODAI_FNV_VICTOR_TALK=1 opens the conversation on the first tick, so the
    // dialogue UI can be checked from a --screenshot run, which cannot press E.
    if (!m_victor.talking && m_victor.placed && !m_victor.tree.nodes.empty() &&
        std::getenv("ODAI_FNV_VICTOR_TALK") != nullptr) {
        static bool autoTalked = false;
        if (!autoTalked) {
            autoTalked = true;
            m_victor.runtime.begin(m_victor.tree, m_victor.context);
            m_victor.talking = true;
            VOX_LOGI("newvegas") << "auto-talk: node="
                                 << (m_victor.runtime.currentNode() != nullptr
                                         ? m_victor.runtime.currentNode()->id
                                         : std::string("<null>"))
                                 << " finished=" << m_victor.runtime.isFinished();
        }
    }

    // Talking to Victor. Held keys are edge-detected by keyDown(), so a choice
    // is taken once per press rather than once per frame.
    // Edge-latched per slot. keyDown() is level-triggered, so an unlatched read
    // takes one choice PER FRAME: a normal ~100 ms press on "1" consumed six
    // choices, ran off the end of the branch and closed the conversation before
    // it could be read, which looked exactly like Victor refusing to talk.
    for (int slot = 0; slot < 9; ++slot) {
        const bool pressed = keyDown(m_window, GLFW_KEY_1 + slot);
        const bool edge = pressed && !m_choiceKeyLatch[slot];
        m_choiceKeyLatch[slot] = pressed;
        if (!edge || !m_victor.talking) {
            continue;
        }
        const auto choices = m_victor.runtime.availableChoices();
        if (static_cast<std::size_t>(slot) < choices.size()) {
            m_victor.runtime.choose(*choices[static_cast<std::size_t>(slot)]);
        }
    }
    // Highlight-and-confirm, alongside the number keys rather than instead of
    // them. The numbers are the fast path for someone at a keyboard; up/down
    // and Accept are the only ones that work from a couch, and they come from
    // UiNavInput so a gamepad drives them identically (pollNavInput already
    // folds the d-pad, the left stick and the arrow keys into the same
    // actions, with auto-repeat, so a held direction scrolls instead of
    // jumping one row per frame).
    if (m_victor.talking) {
        const auto choices = m_victor.runtime.availableChoices();
        const auto choiceCount = static_cast<int>(choices.size());
        // A new node means a new set of replies; leaving the old index in
        // place would highlight an unrelated line, or one that no longer
        // exists.
        const dialogue::DialogueNode* currentNode = m_victor.runtime.currentNode();
        const std::string currentNodeId = currentNode != nullptr ? currentNode->id : std::string();
        if (currentNodeId != m_dialogueChoiceNodeId) {
            m_dialogueChoiceNodeId = currentNodeId;
            m_dialogueChoice = 0;
            // ODAI_FNV_DIALOGUE_SELECT=<n> starts on the nth reply (0-based).
            // The highlight is the whole point of this panel and a --screenshot
            // run cannot press a key, so without this the only row that can
            // ever be photographed is the first one -- and "the highlight is
            // drawn" and "the highlight tracks the selection" are different
            // claims.
            if (const char* fromEnv = std::getenv("ODAI_FNV_DIALOGUE_SELECT")) {
                m_dialogueChoice = std::atoi(fromEnv);
            }
        }
        if (choiceCount > 0) {
            if (m_nav.pressed(ui::UiNavAction::Up)) {
                // Wrapping, not clamping: a four-item list on a controller is
                // faster to reach the end of by going up once.
                m_dialogueChoice = (m_dialogueChoice + choiceCount - 1) % choiceCount;
            }
            if (m_nav.pressed(ui::UiNavAction::Down)) {
                m_dialogueChoice = (m_dialogueChoice + 1) % choiceCount;
            }
            m_dialogueChoice = std::clamp(m_dialogueChoice, 0, choiceCount - 1);
            if (m_nav.pressed(ui::UiNavAction::Accept)) {
                m_victor.runtime.choose(*choices[static_cast<std::size_t>(m_dialogueChoice)]);
                m_dialogueChoice = 0;
            }
        } else {
            m_dialogueChoice = 0;
        }
    }
    if (m_victor.talking) {
        if (m_victor.runtime.isFinished() || m_victor.runtime.currentNode() == nullptr) {
            m_victor.talking = false;
        }
    }
    // One call site rather than one per way a conversation can advance (a
    // choice, opening it, the auto-talk hook). It is a no-op once the current
    // node has been spoken, so polling it costs a map lookup and cannot start a
    // line twice.
    if (m_victor.talking && !m_streamCacheDirectory.empty()) {
        speakVictorLine(
            m_victor, std::filesystem::path(m_streamCacheDirectory) / "voice", m_audio);
    }

    if (keyDown(m_window, GLFW_KEY_ESCAPE)) {
        if (m_victor.talking) {
            m_victor.talking = false;  // leave the conversation, not the game
            return;
        }
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
    const float cameraPosition[3] = {m_cameraX, m_cameraY, m_cameraZ};
    m_victorPromptVisible =
        !m_victor.talking && victorIsInReach(m_victor, cameraPosition, m_yawDegrees * (kPi / 180.0f));
    // Latch BEFORE the branch below. It used to be updated after an early
    // return that the Victor path took, so the latch stayed false while E was
    // held: the next frame saw a fresh "press" and walked the player through
    // Doc Mitchell's door -- which is a step from where Victor stands -- so the
    // conversation opened and an interior load closed it in the same keypress.
    const bool doorPressed = keyDown(m_window, GLFW_KEY_E);
    const bool doorEdge = doorPressed && !m_doorKeyLatch;
    m_doorKeyLatch = doorPressed;
    if (doorEdge && m_victorPromptVisible) {
        m_victor.runtime.begin(m_victor.tree, m_victor.context);
        m_victor.talking = true;
        m_victor.spokenNodeId.clear();
        // The line itself is started by the single speakVictorLine poll above,
        // on the next tick.
        return;  // E opened a conversation; do not also walk through a door
    }
    if (doorEdge) {
        const int doorIndex = findUsableDoor();
        if (doorIndex >= 0) {
            useDoor(m_doors[static_cast<std::size_t>(doorIndex)]);
        }
    }

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

// Greedy word wrap against a baked font's own metrics.
//
// The HUD's addText draws one unwrapped run, which is fine for a status strip
// and wrong for a paragraph: Victor's longer lines ran off the side of the
// screen. Written here rather than reached for through rich_text because that
// path parses <b>/<color=...> markup, and this text comes out of a 1998 game's
// dialogue records -- a stray '<' in a line is content, not a tag.
//
// A single word longer than maxWidth is emitted on its own over-long line
// rather than split mid-word: it cannot be made to fit, and breaking it is
// less readable than letting one line run.
std::vector<std::string> wrapTextToWidth(
    const ui::Font& font, const std::string& text, float maxWidth
) {
    std::vector<std::string> lines;
    if (text.empty()) {
        return lines;
    }
    if (maxWidth <= 0.0f) {
        lines.push_back(text);
        return lines;
    }
    std::string line;
    std::size_t wordStart = 0;
    while (wordStart <= text.size()) {
        std::size_t wordEnd = text.find(' ', wordStart);
        if (wordEnd == std::string::npos) {
            wordEnd = text.size();
        }
        const std::string word = text.substr(wordStart, wordEnd - wordStart);
        if (!word.empty()) {
            const std::string candidate = line.empty() ? word : (line + " " + word);
            if (!line.empty() && font.measureText(candidate) > maxWidth) {
                lines.push_back(line);
                line = word;
            } else {
                line = candidate;
            }
        }
        if (wordEnd == text.size()) {
            break;
        }
        wordStart = wordEnd + 1;
    }
    if (!line.empty()) {
        lines.push_back(line);
    }
    return lines;
}
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

    // All three sources for a direction -- stick, d-pad, arrow key -- combined
    // into ONE level, then set once.
    //
    // This used to call m_navStick.apply() and then fold the digital sources in
    // with `if (key) setAction(action, true)`. Two setAction calls per action
    // per frame, and the second one saw the first's `false` as the previous
    // frame's state, so every frame an arrow key was held produced a fresh
    // press edge. The dialogue list scrolled at frame rate -- about 100 items a
    // second on this machine -- which is what "the selection moves too fast"
    // actually was. The auto-repeat timing was never involved.
    int stickDirectionX = 0;
    int stickDirectionY = 0;
    m_navStick.resolveDirection(stickX, stickY, stickDirectionX, stickDirectionY);
    const bool padUp = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP] == GLFW_PRESS;
    const bool padDown = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN] == GLFW_PRESS;
    const bool padLeft = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT] == GLFW_PRESS;
    const bool padRight = hasPad && pad.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] == GLFW_PRESS;
    m_nav.setAction(ui::UiNavAction::Up, keyUp || padUp || stickDirectionY < 0);
    m_nav.setAction(ui::UiNavAction::Down, keyDownArrow || padDown || stickDirectionY > 0);
    m_nav.setAction(ui::UiNavAction::Left, keyLeft || padLeft || stickDirectionX < 0);
    m_nav.setAction(ui::UiNavAction::Right, keyRight || padRight || stickDirectionX > 0);
    m_nav.setAction(ui::UiNavAction::Accept, accept);
    m_nav.setAction(ui::UiNavAction::Cancel, cancel);
    m_nav.setAction(ui::UiNavAction::Menu, menu);

    m_navRepeat.update(m_nav, deltaSeconds);

    // Any directional or accept input means a controller is driving. The mouse
    // takes it back in updateCamera, which is where mouse motion is already
    // being read.
    m_navDriving = m_navDriving || m_nav.active;
    m_nav.active = false;

    // Not while a conversation is up: Escape is both Menu and Cancel, and this
    // runs at the top of the tick, before the dialogue's own Escape handling.
    // Backing out of a conversation therefore closed it AND opened the menu in
    // one press -- two modal states toggled by one key, which reads as the menu
    // appearing for no reason.
    if (m_nav.pressed(ui::UiNavAction::Menu) && !m_victor.talking) {
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
    // ODAI_FNV_LOG_REGION=1 traces the cell and region under the camera every
    // poll. "No banner fired" and "the walk never left the region" look the
    // same from the outside, and a traversal test has to tell them apart.
    static const bool s_logRegion = std::getenv("ODAI_FNV_LOG_REGION") != nullptr;
    if (s_logRegion) {
        float fallout[3] = {};
        importer::fnv::CellStreamer::engineToFallout(position, fallout);
        std::string names;
        for (const std::string& name : m_streamer->regionNamesAtEngineSpace(position)) {
            names += names.empty() ? name : (", " + name);
        }
        VOX_LOGI("newvegas") << "cell ("
                             << static_cast<int>(std::floor(fallout[0] / 4096.0f)) << ","
                             << static_cast<int>(std::floor(fallout[1] / 4096.0f))
                             << ") regions: " << (names.empty() ? "<none>" : names);
    }
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

    // Compass strip, top-centre, with a marker for Victor.
    //
    // Bearing convention: the camera's forward in XZ is (cos(yaw), sin(yaw)),
    // and Fallout's north is +Y in its own space, which this engine stores as
    // -Z. So north sits at yaw 270 and compass degrees are (yaw + 90) mod 360 --
    // worth writing down because getting it wrong gives a compass that is
    // plausibly wrong by 90 degrees, which is worse than none.
    {
        const auto compassDegrees = [](float yawDegrees) {
            float d = std::fmod(yawDegrees + 90.0f, 360.0f);
            return d < 0.0f ? d + 360.0f : d;
        };
        const float heading = compassDegrees(m_yawDegrees);
        static const char* kPoints[8] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
        const char* cardinal = kPoints[static_cast<int>((heading + 22.5f) / 45.0f) % 8];

        char headingText[64] = {};
        std::snprintf(headingText, sizeof(headingText), "%s  %3d°", cardinal,
                      static_cast<int>(heading + 0.5f) % 360);
        const float headingWidth = m_uiFont.measureText(headingText);
        m_uiDrawList.addText(
            m_uiFont, headingText,
            ui::UiVec2{(static_cast<float>(screenWidth) - headingWidth) * 0.5f, 12.0f * scale},
            kPipGreen);

        // Where Victor is from here, so "I cannot find him" becomes a bearing
        // and a distance rather than a search.
        if (m_victor.placed && !m_victor.talking) {
            const float dx = m_victor.position[0] - m_cameraX;
            const float dz = m_victor.position[2] - m_cameraZ;
            const float distance = std::sqrt((dx * dx) + (dz * dz));
            const float toVictor =
                compassDegrees(std::atan2(dz, dx) * (180.0f / kPi));
            // Signed turn, so the hint says which way to turn rather than
            // leaving the player to subtract two bearings in their head.
            float turn = std::fmod((toVictor - heading) + 540.0f, 360.0f) - 180.0f;
            char victorText[96] = {};
            std::snprintf(victorText, sizeof(victorText), "Victor  %4d u  %s %d°",
                          static_cast<int>(distance),
                          turn >= 0.0f ? "turn right" : "turn left",
                          static_cast<int>(std::fabs(turn) + 0.5f));
            const float victorWidth = m_uiFont.measureText(victorText);
            m_uiDrawList.addText(
                m_uiFont, victorText,
                ui::UiVec2{(static_cast<float>(screenWidth) - victorWidth) * 0.5f, 34.0f * scale},
                std::fabs(turn) < 12.0f ? kPipGreen : kPipGreenDim);
        }
    }

    // Victor. The conversation is drawn straight onto the HUD draw list rather
    // than through DialoguePanel: the panel wants a widget tree, and this app's
    // HUD is immediate-mode text, so one path is simpler than bridging two.
    if (m_victor.talking) {
        if (const dialogue::DialogueNode* node = m_victor.runtime.currentNode()) {
            drawDialoguePanel(*node, screenWidth, screenHeight, scale);
        }
    } else if (m_victorPromptVisible) {
        m_uiDrawList.addText(m_uiFont, "E  talk to Victor",
                             ui::UiVec2{64.0f * scale, static_cast<float>(screenHeight) - (132.0f * scale)},
                             kPipGreen);
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

void NewVegasApp::drawDialoguePanel(
    const dialogue::DialogueNode& node, int screenWidth, int screenHeight, float scale
) {
    // Fall back to the body face when a dialogue bake failed; the layout below
    // measures whatever font it is handed, so it stays correct either way.
    const ui::Font& lineFont = m_dialogueFont.valid() ? m_dialogueFont : m_uiFont;
    const ui::Font& choiceFont = m_dialogueChoiceFont.valid() ? m_dialogueChoiceFont : m_uiFont;

    const auto width = static_cast<float>(screenWidth);
    const auto height = static_cast<float>(screenHeight);

    // Width is capped in *scaled* units as well as as a fraction of the screen.
    // A line of text that spans an entire 4K width is unreadable no matter how
    // big the glyphs are -- the eye loses the start of the next line -- so the
    // card stops growing once it is wide enough for a comfortable measure.
    const float panelWidth = std::min(width * 0.74f, 1500.0f * scale);
    const float padding = 40.0f * scale;
    const float innerWidth = panelWidth - (padding * 2.0f);

    const std::vector<std::string> spokenLines =
        wrapTextToWidth(lineFont, node.text, innerWidth);
    const float spokenLineHeight = lineFont.lineHeightPx() * 1.18f;

    // Replies are indented past their number, and the wrap has to account for
    // that or a long reply overruns the card it is measured against.
    const float choiceIndent = 56.0f * scale;
    const float choiceRowPadding = 14.0f * scale;
    const float choiceLineHeight = choiceFont.lineHeightPx() * 1.15f;
    const auto choices = m_victor.runtime.availableChoices();
    const std::size_t choiceCount = std::min<std::size_t>(choices.size(), 9u);
    std::vector<std::vector<std::string>> choiceLines;
    choiceLines.reserve(choiceCount);
    float choicesHeight = 0.0f;
    for (std::size_t i = 0; i < choiceCount; ++i) {
        choiceLines.push_back(
            wrapTextToWidth(choiceFont, choices[i]->text, innerWidth - choiceIndent));
        const float rows = static_cast<float>(std::max<std::size_t>(choiceLines.back().size(), 1u));
        choicesHeight += (rows * choiceLineHeight) + (choiceRowPadding * 2.0f);
    }

    const float speakerHeight = m_uiFontBold.valid() ? m_uiFontBold.lineHeightPx()
                                                     : m_uiFont.lineHeightPx();
    // Weighted toward the replies: the rule belongs to the block above it, and
    // an equal gap on both sides made the first reply's highlight border read
    // as if it were touching the rule.
    const float ruleGapAbove = 22.0f * scale;
    const float ruleGapBelow = 30.0f * scale;
    const float ruleGap = ruleGapAbove + ruleGapBelow;
    const float footerHeight = m_uiFont.lineHeightPx() + (18.0f * scale);
    const float spokenHeight =
        static_cast<float>(std::max<std::size_t>(spokenLines.size(), 1u)) * spokenLineHeight;
    const float panelHeight = (padding * 2.0f) + speakerHeight + (12.0f * scale) + spokenHeight +
                              ruleGap + choicesHeight + footerHeight;

    const float panelX = (width - panelWidth) * 0.5f;
    const float panelY = (height - panelHeight) * 0.5f;
    const ui::UiRect panel{panelX, panelY, panelX + panelWidth, panelY + panelHeight};
    const float corner = 10.0f * scale;
    // Published for updateCamera, which frames Victor's face above this edge.
    m_dialoguePanelTopPx = panelY;

    // The card sits over the world, so it needs to separate from whatever is
    // behind it: a shadow to lift it off the terrain, a near-opaque fill so
    // text never competes with a bright sky, and a phosphor edge to tie it to
    // the rest of the HUD.
    m_uiDrawList.addDropShadow(panel, ui::UiColor{0.0f, 0.0f, 0.0f, 0.55f}, 18.0f * scale, corner);
    m_uiDrawList.addRoundRectFilled(panel, kPipPanelSolid, corner);
    m_uiDrawList.addRoundRect(panel, kPipGreenDim, corner, 2.0f * scale);

    float y = panel.minY + padding;

    // Speaker, centred over the line, in caps -- a name label, not prose.
    std::string speaker = node.speaker.empty() ? std::string("VICTOR") : node.speaker;
    for (char& c : speaker) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    const ui::Font& speakerFont = m_uiFontBold.valid() ? m_uiFontBold : m_uiFont;
    m_uiDrawList.addText(
        speakerFont, speaker,
        ui::UiVec2{panel.minX + ((panelWidth - speakerFont.measureText(speaker)) * 0.5f), y},
        kPipGreenDim);
    y += speakerHeight + (12.0f * scale);

    // What Victor says: centred, because it is one short block and centring it
    // under the name reads as a single unit.
    for (const std::string& text : spokenLines) {
        m_uiDrawList.addText(
            lineFont, text,
            ui::UiVec2{panel.minX + ((panelWidth - lineFont.measureText(text)) * 0.5f), y},
            kPipGreen);
        y += spokenLineHeight;
    }

    y += ruleGapAbove;
    m_uiDrawList.addRectFilled(
        ui::UiRect{panel.minX + padding, y, panel.maxX - padding, y + (1.0f * scale)},
        ui::UiColor{kPipGreenDim.r, kPipGreenDim.g, kPipGreenDim.b, 0.45f});
    y += ruleGapBelow;

    // The replies: LEFT aligned, unlike the block above. They are a list to be
    // scanned down, and centring a list makes every row start in a different
    // place, which is exactly what the eye uses to track position.
    for (std::size_t i = 0; i < choiceCount; ++i) {
        const float rows = static_cast<float>(std::max<std::size_t>(choiceLines[i].size(), 1u));
        const float rowHeight = (rows * choiceLineHeight) + (choiceRowPadding * 2.0f);
        const bool selected = static_cast<int>(i) == m_dialogueChoice;
        const ui::UiRect row{panel.minX + (padding * 0.5f), y,
                             panel.maxX - (padding * 0.5f), y + rowHeight};

        // Selection is stated three ways at once, because any one of them alone
        // fails somewhere: a fill is invisible to the colour-blind against a
        // green-on-green palette, a colour change alone is easy to miss across
        // a room, and a caret alone is small. Together they are unmistakable at
        // TV distance.
        if (selected) {
            m_uiDrawList.addRoundRectFilled(
                row, ui::UiColor{kPipGreen.r, kPipGreen.g, kPipGreen.b, 0.20f}, corner * 0.6f);
            m_uiDrawList.addRoundRect(row, kPipGreen, corner * 0.6f, 2.0f * scale);
            m_uiDrawList.addText(
                choiceFont, ">",
                ui::UiVec2{row.minX + (16.0f * scale), y + choiceRowPadding}, kPipGreen);
        }

        const std::string number = std::to_string(i + 1) + ".";
        m_uiDrawList.addText(
            choiceFont, number,
            ui::UiVec2{row.minX + (40.0f * scale), y + choiceRowPadding},
            selected ? kPipGreen : kPipGreenDim);

        float choiceY = y + choiceRowPadding;
        for (const std::string& text : choiceLines[i]) {
            m_uiDrawList.addText(
                choiceFont, text,
                ui::UiVec2{row.minX + choiceIndent + (30.0f * scale), choiceY},
                selected ? kPipGreen : kPipGreenDim);
            choiceY += choiceLineHeight;
        }
        y += rowHeight;
    }

    const std::string footer = choiceCount == 0
        ? std::string("Esc  end conversation")
        : std::string("Up/Down  select     Enter  choose     Esc  leave");
    m_uiDrawList.addText(
        m_uiFont, footer,
        ui::UiVec2{panel.minX + ((panelWidth - m_uiFont.measureText(footer)) * 0.5f),
                   panel.maxY - padding - m_uiFont.lineHeightPx() + (10.0f * scale)},
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
    if (!m_menuOpen && !m_victor.talking) {
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
    camera.fovDegrees = m_cameraFovDegrees;
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
