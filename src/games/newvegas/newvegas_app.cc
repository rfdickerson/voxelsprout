#include "games/newvegas/newvegas_app.h"

#include "import/fnv/land_lod.h"

#include "render/upscale/upscale_policy.h"

#include "import/fnv/dialogue_records.h"

#include "import/dds.h"
#include "games/newvegas/newvegas_ogg.h"
#include "import/fnv/bsa_archive.h"

#include <fstream>
#include <sstream>
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
#include <cstring>

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

std::string toLowerAscii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

bool keyDown(GLFWwindow* window, int key) {
    return glfwGetKey(window, key) == GLFW_PRESS;
}

}  // namespace

float NewVegasApp::verticalFovDegreesFor(float horizontalFovDegrees, float aspectRatio) {
    // Guard the degenerate frame: a zero or negative aspect would otherwise
    // divide the tangent into infinity and hand the projection a NaN, which
    // renders as an empty screen rather than as an error.
    const float safeAspect = (aspectRatio > 0.0001f) ? aspectRatio : (16.0f / 9.0f);
    const float clampedHorizontal = std::clamp(horizontalFovDegrees, 1.0f, 179.0f);
    const float halfHorizontalRadians = (clampedHorizontal * 0.5f) * (kPi / 180.0f);
    const float halfVerticalRadians = std::atan(std::tan(halfHorizontalRadians) / safeAspect);
    return halfVerticalRadians * 2.0f * (180.0f / kPi);
}

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

void NewVegasApp::beginConversation(int actorIndex) {
    if (actorIndex < 0 || actorIndex >= static_cast<int>(m_actors.size())) {
        return;
    }
    SkinnedActor& actor = m_actors[static_cast<std::size_t>(actorIndex)];
    if (!actor.canTalk()) {
        return;
    }
    m_talkingActor = actorIndex;
    actor.talking = true;
    actor.runtime.begin(actor.tree, actor.context);
    // Reset the highlight rather than inheriting the last conversation's row,
    // which would open on a reply that belongs to somebody else's branch.
    m_dialogueChoice = 0;
    m_dialogueChoiceNodeId.clear();
    VOX_LOGI("newvegas") << "conversation: " << actor.name << " node="
                         << (actor.runtime.currentNode() != nullptr
                                 ? actor.runtime.currentNode()->id
                                 : std::string("<null>"))
                         << " finished=" << actor.runtime.isFinished();
}

void NewVegasApp::endConversation() {
    if (SkinnedActor* actor = talkingActor()) {
        actor->talking = false;
        // Forget which line was spoken, so returning to this actor replays the
        // greeting instead of opening on a silent node.
        actor->spokenNodeId.clear();
    }
    m_talkingActor = -1;
    m_dialogueChoice = 0;
    m_dialogueChoiceNodeId.clear();
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
    //
    // ODAI_FNV_RT=1 keeps the RT runtime alive anyway. GI is not the only
    // possible TLAS consumer any more: ray-traced sun shadows want one too, and
    // the shading side for those is already written and compiled
    // (sampleRayTracedDirectionalShadow / imported_static_rt.frag.slang.spv).
    // This line is the ONLY thing standing between the streaming path and a
    // TLAS -- the BLAS record block in uploadImportedSceneInternal is not gated
    // on the chunk path at all, it is gated on rayTracingRuntimeReady().
    //
    // Off by default until the acceleration-structure build cost on a STREAMING
    // world is measured; the whole reason this line exists is that those builds
    // were expensive per upload, and a cell stream calls upload constantly.
    const bool rayTracingRequested = std::getenv("ODAI_FNV_RT") != nullptr;
    m_renderer.setRayTracingEnabled(rayTracingRequested);
    if (rayTracingRequested) {
        VOX_LOGI("newvegas") << "ODAI_FNV_RT: ray tracing runtime left enabled "
                                "(acceleration structures will build per cell upload)";
    }
    // Volumetric sun shafts. sun_shafts.comp.slang is a real single-scattering
    // raymarch -- height-falloff density, Henyey-Greenstein phase, shadow-map
    // visibility sampled per step -- so this is the atmosphere pass, not a
    // radial blur.
    //
    // OFF by default, because it is currently paying for nothing: skyConfig4's
    // density/falloff/scatter are near zero for this game, so the effect is
    // invisible while the pass costs 4.7 ms of a 37.9 ms frame at 2560x1440 on
    // the LNL iGPU -- measured by toggling it, not by the GPU timer, which
    // attributes only ~3 ms to the dispatch itself. ODAI_FNV_SHAFTS=1 turns it
    // back on, which is what to do FIRST when tuning those density values: the
    // pass is worth its cost only once they are non-trivial, and this default
    // should flip back the moment they are.
    const bool sunShaftsRequested = [] {
        const char* env = std::getenv("ODAI_FNV_SHAFTS");
        return env != nullptr && env[0] != '0';
    }();
    m_renderer.setSunShaftsEnabled(sunShaftsRequested);
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
    // XeGTAO rather than GTAO, and it is cheaper AS WELL AS cleaner -- which is
    // the opposite of what "it runs three dispatches instead of one" suggests.
    //
    // GTAO marches a fixed sample count per pixel and its only smoothing is the
    // bilateral UPSAMPLE, which exists to reconstruct resolution rather than to
    // denoise; its sample pattern therefore survives into the frame as a
    // stipple, worst on terrain and on alpha-tested foliage where neighbouring
    // samples disagree about depth. XeGTAO marches a prefiltered depth pyramid
    // with adaptive sample counts and then runs a real edge-aware denoise.
    //
    // Measured on Seyda Neen at a pinned camera, as high-frequency energy in the
    // AO channel alone (ODAI_FNV_DEBUGVIEW=ao, mean |pixel - 3x3 mean|, sky
    // excluded), against interleaved A/B/A/B GPU timings:
    //
    //   GTAO            noise 3.95   ssao pass 1.33 ms   frame ~12.9 ms
    //   XeGTAO          noise 1.72   ssao pass 0.43 ms   frame ~11.6 ms
    //   HBAO            noise 2.63
    //   SSAO            noise 0.97   (least noise, crudest estimator)
    //   GTAO full-res   noise 2.70   and ~4.4 ms dearer
    //
    // So 56% less AO noise for 0.9 ms LESS on the pass. The adaptive sample
    // count is why: fewer samples where the term is already smooth.
    //
    // ODAI_XEGTAO_BLUR only affects this mode, which is worth knowing before
    // tuning it -- raising it from 4 to 16 under GTAO changes nothing at all,
    // because GTAO never reaches the XeGTAO denoise.
    //
    // Falls back to the GTAO pipeline on its own if the XeGTAO pipelines or
    // buffer sets are unavailable (see useXeGtao in frame_pass_ssao.cc).
    //
    // ODAI_FNV_AO overrides the mode (off/ssao/hbao/gtao/xegtao) for A/B.
    render::AoMode aoMode = render::AoMode::Xegtao;
    if (const char* aoEnv = std::getenv("ODAI_FNV_AO")) {
        const std::string requested = aoEnv;
        if (requested == "off") {
            aoMode = render::AoMode::Off;
        } else if (requested == "ssao") {
            aoMode = render::AoMode::Ssao;
        } else if (requested == "hbao") {
            aoMode = render::AoMode::Hbao;
        } else if (requested == "gtao") {
            aoMode = render::AoMode::Gtao;
        } else if (requested == "xegtao") {
            aoMode = render::AoMode::Xegtao;
        }
    }
    m_renderer.setSsaoEnabled(aoMode != render::AoMode::Off);
    m_renderer.setAmbientOcclusionMode(aoMode);
    // ODAI_FNV_DEBUGVIEW selects a whole-frame debug visualization by name (see
    // DebugView in renderer_types.h). It exists because a --screenshot run
    // cannot operate the ImGui combo -- F4 is the interactive way in, and this
    // is the only way to photograph a debug view from a script, which is what
    // makes an alpha or material-flags capture attributable in a bug report.
    if (const char* debugViewEnv = std::getenv("ODAI_FNV_DEBUGVIEW")) {
        const std::string requested = debugViewEnv;
        render::DebugView view = render::DebugView::Off;
        if (requested == "albedo") { view = render::DebugView::Albedo; }
        else if (requested == "normal") { view = render::DebugView::Normal; }
        else if (requested == "alpha") { view = render::DebugView::Alpha; }
        else if (requested == "flags") { view = render::DebugView::MaterialFlags; }
        else if (requested == "roughness") { view = render::DebugView::Roughness; }
        else if (requested == "metallic") { view = render::DebugView::Metallic; }
        else if (requested == "mip") { view = render::DebugView::MipLevel; }
        else if (requested == "cascade") { view = render::DebugView::CascadeIndex; }
        else if (requested == "texid") { view = render::DebugView::TextureId; }
        else if (requested == "depth") { view = render::DebugView::LinearDepth; }
        else if (requested == "shadow") { view = render::DebugView::Shadow; }
        else if (requested == "directratio") { view = render::DebugView::DirectRatio; }
        else if (requested == "terrainlayers") { view = render::DebugView::TerrainLayers; }
        else if (requested == "ao") { view = render::DebugView::AmbientOcclusion; }
        else if (requested != "off") {
            VOX_LOGW("newvegas")
                << "ODAI_FNV_DEBUGVIEW=" << requested << " is not a view name; ignoring. "
                << "Valid: albedo normal alpha flags roughness metallic mip cascade texid "
                << "depth shadow directratio terrainlayers ao\n";
        }
        m_renderer.setDebugView(view);
    }
    // ODAI_FNV_DRAW=terrain|statics splits the imported draw list the way the
    // F4 panel's checkboxes do, for the same reason the debug views have an env
    // var: a --screenshot run cannot operate ImGui, and "is this artifact
    // terrain or a static" is unanswerable from a lit frame when the two draw
    // on top of each other.
    if (const char* drawEnv = std::getenv("ODAI_FNV_DRAW")) {
        const std::string requested = drawEnv;
        const bool showTerrain = requested != "statics";
        const bool showStatics = requested != "terrain";
        m_renderer.setImportedSceneDebugState(
            showTerrain, showStatics, /*showTextures=*/true, /*flatShading=*/false,
            /*waterDebug=*/false);
        VOX_LOGI("newvegas") << "imported draws restricted to " << requested;
    }
    // Sweepable, because "too subtle" is a measurable claim: the A/B against
    // AO-off below is what says whether a value actually changed the image.
    //
    // NOTE the intensity is an EXPONENT: sampleSsaoAmbientFactor computes
    // pow(ssaoRaw, intensity) on a value in [0,1]. Anything below 1 pushes the
    // result toward 1, i.e. actively weakens the occlusion -- which is what the
    // inherited 0.85 was doing.
    // 300 units, about 4.3 m at Bethesda's ~70 units per metre.
    //
    // This was briefly dropped to 150 to kill a muddy cast and a dark silhouette
    // fringe on Seyda Neen's shacks, and that was treating the estimator through
    // the radius: the fringe was GTAO's sample pattern, and switching the default
    // to XeGTAO removed it at the source. Re-measured afterwards on the same
    // pinned camera, as high-frequency energy in the AO channel alone
    // (ODAI_FNV_DEBUGVIEW=ao) and as how much of the frame is meaningfully
    // occluded:
    //
    //   radius   noise   occlusion   px below 200
    //     100    1.694     17.40        5.4%
    //     150    1.722     17.76        6.3%
    //     300    1.914     21.27       10.2%
    //     450    2.164     25.51       14.2%
    //
    // 300 costs 11% more AO noise than 150 and buys 60% more occluded frame, and
    // in the LIT frame -- where AO modulates ambient, which is a fraction of the
    // lighting -- the extra noise does not survive while the extra contact
    // darkening does. For scale, GTAO at radius 150 measured 3.95 on this
    // camera: XeGTAO at 300 is half as noisy as the value this replaced.
    //
    // Note a stale claim removed with it: the comment here used to say 128 was
    // "the shader's own clamp ceiling in ssao.comp.slang", which would have made
    // every value above it identical. Both estimators clamp to [0.25, 512]
    // (ssao.comp.slang and frame_pass_ssao.cc), so the sweep above is real.
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
    const std::string colorLook =
        std::getenv("ODAI_FNV_COLOR_LOOK") != nullptr ? std::getenv("ODAI_FNV_COLOR_LOOK") : "";
    if (colorLook == "cinematic") {
        // A measured middle between the neutral grade above and the engine's
        // stylized default, for the landscape flythroughs. Every number here
        // came off a pinned West Gash frame on the Seyda Neen -> Balmora route,
        // and the two it avoids are as informative as the ones it uses:
        //
        //   neutral    sd 0.188  p1 0.135  p99 0.643  sat 0.231
        //   stylized   sd 0.260  p1 0.000  p99 0.638  sat 0.397
        //
        // Stylized buys its contrast by CRUSHING THE SHADOWS TO ZERO, and on a
        // scene whose depth is carried by aerial perspective that deletes the
        // depth cue -- the far ridge and the near rock end up the same black.
        // So shadowDensity stays at 1.0 here and the contrast comes from the
        // midtones and from the white point instead, which lifts the top of the
        // histogram rather than pushing down the bottom.
        //
        // Vibrance is kept well under the stylized 0.12 because it targets the
        // LEAST saturated pixels, and in fog that is most of the frame.
        render::ColorGradingSettings grade;
        grade.midtoneContrast = 1.12f;
        grade.saturation = 1.10f;
        grade.vibrance = 0.05f;
        grade.shadowDensity = 1.0f;
        // Cool the shadows and warm the highlights very slightly: the classic
        // teal/amber split, at a fraction of the usual strength because
        // Morrowind's own palette is already blue-green.
        grade.shadowTint[2] = 0.03f;
        grade.highlightTint[0] = 0.03f;
        m_renderer.setColorGrading(grade);

        // The look INCLUDES its white point, because the grade alone does not
        // fix what is wrong with these frames. Grading on its own measured
        // sd 0.1845 / p99 0.634 -- i.e. it moved saturation and essentially
        // nothing else, since there was no highlight range for contrast to act
        // on. The white point is what supplies that range; the two together are
        // the look. ODAI_FNV_WHITEPOINT still overrides, in applyTonemapSettings.
        render::TonemapSettings tonemap = m_renderer.tonemapSettings();
        tonemap.whitePoint = 0.8f;
        tonemap.highlightShoulder = 1.0f;
        m_renderer.setTonemapSettings(tonemap);
    } else if (colorLook != "stylized") {
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
    applyTonemapSettings();

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
        // "weather" opens the picker sub-page. It needs its own value because it
        // is two keypresses deep and a screenshot run cannot press either.
        m_menuOpen = (demoMode == "menu" || demoMode == "weather");
        if (demoMode == "weather") {
            openWeatherPicker();
        }
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
    // This used to skip the read entirely unless a weather mod or an explicit
    // --weather gave it something to select. The pause menu's weather picker is
    // what made that premise false: vanilla's own records ARE selectable now, so
    // gating on a mod being present left the picker permanently empty on a stock
    // install. The read is a top-level group-header walk (see
    // buildFalloutWeatherTables) rather than a scan of the file, so paying it
    // unconditionally is close to free: measured at 0.49 ms for vanilla's 63
    // WTHR and 31 CLMT, against a ~2.0 s startup. The log line below carries the
    // number so a regression here cannot hide.

    std::vector<std::string> requested;
    requested.push_back(m_streamPlugin);
    requested.insert(requested.end(), m_extraPlugins.begin(), m_extraPlugins.end());

    // Timed because this now runs on every launch rather than only when a
    // weather mod is loaded, and because a plugin scan that forgets to filter on
    // the record header is the classic way to add seconds to startup here
    // without anything looking wrong. See CLAUDE.md on onRecordHeader.
    const core::Stopwatch weatherTimer;

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
                         << m_weatherTables.climates.size() << " CLMT in "
                         << weatherTimer.elapsedMs() << " ms";

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
        // Fall back to the climate of the worldspace WE ARE STREAMING: whichever
        // of its weathers has the highest chance is the closest thing to "what
        // you would normally see here" without running the mod's selection
        // scripts.
        //
        // This used to walk climateByWorldspaceFormId and take the first entry
        // that had any weathers, discarding the worldspace key entirely. With
        // FalloutNV.esm alone that is survivable -- there is not much to pick
        // wrongly from. Add a plugin that pulls the DLCs in as masters, which
        // any "base game + all DLC" patch does, and the first entry in an
        // unordered_map became Lonesome Road's NVDLC04NukedClimate: the Mojave
        // rendered under the Divide's irradiated sky, green from horizon to
        // zenith. The map is keyed by worldspace precisely so it can be asked
        // about ONE worldspace, and the fix is to ask.
        const importer::fnv::FalloutClimateRecord* bestClimate = nullptr;
        const auto worldspaceIt = m_weatherTables.worldspaceFormIdByEditorId.find(
            toLowerAscii(m_streamWorldspace));
        if (worldspaceIt == m_weatherTables.worldspaceFormIdByEditorId.end()) {
            VOX_LOGW("newvegas") << "weather: no worldspace record named \"" << m_streamWorldspace
                                 << "\"; leaving the procedural sky alone";
        } else {
            // A WALLED CITY INHERITS ITS PARENT'S CLIMATE. Skyrim's
            // WhiterunWorld record is an EDID and a WNAM and nothing else, so
            // asking it for a climate finds none and the city renders under the
            // bare procedural sky -- no authored gradient, and no cloud layer at
            // all. Tamriel, one hop up, carries the climate for the whole
            // province. Bounded rather than recursive; the chain is one link in
            // practice and a cycle must not hang startup.
            auto climateIt = m_weatherTables.climateByWorldspaceFormId.find(worldspaceIt->second);
            std::uint32_t inheritedFrom = worldspaceIt->second;
            for (int hop = 0; hop < 8 && climateIt == m_weatherTables.climateByWorldspaceFormId.end();
                 ++hop) {
                const auto parentIt = m_weatherTables.parentWorldspaceFormId.find(inheritedFrom);
                if (parentIt == m_weatherTables.parentWorldspaceFormId.end()) {
                    break;
                }
                inheritedFrom = parentIt->second;
                climateIt = m_weatherTables.climateByWorldspaceFormId.find(inheritedFrom);
            }
            if (climateIt != m_weatherTables.climateByWorldspaceFormId.end() &&
                inheritedFrom != worldspaceIt->second) {
                VOX_LOGI("newvegas") << "weather: " << m_streamWorldspace
                                     << " names no climate; inherited from parent worldspace 0x"
                                     << std::hex << inheritedFrom << std::dec;
            }
            if (climateIt == m_weatherTables.climateByWorldspaceFormId.end()) {
                VOX_LOGW("newvegas") << "weather: worldspace " << m_streamWorldspace
                                     << " names no climate; leaving the procedural sky alone";
            } else {
                const auto found = m_weatherTables.climates.find(climateIt->second);
                if (found == m_weatherTables.climates.end() || found->second.weathers.empty()) {
                    VOX_LOGW("newvegas")
                        << "weather: climate for " << m_streamWorldspace
                        << " is missing or lists no weathers; leaving the procedural sky alone";
                } else {
                    bestClimate = &found->second;
                }
            }
        }
        if (bestClimate != nullptr) {
            // TNAM, in 10-minute units, giving the START and END of each
            // transition; the samplers want the single hour at which the
            // Sunrise and Sunset slots PEAK, which is the midpoint.
            // SkyrimClimate authors 5:30-10:00 and 16:00-20:30, so its dawn
            // peaks at 7:45 rather than the 6:00 default -- close to two hours
            // out, which is a whole slot's worth of colour.
            const auto hoursFromTnam = [](std::uint8_t begin, std::uint8_t end) {
                return ((static_cast<float>(begin) + static_cast<float>(end)) * 0.5f) / 6.0f;
            };
            const float sunrise = hoursFromTnam(bestClimate->sunriseBegin, bestClimate->sunriseEnd);
            const float sunset = hoursFromTnam(bestClimate->sunsetBegin, bestClimate->sunsetEnd);
            // A climate with no TNAM reads as 0 and 0, which would put dusk
            // before dawn and collapse the whole day curve onto one slot.
            if (sunrise > 0.5f && sunset > sunrise + 1.0f && sunset < 23.5f) {
                m_sunriseHour = sunrise;
                m_sunsetHour = sunset;
            }
            VOX_LOGI("newvegas") << "climate " << bestClimate->editorId << ": sunrise peaks "
                                 << m_sunriseHour << "h, sunset " << m_sunsetHour << "h";
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

    // Everything from here -- clouds, sky colours, audio, tonemap -- is the same
    // work a runtime weather change does, so it is one call rather than a copy.
    selectWeather(m_activeWeatherFormId);
}

// The post-processing curve. Deliberately NOT part of selectWeather, where it
// used to live: initWeather returns early on any plugin with no TES4 record, so
// every Morrowind run silently ignored ODAI_FNV_TONEMAP entirely. A renderer
// setting that has nothing to do with weather must not be reachable only
// through the weather path -- the symptom was an env var that measured
// byte-identical to unset and looked like a broken tonemap rather than a
// missing call.
void NewVegasApp::applyTonemapSettings() {
    // ODAI_FNV_WHITEPOINT=<scene linear>[,<shoulder>] pins a scene value to
    // display white on the ACES path. Off by default, because it is a look
    // change and every other game shares this curve.
    //
    // What it is FOR: measured across a Seyda-Neen-to-Balmora flight, the 99th
    // percentile of frame luma sat between 0.64 and 0.70 and moved by under
    // 0.02 under every other knob in the chain -- fog distance, the ENB curve,
    // the stylized colour look. The frame never reached white, and no amount of
    // grading fixes a range the tonemap did not produce.
    if (const char* whiteEnv = std::getenv("ODAI_FNV_WHITEPOINT")) {
        render::TonemapSettings tonemap = m_renderer.tonemapSettings();
        tonemap.whitePoint = std::strtof(whiteEnv, nullptr);
        if (const char* comma = std::strchr(whiteEnv, ',')) {
            tonemap.highlightShoulder = std::strtof(comma + 1, nullptr);
        }
        m_renderer.setTonemapSettings(tonemap);
        VOX_LOGI("newvegas") << "tonemap white point " << tonemap.whitePoint << ", shoulder "
                             << tonemap.highlightShoulder;
    }

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

// Turns Skyrim's semantic cloud band into the slice of sky it covers and the
// projection its art was drawn for. The record says WHERE a layer belongs (see
// FalloutCloudBand); this says what that means to the renderer, which is the
// game's business rather than the importer's.
//
// The windows overlap on purpose: a deck of cloud and the bank under it are not
// separated by a line in the sky. The numbers are dir.y, so 0 is the horizon
// and 1 the zenith, and the horizon band's 0.30 ceiling is about 17 degrees --
// roughly where a real cloud bank stops reading as "on the skyline".
void applySkyrimCloudBand(importer::fnv::FalloutCloudBand band,
                          render::WeatherCloudLayer& target) {
    using Band = importer::fnv::FalloutCloudBand;
    switch (band) {
        case Band::Upper:
            target.mapping = render::WeatherCloudMapping::TilingPlane;
            target.scale = 2.6f;
            target.bandLow = 0.20f;
            target.bandHigh = 1.0f;
            break;
        case Band::Lower:
            target.mapping = render::WeatherCloudMapping::TilingPlane;
            target.scale = 1.5f;
            target.bandLow = 0.10f;
            target.bandHigh = 1.0f;
            break;
        case Band::Horizon:
            target.mapping = render::WeatherCloudMapping::Cylindrical;
            // Bearing repeats: the art is one bank, not a panorama of the whole
            // compass, so it has to go round more than once.
            target.scale = 4.0f;
            target.bandLow = 0.0f;
            target.bandHigh = 0.30f;
            break;
        case Band::Fill:
        case Band::WholeSky:
            target.mapping = render::WeatherCloudMapping::TilingPlane;
            target.scale = 2.0f;
            break;
    }
}

}  // namespace

// Everything that has to happen when the active weather changes: cloud layers
// re-uploaded, sky colours re-decoded, audio re-picked.
//
// Split out of initWeather so the pause menu can change weather at runtime.
// Doing it by hand at the call site was never going to stay correct -- setting
// m_activeWeatherFormId alone leaves the previous weather's cloud textures on
// the GPU and its rain still playing, which reads as "the picker only half
// works" rather than as a missing call.
void NewVegasApp::selectWeather(std::uint32_t weatherFormId) {
    m_activeWeatherFormId = weatherFormId;

    // Cloud layers. These come out of the mod's own BSA, which the streamer's
    // asset source already indexes -- reusing it means no second search path and
    // no second copy of the loose-beats-archive precedence rules.
    const importer::fnv::FalloutWeatherRecord* active =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    if (active != nullptr && m_streamer != nullptr) {
        const importer::fnv::FalloutAssetSource& assets = m_streamer->assets();
        render::WeatherCloudTextures clouds;
        const bool tiling =
            active->cloudMapping == importer::fnv::FalloutCloudMapping::TilingPlane;
        for (int& source : m_cloudLayerSource) {
            source = -1;
        }
        int loadedLayers = 0;

        // The record has already dropped the layers it disables and paired each
        // survivor with its own tint; this only has to fill the renderer's four
        // slots with the ones whose textures actually resolve, REMEMBERING WHICH
        // LAYER EACH SLOT CAME FROM so applyWeather tints it from the same one.
        for (std::size_t index = 0;
             index < active->cloudLayers.size() &&
             static_cast<std::size_t>(loadedLayers) < render::kWeatherCloudLayerCount;
             ++index) {
            const importer::fnv::FalloutWeatherCloudLayer& layer = active->cloudLayers[index];
            const std::size_t slot = static_cast<std::size_t>(loadedLayers);
            std::vector<std::uint8_t> bytes;
            std::string assetError;
            if (!assets.resolveTexture(layer.texture, bytes, assetError)) {
                // Not a warning for Skyrim: a retail record names layers whose
                // textures do not ship (SkyrimCloudy's disabled ones are dead
                // leftovers from the Oblivion and Fallout records it was copied
                // from), and NAM1 has already thrown those away. One that gets
                // this far is worth hearing about.
                VOX_LOGW("newvegas") << "cloud layer " << layer.index << " (" << layer.texture
                                     << ") unresolved: " << assetError;
                continue;
            }
            render::WeatherCloudLayer& target = clouds.layers[slot];
            if (!importer::loadDdsFromMemory(bytes.data(), bytes.size(), target.texture)) {
                VOX_LOGW("newvegas") << "cloud layer " << layer.index << " (" << layer.texture
                                     << ") failed to decode";
                target.texture = importer::ImportedSceneTexture{};
                continue;
            }
            target.texture.sourcePath = layer.texture;
            if (tiling) {
                // Texture units per second. Skyrim's bytes are a rate, not a
                // velocity in any unit this renderer shares, so the scale is
                // chosen for the look: a sheet crosses the sky in a couple of
                // minutes at the speeds SkyrimCloudy authors.
                target.scrollU = layer.driftX * 0.010f;
                target.scrollV = layer.driftY * 0.010f;
                applySkyrimCloudBand(layer.band, target);
            } else {
                // Radians per second about the zenith -- a dome map rotates, it
                // does not translate.
                target.mapping = render::WeatherCloudMapping::DomeFisheye;
                target.scrollU = layer.driftX * 0.0035f;
                // Dome scale: 1.0 puts the horizon exactly on the texture's
                // inscribed circle, which is how these fisheye sky maps are
                // drawn. Slightly under 1 for the upper layers pulls their rim
                // inside the horizon so they read as higher and further away.
                target.scale = (layer.index < 2) ? 1.0f : 0.92f;
            }
            m_cloudLayerSource[slot] = static_cast<int>(index);
            ++loadedLayers;
        }
        VOX_LOGI("newvegas") << "cloud layers: " << loadedLayers << " of "
                             << active->cloudLayers.size() << " authored in use ("
                             << (tiling ? "tiling sheets" : "dome fisheye") << ")";
        m_renderer.setWeatherClouds(clouds);
    }

    applyWeather();
    initWeatherAudio();


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
        // No climate names a weather -- every Oblivion worldspace, and any FNV
        // one before the first weather resolves. The sky is left procedural, but
        // the aerial-perspective distance still has to be published or the
        // shader falls back to 15000 units (~214 m) and a city vista renders
        // behind a wall of milk. Weight stays 0, so only the distance is taken.
        static const float s_fogFar = []() {
            const char* env = std::getenv("ODAI_FNV_FOGFAR");
            const float value = env != nullptr ? static_cast<float>(std::atof(env)) : 160000.0f;
            return value > 1.0f ? value : 160000.0f;
        }();
        render::WeatherSkyParams clear;
        clear.weight = 0.0f;
        clear.fogFarDistance = s_fogFar;
        m_renderer.setWeatherSky(clear);
        return;
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
    // The climate's own dawn and dusk, not the sampler's 6/19 defaults; see
    // m_sunriseHour. Threaded through EVERY sample below, because two samples
    // taken against different day curves disagree about which slot it is --
    // the sky would reach its sunset colour while the clouds were still on Day.
    const float dawn = m_sunriseHour;
    const float dusk = m_sunsetHour;
    const auto skyColor = [&](FalloutWeatherColor channel) {
        return sampleFalloutWeatherColor(*weather, channel, hour, dawn, dusk);
    };
    render::WeatherSkyParams params;
    params.weight = 1.0f;
    decode(skyColor(FalloutWeatherColor::SkyUpper), params.skyUpper);
    decode(skyColor(FalloutWeatherColor::SkyLower), params.skyLower);
    decode(skyColor(FalloutWeatherColor::Horizon), params.horizon);
    // Aerial perspective is the FAR haze, so a record that separates the two
    // (Skyrim) must hand over the far colour; channel 1 is its near fog and is
    // a saturated tint meant for the first few metres.
    if (weather->hasFogFarColor) {
        decode(sampleFalloutWeatherColorRow(weather->fogFarColors, hour, dawn, dusk),
               params.fogColor);
    } else {
        decode(skyColor(FalloutWeatherColor::Fog), params.fogColor);
    }
    // Day fog until dusk, night fog after; the record authors the two
    // separately and there is no third value to interpolate toward.
    const bool daytime = hour >= dawn && hour < dusk;
    params.fogFarDistance = daytime ? weather->fogDayFar : weather->fogNightFar;

    // Sunlight and Ambient light the GROUND. These two channels were read out
    // of every record and then dropped, so a storm rendered as a dark sky over
    // sunlit terrain -- the sky was the only thing the weather touched.
    //
    // Decoded plainly to linear rather than through decode() above: that
    // function's gain and contrast exist to make an EMISSIVE sky readable
    // through an ACES curve, and applying them here would push a light source
    // through a display-referred fudge twice. The renderer takes hue from these
    // and bounds the intensity itself; see WeatherSkyParams::lightingWeight.
    const auto decodeLinear = [](const importer::fnv::FalloutColorRgb& color, float* out) {
        const auto channel = [](std::uint8_t value) {
            const float srgb = static_cast<float>(value) / 255.0f;
            return srgb <= 0.04045f ? (srgb / 12.92f)
                                    : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
        };
        out[0] = channel(color.r);
        out[1] = channel(color.g);
        out[2] = channel(color.b);
    };
    decodeLinear(skyColor(FalloutWeatherColor::Sunlight), params.sunlightColor);
    decodeLinear(skyColor(FalloutWeatherColor::Ambient), params.ambientColor);
    // ODAI_FNV_LIGHT_WEIGHT=0 is the A/B control, and the only way to attribute
    // a brightness change to the weather rather than to the sky gradient.
    static const float s_lightingWeight = []() {
        const char* env = std::getenv("ODAI_FNV_LIGHT_WEIGHT");
        return env != nullptr ? std::clamp(static_cast<float>(std::atof(env)), 0.0f, 1.0f) : 1.0f;
    }();
    params.lightingWeight = s_lightingWeight;
    // DATA's Sun Glare byte, which is the one field in that block the sky can
    // use directly. SkyrimCloudy authors 153 of 255; a fog weather authors far
    // less, and the difference is a sun with a halo against one that is a bare
    // disc in soup.
    params.sunGlare = static_cast<float>(weather->sunGlare) / 255.0f;

    // Cloud tints and opacities come from the layer the slot is DRAWING, not
    // from the slot number -- see m_cloudLayerSource. Both track time of day
    // and are sampled the same way the sky colours are.
    for (int slot = 0; slot < render::kWeatherCloudLayerCount; ++slot) {
        // ODAI_FNV_NOCLOUDS isolates the sky gradient from the cloud layers.
        // Worth keeping: "the sky is black" has two very different causes
        // (an authored-dark gradient vs. total cloud cover) and they are
        // indistinguishable on screen.
        static const bool s_noClouds = std::getenv("ODAI_FNV_NOCLOUDS") != nullptr;
        const int source = m_cloudLayerSource[slot];
        if (source < 0 || static_cast<std::size_t>(source) >= weather->cloudLayers.size() ||
            s_noClouds) {
            params.cloudOpacity[slot] = 0.0f;
            continue;
        }
        const importer::fnv::FalloutWeatherCloudLayer& layer =
            weather->cloudLayers[static_cast<std::size_t>(source)];
        decode(sampleFalloutWeatherCloudTint(layer, hour, dawn, dusk), params.cloudTint[slot]);
        // JNAM, where the record authors one. Skyrim holds its fully-opaque
        // 32x32 fill swatch at 0.4-0.5 here; drawn at 1.0 that layer is a coat
        // of paint over the whole sky.
        params.cloudOpacity[slot] = sampleFalloutWeatherCloudAlpha(layer, hour, dawn, dusk);
    }
    // One line per weather change, not per frame: "the sky is black" is
    // otherwise indistinguishable from "the sky is not being set at all".
    static std::uint32_t s_loggedWeather = 0;
    if (s_loggedWeather != m_activeWeatherFormId) {
        s_loggedWeather = m_activeWeatherFormId;
        VOX_LOGI("newvegas") << "sky linear rgb: upper(" << params.skyUpper[0] << ","
                             << params.skyUpper[1] << "," << params.skyUpper[2] << ") horizon("
                             << params.horizon[0] << "," << params.horizon[1] << ","
                             << params.horizon[2] << ") fog(" << params.fogColor[0] << ","
                             << params.fogColor[1] << "," << params.fogColor[2]
                             << ") fogFar=" << params.fogFarDistance
                             << " weight=" << params.weight;
    }
    m_renderer.setWeatherSky(params);
}

namespace {

// The tour, in engine space: where the camera is and what it is pointed at.
//
// THESE ARE THE TOWN'S OWN LANDMARKS, not invented viewpoints. Each look-at is
// the doorstep of a named interior cell, read out of the plugin by spawning at
// it (--spawn GSProspectorSaloonInterior and friends print the position), so
// the tour is aimed at the buildings Goodsprings actually has rather than at
// coordinates that looked good once and drift the moment anything moves.
//
//   Prospector Saloon      (-67452, 8472, -4900)
//   General Store          (-69319, 8501, -3528)
//   Gas Station            (-75169, 8880, -4076)
//   Doc Mitchell's House   (-73163, 8806, -1312)
//   Schoolhouse            (-74482, 8354,  4780)
//   Victor's Shack         (-72319, 8440,  5928)
//
// Camera heights are absolute rather than ground-relative: the tour is a drone
// shot over a valley whose floor moves 500 units under it, and clamping to the
// terrain would make the camera bob over every rise it crossed.
struct FlyWaypoint {
    float position[3];
    float lookAt[3];
};

// A tour loaded from disk, when --tour-file names one. Empty means the built-in
// Goodsprings list below.
//
// A file rather than more hardcoded arrays because framing a flythrough is
// iterative -- every waypoint is a guess until you watch it -- and a rebuild per
// guess makes that loop useless. It is also what lets one binary tour three
// different games.
std::vector<FlyWaypoint> g_runtimeTour;

constexpr FlyWaypoint kGoodspringsTour[] = {
    // High and south, the whole town in frame.
    {{-70600.0f, 10600.0f, -10200.0f}, {-70600.0f, 8700.0f, -4400.0f}},
    // Down toward the saloon, the first building anyone sees.
    {{-69100.0f,  9500.0f,  -7200.0f}, {-67452.0f, 8620.0f, -4900.0f}},
    // Low across the saloon front, turning onto Easy Pete's spot beside it.
    {{-67500.0f,  8830.0f,  -5300.0f}, {-67845.0f, 8480.0f, -3334.0f}},
    // The general store, on the corner of the main road.
    {{-68400.0f,  8980.0f,  -4100.0f}, {-69319.0f, 8620.0f, -3528.0f}},
    // Along the road toward Doc Mitchell's, Victor parked outside it.
    {{-71400.0f,  9080.0f,  -2600.0f}, {-72943.0f, 8780.0f, -1092.0f}},
    // Over the spawn, then north up the rise to the schoolhouse.
    {{-73200.0f,  9100.0f,   -200.0f}, {-74482.0f, 8560.0f,  4780.0f}},
    {{-73900.0f,  9050.0f,   2900.0f}, {-74482.0f, 8500.0f,  4780.0f}},
    // Victor's shack at the north end, then the turn back south.
    {{-73000.0f,  8900.0f,   4900.0f}, {-72319.0f, 8560.0f,  5928.0f}},
    {{-71200.0f,  8980.0f,   4400.0f}, {-68000.0f, 8500.0f,  2200.0f}},
    // Down to head height over the east end of town, which is where the people
    // are: the five Powder Gangers stand around (-66131, 1645), with a settler
    // and the bighorners between there and the road.
    {{-68900.0f,  8760.0f,   2300.0f}, {-66991.0f, 8300.0f,  1981.0f}},
    {{-67400.0f,  8600.0f,   2350.0f}, {-66500.0f, 8350.0f,  1900.0f}},
    {{-65700.0f,  8700.0f,   2050.0f}, {-65300.0f, 8420.0f,  1500.0f}},
};

// Where the tour stops aiming at coordinates and starts aiming at whoever is
// there. The last waypoints are over the east end of town, and the people
// standing there WANDER -- up to 950 units from where the plugin put them --
// so a fixed look-at is a coin flip that lands on empty dirt about as often as
// it lands on a person. See updateFlythrough.
constexpr float kTourActorTrackStart = 0.72f;
constexpr float kTourActorTrackFull = 0.86f;
constexpr int kBuiltinTourCount =
    static_cast<int>(sizeof(kGoodspringsTour) / sizeof(kGoodspringsTour[0]));

// True once --tour-file has replaced the built-in path. The camera treats an
// authored tour as authoritative: the actor hand-off below is a flourish
// written for Goodsprings, where the tour ends among the townspeople on
// purpose, and it silently overrides whatever a tour file aimed at. In Megaton
// that meant a pan across the shanties turned into a top-down stare at a
// settler standing on the crater floor.
bool tourIsAuthored() {
    return !g_runtimeTour.empty();
}

int tourCount() {
    return g_runtimeTour.empty() ? kBuiltinTourCount
                                 : static_cast<int>(g_runtimeTour.size());
}

// Critically damped smoothing toward a target, with a ceiling on rate of
// change. The standard spring-damper solution with the exponential replaced by
// its Pade approximation, so it is stable at any timestep -- an explicit spring
// integrated with a 40 ms frame overshoots and rings.
//
// `smoothSeconds` is roughly how long it takes to cover the distance, not a
// half-life. `velocity` is carried by the caller because that state is what
// makes it critically damped rather than exponential.
float smoothDampAngle(float current, float target, float& velocity, float smoothSeconds,
                      float maxRatePerSecond, float deltaSeconds) {
    if (deltaSeconds <= 0.0f) {
        return current;
    }
    const float omega = 2.0f / std::max(smoothSeconds, 1e-4f);
    const float x = omega * deltaSeconds;
    const float decay = 1.0f / (1.0f + x + (0.48f * x * x) + (0.235f * x * x * x));
    // Clamping the DISTANCE rather than the step is what makes the rate ceiling
    // behave: the filter then eases toward a target it is allowed to reach,
    // instead of being clipped every frame and arriving with a corner.
    const float maxDistance = maxRatePerSecond * smoothSeconds;
    const float change = std::clamp(current - target, -maxDistance, maxDistance);
    const float clampedTarget = current - change;
    const float temp = (velocity + (omega * change)) * deltaSeconds;
    velocity = (velocity - (omega * temp)) * decay;
    float result = clampedTarget + ((change + temp) * decay);
    // Do not overshoot past the target from the wrong side.
    if (((target - current) > 0.0f) == (result > target)) {
        result = target;
        velocity = (result - target) / deltaSeconds;
    }
    return result;
}

// Duplicate the ends rather than wrapping: this is a path, not a loop, and
// wrapping would curve the first segment toward the last landmark.
const FlyWaypoint& tourWaypoint(int index) {
    const int clamped = std::clamp(index, 0, tourCount() - 1);
    return g_runtimeTour.empty() ? kGoodspringsTour[clamped] : g_runtimeTour[clamped];
}

// CENTRIPETAL Catmull-Rom (Barry-Goldman form, alpha = 0.5), not the uniform
// one this used to use.
//
// Uniform Catmull-Rom takes the tangent at p1 as (p2 - p0)/2 regardless of how
// far apart those points actually are. These waypoints are not evenly spaced --
// the legs run from ~1500 to ~3400 units -- so the curve arrives at a knot with
// one speed and leaves it with another, and every waypoint is a visible kink.
// Uneven spacing also lets the uniform form overshoot and, at a tight corner,
// cusp: the camera briefly reverses. Parameterizing the knots by sqrt(chord
// length) is the standard fix and is guaranteed cusp- and self-intersection-free.
//
// Knot spacings are clamped away from zero because the ends are duplicated, so
// the first and last spans have zero chord length and would divide by it.
void centripetalKnots(const float p0[3], const float p1[3], const float p2[3],
                      const float p3[3], float outKnots[4]) {
    const auto span = [](const float a[3], const float b[3]) {
        const float dx = b[0] - a[0];
        const float dy = b[1] - a[1];
        const float dz = b[2] - a[2];
        return std::max(std::sqrt(std::sqrt((dx * dx) + (dy * dy) + (dz * dz))), 1e-4f);
    };
    outKnots[0] = 0.0f;
    outKnots[1] = outKnots[0] + span(p0, p1);
    outKnots[2] = outKnots[1] + span(p1, p2);
    outKnots[3] = outKnots[2] + span(p2, p3);
}

// One de-Boor-style pyramid step: linear blend of a and b over [ta, tb] at t.
void knotLerp(const float a[3], const float b[3], float ta, float tb, float t, float out[3]) {
    const float denominator = (tb - ta);
    const float w = (std::abs(denominator) < 1e-6f) ? 0.0f : ((t - ta) / denominator);
    for (int axis = 0; axis < 3; ++axis) {
        out[axis] = a[axis] + ((b[axis] - a[axis]) * w);
    }
}

// Evaluate the curve through p1..p2 at local parameter s in [0,1], using a knot
// vector supplied by the caller. The look-at spline is deliberately evaluated
// against the POSITION knots rather than its own: the two must stay paired
// frame for frame, and a look-at sequence has repeated entries (waypoints 5 and
// 6 share one) whose own centripetal knots would advance at a different rate.
void evaluateCentripetal(const float p0[3], const float p1[3], const float p2[3],
                         const float p3[3], const float knots[4], float s, float out[3]) {
    const float t = knots[1] + ((knots[2] - knots[1]) * s);
    float a1[3];
    float a2[3];
    float a3[3];
    knotLerp(p0, p1, knots[0], knots[1], t, a1);
    knotLerp(p1, p2, knots[1], knots[2], t, a2);
    knotLerp(p2, p3, knots[2], knots[3], t, a3);
    float b1[3];
    float b2[3];
    knotLerp(a1, a2, knots[0], knots[2], t, b1);
    knotLerp(a2, a3, knots[1], knots[3], t, b2);
    knotLerp(b1, b2, knots[1], knots[2], t, out);
}

// Sample at a parameter measured in SEGMENTS, i.e. u in [0,1] spans the whole
// waypoint list with each leg getting an equal slice of u regardless of length.
void sampleTourByParameter(float u, float outPosition[3], float outLookAt[3]) {
    const float span = static_cast<float>(tourCount() - 1);
    const float scaled = std::clamp(u, 0.0f, 1.0f) * span;
    const int segment = std::min(static_cast<int>(scaled), tourCount() - 2);
    const float s = scaled - static_cast<float>(segment);
    const FlyWaypoint& w0 = tourWaypoint(segment - 1);
    const FlyWaypoint& w1 = tourWaypoint(segment);
    const FlyWaypoint& w2 = tourWaypoint(segment + 1);
    const FlyWaypoint& w3 = tourWaypoint(segment + 2);
    float knots[4];
    centripetalKnots(w0.position, w1.position, w2.position, w3.position, knots);
    evaluateCentripetal(w0.position, w1.position, w2.position, w3.position, knots, s, outPosition);
    evaluateCentripetal(w0.lookAt, w1.lookAt, w2.lookAt, w3.lookAt, knots, s, outLookAt);
}

// Arc-length reparameterization.
//
// Equal u per leg means equal TIME per leg, and the legs differ in length by
// better than 2x -- so the camera visibly speeds up over the long run to Doc
// Mitchell's and crawls across the short hops at the east end. Constant ground
// speed is what reads as a smooth dolly, so the eased parameter below is a
// distance along the path and this table converts it back to a curve parameter.
constexpr int kTourArcSamples = 1024;

// Rebuilt whenever the loaded tour changes rather than cached forever.
//
// This was a function-local `static const` built on first call, which is
// correct only because --tour-file happens to be parsed before the first frame.
// Any runtime tour swap would keep the OLD path's arc-length table and silently
// reparameterize the new curve by the old one's distances -- a camera that
// speeds up and slows down at the previous tour's waypoints, which is a very
// hard symptom to attribute back to here. Keying on the waypoint count and
// first waypoint costs one comparison per frame and removes the trap.
const std::vector<float>& tourArcLengths() {
    static std::vector<float> table;
    static int builtForWaypointCount = -1;
    static FlyWaypoint builtForFirstWaypoint{};
    const int waypointCount = tourCount();
    const FlyWaypoint firstWaypoint = tourWaypoint(0);
    const bool stale =
        builtForWaypointCount != waypointCount ||
        builtForFirstWaypoint.position[0] != firstWaypoint.position[0] ||
        builtForFirstWaypoint.position[1] != firstWaypoint.position[1] ||
        builtForFirstWaypoint.position[2] != firstWaypoint.position[2];
    if (!stale) {
        return table;
    }
    builtForWaypointCount = waypointCount;
    builtForFirstWaypoint = firstWaypoint;
    table = [&]() {
        std::vector<float> lengths(kTourArcSamples + 1, 0.0f);
        float previous[3] = {};
        float ignoredLookAt[3] = {};
        sampleTourByParameter(0.0f, previous, ignoredLookAt);
        for (int i = 1; i <= kTourArcSamples; ++i) {
            float current[3] = {};
            sampleTourByParameter(static_cast<float>(i) / static_cast<float>(kTourArcSamples),
                                  current, ignoredLookAt);
            const float dx = current[0] - previous[0];
            const float dy = current[1] - previous[1];
            const float dz = current[2] - previous[2];
            lengths[i] = lengths[i - 1] + std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
            previous[0] = current[0];
            previous[1] = current[1];
            previous[2] = current[2];
        }
        const float total = lengths.back();
        if (total > 1e-3f) {
            for (float& entry : lengths) {
                entry /= total;
            }
        }
        return lengths;
    }();
    return table;
}

// distance in [0,1] along the path -> curve parameter in [0,1].
float tourParameterAtDistance(float distance) {
    const std::vector<float>& lengths = tourArcLengths();
    const float target = std::clamp(distance, 0.0f, 1.0f);
    const auto upper = std::upper_bound(lengths.begin(), lengths.end(), target);
    if (upper == lengths.begin()) {
        return 0.0f;
    }
    if (upper == lengths.end()) {
        return 1.0f;
    }
    const auto lower = upper - 1;
    const float lowerValue = *lower;
    const float upperValue = *upper;
    const float denominator = upperValue - lowerValue;
    const float fraction = (denominator > 1e-9f) ? ((target - lowerValue) / denominator) : 0.0f;
    const float index = static_cast<float>(lower - lengths.begin()) + fraction;
    return index / static_cast<float>(kTourArcSamples);
}

void sampleTour(float distance, float outPosition[3], float outLookAt[3]) {
    sampleTourByParameter(tourParameterAtDistance(distance), outPosition, outLookAt);
}

}  // namespace

// "px py pz  lx ly lz" per line; '#' comments and blank lines ignored. Returns
// how many waypoints were loaded so a typo in the path is not silently a
// built-in Goodsprings tour of a Fallout 3 worldspace.
int loadTourFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        return 0;
    }
    std::vector<FlyWaypoint> loaded;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line.resize(hash);
        }
        FlyWaypoint waypoint{};
        std::istringstream stream(line);
        if (stream >> waypoint.position[0] >> waypoint.position[1] >> waypoint.position[2] >>
            waypoint.lookAt[0] >> waypoint.lookAt[1] >> waypoint.lookAt[2]) {
            loaded.push_back(waypoint);
        }
    }
    // Catmull-Rom needs four control points; fewer than four cannot describe a
    // curve and would index past the ends.
    if (loaded.size() < 4u) {
        return 0;
    }
    g_runtimeTour = std::move(loaded);
    return static_cast<int>(g_runtimeTour.size());
}

bool NewVegasApp::updateFlythrough(float deltaSeconds) {
    m_flythroughTime += deltaSeconds;
    const float raw = std::clamp(m_flythroughTime / m_flythroughSeconds, 0.0f, 1.0f);
    // Ease only the ENDS, at constant speed in between.
    //
    // A smoothstep over the whole path is the obvious thing and it is wrong for
    // a tour: it makes the middle -- where all the landmarks are -- rush past
    // at nearly double speed while the first and last waypoints get a third of
    // the running time each. This is the integral of a speed profile that ramps
    // up over the first `kEase` of the run, holds, and ramps back down.
    //
    // THE RAMP-DOWN BRANCH WAS DIVIDING THE WRONG NUMERATOR. The profile's area
    // up to `raw` is `total - remaining^2 / (2 * kEase)`, and it was written as
    // `1 - remaining^2 / (2 * kEase)` -- larger by exactly kEase. So at
    // raw = 1 - kEase the eased parameter jumped from 0.962 straight to 1.0 and
    // the std::min pinned it there: the camera SNAPPED forward over the last
    // stretch of path and then sat frozen for the final 7% of the run, while
    // the actor tracking below went on swinging the aim around. That is the
    // "smooth until the end, then jittery" this had.
    constexpr float kEase = 0.07f;
    const float total = 1.0f - kEase;  // area under that profile
    float eased = raw;
    if (raw < kEase) {
        eased = ((raw * raw) / (2.0f * kEase)) / total;
    } else if (raw > 1.0f - kEase) {
        const float remaining = 1.0f - raw;
        eased = (total - ((remaining * remaining) / (2.0f * kEase))) / total;
    } else {
        eased = ((kEase * 0.5f) + (raw - kEase)) / total;
    }
    eased = std::clamp(eased, 0.0f, 1.0f);

    float position[3] = {};
    float lookAt[3] = {};
    sampleTour(eased, position, lookAt);
    m_cameraX = position[0];
    m_cameraY = position[1];
    m_cameraZ = position[2];

    // Hand the aim over to a real inhabitant for the last stretch. Nearest
    // walker to the camera, aimed at the chest rather than the feet, blended in
    // so the camera drifts onto them instead of snapping.
    if (eased > kTourActorTrackStart && !tourIsAuthored()) {
        // LATCHED, not re-chosen every frame. Picking the nearest actor afresh
        // each frame makes the aim jump the moment two of them swap places in
        // the ordering, and again whenever one crosses the near or far cutoff --
        // the target teleports across the width of the town between one frame
        // and the next. Once someone is chosen the tour stays with them, which
        // is also what a real camera operator would do.
        if (m_tourTrackedActor >= 0 &&
            (static_cast<std::size_t>(m_tourTrackedActor) >= m_actors.size() ||
             !m_actors[static_cast<std::size_t>(m_tourTrackedActor)].placed)) {
            m_tourTrackedActor = -1;
        }
        if (m_tourTrackedActor < 0) {
            float bestDistanceSq = 4000.0f * 4000.0f;
            for (std::size_t i = 0; i < m_actors.size(); ++i) {
                const SkinnedActor& actor = m_actors[i];
                if (!actor.placed || !actor.wanders) {
                    continue;
                }
                const float dx = actor.position[0] - m_cameraX;
                const float dz = actor.position[2] - m_cameraZ;
                const float distanceSq = (dx * dx) + (dz * dz);
                // Not the one under the camera's nose: at a few hundred units an
                // actor fills the frame and the town behind them is gone.
                //
                // 1200, not the 300 this had. The tour flies ~500 units above
                // the townspeople, so 300 units of GROUND distance is a 59
                // degree downward aim -- and since the camera is still moving
                // toward them, it then passes very nearly overhead. Aiming
                // through the nadir makes yaw ill-conditioned: the measured
                // trace swung 140 degrees of yaw in 0.3 s at raw 0.79 while the
                // target itself barely moved. 1200 units keeps the aim under
                // ~23 degrees down and the pass off to one side.
                if (distanceSq < 1200.0f * 1200.0f || distanceSq >= bestDistanceSq) {
                    continue;
                }
                bestDistanceSq = distanceSq;
                m_tourTrackedActor = static_cast<int>(i);
            }
        }
        if (m_tourTrackedActor >= 0) {
            const SkinnedActor& tracked = m_actors[static_cast<std::size_t>(m_tourTrackedActor)];
            const float weight = std::clamp(
                (eased - kTourActorTrackStart) / (kTourActorTrackFull - kTourActorTrackStart),
                0.0f, 1.0f);
            const float smooth = weight * weight * (3.0f - (2.0f * weight));
            const float target[3] = {
                tracked.position[0],
                tracked.position[1] + conversationFaceHeight(tracked),
                tracked.position[2]};
            // Low-pass the aim point before it reaches the camera.
            //
            // An actor is re-settled onto the terrain EVERY frame (see
            // updateActorWandering) and slid out of walls on top of that, so
            // their position carries per-frame steps that a fixed look-at
            // never had. Pointing straight at it hands that noise to the
            // camera's pitch and yaw, which is the shake at the end of the
            // tour. Time-constant form so it behaves the same at any frame rate.
            constexpr float kAimTimeConstantSeconds = 0.35f;
            const float alpha =
                1.0f - std::exp(-std::max(deltaSeconds, 0.0f) / kAimTimeConstantSeconds);
            if (!m_tourAimValid) {
                m_tourAim[0] = target[0];
                m_tourAim[1] = target[1];
                m_tourAim[2] = target[2];
                m_tourAimValid = true;
            }
            for (int axis = 0; axis < 3; ++axis) {
                m_tourAim[axis] += (target[axis] - m_tourAim[axis]) * alpha;
                lookAt[axis] += (m_tourAim[axis] - lookAt[axis]) * smooth;
            }
        }
    }

    const float dx = lookAt[0] - m_cameraX;
    const float dy = lookAt[1] - m_cameraY;
    const float dz = lookAt[2] - m_cameraZ;
    const float horizontal = std::sqrt((dx * dx) + (dz * dz));
    if (horizontal > 1e-3f) {
        const float desiredYaw = std::atan2(dz, dx) * (180.0f / kPi);
        const float desiredPitch = std::clamp(
            std::atan2(dy, horizontal) * (180.0f / kPi), -kPitchLimitDegrees, kPitchLimitDegrees);

        // Snapping the camera onto the aim direction makes the ANGLES only as
        // smooth as the geometry that produced them, and near the nadir that is
        // not smooth at all: a look-at a few hundred units away and several
        // hundred below turns yaw into a badly conditioned function of the
        // target's horizontal position. Smoothing the aim POINT does not help
        // there, because the point is barely moving -- the angle is what
        // explodes.
        //
        // So the last step of the tour camera is a critically damped filter on
        // the angles themselves, with a hard ceiling on turn rate. Critically
        // damped rather than exponential because it is C1: it eases out of a
        // turn instead of arriving with a corner in angular velocity. The rate
        // ceiling is what bounds a whip if the geometry ever goes bad again --
        // the camera lags the target for a moment and catches up, which reads
        // as a camera operator, not as a glitch.
        constexpr float kAngleSmoothSeconds = 0.25f;
        constexpr float kMaxYawRateDegreesPerSecond = 90.0f;
        constexpr float kMaxPitchRateDegreesPerSecond = 60.0f;
        if (!m_tourAnglesValid) {
            m_yawDegrees = desiredYaw;
            m_pitchDegrees = desiredPitch;
            m_tourYawVelocity = 0.0f;
            m_tourPitchVelocity = 0.0f;
            m_tourAnglesValid = true;
        } else {
            // Shortest arc: the tour crosses +/-180 and chasing the long way
            // round would be a full spin.
            float yawTarget = desiredYaw;
            while (yawTarget - m_yawDegrees > 180.0f) {
                yawTarget -= 360.0f;
            }
            while (yawTarget - m_yawDegrees < -180.0f) {
                yawTarget += 360.0f;
            }
            m_yawDegrees = smoothDampAngle(
                m_yawDegrees, yawTarget, m_tourYawVelocity, kAngleSmoothSeconds,
                kMaxYawRateDegreesPerSecond, deltaSeconds);
            m_pitchDegrees = std::clamp(
                smoothDampAngle(m_pitchDegrees, desiredPitch, m_tourPitchVelocity,
                                kAngleSmoothSeconds, kMaxPitchRateDegreesPerSecond, deltaSeconds),
                -kPitchLimitDegrees, kPitchLimitDegrees);
        }
    }

    // ODAI_FNV_TOUR_TRACE=<path> writes one CSV row per tour frame.
    //
    // "Is the camera smooth" cannot be answered from a screenshot and is only
    // half-answerable by watching -- a kink at a waypoint and a shake in the
    // aim look alike at speed and have different causes. The trace makes both
    // measurable: differentiate position for speed, and yaw/pitch twice for the
    // angular jerk that reads as jitter.
    if (const char* tracePath = std::getenv("ODAI_FNV_TOUR_TRACE")) {
        static std::ofstream s_trace(tracePath);
        static bool s_header = false;
        if (!s_header) {
            s_trace << "raw,eased,x,y,z,yaw,pitch\n";
            s_header = true;
        }
        s_trace << raw << ',' << eased << ',' << m_cameraX << ',' << m_cameraY << ','
                << m_cameraZ << ',' << m_yawDegrees << ',' << m_pitchDegrees << '\n';
    }
    return raw < 1.0f;
}

void NewVegasApp::updateCamera(float deltaSeconds) {
    // The scripted tour owns the camera outright -- no input, no collision, no
    // ground clamp. It flies over rooftops on purpose.
    if (m_flythroughSeconds > 0.0f) {
        updateFlythrough(deltaSeconds);
        return;
    }

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
    const bool inConversation = talkingActor() != nullptr;
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
        // The constants are horizontal; the eased value is vertical. Converting
        // the TARGET each frame (rather than easing in horizontal and
        // converting after) also means a window resize retargets smoothly
        // instead of stepping.
        int fovFramebufferWidth = 0;
        int fovFramebufferHeight = 0;
        framebufferSize(fovFramebufferWidth, fovFramebufferHeight);
        const float aspectRatio =
            (fovFramebufferWidth > 0 && fovFramebufferHeight > 0)
                ? (static_cast<float>(fovFramebufferWidth) /
                   static_cast<float>(fovFramebufferHeight))
                : (16.0f / 9.0f);
        const float targetFov = verticalFovDegreesFor(
            inConversation ? kConversationHorizontalFovDegrees : kDefaultHorizontalFovDegrees,
            aspectRatio);
        const float blend = 1.0f - std::exp(-deltaSeconds / kFovTauSeconds);
        m_cameraFovDegrees += (targetFov - m_cameraFovDegrees) * blend;
    }

    if (const SkinnedActor* speakingActor = inConversation ? talkingActor() : nullptr) {
        // Aim at a fraction of the actor's OWN height rather than a constant.
        // A placement is at the FEET, so aiming at the origin points the camera
        // at the ground; and a bighorner, a settler and a Securitron are not
        // the same height, so one constant cannot frame all three.
        const float faceHeightUnits = conversationFaceHeight(*speakingActor);
        // Time constant, not a per-frame fraction: a fixed fraction converges
        // at whatever rate the machine happens to render at, so the turn would
        // be visibly faster on a fast GPU.
        constexpr float kAimTauSeconds = 0.12f;
        const float dx = speakingActor->position[0] - m_cameraX;
        const float dy = (speakingActor->position[1] + faceHeightUnits) - m_cameraY;
        const float dz = speakingActor->position[2] - m_cameraZ;
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
        const SkinnedActor* speakingActor = talkingActor();
        const bool wantDof = speakingActor != nullptr && !s_noDialogueDof;
        const float easeBlend = 1.0f - std::exp(-deltaSeconds / kDofTauSeconds);
        m_dialogueDofBlend += ((wantDof ? 1.0f : 0.0f) - m_dialogueDofBlend) * easeBlend;

        // Focus on whoever is speaking; with nobody speaking the blend is
        // easing to zero and the distance no longer matters.
        const float faceHeightUnits =
            speakingActor != nullptr ? conversationFaceHeight(*speakingActor) : 0.0f;
        const float dx = (speakingActor != nullptr ? speakingActor->position[0] : m_cameraX) - m_cameraX;
        const float dy =
            ((speakingActor != nullptr ? speakingActor->position[1] : m_cameraY) + faceHeightUnits) -
            m_cameraY;
        const float dz = (speakingActor != nullptr ? speakingActor->position[2] : m_cameraZ) - m_cameraZ;
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
    // A spawn interior nobody asked for is a New Vegas default, so do not hunt
    // for it in somebody else's plugin -- Fallout 3 warned about a missing
    // GSDocMitchellHouse on every launch.
    if (!m_streamSpawnInteriorExplicit && toLowerAscii(m_streamPlugin) != "falloutnv.esm") {
        m_streamSpawnInterior.clear();
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
    // Stream across the whole load order when extra plugins are loaded, so a
    // patch's record fixes -- moved statics, corrected models, replaced terrain
    // -- actually reach the scene. YUP alone ships 18871 reference overrides,
    // none of which did anything while cells came from one plugin.
    //
    // Deliberately skipped when nothing was added: re-indexing seven plugins
    // costs startup time for no override, and the single-plugin path is the one
    // every measurement in this project was taken on.
    if (!m_extraPlugins.empty()) {
        std::vector<std::string> requestedPlugins;
        requestedPlugins.push_back(m_streamPlugin);
        requestedPlugins.insert(
            requestedPlugins.end(), m_extraPlugins.begin(), m_extraPlugins.end());
        importer::fnv::FalloutLoadOrder streamOrder;
        for (const std::string& modDirectory : m_modDirectories) {
            streamOrder.addSearchRoot(std::filesystem::path(modDirectory));
        }
        std::string orderError;
        if (!streamOrder.open(
                std::filesystem::path(m_streamDirectory), requestedPlugins, orderError)) {
            VOX_LOGW("newvegas") << "streaming one plugin only: " << orderError;
        } else {
            VOX_LOGI("newvegas") << "streaming across " << streamOrder.size()
                                 << " plugins (record overrides active)";
            // Kept on the app too: actor discovery needs the same order, and a
            // companion mod's NPC/placement/race/armour all live in its plugin.
            m_streamLoadOrder = streamOrder;
            m_streamer->setLoadOrder(std::move(streamOrder));
        }
    }
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
    // From the plugin, not a constant. Fallout and Oblivion exterior cells are
    // 4096 units square (33 height posts at 128-unit spacing); Morrowind's are
    // 8192 (65 posts at the same spacing). Everything about residency is
    // expressed in cells, so a grid built on the wrong figure loads a quarter
    // of the world it believes it is loading.
    config.cellSize = m_streamer->cellWorldSize();
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
    // START INSIDE A ROOM, which is where New Vegas itself begins -- you wake up
    // on Doc Mitchell's table, not on his porch.
    //
    // Built and uploaded here rather than streamed: an interior is ONE room, not
    // a grid, so there is nothing for the residency planner to plan. It goes in
    // as an ordinary scene chunk and into collision as an ordinary cell, so
    // walls, floors and the ground clamp all work with no interior-specific
    // code anywhere downstream.
    if (!m_startInsideInterior.empty()) {
        importer::ImportedScene interiorScene;
        importer::fnv::CellStreamer::InteriorScene interior;
        std::string interiorError;
        if (!m_streamer->buildInteriorScene(
                m_startInsideInterior, interiorScene, interior, interiorError)) {
            VOX_LOGE("newvegas") << "cannot start inside " << m_startInsideInterior << ": "
                                 << interiorError;
            return false;
        }
        if (m_renderer.addImportedSceneChunk(interiorScene) ==
            render::Renderer::kInvalidImportedChunkIndex) {
            VOX_LOGE("newvegas") << "failed to upload interior " << m_startInsideInterior;
            return false;
        }
        // An interior has its own coordinate space with no grid, so it gets one
        // synthetic cell of its own. The coordinate only has to be consistent
        // and not collide with a streamed exterior cell, and an interior sits
        // nowhere near the worldspace grid to begin with.
        const importer::CellCoord interiorCell{
            static_cast<std::int32_t>(std::floor(interior.spawnPosition[0] / 4096.0f)),
            static_cast<std::int32_t>(std::floor(interior.spawnPosition[2] / 4096.0f))};
        m_collision.addCell(interiorCell, interiorScene);

        if (interior.hasSpawn) {
            m_cameraX = interior.spawnPosition[0];
            m_cameraY = interior.spawnPosition[1] + m_collision.tuning().eyeHeight;
            m_cameraZ = interior.spawnPosition[2];
            m_yawDegrees = interior.spawnYawDegrees;
            m_pitchDegrees = 0.0f;
        }
        // XCLL is read and reported but not yet APPLIED: the renderer has no
        // ambient override to hand it to, so the room is lit by the outdoor rig
        // with the roof shadowing it. Stated here so the gap is visible rather
        // than looking like the values were wrong.
        VOX_LOGI("newvegas") << "started inside " << m_startInsideInterior
                             << (interior.hasLighting
                                     ? " (XCLL read; not applied -- no ambient override yet)"
                                     : " (no XCLL lighting on this cell)");
        m_interiorStarted = true;
    }

    float spawn[3] = {0.0f, 0.0f, 0.0f};
    // Doc Mitchell's doorstep first -- that is where New Vegas actually begins.
    // Fall back to the middle of the worldspace if the cell is missing, so a
    // different plugin or a trimmed install still starts somewhere sensible.
    const bool spawnedAtDoorstep =
        !m_interiorStarted && !m_streamSpawnInterior.empty() &&
        m_streamer->spawnAtInteriorDoorEngineSpace(m_streamSpawnInterior, spawn);
    const bool haveSpawn =
        !m_interiorStarted && (spawnedAtDoorstep || m_streamer->suggestedSpawnEngineSpace(spawn));
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
            // Victor is loaded into a local and appended to m_actors AFTER the
            // town, because loadGoodspringsActors clears the list -- and it has
            // to run second anyway, since excluding his base from the generic
            // scan needs the formID his own placement lookup finds.
            SkinnedActor victor;
            const bool victorLoaded =
                loadVictor(dataPath / m_streamPlugin, m_streamer->assets(), victor,
                           m_victorSpawnPosition[1] != 0.0f ? m_victorSpawnPosition : nullptr);
            if (victorLoaded) {
                // Turn him to face wherever the player starts. His authored
                // ACRE rotation is not used: standing him beside the spawn
                // already overrode his authored POSITION, and a robot facing
                // the direction he faces in a different part of town reads as
                // broken rather than as fidelity.
                victor.yawRadians = std::atan2(
                    m_cameraZ - victor.position[2], m_cameraX - victor.position[0]);
                victor.instanceSlot = kVictorSkinnedInstance;
            }
            VOX_LOGI("newvegas") << "Victor: " << victor.status;
            // The rest of the town, discovered from the plugin around wherever
            // the player actually is rather than from a hardcoded list.
            {
                const float engineCentre[3] = {m_cameraX, m_cameraY, m_cameraZ};
                float bethesdaCentre[3] = {};
                importer::fnv::CellStreamer::engineToFallout(engineCentre, bethesdaCentre);
                const float centreXY[3] = {
                    bethesdaCentre[0], bethesdaCentre[1], bethesdaCentre[2]};
                ActorPopulationStats actorStats;
                loadGoodspringsActors(
                    dataPath / m_streamPlugin,
                    m_streamLoadOrder.empty() ? nullptr : &m_streamLoadOrder,
                    m_streamer->assets(), centreXY, kActorLoadRadius,
                    kFirstCrowdSkinnedInstance,
                    render::kMaxSkinnedInstances - kFirstCrowdSkinnedInstance,
                    {victor.baseFormId}, m_actors, actorStats);
                if (victorLoaded) {
                    m_victorIndex = static_cast<int>(m_actors.size());
                    m_actors.push_back(std::move(victor));
                }
                m_actorsUploadPending = !m_actors.empty();
                // Dialogue for everybody who has any, in one plugin walk. Runs
                // after Victor joins the list so his own tree is left alone and
                // his base is not asked for a second time.
                {
                    std::string dialogueDetail;
                    loadActorDialogue(
                        dataPath / m_streamPlugin,
                        m_streamLoadOrder.empty() ? nullptr : &m_streamLoadOrder, m_actors,
                        dialogueDetail);
                    VOX_LOGI("newvegas") << "actor dialogue: " << dialogueDetail;
                    // AFTER the dialogue: an actor with nothing to say needs no
                    // voice index, and skipping those is most of the town.
                    std::string voiceDetail;
                    loadActorVoices(
                        dataPath, m_streamPlugin, m_modDirectories, m_actors, voiceDetail);
                    VOX_LOGI("newvegas") << "actor voices: " << voiceDetail;
                }
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
                        // The terrain under THIS actor, not under the player.
                        // groundHeightAt resolves against the camera's own foot
                        // height, so over a valley it stood the whole parade in
                        // the air at the player's altitude -- which is what
                        // "the actors are floating in the sky" was.
                        float ground = 0.0f;
                        const bool onGround = m_streamer
                            ? m_collision.terrainHeight(actor.position[0], actor.position[2], ground)
                            : groundHeightAt(actor.position[0], actor.position[2], ground);
                        actor.position[1] =
                            onGround ? ground : (m_cameraY - kEyeHeightUnits);
                        // Facing the camera, so a face that failed to build is
                        // visible as a face and not as the back of a head.
                        actor.yawRadians = std::atan2(forwardX, forwardZ);
                    }
                }
                VOX_LOGI("newvegas") << "actors: " << actorStats.detail;
                for (const SkinnedActor& actor : m_actors) {
                    VOX_LOGI("newvegas")
                        << "  actor " << actor.name << " slot=" << actor.instanceSlot << " at ("
                        << actor.position[0] << ", " << actor.position[1] << ", "
                        << actor.position[2] << ") verts=" << actor.character.vertices.size()
                        << " parts=" << actor.character.parts.size()
                        << " unresolvedBones=" << actor.character.unresolvedBoneCount
                        << " bindConflicts=" << actor.character.conflictingInverseBindCount
                        << " clip=" << (actor.idleClip.tracks.empty() ? "none" : "idle")
                        << (actor.canTalk()
                                ? (" topics=" + std::to_string(actor.tree.nodes.size()))
                                : std::string());
                }
            }
            if (m_victorIndex >= 0) {
                const SkinnedActor& loaded = m_actors[static_cast<std::size_t>(m_victorIndex)];
                VOX_LOGI("newvegas") << "Victor animation: " << loaded.animationStatus;
                VOX_LOGI("newvegas") << "Victor load: " << loaded.timing;
                VOX_LOGI("newvegas") << "Victor voice: " << loaded.voice.status;
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
    loadDistantLandLod();
    return true;
}

// Distant landscape, from the game's own LOD pyramid.
//
// OFF BY DEFAULT, because the two obvious placements are both measured wrong
// and the right one is not built yet. Set ODAI_FNV_LOD_TIER=4|8|16|32 to load
// one tier across the whole worldspace, which is how the numbers below were
// taken and is still the fastest way to look at a tier.
//
// What a whole-world single tier does, on the Mojave:
//
//   level16  64 tiles, 112576 triangles, 64 textures, 40 ms
//   level4   1024 tiles, 1976092 triangles, 1020 textures, 1130 ms
//
// and neither is usable:
//
//  * A COARSE TIER SITS ABOVE THE DETAILED TERRAIN. level16 resamples 16 cells
//    per tile, which averages a valley away, so Goodsprings renders drowned in
//    a smooth tan surface with the road and rooftops poking through it. The
//    error is thousands of units, not tens, so the sink below cannot reach it
//    -- sinking that far would bury the distant mountains it exists to draw.
//  * A FINE TIER EXHAUSTS THE TEXTURE TABLE. Terrain LOD names one diffuse per
//    tile, and the bindless table holds kBindlessTargetTextureCapacity = 1024
//    total. level4's 1020 leaves nothing for the world itself, so EVERY surface
//    in the frame loses its texture and falls back to the hashed pastel that
//    stands in for one. It does not look like a texture-budget failure; it
//    looks like the renderer broke.
//
// So the design this needs is per-tile chunks with tier RINGS -- fine tiles
// just outside the loaded cells, coarser further out, tiles overlapping the
// loaded square excluded -- which bounds both the triangle count and, more
// importantly, the texture count. That is the next step, and the two numbers
// above are the budget it has to fit inside.
//
// Morrowind ships no distant land whatsoever, and Oblivion's is a different
// naming scheme with a single 32-cell tier, so this currently covers FNV and
// Fallout 3 only. An absent tier is not an error here -- it logs and leaves the
// horizon as it was.
void NewVegasApp::loadDistantLandLod() {
    if (m_streamer == nullptr) {
        return;
    }
    // 0 disables, and is the default. Tiers are cell widths: 4, 8, 16, 32.
    int tierCells = 0;
    if (const char* env = std::getenv("ODAI_FNV_LOD_TIER")) {
        tierCells = std::atoi(env);
    }
    if (tierCells <= 0) {
        return;
    }
    if (!importer::fnv::landLodTierExists(importer::fnv::LandLodSet::Terrain, tierCells)) {
        VOX_LOGW("newvegas") << "ODAI_FNV_LOD_TIER=" << tierCells
                             << " is not one of 4, 8, 16, 32; distant LOD disabled";
        return;
    }
    std::int32_t minX = 0;
    std::int32_t minZ = 0;
    std::int32_t maxX = 0;
    std::int32_t maxZ = 0;
    if (!m_streamer->cellGridBounds(minX, minZ, maxX, maxZ)) {
        return;
    }
    // Enough to put the LOD surface under the detailed terrain everywhere the
    // two overlap. Small against a cell (4096) and invisible at the distances
    // this geometry is seen from.
    float sinkUnits = 96.0f;
    if (const char* env = std::getenv("ODAI_FNV_LOD_SINK")) {
        sinkUnits = static_cast<float>(std::atof(env));
    }

    const importer::fnv::FalloutAssetSource& assets = m_streamer->assets();
    importer::ImportedScene scene;
    scene.sourceTag = "fnv_lod";
    importer::fnv::LandLodTierStats stats;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = importer::fnv::appendLandLodTier(
        [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
            return assets.resolveMesh(path, bytes, error);
        },
        [&](const std::string& path, std::vector<std::uint8_t>& bytes) {
            return assets.resolveTexture(path, bytes, error);
        },
        m_streamWorldspace, importer::fnv::LandLodSet::Terrain, tierCells,
        minX, minZ, maxX, maxZ, sinkUnits, scene, stats, error);
    if (!ok) {
        VOX_LOGI("newvegas") << "no distant LOD for " << m_streamWorldspace << ": " << error;
        return;
    }
    importer::buildImportedScenePackedRenderData(scene);
    importer::buildImportedScenePageRanges(scene);
    const std::size_t chunk = m_renderer.addImportedSceneChunk(scene);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (chunk == render::Renderer::kInvalidImportedChunkIndex) {
        VOX_LOGW("newvegas") << "distant LOD upload failed";
        return;
    }
    m_distantLodChunk = chunk;
    VOX_LOGI("newvegas") << "distant LOD level" << tierCells << ": " << stats.tilesParsed
                         << " tiles, " << stats.triangles << " triangles, " << stats.textures
                         << " textures, sink " << sinkUnits << " units, in " << ms << " ms";
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
                             << " droppedLayers=" << stats.droppedTerrainLayers
                             << " waterCells=" << stats.waterPatchesLoaded
                             << " blendedDraws=" << stats.blendedPartsLoaded;
    }
}

void NewVegasApp::onTick(float deltaSeconds) {
    // A recording runs on its own clock. Everything downstream of here -- the
    // tour, the wander, the animation, the day cycle -- takes this dt, so the
    // world advances one authored frame per rendered frame however long the
    // rendering took. See setCaptureSequence.
    if (m_captureFixedDt > 0.0f) {
        deltaSeconds = m_captureFixedDt;
    }
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
    if (!m_menuOpen && m_talkingActor < 0) {
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
    // ONE upload path for every actor, Victor included. He had his own copy of
    // this block until the town arrived, and the two had already drifted apart
    // (his remapped texture slots through a helper, the crowd's inline).
    if (m_actorsUploadPending) {
        m_actorsUploadPending = false;
        std::size_t uploaded = 0;
        std::size_t texturedSlots = 0;
        std::size_t textureSlotCount = 0;
        for (SkinnedActor& actor : m_actors) {
            const std::vector<std::uint32_t> slots =
                m_renderer.uploadSkinnedActorTextures(actor.instanceSlot, actor.textures);
            remapActorTextureSlots(actor, slots);
            for (const std::uint32_t slot : slots) {
                texturedSlots += (slot != 0xffffffffu) ? 1u : 0u;
            }
            textureSlotCount += slots.size();

            render::ImportedSkinnedMeshTemplate meshTemplate{};
            meshTemplate.vertices = actor.character.vertices;
            meshTemplate.indices = actor.character.indices;
            meshTemplate.draws = actor.draws;
            meshTemplate.boneCount =
                static_cast<std::uint32_t>(actor.character.skeleton.bones.size());
            actor.uploaded = m_renderer.uploadSkinnedMeshTemplate(actor.instanceSlot, meshTemplate);
            uploaded += actor.uploaded ? 1u : 0u;
        }
        VOX_LOGI("newvegas") << "actors uploaded: " << uploaded << "/" << m_actors.size() << ", "
                             << texturedSlots << "/" << textureSlotCount << " textures bound";
    }

    // Pose every actor every frame, whether or not it is being talked to -- the
    // idle clip is what makes an actor read as someone standing there rather
    // than a statue of them.
    if (!m_actors.empty()) {
        // Move before posing: the pose folds in world placement, so wandering
        // afterwards would draw everyone one frame behind where they are.
        updateActorWandering(
            m_actors, deltaSeconds,
            [this](float x, float z, float referenceY, float& outHeight) {
                // The ACTOR's own foot height is the reference, not the
                // camera's. groundHeight uses it to reject ceilings and to
                // raise onto walkable geometry, so a settler on a porch stays
                // on the porch instead of sinking to the terrain under it --
                // and, crucially, someone across town is not held at whatever
                // altitude the player happens to be standing at.
                return m_streamer ? m_collision.groundHeight(x, z, referenceY, outHeight) : false;
            },
            [this](float& x, float& z, float feetY, float headY, float radius) {
                if (m_streamer) {
                    m_collision.resolveHorizontalFor(
                        x, z, feetY, headY, radius, m_collision.tuning().stepHeight);
                }
            },
            m_talkingActor);
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

    // ODAI_FNV_TALK opens a conversation on the first tick, so the dialogue UI
    // can be checked from a --screenshot run, which cannot press E. With no
    // value it picks Victor; with an EditorID it picks that actor, which is the
    // only way to photograph anyone else's conversation.
    // ODAI_FNV_VICTOR_TALK is the old spelling and still works.
    {
        const char* autoTalkEnv = std::getenv("ODAI_FNV_TALK");
        if (autoTalkEnv == nullptr) {
            autoTalkEnv = std::getenv("ODAI_FNV_VICTOR_TALK");
        }
        static bool autoTalked = false;
        if (autoTalkEnv != nullptr && !autoTalked && m_talkingActor < 0) {
            autoTalked = true;
            const std::string wanted = toLowerAscii(autoTalkEnv);
            // "1" is the historical value of ODAI_FNV_VICTOR_TALK and names
            // nobody, so it means "whoever the default speaker is".
            const bool wantsDefault = wanted.empty() || wanted == "1";
            for (std::size_t i = 0; i < m_actors.size(); ++i) {
                if (!m_actors[i].canTalk()) {
                    continue;
                }
                if (wantsDefault ? (static_cast<int>(i) == m_victorIndex)
                                 : (toLowerAscii(m_actors[i].name) == wanted)) {
                    beginConversation(static_cast<int>(i));
                    break;
                }
            }
            if (m_talkingActor < 0) {
                VOX_LOGW("newvegas") << "auto-talk: no actor with dialogue matching \""
                                     << autoTalkEnv << "\"";
            }
        }
    }

    SkinnedActor* speaker = talkingActor();

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
        if (!edge || speaker == nullptr) {
            continue;
        }
        const auto choices = speaker->runtime.availableChoices();
        if (static_cast<std::size_t>(slot) < choices.size()) {
            speaker->runtime.choose(*choices[static_cast<std::size_t>(slot)]);
        }
    }
    // Highlight-and-confirm, alongside the number keys rather than instead of
    // them. The numbers are the fast path for someone at a keyboard; up/down
    // and Accept are the only ones that work from a couch, and they come from
    // UiNavInput so a gamepad drives them identically (pollNavInput already
    // folds the d-pad, the left stick and the arrow keys into the same
    // actions, with auto-repeat, so a held direction scrolls instead of
    // jumping one row per frame).
    if (speaker != nullptr) {
        const auto choices = speaker->runtime.availableChoices();
        const auto choiceCount = static_cast<int>(choices.size());
        // A new node means a new set of replies; leaving the old index in
        // place would highlight an unrelated line, or one that no longer
        // exists.
        const dialogue::DialogueNode* currentNode = speaker->runtime.currentNode();
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
                speaker->runtime.choose(*choices[static_cast<std::size_t>(m_dialogueChoice)]);
                m_dialogueChoice = 0;
            }
        } else {
            m_dialogueChoice = 0;
        }
    }
    if (speaker != nullptr &&
        (speaker->runtime.isFinished() || speaker->runtime.currentNode() == nullptr)) {
        endConversation();
        speaker = nullptr;
    }
    // One call site rather than one per way a conversation can advance (a
    // choice, opening it, the auto-talk hook). It is a no-op once the current
    // node has been spoken, so polling it costs a map lookup and cannot start a
    // line twice.
    if (speaker != nullptr && !m_streamCacheDirectory.empty()) {
        speakActorLine(
            *speaker, std::filesystem::path(m_streamCacheDirectory) / "voice", m_audio);
    }

    // ESCAPE LEAVES A CONVERSATION AND NOTHING ELSE. It used to quit the game
    // when no one was speaking, and keyDown is a LEVEL read -- a single press
    // spans ~10 frames at 60 fps. So closing a dialogue box with Escape ended
    // the conversation on the first of those frames and then, on the second
    // frame of the same press, found no speaker and quit. Not an edge case:
    // pressing Escape to back out of a conversation quit the game every time.
    //
    // Edge-latched now, and the quit is gone from here entirely -- backing out
    // of something must never be the same keystroke as leaving the game. With
    // no speaker, Escape falls through to the pause menu, which is what the
    // HUD hint has always claimed it does.
    const bool escapeDown = keyDown(m_window, GLFW_KEY_ESCAPE);
    const bool escapePressed = escapeDown && !m_escapeLatch;
    m_escapeLatch = escapeDown;
    if (escapePressed && speaker != nullptr) {
        endConversation();
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

    // P QUITS. A deliberate, single-purpose key, because the alternative was
    // Escape doing double duty as "close this" and "leave the game" -- and a key
    // that both dismisses a panel and exits has no safe way to be pressed.
    //
    // P was the day-cycle pause toggle, which is still on the pause menu's
    // "Day cycle" row; the [ and ] keys still step time directly.
    const bool quitPressed = keyDown(m_window, GLFW_KEY_P);
    if (quitPressed && !m_quitKeyLatch) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }
    m_quitKeyLatch = quitPressed;

    // Edge-latched: holding E must not re-trigger on the door you arrive next
    // to, which is always within range of the one you just came through.
    // ONE activation target per frame, decided here rather than by which `if`
    // happens to run first. An actor wins over a door at equal reach: Victor
    // stands a step from Doc Mitchell's porch, and a player pressing E while
    // facing him means to talk, not to go inside.
    const float cameraPosition[3] = {m_cameraX, m_cameraY, m_cameraZ};
    m_activationActor = (m_talkingActor >= 0)
        ? -1
        : findActorInReach(m_actors, cameraPosition, m_yawDegrees * (kPi / 180.0f));
    // Latch BEFORE the branch below. It used to be updated after an early
    // return that the Victor path took, so the latch stayed false while E was
    // held: the next frame saw a fresh "press" and walked the player through
    // Doc Mitchell's door -- which is a step from where Victor stands -- so the
    // conversation opened and an interior load closed it in the same keypress.
    const bool doorPressed = keyDown(m_window, GLFW_KEY_E);
    const bool doorEdge = doorPressed && !m_doorKeyLatch;
    m_doorKeyLatch = doorPressed;
    if (doorEdge && m_activationActor >= 0) {
        beginConversation(m_activationActor);
        // The line itself is started by the single speakActorLine poll above,
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
    if (m_nav.pressed(ui::UiNavAction::Menu) && m_talkingActor < 0) {
        // The weather picker is a sub-page of the menu, so Escape backs out of
        // it one level rather than closing everything. Closing straight to the
        // world would make the picker feel like a separate mode the player had
        // fallen into, and there would be no way to change your mind about one
        // weather without leaving the menu entirely.
        if (m_menuOpen && m_weatherPickerOpen) {
            m_weatherPickerOpen = false;
        } else {
            m_menuOpen = !m_menuOpen;
            // Releasing the mouse with the menu up is what makes it usable on
            // PC; on a controller it costs nothing.
            setMouseCaptured(!m_menuOpen);
        }
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
        if (m_victorIndex >= 0 && m_talkingActor < 0) {
            const SkinnedActor& victor = m_actors[static_cast<std::size_t>(m_victorIndex)];
            const float dx = victor.position[0] - m_cameraX;
            const float dz = victor.position[2] - m_cameraZ;
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
    if (const SkinnedActor* speaker = talkingActor()) {
        if (const dialogue::DialogueNode* node = speaker->runtime.currentNode()) {
            drawDialoguePanel(*node, screenWidth, screenHeight, scale);
        }
    } else if (m_activationActor >= 0 &&
               m_activationActor < static_cast<int>(m_actors.size())) {
        const std::string prompt =
            "E  talk to " + m_actors[static_cast<std::size_t>(m_activationActor)].displayName();
        m_uiDrawList.addText(m_uiFont, prompt,
                             ui::UiVec2{64.0f * scale, static_cast<float>(screenHeight) - (132.0f * scale)},
                             kPipGreen);
    }

    // Hint line, top-left. Names the buttons of whichever device is in use --
    // showing "Tab" to someone holding a controller is worse than showing
    // nothing.
    const char* hint = m_navDriving
        ? "(Start) menu   (LS) move   (A) use"
        : "Esc menu   [ ] time   P quit   Tab cursor";
    m_uiDrawList.addText(m_uiFont, hint, ui::UiVec2{margin, margin}, kPipGreenDim);
}

void NewVegasApp::buildWeatherChoices() {
    if (!m_weatherChoices.empty()) {
        return;
    }
    // Every weather the load order defines, not just the ones this worldspace's
    // climate runs. Scoping to the climate was the first attempt and it is too
    // narrow to be useful: NVDefaultClimate names exactly TWO, so the picker
    // offered clear-day and clear-night out of the 63 vanilla ships. The point
    // of the picker is looking at skies, and the list scrolls and pages, so
    // there is nothing to be gained by hiding most of them.
    m_weatherChoices.reserve(m_weatherTables.weathers.size());
    for (const auto& [formId, record] : m_weatherTables.weathers) {
        m_weatherChoices.push_back(formId);
    }
    // By name, because the player is reading names. formID order is load-order
    // order, which shuffles when a plugin is added and is meaningless on screen.
    std::sort(
        m_weatherChoices.begin(), m_weatherChoices.end(),
        [this](std::uint32_t a, std::uint32_t b) {
            const auto* ra = m_weatherTables.findWeather(a);
            const auto* rb = m_weatherTables.findWeather(b);
            const std::string& na = ra != nullptr ? ra->editorId : std::string{};
            const std::string& nb = rb != nullptr ? rb->editorId : std::string{};
            return na < nb;
        });
    m_weatherChoices.erase(
        std::unique(m_weatherChoices.begin(), m_weatherChoices.end()), m_weatherChoices.end());
    VOX_LOGI("newvegas") << "weather picker: " << m_weatherChoices.size() << " choices";
}

void NewVegasApp::openWeatherPicker() {
    m_weatherPickerOpen = true;
    buildWeatherChoices();
    // Open ON the active weather rather than at the top. The list is sorted by
    // name, so row 1 is alphabetical happenstance -- with vanilla's 63 that is a
    // Pitt DLC weather, nowhere near whatever is currently over the Mojave --
    // and scrolling back to where you already were is the first thing the player
    // would otherwise have to do.
    const auto found =
        std::find(m_weatherChoices.begin(), m_weatherChoices.end(), m_activeWeatherFormId);
    const int activeIndex = (found != m_weatherChoices.end())
        ? static_cast<int>(found - m_weatherChoices.begin())
        : 0;
    // A few rows of context above it rather than pinned to the top edge.
    m_weatherScrollTop = std::max(0, activeIndex - 4);
    m_weatherFocus.setFocus(std::min(activeIndex, 4));
}

bool NewVegasApp::drawWeatherPicker(const ui::UiRect& panelArea, float scale) {
    buildWeatherChoices();
    if (m_weatherChoices.empty()) {
        m_weatherPickerOpen = false;
        return false;
    }

    const int choiceCount = static_cast<int>(m_weatherChoices.size());
    // A fixed window of rows, sized to what fits comfortably rather than to the
    // list -- the list can be 473 long.
    constexpr int kVisibleRows = 10;
    const int visibleRows = std::min(kVisibleRows, choiceCount);
    const float lineHeight = m_uiFont.lineHeightPx();
    const float rowHeight = lineHeight + (10.0f * scale);
    const float headerBand = lineHeight + (28.0f * scale);
    const float footerBand = lineHeight + (22.0f * scale);

    float contentWidth = m_uiFont.measureText("WEATHER");
    for (int i = 0; i < choiceCount; ++i) {
        const auto* record = m_weatherTables.findWeather(m_weatherChoices[i]);
        if (record != nullptr) {
            contentWidth = std::max(contentWidth, m_uiFont.measureText(record->editorId.c_str()));
        }
    }
    const float panelWidth = std::max(560.0f * scale, contentWidth + (96.0f * scale));
    const float panelHeight =
        headerBand + (rowHeight * static_cast<float>(visibleRows)) + footerBand;
    ui::UiRect panel{};
    panel.minX = ((panelArea.minX + panelArea.maxX) - panelWidth) * 0.5f;
    panel.maxX = panel.minX + panelWidth;
    panel.minY = ((panelArea.minY + panelArea.maxY) - panelHeight) * 0.5f;
    panel.maxY = panel.minY + panelHeight;
    m_uiDrawList.addRoundRectFilled(panel, kPipPanelSolid, 4.0f * scale);
    m_uiDrawList.addRoundRect(panel, kPipGreen, 4.0f * scale, 1.5f * scale);

    char header[64];
    std::snprintf(header, sizeof(header), "WEATHER  (%d)", choiceCount);
    m_uiDrawList.addText(
        m_uiFont, header,
        ui::UiVec2{panel.minX + (24.0f * scale), panel.minY + (14.0f * scale)}, kPipGreen);

    // The focus ring holds only the VISIBLE rows, so navigating off either end
    // has to scroll the window rather than move focus. Registering all 473 would
    // let focus land on a row that is not drawn, and the highlight would simply
    // vanish.
    m_weatherFocus.beginFrame();
    std::vector<ui::UiRect> rows(static_cast<std::size_t>(visibleRows));
    for (int i = 0; i < visibleRows; ++i) {
        ui::UiRect row{};
        row.minX = panel.minX + (16.0f * scale);
        row.maxX = panel.maxX - (16.0f * scale);
        row.minY = panel.minY + headerBand + (static_cast<float>(i) * rowHeight);
        row.maxY = row.minY + rowHeight - (3.0f * scale);
        rows[static_cast<std::size_t>(i)] = row;
        m_weatherFocus.addItem(row);
    }
    if (!m_navDriving) {
        double cursorX = 0.0;
        double cursorY = 0.0;
        glfwGetCursorPos(m_window, &cursorX, &cursorY);
        m_weatherFocus.focusHovered(
            ui::UiVec2{static_cast<float>(cursorX), static_cast<float>(cursorY)});
    }

    // Scroll BEFORE navigating, so a press at the edge of the window moves the
    // list by one instead of being swallowed by the focus ring's own clamp.
    const int maxScroll = std::max(0, choiceCount - visibleRows);
    const int focusedRow = std::max(0, m_weatherFocus.focused());
    if (m_nav.pressed(ui::UiNavAction::Down) && focusedRow == visibleRows - 1 &&
        m_weatherScrollTop < maxScroll) {
        ++m_weatherScrollTop;
    } else if (m_nav.pressed(ui::UiNavAction::Up) && focusedRow == 0 && m_weatherScrollTop > 0) {
        --m_weatherScrollTop;
    } else {
        m_weatherFocus.applyNavigation(m_nav);
    }
    // Shoulder buttons page. 473 entries one row at a time is a minute of
    // holding a stick; this makes the far end of the list reachable.
    if (m_nav.pressed(ui::UiNavAction::NextTab)) {
        m_weatherScrollTop = std::min(maxScroll, m_weatherScrollTop + visibleRows);
    }
    if (m_nav.pressed(ui::UiNavAction::PrevTab)) {
        m_weatherScrollTop = std::max(0, m_weatherScrollTop - visibleRows);
    }
    m_weatherScrollTop = std::clamp(m_weatherScrollTop, 0, maxScroll);

    for (int i = 0; i < visibleRows; ++i) {
        const int choiceIndex = m_weatherScrollTop + i;
        if (choiceIndex >= choiceCount) {
            break;
        }
        const std::uint32_t formId = m_weatherChoices[static_cast<std::size_t>(choiceIndex)];
        const auto* record = m_weatherTables.findWeather(formId);
        const bool focused = m_weatherFocus.isFocused(i);
        const bool isActive = (formId == m_activeWeatherFormId);
        if (focused) {
            m_uiDrawList.addRoundRectFilled(
                rows[static_cast<std::size_t>(i)], ui::UiColor{0.16f, 0.42f, 0.20f, 0.85f},
                3.0f * scale);
        }
        // The active weather is marked as well as highlighted: focus and
        // "currently applied" are different things, and on this palette a single
        // green cue for both is unreadable.
        const std::string label =
            std::string{isActive ? "> " : "  "} + (record != nullptr ? record->editorId : "<?>");
        m_uiDrawList.addText(
            m_uiFont, label.c_str(),
            ui::UiVec2{
                rows[static_cast<std::size_t>(i)].minX + (16.0f * scale),
                rows[static_cast<std::size_t>(i)].minY + (5.0f * scale)},
            focused ? kPipGreen : kPipGreenDim);
    }

    if (m_nav.pressed(ui::UiNavAction::Accept)) {
        const int choiceIndex = m_weatherScrollTop + focusedRow;
        if (choiceIndex >= 0 && choiceIndex < choiceCount) {
            selectWeather(m_weatherChoices[static_cast<std::size_t>(choiceIndex)]);
        }
    }

    char footer[96];
    std::snprintf(
        footer, sizeof(footer), "%s    %d-%d of %d",
        m_navDriving ? "(A) apply   (LB/RB) page   (B) back"
                     : "Enter apply   Q/E page   Esc back",
        m_weatherScrollTop + 1, std::min(choiceCount, m_weatherScrollTop + visibleRows),
        choiceCount);
    m_uiDrawList.addText(
        m_uiFont, footer,
        ui::UiVec2{panel.minX + (24.0f * scale), panel.maxY - footerBand + (12.0f * scale)},
        kPipGreenDim);
    return true;
}

void NewVegasApp::drawPauseMenu() {
    if (!m_menuOpen) {
        // Keep the ring empty so a stale focus index cannot survive a close and
        // reopen and act on the wrong entry.
        m_menuFocus.beginFrame();
        m_weatherFocus.beginFrame();
        m_weatherPickerOpen = false;
        return;
    }
    const float scale = contentScale();
    int screenWidth = 0;
    int screenHeight = 0;
    framebufferSize(screenWidth, screenHeight);

    // Dim the world so the menu is unambiguously modal.
    ui::UiRect full{0.0f, 0.0f, static_cast<float>(screenWidth), static_cast<float>(screenHeight)};
    m_uiDrawList.addRectFilled(full, ui::UiColor{0.0f, 0.02f, 0.0f, 0.55f});

    // The picker REPLACES the menu rather than layering over it: two focus rings
    // reading the same nav input would both move, and backing out would land on
    // whichever row the hidden one had drifted to.
    if (m_weatherPickerOpen) {
        m_menuFocus.beginFrame();
        if (drawWeatherPicker(full, scale)) {
            return;
        }
    }

    struct Entry {
        const char* label;
        const char* value;
    };
    char timeValue[32];
    std::snprintf(timeValue, sizeof(timeValue), "%s", m_dayCyclePaused ? "Paused" : "Running");
    char regionValue[32];
    std::snprintf(regionValue, sizeof(regionValue), "%zu", m_discoveredRegions.size());
    const importer::fnv::FalloutWeatherRecord* activeWeather =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    const std::string weatherValue =
        activeWeather != nullptr ? activeWeather->editorId : std::string{"<none>"};
    const Entry entries[] = {
        {m_walkMode ? "Movement: On Foot" : "Movement: Fly", ""},
        {"Day cycle", timeValue},
        {"Weather", weatherValue.c_str()},
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
            case 2: openWeatherPicker(); break;
            case 3: break;  // a readout, not an action
            case 4: m_menuOpen = false; setMouseCaptured(true); break;
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
    const SkinnedActor* speakingActor = talkingActor();
    const auto choices = speakingActor != nullptr
        ? speakingActor->runtime.availableChoices()
        : decltype(speakingActor->runtime.availableChoices()){};
    const std::size_t choiceCount = std::min<std::size_t>(choices.size(), 9u);
    std::vector<std::vector<std::string>> choiceLines;
    std::vector<float> choiceHeights;
    choiceLines.reserve(choiceCount);
    choiceHeights.reserve(choiceCount);
    for (std::size_t i = 0; i < choiceCount; ++i) {
        choiceLines.push_back(
            wrapTextToWidth(choiceFont, choices[i]->text, innerWidth - choiceIndent));
        const float rows = static_cast<float>(std::max<std::size_t>(choiceLines.back().size(), 1u));
        choiceHeights.push_back((rows * choiceLineHeight) + (choiceRowPadding * 2.0f));
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

    // THE CARD IS CAPPED, AND THE REPLIES SCROLL INSIDE IT.
    //
    // It used to grow with the reply count and stay centred, so Easy Pete's nine
    // replies made a card spanning nearly the whole screen -- and the camera,
    // which frames the speaker's face just above the card's top edge, had
    // nowhere left to put him. He was completely hidden behind his own
    // dialogue. A conversation must not hide the person talking, which is the
    // same rule the pitch offset exists to serve; the offset simply cannot
    // honour it once the card has eaten the frame.
    //
    // 0.62 leaves the top third of the screen clear, which at this card's width
    // is enough for a head and shoulders at conversation distance.
    const float fixedHeight = (padding * 2.0f) + speakerHeight + (12.0f * scale) + spokenHeight +
                              ruleGap + footerHeight;
    const float choiceBudget = std::max(0.0f, (height * 0.62f) - fixedHeight);

    // The window of replies that fits, slid just far enough to keep the
    // highlighted one inside it. Sliding by one rather than paging keeps the
    // list still under the cursor for as long as possible.
    const auto fitFrom = [&](std::size_t start) {
        float used = 0.0f;
        std::size_t shown = 0;
        for (std::size_t i = start; i < choiceCount; ++i) {
            if (shown > 0u && (used + choiceHeights[i]) > choiceBudget) {
                break;
            }
            used += choiceHeights[i];
            ++shown;
        }
        return shown;
    };
    std::size_t firstChoice = 0;
    std::size_t visibleChoices = fitFrom(0);
    const auto selected = static_cast<std::size_t>(std::max(m_dialogueChoice, 0));
    while (selected >= (firstChoice + visibleChoices) && (firstChoice + visibleChoices) < choiceCount) {
        ++firstChoice;
        visibleChoices = fitFrom(firstChoice);
    }
    float choicesHeight = 0.0f;
    for (std::size_t i = firstChoice; i < firstChoice + visibleChoices; ++i) {
        choicesHeight += choiceHeights[i];
    }
    const bool choicesClipped = visibleChoices < choiceCount;

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
    std::string speaker = node.speaker;
    if (speaker.empty()) {
        speaker = speakingActor != nullptr ? speakingActor->displayName() : std::string("?");
    }
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
    for (std::size_t i = firstChoice; i < firstChoice + visibleChoices; ++i) {
        const float rowHeight = choiceHeights[i];
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

    // The count is stated when the list is clipped, because a reply the player
    // cannot see is a reply they do not know to scroll to -- and the numbers on
    // the visible rows are the TRUE indices, so "7." appearing first is only
    // legible next to "of 9".
    std::string footer = choiceCount == 0
        ? std::string("Esc  end conversation")
        : std::string("Up/Down  select     Enter  choose     Esc  leave");
    if (choicesClipped) {
        footer = "Up/Down  select (" + std::to_string(firstChoice + 1u) + "-" +
            std::to_string(firstChoice + visibleChoices) + " of " + std::to_string(choiceCount) +
            ")     Enter  choose     Esc  leave";
    }
    m_uiDrawList.addText(
        m_uiFont, footer,
        ui::UiVec2{panel.minX + ((panelWidth - m_uiFont.measureText(footer)) * 0.5f),
                   panel.maxY - padding - m_uiFont.lineHeightPx() + (10.0f * scale)},
        kPipGreenDim);
}

void NewVegasApp::drawHud() {
    // ODAI_FNV_NOHUD=1 draws the world and nothing else. A screenshot meant to
    // show the RENDERER has its own subject, and the Pip-Boy chrome, the key
    // hints and the debug readouts all sit on top of it.
    static const bool s_noHud = [] {
        const char* env = std::getenv("ODAI_FNV_NOHUD");
        return env != nullptr && env[0] != '0';
    }();
    if (s_noHud) {
        // The software cursor is not part of the HUD -- GameApp draws it after
        // this returns -- and it is drawn wherever the desktop happened to
        // leave the pointer, so it lands in the corner of the capture.
        setCursorVisible(false);
        return;
    }
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
    if (!m_menuOpen && m_talkingActor < 0) {
        const ui::Font& bannerFont = m_uiFontDisplay.valid() ? m_uiFontDisplay : m_uiFont;
        m_banner.draw(m_uiDrawList, bannerFont, m_uiFont, screen, contentScale());
    }
}

void NewVegasApp::updateDebugStats() {
    // Gated on the panel actually being up: this formats a couple of dozen
    // strings and it would otherwise do that every frame for nobody.
    if (!m_renderer.isDebugUiVisible()) {
        return;
    }
    auto number = [](auto value, int decimals = 0) {
        char text[48];
        if (decimals > 0) {
            std::snprintf(text, sizeof(text), "%.*f", decimals, static_cast<double>(value));
        } else {
            std::snprintf(text, sizeof(text), "%lld", static_cast<long long>(value));
        }
        return std::string{text};
    };

    std::vector<render::DebugStatGroup> groups;

    if (m_streamer != nullptr) {
        const importer::fnv::CellStreamerStats stats = m_streamer->stats();
        render::DebugStatGroup streaming{"Cell Streaming", {}};
        streaming.rows.push_back({"Resident cells", number(stats.residentChunks)});
        streaming.rows.push_back({"Loading", number(stats.residency.loadingCount)});
        streaming.rows.push_back({"Loaded / evicted",
            number(stats.scenesLoaded) + " / " + number(stats.residency.evictions)});
        // Wasted loads are the honest cost of prediction: a cell read to
        // completion and then thrown away because the player turned. A number
        // that climbs with distance travelled means the lead time is too long.
        streaming.rows.push_back({"Wasted / unavailable",
            number(stats.residency.wastedLoads) + " / " + number(stats.residency.unavailableCells)});
        streaming.rows.push_back({"Failed loads", number(stats.loadFailures)});
        streaming.rows.push_back({"Empty cells", number(stats.emptyScenes)});
        streaming.rows.push_back({"", ""});
        // Apply is main-thread time; build is worker time. Only the first is
        // felt as a hitch, which is why they are reported separately.
        streaming.rows.push_back({"Apply ms (last / worst)",
            number(stats.lastApplyMs, 2) + " / " + number(stats.worstApplyMs, 2)});
        streaming.rows.push_back({"Build ms (last / worst)",
            number(stats.lastBuildMs, 2) + " / " + number(stats.worstBuildMs, 2)});
        streaming.rows.push_back({"Cell cache hit / miss",
            number(stats.cacheHits) + " / " + number(stats.cacheMisses)});
        streaming.rows.push_back({"Cache load ms", number(stats.lastCacheLoadMs, 2)});
        if (stats.cacheWriteFailures > 0) {
            streaming.rows.push_back({"Cache write failures", number(stats.cacheWriteFailures)});
        }
        groups.push_back(std::move(streaming));
    }

    render::DebugStatGroup world{"World", {}};
    world.rows.push_back({"Camera",
        number(m_cameraX, 0) + ", " + number(m_cameraY, 0) + ", " + number(m_cameraZ, 0)});
    world.rows.push_back({"Cell",
        number(std::floor(m_cameraX / 4096.0f)) + ", " + number(std::floor(m_cameraZ / 4096.0f))});
    world.rows.push_back({"Hour", number(m_timeOfDayHours, 2)});
    const importer::fnv::FalloutWeatherRecord* weather =
        m_activeWeatherFormId != 0u ? m_weatherTables.findWeather(m_activeWeatherFormId) : nullptr;
    world.rows.push_back({"Weather", weather != nullptr ? weather->editorId : "<none>"});
    world.rows.push_back({"Actors", number(m_actors.size())});
    world.rows.push_back({"Regions discovered", number(m_discoveredRegions.size())});
    groups.push_back(std::move(world));

    m_renderer.setDebugStatGroups(std::move(groups));
}

bool NewVegasApp::captureWarmupComplete() const {
    if (m_framesRendered <= m_captureWarmupFrames) {
        return false;
    }
    if (m_framesRendered >= m_captureWarmupFrameCeiling) {
        return true;
    }
    // Streaming is wall-clock work on other threads, so a frame count cannot
    // stand in for it. This mattered the moment frame capture got 28x faster:
    // the same 60 warm-up frames went from over a minute of wall time to about
    // a second, and the opening of every capture became a half-built town.
    return m_streamer == nullptr || m_streamer->isStreamingIdle();
}

void NewVegasApp::onRender(float /*deltaSeconds*/) {
    // Before beginFrameDraw: the backend consumes the pending pose while
    // recording this frame, so setting it afterwards would always be a frame
    // late -- invisible on a still bind pose, and a lag on an animated one.
    if (m_characterMode) {
        updateCharacterPose();
    }
    updateDebugStats();
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
        // A FRAME COUNT IS NOT A STREAMING WAIT, and this path was still
        // counting frames alone long after the video path stopped. On an empty
        // scene a frame costs almost nothing, so several hundred of them elapse
        // in about a second while the cells are still arriving -- and the shot
        // comes out as bare sky over the skybox's ground colour, which reads as
        // "the geometry stopped rendering" rather than "the shot was early".
        // Two hours went into the wrong half of that sentence.
        //
        // Same ceiling as the video path, so a worldspace that never settles
        // still produces a file instead of hanging.
        const bool settled = m_framesRendered >= m_captureWarmupFrameCeiling ||
            m_streamer == nullptr || m_streamer->isStreamingIdle();
        if (m_framesRendered >= m_screenshotWarmupFrames && settled) {
            if (!m_renderer.captureFrameToFile(m_screenshotPath)) {
                VOX_LOGE("newvegas") << "screenshot capture failed";
            }
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        }
    }

    // Frame-sequence recording, straight into an encoder. Same cadence and same
    // warm-up as the stills path below, but the frames never land on disk.
    if (!m_captureVideoPath.empty() && m_captureWritten < m_captureFrames) {
        ++m_framesRendered;
        if (captureWarmupComplete()) {
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            bool ok = m_renderer.captureFrameRgb(m_captureRgb, width, height);
            if (ok && !m_captureVideo.isOpen()) {
                // Opened on the FIRST captured frame rather than up front,
                // because the swapchain extent is what it is: ODAI_WINDOW_SIZE
                // is a request the window manager is free to ignore, and ffmpeg
                // needs the real number baked into its input description.
                ok = m_captureVideo.open(m_captureVideoPath, width, height,
                                         static_cast<int>(m_captureVideoFps + 0.5f));
            }
            if (ok) {
                ok = m_captureVideo.writeFrame(m_captureRgb);
            }
            if (!ok) {
                VOX_LOGE("newvegas") << "video capture failed at frame " << m_captureWritten;
                m_captureVideo.close();
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
                return;
            }
            ++m_captureWritten;
            if ((m_captureWritten % 120) == 0) {
                VOX_LOGI("newvegas")
                    << "captured " << m_captureWritten << "/" << m_captureFrames << " frames";
            }
            if (m_captureWritten >= m_captureFrames) {
                // Closed HERE, not in the destructor: pclose is where a failed
                // encode reports itself, and an hour of rendering should not
                // discover that during teardown.
                m_captureVideo.close();
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            }
        }
    }

    // The older stills path. One PPM per frame, numbered, for ffmpeg to stitch
    // afterwards. Kept because a still sequence is what a frame-by-frame
    // comparison wants -- but for a recording, prefer --capture-video above.
    if (!m_captureDirectory.empty() && m_captureWritten < m_captureFrames) {
        ++m_framesRendered;
        if (captureWarmupComplete()) {
            char leaf[32] = {};
            std::snprintf(leaf, sizeof(leaf), "/frame_%05d.ppm", m_captureWritten);
            if (!m_renderer.captureFrameToFile(m_captureDirectory + leaf)) {
                VOX_LOGE("newvegas") << "sequence capture failed at frame " << m_captureWritten;
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            }
            ++m_captureWritten;
            // Progress, because a 900-frame capture is minutes of silence
            // otherwise and a stalled one looks identical to a slow one.
            if ((m_captureWritten % 60) == 0) {
                VOX_LOGI("newvegas")
                    << "captured " << m_captureWritten << "/" << m_captureFrames << " frames";
            }
            if (m_captureWritten >= m_captureFrames) {
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            }
        }
    }
}

}  // namespace odai::games::newvegas
