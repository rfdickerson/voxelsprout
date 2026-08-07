#include "tools/material_editor/material_editor_app.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>

#include <GLFW/glfw3.h>

#include "content/material_library.h"
#include "core/log.h"
#include "import/imported_material.h"
#include "import/imported_scene_query.h"
#include "math/math.h"
#include "procgen/building_generator.h"
#include "procgen/mesh_emit.h"
#include "ui/ui_draw_list.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/dropdown.h"
#include "ui/widgets/slider.h"

namespace odai::tools::materialeditor {

using ui::UiColor;
using ui::UiRect;

namespace {

constexpr const char* kLibraryPath = "assets/materials/library.json";

// Panel metrics. Plain constants rather than a theme: this is a tool, and the
// values only have to be consistent with each other.
constexpr float kPanelW = 300.0f;
constexpr float kPad = 14.0f;
constexpr float kRowH = 26.0f;
constexpr float kRowGap = 10.0f;
constexpr float kLabelH = 18.0f;

const UiColor kPanelBg{0.10f, 0.11f, 0.13f, 0.96f};
const UiColor kText{0.88f, 0.89f, 0.91f, 1.0f};
const UiColor kTextDim{0.58f, 0.60f, 0.64f, 1.0f};

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Scene
// ─────────────────────────────────────────────────────────────────────────────

// A few buildings of each era on a ground pad. Built rather than loaded so the
// tool is useful in a fresh checkout with no cooked .bin anywhere, and because
// procgen buildings are half the point: their glazing and trim are exactly the
// surfaces whose coefficients were previously C++ literals.
void MaterialEditorApp::buildPreviewScene() {
    m_scene = importer::ImportedScene{};
    m_scene.sourceTag = "material_preview";

    // Ground pad: a large flat quad, deliberately left at material index 0 so
    // there is always an unlit-by-the-library surface to compare against.
    {
        procgen::TriMesh ground;
        const float half = 6.0f;
        const float y = 0.0f;
        const float xs[4] = {-half, half, half, -half};
        const float zs[4] = {-half, -half, half, half};
        for (int i = 0; i < 4; ++i) {
            importer::ImportedScenePackedVertex v{};
            v.position[0] = xs[i];
            v.position[1] = y;
            v.position[2] = zs[i];
            v.normal[1] = 1.0f;
            v.color[0] = 0.26f;
            v.color[1] = 0.28f;
            v.color[2] = 0.25f;
            ground.vertices.push_back(v);
        }
        for (const std::uint32_t o : {0u, 1u, 2u, 0u, 2u, 3u}) ground.indices.push_back(o);
        procgen::appendTriMesh(ground, math::Vector3{0.0f, 0.0f, 0.0f},
                               procgen::Color3{1.0f, 1.0f, 1.0f}, m_scene);
    }

    const procgen::Era eras[3] = {procgen::Era::E1890s, procgen::Era::E1930s,
                                  procgen::Era::E1960s};
    const procgen::BuildingKind kinds[3] = {procgen::BuildingKind::Residential,
                                            procgen::BuildingKind::Commercial,
                                            procgen::BuildingKind::Industrial};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            procgen::BuildingDesc desc;
            desc.era = eras[row];
            desc.kind = kinds[col];
            desc.level = row + 1;
            desc.wealthTier = col;
            desc.lotWidth = 1.5f;
            desc.lotDepth = 1.5f;
            desc.detail = 1;
            desc.seed = static_cast<std::uint32_t>((row * 3 + col + 1) * 0x9E3779B9u);
            // The whole point: facade glazing and trim resolve through the
            // library, so the sliders in this tool move them.
            desc.glassMaterial =
                content::findMaterialIndex(m_materials, "facade_glass");
            desc.mullionMaterial =
                content::findMaterialIndex(m_materials, "facade_mullion");
            desc.wallMaterial = content::findMaterialIndex(m_materials, "building_wall");
            desc.roofMaterial = content::findMaterialIndex(m_materials, "building_roof");
            const procgen::TriMesh mesh = procgen::generateBuilding(desc);
            procgen::appendTriMesh(
                mesh,
                math::Vector3{static_cast<float>(col - 1) * 2.6f, 0.0f,
                              static_cast<float>(row - 1) * 2.6f},
                procgen::Color3{1.0f, 1.0f, 1.0f}, m_scene);
        }
    }

    // appendTriMesh fills packedVertices/packedIndices but does NOT create a
    // draw or maintain scene bounds -- the caller owns both (citybuilder does
    // the same at the end of buildCityScene). Without the draw the renderer has
    // nothing to submit and the scene silently never appears.
    if (!m_scene.packedIndices.empty()) {
        m_scene.packedDraws.push_back(importer::ImportedScenePackedDraw{
            0u, static_cast<std::uint32_t>(m_scene.packedIndices.size())});
    }
    for (int axis = 0; axis < 3; ++axis) {
        m_scene.boundsMin[axis] = std::numeric_limits<float>::max();
        m_scene.boundsMax[axis] = std::numeric_limits<float>::lowest();
    }
    for (const importer::ImportedScenePackedVertex& v : m_scene.packedVertices) {
        for (int axis = 0; axis < 3; ++axis) {
            m_scene.boundsMin[axis] = std::min(m_scene.boundsMin[axis], v.position[axis]);
            m_scene.boundsMax[axis] = std::max(m_scene.boundsMax[axis], v.position[axis]);
        }
    }
    importer::buildImportedScenePageRanges(m_scene);
    // Count how much geometry actually resolves through the library. Zero here
    // would mean the slots never reached the vertices and every slider in this
    // tool would be inert -- worth stating rather than discovering by eye.
    std::size_t libraryVerts = 0;
    for (const importer::ImportedScenePackedVertex& v : m_scene.packedVertices) {
        if (importer::importedSceneMaterialIndex(v.flags) != 0u) ++libraryVerts;
    }
    VOX_LOGI("materialeditor") << "preview scene: " << m_scene.packedVertices.size()
                               << " vertices, " << m_scene.packedDraws.size() << " draw(s), "
                               << libraryVerts << " vertices bound to library materials";
}

bool MaterialEditorApp::loadSceneFromPath() {
    if (!importer::loadImportedScene(std::filesystem::path(m_scenePath), m_scene)) {
        VOX_LOGE("materialeditor") << "failed to load scene '" << m_scenePath
                                   << "': " << importer::getImportedSceneLastError();
        return false;
    }
    VOX_LOGI("materialeditor") << "loaded " << m_scenePath << " ("
                               << m_scene.packedVertices.size() << " vertices, "
                               << m_scene.materials.size() << " materials)";
    // A cooked scene carries its own library; prefer it over the JSON so the
    // editor is editing what the file actually references.
    if (m_scene.materials.size() > 1) {
        m_materials = m_scene.materials;
    }
    return true;
}

void MaterialEditorApp::uploadScene() {
    m_renderer.uploadImportedScene(m_scene);
    m_sceneReady = true;
    // Frame the scene: sit back far enough to see its whole footprint.
    const float spanX = m_scene.boundsMax[0] - m_scene.boundsMin[0];
    const float spanZ = m_scene.boundsMax[2] - m_scene.boundsMin[2];
    const float span = std::max({spanX, spanZ, 1.0f});
    m_camFocus = math::Vector3{(m_scene.boundsMin[0] + m_scene.boundsMax[0]) * 0.5f,
                               (m_scene.boundsMin[1] + m_scene.boundsMax[1]) * 0.5f,
                               (m_scene.boundsMin[2] + m_scene.boundsMax[2]) * 0.5f};
    m_camDistance = span * 1.4f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────

render::CameraPose MaterialEditorApp::cameraPose() const {
    const float yaw = math::radians(m_camYawDeg);
    const float pitch = math::radians(m_camPitchDeg);
    const float cp = std::cos(pitch);
    // The camera looks along +forward, so it sits at focus - forward * distance.
    const math::Vector3 forward{std::cos(yaw) * cp, std::sin(pitch), std::sin(yaw) * cp};
    const math::Vector3 eye = m_camFocus - (forward * m_camDistance);
    render::CameraPose pose{};
    pose.x = eye.x;
    pose.y = eye.y;
    pose.z = eye.z;
    pose.yawDegrees = m_camYawDeg;
    pose.pitchDegrees = m_camPitchDeg;
    pose.fovDegrees = 45.0f;
    return pose;
}

void MaterialEditorApp::handleCamera(float dt) {
    (void)dt;
    // Right-drag orbits, wheel dollies. Left is reserved for picking.
    if (m_uiInput.button(ui::UiMouseButton::Right).down) {
        m_camYawDeg += m_uiInput.mouseDeltaPx.x * 0.4f;
        m_camPitchDeg =
            std::clamp(m_camPitchDeg - m_uiInput.mouseDeltaPx.y * 0.3f, -85.0f, 85.0f);
    }
    if (m_uiInput.scrollDelta != 0.0f) {
        m_camDistance = std::clamp(m_camDistance * (1.0f - m_uiInput.scrollDelta * 0.12f),
                                   0.6f, 200.0f);
    }
}

// Builds a world ray through the cursor and returns the material index of the
// nearest surface it hits. Reuses importer::raycastImportedScene, the same
// function App's `I` key debug inspect now shares.
bool MaterialEditorApp::pickAtCursor(std::uint32_t& outMaterialIndex) const {
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    if (fbW <= 0 || fbH <= 0 || !m_sceneReady) {
        return false;
    }
    const render::CameraPose pose = cameraPose();
    const float yaw = math::radians(pose.yawDegrees);
    const float pitch = math::radians(pose.pitchDegrees);
    const float cp = std::cos(pitch);
    const math::Vector3 forward{std::cos(yaw) * cp, std::sin(pitch), std::sin(yaw) * cp};
    const math::Vector3 worldUp{0.0f, 1.0f, 0.0f};
    const math::Vector3 right = math::normalize(math::cross(forward, worldUp));
    const math::Vector3 up = math::cross(right, forward);

    // Cursor -> NDC -> a direction on the view frustum.
    const float ndcX = (m_uiInput.mousePx.x / static_cast<float>(fbW)) * 2.0f - 1.0f;
    const float ndcY = 1.0f - (m_uiInput.mousePx.y / static_cast<float>(fbH)) * 2.0f;
    const float aspect = static_cast<float>(fbW) / static_cast<float>(fbH);
    const float tanHalf = std::tan(math::radians(pose.fovDegrees) * 0.5f);
    const math::Vector3 dir = math::normalize(forward + (right * (ndcX * tanHalf * aspect)) +
                                              (up * (ndcY * tanHalf)));

    const math::Ray ray{math::Vector3{pose.x, pose.y, pose.z}, dir};
    const importer::ImportedSceneRayHit hit =
        importer::raycastImportedScene(m_scene, ray, 1000.0f);
    if (!hit.hit) {
        return false;
    }
    outMaterialIndex = hit.materialIndex;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Material library
// ─────────────────────────────────────────────────────────────────────────────

void MaterialEditorApp::loadLibrary() {
    const auto result = content::loadMaterialLibraryFromFile(resolveAssetPath(kLibraryPath));
    if (!result.ok()) {
        for (const std::string& e : result.errors) {
            VOX_LOGW("materialeditor") << "material library: " << e;
        }
    }
    m_materials = result.materials;
    if (m_materials.size() <= 1) {
        // Nothing on disk: start from a single editable entry rather than an
        // empty tool with no rows.
        m_materials.resize(2);
        m_materials[1].name = "material_1";
    }
}

// Hot reload. Matches by NAME and keeps slot numbers, because the index is
// baked into vertex flag bits that cannot change without re-extruding the
// scene -- renumbering would silently repaint every surface. A name that is not
// already a slot is ignored rather than appended, for the same reason.
void MaterialEditorApp::reloadLibraryFromDisk() {
    const auto result = content::loadMaterialLibraryFromFile(resolveAssetPath(kLibraryPath));
    if (!result.ok()) {
        m_status = "Library error: " + result.errors.front();
        m_statusTimer = 6.0f;
        VOX_LOGW("materialeditor") << "material library: " << result.errors.front();
        if (result.materials.size() <= 1) {
            return;  // keep the last good values
        }
    }
    int applied = 0;
    for (std::size_t slot = 1; slot < m_materials.size(); ++slot) {
        const std::uint32_t found =
            content::findMaterialIndex(result.materials, m_materials[slot].name);
        if (found == 0u) continue;
        const std::string name = m_materials[slot].name;
        m_materials[slot] = result.materials[found];
        m_materials[slot].name = name;
        m_renderer.setImportedMaterial(static_cast<std::uint32_t>(slot), m_materials[slot]);
        ++applied;
    }
    if (applied > 0) {
        m_status = "Reloaded " + std::to_string(applied) + " material(s) from disk";
        m_statusTimer = 3.0f;
        syncWidgetsFromSelection();
        VOX_LOGI("materialeditor") << "hot reload applied " << applied << " material(s)";
    }
}

void MaterialEditorApp::saveLibrary() {
    const std::filesystem::path path = resolveAssetPath(kLibraryPath);
    std::ofstream out(path);
    if (!out) {
        m_status = "Save failed: cannot open " + path.string();
        m_statusTimer = 6.0f;
        return;
    }
    out << content::materialLibraryToJson(m_materials) << '\n';
    m_status = "Saved " + path.string();
    m_statusTimer = 3.0f;
}

void MaterialEditorApp::selectMaterial(std::uint32_t index) {
    if (index == 0u || index >= m_materials.size()) return;
    m_selected = index;
    syncWidgetsFromSelection();
}

void MaterialEditorApp::pushEdit() {
    if (m_selected == 0u || m_selected >= m_materials.size()) return;
    // The whole edit path: one 32-byte record, no geometry, no upload.
    m_renderer.setImportedMaterial(m_selected, m_materials[m_selected]);
}

void MaterialEditorApp::syncWidgetsFromSelection() {
    if (m_selected >= m_materials.size()) return;
    const importer::ImportedSceneMaterial& m = m_materials[m_selected];
    // Assigning .value fires onChange on some widgets; guard so the sync does
    // not immediately write the same values back and stamp an undo entry.
    m_syncing = true;
    if (m_metallic) m_metallic->value = m.metallic;
    if (m_roughness) m_roughness->value = m.roughness;
    if (m_baseR) m_baseR->value = std::clamp(m.baseColorTint[0], 0.0f, 1.0f);
    if (m_baseG) m_baseG->value = std::clamp(m.baseColorTint[1], 0.0f, 1.0f);
    if (m_baseB) m_baseB->value = std::clamp(m.baseColorTint[2], 0.0f, 1.0f);
    if (m_emissive) m_emissive->value = std::clamp(m.emissiveStrength / 8.0f, 0.0f, 1.0f);
    if (m_materialList) m_materialList->selectedIndex = static_cast<int>(m_selected) - 1;
    m_syncing = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// UI
// ─────────────────────────────────────────────────────────────────────────────

void MaterialEditorApp::buildUi() {
    auto root = std::make_unique<ui::Panel>();
    root->background = UiColor{0.0f, 0.0f, 0.0f, 0.0f};  // the 3-D frame shows through
    ui::Widget* rootPtr = m_uiContext.setRoot(std::move(root));

    auto panel = std::make_unique<ui::Panel>();
    panel->background = kPanelBg;
    ui::Widget* panelPtr = rootPtr->addChild(std::move(panel));

    // Slider factory: every coefficient here is naturally [0,1], which is the
    // only range ui::Slider speaks, so no remapping is needed except emissive
    // strength (scaled by 8 so the useful range is reachable).
    const auto addSlider = [&](ui::Slider*& out, std::function<void(float)> apply) {
        auto slider = std::make_unique<ui::Slider>();
        slider->onChange = [this, apply = std::move(apply)](float v) {
            if (m_syncing) return;
            if (m_selected == 0u || m_selected >= m_materials.size()) return;
            // Snapshot before the first change of a drag would be ideal; per
            // change is simpler and the library is tiny.
            if (m_undo.size() >= kMaxUndo) m_undo.erase(m_undo.begin());
            m_undo.push_back(m_materials);
            apply(v);
            pushEdit();
        };
        out = slider.get();
        panelPtr->addChild(std::move(slider));
    };

    auto list = std::make_unique<ui::Dropdown>(&m_uiFont);
    list->onSelect = [this](int i) { selectMaterial(static_cast<std::uint32_t>(i) + 1u); };
    m_materialList = list.get();
    // zOrder is 1 on Dropdown and its popup draws outside the child tree, so it
    // must be added last to sit above the rows. Added first here and re-parented
    // by draw order would clip it.
    panelPtr->addChild(std::move(list));

    addSlider(m_metallic, [this](float v) { m_materials[m_selected].metallic = v; });
    addSlider(m_roughness, [this](float v) { m_materials[m_selected].roughness = v; });
    addSlider(m_baseR, [this](float v) { m_materials[m_selected].baseColorTint[0] = v; });
    addSlider(m_baseG, [this](float v) { m_materials[m_selected].baseColorTint[1] = v; });
    addSlider(m_baseB, [this](float v) { m_materials[m_selected].baseColorTint[2] = v; });
    addSlider(m_emissive, [this](float v) { m_materials[m_selected].emissiveStrength = v * 8.0f; });

    auto readout = std::make_unique<ui::Label>(m_uiFonts, "");
    m_readout = readout.get();
    panelPtr->addChild(std::move(readout));

    // Populate the dropdown from the library.
    m_materialList->items.clear();
    for (std::size_t i = 1; i < m_materials.size(); ++i) {
        m_materialList->items.push_back(m_materials[i].name);
    }
    layoutUi();
    syncWidgetsFromSelection();
}

void MaterialEditorApp::layoutUi() {
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    const float s = contentScale();
    const float w = kPanelW * s;
    const float x = static_cast<float>(fbW) - w;
    ui::Widget* root = m_uiContext.root();
    if (root == nullptr || root->children().empty()) return;
    root->setRect(UiRect{0.0f, 0.0f, static_cast<float>(fbW), static_cast<float>(fbH)});
    ui::Widget* panel = root->children().front().get();
    panel->setRect(UiRect{x, 0.0f, static_cast<float>(fbW), static_cast<float>(fbH)});

    float y = kPad * s;
    const auto row = [&](ui::Widget* wdg, float height) {
        if (wdg == nullptr) return;
        wdg->setRect(UiRect{x + kPad * s, y, x + w - kPad * s, y + height});
        y += height + kRowGap * s;
    };
    y += kLabelH * s;  // room for the "Material" caption drawn in the overlay
    row(m_materialList, kRowH * s);
    for (ui::Slider* sl : {m_metallic, m_roughness, m_baseR, m_baseG, m_baseB, m_emissive}) {
        y += kLabelH * s;  // caption drawn in the overlay above each slider
        row(sl, kRowH * s * 0.7f);
    }
    y += kLabelH * s;
    row(m_readout, kLabelH * s * 3.0f);
}

// Captions, the swatch and the status line are drawn immediately rather than as
// widgets: they are non-interactive, and a Label per caption would be six more
// borrowed pointers to keep in sync for no benefit.
void MaterialEditorApp::drawStatusOverlay() {
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    const float s = contentScale();
    const float w = kPanelW * s;
    const float x = static_cast<float>(fbW) - w;

    const auto text = [&](const char* str, float px, float py, const UiColor& c) {
        m_uiDrawList.addText(m_uiFont, str, ui::UiVec2{px, py}, c);
    };

    float y = kPad * s;
    text("MATERIAL", x + kPad * s, y, kTextDim);
    y += kLabelH * s + kRowH * s + kRowGap * s;

    const char* names[6] = {"Metallic", "Roughness", "Base R", "Base G", "Base B",
                            "Emissive strength"};
    ui::Slider* sliders[6] = {m_metallic, m_roughness, m_baseR, m_baseG, m_baseB, m_emissive};
    char buf[96];
    for (int i = 0; i < 6; ++i) {
        const float v = sliders[i] != nullptr ? sliders[i]->value : 0.0f;
        std::snprintf(buf, sizeof(buf), "%s  %.2f", names[i],
                      i == 5 ? v * 8.0f : v);
        text(buf, x + kPad * s, y, kText);
        y += kLabelH * s + (kRowH * s * 0.7f) + kRowGap * s;
    }

    // Base-color swatch: three sliders are honest for v1, but a swatch is what
    // makes them readable as a colour rather than three numbers.
    if (m_selected < m_materials.size()) {
        const auto& m = m_materials[m_selected];
        const UiRect sw{x + kPad * s, y, x + kPad * s + 48.0f * s, y + 24.0f * s};
        m_uiDrawList.addRectFilled(sw, UiColor{m.baseColorTint[0], m.baseColorTint[1],
                                               m.baseColorTint[2], 1.0f});
        text("base colour", sw.maxX + 8.0f * s, y + 6.0f * s, kTextDim);
        y += 34.0f * s;
    }

    text("LMB pick   RMB orbit   wheel zoom   Ctrl+S save   Ctrl+Z undo",
         x + kPad * s, static_cast<float>(fbH) - 40.0f * s, kTextDim);
    if (m_statusTimer > 0.0f && !m_status.empty()) {
        text(m_status.c_str(), x + kPad * s, static_cast<float>(fbH) - 22.0f * s, kText);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

bool MaterialEditorApp::onInit() {
    const float s = contentScale();
    if (!loadFonts(resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
                   resolveAssetPath("assets/fonts/Inter-Bold.ttf"),
                   resolveAssetPath("assets/fonts/Inter-Italic.ttf"),
                   resolveAssetPath("assets/fonts/Inter-Regular.ttf"),
                   std::round(15.0f * s), std::round(15.0f * s))) {
        return false;
    }

    loadLibrary();
    m_renderer.setImportedMaterialTable(m_materials);

    if (!m_scenePath.empty() && loadSceneFromPath()) {
        m_renderer.setImportedMaterialTable(m_materials);
    } else {
        if (!m_scenePath.empty()) {
            VOX_LOGW("materialeditor") << "falling back to the procedural preview scene";
        }
        buildPreviewScene();
    }
    uploadScene();
    buildUi();

    m_watch.watch(resolveAssetPath(kLibraryPath), [this] { reloadLibraryFromDisk(); });
    VOX_LOGI("materialeditor") << "ready: " << (m_materials.size() - 1) << " material(s)";
    return true;
}

void MaterialEditorApp::onTick(float dt) {
    m_watch.tick();
    if (m_statusTimer > 0.0f) m_statusTimer -= dt;
    handleCamera(dt);
    layoutUi();

    const bool ctrl = glfwGetKey(m_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(m_window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    static bool prevS = false, prevZ = false;
    const bool sDown = glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS;
    const bool zDown = glfwGetKey(m_window, GLFW_KEY_Z) == GLFW_PRESS;
    if (ctrl && sDown && !prevS) saveLibrary();
    if (ctrl && zDown && !prevZ && !m_undo.empty()) {
        m_materials = m_undo.back();
        m_undo.pop_back();
        m_renderer.setImportedMaterialTable(m_materials);
        syncWidgetsFromSelection();
        m_status = "Undo";
        m_statusTimer = 2.0f;
    }
    prevS = sDown;
    prevZ = zDown;

    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }

    // Click to select the material under the cursor. Skipped over the panel so
    // dragging a slider never re-picks.
    int fbW = 0, fbH = 0;
    framebufferSize(fbW, fbH);
    const float panelX = static_cast<float>(fbW) - kPanelW * contentScale();
    if (m_uiInput.button(ui::UiMouseButton::Left).pressed && m_uiInput.mousePx.x < panelX) {
        std::uint32_t picked = 0;
        if (pickAtCursor(picked)) {
            if (picked == 0u) {
                m_status = "That surface has no library material";
                m_statusTimer = 2.5f;
            } else {
                selectMaterial(picked);
                m_status = "Selected " + m_materials[m_selected].name;
                m_statusTimer = 2.0f;
            }
        }
    }
}

void MaterialEditorApp::onRender(float /*dt*/) {
    beginFrameDraw();
    drawStatusOverlay();
    submitFrame(cameraPose());
}

}  // namespace odai::tools::materialeditor
