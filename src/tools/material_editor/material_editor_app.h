#pragma once

// Live PBR material editor.
//
// Exists because there was no way to see a material change without an edit,
// rebuild and relaunch: coefficients were C++ literals compiled into the procgen
// generators. This puts an actual 3-D surface next to a slider.
//
// Why a new tool rather than extending an existing one: src/tools/ui_editor is
// the only thing in the tree called "editor", but it sets
// wantsMinimalRendering() -> true, which makes the renderer skip creating the
// importedStatic pipeline entirely. It is a 2-D .ui.json layout editor and
// structurally cannot display a lit surface. A fresh GameApp gets the 3-D
// pipelines by default and the src/ui widget set for free.
//
// The editing loop deliberately never calls uploadImportedScene(): a slider
// writes one 32-byte record through Renderer::setImportedMaterial, which lands
// in the next frame's descriptor region. That is the whole reason the material
// table is a storage buffer rather than vertex data.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/file_watch.h"
#include "engine/game_app.h"
#include "import/imported_scene.h"
#include "ui/ui_types.h"

namespace odai::ui {
class Dropdown;
class Label;
class Slider;
}  // namespace odai::ui

namespace odai::tools::materialeditor {

class MaterialEditorApp : public engine::GameApp {
public:
    // A scene path from argv; empty means build the procedural preview instead,
    // so the tool is useful with no cooked assets on disk.
    void setScenePath(std::string path) { m_scenePath = std::move(path); }

protected:
    bool onInit() override;
    void onTick(float dt) override;
    void onRender(float dt) override;

private:
    // ── Scene ────────────────────────────────────────────────────────────────
    void buildPreviewScene();          // procgen buildings on a ground pad
    bool loadSceneFromPath();          // cooked .bin
    void uploadScene();                // the ONE upload; never called from an edit

    // ── Camera ───────────────────────────────────────────────────────────────
    [[nodiscard]] render::CameraPose cameraPose() const;
    void handleCamera(float dt);
    [[nodiscard]] bool pickAtCursor(std::uint32_t& outMaterialIndex) const;

    // ── Material library ─────────────────────────────────────────────────────
    void loadLibrary();
    void reloadLibraryFromDisk();      // hot reload: matches by name, never renumbers
    void saveLibrary();
    void selectMaterial(std::uint32_t index);
    void pushEdit();                   // applies m_materials[m_selected] to the GPU
    void syncWidgetsFromSelection();   // slider positions <- selected material

    // ── UI ───────────────────────────────────────────────────────────────────
    void buildUi();
    void layoutUi();
    void drawStatusOverlay();

    std::string m_scenePath;
    importer::ImportedScene m_scene;
    bool m_sceneReady = false;

    std::vector<importer::ImportedSceneMaterial> m_materials;  // 1-based; [0] is the sentinel
    std::uint32_t m_selected = 1;
    // Undo is a whole-library snapshot per edit, the same shape ui_editor uses:
    // the library is a few hundred bytes, so anything finer would be effort
    // spent on a non-problem.
    std::vector<std::vector<importer::ImportedSceneMaterial>> m_undo;
    static constexpr std::size_t kMaxUndo = 128;

    core::FileWatch m_watch;
    std::string m_status;
    float m_statusTimer = 0.0f;

    // Orbit camera around the scene's centre.
    float m_camYawDeg = 45.0f;
    float m_camPitchDeg = -22.0f;
    float m_camDistance = 6.0f;
    math::Vector3 m_camFocus{0.0f, 0.5f, 0.0f};

    // Widgets are owned by the UiContext root; these are borrowed pointers used
    // to push values in when the selection changes.
    ui::Dropdown* m_materialList = nullptr;
    ui::Slider* m_metallic = nullptr;
    ui::Slider* m_roughness = nullptr;
    ui::Slider* m_baseR = nullptr;
    ui::Slider* m_baseG = nullptr;
    ui::Slider* m_baseB = nullptr;
    ui::Slider* m_emissive = nullptr;
    ui::Label* m_readout = nullptr;
    // True while syncWidgetsFromSelection() is writing: the slider onChange
    // callbacks fire on assignment, and without this the sync would immediately
    // write the values straight back and stamp an undo entry per selection.
    bool m_syncing = false;
};

}  // namespace odai::tools::materialeditor
