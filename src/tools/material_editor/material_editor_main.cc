#include <cstring>
#include <iostream>

#include "tools/material_editor/material_editor_app.h"

int main(int argc, char** argv) {
    odai::tools::materialeditor::MaterialEditorApp app;

    // --scene <path.bin> edits a cooked scene's own material table; with no
    // argument the tool builds a procedural preview so it works in a fresh
    // checkout with nothing cooked.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
            app.setScenePath(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "odai_material_editor [--scene <path.bin>]\n"
                         "  LMB pick surface   RMB orbit   wheel zoom\n"
                         "  Ctrl+S save assets/materials/library.json   Ctrl+Z undo\n"
                         "  The library hot-reloads: edit the JSON with this running.\n";
            return 0;
        }
    }

    if (!app.init("odai material editor")) {
        return 1;
    }
    app.run();
    app.shutdown();
    return 0;
}
