#include "tools/ui_editor/ui_editor_app.h"

int main() {
    odai::tools::ui_editor::UiEditorApp app;
    if (!app.init("odai :: UI Editor")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
