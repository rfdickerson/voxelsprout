#include "tools/retro_theme_demo/retro_theme_demo_app.h"

int main() {
    odai::tools::retro_theme_demo::RetroDemoApp app;
    if (!app.init("odai :: Retro Theme Demo")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
