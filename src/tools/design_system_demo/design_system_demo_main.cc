#include "tools/design_system_demo/design_system_demo_app.h"

int main() {
    odai::tools::design_system_demo::DesignSystemDemoApp app;
    if (!app.init("odai :: Design System Demo")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
