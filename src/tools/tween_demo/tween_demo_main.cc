#include "tools/tween_demo/tween_demo_app.h"

int main() {
    odai::tools::tween_demo::TweenDemoApp app;
    if (!app.init("odai :: Tween Demo")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
