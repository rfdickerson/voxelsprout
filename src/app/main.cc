#include "app/app.h"

#include "core/log.h"

// Program entry point
// Responsible for: creating the app and handing control to the app runtime loop.
// Should NOT do: implement gameplay systems or low-level window/input handling.
int main() {
    VOX_LOGI("main") << "startup";
    voxelsprout::app::App app;

    if (!app.init()) {
        VOX_LOGE("main") << "app init failed, exiting";
        return 1;
    }

    app.run();
    app.shutdown();
    VOX_LOGI("main") << "exit success";
    return 0;
}
