#include "app/app.h"

#include "core/log.h"

int main() {
    VOX_LOGI("main") << "startup";

    voxelsprout::app::App app;
    if (!app.init()) {
        VOX_LOGE("main") << "app initialization failed";
        return 1;
    }

    app.run();
    app.shutdown();
    VOX_LOGI("main") << "shutdown";
    return 0;
}
