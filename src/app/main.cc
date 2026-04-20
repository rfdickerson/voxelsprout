#include "app/app.h"

#include "core/log.h"

#ifndef ODAI_APP_VERSION
#define ODAI_APP_VERSION "dev"
#endif

#ifndef ODAI_RELEASE_PROFILE
#define ODAI_RELEASE_PROFILE "dev_runtime"
#endif

// Program entry point
// Responsible for: creating the app and handing control to the app runtime loop.
// Should NOT do: implement gameplay systems or low-level window/input handling.
int main() {
    VOX_LOGI("main") << "startup"
                     << " version=" << ODAI_APP_VERSION
                     << " profile=" << ODAI_RELEASE_PROFILE;
    odai::app::App app;

    if (!app.init()) {
        VOX_LOGE("main") << "app init failed, exiting";
        return 1;
    }

    app.run();
    app.shutdown();
    VOX_LOGI("main") << "exit success";
    return 0;
}
