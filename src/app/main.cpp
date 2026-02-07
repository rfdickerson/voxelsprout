#include "app/App.hpp"

#include <iostream>

// Program entry point
// Responsible for: creating the app and handing control to the app runtime loop.
// Should NOT do: implement gameplay systems or low-level window/input handling.
int main() {
    std::cerr << "[main] startup\n";
    app::App app;

    if (!app.init()) {
        std::cerr << "[main] app init failed, exiting\n";
        return 1;
    }

    app.run();
    app.shutdown();
    std::cerr << "[main] exit success\n";
    return 0;
}
