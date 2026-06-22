#include "games/stellaris/stellaris_app.h"

int main() {
    odai::games::stellaris::StellarisApp app;
    if (!app.init("Stellaris Clone — Galactic Frontier")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
