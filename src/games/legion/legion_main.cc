#include "games/legion/legion_app.h"

int main() {
    odai::games::legion::LegionApp app;
    if (!app.init("Legion")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
