#include "games/citybuilder/citybuilder_app.h"

int main() {
    odai::games::citybuilder::CityBuilderApp app;
    if (!app.init("OdaiCity — City Builder")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
