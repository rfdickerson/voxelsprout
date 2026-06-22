#include "games/swtor/swtor_app.h"

int main() {
    odai::games::swtor::SwtorApp app;
    if (!app.init("Star Wars: The Old Republic — UI Framework")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
