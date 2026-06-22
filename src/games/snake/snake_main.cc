#include "games/snake/snake_app.h"

int main() {
    odai::games::snake::SnakeApp app;
    if (!app.init("Snake")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
