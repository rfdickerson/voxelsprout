#include "games/minesweeper/minesweeper_app.h"

int main() {
    odai::games::minesweeper::MinesweeperApp app;
    if (!app.init("Minesweeper")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
