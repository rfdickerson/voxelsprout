#include "games/voxelcraft/voxelcraft_app.h"

int main() {
    odai::games::voxelcraft::VoxelCraftApp app;
    if (!app.init("VoxelCraft")) return 1;
    app.run();
    app.shutdown();
    return 0;
}
