#include "games/newvegas/newvegas_app.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
    // MSAA off by default for this game: TAA (taa.comp.slang) does the
    // anti-aliasing work now, MSAA 4x measured ~1.5 ms of main-pass GPU time
    // on the target iGPU, and MSAA cannot fix the two artifacts that actually
    // show here (texture shimmer, alpha-test cutout crawl) anyway. setenv with
    // overwrite=0, so an explicit ODAI_MSAA from the user still wins.
    setenv("ODAI_MSAA", "1", 0);
    // Internal render scale 0.6 with native-resolution UI/tonemap composite,
    // and shadows to 3500 units instead of 6000. Together with TAA, cascade
    // interleaving and the halved shadow atlas these take the target iGPU from
    // 20 ms to ~10 ms of GPU per frame; each is an env the user can override
    // (ODAI_RENDER_SCALE=1 restores native-resolution rendering).
    setenv("ODAI_RENDER_SCALE", "0.6", 0);
    setenv("ODAI_SHADOW_DISTANCE", "3500", 0);
    odai::games::newvegas::NewVegasApp app;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
            app.setScenePath(argv[++i]);
        } else if (std::strcmp(argv[i], "--stream") == 0 && i + 1 < argc) {
            // Stream straight from the game's own Data directory -- no cooking.
            app.setStreamDataPath(argv[++i]);
        } else if (std::strcmp(argv[i], "--plugin") == 0 && i + 1 < argc) {
            app.setStreamPlugin(argv[++i]);
        } else if (std::strcmp(argv[i], "--worldspace") == 0 && i + 1 < argc) {
            app.setStreamWorldspace(argv[++i]);
        } else if (std::strcmp(argv[i], "--cache") == 0 && i + 1 < argc) {
            app.setStreamCacheDirectory(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-cache") == 0) {
            app.setStreamCacheEnabled(false);
        } else if (std::strcmp(argv[i], "--spawn") == 0 && i + 1 < argc) {
            // Interior cell whose doorstep to start on, e.g. GSDocMitchellHouse.
            app.setStreamSpawnInterior(argv[++i]);
        } else if (std::strcmp(argv[i], "--character") == 0) {
            // Optional: a skeleton path, then any number of body-part paths,
            // all relative to Data\Meshes. No arguments means the default male
            // body, which is the case worth having be zero-effort.
            std::string skeletonPath;
            std::vector<std::string> partPaths;
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                if (skeletonPath.empty()) {
                    skeletonPath = argv[++i];
                } else {
                    partPaths.emplace_back(argv[++i]);
                }
            }
            app.setCharacterMode(std::move(skeletonPath), std::move(partPaths));
        } else if (std::strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc) {
            const std::string path = argv[++i];
            // Optional frame count after the path, for scenes that need longer
            // to settle than the default warm-up.
            int warmupFrames = 8;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                warmupFrames = std::atoi(argv[++i]);
                if (warmupFrames < 1) {
                    warmupFrames = 1;
                }
            }
            app.setScreenshotRequest(path, warmupFrames);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "odai_game_newvegas [--scene <path.bin>]\n"
                      << "  Falls back to $ODAI_FNV_SCENE when --scene is absent.\n"
                      << "odai_game_newvegas --screenshot <out.ppm> [frames]\n"
                      << "  Render `frames` frames (default 8), write a PPM, and quit.\n"
                      << "  Cook a scene first with odai_newvegas_cooker.\n"
                      << "odai_game_newvegas --character [<skeleton.nif> <part.nif>...]\n"
                      << "  Stand one GPU-skinned character in bind pose, no world.\n"
                      << "  Defaults to characters\\_male\\skeleton.nif + upperbody.nif.\n";
            return 0;
        }
    }
    if (!app.init("New Vegas")) {
        return 1;
    }
    app.run();
    app.shutdown();
    return 0;
}
