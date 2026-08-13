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
    // Render at native resolution. This used to default to 0.6, which drew the
    // 3D scene at 36% of the pixels and upscaled it into a native-resolution UI
    // composite -- visibly soft, and the first thing anyone notices.
    //
    // Measured on the LNL iGPU (RelWithDebInfo, scripted camera, 1920x1080
    // swapchain), 0.6 -> 1.0 costs ~3.5 ms of GPU: 8.8 ms/frame -> 12.3 ms,
    // about 81 fps. Only two passes scale with it -- main (3.6 -> 5.7) and SSAO
    // (0.46 -> 1.5). Shadow (~1.3) and prepass (~2.3) do not move at all,
    // because they are vertex-bound rather than fill-bound, which is why
    // dropping resolution bought so much less than it looked like it should.
    //
    // ODAI_RENDER_SCALE still dials it back; 0.6 is worth reaching for on a 4K
    // swapchain, where the fill-bound half of that budget quadruples.
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
        } else if (std::strcmp(argv[i], "--plugin-add") == 0 && i + 1 < argc) {
            // An extra plugin loaded after --plugin; masters resolve on their
            // own, so "--plugin-add NevadaSkies.esp" is enough.
            app.addPlugin(argv[++i]);
        } else if (std::strcmp(argv[i], "--weather") == 0 && i + 1 < argc) {
            app.setWeather(argv[++i]);
        } else if (std::strcmp(argv[i], "--mod") == 0 && i + 1 < argc) {
            // A directory laid out like Data itself (textures\..., meshes\...).
            // Repeatable; later ones win, as a mod manager's load order would.
            app.addModDirectory(argv[++i]);
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
        } else if (std::strcmp(argv[i], "--flythrough") == 0) {
            // Scripted tour of Goodsprings. Optional length in seconds.
            float seconds = 40.0f;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                seconds = static_cast<float>(std::atof(argv[++i]));
            }
            app.setFlythroughSeconds(seconds > 1.0f ? seconds : 40.0f);
        } else if (std::strcmp(argv[i], "--capture-seq") == 0 && i + 1 < argc) {
            // <dir> [fps] [seconds]. Frames are numbered PPMs for ffmpeg.
            const std::string directory = argv[++i];
            float fps = 30.0f;
            float seconds = 40.0f;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                fps = static_cast<float>(std::atof(argv[++i]));
            }
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                seconds = static_cast<float>(std::atof(argv[++i]));
            }
            if (fps < 1.0f) {
                fps = 30.0f;
            }
            app.setCaptureSequence(directory, static_cast<int>(fps * seconds), fps);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "odai_game_newvegas [--scene <path.bin>]\n"
                      << "  Falls back to $ODAI_FNV_SCENE when --scene is absent.\n"
                      << "odai_game_newvegas --screenshot <out.ppm> [frames]\n"
                      << "  Render `frames` frames (default 8), write a PPM, and quit.\n"
                      << "  Cook a scene first with odai_newvegas_cooker.\n"
                      << "odai_game_newvegas --flythrough [seconds] --capture-seq <dir> [fps] [seconds]\n"
                      << "  Fly the Goodsprings tour and write numbered PPMs, then quit.\n"
                      << "  Stitch them with: ffmpeg -framerate <fps> -i <dir>/frame_%05d.ppm out.mp4\n"
                      << "odai_game_newvegas --character [<skeleton.nif> <part.nif>...]\n"
                      << "  Stand one GPU-skinned character in bind pose, no world.\n"
                      << "  Defaults to characters\\_male\\skeleton.nif + upperbody.nif.\n"
                      << "odai_game_newvegas --stream <Data> --mod <dir> [--mod <dir>...]\n"
                      << "  Override game assets from directories laid out like Data\n"
                      << "  (textures\\..., meshes\\...); later --mod wins. Also\n"
                      << "  $ODAI_FNV_MODS, ':'-separated.\n"
                      << "  A texture pack needs $ODAI_FNV_TEX_SIZE raised too: the\n"
                      << "  default clamps every texture to 512 px, so higher-resolution\n"
                      << "  art is mip-dropped away before it is ever seen.\n"
                      << "odai_game_newvegas --stream <Data> --plugin-add <Mod.esp>\n"
                      << "  Load an extra plugin for its weather records; masters resolve\n"
                      << "  on their own. The .esp must be in the --stream directory, not\n"
                      << "  in a --mod directory. Also $ODAI_FNV_PLUGINS, ','-separated.\n"
                      << "  --weather <EditorID> forces one weather by name.\n"
                      << "\n"
                      << "See docs/FNV_MODS.md for install + launch recipes.\n";
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
