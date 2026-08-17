#include "games/newvegas/newvegas_app.h"

#include "render/upscale/upscale_policy.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>

namespace {

std::filesystem::path defaultShaderPackWaterNormalPath() {
    std::filesystem::path dataHome;
    if (const char* xdgDataHome = std::getenv("XDG_DATA_HOME")) {
        if (xdgDataHome[0] != '\0') {
            dataHome = xdgDataHome;
        }
    }
    if (dataHome.empty()) {
        if (const char* userHome = std::getenv("HOME")) {
            if (userHome[0] != '\0') {
                dataHome = std::filesystem::path{userHome} / ".local" / "share";
            }
        }
    }
    return dataHome / "odai" / "morrowind" / "shader-packs" /
        "enhanced-pbr-2.0e" / "Enhanced PBR Lighting for OpenMW 0.49-0.52" /
        "water_nm.png";
}

}  // namespace

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
    // NO SHADOW DISTANCE OVERRIDE. There used to be a
    // setenv("ODAI_SHADOW_DISTANCE", "3500", 0) here, sitting under the render
    // scale comment above with no comment of its own, and it silently beat
    // every default the renderer chose. 3500 units is about 50 metres: shadows
    // worked close and midrange and then simply stopped, on every worldspace of
    // every game this viewer opens.
    //
    // It also made the renderer's own cascade tuning unmeasurable from here. An
    // A/B of ODAI_SHADOW_DISTANCE looked like it worked -- because passing the
    // variable explicitly overrode this line -- while changing the DEFAULT in
    // frame_run.cc appeared to do nothing at all. Two experiments that disagree
    // like that are a strong hint that something upstream is pinning the value.
    //
    // The renderer now caps at its own far plane; see frame_run.cc.
    odai::games::newvegas::NewVegasApp app;
    // TAA ON, AT NATIVE RESOLUTION, BY DEFAULT.
    //
    // setTaaEnabled(true) was doing nothing: recordTaaPass returns early with no
    // upscaler object, and the default backend is Off, so every run of this
    // viewer since TAA landed has rendered with the pass costing 0.000 ms in the
    // GPU timings. The give-away was there in every capture and reads as an
    // unused feature rather than a broken one.
    //
    // It matters most for AO. XeGTAO's whole design puts its sampling error in
    // high-frequency noise that a denoiser and a TEMPORAL average are meant to
    // clear; without the temporal half, what is left is stipple. Measured on a
    // Seyda Neen frame, the AO debug view's mean laplacian goes 0.965 -> 0.821,
    // a 15% drop, purely from the resolve running at all.
    //
    // That 15% is the honest number and the first one I took was not. Enabling
    // TAA the only way that worked before this change -- "--upscaler temporal"
    // -- measured 41%, but it also dropped the render scale to 0.667, and a
    // frame rendered at 44% of the pixels is smoother for reasons that have
    // nothing to do with the temporal resolve.
    //
    // Native quality rather than the default Quality preset, because those two
    // things had been welded together: asking for the temporal backend also
    // dropped the render scale to 1/1.5, so "turn TAA on" silently meant "render
    // at 44% of the pixels". A later --upscaler argument still overrides both.
    {
        odai::render::UpscalerSettings upscaler = app.upscalerSettings();
        upscaler.backend = odai::render::UpscalerBackend::Temporal;
        upscaler.quality = odai::render::UpscalerQuality::Native;
        app.setUpscalerSettings(upscaler);
    }
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
        } else if (std::strcmp(argv[i], "--upscaler") == 0 && i + 1 < argc) {
            // off | temporal | xess | fsr | dlss. Unavailable backends report
            // why and fall back rather than failing to launch.
            odai::render::UpscalerSettings upscaler = app.upscalerSettings();
            if (odai::render::parseUpscalerBackend(argv[++i], upscaler.backend)) {
                app.setUpscalerSettings(upscaler);
            } else {
                std::cout << "unknown --upscaler backend: " << argv[i]
                          << " (off|temporal|xess|fsr|dlss)\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--upscaler-quality") == 0 && i + 1 < argc) {
            odai::render::UpscalerSettings upscaler = app.upscalerSettings();
            if (odai::render::parseUpscalerQuality(argv[++i], upscaler.quality)) {
                app.setUpscalerSettings(upscaler);
            } else {
                std::cout << "unknown --upscaler-quality: " << argv[i]
                          << " (native|ultraquality|quality|balanced|performance|ultraperformance)\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--weather") == 0 && i + 1 < argc) {
            app.setWeather(argv[++i]);
        } else if (std::strcmp(argv[i], "--mod") == 0 && i + 1 < argc) {
            // A directory laid out like Data itself (textures\..., meshes\...).
            // Repeatable; later ones win, as a mod manager's load order would.
            app.addModDirectory(argv[++i]);
        } else if (std::strcmp(argv[i], "--shader-pack") == 0 && i + 1 < argc) {
            const std::string preset = argv[++i];
            if (preset != "rafael") {
                std::cout << "unknown --shader-pack preset: " << preset
                          << " (rafael)\n";
                return 1;
            }
            // The app reads this after renderer initialization, while the water
            // texture is created during initialization. Publish both choices now.
            setenv("ODAI_FNV_SHADER_PACK", preset.c_str(), 1);
        } else if (std::strcmp(argv[i], "--cache") == 0 && i + 1 < argc) {
            app.setStreamCacheDirectory(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-cache") == 0) {
            app.setStreamCacheEnabled(false);
        } else if (std::strcmp(argv[i], "--interior") == 0 && i + 1 < argc) {
            // Start INSIDE this interior rather than on its doorstep, which is
            // where New Vegas itself begins: --interior GSDocMitchellHouse.
            app.startInsideInterior(argv[++i]);
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
        } else if (std::strcmp(argv[i], "--tour-file") == 0 && i + 1 < argc) {
            const std::string path = argv[++i];
            const int loaded = odai::games::newvegas::loadTourFile(path);
            if (loaded == 0) {
                std::cout << "could not read a tour from " << path
                          << " (need at least 4 lines of 'px py pz lx ly lz')\n";
                return 1;
            }
            std::cout << "tour: " << loaded << " waypoints from " << path << "\n";
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
        } else if (std::strcmp(argv[i], "--capture-video") == 0 && i + 1 < argc) {
            // <out.mp4> [fps] [seconds]. Frames are piped to ffmpeg as they are
            // rendered; nothing lands on disk but the finished file.
            const std::string outputPath = argv[++i];
            float fps = 60.0f;
            float seconds = 40.0f;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                fps = static_cast<float>(std::atof(argv[++i]));
            }
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                seconds = static_cast<float>(std::atof(argv[++i]));
            }
            if (fps < 1.0f) {
                fps = 60.0f;
            }
            app.setCaptureVideo(outputPath, static_cast<int>(fps * seconds), fps);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "odai_game_newvegas [--scene <path.bin>]\n"
                      << "  Falls back to $ODAI_FNV_SCENE when --scene is absent.\n"
                      << "odai_game_newvegas --screenshot <out.ppm> [frames]\n"
                      << "  Render `frames` frames (default 8), write a PPM, and quit.\n"
                      << "  Cook a scene first with odai_newvegas_cooker.\n"
                      << "odai_game_newvegas --flythrough [seconds] --capture-video <out.mp4> [fps] [secs]\n"
                      << "  Fly the tour and encode it directly, then quit. Needs ffmpeg on PATH;\n"
                      << "  $ODAI_CAPTURE_ENCODER overrides the auto-detected H.264 encoder.\n"
                      << "odai_game_newvegas --flythrough [seconds] --capture-seq <dir> [fps] [seconds]\n"
                      << "  The same, as numbered PPMs. Prefer --capture-video: a still sequence\n"
                      << "  at this resolution is gigabytes.\n"
                      << "odai_game_newvegas --tour-file <path>\n"
                      << "  Replace the built-in Goodsprings path with rows of `px py pz  lx ly lz`.\n"
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
                      << "  --shader-pack rafael enables the native Rafael/Enhanced-PBR\n"
                      << "  preset and its externally installed water normal. Also\n"
                      << "  $ODAI_FNV_SHADER_PACK=rafael and $ODAI_WATER_NORMAL=<png|dds>.\n"
                      << "odai_game_newvegas --stream <Data> --plugin-add <Mod.esp>\n"
                      << "  Load an extra plugin and merge its world-record overrides; masters\n"
                      << "  resolve on their own. Plugins may live in --stream or --mod roots.\n"
                      << "  TES3 load orders merge exterior grids and named interiors. Also\n"
                      << "  $ODAI_FNV_PLUGINS, ','-separated.\n"
                      << "  --weather <EditorID> forces one weather by name.\n"
                      << "\n"
                      << "See docs/FNV_MODS.md and docs/MORROWIND_MODS.md for recipes.\n";
            return 0;
        }
    }
    if (const char* shaderPack = std::getenv("ODAI_FNV_SHADER_PACK")) {
        if (std::strcmp(shaderPack, "rafael") == 0 && std::getenv("ODAI_WATER_NORMAL") == nullptr) {
            const std::filesystem::path waterNormal = defaultShaderPackWaterNormalPath();
            const std::string waterNormalString = waterNormal.string();
            setenv("ODAI_WATER_NORMAL", waterNormalString.c_str(), 0);
        }
    }
    if (!app.init("New Vegas")) {
        return 1;
    }
    app.run();
    app.shutdown();
    return 0;
}
