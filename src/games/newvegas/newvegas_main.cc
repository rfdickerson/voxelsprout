#include "games/newvegas/newvegas_app.h"

#include "render/upscale/upscale_policy.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
    // MSAA off by default for this game: TAA (taa.comp.slang) does the
    // anti-aliasing work now, MSAA 4x measured ~1.5 ms of main-pass GPU time
    // on the target iGPU, and MSAA cannot fix the two artifacts that actually
    // show here (texture shimmer, alpha-test cutout crawl) anyway. setenv with
    // overwrite=0, so an explicit ODAI_MSAA from the user still wins.
    setenv("ODAI_MSAA", "1", 0);
    // Keep the window, presentation and UI at the display resolution, but shade
    // the 3D scene at 1920x1080 and let the temporal backend reconstruct it.
    // The measured 3200x1800 Skyrim frame is fill-bound at ~35 ms; this removes
    // 62.5% of its scene pixels without softening native-resolution text.
    // ODAI_RENDER_SIZE and the older ODAI_RENDER_SCALE both remain explicit
    // overrides, with scale taking precedence when both are supplied.
    setenv("ODAI_RENDER_SIZE", "1920x1080", 0);
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
    bool profileSpecified = false;
    bool loadOrderSpecified = false;
    bool listProfiles = false;
    bool profilePicker = false;
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
        } else if (std::strcmp(argv[i], "--load-order") == 0 && i + 1 < argc) {
            app.setLoadOrderPath(argv[++i]);
            loadOrderSpecified = true;
        } else if (std::strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            app.setContentProfilePath(argv[++i]);
            profileSpecified = true;
        } else if (std::strcmp(argv[i], "--mods-root") == 0 && i + 1 < argc) {
            app.setContentProfileModsRoot(argv[++i]);
        } else if (std::strcmp(argv[i], "--compat-report") == 0 && i + 1 < argc) {
            app.setCompatibilityReportPath(argv[++i]);
        } else if (std::strcmp(argv[i], "--reindex-content") == 0) {
            app.setForceContentReindex(true);
        } else if (std::strcmp(argv[i], "--list-profiles") == 0) {
            listProfiles = true;
        } else if (std::strcmp(argv[i], "--profile-picker") == 0) {
            profilePicker = true;
        } else if (std::strcmp(argv[i], "--no-profile-picker") == 0) {
            profilePicker = false;
        } else if (std::strcmp(argv[i], "--state") == 0 && i + 1 < argc) {
            app.setTraversalStatePath(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-resume") == 0) {
            app.setResumeEnabled(false);
        } else if (std::strcmp(argv[i], "--scenario") == 0 && i + 1 < argc) {
            app.setScenario(argv[++i]);
        } else if (std::strcmp(argv[i], "--save-game") == 0 && i + 1 < argc) {
            app.setGameplaySavePath(argv[++i]);
        } else if (std::strcmp(argv[i], "--load-game") == 0 && i + 1 < argc) {
            app.setGameplayLoadPath(argv[++i]);
        } else if (std::strcmp(argv[i], "--worldspace") == 0 && i + 1 < argc) {
            app.setStreamWorldspace(argv[++i]);
        } else if (std::strcmp(argv[i], "--plugin-add") == 0 && i + 1 < argc) {
            // An extra plugin loaded after --plugin; masters resolve on their
            // own, so "--plugin-add NevadaSkies.esp" is enough.
            app.addPlugin(argv[++i]);
        } else if (std::strcmp(argv[i], "--upscaler") == 0 && i + 1 < argc) {
            // off | temporal | xess. Unavailable backends report
            // why and fall back rather than failing to launch.
            odai::render::UpscalerSettings upscaler = app.upscalerSettings();
            if (odai::render::parseUpscalerBackend(argv[++i], upscaler.backend)) {
                app.setUpscalerSettings(upscaler);
            } else {
                std::cout << "unknown --upscaler backend: " << argv[i]
                          << " (off|temporal|xess)\n";
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
        } else if (std::strcmp(argv[i], "--hour") == 0 && i + 1 < argc) {
            const float hour = static_cast<float>(std::atof(argv[++i]));
            if (hour < 0.0f || hour >= 24.0f) {
                std::cout << "--hour must be in [0,24)\n";
                return 1;
            }
            app.setTimeOfDayHours(hour);
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
        } else if (std::strcmp(argv[i], "--tes3-start-quest") == 0 && i + 2 < argc) {
            const std::string questId = argv[++i];
            char* end = nullptr;
            const long index = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' ||
                index < std::numeric_limits<std::int32_t>::min() ||
                index > std::numeric_limits<std::int32_t>::max()) {
                std::cout << "--tes3-start-quest requires <journal-id> <index>\n";
                return 1;
            }
            app.setTes3StartQuest(questId, static_cast<std::int32_t>(index));
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
        } else if (std::strcmp(argv[i], "--capture-audio") == 0) {
            app.setCaptureAudio(true);
        } else if (std::strcmp(argv[i], "--capture-seed") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long value = std::strtoul(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || value > 0xfffffffful) {
                std::cout << "--capture-seed must be a 32-bit unsigned integer\n";
                return 1;
            }
            app.setCaptureSeed(static_cast<std::uint32_t>(value));
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "odai [--scene <path.bin>]\n"
                      << "  Falls back to $ODAI_FNV_SCENE when --scene is absent.\n"
                      << "odai --screenshot <out.ppm> [frames]\n"
                      << "  Render `frames` frames (default 8), write a PPM, and quit.\n"
                      << "  Cook a scene first with odai_newvegas_cooker.\n"
                      << "odai --flythrough [seconds] --capture-video <out.mp4> [fps] [secs]\n"
                      << "  Fly the tour and encode it directly, then quit. Needs ffmpeg on PATH;\n"
                      << "  $ODAI_CAPTURE_ENCODER overrides the auto-detected H.264 encoder.\n"
                      << "  Add --capture-audio for deterministic 48 kHz engine ambience;\n"
                      << "  --capture-seed <u32> fixes regional sound choices.\n"
                      << "  --hour <0..24) fixes the authored time of day.\n"
                      << "odai --flythrough [seconds] --capture-seq <dir> [fps] [seconds]\n"
                      << "  The same, as numbered PPMs. Prefer --capture-video: a still sequence\n"
                      << "  at this resolution is gigabytes.\n"
                      << "odai --tour-file <path>\n"
                      << "  Replace the built-in Goodsprings path with rows of `px py pz  lx ly lz`.\n"
                      << "odai --character [<skeleton.nif> <part.nif>...]\n"
                      << "  Stand one GPU-skinned character in bind pose, no world.\n"
                      << "  Defaults to characters\\_male\\skeleton.nif + upperbody.nif.\n"
                      << "odai --stream <Data> --mod <dir> [--mod <dir>...]\n"
                      << "  Override game assets from directories laid out like Data\n"
                      << "  (textures\\..., meshes\\...); later --mod wins. Also\n"
                      << "  $ODAI_FNV_MODS, ':'-separated.\n"
                      << "  A texture pack needs $ODAI_FNV_TEX_SIZE raised too: the\n"
                      << "  default clamps every texture to 512 px, so higher-resolution\n"
                      << "  art is mip-dropped away before it is ever seen.\n"
                      << "  --shader-pack rafael enables the native Rafael/Enhanced-PBR\n"
                      << "  preset. The engine uses its bundled tileable water normal;\n"
                      << "  $ODAI_FNV_SHADER_PACK=rafael and $ODAI_WATER_NORMAL=<png|dds>.\n"
                      << "odai --stream <Data> --plugin-add <Mod.esp>\n"
                      << "  Load an extra plugin and merge its world-record overrides; masters\n"
                      << "  resolve on their own. Plugins may live in --stream or --mod roots.\n"
                      << "  TES3 load orders merge exterior grids and named interiors. Also\n"
                      << "  $ODAI_FNV_PLUGINS, ','-separated.\n"
                      << "  --load-order <plugins.txt> selects Skyrim's active profile;\n"
                      << "  otherwise it auto-discovers Proton/native profiles and falls\n"
                      << "  back to installed official content. Also $ODAI_FNV_LOAD_ORDER.\n"
                      << "  --state <path> overrides the traversal save; --no-resume skips\n"
                      << "  loading it without deleting it. Explicit world/interior/spawn wins.\n"
                      << "  --scenario skyrim-bleak-falls starts the Skyrim-first gameplay\n"
                      << "  scenario at Riverwood with MQ101 post-Helgen and authored MQ102:10 startup. F5/F9 save/load\n"
                      << "  a checksummed ODAI save; --save-game/--load-game override its path.\n"
                      << "  --weather <EditorID> forces one weather by name.\n"
                      << "  --tes3-start-quest <journal-id> <index> seeds an authored TES3 journal entry;\n"
                      << "  press J for the journal and E while facing an actor to talk.\n"
                      << "  --profile <path> loads an ODAI JSON, MO2 profile directory,\n"
                      << "  or OpenMW openmw.cfg as one authoritative content graph.\n"
                      << "  --mods-root <dir> resolves nonstandard MO2 instances; --mod and\n"
                      << "  --plugin-add remain highest-priority overlays.\n"
                      << "  --list-profiles lists discovered profiles; --profile-picker selects\n"
                      << "  the sole match or lists ambiguous matches. --compat-report <json> writes validation.\n"
                      << "  --reindex-content rebuilds persistent content indexes.\n"
                      << "\n"
                      << "See docs/FNV_MODS.md and docs/MORROWIND_MODS.md for recipes.\n";
            return 0;
        }
    }
    if (profileSpecified && loadOrderSpecified) {
        std::cout << "--profile and --load-order are both authoritative; choose one\n";
        return 1;
    }
    if (listProfiles || (profilePicker && !profileSpecified)) {
        const auto profiles = odai::importer::fnv::discoverContentProfiles();
        if (profiles.empty()) {
            std::cout << "no ODAI, MO2, or OpenMW profiles discovered\n";
            return listProfiles ? 0 : 1;
        }
        for (std::size_t index = 0; index < profiles.size(); ++index) {
            std::cout << index + 1u << "  " << profiles[index].string() << "\n";
        }
        if (listProfiles) return 0;
        if (profiles.size() != 1u) {
            std::cout << "multiple profiles found; pass --profile <path> to select one\n";
            return 1;
        }
        app.setContentProfilePath(profiles.front().string());
    }
    if (!app.init("New Vegas")) {
        return 1;
    }
    app.run();
    app.shutdown();
    return 0;
}
