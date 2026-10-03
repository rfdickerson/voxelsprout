#pragma once

// Synthetic GI verification uses the production scene upload and compute passes.
// Readback is explicit, linear, and independent of exposure and camera visibility.
inline bool runImportedGiFixture(odai::render::Renderer& renderer) {
    using namespace odai::render;
    using namespace odai::importer;
    const auto scene = [](bool blue, bool lit) {
        ImportedScene s;
        s.sourceTag = "morrowind_interior";
        ImportedSceneTexture t;
        t.sourcePath = blue ? "synthetic/gi/blue" : "synthetic/gi/red";
        t.width = t.height = 4;
        t.format = TextureFormat::BC1;
        t.rgba8 = blue ? std::vector<std::uint8_t>{31,0,0,0,0,0,0,0} :
                        std::vector<std::uint8_t>{0,248,0,0,0,0,0,0};
        s.textures.push_back(t);
        t.sourcePath = "synthetic/gi/neutral";
        t.format = TextureFormat::RGBA8;
        t.width = t.height = 1;
        t.rgba8 = {180,180,180,255};
        s.textures.push_back(t);
        const auto quad = [&](std::array<float,3> a, std::array<float,3> b,
                              std::array<float,3> c, std::array<float,3> d,
                              std::array<float,3> n, unsigned texture) {
            const auto base = std::uint32_t(s.packedVertices.size());
            const auto first = std::uint32_t(s.packedIndices.size());
            for (auto p : {a,b,c,d}) {
                ImportedScenePackedVertex v{};
                std::copy(p.begin(),p.end(),v.position);
                std::copy(n.begin(),n.end(),v.normal);
                v.color[0] = v.color[1] = v.color[2] = 1;
                v.textureIndex = texture;
                s.packedVertices.push_back(v);
            }
            for (auto index : {0u,1u,2u,0u,2u,3u}) s.packedIndices.push_back(base + index);
            s.packedDraws.push_back({first,6});
        };
        quad({-16,-64,-320},{-16,320,-320},{-16,320,320},{-16,-64,320},{1,0,0},0);
        quad({-64,-48,-320},{-64,-48,320},{320,-48,320},{320,-48,-320},{0,1,0},1);
        quad({16,-64,64},{16,320,64},{16,320,320},{16,-64,320},{1,0,0},1);
        // The light is inside its own lamp geometry. Endpoint occupancy must
        // not extinguish it, while the independent blocker must still occlude.
        quad({144,120,-152},{144,136,-152},{144,136,-136},{144,120,-136},{1,0,0},1);
        ImportedSceneLight light;
        light.position[0] = 144; light.position[1] = 128; light.position[2] = -144;
        light.radius = 600; light.intensity = lit ? 1 : 0;
        s.lights.push_back(light);
        std::fill_n(s.boundsMin,3,-320.f); std::fill_n(s.boundsMax,3,320.f);
        return s;
    };
    CameraPose camera{};
    camera.x = 240; camera.y = 96; camera.z = 320; camera.yawDegrees = -125;
    renderer.setTaaEnabled(false);
    renderer.setRayTracingEnabled(false);
    renderer.setAutoExposureEnabled(false);
    renderer.setNeutralColorGrading();
    renderer.setImportedSceneInteriorMode(true);
    renderer.setGlobalIlluminationEnabled(true);
    const auto read = [&](VoxelGiCapture& capture) {
        // A window-system resize may consume a render call without submission.
        // Bound startup and each change to four calls, including that recovery.
        for (int frame = 0; frame < 4; ++frame) renderer.renderFrame(camera);
        if (!renderer.captureVoxelGi(capture)) return false;
        return std::all_of(capture.rgb.begin(), capture.rgb.end(), [](float x) { return std::isfinite(x) && x >= 0; });
    };
    const auto sample = [](const VoxelGiCapture& c, float x, float y, float z) {
        const int gx = int(std::floor((x-c.origin[0])/c.cellSize));
        const int gy = int(std::floor((y-c.origin[1])/c.cellSize));
        const int gz = int(std::floor((z-c.origin[2])/c.cellSize));
        const std::size_t index = (gx + c.resolution * (gy + c.resolution * gz)) * 3;
        return std::array<float,3>{c.rgb.at(index),c.rgb.at(index+1),c.rgb.at(index+2)};
    };
    VoxelGiCapture red, blue, dark, turned, stable;
    if (!renderer.uploadImportedScene(scene(false,true)) || !renderer.waitForImportedSceneUploads() || !read(red)) return false;
    const auto r = sample(red,48,16,-144), blocked = sample(red,48,16,176);
    const auto image = [&](std::vector<std::uint8_t>& pixels) {
        renderer.renderFrame(camera);
        if (!renderer.prepareFrameCapture()) return false;
        renderer.renderFrame(camera);
        std::uint32_t width = 0, height = 0;
        return renderer.captureFrameRgb(pixels, width, height) && width && height;
    };
    std::vector<std::uint8_t> onImage, offImage, repeatedOff;
    if (!image(onImage)) return false;
    renderer.setGlobalIlluminationEnabled(false);
    if (!image(offImage) || !image(repeatedOff) || onImage.size() != offImage.size() ||
        offImage.size() != repeatedOff.size()) return false;
    std::size_t changedPixels = 0;
    for (std::size_t i = 0; i < onImage.size(); ++i) {
        if (std::abs(int(onImage[i]) - int(offImage[i])) > 2) ++changedPixels;
        if (std::abs(int(offImage[i]) - int(repeatedOff[i])) > 1) return false;
    }
    if (changedPixels < 10) { std::cerr << "GI toggle did not affect rendered receivers\n"; return false; }
    renderer.setGlobalIlluminationEnabled(true);
    camera.yawDegrees += 180;
    if (!read(turned) || !read(stable)) return false;
    const auto offscreen = sample(turned,48,16,-144);
    if (!renderer.uploadImportedScene(scene(true,true)) || !renderer.waitForImportedSceneUploads() || !read(blue)) return false;
    const auto b = sample(blue,48,16,-144), blockedBlue = sample(blue,48,16,176);
    if (!renderer.uploadImportedScene(scene(true,false)) || !renderer.waitForImportedSceneUploads() || !read(dark)) return false;
    const auto unlit = sample(dark,48,16,-144);
    // Residency, volume movement, and re-entry must replace the old field.
    const auto chunk = renderer.addImportedSceneChunk(scene(false,true));
    VoxelGiCapture added, moved, removed, reentered;
    if (chunk == Renderer::kInvalidImportedChunkIndex || !renderer.waitForImportedSceneUploads() || !read(added)) return false;
    camera.x += 320;
    if (!read(moved) || moved.origin == added.origin) return false;
    renderer.removeImportedSceneChunk(chunk);
    if (!read(removed)) return false;
    renderer.setImportedSceneInteriorMode(false);
    renderer.renderFrame(camera);
    renderer.setImportedSceneInteriorMode(true);
    if (!read(reentered)) return false;
    const bool residency = sample(added,48,16,-144)[0] > .001f &&
        sample(moved,48,16,-144)[0] > .001f &&
        sample(removed,48,16,-144)[0] < .0001f &&
        sample(reentered,48,16,-144)[0] < .0001f;
    std::cout << "GI fixture linear receiver: red=" << r[0] << ',' << r[1] << ',' << r[2]
              << " blue=" << b[0] << ',' << b[1] << ',' << b[2]
              << " blockedTransfer=" << std::abs(blocked[0]-blockedBlue[0]) << " offscreen=" << offscreen[0]
              << " unlit=" << unlit[2] << '\n';
    const bool pass = r[0] > .001f && r[0] > 2*r[2] && b[2] > .001f && b[2] > 2*b[0] &&
        std::abs(blocked[0]-blockedBlue[0]) < (r[0]-b[0])*.25f && std::abs(offscreen[0]-r[0]) < .0001f &&
        unlit[2] < b[2]*.01f && turned.rgb == stable.rgb && residency;
    if (!pass) std::cerr << "GI color transfer, blocker, offscreen, light response or stationary stability failed\n";
    return pass;
}
