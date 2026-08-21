#include "games/newvegas/newvegas_traversal_state.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    namespace nv = odai::games::newvegas;
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "odai-traversal-state-tests";
    std::filesystem::create_directories(root);
    const std::filesystem::path path = root / "state.json";

    nv::TraversalState state;
    state.interior = true;
    state.worldspaceEditorId = "WhiterunWorld";
    state.interiorEditorId = "WhiterunBanneredMare";
    state.position[0] = 1.0f;
    state.position[1] = 2.0f;
    state.position[2] = 3.0f;
    state.yawDegrees = 42.0f;
    state.pitchDegrees = -5.0f;
    state.timeOfDayHours = 18.25f;
    state.weatherEditorId = "SkyrimCloudy";
    state.loadOrderFingerprint = "fingerprint";
    state.discoveries.push_back({0xfe001123u, "Tamriel", "Whiterun"});
    std::string error;
    assert(nv::saveTraversalStateAtomic(path, state, error));
    assert(!std::filesystem::exists(path.string() + ".tmp"));

    nv::TraversalState loaded;
    assert(nv::loadTraversalState(path, loaded, error));
    assert(loaded.interior);
    assert(loaded.worldspaceEditorId == "WhiterunWorld");
    assert(loaded.interiorEditorId == "WhiterunBanneredMare");
    assert(loaded.position[2] == 3.0f);
    assert(loaded.discoveries.size() == 1u);
    assert(loaded.discoveries[0].sourceReferenceFormId == 0xfe001123u);

    {
        std::ofstream corrupt(path, std::ios::trunc);
        corrupt << "{\"version\":1,\"camera\":";
    }
    assert(!nv::loadTraversalState(path, loaded, error));
    assert(error.find("invalid traversal state") != std::string::npos);

    std::filesystem::remove_all(root);
    std::cout << "traversal state tests passed\n";
    return 0;
}
