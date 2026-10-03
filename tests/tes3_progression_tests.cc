#include "bethesda/bethesda_session.h"
#include "bethesda/save_game.h"
#include "bethesda/tes3_level_ui_flow.h"
#include "bethesda/tes3_class_ui_flow.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
namespace {
using namespace odai::bethesda;
using namespace odai::importer::bethesda;
namespace fs = std::filesystem;
int failures = 0;
void check(bool value, const std::string& message) {
    if (!value) { std::cerr << "[TES3 progression] " << message << '\n'; ++failures; }
}
template <typename T>
void append(std::vector<std::uint8_t>& bytes, const T& value) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
}

void sub(std::vector<std::uint8_t>& body, const char* type,
         std::vector<std::uint8_t> payload) {
    body.insert(body.end(), type, type + 4);
    append(body, static_cast<std::uint32_t>(payload.size()));
    body.insert(body.end(), payload.begin(), payload.end());
}

void sub(std::vector<std::uint8_t>& body, const char* type, const std::string& text) {
    std::vector<std::uint8_t> payload(text.begin(), text.end());
    payload.push_back('\0');
    sub(body, type, std::move(payload));
}

void record(std::vector<std::uint8_t>& file, const char* type,
            const std::vector<std::uint8_t>& body) {
    file.insert(file.end(), type, type + 4);
    append(file, static_cast<std::uint32_t>(body.size()));
    append(file, std::uint32_t{0});
    append(file, std::uint32_t{0});
    file.insert(file.end(), body.begin(), body.end());
}

void header(std::vector<std::uint8_t>& file) {
    std::vector<std::uint8_t> body;
    std::vector<std::uint8_t> hedr(300u, 0u);
    const float version = 1.3f;
    const std::uint32_t master = 1u;
    std::memcpy(hedr.data(), &version, sizeof(version));
    std::memcpy(hedr.data() + 4u, &master, sizeof(master));
    sub(body, "HEDR", std::move(hedr));
    record(file, "TES3", body);
}

template<class T> void put(std::vector<std::uint8_t>& bytes, std::size_t at, T value) {
    std::memcpy(bytes.data() + at, &value, sizeof(value));
}
std::shared_ptr<Tes3ContentStore> fixture(const fs::path& root, bool overridden = false) {
    fs::create_directories(root);
    std::vector<std::uint8_t> file; header(file);
    for (int s = 0; s < 27; ++s) {
        std::vector<std::uint8_t> body, index; append(index, s); sub(body, "INDX", index);
        std::vector<std::uint8_t> data(24);
        put(data, 0, std::int32_t(s == 8 ? 4 : s == 20 ? 0 : s % 7));
        put(data, 4, std::int32_t(s % 3));
        for (int u = 0; u < 4; ++u) put(data, 8 + u * 4, float(u + 1));
        sub(body, "SKDT", data); record(file, "SKIL", body);
    }
    std::vector<std::uint8_t> body, data(60);
    sub(body, "NAME", "test_class");
    put(data, 0, std::int32_t(0)); put(data, 4, std::int32_t(5)); put(data, 8, std::int32_t(0));
    for (int i = 0; i < 5; ++i) { put(data, 12 + i * 8, std::int32_t(i)); put(data, 16 + i * 8, std::int32_t(i + 5)); }
    put(data, 52, std::int32_t(1)); sub(body, "CLDT", data); record(file, "CLAS", body);
    body.clear(); data.assign(140, 0); sub(body, "NAME", "test_race");
    for (int i = 0; i < 7; ++i) { put(data, i * 8, std::int32_t(-1)); }
    put(data, 0, std::int32_t(8)); put(data, 4, std::int32_t(5));
    for (int a = 0; a < 8; ++a) { put(data, 56 + a * 8, std::int32_t(40)); put(data, 60 + a * 8, std::int32_t(45)); }
    sub(body, "RADT", data); record(file, "RACE", body);
    body.clear(); sub(body, "NAME", "passive_race"); sub(body, "RADT", data);
    sub(body, "NPCS", "trainer_ability"); record(file, "RACE", body);
    body.clear(); sub(body, "NAME", "trainer_ability"); data.assign(12, 0);
    put(data, 0, std::int32_t(1)); sub(body, "SPDT", data);
    for (const auto [effect, skill, attribute, magnitude] : std::array<std::array<int, 4>, 3>{{
        {83, 8, -1, 30}, {83, 24, -1, 10}, {79, -1, 3, 20}}}) {
        data.assign(24, 0); put(data, 0, std::int16_t(effect)); data[2] = std::uint8_t(skill); data[3] = std::uint8_t(attribute);
        put(data, 16, std::int32_t(magnitude)); put(data, 20, std::int32_t(magnitude)); sub(body, "ENAM", data);
    }
    record(file, "SPEL", body);
    body.clear(); sub(body, "NAME", "speech_fortify"); sub(body, "SPDT", std::vector<std::uint8_t>(12, 0));
    data.assign(24, 0); put(data, 0, std::int16_t(83)); data[2] = 25; data[3] = 0xff;
    put(data, 12, std::int32_t(100)); put(data, 16, std::int32_t(200)); put(data, 20, std::int32_t(200));
    sub(body, "ENAM", data); record(file, "SPEL", body);
    body.clear(); sub(body, "NAME", "test_sign"); record(file, "BSGN", body);
    for (int n = 0; n < 4; ++n) {
        body.clear(); data.assign(12, 0); sub(body, "NAME", n == 3 ? "dry_interior" : n == 2 ? "water_interior" : n ? "no_sleep" : "wilderness");
        put(data, 0, std::uint32_t(n == 3 ? 1 : n == 2 ? 3 : n ? 4 : 0)); sub(body, "DATA", data);
        if (n == 2) { std::vector<std::uint8_t> height; append(height, 100.0f); sub(body, "WHGT", height); }
        if (!n) {
            std::vector<std::uint8_t> ref; append(ref, std::uint32_t(0x99)); sub(body, "FRMR", ref);
            sub(body, "NAME", "trainer"); sub(body, "DATA", std::vector<std::uint8_t>(24, 0));
            ref.clear(); append(ref, std::uint32_t(0x100)); sub(body, "FRMR", ref);
            sub(body, "NAME", "auto_female"); sub(body, "DATA", std::vector<std::uint8_t>(24, 0));
            ref.clear(); append(ref, std::uint32_t(0x101)); sub(body, "FRMR", ref);
            sub(body, "NAME", "faction_trainer"); sub(body, "DATA", std::vector<std::uint8_t>(24, 0));
            ref.clear(); append(ref, std::uint32_t(0x102)); sub(body, "FRMR", ref);
            sub(body, "NAME", "placed_repair_weapon"); data.assign(4, 0); put(data, 0, std::int32_t(20)); sub(body, "INTV", data);
            sub(body, "DATA", std::vector<std::uint8_t>(24, 0));
            ref.clear(); append(ref, std::uint32_t(0x103)); sub(body, "FRMR", ref);
            sub(body, "NAME", "repair_tool"); data.assign(4, 0); put(data, 0, std::int32_t(1)); sub(body, "INTV", data);
            sub(body, "DATA", std::vector<std::uint8_t>(24, 0));
        }
        record(file, "CELL", body);
    }
    body.clear(); data.assign(20, 0); sub(body, "NAME", "skill_book"); put(data, 12, std::int32_t(8));
    sub(body, "BKDT", data); sub(body, "TEXT", "A synthetic skill book."); record(file, "BOOK", body);
    body.clear(); sub(body, "NAME", "gold_001"); record(file, "MISC", body);
    constexpr std::array<float, 11> armorWeights{5,30,10,10,15,20,5,5,15,5,5};
    for (int kind = 0; kind < 3; ++kind) for (int type = 0; type < 11; ++type) {
        body.clear(); data.assign(24, 0);
        sub(body, "NAME", "armor_" + std::to_string(kind) + "_" + std::to_string(type));
        put(data, 0, std::int32_t(type));
        put(data, 4, armorWeights[type] * (kind == 0 ? .3f : kind == 1 ? .75f : 1.f));
        put(data, 12, std::int32_t(100)); sub(body, "AODT", data); record(file, "ARMO", body);
    }
    body.clear(); sub(body, "NAME", "bad_armor"); sub(body, "AODT", std::vector<std::uint8_t>(3)); record(file, "ARMO", body);
    for (int type = 0; type < 14; ++type) {
        body.clear(); data.assign(32, 0); sub(body, "NAME", "weapon_" + std::to_string(type));
        put(data, 8, std::int16_t(type)); sub(body, "WPDT", data); record(file, "WEAP", body);
    }
    for (const auto& [id, quality] : std::array<std::pair<const char*, float>, 2>{{{"repair_tool", .1f}, {"repair_good", 1.f}}}) {
        body.clear(); sub(body, "NAME", id); data.assign(16, 0);
        put(data, 8, std::int32_t(2)); put(data, 12, quality); sub(body, "RIDT", data); record(file, "REPA", body);
    }
    body.clear(); sub(body, "NAME", "repair_bad"); sub(body, "RIDT", std::vector<std::uint8_t>(3)); record(file, "REPA", body);
    body.clear(); sub(body, "NAME", "repair_weapon"); data.assign(32, 0);
    put(data, 8, std::int16_t(0)); put(data, 10, std::uint16_t(100)); sub(body, "WPDT", data);
    sub(body, "SCRI", "RepairProbe"); record(file, "WEAP", body);
    body.clear(); sub(body, "NAME", "placed_repair_weapon"); sub(body, "WPDT", data); record(file, "WEAP", body);
    body.clear(); data.assign(52, 0); std::memcpy(data.data(), "RepairProbe", 11); sub(body, "SCHD", data);
    sub(body, "SCTX", "begin RepairProbe\nshort onpcrepair\nshort repairs\nif ( onpcrepair == 1 )\nset repairs to repairs + 1\nset onpcrepair to 0\nendif\nend"); record(file, "SCPT", body);
    for (int edge = 0; edge < 3; ++edge) {
        body.clear(); data.assign(24, 0); sub(body, "NAME", "armor_edge_" + std::to_string(edge));
        put(data, 0, std::int32_t(1)); put(data, 4, edge == 0 ? 18.0004f : edge == 1 ? 27.0004f : 27.001f);
        sub(body, "AODT", data); record(file, "ARMO", body);
    }
    body.clear(); sub(body, "NAME", "trainer"); sub(body, "FNAM", "Test Trainer");
    data.assign(52, 0); put(data, 0, std::int16_t(10));
    for (int a = 0; a < 8; ++a) data[2 + a] = 60;
    for (int s = 0; s < 27; ++s) data[10 + s] = s == 8 ? 95 : s == 20 ? 90 : s == 5 ? 85 : 10;
    put(data, 38, std::uint16_t(100)); put(data, 40, std::uint16_t(100)); put(data, 42, std::uint16_t(100)); data[44] = 50;
    sub(body, "NPDT", data); data.assign(12, 0); put(data, 8, std::uint32_t(0x4000));
    sub(body, "AIDT", data); record(file, "NPC_", body);
    // Both the race and NPC explicitly grant the same spell; it is owned once.
    body.erase(body.begin(), body.begin() + 8 + sizeof("trainer"));
    sub(body, "NAME", "passive_trainer"); sub(body, "RNAM", "passive_race");
    sub(body, "NPCS", "trainer_ability"); record(file, "NPC_", body);
    // Use a separate authored NPC for disposition and faction-service checks.
    body.clear(); sub(body, "NAME", "faction_trainer"); sub(body, "RNAM", "test_race"); sub(body, "ANAM", "guild");
    data.assign(52, 0); for (int a = 0; a < 8; ++a) data[2 + a] = 60;
    for (int s = 0; s < 27; ++s) data[10 + s] = s == 8 ? 95 : s == 20 ? 90 : s == 5 ? 85 : 10;
    data[44] = 50; sub(body, "NPDT", data); data.assign(12, 0); put(data, 8, std::uint32_t(0x4000));
    sub(body, "AIDT", data); record(file, "NPC_", body);
    for (const auto* id : {"guild", "friend", "enemy"}) {
        body.clear(); sub(body, "NAME", id); sub(body, "FADT", std::vector<std::uint8_t>(240, 0));
        if (std::string_view(id) == "guild") for (const auto& [target, reaction] :
            std::array<std::pair<const char*, int>, 3>{{{"guild", 1}, {"friend", 4}, {"enemy", -2}}}) {
            sub(body, "ANAM", target); data.assign(4, 0); put(data, 0, std::int32_t(reaction)); sub(body, "INTV", data);
        }
        record(file, "FACT", body);
    }
    body.clear(); sub(body, "NAME", "test_disease"); data.assign(12, 0); put(data, 0, std::int32_t(3));
    sub(body, "SPDT", data); record(file, "SPEL", body);
    for (int greeting = 0; greeting < 2; ++greeting) {
        body.clear(); sub(body, "NAME", greeting ? "greeting 0" : "disposition_gate");
        sub(body, "DATA", std::vector<std::uint8_t>{std::uint8_t(greeting ? 2 : 0)}); record(file, "DIAL", body);
        body.clear(); sub(body, "INAM", greeting ? "disp_greeting" : "disp_gate"); sub(body, "PNAM", ""); sub(body, "NNAM", "");
        data.assign(12, 0xff); put(data, 0, std::int32_t(greeting ? 2 : 0)); put(data, 4, std::int32_t(greeting ? 0 : 60)); data[11] = 0;
        sub(body, "DATA", data); sub(body, "NAME", greeting ? "Hello." : "Disposition gate passed."); record(file, "INFO", body);
    }
    body.clear(); sub(body, "NAME", "persuasion_response_mode"); sub(body, "FNAM", "s");
    data.assign(4, 0); sub(body, "FLTV", data); record(file, "GLOB", body);
    for (const auto* outcome : {"Admire Success", "Admire Fail", "Intimidate Success", "Intimidate Fail",
                               "Taunt Success", "Taunt Fail", "Bribe Success", "Bribe Fail"}) {
        body.clear(); sub(body, "NAME", outcome); sub(body, "DATA", std::vector<std::uint8_t>{3}); record(file, "DIAL", body);
        const int variants = std::string_view(outcome) == "Admire Success" ? 4 : 1;
        for (int variant = 0; variant < variants; ++variant) {
            const auto key = std::string(outcome) + std::to_string(variant);
            body.clear(); sub(body, "INAM", key);
            sub(body, "PNAM", variant ? std::string(outcome) + std::to_string(variant - 1) : "");
            sub(body, "NNAM", variant + 1 < variants ? std::string(outcome) + std::to_string(variant + 1) : "");
            data.assign(12, 0xff); put(data, 0, std::int32_t(3)); put(data, 4, std::int32_t(0)); data[11] = 0;
            sub(body, "DATA", data); sub(body, "NAME", std::string(outcome) + " authored reply.");
            if (variant < variants - 1) {
                sub(body, "SCVR", "02000persuasion_response_mode"); data.assign(4, 0); put(data, 0, std::int32_t(variant + 1)); sub(body, "INTV", data);
                sub(body, "BNAM", variant == 0 ? "Player->ModReputation 10\nMessageBox \"Persuasion continuation\", \"Continue\"\nPlayer->ModReputation 20" :
                    variant == 1 ? "Player->ModReputation 100\nmissing_actor->SetHealth 5" : "Player->ModReputation 1\nGoodbye");
            }
            record(file, "INFO", body);
        }
    }
    for (int profile = 0; profile < 4; ++profile) {
        body.clear(); sub(body, "NAME", "reputation_" + std::to_string(profile));
        sub(body, "RNAM", "test_race"); sub(body, "CNAM", "test_class");
        if (profile != 1) sub(body, "ANAM", "guild");
        data.assign(profile == 3 ? 12 : 52, 0); put(data, 0, std::int16_t(profile == 2 ? 0 : 6));
        data[profile == 3 ? 3 : 45] = 47;
        data[profile == 3 ? 4 : 46] = std::uint8_t(profile == 2 ? -1 : 2);
        sub(body, "NPDT", data); record(file, "NPC_", body);
    }
    for (int profile = 0; profile < 4; ++profile) {
        body.clear(); sub(body, "NAME", profile == 0 ? "auto_female" : profile == 1 ? "auto_male" : profile == 2 ? "auto_zero" : "auto_high");
        sub(body, "RNAM", "test_race"); sub(body, "CNAM", "test_class");
        data.assign(12, 0); put(data, 0, std::int16_t(profile == 2 ? 0 : profile == 3 ? 200 : 6)); data[2] = 50;
        sub(body, "NPDT", data); data.assign(4, 0); put(data, 0, std::uint32_t(profile == 0 ? 0x11 : 0x10)); sub(body, "FLAG", data);
        data.assign(12, 0); put(data, 8, std::uint32_t(0x4000)); sub(body, "AIDT", data); record(file, "NPC_", body);
    }
    body.clear(); sub(body, "NAME", "player"); data.assign(52, 0); put(data, 0, std::int16_t(1));
    put(data, 38, std::uint16_t(100)); put(data, 40, std::uint16_t(100)); put(data, 42, std::uint16_t(100));
    sub(body, "NPDT", data); record(file, "NPC_", body);
    body.clear(); sub(body, "NAME", "test_creature"); data.assign(96, 0);
    for (int a = 0; a < 8; ++a) put(data, 8 + a * 4, std::int32_t(40));
    put(data, 40, std::int32_t(100)); put(data, 44, std::int32_t(100)); put(data, 48, std::int32_t(100));
    put(data, 56, std::int32_t(60)); put(data, 60, std::int32_t(50)); put(data, 64, std::int32_t(40));
    sub(body, "NPDT", data); record(file, "CREA", body);
    body.clear(); data.assign(52, 0); std::memcpy(data.data(), "LevelProbe", 10);
    sub(body, "SCHD", data); sub(body, "SCTX", "begin LevelProbe\nshort observed\nset observed to Player->GetLevel\nend");
    record(file, "SCPT", body);
    body.clear(); data.assign(52, 0); std::memcpy(data.data(), "DispositionProbe", 16); sub(body, "SCHD", data);
    sub(body, "SCTX", "begin DispositionProbe\nshort observed\nset observed to GetDisposition\nend"); record(file, "SCPT", body);
    body.clear(); data.assign(52, 0); std::memcpy(data.data(), "DispositionChange", 17); sub(body, "SCHD", data);
    sub(body, "SCTX", "begin DispositionChange\nshort observed\nSetDisposition 80\nModDisposition -10\nset observed to GetDisposition\nend"); record(file, "SCPT", body);
    body.clear(); data.assign(52, 0); std::memcpy(data.data(), "ReputationProbe", 15); sub(body, "SCHD", data);
    sub(body, "SCTX", "begin ReputationProbe\nshort baseline\nshort observed\nset baseline to GetReputation\nModReputation 5\nset observed to GetReputation\nend"); record(file, "SCPT", body);
    if (overridden) {
        const auto setting = [&](const char* id, float value) {
            std::vector<std::uint8_t> b, v; sub(b, "NAME", id); append(v, value); sub(b, "FLTV", v); record(file, "GMST", b);
        };
        setting("fMajorSkillBonus", .5f); setting("fSpecialSkillBonus", .5f);
        setting("iLevelUpTotal", 3); setting("iLevelUp01Mult", 4);
        setting("fLevelUpHealthEndMult", .2f);
        setting("iCuirassWeight", 20.9f); setting("fLightMaxMod", .25f); setting("fMedMaxMod", .5f);
        setting("fDispCrimeMod", .1f);
        setting("iPerMinChance", 69); setting("fPerTempMult", 2);
        setting("iAutoRepFacMod", 7); setting("iAutoRepLevMod", 3);
    }
    std::ofstream out(root / "Morrowind.esm", std::ios::binary);
    out.write(reinterpret_cast<const char*>(file.data()), file.size()); out.close();
    FalloutLoadOrder order; std::string error;
    check(order.open(root, {"Morrowind.esm"}, error), error);
    auto content = std::make_shared<Tes3ContentStore>(); check(content->load(order, "windows-1252", error), error);
    return content;
}
void setup(BethesdaSession& session, std::shared_ptr<const Tes3ContentStore> content) {
    std::string error; BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind; config.contentFingerprint = "tes3-progression"; config.livingWorldEnabled = false;
    check(session.configure(config, error), error); check(session.configureTes3Content(content, error), error);
    RuntimeObject player; player.id = session.playerObject(); player.kind = RuntimeObjectKind::Actor;
    player.base = makeTes3RecordKey("NPC_", "player"); player.actorValues.emplace();
    player.currentSpace.cell = makeTes3RecordKey("CELL", "wilderness"); player.originSpace = player.currentSpace;
    player.inventory.push_back({makeTes3RecordKey("MISC", "gold_001"), 100000, false});
    player.inventory.push_back({makeTes3RecordKey("BOOK", "skill_book"), 1, false});
    check(session.world().addInitialObject(player, error), error);
    auto& p = session.tes3().playerState(); p.race = "test_race"; p.actorClass = "test_class"; p.birthsign = "test_sign"; p.gender = 0;
    check(session.initializeTes3PlayerProgression(error), error);
    RuntimeObject trainer; trainer.id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99)); trainer.kind = RuntimeObjectKind::Actor;
    trainer.base = makeTes3RecordKey("NPC_", "trainer"); trainer.actorValues.emplace(); trainer.currentSpace = player.currentSpace;
    check(session.world().addInitialObject(trainer, error), error);
}
std::string checksum(const std::string& bytes) {
    std::uint64_t hash = 1469598103934665603ull;
    for (unsigned char byte : bytes) { hash ^= byte; hash *= 1099511628211ull; }
    std::ostringstream out; out << std::hex << std::setfill('0') << std::setw(16) << hash; return out.str();
}
void testImportedRepairCondition(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    BethesdaSession session; setup(session, content); std::string error;
    check(content->references().at(ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99))).itemCondition == -1,
        "absent reference INTV preserves the imported-full condition sentinel");
    for (const auto [number, condition] : std::array<std::pair<unsigned, int>,2>{{{0x102,20},{0x103,1}}}) {
        const auto id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", number));
        const auto& reference = content->references().at(id);
        check(reference.itemCondition == condition, "placed item imports TES3 INTV condition/remaining tool uses");
        RuntimeObject object; object.id = id; object.base = reference.base; object.kind = RuntimeObjectKind::Item;
        object.itemCondition = reference.itemCondition; object.currentSpace = session.world().find(session.playerObject())->currentSpace;
        check(session.world().addInitialObject(object, error), error);
        check(session.activateTes3Reference(id, false, error), error);
        (void)session.advance(1.0 / 60);
        check(!session.world().find(id)->enabled, "ordinary pickup consumes the authored placed source");
    }
    const auto tools = session.tes3RepairTools(), targets = session.tes3RepairTargets();
    check(tools.size() == 1 && targets.size() == 1 && tools[0].condition == 1 && targets[0].condition == 20,
        "ordinary pickup exposes imported partial tool and damaged weapon to repair selection");
    if (tools.empty() || targets.empty()) return;
    session.tes3().playerState().numericFilters["armorer"] = 60;
    auto& values = *session.world().find(session.playerObject())->actorValues;
    values.stamina = 50; values.maxStamina = 100; session.setRandomState(1);
    const auto result = session.repairTes3PlayerItem(targets[0].inventoryIndex, tools[0].inventoryIndex, error);
    check(result.accepted && result.success && result.repaired == 20 && result.toolConsumed &&
          session.tes3().playerState().progression.skillProgress[1] > 0 && session.tes3RepairTools().empty(),
        "picked-up authored damage and remaining uses feed an actual repair and Armorer award");
    check(saveOdaiGameAtomic(root / "placed-repair.json", session, error), error);
    BethesdaSession loaded; setup(loaded, content); SaveLoadReport report;
    check(loadOdaiGame(root / "placed-repair.json", loaded, {}, report, error) &&
          loaded.world().find(loaded.playerObject())->inventory == session.world().find(session.playerObject())->inventory &&
          loaded.tes3().playerState().progression == session.tes3().playerState().progression,
        "imported pickup/repair outcome persists without restoring consumed source or tool");
    const auto id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x102));
    check(!loaded.activateTes3Reference(id, false, error), "loading cannot pick up the authored item a second time");
}
void testGameplayRepair(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    const auto weapon = makeTes3RecordKey("WEAP", "repair_weapon");
    const auto tool = makeTes3RecordKey("REPA", "repair_tool");
    const auto prepare = [&](BethesdaSession& session, bool skilled) {
        setup(session, content);
        auto* player = session.world().find(session.playerObject());
        InventoryEntry damaged{weapon, 2, false}; damaged.condition = 20;
        player->inventory.push_back(damaged); player->inventory.push_back({tool, 2, false});
        if (skilled) {
            session.tes3().playerState().numericFilters["armorer"] = 60;
            player->actorValues->stamina = 50; player->actorValues->maxStamina = 100;
        }
        session.setRandomState(1);
    };
    const auto count = [&](const BethesdaSession& session, const RecordKey& key, int condition) {
        int total = 0;
        for (const auto& entry : session.world().find(session.playerObject())->inventory)
            if (entry.item == key && entry.condition == condition) total += entry.count;
        return total;
    };
    BethesdaSession success; prepare(success, true); std::string error;
    const auto tools = success.tes3RepairTools(), targets = success.tes3RepairTargets();
    const auto requirement = success.tes3().playerSkillRequirement(1);
    const auto repaired = success.repairTes3PlayerItem(targets[0].inventoryIndex, tools[0].inventoryIndex, error);
    check(repaired.accepted && repaired.success && repaired.repaired == 20 && !repaired.toolConsumed,
        "repair roll equal to modified Armorer/Strength/Luck/fatigue chance succeeds with imported tool quality");
    check(count(success, weapon, 20) == 1 && count(success, weapon, 40) == 1 &&
          count(success, tool, -1) == 1 && count(success, tool, 1) == 1,
        "repair changes one damaged copy and consumes one use from one tool in stacked inventories");
    check(std::abs(success.tes3().playerState().progression.skillProgress[1] - 1 / requirement) < 1e-6,
        "successful actual repair awards imported Armorer use exactly once");
    (void)success.advance(1.0 / 60);
    bool repairedEvent = false;
    for (const auto& [id, thread] : success.tes3().scripts().threads()) if (thread.program == "repairprobe")
        repairedEvent = thread.locals.at("repairs").number == 1 && thread.locals.at("onpcrepair").number == 0;
    check(repairedEvent, "successful repair delivers authored OnPCRepair once through the item script path");
    const auto afterEvent = success.tes3().playerState().progression;
    (void)success.advance(1.0 / 60);
    check(success.tes3().playerState().progression == afterEvent, "later ticks do not repeat the repair award");
    const auto path = root / "repair.json";
    check(saveOdaiGameAtomic(path, success, error), error);
    BethesdaSession restored; setup(restored, content); SaveLoadReport report;
    check(loadOdaiGame(path, restored, {}, report, error) &&
          restored.world().find(restored.playerObject())->inventory == success.world().find(success.playerObject())->inventory &&
          restored.tes3().playerState().progression == afterEvent && restored.randomState() == success.randomState(),
        "repair state, remaining tools, progression, script state and random outcome persist across save/load");
    BethesdaSession failed; prepare(failed, false);
    const auto before = failed.tes3().playerState().progression;
    const auto failedTargets = failed.tes3RepairTargets(), failedTools = failed.tes3RepairTools();
    Tes3ActiveSpell invisibility; invisibility.effects.push_back({39,-1,-1,1,1000});
    failed.tes3().activeSpellsForRestore()[failed.playerObject()].push_back(invisibility);
    const auto failure = failed.repairTes3PlayerItem(failedTargets[0].inventoryIndex, failedTools[0].inventoryIndex, error);
    check(failure.accepted && !failure.success && failure.repaired == 0 && count(failed, weapon, 20) == 2 &&
          count(failed, tool, 1) == 1 && failed.tes3().playerState().progression == before,
        "failed repair consumes a tool use and random roll without durability gain or Armorer award");
    check(failed.tes3MagicEffectMagnitude(failed.playerObject(), 39) == 0,
        "ordinary repair use breaks invisibility even when the outcome fails");
    BethesdaSession threshold; prepare(threshold, true);
    threshold.tes3().playerState().progression.skillProgress[1] = .999;
    const auto thresholdTools = threshold.tes3RepairTools(), thresholdTargets = threshold.tes3RepairTargets();
    check(threshold.repairTes3PlayerItem(thresholdTargets[0].inventoryIndex, thresholdTools[0].inventoryIndex, error).success &&
          threshold.tes3().playerState().numericFilters.at("armorer") == 61 &&
          threshold.tes3().playerState().progression.levelProgress == 1 &&
          threshold.tes3().playerState().progression.attributeIncreases[1] == 1,
        "actual repair threshold contributes a minor-skill character point and governing attribute bonus");
    BethesdaSession capped; prepare(capped, true);
    capped.tes3().playerState().numericFilters["armorer"] = 100;
    auto* pc = capped.world().find(capped.playerObject());
    for (auto& entry : pc->inventory) {
        if (entry.item == weapon) { entry.count = 1; entry.condition = 90; }
        if (entry.item == tool) { entry.count = 1; entry.condition = 1; entry.item = makeTes3RecordKey("REPA", "repair_good"); }
    }
    pc->inventory.push_back({weapon, 1, false});
    const auto cappedProgress = capped.tes3().playerState().progression;
    const auto cappedTools = capped.tes3RepairTools(), cappedTargets = capped.tes3RepairTargets();
    const auto full = capped.repairTes3PlayerItem(cappedTargets[0].inventoryIndex, cappedTools[0].inventoryIndex, error);
    check(full.accepted && full.success && full.repaired == 10 && full.toolConsumed && count(capped, weapon, -1) == 2 &&
          capped.tes3RepairTools().empty() && capped.tes3().playerState().progression == cappedProgress,
        "repair clamps at full condition, restacks identical copies, removes exhausted tool, and respects skill cap");
    const auto hash = capped.deterministicHash(); const auto random = capped.randomState();
    check(!capped.repairTes3PlayerItem(cappedTargets[0].inventoryIndex, cappedTools[0].inventoryIndex, error).accepted &&
          capped.deterministicHash() == hash && capped.randomState() == random,
        "repeated input on exhausted tool or fully repaired target rejects without award or roll");
    BethesdaSession blocked; prepare(blocked, true);
    const auto blockedTargets = blocked.tes3RepairTargets(), blockedTools = blocked.tes3RepairTools();
    blocked.tes3().playerState().progression.selectionOpen = true;
    const auto blockedHash = blocked.deterministicHash();
    check(!blocked.repairTes3PlayerItem(blockedTargets[0].inventoryIndex, blockedTools[0].inventoryIndex, error).accepted &&
          blocked.deterministicHash() == blockedHash, "level selection blocks repair without mutation");
}
void testInventoryCondition(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    BethesdaSession session; setup(session, content); std::string error;
    const auto pc = session.playerObject();
    const auto npc = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99));
    const auto item = makeTes3RecordKey("WEAP", "weapon_0");
    const auto command = [&](WorldCommandType type, ObjectId owner, int count, std::optional<int> condition = {}) {
        WorldCommand action; action.type = type; action.target = owner; action.item = item; action.itemCount = count;
        action.itemCondition = condition; action.other = npc; (void)session.world().queue(action);
        return session.world().applyQueuedCommands();
    };
    const auto quantity = [&](const BethesdaSession& source, ObjectId owner, int condition) {
        int count = 0;
        for (const auto& entry : source.world().find(owner)->inventory) if (entry.item == item && entry.condition == condition) count += entry.count;
        return count;
    };
    check(command(WorldCommandType::AddItem, pc, 1, 31).applied == 1 &&
          command(WorldCommandType::AddItem, pc, 2).applied == 1 &&
          quantity(session, pc, 31) == 1 && quantity(session, pc, -1) == 2,
        "damaged and imported-full copies retain separate inventory stacks");
    check(command(WorldCommandType::TransferItem, pc, 2).applied == 1 &&
          quantity(session, npc, 31) == 1 && quantity(session, npc, -1) == 1 && quantity(session, pc, -1) == 1,
        "aggregate transfer spans stacks and preserves each moved condition");
    const auto dropped = command(WorldCommandType::DropItem, npc, 1, 31);
    ObjectId worldItem;
    for (const auto& object : session.world().orderedObjects()) if (object.kind == RuntimeObjectKind::Item && object.base == item) worldItem = object.id;
    check(dropped.applied == 1 && worldItem.valid() && session.world().find(worldItem)->itemCondition == 31 &&
          quantity(session, npc, -1) == 1 && quantity(session, npc, 31) == 0,
        "selected damaged stack drops one conditioned world item and retains other copies");
    check(command(WorldCommandType::AddItem, pc, 1, 9).applied == 1, "distinct damaged copy added");
    check(session.activateTes3Reference(worldItem, false, error), error);
    (void)session.advance(1.0 / 60);
    check(quantity(session, pc, 31) == 1 && quantity(session, pc, 9) == 1 && quantity(session, pc, -1) == 1 &&
          !session.world().find(worldItem)->enabled,
        "ordinary world pickup preserves dropped condition without merging different copies");
    check(!session.dropInventoryItem(pc, item, error, 99), "unowned condition cannot queue a misleading drop action");
    const auto path = root / "inventory-condition.json";
    check(saveOdaiGameAtomic(path, session, error), error);
    BethesdaSession loaded; setup(loaded, content); SaveLoadReport report;
    const bool conditionLoaded = loadOdaiGame(path, loaded, {}, report, error);
    check(conditionLoaded && loaded.world().find(pc)->inventory == session.world().find(pc)->inventory &&
          loaded.world().find(worldItem)->itemCondition == 31 && loaded.deterministicHash() == session.deterministicHash(),
        "inventory and world item conditions round-trip with deterministic state");
    nlohmann::json saved; { std::ifstream in(path); in >> saved; }
    auto legacy = saved;
    for (auto& object : legacy["payload"]["world"]["objects"]) {
        object.erase("item_condition");
        for (auto& entry : object["inventory"]) entry.erase("condition");
    }
    legacy["checksum"] = checksum(legacy["payload"].dump());
    { std::ofstream out(root / "old-condition.json"); out << legacy.dump(); }
    BethesdaSession old; setup(old, content);
    const bool oldConditionLoaded = loadOdaiGame(root / "old-condition.json", old, {}, report, error);
    check(oldConditionLoaded && quantity(old, pc, -1) == 3 &&
          old.world().find(worldItem)->itemCondition == -1,
        "older saves initialize missing condition as imported-full without inventing damage");
    for (bool world : {false, true}) {
        auto malformed = saved;
        auto& objects = malformed["payload"]["world"]["objects"];
        auto owned = std::find_if(objects.begin(), objects.end(), [](const auto& object) { return !object.at("inventory").empty(); });
        if (world) objects[0]["item_condition"] = std::numeric_limits<std::uint64_t>::max();
        else (*owned)["inventory"][0]["condition"] = -2;
        malformed["checksum"] = checksum(malformed["payload"].dump());
        { std::ofstream out(root / "bad-condition.json"); out << malformed.dump(); }
        const auto hash = loaded.deterministicHash();
        check(!loadOdaiGame(root / "bad-condition.json", loaded, {}, report, error) && loaded.deterministicHash() == hash,
            "malformed item condition rejects before session mutation");
    }
    const auto inventoryBefore = session.world().find(pc)->inventory;
    const auto recipientBefore = session.world().find(npc)->inventory;
    auto& recipientInventory = session.world().find(npc)->inventory;
    const auto recipient = std::find_if(recipientInventory.begin(), recipientInventory.end(), [&](const auto& entry) { return entry.item == item; });
    recipient->count = std::numeric_limits<std::int32_t>::max();
    const auto fullRecipient = recipientInventory;
    check(command(WorldCommandType::TransferItem, pc, 1, 31).applied == 0 &&
          session.world().find(pc)->inventory == inventoryBefore && recipientInventory == fullRecipient,
        "mixed-condition transfer rejects aggregate count overflow atomically");
    recipientInventory = recipientBefore;
    check(command(WorldCommandType::AddItem, pc, std::numeric_limits<std::int32_t>::max(), 50).applied == 0 &&
          session.world().find(pc)->inventory == inventoryBefore,
        "new condition stack cannot overflow shared base-record inventory count");
    check(command(WorldCommandType::RemoveItem, pc, 2).applied == 1 &&
          quantity(session, pc, -1) + quantity(session, pc, 9) + quantity(session, pc, 31) == 1,
        "base-record removal spans conditioned stacks without deleting untouched copies");
}
void testNpcReputation(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    const auto verify = [&](const Tes3ContentStore& source, bool modified) {
        check(source.findActor("NPC_", "reputation_0")->reputation == (modified ? 36 : 6),
            "faction reputation replaces explicit authored reputation using rank and imported GMSTs");
        check(source.findActor("NPC_", "reputation_1")->reputation == 47,
            "unaffiliated NPC retains explicit authored reputation");
        check(source.findActor("NPC_", "reputation_2")->reputation == (modified ? -3 : 0),
            "rank minus one and level zero retain signed reputation arithmetic without a zero clamp");
        check(source.findActor("NPC_", "reputation_3")->reputation == (modified ? 36 : 6),
            "automatic and explicit NPCs use the same faction reputation rule");
    };
    verify(*content, false);
    verify(*fixture(root / "reputation-settings", true), true);
    BethesdaSession session; setup(session, content); std::string error;
    const auto id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99));
    session.world().find(id)->base = makeTes3RecordKey("NPC_", "reputation_0");
    const auto thread = session.tes3().scripts().start("ReputationProbe", id, error);
    check(thread != 0, error); (void)session.tes3().step(0);
    check(session.tes3().scripts().threads().at(thread).locals.at("baseline").number == 6 &&
          session.tes3().scripts().threads().at(thread).locals.at("observed").number == 11,
        "Get/ModReputation use the calculated faction baseline and retain per-instance changes");
    check(saveOdaiGameAtomic(root / "npc-reputation.json", session, error), error);
    BethesdaSession restored; setup(restored, content); SaveLoadReport report;
    check(loadOdaiGame(root / "npc-reputation.json", restored, {}, report, error) &&
          restored.tes3().referenceOverrides().at(id).locals.at("stat:reputation").number == 11,
        "loading preserves reputation overrides without recalculating away script changes");

}
void testAutomaticTrainers(std::shared_ptr<const Tes3ContentStore> content) {
    const auto* female = content->findActor("NPC_", "auto_female");
    const auto* male = content->findActor("NPC_", "auto_male");
    const auto* zero = content->findActor("NPC_", "auto_zero");
    const auto* high = content->findActor("NPC_", "auto_high");
    check(female && female->attributes.size() == 8 && female->skills.size() == 27,
        "auto NPC imports complete derived attributes and skills");
    if (!female || female->attributes.size() != 8 || female->skills.size() != 27 || !male || !zero || !high) return;
    check(female->attributes.at("strength") == 66 && female->attributes.at("intelligence") == 50 &&
          female->attributes.at("willpower") == 54 && female->attributes.at("luck") == 45 && female->health == 94,
        "automatic female NPC uses race/class, governing-skill growth, ties-to-even and NPC health growth");
    check(male->attributes.at("strength") == 60 && male->attributes.at("intelligence") == 44 && male->health == 89,
        "automatic male NPC uses its gender-specific race attributes and even rounding");
    check(female->skills.at("axe") == 42 && female->skills.at("block") == 28 &&
          female->skills.at("armorer") == 20 && female->skills.at("destruction") == 6 &&
          female->skills.at("illusion") == 13 && female->skills.at("athletics") == 40,
        "automatic skills combine class, specialization, level and race with even rounding");
    check(zero->skills.at("axe") == 34 && zero->skills.at("block") == 18 &&
          high->attributes.at("strength") == 100 && high->skills.at("axe") == 100,
        "automatic NPC level zero and normal caps are handled");
    BethesdaSession session; setup(session, content); std::string error;
    const auto id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x100));
    RuntimeObject trainer; trainer.id = id; trainer.kind = RuntimeObjectKind::Actor; trainer.base = female->record;
    trainer.actorValues.emplace(); trainer.currentSpace = session.world().find(session.playerObject())->currentSpace;
    trainer.actorValues->stamina = trainer.actorValues->maxStamina = female->fatigue;
    check(session.world().addInitialObject(trainer, error), error);
    auto offers = session.tes3TrainingOffers(id);
    check(offers.size() == 3 && offers[0].skill == 6 && offers[1].skill == 9 && offers[2].skill == 8,
        "automatic trainer offers its three highest calculated skills");
    auto& p = session.tes3().playerState();
    const auto before = p.progression;
    check(session.trainTes3PlayerSkill(id, 8, error) && p.numericFilters["athletics"] == 36 &&
          p.progression.levelProgress == before.levelProgress + 1,
        "automatic trainer transaction grants one earned major skill point");
    session.tes3().referenceOverridesForRestore()[id].locals["stat:athletics"] = Tes3Value::fromNumber(35);
    const auto saved = p;
    check(!session.trainTes3PlayerSkill(id, 8, error) && p == saved,
        "runtime trainer stat override takes precedence over automatic baseline");
    auto invalid = *female; invalid.race = "missing_race";
    const auto skills = invalid.skills;
    check(!tes3AutoCalculateNpc(*content, invalid) && invalid.skills == skills,
        "incomplete automatic data does not mutate an existing stat baseline");
}
void testTrainerAbilities(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    const auto* definition = content->findActor("NPC_", "passive_trainer");
    check(definition && definition->inventory.size() == 1 && definition->inventory[0].first.recordType == "SPEL",
        "NPC and racial spell grants import once as owned spells");
    if (!definition) return;
    BethesdaSession session; setup(session, content); std::string error;
    const auto id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99));
    auto* trainer = session.world().find(id); trainer->base = definition->record;
    auto& p = session.tes3().playerState(); p.numericFilters["athletics"] = 96; p.numericFilters["speed"] = 100;
    const auto ledger = p.progression;
    const auto baseline = session.tes3TrainingOffers(id);
    check(baseline.size() == 3 && !baseline[0].eligible, "unmodified trainer cannot teach above its base skill");
    for (const auto& [spell, count] : definition->inventory) trainer->inventory.push_back({spell, count, false});
    const auto modified = session.tes3TrainingOffers(id);
    check(modified.size() == 3 && modified[0].eligible && modified[0].price > baseline[0].price,
        "owned ability fortifies trainer skill limits and seller mercantile pricing");
    check(p.progression == ledger && definition->skills.at("athletics") == 95,
        "passive modifiers leave earned counters and authored base skills unchanged");
    const double hit = session.tes3MeleeHitChance(id, session.playerObject());
    Tes3ActiveSpell passive; passive.spell = makeTes3RecordKey("SPEL", "trainer_ability"); passive.caster = id;
    for (const auto& authored : content->findSpell("trainer_ability")->effects) {
        Tes3ActiveSpellEffect effect; effect.effectId = authored.effectId; effect.skill = authored.skill;
        effect.attribute = authored.attribute; effect.magnitude = authored.magnitudeMin;
        effect.expiresTick = std::numeric_limits<std::uint64_t>::max(); passive.effects.push_back(effect);
    }
    session.tes3().activeSpellsForRestore()[id].push_back(passive);
    const auto represented = session.tes3TrainingOffers(id);
    check(represented[0].price == modified[0].price && session.tes3MeleeHitChance(id, session.playerObject()) == hit,
        "active representation of an owned passive applies its stat modifiers once");
    Tes3ActiveSpell drain; drain.spell = makeTes3RecordKey("SPEL", "temporary_drain");
    Tes3ActiveSpellEffect effect; effect.effectId = 21; effect.skill = 8; effect.magnitude = 40; effect.expiresTick = 10;
    drain.effects.push_back(effect); session.tes3().activeSpellsForRestore()[id].push_back(drain);
    const auto drained = session.tes3TrainingOffers(id);
    const auto athletics = std::find_if(drained.begin(), drained.end(), [](const auto& offer) { return offer.skill == 8; });
    check(athletics != drained.end() && !athletics->eligible, "temporary Drain Skill reduces effective training limit");
    session.tes3().activeSpellsForRestore()[id].back().effects[0].expiresTick = 0;
    check(session.tes3TrainingOffers(id)[0].eligible, "expired Drain Skill stops reducing the trainer limit");
    session.tes3().activeSpellsForRestore()[id].pop_back();
    const auto path = root / "trainer-ability-save.json";
    check(saveOdaiGameAtomic(path, session, error), error);
    BethesdaSession loaded; setup(loaded, content); SaveLoadReport report;
    check(loadOdaiGame(path, loaded, {}, report, error), error);
    check(loaded.tes3TrainingOffers(id)[0].price == modified[0].price && loaded.tes3TrainingOffers(id)[0].eligible,
        "owned and active trainer abilities retain the same offer after save/load");
    // Character creation already incorporated ability Fortify Attribute into
    // the player's baseline; only a later timed modifier belongs above it.
    p.numericFilters["agility"] = 60;
    const double createdChance = session.tes3MeleeHitChance(session.playerObject(), id);
    session.world().find(session.playerObject())->inventory.push_back({passive.spell, 1, false});
    check(session.tes3MeleeHitChance(session.playerObject(), id) == createdChance,
        "creation ability attributes are not counted twice in player gameplay stats");
    Tes3ActiveSpell fortify; fortify.spell = makeTes3RecordKey("SPEL", "timed_fortify");
    effect.effectId = 79; effect.skill = -1; effect.attribute = 3; effect.magnitude = 20; effect.expiresTick = 10;
    fortify.effects.push_back(effect); session.tes3().activeSpellsForRestore()[session.playerObject()].push_back(fortify);
    check(session.tes3MeleeHitChance(session.playerObject(), id) == createdChance + 5,
        "timed Fortify Attribute still applies above the creation baseline");
    check(p.numericFilters["agility"] == 60 && p.progression == ledger,
        "ability and timed stat modifiers never rewrite base stats or advancement counters");
}
void testDerivedDisposition(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    BethesdaSession session; setup(session, content); std::string error;
    const auto id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x101));
    const auto addTrainer = [&](BethesdaSession& target) {
        RuntimeObject trainer = *target.world().find(ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99)));
        trainer.id = id; trainer.base = makeTes3RecordKey("NPC_", "faction_trainer");
        check(target.world().addInitialObject(trainer, error), error);
    };
    addTrainer(session);
    auto& p = session.tes3().playerState(); auto& filters = p.numericFilters;
    const auto ledger = p.progression;
    check(session.tes3DerivedDisposition(id) == 50, "shared race and Personality use imported disposition settings");
    const auto basePrice = session.tes3TrainingOffers(id)[0].price;
    filters["personality"] = 70;
    check(session.tes3DerivedDisposition(id) == 65 && session.tes3TrainingOffers(id)[0].price < basePrice,
        "modified Personality changes disposition and training price");
    p.factionRanks["friend"] = 2;
    check(session.tes3DerivedDisposition(id) == 89, "positive faction reaction uses associated player rank");
    p.factionRanks["enemy"] = 1;
    check(session.tes3DerivedDisposition(id) == 56, "worst faction reaction and its rank take precedence");
    filters["expelled:enemy"] = 1;
    check(session.tes3DerivedDisposition(id) == 89, "expelled factions do not affect disposition");
    p.factionRanks["guild"] = 3;
    check(session.tes3DerivedDisposition(id) == 72, "shared faction uses its authored self-reaction and player rank");
    filters["faction_reaction:guild:guild"] = -3;
    check(session.tes3DerivedDisposition(id) == 50, "script faction reaction modifier contributes to derived disposition");
    filters["expelled:guild"] = 1;
    check(session.tes3DerivedDisposition(id) == 65, "expelled shared membership suppresses faction terms");
    auto* player = session.world().find(session.playerObject());
    player->equipment.drawn = true;
    check(session.tes3DerivedDisposition(id) == 60, "drawn weapon lowers disposition");
    player->inventory.push_back({makeTes3RecordKey("SPEL", "test_disease"), 1, false});
    check(session.tes3DerivedDisposition(id) == 50, "owned common disease lowers disposition once");
    Tes3ActiveSpell charm; charm.spell = makeTes3RecordKey("SPEL", "charm");
    Tes3ActiveSpellEffect effect; effect.effectId = 44; effect.magnitude = 12; effect.expiresTick = 10;
    charm.effects.push_back(effect); session.tes3().activeSpellsForRestore()[id].push_back(charm);
    check(session.tes3DerivedDisposition(id) == 62, "Charm applies to recipient disposition");
    const auto beforeRead = session.tes3().referenceOverrides();
    const auto thread = session.tes3().scripts().start("DispositionProbe", id, error);
    check(thread != 0, error); (void)session.tes3().step(0);
    check(session.tes3().scripts().threads().at(thread).locals.at("observed").number == 62 &&
          session.tes3().referenceOverrides() == beforeRead,
        "GetDisposition reads derived value without installing a zero base override");
    Tes3DialogueActorState speaker; speaker.id = "faction_trainer"; speaker.object = id;
    check(session.startTes3Dialogue(speaker, p).accepted, "authored disposition fixture greeting opens");
    check(session.tes3().addTopic("disposition_gate"), "disposition gate topic exists");
    check(session.selectTes3Topic("disposition_gate").accepted, "dialogue disposition filter shares trainer/native value");
    session.tes3().activeSpellsForRestore()[id][0].effects[0].expiresTick = 0;
    check(!session.selectTes3Topic("disposition_gate").accepted, "expired Charm immediately changes dialogue eligibility");
    session.tes3().activeSpellsForRestore()[id].clear();
    session.tes3().referenceOverridesForRestore()[id].locals["stat:disposition"] = Tes3Value::fromNumber(70.9);
    check(session.tes3DerivedDisposition(id) == 70, "derived disposition truncates fractional base before use");
    check(p.progression == ledger, "disposition modifiers leave earned advancement unchanged");
    check(saveOdaiGameAtomic(root / "disposition.json", session, error), error);
    BethesdaSession loaded; setup(loaded, content); SaveLoadReport report;
    check(loadOdaiGame(root / "disposition.json", loaded, {}, report, error), error);
    check(loaded.tes3DerivedDisposition(id) == 70 &&
          loaded.tes3TrainingOffers(id)[0].price == session.tes3TrainingOffers(id)[0].price,
        "base disposition and all saved modifier inputs retain training price after save/load");
    filters["personality"] = 1000; check(session.tes3DerivedDisposition(id) == 100, "disposition clamps high values");
    filters["personality"] = 0; filters["damage:personality"] = 100;
    session.tes3().referenceOverridesForRestore()[id].locals["stat:disposition"] = Tes3Value::fromNumber(-100);
    check(session.tes3DerivedDisposition(id) == 0, "disposition clamps negative values");
    auto altered = fixture(root / "disposition-settings", true);
    BethesdaSession crimeSession; setup(crimeSession, altered);
    addTrainer(crimeSession);
    crimeSession.tes3().playerState().numericFilters["crimelevel"] = 50;
    check(crimeSession.tes3DerivedDisposition(id) == 45,
        "imported crime modifier and TES3 float intermediate rounding determine disposition");
    const auto changed = crimeSession.tes3().scripts().start("DispositionChange", id, error);
    check(changed != 0, error); (void)crimeSession.tes3().step(0);
    check(crimeSession.tes3().referenceOverrides().at(id).locals.at("stat:disposition").number == 70 &&
          crimeSession.tes3().scripts().threads().at(changed).locals.at("observed").number == 65,
        "Set/ModDisposition change only the base while GetDisposition applies current modifiers");
}
void testPersuasion(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    using Action = BethesdaSession::Tes3PersuasionAction;
    const auto id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x101));
    const auto prepare = [&](BethesdaSession& session, bool fortified, std::shared_ptr<const Tes3ContentStore> source = {}) {
        setup(session, source ? source : content); std::string error;
        RuntimeObject npc = *session.world().find(ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99)));
        npc.id = id; npc.base = makeTes3RecordKey("NPC_", "faction_trainer");
        check(session.world().addInitialObject(npc, error), error);
        Tes3DialogueActorState actor; actor.id = "faction_trainer"; actor.object = id;
        check(session.startTes3Dialogue(actor, session.tes3().playerState()).accepted, "persuasion conversation opens through authored greeting");
        if (fortified) {
            Tes3ActiveSpell fortify; fortify.spell = makeTes3RecordKey("SPEL", "speech_fortify");
            Tes3ActiveSpellEffect effect; effect.effectId = 83; effect.skill = 25; effect.magnitude = 200; effect.expiresTick = 100;
            fortify.effects.push_back(effect); session.tes3().activeSpellsForRestore()[session.playerObject()].push_back(fortify);
        }
        session.setRandomState(1);
    };
    const auto gold = makeTes3RecordKey("MISC", "gold_001");
    const auto money = [&](const BethesdaSession& session, ObjectId actor) {
        int total = 0; for (const auto& item : session.world().find(actor)->inventory) if (item.item == gold) total += item.count; return total;
    };
    for (int action = 0; action < 6; ++action) {
        BethesdaSession session; prepare(session, true); std::string error;
        const auto before = session.tes3().playerState(); const auto random = session.randomState();
        const int initial = session.tes3DerivedDisposition(id), balance = money(session, session.playerObject());
        const double requirement = session.tes3().playerSkillRequirement(25);
        const auto result = session.persuadeTes3Npc(Action(action), error);
        check(result.accepted && result.success && session.randomState() != random, "all six persuasion actions resolve a successful gameplay attempt");
        const auto reply = std::string(action == 0 ? "Admire" : action == 1 ? "Intimidate" : action == 2 ? "Taunt" : "Bribe") + " Success authored reply.";
        check(result.response.accepted && result.response.text == reply,
            "successful persuasion selects the matching authored service response");
        check(std::abs(session.tes3().playerState().progression.skillProgress[25] - 1 / requirement) < 1e-6 &&
              session.tes3().playerState().numericFilters.at("speechcraft") == before.numericFilters.at("speechcraft"),
            "successful persuasion advances only the imported success-use fraction, excluding Fortify Skill");
        const int bribe = action == 3 ? 10 : action == 4 ? 100 : action == 5 ? 1000 : 0;
        check(money(session, session.playerObject()) == balance - bribe && money(session, id) == bribe,
            "successful bribe transfers exactly the offered gold; other actions transfer none");
        check(session.tes3DerivedDisposition(id) == std::clamp(initial + result.temporaryChange, 0, 100),
            "temporary persuasion affects the shared current conversation disposition");
        if (action == 1 || action == 2) {
            const auto& locals = session.tes3().referenceOverrides().at(id).locals;
            check(action == 1 ? locals.at("stat:flee").number > 0 : locals.at("stat:fight").number > 0,
                "successful intimidate/taunt update the appropriate authored AI setting");
        }
        const auto savedState = session.tes3().dialogue();
        const auto path = root / ("persuasion-" + std::to_string(action) + ".json");
        check(saveOdaiGameAtomic(path, session, error), error);
        BethesdaSession loaded; setup(loaded, content); SaveLoadReport report;
        check(loadOdaiGame(path, loaded, {}, report, error), error);
        check(loaded.tes3().dialogue() == savedState && loaded.tes3DerivedDisposition(id) == session.tes3DerivedDisposition(id) &&
              loaded.randomState() == session.randomState(), "pending persuasion changes and random outcome persist across save/load");
        if (action == 0) {
            nlohmann::json saved; { std::ifstream in(path); in >> saved; }
            auto old = saved; old["payload"]["tes3"]["dialogue"].erase("persuasion_temporary");
            old["payload"]["tes3"]["dialogue"].erase("persuasion_permanent"); old["checksum"] = checksum(old["payload"].dump());
            { std::ofstream out(root / "old-persuasion.json"); out << old.dump(); }
            BethesdaSession legacy; setup(legacy, content);
            check(loadOdaiGame(root / "old-persuasion.json", legacy, {}, report, error) &&
                  legacy.tes3().dialogue().persuasionTemporary == 0 && legacy.tes3().dialogue().persuasionPermanent == 0,
                "older dialogue saves initialize missing persuasion changes without inventing history");
            auto bad = saved; bad["payload"]["tes3"]["dialogue"]["persuasion_temporary"] = std::numeric_limits<std::uint64_t>::max();
            bad["checksum"] = checksum(bad["payload"].dump());
            { std::ofstream out(root / "bad-persuasion.json"); out << bad.dump(); }
            const auto hash = loaded.deterministicHash();
            check(!loadOdaiGame(root / "bad-persuasion.json", loaded, {}, report, error) && loaded.deterministicHash() == hash,
                "malformed persuasion counter rejects atomically, including unsigned overflow");
        }
        session.tes3().endDialogue(); loaded.tes3().endDialogue();
        check(loaded.tes3().playerState() == session.tes3().playerState(),
            "closing restored dialogue preserves committed race/class and the full earned progression ledger");
        check(session.tes3DerivedDisposition(id) == std::clamp(initial + result.permanentChange, 0, 100) &&
              loaded.tes3DerivedDisposition(id) == session.tes3DerivedDisposition(id),
            "closing restored conversation applies the permanent portion once and discards temporary change");
        const auto base = session.tes3().referenceOverrides();
        session.tes3().endDialogue(); check(session.tes3().referenceOverrides() == base, "repeated close cannot reapply persuasion");
        const auto rejected = session.tes3().playerState(); const auto stoppedRandom = session.randomState();
        check(!session.persuadeTes3Npc(Action(action), error).accepted && session.tes3().playerState() == rejected &&
              session.randomState() == stoppedRandom, "persuasion outside dialogue awards no use and consumes no roll");
    }
    BethesdaSession factionRated; prepare(factionRated, false); std::string reputationError;
    const auto factionBribe = factionRated.persuadeTes3Npc(Action::Bribe10, reputationError);
    BethesdaSession reputationOverride; prepare(reputationOverride, false);
    reputationOverride.tes3().referenceOverridesForRestore()[id].locals["stat:reputation"] = Tes3Value::fromNumber(47);
    const auto overrideBribe = reputationOverride.persuadeTes3Npc(Action::Bribe10, reputationError);
    check(factionBribe.accepted && factionBribe.success && overrideBribe.accepted && !overrideBribe.success,
        "persuasion uses initialized faction reputation and respects instance reputation overrides");
    BethesdaSession failed; prepare(failed, false); std::string error;
    const auto balance = money(failed, failed.playerObject());
    const double requirement = failed.tes3().playerSkillRequirement(25);
    const auto failure = failed.persuadeTes3Npc(Action::Admire, error);
    check(failure.accepted && !failure.success && std::abs(failed.tes3().playerState().progression.skillProgress[25] - 2 / requirement) < 1e-6,
        "failed persuasion uses the imported failure-use value rather than success value");
    check(failure.response.accepted && failure.response.text == "Admire Fail authored reply.",
        "failed admire selects its authored failure response");
    check(money(failed, failed.playerObject()) == balance && failure.temporaryChange < 0, "failed admire lowers disposition without charging gold");
    failed.setRandomState(4);
    const auto bribeFailure = failed.persuadeTes3Npc(Action::Bribe10, error);
    check(bribeFailure.accepted && !bribeFailure.success && money(failed, failed.playerObject()) == balance && money(failed, id) == 0,
        "failed affordable bribe consumes an attempt but transfers no gold");
    check(bribeFailure.response.accepted && bribeFailure.response.text == "Bribe Fail authored reply.",
        "failed bribe selects its authored failure response");
    const auto replyState = failed.tes3().dialogue();
    check(!failed.tes3().selectPersuasionResponse("disposition_gate").accepted &&
          !failed.selectTes3Topic("Admire Success").accepted && failed.tes3().dialogue() == replyState,
        "persuasion replies remain service outcomes and ordinary topics cannot enter that path");
    failed.world().find(failed.playerObject())->inventory[0].count = 0;
    const auto before = failed.tes3().playerState(); const auto random = failed.randomState(); const auto disposition = failed.tes3().dialogue();
    check(!failed.persuadeTes3Npc(Action::Bribe10, error).accepted && failed.randomState() == random &&
          failed.tes3().playerState() == before && failed.tes3().dialogue() == disposition,
        "unaffordable bribe is rejected before progression, disposition or random state changes");
    failed.tes3().dialogueForRestore().choices.push_back({"An authored choice", 1});
    check(!failed.persuadeTes3Npc(Action::Taunt, error).accepted, "authored dialogue choices block persuasion");
    failed.tes3().dialogueForRestore().choices.clear();
    failed.world().find(id)->transform.position[0] = 1000;
    check(!failed.persuadeTes3Npc(Action::Admire, error).accepted, "out-of-reach conversation cannot award Speechcraft");
    for (int mode = 1; mode <= 3; ++mode) {
        BethesdaSession scripted; prepare(scripted, true);
        scripted.tes3().scripts().globals()["persuasion_response_mode"] = Tes3Value::fromNumber(mode);
        const auto reputation = scripted.tes3().playerState().numericFilters["reputation"];
        const auto attempt = scripted.persuadeTes3Npc(Action::Admire, error);
        const auto ledger = scripted.tes3().playerState().progression;
        const auto rolled = scripted.randomState();
        check(attempt.accepted && attempt.success && ledger.skillProgress[25] > 0,
            "authored result scripts execute after the gameplay attempt commits");
        if (mode == 1) {
            check(attempt.response.accepted && attempt.response.choices.size() == 1 &&
                  scripted.tes3().playerState().numericFilters["reputation"] == reputation + 10,
                "persuasion MessageBox suspends its result transaction");
            check(!scripted.persuadeTes3Npc(Action::Admire, error).accepted && scripted.randomState() == rolled &&
                  scripted.tes3().playerState().progression == ledger,
                "suspended reply blocks another attempt without consuming random or skill progress");
            check(scripted.answerTes3Choice(0).accepted &&
                  scripted.tes3().playerState().numericFilters["reputation"] == reputation + 30 &&
                  scripted.tes3().playerState().progression == ledger && scripted.randomState() == rolled,
                "reply continuation commits once without replaying persuasion");
            check(!scripted.answerTes3Choice(0).accepted &&
                  scripted.tes3().playerState().numericFilters["reputation"] == reputation + 30,
                "repeated continuation cannot repeat result effects");
        } else if (mode == 2) {
            check(!attempt.response.accepted && !attempt.response.diagnostics.empty() &&
                  scripted.tes3().playerState().numericFilters["reputation"] == reputation &&
                  scripted.tes3().dialogue().persuasionTemporary == attempt.temporaryChange,
                "failed reply rolls back script effects while retaining the resolved gameplay attempt");
            scripted.tes3().scripts().globals()["persuasion_response_mode"] = Tes3Value::fromNumber(0);
            check(scripted.persuadeTes3Npc(Action::Admire, error).response.accepted,
                "failed reply releases the result transaction for a subsequent attempt");
        } else {
            check(attempt.response.accepted && attempt.response.goodbye && !scripted.tes3().dialogue().active &&
                  scripted.tes3().playerState().numericFilters["reputation"] == reputation + 1 &&
                  scripted.tes3().dialogue().persuasionPermanent == 0,
                "authored Goodbye closes the conversation and commits permanent persuasion");
            const auto overrides = scripted.tes3().referenceOverrides();
            scripted.tes3().endDialogue();
            check(scripted.tes3().referenceOverrides() == overrides,
                "closing again after authored Goodbye cannot duplicate permanent changes");
        }
    }
    BethesdaSession cancelled; prepare(cancelled, true);
    cancelled.tes3().scripts().globals()["persuasion_response_mode"] = Tes3Value::fromNumber(1);
    const auto oldReputation = cancelled.tes3().playerState().numericFilters["reputation"];
    const auto cancelledAttempt = cancelled.persuadeTes3Npc(Action::Admire, error);
    const auto cancelledLedger = cancelled.tes3().playerState().progression;
    const auto cancelledRoll = cancelled.randomState();
    cancelled.tes3().endDialogue();
    check(cancelledAttempt.response.accepted && cancelled.tes3().playerState().numericFilters["reputation"] == oldReputation &&
          cancelled.tes3().playerState().progression == cancelledLedger && cancelled.randomState() == cancelledRoll &&
          cancelled.tes3().referenceOverrides().at(id).locals.at("stat:disposition").number == 50 + cancelledAttempt.permanentChange,
        "closing a suspended reply rolls back result effects and commits the prior persuasion attempt once");
    BethesdaSession capped; prepare(capped, true); capped.tes3().playerState().numericFilters["speechcraft"] = 100;
    const auto cappedLedger = capped.tes3().playerState().progression;
    check(capped.persuadeTes3Npc(Action::Admire, error).accepted && capped.tes3().playerState().progression == cappedLedger,
        "capped Speechcraft still permits gameplay persuasion without advancement");
    BethesdaSession major; prepare(major, true);
    auto custom = *tes3ClassDefinition(*content, "test_class"); custom.major[0] = 25;
    major.tes3().playerState().progression.customClass = custom;
    check(major.initializeTes3PlayerProgression(error), error);
    major.tes3().playerState().progression.skillProgress[25] = .999;
    const auto majorBase = major.tes3().playerState().numericFilters.at("speechcraft");
    check(major.persuadeTes3Npc(Action::Admire, error).accepted &&
          major.tes3().playerState().numericFilters.at("speechcraft") == majorBase + 1 &&
          major.tes3().playerState().progression.levelProgress == 1 &&
          major.tes3().playerState().progression.attributeIncreases[4] == 1,
        "actual persuasion threshold feeds major-skill level progress and governing attribute bonus");
    const auto modified = fixture(root / "persuasion-settings", true);
    BethesdaSession boundary; prepare(boundary, false, modified);
    boundary.tes3().referenceOverridesForRestore()[id].locals["stat:speechcraft"] = Tes3Value::fromNumber(1000);
    const auto equal = boundary.persuadeTes3Npc(Action::Admire, error);
    check(equal.accepted && equal.success && equal.temporaryChange == 20 && equal.permanentChange == 10,
        "roll equal to imported minimum chance succeeds; temporary multiplier does not inflate permanent change");
    boundary.tes3().endDialogue();
    check(boundary.tes3DerivedDisposition(id) == 60, "only permanent part remains after ending amplified temporary persuasion");
    BethesdaSession marginal; prepare(marginal, false, modified);
    marginal.tes3().referenceOverridesForRestore()[id].locals["stat:speechcraft"] = Tes3Value::fromNumber(1000);
    const auto intimidation = marginal.persuadeTes3Npc(Action::Intimidate, error);
    check(intimidation.accepted && intimidation.success && intimidation.temporaryChange == 0 && intimidation.permanentChange == -10,
        "vanilla marginal intimidation succeeds without temporary disposition gain and preserves permanent loss");
}
void testCustomClassInput(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    BethesdaSession session; setup(session, content); std::string error;
    Tes3ClassUiFlow flow;
    const auto before = session.tes3().playerState();
    check(!flow.update({false, false, false, true}, session) && flow.stage() == 0,
        "back from first class stage does not commit");
    // Choose Magic, Strength, Intelligence, then five major and minor skills.
    (void)flow.update({false, true, false, false}, session);
    for (int stage = 0; stage < 13; ++stage) {
        check(!flow.update({false, false, true, false}, session), "class draft remains uncommitted");
        check(session.tes3().playerState() == before, "custom-class choices do not mutate player stats or ledger");
    }
    check(flow.stage() == 13 && validTes3ProgressionClass(flow.draft()), "wizard constructs a valid distinct class");
    check(flow.draft().specialization == 1 && flow.draft().major[0] == 0 && flow.draft().minor[0] == 5,
        "class wizard retains the chosen specialization and groups");
    (void)flow.update({false, false, false, true}, session);
    check(flow.stage() == 12 && flow.draft().minor[4] == -1, "back from review clears the last choice");
    (void)flow.update({false, true, true, false}, session);
    check(flow.draft().minor[4] == 10, "class choice can be revised before committing");
    const auto expected = flow.draft();
    check(flow.update({false, false, true, false}, session), "final review commits class");
    auto& p = session.tes3().playerState();
    check(p.progression.customClass == expected && p.numericFilters == before.numericFilters,
        "class commit preserves existing stats until character review");
    check(session.initializeTes3PlayerProgression(error), error);
    check(p.progression.customClass == expected && p.numericFilters["strength"] == 50 &&
          p.numericFilters["intelligence"] == 50 && p.numericFilters["endurance"] == 40,
        "character review uses custom favored attributes");
    check(p.numericFilters["block"] == 30 && p.numericFilters["armorer"] == 35 &&
          p.numericFilters["longblade"] == 15 && p.numericFilters["destruction"] == 20,
        "custom major/minor/specialization determine starting skills");
    check(p.progression.levelProgress == 0 && p.progression.attributeIncreases == std::array<int, 8>{},
        "custom-class character creation does not earn increases");
    check(session.tes3().advancePlayerSkill(0, Tes3SkillAdvanceSource::Training, error) &&
          session.tes3().advancePlayerSkill(10, Tes3SkillAdvanceSource::Book, error) &&
          session.tes3().advancePlayerSkill(20, Tes3SkillAdvanceSource::Usage, error) &&
          p.progression.levelProgress == 2, "earned increases use committed custom-class membership");
    const auto committed = p;
    check(!flow.update({false, false, true, false}, session) && p == committed,
        "repeated confirmation does not recommit or reset progression");
    check(saveOdaiGameAtomic(root / "custom-class.json", session, error), error);
    BethesdaSession loaded; setup(loaded, content); SaveLoadReport report;
    check(loadOdaiGame(root / "custom-class.json", loaded, {}, report, error) &&
          loaded.tes3().playerState() == committed,
        "input-created custom class and earned counters persist across save/load");
}
void testRestEffectTime(std::shared_ptr<const Tes3ContentStore> content) {
    BethesdaSession session; setup(session, content); std::string error;
    auto& runtime = session.tes3();
    auto& p = runtime.playerState();
    auto& stats = p.numericFilters;
    stats["magicka"] = 0; session.synchronizeTes3PlayerValues();
    runtime.scripts().globals()["timescale"] = Tes3Value::fromNumber(30);
    const auto now = session.clock().tick();
    Tes3ActiveSpell spell;
    spell.effects = {{136, -1, -1, 1, now + 3600},
                     {79, -1, 5, 20, now + 7200},
                     {79, -1, 0, 10, now + 14400},
                     {79, -1, 1, 10, std::numeric_limits<std::uint64_t>::max()}};
    runtime.activeSpellsForRestore()[session.playerObject()].push_back(spell);
    const auto trainer = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99));
    runtime.activeSpellsForRestore()[trainer].push_back(spell);
    const auto before = runtime.activeSpells();
    session.world().find(session.playerObject())->currentSpace.cell = makeTes3RecordKey("CELL", "no_sleep");
    check(!session.restTes3Player(1, true, error) && runtime.activeSpells() == before,
        "rejected rest does not expire effects");
    session.world().find(session.playerObject())->currentSpace.cell = makeTes3RecordKey("CELL", "wilderness");
    check(session.restTes3Player(1, true, error), error);
    check(session.clock().tick() == now, "rest effect aging does not advance simulation ticks");
    check(std::abs(stats["magicka"] - 3.75) < 1e-5,
        "Stunted Magicka blocks only its remaining half hour of sleep");
    check(session.modifiedTes3Stat(session.playerObject(), "endurance") == 50 &&
          session.modifiedTes3Stat(session.playerObject(), "strength") == 60,
        "sleep expires a short effect and retains a longer one");
    check(runtime.activeSpells().at(trainer).front().effects ==
          runtime.activeSpells().at(session.playerObject()).front().effects,
        "rest also ages active effects on other actors");
    check(session.restTes3Player(1, false, error), error);
    check(session.modifiedTes3Stat(session.playerObject(), "strength") == 50 &&
          session.modifiedTes3Stat(session.playerObject(), "intelligence") == 50,
        "waiting ages timed effects while retaining permanent effects");
    check(std::abs(stats["magicka"] - 3.75) < 1e-5, "waiting does not restore magicka");
    runtime.scripts().globals()["timescale"] = Tes3Value::fromNumber(0);
    runtime.activeSpellsForRestore()[session.playerObject()].front().effects.push_back({79, -1, 5, 20, now + 216000});
    check(session.restTes3Player(1, false, error) && session.modifiedTes3Stat(session.playerObject(), "endurance") == 50,
        "zero time scale uses real elapsed seconds for rest effects");
    runtime.activeSpellsForRestore()[session.playerObject()].front().effects.push_back(
        {136, -1, -1, 1, std::numeric_limits<std::uint64_t>::max()});
    check(session.restTes3Player(1, true, error) && std::abs(stats["magicka"] - 3.75) < 1e-5,
        "permanent Stunted Magicka blocks the entire sleep interval");
}
void testGameplayFalls(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    const std::vector<odai::math::Vector3> floor{{-10000,0,-10000},{10000,0,-10000},{10000,0,10000},{-10000,0,10000}};
    const std::vector<std::uint32_t> indices{0,1,2,0,2,3};
    std::string error;
    const auto prepare = [&](BethesdaSession& session, float height) {
        setup(session, content);
        check(session.physics().addStaticCollision(ObjectId::runtime(123), floor, indices, error), error);
        PhysicsCharacterConfig config; config.position = {0,height,0};
        check(session.registerActorController(session.playerObject(), config, error), error);
    };
    const auto land = [&](BethesdaSession& session) {
        for (int i = 0; i < 600; ++i) {
            (void)session.advance(1.0 / 60);
            if (session.physics().characterState(session.playerObject())->landed) return;
        }
        check(false, "fall fixture lands on physical floor");
    };
    BethesdaSession shortFall; prepare(shortFall, 100);
    const float shortHealth = shortFall.world().find(shortFall.playerObject())->actorValues->health;
    land(shortFall);
    check(shortFall.tes3().playerState().progression.skillProgress[20] == 0 &&
          shortFall.world().find(shortFall.playerObject())->actorValues->health == shortHealth,
        "harmless landing does not award Acrobatics fall use");
    BethesdaSession damaging; prepare(damaging, 600);
    damaging.tes3().playerState().numericFilters["acrobatics"] = 40;
    const float priorHealth = damaging.world().find(damaging.playerObject())->actorValues->health;
    for (int i = 0; i < 30; ++i) (void)damaging.advance(1.0 / 60);
    check(damaging.physicsSnapshots().front().fallDistanceUnits > 0, "airborne fall accumulates measured distance");
    check(saveOdaiGameAtomic(root / "fall.json", damaging, error), error);
    BethesdaSession resumed; prepare(resumed, 600); SaveLoadReport report;
    check(loadOdaiGame(root / "fall.json", resumed, {}, report, error), error);
    check(resumed.physicsSnapshots() == damaging.physicsSnapshots(), "mid-fall save preserves distance and physical state");
    nlohmann::json savedFall; { std::ifstream in(root / "fall.json"); in >> savedFall; }
    auto legacy = savedFall;
    for (auto& character : legacy["payload"]["physics"]) character.erase("fall_distance");
    legacy["checksum"] = checksum(legacy["payload"].dump());
    { std::ofstream out(root / "old-fall.json"); out << legacy.dump(); }
    BethesdaSession oldFall; prepare(oldFall, 600);
    check(loadOdaiGame(root / "old-fall.json", oldFall, {}, report, error) &&
          oldFall.physicsSnapshots().front().fallDistanceUnits == 0,
        "older physics saves initialize missing fall distance without inventing a fall");
    auto malformed = savedFall; malformed["payload"]["physics"][0]["fall_distance"] = -1;
    malformed["checksum"] = checksum(malformed["payload"].dump());
    { std::ofstream out(root / "bad-fall.json"); out << malformed.dump(); }
    const auto beforeBad = resumed.deterministicHash();
    check(!loadOdaiGame(root / "bad-fall.json", resumed, {}, report, error) && resumed.deterministicHash() == beforeBad,
        "negative fall distance is rejected without session mutation");
    land(damaging); land(resumed);
    const auto physical = damaging.physics().characterState(damaging.playerObject());
    const float health = damaging.world().find(damaging.playerObject())->actorValues->health;
    const double expected = std::max(0.0, double(physical->fallDistanceUnits) - 400 - 60) * .07 * .85 * .6875;
    check(std::abs((priorHealth - health) - expected) < .001, "fall damage uses measured height, Acrobatics, and fatigue");
    const double progress = damaging.tes3().playerState().progression.skillProgress[20];
    check(progress > 0 && std::abs(progress - 2.0 / damaging.tes3().playerSkillRequirement(20)) < 1e-6,
        "eligible damaging landing uses the imported fall-use value once");
    check(std::abs(resumed.world().find(resumed.playerObject())->actorValues->health - health) < .01 &&
          resumed.tes3().playerState().progression.skillProgress[20] == progress,
        "mid-fall load preserves landing damage and skill-use outcome");
    for (int i = 0; i < 30; ++i) (void)damaging.advance(1.0 / 60);
    check(damaging.tes3().playerState().progression.skillProgress[20] == progress &&
          damaging.world().find(damaging.playerObject())->actorValues->health == health,
        "continued grounded simulation does not repeat landing damage or skill use");
    BethesdaSession knockdown; prepare(knockdown, 600); land(knockdown);
    check(knockdown.tes3().playerState().progression.skillProgress[20] == 0 &&
          knockdown.world().find(knockdown.playerObject())->actorValues->health < shortHealth,
        "landing above the Acrobatics/fatigue knockdown threshold damages without advancing skill");
    BethesdaSession protectedFall; prepare(protectedFall, 600);
    Tes3ActiveSpell slow; slow.effects.push_back({11, -1, -1, 1, 100000});
    protectedFall.tes3().activeSpellsForRestore()[protectedFall.playerObject()].push_back(slow);
    land(protectedFall);
    check(protectedFall.tes3().playerState().progression.skillProgress[20] == 0 &&
          protectedFall.world().find(protectedFall.playerObject())->actorValues->health == shortHealth,
        "Slow Fall does not produce damaging-fall progression");
    BethesdaSession jumpProtected; prepare(jumpProtected, 600);
    Tes3ActiveSpell jump; jump.effects.push_back({9, -1, -1, 500, 100000});
    jumpProtected.tes3().activeSpellsForRestore()[jumpProtected.playerObject()].push_back(jump);
    land(jumpProtected);
    check(jumpProtected.tes3().playerState().progression.skillProgress[20] == 0 &&
          jumpProtected.world().find(jumpProtected.playerObject())->actorValues->health == shortHealth,
        "Jump magnitude subtracts from fall damage distance");
    BethesdaSession waterLanding; prepare(waterLanding, 600);
    waterLanding.world().find(waterLanding.playerObject())->currentSpace.cell = makeTes3RecordKey("CELL", "water_interior");
    land(waterLanding);
    check(waterLanding.tes3().playerState().progression.skillProgress[20] == 0 &&
          waterLanding.world().find(waterLanding.playerObject())->actorValues->health == shortHealth,
        "landing underwater does not produce damaging-fall progression");
}
void testArmorContacts(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    std::array<int, 9> frequencies{};
    for (unsigned roll = 0; roll < 100; ++roll) ++frequencies[tes3ArmorHitSlot(roll)];
    check(frequencies == std::array<int,9>{10,30,10,10,10,10,5,5,10} && tes3ArmorHitSlot(100) == -1,
        "armor hit slot distribution covers the TES3 body weights");
    for (int kind = 0; kind < 3; ++kind) for (int type = 0; type < 11; ++type) {
        const auto def = tes3ArmorDefinition(*content, "armor_" + std::to_string(kind) + "_" + std::to_string(type));
        check(def && def->skill == (kind == 0 ? 21 : kind == 1 ? 2 : 3) &&
              def->slot == (type == 9 ? 6 : type == 10 ? 7 : type), "armor records classify weight and bracer slots");
    }
    check(!tes3ArmorDefinition(*content, "bad_armor"), "malformed armor is not a progression producer");
    for (int edge = 0; edge < 3; ++edge) {
        const auto def = tes3ArmorDefinition(*content, "armor_edge_" + std::to_string(edge));
        check(def && def->skill == (edge == 0 ? 21 : edge == 1 ? 2 : 3),
            "armor classification preserves TES3 weight-boundary tolerance");
    }
    std::string error;
    for (int kind = -1; kind < 3; ++kind) {
        BethesdaSession session; setup(session, content);
        auto* player = session.world().find(session.playerObject());
        if (kind >= 0) for (int type = 0; type < 9; ++type) {
            const auto key = makeTes3RecordKey("ARMO", "armor_" + std::to_string(kind) + "_" + std::to_string(type));
            player->inventory.push_back({key, 1, false});
            check(session.equipActorItem(session.playerObject(), key, true, false, error), error);
        }
        (void)session.advance(1.0 / 60);
        const std::vector<odai::math::Vector3> floor{{-1000,0,-1000},{1000,0,-1000},{1000,0,1000},{-1000,0,1000}};
        const std::vector<std::uint32_t> triangles{0,1,2,0,2,3};
        check(session.physics().addStaticCollision(ObjectId::runtime(991), floor, triangles, error), error);
        PhysicsCharacterConfig config; config.position = {0,10,0};
        check(session.registerActorController(session.playerObject(), config, error), error);
        const auto trainer = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99));
        session.tes3().referenceOverridesForRestore()[trainer].locals["stat:handtohand"] = Tes3Value::fromNumber(100);
        config.position = {100,10,0}; check(session.registerActorController(trainer, config, error), error);
        auto& runtime = session.tes3(); auto& p = runtime.playerState();
        const int skill = kind == -1 ? 17 : kind == 0 ? 21 : kind == 1 ? 2 : 3;
        const auto before = p.progression;
        const auto miss = session.performMeleeAttack(trainer, {1,0,0}, 1, 180);
        check(!miss.hit && p.progression == before, "missed contact does not advance defensive skills");
        for (int i = 0; i < 30; ++i) (void)session.advance(1.0 / 60);
        Tes3ActiveSpell blinded; blinded.effects.push_back({47,-1,-1,200,100000});
        runtime.activeSpellsForRestore()[trainer].push_back(blinded);
        const auto failedRoll = session.performMeleeAttack(trainer, {-1,0,0}, 1, 180);
        check(failedRoll.accepted && !failedRoll.hit && p.progression == before,
            "incoming contact failing hit eligibility does not award armor or Unarmored use");
        runtime.activeSpellsForRestore().erase(trainer);
        for (int i = 0; i < 30; ++i) (void)session.advance(1.0 / 60);
        const auto hit = session.performMeleeAttack(trainer, {-1,0,0}, 1, 180);
        check(hit.hit && hit.target == session.playerObject(), "armor fixture resolves incoming melee contact");
        check(std::abs(p.progression.skillProgress[skill] - 1.0 / runtime.playerSkillRequirement(skill)) < 1e-6,
            "resolved health hit advances exactly the equipped armor category or Unarmored");
        for (int other : {2,3,17,21}) if (other != skill)
            check(p.progression.skillProgress[other] == 0, "one hit does not advance other armor categories");
        (void)session.advance(1.0 / 60);
        check(saveOdaiGameAtomic(root / "armor.json", session, error), error);
        BethesdaSession restored; setup(restored, content); SaveLoadReport report;
        check(loadOdaiGame(root / "armor.json", restored, {}, report, error) && restored.tes3().playerState() == p,
            "defensive skill use persists across gameplay save/load");
        const auto earned = p.progression;
        for (int i = 0; i < 30; ++i) (void)session.advance(1.0 / 60);
        check(p.progression == earned, "contact does not repeatedly award armor use on later ticks");
    }
    BethesdaSession slots; setup(slots, content);
    const auto gauntlet = makeTes3RecordKey("ARMO", "armor_0_6"), bracer = makeTes3RecordKey("ARMO", "armor_0_9");
    slots.world().find(slots.playerObject())->inventory.push_back({gauntlet,1,false});
    slots.world().find(slots.playerObject())->inventory.push_back({bracer,1,false});
    check(slots.equipActorItem(slots.playerObject(), gauntlet, true, false, error), error);
    (void)slots.advance(1.0 / 60);
    check(slots.equipActorItem(slots.playerObject(), bracer, true, false, error), error);
    (void)slots.advance(1.0 / 60);
    const auto& inventory = slots.world().find(slots.playerObject())->inventory;
    check(std::count_if(inventory.begin(), inventory.end(), [](const auto& e) { return e.equipped; }) == 1,
        "bracers replace gauntlets in the same equipment slot");
    for (auto& entry : slots.world().find(slots.playerObject())->inventory)
        if (entry.item == bracer) entry.equipmentSlots = 1ull << 9;
    check(saveOdaiGameAtomic(root / "old-bracer.json", slots, error), error);
    BethesdaSession oldBracer; setup(oldBracer, content); SaveLoadReport report;
    check(loadOdaiGame(root / "old-bracer.json", oldBracer, {}, report, error), error);
    check(oldBracer.equipActorItem(oldBracer.playerObject(), gauntlet, true, false, error), error);
    (void)oldBracer.advance(1.0 / 60);
    const auto& migrated = oldBracer.world().find(oldBracer.playerObject())->inventory;
    check(std::count_if(migrated.begin(), migrated.end(), [](const auto& e) { return e.equipped; }) == 1 &&
        std::find_if(migrated.begin(), migrated.end(), [&](const auto& e) { return e.item == gauntlet; })->equipped,
        "legacy bracer equipment slots migrate to the gauntlet slot on load");
}
void testWeaponHitEligibility(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    constexpr std::array<int,12> skillIds{22,5,5,4,4,4,7,6,6,23,23,23};
    for (int type = 0; type < 14; ++type) {
        const auto skill = tes3WeaponSkill(*content, "weapon_" + std::to_string(type));
        check(type < 12 ? skill && *skill == skillIds[type] : !skill,
            "TES3 weapon record type maps to its actual skill; ammunition is not a weapon use");
    }
    std::string error;
    const auto trainer = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99));
    const auto controllers = [&](BethesdaSession& session) {
        PhysicsCharacterConfig config; config.position = {0,10,0};
        check(session.registerActorController(session.playerObject(), config, error), error);
        config.position = {100,10,0}; check(session.registerActorController(trainer, config, error), error);
    };
    BethesdaSession chance; setup(chance, content); controllers(chance);
    auto& stats = chance.tes3().playerState().numericFilters;
    stats["handtohand"] = 20;
    check(chance.tes3MeleeHitChance(chance.playerObject(), trainer) == 18,
        "hit chance combines skill, both actors' agility/luck and fatigue, then rounds");
    const float maximumFatigue = chance.world().find(chance.playerObject())->actorValues->maxStamina;
    chance.world().find(chance.playerObject())->actorValues->stamina = maximumFatigue * .5f;
    check(chance.tes3MeleeHitChance(chance.playerObject(), trainer) == 10, "fatigue reduces hit chance");
    chance.world().find(chance.playerObject())->actorValues->stamina = maximumFatigue;
    auto& spells = chance.tes3().activeSpellsForRestore();
    Tes3ActiveSpell sanctuary; sanctuary.effects.push_back({42,-1,-1,30,100000});
    spells[trainer].push_back(sanctuary);
    check(chance.tes3MeleeHitChance(chance.playerObject(), trainer) == -13, "Sanctuary raises evasion");
    Tes3ActiveSpell paralyze; paralyze.effects.push_back({45,-1,-1,1,100000}); spells[trainer].push_back(paralyze);
    check(chance.tes3MeleeHitChance(chance.playerObject(), trainer) == 40, "paralysis removes normal evasion");
    spells.clear();
    Tes3ActiveSpell blind; blind.effects.push_back({47,-1,-1,200,100000}); spells[chance.playerObject()].push_back(blind);
    const auto before = chance.tes3().playerState().progression;
    const float health = chance.world().find(trainer)->actorValues->health;
    const auto missed = chance.performMeleeAttack(chance.playerObject(), {1,0,0}, 1, 180);
    check(missed.accepted && !missed.hit && missed.target == trainer &&
          chance.tes3().playerState().progression == before,
        "geometric contact that fails hit chance grants no weapon or defensive skill use");
    (void)chance.advance(1.0 / 60);
    check(chance.world().find(trainer)->actorValues->health == health, "failed hit chance applies no damage");
    for (int type = 0; type < 12; ++type) {
        BethesdaSession session; setup(session, content);
        const auto weapon = makeTes3RecordKey("WEAP", "weapon_" + std::to_string(type));
        session.world().find(session.playerObject())->inventory.push_back({weapon,1,false});
        check(session.equipActorItem(session.playerObject(), weapon, true, false, error), error);
        (void)session.advance(1.0 / 60); controllers(session);
        Tes3ActiveSpell accurate; accurate.effects.push_back({117,-1,-1,200,100000});
        session.tes3().activeSpellsForRestore()[session.playerObject()].push_back(accurate);
        const auto baseline = session.tes3().playerState().progression;
        const auto hit = session.performMeleeAttack(session.playerObject(), {1,0,0}, 1, 180);
        const auto& progress = session.tes3().playerState().progression;
        if (type >= 9) {
            check(!hit.hit && progress == baseline, "bow/crossbow/thrown equipment does not earn Marksman through a melee action");
        } else {
            check(hit.hit && progress.skillProgress[skillIds[type]] > 0, "resolved equipped weapon contact advances its imported weapon skill");
            for (int skill = 0; skill < 27; ++skill) if (skill != skillIds[type])
                check(progress.skillProgress[skill] == baseline.skillProgress[skill], "weapon contact does not advance another skill");
        }
    }
    BethesdaSession replay; setup(replay, content); controllers(replay);
    check(saveOdaiGameAtomic(root / "hit-chance.json", replay, error), error);
    BethesdaSession loaded; setup(loaded, content); controllers(loaded); SaveLoadReport report;
    check(loadOdaiGame(root / "hit-chance.json", loaded, {}, report, error), error);
    const auto original = replay.performMeleeAttack(replay.playerObject(), {1,0,0}, 1, 180);
    const auto repeated = loaded.performMeleeAttack(loaded.playerObject(), {1,0,0}, 1, 180);
    check(original.hit == repeated.hit && original.target == repeated.target &&
          replay.tes3().playerState().progression == loaded.tes3().playerState().progression,
        "save/load preserves deterministic hit outcome and earned progression");
    const auto* creature = content->findActor("CREA", "test_creature");
    check(creature && creature->attributes.at("agility") == 40 && creature->skills.at("combat") == 60 &&
          creature->skills.at("magic") == 50 && creature->skills.at("stealth") == 40,
        "creature hit eligibility imports authored attributes and specialization skills");
}
void testGameplaySwimming(std::shared_ptr<const Tes3ContentStore> content, const fs::path& root) {
    const auto prepare = [&](BethesdaSession& session) {
        setup(session, content); std::string error;
        session.world().find(session.playerObject())->currentSpace.cell = makeTes3RecordKey("CELL", "water_interior");
        PhysicsCharacterConfig config; config.position = {0,-200,0};
        check(session.registerActorController(session.playerObject(), config, error), error);
    };
    BethesdaSession blocked; prepare(blocked); std::string wallError;
    const std::vector<odai::math::Vector3> wall{{50,-500,-1000},{50,500,-1000},{50,500,1000},{50,-500,1000}};
    const std::vector<std::uint32_t> faces{0,1,2,0,2,3};
    check(blocked.physics().addStaticCollision(ObjectId::runtime(992), wall, faces, wallError), wallError);
    PhysicsCharacterInput againstWall; againstWall.desiredVelocity = {100,0,0};
    for (int i = 0; i < 120; ++i) {
        (void)blocked.setActorControllerInput(blocked.playerObject(), againstWall);
        (void)blocked.advance(1.0 / 60);
    }
    const auto blockedProgress = blocked.tes3().playerState().progression;
    check(blocked.physics().characterState(blocked.playerObject())->blocked && blockedProgress.skillProgress[8] > 0,
        "swimming contacts a real wall after awarding only the approach movement");
    for (int i = 0; i < 120; ++i) {
        (void)blocked.setActorControllerInput(blocked.playerObject(), againstWall);
        (void)blocked.advance(1.0 / 60);
    }
    check(blocked.tes3().playerState().progression == blockedProgress,
        "held swimming input against a wall grants no further skill use");
    BethesdaSession session; prepare(session); std::string error;
    auto& p = session.tes3().playerState();
    const auto step = [&](PhysicsCharacterInput input, int ticks, bool running = false) {
        for (int i = 0; i < ticks; ++i) {
            (void)session.setActorControllerInput(session.playerObject(), input, running);
            (void)session.advance(1.0 / 60);
        }
    };
    step({}, 120);
    auto physical = session.physics().characterState(session.playerObject());
    check(physical->swimming && std::abs(physical->position.y + 200) < .01 && p.progression.skillProgress[8] == 0,
        "immersed idle player floats without gravity or Athletics use");
    const double requirement = session.tes3().playerSkillRequirement(8);
    PhysicsCharacterInput move; move.desiredVelocity = {100,0,0}; step(move, 120, true);
    check(std::abs(p.progression.skillProgress[8] - 4 / requirement) < 1e-5 && p.progression.skillProgress[20] == 0,
        "resolved swimming uses imported swim index once per elapsed second and excludes run/jump awards");
    physical = session.physics().characterState(session.playerObject());
    check(physical->position.x > 199 && std::abs(physical->position.y + 200) < .01,
        "swimming moves through actual collision-backed character controller without sinking");
    move.desiredVelocity = {0,100,0}; step(move, 180);
    physical = session.physics().characterState(session.playerObject());
    check(physical->swimming && std::abs(physical->position.y - (100 - 128 * .9)) < .02 &&
          p.progression.skillProgress[20] == 0,
        "swim ascent reaches imported height scale and cannot launch a jump above the surface");
    const auto surfaceProgress = p.progression.skillProgress[8]; step(move, 120);
    check(std::abs(p.progression.skillProgress[8] - surfaceProgress) < 1e-6,
        "holding ascent at the surface awards no stationary swim use");
    step({}, 1);
    check(saveOdaiGameAtomic(root / "swimming.json", session, error), error);
    BethesdaSession restored; prepare(restored); SaveLoadReport report;
    check(loadOdaiGame(root / "swimming.json", restored, {}, report, error), error);
    for (int i = 0; i < 60; ++i) {
        move.desiredVelocity = {100,-10,0};
        (void)session.setActorControllerInput(session.playerObject(), move);
        (void)restored.setActorControllerInput(restored.playerObject(), move);
        (void)session.advance(1.0 / 60); (void)restored.advance(1.0 / 60);
    }
    check(restored.tes3().playerState().progression == p.progression &&
          std::abs(restored.physics().characterState(restored.playerObject())->position.y -
              session.physics().characterState(session.playerObject())->position.y) < .01,
        "save/load derives swimming again from saved cell and physical state without duplicate uses");
    session.world().find(session.playerObject())->currentSpace.cell = makeTes3RecordKey("CELL", "dry_interior");
    const auto before = p.progression; step({}, 60);
    check(!session.physics().characterState(session.playerObject())->swimming && p.progression == before &&
          session.physics().characterState(session.playerObject())->position.y < physical->position.y - 100,
        "leaving a water cell restores falling and awards no swimming use");
}
void testGameplayMotion(std::shared_ptr<const Tes3ContentStore> content) {
    BethesdaSession session; setup(session, content); std::string error;
    const std::vector<odai::math::Vector3> floor{{-10000,0,-10000},{10000,0,-10000},{10000,0,10000},{-10000,0,10000}};
    const std::vector<std::uint32_t> indices{0,1,2,0,2,3};
    check(session.physics().addStaticCollision(ObjectId::runtime(123), floor, indices, error), error);
    PhysicsCharacterConfig config; config.position = {0,.1f,0};
    check(session.registerActorController(session.playerObject(), config, error), error);
    for (int i = 0; i < 10; ++i) (void)session.advance(1.0 / 60);
    check(session.physics().characterState(session.playerObject())->grounded, "movement fixture has physical support");
    auto& p = session.tes3().playerState();
    PhysicsCharacterInput input; input.desiredVelocity = {100,0,0};
    for (int i = 0; i < 120; ++i) { check(session.setActorControllerInput(session.playerObject(), input), "walk input accepted"); (void)session.advance(1.0 / 60); }
    check(p.progression.skillProgress[8] == 0, "walking does not award Athletics run progress");
    for (int i = 0; i < 120; ++i) { check(session.setActorControllerInput(session.playerObject(), input, true), "run input accepted"); (void)session.advance(1.0 / 60); }
    const double runProgress = p.progression.skillProgress[8];
    check(runProgress > .05 && runProgress < .1, "actual running awards time-scaled skill use");
    input.desiredVelocity = {};
    for (int i = 0; i < 120; ++i) { (void)session.setActorControllerInput(session.playerObject(), input, true); (void)session.advance(1.0 / 60); }
    check(std::abs(p.progression.skillProgress[8] - runProgress) < 1e-6, "stationary run intent grants no use");
    input.desiredVelocity.y = jumpSpeedForHeightBethesdaUnits(kMorrowindBaselineJumpHeightUnits);
    (void)session.setActorControllerInput(session.playerObject(), input); (void)session.advance(1.0 / 60);
    input.desiredVelocity = {};
    for (int i = 0; i < 180; ++i) { (void)session.setActorControllerInput(session.playerObject(), input); (void)session.advance(1.0 / 60); }
    check(p.progression.skillProgress[20] > 0 && p.progression.skillProgress[20] < .2, "successful physical takeoff awards Acrobatics once");
    const auto trainer = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99));
    session.world().find(trainer)->transform.position = {800, 0, 0};
    config.position = {800,.1f,0}; check(session.registerActorController(trainer, config, error), error);
    for (int i = 0; i < 10; ++i) (void)session.advance(1.0 / 60);
    const auto attacker = session.physics().characterState(session.playerObject());
    const auto target = session.physics().characterState(trainer);
    Tes3ActiveSpell accurate; accurate.effects.push_back({117,-1,-1,200,100000});
    session.tes3().activeSpellsForRestore()[session.playerObject()].push_back(accurate);
    const auto hit = session.performMeleeAttack(session.playerObject(), target->position - attacker->position, 5, 500);
    check(hit.hit && p.progression.skillProgress[26] > 0, "resolved unarmed contact advances Hand-to-hand");
    const double hitProgress = p.progression.skillProgress[26];
    for (int i = 0; i < 30; ++i) (void)session.advance(1.0 / 60);
    const auto miss = session.performMeleeAttack(session.playerObject(), {-1,0,0}, 5, 500);
    check(!miss.hit && p.progression.skillProgress[26] == hitProgress, "miss cannot advance a weapon skill");
    auto* npc = session.world().find(trainer); npc->combatState.emplace(); npc->combatState->combatTarget = session.playerObject();
    p.progression.levelProgress = 10;
    const auto before = p;
    check(!session.restTes3Player(1, true, error) && p == before, "nearby hostile actor blocks rest without consuming eligibility");
}
void retailRecords(const fs::path& dataRoot) {
    FalloutLoadOrder order; std::string error;
    check(order.open(dataRoot, {"Morrowind.esm"}, error), error);
    auto content = std::make_shared<Tes3ContentStore>(); check(content->load(order, "windows-1252", error), error);
    for (int skill = 0; skill < 27; ++skill) check(tes3SkillDefinition(*content, skill).has_value(), "retail skill " + std::to_string(skill));
    for (const auto& [key, record] : content->namedRecords()) if (key.recordType == "CLAS")
        check(tes3ClassDefinition(*content, record.id).has_value(), "retail class " + record.id);
    std::size_t automatic = 0;
    for (const auto& [key, actor] : content->actors()) if (actor.autoCalculate && !actor.creature) {
        ++automatic;
        check(actor.attributes.size() == 8 && actor.skills.size() == 27, "retail automatic stats " + actor.id);
    }
    check(automatic > 0, "local retail check includes automatically calculated NPCs");
    const double factionRep = tes3NumericSetting(*content, "iAutoRepFacMod", 2);
    const double levelRep = tes3NumericSetting(*content, "iAutoRepLevMod", 0);
    std::size_t factionNpcs = 0;
    for (const auto& [key, actor] : content->actors()) if (!actor.creature && actor.faction.valid()) {
        ++factionNpcs;
        check(actor.reputation == factionRep * (actor.rank + 1) + levelRep * (actor.level - 1),
            "retail faction NPC reputation " + actor.id);
    }
    check(factionNpcs > 0, "local retail check includes faction NPC reputation initialization");

    BethesdaSession session; BethesdaSessionConfig config; config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "local-retail-leveling-records";
    check(session.configure(config, error), error); check(session.configureTes3Content(content, error), error);
    RuntimeObject player; player.id = session.playerObject(); player.kind = RuntimeObjectKind::Actor;
    player.base = makeTes3RecordKey("NPC_", "player"); player.actorValues.emplace();
    check(session.world().addInitialObject(player, error), error);
    auto& p = session.tes3().playerState(); p.race = "breton"; p.actorClass = "warrior"; p.birthsign = "Lady's Favor"; p.gender = 0;
    check(session.initializeTes3PlayerProgression(error), error);
    check(p.numericFilters["level"] == 1 && p.progression.levelProgress == 0, "retail choices initialize without fake gains");
    check(p.numericFilters["endurance"] == 65 && p.numericFilters["personality"] == 65 && p.numericFilters["maxhealth"] == 57,
        "retail Breton Warrior/Lady baseline includes ability attributes");
    check(p.numericFilters["maxmagicka"] == 75, "retail Breton magicka ability feeds creation baseline");
    std::cout << "Local retail record and initialization checks completed; this is not a normal-input playthrough.\n";
}
void tests(const fs::path& root) {
    const auto content = fixture(root); testImportedRepairCondition(content, root); testGameplayRepair(content, root); testInventoryCondition(content, root); testNpcReputation(content, root); testAutomaticTrainers(content); testTrainerAbilities(content, root); testDerivedDisposition(content, root); testPersuasion(content, root); testCustomClassInput(content, root); testRestEffectTime(content);
    testGameplayFalls(content, root); testArmorContacts(content, root); testWeaponHitEligibility(content, root);
    testGameplaySwimming(content, root); testGameplayMotion(content); BethesdaSession session; setup(session, content);
    auto& runtime = session.tes3(); auto& p = runtime.playerState(); std::string error;
    check(content->findRecord("SKIL", "26") && !content->findRecord("SKIL", "27"), "numeric SKIL import covers all 27 skills");
    check(p.numericFilters["strength"] == 50 && p.numericFilters["endurance"] == 50 && p.numericFilters["athletics"] == 35,
        "race/class initialization grants baseline without earned counters");
    check(p.progression.levelProgress == 0 && p.numericFilters["level"] == 1, "initial level and empty progression");
    const auto baseline = p;
    for (int skill = 0; skill < 27; ++skill) {
        check(runtime.usePlayerSkill(skill, 0, runtime.playerSkillRequirement(skill), error), "imported use values advance skill " + std::to_string(skill));
    }
    check(p.progression.levelProgress == 10, "all 27 skill records use class membership consistently");
    p = baseline;
    check(std::abs(runtime.playerSkillRequirement(6) - 21.6) < 1e-6, "major specialization threshold");
    check(std::abs(runtime.playerSkillRequirement(0) - 16.8) < 1e-6, "minor specialization threshold");
    check(std::abs(runtime.playerSkillRequirement(10) - 7.5) < 1e-6, "misc threshold");
    check(runtime.usePlayerSkill(6, 0, 10.8, error) && std::abs(p.progression.skillProgress[6] - .5) < 1e-6, "fractional use gain");
    check(runtime.advancePlayerSkill(6, Tes3SkillAdvanceSource::Training, error) && p.progression.skillProgress[6] == .5,
        "training preserves normalized progress");
    check(runtime.usePlayerSkill(6, 0, 1000, error) && p.numericFilters["axe"] == 37 && p.progression.skillProgress[6] == 0,
        "usage gains one point and discards overflow");
    check(!runtime.usePlayerSkill(-1, 0, 1, error) && !runtime.usePlayerSkill(0, 4, 1, error) &&
        !runtime.usePlayerSkill(0, 0, std::numeric_limits<double>::quiet_NaN(), error), "invalid use input rejected");
    p.progression = {}; p.numericFilters["axe"] = 40; p.numericFilters["block"] = 40; p.numericFilters["destruction"] = 40;
    check(runtime.advancePlayerSkill(6, Tes3SkillAdvanceSource::Book, error) && runtime.advancePlayerSkill(0, Tes3SkillAdvanceSource::Training, error) &&
        runtime.advancePlayerSkill(10, Tes3SkillAdvanceSource::Usage, error), "all positive skill advancement sources accepted");
    check(p.progression.levelProgress == 2 && p.progression.attributeIncreases[3] == 1, "misc increases attribute only");
    for (int count : {0, 1, 4, 5, 7, 8, 9, 10, 15}) {
        p.progression.attributeIncreases[0] = count;
        const int expected = count == 0 ? 1 : count <= 4 ? 2 : count <= 7 ? 3 : count <= 9 ? 4 : 5;
        check(runtime.playerAttributeGain(0) == expected, "multiplier boundary " + std::to_string(count));
    }
    p.progression.attributeIncreases[7] = 99; check(runtime.playerAttributeGain(7) == 1, "Luck always gains one");
    p.numericFilters["strength"] = 98; check(runtime.playerAttributeGain(0) == 2, "gain clamped near 100");
    p.numericFilters["strength"] = 100; check(runtime.playerAttributeGain(0) == 0, "capped attribute unavailable");
    p.numericFilters["strength"] = 50;
    p.progression = {}; p.numericFilters["athletics"] = 35;
    p.progression.skillProgress[8] = .3;
    check(session.readTes3SkillBook(makeTes3RecordKey("BOOK", "skill_book"), error), error);
    const auto afterBook = p.progression;
    check(session.readTes3SkillBook(makeTes3RecordKey("BOOK", "SKILL_BOOK"), error) && p.progression == afterBook,
        "case-insensitive repeated book read awards once");
    check(p.numericFilters["athletics"] == 36 && p.progression.skillProgress[8] == .3, "book preserves fractional progress");
    const auto offers = session.tes3TrainingOffers(ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99)));
    check(offers.size() == 3 && offers[0].skill == 8 && offers[0].eligible, "trainer offers top three authored skills");
    const int goldBefore = session.world().find(session.playerObject())->inventory[0].count;
    const auto minuteBefore = session.livingWorld().absoluteGameMinute();
    check(session.trainTes3PlayerSkill(ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99)), 8, error), error);
    check(session.world().find(session.playerObject())->inventory[0].count == goldBefore - offers[0].price &&
        p.numericFilters["athletics"] == 37 && p.progression.skillProgress[8] == .3, "training charges once and preserves progress");
    check(session.livingWorld().absoluteGameMinute() == minuteBefore + 120, "training advances two game hours even without schedules");
    check(!session.trainTes3PlayerSkill(ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x99)), 26, error), "unoffered training rejected");
    p.numericFilters["axe"] = 100; const auto capped = p.progression;
    check(!runtime.advancePlayerSkill(6, Tes3SkillAdvanceSource::Usage, error) && p.progression == capped, "skill cap awards no counters");
    check(runtime.advancePlayerSkill(6, Tes3SkillAdvanceSource::Jail, error) && p.numericFilters["axe"] == 99 && p.progression == capped,
        "jail loss preserves progression counters");
    check(runtime.advancePlayerSkill(6, Tes3SkillAdvanceSource::Training, error) && p.progression.levelProgress == capped.levelProgress + 1,
        "recovering a jail loss counts as an earned increase");
    p.progression = {}; p.progression.levelProgress = 9;
    check(!runtime.playerLevelReady() && !runtime.confirmPlayerLevel({0, 1, 5}, error), "9/10 is ineligible");
    p.numericFilters["block"] = 30; check(runtime.advancePlayerSkill(0, Tes3SkillAdvanceSource::Usage, error), error);
    check(runtime.playerLevelReady() && p.numericFilters["level"] == 1 && p.progression.readyNotification, "10/10 signals readiness without granting level");
    p.progression.levelProgress = 15; p.progression.attributeIncreases[5] = 10; p.numericFilters["endurance"] = 45;
    p.numericFilters["maxhealth"] = 50; p.numericFilters["health"] = 40; session.synchronizeTes3PlayerValues();
    check(session.restTes3Player(1, false, error) && !p.progression.selectionOpen, "waiting cannot open level selection");
    session.world().find(session.playerObject())->currentSpace.cell = makeTes3RecordKey("CELL", "no_sleep");
    check(!session.restTes3Player(1, true, error) && !p.progression.selectionOpen, "illegal sleeping rejected");
    session.world().find(session.playerObject())->currentSpace.cell = makeTes3RecordKey("CELL", "wilderness");
    check(session.restTes3Player(1, true, error) && p.progression.selectionOpen, error);
    const auto beforeConfirm = p;
    check(!session.confirmTes3PlayerLevel({0, 0, 5}, error) && !session.confirmTes3PlayerLevel({0, 5}, error) && p == beforeConfirm,
        "duplicate/incomplete confirmation is atomic");
    check(session.confirmTes3PlayerLevel({0, 1, 5}, error), error);
    check(p.numericFilters["level"] == 2 && p.numericFilters["endurance"] == 50 && p.progression.levelProgress == 5,
        "post-selection level and overflow");
    check(p.numericFilters["maxhealth"] == 55 && std::all_of(p.progression.attributeIncreases.begin(), p.progression.attributeIncreases.end(), [](int n){return n == 0;}),
        "post-selection health and all bonus counters cleared");
    check(!session.confirmTes3PlayerLevel({0, 1, 5}, error), "repeated confirmation cannot grant another level");
    const auto levelThread = runtime.scripts().start("LevelProbe", session.playerObject(), error);
    check(levelThread != 0, error); (void)session.advance(1.0 / 60);
    check(runtime.scripts().threads().at(levelThread).locals.at("observed").number == 2,
        "MWScript GetLevel reads committed progression, not the NPC template level");
    p.progression.levelProgress = 20; p.numericFilters["endurance"] = 45; p.progression.attributeIncreases.fill(0);
    Tes3ActiveSpell fortify; fortify.effects.push_back({79, -1, 5, 100, std::numeric_limits<std::uint64_t>::max()});
    runtime.activeSpellsForRestore()[session.playerObject()].push_back(fortify);
    check(session.modifiedTes3Stat(session.playerObject(), "endurance") == 145, "temporary Endurance fortification is visible");
    check(session.restTes3Player(1, true, error) && session.confirmTes3PlayerLevel({0, 1, 7}, error), error);
    check(p.numericFilters["maxhealth"] == 59.5 && p.progression.levelProgress == 10 && !p.progression.selectionOpen,
        "fractional base-Endurance health excludes temporary fortification and banked levels require another rest");
    runtime.activeSpellsForRestore().erase(session.playerObject());
    check(session.restTes3Player(1, true, error), error);
    Tes3LevelUiFlow flow;
    (void)flow.update({false, false, true}, session, error);
    (void)flow.update({false, false, true}, session, error); check(flow.selected().empty(), "reselection toggles preview without mutation");
    (void)flow.update({false, false, true}, session, error);
    (void)flow.update({false, true, true}, session, error);
    (void)flow.update({false, true, true}, session, error);
    check(flow.selected().size() == 3 && p.numericFilters["level"] == 3, "UI selection is a preview");
    for (int i = 0; i < 6; ++i) (void)flow.update({false, true, false}, session, error);
    check(flow.update({false, false, true}, session, error) && p.numericFilters["level"] == 4, "headless UI confirmation commits level");
    p.progression.levelProgress = 10;
    for (auto a : tes3AttributeNames) p.numericFilters[std::string(a)] = 100;
    p.numericFilters["luck"] = 99;
    check(runtime.levelChoiceCount() == 1 && session.restTes3Player(1, true, error) && session.confirmTes3PlayerLevel({7}, error), "one uncapped attribute needs one choice");
    p.progression.levelProgress = 10;
    check(runtime.levelChoiceCount() == 0 && session.restTes3Player(1, true, error) && session.confirmTes3PlayerLevel({}, error), "fully capped character can still level");
    p.progression.levelProgress = 21; p.progression.skillProgress[8] = .25; p.progression.attributeIncreases[4] = 9;
    p.progression.customClass = Tes3ProgressionClass{};
    check(session.restTes3Player(1, true, error), error);
    const auto saved = p;
    check(saveOdaiGameAtomic(root / "save.json", session, error), error);
    BethesdaSession restored; setup(restored, content); SaveLoadReport report;
    check(loadOdaiGame(root / "save.json", restored, {}, report, error), error);
    check(restored.tes3().playerState() == saved, "save round-trip preserves entire player ledger and fractional stats");
    check(restored.tes3().playerState().progression.selectionOpen && restored.confirmTes3PlayerLevel({}, error), "saved pending selection remains confirmable once");
    check(!restored.confirmTes3PlayerLevel({}, error), "load cannot duplicate committed level");
    nlohmann::json json; { std::ifstream in(root / "save.json"); in >> json; }
    auto old = json; old["payload"]["tes3"]["player"].erase("progression");
    old["checksum"] = checksum(old["payload"].dump());
    { std::ofstream out(root / "old.json"); out << old.dump(); }
    BethesdaSession migrated; setup(migrated, content);
    check(loadOdaiGame(root / "old.json", migrated, {}, report, error), error);
    check(migrated.tes3().playerState().progression == Tes3ProgressionState{} &&
        migrated.tes3().playerState().numericFilters == saved.numericFilters, "older saves preserve base stats without inventing earned counters");
    auto bad = json; bad["payload"]["tes3"]["player"]["progression"]["skills"][0] = -1;
    bad["checksum"] = checksum(bad["payload"].dump());
    { std::ofstream out(root / "bad.json"); out << bad.dump(); }
    const auto hash = migrated.deterministicHash();
    check(!loadOdaiGame(root / "bad.json", migrated, {}, report, error) && migrated.deterministicHash() == hash,
        "malformed progression fails without mutating the session");
    const auto overridden = fixture(root / "override", true); BethesdaSession modified; setup(modified, overridden);
    const auto modArmor = tes3ArmorDefinition(*overridden, "armor_0_1");
    check(modArmor && modArmor->skill == 2, "imported weight and category GMSTs override armor classification");
    auto& mr = modified.tes3(); auto& mp = mr.playerState();
    check(std::abs(mr.playerSkillRequirement(6) - 9) < 1e-6 && mr.levelThreshold() == 3, "imported class factors and level threshold override defaults");
    mp.progression.attributeIncreases[0] = 1; check(mr.playerAttributeGain(0) == 4, "GMST attribute multiplier override");
    mp.progression.levelProgress = 3; check(modified.restTes3Player(1, true, error) && modified.confirmTes3PlayerLevel({0, 1, 7}, error), error);
    check(std::abs(mp.numericFilters["maxhealth"] - 60) < 1e-5, "health factor override");
}
}
int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--retail") { retailRecords(argv[2]); return failures ? 1 : 0; }
    const auto root = fs::temp_directory_path() / "odai_tes3_progression_tests";
    fs::remove_all(root); tests(root); fs::remove_all(root);
    if (!failures) std::cout << "TES3 progression tests passed\n";
    return failures ? 1 : 0;
}
