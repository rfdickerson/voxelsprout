#include "bethesda/tes3_content.h"
#include "bethesda/gameplay_catalog.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace odai::bethesda;
using namespace odai::importer::fnv;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[tes3 content test] FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename T>
void append(std::vector<std::uint8_t>& bytes, const T& value) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
}

std::vector<std::uint8_t> subrecord(
    const char* type, const std::vector<std::uint8_t>& payload) {
    std::vector<std::uint8_t> out(type, type + 4);
    append(out, static_cast<std::uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

std::vector<std::uint8_t> subrecord(const char* type, const std::string& text) {
    std::vector<std::uint8_t> bytes(text.begin(), text.end());
    bytes.push_back('\0');
    return subrecord(type, bytes);
}

void addSubrecord(std::vector<std::uint8_t>& body, const char* type, const std::string& text) {
    const auto sub = subrecord(type, text);
    body.insert(body.end(), sub.begin(), sub.end());
}

void addSubrecord(
    std::vector<std::uint8_t>& body, const char* type,
    const std::vector<std::uint8_t>& bytes) {
    const auto sub = subrecord(type, bytes);
    body.insert(body.end(), sub.begin(), sub.end());
}

void addRecord(
    std::vector<std::uint8_t>& file, const char* type,
    const std::vector<std::uint8_t>& body) {
    file.insert(file.end(), type, type + 4);
    append(file, static_cast<std::uint32_t>(body.size()));
    append(file, std::uint32_t{0});
    append(file, std::uint32_t{0});
    file.insert(file.end(), body.begin(), body.end());
}

void addHeader(
    std::vector<std::uint8_t>& file, bool master,
    const std::vector<std::string>& masters, std::uint32_t recordCount) {
    std::vector<std::uint8_t> body;
    std::vector<std::uint8_t> hedr(300u, 0u);
    const float version = 1.3f;
    std::memcpy(hedr.data(), &version, sizeof(version));
    const std::uint32_t type = master ? 1u : 0u;
    std::memcpy(hedr.data() + 4u, &type, sizeof(type));
    std::memcpy(hedr.data() + 296u, &recordCount, sizeof(recordCount));
    addSubrecord(body, "HEDR", hedr);
    for (const std::string& dependency : masters) {
        addSubrecord(body, "MAST", dependency);
        addSubrecord(body, "DATA", std::vector<std::uint8_t>(8u, 0u));
    }
    addRecord(file, "TES3", body);
}

std::vector<std::uint8_t> infoData(std::int32_t journalIndex) {
    std::vector<std::uint8_t> bytes(12u, 0xffu);
    const std::int32_t type = 4;
    std::memcpy(bytes.data(), &type, sizeof(type));
    std::memcpy(bytes.data() + 4u, &journalIndex, sizeof(journalIndex));
    bytes[11u] = 0u;
    return bytes;
}

void addDialogue(std::vector<std::uint8_t>& file) {
    std::vector<std::uint8_t> body;
    addSubrecord(body, "NAME", "TR_TestQuest");
    addSubrecord(body, "DATA", std::vector<std::uint8_t>{4u});
    addRecord(file, "DIAL", body);
}

void addInfo(
    std::vector<std::uint8_t>& file, const std::string& id,
    const std::string& previous, const std::string& next,
    std::int32_t index, const std::string& response,
    const char* questFlag = nullptr, bool deleted = false) {
    std::vector<std::uint8_t> body;
    addSubrecord(body, "INAM", id);
    addSubrecord(body, "PNAM", previous);
    addSubrecord(body, "NNAM", next);
    if (deleted) {
        addSubrecord(body, "DELE", std::vector<std::uint8_t>{0u, 0u, 0u});
    } else {
        addSubrecord(body, "DATA", infoData(index));
        addSubrecord(body, "NAME", response);
        addSubrecord(body, "SCVR", "04JX0TR_Prerequisite");
        std::vector<std::uint8_t> comparison;
        append(comparison, std::int32_t{20});
        addSubrecord(body, "INTV", comparison);
        addSubrecord(body, "BNAM", "Journal TR_TestQuest 30");
        if (questFlag != nullptr) addSubrecord(body, questFlag, std::vector<std::uint8_t>{1u});
    }
    addRecord(file, "INFO", body);
}

void addScript(std::vector<std::uint8_t>& file, const std::string& source) {
    std::vector<std::uint8_t> body;
    std::vector<std::uint8_t> header(52u, 0u);
    const std::string id = "TR_TestScript";
    std::memcpy(header.data(), id.data(), id.size());
    const std::uint32_t shorts = 1u;
    std::memcpy(header.data() + 32u, &shorts, sizeof(shorts));
    addSubrecord(body, "SCHD", header);
    addSubrecord(body, "SCVR", std::string("state\0", 6u));
    addSubrecord(body, "SCDT", std::vector<std::uint8_t>{1u, 2u, 3u});
    addSubrecord(body, "SCTX", source);
    addRecord(file, "SCPT", body);
}

void addGlobal(std::vector<std::uint8_t>& file, float value) {
    std::vector<std::uint8_t> body;
    addSubrecord(body, "NAME", "TR_TestGlobal");
    addSubrecord(body, "FNAM", std::vector<std::uint8_t>{'f'});
    std::vector<std::uint8_t> bytes;
    append(bytes, value);
    addSubrecord(body, "FLTV", bytes);
    addRecord(file, "GLOB", body);
}

void addStatic(std::vector<std::uint8_t>& file) {
    std::vector<std::uint8_t> body;
    addSubrecord(body, "NAME", "tr_test_stat");
    addSubrecord(body, "MODL", "x\\tr_test.nif");
    addRecord(file, "STAT", body);
}

void addActorAndItem(std::vector<std::uint8_t>& file) {
    std::vector<std::uint8_t> item;
    addSubrecord(item, "NAME", "tr_test_item");
    addSubrecord(item, "FNAM", "Test Relic");
    addRecord(file, "MISC", item);

    std::vector<std::uint8_t> actor;
    addSubrecord(actor, "NAME", "tr_test_actor");
    addSubrecord(actor, "FNAM", "Test Pilgrim");
    addSubrecord(actor, "RNAM", "dark elf");
    addSubrecord(actor, "CNAM", "priest");
    addSubrecord(actor, "ANAM", "temple");
    addSubrecord(actor, "SCRI", "TR_TestScript");
    std::vector<std::uint8_t> stats(52u, 0u);
    const std::int16_t level = 8;
    const std::int16_t health = 75;
    const std::int16_t magicka = 60;
    const std::int16_t fatigue = 90;
    std::memcpy(stats.data(), &level, sizeof(level));
    stats[2u] = 55u;   // Strength.
    stats[10u] = 42u;  // Block.
    std::memcpy(stats.data() + 38u, &health, sizeof(health));
    std::memcpy(stats.data() + 40u, &magicka, sizeof(magicka));
    std::memcpy(stats.data() + 42u, &fatigue, sizeof(fatigue));
    stats[46u] = 3u;
    addSubrecord(actor, "NPDT", std::move(stats));
    std::vector<std::uint8_t> inventory(36u, 0u);
    const std::int32_t count = 2;
    std::memcpy(inventory.data(), &count, sizeof(count));
    std::memcpy(inventory.data() + 4u, "tr_test_item", 12u);
    addSubrecord(actor, "NPCO", std::move(inventory));
    std::vector<std::uint8_t> aiData(12u, 0u);
    const std::uint32_t travelService = 0x00001000u;
    std::memcpy(aiData.data() + 8u, &travelService, sizeof(travelService));
    addSubrecord(actor, "AIDT", std::move(aiData));
    std::vector<std::uint8_t> destination(24u, 0u);
    const float destinationX = 4096.0f;
    std::memcpy(destination.data(), &destinationX, sizeof(destinationX));
    addSubrecord(actor, "DODT", std::move(destination));
    addSubrecord(actor, "DNAM", "Test Destination");
    addRecord(file, "NPC_", actor);
}

void addCell(std::vector<std::uint8_t>& file, std::uint32_t frmr, bool deleted) {
    std::vector<std::uint8_t> body;
    addSubrecord(body, "NAME", "Test Sanctuary");
    std::vector<std::uint8_t> cellData(12u, 0u);
    const std::uint32_t interior = 1u;
    std::memcpy(cellData.data(), &interior, sizeof(interior));
    addSubrecord(body, "DATA", cellData);
    std::vector<std::uint8_t> referenceId;
    append(referenceId, frmr);
    addSubrecord(body, "FRMR", referenceId);
    addSubrecord(body, "NAME", "tr_test_stat");
    if (deleted) {
        addSubrecord(body, "DELE", std::vector<std::uint8_t>{0u, 0u, 0u});
    } else {
        std::vector<std::uint8_t> transform(24u, 0u);
        const float x = 12.5f;
        std::memcpy(transform.data(), &x, sizeof(x));
        addSubrecord(body, "DATA", transform);
    }
    addRecord(file, "CELL", body);
}

void addNamedExteriorCell(std::vector<std::uint8_t>& file, std::uint32_t frmr) {
    std::vector<std::uint8_t> body;
    addSubrecord(body, "NAME", "Test Town");
    std::vector<std::uint8_t> cellData(12u, 0u);
    const std::int32_t gridX = 2;
    const std::int32_t gridZ = -3;
    std::memcpy(cellData.data() + 4u, &gridX, sizeof(gridX));
    std::memcpy(cellData.data() + 8u, &gridZ, sizeof(gridZ));
    addSubrecord(body, "DATA", cellData);
    std::vector<std::uint8_t> referenceId;
    append(referenceId, frmr);
    addSubrecord(body, "FRMR", referenceId);
    addSubrecord(body, "NAME", "tr_test_actor");
    addSubrecord(body, "DATA", std::vector<std::uint8_t>(24u, 0u));
    addRecord(file, "CELL", body);
}

void writePlugin(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
}

void testContentStore() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_content_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);

    std::vector<std::uint8_t> base;
    addHeader(base, true, {}, 8u);
    addDialogue(base);
    addInfo(base, "a", "", "b", 10, "Original", "QSTN");
    addInfo(base, "b", "a", "", 20, "Terminal", "QSTF");
    addScript(base, "begin TR_TestScript\nshort state\nend");
    addGlobal(base, 2.0f);
    addStatic(base);
    addActorAndItem(base);
    addCell(base, 0x00000042u, false);
    addNamedExteriorCell(base, 0x00000043u);
    writePlugin(root / "Morrowind.esm", base);

    std::vector<std::uint8_t> patch;
    addHeader(patch, false, {"Morrowind.esm"}, 5u);
    addDialogue(patch);
    addInfo(patch, "a", "", "c", 10, "Patched", "QSTN");
    addInfo(patch, "b", "a", "", 20, "", nullptr, true);
    addInfo(patch, "c", "a", "", 30, "Restart", "QSTR");
    addCell(patch, 0x01000042u, true);
    writePlugin(root / "Patch.esp", patch);

    FalloutLoadOrder order;
    std::string error;
    check(order.open(root, {"Patch.esp"}, error), error);
    check(order.size() == 2u, "master dependency is loaded before patch");

    Tes3ContentStore content;
    check(content.load(order, "windows-1252", error), error);
    const Tes3DialogueDefinition* quest = content.findDialogue("tr_testquest");
    check(quest != nullptr && quest->type == Tes3DialogueType::Journal,
          "journal DIAL is keyed case-insensitively");
    check(quest != nullptr && quest->infos.size() == 2u &&
              quest->infos[0].id == "a" && quest->infos[1].id == "c",
          "later INFO override/deletion and linked ordering are applied");
    check(quest != nullptr && quest->infos[0].response == "Patched" &&
              quest->infos[1].questStatus == Tes3QuestStatus::Restart,
          "typed response and Tribunal quest status survive import");
    check(quest != nullptr && !quest->infos[0].conditions.empty() &&
              quest->infos[0].conditions[0].valid &&
              quest->infos[0].conditions[0].function == Tes3ConditionFunction::Journal,
          "SCVR journal condition and comparison value are decoded");

    const Tes3ScriptDefinition* script = content.findScript("TR_TESTSCRIPT");
    check(script != nullptr && script->shortCount == 1u &&
              script->variableNames.size() == 1u && !script->bytecode.empty() &&
              script->source.find("begin") == 0u,
          "SCPT source, bytecode, and locals are retained together");
    check(content.globals().at(makeTes3RecordKey("GLOB", "tr_testglobal")).value == 2.0f,
          "GLOB initial value is typed");
    check(content.findRecord("STAT", "TR_TEST_STAT") != nullptr,
          "generic named records retain later-wins string identity");
    const Tes3ActorDefinition* actor = content.findActor("NPC_", "TR_TEST_ACTOR");
    check(actor != nullptr && actor->level == 8 && actor->rank == 3 &&
              actor->health == 75.0f && actor->magicka == 60.0f &&
              actor->fatigue == 90.0f && actor->attributes.at("strength") == 55.0f &&
              actor->skills.at("block") == 42.0f &&
              actor->faction == makeTes3RecordKey("FACT", "temple") &&
              actor->script == makeTes3RecordKey("SCPT", "TR_TestScript"),
          "TES3 actor identity, stats, faction, rank, and local script are typed");
    check(actor != nullptr && actor->serviceFlags == 0x00001000u &&
              actor->travelDestinations.size() == 1u &&
              actor->travelDestinations[0].cell == "Test Destination" &&
              actor->travelDestinations[0].position[0] == 4096.0f,
          "TES3 AIDT services and DODT/DNAM travel destinations are typed");
    check(actor != nullptr && actor->inventory.size() == 1u &&
              actor->inventory[0].first == makeTes3RecordKey("MISC", "tr_test_item") &&
              actor->inventory[0].second == 2,
          "TES3 NPCO inventory resolves string IDs to the winning typed record");
    check(content.references().size() == 1u,
          "a later plugin can delete a master-owned FRMR reference");
    if (!content.references().empty()) {
        const Tes3ReferenceDefinition& townActor = content.references().begin()->second;
        check(townActor.cell == makeTes3RecordKey("CELL", "Test Town") &&
                  !townActor.interior &&
                  townActor.hasCellGrid && townActor.cellGridX == 2 &&
                  townActor.cellGridZ == -3 &&
                  townActor.base == makeTes3RecordKey("NPC_", "tr_test_actor"),
              "named exterior CELL references retain identity, grid, and typed actor bases");
    }
    GameplayCellPayload gameplay;
    check(compileTes3GameplayCell(content,
              makeTes3RecordKey("CELL", "Test Town"), order.fingerprint(),
              gameplay, error), error);
    check(gameplay.actors.size() == 1u &&
              hasRole(gameplay.actors[0].roles, ActorRole::Merchant) &&
              hasRole(gameplay.actors[0].roles, ActorRole::GuildMember) &&
              hasRole(gameplay.actors[0].roles, ActorRole::Priest) &&
              hasRole(gameplay.actors[0].roles, ActorRole::Traveller),
          "TES3 services, faction, class, and travel records compile to common actor roles");
    check(gameplay.actors.size() == 1u && gameplay.actors[0].questConstrained &&
              !gameplay.actors[0].authoredPackages.empty() &&
              gameplay.actors[0].authoredPackages[0].source ==
                  BehaviorPackageSource::QuestOrScript,
          "scripted TES3 actors retain authored-first package precedence");
    check(gameplay.anchors.size() == 2u &&
              gameplay.anchors[0].kind == ActivityAnchorKind::Idle &&
              gameplay.anchors[1].kind == ActivityAnchorKind::TravelService,
          "TES3 actor origin and travel service compile as stable anchors");
    GameplayCellPayload streamedGameplay;
    check(compileTes3GameplayExteriorCell(content, 2, -3,
              order.fingerprint(), streamedGameplay, error), error);
    check(streamedGameplay.space.kind == RuntimeSpaceKind::Exterior &&
              streamedGameplay.space.gridX == 2 &&
              streamedGameplay.space.gridZ == -3 &&
              streamedGameplay.space.cell == makeTes3RecordKey("CELL", "Test Town"),
          "streaming grid resolves a named TES3 exterior gameplay sidecar");

    const std::string cp1252("Pilgrimage \x97 complete", 21u);
    check(decodeTes3Text(cp1252, "win1252").find("\xe2\x80\x94") != std::string::npos,
          "Windows-1252 punctuation decodes deterministically to UTF-8");
    fs::remove_all(root, ec);
}

}  // namespace

int main() {
    testContentStore();
    if (failures == 0) std::cout << "tes3 content tests passed\n";
    return failures == 0 ? 0 : 1;
}
