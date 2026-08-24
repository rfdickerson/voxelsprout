#include "bethesda/bethesda_session.h"
#include "bethesda/save_game.h"
#include "bethesda/tes3_runtime.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace odai::bethesda;
using namespace odai::importer::fnv;

int failures = 0;

void check(bool value, const std::string& message) {
    if (!value) {
        std::cerr << "[tes3 runtime test] FAIL: " << message << '\n';
        ++failures;
    }
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

void dial(std::vector<std::uint8_t>& file, const std::string& id, std::uint8_t type) {
    std::vector<std::uint8_t> body;
    sub(body, "NAME", id);
    sub(body, "DATA", std::vector<std::uint8_t>{type});
    record(file, "DIAL", body);
}

std::vector<std::uint8_t> infoData(std::int32_t type, std::int32_t value) {
    std::vector<std::uint8_t> bytes(12u, 0xffu);
    std::memcpy(bytes.data(), &type, sizeof(type));
    std::memcpy(bytes.data() + 4u, &value, sizeof(value));
    bytes[11u] = 0u;
    return bytes;
}

void info(std::vector<std::uint8_t>& file, const std::string& id,
          const std::string& previous, const std::string& next,
          std::int32_t type, std::int32_t value, const std::string& response,
          const std::string& result = {}, const char* questFlag = nullptr,
          const std::string& select = {}, std::int32_t selectValue = 0) {
    std::vector<std::uint8_t> body;
    sub(body, "INAM", id);
    sub(body, "PNAM", previous);
    sub(body, "NNAM", next);
    sub(body, "DATA", infoData(type, value));
    sub(body, "NAME", response);
    if (!select.empty()) {
        sub(body, "SCVR", select);
        std::vector<std::uint8_t> bytes;
        append(bytes, selectValue);
        sub(body, "INTV", std::move(bytes));
    }
    if (!result.empty()) sub(body, "BNAM", result);
    if (questFlag != nullptr) sub(body, questFlag, std::vector<std::uint8_t>{1u});
    record(file, "INFO", body);
}

void addUnloadedScriptedReference(std::vector<std::uint8_t>& file) {
    std::vector<std::uint8_t> blessing;
    sub(blessing, "NAME", "TestBlessing");
    sub(blessing, "FNAM", "Test Blessing");
    std::vector<std::uint8_t> spellData(12u, 0u);
    sub(blessing, "SPDT", std::move(spellData));
    std::vector<std::uint8_t> effect(24u, 0u);
    const std::int16_t fortifyAttribute = 79;
    const std::int32_t duration = 10;
    const std::int32_t magnitude = 5;
    std::memcpy(effect.data(), &fortifyAttribute, sizeof(fortifyAttribute));
    effect[2u] = 0xffu;
    effect[3u] = 0u;
    std::memcpy(effect.data() + 12u, &duration, sizeof(duration));
    std::memcpy(effect.data() + 16u, &magnitude, sizeof(magnitude));
    std::memcpy(effect.data() + 20u, &magnitude, sizeof(magnitude));
    sub(blessing, "ENAM", std::move(effect));
    record(file, "SPEL", blessing);

    std::vector<std::uint8_t> object;
    sub(object, "NAME", "quest_switch");
    record(file, "ACTI", object);

    std::vector<std::uint8_t> script;
    std::vector<std::uint8_t> scriptHeader(52u, 0u);
    std::memcpy(scriptHeader.data(), "UnloadedDisable", 15u);
    sub(script, "SCHD", std::move(scriptHeader));
    sub(script, "SCTX", "begin UnloadedDisable\n\"quest_switch\"->Disable\nend");
    record(file, "SCPT", script);

    std::vector<std::uint8_t> cell;
    sub(cell, "NAME", "Unloaded Cell");
    std::vector<std::uint8_t> cellData(12u, 0u);
    const std::uint32_t interior = 1u;
    std::memcpy(cellData.data(), &interior, sizeof(interior));
    sub(cell, "DATA", std::move(cellData));
    std::vector<std::uint8_t> frmr;
    append(frmr, std::uint32_t{0x42u});
    sub(cell, "FRMR", std::move(frmr));
    sub(cell, "NAME", "quest_switch");
    record(file, "CELL", cell);

    std::vector<std::uint8_t> statsScript;
    std::vector<std::uint8_t> statsHeader(52u, 0u);
    std::memcpy(statsHeader.data(), "PlayerStats", 11u);
    sub(statsScript, "SCHD", std::move(statsHeader));
    sub(statsScript, "SCTX",
        "begin PlayerStats\nshort fortified\nplayer->SetStrength 40\n"
        "player->ModStrength 2\nCast \"TestBlessing\" Player\n"
        "set fortified to Player->GetStrength\nend");
    record(file, "SCPT", statsScript);
}

std::shared_ptr<Tes3ContentStore> makeContent(const fs::path& root) {
    std::vector<std::uint8_t> file;
    header(file);
    dial(file, "TR_RuntimeQuest", 4u);
    info(file, "q10", "", "q20", 4, 10, "Quest begins", {}, "QSTN");
    info(file, "q20", "q10", "", 4, 20, "Quest ends", {}, "QSTF");
    dial(file, "LegacyQuest", 4u);
    info(file, "legacy10", "", "", 4, 10, "Old journal entry");
    dial(file, "Greeting 1", 2u);
    info(file, "greet", "", "", 2, 0,
         "Would you ask about the Sanctuary?", "AddTopic \"Sanctuary\"");
    dial(file, "Sanctuary", 0u);
    info(file, "topic_a", "", "topic_b", 0, 0,
         "Will you help?", "Choice \"Accept\" 1 \"Decline\" 2");
    // SCVR: index 0, built-in function 50 (Choice), equality, no variable.
    info(file, "topic_b", "topic_a", "", 0, 0,
         "Then the work begins.", "Journal \"TR_RuntimeQuest\" 10",
         nullptr, "01500", 1);
    addUnloadedScriptedReference(file);
    const fs::path plugin = root / "Morrowind.esm";
    std::ofstream output(plugin, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    output.close();

    FalloutLoadOrder order;
    std::string error;
    check(order.open(root, {"Morrowind.esm"}, error), error);
    auto content = std::make_shared<Tes3ContentStore>();
    check(content->load(order, "windows-1252", error), error);
    return content;
}

void testJournalAndDialogue() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_runtime_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const std::shared_ptr<Tes3ContentStore> content = makeContent(root);

    Tes3Runtime runtime;
    std::string error;
    const ObjectId player = ObjectId::persistent(makeTes3RecordKey("NPC_", "player"));
    check(runtime.configure(content, player, error), error);
    check(runtime.scriptCheckReport().strictPass(),
          "all synthetic result scripts compile and close over implemented natives");

    Tes3DialogueActorState actor;
    actor.object = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x42u));
    actor.id = "temple priest";
    actor.cell = "Test Temple";
    Tes3DialoguePlayerState playerState;
    playerState.object = player;
    Tes3DialogueResponse greeting = runtime.startDialogue(actor, playerState);
    check(greeting.accepted && greeting.text.find("Sanctuary") != std::string::npos,
          "OpenMW-style greeting selects dynamically for actor/player state");
    check(runtime.knownTopics().contains(makeTes3RecordKey("DIAL", "sanctuary")) &&
              runtime.availableTopics().size() == 1u,
          "response discovery and AddTopic expose a known topic");

    Tes3DialogueResponse topic = runtime.selectTopic("SANCTUARY");
    check(topic.accepted && topic.choices.size() == 2u && topic.choices[0].value == 1,
          "topic result script synchronously creates authored Choice options");
    Tes3DialogueResponse accepted = runtime.answerChoice(1);
    check(accepted.accepted && accepted.text == "Then the work begins.",
          "Choice filter selects the matching linked INFO");
    const Tes3JournalQuestState* quest = runtime.journal().find("TR_RuntimeQuest");
    check(quest != nullptr && quest->currentIndex == 10 &&
              quest->classification == Tes3JournalQuestClassification::Active &&
              runtime.journal().chronology().size() == 1u,
          "Journal native records chronology and QSTN active state");

    check(runtime.journal().addEntry(*content->findDialogue("TR_RuntimeQuest"), 20u, 9u, error), error);
    quest = runtime.journal().find("TR_RuntimeQuest");
    check(quest != nullptr && quest->classification == Tes3JournalQuestClassification::Completed,
          "QSTF marks a modern quest complete without inventing objectives");
    check(runtime.journal().addEntry(*content->findDialogue("LegacyQuest"), 10u, 10u, error), error);
    const Tes3JournalQuestState* legacy = runtime.journal().find("legacyquest");
    check(legacy != nullptr &&
              legacy->classification == Tes3JournalQuestClassification::Legacy &&
              !legacy->hasStatusFlags,
          "pre-Tribunal journals remain chronological instead of guessed complete");
    fs::remove_all(root, ec);
}

void testSessionSaveReloadMidDialogue() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_save_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const std::shared_ptr<Tes3ContentStore> content = makeContent(root);

    std::string error;
    BethesdaSession original;
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-save-fixture";
    config.randomSeed = 91u;
    check(original.configure(config, error), error);
    check(original.configureTes3Content(content, error), error);
    check(original.tes3().scripts().start(
              "PlayerStats", original.playerObject(), error) != 0u, error);
    (void)original.advance(1.0 / 60.0);
    check(original.tes3().playerState().numericFilters.at("strength") == 42.0,
          "authored player attribute set/mod commands mutate persistent TES3 state");
    const auto statsThread = std::find_if(original.tes3().scripts().threads().begin(),
        original.tes3().scripts().threads().end(), [](const auto& item) {
            return item.second.program == "playerstats";
        });
    check(statsThread != original.tes3().scripts().threads().end() &&
              statsThread->second.locals.at("fortified").number == 47.0 &&
              !original.tes3().activeSpells().empty(),
          "Cast applies a deterministic timed fortify effect visible to TES3 stat queries");

    Tes3DialogueActorState actor;
    actor.object = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x42u));
    actor.id = "temple priest";
    Tes3DialoguePlayerState player;
    player.object = original.playerObject();
    check(original.tes3().startDialogue(actor, player).accepted,
          "session starts TES3 dialogue before save");
    const Tes3DialogueResponse topic = original.tes3().selectTopic("Sanctuary");
    check(topic.accepted && topic.choices.size() == 2u,
          "session is suspended at an authored TES3 choice");
    check(original.tes3().journal().addEntry(
              *content->findDialogue("TR_RuntimeQuest"), 10u,
              original.clock().tick(), error), error);
    original.tes3().scripts().globals()["tr_test_global"] = Tes3Value::fromNumber(7.0);
    const ObjectId unloaded = ObjectId::persistent(
        makeTes3ReferenceKey("Morrowind.esm", 0x42u));
    check(original.tes3().scripts().start("UnloadedDisable", unloaded, error) != 0u, error);
    (void)original.advance(1.0 / 60.0);
    check(original.tes3().referenceOverrides().contains(unloaded) &&
              original.tes3().referenceOverrides().at(unloaded).enabled == false,
          "target-qualified native mutates an unloaded TES3 reference overlay");

    const fs::path savePath = root / "mid-dialogue.odai";
    const std::uint64_t expectedHash = original.deterministicHash();
    check(saveOdaiGameAtomic(savePath, original, error), error);

    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash,
          "ODAI save v8 restores deterministic TES3 journal, VM, and dialogue state");
    check(restored.tes3().dialogue().active &&
              restored.tes3().dialogue().choices == topic.choices,
          "mid-dialogue authored Choice survives save/reload");
    check(restored.tes3().journal().index("TR_RuntimeQuest") == 10 &&
              restored.tes3().scripts().globals().at("tr_test_global").number == 7.0,
          "TES3 journal and globals survive save/reload");
    check(restored.tes3().referenceOverrides() == original.tes3().referenceOverrides(),
          "sparse unloaded-reference overrides survive save/reload");
    fs::remove_all(root, ec);
}

}  // namespace

int main() {
    testJournalAndDialogue();
    testSessionSaveReloadMidDialogue();
    if (failures == 0) std::cout << "tes3 runtime tests passed\n";
    return failures == 0 ? 0 : 1;
}
