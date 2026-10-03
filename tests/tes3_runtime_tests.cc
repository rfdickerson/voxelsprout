#include "bethesda/bethesda_session.h"
#include "bethesda/save_game.h"
#include "bethesda/tes3_runtime.h"

#include <algorithm>
#include <cmath>
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
using namespace odai::importer::bethesda;

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

    const auto addSpell = [&](const char* id, std::int32_t type,
                              std::int16_t effectId) {
        std::vector<std::uint8_t> body;
        sub(body, "NAME", id);
        std::vector<std::uint8_t> spellData(12u, 0u);
        std::memcpy(spellData.data(), &type, sizeof(type));
        sub(body, "SPDT", std::move(spellData));
        std::vector<std::uint8_t> effect(24u, 0u);
        std::memcpy(effect.data(), &effectId, sizeof(effectId));
        effect[2u] = 0xffu;
        effect[3u] = 0xffu;
        const std::int32_t magnitude = 1;
        std::memcpy(effect.data() + 16u, &magnitude, sizeof(magnitude));
        std::memcpy(effect.data() + 20u, &magnitude, sizeof(magnitude));
        sub(body, "ENAM", std::move(effect));
        record(file, "SPEL", body);
    };
    addSpell("corprus", 2, 132);
    addSpell("ash-chancre", 2, 17);
    addSpell("common flu", 3, 17);
    addSpell("corprus immunity", 1, 96);
    addSpell("Cure Blight Disease", 0, 70);
    addSpell("Cure Common Disease Other", 0, 69);
    std::vector<std::uint8_t> shieldSpell;
    sub(shieldSpell, "NAME", "test_route_shield");
    std::vector<std::uint8_t> shieldData(12u, 0u);
    const std::int32_t ability = 1;
    std::memcpy(shieldData.data(), &ability, sizeof(ability));
    sub(shieldSpell, "SPDT", std::move(shieldData));
    for (const std::int16_t effectId : {3, 4, 5, 6}) {
        std::vector<std::uint8_t> effect(24u, 0u);
        std::memcpy(effect.data(), &effectId, sizeof(effectId));
        effect[2u] = effect[3u] = 0xffu;
        const std::int32_t magnitude = 50;
        std::memcpy(effect.data() + 16u, &magnitude, sizeof(magnitude));
        std::memcpy(effect.data() + 20u, &magnitude, sizeof(magnitude));
        sub(shieldSpell, "ENAM", std::move(effect));
    }
    record(file, "SPEL", shieldSpell);
    std::vector<std::uint8_t> restoreSpell;
    sub(restoreSpell, "NAME", "test_restore_attribute");
    sub(restoreSpell, "SPDT", std::vector<std::uint8_t>(12u, 0u));
    std::vector<std::uint8_t> restoreEffect(24u, 0u);
    const std::int16_t restoreAttribute = 74;
    const std::int32_t restoreMagnitude = 100;
    std::memcpy(restoreEffect.data(), &restoreAttribute, sizeof(restoreAttribute));
    restoreEffect[2u] = 0xffu;
    restoreEffect[3u] = 3u;  // Agility
    std::memcpy(restoreEffect.data() + 16u, &restoreMagnitude, sizeof(restoreMagnitude));
    std::memcpy(restoreEffect.data() + 20u, &restoreMagnitude, sizeof(restoreMagnitude));
    sub(restoreSpell, "ENAM", std::move(restoreEffect));
    record(file, "SPEL", restoreSpell);
    std::vector<std::uint8_t> finaleScript;
    std::vector<std::uint8_t> finaleHeader(52u, 0u);
    std::memcpy(finaleHeader.data(), "FinaleEffectProbe", 17u);
    sub(finaleScript, "SCHD", std::move(finaleHeader));
    sub(finaleScript, "SCTX",
        "begin FinaleEffectProbe\n"
        "quest_actor->AddSpell \"test_route_shield\"\n"
        "set shieldSeen to quest_actor->GetSpell \"test_route_shield\"\n"
        "set beforeRestore to Player->GetAgility\n"
        "quest_actor->Cast \"test_restore_attribute\" Player\n"
        "set afterRestore to Player->GetAgility\nend");
    record(file, "SCPT", finaleScript);
    std::vector<std::uint8_t> heartScript;
    std::vector<std::uint8_t> heartHeader(52u, 0u);
    std::memcpy(heartHeader.data(), "HeartProbe", 10u);
    sub(heartScript, "SCHD", std::move(heartHeader));
    sub(heartScript, "SCTX",
        "begin HeartProbe\nshort countHits\n"
        "set countHits to HitOnMe sunder\n"
        "if ( countHits == 1 )\nset countHits to 2\nendif\nend");
    record(file, "SCPT", heartScript);
    std::vector<std::uint8_t> watcherScript;
    std::vector<std::uint8_t> watcherHeader(52u, 0u);
    std::memcpy(watcherHeader.data(), "HeartWatcher", 12u);
    sub(watcherScript, "SCHD", std::move(watcherHeader));
    sub(watcherScript, "SCTX",
        "begin HeartWatcher\n"
        "set observedHits to \"quest_actor\".countHits\nend");
    record(file, "SCPT", watcherScript);

    std::vector<std::uint8_t> package;
    sub(package, "NAME", "bk_a1_1_caiuspackage");
    record(file, "MISC", package);

    std::vector<std::uint8_t> object;
    sub(object, "NAME", "quest_switch");
    record(file, "ACTI", object);
    std::vector<std::uint8_t> questActor;
    sub(questActor, "NAME", "quest_actor");
    record(file, "NPC_", questActor);

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
    std::vector<std::uint8_t> actorFrmr;
    append(actorFrmr, std::uint32_t{0x43u});
    sub(cell, "FRMR", std::move(actorFrmr));
    sub(cell, "NAME", "quest_actor");
    for (const std::uint32_t itemId : {0x78u, 0x79u}) {
        std::vector<std::uint8_t> itemFrmr;
        append(itemFrmr, itemId);
        sub(cell, "FRMR", std::move(itemFrmr));
        sub(cell, "NAME", "test_scripted_paper");
    }
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

    std::vector<std::uint8_t> diseaseScript;
    std::vector<std::uint8_t> diseaseHeader(52u, 0u);
    std::memcpy(diseaseHeader.data(), "DiseaseRoute", 12u);
    sub(diseaseScript, "SCHD", std::move(diseaseHeader));
    sub(diseaseScript, "SCTX",
        "begin DiseaseRoute\nshort before\nshort after\n"
        "short ashAfter\nshort fluAfter\n"
        "player->AddSpell corprus\nplayer->AddSpell ash-chancre\n"
        "set before to player->GetBlightDisease\n"
        "Cast \"Cure Blight Disease\" Player\n"
        "set after to player->GetBlightDisease\n"
        "set ashAfter to player->GetSpell \"ash-chancre\"\n"
        "player->RemoveSpell corprus\n"
        "player->AddSpell \"corprus immunity\"\n"
        "player->AddSpell \"common flu\"\n"
        "Cast \"Cure Common Disease Other\" Player\n"
        "set fluAfter to player->GetCommonDisease\nend");
    record(file, "SCPT", diseaseScript);

    std::vector<std::uint8_t> standingScript;
    std::vector<std::uint8_t> standingHeader(52u, 0u);
    std::memcpy(standingHeader.data(), "StandingCheck", 13u);
    sub(standingScript, "SCHD", std::move(standingHeader));
    sub(standingScript, "SCTX",
        "begin StandingCheck\nshort standing\nset standing to GetStandingPC\nend");
    record(file, "SCPT", standingScript);

    std::vector<std::uint8_t> weatherScript;
    std::vector<std::uint8_t> weatherHeader(52u, 0u);
    std::memcpy(weatherHeader.data(), "WeatherRoute", 12u);
    sub(weatherScript, "SCHD", std::move(weatherHeader));
    sub(weatherScript, "SCTX",
        "begin WeatherRoute\nshort weather\n"
        "ModRegion \"Red Mountain Region\" 50 50 0 0 0 0 0 0\n"
        "ChangeWeather \"Red Mountain Region\" 3\n"
        "set weather to GetCurrentWeather\nend");
    record(file, "SCPT", weatherScript);

    std::vector<std::uint8_t> invalidWeatherScript;
    std::vector<std::uint8_t> invalidWeatherHeader(52u, 0u);
    std::memcpy(invalidWeatherHeader.data(), "BadWeather", 10u);
    sub(invalidWeatherScript, "SCHD", std::move(invalidWeatherHeader));
    sub(invalidWeatherScript, "SCTX",
        "begin BadWeather\n"
        "ModRegion \"Red Mountain Region\" 120 0 0 0 0 0 0 0\nend");
    record(file, "SCPT", invalidWeatherScript);

    std::vector<std::uint8_t> healthScript;
    std::vector<std::uint8_t> healthHeader(52u, 0u);
    std::memcpy(healthHeader.data(), "UnloadedHealth", 14u);
    sub(healthScript, "SCHD", std::move(healthHeader));
    sub(healthScript, "SCTX",
        "begin UnloadedHealth\nshort health\n"
        "quest_actor->SetHealth 0\n"
        "set health to quest_actor->GetHealth\nend");
    record(file, "SCPT", healthScript);

    std::vector<std::uint8_t> aiScript;
    std::vector<std::uint8_t> aiHeader(52u, 0u);
    std::memcpy(aiHeader.data(), "AiRouteProbe", 12u);
    sub(aiScript, "SCHD", std::move(aiHeader));
    sub(aiScript, "SCTX",
        "begin AiRouteProbe\n"
        "quest_actor->AiTravel 1 2 3\n"
        "set package to quest_actor->GetCurrentAIPackage\n"
        "set done to quest_actor->GetAIPackageDone\nend");
    record(file, "SCPT", aiScript);

    std::vector<std::uint8_t> escortScript;
    std::vector<std::uint8_t> escortHeader(52u, 0u);
    std::memcpy(escortHeader.data(), "EscortRouteProbe", 16u);
    sub(escortScript, "SCHD", std::move(escortHeader));
    sub(escortScript, "SCTX",
        "begin EscortRouteProbe\n"
        "quest_actor->AIEscort Player 12 195 100 170\n"
        "set package to quest_actor->GetCurrentAIPackage\n"
        "set done to quest_actor->GetAIPackageDone\nend");
    record(file, "SCPT", escortScript);

    std::vector<std::uint8_t> menuGateScript;
    std::vector<std::uint8_t> menuGateHeader(52u, 0u);
    std::memcpy(menuGateHeader.data(), "TutorialMenuGate", 16u);
    sub(menuGateScript, "SCHD", std::move(menuGateHeader));
    sub(menuGateScript, "SCTX",
        "begin TutorialMenuGate\nshort advanced\n"
        "if ( MenuMode == 1 )\nreturn\nendif\n"
        "set advanced to 1\nend");
    record(file, "SCPT", menuGateScript);

    std::vector<std::uint8_t> openingScript;
    std::vector<std::uint8_t> openingHeader(52u, 0u);
    std::memcpy(openingHeader.data(), "CharGen", 7u);
    sub(openingScript, "SCHD", std::move(openingHeader));
    sub(openingScript, "SCTX",
        "begin CharGen\nDisablePlayerControls\n"
        "Player->PositionCell 61, -135, 24, 340, \"Imperial Prison Ship\"\n"
        "set CharGenState to 10\nStopScript CharGen\nend");
    record(file, "SCPT", openingScript);

    std::vector<std::uint8_t> openingGuard;
    sub(openingGuard, "NAME", "opening_guard");
    sub(openingGuard, "SCRI", "TestOpeningGuard");
    record(file, "NPC_", openingGuard);
    std::vector<std::uint8_t> guardScript;
    std::vector<std::uint8_t> guardHeader(52u, 0u);
    std::memcpy(guardHeader.data(), "TestOpeningGuard", 16u);
    sub(guardScript, "SCHD", std::move(guardHeader));
    sub(guardScript, "SCTX",
        "begin TestOpeningGuard\nshort introductions\n"
        "set introductions to introductions + 1\nEnablePlayerControls\n"
        "StopScript TestOpeningGuard\nend");
    record(file, "SCPT", guardScript);

    std::vector<std::uint8_t> gauntlet;
    sub(gauntlet, "NAME", "test_wraithguard");
    sub(gauntlet, "FNAM", "Test Wraithguard");
    std::vector<std::uint8_t> armorData(24u, 0u);
    const std::uint32_t rightGauntlet = 7u;
    std::memcpy(armorData.data(), &rightGauntlet, sizeof(rightGauntlet));
    sub(gauntlet, "AODT", std::move(armorData));
    sub(gauntlet, "SCRI", "TestArtifactScript");
    record(file, "ARMO", gauntlet);
    std::vector<std::uint8_t> artifactScript;
    std::vector<std::uint8_t> artifactHeader(52u, 0u);
    std::memcpy(artifactHeader.data(), "TestArtifactScript", 18u);
    sub(artifactScript, "SCHD", std::move(artifactHeader));
    sub(artifactScript, "SCTX",
        "begin TestArtifactScript\nshort OnPCEquip\n"
        "set testArtifactEquipped to OnPCEquip\nend");
    record(file, "SCPT", artifactScript);

    std::vector<std::uint8_t> scriptedPaper;
    sub(scriptedPaper, "NAME", "test_scripted_paper");
    sub(scriptedPaper, "FNAM", "Scripted Paper");
    sub(scriptedPaper, "SCRI", "ScriptedPickup");
    record(file, "MISC", scriptedPaper);
    std::vector<std::uint8_t> pickupScript;
    std::vector<std::uint8_t> pickupHeader(52u, 0u);
    std::memcpy(pickupHeader.data(), "ScriptedPickup", 14u);
    sub(pickupScript, "SCHD", std::move(pickupHeader));
    sub(pickupScript, "SCTX",
        "begin ScriptedPickup\nshort OnPCAdd\n"
        "if ( OnActivate == 1 )\nActivate\n"
        "set pickupCount to pickupCount + 1\nendif\n"
        "if ( OnPCAdd == 1 )\nset pickedUp to 1\nendif\nend");
    record(file, "SCPT", pickupScript);
    std::vector<std::uint8_t> doorScript;
    std::vector<std::uint8_t> doorHeader(52u, 0u);
    std::memcpy(doorHeader.data(), "ScriptedDoor", 12u);
    sub(doorScript, "SCHD", std::move(doorHeader));
    sub(doorScript, "SCTX",
        "begin ScriptedDoor\nif ( OnActivate == 1 )\nActivate\nendif\nend");
    record(file, "SCPT", doorScript);

    std::vector<std::uint8_t> voiceScript;
    std::vector<std::uint8_t> voiceHeader(52u, 0u);
    std::memcpy(voiceHeader.data(), "VoiceGate", 9u);
    sub(voiceScript, "SCHD", std::move(voiceHeader));
    sub(voiceScript, "SCTX",
        "begin VoiceGate\nshort state\n"
        "if ( state == 0 )\nSay test.wav \"Wait for this line.\"\n"
        "set state to 1\nelseif ( state == 1 )\n"
        "if ( SayDone == 1 )\nset state to 2\nendif\nendif\nend");
    record(file, "SCPT", voiceScript);

    std::vector<std::uint8_t> menuScript;
    std::vector<std::uint8_t> menuHeader(52u, 0u);
    std::memcpy(menuHeader.data(), "MenuRequest", 11u);
    sub(menuScript, "SCHD", std::move(menuHeader));
    sub(menuScript, "SCTX",
        "begin MenuRequest\nshort opened\nEnableNameMenu\n"
        "set opened to MenuMode\nend");
    record(file, "SCPT", menuScript);

    std::vector<std::uint8_t> promptScript;
    std::vector<std::uint8_t> promptHeader(52u, 0u);
    std::memcpy(promptHeader.data(), "MessagePrompt", 13u);
    sub(promptScript, "SCHD", std::move(promptHeader));
    sub(promptScript, "SCTX",
        "begin MessagePrompt\nshort selected\n"
        "MessageBox \"Continue?\" \"Yes\" \"No\"\n"
        "set selected to GetButtonPressed\nend");
    record(file, "SCPT", promptScript);

    std::vector<std::uint8_t> transientScript;
    std::vector<std::uint8_t> transientHeader(52u, 0u);
    std::memcpy(transientHeader.data(), "TransientObject", 15u);
    sub(transientScript, "SCHD", std::move(transientHeader));
    sub(transientScript, "SCTX", "begin TransientObject\nDontSaveObject\nend");
    record(file, "SCPT", transientScript);

    std::vector<std::uint8_t> detectionScript;
    std::vector<std::uint8_t> detectionHeader(52u, 0u);
    std::memcpy(detectionHeader.data(), "DetectionProbe", 14u);
    sub(detectionScript, "SCHD", std::move(detectionHeader));
    sub(detectionScript, "SCTX",
        "begin DetectionProbe\nshort seen\n"
        "set seen to GetDetected target_actor\nend");
    record(file, "SCPT", detectionScript);
}

std::shared_ptr<Tes3ContentStore> makeContent(
    const fs::path& root, bool blockedQuest = false) {
    std::vector<std::uint8_t> file;
    header(file);
    std::vector<std::uint8_t> playerRecord;
    sub(playerRecord, "NAME", "player");
    record(file, "NPC_", playerRecord);
    std::vector<std::uint8_t> region;
    sub(region, "NAME", "Red Mountain Region");
    record(file, "REGN", region);
    dial(file, "TR_RuntimeQuest", 4u);
    info(file, "q10", "", "q20", 4, 10, "Quest begins", {}, "QSTN");
    info(file, "q20", "q10", "", 4, 20, "Quest ends", {}, "QSTF");
    dial(file, "LegacyQuest", 4u);
    info(file, "legacy10", "", "", 4, 10, "Old journal entry");
    dial(file, "Greeting 0", 2u);
    info(file, "package_greeting", "", "", 2, 0,
         "You have the package.", {}, nullptr,
         "05IX2bk_a1_1_caiuspackage", 0);
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
    dial(file, "Package Grant", 0u);
    info(file, "package_grant", "", "", 0, 0,
         "Take this package.", "Player->AddItem \"bk_a1_1_caiuspackage\" 1");
    dial(file, "Package Receipt", 0u);
    info(file, "package_receipt", "", "", 0, 0,
         "The package is in your inventory.", {}, nullptr,
         "05IX2bk_a1_1_caiuspackage", 0);
    dial(file, "Runtime Failure", 0u);
    info(file, "runtime_failure", "", "", 0, 0,
        "This transition must fail cleanly.",
        "Journal \"TR_RuntimeQuest\" 10\n"
        "Player->AddItem \"bk_a1_1_caiuspackage\" 1\n"
        "ModRegion \"Red Mountain Region\" 120 0 0 0 0 0 0 0");
    dial(file, "Suspended Failure", 0u);
    info(file, "suspended_failure", "", "", 0, 0,
        "This choice must also fail cleanly.",
        "Journal \"TR_RuntimeQuest\" 10\n"
        "Player->AddItem \"bk_a1_1_caiuspackage\" 1\n"
        "MessageBox \"Choose\" \"Yes\" \"No\"\n"
        "ModRegion \"Red Mountain Region\" 120 0 0 0 0 0 0 0");
    dial(file, "Suspended Success", 0u);
    info(file, "suspended_success", "", "", 0, 0,
        "This choice completes its transition.",
        "MessageBox \"Choose\" \"Yes\" \"No\"\n"
        "Journal \"TR_RuntimeQuest\" 10");
    dial(file, "Attribution", 0u);
    info(file, "clear_actor", "", "", 0, 0,
        "The response has no actor attribution.", "ClearInfoActor");
    if (blockedQuest) {
        dial(file, "Blocked Quest", 0u);
        info(file, "blocked_info", "", "", 0, 0, "Quest offer",
             "Journal \"TR_RuntimeQuest\" 10\n"
             "quest_switch->UnsupportedQuestNative\n"
             "Journal \"TR_RuntimeQuest\" 20");
    }
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

void testUnsupportedResultScriptDoesNotAdvanceQuest() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_blocked_quest_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root, true);
    Tes3Runtime runtime;
    std::string error;
    const ObjectId player = ObjectId::persistent(makeTes3RecordKey("NPC_", "player"));
    check(runtime.configure(content, player, error), error);
    check(runtime.scriptCheckReport().unsupportedCommands.contains("unsupportedquestnative"),
          "unsupported result operation appears in the content closure audit");
    Tes3DialogueActorState actor;
    actor.object = player;
    actor.id = "quest giver";
    Tes3DialoguePlayerState playerState;
    playerState.object = player;
    check(runtime.startDialogue(actor, playerState).accepted, "blocked quest dialogue opens");
    check(runtime.addTopic("Blocked Quest"), "blocked quest topic is available");
    const auto exhaustedBefore = runtime.dialogue().exhaustedInfos;
    const Tes3DialogueResponse response = runtime.selectTopic("Blocked Quest");
    check(!response.accepted && !response.diagnostics.empty() &&
              response.diagnostics.front().find("unsupported operation unsupportedquestnative")
                  != std::string::npos &&
              runtime.journal().index("TR_RuntimeQuest") == 0 &&
              runtime.dialogue().exhaustedInfos == exhaustedBefore,
          "unsupported result script fails before journal and one-time dialogue state change");
    fs::remove_all(root, ec);
}

void testRuntimeFailureRollsBackDialogueTransition() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_result_rollback_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-result-rollback-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeTes3RecordKey("NPC_", "player");
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    check(session.world().addInitialObject(std::move(player), error), error);
    Tes3DialogueActorState actor;
    actor.object = session.playerObject();
    actor.id = "quest giver";
    Tes3DialoguePlayerState playerState;
    playerState.object = session.playerObject();
    check(session.startTes3Dialogue(actor, playerState).accepted,
          "runtime failure fixture opens dialogue");
    check(session.tes3().addTopic("Runtime Failure"), "runtime failure topic is available");
    const auto exhaustedBefore = session.tes3().dialogue().exhaustedInfos;
    const auto response = session.tes3().selectTopic("Runtime Failure");
    const RecordKey package = makeTes3RecordKey("BOOK", "bk_a1_1_caiuspackage");
    check(!response.accepted && !response.diagnostics.empty() &&
              response.diagnostics.front().find("modregion") != std::string::npos &&
              session.tes3().journal().index("TR_RuntimeQuest") == 0 &&
              session.tes3().dialogue().exhaustedInfos == exhaustedBefore &&
              !session.tes3().playerState().inventory.contains(package) &&
              session.world().find(session.playerObject())->inventory.empty() &&
              !session.world().hasPendingCommands(),
          "failed result restores journal, response eligibility, inventory and world commands");
    check(session.tes3().addTopic("Suspended Failure"),
          "suspended failure topic is available");
    const auto suspended = session.tes3().selectTopic("Suspended Failure");
    check(suspended.accepted && suspended.choices.size() == 2u &&
              suspended.text.find("Choose") != std::string::npos &&
              session.tes3().hasPendingResultTransaction() &&
              session.tes3().journal().index("TR_RuntimeQuest") == 10,
          "dialogue result remains transactional while MessageBox awaits a choice");
    const auto competing = session.tes3().selectTopic("Package Grant");
    check(!competing.accepted && !competing.diagnostics.empty() &&
              session.tes3().hasPendingResultTransaction(),
          "another topic cannot alter a suspended result transition");
    const fs::path suspendedSave = root / "suspended.odai";
    check(!saveOdaiGameAtomic(suspendedSave, session, error) &&
              error.find("suspended TES3 dialogue result") != std::string::npos,
          "save rejects a suspended one-time transition");
    const auto failedChoice = session.tes3().answerChoice(0);
    check(!failedChoice.accepted && !failedChoice.diagnostics.empty() &&
              failedChoice.diagnostics.front().find("modregion") != std::string::npos &&
              !session.tes3().hasPendingResultTransaction() &&
              session.tes3().journal().index("TR_RuntimeQuest") == 0 &&
              session.tes3().dialogue().exhaustedInfos == exhaustedBefore &&
              !session.tes3().playerState().inventory.contains(package) &&
              session.world().find(session.playerObject())->inventory.empty() &&
              !session.world().hasPendingCommands(),
          "failure after MessageBox restores the complete dialogue transition");
    check(session.tes3().addTopic("Suspended Success"),
          "successful suspended result topic is available");
    const auto success = session.tes3().selectTopic("Suspended Success");
    check(success.accepted && success.choices.size() == 2u &&
              session.tes3().hasPendingResultTransaction(),
          "successful result also suspends until its choice arrives");
    const auto completedChoice = session.tes3().answerChoice(1);
    check(completedChoice.accepted && completedChoice.diagnostics.empty() &&
              !session.tes3().hasPendingResultTransaction() &&
              session.tes3().journal().index("TR_RuntimeQuest") == 10,
          "successful suspended result commits only after the button choice");
    fs::remove_all(root, ec);
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
    check(runtime.topicResponseActors().contains(topic.info) &&
              runtime.topicResponseActors().at(topic.info) == "temple priest",
          "topic response records the actor attribution");
    Tes3DialogueResponse accepted = runtime.answerChoice(1);
    check(accepted.accepted && accepted.text == "Then the work begins.",
          "Choice filter selects the matching linked INFO");
    check(runtime.dialogue().choice == -1 && !runtime.availableTopics().empty(),
          "completed choice restores ordinary topic eligibility");
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
    check(runtime.journal().addEntry(*content->findDialogue("TR_RuntimeQuest"), 10u, 11u, error), error);
    check(runtime.journal().find("TR_RuntimeQuest")->classification == Tes3JournalQuestClassification::Completed &&
              runtime.journal().chronology().size() == 3u,
          "duplicate earlier journal entry cannot reopen a completed quest");
    check(runtime.selectTopic("Sanctuary").choices.size() == 2,
          "authored prompt may repeat without artificial INFO exhaustion");
    const auto offeredChoices = runtime.dialogue().choices;
    check(!runtime.answerChoice(2).accepted && runtime.dialogue().choices == offeredChoices,
          "unmatched choice preserves the authored prompt");
    check(!runtime.selectTopic("Sanctuary").accepted && runtime.dialogue().choices == offeredChoices,
          "topic selection cannot bypass an outstanding choice");
    check(runtime.answerChoice(1).accepted, "authored choice can be retried");
    check(runtime.addTopic("Attribution"), "attribution fixture topic is available");
    const auto cleared = runtime.selectTopic("Attribution");
    check(cleared.accepted && !runtime.topicResponseActors().contains(cleared.info),
          "ClearInfoActor removes actor attribution from the current topic response");
    fs::remove_all(root, ec);
}

void testAuthoredOpeningPositionAndSave() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_opening_tests";
    std::error_code ec;
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-opening-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeTes3RecordKey("NPC_", "player");
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    check(session.world().addInitialObject(std::move(player), error), error);
    check(session.tes3().scripts().start("CharGen", session.playerObject(), error) != 0u, error);
    const auto step = session.advance(1.0 / 60.0);
    check(step.diagnostics.empty(), "authored opening executes without diagnostics");
    const auto* spawned = session.world().find(session.playerObject());
    check(spawned != nullptr && spawned->transform.position[0] == 61.0 &&
              spawned->transform.position[1] == 24.0 &&
              spawned->transform.position[2] == 135.0,
          "PositionCell converts TES3 Z-up coordinates into runtime space");
    check(spawned != nullptr &&
              std::abs(spawned->transform.rotationRadians[1] +
                       340.0 * (3.14159265358979323846 / 180.0)) < 0.0001,
          "PositionCell maps TES3 heading onto the runtime up axis");
    check(spawned != nullptr && spawned->currentSpace.kind == RuntimeSpaceKind::Interior &&
              spawned->currentSpace.cell == makeTes3RecordKey("CELL", "Imperial Prison Ship"),
          "opening places the player in the authored interior");
    check(session.tes3().scripts().globals().at("chargenstate").number == 10.0 &&
              session.tes3().playerState().numericFilters.at("control:playercontrols") == 0.0,
          "opening advances CharGenState and disables controls");

    RuntimeObject guard;
    guard.id = session.world().allocateRuntimeId();
    const auto guardId = guard.id;
    guard.base = makeTes3RecordKey("NPC_", "opening_guard");
    guard.kind = RuntimeObjectKind::Actor;
    guard.currentSpace = spawned->currentSpace;
    check(session.world().addInitialObject(std::move(guard), error), error);
    check(session.bindTes3ActorLocalScript(guardId, error), error);
    const auto threadCount = session.tes3().scripts().threads().size();
    check(session.bindTes3ActorLocalScript(guardId, error) &&
              session.tes3().scripts().threads().size() == threadCount,
          "actor residency binds exactly one local opening script");
    check(session.advance(1.0 / 60.0).diagnostics.empty(), "guard introduction executes");
    check(session.tes3().playerState().numericFilters.at("control:playercontrols") == 1.0,
          "resident guard script releases the opening movement lock");
    check(session.bindTes3ActorLocalScript(guardId, error) &&
              session.tes3().scripts().threads().size() == threadCount,
          "stopped guard script remains stopped on residency updates");

    const fs::path savePath = root / "opening.odai";
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    const auto* restoredPlayer = restored.world().find(restored.playerObject());
    check(restoredPlayer != nullptr && restoredPlayer->transform.position[2] == 135.0 &&
              restored.tes3().scripts().globals().at("chargenstate").number == 10.0,
          "save/load keeps authored opening position and quest gate");
    check(restored.bindTes3ActorLocalScript(guardId, error) &&
              restored.tes3().scripts().threads().size() == threadCount,
          "restored actor script is not duplicated or restarted");
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

void testCorprusCureSpellStateAndSave() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_corprus_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-corprus-fixture";
    BethesdaSession original;
    std::string error;
    check(original.configure(config, error), error);
    check(original.configureTes3Content(content, error), error);
    RuntimeObject player;
    player.id = original.playerObject();
    player.base = makeTes3RecordKey("NPC_", "player");
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    check(original.world().addInitialObject(std::move(player), error), error);
    const std::uint64_t threadId = original.tes3().scripts().start(
        "DiseaseRoute", original.playerObject(), error);
    check(threadId != 0u, error);
    (void)original.advance(1.0 / 60.0);
    const Tes3ScriptThread& thread = original.tes3().scripts().threads().at(threadId);
    check(thread.state == Tes3ThreadState::Completed &&
              thread.locals.at("before").number == 1.0 &&
              thread.locals.at("after").number == 1.0 &&
              thread.locals.at("ashafter").number == 0.0 &&
              thread.locals.at("fluafter").number == 0.0,
          "same-tick disease and spell queries see AddSpell, cure, and RemoveSpell");
    const auto& inventory = original.tes3().playerState().inventory;
    check(!inventory.contains(makeTes3RecordKey("SPEL", "corprus")) &&
              !inventory.contains(makeTes3RecordKey("SPEL", "ash-chancre")) &&
              !inventory.contains(makeTes3RecordKey("SPEL", "common flu")) &&
              inventory.contains(makeTes3RecordKey("SPEL", "corprus immunity")),
          "Corprus Cure retains immunity and removes disease spells");
    const auto active = original.tes3().activeSpells().find(original.playerObject());
    check(active != original.tes3().activeSpells().end() &&
              active->second.size() == 1u &&
              active->second.front().spell == makeTes3RecordKey("SPEL", "corprus immunity"),
          "permanent immunity effect remains active after cure");
    const fs::path savePath = root / "cured.odai";
    const std::uint64_t expectedHash = original.deterministicHash();
    check(saveOdaiGameAtomic(savePath, original, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash &&
              restored.tes3().activeSpells() == original.tes3().activeSpells(),
          "Corprus Cure spellbook and permanent effect survive save/load");
    fs::remove_all(root, ec);
}

void testGetStandingPcUsesPhysicalSupport() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_standing_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-standing-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    const ObjectId platform = ObjectId::persistent(
        makeTes3ReferenceKey("Morrowind.esm", 0x42u));
    PhysicsCharacterSnapshot character;
    character.object = session.playerObject();
    character.grounded = true;
    character.supportingObject = platform;
    check(session.restorePhysicsSnapshots(std::span<const PhysicsCharacterSnapshot>(&character, 1u),
              error), error);
    const std::uint64_t supported = session.tes3().scripts().start(
        "StandingCheck", platform, error);
    check(supported != 0u, error);
    (void)session.advance(1.0 / 60.0);
    check(session.tes3().scripts().threads().at(supported).locals.at("standing").number == 1.0,
          "GetStandingPC reads the player's current supporting object");
    character.supportingObject.reset();
    character.grounded = false;
    check(session.restorePhysicsSnapshots(std::span<const PhysicsCharacterSnapshot>(&character, 1u),
              error), error);
    const std::uint64_t airborne = session.tes3().scripts().start(
        "StandingCheck", platform, error);
    check(airborne != 0u, error);
    (void)session.advance(1.0 / 60.0);
    check(session.tes3().scripts().threads().at(airborne).locals.at("standing").number == 0.0,
          "GetStandingPC clears when the player leaves the supporting object");
    fs::remove_all(root, ec);
}

void testRegionWeatherMutationAndSave() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_weather_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-weather-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    const std::uint64_t invalid = session.tes3().scripts().start(
        "BadWeather", session.playerObject(), error);
    check(invalid != 0u, error);
    (void)session.advance(1.0 / 60.0);
    check(session.tes3().scripts().threads().at(invalid).state == Tes3ThreadState::Failed &&
              !session.tes3().playerState().numericFilters.contains(
                  "regionweather:red mountain region:count"),
          "invalid ModRegion chance fails before mutating region weather");
    const std::uint64_t valid = session.tes3().scripts().start(
        "WeatherRoute", session.playerObject(), error);
    check(valid != 0u, error);
    (void)session.advance(1.0 / 60.0);
    const auto& state = session.tes3().playerState().numericFilters;
    check(session.tes3().scripts().threads().at(valid).locals.at("weather").number == 3.0 &&
              state.at("regionweather:red mountain region:count") == 8.0 &&
              state.at("regionweather:red mountain region:0") == 50.0 &&
              state.at("regionweather:red mountain region:1") == 50.0,
          "ModRegion and ChangeWeather update same-tick TES3 weather state");
    const fs::path savePath = root / "weather.odai";
    const std::uint64_t expectedHash = session.deterministicHash();
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash &&
              restored.tes3().playerState().numericFilters == state,
          "region weather chances and forced weather survive save/load");
    fs::remove_all(root, ec);
}

void testUnloadedActorHealthPersists() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_unloaded_health_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-unloaded-health-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    const std::uint64_t threadId = session.tes3().scripts().start(
        "UnloadedHealth", session.playerObject(), error);
    check(threadId != 0u, error);
    (void)session.advance(1.0 / 60.0);
    const ObjectId actor = ObjectId::persistent(
        makeTes3ReferenceKey("Morrowind.esm", 0x43u));
    const auto saved = session.tes3().referenceOverrides().find(actor);
    check(session.tes3().scripts().threads().at(threadId).state == Tes3ThreadState::Completed &&
              session.tes3().scripts().threads().at(threadId).locals.at("health").number == 0.0 &&
              saved != session.tes3().referenceOverrides().end() &&
              saved->second.locals.at("actor:health").number == 0.0 &&
              saved->second.locals.at("actor:dead").number == 1.0,
          "SetHealth affects an unloaded actor and same-tick GetHealth reads the override");
    const fs::path savePath = root / "actor.odai";
    const std::uint64_t expectedHash = session.deterministicHash();
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash &&
              restored.tes3().referenceOverrides().at(actor).locals.at("actor:health").number == 0.0,
          "unloaded actor health and death state survive save/load");
    fs::remove_all(root, ec);
}

void testDialogueUsesCurrentPlayerInventory() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_inventory_dialogue_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const std::shared_ptr<Tes3ContentStore> content = makeContent(root);

    BethesdaSession session;
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-inventory-dialogue-fixture";
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeTes3RecordKey("NPC_", "player");
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    check(session.world().addInitialObject(std::move(player), error), error);

    Tes3DialogueActorState actor;
    actor.object = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x42u));
    actor.id = "temple priest";
    Tes3DialoguePlayerState stalePlayer;
    stalePlayer.object = session.playerObject();
    const RecordKey package = makeTes3RecordKey("MISC", "bk_a1_1_caiuspackage");
    WorldCommand add;
    add.type = WorldCommandType::AddItem;
    add.target = session.playerObject();
    add.item = package;
    add.itemCount = 1;
    check(session.world().queue(std::move(add)), "queue a package pickup");
    (void)session.advance(1.0 / 60.0);
    check(session.startTes3Dialogue(actor, stalePlayer).text == "You have the package.",
          "TES3 item filter observes the player's current world inventory");
    check(session.tes3().playerState().inventory.at(package) == 1,
          "TES3 player state receives the package after world command application");
    session.tes3().endDialogue();

    WorldCommand remove;
    remove.type = WorldCommandType::RemoveItem;
    remove.target = session.playerObject();
    remove.item = package;
    remove.itemCount = 1;
    check(session.world().queue(std::move(remove)), "queue delivery of the package");
    (void)session.advance(1.0 / 60.0);
    check(session.startTes3Dialogue(actor, stalePlayer).text != "You have the package.",
          "TES3 item filter stops matching after the package is delivered");
    check(!session.tes3().playerState().inventory.contains(package),
          "TES3 player state removes delivered items");

    session.tes3().endDialogue();
    check(session.startTes3Dialogue(actor, stalePlayer).accepted,
          "conversation can restart without the package");
    check(session.tes3().addTopic("Package Grant") &&
              session.tes3().addTopic("Package Receipt"),
          "fixture topics are available for the scripted transfer");
    check(session.selectTes3Topic("Package Grant").accepted,
          "authored dialogue result grants the package");
    check(session.selectTes3Topic("Package Receipt").accepted,
          "next topic sees an item granted by the preceding result script");
    check(session.tes3().playerState().inventory.at(package) == 1,
          "TES3 item mirror updates before the queued world command is applied");
    (void)session.advance(1.0 / 60.0);
    const RuntimeObject* worldPlayer = session.world().find(session.playerObject());
    check(worldPlayer != nullptr && worldPlayer->inventory.size() == 1u &&
              worldPlayer->inventory.front().item == package,
          "world inventory receives the scripted item at the next tick");
    fs::remove_all(root, ec);
}

void testArtifactEquipScriptAndSave() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_artifact_equip_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-artifact-equip-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    const RecordKey artifact = makeTes3RecordKey("ARMO", "test_wraithguard");
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeTes3RecordKey("NPC_", "player");
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    player.inventory.push_back({artifact, 1});
    check(session.world().addInitialObject(std::move(player), error), error);
    check(session.useInventoryItem(session.playerObject(), artifact, error), error);
    (void)session.advance(1.0 / 60.0);
    const auto* worldPlayer = session.world().find(session.playerObject());
    check(worldPlayer != nullptr && worldPlayer->inventory.front().equipped,
          "TES3 armor equip applies to the owned inventory instance");
    (void)session.advance(1.0 / 60.0);
    check(session.tes3().scripts().globals().at("testartifactequipped").number == 1.0,
          "equipped item's authored OnPCEquip script observes the event");
    const fs::path savePath = root / "artifact.odai";
    const std::uint64_t expectedHash = session.deterministicHash();
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash &&
              restored.world().find(restored.playerObject())->inventory.front().equipped,
          "equipped TES3 artifact and its running script survive save/load");
    check(restored.useInventoryItem(restored.playerObject(), artifact, error), error);
    (void)restored.advance(1.0 / 60.0);
    (void)restored.advance(1.0 / 60.0);
    check(!restored.world().find(restored.playerObject())->inventory.front().equipped &&
              restored.tes3().scripts().globals().at("testartifactequipped").number == 0.0,
          "unequip sends OnPCEquip zero to the persistent item script");
    fs::remove_all(root, ec);
}

void testScriptedWorldPickup() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_scripted_pickup_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-scripted-pickup-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeTes3RecordKey("NPC_", "player");
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    check(session.world().addInitialObject(std::move(player), error), error);
    const ObjectId id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x78u));
    RuntimeObject paper;
    paper.id = id;
    paper.base = makeTes3RecordKey("MISC", "test_scripted_paper");
    paper.kind = RuntimeObjectKind::Item;
    check(session.world().addInitialObject(std::move(paper), error), error);
    check(session.tes3().scripts().start("ScriptedPickup", id, error, true, true) != 0u,
          error);
    check(session.activateTes3Reference(id, false, error), error);
    const auto first = session.advance(1.0 / 60.0);
    check(first.diagnostics.empty() &&
              session.tes3().playerState().inventory.at(
                  makeTes3RecordKey("MISC", "test_scripted_paper")) == 1 &&
              !session.world().find(id)->enabled,
          "scripted activation transfers the world item once and hides its source");
    (void)session.advance(1.0 / 60.0);
    check(session.tes3().scripts().globals().at("pickupcount").number == 1.0 &&
              session.tes3().scripts().globals().at("pickedup").number == 1.0,
          "OnActivate is consumed while OnPCAdd reaches the attached script");
    const ObjectId plainId = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x79u));
    RuntimeObject plain;
    plain.id = plainId;
    plain.base = makeTes3RecordKey("MISC", "test_scripted_paper");
    plain.kind = RuntimeObjectKind::Item;
    check(session.world().addInitialObject(std::move(plain), error), error);
    check(session.activateTes3Reference(plainId, false, error), error);
    check(!session.activateTes3Reference(plainId, false, error) &&
              session.tes3().playerState().inventory.at(
                  makeTes3RecordKey("MISC", "test_scripted_paper")) == 2,
          "a second pickup before the world command applies cannot duplicate the item");
    (void)session.advance(1.0 / 60.0);
    const fs::path savePath = root / "pickup.odai";
    const std::uint64_t expectedHash = session.deterministicHash();
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash &&
              !restored.world().find(id)->enabled,
          "scripted world pickup persists after save/load");
    fs::remove_all(root, ec);
}

void testAiPackageSameTickAndSave() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_ai_package_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-ai-package-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    const ObjectId actor = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x43u));
    RuntimeObject object;
    object.id = actor;
    object.base = makeTes3RecordKey("NPC_", "quest_actor");
    object.kind = RuntimeObjectKind::Actor;
    object.actorValues.emplace();
    check(session.world().addInitialObject(std::move(object), error), error);
    check(session.tes3().scripts().start("AiRouteProbe", actor, error) != 0u, error);
    const auto step = session.advance(1.0 / 60.0);
    check(step.diagnostics.empty() &&
              session.tes3().scripts().globals().at("package").number == 1.0 &&
              session.tes3().scripts().globals().at("done").number == 0.0,
          "AI package type and pending completion are visible in the issuing tick");
    check(session.world().find(actor)->aiState->wanderTarget ==
              std::array<float, 3>{1.0f, 3.0f, -2.0f},
          "AITravel maps the authored Z-up destination to engine space");
    const fs::path savePath = root / "ai.odai";
    const std::uint64_t expectedHash = session.deterministicHash();
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash,
          "AI package state and same-tick script result survive save/load");
    RuntimeObject player;
    player.id = restored.playerObject();
    player.base = makeTes3RecordKey("NPC_", "player");
    player.kind = RuntimeObjectKind::Actor;
    check(restored.world().addInitialObject(std::move(player), error), error);
    check(restored.tes3().scripts().start("EscortRouteProbe", actor, error) != 0u, error);
    const auto escort = restored.advance(1.0 / 60.0);
    check(escort.diagnostics.empty() &&
              restored.world().find(actor)->aiState->wanderTarget ==
                  std::array<float, 3>{195.0f, 170.0f, -100.0f} &&
              !restored.world().find(actor)->navigationRequest.has_value() &&
              restored.tes3().scripts().globals().at("package").number == 2.0 &&
              restored.tes3().scripts().globals().at("done").number == 0.0,
          "ship escort leads to the authored hatch rather than following the player");
    fs::remove_all(root, ec);
}

void testScriptedDoorActivation() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_scripted_door_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-scripted-door-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeTes3RecordKey("NPC_", "player");
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    check(session.world().addInitialObject(std::move(player), error), error);
    const ObjectId id = ObjectId::persistent(makeTes3ReferenceKey("fixture.esp", 0x80u));
    RuntimeObject door;
    door.id = id;
    door.kind = RuntimeObjectKind::Door;
    door.base = makeTes3RecordKey("DOOR", "scripted_door");
    check(session.world().addInitialObject(std::move(door), error), error);
    check(session.tes3().scripts().start("ScriptedDoor", id, error, true, true) != 0u,
          error);
    check(session.activateTes3Reference(id, false, error) &&
              session.takeTes3DoorActivations().empty(),
          "player activation waits for the door's local script");
    const auto step = session.advance(1.0 / 60.0);
    const auto activations = session.takeTes3DoorActivations();
    check(step.diagnostics.empty() && activations.size() == 1u && activations.front() == id,
          "the script's Activate requests the door transition once");
    (void)session.advance(1.0 / 60.0);
    check(session.takeTes3DoorActivations().empty(),
          "consumed OnActivate does not reopen the door on later ticks");
    fs::remove_all(root, ec);
}

void testFinaleShieldAndRestoreAttribute() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_finale_effect_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-finale-effects-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    RuntimeObject player;
    player.id = session.playerObject();
    player.base = makeTes3RecordKey("NPC_", "player");
    player.kind = RuntimeObjectKind::Actor;
    player.actorValues.emplace();
    check(session.world().addInitialObject(std::move(player), error), error);
    const ObjectId actor = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x43u));
    RuntimeObject npc;
    npc.id = actor;
    npc.base = makeTes3RecordKey("NPC_", "quest_actor");
    npc.kind = RuntimeObjectKind::Actor;
    npc.actorValues.emplace();
    check(session.world().addInitialObject(std::move(npc), error), error);
    session.tes3().playerState().numericFilters["agility"] = 40.0;
    session.tes3().playerState().numericFilters["damage:agility"] = 25.0;
    check(session.tes3().scripts().start(
              "FinaleEffectProbe", session.playerObject(), error) != 0u, error);
    const auto effectStep = session.advance(1.0 / 60.0);
    check(effectStep.diagnostics.empty(),
          "Dagoth-style shield and RestoreAttribute script completes");
    const auto active = session.tes3().activeSpells().find(actor);
    check(active != session.tes3().activeSpells().end() &&
              active->second.size() == 1u && active->second.front().effects.size() == 4u,
          "shield and three elemental shields become active passive effects");
    const auto& globals = session.tes3().scripts().globals();
    check(globals.at("shieldseen").number == 1.0 &&
              globals.at("beforerestore").number == 15.0 &&
              globals.at("afterrestore").number == 40.0,
          "RestoreAttribute clears damage between same-tick script queries");
    const fs::path savePath = root / "effects.odai";
    const std::uint64_t expectedHash = session.deterministicHash();
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash &&
              restored.tes3().activeSpells().at(actor).front().effects.size() == 4u,
          "finale shield state and restored attribute persist after save/load");
    fs::remove_all(root, ec);
}

void testWeaponHitAndCrossReferenceLocal() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_heart_hit_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-heart-hit-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    RuntimeObject player;
    player.id = session.playerObject();
    player.kind = RuntimeObjectKind::Actor;
    player.base = makeTes3RecordKey("NPC_", "player");
    player.actorValues.emplace();
    check(session.world().addInitialObject(std::move(player), error), error);
    const ObjectId heart = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 0x43u));
    RuntimeObject actor;
    actor.id = heart;
    actor.kind = RuntimeObjectKind::Actor;
    actor.base = makeTes3RecordKey("NPC_", "quest_actor");
    actor.actorValues.emplace();
    check(session.world().addInitialObject(std::move(actor), error), error);
    check(session.tes3().scripts().start("HeartProbe", heart, error, true, true) != 0u,
          error);
    check(session.tes3().scripts().start(
              "HeartWatcher", session.playerObject(), error) != 0u, error);
    session.dispatchTes3GameplayEvent("hitonme", heart,
        Tes3Value::fromString("sunder"));
    const auto step = session.advance(1.0 / 60.0);
    check(step.diagnostics.empty() &&
              session.tes3().scripts().globals().at("observedhits").number == 2.0,
          "HitOnMe consumes the matching weapon and another script reads the heart local");
    const auto thread = std::find_if(session.tes3().scripts().threads().begin(),
        session.tes3().scripts().threads().end(), [](const auto& pair) {
            return pair.second.program == "heartprobe";
        });
    check(thread != session.tes3().scripts().threads().end() &&
              !thread->second.eventVariables.contains("hitonme"),
          "matching HitOnMe clears the attack event");
    fs::remove_all(root, ec);
}

void testSayDoneWaitsAcrossTicks() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_voice_gate_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-voice-gate-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    const std::uint64_t id = session.tes3().scripts().start(
        "VoiceGate", session.playerObject(), error, true);
    check(id != 0u, error);
    (void)session.advance(1.0 / 60.0);
    (void)session.advance(1.0 / 60.0);
    check(session.tes3().scripts().threads().at(id).locals.at("state").number == 1.0,
          "SayDone holds the authored branch while the voiced line is active");
    for (int i = 0; i < 90; ++i) (void)session.advance(1.0 / 60.0);
    check(session.tes3().scripts().threads().at(id).locals.at("state").number == 2.0,
          "SayDone allows the next script state after the voice gate ends");
    fs::remove_all(root, ec);
}

void testCharacterMenuRequestAndSave() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_character_menu_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-character-menu-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    const std::uint64_t id = session.tes3().scripts().start(
        "MenuRequest", session.playerObject(), error);
    check(id != 0u, error);
    const auto step = session.advance(1.0 / 60.0);
    check(step.diagnostics.empty() &&
              session.tes3().scripts().threads().at(id).locals.at("opened").number == 1.0 &&
              session.tes3().playerState().numericFilters.at("chargen:menu") == 1.0,
          "EnableNameMenu opens the UI request observed by same-tick MenuMode");
    auto& player = session.tes3().playerState();
    player.name = "Nerevarine";
    player.race = "Dark Elf";
    player.actorClass = "Warrior";
    player.birthsign = "The Lady";
    player.head = "1";
    player.hair = "2";
    player.gender = 1;
    player.numericFilters["chargen:menu"] = 0.0;
    const fs::path savePath = root / "character.odai";
    const std::uint64_t expectedHash = session.deterministicHash();
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash &&
              restored.tes3().playerState().name == "Nerevarine" &&
              restored.tes3().playerState().race == "Dark Elf" &&
              restored.tes3().playerState().actorClass == "Warrior" &&
              restored.tes3().playerState().birthsign == "The Lady" &&
              restored.tes3().playerState().head == "1" &&
              restored.tes3().playerState().hair == "2" &&
              restored.tes3().playerState().gender == 1,
          "authored character choices and closed menu survive save/load");
    fs::remove_all(root, ec);
}

void testWorldMessageChoiceAndSave() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_world_prompt_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-world-prompt-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    const std::uint64_t thread = session.tes3().scripts().start(
        "MessagePrompt", session.playerObject(), error);
    check(thread != 0u, error);
    const auto pending = session.advance(1.0 / 60.0);
    check(pending.diagnostics.empty() &&
              session.tes3().scripts().threads().at(thread).state ==
                  Tes3ThreadState::Suspended &&
              !session.tes3().dialogue().active &&
              session.tes3().dialogue().messageBoxText == "Continue?" &&
              session.tes3().dialogue().choices.size() == 2u,
          "a world script exposes its authored MessageBox text and choices");
    const auto gate = session.tes3().scripts().start(
        "TutorialMenuGate", session.playerObject(), error, true);
    check(gate != 0u, error);
    check(session.advance(1.0 / 60.0).diagnostics.empty() &&
              session.tes3().scripts().threads().at(gate).locals.at("advanced").number == 0.0,
          "other opening scripts wait while a tutorial MessageBox is open");
    const fs::path savePath = root / "prompt.odai";
    const std::uint64_t expectedHash = session.deterministicHash();
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.deterministicHash() == expectedHash &&
              restored.tes3().dialogue().messageBoxText == "Continue?",
          "pending world MessageBox and text survive save/load");
    check(restored.answerTes3Choice(1, false).accepted &&
              restored.tes3().dialogue().messageBoxText.empty(),
          "answering the world MessageBox clears its presentation");
    const auto completed = restored.advance(1.0 / 60.0);
    check(completed.diagnostics.empty() &&
              restored.tes3().scripts().threads().at(thread).state ==
                  Tes3ThreadState::Completed &&
              restored.tes3().scripts().threads().at(thread).locals.at("selected").number == 1.0,
          "the saved script resumes with the selected button value");
    check(restored.tes3().scripts().threads().at(gate).locals.at("advanced").number == 1.0,
          "opening scripts advance after the saved tutorial prompt is acknowledged");
    fs::remove_all(root, ec);
}

void testDontSaveObjectSkipsWorldSnapshot() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_transient_object_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-transient-object-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    const ObjectId id = ObjectId::persistent(makeTes3ReferenceKey("fixture.esp", 0x77u));
    RuntimeObject object;
    object.id = id;
    object.base = makeTes3RecordKey("CREA", "kwama_worker");
    object.kind = RuntimeObjectKind::Actor;
    object.actorValues.emplace();
    check(session.world().addInitialObject(std::move(object), error), error);
    const std::uint64_t threadId = session.tes3().scripts().start("TransientObject", id, error);
    check(threadId != 0u, error);
    const auto step = session.advance(1.0 / 60.0);
    check(step.diagnostics.empty() && !session.world().find(id)->saveObject,
          "DontSaveObject marks its resident owner as transient");
    const fs::path savePath = root / "transient.odai";
    const std::uint64_t expectedHash = session.deterministicHash();
    check(saveOdaiGameAtomic(savePath, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error);
    check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(savePath, restored, {}, report, error), error);
    check(restored.world().find(id) == nullptr &&
              restored.deterministicHash() == expectedHash,
          "transient object is absent after load and deterministic saved state is retained");
    fs::remove_all(root, ec);
}

void testGetDetectedUsesResidentVisibility() {
    const fs::path root = fs::temp_directory_path() / "odai_tes3_detected_tests";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const auto content = makeContent(root);
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind;
    config.contentFingerprint = "tes3-detected-fixture";
    BethesdaSession session;
    std::string error;
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    RuntimeObject observer;
    observer.id = ObjectId::persistent(makeTes3ReferenceKey("fixture.esp", 0x81u));
    observer.base = makeTes3RecordKey("NPC_", "observer_actor");
    observer.kind = RuntimeObjectKind::Actor;
    observer.actorValues.emplace();
    observer.currentSpace.cell = makeTes3RecordKey("CELL", "Test Cell");
    RuntimeObject target;
    target.id = ObjectId::persistent(makeTes3ReferenceKey("fixture.esp", 0x82u));
    target.base = makeTes3RecordKey("NPC_", "target_actor");
    target.kind = RuntimeObjectKind::Actor;
    target.actorValues.emplace();
    target.currentSpace.cell = observer.currentSpace.cell;
    target.transform.position[0] = 100.0;
    check(session.world().addInitialObject(observer, error), error);
    check(session.world().addInitialObject(target, error), error);
    const std::uint64_t nearThread = session.tes3().scripts().start(
        "DetectionProbe", observer.id, error);
    check(nearThread != 0u, error);
    const auto nearStep = session.advance(1.0 / 60.0);
    check(nearStep.diagnostics.empty() &&
              session.tes3().scripts().threads().at(nearThread).locals.at("seen").number == 1.0,
          "GetDetected sees a nearby enabled actor in the same clear cell");
    session.world().find(target.id)->currentSpace.cell =
        makeTes3RecordKey("CELL", "Another Cell");
    const std::uint64_t farThread = session.tes3().scripts().start(
        "DetectionProbe", observer.id, error);
    const auto farStep = session.advance(1.0 / 60.0);
    check(farThread != 0u && farStep.diagnostics.empty() &&
              session.tes3().scripts().threads().at(farThread).locals.at("seen").number == 0.0,
          "GetDetected rejects an actor in another cell");
    fs::remove_all(root, ec);
}

void testMainQuestDialogueFiltersAndRecognitions() {
    const fs::path root = fs::temp_directory_path() / "odai_mech_tes3_002";
    fs::create_directories(root);
    std::vector<std::uint8_t> file;
    header(file);
    std::vector<std::uint8_t> enemyRecord;
    sub(enemyRecord, "NAME", "route_enemy"); record(file, "NPC_", enemyRecord);
    std::vector<std::uint8_t> playerRecord;
    sub(playerRecord, "NAME", "player"); record(file, "NPC_", playerRecord);
    const auto factionRecord = [&](const char* id) {
        std::vector<std::uint8_t> body;
        sub(body, "NAME", id);
        std::vector<std::uint8_t> fadt;
        append(fadt, std::int32_t{0}); append(fadt, std::int32_t{1});
        for (int i = 0; i < 10; ++i) {
            for (int value : {30, 35, 40, 30, 10}) append(fadt, std::int32_t(value));
        }
        for (int value : {0, 1, 2, -1, -1, -1, -1})
            append(fadt, std::int32_t(std::string(id) == "sparse" ? -1 : value));
        append(fadt, std::int32_t{0}); sub(body, "FADT", fadt);
        sub(body, "ANAM", "friendly"); std::vector<std::uint8_t> reaction;
        append(reaction, std::int32_t{10}); sub(body, "INTV", reaction);
        sub(body, "ANAM", "hostile"); reaction.clear();
        append(reaction, std::int32_t{-10}); sub(body, "INTV", reaction);
        record(file, "FACT", body);
    };
    factionRecord("house"); factionRecord("friendly"); factionRecord("hostile");
    factionRecord("sparse");
    std::vector<std::uint8_t> clothingRecord;
    sub(clothingRecord, "NAME", "valuable_robe");
    std::vector<std::uint8_t> clothingData;
    append(clothingData, std::int32_t{4}); append(clothingData, 1.0f);
    append(clothingData, std::uint16_t{200}); append(clothingData, std::uint16_t{0});
    sub(clothingRecord, "CTDT", clothingData); record(file, "CLOT", clothingRecord);
    dial(file, "Greeting 1", 2);
    info(file, "greeting", "", "", 2, 0, "Welcome.");
    const auto filtered = [&](const std::string& id, const std::string& rule, int expected) {
        dial(file, id, 0);
        info(file, id + "_info", "", "", 0, 0, id, {}, nullptr, rule, expected);
    };
    filtered("Global Gate", "02000routeglobal", 0);
    filtered("Missing Global", "02000missing", 0);
    filtered("Local Gate", "03000stage", 2);
    filtered("Missing Local", "03000missing", 0);
    filtered("Missing Not Local", "0C000missing", 0);
    filtered("Invalid Condition", "broken", 0);
    filtered("Not Local", "0C003stage", 2); // NOT(stage >= 2), not NOT(stage == 0)
    filtered("Dead Gate", "06002route_enemy", 0);
    filtered("Journal Gate", "04000Recognition0", 0);
    filtered("Missing Journal", "04000missing", 0);
    filtered("Choice Without Prompt", "01501", 99);
    filtered("Talked Before", "01630", 1);
    filtered("Unknown Function", "01700", 0);
    filtered("Not Cell", "0B000Balmora", 0);
    filtered("Not Id", "07000other_actor", 0);
    filtered("Not Faction", "08000other_faction", 0);
    filtered("Not Class", "09000other_class", 0);
    filtered("Not Race", "0A000other_race", 0);
    filtered("Player Level", "01063", 3);
    filtered("Player Reputation", "01053", 5);
    filtered("Player Strength", "01103", 40);
    filtered("Same Faction", "01460", 1);
    filtered("Corprus Gate", "01580", 1);
    filtered("Health Gate", "01073", 50);
    filtered("Exact Health Percent", "01070", 50);
    filtered("Actor Reputation", "01030", 5);
    filtered("Actor Health Percent", "01040", 50);
    filtered("Faction Rank Difference", "01470", -4);
    filtered("Player Health Raw", "01643", 50);
    filtered("Expelled Gate", "01390", 1);
    filtered("Rank Requirements", "01020", 3);
    filtered("Lowest Reaction", "01000", -10);
    filtered("Highest Reaction", "01010", 10);
    filtered("Clothing Gate", "01423", 200);
    // Static filters in imported DATA and actor/player/cell subrecords.
    dial(file, "Actor Gate", 0);
    std::vector<std::uint8_t> body;
    sub(body, "INAM", "actor_gate");
    auto data = infoData(0, 60);
    data[8] = 2; data[9] = 1; data[10] = 1;
    sub(body, "DATA", data);
    sub(body, "ONAM", "quest_actor"); sub(body, "RNAM", "race");
    sub(body, "CNAM", "class"); sub(body, "FNAM", "house");
    sub(body, "ANAM", "Balmora"); sub(body, "NAME", "All actor filters pass.");
    record(file, "INFO", body);
    std::vector<std::uint8_t> global;
    sub(global, "NAME", "routeglobal"); sub(global, "FNAM", std::vector<std::uint8_t>{'s'});
    std::vector<std::uint8_t> zero; append(zero, 0.0f); sub(global, "FLTV", zero);
    record(file, "GLOB", global);
    // A declared speaker local starts at zero, changes in the result, and gates rewards on reopen.
    std::vector<std::uint8_t> actorRecord;
    sub(actorRecord, "NAME", "local_actor"); sub(actorRecord, "SCRI", "SpeakerScript");
    record(file, "NPC_", actorRecord);
    std::vector<std::uint8_t> cell;
    sub(cell, "NAME", "Test Cell");
    std::vector<std::uint8_t> cellData(12, 0); cellData[0] = 1;
    sub(cell, "DATA", cellData);
    std::vector<std::uint8_t> frmr; append(frmr, std::uint32_t{42});
    sub(cell, "FRMR", frmr); sub(cell, "NAME", "local_actor");
    std::vector<std::uint8_t> enemyFrmr; append(enemyFrmr, std::uint32_t{43});
    sub(cell, "FRMR", enemyFrmr); sub(cell, "NAME", "route_enemy");
    record(file, "CELL", cell);
    std::vector<std::uint8_t> script;
    std::vector<std::uint8_t> scriptHeader(52, 0);
    std::memcpy(scriptHeader.data(), "SpeakerScript", 13);
    sub(script, "SCHD", scriptHeader);
    sub(script, "SCTX", "begin SpeakerScript\nshort rewarded\nend");
    record(file, "SCPT", script);
    dial(file, "Local Reward", 0);
    info(file, "local_reward", "", "", 0, 0, "Reward.",
        "set rewarded to 1\nModReputation 5\nPlayer->ModReputation 2\nModDisposition 10\n"
        "route_enemy->ModDisposition 20\nroute_enemy->ModReputation 30\nPlayer->SetHealth 25", nullptr, "03000rewarded", 0);
    for (int i = 0; i < 7; ++i) {
        const std::string quest = "Recognition" + std::to_string(i);
        dial(file, quest, 4);
        info(file, quest + "10", "", "", 4, 10, "Recognized.");
        dial(file, "Recognize " + std::to_string(i), 0);
        info(file, "recognize" + std::to_string(i), "", "", 0, 0, "Recognition.",
            "Journal " + quest + " 10\nPlayer->ModReputation 1", nullptr, "04000" + quest, 0);
    }
    filtered("Inventory Gate", "05IX2route_item", 0);
    const fs::path plugin = root / "Morrowind.esm";
    { std::ofstream out(plugin, std::ios::binary); out.write(reinterpret_cast<const char*>(file.data()), file.size()); }
    FalloutLoadOrder order;
    std::string error;
    check(order.open(root, {"Morrowind.esm"}, error), error);
    auto content = std::make_shared<Tes3ContentStore>();
    check(content->load(order, "windows-1252", error), error);
    BethesdaSession session;
    BethesdaSessionConfig config;
    config.game = BethesdaGame::Morrowind; config.contentFingerprint = "mech-tes3-002-fixture";
    check(session.configure(config, error), error);
    check(session.configureTes3Content(content, error), error);
    RuntimeObject worldPlayer;
    worldPlayer.id = session.playerObject();
    worldPlayer.base = makeTes3RecordKey("NPC_", "player");
    worldPlayer.kind = RuntimeObjectKind::Actor; worldPlayer.actorValues.emplace();
    check(session.world().addInitialObject(worldPlayer, error), error);
    Tes3DialogueActorState actor;
    actor.object = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 42));
    actor.id = "quest_actor"; actor.race = "race"; actor.actorClass = "class";
    actor.faction = "house"; actor.rank = 3; actor.gender = 1;
    actor.disposition = 60; actor.cell = "Balmora, Council Club";
    actor.locals["stage"] = 2;
    Tes3DialoguePlayerState player;
    player.object = session.playerObject(); player.factionRanks["house"] = 2;
    player.numericFilters = {{"level", 3}, {"reputation", 5}, {"strength", 40}, {"intelligence", 35},
        {"block", 40}, {"armorer", 30}, {"mediumarmor", 30}, {"faction_rep:house", 10},
        {"health", 50}, {"maxhealth", 100}, {"expelled:house", 1}};
    player.factionRanks["friendly"] = 0; player.factionRanks["hostile"] = 0;
    player.deathCounts["route_enemy"] = 1;
    auto& runtime = session.tes3();
    check(runtime.startDialogue(actor, player).accepted && runtime.addTopic("Talked Before") &&
        !runtime.selectTopic("Talked Before").accepted,
        "TalkedToPc snapshots history before the greeting");
    check(runtime.startDialogue(actor, player).accepted && runtime.selectTopic("Talked Before").accepted,
        "a reopened conversation observes the prior greeting");
    const auto gate = [&](const std::string& topic, bool expected, bool strict = true) {
        check(runtime.startDialogue(actor, player, strict).accepted, "filter fixture greeting");
        check(runtime.addTopic(topic), "fixture topic imported: " + topic);
        check(runtime.selectTopic(topic, strict).accepted == expected, "filter eligibility: " + topic);
    };
    for (const char* topic : {"Global Gate", "Local Gate", "Dead Gate", "Journal Gate", "Actor Gate",
            "Not Id", "Not Faction", "Not Class", "Not Race", "Player Level", "Player Reputation",
            "Player Strength", "Same Faction", "Health Gate", "Expelled Gate", "Rank Requirements",
            "Lowest Reaction", "Highest Reaction"}) gate(topic, true);
    player.numericFilters.erase("intelligence"); gate("Rank Requirements", false);
    player.numericFilters["intelligence"] = 35;
    player.numericFilters["block"] = 39; gate("Rank Requirements", false);
    player.numericFilters["block"] = 40;
    player.numericFilters["faction_rep:house"] = 9; gate("Rank Requirements", false);
    player.numericFilters["faction_rep:house"] = 10;
    actor.faction = "sparse"; player.numericFilters["faction_rep:sparse"] = 10;
    gate("Rank Requirements", true);
    actor.faction = "house";
    player.factionRanks["house"] = 9; gate("Rank Requirements", false);
    player.factionRanks["house"] = 2;
    auto* owned = session.world().find(session.playerObject());
    const auto robe = makeTes3RecordKey("CLOT", "valuable_robe");
    owned->inventory.push_back({robe, 1});
    check(session.equipActorItem(session.playerObject(), robe, true, false, error), error);
    check(session.advance(1.0 / 60.0).diagnostics.empty(), "clothing equip applies");
    auto clothPlayer = runtime.playerState();
    check(session.startTes3Dialogue(actor, clothPlayer).accepted, "clothing fixture greeting");
    check(runtime.addTopic("Clothing Gate") && session.selectTes3Topic("Clothing Gate").accepted,
        "clothing modifier reads equipped item value from imported CTDT");
    check(session.equipActorItem(session.playerObject(), robe, false, false, error), error);
    (void)session.advance(1.0 / 60.0);
    check(session.startTes3Dialogue(actor, runtime.playerState()).accepted &&
        !session.selectTes3Topic("Clothing Gate").accepted, "unequipping updates clothing eligibility");
    gate("Corprus Gate", false);
    Tes3ActiveSpell corprus;
    Tes3ActiveSpellEffect disease;
    disease.effectId = 132; disease.magnitude = 1;
    disease.expiresTick = std::numeric_limits<std::uint64_t>::max();
    corprus.effects.push_back(disease);
    runtime.activeSpellsForRestore()[player.object].push_back(corprus);
    gate("Corprus Gate", true);
    runtime.activeSpellsForRestore().clear();
    player.numericFilters["health"] = 49; gate("Health Gate", false);
    player.numericFilters["health"] = 50.9; gate("Exact Health Percent", true);
    player.numericFilters["health"] = 51; gate("Exact Health Percent", false);
    player.numericFilters["health"] = 50;
    player.numericFilters["damage:strength"] = 1; gate("Player Strength", false);
    Tes3ActiveSpell fortify;
    Tes3ActiveSpellEffect attributeEffect;
    attributeEffect.effectId = 79; attributeEffect.attribute = 0; attributeEffect.magnitude = 1;
    attributeEffect.expiresTick = std::numeric_limits<std::uint64_t>::max();
    fortify.effects.push_back(attributeEffect);
    runtime.activeSpellsForRestore()[player.object].push_back(fortify);
    gate("Player Strength", true);
    runtime.activeSpellsForRestore().clear(); player.numericFilters.erase("damage:strength");
    player.numericFilters["strength"] = std::numeric_limits<double>::quiet_NaN();
    gate("Player Strength", false); player.numericFilters["strength"] = 40;
    player.numericFilters["expelled:house"] = 0; gate("Expelled Gate", false);
    for (const char* topic : {"Missing Global", "Missing Local", "Missing Journal", "Not Local",
            "Choice Without Prompt", "Unknown Function", "Not Cell", "Inventory Gate", "Missing Not Local", "Invalid Condition"}) {
        gate(topic, false); gate(topic, false, false);
    }
    runtime.playerState().inventory[makeTes3RecordKey("MISC", "route_item")] = 1;
    player.inventory = runtime.playerState().inventory;
    gate("Inventory Gate", true);
    actor.locals["stage"] = 1; gate("Local Gate", false); gate("Not Local", true);
    player.deathCounts.clear(); gate("Dead Gate", false);
    actor.disposition = 59; gate("Actor Gate", false); actor.disposition = 60;
    actor.rank = 1; gate("Actor Gate", false); actor.rank = 3;
    actor.gender = 0; gate("Actor Gate", false); actor.gender = 1;
    actor.id = "other_actor"; gate("Actor Gate", false); actor.id = "quest_actor";
    actor.race = "other_race"; gate("Actor Gate", false); actor.race = "race";
    actor.actorClass = "other_class"; gate("Actor Gate", false); actor.actorClass = "class";
    actor.faction = "other_faction"; gate("Actor Gate", false); actor.faction = "house";
    actor.cell = "Ald'ruhn"; gate("Actor Gate", false); gate("Not Cell", true);
    actor.cell = "Balmora"; player.factionRanks.clear(); gate("Actor Gate", false);
    gate("Faction Rank Difference", true);
    actor.id = "local_actor"; actor.locals.clear();
    actor.object = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 42));
    gate("Missing Not Local", true);
    gate("Missing Local", false);
    gate("Local Reward", true);
    check(runtime.playerState().numericFilters.at("reputation") == 7 &&
        runtime.referenceOverrides().at(actor.object).locals.at("stat:reputation").number == 5,
        "speaker and qualified player reputation mutations target separate actors");
    const auto& otherStats = runtime.referenceOverrides().at(
        ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 43))).locals;
    check(otherStats.at("stat:disposition").number == 70 && otherStats.at("stat:reputation").number == 30,
        "qualified disposition and reputation target an unloaded actor rather than the speaker");
    check(runtime.addTopic("Actor Reputation") && runtime.selectTopic("Actor Reputation").accepted,
        "speaker reputation filter observes its preceding result");
    actor.locals["actor:health"] = 50.9; actor.locals["actor:maxhealth"] = 100;
    player = runtime.playerState(); gate("Actor Health Percent", true);
    actor.locals["actor:health"] = 51; gate("Actor Health Percent", false);
    actor.locals.erase("actor:health"); actor.locals.erase("actor:maxhealth");
    check(!runtime.selectTopic("Local Reward").accepted,
        "an earlier result updates local reward eligibility in the same conversation");
    check(runtime.addTopic("Player Health Raw") && !runtime.selectTopic("Player Health Raw").accepted &&
        runtime.playerState().numericFilters.at("health") == 25,
        "result-script health changes update player filters within the same conversation");
    player = runtime.playerState();
    gate("Local Reward", false);
    check(runtime.scripts().globals().find("rewarded") == runtime.scripts().globals().end(),
        "speaker-local result assignment never becomes a global");
    check(runtime.dialogue().actor.disposition == 60, "scripted disposition survives conversation reopen");
    // No stage setup: all seven independent journal transitions run through topic results.
    const int recognitionOrder[] = {4, 0, 6, 1, 5, 3, 2};
    for (int i : recognitionOrder) {
        gate("Recognize " + std::to_string(i), true);
        player = runtime.playerState();
        gate("Recognize " + std::to_string(i), false);
        check(runtime.journal().index("Recognition" + std::to_string(i)) == 10,
            "recognition progresses independently");
    }
    check(runtime.journal().chronology().size() == 7, "recognition journal entries recorded once");
    RuntimeObject enemy;
    enemy.id = ObjectId::persistent(makeTes3ReferenceKey("Morrowind.esm", 43));
    enemy.base = makeTes3RecordKey("NPC_", "route_enemy");
    enemy.kind = RuntimeObjectKind::Actor; enemy.actorValues.emplace();
    check(session.world().addInitialObject(enemy, error), error);
    WorldCommand kill;
    kill.type = WorldCommandType::SetActorValue; kill.target = enemy.id;
    kill.actorValue = ActorValue::Health; kill.actorValueAbsolute = 0;
    (void)session.world().queue(kill);
    check(session.advance(1.0 / 60.0).diagnostics.empty(), "authored enemy death applies");
    (void)session.advance(1.0 / 60.0);
    check(runtime.playerState().deathCounts.at("route_enemy") == 1,
        "world death contributes once to dialogue death count");
    runtime.endDialogue();
    WorldCommand transition;
    transition.type = WorldCommandType::SetCurrentSpace;
    transition.target = session.playerObject();
    transition.currentSpace.kind = RuntimeSpaceKind::Interior;
    transition.currentSpace.cell = makeTes3RecordKey("CELL", "Test Cell");
    (void)session.world().queue(transition);
    check(session.advance(1.0 / 60.0).diagnostics.empty() && runtime.journal().chronology().size() == 7 &&
        runtime.knownTopics().contains(makeTes3RecordKey("DIAL", "Recognize 2")),
        "cell transition retains dialogue topics and independent journal progress");
    const fs::path save = root / "recognitions.odai";
    check(saveOdaiGameAtomic(save, session, error), error);
    BethesdaSession restored;
    check(restored.configure(config, error), error); check(restored.configureTes3Content(content, error), error);
    SaveLoadReport report;
    check(loadOdaiGame(save, restored, {}, report, error), error);
    check(restored.tes3().playerState().deathCounts.contains("route_enemy") &&
        restored.tes3().playerState().deathCounts.at("route_enemy") == 1,
        "world death count survives save/load");
    check(restored.tes3().journal().chronology() == runtime.journal().chronology(),
        "recognition chronology survives save/load");
    check(restored.tes3().startDialogue(actor, restored.tes3().playerState()).accepted,
        "saved conversation reopens");
    check(!restored.tes3().selectTopic("Local Reward").accepted &&
        !restored.tes3().selectTopic("Recognize 2").accepted,
        "saved authored guards prevent repeated rewards");
    check(restored.tes3().selectTopic("Talked Before").accepted,
        "per-reference conversation history survives save/load");
    fs::remove_all(root);
}

}  // namespace

int main() {
    testMainQuestDialogueFiltersAndRecognitions();
    testUnsupportedResultScriptDoesNotAdvanceQuest();
    testRuntimeFailureRollsBackDialogueTransition();
    testJournalAndDialogue();
    testAuthoredOpeningPositionAndSave();
    testSessionSaveReloadMidDialogue();
    testCorprusCureSpellStateAndSave();
    testGetStandingPcUsesPhysicalSupport();
    testRegionWeatherMutationAndSave();
    testUnloadedActorHealthPersists();
    testDialogueUsesCurrentPlayerInventory();
    testArtifactEquipScriptAndSave();
    testScriptedWorldPickup();
    testAiPackageSameTickAndSave();
    testScriptedDoorActivation();
    testFinaleShieldAndRestoreAttribute();
    testWeaponHitAndCrossReferenceLocal();
    testSayDoneWaitsAcrossTicks();
    testCharacterMenuRequestAndSave();
    testWorldMessageChoiceAndSave();
    testDontSaveObjectSkipsWorldSnapshot();
    testGetDetectedUsesResidentVisibility();
    if (failures == 0) std::cout << "tes3 runtime tests passed\n";
    return failures == 0 ? 0 : 1;
}
