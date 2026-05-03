local TAXMAN_QUEST = "MV_DeadTaxman"
local FARGOTH_RING_QUEST = "MV_FargothRing"
local FARGOTH_HIDING_QUEST = "MV_FargothHiding"
local VODUNIUS_QUEST = "MV_Vodunius"
local CAIUS_QUEST = "MV_ReportToCaius"
local CAIUS_ANTABOLIS_QUEST = "MV_CaiusAntabolis"
local CAIUS_VIVEC_QUEST = "MV_CaiusVivecInformants"
local FG_RATS_QUEST = "MV_FG_Rats"
local FG_EGGMINE_QUEST = "MV_FG_Eggmine"
local MG_MUSHROOMS_QUEST = "MV_MG_Mushrooms"
local MG_FAKE_SOULGEM_QUEST = "MV_MG_FakeSoulgem"
local TG_DIAMOND_QUEST = "MV_TG_Diamond"
local TG_KEY_QUEST = "MV_TG_NeranoKey"
local HH_RECORDS_QUEST = "MV_HH_Records"
local HH_DEBT_QUEST = "MV_HH_Debt"

local TAX_RECORD = "bk_seydaneentaxrecord"
local PROCESSUS = "processus vitellius"
local SOCUCIUS = "socucius ergalla"
local FORYN = "foryn gilnith"
local FARGOTH = "fargoth"
local HRISSKAR = "hrisskar flat-foot"
local SELLUS = "sellus gravius"
local VODUNIUS = "vodunius nuccius"
local PROCESSUS_RING = "processus_ring"
local FARGOTH_RING = "engraved_ring_of_healing"
local PACKAGE = "package_for_caius"
local FARGOTH_STASH = "fargoth_stash"
local FARGOTH_STASH_REF = "fargoth_stash"
local FARGOTH_STASH_X = -13352.96
local FARGOTH_STASH_Y = 0.0
local FARGOTH_STASH_Z = -68649.04

local CAIUS = "caius cosades"
local EYDIA = "eydis fire-eye"
local AJIRA = "ajira"
local SOTtilde = "sottilde"
local NILENO = "nileno dorvayn"

local function result(message)
    return { handled = true, message = message }
end

local function ignored()
    return { handled = false }
end

local function topic(id, text)
    return { id = id, text = text }
end

local function choice(id, text)
    return { id = id, text = text }
end

local function lower(value)
    return string.lower(value or "")
end

local function route_point(x, y, z)
    return { x = x, y = y, z = z }
end

local function common_topics()
    return {
        topic("region quests", "Region quests"),
        topic("latest rumors", "Latest rumors"),
        topic("background", "Background"),
        topic("statistics", "Statistics"),
        topic("services", "Services"),
    }
end

local function append_topic(topics, id, text)
    topics[#topics + 1] = topic(id, text)
end

local function dialogue(text, topics, choices)
    return {
        handled = true,
        text = text,
        topics = topics or {},
        choices = choices or {},
    }
end

local ITEM_NAMES = {
    [TAX_RECORD] = "Seyda Neen Tax Record",
    [PROCESSUS_RING] = "Processus' Ring",
    [FARGOTH_RING] = "Engraved Ring of Healing",
    [PACKAGE] = "Package for Caius Cosades",
    [FARGOTH_STASH] = "Fargoth's Stash",
    ["combat_trophy"] = "Proof of Victory",
    ["code_book"] = "Code Book",
    ["bittergreen_petals"] = "Bittergreen Petals",
    ["luminous_russula"] = "Luminous Russula",
    ["violet_coprinus"] = "Violet Coprinus",
    ["fake_soulgem"] = "Fake Soulgem",
    ["diamond"] = "Diamond",
    ["nerano_key"] = "Nerano Manor Key",
    ["hlaalu_records"] = "Hlaalu Records",
    ["debt_note"] = "Debt Note",
}

local QUEST_DEFS = {
    { FARGOTH_RING_QUEST, "Fargoth's Ring", "Find Fargoth's engraved ring in Seyda Neen." },
    { FARGOTH_HIDING_QUEST, "Fargoth's Hiding Place", "Help Hrisskar find Fargoth's hidden stash." },
    { TAXMAN_QUEST, "Death of a Taxman", "Find out what happened to Processus Vitellius." },
    { VODUNIUS_QUEST, "Vodunius Nuccius", "Help Vodunius get out of Seyda Neen." },
    { CAIUS_QUEST, "Report to Caius Cosades", "Take Sellus Gravius' package to Caius in Balmora." },
    { CAIUS_ANTABOLIS_QUEST, "Caius: Antabolis Informant", "Run Caius' first local intelligence errand." },
    { CAIUS_VIVEC_QUEST, "Caius: Informants", "Gather a simplified local report for Caius." },
    { FG_RATS_QUEST, "Fighters Guild: Cave Rats", "Clear rats from a Balmora store room." },
    { FG_EGGMINE_QUEST, "Fighters Guild: Egg Mine Trouble", "Deal with a hostile egg mine scout near Balmora." },
    { MG_MUSHROOMS_QUEST, "Mages Guild: Ajira's Mushrooms", "Collect mushroom samples for Ajira." },
    { MG_FAKE_SOULGEM_QUEST, "Mages Guild: Fake Soulgem", "Place Ajira's fake soulgem." },
    { TG_DIAMOND_QUEST, "Thieves Guild: Diamonds", "Acquire a diamond for Sottilde." },
    { TG_KEY_QUEST, "Thieves Guild: Nerano Manor Key", "Acquire a local manor key for the guild." },
    { HH_RECORDS_QUEST, "House Hlaalu: Records", "Recover useful records for Nileno Dorvayn." },
    { HH_DEBT_QUEST, "House Hlaalu: Debt Money", "Collect or cover a local debt for House Hlaalu." },
}

for id, name in pairs(ITEM_NAMES) do
    game.set_item_name(id, name)
end
for _, quest in ipairs(QUEST_DEFS) do
    game.define_quest(quest[1], quest[2], quest[3])
end
game.set_player_max_stats(100, 60, 100)
game.recover_player()

local SEYDA_NEEN_PEOPLE = {
    ["adraria vandacia"] = { race = "Imperial", gender = "Female", class = "Agent", faction = "Census and Excise", rank = "Taxman", level = 11, health = 98, magicka = 102, location = "Census and Excise Warehouse", trainer = true },
    ["albecius colollius"] = { race = "Imperial", gender = "Male", class = "Battlemage", level = 6, health = 68, magicka = 122, location = "Arrille's Tradehouse" },
    ["aronil"] = { race = "High Elf", gender = "Male", class = "Battlemage", level = 12, health = 180, magicka = 500, location = "Outside [-1,-10]" },
    ["arrille"] = { race = "High Elf", gender = "Male", class = "Trader Service", level = 15, health = 141, magicka = 42, location = "Arrille's Tradehouse", merchant = true, spell_merchant = true },
    ["darvame hleran"] = { race = "Dark Elf", gender = "Female", class = "Caravaner", level = 0, health = 32, magicka = 96, location = "Outside [-2,-9]", travel = true },
    ["draren thiralas"] = { race = "Dark Elf", gender = "Male", class = "Commoner", level = 4, health = 63, magicka = 86, location = "Draren Thiralas' House" },
    ["eldafire"] = { race = "High Elf", gender = "Female", class = "Commoner", level = 5, health = 58, magicka = 108, location = "Outside [-2,-9]" },
    ["elone"] = { race = "Redguard", gender = "Female", class = "Scout", faction = "Blades", rank = "Journeyman", level = 9, health = 110, magicka = 78, location = "Arrille's Tradehouse", trainer = true },
    ["erene llenim"] = { race = "Dark Elf", gender = "Male", class = "Commoner", level = 3, health = 57, magicka = 84, location = "Outside [-2,-9]" },
    [FARGOTH] = { race = "Wood Elf", gender = "Male", class = "Commoner", level = 2, health = 41, magicka = 82, location = "Outside [-2,-9]" },
    ["fine-mouth"] = { race = "Argonian", gender = "Male", class = "Commoner", level = 2, health = 46, magicka = 82, location = "Fine-Mouth's Shack" },
    [FORYN] = { race = "Dark Elf", gender = "Male", class = "Commoner", level = 5, health = 68, magicka = 88, location = "Foryn Gilnith's Shack", hostile_health = 55 },
    ["ganciele douar"] = { race = "Imperial", gender = "Male", class = "Guard", faction = "Imperial Legion", rank = "Spearman", level = 15, health = 170, magicka = 110, location = "Census and Excise Office" },
    ["hjrondir"] = { race = "Nord", gender = "Male", class = "Warrior", level = 10, health = 220, magicka = 74, location = "Outside [-1,-10]" },
    [HRISSKAR] = { race = "Nord", gender = "Male", class = "Rogue", faction = "Imperial Legion", rank = "Trooper", level = 8, health = 81, magicka = 18, location = "Arrille's Tradehouse" },
    ["indrele rathryon"] = { race = "Dark Elf", gender = "Female", class = "Commoner", level = 1, health = 40, magicka = 80, location = "Outside [-2,-9]" },
    ["mara"] = { race = "Wood Elf", gender = "Female", class = "Archer", level = 10, health = 180, magicka = 72, location = "Outside [-1,-10]" },
    ["raflod the braggart"] = { race = "Nord", gender = "Male", class = "Scout", faction = "Thieves Guild", rank = "Wet Ear", level = 7, health = 100, magicka = 74, location = "Arrille's Tradehouse", trainer = true },
    [SELLUS] = { race = "Imperial", gender = "Male", class = "Guard", faction = "Imperial Legion", rank = "Knight Errant", level = 17, health = 186, magicka = 116, location = "Census and Excise Office" },
    ["silm-dar"] = { race = "Argonian", gender = "Male", class = "Commoner", level = 6, health = 150, magicka = 68, location = "Outside [-1,-10]" },
    [SOCUCIUS] = { race = "Breton", gender = "Male", class = "Agent", faction = "Census and Excise", rank = "Agent", level = 14, health = 102, magicka = 128, location = "Census and Excise Office", trainer = true },
    ["tandram andalen"] = { race = "Dark Elf", gender = "Male", class = "Bard", level = 3, health = 51, magicka = 108, location = "Arrille's Tradehouse" },
    ["teleri helvi"] = { race = "Dark Elf", gender = "Female", class = "Commoner", level = 4, health = 58, magicka = 86, location = "Outside [-2,-9]" },
    ["teruise girvayne"] = { race = "Dark Elf", gender = "Female", class = "Commoner", level = 3, health = 52, magicka = 84, location = "Terurise Girvayne's House" },
    ["thavere vedrano"] = { race = "Dark Elf", gender = "Female", class = "Commoner", level = 5, health = 63, magicka = 88, location = "Lighthouse" },
    ["tolvise othralen"] = { race = "Dark Elf", gender = "Female", class = "Commoner", level = 3, health = 52, magicka = 84, location = "Arrille's Tradehouse" },
    [VODUNIUS] = { race = "Imperial", gender = "Male", class = "Commoner", level = 4, health = 63, magicka = 86, location = "Outside [-2,-9]" },
}

local BALMORA_PEOPLE = {
    [CAIUS] = { race = "Imperial", gender = "Male", class = "Blades Spymaster", faction = "Blades", rank = "Spymaster", level = 20, health = 190, magicka = 120, location = "Caius Cosades' House", trainer = true },
    [EYDIA] = { race = "Nord", gender = "Female", class = "Warrior", faction = "Fighters Guild", rank = "Steward", level = 14, health = 160, magicka = 25, location = "Balmora Fighters Guild", trainer = true },
    [AJIRA] = { race = "Khajiit", gender = "Female", class = "Mage", faction = "Mages Guild", rank = "Associate", level = 8, health = 75, magicka = 180, location = "Balmora Mages Guild", merchant = true, spell_merchant = true },
    [SOTtilde] = { race = "Nord", gender = "Female", class = "Thief", faction = "Thieves Guild", rank = "Operative", level = 10, health = 95, magicka = 30, location = "South Wall Cornerclub", trainer = true },
    [NILENO] = { race = "Dark Elf", gender = "Female", class = "Agent", faction = "House Hlaalu", rank = "Retainer", level = 13, health = 105, magicka = 90, location = "Balmora Council Club", trainer = true },
    ["ranis athrys"] = { race = "Dark Elf", gender = "Female", class = "Mage", faction = "Mages Guild", rank = "Guild Guide", level = 12, health = 95, magicka = 220, location = "Balmora Mages Guild", trainer = true, spell_merchant = true },
    ["habasi"] = { race = "Khajiit", gender = "Female", class = "Thief", faction = "Thieves Guild", rank = "Mastermind", level = 15, health = 120, magicka = 55, location = "South Wall Cornerclub", trainer = true },
    ["fire-eye rat"] = { race = "Creature", gender = "None", class = "Hostile", level = 2, health = 25, magicka = 0, location = "Balmora Store Room", hostile_health = 25 },
    ["egg mine scout"] = { race = "Dark Elf", gender = "Male", class = "Bandit", level = 5, health = 70, magicka = 20, location = "Balmora Outskirts", hostile_health = 70 },
}

local function actor_info(actor)
    actor = lower(actor)
    return SEYDA_NEEN_PEOPLE[actor] or BALMORA_PEOPLE[actor]
end

local function append_actor_service_choices(actor, choices)
    local info = actor_info(actor)
    choices = choices or {}
    if info == nil then return choices end
    if info.trainer then choices[#choices + 1] = choice("train:" .. lower(actor), "Ask about training.") end
    if info.merchant then choices[#choices + 1] = choice("barter:" .. lower(actor), "Ask to barter.") end
    if info.spell_merchant then choices[#choices + 1] = choice("spells:" .. lower(actor), "Ask about spells.") end
    if info.travel then choices[#choices + 1] = choice("travel:" .. lower(actor), "Ask about travel.") end
    return choices
end

local function actor_topics(actor)
    local topics = common_topics()
    local info = actor_info(actor)
    if info ~= nil and (info.trainer or info.merchant or info.spell_merchant or info.travel) then
        append_topic(topics, "training", "Training")
    end
    return topics
end

local function actor_summary(actor)
    local info = actor_info(actor)
    if info == nil then return nil end
    local faction = ""
    if info.faction ~= nil then
        faction = " " .. info.faction
        if info.rank ~= nil then faction = faction .. " (" .. info.rank .. ")" end
        faction = faction .. "."
    end
    return actor .. " is a level " .. tostring(info.level) .. " " .. info.gender .. " " ..
        info.race .. " " .. info.class .. " found at " .. info.location .. "." .. faction
end

local function actor_statistics_text(actor)
    local info = actor_info(actor)
    if info == nil then return nil end
    return "Level " .. tostring(info.level) .. ". Health " .. tostring(info.health) .. ". Magicka " .. tostring(info.magicka) .. "."
end

local function actor_services_text(actor)
    local info = actor_info(actor)
    if info == nil then return nil end
    local services = {}
    if info.trainer then services[#services + 1] = "training" end
    if info.merchant then services[#services + 1] = "barter" end
    if info.spell_merchant then services[#services + 1] = "spell merchant" end
    if info.travel then services[#services + 1] = "silt strider travel" end
    if #services == 0 then return "I do not offer services." end
    return "Available services: " .. table.concat(services, ", ") .. "."
end

local function start_quest(id, stage, objective)
    if game.journal(id) < stage then
        game.set_journal(id, stage)
    end
    game.set_quest_objective(id, objective)
end

local function finish_quest(id, stage, objective)
    if game.journal(id) < stage then
        game.set_journal(id, stage)
    end
    game.set_quest_objective(id, objective)
    game.complete_quest(id)
end

local function grant_item(id, count)
    if game.item_count(id) == 0 then
        game.add_item(id, count or 1)
    end
end

local function combat_choice(target, label)
    return choice("combat:" .. target, label or ("Attack " .. target .. "."))
end

local function resolve_combat(target, reward_item, reward_gold, quest_id, complete_stage, complete_text)
    target = lower(target)
    local info = actor_info(target)
    if info == nil then return ignored() end
    if game.is_ref_dead(target) then return result(target .. " is already down.") end
    if game.ref_health(target) <= 0 then
        game.set_ref_health(target, info.hostile_health or info.health or 50)
    end
    if not game.spend_player_fatigue(12) then
        game.damage_player(8)
        return result("You are too exhausted to press the attack. You take 8 damage while backing away.")
    end
    game.damage_ref(target, 34)
    if not game.is_ref_dead(target) then
        game.damage_player(10)
        return result("You hit " .. target .. ". They are hurt but still fighting.")
    end
    if reward_item ~= nil then grant_item(reward_item, 1) end
    if reward_gold ~= nil and reward_gold > 0 then game.add_gold(reward_gold) end
    if quest_id ~= nil then finish_quest(quest_id, complete_stage or 100, complete_text or "Completed.") end
    return result(target .. " is defeated.")
end

local SEYDA_NEEN_SCHEDULES = {
    [FARGOTH] = { day = "village_square", night = "fargoth_home", speed = 68.0 },
    ["eldafire"] = { day = "eldafire_home", night = "eldafire_home", speed = 62.0 },
    ["erene llenim"] = { day = "erene_home", night = "erene_home", speed = 60.0 },
    ["indrele rathryon"] = { day = "indrele_shack", night = "indrele_shack", speed = 58.0 },
    ["teleri helvi"] = { day = "teleri_home", night = "teleri_home", speed = 60.0 },
    [VODUNIUS] = { day = "tradehouse", night = "vodunius_home", speed = 66.0 },
    ["darvame hleran"] = { day = "darvame_silt_strider", night = "darvame_silt_strider", speed = 56.0 },
    ["aronil"] = { day = "road_east", night = "road_east", speed = 72.0 },
    ["hjrondir"] = { day = "road_east", night = "road_east", speed = 74.0 },
    ["mara"] = { day = "road_east", night = "road_east", speed = 74.0 },
    ["silm-dar"] = { day = "road_east", night = "road_east", speed = 64.0 },
}

local function is_seyda_neen_guard(actor)
    return string.sub(actor, 1, 17) == "seyda_neen_guard_"
        or actor == "imperial guard"
        or actor == "chargen dock guard"
        or actor == "chargen boat guard 1"
end

local function generic_schedule(actor, game_hour)
    local hour = game_hour or 14.0
    local is_night = hour >= 21.0 or hour < 6.0
    if is_seyda_neen_guard(actor) then
        return {
            handled = true,
            state = "wander",
            anchor = is_night and "census_office" or "dock",
            speed = is_night and 86.0 or 92.0,
            wait_seconds = is_night and 3.0 or 2.0,
            wander_radius = is_night and 260.0 or 320.0,
            priority = 10,
        }
    end
    local schedule = SEYDA_NEEN_SCHEDULES[actor]
    if schedule == nil then return ignored() end
    if is_night then
        return { handled = true, state = "travel", anchor = schedule.night, speed = schedule.speed, wait_seconds = 8.0, priority = 1 }
    end
    return { handled = true, state = "wander", anchor = schedule.day, speed = schedule.speed, wait_seconds = 4.0, wander_radius = 180.0, priority = 1 }
end

local function fargoth_topics()
    local topics = actor_topics(FARGOTH)
    append_topic(topics, "missing ring", "Missing ring")
    if game.journal(FARGOTH_HIDING_QUEST) >= 10 then append_topic(topics, "hiding place", "Hiding place") end
    return topics
end

local function hrisskar_topics()
    local topics = actor_topics(HRISSKAR)
    if game.journal(FARGOTH_RING_QUEST) >= 30 then append_topic(topics, "fargoth's hiding place", "Fargoth's hiding place") end
    return topics
end

local function taxman_topics()
    local topics = actor_topics(SOCUCIUS)
    append_topic(topics, "murder of processus vitellius", "Murder of Processus Vitellius")
    return topics
end

local function region_quest_text()
    return "Playable regional quests are active for Seyda Neen and Balmora. Off-region errands are folded into local objectives so you can finish a loop now."
end

function on_activate(ref_id)
    local id = lower(ref_id)
    if game.is_player_dead() then
        game.recover_player()
        return result("You recover at the nearest safe place.")
    end
    if id == PROCESSUS then
        if game.journal(TAXMAN_QUEST) < 10 then
            game.set_ref_dead(PROCESSUS, true)
            game.add_item(TAX_RECORD, 1)
            game.add_gold(200)
            start_quest(TAXMAN_QUEST, 10, "Return the recovered tax money and records to Socucius Ergalla.")
            game.log("Found Processus Vitellius and recovered the tax record.")
            return result("You found Processus Vitellius' corpse. Tax record and 200 gold recovered.")
        end
        return result("Processus Vitellius' corpse has already been searched.")
    end
    if id == "fargoth ring barrel" then
        if game.item_count(FARGOTH_RING) == 0 and game.journal(FARGOTH_RING_QUEST) == 0 then
            game.add_item(FARGOTH_RING, 1)
            start_quest(FARGOTH_RING_QUEST, 10, "Return the engraved ring to Fargoth.")
            return result("You found an engraved ring of healing in the barrel.")
        end
        return result("The barrel has already been searched.")
    end
    if id == "sellus package" then
        if game.item_count(PACKAGE) == 0 and game.journal(CAIUS_QUEST) == 0 then
            game.add_item(PACKAGE, 1)
            start_quest(CAIUS_QUEST, 10, "Take the package to Caius Cosades in Balmora.")
            return result("You received a package for Caius Cosades.")
        end
        return result("You already have your orders.")
    end
    if id == FARGOTH_STASH_REF then
        if game.journal(FARGOTH_HIDING_QUEST) >= 15 and game.journal(FARGOTH_HIDING_QUEST) < 20 then
            if game.item_count(FARGOTH_STASH) == 0 then
                game.add_item(FARGOTH_STASH, 1)
                game.set_quest_objective(FARGOTH_HIDING_QUEST, "Bring Fargoth's stash to Hrisskar.")
                return result("You recover Fargoth's hidden stash.")
            end
            return result("You already recovered Fargoth's hidden stash.")
        end
        return ignored()
    end
    return ignored()
end

local function balmora_dialogue(actor, topic_id)
    local topics = actor_topics(actor)
    append_topic(topics, "balmora work", "Balmora work")
    if actor == CAIUS then
        append_topic(topics, "orders", "Orders")
        if topic_id == "orders" or topic_id == "" then
            if game.has_item(PACKAGE) and game.journal(CAIUS_QUEST) < 100 then
                return dialogue("You found Caius. Hand over the package and he will put you to work in Balmora.", topics, { choice("caius_deliver_package", "Deliver the package.") })
            end
            return dialogue("Caius has more local work ready: Antabolis, informants, and a few errands that keep the Blades moving.", topics, {
                choice("caius_start_antabolis", "Take the Antabolis errand."),
                choice("caius_start_informants", "Gather local informant notes."),
            })
        end
    end
    if actor == EYDIA then
        if topic_id == "balmora work" or topic_id == "" then
            return dialogue("The Fighters Guild needs simple work done now: rats in a store room and trouble near the egg mine.", topics, {
                choice("fg_start_rats", "Take the cave rats job."),
                choice("fg_start_eggmine", "Take the egg mine job."),
            })
        end
    end
    if actor == AJIRA then
        if topic_id == "balmora work" or topic_id == "" then
            return dialogue("Ajira needs mushroom samples and a fake soulgem placed before Galbedir notices.", topics, {
                choice("mg_start_mushrooms", "Collect mushroom samples."),
                choice("mg_start_soulgem", "Take the fake soulgem."),
            })
        end
    end
    if actor == SOTtilde or actor == "habasi" then
        if topic_id == "balmora work" or topic_id == "" then
            return dialogue("The guild wants quiet hands: a diamond and a manor key. This version keeps both jobs local to Balmora.", topics, {
                choice("tg_start_diamond", "Find a diamond."),
                choice("tg_start_key", "Acquire Nerano's key."),
            })
        end
    end
    if actor == NILENO then
        if topic_id == "balmora work" or topic_id == "" then
            return dialogue("House Hlaalu rewards useful paperwork and collected debts.", topics, {
                choice("hh_start_records", "Recover Hlaalu records."),
                choice("hh_start_debt", "Collect the local debt."),
            })
        end
    end
    if actor == "fire-eye rat" then
        return dialogue("The rat hisses from the corner.", topics, { combat_choice("fire-eye rat", "Attack the rat.") })
    end
    if actor == "egg mine scout" then
        return dialogue("The scout grips a chipped blade and blocks the path.", topics, { combat_choice("egg mine scout", "Fight the scout.") })
    end
    return nil
end

function get_dialogue(actor_id, topic_id)
    local actor = lower(actor_id)
    topic_id = lower(topic_id)
    if topic_id == "region quests" then
        return dialogue(region_quest_text(), common_topics(), {})
    end
    if topic_id == "latest rumors" then
        return dialogue("People talk about Fargoth's ring, the dead taxman, guild work in Balmora, and Caius Cosades waiting for a package.", common_topics(), {})
    end
    if topic_id == "background" then
        local summary = actor_summary(actor)
        if summary ~= nil then return dialogue(summary, actor_topics(actor), append_actor_service_choices(actor, {})) end
    end
    if topic_id == "statistics" then
        local stats = actor_statistics_text(actor)
        if stats ~= nil then return dialogue(stats, actor_topics(actor), append_actor_service_choices(actor, {})) end
    end
    if topic_id == "services" or topic_id == "training" then
        local services = actor_services_text(actor)
        if services ~= nil then return dialogue(services, actor_topics(actor), append_actor_service_choices(actor, {})) end
    end

    local balmora = balmora_dialogue(actor, topic_id)
    if balmora ~= nil then return balmora end

    if actor == FARGOTH then
        if topic_id == "missing ring" then
            if game.journal(FARGOTH_RING_QUEST) >= 30 then return dialogue("You found my ring. I will not forget that kindness.", fargoth_topics(), {}) end
            if game.has_item(FARGOTH_RING) then
                return dialogue("That is it! My engraved ring. Please, sera, may I have it back?", fargoth_topics(), { choice("fargoth_return_ring", "Return the engraved ring.") })
            end
            return dialogue("I lost an engraved ring of healing. The guards have been laughing about it.", fargoth_topics(), {})
        end
        if topic_id == "hiding place" then
            return dialogue("Why would you ask me that? I keep to myself, and still Hrisskar will not leave me be.", fargoth_topics(), {})
        end
        return dialogue("Are you the one that boat dropped off? Odd time for a ship to arrive. If you find my ring, please bring it to me.", fargoth_topics(), append_actor_service_choices(actor, {}))
    end

    if actor == HRISSKAR then
        if topic_id == "fargoth's hiding place" then
            if game.journal(FARGOTH_RING_QUEST) < 30 then return dialogue("Talk to the little Bosmer first. He is jumpier than usual.", hrisskar_topics(), {}) end
            if game.journal(FARGOTH_HIDING_QUEST) >= 20 then return dialogue("You found the stash. Fargoth will be missing more than his ring now.", hrisskar_topics(), {}) end
            if game.has_item(FARGOTH_STASH) then return dialogue("So you saw him hide it? Hand it over, and I will make it worth your time.", hrisskar_topics(), { choice("hrisskar_turn_in_stash", "Give Hrisskar Fargoth's stash.") }) end
            if game.journal(FARGOTH_HIDING_QUEST) >= 15 then return dialogue("You saw where he hid it. Get the stash and bring it to me.", hrisskar_topics(), {}) end
            return dialogue("Fargoth has a hiding place. Watch him from the lighthouse after dark, then bring me what he stashed.", hrisskar_topics(), { choice("hrisskar_accept_hiding", "Agree to watch Fargoth.") })
        end
        return dialogue("Hrisskar Flat-Foot. I keep order upstairs at Arrille's when order needs keeping.", hrisskar_topics(), append_actor_service_choices(actor, {}))
    end

    if actor == SOCUCIUS and (topic_id == "" or topic_id == "murder of processus vitellius") then
        if game.journal(TAXMAN_QUEST) == 10 and game.has_item(TAX_RECORD) then
            return dialogue("Processus is dead? Did you find the tax money he collected?", taxman_topics(), { choice("taxman_truth", "Tell him about the 200 gold."), choice("taxman_lie", "Say there was no money.") })
        elseif game.journal(TAXMAN_QUEST) >= 30 and game.journal(TAXMAN_QUEST) < 70 then
            return dialogue("Find whoever murdered Processus Vitellius and report back to me.", taxman_topics(), {})
        elseif game.journal(TAXMAN_QUEST) >= 70 then
            return dialogue("You have done the Census and Excise Office a service.", taxman_topics(), {})
        end
        return dialogue("Official business belongs at the Census and Excise Office. If you hear anything about Processus, report it.", taxman_topics(), {})
    end

    if actor == FORYN and game.journal(TAXMAN_QUEST) >= 30 and game.journal(TAXMAN_QUEST) < 70 then
        return dialogue("Processus pushed people too far. I did what had to be done.", taxman_topics(), { combat_choice(FORYN, "Attack Foryn."), choice("taxman_spare_foryn", "Leave him for now.") })
    end

    if actor == SELLUS then
        local topics = common_topics()
        append_topic(topics, "orders", "Orders")
        if topic_id == "orders" then
            if game.journal(CAIUS_QUEST) >= 30 then return dialogue("Your orders are clear. Find Caius Cosades in Balmora.", topics, {}) end
            return dialogue("Take the package to Caius Cosades in Balmora. The silt strider is the sensible road from here.", topics, { choice("sellus_take_package", "Take the package for Caius.") })
        end
        return dialogue("You are released by order of the Emperor. Keep your papers and follow your instructions.", topics, {})
    end

    if actor == VODUNIUS then
        local topics = common_topics()
        append_topic(topics, "leaving seyda neen", "Leaving Seyda Neen")
        if topic_id == "leaving seyda neen" then
            if game.journal(VODUNIUS_QUEST) >= 30 then return dialogue("Thanks to you, I can finally get out of this swamp.", topics, {}) end
            return dialogue("I have had enough of Seyda Neen. A little money would get me closer to the road home.", topics, { choice("vodunius_give_gold", "Give Vodunius 100 gold.") })
        end
        return dialogue("Vodunius Nuccius. Down on my luck, and not much fond of this place.", topics, append_actor_service_choices(actor, {}))
    end

    local summary = actor_summary(actor)
    if summary ~= nil then return dialogue(summary, actor_topics(actor), append_actor_service_choices(actor, {})) end
    return ignored()
end

function choose_dialogue(response_id)
    local response = lower(response_id)
    if string.sub(response, 1, 7) == "combat:" then
        local target = string.sub(response, 8)
        if target == FORYN then return resolve_combat(FORYN, PROCESSUS_RING, 500, TAXMAN_QUEST, 70, "Foryn is dead. Report back to Socucius.") end
        if target == "fire-eye rat" then return resolve_combat(target, "combat_trophy", 35, FG_RATS_QUEST, 100, "The store room rats are cleared.") end
        if target == "egg mine scout" then return resolve_combat(target, "code_book", 75, FG_EGGMINE_QUEST, 100, "The egg mine scout is defeated.") end
        return ignored()
    end
    if string.sub(response, 1, 6) == "train:" then
        local actor = string.sub(response, 7)
        local info = actor_info(actor)
        if info ~= nil and info.trainer then return result(actor .. " offers practical training appropriate to " .. info.class .. ".") end
        return ignored()
    end
    if string.sub(response, 1, 7) == "barter:" then
        local actor = string.sub(response, 8)
        local info = actor_info(actor)
        if info ~= nil and info.merchant then return result(actor .. " is willing to barter.") end
        return ignored()
    end
    if string.sub(response, 1, 7) == "spells:" then
        local actor = string.sub(response, 8)
        local info = actor_info(actor)
        if info ~= nil and info.spell_merchant then return result(actor .. " offers spells for sale.") end
        return ignored()
    end
    if string.sub(response, 1, 7) == "travel:" then
        local actor = string.sub(response, 8)
        local info = actor_info(actor)
        if info ~= nil and info.travel then return result(actor .. " offers silt strider travel to Balmora, Gnisis, Suran, and Vivec.") end
        return ignored()
    end

    if response == "fargoth_return_ring" and game.journal(FARGOTH_RING_QUEST) < 30 then
        if game.remove_item(FARGOTH_RING, 1) then
            finish_quest(FARGOTH_RING_QUEST, 30, "Returned Fargoth's ring.")
            game.log("Returned Fargoth's engraved ring.")
            return result("Fargoth beams with relief. Arrille may look more kindly on you now.")
        end
        return ignored()
    end
    if response == "hrisskar_accept_hiding" and game.journal(FARGOTH_RING_QUEST) >= 30 and game.journal(FARGOTH_HIDING_QUEST) < 10 then
        start_quest(FARGOTH_HIDING_QUEST, 10, "Watch Fargoth from the lighthouse after dark.")
        return result("Hrisskar tells you to watch Fargoth from the lighthouse after dark.")
    end
    if response == "hrisskar_turn_in_stash" and game.journal(FARGOTH_HIDING_QUEST) >= 15 and game.journal(FARGOTH_HIDING_QUEST) < 20 then
        if game.remove_item(FARGOTH_STASH, 1) then
            finish_quest(FARGOTH_HIDING_QUEST, 20, "Delivered the stash to Hrisskar.")
            game.add_gold(100)
            return result("Hrisskar takes the stash and pays you 100 gold.")
        end
        return ignored()
    end
    if response == "sellus_take_package" and game.journal(CAIUS_QUEST) < 30 then
        if game.item_count(PACKAGE) == 0 then game.add_item(PACKAGE, 1) end
        start_quest(CAIUS_QUEST, 30, "Find Caius Cosades in Balmora.")
        return result("Sellus gives you the package for Caius Cosades in Balmora.")
    end
    if response == "vodunius_give_gold" and game.journal(VODUNIUS_QUEST) < 30 then
        if game.spend_gold(100) then
            finish_quest(VODUNIUS_QUEST, 30, "Helped Vodunius leave Seyda Neen.")
            return result("Vodunius accepts the gold and starts planning his way out of Seyda Neen.")
        end
        return ignored()
    end
    if response == "taxman_truth" and game.journal(TAXMAN_QUEST) == 10 then
        game.spend_gold(200)
        start_quest(TAXMAN_QUEST, 30, "Confront Foryn Gilnith about Processus' murder.")
        return result("You returned the tax money. Socucius asks you to find the killer.")
    end
    if response == "taxman_lie" and game.journal(TAXMAN_QUEST) == 10 then
        start_quest(TAXMAN_QUEST, 40, "You kept the tax money quiet. Find the killer anyway.")
        return result("You kept the tax money quiet. Socucius still wants the killer found.")
    end
    if response == "taxman_kill_foryn" then
        if game.journal(TAXMAN_QUEST) >= 30 and game.journal(TAXMAN_QUEST) < 70 then
            game.set_ref_dead(FORYN, true)
            game.add_item(PROCESSUS_RING, 1)
            finish_quest(TAXMAN_QUEST, 70, "Foryn is dead. Report back to Socucius.")
            game.add_gold(500)
            return result("Foryn Gilnith is dead. Socucius' reward has been paid.")
        end
        return ignored()
    end
    if response == "taxman_spare_foryn" then return result("You step back from the accusation for now.") end

    if response == "caius_deliver_package" then
        if game.remove_item(PACKAGE, 1) then
            finish_quest(CAIUS_QUEST, 100, "Delivered the package to Caius Cosades.")
            game.add_gold(200)
            game.restore_player(25, 20, 25)
            return result("Caius takes the package, pays 200 gold, and tells you to build useful local contacts.")
        end
        return ignored()
    end
    if response == "caius_start_antabolis" then
        if game.journal(CAIUS_ANTABOLIS_QUEST) < 100 then
            grant_item("hlaalu_records", 1)
            finish_quest(CAIUS_ANTABOLIS_QUEST, 100, "Collected a local Antabolis-style intelligence note for Caius.")
            game.add_gold(150)
            return result("You gather a local intelligence note for Caius. He pays 150 gold.")
        end
        return ignored()
    end
    if response == "caius_start_informants" then
        if game.journal(CAIUS_VIVEC_QUEST) < 100 then
            finish_quest(CAIUS_VIVEC_QUEST, 100, "Compiled local informant notes for Caius.")
            game.add_gold(175)
            return result("You collect enough local rumors to stand in for the longer informant run. Caius pays 175 gold.")
        end
        return ignored()
    end
    if response == "fg_start_rats" then
        start_quest(FG_RATS_QUEST, 10, "Find and defeat the store room rat.")
        game.set_ref_health("fire-eye rat", 25)
        return result("Eydis marks the rat job on your journal. Find the rat and use the combat dialogue to defeat it.")
    end
    if response == "fg_start_eggmine" then
        start_quest(FG_EGGMINE_QUEST, 10, "Defeat the hostile egg mine scout.")
        game.set_ref_health("egg mine scout", 70)
        return result("Eydis sends you after a hostile scout near Balmora.")
    end
    if response == "mg_start_mushrooms" then
        if game.journal(MG_MUSHROOMS_QUEST) < 100 then
            grant_item("bittergreen_petals", 1)
            grant_item("luminous_russula", 1)
            grant_item("violet_coprinus", 1)
            finish_quest(MG_MUSHROOMS_QUEST, 100, "Collected local mushroom samples for Ajira.")
            game.add_gold(75)
            return result("You collect useful mushroom samples near the guild. Ajira pays 75 gold.")
        end
        return ignored()
    end
    if response == "mg_start_soulgem" then
        if game.journal(MG_FAKE_SOULGEM_QUEST) < 100 then
            grant_item("fake_soulgem", 1)
            finish_quest(MG_FAKE_SOULGEM_QUEST, 100, "Placed Ajira's fake soulgem.")
            game.add_gold(100)
            return result("You place Ajira's fake soulgem and return before anyone notices.")
        end
        return ignored()
    end
    if response == "tg_start_diamond" then
        if game.journal(TG_DIAMOND_QUEST) < 100 then
            grant_item("diamond", 1)
            finish_quest(TG_DIAMOND_QUEST, 100, "Acquired a diamond for the Thieves Guild.")
            game.add_gold(100)
            return result("You acquire a diamond locally and Sottilde pays 100 gold.")
        end
        return ignored()
    end
    if response == "tg_start_key" then
        if game.journal(TG_KEY_QUEST) < 100 then
            grant_item("nerano_key", 1)
            finish_quest(TG_KEY_QUEST, 100, "Acquired the Nerano manor key.")
            game.add_gold(125)
            return result("You quietly acquire Nerano's key. The guild pays 125 gold.")
        end
        return ignored()
    end
    if response == "hh_start_records" then
        if game.journal(HH_RECORDS_QUEST) < 100 then
            grant_item("hlaalu_records", 1)
            finish_quest(HH_RECORDS_QUEST, 100, "Recovered useful records for House Hlaalu.")
            game.add_gold(125)
            return result("You recover the records Nileno wanted. House Hlaalu pays 125 gold.")
        end
        return ignored()
    end
    if response == "hh_start_debt" then
        if game.journal(HH_DEBT_QUEST) < 100 then
            grant_item("debt_note", 1)
            finish_quest(HH_DEBT_QUEST, 100, "Settled a local debt for House Hlaalu.")
            game.add_gold(100)
            return result("You settle the debt locally and receive a 100 gold cut.")
        end
        return ignored()
    end
    return ignored()
end

function on_actor_death(actor_id)
    local actor = lower(actor_id)
    if actor == FORYN and game.journal(TAXMAN_QUEST) >= 30 and game.journal(TAXMAN_QUEST) < 70 then
        game.add_item(PROCESSUS_RING, 1)
        finish_quest(TAXMAN_QUEST, 70, "Foryn is dead. Report back to Socucius.")
        game.add_gold(500)
        return result("Foryn Gilnith is dead.")
    end
    return ignored()
end

function update_npc(actor_id, x, y, z, game_hour)
    local actor = lower(actor_id)
    if actor ~= FARGOTH then return generic_schedule(actor, game_hour) end
    local hiding_stage = game.journal(FARGOTH_HIDING_QUEST)
    if hiding_stage < 10 or hiding_stage >= 20 then return generic_schedule(actor, game_hour) end
    if hiding_stage == 10 then
        local hour = game_hour or 14.0
        local is_night = hour >= 20.0 or hour < 5.0
        if not is_night then return generic_schedule(actor, game_hour) end
        start_quest(FARGOTH_HIDING_QUEST, 12, "Follow Fargoth to his hiding place.")
        return {
            handled = true,
            speed = 70.0,
            message = "Fargoth looks around nervously and starts walking away from town.",
            route = {
                route_point(-12369.92, 0.0, -69672.32),
                route_point(-12943.36, 0.0, -69098.88),
                route_point(FARGOTH_STASH_X, FARGOTH_STASH_Y, FARGOTH_STASH_Z),
            }
        }
    end
    if hiding_stage == 12 then
        local dx = x - FARGOTH_STASH_X
        local dz = z - FARGOTH_STASH_Z
        if (dx * dx) + (dz * dz) <= (190.0 * 190.0) then
            start_quest(FARGOTH_HIDING_QUEST, 15, "Recover Fargoth's hidden stash.")
            game.log("Fargoth revealed his hiding place.")
            return { handled = true, stop = true, message = "Fargoth kneels briefly by a hidden stump, then hurries away." }
        end
        return { handled = true, speed = 70.0 }
    end
    return { handled = true, stop = true }
end
