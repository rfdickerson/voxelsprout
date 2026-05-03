local TAXMAN_QUEST = "MV_DeadTaxman"
local FARGOTH_RING_QUEST = "MV_FargothRing"
local FARGOTH_HIDING_QUEST = "MV_FargothHiding"
local VODUNIUS_QUEST = "MV_Vodunius"
local CAIUS_QUEST = "MV_ReportToCaius"

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

local function common_topics()
    return {
        topic("seyda neen", "Seyda Neen"),
        topic("latest rumors", "Latest rumors"),
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

local function route_point(x, y, z)
    return { x = x, y = y, z = z }
end

local function lower(value)
    return string.lower(value or "")
end

local function fargoth_topics()
    local topics = common_topics()
    append_topic(topics, "missing ring", "Missing ring")
    if game.journal(FARGOTH_HIDING_QUEST) >= 10 then
        append_topic(topics, "hiding place", "Hiding place")
    end
    return topics
end

local function hrisskar_topics()
    local topics = common_topics()
    if game.journal(FARGOTH_RING_QUEST) >= 30 then
        append_topic(topics, "fargoth's hiding place", "Fargoth's hiding place")
    end
    return topics
end

local function taxman_topics()
    local topics = common_topics()
    append_topic(topics, "murder of processus vitellius", "Murder of Processus Vitellius")
    return topics
end

function on_activate(ref_id)
    local id = lower(ref_id)
    if id == PROCESSUS then
        if game.journal(TAXMAN_QUEST) < 10 then
            game.set_ref_dead(PROCESSUS, true)
            game.add_item(TAX_RECORD, 1)
            game.add_gold(200)
            game.set_journal(TAXMAN_QUEST, 10)
            game.log("Found Processus Vitellius and recovered the tax record.")
            return result("You found Processus Vitellius' corpse. Tax record and 200 gold recovered.")
        end
        return result("Processus Vitellius' corpse has already been searched.")
    end

    if id == "fargoth ring barrel" then
        if game.item_count(FARGOTH_RING) == 0 and game.journal(FARGOTH_RING_QUEST) == 0 then
            game.add_item(FARGOTH_RING, 1)
            game.set_journal(FARGOTH_RING_QUEST, 10)
            return result("You found an engraved ring of healing in the barrel.")
        end
        return result("The barrel has already been searched.")
    end

    if id == "sellus package" then
        if game.item_count(PACKAGE) == 0 and game.journal(CAIUS_QUEST) == 0 then
            game.add_item(PACKAGE, 1)
            game.set_journal(CAIUS_QUEST, 10)
            return result("You received a package for Caius Cosades.")
        end
        return result("You already have your orders.")
    end

    if id == FARGOTH_STASH_REF then
        if game.journal(FARGOTH_HIDING_QUEST) >= 15 and game.journal(FARGOTH_HIDING_QUEST) < 20 then
            if game.item_count(FARGOTH_STASH) == 0 then
                game.add_item(FARGOTH_STASH, 1)
                return result("You recover Fargoth's hidden stash.")
            end
            return result("You already recovered Fargoth's hidden stash.")
        end
        return ignored()
    end

    return ignored()
end

function get_dialogue(actor_id, topic_id)
    local actor = lower(actor_id)
    local topic_id = lower(topic_id)

    if topic_id == "seyda neen" then
        return dialogue(
            "Seyda Neen is the Empire's damp little doorway into Vvardenfell: Census office, tradehouse, lighthouse, and the silt strider.",
            common_topics(),
            {}
        )
    end

    if topic_id == "latest rumors" then
        return dialogue(
            "People talk about Fargoth's missing ring, Hrisskar's schemes, and a tax collector who has not been seen.",
            common_topics(),
            {}
        )
    end

    if actor == FARGOTH then
        if topic_id == "missing ring" then
            if game.journal(FARGOTH_RING_QUEST) >= 30 then
                return dialogue(
                    "You found my ring. I will not forget that kindness.",
                    fargoth_topics(),
                    {}
                )
            end
            if game.has_item(FARGOTH_RING) then
                return dialogue(
                    "That is it! My engraved ring. Please, sera, may I have it back?",
                    fargoth_topics(),
                    { choice("fargoth_return_ring", "Return the engraved ring.") }
                )
            end
            return dialogue(
                "I lost an engraved ring of healing. The guards have been laughing about it.",
                fargoth_topics(),
                {}
            )
        end
        if topic_id == "hiding place" then
            return dialogue(
                "Why would you ask me that? I keep to myself, and still Hrisskar will not leave me be.",
                fargoth_topics(),
                {}
            )
        end
        return dialogue(
            "Are you the one that boat dropped off? Odd time for a ship to arrive. If you find my ring, please bring it to me.",
            fargoth_topics(),
            {}
        )
    end

    if actor == HRISSKAR then
        if topic_id == "fargoth's hiding place" then
            if game.journal(FARGOTH_RING_QUEST) < 30 then
                return dialogue(
                    "Talk to the little Bosmer first. He is jumpier than usual.",
                    hrisskar_topics(),
                    {}
                )
            end
            if game.journal(FARGOTH_HIDING_QUEST) >= 20 then
                return dialogue(
                    "You found the stash. Fargoth will be missing more than his ring now.",
                    hrisskar_topics(),
                    {}
                )
            end
            if game.has_item(FARGOTH_STASH) then
                return dialogue(
                    "So you saw him hide it? Hand it over, and I will make it worth your time.",
                    hrisskar_topics(),
                    { choice("hrisskar_turn_in_stash", "Give Hrisskar Fargoth's stash.") }
                )
            end
            if game.journal(FARGOTH_HIDING_QUEST) >= 15 then
                return dialogue(
                    "You saw where he hid it. Get the stash and bring it to me.",
                    hrisskar_topics(),
                    {}
                )
            end
            return dialogue(
                "Fargoth has a hiding place. Watch him from the lighthouse after dark, then bring me what he stashed.",
                hrisskar_topics(),
                { choice("hrisskar_accept_hiding", "Agree to watch Fargoth.") }
            )
        end
        return dialogue(
            "Hrisskar Flat-Foot. I keep order upstairs at Arrille's when order needs keeping.",
            hrisskar_topics(),
            {}
        )
    end

    if actor == SOCUCIUS and (topic_id == "" or topic_id == "murder of processus vitellius") then
        if game.journal(TAXMAN_QUEST) == 10 and game.has_item(TAX_RECORD) then
            return dialogue(
                "Processus is dead? Did you find the tax money he collected?",
                taxman_topics(),
                {
                    choice("taxman_truth", "Tell him about the 200 gold."),
                    choice("taxman_lie", "Say there was no money."),
                }
            )
        elseif game.journal(TAXMAN_QUEST) >= 30 and game.journal(TAXMAN_QUEST) < 70 then
            return dialogue(
                "Find whoever murdered Processus Vitellius and report back to me.",
                taxman_topics(),
                {}
            )
        elseif game.journal(TAXMAN_QUEST) >= 70 then
            return dialogue(
                "You have done the Census and Excise Office a service.",
                taxman_topics(),
                {}
            )
        end
        return dialogue(
            "Official business belongs at the Census and Excise Office. If you hear anything about Processus, report it.",
            taxman_topics(),
            {}
        )
    end

    if actor == FORYN and game.journal(TAXMAN_QUEST) >= 30 and game.journal(TAXMAN_QUEST) < 70 then
        return dialogue(
            "Processus pushed people too far. I did what had to be done.",
            taxman_topics(),
            {
                choice("taxman_kill_foryn", "Attack Foryn."),
                choice("taxman_spare_foryn", "Leave him for now."),
            }
        )
    end

    if actor == SELLUS then
        local topics = common_topics()
        append_topic(topics, "orders", "Orders")
        if topic_id == "orders" then
            if game.journal(CAIUS_QUEST) >= 30 then
                return dialogue(
                    "Your orders are clear. Find Caius Cosades in Balmora.",
                    topics,
                    {}
                )
            end
            return dialogue(
                "Take the package to Caius Cosades in Balmora. The silt strider is the sensible road from here.",
                topics,
                { choice("sellus_take_package", "Take the package for Caius.") }
            )
        end
        return dialogue(
            "You are released by order of the Emperor. Keep your papers and follow your instructions.",
            topics,
            {}
        )
    end

    if actor == VODUNIUS then
        local topics = common_topics()
        append_topic(topics, "leaving seyda neen", "Leaving Seyda Neen")
        if topic_id == "leaving seyda neen" then
            if game.journal(VODUNIUS_QUEST) >= 30 then
                return dialogue(
                    "Thanks to you, I can finally get out of this swamp.",
                    topics,
                    {}
                )
            end
            return dialogue(
                "I have had enough of Seyda Neen. A little money would get me closer to the road home.",
                topics,
                { choice("vodunius_give_gold", "Give Vodunius 100 gold.") }
            )
        end
        return dialogue(
            "Vodunius Nuccius. Down on my luck, and not much fond of this place.",
            topics,
            {}
        )
    end

    return ignored()
end

function choose_dialogue(response_id)
    if response_id == "fargoth_return_ring" and game.journal(FARGOTH_RING_QUEST) < 30 then
        if game.remove_item(FARGOTH_RING, 1) then
            game.set_journal(FARGOTH_RING_QUEST, 30)
            game.log("Returned Fargoth's engraved ring.")
            return result("Fargoth beams with relief. Arrille may look more kindly on you now.")
        end
        return ignored()
    end

    if response_id == "hrisskar_accept_hiding" and game.journal(FARGOTH_RING_QUEST) >= 30 and game.journal(FARGOTH_HIDING_QUEST) < 10 then
        game.set_journal(FARGOTH_HIDING_QUEST, 10)
        return result("Hrisskar tells you to watch Fargoth from the lighthouse after dark.")
    end

    if response_id == "hrisskar_turn_in_stash" and game.journal(FARGOTH_HIDING_QUEST) >= 15 and game.journal(FARGOTH_HIDING_QUEST) < 20 then
        if game.remove_item(FARGOTH_STASH, 1) then
            game.set_journal(FARGOTH_HIDING_QUEST, 20)
            game.add_gold(100)
            return result("Hrisskar takes the stash and pays you 100 gold.")
        end
        return ignored()
    end

    if response_id == "sellus_take_package" and game.journal(CAIUS_QUEST) < 30 then
        if game.item_count(PACKAGE) == 0 then
            game.add_item(PACKAGE, 1)
        end
        game.set_journal(CAIUS_QUEST, 30)
        return result("Sellus gives you the package for Caius Cosades in Balmora.")
    end

    if response_id == "vodunius_give_gold" and game.journal(VODUNIUS_QUEST) < 30 then
        if game.spend_gold(100) then
            game.set_journal(VODUNIUS_QUEST, 30)
            return result("Vodunius accepts the gold and starts planning his way out of Seyda Neen.")
        end
        return ignored()
    end

    if response_id == "taxman_truth" and game.journal(TAXMAN_QUEST) == 10 then
        game.spend_gold(200)
        game.set_journal(TAXMAN_QUEST, 30)
        return result("You returned the tax money. Socucius asks you to find the killer.")
    end

    if response_id == "taxman_lie" and game.journal(TAXMAN_QUEST) == 10 then
        game.set_journal(TAXMAN_QUEST, 40)
        return result("You kept the tax money quiet. Socucius still wants the killer found.")
    end

    if response_id == "taxman_kill_foryn" and game.journal(TAXMAN_QUEST) >= 30 and game.journal(TAXMAN_QUEST) < 70 then
        game.set_ref_dead(FORYN, true)
        game.add_item(PROCESSUS_RING, 1)
        game.set_journal(TAXMAN_QUEST, 70)
        game.add_gold(500)
        return result("Foryn Gilnith is dead. Socucius' reward has been paid.")
    end

    if response_id == "taxman_spare_foryn" then
        return result("You step back from the accusation for now.")
    end

    return ignored()
end

function on_actor_death(actor_id)
    if lower(actor_id) == FORYN and game.journal(TAXMAN_QUEST) >= 30 and game.journal(TAXMAN_QUEST) < 70 then
        game.add_item(PROCESSUS_RING, 1)
        game.set_journal(TAXMAN_QUEST, 70)
        game.add_gold(500)
        return result("Foryn Gilnith is dead.")
    end
    return ignored()
end

function update_npc(actor_id, x, y, z, game_hour)
    local actor = lower(actor_id)
    if actor ~= FARGOTH then
        return { handled = false }
    end

    local hiding_stage = game.journal(FARGOTH_HIDING_QUEST)
    if hiding_stage < 10 or hiding_stage >= 20 then
        return { handled = true, stop = true }
    end

    if hiding_stage == 10 then
        local hour = game_hour or 14.0
        local is_night = hour >= 20.0 or hour < 5.0
        if not is_night then
            return { handled = true, stop = true }
        end
        game.set_journal(FARGOTH_HIDING_QUEST, 12)
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
            game.set_journal(FARGOTH_HIDING_QUEST, 15)
            game.log("Fargoth revealed his hiding place.")
            return {
                handled = true,
                stop = true,
                message = "Fargoth kneels briefly by a hidden stump, then hurries away."
            }
        end
        return { handled = true, speed = 70.0 }
    end

    return { handled = true, stop = true }
end
