local QUEST = "MV_DeadTaxman"
local TAX_RECORD = "bk_seydaneentaxrecord"
local PROCESSUS = "processus vitellius"
local SOCUCIUS = "socucius ergalla"
local FORYN = "foryn gilnith"
local RING = "processus_ring"

local function result(message)
    return { handled = true, message = message }
end

local function ignored()
    return { handled = false }
end

function on_activate(ref_id)
    local id = string.lower(ref_id)
    if id == PROCESSUS then
        if game.journal(QUEST) < 10 then
            game.set_ref_dead(PROCESSUS, true)
            game.add_item(TAX_RECORD, 1)
            game.add_gold(200)
            game.set_journal(QUEST, 10)
            game.log("Found Processus Vitellius and recovered the tax record.")
            return result("You found Processus Vitellius' corpse. Tax record and 200 gold recovered.")
        end
        return result("Processus Vitellius' corpse has already been searched.")
    end
    return ignored()
end

function get_dialogue(actor_id, topic_id)
    local actor = string.lower(actor_id)
    local topic = string.lower(topic_id)
    if actor == SOCUCIUS and (topic == "" or topic == "murder of processus vitellius") then
        if game.journal(QUEST) == 10 and game.has_item(TAX_RECORD) then
            return {
                handled = true,
                text = "Processus is dead? Did you find the tax money he collected?",
                choices = {
                    { id = "taxman_truth", text = "Tell him about the 200 gold." },
                    { id = "taxman_lie", text = "Say there was no money." },
                },
            }
        elseif game.journal(QUEST) >= 30 and game.journal(QUEST) < 70 then
            return {
                handled = true,
                text = "Find whoever murdered Processus Vitellius and report back to me.",
                choices = {},
            }
        elseif game.journal(QUEST) >= 70 then
            return {
                handled = true,
                text = "You have done the Census and Excise Office a service.",
                choices = {},
            }
        end
    end

    if actor == FORYN and game.journal(QUEST) >= 30 and game.journal(QUEST) < 70 then
        return {
            handled = true,
            text = "Processus pushed people too far. I did what had to be done.",
            choices = {
                { id = "taxman_kill_foryn", text = "Attack Foryn." },
                { id = "taxman_spare_foryn", text = "Leave him for now." },
            },
        }
    end

    return { handled = false }
end

function choose_dialogue(response_id)
    if response_id == "taxman_truth" and game.journal(QUEST) == 10 then
        game.spend_gold(200)
        game.set_journal(QUEST, 30)
        return result("You returned the tax money. Socucius asks you to find the killer.")
    end

    if response_id == "taxman_lie" and game.journal(QUEST) == 10 then
        game.set_journal(QUEST, 40)
        return result("You kept the tax money quiet. Socucius still wants the killer found.")
    end

    if response_id == "taxman_kill_foryn" and game.journal(QUEST) >= 30 and game.journal(QUEST) < 70 then
        game.set_ref_dead(FORYN, true)
        game.add_item(RING, 1)
        game.set_journal(QUEST, 70)
        game.add_gold(500)
        return result("Foryn Gilnith is dead. Socucius' reward has been paid.")
    end

    if response_id == "taxman_spare_foryn" then
        return result("You step back from the accusation for now.")
    end

    return ignored()
end

function on_actor_death(actor_id)
    if string.lower(actor_id) == FORYN and game.journal(QUEST) >= 30 and game.journal(QUEST) < 70 then
        game.add_item(RING, 1)
        game.set_journal(QUEST, 70)
        game.add_gold(500)
        return result("Foryn Gilnith is dead.")
    end
    return ignored()
end
