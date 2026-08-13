#include "import/fnv/dialogue_records.h"

#include "import/fnv/esm_reader.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace odai::importer::fnv {

namespace {

std::string toLowerAsciiCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

// Subrecord payloads are zero-terminated in the file; the terminator is part of
// the declared size, so it has to come off or every string carries a trailing
// NUL into the UI.
std::string subrecordString(const EsmSubrecordView& sub) {
    if (sub.data == nullptr || sub.size == 0u) {
        return {};
    }
    std::string value(reinterpret_cast<const char*>(sub.data), static_cast<std::size_t>(sub.size));
    while (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

std::uint32_t readU32(const std::uint8_t* data) {
    std::uint32_t value = 0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

std::string nodeIdFor(std::uint32_t infoFormId) {
    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "info_%08X", infoFormId);
    return buffer;
}

struct RawInfo {
    std::uint32_t formId = 0;
    std::uint32_t topicFormId = 0;
    std::string text;
    std::vector<std::uint32_t> linkedTopics;
};

}  // namespace

bool findSpeakerPlacement(
    const std::filesystem::path& pluginPath,
    const std::string& speakerEditorId,
    SpeakerPlacement& outPlacement,
    std::string& outError
) {
    outPlacement = SpeakerPlacement{};
    outError.clear();

    EsmReader reader;
    if (!reader.open(pluginPath)) {
        outError = "cannot open plugin: " + reader.lastError();
        return false;
    }
    const std::string wantedEditorId = toLowerAsciiCopy(speakerEditorId);

    std::uint32_t speakerFormId = 0;
    {
        EsmReader::Visitor visitor;
        // Reject by TYPE before the body is touched. Without this the walk
        // materializes every record in the file to hand it to onRecord, which
        // means inflating FalloutNV.esm's 29363 compressed LAND records to look
        // for a creature -- ~3.4 s and hundreds of MB of heap, per scan. See
        // EsmReader::Visitor::onRecordHeader.
        visitor.onRecordHeader = [&](const EsmRecordHeaderView& header) {
            return speakerFormId == 0u && (header.type == "CREA" || header.type == "NPC_");
        };
        visitor.onRecord = [&](const EsmRecordView& record) {
            if (speakerFormId != 0u || (record.type != "CREA" && record.type != "NPC_")) {
                return;
            }
            bool isWanted = false;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "EDID" && toLowerAsciiCopy(subrecordString(sub)) == wantedEditorId) {
                    isWanted = true;
                }
            }
            if (!isWanted) {
                return;
            }
            speakerFormId = record.formId;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "MODL") {
                    outPlacement.skeletonPath = subrecordString(sub);
                } else if (sub.type == "NIFZ" && sub.size != 0u) {
                    // NUL-separated body-part filenames, relative to the
                    // skeleton's own directory.
                    const std::string blob(
                        reinterpret_cast<const char*>(sub.data), static_cast<std::size_t>(sub.size));
                    std::size_t begin = 0;
                    while (begin < blob.size()) {
                        const std::size_t end = blob.find('\0', begin);
                        std::string part = blob.substr(
                            begin, end == std::string::npos ? std::string::npos : end - begin);
                        if (!part.empty()) {
                            outPlacement.bodyPartPaths.push_back(std::move(part));
                        }
                        if (end == std::string::npos) {
                            break;
                        }
                        begin = end + 1;
                    }
                }
            }
        };
        reader.walk(visitor);
    }
    if (speakerFormId == 0u) {
        outError = "no CREA/NPC_ with EditorID \"" + speakerEditorId + "\"";
        return false;
    }

    outPlacement.baseFormId = speakerFormId;

    // ACRE places a creature, ACHR an NPC; both carry the same NAME (base) and
    // DATA (position + rotation) subrecords a REFR does.
    std::uint32_t currentCell = 0;
    EsmReader::Visitor visitor;
    // CELL for the enclosing-cell id, ACRE/ACHR for the placement itself.
    // Everything else -- crucially the compressed LAND records interleaved
    // among them -- is rejected on its header.
    visitor.onRecordHeader = [&](const EsmRecordHeaderView& header) {
        return !outPlacement.found &&
               (header.type == "CELL" || header.type == "ACRE" || header.type == "ACHR");
    };
    visitor.onRecord = [&](const EsmRecordView& record) {
        if (record.type == "CELL") {
            currentCell = record.formId;
            return;
        }
        if (outPlacement.found || (record.type != "ACRE" && record.type != "ACHR")) {
            return;
        }
        std::uint32_t base = 0;
        bool hasData = false;
        float data[6] = {};
        for (const auto& sub : record.subrecords) {
            if (sub.type == "NAME" && sub.size >= 4u) {
                base = readU32(sub.data);
            } else if (sub.type == "DATA" && sub.size >= 24u) {
                std::memcpy(data, sub.data, sizeof(data));
                hasData = true;
            }
        }
        if (base != speakerFormId || !hasData) {
            return;
        }
        outPlacement.referenceFormId = record.formId;
        outPlacement.cellFormId = currentCell;
        for (int i = 0; i < 3; ++i) {
            outPlacement.position[i] = data[i];
            outPlacement.rotationRadians[i] = data[3 + i];
        }
        outPlacement.found = true;
    };
    if (!reader.walk(visitor)) {
        outError = "plugin walk failed while looking for the speaker's reference";
        return false;
    }
    if (!outPlacement.found) {
        outError = "no ACRE/ACHR reference places " + speakerEditorId;
        return false;
    }
    return true;
}

bool buildSpeakerDialogueTree(
    const std::filesystem::path& pluginPath,
    const std::string& speakerEditorId,
    odai::dialogue::DialogueTree& outTree,
    DialogueImportStats& outStats,
    std::string& outError
) {
    outTree = odai::dialogue::DialogueTree{};
    outStats = DialogueImportStats{};
    outError.clear();

    EsmReader reader;
    if (!reader.open(pluginPath)) {
        outError = "cannot open plugin: " + reader.lastError();
        return false;
    }
    const std::string wantedEditorId = toLowerAsciiCopy(speakerEditorId);

    // Pass 1: the speaker's formID. An exact EDID match, not a substring --
    // "Victor" alone also matches VictorLauncher, VictorLaser and eight others,
    // which are his weapon variants and a disabled duplicate.
    std::uint32_t speakerFormId = 0;
    std::string speakerName;
    {
        EsmReader::Visitor visitor;
        visitor.onRecordHeader = [&](const EsmRecordHeaderView& header) {
            return speakerFormId == 0u && (header.type == "CREA" || header.type == "NPC_");
        };
        visitor.onRecord = [&](const EsmRecordView& record) {
            if (speakerFormId != 0u || (record.type != "CREA" && record.type != "NPC_")) {
                return;
            }
            for (const auto& sub : record.subrecords) {
                if (sub.type != "EDID") {
                    continue;
                }
                const std::string edid = subrecordString(sub);
                if (toLowerAsciiCopy(edid) == wantedEditorId) {
                    speakerFormId = record.formId;
                    speakerName = edid;
                }
            }
        };
        if (!reader.walk(visitor)) {
            outError = "plugin walk failed while looking for the speaker";
            return false;
        }
    }
    if (speakerFormId == 0u) {
        outError = "no CREA/NPC_ with EditorID \"" + speakerEditorId + "\"";
        return false;
    }

    // Pass 2 is the batch reader, which is the same walk with one entry in the
    // wanted set. Keeping one implementation means the single-speaker path
    // cannot drift away from the one the town actually uses.
    std::unordered_map<std::uint32_t, odai::dialogue::DialogueTree> trees;
    std::unordered_map<std::uint32_t, DialogueImportStats> stats;
    if (!buildSpeakerDialogueTrees(
            pluginPath,
            std::vector<SpeakerDialogueRequest>{
                SpeakerDialogueRequest{speakerFormId, /*voiceTypeFormId=*/0u, speakerName}},
            trees, stats, outError)) {
        return false;
    }
    const auto tree = trees.find(speakerFormId);
    if (tree == trees.end()) {
        outError = "no dialogue responses found for " + speakerName;
        return false;
    }
    outTree = std::move(tree->second);
    const auto stat = stats.find(speakerFormId);
    if (stat != stats.end()) {
        outStats = stat->second;
    }
    return true;
}

bool buildSpeakerDialogueTreesImpl(
    const std::filesystem::path& pluginPath,
    const FalloutLoadOrder* order,
    const std::vector<SpeakerDialogueRequest>& speakers,
    std::unordered_map<std::uint32_t, odai::dialogue::DialogueTree>& outTrees,
    std::unordered_map<std::uint32_t, DialogueImportStats>& outStats,
    std::string& outError
) {
    outTrees.clear();
    outStats.clear();
    outError.clear();
    if (speakers.empty()) {
        return true;
    }

    // One entry per plugin to walk. A single-plugin call is one entry with the
    // identity rewrite, so both paths run the same code and cannot drift.
    struct DialogueSource {
        std::filesystem::path path;
        std::size_t pluginIndex = 0;
    };
    std::vector<DialogueSource> sources;
    if (order != nullptr && !order->empty()) {
        for (std::size_t i = 0; i < order->entries().size(); ++i) {
            sources.push_back(DialogueSource{order->entries()[i].path, i});
        }
    } else {
        sources.push_back(DialogueSource{pluginPath, 0u});
    }

    std::unordered_map<std::uint32_t, std::string> nameByFormId;
    // Voice type -> every speaker that uses it. One line naming a voice type is
    // a line for all of them, so this is a list rather than a single owner.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> speakersByVoiceType;
    for (const SpeakerDialogueRequest& speaker : speakers) {
        if (speaker.baseFormId == 0u) {
            continue;
        }
        nameByFormId.emplace(speaker.baseFormId, speaker.displayName);
        if (speaker.voiceTypeFormId != 0u) {
            speakersByVoiceType[speaker.voiceTypeFormId].push_back(speaker.baseFormId);
        }
    }
    if (nameByFormId.empty()) {
        return true;
    }

    // One walk over DIAL/INFO for everybody. INFO records stream immediately
    // after the DIAL that owns them (they live in its child group), so tracking
    // the most recent DIAL is enough to know each response's topic -- no group
    // bookkeeping required.
    std::unordered_map<std::uint32_t, std::string> topicPlayerText;
    std::unordered_map<std::uint32_t, std::string> topicEditorId;
    std::unordered_map<std::uint32_t, std::uint8_t> topicType;
    std::unordered_map<std::uint32_t, std::vector<RawInfo>> infosBySpeaker;

    // A line whose speaker is named by its QUEST rather than by itself.
    //
    // The GECK lets a quest carry conditions that apply to every INFO in it, and
    // that is where a companion mod puts "GetIsID <me>" -- once, on the quest,
    // instead of on each of hundreds of lines. The INFOs then carry only what
    // varies between them (quest variables, faction, karma), so a reader that
    // looks for the speaker on the INFO alone finds NOBODY and drops the lot.
    //
    // Measured on Willow: 82 of her 90 greeting INFOs name no speaker at all and
    // belong to quest AWillowD, whose single condition is GetIsID 0x1000ADD --
    // her. That is why she had 84 recorded greetings on disk and none in her
    // tree.
    //
    // Resolved after every plugin is walked, because nothing orders a quest
    // ahead of the dialogue that cites it -- and the quest may be in a different
    // plugin than the line.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> ownersByQuest;
    // Every line in walk order, with its speakers left unresolved. Collected
    // rather than filed immediately so that quest-attributed lines keep their
    // position among the rest: "the first INFO under this topic" and "the first
    // greeting" both depend on this order.
    struct CollectedInfo {
        RawInfo info;
        std::vector<std::uint32_t> owners;
        std::uint32_t questFormId = 0;
        std::uint32_t responses = 0;
    };
    std::vector<CollectedInfo> collected;
    std::uint32_t currentTopic = 0;
    std::uint32_t topicsSeen = 0;
    std::unordered_map<std::uint32_t, DialogueImportStats> stats;
    for (const DialogueSource& source : sources) {
        EsmReader reader;
        if (!reader.open(source.path)) {
            if (sources.size() == 1u) {
                outError = "cannot open plugin: " + reader.lastError();
                return false;
            }
            continue;  // one unreadable plugin costs its lines, not everyone's
        }
        // Rewrites this plugin's local formIDs into the order's global space.
        // EVERY id leaving this walk goes through it: the topic a response hangs
        // under, the response itself, the speaker or voice type a condition
        // names, and each linked topic. A missed one silently attributes a line
        // to a different actor.
        const auto remap = [&](std::uint32_t formId) {
            if (order == nullptr || formId == 0u) {
                return formId;
            }
            return order->remapFormId(source.pluginIndex, formId);
        };
        currentTopic = 0u;  // a topic does not carry across plugins
        EsmReader::Visitor visitor;
        // A topic's INFOs live in its TOPIC CHILDREN group (type 7), whose label
        // is the owning DIAL's formID. Taking the topic from the group rather
        // than from "the last DIAL record seen" is what makes an override
        // plugin work: a mod that adds lines to a shared topic ships the child
        // group and NOT the DIAL, so the record-order heuristic attributes every
        // one of them to whatever topic happened to precede them.
        //
        // Willow is exactly that case. Her plugin defines no GREETING topic --
        // her 84 greeting lines hang under the base game's -- so her greeting
        // was being filed under an unrelated topic and could never be found.
        constexpr std::int32_t kTopicChildrenGroup = 7;
        visitor.onGroupEnter = [&](const EsmGroupView& group) {
            if (group.groupType == kTopicChildrenGroup && group.rawLabel.size() == 4u) {
                std::uint32_t topicFormId = 0;
                std::memcpy(&topicFormId, group.rawLabel.data(), 4u);
                currentTopic = remap(topicFormId);
            }
            return true;
        };
        visitor.onRecordHeader = [](const EsmRecordHeaderView& header) {
            return header.type == "DIAL" || header.type == "INFO" || header.type == "QUST";
        };
        // Every speaker a record's CTDA conditions hand it to. Shared by INFO
        // and QUST because the conditions are the same format and mean the same
        // thing in both places -- the only difference is that a quest's apply to
        // all of its dialogue rather than to one line.
        //
        // outNamedSomeone reports whether the record named ANY speaker, matched
        // or not. That is the difference between "this line says who says it and
        // it is not one of ours" and "this line says nothing", and only the
        // second may fall back to the quest -- see the QSTI handling below.
        const auto collectOwners = [&](const EsmRecordView& record,
                                       std::vector<std::uint32_t>& owners,
                                       bool* outNamedSomeone = nullptr) {
            for (const auto& sub : record.subrecords) {
                if (sub.type != "CTDA" || sub.size < 28u) {
                    continue;
                }
                // CTDA (FO3/FNV, 28 bytes): type u8 @0, 3 unused, comparison
                // f32 @4, function index u32 @8, param1 u32 @12, param2, runOn,
                // reference.
                const std::uint32_t function = readU32(sub.data + 8);
                const bool byActor = function == kCtdaFunctionGetIsId;
                const bool byVoiceType = function == kCtdaFunctionGetIsVoiceType;
                if (!byActor && !byVoiceType) {
                    continue;
                }
                const std::uint32_t named = remap(readU32(sub.data + 12));
                // The operator in the type byte's top 3 bits has to be read, not
                // just the function and its parameter. Fallout writes "everyone
                // EXCEPT this actor" as GetIsID <actor> == 0, which is the same
                // function and the same parameter as the line that belongs to
                // him -- only the comparison differs. Ignoring it attributed
                // other characters' lines to the speaker.
                float comparisonValue = 0.0f;
                std::memcpy(&comparisonValue, sub.data + 4, sizeof(comparisonValue));
                const auto comparisonOperator = static_cast<std::uint8_t>((sub.data[0] >> 5) & 0x7u);
                // 0 = EQUAL, 2 = GREATER, 3 = GREATER-OR-EQUAL: all read as
                // "is this actor" when tested against 1.
                const bool positive = (comparisonOperator == 0u && comparisonValue == 1.0f) ||
                    (comparisonOperator == 2u && comparisonValue == 0.0f) ||
                    (comparisonOperator == 3u && comparisonValue == 1.0f);
                if (!positive) {
                    continue;
                }
                // Tested BEFORE matching against the wanted speakers: a line
                // that names an actor nobody asked about has still named one,
                // and must not be handed to whoever the quest belongs to.
                if (outNamedSomeone != nullptr) {
                    *outNamedSomeone = true;
                }
                // Whom this condition hands the line to. For GetIsID that is one
                // actor; for GetIsVoiceType it is everyone sharing the voice.
                const std::vector<std::uint32_t>* claimants = nullptr;
                std::vector<std::uint32_t> single;
                if (byActor) {
                    if (nameByFormId.find(named) == nameByFormId.end()) {
                        continue;
                    }
                    single.push_back(named);
                    claimants = &single;
                } else {
                    const auto sharing = speakersByVoiceType.find(named);
                    if (sharing == speakersByVoiceType.end()) {
                        continue;
                    }
                    claimants = &sharing->second;
                }
                for (const std::uint32_t claimant : *claimants) {
                    if (std::find(owners.begin(), owners.end(), claimant) == owners.end()) {
                        owners.push_back(claimant);
                    }
                }
            }
        };
        visitor.onRecord = [&](const EsmRecordView& record) {
            if (record.type == "QUST") {
                std::vector<std::uint32_t> owners;
                collectOwners(record, owners);
                if (!owners.empty()) {
                    ownersByQuest[remap(record.formId)] = std::move(owners);
                }
                return;
            }
            if (record.type == "DIAL") {
                currentTopic = remap(record.formId);
                ++topicsSeen;
                for (const auto& sub : record.subrecords) {
                    if (sub.type == "FULL") {
                        topicPlayerText[currentTopic] = subrecordString(sub);
                    } else if (sub.type == "EDID") {
                        // The EditorID, NOT the player-facing FULL. A greeting
                        // has no FULL at all -- the player never picks it, the
                        // game opens on it -- so FULL cannot identify one.
                        topicEditorId[currentTopic] = subrecordString(sub);
                    } else if (sub.type == "DATA" && sub.size >= 1u) {
                        // Byte 0 is the dialogue TYPE: 0 Topic, 1 Conversation,
                        // 2 Combat, 3 Persuasion, 4 Detection, 5 Service,
                        // 6 Miscellaneous. Only type 0 is something a player
                        // can be shown; the rest are barks the game triggers.
                        topicType[currentTopic] = sub.data[0];
                    }
                }
                return;
            }
            if (record.type != "INFO") {
                return;
            }
            // Which of the wanted speakers this line belongs to. A line can name
            // more than one, so this collects rather than stopping at the first.
            std::vector<std::uint32_t> owners;
            bool namedSomeone = false;
            collectOwners(record, owners, &namedSomeone);
            // QSTI is the quest this line belongs to, and the fallback when the
            // line names NOBODY: the quest's conditions then name the speaker
            // for all of its dialogue at once.
            //
            // Only when it names nobody. A quest holds one character's lines but
            // not exclusively -- companion mods put other actors' responses in
            // the same quest, each named on its own INFO -- so falling back
            // whenever the wanted speaker simply did not match would hand the
            // quest's owner every other character's dialogue too.
            std::uint32_t questFormId = 0;
            if (!namedSomeone) {
                for (const auto& sub : record.subrecords) {
                    if (sub.type == "QSTI" && sub.size >= 4u) {
                        questFormId = remap(readU32(sub.data));
                        break;
                    }
                }
            }
            if (owners.empty() && questFormId == 0u) {
                return;
            }

            RawInfo info;
            info.formId = remap(record.formId);
            info.topicFormId = currentTopic;
            std::uint32_t responses = 0;
            for (const auto& sub : record.subrecords) {
                if (sub.type == "NAM1") {
                    const std::string line = subrecordString(sub);
                    if (line.empty()) {
                        continue;
                    }
                    // One INFO can hold several responses, separated by NEXT;
                    // they are spoken back to back, so they join into one node
                    // rather than becoming nodes the player has to click through.
                    if (!info.text.empty()) {
                        info.text += "  ";
                    }
                    info.text += line;
                    ++responses;
                } else if (sub.type == "TCLT" && sub.size >= 4u) {
                    info.linkedTopics.push_back(remap(readU32(sub.data)));
                }
            }
            if (info.text.empty()) {
                return;  // a silent INFO (script-only) is not a line to show
            }
            collected.push_back(CollectedInfo{std::move(info), std::move(owners), questFormId,
                                             responses});
        };
        if (!reader.walk(visitor)) {
            outError = "plugin walk failed while reading dialogue";
            return false;
        }
    }

    // Now that every plugin's quests are known, file each line under its
    // speakers -- in walk order, so a quest-attributed line sits where it was
    // read rather than after everything else.
    for (CollectedInfo& entry : collected) {
        const std::vector<std::uint32_t>* owners = &entry.owners;
        if (owners->empty()) {
            const auto questIt = ownersByQuest.find(entry.questFormId);
            if (questIt == ownersByQuest.end()) {
                continue;  // names nobody, and its quest names nobody either
            }
            owners = &questIt->second;
        }
        for (const std::uint32_t owner : *owners) {
            infosBySpeaker[owner].push_back(entry.info);
            DialogueImportStats& speakerStats = stats[owner];
            ++speakerStats.infosForSpeaker;
            speakerStats.responsesConcatenated += entry.responses;
        }
    }

    for (const auto& [speakerFormId, infos] : infosBySpeaker) {
        if (infos.empty()) {
            continue;
        }
        const auto nameIt = nameByFormId.find(speakerFormId);
        const std::string& speakerName =
            nameIt != nameByFormId.end() ? nameIt->second : std::string();
        DialogueImportStats& speakerStats = stats[speakerFormId];
        speakerStats.topicsSeen = topicsSeen;

        // Topic -> the speaker's first response under it. First rather than a
        // list because choosing a topic has to land somewhere definite and
        // conditions (which would pick between them) are not evaluated -- see
        // the header.
        std::unordered_map<std::uint32_t, std::uint32_t> firstInfoForTopic;
        for (const RawInfo& info : infos) {
            firstInfoForTopic.emplace(info.topicFormId, info.formId);
        }

        odai::dialogue::DialogueTree tree;
        for (const RawInfo& info : infos) {
            odai::dialogue::DialogueNode node;
            node.id = nodeIdFor(info.formId);
            node.speaker = speakerName;
            node.text = info.text;
            for (const std::uint32_t topic : info.linkedTopics) {
                ++speakerStats.choiceLinks;
                const auto targetIt = firstInfoForTopic.find(topic);
                if (targetIt == firstInfoForTopic.end()) {
                    // The topic exists but this speaker has nothing to say under
                    // it -- usually because the line that would answer is gated
                    // behind a quest condition we cannot evaluate. Dropped
                    // rather than offered as a choice that would dead-end.
                    ++speakerStats.danglingLinks;
                    continue;
                }
                odai::dialogue::DialogueChoice choice;
                const auto textIt = topicPlayerText.find(topic);
                choice.text = (textIt != topicPlayerText.end() && !textIt->second.empty())
                    ? textIt->second
                    : std::string("...");
                choice.targetNode = nodeIdFor(targetIt->second);
                node.choices.push_back(std::move(choice));
            }
            tree.nodes.emplace(node.id, std::move(node));
        }

        // Open on a GREETING, which is what the game does.
        //
        // Matched on the topic's EDITOR ID. This used to compare the topic's
        // FULL against "greeting", which a greeting never has: FULL is the line
        // the PLAYER picks, and nobody picks a greeting -- the game opens on it.
        // So the test never fired for anyone and every speaker fell through to
        // "first INFO seen". On a companion with 209 topics that meant Willow
        // opened on a line addressed to another NPC entirely.
        for (const RawInfo& info : infos) {
            const auto edidIt = topicEditorId.find(info.topicFormId);
            if (edidIt != topicEditorId.end() && toLowerAsciiCopy(edidIt->second) == "greeting") {
                tree.startNode = nodeIdFor(info.formId);
                break;
            }
        }
        // No greeting: prefer a real TOPIC over a bark. Types 1-6 are lines the
        // game fires at combat, detection or idle chatter -- "Attack",
        // "IdleChatter", "HELLO" -- and opening a conversation on one is how a
        // companion greets you by shouting mid-fight dialogue.
        if (tree.startNode.empty()) {
            for (const RawInfo& info : infos) {
                const auto typeIt = topicType.find(info.topicFormId);
                if (typeIt != topicType.end() && typeIt->second == 0u) {
                    tree.startNode = nodeIdFor(info.formId);
                    break;
                }
            }
        }
        // Still nothing: a speaker whose every line is a bark still yields a
        // usable tree rather than none.
        if (tree.startNode.empty()) {
            tree.startNode = nodeIdFor(infos.front().formId);
        }
        // ODAI_FNV_DIALOGUE_DEBUG names the topics a speaker's lines hang under.
        // "which topic did the entry line come from" is otherwise unanswerable
        // from outside, and it is the only thing that distinguishes a greeting
        // that was not found from one that does not exist.
        if (std::getenv("ODAI_FNV_DIALOGUE_DEBUG") != nullptr) {
            std::map<std::string, std::size_t> byTopic;
            for (const RawInfo& info : infos) {
                const auto edidIt = topicEditorId.find(info.topicFormId);
                char key[32] = {};
                std::snprintf(key, sizeof(key), "0x%08X", info.topicFormId);
                byTopic[edidIt != topicEditorId.end() ? edidIt->second : std::string(key)] += 1u;
            }
            std::cerr << "[fnv] dialogue " << speakerName << ": " << infos.size()
                      << " lines over " << byTopic.size() << " topics; start=" << tree.startNode
                      << "\n";
            std::size_t shown = 0;
            for (const auto& [topic, count] : byTopic) {
                if (shown++ >= 8u) {
                    break;
                }
                std::cerr << "[fnv]     " << topic << " x" << count << "\n";
            }
        }
        tree.id = speakerName;
        outTrees.emplace(speakerFormId, std::move(tree));
    }
    outStats = std::move(stats);
    return true;
}

bool buildSpeakerDialogueTrees(
    const std::filesystem::path& pluginPath,
    const std::vector<SpeakerDialogueRequest>& speakers,
    std::unordered_map<std::uint32_t, odai::dialogue::DialogueTree>& outTrees,
    std::unordered_map<std::uint32_t, DialogueImportStats>& outStats,
    std::string& outError
) {
    return buildSpeakerDialogueTreesImpl(
        pluginPath, nullptr, speakers, outTrees, outStats, outError);
}

bool buildSpeakerDialogueTreesAcrossOrder(
    const FalloutLoadOrder& order,
    const std::vector<SpeakerDialogueRequest>& speakers,
    std::unordered_map<std::uint32_t, odai::dialogue::DialogueTree>& outTrees,
    std::unordered_map<std::uint32_t, DialogueImportStats>& outStats,
    std::string& outError
) {
    return buildSpeakerDialogueTreesImpl(
        order.empty() ? std::filesystem::path{} : order.entries().front().path, &order, speakers,
        outTrees, outStats, outError);
}

}  // namespace odai::importer::fnv
