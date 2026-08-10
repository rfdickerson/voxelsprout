#include "import/fnv/dialogue_records.h"

#include "import/fnv/esm_reader.h"

#include <algorithm>
#include <cctype>
#include <cstring>
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

    // Pass 2: topics, and the responses under them that belong to this speaker.
    //
    // INFO records stream immediately after the DIAL that owns them (they live
    // in its child group), so tracking the most recent DIAL is enough to know
    // each response's topic -- no group bookkeeping required.
    std::unordered_map<std::uint32_t, std::string> topicPlayerText;
    std::vector<RawInfo> infos;
    std::uint32_t currentTopic = 0;
    {
        EsmReader::Visitor visitor;
        visitor.onRecord = [&](const EsmRecordView& record) {
            if (record.type == "DIAL") {
                currentTopic = record.formId;
                ++outStats.topicsSeen;
                for (const auto& sub : record.subrecords) {
                    if (sub.type == "FULL") {
                        topicPlayerText[record.formId] = subrecordString(sub);
                    }
                }
                return;
            }
            if (record.type != "INFO") {
                return;
            }
            bool isSpeaker = false;
            for (const auto& sub : record.subrecords) {
                if (sub.type != "CTDA" || sub.size < 28u) {
                    continue;
                }
                // CTDA (FO3/FNV, 28 bytes): type u8 @0, 3 unused, comparison
                // f32 @4, function index u32 @8, param1 u32 @12, param2, runOn,
                // reference.
                //
                // The operator in the type byte's top 3 bits has to be read,
                // not just the function and its parameter. Fallout writes
                // "everyone EXCEPT this actor" as GetIsID <actor> == 0, which
                // is the same function and the same parameter as the line that
                // belongs to him -- only the comparison differs. Ignoring it
                // attributed other characters' lines to the speaker.
                if (readU32(sub.data + 8) != kCtdaFunctionGetIsId ||
                    readU32(sub.data + 12) != speakerFormId) {
                    continue;
                }
                float comparisonValue = 0.0f;
                std::memcpy(&comparisonValue, sub.data + 4, sizeof(comparisonValue));
                const std::uint8_t comparisonOperator =
                    static_cast<std::uint8_t>((sub.data[0] >> 5) & 0x7u);
                // 0 = EQUAL, 2 = GREATER, 3 = GREATER-OR-EQUAL: all read as
                // "is this actor" when tested against 1.
                const bool positive =
                    (comparisonOperator == 0u && comparisonValue == 1.0f) ||
                    (comparisonOperator == 2u && comparisonValue == 0.0f) ||
                    (comparisonOperator == 3u && comparisonValue == 1.0f);
                if (positive) {
                    isSpeaker = true;
                }
            }
            if (!isSpeaker) {
                return;
            }
            RawInfo info;
            info.formId = record.formId;
            info.topicFormId = currentTopic;
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
                    ++outStats.responsesConcatenated;
                } else if (sub.type == "TCLT" && sub.size >= 4u) {
                    info.linkedTopics.push_back(readU32(sub.data));
                }
            }
            if (info.text.empty()) {
                return;  // a silent INFO (script-only) is not a line to show
            }
            infos.push_back(std::move(info));
            ++outStats.infosForSpeaker;
        };
        if (!reader.walk(visitor)) {
            outError = "plugin walk failed while reading dialogue";
            return false;
        }
    }
    if (infos.empty()) {
        outError = "no dialogue responses found for " + speakerName;
        return false;
    }

    // Topic -> the speaker's first response under it. First rather than a list
    // because choosing a topic has to land somewhere definite and conditions
    // (which would pick between them) are not evaluated -- see the header.
    std::unordered_map<std::uint32_t, std::uint32_t> firstInfoForTopic;
    for (const RawInfo& info : infos) {
        firstInfoForTopic.emplace(info.topicFormId, info.formId);
    }

    for (const RawInfo& info : infos) {
        odai::dialogue::DialogueNode node;
        node.id = nodeIdFor(info.formId);
        node.speaker = speakerName;
        node.text = info.text;
        for (const std::uint32_t topic : info.linkedTopics) {
            ++outStats.choiceLinks;
            const auto targetIt = firstInfoForTopic.find(topic);
            if (targetIt == firstInfoForTopic.end()) {
                // The topic exists but this speaker has nothing to say under
                // it -- usually because the line that would answer is gated
                // behind a quest condition we cannot evaluate. Dropped rather
                // than offered as a choice that would dead-end.
                ++outStats.danglingLinks;
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
        outTree.nodes.emplace(node.id, std::move(node));
    }

    // Open on a GREETING, which is what the game does. Falls back to any node
    // so a speaker whose greeting is gated still yields a usable tree.
    for (const RawInfo& info : infos) {
        const auto textIt = topicPlayerText.find(info.topicFormId);
        if (textIt != topicPlayerText.end() && toLowerAsciiCopy(textIt->second) == "greeting") {
            outTree.startNode = nodeIdFor(info.formId);
            break;
        }
    }
    if (outTree.startNode.empty()) {
        outTree.startNode = nodeIdFor(infos.front().formId);
    }
    outTree.id = speakerName;
    return true;
}

}  // namespace odai::importer::fnv
