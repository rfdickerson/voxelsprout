#include "import/fnv/actor_records.h"

#include "import/fnv/esm_reader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string_view>

namespace odai::importer::fnv {

namespace {

std::uint32_t readU32(const EsmSubrecordView& sub, std::size_t offset = 0) {
    std::uint32_t value = 0;
    if (sub.size >= offset + 4u) {
        std::memcpy(&value, sub.data + offset, 4u);
    }
    return value;
}

std::string subrecordText(const EsmSubrecordView& sub) {
    std::string out(reinterpret_cast<const char*>(sub.data), static_cast<std::size_t>(sub.size));
    while (!out.empty() && out.back() == '\0') {
        out.pop_back();
    }
    return out;
}

// NIFZ is a NUL-separated list of filenames relative to the skeleton's own
// directory -- not full paths.
void appendNifzParts(const EsmSubrecordView& sub, std::vector<std::string>& out) {
    const std::string blob(
        reinterpret_cast<const char*>(sub.data), static_cast<std::size_t>(sub.size));
    std::size_t begin = 0;
    while (begin < blob.size()) {
        const std::size_t end = blob.find('\0', begin);
        std::string part =
            blob.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!part.empty()) {
            out.push_back(std::move(part));
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
}

std::string toLowerAscii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string directoryOf(const std::string& path) {
    const std::size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

bool endsWith(const std::string& value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        toLowerAscii(value.substr(value.size() - suffix.size())) == suffix;
}

// NIFZ names are relative to the skeleton's own directory; everything else in
// the format stores a full path. resolve() hands out full paths only.
std::vector<std::string> fullPartPaths(
    const std::string& skeletonPath, const std::vector<std::string>& relativeNames
) {
    const std::string directory = directoryOf(skeletonPath);
    std::vector<std::string> out;
    out.reserve(relativeNames.size());
    for (const std::string& name : relativeNames) {
        out.push_back(directory + name);
    }
    return out;
}

// RACE's part models are a POSITIONAL format, not a keyed one: NAM0 opens the
// head section and NAM1 the body section, MNAM and FNAM switch sex inside
// whichever is open, and each INDX names the slot the MODL after it fills.
// Reading MODL alone gets you the last model in the record for everybody.
//
// The same four subrecord types appear again AFTER the section list -- MNAM and
// FNAM reappear to introduce FaceGen data (FGGS/FGGA/FGTS), where an INDX means
// something else entirely. HNAM (hair list) is the first subrecord past the
// parts, so it is where the state machine stops.
FalloutRaceParts parseRaceParts(const EsmRecordView& record) {
    FalloutRaceParts race;
    race.formId = record.formId;

    enum class Section : std::uint8_t { None, Head, Body };
    Section section = Section::None;
    bool female = false;
    std::size_t slot = 0;
    bool done = false;

    for (const EsmSubrecordView& sub : record.subrecords) {
        if (sub.type == "EDID") {
            race.editorId = subrecordText(sub);
            continue;
        }
        if (sub.type == "VTCK" && sub.size >= 8u) {
            race.maleVoiceTypeFormId = readU32(sub, 0);
            race.femaleVoiceTypeFormId = readU32(sub, 4);
            continue;
        }
        if (done) {
            continue;
        }
        if (sub.type == "NAM0") {
            section = Section::Head;
            female = false;
        } else if (sub.type == "NAM1") {
            section = Section::Body;
            female = false;
        } else if (sub.type == "HNAM" || sub.type == "ENAM") {
            done = true;
        } else if (sub.type == "MNAM") {
            female = false;
        } else if (sub.type == "FNAM") {
            female = true;
        } else if (sub.type == "INDX" && sub.size >= 4u) {
            slot = static_cast<std::size_t>(readU32(sub));
        } else if (sub.type == "MODL") {
            const std::string model = subrecordText(sub);
            // The last body slot holds a FaceGen texture, not a mesh.
            if (model.empty() || endsWith(model, ".egt")) {
                continue;
            }
            if (section == Section::Head && slot < kRaceHeadPartCount) {
                (female ? race.femaleHeadModels : race.maleHeadModels)[slot] = model;
            } else if (section == Section::Body && slot < kRaceBodyPartCount) {
                (female ? race.femaleBodyModels : race.maleBodyModels)[slot] = model;
            }
        }
    }
    return race;
}

void appendUnique(std::vector<std::string>& out, const std::string& path) {
    if (path.empty()) {
        return;
    }
    // An outfit that covers upper body AND both hands is ONE model claiming
    // three slots -- appending it per slot draws the same body three times.
    const std::string key = toLowerAscii(path);
    for (const std::string& existing : out) {
        if (toLowerAscii(existing) == key) {
            return;
        }
    }
    out.push_back(path);
}

}  // namespace

ResolvedActorBase FalloutActorScan::resolve(std::uint32_t baseFormId) const {
    ResolvedActorBase resolved;
    const auto found = bases.find(baseFormId);
    if (found == bases.end()) {
        return resolved;
    }
    resolved.base = &found->second;

    // Own geometry wins outright.
    if (!found->second.bodyPartPaths.empty() && !found->second.skeletonPath.empty()) {
        resolved.geometrySource = ActorGeometrySource::OwnBodyParts;
        resolved.resolvedBaseFormId = baseFormId;
        resolved.skeletonPath = found->second.skeletonPath;
        resolved.bodyPartPaths =
            fullPartPaths(found->second.skeletonPath, found->second.bodyPartPaths);
        return resolved;
    }

    // Otherwise follow TPLT. Bounded and cycle-guarded: a template chain is
    // author data and nothing in the format forbids it pointing at itself.
    std::uint32_t current = found->second.templateFormId;
    for (int hop = 0; hop < 8 && current != 0u && current != baseFormId; ++hop) {
        // A levelled list is the usual first hop. Take the first entry that
        // carries geometry: which one the game would roll depends on player
        // level and a die, and any of them is a truthful answer to "what does
        // this spawn point put here".
        const auto list = leveledLists.find(current);
        if (list != leveledLists.end()) {
            for (const std::uint32_t candidate : list->second) {
                const auto entry = bases.find(candidate);
                if (entry == bases.end() || entry->second.bodyPartPaths.empty() ||
                    entry->second.skeletonPath.empty()) {
                    continue;
                }
                resolved.geometrySource = ActorGeometrySource::Template;
                resolved.resolvedBaseFormId = candidate;
                resolved.skeletonPath = entry->second.skeletonPath;
                resolved.bodyPartPaths =
                    fullPartPaths(entry->second.skeletonPath, entry->second.bodyPartPaths);
                return resolved;
            }
            break;
        }
        const auto step = bases.find(current);
        if (step == bases.end()) {
            break;
        }
        if (!step->second.bodyPartPaths.empty() && !step->second.skeletonPath.empty()) {
            resolved.geometrySource = ActorGeometrySource::Template;
            resolved.resolvedBaseFormId = current;
            resolved.skeletonPath = step->second.skeletonPath;
            resolved.bodyPartPaths =
                fullPartPaths(step->second.skeletonPath, step->second.bodyPartPaths);
            return resolved;
        }
        current = step->second.templateFormId;
    }

    // An NPC_ (or a creature whose chain ran dry) with a race to build from.
    // Its race, its sex and its wardrobe are each individually inheritable from
    // its TPLT, so each is looked up rather than read straight off the record.
    // Both are non-null here -- inheritedFrom returns null only for a formID
    // this scan has no base for, and `found` already proved otherwise -- but a
    // fallback to the record itself keeps that reasoning local.
    const FalloutActorBase* traitsSource = inheritedFrom(baseFormId, kActorTemplateUseTraits);
    const FalloutActorBase* wardrobeSource =
        inheritedFrom(baseFormId, kActorTemplateUseInventory);
    const FalloutActorBase& traits = traitsSource != nullptr ? *traitsSource : found->second;
    const FalloutActorBase& wardrobe =
        wardrobeSource != nullptr ? *wardrobeSource : found->second;
    // The skeleton is a templated field too, and taking the record's own MODL
    // when it does not own one is how an actor ends up bound to
    // marker_creature.nif with every bone unresolved.
    const FalloutActorBase* modelSource =
        inheritedFrom(baseFormId, kActorTemplateUseModelAnimation);
    const std::string& skeletonPath =
        modelSource != nullptr ? modelSource->skeletonPath : found->second.skeletonPath;

    const auto race = races.find(traits.raceFormId);
    if (race == races.end() || skeletonPath.empty()) {
        return resolved;
    }

    resolved.geometrySource = ActorGeometrySource::Race;
    resolved.resolvedBaseFormId = baseFormId;
    resolved.skeletonPath = skeletonPath;

    const bool female = traits.isFemale;
    const auto* headModels = female ? race->second.femaleHeadModels : race->second.maleHeadModels;
    const auto* bodyModels = female ? race->second.femaleBodyModels : race->second.maleBodyModels;

    // The four slots a body is assembled from, seeded with bare skin. Worn
    // armour overwrites a slot rather than being added to it: an outfit's NIF
    // already contains the body underneath it, so drawing both puts the race's
    // underwear through the clothes.
    std::string head = headModels[kRaceHeadSlot];
    std::string upperBody = bodyModels[kRaceUpperBodySlot];
    std::string leftHand = bodyModels[kRaceLeftHandSlot];
    std::string rightHand = bodyModels[kRaceRightHandSlot];
    bool upperBodyTaken = false;
    bool leftHandTaken = false;
    bool rightHandTaken = false;
    bool headTaken = false;
    bool hatTaken = false;
    // Additive pieces -- a hat sits ON a head rather than replacing it.
    std::vector<std::string> accessories;

    // Levelled lists expand in place, so a settler's "OutfitSettlerFemale"
    // becomes the outfits it can hand out and the loop below sees armour.
    std::vector<std::uint32_t> wardrobeItems = wardrobe.inventoryFormIds;
    for (std::size_t i = 0; i < wardrobeItems.size() && wardrobeItems.size() < 256u; ++i) {
        const auto list = leveledItems.find(wardrobeItems[i]);
        if (list == leveledItems.end()) {
            continue;
        }
        // Which entry the game would roll depends on player level and a die.
        // Any of them is a truthful answer to "what is this actor wearing",
        // and the first that resolves to armour is the one that gets worn.
        wardrobeItems.insert(wardrobeItems.end(), list->second.begin(), list->second.end());
    }

    for (const std::uint32_t itemFormId : wardrobeItems) {
        const auto armor = armors.find(itemFormId);
        if (armor == armors.end()) {
            // Not armour: a weapon, ammo, caps, or a list already expanded. An
            // actor's inventory is mostly this.
            continue;
        }
        const std::string& model = female && !armor->second.femaleModel.empty()
            ? armor->second.femaleModel
            : armor->second.maleModel;
        if (model.empty()) {
            continue;
        }
        // First claim on a slot wins, and the actor's own CNTO entries are ahead
        // of anything a list expanded into -- so an explicitly carried outfit
        // beats one the game would have rolled for.
        const std::uint32_t slots = armor->second.bipedFlags;
        bool worn = false;
        const auto claim = [&](std::uint32_t slot, bool& taken, std::string& target) {
            if ((slots & slot) == 0u || taken) {
                return;
            }
            target = model;
            taken = true;
            worn = true;
        };
        claim(kBipedSlotUpperBody, upperBodyTaken, upperBody);
        claim(kBipedSlotLeftHand, leftHandTaken, leftHand);
        claim(kBipedSlotRightHand, rightHandTaken, rightHand);
        claim(kBipedSlotHead, headTaken, head);
        if ((slots & (kBipedSlotHat | kBipedSlotHair)) != 0u && (slots & kBipedSlotHead) == 0u &&
            !hatTaken) {
            accessories.push_back(model);
            hatTaken = true;
            worn = true;
        }
        if (worn) {
            resolved.wornArmorFormIds.push_back(itemFormId);
        }
    }

    appendUnique(resolved.bodyPartPaths, upperBody);
    appendUnique(resolved.bodyPartPaths, leftHand);
    appendUnique(resolved.bodyPartPaths, rightHand);
    appendUnique(resolved.bodyPartPaths, head);
    for (const std::string& accessory : accessories) {
        appendUnique(resolved.bodyPartPaths, accessory);
    }
    return resolved;
}

std::uint32_t FalloutActorScan::voiceTypeFormIdFor(std::uint32_t baseFormId) const {
    const FalloutActorBase* traits = inheritedFrom(baseFormId, kActorTemplateUseTraits);
    if (traits == nullptr) {
        return 0u;
    }
    // Its own, but only if the table actually knows it -- a VTCK pointing at a
    // VTYP this scan never read would resolve to an empty folder name while
    // looking resolved.
    if (traits->voiceTypeFormId != 0u &&
        voiceTypes.find(traits->voiceTypeFormId) != voiceTypes.end()) {
        return traits->voiceTypeFormId;
    }
    const auto race = races.find(traits->raceFormId);
    if (race == races.end()) {
        return 0u;
    }
    return traits->isFemale ? race->second.femaleVoiceTypeFormId
                            : race->second.maleVoiceTypeFormId;
}

std::string FalloutActorScan::voiceFolderFor(std::uint32_t baseFormId) const {
    const auto found = voiceTypes.find(voiceTypeFormIdFor(baseFormId));
    return found != voiceTypes.end() ? found->second : std::string();
}

const FalloutActorBase* FalloutActorScan::firstActorFrom(
    std::uint32_t formId, std::uint32_t excludeFormId
) const {
    // A template hop is usually NOT to another actor: it lands on a levelled
    // list (LVLC for creatures, LVLN for NPCs) whose entries are the real ones.
    // And those entries are routinely MORE LISTS -- Fallout 3's EncRaiderMelee
    // is a list of lists -- so following one level finds nothing and hands back
    // the marker record, which is a skeleton with none of the bones a body is
    // weighted to.
    //
    // Breadth-first so the shallowest real actor wins, and bounded because
    // nothing in the format forbids a list containing itself.
    std::vector<std::uint32_t> queue{formId};
    std::vector<std::uint32_t> seen;
    for (std::size_t i = 0; i < queue.size() && queue.size() < 256u; ++i) {
        const std::uint32_t candidate = queue[i];
        if (candidate == 0u || candidate == excludeFormId ||
            std::find(seen.begin(), seen.end(), candidate) != seen.end()) {
            continue;
        }
        seen.push_back(candidate);
        const auto actor = bases.find(candidate);
        if (actor != bases.end()) {
            return &actor->second;
        }
        const auto list = leveledLists.find(candidate);
        if (list != leveledLists.end()) {
            queue.insert(queue.end(), list->second.begin(), list->second.end());
        }
    }
    return nullptr;
}

const FalloutActorBase* FalloutActorScan::inheritedFrom(
    std::uint32_t baseFormId, std::uint16_t templateUseFlag
) const {
    const auto found = bases.find(baseFormId);
    if (found == bases.end()) {
        return nullptr;
    }
    const FalloutActorBase* current = &found->second;
    // Bounded and cycle-guarded for the same reason the geometry walk above is:
    // a template chain is author data.
    for (int hop = 0; hop < 8; ++hop) {
        if ((current->templateFlags & templateUseFlag) == 0u || current->templateFormId == 0u) {
            return current;
        }
        const FalloutActorBase* next =
            firstActorFrom(current->templateFormId, current->formId);
        if (next == nullptr) {
            return current;
        }
        current = next;
    }
    return current;
}

bool findActorsNear(
    const std::filesystem::path& pluginPath,
    float centreX,
    float centreY,
    float radius,
    FalloutActorScan& outScan,
    std::string& outError
) {
    outScan = FalloutActorScan{};
    outError.clear();

    EsmReader reader;
    if (!reader.open(pluginPath)) {
        outError = "cannot open plugin: " + reader.lastError();
        return false;
    }

    // Pass 1: every actor base, plus the races and armour they are assembled
    // from. Collected wholesale rather than on demand because a placement names
    // its base by formID, and resolving those one at a time would be one plugin
    // walk each.
    {
        EsmReader::Visitor visitor;
        visitor.onRecordHeader = [](const EsmRecordHeaderView& header) {
            return header.type == "CREA" || header.type == "NPC_" || header.type == "LVLC" ||
                header.type == "LVLN" || header.type == "LVLI" || header.type == "RACE" ||
                header.type == "ARMO" || header.type == "VTYP";
        };
        visitor.onRecord = [&](const EsmRecordView& record) {
            if (record.type == "VTYP") {
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "EDID") {
                        outScan.voiceTypes[record.formId] = subrecordText(sub);
                    }
                }
                return;
            }
            if (record.type == "RACE") {
                outScan.races[record.formId] = parseRaceParts(record);
                return;
            }
            if (record.type == "ARMO") {
                FalloutArmorPiece armor;
                armor.formId = record.formId;
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "EDID") {
                        armor.editorId = subrecordText(sub);
                    } else if (sub.type == "BMDT" && sub.size >= 4u) {
                        armor.bipedFlags = readU32(sub);
                    } else if (sub.type == "MODL") {
                        armor.maleModel = subrecordText(sub);
                    } else if (sub.type == "MOD3") {
                        armor.femaleModel = subrecordText(sub);
                    }
                }
                if (armor.bipedFlags != 0u &&
                    (!armor.maleModel.empty() || !armor.femaleModel.empty())) {
                    outScan.armors[record.formId] = std::move(armor);
                }
                return;
            }
            if (record.type == "LVLC" || record.type == "LVLN" || record.type == "LVLI") {
                // LVLO is a 12-byte entry; the referenced formID sits at offset
                // 4, after a u16 level and two unused bytes.
                std::vector<std::uint32_t> entries;
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "LVLO" && sub.size >= 8u) {
                        entries.push_back(readU32(sub, 4));
                    }
                }
                if (!entries.empty()) {
                    (record.type == "LVLI" ? outScan.leveledItems
                                           : outScan.leveledLists)[record.formId] =
                        std::move(entries);
                }
                return;
            }
            FalloutActorBase base;
            base.formId = record.formId;
            base.recordType = std::string(record.type);
            for (const EsmSubrecordView& sub : record.subrecords) {
                if (sub.type == "EDID") {
                    base.editorId = subrecordText(sub);
                } else if (sub.type == "FULL") {
                    base.fullName = subrecordText(sub);
                } else if (sub.type == "MODL") {
                    base.skeletonPath = subrecordText(sub);
                } else if (sub.type == "NIFZ" && sub.size != 0u) {
                    appendNifzParts(sub, base.bodyPartPaths);
                } else if (sub.type == "TPLT") {
                    base.templateFormId = readU32(sub);
                } else if (sub.type == "RNAM") {
                    base.raceFormId = readU32(sub);
                } else if (sub.type == "VTCK" && sub.size >= 4u) {
                    base.voiceTypeFormId = readU32(sub);
                } else if (sub.type == "CNTO" && sub.size >= 4u) {
                    base.inventoryFormIds.push_back(readU32(sub));
                } else if (sub.type == "ACBS" && sub.size >= 4u) {
                    // Flags bit 0 is Female. Which of the RACE's two part sets
                    // an NPC_ uses hangs on it.
                    base.isFemale = (readU32(sub) & 0x00000001u) != 0u;
                    // The template-use flags are the record's trailing u16.
                    if (sub.size >= 24u) {
                        std::memcpy(&base.templateFlags, sub.data + 22, sizeof(base.templateFlags));
                    }
                }
            }
            outScan.bases[record.formId] = std::move(base);
        };
        if (!reader.walk(visitor)) {
            outError = "actor base scan failed: " + reader.lastError();
            return false;
        }
    }

    // Pass 2: placements inside the radius.
    {
        EsmReader::Visitor visitor;
        visitor.onRecordHeader = [](const EsmRecordHeaderView& header) {
            return header.type == "ACRE" || header.type == "ACHR";
        };
        visitor.onRecord = [&](const EsmRecordView& record) {
            FalloutActorPlacement placement;
            placement.refFormId = record.formId;
            placement.initiallyDisabled = (record.flags & 0x00000800u) != 0u;
            bool hasPosition = false;
            for (const EsmSubrecordView& sub : record.subrecords) {
                if (sub.type == "NAME") {
                    placement.baseFormId = readU32(sub);
                } else if (sub.type == "DATA" && sub.size >= 24u) {
                    std::memcpy(placement.position, sub.data, sizeof(placement.position));
                    std::memcpy(
                        placement.rotationRadians, sub.data + 12, sizeof(placement.rotationRadians));
                    hasPosition = true;
                }
            }
            if (!hasPosition || placement.baseFormId == 0u) {
                return;
            }
            const float dx = placement.position[0] - centreX;
            const float dy = placement.position[1] - centreY;
            if (((dx * dx) + (dy * dy)) > (radius * radius)) {
                return;
            }
            outScan.placements.push_back(placement);
        };
        if (!reader.walk(visitor)) {
            outError = "actor placement scan failed: " + reader.lastError();
            return false;
        }
    }

    // Nearest first, so a caller that can only afford N actors spends them on
    // the ones the player is standing among.
    std::sort(
        outScan.placements.begin(), outScan.placements.end(),
        [centreX, centreY](const FalloutActorPlacement& a, const FalloutActorPlacement& b) {
            const float ax = a.position[0] - centreX;
            const float ay = a.position[1] - centreY;
            const float bx = b.position[0] - centreX;
            const float by = b.position[1] - centreY;
            return ((ax * ax) + (ay * ay)) < ((bx * bx) + (by * by));
        });
    return true;
}

}  // namespace odai::importer::fnv
