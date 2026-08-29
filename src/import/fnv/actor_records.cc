#include "import/fnv/actor_records.h"

#include "import/fnv/esm_reader.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cctype>
#include <cmath>
#include <cstring>
#include <sstream>
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
FalloutRaceParts parseRaceParts(
    const EsmRecordView& record, EsmPluginFormat pluginFormat
) {
    FalloutRaceParts race;
    race.formId = record.formId;
    race.usesOblivionBodyLayout = pluginFormat == EsmPluginFormat::kOblivion;

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
        if (sub.type == "WNAM" && sub.size >= 4u) {
            race.defaultSkinFormId = readU32(sub);
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
        } else if (sub.type == "ANAM") {
            // Skyrim: the first MNAM/ANAM and FNAM/ANAM pairs name the two
            // skeleton NIFs. Fallout's race layout has no ANAM model path, so
            // accepting only .nif keeps this an additive generation branch.
            const std::string model = subrecordText(sub);
            if (endsWith(model, ".nif")) {
                (female ? race.femaleSkeletonPath : race.maleSkeletonPath) = model;
            }
        } else if (sub.type == "INDX" && sub.size >= 4u) {
            slot = static_cast<std::size_t>(readU32(sub));
        } else if (sub.type == "MODL") {
            const std::string model = subrecordText(sub);
            // The last body slot holds a FaceGen texture, not a mesh.
            if (model.empty() || !endsWith(model, ".nif")) {
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
    const auto race = races.find(traits.raceFormId);
    if (race == races.end()) {
        return resolved;
    }

    // A RACE-ASSEMBLED ACTOR'S SKELETON LIVES BESIDE ITS RACE'S BODY PARTS, not
    // in its own MODL.
    //
    // MODL looks like the answer and mostly reads like one -- every New Vegas
    // NPC_ stores "Characters\_Male\Skeleton.NIF" there -- but that is
    // convention, not the source of truth. Fallout 3 is full of records storing
    // "marker_creature.nif" instead, and not only the ones whose template flags
    // say they borrow a model: Stockholm's flags claim he owns his (0x0002,
    // stats only) while his MODL is a marker. A marker is a real, parseable NIF
    // carrying none of the bones a body is weighted to, so using it does not
    // fail -- it binds a character with every bone unresolved. Measured: 85 of
    // them on Stockholm, 91 on a Springvale raider.
    //
    // Deriving it from the race's own body-part directory is the same
    // convention creatures already rely on for mtidle.kf, and it agrees with
    // MODL wherever MODL was right: characters\_Male\UpperBody.nif and
    // characters\_Male\Skeleton.NIF are the same directory.
    const auto* bodyModels =
        traits.isFemale ? race->second.femaleBodyModels : race->second.maleBodyModels;
    std::string skeletonPath;
    // Skyrim's RACE states the answer directly. This also distinguishes its
    // actor assembly from Fallout's: TES5 has no naked-body NIF list on RACE;
    // the body arrives through the default outfit's ARMO -> ARMA chain.
    const std::string& skyrimSkeleton =
        traits.isFemale ? race->second.femaleSkeletonPath : race->second.maleSkeletonPath;
    if (!skyrimSkeleton.empty()) {
        skeletonPath = skyrimSkeleton;
    } else if (!bodyModels[kRaceUpperBodySlot].empty()) {
        skeletonPath = directoryOf(bodyModels[kRaceUpperBodySlot]) + "skeleton.nif";
    } else {
        // No body to stand beside: fall back to what the record claims, through
        // any template that owns the model on its behalf.
        const FalloutActorBase* modelSource =
            inheritedFrom(baseFormId, kActorTemplateUseModelAnimation);
        skeletonPath = modelSource != nullptr ? modelSource->skeletonPath : found->second.skeletonPath;
    }
    if (skeletonPath.empty()) {
        return resolved;
    }

    resolved.geometrySource = ActorGeometrySource::Race;
    resolved.resolvedBaseFormId = baseFormId;
    resolved.skeletonPath = skeletonPath;

    const bool female = traits.isFemale;
    const auto* headModels = female ? race->second.femaleHeadModels : race->second.maleHeadModels;

    // The four slots a body is assembled from, seeded with bare skin. Worn
    // armour overwrites a slot rather than being added to it: an outfit's NIF
    // already contains the body underneath it, so drawing both puts the race's
    // underwear through the clothes.
    std::string head = headModels[kRaceHeadSlot];
    std::string upperBody = bodyModels[kRaceUpperBodySlot];
    std::string leftHand = bodyModels[kRaceLeftHandSlot];
    std::string rightHand = bodyModels[kRaceRightHandSlot];
    // The arrays are positional in both generations but the positions do not
    // mean the same thing. In TES4 these are lower body, one two-handed mesh,
    // and feet. Preserve Fallout's left/right interpretation everywhere else.
    const bool oblivionBodyLayout = race->second.usesOblivionBodyLayout;
    std::string lowerBody = oblivionBodyLayout ? bodyModels[1] : std::string();
    std::string hands = oblivionBodyLayout ? bodyModels[2] : std::string();
    std::string feet = oblivionBodyLayout ? bodyModels[3] : std::string();
    bool upperBodyTaken = false;
    bool leftHandTaken = false;
    bool rightHandTaken = false;
    bool lowerBodyTaken = false;
    bool handsTaken = false;
    bool feetTaken = false;
    bool headTaken = false;
    bool hatTaken = false;
    // Additive pieces -- a hat sits ON a head rather than replacing it.
    std::vector<std::string> accessories;
    std::uint32_t skyrimCoveredSlots = 0u;

    // Levelled lists expand in place, so a settler's "OutfitSettlerFemale"
    // becomes the outfits it can hand out and the loop below sees armour.
    std::vector<std::uint32_t> wardrobeItems = wardrobe.inventoryFormIds;
    // TES5 DOFT points at an OTFT whose INAM entries are the worn set. Follow
    // the template chain opportunistically: Skyrim's ACBS template flags do
    // not share Fallout's trailing-u16 layout, but an empty DOFT on a derived
    // guard unambiguously means to keep walking until a template owns one.
    const FalloutActorBase* outfitSource = &wardrobe;
    for (int hop = 0; hop < 8 && outfitSource != nullptr &&
         outfitSource->defaultOutfitFormId == 0u && outfitSource->templateFormId != 0u; ++hop) {
        outfitSource = firstActorFrom(outfitSource->templateFormId, outfitSource->formId);
    }
    if (outfitSource != nullptr && outfitSource->defaultOutfitFormId != 0u) {
        const auto outfit = outfits.find(outfitSource->defaultOutfitFormId);
        if (outfit != outfits.end()) {
            wardrobeItems.insert(wardrobeItems.end(), outfit->second.begin(), outfit->second.end());
        }
    }
    for (std::size_t i = 0; i < wardrobeItems.size() && wardrobeItems.size() < 256u; ++i) {
        const auto list = leveledItems.find(wardrobeItems[i]);
        if (list == leveledItems.end()) {
            continue;
        }
        const auto useAll = leveledItemUseAll.find(wardrobeItems[i]);
        if (useAll != leveledItemUseAll.end() && useAll->second) {
            wardrobeItems.insert(wardrobeItems.end(), list->second.begin(), list->second.end());
        } else if (!list->second.empty()) {
            // Player level and a die pick the real entry. A stable first choice
            // keeps captures reproducible while still selecting exactly one of
            // the mutually-exclusive complete outfits.
            wardrobeItems.push_back(list->second.front());
        }
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
        // Skyrim's ARMO is a container. Each binary MODL is an ARMA reference,
        // and the third-person model is MOD2/MOD3 on that record. Append every
        // applicable addon: boots, cuirass, gloves and helmet are separate
        // skinned meshes and none can substitute for another.
        if (!armor->second.armatureFormIds.empty()) {
            for (const std::uint32_t addonId : armor->second.armatureFormIds) {
                const auto addon = armorAddons.find(addonId);
                if (addon == armorAddons.end()) {
                    continue;
                }
                if (!addon->second.raceFormIds.empty() &&
                    std::find(addon->second.raceFormIds.begin(), addon->second.raceFormIds.end(),
                              traits.raceFormId) == addon->second.raceFormIds.end()) {
                    continue;
                }
                const std::string& addonModel = female && !addon->second.femaleModel.empty()
                    ? addon->second.femaleModel
                    : addon->second.maleModel;
                appendUnique(resolved.bodyPartPaths, addonModel);
                skyrimCoveredSlots |= addon->second.bipedFlags;
            }
            resolved.wornArmorFormIds.push_back(itemFormId);
            continue;
        }
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
        if (oblivionBodyLayout) {
            claim(kOblivionBipedSlotLowerBody, lowerBodyTaken, lowerBody);
            claim(kOblivionBipedSlotHands, handsTaken, hands);
            claim(kOblivionBipedSlotFeet, feetTaken, feet);
        } else {
            claim(kBipedSlotLeftHand, leftHandTaken, leftHand);
            claim(kBipedSlotRightHand, rightHandTaken, rightHand);
            claim(kBipedSlotHead, headTaken, head);
        }
        // A TES4 helmet hides hair but is drawn around the race's FaceGen head;
        // it does not contain a replacement face. Treating the head bit as a
        // replacement made every helmeted Anvil guard headless.
        if (oblivionBodyLayout &&
            (slots & (kBipedSlotHead | kBipedSlotHair)) != 0u && !hatTaken) {
            accessories.push_back(model);
            hatTaken = true;
            worn = true;
        }
        if ((slots & (kBipedSlotHat | kBipedSlotHair)) != 0u && (slots & kBipedSlotHead) == 0u &&
            !hatTaken && !oblivionBodyLayout) {
            accessories.push_back(model);
            hatTaken = true;
            worn = true;
        }
        if (worn) {
            resolved.wornArmorFormIds.push_back(itemFormId);
        }
    }

    // Skyrim clothing contains only the skin it explicitly covers. The race's
    // WNAM skin ARMO supplies exposed hands, feet and (for sparse outfits) body
    // pieces. Add only ARMA records whose slots remain uncovered so a robe does
    // not receive a second naked torso underneath it.
    if (!skyrimSkeleton.empty() && race->second.defaultSkinFormId != 0u) {
        const auto skin = armors.find(race->second.defaultSkinFormId);
        if (skin != armors.end()) {
            for (const std::uint32_t addonId : skin->second.armatureFormIds) {
                const auto addon = armorAddons.find(addonId);
                if (addon == armorAddons.end()) continue;
                if (!addon->second.raceFormIds.empty() &&
                    std::find(addon->second.raceFormIds.begin(),
                        addon->second.raceFormIds.end(), traits.raceFormId) ==
                        addon->second.raceFormIds.end()) {
                    continue;
                }
                if (addon->second.bipedFlags != 0u &&
                    (addon->second.bipedFlags & ~skyrimCoveredSlots) == 0u) {
                    continue;
                }
                const std::string& model = female && !addon->second.femaleModel.empty()
                    ? addon->second.femaleModel
                    : addon->second.maleModel;
                appendUnique(resolved.bodyPartPaths, model);
            }
        }
    }

    appendUnique(resolved.bodyPartPaths, upperBody);
    if (oblivionBodyLayout) {
        appendUnique(resolved.bodyPartPaths, lowerBody);
        appendUnique(resolved.bodyPartPaths, hands);
        appendUnique(resolved.bodyPartPaths, feet);
        // TES4's head section is an additive set: face, ears, mouth, teeth,
        // tongue, and eyes. The first entry alone produces a blank mask.
        for (std::size_t slot = 0u; slot < kRaceHeadPartCount; ++slot) {
            appendUnique(resolved.bodyPartPaths, headModels[slot]);
        }
    } else {
        appendUnique(resolved.bodyPartPaths, leftHand);
        appendUnique(resolved.bodyPartPaths, rightHand);
        appendUnique(resolved.bodyPartPaths, head);
    }
    for (const std::string& accessory : accessories) {
        appendUnique(resolved.bodyPartPaths, accessory);
    }
    // Skyrim's RACE supplies no head NIF because the face is generated per
    // NPC. The CK persists that result under FaceGeom/<plugin>/<localForm>.nif
    // rather than putting a reference on NPC_. Append it only on the TES5
    // assembly branch: Fallout's race head model above remains authoritative.
    if (!skyrimSkeleton.empty()) {
        for (const std::string& faceGeometryPath : traits.faceGeometryPaths) {
            appendUnique(resolved.bodyPartPaths, faceGeometryPath);
        }
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

std::vector<std::uint32_t> FalloutActorScan::materializeInventory(
    std::uint32_t baseFormId,
    std::uint32_t seed) const {
    const FalloutActorBase* source =
        inheritedFrom(baseFormId, kActorTemplateUseInventory);
    if (source == nullptr) return {};
    std::vector<std::uint32_t> pending = source->inventoryFormIds;
    std::vector<std::uint32_t> resolved;
    std::uint32_t expansionCount = 0u;
    for (std::size_t cursor = 0u;
         cursor < pending.size() && cursor < 256u && resolved.size() < 256u;
         ++cursor) {
        const std::uint32_t token = pending[cursor];
        const auto list = leveledItems.find(token);
        if (list == leveledItems.end()) {
            if (token != 0u) resolved.push_back(token);
            continue;
        }
        ++expansionCount;
        if (list->second.empty()) continue;
        const auto useAll = leveledItemUseAll.find(token);
        if (useAll != leveledItemUseAll.end() && useAll->second) {
            if (pending.size() < 256u) {
                const std::size_t remaining = 256u - pending.size();
                pending.insert(pending.end(), list->second.begin(),
                    list->second.begin() + static_cast<std::ptrdiff_t>(
                        std::min(remaining, list->second.size())));
            }
        } else {
            std::uint32_t choice = seed ^ token ^
                (expansionCount * 0x9e3779b9u);
            choice ^= choice << 13u;
            choice ^= choice >> 17u;
            choice ^= choice << 5u;
            if (pending.size() < 256u) {
                pending.push_back(list->second[choice % list->second.size()]);
            }
        }
    }
    return resolved;
}

namespace {

// Rewrites every formID one plugin's scan produced into the load order's global
// space. Exhaustive on purpose, for the reason the cell merge states: an
// un-remapped ID does not fail, it addresses a different record -- the wrong
// race, the wrong armour, the wrong voice.
void remapActorScan(const FalloutLoadOrder& order, std::size_t pluginIndex,
                    FalloutActorScan& scan) {
    const auto remap = [&](std::uint32_t formId) {
        return formId == 0u ? 0u : order.remapFormId(pluginIndex, formId);
    };
    const auto remapKeyed = [&](auto& map, auto rewriteValue) {
        std::decay_t<decltype(map)> rebuilt;
        rebuilt.reserve(map.size());
        for (auto& [key, value] : map) {
            rewriteValue(value);
            rebuilt.emplace(remap(key), std::move(value));
        }
        map = std::move(rebuilt);
    };

    for (FalloutActorPlacement& placement : scan.placements) {
        placement.refFormId = remap(placement.refFormId);
        placement.baseFormId = remap(placement.baseFormId);
        for (std::uint32_t& referenceType : placement.referenceTypeFormIds) {
            referenceType = remap(referenceType);
        }
    }
    remapKeyed(scan.bases, [&](FalloutActorBase& base) {
        base.formId = remap(base.formId);
        base.templateFormId = remap(base.templateFormId);
        base.raceFormId = remap(base.raceFormId);
        base.voiceTypeFormId = remap(base.voiceTypeFormId);
        base.defaultOutfitFormId = remap(base.defaultOutfitFormId);
        for (std::uint32_t& item : base.inventoryFormIds) {
            item = remap(item);
        }
    });
    remapKeyed(scan.leveledLists, [&](std::vector<std::uint32_t>& entries) {
        for (std::uint32_t& entry : entries) {
            entry = remap(entry);
        }
    });
    remapKeyed(scan.leveledItems, [&](std::vector<std::uint32_t>& entries) {
        for (std::uint32_t& entry : entries) {
            entry = remap(entry);
        }
    });
    remapKeyed(scan.leveledItemUseAll, [](bool&) {});
    remapKeyed(scan.races, [&](FalloutRaceParts& race) {
        race.formId = remap(race.formId);
        race.maleVoiceTypeFormId = remap(race.maleVoiceTypeFormId);
        race.femaleVoiceTypeFormId = remap(race.femaleVoiceTypeFormId);
    });
    remapKeyed(scan.armors, [&](FalloutArmorPiece& armor) {
        armor.formId = remap(armor.formId);
        for (std::uint32_t& addon : armor.armatureFormIds) {
            addon = remap(addon);
        }
    });
    remapKeyed(scan.outfits, [&](std::vector<std::uint32_t>& entries) {
        for (std::uint32_t& entry : entries) {
            entry = remap(entry);
        }
    });
    remapKeyed(scan.outfitEditorIds, [](std::string&) {});
    remapKeyed(scan.armorAddons, [&](SkyrimArmorAddon& addon) {
        addon.formId = remap(addon.formId);
        for (std::uint32_t& race : addon.raceFormIds) {
            race = remap(race);
        }
    });
    remapKeyed(scan.voiceTypes, [](std::string&) {});
}

}  // namespace

bool findActorsNearAcrossOrder(
    const FalloutLoadOrder& order,
    float centreX,
    float centreY,
    float radius,
    FalloutActorScan& outScan,
    std::unordered_map<std::uint32_t, std::string>& outVoiceFolderPlugin,
    std::string& outError) {
    outScan = FalloutActorScan{};
    outVoiceFolderPlugin.clear();
    outError.clear();
    if (order.empty()) {
        outError = "empty load order";
        return false;
    }

    // Placements keyed by reference formID: an override MOVES an actor, it does
    // not add a second one standing inside the first.
    std::unordered_map<std::uint32_t, std::size_t> placementSlotByFormId;

    for (std::size_t pluginIndex = 0; pluginIndex < order.entries().size(); ++pluginIndex) {
        const FalloutLoadOrderEntry& entry = order.entries()[pluginIndex];
        FalloutActorScan scan;
        std::string error;
        if (!findActorsNear(entry.path, centreX, centreY, radius, scan, error)) {
            // A plugin that will not scan costs its actors, not everyone else's.
            // std::cerr, not VOX_LOGW: this file links into the probe and the
            // cooker, neither of which links core/log.cc.
            std::cerr << "[fnv] actors: skipping " << entry.header.fileName << ": " << error
                      << "\n";
            continue;
        }
        remapActorScan(order, pluginIndex, scan);

        for (auto& [formId, base] : scan.bases) {
            outScan.bases[formId] = std::move(base);
            outVoiceFolderPlugin[formId] = entry.header.fileName;
        }
        for (auto& [formId, list] : scan.leveledLists) {
            outScan.leveledLists[formId] = std::move(list);
        }
        for (auto& [formId, list] : scan.leveledItems) {
            outScan.leveledItems[formId] = std::move(list);
        }
        for (const auto& [formId, useAll] : scan.leveledItemUseAll) {
            outScan.leveledItemUseAll[formId] = useAll;
        }
        for (auto& [formId, race] : scan.races) {
            outScan.races[formId] = std::move(race);
        }
        for (auto& [formId, armor] : scan.armors) {
            outScan.armors[formId] = std::move(armor);
        }
        for (auto& [formId, outfit] : scan.outfits) {
            outScan.outfits[formId] = std::move(outfit);
        }
        for (auto& [formId, editorId] : scan.outfitEditorIds) {
            outScan.outfitEditorIds[formId] = std::move(editorId);
        }
        for (auto& [formId, addon] : scan.armorAddons) {
            outScan.armorAddons[formId] = std::move(addon);
        }
        for (auto& [formId, voice] : scan.voiceTypes) {
            outScan.voiceTypes[formId] = std::move(voice);
        }
        for (FalloutActorPlacement& placement : scan.placements) {
            const auto slot = placementSlotByFormId.find(placement.refFormId);
            if (slot == placementSlotByFormId.end()) {
                placementSlotByFormId.emplace(placement.refFormId, outScan.placements.size());
                outScan.placements.push_back(std::move(placement));
            } else {
                outScan.placements[slot->second] = std::move(placement);
            }
        }
    }

    // findActorsNear returns its own placements nearest-first; merging several
    // scans destroys that, so restore it here rather than leaving callers to
    // discover the order silently changed when a second plugin loaded.
    std::sort(
        outScan.placements.begin(), outScan.placements.end(),
        [&](const FalloutActorPlacement& a, const FalloutActorPlacement& b) {
            const float ax = a.position[0] - centreX;
            const float ay = a.position[1] - centreY;
            const float bx = b.position[0] - centreX;
            const float by = b.position[1] - centreY;
            return ((ax * ax) + (ay * ay)) < ((bx * bx) + (by * by));
        });
    return true;
}

bool findAllActorsAcrossOrder(
    const FalloutLoadOrder& order,
    FalloutActorScan& outScan,
    std::unordered_map<std::uint32_t, std::string>& outVoiceFolderPlugin,
    std::string& outError) {
    // A negative radius is the private sentinel consumed by findActorsNear's
    // placement pass. Keeping the full-content operation behind a named API
    // prevents callers from relying on that implementation detail.
    return findActorsNearAcrossOrder(
        order, 0.0f, 0.0f, -1.0f, outScan, outVoiceFolderPlugin, outError);
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
                header.type == "ARMO" || header.type == "CLOT" ||
                header.type == "ARMA" || header.type == "OTFT" ||
                header.type == "VTYP";
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
                outScan.races[record.formId] =
                    parseRaceParts(record, reader.pluginFormat());
                return;
            }
            // TES4 CLOT has the same wearable BMDT + male/female biped-model
            // shape as ARMO. Compile both into the common wardrobe table: an
            // Oblivion citizen generally gets their entire visible body from
            // clothing, so omitting CLOT leaves an otherwise valid actor as a
            // walking FaceGen head.
            if (record.type == "ARMO" || record.type == "CLOT") {
                FalloutArmorPiece armor;
                armor.formId = record.formId;
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "EDID") {
                        armor.editorId = subrecordText(sub);
                    } else if (sub.type == "BMDT" && sub.size >= 4u) {
                        armor.bipedFlags = readU32(sub);
                    } else if (sub.type == "BOD2" && sub.size >= 4u) {
                        armor.bipedFlags = readU32(sub);
                    } else if (sub.type == "MODL" && sub.size == 4u) {
                        armor.armatureFormIds.push_back(readU32(sub));
                    } else if (sub.type == "MODL") {
                        armor.maleModel = subrecordText(sub);
                    } else if (sub.type == "MOD3") {
                        armor.femaleModel = subrecordText(sub);
                    }
                }
                if ((!armor.maleModel.empty() || !armor.femaleModel.empty()) ||
                    !armor.armatureFormIds.empty()) {
                    outScan.armors[record.formId] = std::move(armor);
                }
                return;
            }
            if (record.type == "ARMA") {
                SkyrimArmorAddon addon;
                addon.formId = record.formId;
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "EDID") {
                        addon.editorId = subrecordText(sub);
                    } else if ((sub.type == "BODT" || sub.type == "BOD2") && sub.size >= 4u) {
                        addon.bipedFlags = readU32(sub);
                    } else if (sub.type == "MOD2") {
                        addon.maleModel = subrecordText(sub);
                    } else if (sub.type == "MOD3") {
                        addon.femaleModel = subrecordText(sub);
                    } else if (sub.type == "MOD4") {
                        addon.maleFirstPersonModel = subrecordText(sub);
                    } else if (sub.type == "MOD5") {
                        addon.femaleFirstPersonModel = subrecordText(sub);
                    } else if (sub.type == "RNAM" && sub.size >= 4u) {
                        addon.raceFormIds.push_back(readU32(sub));
                    } else if (sub.type == "MODL" && sub.size == 4u) {
                        addon.raceFormIds.push_back(readU32(sub));
                    }
                }
                if (!addon.maleModel.empty() || !addon.femaleModel.empty() ||
                    !addon.maleFirstPersonModel.empty() || !addon.femaleFirstPersonModel.empty()) {
                    outScan.armorAddons[record.formId] = std::move(addon);
                }
                return;
            }
            if (record.type == "OTFT") {
                std::vector<std::uint32_t> items;
                std::string editorId;
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "EDID") {
                        editorId = subrecordText(sub);
                    } else if (sub.type == "INAM" && sub.size >= 4u) {
                        // Skyrim writes the outfit as one packed form-id array,
                        // not necessarily one INAM subrecord per entry.
                        for (std::size_t offset = 0u;
                             offset + 4u <= sub.size; offset += 4u) {
                            items.push_back(readU32(sub, offset));
                        }
                    }
                }
                if (!editorId.empty()) {
                    outScan.outfitEditorIds[record.formId] = std::move(editorId);
                }
                if (!items.empty()) {
                    outScan.outfits[record.formId] = std::move(items);
                }
                return;
            }
            if (record.type == "LVLC" || record.type == "LVLN" || record.type == "LVLI") {
                // LVLO is a 12-byte entry; the referenced formID sits at offset
                // 4, after a u16 level and two unused bytes.
                std::vector<std::uint32_t> entries;
                bool useAll = false;
                for (const EsmSubrecordView& sub : record.subrecords) {
                    if (sub.type == "LVLO" && sub.size >= 8u) {
                        entries.push_back(readU32(sub, 4));
                    } else if (sub.type == "LVLF" && sub.size >= 1u) {
                        useAll = (sub.data[0] & 0x04u) != 0u;
                    }
                }
                if (!entries.empty()) {
                    (record.type == "LVLI" ? outScan.leveledItems
                                           : outScan.leveledLists)[record.formId] =
                        std::move(entries);
                    if (record.type == "LVLI") {
                        outScan.leveledItemUseAll[record.formId] = useAll;
                    }
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
                } else if (sub.type == "DOFT" && sub.size >= 4u) {
                    base.defaultOutfitFormId = readU32(sub);
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
            if (record.type == "NPC_") {
                std::ostringstream localForm;
                localForm << std::hex << std::nouppercase << std::setfill('0')
                          << std::setw(8) << record.formId;
                base.faceGeometryPaths.push_back(
                    "actors\\character\\facegendata\\facegeom\\" +
                    pluginPath.filename().string() + "\\" + localForm.str() + ".nif");
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
                } else if (sub.type == "XLRT" && sub.size >= 4u) {
                    placement.referenceTypeFormIds.push_back(readU32(sub));
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
            std::sort(placement.referenceTypeFormIds.begin(),
                placement.referenceTypeFormIds.end());
            placement.referenceTypeFormIds.erase(std::unique(
                placement.referenceTypeFormIds.begin(),
                placement.referenceTypeFormIds.end()),
                placement.referenceTypeFormIds.end());
            const float dx = placement.position[0] - centreX;
            const float dy = placement.position[1] - centreY;
            if (radius >= 0.0f && ((dx * dx) + (dy * dy)) > (radius * radius)) {
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

bool findAllActors(
    const std::filesystem::path& pluginPath,
    FalloutActorScan& outScan,
    std::string& outError) {
    return findActorsNear(pluginPath, 0.0f, 0.0f, -1.0f, outScan, outError);
}

}  // namespace odai::importer::fnv
