#pragma once

// Finds the actors a plugin places in a region, and what each one needs to be
// built into geometry.
//
// This is the discovery half of populating a settlement; character_builder.h is
// the assembly half. Kept separate from dialogue_records.h, which answers a
// different question (who speaks, and what) about a single named actor.
//
// THREE KINDS OF ACTOR, and the difference is the whole reason this file is not
// a one-liner. Measured on Goodsprings (37 placements):
//
//   * A CREA with a NIFZ list carries its own geometry: MODL is its skeleton
//     and NIFZ names the body parts beside it. Victor is one. 16 of the 37.
//   * A CREA with no NIFZ is a TEMPLATE actor -- a levelled/spawn variant whose
//     geometry comes from its TPLT. Note the hop is usually NOT to another
//     actor: it lands on a LEVELLED CREATURE list (LVLC), whose entries are the
//     real actors. Following TPLT only through CREA/NPC_ resolves none of the
//     VSpawnTier1* coyotes and bloatflies, because every one of them points at
//     a list.
//   * An NPC_ carries a skeleton and NOTHING else. Its body is assembled from
//     its RACE record's part models, and then largely replaced by whatever it
//     is wearing. Every human in the town is one of these.
//
// A caller that handles only the first kind renders the wildlife and none of
// the people, which is why `geometrySource` is reported rather than left for
// the caller to infer from an empty part list.
//
// PATHS OUT OF `resolve()` ARE ALWAYS FULL MESH PATHS, whichever kind they came
// from. NIFZ stores names relative to the skeleton's own directory and RACE and
// ARMO store full ones; making every caller know which is which is how you get
// a creature that loads and an NPC that silently does not.

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "import/fnv/plugin_load_order.h"

namespace odai::importer::fnv {

enum class ActorGeometrySource : std::uint8_t {
    // MODL + NIFZ on the base record itself.
    OwnBodyParts = 0,
    // Reached through TPLT; `resolvedBaseFormId` is where the parts came from.
    Template = 1,
    // NPC_: needs RACE part models plus equipment. `raceFormId` is set.
    Race = 2,
    // Nothing usable found.
    None = 3,
};

struct FalloutActorBase {
    std::uint32_t formId = 0;
    std::string editorId;
    // FULL -- the name the game shows. An EditorID reads as "GSSettlerAM"; a
    // prompt offering to talk to that is a debug string on screen.
    std::string fullName;
    std::string recordType;   // "CREA" or "NPC_"
    std::string skeletonPath; // MODL -- the skeleton for a CREA, and for an NPC_ too
    // TES5 stores each NPC's generated head as a separate mesh named by the
    // NPC record's LOCAL form ID. It is not referenced by a subrecord, so the
    // path must be captured before load-order remapping changes the form ID.
    // The ordinary case has one entry. The customizable Player has no baked
    // FaceGeom asset, so the explicit showcase replaces it with the ordered
    // retail head/eyes/mouth/brows/hair pieces used for its stock appearance.
    std::vector<std::string> faceGeometryPaths;
    std::vector<std::string> bodyPartPaths;  // NIFZ, relative to the skeleton's directory
    std::uint32_t templateFormId = 0;        // TPLT
    std::uint32_t raceFormId = 0;            // RNAM
    // Worn/carried items (CNTO). An NPC_'s clothing lives here; resolving these
    // to ARMO biped models is what stops the townsfolk rendering naked.
    std::vector<std::uint32_t> inventoryFormIds;
    // TES5 DOFT. Skyrim moved an NPC's worn set out of CNTO and into an OTFT
    // record; guards usually carry only weapons in CNTO, so ignoring this
    // leaves every otherwise-valid actor undressed.
    std::uint32_t defaultOutfitFormId = 0;
    bool isFemale = false;  // ACBS flag bit 0, picks RACE's FNAM parts over MNAM
    // VTCK. Names a VTYP record whose EditorID IS the voice folder under
    // sound\voice\<plugin>\ -- so this, not the actor's name, is what finds a
    // recorded line. Zero means "inherit the race's", which most actors do.
    std::uint32_t voiceTypeFormId = 0;
    // ACBS's trailing u16. Which fields the record actually OWNS rather than
    // borrows from its TPLT -- see kActorTemplateUse* below. A record that
    // borrows its traits still stores a race and a sex of its own, and they are
    // stale data the game never reads.
    std::uint16_t templateFlags = 0;
};

// ACBS template flags: which fields the record BORROWS from its TPLT rather
// than owning. These three decide where an actor's body comes from.
inline constexpr std::uint16_t kActorTemplateUseTraits = 0x0001;     // race, sex
inline constexpr std::uint16_t kActorTemplateUseInventory = 0x0100;  // CNTO, so clothing
// The SKELETON. A record that borrows its model stores "marker_creature.nif"
// as its own MODL -- a real, parseable NIF with none of the bones a body is
// weighted to, so using it does not fail: it binds a character whose bones are
// all unresolved and which silently collapses. Measured on Fallout 3, where a
// levelled raider reported 71 unresolved bones and stood in bind pose because
// of exactly this.
inline constexpr std::uint16_t kActorTemplateUseModelAnimation = 0x0040;

// A race's body, which FNV stores as a set of SLOTS rather than one model: the
// game assembles a human from an upper body, two hands and a head, then swaps
// individual slots out for whatever is worn. Indices are the record's own INDX
// values, kept as-is so a slot with no model (human ears) stays empty rather
// than shifting everything after it.
inline constexpr std::size_t kRaceHeadPartCount = 8;  // head, ears, mouth, teeth x2, tongue, eyes x2
inline constexpr std::size_t kRaceBodyPartCount = 4;  // upper body, left hand, right hand, (.egt)
inline constexpr std::size_t kRaceHeadSlot = 0;
inline constexpr std::size_t kRaceUpperBodySlot = 0;
inline constexpr std::size_t kRaceLeftHandSlot = 1;
inline constexpr std::size_t kRaceRightHandSlot = 2;

struct FalloutRaceParts {
    std::uint32_t formId = 0;
    std::string editorId;
    // TES4 uses its four NAM1 body indices as upper body, lower body, hands,
    // and feet. Fallout reuses the same positional record shape for upper
    // body, left hand, right hand, and a non-mesh FaceGen texture. Keep the
    // generation with the compiled race so resolution never has to guess from
    // filenames or from which slots happen to be populated.
    bool usesOblivionBodyLayout = false;
    // VTCK, which on a RACE is a PAIR: male voice type then female, 8 bytes.
    // An actor with no VTCK of its own takes whichever its sex selects.
    std::uint32_t maleVoiceTypeFormId = 0;
    std::uint32_t femaleVoiceTypeFormId = 0;
    // TES5 WNAM: the race's default skin ARMO. Its ARMA records provide the
    // naked hands/feet/body for slots not covered by the actor's outfit.
    std::uint32_t defaultSkinFormId = 0;
    // TES5 RACE stores the male/female skeleton directly in ANAM after the
    // corresponding MNAM/FNAM marker. Earlier generations leave these empty.
    std::string maleSkeletonPath;
    std::string femaleSkeletonPath;
    std::string maleHeadModels[kRaceHeadPartCount];
    std::string femaleHeadModels[kRaceHeadPartCount];
    std::string maleBodyModels[kRaceBodyPartCount];
    std::string femaleBodyModels[kRaceBodyPartCount];
};

// Biped slots an ARMO covers (BMDT's first u32). Only the ones that decide
// whether a body slot is replaced are named; the rest (weapon, pipboy,
// jewellery) are carried through untouched.
inline constexpr std::uint32_t kBipedSlotHead = 0x00000001u;
inline constexpr std::uint32_t kBipedSlotHair = 0x00000002u;
inline constexpr std::uint32_t kBipedSlotUpperBody = 0x00000004u;
inline constexpr std::uint32_t kBipedSlotLeftHand = 0x00000008u;
inline constexpr std::uint32_t kBipedSlotRightHand = 0x00000010u;
inline constexpr std::uint32_t kBipedSlotHat = 0x00000400u;

// Oblivion predates Fallout's split left/right-hand biped objects. Its BMDT
// word uses the next three bits for lower body, hands, and feet respectively.
inline constexpr std::uint32_t kOblivionBipedSlotLowerBody = 0x00000008u;
inline constexpr std::uint32_t kOblivionBipedSlotHands = 0x00000010u;
inline constexpr std::uint32_t kOblivionBipedSlotFeet = 0x00000020u;

struct FalloutArmorPiece {
    std::uint32_t formId = 0;
    std::string editorId;
    std::uint32_t bipedFlags = 0;  // BMDT
    std::string maleModel;         // MODL -- the male BIPED model, not the ground model
    std::string femaleModel;       // MOD3
    // TES5 ARMO names one or more ARMA records in binary MODL subrecords.
    // The ARMA, not ARMO's MOD2 ground model, owns the skinned third-person
    // body mesh.
    std::vector<std::uint32_t> armatureFormIds;
};

struct SkyrimArmorAddon {
    std::uint32_t formId = 0;
    std::string editorId;
    std::uint32_t bipedFlags = 0;
    std::string maleModel;    // ARMA MOD2
    std::string femaleModel;  // ARMA MOD3
    std::string maleFirstPersonModel;    // ARMA MOD4
    std::string femaleFirstPersonModel;  // ARMA MOD5
    // RNAM plus the binary MODL tail. Several ARMA records can hang off one
    // helmet ARMO (human, Argonian, Khajiit); only the addon naming the actor's
    // race is applicable.
    std::vector<std::uint32_t> raceFormIds;
};

struct FalloutActorPlacement {
    std::uint32_t refFormId = 0;
    std::uint32_t baseFormId = 0;
    // TES5 XLRT location-reference types used by quest alias filling.
    std::vector<std::uint32_t> referenceTypeFormIds;
    float position[3] = {};         // Bethesda space, as stored
    float rotationRadians[3] = {};
    // Record header flag 0x800. These are dormant until quest state enables
    // them -- rendering them puts duplicate actors in the world, which is how
    // Goodsprings ends up with two Victors.
    bool initiallyDisabled = false;
};

// A base with its geometry question already answered.
struct ResolvedActorBase {
    const FalloutActorBase* base = nullptr;
    ActorGeometrySource geometrySource = ActorGeometrySource::None;
    // For Template: the base the parts actually came from. Equals base->formId
    // for OwnBodyParts and Race.
    std::uint32_t resolvedBaseFormId = 0;
    // Skeleton + parts to build, already followed through any TPLT chain. Full
    // mesh paths in every case.
    std::string skeletonPath;
    std::vector<std::string> bodyPartPaths;
    // Race only: which ARMO records supplied a body slot. Empty means the actor
    // is standing there in the race's underwear, which is a truthful render of
    // an actor with no clothes resolvable rather than a failure.
    std::vector<std::uint32_t> wornArmorFormIds;
};

struct FalloutActorScan {
    std::vector<FalloutActorPlacement> placements;  // sorted nearest-first
    std::unordered_map<std::uint32_t, FalloutActorBase> bases;
    // LVLC/LVLN formID -> the actor formIDs it can spawn, in list order. A
    // template actor's TPLT usually lands here rather than on another actor:
    // LVLC for creatures, LVLN for NPCs. One map because the two play the same
    // role and a chain never cares which kind it landed on.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> leveledLists;
    // LVLI formID -> the item formIDs it can hand out. A settler does not carry
    // an outfit, she carries "OutfitSettlerFemale", which is one of these; an
    // inventory walk that does not expand them dresses half the town in
    // underwear. Entries can themselves be lists, so expansion recurses.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> leveledItems;
    // LVLF bit 2 is "use all". A set such as ArmorStormcloakSet contributes
    // boots + cuirass + gloves + helmet; its parent OutfitListSoldierSons has
    // the bit clear and chooses one complete set. Flattening both kinds puts
    // every mutually-exclusive outfit on the actor at once.
    std::unordered_map<std::uint32_t, bool> leveledItemUseAll;
    std::unordered_map<std::uint32_t, FalloutRaceParts> races;
    std::unordered_map<std::uint32_t, FalloutArmorPiece> armors;
    // OTFT formID -> its INAM item list.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> outfits;
    // OTFT formID -> EditorID. The read-only avatar catalog uses this to
    // select an authored Skyrim outfit without publishing Skyrim records into
    // the active game's mutable runtime world.
    std::unordered_map<std::uint32_t, std::string> outfitEditorIds;
    std::unordered_map<std::uint32_t, SkyrimArmorAddon> armorAddons;
    // VTYP formID -> its EditorID, which is the voice folder's name verbatim
    // ("MaleAdult11", "RobotVictor"). Collected wholesale for the same reason
    // the bases are: resolving them one at a time would be one walk each.
    std::unordered_map<std::uint32_t, std::string> voiceTypes;

    // Follows TPLT to whichever base actually carries geometry, and assembles
    // an NPC_'s body from its race and its wardrobe. Returns a source of None
    // when nothing in the chain does.
    [[nodiscard]] ResolvedActorBase resolve(std::uint32_t baseFormId) const;

    // The record an actor's `templateUseFlag`-governed fields actually come
    // from -- itself, when it owns them. Returns null only for an unknown
    // formID; a base with no template is its own answer.
    // The first real actor reachable from `formId`, following levelled lists
    // through as many levels as they nest. Returns null when the chain holds
    // no actor this scan knows.
    [[nodiscard]] const FalloutActorBase* firstActorFrom(
        std::uint32_t formId, std::uint32_t excludeFormId = 0) const;

    [[nodiscard]] const FalloutActorBase* inheritedFrom(
        std::uint32_t baseFormId, std::uint16_t templateUseFlag) const;

    // Resolves inherited CNTO inventory and deterministically expands nested
    // LVLI records. List records themselves are never returned. `seed` should
    // be the persistent placed-reference ID so stream order cannot alter loot.
    [[nodiscard]] std::vector<std::uint32_t> materializeInventory(
        std::uint32_t baseFormId, std::uint32_t seed) const;

    // The VTYP an actor speaks with, or 0. Its own VTCK first, then its RACE's
    // male/female pair by sex -- most actors carry no VTCK and would otherwise
    // resolve to nothing. Both are inheritable from a TPLT.
    //
    // The formID rather than the folder name, because this is also the key a
    // CTDA names when it binds a generic line to a voice type.
    [[nodiscard]] std::uint32_t voiceTypeFormIdFor(std::uint32_t baseFormId) const;

    // The voice folder an actor's recorded lines live under, or empty.
    //
    // Its own VTCK first, then its RACE's male/female pair by sex -- most
    // actors carry no VTCK and would otherwise resolve to silence. Both are
    // inheritable from a TPLT, so both go through inheritedFrom.
    [[nodiscard]] std::string voiceFolderFor(std::uint32_t baseFormId) const;
};

// Scans `pluginPath` for every ACRE/ACHR placed within `radius` of the Bethesda
// XY (`centreX`, `centreY`), plus every actor base, race and armor record so
// the placements can be resolved without a second pass.
//
// Two passes over the plugin, both filtered on the record header -- see the
// note in dialogue_records.cc about what omitting that filter costs.
bool findActorsNear(
    const std::filesystem::path& pluginPath,
    float centreX,
    float centreY,
    float radius,
    FalloutActorScan& outScan,
    std::string& outError);

// As above, across a whole load order. Every plugin is scanned and its formIDs
// rewritten into the order's global space; bases, races, armour and voice types
// merge later-wins, and placements merge by reference formID so an override
// moves an actor rather than duplicating it.
//
// A companion mod is the reason this exists: its NPC, its placement, its race
// and its armour all live in ITS plugin, so scanning only the worldspace's
// plugin finds nothing at all.
//
// outVoiceFolderPlugin maps each base formID to the FILE NAME of the plugin that
// defined it. Voice paths start with the defining plugin's own name
// (sound\voice\NVWillow.esp\...), so this cannot be derived from the load
// order's first entry -- see FalloutActorScan::voiceFolderFor.
bool findActorsNearAcrossOrder(
    const FalloutLoadOrder& order,
    float centreX,
    float centreY,
    float radius,
    FalloutActorScan& outScan,
    std::unordered_map<std::uint32_t, std::string>& outVoiceFolderPlugin,
    std::string& outError);

// Builds the immutable actor-content catalog for a plugin/load order. Unlike
// findActorsNear*, these retain every winning ACRE/ACHR placement regardless
// of its authored coordinates. Runtime streaming must start from this catalog:
// a package or MoveTo can put an interior-owned actor into a distant exterior
// cell, and filtering the authored placement before consulting runtime state
// makes that actor impossible to discover.
bool findAllActors(
    const std::filesystem::path& pluginPath,
    FalloutActorScan& outScan,
    std::string& outError);

bool findAllActorsAcrossOrder(
    const FalloutLoadOrder& order,
    FalloutActorScan& outScan,
    std::unordered_map<std::uint32_t, std::string>& outVoiceFolderPlugin,
    std::string& outError);

}  // namespace odai::importer::fnv
