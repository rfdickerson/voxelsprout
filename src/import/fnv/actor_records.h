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
    std::string recordType;   // "CREA" or "NPC_"
    std::string skeletonPath; // MODL -- the skeleton for a CREA, and for an NPC_ too
    std::vector<std::string> bodyPartPaths;  // NIFZ, relative to the skeleton's directory
    std::uint32_t templateFormId = 0;        // TPLT
    std::uint32_t raceFormId = 0;            // RNAM
    // Worn/carried items (CNTO). An NPC_'s clothing lives here; resolving these
    // to ARMO biped models is what stops the townsfolk rendering naked.
    std::vector<std::uint32_t> inventoryFormIds;
    bool isFemale = false;  // ACBS flag bit 0, picks RACE's FNAM parts over MNAM
    // ACBS's trailing u16. Which fields the record actually OWNS rather than
    // borrows from its TPLT -- see kActorTemplateUse* below. A record that
    // borrows its traits still stores a race and a sex of its own, and they are
    // stale data the game never reads.
    std::uint16_t templateFlags = 0;
};

// ACBS template flags, the two that decide where an NPC_'s body comes from.
inline constexpr std::uint16_t kActorTemplateUseTraits = 0x0001;     // race, sex
inline constexpr std::uint16_t kActorTemplateUseInventory = 0x0100;  // CNTO, so clothing

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

struct FalloutArmorPiece {
    std::uint32_t formId = 0;
    std::string editorId;
    std::uint32_t bipedFlags = 0;  // BMDT
    std::string maleModel;         // MODL -- the male BIPED model, not the ground model
    std::string femaleModel;       // MOD3
};

struct FalloutActorPlacement {
    std::uint32_t refFormId = 0;
    std::uint32_t baseFormId = 0;
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
    // LVLC formID -> the actor formIDs it can spawn, in list order. A template
    // actor's TPLT lands here rather than on another actor.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> leveledLists;
    // LVLI formID -> the item formIDs it can hand out. A settler does not carry
    // an outfit, she carries "OutfitSettlerFemale", which is one of these; an
    // inventory walk that does not expand them dresses half the town in
    // underwear. Entries can themselves be lists, so expansion recurses.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> leveledItems;
    std::unordered_map<std::uint32_t, FalloutRaceParts> races;
    std::unordered_map<std::uint32_t, FalloutArmorPiece> armors;

    // Follows TPLT to whichever base actually carries geometry, and assembles
    // an NPC_'s body from its race and its wardrobe. Returns a source of None
    // when nothing in the chain does.
    [[nodiscard]] ResolvedActorBase resolve(std::uint32_t baseFormId) const;

    // The record an actor's `templateUseFlag`-governed fields actually come
    // from -- itself, when it owns them. Returns null only for an unknown
    // formID; a base with no template is its own answer.
    [[nodiscard]] const FalloutActorBase* inheritedFrom(
        std::uint32_t baseFormId, std::uint16_t templateUseFlag) const;
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

}  // namespace odai::importer::fnv
