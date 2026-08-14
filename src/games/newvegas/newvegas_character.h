#pragma once

// The player character: what New Vegas asks you to decide before it lets you
// out of Doc Mitchell's house, read from the game's own records rather than
// transcribed from a wiki.
//
// WHY READ IT AT ALL, when seven attribute names could be typed out in a line:
// because the names in the files are not the names anybody would type. FNV's
// Survival skill is stored as "AVThrowing" -- Fallout 3's skill, renamed in the
// FULL field and never in the EditorID -- and its Guns skill is "AVSmallGuns".
// A hardcoded list gets those two wrong in a way that looks like a typo and is
// actually the game disagreeing with the wiki. The same goes for traits, which
// carry their own rules text.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace odai::games::newvegas {

// One SPECIAL attribute, skill, or trait, as the game names it.
struct CharacterStatDefinition {
    std::uint32_t formId = 0;
    std::string editorId;   // "AVThrowing"
    std::string name;       // "Survival" -- FULL, which is what a player sees
    std::string description;
};

// Everything character creation offers, loaded once from the plugin.
struct CharacterDefinitions {
    // Exactly seven, in S.P.E.C.I.A.L. order. That order is not a convention
    // this code imposes -- the records are consecutive, AVStrength at 0x3E8
    // through AVLuck at 0x3EE, in exactly that sequence.
    std::vector<CharacterStatDefinition> attributes;
    // Thirteen. AVBigGuns (0x4B1) is deliberately absent: its own FULL says
    // "Big Guns - OBSOLETE", the record having survived the FO3 -> FNV skill
    // merge that folded it into Guns and Explosives.
    std::vector<CharacterStatDefinition> skills;
    // PERK records flagged as traits. Read rather than listed because the DLC
    // add their own, and because each carries the rules text to show.
    std::vector<CharacterStatDefinition> traits;

    [[nodiscard]] bool valid() const {
        return attributes.size() == 7u && !skills.empty();
    }
};

// Reads the definitions out of a plugin. Returns false only if the file cannot
// be walked; a plugin missing the records yields an invalid() result instead.
bool loadCharacterDefinitions(
    const std::filesystem::path& pluginPath,
    CharacterDefinitions& outDefinitions,
    std::string& outError);

// The rules of the creation screen, which are the game's own.
inline constexpr int kSpecialPointPool = 40;   // what the Vit-o-Matic hands out
inline constexpr int kSpecialMinimum = 1;
inline constexpr int kSpecialMaximum = 10;
inline constexpr std::size_t kTagSkillCount = 3;
inline constexpr std::size_t kMaxTraits = 2;   // and taking none is allowed

// The character being made, and afterwards the one being played.
struct PlayerCharacter {
    std::string name = "Courier";
    bool isFemale = false;
    // Indexed the same as CharacterDefinitions::attributes. Five across the
    // board is 35 of the 40 points, which is where the game starts you.
    int special[7] = {5, 5, 5, 5, 5, 5, 5};
    // Indices into CharacterDefinitions::skills / ::traits.
    std::vector<std::size_t> taggedSkills;
    std::vector<std::size_t> traits;

    [[nodiscard]] int spentPoints() const {
        int spent = 0;
        for (const int value : special) {
            spent += value;
        }
        return spent;
    }
    [[nodiscard]] int remainingPoints() const { return kSpecialPointPool - spentPoints(); }

    [[nodiscard]] bool hasTag(std::size_t skillIndex) const;
    [[nodiscard]] bool hasTrait(std::size_t traitIndex) const;
    // Toggles, refusing anything that would exceed the caps. Returns whether the
    // set changed, so a UI can decline silently rather than lying about it.
    bool toggleTag(std::size_t skillIndex);
    bool toggleTrait(std::size_t traitIndex);

    // Creation is finishable only when every point is spent and the tag skills
    // are chosen. Traits are optional -- taking none is a legitimate build.
    [[nodiscard]] bool isComplete() const {
        return remainingPoints() == 0 && taggedSkills.size() == kTagSkillCount;
    }

    // Base skill value from SPECIAL, before tags. Fallout's own formula:
    // 2 + (2 x attribute) + ceil(Luck / 2). Tagged skills get +15.
    [[nodiscard]] int skillValue(
        const CharacterDefinitions& definitions, std::size_t skillIndex) const;
};

}  // namespace odai::games::newvegas
