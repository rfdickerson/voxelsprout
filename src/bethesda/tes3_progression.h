#pragma once

#include "bethesda/runtime_ids.h"
#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>

namespace odai::bethesda {
inline constexpr std::array<std::string_view, 8> tes3AttributeNames{
    "strength", "intelligence", "willpower", "agility", "speed", "endurance", "personality", "luck"};
inline constexpr std::array<std::string_view, 27> tes3SkillNames{
    "block", "armorer", "mediumarmor", "heavyarmor", "bluntweapon", "longblade", "axe", "spear",
    "athletics", "enchant", "destruction", "alteration", "illusion", "conjuration", "mysticism",
    "restoration", "alchemy", "unarmored", "security", "sneak", "acrobatics", "lightarmor",
    "shortblade", "marksman", "mercantile", "speechcraft", "handtohand"};

struct Tes3ProgressionClass {
    std::array<int, 2> attributes{0, 5};
    int specialization = 0;
    std::array<int, 5> minor{0, 1, 2, 3, 4};
    std::array<int, 5> major{5, 6, 7, 8, 9};
    friend bool operator==(const Tes3ProgressionClass&, const Tes3ProgressionClass&) = default;
};
struct Tes3SkillDefinition {
    int attribute = -1;
    int specialization = -1;
    std::array<double, 4> useValues{};
};
struct Tes3ProgressionState {
    // Base attributes/skills remain in the shared player numeric stat map.
    // Progress is normalized [0,1); TES3 discards usage overflow on advancement.
    std::array<double, 27> skillProgress{};
    std::array<int, 8> attributeIncreases{};
    int levelProgress = 0;
    bool selectionOpen = false;
    ObjectId restBed;
    bool readyNotification = false;
    std::set<std::string> readBooks;
    std::optional<Tes3ProgressionClass> customClass;
    friend bool operator==(const Tes3ProgressionState&, const Tes3ProgressionState&) = default;
};
enum class Tes3SkillAdvanceSource { Usage, Training, Book, Jail };
class Tes3ContentStore;
struct Tes3ActorDefinition;
bool tes3AutoCalculateNpc(const Tes3ContentStore&, Tes3ActorDefinition&);
struct Tes3ArmorDefinition { int slot = -1; int skill = -1; };
// Armor types 9/10 (bracers) share the gauntlet slots 6/7.
[[nodiscard]] std::optional<Tes3ArmorDefinition> tes3ArmorDefinition(const Tes3ContentStore&, std::string_view id);
[[nodiscard]] int tes3ArmorHitSlot(unsigned roll);
[[nodiscard]] std::optional<int> tes3WeaponSkill(const Tes3ContentStore&, std::string_view id);
[[nodiscard]] std::optional<Tes3SkillDefinition> tes3SkillDefinition(const Tes3ContentStore&, int skill);
[[nodiscard]] std::optional<Tes3ProgressionClass> tes3ClassDefinition(const Tes3ContentStore&, std::string_view id);
[[nodiscard]] bool validTes3ProgressionClass(const Tes3ProgressionClass&);
// Vanilla fallback only for absent settings; malformed imported values return NaN.
[[nodiscard]] double tes3NumericSetting(const Tes3ContentStore&, std::string_view id, double fallback);
} // namespace odai::bethesda
