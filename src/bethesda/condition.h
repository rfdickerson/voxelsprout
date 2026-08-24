#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace odai::bethesda {

enum class ConditionComparison : std::uint8_t {
    Equal = 0u,
    NotEqual = 1u,
    Greater = 2u,
    GreaterOrEqual = 3u,
    Less = 4u,
    LessOrEqual = 5u,
};

struct Condition {
    ConditionComparison comparison = ConditionComparison::Equal;
    bool orWithNext = false;
    float comparisonValue = 0.0f;
    std::uint16_t function = 0u;
    std::uint32_t parameter1 = 0u;
    std::uint32_t parameter2 = 0u;
    // TES5 stores string operands in CIS1/CIS2 subrecords immediately after
    // CTDA. The raw 32-bit parameter slot is not a string-table offset.
    std::string stringParameter1;
    std::string stringParameter2;
    std::uint32_t runOn = 0u;
    std::uint32_t reference = 0u;
};

using ConditionFunction = std::function<std::optional<float>(const Condition&)>;

struct ConditionEvaluation {
    bool matched = true;
    std::vector<std::string> diagnostics;
};

bool readCondition(std::span<const std::uint8_t> ctda, Condition& out, std::string& outError);
ConditionEvaluation evaluateConditions(
    std::span<const Condition> conditions,
    const ConditionFunction& function,
    bool strict);

}  // namespace odai::bethesda
