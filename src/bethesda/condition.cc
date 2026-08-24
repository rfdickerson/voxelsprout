#include "bethesda/condition.h"

#include <bit>
#include <cmath>
#include <cstring>

namespace odai::bethesda {
namespace {

std::uint16_t u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8u));
}

std::uint32_t u32(const std::uint8_t* bytes) {
    std::uint32_t value = 0u;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

bool compare(float actual, ConditionComparison operation, float expected) {
    switch (operation) {
        case ConditionComparison::Equal: return actual == expected;
        case ConditionComparison::NotEqual: return actual != expected;
        case ConditionComparison::Greater: return actual > expected;
        case ConditionComparison::GreaterOrEqual: return actual >= expected;
        case ConditionComparison::Less: return actual < expected;
        case ConditionComparison::LessOrEqual: return actual <= expected;
    }
    return false;
}

}  // namespace

bool readCondition(
    std::span<const std::uint8_t> ctda, Condition& out, std::string& outError) {
    if (ctda.size() < 28u) { outError = "CTDA is shorter than 28 bytes"; return false; }
    const std::uint8_t operation = static_cast<std::uint8_t>((ctda[0] >> 5u) & 0x7u);
    if (operation > static_cast<std::uint8_t>(ConditionComparison::LessOrEqual)) {
        outError = "CTDA has unsupported comparison operator " + std::to_string(operation);
        return false;
    }
    Condition parsed;
    parsed.comparison = static_cast<ConditionComparison>(operation);
    parsed.orWithNext = (ctda[0] & 0x1u) != 0u;
    const std::uint32_t comparisonBits = u32(ctda.data() + 4u);
    parsed.comparisonValue = std::bit_cast<float>(comparisonBits);
    if (!std::isfinite(parsed.comparisonValue)) {
        outError = "CTDA comparison value is not finite"; return false;
    }
    parsed.function = u16(ctda.data() + 8u);
    parsed.parameter1 = u32(ctda.data() + 12u);
    parsed.parameter2 = u32(ctda.data() + 16u);
    parsed.runOn = u32(ctda.data() + 20u);
    parsed.reference = u32(ctda.data() + 24u);
    out = parsed; outError.clear(); return true;
}

ConditionEvaluation evaluateConditions(
    std::span<const Condition> conditions,
    const ConditionFunction& function,
    bool strict) {
    ConditionEvaluation result;
    bool groupMatched = false;
    for (std::size_t index = 0u; index < conditions.size(); ++index) {
        const Condition& condition = conditions[index];
        const std::optional<float> actual = function ? function(condition) : std::nullopt;
        bool matched = true;
        if (!actual.has_value()) {
            result.diagnostics.push_back(
                "unsupported CTDA function " + std::to_string(condition.function));
            matched = !strict;
        } else if (!std::isfinite(*actual)) {
            result.diagnostics.push_back(
                "CTDA function " + std::to_string(condition.function) + " returned non-finite data");
            matched = false;
        } else {
            matched = compare(*actual, condition.comparison, condition.comparisonValue);
        }
        groupMatched = groupMatched || matched;
        if (!condition.orWithNext) {
            result.matched = result.matched && groupMatched;
            groupMatched = false;
        }
    }
    if (!conditions.empty() && conditions.back().orWithNext) {
        result.diagnostics.push_back("CTDA OR chain ends without a following condition");
        result.matched = false;
    }
    return result;
}

}  // namespace odai::bethesda
