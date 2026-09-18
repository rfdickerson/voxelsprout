#pragma once

#include "anim/hkx_packfile.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <optional>

namespace odai::anim {

struct BehaviorExpression {
    enum class Op { Constant, Variable, Negate, Not, Add, Subtract, Multiply, Divide,
        Less, LessEqual, Greater, GreaterEqual, Equal, NotEqual, And, Or };
    struct Term {
        Op op = Op::Constant;
        float value = 0;
        std::string variable;
        std::uint32_t left = 0, right = 0;
    };
    std::vector<Term> terms;
    std::uint32_t root = 0;
    [[nodiscard]] std::optional<float> evaluate(const std::map<std::string, float>& variables) const;
};

std::optional<BehaviorExpression> compileBehaviorExpression(std::string_view expression,
    const std::vector<std::string>& variables, std::string& error);

// Admission is deliberately separate from structural decoding. A class name in
// an HKX file is not evidence that its runtime semantics have been implemented.
struct BehaviorProgram {
    HkxDecodedBehaviorGraph graph;
    std::vector<std::string> unsupported;
    std::map<std::pair<std::uint32_t, std::size_t>, BehaviorExpression> conditions;
    [[nodiscard]] bool executable() const { return unsupported.empty(); }
};

std::shared_ptr<const BehaviorProgram> compileBehaviorProgram(HkxDecodedBehaviorGraph graph);

} // namespace odai::anim
