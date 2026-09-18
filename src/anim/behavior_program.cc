#include "anim/behavior_program.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <map>
#include <charconv>
#include <cctype>
#include <stdexcept>

namespace odai::anim {

namespace {
class ExpressionParser {
public:
    ExpressionParser(std::string_view text, const std::vector<std::string>& names)
        : m_text(text), m_names(names) {}
    BehaviorExpression parse() {
        if (m_text.size() > 4096) throw std::runtime_error("expression length exceeds 4096");
        m_result.root = expression(1, 0);
        whitespace();
        if (m_position != m_text.size()) throw std::runtime_error("unsupported expression token");
        return std::move(m_result);
    }
private:
    using Op = BehaviorExpression::Op;
    struct Binary { std::string_view token; Op op; unsigned precedence; };
    static constexpr Binary operators[] = {
        {"||", Op::Or, 1}, {"&&", Op::And, 2}, {"==", Op::Equal, 3}, {"!=", Op::NotEqual, 3},
        {"<=", Op::LessEqual, 4}, {">=", Op::GreaterEqual, 4}, {"<", Op::Less, 4}, {">", Op::Greater, 4},
        {"+", Op::Add, 5}, {"-", Op::Subtract, 5}, {"*", Op::Multiply, 6}, {"/", Op::Divide, 6}};
    void whitespace() {
        while (m_position < m_text.size() && std::isspace(static_cast<unsigned char>(m_text[m_position]))) ++m_position;
    }
    bool take(std::string_view token) {
        whitespace();
        if (!m_text.substr(m_position).starts_with(token)) return false;
        m_position += token.size();
        return true;
    }
    std::uint32_t term(Op op, std::uint32_t left = 0, std::uint32_t right = 0,
                       float value = 0, std::string variable = {}) {
        if (m_result.terms.size() >= 256) throw std::runtime_error("expression complexity exceeds 256 terms");
        m_result.terms.push_back({op, value, std::move(variable), left, right});
        return static_cast<std::uint32_t>(m_result.terms.size() - 1);
    }
    std::uint32_t primary(unsigned depth) {
        if (depth > 64) throw std::runtime_error("expression nesting exceeds 64");
        if (take("!")) return term(Op::Not, primary(depth + 1));
        if (take("-")) return term(Op::Negate, primary(depth + 1));
        if (take("+")) return primary(depth + 1);
        if (take("(")) {
            const auto result = expression(1, depth + 1);
            if (!take(")")) throw std::runtime_error("missing expression closing parenthesis");
            return result;
        }
        whitespace();
        const auto begin = m_position;
        while (m_position < m_text.size() &&
            (std::isalnum(static_cast<unsigned char>(m_text[m_position])) || m_text[m_position] == '_')) ++m_position;
        const auto name = m_text.substr(begin, m_position - begin);
        if (!name.empty() && (std::isalpha(static_cast<unsigned char>(name.front())) || name.front() == '_')) {
            if (name == "true" || name == "false") return term(Op::Constant, 0, 0, name == "true" ? 1.f : 0.f);
            if (std::find(m_names.begin(), m_names.end(), name) == m_names.end())
                throw std::runtime_error("unknown expression variable: " + std::string(name));
            return term(Op::Variable, 0, 0, 0, std::string(name));
        }
        m_position = begin;
        float value = 0;
        const auto parsed = std::from_chars(m_text.data() + begin, m_text.data() + m_text.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr == m_text.data() + begin || !std::isfinite(value))
            throw std::runtime_error("invalid expression operand");
        m_position = static_cast<std::size_t>(parsed.ptr - m_text.data());
        return term(Op::Constant, 0, 0, value);
    }
    std::uint32_t expression(unsigned precedence, unsigned depth) {
        auto left = primary(depth);
        while (true) {
            whitespace();
            const Binary* next = nullptr;
            for (const auto& candidate : operators)
                if (m_text.substr(m_position).starts_with(candidate.token)) { next = &candidate; break; }
            if (!next || next->precedence < precedence) return left;
            m_position += next->token.size();
            left = term(next->op, left, expression(next->precedence + 1, depth + 1));
        }
    }
    std::string_view m_text;
    const std::vector<std::string>& m_names;
    std::size_t m_position = 0;
    BehaviorExpression m_result;
};
} // namespace

std::optional<BehaviorExpression> compileBehaviorExpression(std::string_view expression,
    const std::vector<std::string>& variables, std::string& error) {
    error.clear();
    try { return ExpressionParser(expression, variables).parse(); }
    catch (const std::runtime_error& exception) { error = exception.what(); return std::nullopt; }
}

std::optional<float> BehaviorExpression::evaluate(const std::map<std::string, float>& variables) const {
    std::function<std::optional<float>(std::uint32_t, unsigned)> run =
        [&](std::uint32_t index, unsigned depth) -> std::optional<float> {
        if (index >= terms.size() || depth > 256) return std::nullopt;
        const auto& term = terms[index];
        if (term.op == Op::Constant) return std::isfinite(term.value) ? std::optional(term.value) : std::nullopt;
        if (term.op == Op::Variable) {
            const auto found = variables.find(term.variable);
            if (found == variables.end() || !std::isfinite(found->second)) return std::nullopt;
            return found->second;
        }
        const auto left = run(term.left, depth + 1);
        if (!left) return std::nullopt;
        if (term.op == Op::Not) return *left == 0 ? 1.f : 0.f;
        if (term.op == Op::Negate) return -*left;
        if (term.op == Op::And && *left == 0) return 0.f;
        if (term.op == Op::Or && *left != 0) return 1.f;
        const auto right = run(term.right, depth + 1);
        if (!right) return std::nullopt;
        float result = 0;
        switch (term.op) {
        case Op::Add: result = *left + *right; break;
        case Op::Subtract: result = *left - *right; break;
        case Op::Multiply: result = *left * *right; break;
        case Op::Divide: if (*right == 0) return std::nullopt; result = *left / *right; break;
        case Op::Less: return *left < *right ? 1.f : 0.f;
        case Op::LessEqual: return *left <= *right ? 1.f : 0.f;
        case Op::Greater: return *left > *right ? 1.f : 0.f;
        case Op::GreaterEqual: return *left >= *right ? 1.f : 0.f;
        case Op::Equal: return *left == *right ? 1.f : 0.f;
        case Op::NotEqual: return *left != *right ? 1.f : 0.f;
        case Op::And: case Op::Or: return *right != 0 ? 1.f : 0.f;
        default: return std::nullopt;
        }
        return std::isfinite(result) ? std::optional(result) : std::nullopt;
    };
    return run(root, 0);
}

std::shared_ptr<const BehaviorProgram> compileBehaviorProgram(HkxDecodedBehaviorGraph graph) {
    auto result = std::make_shared<BehaviorProgram>();
    result->graph = std::move(graph);
    result->unsupported = result->graph.unsupportedVariables;
    const auto& nodes = result->graph.nodes;
    std::vector<std::uint8_t> visited(nodes.size());
    std::map<std::string, std::pair<float, std::uint8_t>> clipSettings;
    const auto gap = [&](std::string message) { result->unsupported.push_back(std::move(message)); };
    std::function<void(std::uint32_t, unsigned)> visit = [&](std::uint32_t index, unsigned depth) {
        if (index >= nodes.size()) { gap("invalid node reference"); return; }
        if (depth > 128u) { gap("generator nesting exceeds 128"); return; }
        if (visited[index] == 1) { gap("cyclic generator topology"); return; }
        if (visited[index] == 2) return;
        visited[index] = 1;
        const auto& node = nodes[index];
        const auto nodeGap = [&](const std::string& reason) { gap(node.name + ": " + reason); };
        if (node.enableBindingIndex != -1) nodeGap("enable binding is not executable");
        if (node.hasBindings && node.bindings.empty()) nodeGap("variable bindings were not decoded");
        std::set<std::string> boundMembers;
        for (const auto& binding : node.bindings) {
            if (binding.bindingType != 0 || binding.bitIndex != -1 || binding.variableIndex < 0 ||
                static_cast<std::size_t>(binding.variableIndex) >= result->graph.variableNames.size())
                nodeGap("unsupported variable binding source: " + binding.memberPath);
            if (node.kind != HkxBehaviorNodeKind::ManualSelector || binding.memberPath != "selectedGeneratorIndex")
                nodeGap("unsupported bound member: " + binding.memberPath);
            if (!boundMembers.insert(binding.memberPath).second) nodeGap("duplicate bound member: " + binding.memberPath);
        }
        if (node.hasUnsupportedSettings) nodeGap("authored node settings/notifications are not executable");
        if (node.hasUndecodedChildren) nodeGap("undecoded generator/trigger dependency");
        switch (node.kind) {
        case HkxBehaviorNodeKind::Graph:
        case HkxBehaviorNodeKind::State:
            if (node.children.size() != 1) nodeGap("requires exactly one generator");
            break;
        case HkxBehaviorNodeKind::StateMachine: {
            std::set<std::int32_t> states;
            for (auto child : node.children) {
                if (child >= nodes.size() || nodes[child].kind != HkxBehaviorNodeKind::State ||
                    !states.insert(nodes[child].stateId).second) nodeGap("invalid/duplicate state");
            }
            if (!states.contains(node.startStateId)) nodeGap("missing start state");
            const auto targets = [&](const HkxBehaviorNode& owner) {
                for (const auto& rule : owner.transitions)
                    if (!states.contains(rule.toStateId)) nodeGap("transition target is not a sibling state");
            };
            targets(node);
            for (auto child : node.children) if (child < nodes.size()) targets(nodes[child]);
            break;
        }
        case HkxBehaviorNodeKind::Clip:
            if (const auto [entry, inserted] = clipSettings.emplace(node.assetPath,
                    std::pair{node.playbackSpeed, node.playbackMode}); !inserted &&
                entry->second != std::pair{node.playbackSpeed, node.playbackMode})
                nodeGap("shared clip has differing generator playback settings");
            if (node.assetPath.empty()) nodeGap("empty clip path");
            if (!std::isfinite(node.playbackSpeed) || node.playbackSpeed <= 0 ||
                node.playbackMode > 1 || node.clipFlags || node.cropStart != 0 ||
                node.cropEnd != 0 || node.enforcedDuration != 0 || node.startTime != 0)
                nodeGap("unsupported clip playback settings");
            break;
        case HkxBehaviorNodeKind::TransitionEffect:
            if (!std::isfinite(node.transitionDuration) || node.transitionDuration < 0)
                nodeGap("invalid blend duration");
            break;
        case HkxBehaviorNodeKind::ManualSelector:
            if (node.selectedGeneratorIndex < 0 ||
                static_cast<std::size_t>(node.selectedGeneratorIndex) >= node.children.size())
                nodeGap("invalid selected generator index");
            break;
        default:
            nodeGap("generator/modifier semantics are not executable");
            break;
        }
        for (std::size_t ruleIndex = 0; ruleIndex < node.transitions.size(); ++ruleIndex) {
            const auto& rule = node.transitions[ruleIndex];
            if (rule.flags & HkxBehaviorTransition::Disabled) continue;
            if (rule.hasCondition && !(rule.flags & HkxBehaviorTransition::DisableCondition)) {
                std::string error;
                auto expression = compileBehaviorExpression(rule.conditionExpression, result->graph.variableNames, error);
                if (rule.conditionClass != "hkbExpressionCondition") nodeGap("unsupported condition class: " + rule.conditionClass);
                else if (!expression) nodeGap("condition: " + error);
                else result->conditions.emplace(std::pair{index, ruleIndex}, std::move(*expression));
            }
            constexpr auto supportedFlags = HkxBehaviorTransition::DisableCondition |
                HkxBehaviorTransition::AllowWildcardSelfTransition | HkxBehaviorTransition::LocalWildcard;
            if (rule.flags & ~supportedFlags)
                nodeGap("transition flags/nested override are not executable");
            if (rule.eventId < -1 || (rule.eventId >= 0 &&
                static_cast<std::size_t>(rule.eventId) >= result->graph.eventNames.size()))
                nodeGap("invalid transition event index");
            // Window flags are not yet admitted; never silently ignore authored windows.
            for (const auto* interval : {&rule.triggerInterval, &rule.initiateInterval})
                if (interval->enterEventId != -1 || interval->exitEventId != -1 ||
                    interval->enterTime != 0 || interval->exitTime != 0)
                    nodeGap("transition interval is not executable");
            if (rule.hasEffect) {
                if (rule.effectNode < 0 || static_cast<std::size_t>(rule.effectNode) >= nodes.size() ||
                    nodes[rule.effectNode].kind != HkxBehaviorNodeKind::TransitionEffect)
                    nodeGap("unknown transition effect");
                else visit(static_cast<std::uint32_t>(rule.effectNode), depth + 1);
            }
        }
        for (const auto& trigger : node.triggers) {
            if (trigger.hasPayload) nodeGap("clip trigger payload is not executable");
            if (trigger.eventId < 0 || static_cast<std::size_t>(trigger.eventId) >= result->graph.eventNames.size())
                nodeGap("invalid clip trigger event index");
        }
        for (auto child : node.children) visit(child, depth + 1);
        visited[index] = 2;
    };
    visit(result->graph.rootNode, 0);
    std::sort(result->unsupported.begin(), result->unsupported.end());
    result->unsupported.erase(std::unique(result->unsupported.begin(), result->unsupported.end()),
        result->unsupported.end());
    return result;
}

} // namespace odai::anim
