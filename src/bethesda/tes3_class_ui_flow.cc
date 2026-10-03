#include "bethesda/tes3_class_ui_flow.h"
#include "bethesda/bethesda_session.h"
#include <algorithm>

namespace odai::bethesda {
void Tes3ClassUiFlow::reset() {
    m_stage = m_cursor = 0;
    m_draft = {};
    m_draft.specialization = -1;
    m_draft.attributes.fill(-1);
    m_draft.major.fill(-1);
    m_draft.minor.fill(-1);
}
std::string Tes3ClassUiFlow::prompt() const {
    if (m_stage == 0) return "Choose specialization";
    if (m_stage <= 2) return "Favored attribute " + std::to_string(m_stage) + " of 2";
    if (m_stage <= 7) return "Major skill " + std::to_string(m_stage - 2) + " of 5";
    if (m_stage <= 12) return "Minor skill " + std::to_string(m_stage - 7) + " of 5";
    return "Review custom class";
}
std::vector<int> Tes3ClassUiFlow::choices() const {
    if (m_stage == 13) return {};
    std::vector<int> result;
    const int limit = m_stage == 0 ? 3 : m_stage <= 2 ? 8 : 27;
    for (int i = 0; i < limit; ++i) {
        if (m_stage == 2 && i == m_draft.attributes[0]) continue;
        if (m_stage >= 3 &&
            (std::find(m_draft.major.begin(), m_draft.major.end(), i) != m_draft.major.end() ||
             std::find(m_draft.minor.begin(), m_draft.minor.end(), i) != m_draft.minor.end())) continue;
        result.push_back(i);
    }
    return result;
}
std::string Tes3ClassUiFlow::label(int value) const {
    if (m_stage == 0) {
        constexpr const char* labels[]{"Combat", "Magic", "Stealth"};
        return value >= 0 && value < 3 ? labels[value] : "";
    }
    if (m_stage <= 2) return value >= 0 && value < 8 ? std::string(tes3AttributeNames[value]) : "";
    return value >= 0 && value < 27 ? std::string(tes3SkillNames[value]) : "";
}
bool Tes3ClassUiFlow::update(const Tes3ClassUiInput& input, BethesdaSession& session) {
    if (input.back) {
        if (m_stage == 0) return false;
        --m_stage; m_cursor = 0;
        if (m_stage == 0) m_draft.specialization = -1;
        else if (m_stage <= 2) m_draft.attributes[m_stage - 1] = -1;
        else if (m_stage <= 7) m_draft.major[m_stage - 3] = -1;
        else m_draft.minor[m_stage - 8] = -1;
        return false;
    }
    if (m_stage == 13) {
        if (!input.accept || !validTes3ProgressionClass(m_draft)) return false;
        auto& player = session.tes3().playerState();
        player.actorClass = "Custom Class";
        player.progression.customClass = m_draft;
        session.tes3().synchronizePlayerDialogue();
        reset();
        return true;
    }
    const auto options = choices();
    if (options.empty()) return false;
    if (input.up) m_cursor = (m_cursor + int(options.size()) - 1) % int(options.size());
    if (input.down) m_cursor = (m_cursor + 1) % int(options.size());
    if (!input.accept) return false;
    const int value = options[m_cursor];
    if (m_stage == 0) m_draft.specialization = value;
    else if (m_stage <= 2) m_draft.attributes[m_stage - 1] = value;
    else if (m_stage <= 7) m_draft.major[m_stage - 3] = value;
    else m_draft.minor[m_stage - 8] = value;
    ++m_stage; m_cursor = 0;
    return false;
}
}
