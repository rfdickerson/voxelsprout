#pragma once

#include "bethesda/tes3_progression.h"
#include <string>
#include <vector>

namespace odai::bethesda {
class BethesdaSession;
struct Tes3ClassUiInput { bool up = false, down = false, accept = false, back = false; };
// A character-generation preview. Only the final review commits the class;
// creation of starting stats remains the character-review operation.
class Tes3ClassUiFlow {
public:
    Tes3ClassUiFlow() { reset(); }
    void reset();
    bool update(const Tes3ClassUiInput&, BethesdaSession&);
    [[nodiscard]] int stage() const { return m_stage; }
    [[nodiscard]] int cursor() const { return m_cursor; }
    [[nodiscard]] const Tes3ProgressionClass& draft() const { return m_draft; }
    [[nodiscard]] std::string prompt() const;
    [[nodiscard]] std::vector<int> choices() const;
    [[nodiscard]] std::string label(int value) const;
private:
    Tes3ProgressionClass m_draft;
    int m_stage = 0, m_cursor = 0;
};
}
