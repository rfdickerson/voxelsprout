#pragma once
#include <string>
#include <vector>

namespace odai::bethesda {
class BethesdaSession;
struct Tes3LevelUiInput { bool up = false; bool down = false; bool accept = false; };
// Same input intents drive the retained UI and headless verification. Selection
// is a preview only; reload safely starts it again from unchanged session stats.
class Tes3LevelUiFlow {
public:
    bool update(const Tes3LevelUiInput&, BethesdaSession&, std::string& error);
    [[nodiscard]] int cursor() const { return m_cursor; }
    [[nodiscard]] const std::vector<int>& selected() const { return m_selected; }
    void reset() { m_cursor = 0; m_selected.clear(); m_open = false; }
private:
    bool m_open = false;
    int m_cursor = 0;
    std::vector<int> m_selected;
};
}
