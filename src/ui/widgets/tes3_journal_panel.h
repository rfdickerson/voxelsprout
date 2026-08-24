#pragma once

#include "ui/font.h"
#include "ui/widget.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace odai::ui {

// Pure presentation model for the native TES3 journal. The Bethesda runtime
// supplies authored journal text and classification; this widget never guesses
// Skyrim-style objectives for Morrowind quests.
class Tes3JournalPanel : public Widget {
public:
    enum class View : std::uint8_t { Chronological, Active, Completed, KnownTopics };
    enum class QuestState : std::uint8_t { Legacy, Active, Completed };

    struct Entry {
        std::uint64_t sequence = 0u;
        std::string questId;
        std::string title;
        std::string text;
        std::int32_t index = 0;
        QuestState state = QuestState::Legacy;
        bool hasStatusFlags = false;
    };

    explicit Tes3JournalPanel(FontSet fonts) : fonts_(fonts) {}

    void setJournal(std::vector<Entry> entries, std::vector<std::string> knownTopics);
    void setView(View view);
    [[nodiscard]] View view() const { return view_; }
    void setSearch(std::string text);
    [[nodiscard]] const std::string& search() const { return search_; }
    void moveSelection(int delta);
    void pinSelected();
    bool pinQuest(std::string_view questId);
    [[nodiscard]] const std::string& pinnedQuest() const { return pinnedQuest_; }
    [[nodiscard]] std::optional<Entry> latestPinnedEntry() const;
    [[nodiscard]] std::size_t visibleCount() const { return visible_.size(); }
    [[nodiscard]] std::size_t selectedIndex() const { return selected_; }
    [[nodiscard]] const std::vector<Entry>& visibleEntries() const { return visible_; }

    // Rebuilds the retained children after data, view, search, or layout changes.
    void rebuild(const UiRect& rect, float scale);

private:
    void refilter();

    FontSet fonts_;
    std::vector<Entry> entries_;
    std::vector<std::string> topics_;
    std::vector<Entry> visible_;
    View view_ = View::Chronological;
    std::string search_;
    std::string pinnedQuest_;
    std::size_t selected_ = 0u;
};

}  // namespace odai::ui
