#include "ui/widgets/tes3_journal_panel.h"

#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <utility>

namespace odai::ui {
namespace {

std::string lowerAscii(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string escapeMarkup(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    for (const char ch : text) {
        if (ch == '&') result += "&amp;";
        else if (ch == '<') result += "&lt;";
        else if (ch == '>') result += "&gt;";
        else result.push_back(ch);
    }
    return result;
}

const char* viewName(Tes3JournalPanel::View view) {
    switch (view) {
        case Tes3JournalPanel::View::Chronological: return "Chronological";
        case Tes3JournalPanel::View::Active: return "Active Quests";
        case Tes3JournalPanel::View::Completed: return "Completed Quests";
        case Tes3JournalPanel::View::KnownTopics: return "Known Topics";
    }
    return "Journal";
}

}  // namespace

void Tes3JournalPanel::setJournal(
    std::vector<Entry> entries, std::vector<std::string> knownTopics) {
    entries_ = std::move(entries);
    topics_ = std::move(knownTopics);
    std::stable_sort(entries_.begin(), entries_.end(), [](const Entry& left, const Entry& right) {
        return left.sequence < right.sequence;
    });
    std::sort(topics_.begin(), topics_.end(), [](const std::string& left, const std::string& right) {
        return lowerAscii(left) < lowerAscii(right);
    });
    refilter();
}

void Tes3JournalPanel::setView(View view) {
    view_ = view;
    selected_ = 0u;
    refilter();
}

void Tes3JournalPanel::setSearch(std::string text) {
    search_ = lowerAscii(std::move(text));
    selected_ = 0u;
    refilter();
}

void Tes3JournalPanel::moveSelection(int delta) {
    if (visible_.empty()) {
        selected_ = 0u;
        return;
    }
    const int size = static_cast<int>(visible_.size());
    int next = static_cast<int>(selected_) + delta;
    next %= size;
    if (next < 0) next += size;
    selected_ = static_cast<std::size_t>(next);
}

void Tes3JournalPanel::pinSelected() {
    if (selected_ < visible_.size() && !visible_[selected_].questId.empty()) {
        (void)pinQuest(visible_[selected_].questId);
    }
}

bool Tes3JournalPanel::pinQuest(std::string_view questId) {
    const std::string wanted = lowerAscii(std::string(questId));
    const auto found = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& entry) {
        return lowerAscii(entry.questId) == wanted;
    });
    if (found == entries_.end()) return false;
    pinnedQuest_ = found->questId;
    return true;
}

std::optional<Tes3JournalPanel::Entry> Tes3JournalPanel::latestPinnedEntry() const {
    const std::string wanted = lowerAscii(pinnedQuest_);
    std::optional<Entry> result;
    for (const Entry& entry : entries_) {
        if (lowerAscii(entry.questId) == wanted &&
            (!result.has_value() || entry.sequence > result->sequence)) {
            result = entry;
        }
    }
    return result;
}

void Tes3JournalPanel::refilter() {
    visible_.clear();
    const auto matchesSearch = [&](const Entry& entry) {
        if (search_.empty()) return true;
        return lowerAscii(entry.questId).find(search_) != std::string::npos ||
            lowerAscii(entry.title).find(search_) != std::string::npos ||
            lowerAscii(entry.text).find(search_) != std::string::npos;
    };
    if (view_ == View::KnownTopics) {
        std::uint64_t sequence = 0u;
        for (const std::string& topic : topics_) {
            Entry entry;
            entry.sequence = sequence++;
            entry.title = topic;
            if (matchesSearch(entry)) visible_.push_back(std::move(entry));
        }
    } else if (view_ == View::Chronological) {
        std::copy_if(entries_.begin(), entries_.end(), std::back_inserter(visible_), matchesSearch);
    } else {
        // Quest views show the most recent authored entry for each quest.
        for (const Entry& entry : entries_) {
            const bool stateMatches = view_ == View::Active
                ? entry.state == QuestState::Active
                : entry.state == QuestState::Completed;
            if (!stateMatches || !matchesSearch(entry)) continue;
            const auto existing = std::find_if(visible_.begin(), visible_.end(),
                [&](const Entry& item) { return lowerAscii(item.questId) == lowerAscii(entry.questId); });
            if (existing == visible_.end()) visible_.push_back(entry);
            else if (existing->sequence < entry.sequence) *existing = entry;
        }
        std::sort(visible_.begin(), visible_.end(), [](const Entry& left, const Entry& right) {
            return lowerAscii(left.title.empty() ? left.questId : left.title) <
                lowerAscii(right.title.empty() ? right.questId : right.title);
        });
    }
    if (visible_.empty()) selected_ = 0u;
    else selected_ = std::min(selected_, visible_.size() - 1u);
}

void Tes3JournalPanel::rebuild(const UiRect& rect, float scale) {
    children_.clear();
    setRect(rect);
    const Font* regular = fonts_.regular;
    const Font* bold = fonts_.bold != nullptr ? fonts_.bold : regular;
    const float line = regular ? regular->lineHeightPx() : 28.0f * scale;
    const float padding = 24.0f * scale;

    // Dark carved surround, double brass line and corner ticks. The inner page
    // is deliberately matte parchment so long quest text retains contrast in
    // a bright room and does not inherit the radioactive HUD accent.
    auto frame = std::make_unique<Panel>();
    frame->setRect(rect);
    frame->styleOrnate(scale, 0.99f);
    frame->bgTop = UiColor{0.20f, 0.16f, 0.10f, 0.99f};
    frame->bgBottom = UiColor{0.075f, 0.055f, 0.032f, 0.99f};
    frame->borderColor = {0.12f, 0.09f, 0.05f, 1.0f};
    frame->innerBorderColor = {0.68f, 0.52f, 0.27f, 0.92f};
    frame->cornerAccentColor = {0.79f, 0.64f, 0.36f, 0.95f};
    addChild(std::move(frame));

    const UiRect page{
        rect.minX + (12.0f * scale), rect.minY + (12.0f * scale),
        rect.maxX - (12.0f * scale), rect.maxY - (12.0f * scale)};
    auto parchment = std::make_unique<Panel>();
    parchment->setRect(page);
    parchment->bgTop = UiColor{0.78f, 0.71f, 0.57f, 1.0f};
    parchment->bgBottom = UiColor{0.59f, 0.51f, 0.39f, 1.0f};
    parchment->background = *parchment->bgBottom;
    parchment->borderColor = {0.24f, 0.18f, 0.11f, 0.95f};
    parchment->borderThicknessPx = 1.5f * scale;
    parchment->innerBorderColor = {0.91f, 0.84f, 0.67f, 0.42f};
    parchment->innerBorderInsetPx = 4.0f * scale;
    parchment->showShadow = false;
    addChild(std::move(parchment));

    float y = page.minY + padding * 0.70f;
    auto heading = std::make_unique<Label>(bold, "THE JOURNAL");
    heading->align = UiTextAlign::Center;
    heading->wrap = false;
    heading->color = {0.16f, 0.11f, 0.07f, 1.0f};
    heading->setRect(UiRect::fromXYWH(
        page.minX + padding, y, page.width() - (2.0f * padding), line * 1.2f));
    addChild(std::move(heading));
    y += line * 1.25f;

    // Four wide tabs remain legible from couch distance. The bumper hints live
    // on the same visual axis as the thing they change.
    const float tabGap = 5.0f * scale;
    const float tabsX = page.minX + padding;
    const float tabsWidth = page.width() - (2.0f * padding);
    const float tabWidth = (tabsWidth - (3.0f * tabGap)) / 4.0f;
    const float tabHeight = std::max(46.0f * scale, line * 1.20f);
    static constexpr View kViews[] = {
        View::Chronological, View::Active, View::Completed, View::KnownTopics};
    for (std::size_t tab = 0u; tab < 4u; ++tab) {
        const float x = tabsX + static_cast<float>(tab) * (tabWidth + tabGap);
        const UiRect tabRect = UiRect::fromXYWH(x, y, tabWidth, tabHeight);
        auto tabPanel = std::make_unique<Panel>();
        tabPanel->setRect(tabRect);
        const bool active = view_ == kViews[tab];
        tabPanel->background = active
            ? UiColor{0.25f, 0.18f, 0.10f, 0.96f}
            : UiColor{0.48f, 0.40f, 0.29f, 0.48f};
        tabPanel->borderColor = active
            ? UiColor{0.78f, 0.63f, 0.36f, 1.0f}
            : UiColor{0.25f, 0.19f, 0.12f, 0.62f};
        tabPanel->borderThicknessPx = active ? 2.0f * scale : 1.0f * scale;
        tabPanel->showShadow = false;
        addChild(std::move(tabPanel));
        auto tabLabel = std::make_unique<Label>(active ? bold : regular, viewName(kViews[tab]));
        tabLabel->align = UiTextAlign::Center;
        tabLabel->wrap = false;
        tabLabel->color = active
            ? UiColor{0.96f, 0.88f, 0.70f, 1.0f}
            : UiColor{0.20f, 0.15f, 0.10f, 0.96f};
        tabLabel->padding = {5.0f * scale, 7.0f * scale};
        tabLabel->setRect(tabRect);
        addChild(std::move(tabLabel));
    }
    y += tabHeight + (14.0f * scale);

    if (!search_.empty()) {
        auto searchLabel = std::make_unique<Label>(fonts_,
            "<i>Search: " + escapeMarkup(search_) + "</i>");
        searchLabel->color = {0.27f, 0.20f, 0.13f, 0.82f};
        searchLabel->setRect(UiRect::fromXYWH(
            tabsX, y, tabsWidth, line));
        addChild(std::move(searchLabel));
        y += line;
    }

    const float footerHeight = std::max(42.0f * scale, line * 1.05f);
    const float contentBottom = page.maxY - padding * 0.65f - footerHeight;
    const float gutter = 18.0f * scale;
    const float indexWidth = tabsWidth * 0.35f;
    const UiRect indexRect{tabsX, y, tabsX + indexWidth, contentBottom};
    const UiRect detailRect{
        indexRect.maxX + gutter, y, tabsX + tabsWidth, contentBottom};

    auto indexPage = std::make_unique<Panel>();
    indexPage->setRect(indexRect);
    indexPage->background = {0.39f, 0.31f, 0.21f, 0.16f};
    indexPage->borderColor = {0.25f, 0.18f, 0.11f, 0.58f};
    indexPage->borderThicknessPx = 1.0f * scale;
    indexPage->showShadow = false;
    addChild(std::move(indexPage));
    // Book gutter: a shadow line and a fine highlight make the two panes read
    // as facing pages without wasting controller navigation on another column.
    auto gutterShadow = std::make_unique<Panel>();
    gutterShadow->setRect(UiRect{
        indexRect.maxX + (gutter * 0.43f), y,
        indexRect.maxX + (gutter * 0.56f), contentBottom});
    gutterShadow->background = {0.20f, 0.14f, 0.08f, 0.42f};
    gutterShadow->borderThicknessPx = 0.0f;
    gutterShadow->showShadow = false;
    addChild(std::move(gutterShadow));

    const float rowHeight = std::max(54.0f * scale, line * 1.38f);
    const std::size_t rowCount = std::max<std::size_t>(
        1u, static_cast<std::size_t>(std::max(0.0f, indexRect.height()) / rowHeight));
    std::size_t first = selected_ > rowCount / 2u ? selected_ - rowCount / 2u : 0u;
    if (first + rowCount > visible_.size()) {
        first = visible_.size() > rowCount ? visible_.size() - rowCount : 0u;
    }
    for (std::size_t index = first;
         index < visible_.size() && index < first + rowCount; ++index) {
        const Entry& entry = visible_[index];
        const bool selected = index == selected_;
        const bool pinned = !entry.questId.empty() &&
            lowerAscii(entry.questId) == lowerAscii(pinnedQuest_);
        const float rowY = indexRect.minY +
            static_cast<float>(index - first) * rowHeight;
        const UiRect rowRect{
            indexRect.minX + (5.0f * scale), rowY + (3.0f * scale),
            indexRect.maxX - (5.0f * scale), rowY + rowHeight - (3.0f * scale)};
        if (selected) {
            auto selection = std::make_unique<Panel>();
            selection->setRect(rowRect);
            selection->background = {0.25f, 0.18f, 0.10f, 0.94f};
            selection->borderColor = {0.76f, 0.60f, 0.34f, 0.92f};
            selection->borderThicknessPx = 2.0f * scale;
            selection->showShadow = false;
            addChild(std::move(selection));
        }
        std::string title = escapeMarkup(
            entry.title.empty() ? entry.questId : entry.title);
        if (pinned) title = "* " + title;
        auto label = std::make_unique<Label>(selected ? bold : regular, std::move(title));
        label->wrap = true;
        label->color = selected
            ? UiColor{0.97f, 0.89f, 0.72f, 1.0f}
            : UiColor{0.20f, 0.14f, 0.09f, 0.96f};
        label->padding = {12.0f * scale, 8.0f * scale};
        label->setRect(rowRect);
        addChild(std::move(label));
    }

    if (visible_.empty()) {
        auto empty = std::make_unique<Label>(fonts_, "<i>No entries on this page.</i>");
        empty->align = UiTextAlign::Center;
        empty->color = {0.28f, 0.20f, 0.13f, 0.78f};
        empty->setRect(detailRect);
        addChild(std::move(empty));
    } else {
        const Entry& entry = visible_[selected_];
        const bool pinned = !entry.questId.empty() &&
            lowerAscii(entry.questId) == lowerAscii(pinnedQuest_);
        auto detailTitle = std::make_unique<Label>(fonts_,
            "<b>" + escapeMarkup(entry.title.empty() ? entry.questId : entry.title) + "</b>");
        detailTitle->color = {0.16f, 0.11f, 0.07f, 1.0f};
        detailTitle->setRect(UiRect::fromXYWH(
            detailRect.minX, detailRect.minY, detailRect.width(), line * 1.35f));
        addChild(std::move(detailTitle));

        std::string metadata;
        if (entry.index != 0) metadata = "Entry " + std::to_string(entry.index);
        if (entry.state == QuestState::Active) metadata +=
            (metadata.empty() ? "" : "  -  ") + std::string("Active quest");
        else if (entry.state == QuestState::Completed) metadata +=
            (metadata.empty() ? "" : "  -  ") + std::string("Completed");
        if (pinned) metadata += (metadata.empty() ? "" : "  -  ") + std::string("Pinned");
        if (!metadata.empty()) {
            auto meta = std::make_unique<Label>(fonts_, "<i>" + metadata + "</i>");
            meta->color = {0.31f, 0.22f, 0.14f, 0.76f};
            meta->setRect(UiRect::fromXYWH(
                detailRect.minX, detailRect.minY + line * 1.2f,
                detailRect.width(), line));
            addChild(std::move(meta));
        }
        auto body = std::make_unique<Label>(fonts_, escapeMarkup(entry.text));
        body->color = {0.13f, 0.095f, 0.065f, 0.98f};
        body->padding = {2.0f * scale, 8.0f * scale};
        body->setRect(UiRect{
            detailRect.minX, detailRect.minY + line * 2.25f,
            detailRect.maxX, detailRect.maxY});
        addChild(std::move(body));
    }

    auto footerRule = std::make_unique<Panel>();
    footerRule->setRect(UiRect{
        tabsX, contentBottom + (8.0f * scale), tabsX + tabsWidth,
        contentBottom + (9.5f * scale)});
    footerRule->background = {0.28f, 0.20f, 0.12f, 0.62f};
    footerRule->borderThicknessPx = 0.0f;
    footerRule->showShadow = false;
    addChild(std::move(footerRule));
}

}  // namespace odai::ui
