#include "ui/widgets/dialogue_panel.h"

#include "ui/widgets/button.h"
#include "ui/widgets/label.h"
#include "ui/widgets/panel.h"
#include "ui/widgets/rich_text_view.h"

#include <algorithm>
#include <utility>

namespace odai::ui {

void DialoguePanel::setContent(const UiRect& rect, float s, const Line& line,
                                const std::vector<ChoiceEntry>& choices) {
    children_.clear();
    choiceButtons_.clear();
    speakerLabel_ = nullptr;
    bodyView_ = nullptr;
    setRect(rect);

    const float pad = 14.0f * s;

    auto bg = std::make_unique<Panel>();
    bg->setRect(rect);
    bg->styleOrnate(s);
    bg_ = static_cast<Panel*>(addChild(std::move(bg)));

    const Font* boldFont = fonts_.bold ? fonts_.bold : fonts_.regular;
    const float speakerH = (line.speaker.empty() || boldFont == nullptr)
                                ? 0.0f
                                : boldFont->lineHeightPx() + 6.0f * s;

    float y = rect.minY + pad;
    if (speakerH > 0.0f) {
        auto speaker = std::make_unique<Label>(fonts_, "<b><color=#e6d6a4>" + line.speaker + "</color></b>");
        speaker->setRect(UiRect::fromXYWH(rect.minX + pad, y, rect.width() - 2.0f * pad, speakerH));
        speakerLabel_ = static_cast<Label*>(addChild(std::move(speaker)));
        y += speakerH;
    }

    const float btnH = 34.0f * s;
    const float btnGap = 6.0f * s;
    const float choicesBandH = choices.empty()
                                    ? 0.0f
                                    : static_cast<float>(choices.size()) * btnH +
                                          static_cast<float>(choices.size() - 1) * btnGap + 10.0f * s;
    const float bodyBottom = rect.maxY - pad - choicesBandH;

    auto body = std::make_unique<RichTextView>(fonts_, line.text);
    body->padding = {4.0f * s, 2.0f * s};
    body->scrollBarThumbColor = UiColor{0.70f, 0.56f, 0.28f, 0.6f};
    body->scrollBarTrackColor = UiColor{0.0f, 0.0f, 0.0f, 0.15f};
    body->setRect(UiRect::fromXYWH(rect.minX + pad, y, rect.width() - 2.0f * pad,
                                    std::max(0.0f, bodyBottom - y)));
    bodyView_ = static_cast<RichTextView*>(addChild(std::move(body)));

    const Font* btnFont = fonts_.regular ? fonts_.regular : fonts_.bold;
    float btnY = bodyBottom + 10.0f * s;
    for (const ChoiceEntry& choice : choices) {
        std::function<void()> onChoose = choice.onChoose;
        auto btn = std::make_unique<Button>(btnFont, choice.text, [onChoose]() {
            if (onChoose) onChoose();
        });
        btn->setRect(UiRect::fromXYWH(rect.minX + pad, btnY, rect.width() - 2.0f * pad, btnH));
        choiceButtons_.push_back(addChild(std::move(btn)));
        btnY += btnH + btnGap;
    }
}

}  // namespace odai::ui
