#include "ui/toast_host.h"

#include "ui/animation.h"

#include <algorithm>

namespace odai::ui {

const std::string ToastHost::kEmpty;

ToastStyle makeBannerStyle() {
    ToastStyle style{};
    style.placement = ToastPlacement::ScreenCenter;
    style.drawChrome = false;
    style.accentBarPx = 0.0f;
    // Near-white rather than the Pip-Boy green the corner toasts use. A banner
    // sits directly on the world with no panel behind it, so it has to hold
    // contrast against desert sand, night sky and interior gloom alike; a
    // saturated green does not, and tinting it to suit one of those loses the
    // others.
    style.title = UiColor{0.96f, 0.96f, 0.94f, 1.00f};
    style.body = UiColor{0.82f, 0.82f, 0.78f, 1.00f};
    style.bodyLeadingScale = 1.45f;
    return style;
}

ToastTiming makeBannerTiming() {
    ToastTiming timing{};
    // Slow on both ends. This is the whole difference between "a notification
    // appeared" and "something happened in the world": a 0.2s pop reads as UI,
    // a 1.2s bloom reads as an event. Skyrim's location banner is in this
    // range and it is why it feels like part of the game rather than part of
    // the menu.
    timing.fadeInSeconds = 1.10f;
    timing.holdSeconds = 2.20f;
    timing.fadeOutSeconds = 1.60f;
    // No slide. Motion across the screen fights the fade at this size and
    // draws the eye to the movement rather than the words.
    timing.slideDistancePx = 0.0f;
    return timing;
}

void ToastHost::push(ToastDesc desc) {
    // Coalesce against anything already on screen with the same key: restart
    // its clock rather than queue a second copy. Without this, standing on a
    // cell boundary re-announces the same region every time the streamer
    // changes its mind about which cell you are in.
    if (!desc.key.empty()) {
        for (Live& live : m_visible) {
            if (live.desc.key == desc.key) {
                live.desc = std::move(desc);
                // Back to Hold rather than FadeIn: the toast is already on
                // screen, and replaying the slide would read as a second,
                // different notification.
                live.phase = Phase::Hold;
                live.phaseSeconds = 0.0f;
                live.presence = 1.0f;
                return;
            }
        }
        for (ToastDesc& queued : m_queued) {
            if (queued.key == desc.key) {
                queued = std::move(desc);
                return;
            }
        }
    }
    m_queued.push_back(std::move(desc));
}

void ToastHost::update(float deltaSeconds) {
    if (deltaSeconds < 0.0f) {
        deltaSeconds = 0.0f;
    }

    for (Live& toast : m_visible) {
        toast.phaseSeconds += deltaSeconds;
        switch (toast.phase) {
            case Phase::FadeIn: {
                const float duration = std::max(m_timing.fadeInSeconds, 1e-4f);
                const float t = std::min(toast.phaseSeconds / duration, 1.0f);
                toast.presence = applyEasing(Easing::CubicOut, t);
                if (t >= 1.0f) {
                    toast.phase = Phase::Hold;
                    toast.phaseSeconds = 0.0f;
                    toast.presence = 1.0f;
                }
                break;
            }
            case Phase::Hold: {
                toast.presence = 1.0f;
                if (toast.phaseSeconds >= m_timing.holdSeconds) {
                    toast.phase = Phase::FadeOut;
                    toast.phaseSeconds = 0.0f;
                }
                break;
            }
            case Phase::FadeOut: {
                const float duration = std::max(m_timing.fadeOutSeconds, 1e-4f);
                const float t = std::min(toast.phaseSeconds / duration, 1.0f);
                toast.presence = 1.0f - applyEasing(Easing::CubicIn, t);
                break;
            }
        }
    }

    // Retire finished toasts. erase-remove rather than an index loop so a toast
    // expiring in the same frame another is admitted cannot shuffle indices
    // underneath the admission below.
    std::erase_if(m_visible, [](const Live& toast) {
        return toast.phase == Phase::FadeOut && toast.presence <= 0.0f;
    });

    // Admit queued toasts into the freed slots, oldest first.
    while (m_visible.size() < m_maxVisible && !m_queued.empty()) {
        Live toast;
        toast.desc = std::move(m_queued.front());
        m_queued.pop_front();
        m_visible.push_back(std::move(toast));
    }
}

void ToastHost::clear() {
    m_visible.clear();
    m_queued.clear();
}

const std::string& ToastHost::visibleTitle(std::size_t index) const {
    if (index >= m_visible.size()) {
        return kEmpty;
    }
    return m_visible[index].desc.title;
}

float ToastHost::toastHeightPx(
    const Live& toast, const Font& titleFont, const Font& bodyFont, float scale) const {
    float height = titleFont.lineHeightPx();
    if (!toast.desc.body.empty()) {
        height += bodyFont.lineHeightPx() * m_style.bodyLeadingScale;
    }
    return height + (m_style.paddingPx * 2.0f * scale);
}

void ToastHost::draw(
    UiDrawList& drawList, const Font& titleFont, const Font& bodyFont, const UiRect& screen,
    float scale, float marginPx) const {
    if (m_visible.empty()) {
        return;
    }
    if (m_style.placement == ToastPlacement::ScreenCenter) {
        drawBanner(drawList, titleFont, bodyFont, screen, scale);
        return;
    }
    drawCorner(drawList, titleFont, bodyFont, screen, scale, marginPx);
}

void ToastHost::drawCorner(
    UiDrawList& drawList, const Font& titleFont, const Font& bodyFont, const UiRect& screen,
    float scale, float marginPx) const {
    const float width = m_style.widthPx * scale;
    const float padding = m_style.paddingPx * scale;
    const float spacing = m_style.spacingPx * scale;
    const float radius = m_style.cornerRadiusPx * scale;
    const float margin = marginPx * scale;
    const float right = screen.maxX - margin;
    float cursorY = screen.minY + margin;

    for (const Live& toast : m_visible) {
        const float height = toastHeightPx(toast, titleFont, bodyFont, scale);
        // Slide in from the right by the remaining presence. Using (1 -
        // presence) rather than a separate tween keeps motion and opacity
        // locked together, so a toast never sits half-transparent and still.
        const float slide = (1.0f - toast.presence) * m_timing.slideDistancePx * scale;
        const float alpha = toast.presence;

        UiRect rect{};
        rect.minX = right - width + slide;
        rect.maxX = right + slide;
        rect.minY = cursorY;
        rect.maxY = cursorY + height;

        const auto withAlpha = [alpha](UiColor color) {
            color.a *= alpha;
            return color;
        };

        float textLeft = rect.minX + padding;
        if (m_style.drawChrome) {
            drawList.addRoundRectFilled(rect, withAlpha(m_style.panel), radius);
            drawList.addRoundRect(
                rect, withAlpha(m_style.border), radius, m_style.borderThicknessPx * scale);
            if (m_style.accentBarPx > 0.0f) {
                UiRect accent{};
                accent.minX = rect.minX + (2.0f * scale);
                accent.maxX = accent.minX + (m_style.accentBarPx * scale);
                accent.minY = rect.minY + (4.0f * scale);
                accent.maxY = rect.maxY - (4.0f * scale);
                drawList.addRectFilled(accent, withAlpha(m_style.accent));
                textLeft = accent.maxX + padding;
            }
        }

        UiVec2 titlePosition{textLeft, rect.minY + padding};
        drawList.addText(titleFont, toast.desc.title, titlePosition, withAlpha(m_style.title));
        if (!toast.desc.body.empty()) {
            UiVec2 bodyPosition{
                textLeft, titlePosition.y + (titleFont.lineHeightPx() * m_style.bodyLeadingScale)};
            drawList.addText(bodyFont, toast.desc.body, bodyPosition, withAlpha(m_style.body));
        }

        cursorY += height + spacing;
    }
}

void ToastHost::drawBanner(
    UiDrawList& drawList, const Font& titleFont, const Font& bodyFont, const UiRect& screen,
    float scale) const {
    // ONE banner, ever -- the newest. Stacking centred announcements would put
    // two pieces of large type on top of each other in the middle of the
    // screen; the rest of the queue simply waits, which is what update()
    // already does for corner toasts too.
    const Live& toast = m_visible.front();
    const float alpha = toast.presence;
    const auto withAlpha = [alpha](UiColor color) {
        color.a *= alpha;
        return color;
    };

    const float centerX = (screen.minX + screen.maxX) * 0.5f;
    // Slightly above centre. Dead centre collides with whatever the player is
    // actually looking at (and with a crosshair); the upper third is where the
    // eye already rests and where every game that does this puts it.
    const float centerY = screen.minY + ((screen.maxY - screen.minY) * 0.38f);

    const float titleWidth = titleFont.measureText(toast.desc.title);
    UiVec2 titlePosition{centerX - (titleWidth * 0.5f), centerY};
    drawList.addText(titleFont, toast.desc.title, titlePosition, withAlpha(m_style.title));

    if (!toast.desc.body.empty()) {
        const float bodyWidth = bodyFont.measureText(toast.desc.body);
        UiVec2 bodyPosition{
            centerX - (bodyWidth * 0.5f),
            centerY + (titleFont.lineHeightPx() * m_style.bodyLeadingScale)};
        drawList.addText(bodyFont, toast.desc.body, bodyPosition, withAlpha(m_style.body));
    }

    if (m_style.drawChrome) {
        // Not the default for a banner, but honoured if a caller turns it on.
        UiRect rect{};
        const float padding = m_style.paddingPx * scale;
        rect.minX = titlePosition.x - padding;
        rect.maxX = titlePosition.x + titleWidth + padding;
        rect.minY = centerY - padding;
        rect.maxY = centerY + toastHeightPx(toast, titleFont, bodyFont, scale);
        drawList.addRoundRect(
            rect, withAlpha(m_style.border), m_style.cornerRadiusPx * scale,
            m_style.borderThicknessPx * scale);
    }
}

}  // namespace odai::ui
