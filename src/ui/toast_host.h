#pragma once

#include "ui/font.h"
#include "ui/ui_draw_list.h"
#include "ui/ui_types.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

// Transient corner notifications -- "Mojave Outpost discovered", "Quest
// updated", "Item added".
//
// Vulkan-free and platform-free like the rest of src/ui/: this owns timing,
// stacking and layout, and emits geometry into a UiDrawList. All of the
// interesting behaviour (queueing, coalescing, expiry) is plain data that a
// headless test drives with a fake clock, which is why update() takes a delta
// rather than reading one.
namespace odai::ui {

// Where a notification appears, and therefore what shape it takes.
//
// These are two genuinely different notification idioms, not one with a moved
// anchor. A corner toast is chrome: panelled, stackable, safe to show three of
// while the player keeps playing. A centre banner is an ANNOUNCEMENT -- the
// thing Skyrim does when you find a new location -- and it works precisely
// because it has no panel, no stack and no border: large type fading up over
// the world and back out, occupying the middle of the screen for a moment.
// Give a banner a panel and it stops reading as part of the world; stack two
// and neither gets read at all.
enum class ToastPlacement : std::uint8_t {
    TopRight,      // Corner toasts, stacking downward.
    ScreenCenter,  // One large announcement, centred, no chrome.
};

struct ToastStyle {
    // Fallout's Pip-Boy phosphor green. Overridable because this widget is not
    // Fallout-specific -- it just ships with a palette that suits the first
    // game using it.
    UiColor panel{0.02f, 0.06f, 0.03f, 0.86f};
    UiColor border{0.35f, 0.98f, 0.45f, 0.90f};
    UiColor title{0.62f, 1.00f, 0.68f, 1.00f};
    UiColor body{0.38f, 0.86f, 0.45f, 1.00f};
    UiColor accent{0.35f, 0.98f, 0.45f, 1.00f};

    float widthPx = 320.0f;
    float paddingPx = 12.0f;
    float spacingPx = 8.0f;
    float cornerRadiusPx = 3.0f;
    float borderThicknessPx = 1.0f;
    // Width of the accent bar down the leading edge. Zero removes it.
    float accentBarPx = 3.0f;

    ToastPlacement placement = ToastPlacement::TopRight;
    // Panel, border and accent bar. Off for a centre banner, where the type is
    // meant to sit on the world rather than in a box.
    bool drawChrome = true;
    // Extra leading between the title and the body line, as a multiple of the
    // body font's line height. A banner wants the subtitle to breathe; a corner
    // toast wants it tight.
    float bodyLeadingScale = 1.0f;
};


struct ToastTiming {
    float fadeInSeconds = 0.22f;
    float holdSeconds = 4.0f;
    float fadeOutSeconds = 0.45f;
    // How far a toast slides in from, in pixels. Motion is what makes a new
    // toast register in peripheral vision; a fade alone is easy to miss while
    // the player is looking at the centre of the screen.
    float slideDistancePx = 28.0f;
};
// The look Skyrim-style location discovery wants: big centred type, no box,
// and a fade slow enough to register as an event rather than a popup.
[[nodiscard]] ToastStyle makeBannerStyle();
[[nodiscard]] ToastTiming makeBannerTiming();

struct ToastDesc {
    std::string title;
    std::string body;
    // Toasts with the same non-empty key coalesce: pushing one while an
    // identical key is still alive restarts that toast instead of stacking a
    // duplicate. Region discovery needs this -- walking back and forth across a
    // cell boundary would otherwise queue "Nipton" over and over.
    std::string key;
};

class ToastHost {
public:
    ToastHost() = default;
    explicit ToastHost(ToastStyle style, ToastTiming timing = {})
        : m_style(style), m_timing(timing) {}

    void setStyle(const ToastStyle& style) { m_style = style; }
    void setTiming(const ToastTiming& timing) { m_timing = timing; }
    [[nodiscard]] const ToastStyle& style() const { return m_style; }

    // At most this many are on screen at once; the rest wait their turn. A
    // stack that grows without bound covers the game.
    void setMaxVisible(std::size_t count) { m_maxVisible = count == 0 ? 1 : count; }

    void push(ToastDesc desc);

    // Convenience for the common two-line shape.
    void push(std::string title, std::string body, std::string key = {}) {
        push(ToastDesc{std::move(title), std::move(body), std::move(key)});
    }

    void update(float deltaSeconds);
    void clear();

    // Emits the visible stack. `screen` is the full framebuffer rect.
    //
    // Title and body take separate fonts because the two placements want
    // different type: a corner toast reads fine in the body face, and a centre
    // banner needs the display face to carry the screen. Pass the same font
    // twice when they should match.
    void draw(UiDrawList& drawList, const Font& titleFont, const Font& bodyFont,
              const UiRect& screen, float scale = 1.0f, float marginPx = 24.0f) const;

    // Same font for both, for callers with only one face.
    void draw(UiDrawList& drawList, const Font& font, const UiRect& screen,
              float scale = 1.0f, float marginPx = 24.0f) const {
        draw(drawList, font, font, screen, scale, marginPx);
    }

    [[nodiscard]] std::size_t visibleCount() const { return m_visible.size(); }
    [[nodiscard]] std::size_t queuedCount() const { return m_queued.size(); }
    // Title of the nth visible toast, newest last. For tests and debug readouts.
    [[nodiscard]] const std::string& visibleTitle(std::size_t index) const;

private:
    enum class Phase : std::uint8_t { FadeIn, Hold, FadeOut };

    struct Live {
        ToastDesc desc;
        Phase phase = Phase::FadeIn;
        float phaseSeconds = 0.0f;
        // Eased 0..1, driving both opacity and the slide offset.
        float presence = 0.0f;
    };

    [[nodiscard]] float toastHeightPx(
        const Live& toast, const Font& titleFont, const Font& bodyFont, float scale) const;
    void drawCorner(UiDrawList& drawList, const Font& titleFont, const Font& bodyFont,
                    const UiRect& screen, float scale, float marginPx) const;
    void drawBanner(UiDrawList& drawList, const Font& titleFont, const Font& bodyFont,
                    const UiRect& screen, float scale) const;

    ToastStyle m_style{};
    ToastTiming m_timing{};
    std::size_t m_maxVisible = 4;
    std::vector<Live> m_visible;
    std::deque<ToastDesc> m_queued;
    static const std::string kEmpty;
};

}  // namespace odai::ui
