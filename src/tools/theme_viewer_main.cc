// Theme Viewer — artist tool for iterating on odai_ui theme JSON files.
//
// Usage:  odai_theme_viewer <theme.json>
//
// On start, prints all color, font, size, and frame tokens from the theme.
// Then polls the file every second; when the mtime changes, reloads and
// reprints so artists see their edits without restarting.
//
// Color tokens are shown as ANSI 24-bit color blocks where the terminal
// supports them (Windows Terminal, most Linux terminals).
//
// Theme JSON format — place next to any TTF fonts it references:
//   {
//     "name": "MyTheme",
//     "fonts": {
//       "body":    { "path": "fonts/Regular.ttf",  "size": 18 },
//       "heading": { "path": "fonts/Bold.ttf",     "size": 24 }
//     },
//     "colors": {
//       "text":      "#E8D9B0",
//       "panel.bg":  "#2C1F0E",
//       "accent":    "#C8963A"
//     },
//     "frames": {
//       "Panel.Default": { "texture": "frames/panel.png", "border": [12,12,12,12] }
//     },
//     "sizes": {
//       "padding":   [8, 8],
//       "gap":       6,
//       "titleBarH": 30
//     },
//     "icons": {
//       "sword": { "path": "icons/sword.png" }
//     }
//   }

#include "ui/theme/ui_theme.h"
#include "ui/ui_types.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

namespace {

// Null upload function for the headless theme viewer — we don't need actual GPU
// textures; returning kUiNoTexture is fine for the inspection workflow.
odai::ui::UiTextureId nullUpload(const std::uint8_t*, std::uint32_t, std::uint32_t, bool) {
    return odai::ui::kUiNoTexture;
}

// Emit an ANSI 24-bit background color block followed by a reset.
void printColorSwatch(const odai::ui::UiColor& c) {
    const int r = static_cast<int>(c.r * 255.0f);
    const int g = static_cast<int>(c.g * 255.0f);
    const int b = static_cast<int>(c.b * 255.0f);
    // Foreground auto-contrasted (dark bg → white text, light bg → dark text).
    const float luminance = 0.299f * c.r + 0.587f * c.g + 0.114f * c.b;
    const int fr = luminance > 0.5f ? 0 : 255;
    const int fg = luminance > 0.5f ? 0 : 255;
    const int fb = luminance > 0.5f ? 0 : 255;
    std::printf("\x1b[48;2;%d;%d;%dm\x1b[38;2;%d;%d;%dm  ##  \x1b[0m", r, g, b, fr, fg, fb);
}

// Convert a UiColor to "#RRGGBB" hex for display.
std::string toHex(const odai::ui::UiColor& c) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "#%02X%02X%02X",
        static_cast<int>(c.r * 255.0f),
        static_cast<int>(c.g * 255.0f),
        static_cast<int>(c.b * 255.0f));
    return std::string(buf);
}

void printTheme(const odai::ui::UiTheme& theme, const std::string& path) {
    std::printf("\n\x1b[1m========================================\x1b[0m\n");
    std::printf("\x1b[1mTheme: %s\x1b[0m\n", theme.name().c_str());
    std::printf("File:  %s\n", path.c_str());
    std::printf("\x1b[1m========================================\x1b[0m\n\n");

    // The theme stores tokens in private maps; we rediscover them via a known
    // set of convention-based keys. Artists extending this viewer can add more.
    const char* kColorKeys[] = {
        "text", "text.dim", "text.positive", "text.negative", "text.warning",
        "panel.bg", "panel.border", "panel.shadow",
        "accent", "accent.dim", "accent.bright",
        "button.normal", "button.hover", "button.pressed", "button.disabled",
        "window.titlebar", "window.body", "window.border",
        "tooltip.bg", "tooltip.border",
        "progress.fill", "progress.bg",
        "slider.track", "slider.thumb",
        "input.bg", "input.border", "input.text",
        "tab.active", "tab.inactive",
        "chart.line1", "chart.line2", "chart.line3",
        nullptr
    };

    std::printf("\x1b[1;4mColors\x1b[0m\n");
    std::printf("  (white swatch = token not defined in theme)\n");
    for (int i = 0; kColorKeys[i] != nullptr; ++i) {
        const odai::ui::UiColor c = theme.color(kColorKeys[i]);
        printColorSwatch(c);
        std::printf("  %-28s  %s  (a=%.2f)\n", kColorKeys[i], toHex(c).c_str(), c.a);
    }

    std::printf("\n\x1b[1;4mFonts\x1b[0m\n");
    const char* kFontKeys[] = { "body", "bold", "italic", "boldItalic", "heading", "mono", nullptr };
    bool anyFont = false;
    for (int i = 0; kFontKeys[i] != nullptr; ++i) {
        const odai::ui::Font* f = theme.font(kFontKeys[i]);
        if (!f) continue;
        anyFont = true;
        std::printf("  %-16s  ascent=%.1fpx  descent=%.1fpx  lineH=%.1fpx\n",
            kFontKeys[i],
            f->ascentPx(),
            f->descentPx(),
            f->lineHeightPx());
    }
    if (!anyFont) {
        std::printf("  (no fonts loaded — check that TTF paths are correct)\n");
    }

    std::printf("\n\x1b[1;4mSizes\x1b[0m\n");
    const char* kSizeKeys[] = {
        "padding", "gap", "titleBarH", "borderRadius", "iconSize",
        "rowH", "sectionPad", "scrollbarW", nullptr
    };
    bool anySize = false;
    for (int i = 0; kSizeKeys[i] != nullptr; ++i) {
        const float s = theme.size(kSizeKeys[i]);
        const odai::ui::UiVec2 sv = theme.sizeVec2(kSizeKeys[i]);
        if (s == 0.0f && sv.x == 0.0f && sv.y == 0.0f) continue;
        anySize = true;
        if (sv.x != 0.0f || sv.y != 0.0f) {
            std::printf("  %-20s  [%.1f, %.1f]\n", kSizeKeys[i], sv.x, sv.y);
        } else {
            std::printf("  %-20s  %.1f\n", kSizeKeys[i], s);
        }
    }
    if (!anySize) {
        std::printf("  (no recognized size keys)\n");
    }

    std::printf("\n\x1b[1;4mFrames (9-slice)\x1b[0m\n");
    const char* kFrameKeys[] = {
        "Panel.Default", "Window.Default", "Tooltip", "Button.Normal",
        "Button.Hover", "Button.Pressed", "Input.Default", nullptr
    };
    bool anyFrame = false;
    for (int i = 0; kFrameKeys[i] != nullptr; ++i) {
        const auto fr = theme.frame(kFrameKeys[i]);
        if (!fr) continue;
        anyFrame = true;
        std::printf("  %-20s  border=[%.0f %.0f %.0f %.0f]  texId=%u\n",
            kFrameKeys[i],
            fr->borderLeftPx, fr->borderTopPx, fr->borderRightPx, fr->borderBottomPx,
            fr->textureId);
    }
    if (!anyFrame) {
        std::printf("  (no recognized frame keys)\n");
    }

    std::printf("\n\x1b[2mWatching for changes... (Ctrl+C to exit)\x1b[0m\n");
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: odai_theme_viewer <theme.json>\n\n"
            "Loads the theme, prints all tokens with ANSI color swatches,\n"
            "then hot-reloads and reprints whenever the file changes.\n");
        return 1;
    }

    const std::string path = argv[1];
    if (!std::filesystem::exists(path)) {
        std::fprintf(stderr, "Error: file not found: %s\n", path.c_str());
        return 1;
    }

    std::filesystem::file_time_type lastWriteTime{};

    while (true) {
        std::error_code ec;
        const auto mtime = std::filesystem::last_write_time(path, ec);
        if (!ec && mtime != lastWriteTime) {
            lastWriteTime = mtime;

            odai::ui::UiTheme theme;
            if (theme.loadFromFile(path, nullUpload)) {
                printTheme(theme, path);
            } else {
                std::fprintf(stderr, "Failed to load theme: %s\n", path.c_str());
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
