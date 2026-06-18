#pragma once

#include "ui/document/ui_document.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace odai::ui {

// Polls watched .ui.json files each frame (one GetFileAttributesEx call per
// file — negligible cost). On modification, reloads and fires a callback.
//
// Usage:
//   UiHotReload hr;
//   hr.watch("assets/ui/city_panel.ui.json", loader, ctx,
//            [&](std::unique_ptr<Widget> w) { m_cityPanel = std::move(w); });
//   // each frame:
//   hr.tick();
class UiHotReload {
public:
    using Callback = std::function<void(std::unique_ptr<Widget>)>;

    void watch(const std::filesystem::path& path,
               const UiDocumentLoader& loader,
               const BindingContext& ctx,
               Callback onReload);

    // Call every frame. Detects file changes and fires callbacks.
    void tick();

    void clear() { m_entries.clear(); }

private:
    struct Entry {
        std::filesystem::path path;
        const UiDocumentLoader* loader = nullptr;
        BindingContext ctx;
        Callback callback;
        std::filesystem::file_time_type lastWriteTime{};
    };
    std::vector<Entry> m_entries;
};

}  // namespace odai::ui
