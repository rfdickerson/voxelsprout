#include "ui/document/ui_hot_reload.h"

#include <system_error>

namespace odai::ui {

void UiHotReload::watch(const std::filesystem::path& path,
                         const UiDocumentLoader& loader,
                         const BindingContext& ctx,
                         Callback onReload) {
    Entry e;
    e.path = path;
    e.loader = &loader;
    e.ctx = ctx;
    e.callback = std::move(onReload);

    std::error_code ec;
    e.lastWriteTime = std::filesystem::last_write_time(path, ec);

    m_entries.push_back(std::move(e));
}

void UiHotReload::tick() {
    for (Entry& e : m_entries) {
        if (!e.loader || !e.callback) continue;

        std::error_code ec;
        const auto mtime = std::filesystem::last_write_time(e.path, ec);
        if (ec || mtime == e.lastWriteTime) continue;

        e.lastWriteTime = mtime;

        auto w = e.loader->load(e.path, e.ctx);
        if (w) {
            e.callback(std::move(w));
        }
    }
}

}  // namespace odai::ui
