#pragma once

// Poll-based file-change watcher.
//
// Generalized from ui::UiHotReload (src/ui/document/ui_hot_reload.cc), which was
// the only runtime hot-reload mechanism in the tree but is welded to
// UiDocumentLoader and BindingContext. This is the same forty lines of
// mtime-polling with the UI types removed, so anything — materials, themes,
// scripts — can watch a file without pulling in the widget system.
//
// UiHotReload is deliberately left alone: untangling it is a separate refactor,
// and duplicating a poll loop is cheaper than a bad abstraction over both.
//
// Polling, not inotify/ReadDirectoryChangesW: it is portable, it needs no
// threads, and at editor cadence a stat() per watched file per frame is free.
// A save that lands mid-poll is picked up on the next tick.

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace odai::core {

class FileWatch {
public:
    // The callback fires on the polling thread (whoever calls tick()), never
    // concurrently, so watchers can touch caller state without locking.
    void watch(std::filesystem::path path, std::function<void()> onChange);

    // Checks every watched file once. Cheap enough to call every frame.
    void tick();

    void clear();

    [[nodiscard]] std::size_t count() const { return m_entries.size(); }

private:
    struct Entry {
        std::filesystem::path path;
        std::filesystem::file_time_type lastWriteTime{};
        std::function<void()> onChange;
        // True once the file has been seen at least once. A file that does not
        // exist yet is watched anyway and fires when it appears — which is what
        // you want when the editor is about to write it for the first time.
        bool seen = false;
    };

    std::vector<Entry> m_entries;
};

}  // namespace odai::core
