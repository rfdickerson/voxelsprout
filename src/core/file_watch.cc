#include "core/file_watch.h"

#include <system_error>
#include <utility>

namespace odai::core {

void FileWatch::watch(std::filesystem::path path, std::function<void()> onChange) {
    if (path.empty() || !onChange) {
        return;
    }
    Entry entry;
    entry.path = std::move(path);
    entry.onChange = std::move(onChange);
    // Seed from the current mtime so registering a watch does not immediately
    // fire on a file that has not changed. A file that does not exist yet stays
    // unseen and fires the first time it appears.
    std::error_code ec;
    const auto mtime = std::filesystem::last_write_time(entry.path, ec);
    if (!ec) {
        entry.lastWriteTime = mtime;
        entry.seen = true;
    }
    m_entries.push_back(std::move(entry));
}

void FileWatch::tick() {
    for (Entry& entry : m_entries) {
        std::error_code ec;
        const auto mtime = std::filesystem::last_write_time(entry.path, ec);
        if (ec) {
            // Vanished, or being rewritten right now. Mark it unseen so the
            // next successful stat counts as a change — an editor that saves by
            // rename briefly leaves no file at the path, and that must not eat
            // the update.
            entry.seen = false;
            continue;
        }
        if (entry.seen && mtime == entry.lastWriteTime) {
            continue;
        }
        entry.lastWriteTime = mtime;
        entry.seen = true;
        entry.onChange();
    }
}

void FileWatch::clear() {
    m_entries.clear();
}

}  // namespace odai::core
