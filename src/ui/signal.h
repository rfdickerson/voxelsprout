#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Lightweight signal/slot system for the UI framework.
//
// Signal<Args...>: a multi-listener broadcast. connect() adds a listener;
// emit() fires all listeners in connection order. clear() disconnects all.
//
// SlotRegistry: maps string slot names to void() callables. wire(root) walks
// a widget tree and connects each widget's slotName to its activated signal.
// Use this to wire JSON-authored "on_click" strings to game callbacks:
//
//   SlotRegistry reg;
//   reg.on("build_farm", [&]{ game.buildFarm(city); });
//   reg.wire(*root);
//
// JSON schema:
//   { "type": "Button", "id": "build_btn", "on_click": "build_farm", ... }

namespace odai::ui {

class Widget;

template<typename... Args>
class Signal {
public:
    using Fn = std::function<void(Args...)>;

    void connect(Fn fn) { m_listeners.push_back(std::move(fn)); }
    void emit(Args... args) const {
        auto snapshot = m_listeners;  // safe if a listener calls connect/clear
        for (const Fn& fn : snapshot) fn(args...);
    }
    void clear() { m_listeners.clear(); }
    [[nodiscard]] bool empty() const { return m_listeners.empty(); }

private:
    std::vector<Fn> m_listeners;
};

class SlotRegistry {
public:
    // Register a named callback. Overwrites any existing registration.
    void on(std::string slotName, std::function<void()> fn) {
        m_slots[std::move(slotName)] = std::move(fn);
    }

    // Walk the widget tree rooted at `root` and connect each widget whose
    // slotName is non-empty to the registered callable. Unregistered names
    // are silently ignored (keeps JSON forward-compatible).
    void wire(Widget& root) const;

private:
    std::unordered_map<std::string, std::function<void()>> m_slots;
};

}  // namespace odai::ui
