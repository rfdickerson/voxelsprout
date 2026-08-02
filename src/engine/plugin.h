#pragma once

#include <memory>
#include <vector>

namespace odai::engine {

class GameApp;

// Lifecycle hooks an engine plugin may implement. Mirrors GameApp's own
// onInit/onTick/onRender/onShutdown shape so a plugin composes the same way
// a GameApp subclass would, without requiring inheritance from GameApp
// itself and without GameApp needing to know anything about the plugin's
// concrete type. All methods default to no-ops; a plugin overrides only
// what it needs.
//
// onRender is NOT fanned out automatically by GameApp::run() -- a game's
// onRender() ends by calling submitFrame(), which flushes the UI draw list
// and submits the frame to the GPU, so anything drawn after that point never
// makes it into the frame. A game that wants plugins to contribute UI must
// call plugins().renderAll(*this, dt) itself, after beginFrameDraw() and
// before submitFrame(). onTick has no such ordering hazard and is called
// automatically.
class IEnginePlugin {
public:
    virtual ~IEnginePlugin() = default;

    // Short, unique identifier for logging/diagnostics (e.g. "frame-stats").
    virtual const char* name() const = 0;

    // Called once by GameApp::init(), after onInit() has succeeded. Return
    // false to abort startup, mirroring GameApp::onInit()'s own contract.
    virtual bool onAttach(GameApp& /*app*/) { return true; }

    // Called once per frame from GameApp::run(), right after the game's own
    // onTick(dt).
    virtual void onTick(GameApp& /*app*/, float /*dt*/) {}

    // Opt-in: see the class comment. Not called unless the game explicitly
    // invokes plugins().renderAll(*this, dt) from its own onRender().
    virtual void onRender(GameApp& /*app*/, float /*dt*/) {}

    // Called once by GameApp::shutdown(), after the game's own onShutdown().
    virtual void onDetach(GameApp& /*app*/) {}
};

// Owns a set of plugins and fans lifecycle calls out to all of them in
// registration order. A GameApp holds one; a subclass registers plugins
// during its own onInit() (before GameApp::init() calls attachAll()).
//
// Deliberately a flat vector<unique_ptr<>>, not a name-keyed registry with
// lookup/replace semantics -- add that only when a real use case needs
// dynamic (un)registration by name.
class PluginRegistry {
public:
    void add(std::unique_ptr<IEnginePlugin> plugin);

    // Calls onAttach() on every plugin in registration order. Stops and
    // returns false at the first failure (logging which plugin failed).
    // Does NOT roll back the plugins already attached itself -- on failure,
    // the caller must call detachAll() before tearing anything else down
    // (GameApp::init() does this).
    bool attachAll(GameApp& app);

    void tickAll(GameApp& app, float dt);
    void renderAll(GameApp& app, float dt);

    // Calls onDetach() on every plugin in reverse registration order, then
    // clears the registry.
    void detachAll(GameApp& app);

    std::size_t size() const { return m_plugins.size(); }

private:
    std::vector<std::unique_ptr<IEnginePlugin>> m_plugins;
};

} // namespace odai::engine
