#include "engine/plugin.h"

#include "core/log.h"

namespace odai::engine {

void PluginRegistry::add(std::unique_ptr<IEnginePlugin> plugin) {
    m_plugins.push_back(std::move(plugin));
}

bool PluginRegistry::attachAll(GameApp& app) {
    for (auto& plugin : m_plugins) {
        if (!plugin->onAttach(app)) {
            VOX_LOGE("engine") << "plugin '" << plugin->name() << "' failed to attach";
            return false;
        }
    }
    return true;
}

void PluginRegistry::tickAll(GameApp& app, float dt) {
    for (auto& plugin : m_plugins)
        plugin->onTick(app, dt);
}

void PluginRegistry::renderAll(GameApp& app, float dt) {
    for (auto& plugin : m_plugins)
        plugin->onRender(app, dt);
}

void PluginRegistry::detachAll(GameApp& app) {
    for (auto it = m_plugins.rbegin(); it != m_plugins.rend(); ++it)
        (*it)->onDetach(app);
    m_plugins.clear();
}

} // namespace odai::engine
