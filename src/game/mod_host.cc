#include "game/mod_host.h"

namespace odai::game {

namespace {

// Does nothing: the default host so the base simulation runs identically whether
// or not a scripting engine is installed.
class NullModHost final : public IModHost {};

IModHost& nullHost() {
    static NullModHost host;
    return host;
}

IModHost* g_host = nullptr;

}  // namespace

IModHost& modHost() {
    return g_host != nullptr ? *g_host : nullHost();
}

void setModHost(IModHost* host) {
    g_host = host;
}

}  // namespace odai::game
