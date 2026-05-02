#include "game/lua_script.h"

#if __has_include(<lua.hpp>)
#include <lua.hpp>
#elif __has_include(<lua/lua.hpp>)
#include <lua/lua.hpp>
#else
extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}
#endif

#include <algorithm>
#include <cstddef>
#include <utility>

namespace odai::game {

namespace {

GameState* stateFromLua(lua_State* lua) {
    lua_getfield(lua, LUA_REGISTRYINDEX, "odai_game_state");
    auto* state = static_cast<GameState*>(lua_touserdata(lua, -1));
    lua_pop(lua, 1);
    return state;
}

int luaLog(lua_State* lua) {
    if (GameState* state = stateFromLua(lua)) {
        state->appendLog(luaL_checkstring(lua, 1));
    }
    return 0;
}

int luaJournal(lua_State* lua) {
    GameState* state = stateFromLua(lua);
    lua_pushinteger(lua, state == nullptr ? 0 : state->journalStage(luaL_checkstring(lua, 1)));
    return 1;
}

int luaSetJournal(lua_State* lua) {
    if (GameState* state = stateFromLua(lua)) {
        state->setJournalStage(luaL_checkstring(lua, 1), static_cast<int>(luaL_checkinteger(lua, 2)));
    }
    return 0;
}

int luaItemCount(lua_State* lua) {
    GameState* state = stateFromLua(lua);
    lua_pushinteger(lua, state == nullptr ? 0 : state->itemCount(luaL_checkstring(lua, 1)));
    return 1;
}

int luaHasItem(lua_State* lua) {
    GameState* state = stateFromLua(lua);
    const int needed = static_cast<int>(luaL_optinteger(lua, 2, 1));
    lua_pushboolean(lua, state != nullptr && state->itemCount(luaL_checkstring(lua, 1)) >= needed);
    return 1;
}

int luaAddItem(lua_State* lua) {
    if (GameState* state = stateFromLua(lua)) {
        state->addItem(luaL_checkstring(lua, 1), static_cast<int>(luaL_checkinteger(lua, 2)));
    }
    return 0;
}

int luaRemoveItem(lua_State* lua) {
    bool removed = false;
    if (GameState* state = stateFromLua(lua)) {
        removed = state->removeItem(luaL_checkstring(lua, 1), static_cast<int>(luaL_checkinteger(lua, 2)));
    }
    lua_pushboolean(lua, removed);
    return 1;
}

int luaGold(lua_State* lua) {
    GameState* state = stateFromLua(lua);
    lua_pushinteger(lua, state == nullptr ? 0 : state->gold());
    return 1;
}

int luaAddGold(lua_State* lua) {
    if (GameState* state = stateFromLua(lua)) {
        state->addGold(static_cast<int>(luaL_checkinteger(lua, 1)));
    }
    return 0;
}

int luaSpendGold(lua_State* lua) {
    bool spent = false;
    if (GameState* state = stateFromLua(lua)) {
        spent = state->spendGold(static_cast<int>(luaL_checkinteger(lua, 1)));
    }
    lua_pushboolean(lua, spent);
    return 1;
}

int luaIsRefDead(lua_State* lua) {
    GameState* state = stateFromLua(lua);
    lua_pushboolean(lua, state != nullptr && state->isRefDead(luaL_checkstring(lua, 1)));
    return 1;
}

int luaSetRefDead(lua_State* lua) {
    if (GameState* state = stateFromLua(lua)) {
        state->setRefDead(luaL_checkstring(lua, 1), lua_toboolean(lua, 2) != 0);
    }
    return 0;
}

int luaIsRefDisabled(lua_State* lua) {
    GameState* state = stateFromLua(lua);
    lua_pushboolean(lua, state != nullptr && state->isRefDisabled(luaL_checkstring(lua, 1)));
    return 1;
}

int luaSetRefDisabled(lua_State* lua) {
    if (GameState* state = stateFromLua(lua)) {
        state->setRefDisabled(luaL_checkstring(lua, 1), lua_toboolean(lua, 2) != 0);
    }
    return 0;
}

void registerGameApi(lua_State* lua, GameState& state) {
    lua_pushlightuserdata(lua, &state);
    lua_setfield(lua, LUA_REGISTRYINDEX, "odai_game_state");

    lua_newtable(lua);
    lua_pushcfunction(lua, luaLog);
    lua_setfield(lua, -2, "log");
    lua_pushcfunction(lua, luaJournal);
    lua_setfield(lua, -2, "journal");
    lua_pushcfunction(lua, luaSetJournal);
    lua_setfield(lua, -2, "set_journal");
    lua_pushcfunction(lua, luaItemCount);
    lua_setfield(lua, -2, "item_count");
    lua_pushcfunction(lua, luaHasItem);
    lua_setfield(lua, -2, "has_item");
    lua_pushcfunction(lua, luaAddItem);
    lua_setfield(lua, -2, "add_item");
    lua_pushcfunction(lua, luaRemoveItem);
    lua_setfield(lua, -2, "remove_item");
    lua_pushcfunction(lua, luaGold);
    lua_setfield(lua, -2, "gold");
    lua_pushcfunction(lua, luaAddGold);
    lua_setfield(lua, -2, "add_gold");
    lua_pushcfunction(lua, luaSpendGold);
    lua_setfield(lua, -2, "spend_gold");
    lua_pushcfunction(lua, luaIsRefDead);
    lua_setfield(lua, -2, "is_ref_dead");
    lua_pushcfunction(lua, luaSetRefDead);
    lua_setfield(lua, -2, "set_ref_dead");
    lua_pushcfunction(lua, luaIsRefDisabled);
    lua_setfield(lua, -2, "is_ref_disabled");
    lua_pushcfunction(lua, luaSetRefDisabled);
    lua_setfield(lua, -2, "set_ref_disabled");
    lua_setglobal(lua, "game");
}

std::string optionalTableString(lua_State* lua, int tableIndex, const char* fieldName) {
    lua_getfield(lua, tableIndex, fieldName);
    std::string result;
    if (lua_isstring(lua, -1)) {
        result = lua_tostring(lua, -1);
    }
    lua_pop(lua, 1);
    return result;
}

bool optionalTableBool(lua_State* lua, int tableIndex, const char* fieldName) {
    lua_getfield(lua, tableIndex, fieldName);
    const bool result = lua_toboolean(lua, -1) != 0;
    lua_pop(lua, 1);
    return result;
}

float optionalTableFloat(lua_State* lua, int tableIndex, const char* fieldName, float fallback) {
    lua_getfield(lua, tableIndex, fieldName);
    const float result = lua_isnumber(lua, -1) ? static_cast<float>(lua_tonumber(lua, -1)) : fallback;
    lua_pop(lua, 1);
    return result;
}

float routePointComponent(lua_State* lua, int tableIndex, const char* fieldName, int arrayIndex) {
    lua_getfield(lua, tableIndex, fieldName);
    if (lua_isnumber(lua, -1)) {
        const float result = static_cast<float>(lua_tonumber(lua, -1));
        lua_pop(lua, 1);
        return result;
    }
    lua_pop(lua, 1);

    lua_rawgeti(lua, tableIndex, arrayIndex);
    const float result = lua_isnumber(lua, -1) ? static_cast<float>(lua_tonumber(lua, -1)) : 0.0f;
    lua_pop(lua, 1);
    return result;
}

}  // namespace

LuaScriptRuntime::LuaScriptRuntime() = default;

LuaScriptRuntime::~LuaScriptRuntime() {
    shutdown();
}

bool LuaScriptRuntime::init(GameState& state) {
    shutdown();
    m_lua = luaL_newstate();
    if (m_lua == nullptr) {
        setError("failed to allocate Lua state");
        return false;
    }
    luaL_openlibs(m_lua);
    m_state = &state;
    registerGameApi(m_lua, state);
    m_lastError.clear();
    return true;
}

void LuaScriptRuntime::shutdown() {
    if (m_lua != nullptr) {
        lua_close(m_lua);
        m_lua = nullptr;
    }
    m_state = nullptr;
}

bool LuaScriptRuntime::initialized() const {
    return m_lua != nullptr && m_state != nullptr;
}

const std::string& LuaScriptRuntime::lastError() const {
    return m_lastError;
}

bool LuaScriptRuntime::loadScriptFile(const std::filesystem::path& scriptPath) {
    if (!initialized()) {
        setError("Lua runtime is not initialized");
        return false;
    }
    if (luaL_dofile(m_lua, scriptPath.string().c_str()) != LUA_OK) {
        setError(lua_tostring(m_lua, -1) != nullptr ? lua_tostring(m_lua, -1) : "Lua script load failed");
        lua_pop(m_lua, 1);
        return false;
    }
    m_lastError.clear();
    return true;
}

bool LuaScriptRuntime::loadScriptString(const std::string& scriptText, const std::string& chunkName) {
    if (!initialized()) {
        setError("Lua runtime is not initialized");
        return false;
    }
    if (luaL_loadbuffer(m_lua, scriptText.data(), scriptText.size(), chunkName.c_str()) != LUA_OK ||
        lua_pcall(m_lua, 0, 0, 0) != LUA_OK) {
        setError(lua_tostring(m_lua, -1) != nullptr ? lua_tostring(m_lua, -1) : "Lua script load failed");
        lua_pop(m_lua, 1);
        return false;
    }
    m_lastError.clear();
    return true;
}

bool LuaScriptRuntime::callStringFunction(const char* functionName, const std::string& arg, int expectedResults) {
    if (!initialized()) {
        setError("Lua runtime is not initialized");
        return false;
    }
    lua_getglobal(m_lua, functionName);
    if (!lua_isfunction(m_lua, -1)) {
        lua_pop(m_lua, 1);
        return false;
    }
    const GameState snapshot = *m_state;
    lua_pushlstring(m_lua, arg.data(), arg.size());
    if (lua_pcall(m_lua, 1, expectedResults, 0) != LUA_OK) {
        *m_state = snapshot;
        setError(lua_tostring(m_lua, -1) != nullptr ? lua_tostring(m_lua, -1) : "Lua call failed");
        lua_pop(m_lua, 1);
        return false;
    }
    m_lastError.clear();
    return true;
}

ScriptCallResult LuaScriptRuntime::readScriptCallResult(int stackIndex) const {
    ScriptCallResult result{};
    if (!lua_istable(m_lua, stackIndex)) {
        return result;
    }
    const int tableIndex = lua_absindex(m_lua, stackIndex);
    result.handled = optionalTableBool(m_lua, tableIndex, "handled");
    result.message = optionalTableString(m_lua, tableIndex, "message");
    return result;
}

DialogueResult LuaScriptRuntime::readDialogueResult(int stackIndex) const {
    DialogueResult result{};
    if (!lua_istable(m_lua, stackIndex)) {
        return result;
    }
    const int tableIndex = lua_absindex(m_lua, stackIndex);
    result.handled = optionalTableBool(m_lua, tableIndex, "handled");
    result.text = optionalTableString(m_lua, tableIndex, "text");
    lua_getfield(m_lua, tableIndex, "choices");
    if (lua_istable(m_lua, -1)) {
        const int choicesIndex = lua_absindex(m_lua, -1);
        const int choiceCount = static_cast<int>(lua_rawlen(m_lua, choicesIndex));
        for (int i = 1; i <= choiceCount; ++i) {
            lua_rawgeti(m_lua, choicesIndex, i);
            if (lua_istable(m_lua, -1)) {
                const int choiceIndex = lua_absindex(m_lua, -1);
                DialogueChoice choice{};
                choice.id = optionalTableString(m_lua, choiceIndex, "id");
                choice.text = optionalTableString(m_lua, choiceIndex, "text");
                if (!choice.id.empty()) {
                    result.choices.push_back(std::move(choice));
                }
            }
            lua_pop(m_lua, 1);
        }
    }
    lua_pop(m_lua, 1);
    return result;
}

LuaScriptRuntime::NpcUpdateCommand LuaScriptRuntime::readNpcUpdateCommand(int stackIndex) const {
    NpcUpdateCommand result{};
    if (!lua_istable(m_lua, stackIndex)) {
        return result;
    }

    const int tableIndex = lua_absindex(m_lua, stackIndex);
    result.handled = optionalTableBool(m_lua, tableIndex, "handled");
    result.stop = optionalTableBool(m_lua, tableIndex, "stop");
    result.speed = optionalTableFloat(m_lua, tableIndex, "speed", -1.0f);
    result.message = optionalTableString(m_lua, tableIndex, "message");

    lua_getfield(m_lua, tableIndex, "route");
    if (lua_istable(m_lua, -1)) {
        const int routeIndex = lua_absindex(m_lua, -1);
        const int routeLength = static_cast<int>(lua_rawlen(m_lua, routeIndex));
        result.route.reserve(static_cast<std::size_t>(std::max(routeLength, 0)));
        for (int i = 1; i <= routeLength; ++i) {
            lua_rawgeti(m_lua, routeIndex, i);
            if (lua_istable(m_lua, -1)) {
                const int pointIndex = lua_absindex(m_lua, -1);
                result.route.push_back({
                    routePointComponent(m_lua, pointIndex, "x", 1),
                    routePointComponent(m_lua, pointIndex, "y", 2),
                    routePointComponent(m_lua, pointIndex, "z", 3)
                });
            }
            lua_pop(m_lua, 1);
        }
    }
    lua_pop(m_lua, 1);
    return result;
}

ScriptCallResult LuaScriptRuntime::onActivate(const std::string& refId) {
    if (!callStringFunction("on_activate", refId, 1)) {
        return {};
    }
    ScriptCallResult result = readScriptCallResult(-1);
    lua_pop(m_lua, 1);
    return result;
}

DialogueResult LuaScriptRuntime::getDialogue(const std::string& actorId, const std::string& topicId) {
    if (!initialized()) {
        setError("Lua runtime is not initialized");
        return {};
    }
    lua_getglobal(m_lua, "get_dialogue");
    if (!lua_isfunction(m_lua, -1)) {
        lua_pop(m_lua, 1);
        return {};
    }
    const GameState snapshot = *m_state;
    lua_pushlstring(m_lua, actorId.data(), actorId.size());
    lua_pushlstring(m_lua, topicId.data(), topicId.size());
    if (lua_pcall(m_lua, 2, 1, 0) != LUA_OK) {
        *m_state = snapshot;
        setError(lua_tostring(m_lua, -1) != nullptr ? lua_tostring(m_lua, -1) : "Lua dialogue call failed");
        lua_pop(m_lua, 1);
        return {};
    }
    DialogueResult result = readDialogueResult(-1);
    lua_pop(m_lua, 1);
    m_lastError.clear();
    return result;
}

ScriptCallResult LuaScriptRuntime::chooseDialogue(const std::string& responseId) {
    if (!callStringFunction("choose_dialogue", responseId, 1)) {
        return {};
    }
    ScriptCallResult result = readScriptCallResult(-1);
    lua_pop(m_lua, 1);
    return result;
}

ScriptCallResult LuaScriptRuntime::onActorDeath(const std::string& actorId) {
    if (!callStringFunction("on_actor_death", actorId, 1)) {
        return {};
    }
    ScriptCallResult result = readScriptCallResult(-1);
    lua_pop(m_lua, 1);
    return result;
}

LuaScriptRuntime::NpcUpdateCommand LuaScriptRuntime::updateNpc(
    const std::string& actorId,
    float x,
    float y,
    float z
) {
    if (!initialized()) {
        setError("Lua runtime is not initialized");
        return {};
    }
    lua_getglobal(m_lua, "update_npc");
    if (!lua_isfunction(m_lua, -1)) {
        lua_pop(m_lua, 1);
        return {};
    }

    const GameState snapshot = *m_state;
    lua_pushlstring(m_lua, actorId.data(), actorId.size());
    lua_pushnumber(m_lua, x);
    lua_pushnumber(m_lua, y);
    lua_pushnumber(m_lua, z);
    if (lua_pcall(m_lua, 4, 1, 0) != LUA_OK) {
        *m_state = snapshot;
        setError(lua_tostring(m_lua, -1) != nullptr ? lua_tostring(m_lua, -1) : "Lua NPC update call failed");
        lua_pop(m_lua, 1);
        return {};
    }
    NpcUpdateCommand result = readNpcUpdateCommand(-1);
    lua_pop(m_lua, 1);
    m_lastError.clear();
    return result;
}

void LuaScriptRuntime::setError(std::string error) {
    m_lastError = std::move(error);
}

}  // namespace odai::game
