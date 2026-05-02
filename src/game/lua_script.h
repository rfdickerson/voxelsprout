#pragma once

#include "game/game_state.h"

#include <filesystem>
#include <string>

struct lua_State;

namespace odai::game {

class LuaScriptRuntime {
public:
    LuaScriptRuntime();
    ~LuaScriptRuntime();
    LuaScriptRuntime(const LuaScriptRuntime&) = delete;
    LuaScriptRuntime& operator=(const LuaScriptRuntime&) = delete;

    bool init(GameState& state);
    void shutdown();

    [[nodiscard]] bool initialized() const;
    [[nodiscard]] const std::string& lastError() const;

    bool loadScriptFile(const std::filesystem::path& scriptPath);
    bool loadScriptString(const std::string& scriptText, const std::string& chunkName);

    ScriptCallResult onActivate(const std::string& refId);
    DialogueResult getDialogue(const std::string& actorId, const std::string& topicId);
    ScriptCallResult chooseDialogue(const std::string& responseId);
    ScriptCallResult onActorDeath(const std::string& actorId);

private:
    [[nodiscard]] bool callStringFunction(
        const char* functionName,
        const std::string& arg,
        int expectedResults
    );
    [[nodiscard]] ScriptCallResult readScriptCallResult(int stackIndex) const;
    [[nodiscard]] DialogueResult readDialogueResult(int stackIndex) const;
    void setError(std::string error);

    lua_State* m_lua = nullptr;
    GameState* m_state = nullptr;
    std::string m_lastError;
};

}  // namespace odai::game
