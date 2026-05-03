#pragma once

#include "game/game_state.h"

#include <filesystem>
#include <string>
#include <vector>

struct lua_State;

namespace odai::game {

class LuaScriptRuntime {
public:
    struct NpcRoutePoint {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    struct NpcUpdateCommand {
        bool handled = false;
        bool stop = false;
        float speed = -1.0f;
        float waitSeconds = -1.0f;
        float wanderRadius = -1.0f;
        int priority = 0;
        std::string state;
        std::string anchor;
        std::string message;
        std::vector<NpcRoutePoint> route;
    };

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
    NpcUpdateCommand updateNpc(const std::string& actorId, float x, float y, float z, float gameHour);

private:
    [[nodiscard]] bool callStringFunction(
        const char* functionName,
        const std::string& arg,
        int expectedResults
    );
    [[nodiscard]] ScriptCallResult readScriptCallResult(int stackIndex) const;
    [[nodiscard]] DialogueResult readDialogueResult(int stackIndex) const;
    [[nodiscard]] NpcUpdateCommand readNpcUpdateCommand(int stackIndex) const;
    void setError(std::string error);

    lua_State* m_lua = nullptr;
    GameState* m_state = nullptr;
    std::string m_lastError;
};

}  // namespace odai::game
