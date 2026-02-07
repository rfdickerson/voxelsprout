#pragma once

// Core Input subsystem
// Responsible for: defining minimal input state placeholders used by the app.
// Should NOT do: poll OS APIs, bind actions, or process gameplay commands yet.
namespace core {

struct InputState {
    bool quitRequested = false;
    bool moveForward = false;
    bool moveBackward = false;
    bool moveLeft = false;
    bool moveRight = false;
    bool moveUp = false;
    bool moveDown = false;
    bool placeBlockDown = false;
    bool removeBlockDown = false;

    float mouseDeltaX = 0.0f;
    float mouseDeltaY = 0.0f;
};

} // namespace core
