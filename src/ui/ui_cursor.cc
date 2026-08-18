#include "ui/ui_cursor.h"

#include "ui/ui_draw_list.h"

namespace odai::ui {

namespace {

void addArrowTriangle(UiDrawList& drawList, const UiVec2& position, float scale,
                      const UiColor& color, float inset) {
    const std::uint32_t rgba = color.packAbgr8();
    const std::uint32_t mode = static_cast<std::uint32_t>(UiDrawMode::SolidColor);
    const UiVertex vertices[] = {
        {{position.x + inset * scale, position.y + inset * scale}, {}, rgba, mode, {}},
        {{position.x + inset * scale, position.y + (14.0f - inset) * scale}, {}, rgba, mode, {}},
        {{position.x + (11.0f - inset) * scale, position.y + (10.0f - inset) * scale}, {}, rgba, mode, {}},
    };
    constexpr std::uint32_t indices[] = {0, 1, 2};
    drawList.addTriangleMesh(vertices, 3, indices, 3);
}

}  // namespace

void drawCursor(UiDrawList& drawList, const UiVec2& position, float scale) {
    addArrowTriangle(drawList, position, scale, {1, 1, 1, 1}, 0.0f);
    addArrowTriangle(drawList, position, scale, {0.05f, 0.05f, 0.05f, 1}, 1.2f);
}

}  // namespace odai::ui
