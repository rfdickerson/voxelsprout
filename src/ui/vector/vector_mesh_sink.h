#pragma once

#include "ui/ui_draw_list.h"

#include <cstdint>
#include <vector>

// Decouples the tessellator from the destination geometry buffer. The same
// tessellation code can feed a live UiDrawList (runtime drawing) or a
// UiGeometryBlock (offline baking / cached icons) without knowing which.
//
// All emitted geometry uses UiDrawMode::SolidColor with per-vertex color, so the
// anti-aliased fringe is just vertices whose packed alpha byte is zero.
namespace odai::ui {

class IMeshSink {
public:
    virtual ~IMeshSink() = default;
    // Push a vertex in pixel space with a packed ABGR8 color; returns its index.
    virtual std::uint32_t pushVertex(float x, float y, std::uint32_t rgba8) = 0;
    virtual void pushTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) = 0;
};

// Sink that appends straight into a UiGeometryBlock's vertex/index arrays and a
// single SolidColor command. Used by the offline bundler and icon baking.
class GeometryBlockMeshSink final : public IMeshSink {
public:
    explicit GeometryBlockMeshSink(UiGeometryBlock& block) : m_block(block) {
        if (m_block.commands.empty()) {
            UiDrawCmd cmd{};
            cmd.indexOffset = static_cast<std::uint32_t>(m_block.indices.size());
            cmd.indexCount = 0;
            cmd.textureId = kUiNoTexture;
            m_block.commands.push_back(cmd);
        }
    }

    std::uint32_t pushVertex(float x, float y, std::uint32_t rgba8) override {
        const auto index = static_cast<std::uint32_t>(m_block.vertices.size());
        UiVertex v{};
        v.posPx[0] = x;
        v.posPx[1] = y;
        v.rgba8 = rgba8;
        v.mode = static_cast<std::uint32_t>(UiDrawMode::SolidColor);
        m_block.vertices.push_back(v);
        return index;
    }

    void pushTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) override {
        m_block.indices.push_back(a);
        m_block.indices.push_back(b);
        m_block.indices.push_back(c);
        m_block.commands.back().indexCount += 3;
    }

private:
    UiGeometryBlock& m_block;
};

// Sink that buffers geometry and flushes it into a live UiDrawList via
// addTriangleMesh (which applies the current opacity and clip). Flushes on
// destruction; call flush() explicitly if you need the geometry sooner.
class DrawListMeshSink final : public IMeshSink {
public:
    explicit DrawListMeshSink(UiDrawList& drawList) : m_drawList(drawList) {}
    ~DrawListMeshSink() override { flush(); }

    DrawListMeshSink(const DrawListMeshSink&) = delete;
    DrawListMeshSink& operator=(const DrawListMeshSink&) = delete;

    std::uint32_t pushVertex(float x, float y, std::uint32_t rgba8) override {
        const auto index = static_cast<std::uint32_t>(m_vertices.size());
        UiVertex v{};
        v.posPx[0] = x;
        v.posPx[1] = y;
        v.rgba8 = rgba8;
        v.mode = static_cast<std::uint32_t>(UiDrawMode::SolidColor);
        m_vertices.push_back(v);
        return index;
    }

    void pushTriangle(std::uint32_t a, std::uint32_t b, std::uint32_t c) override {
        m_indices.push_back(a);
        m_indices.push_back(b);
        m_indices.push_back(c);
    }

    void flush() {
        if (!m_vertices.empty() && !m_indices.empty()) {
            m_drawList.addTriangleMesh(m_vertices.data(), m_vertices.size(), m_indices.data(),
                                       m_indices.size());
        }
        m_vertices.clear();
        m_indices.clear();
    }

private:
    UiDrawList& m_drawList;
    std::vector<UiVertex> m_vertices;
    std::vector<std::uint32_t> m_indices;
};

}  // namespace odai::ui
