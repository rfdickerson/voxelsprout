#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstring>
#include <vector>

#include "sim/simulation.h"
#include "sim/network_procedural.h"

namespace voxelsprout::render {

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "render/renderer_shared.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

RendererBackend::FrameInstanceDrawData RendererBackend::prepareFrameInstanceDrawData(
    const voxelsprout::sim::Simulation& simulation,
    float simulationAlpha
) {
    FrameInstanceDrawData out{};
    const bool buildPipeAndTransportInstances = (m_pipeIndexCount > 0 || m_transportIndexCount > 0);
    const float clampedSimulationAlpha = std::clamp(simulationAlpha, 0.0f, 1.0f);

    if (buildPipeAndTransportInstances) {
        const std::vector<voxelsprout::sim::Pipe>& pipes = simulation.pipes();
        const std::vector<voxelsprout::sim::Belt>& belts = simulation.belts();
        const std::vector<voxelsprout::sim::Track>& tracks = simulation.tracks();
        const std::vector<voxelsprout::sim::BeltCargo>& beltCargoes = simulation.beltCargoes();
        const std::vector<PipeEndpointState> endpointStates =
            pipes.empty() ? std::vector<PipeEndpointState>{} : buildPipeEndpointStates(pipes);

        std::vector<PipeInstance> pipeInstances;
        pipeInstances.reserve(pipes.size());
        for (std::size_t pipeIndex = 0; pipeIndex < pipes.size(); ++pipeIndex) {
            const voxelsprout::sim::Pipe& pipe = pipes[pipeIndex];
            const PipeEndpointState& endpointState = endpointStates[pipeIndex];
            PipeInstance instance{};
            instance.originLength[0] = static_cast<float>(pipe.x);
            instance.originLength[1] = static_cast<float>(pipe.y);
            instance.originLength[2] = static_cast<float>(pipe.z);
            instance.originLength[3] = std::max(pipe.length, 0.05f);
            instance.axisRadius[0] = endpointState.axis.x;
            instance.axisRadius[1] = endpointState.axis.y;
            instance.axisRadius[2] = endpointState.axis.z;
            instance.axisRadius[3] = endpointState.renderedRadius;
            instance.tint[0] = std::clamp(pipe.tint.x, 0.0f, 1.0f);
            instance.tint[1] = std::clamp(pipe.tint.y, 0.0f, 1.0f);
            instance.tint[2] = std::clamp(pipe.tint.z, 0.0f, 1.0f);
            instance.tint[3] = 0.0f;
            instance.extensions[0] = endpointState.startExtension;
            instance.extensions[1] = endpointState.endExtension;
            instance.extensions[2] = 1.0f;
            instance.extensions[3] = 1.0f;
            pipeInstances.push_back(instance);
        }

        std::vector<PipeInstance> transportInstances;
        transportInstances.reserve(belts.size() + tracks.size());
        for (const voxelsprout::sim::Belt& belt : belts) {
            PipeInstance instance{};
            instance.originLength[0] = static_cast<float>(belt.x);
            instance.originLength[1] = static_cast<float>(belt.y);
            instance.originLength[2] = static_cast<float>(belt.z);
            instance.originLength[3] = 1.0f;
            const voxelsprout::math::Vector3 axis = beltDirectionAxis(belt.direction);
            instance.axisRadius[0] = axis.x;
            instance.axisRadius[1] = axis.y;
            instance.axisRadius[2] = axis.z;
            instance.axisRadius[3] = kBeltRadius;
            instance.tint[0] = kBeltTint.x;
            instance.tint[1] = kBeltTint.y;
            instance.tint[2] = kBeltTint.z;
            instance.tint[3] = 1.0f;
            instance.extensions[0] = 0.0f;
            instance.extensions[1] = 0.0f;
            instance.extensions[2] = 2.0f;
            instance.extensions[3] = 0.25f;
            transportInstances.push_back(instance);
        }

        for (const voxelsprout::sim::Track& track : tracks) {
            PipeInstance instance{};
            instance.originLength[0] = static_cast<float>(track.x);
            instance.originLength[1] = static_cast<float>(track.y);
            instance.originLength[2] = static_cast<float>(track.z);
            instance.originLength[3] = 1.0f;
            const voxelsprout::math::Vector3 axis = trackDirectionAxis(track.direction);
            instance.axisRadius[0] = axis.x;
            instance.axisRadius[1] = axis.y;
            instance.axisRadius[2] = axis.z;
            instance.axisRadius[3] = kTrackRadius;
            instance.tint[0] = kTrackTint.x;
            instance.tint[1] = kTrackTint.y;
            instance.tint[2] = kTrackTint.z;
            instance.tint[3] = 2.0f;
            instance.extensions[0] = 0.0f;
            instance.extensions[1] = 0.0f;
            instance.extensions[2] = 2.0f;
            instance.extensions[3] = 0.25f;
            transportInstances.push_back(instance);
        }

        std::vector<PipeInstance> beltCargoInstances;
        beltCargoInstances.reserve(beltCargoes.size());
        for (const voxelsprout::sim::BeltCargo& cargo : beltCargoes) {
            if (cargo.beltIndex < 0 || static_cast<std::size_t>(cargo.beltIndex) >= belts.size()) {
                continue;
            }
            const float worldX = std::lerp(cargo.prevWorldPos[0], cargo.currWorldPos[0], clampedSimulationAlpha);
            const float worldY = std::lerp(cargo.prevWorldPos[1], cargo.currWorldPos[1], clampedSimulationAlpha);
            const float worldZ = std::lerp(cargo.prevWorldPos[2], cargo.currWorldPos[2], clampedSimulationAlpha);
            const voxelsprout::sim::Belt& belt = belts[static_cast<std::size_t>(cargo.beltIndex)];
            const voxelsprout::math::Vector3 axis = beltDirectionAxis(belt.direction);
            const voxelsprout::math::Vector3 tint =
                kBeltCargoTints[static_cast<std::size_t>(cargo.typeId % kBeltCargoTints.size())];

            PipeInstance instance{};
            instance.originLength[0] = worldX - 0.5f;
            instance.originLength[1] = worldY - 0.5f;
            instance.originLength[2] = worldZ - 0.5f;
            instance.originLength[3] = kBeltCargoLength;
            instance.axisRadius[0] = axis.x;
            instance.axisRadius[1] = axis.y;
            instance.axisRadius[2] = axis.z;
            instance.axisRadius[3] = kBeltCargoRadius;
            instance.tint[0] = tint.x;
            instance.tint[1] = tint.y;
            instance.tint[2] = tint.z;
            instance.tint[3] = 2.0f;
            instance.extensions[0] = 0.0f;
            instance.extensions[1] = 0.0f;
            instance.extensions[2] = 1.0f;
            instance.extensions[3] = 1.0f;
            beltCargoInstances.push_back(instance);
        }

        if (!pipeInstances.empty() && m_pipeIndexCount > 0) {
            out.pipeInstanceSliceOpt = m_frameArena.allocateUpload(
                static_cast<VkDeviceSize>(pipeInstances.size() * sizeof(PipeInstance)),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::InstanceData
            );
            if (out.pipeInstanceSliceOpt.has_value() && out.pipeInstanceSliceOpt->mapped != nullptr) {
                std::memcpy(
                    out.pipeInstanceSliceOpt->mapped,
                    pipeInstances.data(),
                    static_cast<size_t>(out.pipeInstanceSliceOpt->size)
                );
                out.pipeInstanceCount = static_cast<uint32_t>(pipeInstances.size());
            }
        }

        if (!transportInstances.empty() && m_transportIndexCount > 0) {
            out.transportInstanceSliceOpt = m_frameArena.allocateUpload(
                static_cast<VkDeviceSize>(transportInstances.size() * sizeof(PipeInstance)),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::InstanceData
            );
            if (out.transportInstanceSliceOpt.has_value() && out.transportInstanceSliceOpt->mapped != nullptr) {
                std::memcpy(
                    out.transportInstanceSliceOpt->mapped,
                    transportInstances.data(),
                    static_cast<size_t>(out.transportInstanceSliceOpt->size)
                );
                out.transportInstanceCount = static_cast<uint32_t>(transportInstances.size());
            }
        }

        if (!beltCargoInstances.empty() && m_transportIndexCount > 0) {
            out.beltCargoInstanceSliceOpt = m_frameArena.allocateUpload(
                static_cast<VkDeviceSize>(beltCargoInstances.size() * sizeof(PipeInstance)),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::InstanceData
            );
            if (out.beltCargoInstanceSliceOpt.has_value() && out.beltCargoInstanceSliceOpt->mapped != nullptr) {
                std::memcpy(
                    out.beltCargoInstanceSliceOpt->mapped,
                    beltCargoInstances.data(),
                    static_cast<size_t>(out.beltCargoInstanceSliceOpt->size)
                );
                out.beltCargoInstanceCount = static_cast<uint32_t>(beltCargoInstances.size());
            }
        }
    }

    out.readyMagicaDraws.reserve(m_magicaMeshDraws.size());
    for (const MagicaMeshDraw& draw : m_magicaMeshDraws) {
        if (draw.indexCount == 0 ||
            draw.vertexBufferHandle == kInvalidBufferHandle ||
            draw.indexBufferHandle == kInvalidBufferHandle) {
            continue;
        }
        const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(draw.vertexBufferHandle);
        const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(draw.indexBufferHandle);
        if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE) {
            continue;
        }
        out.readyMagicaDraws.push_back(ReadyMagicaDraw{
            vertexBuffer,
            indexBuffer,
            draw.indexCount,
            draw.offsetX,
            draw.offsetY,
            draw.offsetZ
        });
    }

    return out;
}

} // namespace voxelsprout::render
