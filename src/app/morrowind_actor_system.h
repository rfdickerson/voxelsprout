#pragma once

#include "import/imported_scene.h"
#include "import/morrowind_nif.h"
#include "math/math.h"
#include "render/renderer_types.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace odai::app {

class MorrowindActorSystem {
public:
    using TextureSlotFn = std::function<std::uint32_t(const std::string&)>;

    enum class PoseMode {
        Movement,
        BindPose,
        WalkClip
    };

    struct BuildStats {
        std::uint32_t prototypeCount = 0u;
        std::uint32_t skeletonNodeCount = 0u;
        std::uint32_t nodeAnimationCount = 0u;
        std::uint32_t clipCount = 0u;
        std::uint32_t weightedVertexCount = 0u;
        std::uint32_t unweightedVertexCount = 0u;
        std::string firstBoneNames;
        std::string firstClipNames;
    };

    struct FrameInput {
        std::uint32_t prototypeIndex = std::numeric_limits<std::uint32_t>::max();
        odai::math::Vector3 position{};
        float yawRadians = 0.0f;
        float animationTime = 0.0f;
        bool moving = false;
    };

    void clear();
    [[nodiscard]] bool loadHumanoidSkeleton(
        const std::filesystem::path& dataFilesPath,
        std::string& outError
    );
    [[nodiscard]] std::uint32_t findOrBuildHumanoidPrototype(
        const std::filesystem::path& dataFilesPath,
        const odai::importer::MorrowindActorRecord& actor,
        const odai::importer::MorrowindEquipmentCatalog& equipmentCatalog,
        const TextureSlotFn& textureSlotFn,
        std::string& outError
    );
    [[nodiscard]] std::uint32_t findOrBuildHumanoidPrototypeFromParts(
        const std::filesystem::path& dataFilesPath,
        const std::string& actorId,
        const std::vector<odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart>& parts,
        const TextureSlotFn& textureSlotFn,
        std::string& outError
    );

    void beginFrame(std::size_t visibleActorCapacity, bool debugBoneLines);
    void appendFrameInstance(
        const FrameInput& input,
        PoseMode poseMode,
        float nowSeconds
    );

    [[nodiscard]] odai::render::ImportedActorRenderAssetData renderAssetData() const;
    [[nodiscard]] odai::render::ImportedActorFrameData frameData() const;
    [[nodiscard]] bool hasRenderableAsset() const;
    [[nodiscard]] bool assetDirty() const { return m_assetDirty; }
    void markAssetUploaded() { m_assetDirty = false; }

    [[nodiscard]] const BuildStats& stats() const { return m_stats; }
    [[nodiscard]] std::uint32_t prototypeCount() const;
    [[nodiscard]] std::uint32_t vertexCount() const;
    [[nodiscard]] std::uint32_t indexCount() const;
    [[nodiscard]] std::uint32_t drawCount() const;
    [[nodiscard]] std::uint32_t skeletonNodeCount() const;
    [[nodiscard]] std::uint32_t paletteMatrixCount() const;
    [[nodiscard]] std::uint32_t visibleInstanceCount() const;
    [[nodiscard]] const odai::importer::ImportedAnimationClip* walkClip() const;
    [[nodiscard]] const odai::importer::ImportedAnimationClip* idleClip() const;

private:
    struct PrototypeRange {
        std::uint32_t firstDraw = 0u;
        std::uint32_t drawCount = 0u;
    };

    [[nodiscard]] std::string makeAppearanceSignature(
        const odai::importer::MorrowindActorRecord& actor,
        std::span<const odai::importer::MorrowindEquipmentCatalog::ResolvedActorPart> parts
    ) const;
    [[nodiscard]] bool appendBuiltActor(
        const std::string& actorId,
        const odai::importer::ImportedSkinnedActorAsset& actorAsset,
        const TextureSlotFn& textureSlotFn,
        std::uint32_t& outPrototypeIndex,
        std::string& outError
    );
    void refreshStats();

    odai::importer::ImportedSkinnedActorAsset m_skeletonTemplate;
    std::vector<odai::importer::ImportedScenePackedVertex> m_vertices;
    std::vector<std::uint32_t> m_indices;
    std::vector<odai::importer::ImportedScenePackedDraw> m_draws;
    std::vector<std::array<std::uint16_t, 4>> m_boneIndices;
    std::vector<std::array<float, 4>> m_boneWeights;
    std::vector<PrototypeRange> m_prototypes;
    std::unordered_map<std::string, std::uint32_t> m_prototypeBySignature;
    std::vector<odai::render::ImportedActorInstanceData> m_frameInstances;
    std::vector<odai::render::ImportedActorBonePaletteMatrix> m_frameBonePalette;
    std::vector<odai::render::ImportedActorDebugLineVertex> m_frameDebugBoneLines;
    BuildStats m_stats{};
    bool m_skeletonLoaded = false;
    bool m_assetDirty = false;
    bool m_frameDebugBoneLinesEnabled = false;
};

} // namespace odai::app
