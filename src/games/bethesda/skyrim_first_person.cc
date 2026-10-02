#include "games/bethesda/bethesda_app.h"
#include "import/bethesda/character_asset_manifest.h"
#include "core/log.h"
#include <algorithm>
#include <cmath>

namespace odai::games::bethesda {
void BethesdaApp::updateSkyrimFirstPersonWeapon(float deltaSeconds) {
    if (!m_streamer || !m_bethesdaSessionConfigured) return;
    const auto* player = m_bethesdaSession.world().find(m_bethesdaSession.playerObject());
    std::string model, shield;
    if (player) for (const auto& entry : player->inventory) {
        if (!entry.equipped || entry.count <= 0) continue;
        const auto* item = m_bethesdaSession.skyrimItem(entry.item);
        if (item && item->recordType == "WEAP" && item->weaponAnimationType == 1) { model = item->model; }
        if (item && item->recordType == "ARMO" && (item->bipedSlots & (1u << 9))) shield = item->model;
    }
    const std::string visualKey = model + "|" + shield;
    if (visualKey != m_firstPersonWeaponPath) {
        m_renderer.setSkinnedActorVisible(kFirstPersonWeaponInstance, false);
        m_skyrimFirstPersonWeapon.reset();
        m_firstPersonWeaponPath = visualKey;
        m_firstPersonPoseState = -1;
        m_firstPersonLocalPose.clear();
        m_firstPersonBlockClip = {};
        m_firstPersonAttackTime = -1;
        m_firstPersonWasVisible = false;
        if (!model.empty()) {
            using namespace importer::bethesda;
            const auto& assets = m_streamer->assets();
            SkinnedActor weapon;
            weapon.instanceSlot = kFirstPersonWeaponInstance;
            std::vector<std::string> parts{model}, attachments{"WEAPON"};
            std::vector<std::uint8_t> modes{1};
            if (!shield.empty()) { parts.push_back(shield); attachments.push_back("SHIELD"); modes.push_back(1); } // drawn geometry; omit scabbard
            std::string error;
            anim::HkxDecodedSkeleton source;
            FalloutAssetSource::ResolvedAsset asset;
            anim::HkxDecodedClipMetadata metadata;
            const std::string root = "meshes\\actors\\character\\_1stperson\\";
            if (!buildSkinnedActor(assets, root.substr(7) + "skeleton.nif", parts, weapon.character, weapon.textures,
                    weapon.draws, error, &attachments, nullptr, 0, nullptr, &modes) ||
                !assets.resolveAssetWithProvider(root + "characterassets\\skeletonfirst.hkx", asset, error) ||
                !anim::decodeHkxAnimationSkeleton(asset.bytes, source, error) ||
                !loadCharacterSourceClip(assets, root + "animations\\1hm_idle.hkx", weapon.character.skeleton,
                    &source, false, weapon.idleClip, metadata, asset, error) ||
                !loadCharacterSourceClip(assets, root + "animations\\1hm_attackright.hkx", weapon.character.skeleton,
                    &source, false, weapon.walkClip, metadata, asset, error)) {
                VOX_LOGW("first-person") << "sword view unavailable: " << error;
            } else {
                if (!shield.empty() && !loadCharacterSourceClip(assets, root + "animations\\shd_blockidle.hkx",
                        weapon.character.skeleton, &source, false, m_firstPersonBlockClip, metadata, asset, error)) {
                    VOX_LOGW("first-person") << "shield animation unavailable: " << error;
                }
                m_firstPersonBlockClip.loop = true;
                weapon.idleClip.loop = true;
                weapon.walkClip.loop = false;
                weapon.sampler.bindSkeleton(weapon.character.skeleton, weapon.character.inverseBindMatrices);
                const auto slots = m_renderer.uploadSkinnedActorTextures(weapon.instanceSlot, weapon.textures);
                remapActorTextureSlots(weapon, slots);
                render::ImportedSkinnedMeshTemplate mesh{};
                mesh.vertices = weapon.character.vertices; mesh.indices = weapon.character.indices; mesh.draws = weapon.draws;
                mesh.boneCount = weapon.character.skeleton.bones.size();
                weapon.uploaded = m_renderer.uploadSkinnedMeshTemplate(weapon.instanceSlot, mesh);
                if (weapon.uploaded) {
                    VOX_LOGI("first-person") << "authored sword view ready: " << model
                        << "; shield=" << shield << "; attack duration=" << weapon.walkClip.duration;
                    m_skyrimFirstPersonWeapon = std::move(weapon);
                }
            }
        }
    }
    if (!m_skyrimFirstPersonWeapon) return;
    auto& weapon = *m_skyrimFirstPersonWeapon;
    const bool visible = !m_thirdPersonView && player && player->equipment.drawn &&
        (!player->actorValues || !player->actorValues->dead) &&
        !m_bethesdaSession.physics().hasActiveRagdoll(m_bethesdaSession.playerObject());
    m_renderer.setSkinnedActorVisible(weapon.instanceSlot, visible);
    // Cosmetic first-person playback follows accepted gameplay attacks. It does
    // not emit a second HitFrame or apply damage independently of the session.
    const bool paused = m_menuOpen || m_playerInventoryOpen || m_talkingActor >= 0;
    const float dt = paused ? 0.0f : std::max(deltaSeconds, 0.0f);
    weapon.animationSeconds += dt;
    if (m_firstPersonAttackTime >= 0) {
        m_firstPersonAttackTime += dt;
        if (m_firstPersonAttackTime >= weapon.walkClip.duration) m_firstPersonAttackTime = -1;
    }
    if (!visible) { m_firstPersonWasVisible = false; return; }
    const bool attacking = m_firstPersonAttackTime >= 0;
    const int state = attacking ? 1 : m_firstPersonGuardHeld && m_firstPersonBlockClip.duration > 0 ? 2 : 0;
    if (state != m_firstPersonPoseState) {
        m_firstPersonBlendFrom = m_firstPersonLocalPose;
        m_firstPersonBlendTime = 0;
        m_firstPersonPoseState = state;
    }
    const auto& clip = state == 1 ? weapon.walkClip : state == 2 ? m_firstPersonBlockClip : weapon.idleClip;
    auto target = anim::sampleLocalPose(weapon.character.skeleton, clip,
        attacking ? m_firstPersonAttackTime : weapon.animationSeconds);
    // Blend local TRS (including quaternion slerp), never skinning matrices.
    // Smooth both attack recovery and shield raise/lower without filtering the camera.
    m_firstPersonBlendTime += dt;
    const float t = std::clamp(m_firstPersonBlendTime / 0.16f, 0.0f, 1.0f);
    m_firstPersonLocalPose = m_firstPersonBlendFrom.empty() ? std::move(target) :
        anim::blendLocalPoses(m_firstPersonBlendFrom, target, t * t * (3 - 2 * t));
    weapon.sampler.paletteFromLocal(weapon.character.skeleton, m_firstPersonLocalPose, weapon.poseScratch);
    // Skyrim's first-person rig is authored facing engine -Z. Anchor its camera
    // translation to the actual eye while retaining authored hand/weapon motion.
    odai::math::Vector3 authoredEye{};
    const int camera = weapon.character.skeleton.findBone("Camera1st [Cam1]");
    if (camera >= 0 && static_cast<std::size_t>(camera) < weapon.poseScratch.size()) {
        const auto world = weapon.poseScratch[camera] * odai::math::inverse(weapon.character.inverseBindMatrices[camera]);
        authoredEye = {world(0,3), world(1,3), world(2,3)};
    }
    constexpr float radians = 3.14159265358979323846f / 180.0f;
    const float yaw = m_yawDegrees * radians, pitch = m_pitchDegrees * radians;
    const odai::math::Vector3 eye{m_cameraX, m_cameraY, m_cameraZ};
    const odai::math::Vector3 forward{std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};
    const auto transform = odai::math::inverse(odai::math::lookAt(eye, eye + forward, {0,1,0})) *
        odai::math::Matrix4::translation({-authoredEye.x, -authoredEye.y, -authoredEye.z});
    for (auto& bone : weapon.poseScratch) bone = transform * bone;
    render::ImportedSkinnedActorFrameData frame{};
    frame.boneMatrices = weapon.poseScratch;
    frame.resetHistory = !m_firstPersonWasVisible;
    m_renderer.setSkinnedActorPose(weapon.instanceSlot, frame);
    m_firstPersonWasVisible = true;
}
}
