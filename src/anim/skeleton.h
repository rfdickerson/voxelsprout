#pragma once

#include "math/math.h"

#include <string>
#include <vector>

// Skeleton/bone-hierarchy data model — the Dragon Age: Origins touchstone's
// animation foundation (see docs/ROADMAP.md's Party RPG / Narrative section).
// Pure flat data, no Vulkan, no game dependency: a bind-pose bone hierarchy
// that any skinned mesh (a future real NIF import, or a hand-authored test
// rig today) can reference by bone index.
namespace odai::anim {

// A single joint's bind-pose local transform, relative to its parent.
struct Bone {
    std::string name;
    int parentIndex = -1;  // -1 = root
    odai::math::Vector3 localTranslation{};
    odai::math::Quaternion localRotation{};
    odai::math::Vector3 localScale{1.0f, 1.0f, 1.0f};
};

// bones must be stored parent-before-child (every bone's parentIndex refers
// to an earlier index in the vector) so hierarchy composition is a single
// forward pass with no dependency ordering to resolve.
struct Skeleton {
    std::vector<Bone> bones;

    [[nodiscard]] int findBone(const std::string& name) const {
        for (std::size_t i = 0; i < bones.size(); ++i) {
            if (bones[i].name == name) return static_cast<int>(i);
        }
        return -1;
    }
};

}  // namespace odai::anim
