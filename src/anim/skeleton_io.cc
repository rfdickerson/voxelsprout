#include "anim/skeleton_io.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iterator>

namespace odai::anim {

namespace {

using json = nlohmann::json;
using odai::math::Quaternion;
using odai::math::Vector3;

Vector3 parseVector3(const json& parent, const char* key, const Vector3& fallback) {
    if (!parent.contains(key) || !parent[key].is_array() || parent[key].size() != 3) return fallback;
    const json& arr = parent[key];
    return Vector3{arr[0].get<float>(), arr[1].get<float>(), arr[2].get<float>()};
}

Quaternion parseQuaternion(const json& parent, const char* key, const Quaternion& fallback) {
    if (!parent.contains(key) || !parent[key].is_array() || parent[key].size() != 4) return fallback;
    const json& arr = parent[key];
    return Quaternion{arr[0].get<float>(), arr[1].get<float>(), arr[2].get<float>(), arr[3].get<float>()};
}

std::vector<Vector3Key> parseVector3Keys(const json& arr) {
    std::vector<Vector3Key> keys;
    if (!arr.is_array()) return keys;
    for (const json& keyJson : arr) {
        if (!keyJson.is_object()) continue;
        Vector3Key key;
        key.time = keyJson.value("time", 0.0f);
        key.value = parseVector3(keyJson, "value", Vector3{});
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end(), [](const Vector3Key& a, const Vector3Key& b) { return a.time < b.time; });
    return keys;
}

std::vector<QuaternionKey> parseQuaternionKeys(const json& arr) {
    std::vector<QuaternionKey> keys;
    if (!arr.is_array()) return keys;
    for (const json& keyJson : arr) {
        if (!keyJson.is_object()) continue;
        QuaternionKey key;
        key.time = keyJson.value("time", 0.0f);
        key.value = parseQuaternion(keyJson, "value", Quaternion{});
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end(),
              [](const QuaternionKey& a, const QuaternionKey& b) { return a.time < b.time; });
    return keys;
}

std::string readWholeFile(const std::filesystem::path& path, std::vector<std::string>& errors) {
    std::ifstream file(path);
    if (!file) {
        errors.push_back("anim: cannot open " + path.string());
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

}  // namespace

SkeletonLoadResult loadSkeletonFromJson(const std::string& jsonText, const std::string& sourceLabel) {
    SkeletonLoadResult result;
    json root;
    try {
        root = json::parse(jsonText);
    } catch (const json::exception& e) {
        result.errors.push_back("anim: JSON parse error in " + sourceLabel + ": " + e.what());
        return result;
    }
    if (!root.is_object() || !root.contains("bones") || !root["bones"].is_array()) {
        result.errors.push_back("anim: " + sourceLabel + " has no 'bones' array");
        return result;
    }

    for (const json& boneJson : root["bones"]) {
        if (!boneJson.is_object()) {
            result.errors.push_back("anim: a bone entry in " + sourceLabel + " is not an object, skipped");
            continue;
        }
        Bone bone;
        bone.name = boneJson.value("name", std::string());
        const std::string parentName = boneJson.value("parent", std::string());
        if (parentName.empty()) {
            bone.parentIndex = -1;
        } else {
            bone.parentIndex = result.skeleton.findBone(parentName);
            if (bone.parentIndex < 0) {
                result.errors.push_back("anim: bone '" + bone.name + "' in " + sourceLabel +
                                         " references unknown or not-yet-declared parent '" + parentName + "'");
            }
        }
        bone.localTranslation = parseVector3(boneJson, "translation", Vector3{});
        bone.localRotation = parseQuaternion(boneJson, "rotation", Quaternion{});
        bone.localScale = parseVector3(boneJson, "scale", Vector3{1.0f, 1.0f, 1.0f});
        result.skeleton.bones.push_back(std::move(bone));
    }
    return result;
}

SkeletonLoadResult loadSkeletonFromFile(const std::filesystem::path& path) {
    SkeletonLoadResult result;
    std::vector<std::string> readErrors;
    const std::string text = readWholeFile(path, readErrors);
    if (!readErrors.empty()) {
        result.errors = std::move(readErrors);
        return result;
    }
    return loadSkeletonFromJson(text, path.string());
}

AnimationClipLoadResult loadAnimationClipFromJson(const std::string& jsonText, const Skeleton& skeleton,
                                                   const std::string& sourceLabel) {
    AnimationClipLoadResult result;
    json root;
    try {
        root = json::parse(jsonText);
    } catch (const json::exception& e) {
        result.errors.push_back("anim: JSON parse error in " + sourceLabel + ": " + e.what());
        return result;
    }
    if (!root.is_object()) {
        result.errors.push_back("anim: " + sourceLabel + " root is not an object");
        return result;
    }

    result.clip.name = root.value("name", std::string());
    result.clip.duration = root.value("duration", 0.0f);
    result.clip.loop = root.value("loop", true);

    if (!root.contains("tracks") || !root["tracks"].is_array()) {
        result.errors.push_back("anim: " + sourceLabel + " has no 'tracks' array");
        return result;
    }
    for (const json& trackJson : root["tracks"]) {
        if (!trackJson.is_object()) {
            result.errors.push_back("anim: a track in " + sourceLabel + " is not an object, skipped");
            continue;
        }
        const std::string boneName = trackJson.value("bone", std::string());
        const int boneIndex = skeleton.findBone(boneName);
        if (boneIndex < 0) {
            result.errors.push_back("anim: " + sourceLabel + " track references unknown bone '" + boneName + "'");
            continue;
        }
        BoneTrack track;
        track.boneIndex = boneIndex;
        if (trackJson.contains("translationKeys")) {
            track.translationKeys = parseVector3Keys(trackJson["translationKeys"]);
        }
        if (trackJson.contains("rotationKeys")) {
            track.rotationKeys = parseQuaternionKeys(trackJson["rotationKeys"]);
        }
        if (trackJson.contains("scaleKeys")) {
            track.scaleKeys = parseVector3Keys(trackJson["scaleKeys"]);
        }
        result.clip.tracks.push_back(std::move(track));
    }
    return result;
}

AnimationClipLoadResult loadAnimationClipFromFile(const std::filesystem::path& path, const Skeleton& skeleton) {
    AnimationClipLoadResult result;
    std::vector<std::string> readErrors;
    const std::string text = readWholeFile(path, readErrors);
    if (!readErrors.empty()) {
        result.errors = std::move(readErrors);
        return result;
    }
    return loadAnimationClipFromJson(text, skeleton, path.string());
}

}  // namespace odai::anim
