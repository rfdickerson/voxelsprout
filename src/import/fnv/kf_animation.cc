#include "import/fnv/kf_animation.h"

#include "import/fnv/nif_scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace odai::importer::fnv {

namespace {

using odai::math::Quaternion;
using odai::math::Vector3;

// NIF KeyType, shared by every KeyGroup in the file.
constexpr std::uint32_t kKeyLinear = 1;
constexpr std::uint32_t kKeyQuadratic = 2;
constexpr std::uint32_t kKeyTbc = 3;
constexpr std::uint32_t kKeyXyzRotation = 4;
constexpr std::uint32_t kKeyConst = 5;

// NiTransformInterpolator stores -FLT_MAX in a channel it does not provide a
// static value for. Testing against the exact bit pattern would work too, but a
// magnitude test also catches the +FLT_MAX form some exporters write.
bool isUnsetFloat(float value) {
    return !std::isfinite(value) || std::fabs(value) >= (std::numeric_limits<float>::max() * 0.99f);
}

// Little-endian cursor over one block's bytes. Every read is bounds-checked
// against the block's own end, so a layout mistake stops at the block boundary
// instead of walking into the next one and producing plausible garbage.
class BlockCursor {
public:
    BlockCursor(const std::uint8_t* data, std::size_t size) : m_data(data), m_size(size) {}

    bool readU32(std::uint32_t& out) { return readRaw(&out, 4); }
    bool readI32(std::int32_t& out) { return readRaw(&out, 4); }
    bool readU8(std::uint8_t& out) { return readRaw(&out, 1); }
    bool readFloat(float& out) { return readRaw(&out, 4); }

    bool readVector3(Vector3& out) {
        return readFloat(out.x) && readFloat(out.y) && readFloat(out.z);
    }

    // NIF stores a quaternion W FIRST; odai::math::Quaternion is (x, y, z, w).
    bool readQuaternion(Quaternion& out) {
        float w = 0.0f;
        if (!readFloat(w) || !readFloat(out.x) || !readFloat(out.y) || !readFloat(out.z)) {
            return false;
        }
        out.w = w;
        return true;
    }

    bool skip(std::size_t count) {
        if (m_offset + count > m_size) {
            return false;
        }
        m_offset += count;
        return true;
    }

    // Bulk copy for an array whose element type is fixed-width and needs no
    // conversion -- the B-spline control point array is 60 KB of shorts.
    bool readRawBytes(void* out, std::size_t count) { return readRaw(out, count); }

    [[nodiscard]] std::size_t remaining() const { return m_size - m_offset; }

private:
    bool readRaw(void* out, std::size_t count) {
        if (m_offset + count > m_size) {
            return false;
        }
        std::memcpy(out, m_data + m_offset, count);
        m_offset += count;
        return true;
    }

    const std::uint8_t* m_data;
    std::size_t m_size;
    std::size_t m_offset = 0;
};

// Trailing per-key data that carries no value this reader uses: quadratic
// tangents, TBC parameters. Returned in bytes, or SIZE_MAX for an interpolation
// mode whose stride is unknown -- which must abort the track rather than guess.
std::size_t extraKeyBytes(std::uint32_t interpolation, std::size_t valueFloats) {
    switch (interpolation) {
        case kKeyLinear:
        case kKeyConst:
            return 0;
        // Forward and backward tangents, one per value component.
        case kKeyQuadratic:
            return valueFloats * 2u * sizeof(float);
        // Tension, bias, continuity.
        case kKeyTbc:
            return 3u * sizeof(float);
        default:
            return std::numeric_limits<std::size_t>::max();
    }
}

// ---------------------------------------------------------------------------
// B-spline-compressed interpolators
//
// Bethesda stores a human's animation twice over: the bones that barely move
// (pelvis, spine, toes) as ordinary keyframes, and the ones that carry the
// motion -- both arms, both legs, the head -- as CUBIC B-SPLINES with 16-bit
// quantized control points. Decoding only the first kind gets you a clip with
// 42 of 58 tracks bound that still renders a T-pose, because the 16 it skipped
// are exactly the limbs. That was the state before this, and the track count
// is what made it look nearly right.
//
// Three blocks per curve: the interpolator names an offset per channel into a
// shared NiBSplineData (one flat array of shorts for the whole file), and a
// shared NiBSplineBasisData says how many control points each channel spends.
// Decoded curves are SAMPLED into ordinary keys here, so nothing downstream --
// the sampler, the basis change, the clip type -- needs to know they existed.

// Cubic. Every B-spline in these files is degree 3; the format does not carry
// a degree field, which is a strong hint it was never anything else.
constexpr std::uint32_t kBSplineDegree = 3;
// Two per curve segment. Sampling at the control points alone visibly flattens
// a fast joint, and the keys are cheap next to the geometry they pose.
constexpr std::uint32_t kBSplineSamplesPerSegment = 2;
constexpr std::uint32_t kBSplineNoChannel = 0xffffffffu;

struct BSplineChannel {
    std::uint32_t offset = kBSplineNoChannel;
    float bias = 0.0f;
    float multiplier = 0.0f;
};

// The clamped uniform knot vector these curves use: `degree + 1` repeats at
// each end, integers in between. Repeating the ends is what makes the curve
// pass THROUGH its first and last control point, which an animation needs --
// an unclamped spline starts a quarter of the way into the pose.
float bSplineKnot(std::uint32_t index, std::uint32_t controlPointCount) {
    const auto span = static_cast<float>(controlPointCount - kBSplineDegree);
    const auto shifted = static_cast<float>(static_cast<int>(index) - static_cast<int>(kBSplineDegree));
    return std::clamp(shifted, 0.0f, span);
}

// De Boor's algorithm on one channel's control points. `u` is in [0, n-degree],
// the knot vector's own domain.
//
// `components` are interleaved per control point, and all of them are evaluated
// together because they share the basis -- a quaternion's four channels must be
// weighted identically or the result is not a rotation.
void evaluateBSpline(
    const std::vector<float>& controlPoints,
    std::uint32_t controlPointCount,
    std::uint32_t components,
    float u,
    float* out
) {
    const auto lastSpan = controlPointCount - kBSplineDegree - 1u;
    auto span = static_cast<std::uint32_t>(std::floor(u));
    span = std::min(span, lastSpan);

    // Working set: the degree+1 control points that influence this span.
    float working[(kBSplineDegree + 1u) * 4u] = {};
    for (std::uint32_t i = 0; i <= kBSplineDegree; ++i) {
        const std::uint32_t point = span + i;
        for (std::uint32_t c = 0; c < components; ++c) {
            const std::size_t source = (static_cast<std::size_t>(point) * components) + c;
            working[(i * 4u) + c] =
                source < controlPoints.size() ? controlPoints[source] : 0.0f;
        }
    }

    for (std::uint32_t round = 1; round <= kBSplineDegree; ++round) {
        for (std::uint32_t i = kBSplineDegree; i >= round; --i) {
            const std::uint32_t knotIndex = span + i;
            const float low = bSplineKnot(knotIndex, controlPointCount);
            const float high = bSplineKnot(knotIndex + kBSplineDegree + 1u - round, controlPointCount);
            const float denominator = high - low;
            const float alpha = denominator > 0.0f ? ((u - low) / denominator) : 0.0f;
            for (std::uint32_t c = 0; c < components; ++c) {
                working[(i * 4u) + c] = ((1.0f - alpha) * working[((i - 1u) * 4u) + c]) +
                    (alpha * working[(i * 4u) + c]);
            }
        }
    }
    for (std::uint32_t c = 0; c < components; ++c) {
        out[c] = working[(kBSplineDegree * 4u) + c];
    }
}

// Dequantizes one channel's control points out of the file's shared short
// array. 32767 rather than 32768: the format's own range is symmetric, and the
// half-LSB it costs is far below what a 16-bit rotation resolves anyway.
bool readBSplineChannel(
    const BSplineChannel& channel,
    const std::vector<std::int16_t>& shortControlPoints,
    std::uint32_t controlPointCount,
    std::uint32_t components,
    std::vector<float>& out
) {
    if (channel.offset == kBSplineNoChannel || controlPointCount <= kBSplineDegree) {
        return false;
    }
    const std::size_t needed =
        static_cast<std::size_t>(controlPointCount) * components;
    if (static_cast<std::size_t>(channel.offset) + needed > shortControlPoints.size()) {
        return false;
    }
    out.resize(needed);
    for (std::size_t i = 0; i < needed; ++i) {
        const float quantized =
            static_cast<float>(shortControlPoints[channel.offset + i]) / 32767.0f;
        out[i] = channel.bias + (channel.multiplier * quantized);
    }
    return true;
}

// Reads a NiBSplineCompTransformInterpolator and bakes its curves into keys.
//
// Layout, in order, all little-endian:
//   NiBSplineInterpolator          startTime, stopTime, splineData, basisData
//   NiBSplineTransformInterpolator translation(3), rotation(4, W first),
//                                  scale, then one u32 offset per channel
//   ...CompTransformInterpolator   bias/multiplier per channel, in the same
//                                  channel order
// An offset of 0xFFFFFFFF means the channel is not animated, and the static
// value ahead of the offsets is the pose for it.
bool readBSplineTransformInterpolator(
    const std::vector<std::uint8_t>& bytes,
    const NifBlockSummary& summary,
    std::size_t interpolatorBlock,
    KfBoneTrack& outTrack
) {
    BlockCursor cursor(
        bytes.data() + summary.blockStarts[interpolatorBlock],
        summary.blockSizes[interpolatorBlock]);

    float startTime = 0.0f;
    float stopTime = 0.0f;
    std::int32_t splineDataRef = -1;
    std::int32_t basisDataRef = -1;
    Vector3 staticTranslation{};
    Quaternion staticRotation{};
    float staticScale = 1.0f;
    BSplineChannel translation;
    BSplineChannel rotation;
    BSplineChannel scale;
    if (!cursor.readFloat(startTime) || !cursor.readFloat(stopTime) ||
        !cursor.readI32(splineDataRef) || !cursor.readI32(basisDataRef) ||
        !cursor.readVector3(staticTranslation) || !cursor.readQuaternion(staticRotation) ||
        !cursor.readFloat(staticScale) || !cursor.readU32(translation.offset) ||
        !cursor.readU32(rotation.offset) || !cursor.readU32(scale.offset) ||
        !cursor.readFloat(translation.bias) || !cursor.readFloat(translation.multiplier) ||
        !cursor.readFloat(rotation.bias) || !cursor.readFloat(rotation.multiplier) ||
        !cursor.readFloat(scale.bias) || !cursor.readFloat(scale.multiplier)) {
        return false;
    }
    if (!std::isfinite(startTime) || !std::isfinite(stopTime) || stopTime <= startTime) {
        return false;
    }
    if (splineDataRef < 0 || basisDataRef < 0 ||
        static_cast<std::size_t>(splineDataRef) >= summary.blockTypeNames.size() ||
        static_cast<std::size_t>(basisDataRef) >= summary.blockTypeNames.size() ||
        summary.blockTypeNames[static_cast<std::size_t>(splineDataRef)] != "NiBSplineData" ||
        summary.blockTypeNames[static_cast<std::size_t>(basisDataRef)] != "NiBSplineBasisData") {
        return false;
    }

    // NiBSplineBasisData: how many control points EVERY channel in the file
    // spends. It is shared, which is why one number covers three channels.
    std::uint32_t controlPointCount = 0;
    {
        const auto basisBlock = static_cast<std::size_t>(basisDataRef);
        BlockCursor basisCursor(
            bytes.data() + summary.blockStarts[basisBlock], summary.blockSizes[basisBlock]);
        if (!basisCursor.readU32(controlPointCount) ||
            controlPointCount <= kBSplineDegree || controlPointCount > 4096u) {
            return false;
        }
    }

    // NiBSplineData: floats first, then the shorts every compressed channel
    // indexes into.
    std::vector<std::int16_t> shortControlPoints;
    {
        const auto dataBlock = static_cast<std::size_t>(splineDataRef);
        BlockCursor dataCursor(
            bytes.data() + summary.blockStarts[dataBlock], summary.blockSizes[dataBlock]);
        std::uint32_t floatCount = 0;
        if (!dataCursor.readU32(floatCount) ||
            !dataCursor.skip(static_cast<std::size_t>(floatCount) * sizeof(float))) {
            return false;
        }
        std::uint32_t shortCount = 0;
        if (!dataCursor.readU32(shortCount) ||
            (static_cast<std::size_t>(shortCount) * sizeof(std::int16_t)) > dataCursor.remaining()) {
            return false;
        }
        shortControlPoints.resize(shortCount);
        if (shortCount != 0u &&
            !dataCursor.readRawBytes(
                shortControlPoints.data(),
                static_cast<std::size_t>(shortCount) * sizeof(std::int16_t))) {
            return false;
        }
    }

    std::vector<float> translationPoints;
    std::vector<float> rotationPoints;
    std::vector<float> scalePoints;
    const bool hasTranslation = readBSplineChannel(
        translation, shortControlPoints, controlPointCount, 3u, translationPoints);
    const bool hasRotation =
        readBSplineChannel(rotation, shortControlPoints, controlPointCount, 4u, rotationPoints);
    const bool hasScale =
        readBSplineChannel(scale, shortControlPoints, controlPointCount, 1u, scalePoints);

    const std::uint32_t segments = controlPointCount - kBSplineDegree;
    const std::uint32_t sampleCount = (segments * kBSplineSamplesPerSegment) + 1u;
    if (hasTranslation || hasRotation || hasScale) {
        for (std::uint32_t s = 0; s < sampleCount; ++s) {
            const float fraction = static_cast<float>(s) / static_cast<float>(sampleCount - 1u);
            const float time = startTime + (fraction * (stopTime - startTime));
            // Nudged off the very end of the domain: the last knot is the first
            // parameter with no span to its right, and de Boor there indexes one
            // control point past the curve.
            const float u = std::min(
                fraction * static_cast<float>(segments),
                static_cast<float>(segments) - 1e-4f);
            float value[4] = {};
            if (hasTranslation) {
                evaluateBSpline(translationPoints, controlPointCount, 3u, u, value);
                outTrack.translationKeys.push_back(
                    KfVector3Key{time, Vector3{value[0], value[1], value[2]}});
            }
            if (hasRotation) {
                evaluateBSpline(rotationPoints, controlPointCount, 4u, u, value);
                // W first in the file, as everywhere else in NIF. A blended
                // quaternion is not unit-length, and the sampler's slerp needs
                // one that is.
                Quaternion sampled{value[1], value[2], value[3], value[0]};
                const float length = std::sqrt(
                    (sampled.x * sampled.x) + (sampled.y * sampled.y) + (sampled.z * sampled.z) +
                    (sampled.w * sampled.w));
                if (length > 1e-6f) {
                    sampled.x /= length;
                    sampled.y /= length;
                    sampled.z /= length;
                    sampled.w /= length;
                    outTrack.rotationKeys.push_back(KfQuaternionKey{time, sampled});
                }
            }
            if (hasScale) {
                evaluateBSpline(scalePoints, controlPointCount, 1u, u, value);
                outTrack.scaleKeys.push_back(
                    KfVector3Key{time, Vector3{value[0], value[0], value[0]}});
            }
        }
    }

    // Channels the curve did not carry fall back to the interpolator's own
    // static pose, exactly as they do for an uncompressed track.
    if (outTrack.translationKeys.empty() && !isUnsetFloat(staticTranslation.x) &&
        !isUnsetFloat(staticTranslation.y) && !isUnsetFloat(staticTranslation.z)) {
        outTrack.translationKeys.push_back(KfVector3Key{startTime, staticTranslation});
    }
    if (outTrack.rotationKeys.empty() && !isUnsetFloat(staticRotation.w) &&
        !isUnsetFloat(staticRotation.x) && !isUnsetFloat(staticRotation.y) &&
        !isUnsetFloat(staticRotation.z)) {
        outTrack.rotationKeys.push_back(KfQuaternionKey{startTime, staticRotation});
    }
    if (outTrack.scaleKeys.empty() && !isUnsetFloat(staticScale) && staticScale > 0.0f) {
        outTrack.scaleKeys.push_back(
            KfVector3Key{startTime, Vector3{staticScale, staticScale, staticScale}});
    }
    return !outTrack.translationKeys.empty() || !outTrack.rotationKeys.empty() ||
        !outTrack.scaleKeys.empty();
}

// A KeyGroup<T>: count, then (only when non-empty) the interpolation mode, then
// the keys. The "interpolation is absent when the count is zero" rule is easy
// to miss and shifts everything after it by four bytes.
template <typename ReadValueFn>
bool readKeyGroup(
    BlockCursor& cursor, std::size_t valueFloats, ReadValueFn readValue, std::size_t& outCount
) {
    std::uint32_t keyCount = 0;
    if (!cursor.readU32(keyCount)) {
        return false;
    }
    outCount = keyCount;
    if (keyCount == 0) {
        return true;
    }
    std::uint32_t interpolation = 0;
    if (!cursor.readU32(interpolation)) {
        return false;
    }
    const std::size_t extra = extraKeyBytes(interpolation, valueFloats);
    if (extra == std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    for (std::uint32_t i = 0; i < keyCount; ++i) {
        float time = 0.0f;
        if (!cursor.readFloat(time) || !readValue(cursor, time) || !cursor.skip(extra)) {
            return false;
        }
    }
    return true;
}

// NiTransformData: the rotation channel first (its count and type are inline
// rather than in a KeyGroup, and XYZ_ROTATION_KEY replaces the quaternion keys
// with three separate float KeyGroups), then translation, then scale.
bool readTransformData(
    const std::uint8_t* blockData, std::size_t blockSize, KfBoneTrack& track, std::string& outError
) {
    BlockCursor cursor(blockData, blockSize);
    std::uint32_t rotationKeyCount = 0;
    if (!cursor.readU32(rotationKeyCount)) {
        outError = "NiTransformData truncated at rotation key count";
        return false;
    }
    std::uint32_t rotationType = 0;
    if (rotationKeyCount != 0 && !cursor.readU32(rotationType)) {
        outError = "NiTransformData truncated at rotation key type";
        return false;
    }

    if (rotationKeyCount != 0 && rotationType == kKeyXyzRotation) {
        // Euler form: three independent float channels, each its own KeyGroup,
        // sampled and recombined here into one quaternion track. The counts
        // need not match across axes, so the merge is by axis at its own key
        // times -- and rather than resample, this reader takes the union of the
        // three time sets, which for Bethesda's exports are identical anyway.
        std::vector<KfVector3Key> eulerByAxis[3];
        for (int axis = 0; axis < 3; ++axis) {
            std::size_t count = 0;
            const int capturedAxis = axis;
            if (!readKeyGroup(
                    cursor, 1u,
                    [&](BlockCursor& keyCursor, float time) {
                        float value = 0.0f;
                        if (!keyCursor.readFloat(value)) {
                            return false;
                        }
                        KfVector3Key key{};
                        key.time = time;
                        // Parked in the matching component so the merge below
                        // can read all three tracks the same way.
                        (&key.value.x)[capturedAxis] = value;
                        eulerByAxis[capturedAxis].push_back(key);
                        return true;
                    },
                    count)) {
                outError = "NiTransformData XYZ rotation channel unreadable";
                return false;
            }
        }
        std::vector<float> times;
        for (const auto& axisKeys : eulerByAxis) {
            for (const KfVector3Key& key : axisKeys) {
                times.push_back(key.time);
            }
        }
        std::sort(times.begin(), times.end());
        times.erase(std::unique(times.begin(), times.end()), times.end());
        const auto sampleAxis = [](const std::vector<KfVector3Key>& keys, int axis, float time) {
            if (keys.empty()) {
                return 0.0f;
            }
            if (time <= keys.front().time) {
                return (&keys.front().value.x)[axis];
            }
            if (time >= keys.back().time) {
                return (&keys.back().value.x)[axis];
            }
            for (std::size_t i = 1; i < keys.size(); ++i) {
                if (keys[i].time < time) {
                    continue;
                }
                const float span = keys[i].time - keys[i - 1].time;
                const float t = span > 0.0f ? (time - keys[i - 1].time) / span : 0.0f;
                const float a = (&keys[i - 1].value.x)[axis];
                const float b = (&keys[i].value.x)[axis];
                return a + ((b - a) * t);
            }
            return (&keys.back().value.x)[axis];
        };
        for (const float time : times) {
            const float rx = sampleAxis(eulerByAxis[0], 0, time);
            const float ry = sampleAxis(eulerByAxis[1], 1, time);
            const float rz = sampleAxis(eulerByAxis[2], 2, time);
            // NIF applies X, then Y, then Z, each about the parent's axes.
            const float halfX = rx * 0.5f;
            const float halfY = ry * 0.5f;
            const float halfZ = rz * 0.5f;
            const float sx = std::sin(halfX), cx = std::cos(halfX);
            const float sy = std::sin(halfY), cy = std::cos(halfY);
            const float sz = std::sin(halfZ), cz = std::cos(halfZ);
            KfQuaternionKey key{};
            key.time = time;
            key.value.w = (cx * cy * cz) + (sx * sy * sz);
            key.value.x = (sx * cy * cz) - (cx * sy * sz);
            key.value.y = (cx * sy * cz) + (sx * cy * sz);
            key.value.z = (cx * cy * sz) - (sx * sy * cz);
            track.rotationKeys.push_back(key);
        }
    } else if (rotationKeyCount != 0) {
        const std::size_t extra = (rotationType == kKeyTbc) ? (3u * sizeof(float)) : 0u;
        if (rotationType != kKeyLinear && rotationType != kKeyQuadratic &&
            rotationType != kKeyTbc && rotationType != kKeyConst) {
            outError = "NiTransformData unknown rotation key type";
            return false;
        }
        track.rotationKeys.reserve(rotationKeyCount);
        for (std::uint32_t i = 0; i < rotationKeyCount; ++i) {
            KfQuaternionKey key{};
            if (!cursor.readFloat(key.time) || !cursor.readQuaternion(key.value) ||
                !cursor.skip(extra)) {
                outError = "NiTransformData truncated in rotation keys";
                return false;
            }
            track.rotationKeys.push_back(key);
        }
    }

    std::size_t translationCount = 0;
    if (!readKeyGroup(
            cursor, 3u,
            [&](BlockCursor& keyCursor, float time) {
                KfVector3Key key{};
                key.time = time;
                if (!keyCursor.readVector3(key.value)) {
                    return false;
                }
                track.translationKeys.push_back(key);
                return true;
            },
            translationCount)) {
        outError = "NiTransformData translation channel unreadable";
        return false;
    }

    std::size_t scaleCount = 0;
    if (!readKeyGroup(
            cursor, 1u,
            [&](BlockCursor& keyCursor, float time) {
                float value = 1.0f;
                if (!keyCursor.readFloat(value)) {
                    return false;
                }
                KfVector3Key key{};
                key.time = time;
                key.value = Vector3{value, value, value};
                track.scaleKeys.push_back(key);
                return true;
            },
            scaleCount)) {
        outError = "NiTransformData scale channel unreadable";
        return false;
    }
    return true;
}

}  // namespace

bool parseKfAnimation(
    const std::vector<std::uint8_t>& bytes, KfAnimation& outAnimation, std::string& outError
) {
    outAnimation = KfAnimation{};

    NifBlockSummary summary;
    if (!parseNifBlockSummary(bytes, summary, outError)) {
        return false;
    }

    std::size_t sequenceBlock = summary.blockTypeNames.size();
    for (std::size_t i = 0; i < summary.blockTypeNames.size(); ++i) {
        if (summary.blockTypeNames[i] == "NiControllerSequence") {
            sequenceBlock = i;
            break;
        }
    }
    if (sequenceBlock == summary.blockTypeNames.size()) {
        outError = "no NiControllerSequence block (not an animation .kf?)";
        return false;
    }

    const auto stringAt = [&](std::int32_t index) -> const std::string* {
        if (index < 0 || static_cast<std::size_t>(index) >= summary.strings.size()) {
            return nullptr;
        }
        return &summary.strings[static_cast<std::size_t>(index)];
    };

    BlockCursor cursor(
        bytes.data() + summary.blockStarts[sequenceBlock], summary.blockSizes[sequenceBlock]);

    std::int32_t nameIndex = 0;
    std::uint32_t controlledBlockCount = 0;
    std::uint32_t arrayGrowBy = 0;
    if (!cursor.readI32(nameIndex) || !cursor.readU32(controlledBlockCount) ||
        !cursor.readU32(arrayGrowBy)) {
        outError = "NiControllerSequence truncated at header";
        return false;
    }
    if (const std::string* name = stringAt(nameIndex)) {
        outAnimation.name = *name;
    }
    // 29 bytes each; a count that cannot fit is the first sign the layout
    // assumption is wrong for this file's version.
    if (static_cast<std::size_t>(controlledBlockCount) * 29u > cursor.remaining()) {
        outError = "NiControllerSequence controlled-block count does not fit the block";
        return false;
    }

    struct ControlledBlock {
        std::int32_t interpolator = -1;
        std::string nodeName;
    };
    std::vector<ControlledBlock> controlledBlocks;
    controlledBlocks.reserve(controlledBlockCount);
    for (std::uint32_t i = 0; i < controlledBlockCount; ++i) {
        std::int32_t interpolator = 0;
        std::int32_t controller = 0;
        std::uint8_t priority = 0;
        std::int32_t nodeName = 0;
        std::int32_t propertyType = 0;
        std::int32_t controllerType = 0;
        std::int32_t controllerId = 0;
        std::int32_t interpolatorId = 0;
        if (!cursor.readI32(interpolator) || !cursor.readI32(controller) ||
            !cursor.readU8(priority) || !cursor.readI32(nodeName) ||
            !cursor.readI32(propertyType) || !cursor.readI32(controllerType) ||
            !cursor.readI32(controllerId) || !cursor.readI32(interpolatorId)) {
            outError = "NiControllerSequence truncated in controlled blocks";
            return false;
        }
        // The layout check that matters. Every one of these is either a valid
        // index or the -1 "absent" marker; a stride error makes them wander out
        // of range within the first few entries.
        if (interpolator >= static_cast<std::int32_t>(summary.blockTypeNames.size()) ||
            nodeName >= static_cast<std::int32_t>(summary.strings.size()) ||
            interpolator < -1 || nodeName < -1) {
            outError = "NiControllerSequence controlled block out of range (layout mismatch)";
            return false;
        }
        ControlledBlock entry;
        entry.interpolator = interpolator;
        if (const std::string* resolved = stringAt(nodeName)) {
            entry.nodeName = *resolved;
        }
        controlledBlocks.push_back(std::move(entry));
    }

    float weight = 0.0f;
    std::int32_t textKeys = 0;
    if (!cursor.readFloat(weight) || !cursor.readI32(textKeys) ||
        !cursor.readU32(outAnimation.cycleType)) {
        outError = "NiControllerSequence truncated after controlled blocks";
        return false;
    }
    float frequency = 0.0f;
    if (!cursor.readFloat(frequency) || !cursor.readFloat(outAnimation.startTime) ||
        !cursor.readFloat(outAnimation.stopTime)) {
        outError = "NiControllerSequence truncated at timing fields";
        return false;
    }
    if (!std::isfinite(outAnimation.startTime) || !std::isfinite(outAnimation.stopTime) ||
        outAnimation.stopTime < outAnimation.startTime) {
        outError = "NiControllerSequence implausible start/stop time (layout mismatch)";
        return false;
    }

    outAnimation.stats.controlledBlocks = controlledBlocks.size();
    outAnimation.tracks.reserve(controlledBlocks.size());
    for (const ControlledBlock& entry : controlledBlocks) {
        if (entry.interpolator < 0 || entry.nodeName.empty()) {
            continue;
        }
        const std::size_t interpolatorBlock = static_cast<std::size_t>(entry.interpolator);
        if (summary.blockTypeNames[interpolatorBlock] == "NiBSplineCompTransformInterpolator") {
            KfBoneTrack track;
            track.nodeName = entry.nodeName;
            if (readBSplineTransformInterpolator(bytes, summary, interpolatorBlock, track)) {
                ++outAnimation.stats.bSplineInterpolators;
                outAnimation.tracks.push_back(std::move(track));
            } else {
                ++outAnimation.stats.unsupportedInterpolators;
                outAnimation.stats.unsupportedNodes.push_back(entry.nodeName);
            }
            continue;
        }
        if (summary.blockTypeNames[interpolatorBlock] != "NiTransformInterpolator") {
            // Float and point interpolators land here: a bone's visibility or a
            // morph weight, neither of which is a transform.
            ++outAnimation.stats.unsupportedInterpolators;
            outAnimation.stats.unsupportedNodes.push_back(entry.nodeName);
            continue;
        }
        ++outAnimation.stats.transformInterpolators;

        BlockCursor interpolatorCursor(
            bytes.data() + summary.blockStarts[interpolatorBlock],
            summary.blockSizes[interpolatorBlock]);
        Vector3 staticTranslation{};
        Quaternion staticRotation{};
        float staticScale = 1.0f;
        std::int32_t dataRef = -1;
        if (!interpolatorCursor.readVector3(staticTranslation) ||
            !interpolatorCursor.readQuaternion(staticRotation) ||
            !interpolatorCursor.readFloat(staticScale) || !interpolatorCursor.readI32(dataRef)) {
            outError = "NiTransformInterpolator truncated";
            return false;
        }

        KfBoneTrack track;
        track.nodeName = entry.nodeName;
        if (dataRef >= 0 && static_cast<std::size_t>(dataRef) < summary.blockTypeNames.size() &&
            summary.blockTypeNames[static_cast<std::size_t>(dataRef)] == "NiTransformData") {
            const std::size_t dataBlock = static_cast<std::size_t>(dataRef);
            if (!readTransformData(
                    bytes.data() + summary.blockStarts[dataBlock], summary.blockSizes[dataBlock],
                    track, outError)) {
                return false;
            }
        }
        // The interpolator's own values are the pose for any channel the data
        // block did not key -- a bone held at a fixed offset for the whole clip
        // carries no keys at all, only these. Skipping them leaves such a bone
        // at bind pose, which is wrong by exactly the offset the clip meant to
        // apply.
        if (track.translationKeys.empty() && !isUnsetFloat(staticTranslation.x) &&
            !isUnsetFloat(staticTranslation.y) && !isUnsetFloat(staticTranslation.z)) {
            track.translationKeys.push_back(KfVector3Key{0.0f, staticTranslation});
        }
        if (track.rotationKeys.empty() && !isUnsetFloat(staticRotation.w) &&
            !isUnsetFloat(staticRotation.x) && !isUnsetFloat(staticRotation.y) &&
            !isUnsetFloat(staticRotation.z)) {
            track.rotationKeys.push_back(KfQuaternionKey{0.0f, staticRotation});
        }
        if (track.scaleKeys.empty() && !isUnsetFloat(staticScale) && staticScale > 0.0f) {
            track.scaleKeys.push_back(
                KfVector3Key{0.0f, Vector3{staticScale, staticScale, staticScale}});
        }
        if (track.translationKeys.empty() && track.rotationKeys.empty() &&
            track.scaleKeys.empty()) {
            continue;
        }
        outAnimation.tracks.push_back(std::move(track));
    }

    // Rebase to zero so the clip's own duration is what the sampler wraps on.
    const float startTime = outAnimation.startTime;
    if (startTime != 0.0f) {
        for (KfBoneTrack& track : outAnimation.tracks) {
            for (KfVector3Key& key : track.translationKeys) {
                key.time -= startTime;
            }
            for (KfQuaternionKey& key : track.rotationKeys) {
                key.time -= startTime;
            }
            for (KfVector3Key& key : track.scaleKeys) {
                key.time -= startTime;
            }
        }
        outAnimation.stopTime -= startTime;
        outAnimation.startTime = 0.0f;
    }
    return true;
}

}  // namespace odai::importer::fnv
