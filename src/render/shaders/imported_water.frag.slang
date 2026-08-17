import camera_uniform;
import chunk_push_constants;

[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

[[vk::binding(6, 0)]]
Sampler2D<float4> normalDepthTexture;
[[vk::binding(5, 0)]]
Sampler2D<float4> opaqueSceneColorTexture;
[[vk::binding(18, 0)]]
Sampler2D<float4> planarReflectionTexture;

#ifdef ODAI_RT_REFLECTIONS
[[vk::binding(12, 0)]]
RaytracingAccelerationStructure reflectionScene;
#endif

static const float kPi = 3.14159265359;
static const float kAirIor = 1.0;
static const float kWaterIor = 1.333;
static const float kAirToWaterEta = kAirIor / kWaterIor;
static const uint kBindlessTextureIndexWaterNormal = 8u;
// Bethesda rivers are fed by dark rock and conifer forest, not a white sand
// shelf.  The previous Mediterranean palette was bright enough that the
// tonemapper turned a broad river into an almost emissive cyan card.  Keep the
// shallow water readable, but let reflected sky and the actual riverbed supply
// most of its colour.
static const float3 kWaterShallowTint = float3(0.025, 0.055, 0.045);
static const float3 kWaterDeepTint = float3(0.006, 0.020, 0.020);
// Bethesda units are much smaller than metres. The previous coefficients
// removed most red over a typical 100-unit river column while retaining blue,
// which made even a brown riverbed read as an emissive cyan sheet. Keep the
// wavelength dependence restrained: Skyrim's water is mostly the refracted
// bottom and reflected environment, not a body-colour swatch.
static const float3 kWaterAbsorption = float3(0.0085, 0.0045, 0.0030);

struct FragmentInput
{
    float4 inPosition : SV_Position;
    float3 inWorldPosition : TEXCOORD0;
    float2 inUv : TEXCOORD1;
    nointerpolation uint inNormalTextureSlot : TEXCOORD2;
    nointerpolation uint inFlowTextureSlot : TEXCOORD3;
};

struct SceneWaterSample
{
    float valid;
    float thickness;
    float contact;
    float depthEdge;
    float3 bottomTint;
};

struct WaterSurfaceSample
{
    float3 normal;
    float2 flow;
    float speed;
};

float3 proceduralSkyNoSunDisk(float3 direction)
{
    const float3 dir = normalize(direction);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float rayleighStrength = max(camera.skyConfig0.x, 0.01);
    const float mieStrength = max(camera.skyConfig0.y, 0.01);
    const float skyExposure = max(camera.skyConfig0.w, 0.01);
    const float sunAltitude = clamp(toSun.y, -1.0, 1.0);
    const float dayFactor = smoothstep(0.05, 0.65, sunAltitude);
    const float twilightFactor = 1.0 - dayFactor;
    const float horizonBand = clamp(1.0 - (abs(sunAltitude) / 0.35), 0.0, 1.0);
    const float warmAmount = twilightFactor * pow(horizonBand, 1.2);
    const float pinkAmount = warmAmount * clamp((0.10 - sunAltitude) / 0.30, 0.0, 1.0);

    const float horizonT = clamp((dir.y * 0.5) + 0.5, 0.0, 1.0);
    const float skyT = pow(horizonT, 0.35);

    float3 horizonRayleigh = lerp(float3(0.54, 0.70, 1.00), float3(0.74, 0.44, 0.52), warmAmount);
    float3 horizonMie = lerp(float3(1.00, 0.74, 0.42), float3(1.18, 0.54, 0.30), warmAmount);
    horizonRayleigh = lerp(horizonRayleigh, float3(0.70, 0.36, 0.68), pinkAmount * 0.70);
    horizonMie = lerp(horizonMie, float3(1.08, 0.46, 0.72), pinkAmount * 0.85);

    const float zenithWarm = twilightFactor * 0.58;
    const float3 zenithRayleigh = lerp(float3(0.06, 0.24, 0.54), float3(0.16, 0.12, 0.30), zenithWarm);
    const float3 zenithMie = lerp(float3(0.22, 0.20, 0.15), float3(0.30, 0.18, 0.24), zenithWarm);

    const float3 horizonColor =
        (horizonRayleigh * rayleighStrength) +
        (horizonMie * (mieStrength * 0.58));
    const float3 zenithColor =
        (zenithRayleigh * rayleighStrength) +
        (zenithMie * (mieStrength * 0.25));
    const float3 sky = lerp(horizonColor, zenithColor, skyT);

    const float3 groundColor = float3(0.05, 0.06, 0.07);
    const float belowHorizon = clamp(-dir.y, 0.0, 1.0);
    const float3 horizonGroundColor = horizonColor * 0.32;
    const float3 ground = lerp(horizonGroundColor, groundColor, pow(belowHorizon, 0.55));
    const float skyWeight = smoothstep(-0.18, 0.02, dir.y);
    const float3 color = lerp(ground, sky, skyWeight);
    return max(color * skyExposure, float3(0.0, 0.0, 0.0));
}

float3 proceduralWaterReflectionSky(float3 direction)
{
    const float3 dir = normalize(direction);
    const float3 toSun = -normalize(camera.sunDirectionIntensity.xyz);
    const float sunDot = saturate(dot(dir, toSun));
    const float sunDisk = pow(sunDot, 1200.0);
    const float sunHalo = pow(sunDot, 42.0);
    const float3 proceduralSky = proceduralSkyNoSunDisk(dir);
    // Reflections must use the same authored weather family as the visible
    // sky.  Sampling only the procedural clear sky made an overcast Skyrim
    // river reflect saturated summer blue while every object around it was
    // lit by the WTHR record.
    const float weatherWeight = saturate(camera.weatherSkyUpper.w);
    const float height = saturate(dir.y);
    const float lowerToUpper = smoothstep(0.10, 0.72, height);
    const float horizonToLower = smoothstep(0.00, 0.18, height);
    const float3 weatherUpperLower = lerp(
        camera.weatherSkyLower.xyz,
        camera.weatherSkyUpper.xyz,
        lowerToUpper);
    const float3 weatherSky = lerp(
        camera.weatherHorizon.xyz,
        weatherUpperLower,
        horizonToLower);
    // The fullscreen sky subsequently composites cloud sheets, but this pass
    // has only the WTHR gradient. Reflecting that unobscured HDR gradient at
    // full strength is especially wrong for SkyrimCloudy: it substitutes a
    // saturated cyan emitter for the missing cloud cover. Approximate the
    // cover by reducing and partially neutralising the reflected gradient;
    // sharp sun glints are added separately below and remain energetic.
    const float weatherLuminance = dot(weatherSky, float3(0.2126, 0.7152, 0.0722));
    const float3 cloudyWeatherReflection =
        lerp(weatherLuminance.xxx, weatherSky, 0.38) * 0.50;
    const float3 baseSky = lerp(proceduralSky, cloudyWeatherReflection, weatherWeight);
    return baseSky + (camera.sunColorShadow.xyz * ((sunHalo * 0.032) + (sunDisk * 0.24)));
}

float3 applyAtmosphericFog(float3 litColor, float3 worldPosition, float viewDepth, float3 sunDirection)
{
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(worldPosition - cameraWorld);
    // Use the same Bethesda-unit calibration as imported_static. This used to
    // apply metre-scale Rayleigh/Mie coefficients directly to game units, for
    // an e-fold distance of only a few hundred units. Riverwood's water viewed
    // from 1-2k units away was therefore almost 100% sky inscatter: a flat
    // white polygon even though its geometry, normals and XCLW were correct.
    const float weatherWeight = saturate(camera.weatherSkyUpper.w);
    const float authoredFogFar = camera.weatherFog.w;
    const float aerialEFoldDistance =
        (authoredFogFar > 1.0) ? (authoredFogFar / 1.4) : 15000.0;
    const float3 extinctionPerUnit =
        float3(1.35, 1.12, 0.85) / aerialEFoldDistance;
    const float3 transmittance = exp(-max(viewDepth, 0.0) * extinctionPerUnit);
    const float sunAlignment = saturate(dot(viewDir, -sunDirection));
    const float3 skyInscatter =
        lerp(float3(0.40, 0.56, 0.82), camera.weatherFog.xyz, weatherWeight);
    const float3 sunInscatter = float3(1.00, 0.86, 0.66);
    const float3 inscatterColor =
        lerp(skyInscatter, sunInscatter, pow(sunAlignment, 6.0) * 0.85);
    const float skyStrength = max(camera.skyConfig0.x, 0.01);
    const float sunLuminance = dot(camera.sunColorShadow.xyz, float3(0.30, 0.59, 0.11));
    const float3 inscatter =
        inscatterColor * (skyStrength * 0.42) * max(sunLuminance, 0.03);
    return (litColor * transmittance) +
        (inscatter * (float3(1.0, 1.0, 1.0) - transmittance));
}

float stableWaterNoise(float2 p)
{
    const float2 i = floor(p);
    const float2 f = frac(p);
    const float2 u = f * f * (3.0 - (2.0 * f));
    const float a = frac(sin(dot(i + float2(0.0, 0.0), float2(127.1, 311.7))) * 43758.5453123);
    const float b = frac(sin(dot(i + float2(1.0, 0.0), float2(127.1, 311.7))) * 43758.5453123);
    const float c = frac(sin(dot(i + float2(0.0, 1.0), float2(127.1, 311.7))) * 43758.5453123);
    const float d = frac(sin(dot(i + float2(1.0, 1.0), float2(127.1, 311.7))) * 43758.5453123);
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float3 decodeViewNormal(float3 encodedNormal)
{
    return normalize((encodedNormal * 2.0) - 1.0);
}

float2 waterScreenUv(float4 screenPosition)
{
    // SV_Position is always in render-target pixels.  normalDepth used to be
    // unconditionally half resolution, but the merged depth prepass makes it
    // full resolution.  Multiplying its dimensions by two therefore addressed
    // the upper-left quarter of the scene and classified unrelated terrain as
    // bright shallow water in rectangular blocks.  Normalized screen UV is
    // independent of the sampled texture's resolution.
    const float2 renderExtent = max(camera.contactShadowConfig.xy, float2(1.0, 1.0));
    return clamp(screenPosition.xy / renderExtent, float2(0.0, 0.0), float2(1.0, 1.0));
}

float3 viewRayFromUv(float2 uv)
{
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6) projX = (projX < 0.0) ? -1e-6 : 1e-6;
    if (abs(projY) < 1e-6) projY = (projY < 0.0) ? -1e-6 : 1e-6;

    const float2 ndc = (uv * 2.0) - 1.0;
    return normalize(float3(ndc.x / projX, ndc.y / projY, -1.0));
}

SceneWaterSample sampleSceneWater(float4 screenPosition, float3 waterWorldPosition, float3 normal)
{
    SceneWaterSample result;
    result.valid = 0.0;
    result.thickness = 280.0;
    result.contact = 0.0;
    result.depthEdge = 0.0;
    result.bottomTint = float3(0.22, 0.22, 0.17);

    uint texW = 0u;
    uint texH = 0u;
    normalDepthTexture.GetDimensions(texW, texH);
    const float2 texelSize = 1.0 / float2(float(max(texW, 1u)), float(max(texH, 1u)));
    const float2 renderTexel = 1.0 / max(camera.contactShadowConfig.xy, float2(1.0, 1.0));
    const float2 baseUv = waterScreenUv(screenPosition);
    const float2 refractedUv = clamp(
        baseUv + (normal.xz * renderTexel * 7.0),
        float2(0.0, 0.0),
        float2(1.0, 1.0));
    const float4 sceneNormalDepth = normalDepthTexture.SampleLevel(refractedUv, 0.0);
    const float sceneViewDepth = sceneNormalDepth.w;
    if (sceneViewDepth <= 0.0001)
    {
        return result;
    }

    const float waterViewDepth = max(-mul(camera.view, float4(waterWorldPosition, 1.0)).z, 0.0);
    const float depthBehindWater = sceneViewDepth - waterViewDepth;
    if (depthBehindWater <= 0.02)
    {
        return result;
    }

    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewRay = viewRayFromUv(refractedUv);
    const float3x3 invViewRotation = transpose((float3x3)camera.view);
    const float3 worldViewDir = normalize(mul(invViewRotation, viewRay));
    const float travelDistance = sceneViewDepth / max(-viewRay.z, 1e-4);
    const float3 sceneWorldPosition = cameraWorld + (worldViewDir * travelDistance);
    const float bottomBelowSurface = step(sceneWorldPosition.y, waterWorldPosition.y + 0.35);
    if (bottomBelowSurface <= 0.0)
    {
        return result;
    }

    const float3 sceneViewNormal = decodeViewNormal(sceneNormalDepth.xyz);
    const float3 sceneWorldNormal = normalize(mul(invViewRotation, sceneViewNormal));
    const float floorFactor = saturate(sceneWorldNormal.y * 1.35);
    const float sunFacing = saturate(dot(
        normalize(float3(-worldViewDir.x, 0.45, -worldViewDir.z)),
        -normalize(camera.sunDirectionIntensity.xyz)));
    const float3 sandTint = lerp(
        float3(0.44, 0.37, 0.24),
        float3(0.68, 0.58, 0.36),
        sunFacing);
    const float3 rockTint = lerp(
        float3(0.20, 0.22, 0.18),
        float3(0.36, 0.38, 0.28),
        sunFacing);
    const float3 wallTint = lerp(
        float3(0.14, 0.15, 0.13),
        float3(0.30, 0.28, 0.20),
        sunFacing);
    // This is the path the transmitted light travels through water, not merely
    // the vertical column.  It is the distance Beer-Lambert attenuation needs.
    const float depthThickness = length(sceneWorldPosition - waterWorldPosition);
    const float sedimentNoise =
        (stableWaterNoise(sceneWorldPosition.xz * 0.035) * 0.65) +
        (stableWaterNoise(sceneWorldPosition.zx * 0.083) * 0.35);

    result.valid = 1.0;
    // Do not clamp a real deep column to the old 160-unit artistic ceiling:
    // that ceiling also capped Beer-Lambert absorption and let a bright HDR
    // riverbed leak through deep water as a pale cyan/green sheet.  The upper
    // bound is only a finite-value guard for reconstruction mistakes.
    result.thickness = clamp(depthThickness, 0.25, 2048.0);
    result.contact = saturate(1.0 - smoothstep(2.0, 18.0, result.thickness));
    // A four-neighbour depth discontinuity distinguishes a bank/rock contact
    // from uniformly shallow water.  The sampler is nearest-filtered, so these
    // values never invent a depth between a foreground plank and the riverbed.
    float maxDepthDelta = 0.0;
    const float centerDepth = sceneViewDepth;
    maxDepthDelta = max(maxDepthDelta, abs(normalDepthTexture.SampleLevel(baseUv + float2(texelSize.x, 0.0), 0.0).w - centerDepth));
    maxDepthDelta = max(maxDepthDelta, abs(normalDepthTexture.SampleLevel(baseUv - float2(texelSize.x, 0.0), 0.0).w - centerDepth));
    maxDepthDelta = max(maxDepthDelta, abs(normalDepthTexture.SampleLevel(baseUv + float2(0.0, texelSize.y), 0.0).w - centerDepth));
    maxDepthDelta = max(maxDepthDelta, abs(normalDepthTexture.SampleLevel(baseUv - float2(0.0, texelSize.y), 0.0).w - centerDepth));
    result.depthEdge = saturate(maxDepthDelta / max(8.0, centerDepth * 0.012));
    result.bottomTint = lerp(wallTint, lerp(sandTint, rockTint, 0.22 + (0.24 * sedimentNoise)), floorFactor);
    result.bottomTint *= 0.96 + (0.16 * sedimentNoise);
    return result;
}

struct ScreenRefractionSample
{
    float3 color;
    float weight;
    float distortionPixels;
};

ScreenRefractionSample sampleAnimatedSceneRefraction(
    float2 baseUv,
    float3 waterWorldPosition,
    float3 normal,
    float sceneThickness,
    float shallowFactor,
    float bottomValid,
    float overheadView,
    float contactFactor,
    float3 shallowWater,
    float3 fallbackColor)
{
    ScreenRefractionSample result;
    result.color = fallbackColor;
    result.weight = 0.0;
    result.distortionPixels = 0.0;

    uint texW = 0u;
    uint texH = 0u;
    opaqueSceneColorTexture.GetDimensions(texW, texH);
    const float2 texelSize = 1.0 / float2(float(max(texW, 1u)), float(max(texH, 1u)));
    const float refractionStrength = max(camera.dofConfig2.y, 0.0);
    const float refractionDistortionPixels = max(camera.dofConfig2.z, 0.0);
    if (refractionStrength <= 0.0001)
    {
        return result;
    }
    if (bottomValid < 0.5)
    {
        // No opaque depth below this piece of the full-cell water plane. The
        // copied HDR image contains clear colour or unrelated dry geometry at
        // this pixel; treating it as a riverbed produces white wedges and
        // stretched trees. Keep the physically dark deep-water fallback.
        return result;
    }

    const float depthFade = lerp(0.24, 1.0, 1.0 - smoothstep(44.0, 180.0, sceneThickness));
    const float shorelineDamping = lerp(1.0, 0.42, contactFactor);
    const float viewFade = 0.48 + (0.52 * overheadView);
    const float2 refractiveSlope = normal.xz / max(normal.y, 0.35);
    const float distortionPixels =
        refractionDistortionPixels *
        lerp(0.52, 1.0, shallowFactor) *
        depthFade *
        shorelineDamping *
        viewFade;
    result.distortionPixels = distortionPixels;
    float2 refractedUv = clamp(
        baseUv + (refractiveSlope * texelSize * distortionPixels),
        float2(0.0, 0.0),
        float2(1.0, 1.0));
    // A screen-space offset near a shoreline can land on a tree, bank, or the
    // sky. Sampling that colour as transmission puts dry geometry underneath
    // the water as a large pale smear. Accept the displaced sample only when
    // its depth reconstructs to a point both behind and below the water plane;
    // otherwise use the undistorted pixel, whose opaque surface is the actual
    // riverbed along this view ray.
    const float candidateViewDepth = normalDepthTexture.SampleLevel(refractedUv, 0.0).w;
    const float waterViewDepth = max(-mul(camera.view, float4(waterWorldPosition, 1.0)).z, 0.0);
    bool validDisplacedBottom = candidateViewDepth > waterViewDepth + 0.02;
    if (validDisplacedBottom)
    {
        const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
        const float3 candidateViewRay = viewRayFromUv(refractedUv);
        const float3x3 invViewRotation = transpose((float3x3)camera.view);
        const float3 candidateWorldDir = normalize(mul(invViewRotation, candidateViewRay));
        const float candidateTravel = candidateViewDepth / max(-candidateViewRay.z, 1e-4);
        const float3 candidateWorld = cameraWorld + (candidateWorldDir * candidateTravel);
        validDisplacedBottom = candidateWorld.y <= waterWorldPosition.y + 0.35;
    }
    if (!validDisplacedBottom)
    {
        refractedUv = baseUv;
    }
    const float3 opaqueSceneColor = opaqueSceneColorTexture.SampleLevel(refractedUv, 0.0).xyz;
    const float3 attenuation = exp(-kWaterAbsorption * max(sceneThickness, 0.0));
    const float3 volumeScatter = lerp(kWaterDeepTint, shallowWater, shallowFactor);
    const float3 refractedSceneColor =
        (opaqueSceneColor * attenuation * lerp(0.90, 1.02, shallowFactor)) +
        (volumeScatter * (float3(1.0, 1.0, 1.0) - attenuation) * 0.38);
    const float sceneWeight =
        refractionStrength *
        0.78 *
        viewFade *
        depthFade *
        lerp(0.58, 1.0, shallowFactor) *
        lerp(1.0, 0.72, contactFactor);
    result.weight = saturate(sceneWeight);
    result.color = lerp(fallbackColor, refractedSceneColor, result.weight);
    return result;
}

struct ScreenReflectionSample
{
    float3 color;
    float weight;
};

// A short screen-space reflection closes the largest gap between the raster
// and RT water paths: without it, nearby trees, rocks and bridges disappear
// from the river and only the analytic sky remains.  The opaque colour copy is
// already made for refraction and normalDepthTexture is already sampled by the
// water pass, so this adds no descriptor or attachment lifetime.  Steps grow
// geometrically in Bethesda units; a crossing of the opaque depth surface is
// accepted only inside a distance-scaled thickness to avoid reflecting objects
// hidden far behind the first visible bank.
ScreenReflectionSample sampleScreenReflection(
    float3 waterWorldPosition,
    float3 normal,
    float3 reflectionDir,
    float3 fallbackColor)
{
    ScreenReflectionSample result;
    result.color = fallbackColor;
    result.weight = 0.0;

    // A steep reflection ray sees sky, not a bank. Avoid spending ten depth
    // reads per pixel on the large overhead parts of lakes and rivers.
    if (reflectionDir.y > 0.48)
    {
        return result;
    }

    const float3 origin = waterWorldPosition + (normal * 0.65);
    float previousDelta = -1.0;
    [unroll]
    for (int stepIndex = 0; stepIndex < 10; ++stepIndex)
    {
        const float fi = float(stepIndex);
        const float rayDistance = 14.0 * pow(1.58, fi);
        const float3 sampleWorld = origin + (reflectionDir * rayDistance);
        const float4 sampleView = mul(camera.view, float4(sampleWorld, 1.0));
        const float rayDepth = -sampleView.z;
        if (rayDepth <= 0.01)
        {
            continue;
        }

        const float4 sampleClip = mul(camera.proj, sampleView);
        if (sampleClip.w <= 0.0001)
        {
            continue;
        }
        const float2 sampleUv = (sampleClip.xy / sampleClip.w) * 0.5 + 0.5;
        if (any(sampleUv <= float2(0.002, 0.002)) || any(sampleUv >= float2(0.998, 0.998)))
        {
            continue;
        }

        const float sceneDepth = normalDepthTexture.SampleLevel(sampleUv, 0.0).w;
        if (sceneDepth <= 0.0001)
        {
            previousDelta = -1.0;
            continue;
        }

        const float depthDelta = rayDepth - sceneDepth;
        const float hitThickness = 3.0 + (rayDistance * 0.035);
        const bool crossedSurface = depthDelta >= 0.0 && previousDelta < 0.0;
        if (crossedSurface && depthDelta <= hitThickness)
        {
            const float2 edgeDistance = min(sampleUv, 1.0 - sampleUv);
            const float edgeFade = smoothstep(0.01, 0.10, min(edgeDistance.x, edgeDistance.y));
            const float distanceFade = exp(-rayDistance * 0.0015);
            const float normalConfidence = smoothstep(-0.10, 0.16, reflectionDir.y);
            result.color = opaqueSceneColorTexture.SampleLevel(sampleUv, 0.0).xyz;
            result.weight = saturate(edgeFade * distanceFade * normalConfidence * 0.72);
            return result;
        }
        previousDelta = depthDelta;
    }
    return result;
}

float waterCausticPattern(float3 worldPosition, float depthDistance)
{
    const float time = camera.skyConfig1.z * max(camera.waterConfig.x, 0.01);
    const float2 p0 = (worldPosition.xz * 0.033) + float2(time * 0.030, -time * 0.021);
    const float2 p1 = (worldPosition.zx * 0.047) + float2(-time * 0.026, time * 0.018);
    const float2 p2 = (worldPosition.xz * 0.021) + float2(-time * 0.013, time * 0.016);
    const float line0 = pow(1.0 - abs(frac(p0.x + sin(p0.y) * 0.18) - 0.5) * 2.0, 8.0);
    const float line1 = pow(1.0 - abs(frac(p1.x + cos(p1.y) * 0.16) - 0.5) * 2.0, 10.0);
    const float breakup = 0.72 + (0.28 * stableWaterNoise(p2 * 2.7));
    const float depthFade = 1.0 - smoothstep(5.0, 34.0, depthDistance);
    return (line0 * 0.58 + line1 * 0.42) * breakup * depthFade;
}

WaterSurfaceSample waterSurface(
    float3 worldPosition,
    float2 uv,
    uint authoredNormalTextureSlot,
    uint authoredFlowTextureSlot)
{
    const float time = camera.skyConfig1.z;
    const float animationSpeed = max(camera.waterConfig.x, 0.01);
    const float normalStrength = max(camera.waterConfig.y, 0.01);
    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float viewDistance = length(cameraWorld - worldPosition);
    const float detailFade = 1.0 - smoothstep(220.0, 1800.0, viewDistance);
    // Treat the normal map as a moving river surface rather than three pieces
    // of ocean chop scrolling in unrelated directions.  Low-frequency noise
    // bends a stable downstream axis gradually, while the two main octaves
    // both advect along it.  The small cross-current octave breaks repetition
    // without destroying the visual direction of travel.
    const bool hasAuthoredFlow = authoredFlowTextureSlot != 0xffffffffu;
    float2 baseFlow = normalize(float2(0.80, -0.60));
    float flowSpeed = 0.35;
    if (hasAuthoredFlow)
    {
        // Skyrim stores a 64x64 ARGB flow-vector field for each exterior
        // water cell. DDS decoding puts its red/green components in linear
        // RG; Bethesda +Y becomes engine -Z.
        const float2 flowVector =
            (bindlessTextures[NonUniformResourceIndex(authoredFlowTextureSlot)].SampleLevel(uv, 0.0).xy * 2.0) - 1.0;
        const float authoredSpeed = length(flowVector);
        if (authoredSpeed > 0.05)
        {
            baseFlow = normalize(float2(flowVector.x, -flowVector.y));
            flowSpeed = saturate(authoredSpeed);
        }
    }
    const float bend = hasAuthoredFlow ? 0.0 :
        (stableWaterNoise(worldPosition.xz * 0.00042 + float2(8.3, -5.7)) - 0.5) * 0.70;
    const float cs = cos(bend);
    const float sn = sin(bend);
    const float2 flow = normalize(float2(
        (baseFlow.x * cs) - (baseFlow.y * sn),
        (baseFlow.x * sn) + (baseFlow.y * cs)));
    const float2 across = float2(-flow.y, flow.x);
    const float alongPosition = dot(worldPosition.xz, flow);
    const float acrossPosition = dot(worldPosition.xz, across);
    const float currentWave0 = sin((acrossPosition * 0.0105) + (alongPosition * 0.0022) - (time * 0.30 * animationSpeed));
    const float currentWave1 = sin((acrossPosition * 0.0210) - (alongPosition * 0.0040) - (time * 0.48 * animationSpeed));
    const float2 proceduralSlope =
        across * ((currentWave0 * 0.012) + (currentWave1 * 0.004));

    const float2 flowUv = float2(alongPosition, acrossPosition) * 0.00095;
    const float speedScale = lerp(0.65, 1.15, flowSpeed);
    const float2 sampleUv0 = flowUv + float2(-time * 0.028 * animationSpeed * speedScale, 0.0);
    const float2 sampleUv1 = (float2(alongPosition, acrossPosition) * 0.0018) +
        float2(-time * 0.045 * animationSpeed * speedScale, 0.037);
    const uint normalTextureSlot = authoredNormalTextureSlot != 0xffffffffu
        ? authoredNormalTextureSlot
        : kBindlessTextureIndexWaterNormal;
    // Tamriel's default WATR references DefaultWater.dds for all three normal
    // layers. Sample that original DXT1 vector map at the material's three
    // scales; its image view is linear rather than sRGB.
    const float2 waterNormalRg0 = bindlessTextures[NonUniformResourceIndex(normalTextureSlot)].Sample(sampleUv0).xy * 2.0 - 1.0;
    const float2 waterNormalRg1 = bindlessTextures[NonUniformResourceIndex(normalTextureSlot)].Sample(sampleUv1).xy * 2.0 - 1.0;
    const float2 flowTextureSlope =
        (waterNormalRg0 * 0.62) +
        (waterNormalRg1 * (0.30 + (0.08 * detailFade)));
    // Normal texture X/Y are in the flow/across frame. Rotate the slope back
    // into world X/Z before lighting and refraction use it.
    const float2 textureSlope =
        ((flow * flowTextureSlope.x) + (across * flowTextureSlope.y)) *
        (0.34 + (0.22 * detailFade)) * normalStrength;
    const float distanceFlattenFactor = 1.0 - smoothstep(500.0, 3500.0, viewDistance) * 0.72;
    const float2 waveSlope = (textureSlope + (proceduralSlope * normalStrength)) * distanceFlattenFactor;
    WaterSurfaceSample result;
    result.normal = normalize(float3(waveSlope.x, 1.0, waveSlope.y));
    result.flow = flow;
    result.speed = flowSpeed;
    return result;
}

#ifdef ODAI_RT_REFLECTIONS
float3 traceWaterReflection(float3 origin, float3 reflectionDir, float3 skyReflection)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = reflectionDir;
    ray.TMin = 0.05;
    ray.TMax = max(camera.shadowCascadeSplits.w * 1.8, 768.0);

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rayQuery;
    rayQuery.TraceRayInline(reflectionScene, 0, 0xFF, ray);
    while (rayQuery.Proceed()) {}

    if (rayQuery.CommittedStatus() == COMMITTED_NOTHING)
    {
        return skyReflection;
    }

    const float hitDistance = rayQuery.CommittedRayT();
    const float3 hitPosition = origin + (reflectionDir * hitDistance);
    const float heightAboveWater = max(hitPosition.y - origin.y, 0.0);
    const float buildingFactor = smoothstep(120.0, 900.0, heightAboveWater);
    const float distanceFade = exp(-hitDistance * 0.0035);
    const float skyOcclusion = saturate(1.0 - (abs(reflectionDir.y) * 1.35));
    const float sunFacing = saturate(dot(
        normalize(float3(-reflectionDir.x, 0.35, -reflectionDir.z)),
        -normalize(camera.sunDirectionIntensity.xyz)));
    const float3 terrainTint = lerp(
        float3(0.14, 0.13, 0.11),
        float3(0.34, 0.27, 0.18),
        sunFacing);
    const float3 buildingTint = lerp(
        float3(0.26, 0.22, 0.18),
        float3(0.66, 0.50, 0.32),
        sunFacing);
    const float3 reflectedSurface = lerp(terrainTint, buildingTint, buildingFactor);
    const float sceneWeight = saturate((0.30 + (skyOcclusion * 0.55)) * (0.65 + (0.35 * distanceFade)));
    return lerp(skyReflection, reflectedSurface + (skyReflection * 0.18), sceneWeight);
}

float3 traceWaterRefraction(
    float3 origin,
    float3 refractionDir,
    float3 deepWater,
    float3 shallowWater,
    out float outHitDistance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = refractionDir;
    ray.TMin = 0.02;
    ray.TMax = 224.0;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> rayQuery;
    rayQuery.TraceRayInline(reflectionScene, 0, 0xFF, ray);
    while (rayQuery.Proceed()) {}

    if (rayQuery.CommittedStatus() == COMMITTED_NOTHING)
    {
        outHitDistance = ray.TMax;
        return deepWater * 0.88;
    }

    const float hitDistance = rayQuery.CommittedRayT();
    const float3 hitPosition = origin + (refractionDir * hitDistance);
    outHitDistance = hitDistance;
    const float depthFactor = saturate(hitDistance / 150.0);
    const float shallowFactor = 1.0 - depthFactor;
    const float refractionDecay = max(camera.waterConfig.w, 0.01);
    const float murkFactor = saturate(hitDistance / max(34.0 / refractionDecay, 1.0));
    const float riverbedHeightFactor = saturate((hitPosition.y + 256.0) / 768.0);
    const float riverbedSlopeFactor = saturate(abs(refractionDir.y) * 1.35);
    const float sunFacing = saturate(dot(
        normalize(float3(-refractionDir.x, 0.55, -refractionDir.z)),
        -normalize(camera.sunDirectionIntensity.xyz)));
    const float3 siltTint = lerp(
        float3(0.42, 0.35, 0.22),
        float3(0.62, 0.53, 0.33),
        riverbedHeightFactor);
    const float3 rockTint = lerp(
        float3(0.20, 0.22, 0.18),
        float3(0.35, 0.38, 0.28),
        sunFacing);
    const float3 hitSurfaceTint = lerp(siltTint, rockTint, riverbedSlopeFactor * 0.48);
    const float caustics = waterCausticPattern(hitPosition, hitDistance) * (0.20 + (0.42 * sunFacing));
    const float3 bottomTint = lerp((hitSurfaceTint * (1.06 + caustics)), deepWater * 0.22, depthFactor);
    const float3 attenuation = exp(-kWaterAbsorption * (hitDistance * refractionDecay * 0.42));
    const float3 transmittedColor =
        (bottomTint * attenuation) +
        (lerp(deepWater, shallowWater, shallowFactor) *
            (float3(1.0, 1.0, 1.0) - attenuation) * 0.36);
    const float3 murkyVolumeColor = lerp(shallowWater * 0.42, deepWater * 0.92, murkFactor);
    return lerp(transmittedColor, murkyVolumeColor, saturate(murkFactor * 0.22));
}
#endif

float3 largeScaleWaterColor(float3 worldPosition, float3 shallowWater, float3 deepWater)
{
    const float v0 = stableWaterNoise(worldPosition.xz * 0.00110);
    const float v1 = stableWaterNoise(worldPosition.zx * 0.00175 + float2(17.3, -8.6));
    const float v2 = stableWaterNoise(worldPosition.xz * 0.00062 + float2(-5.1, 22.4));
    const float variation = (v0 * 0.50) + (v1 * 0.32) + (v2 * 0.18);
    const float3 midTone = lerp(deepWater * 1.45, shallowWater * 0.85, variation);
    return lerp(deepWater, midTone, variation * 0.60);
}

float4 main(FragmentInput input) : SV_Target0
{
    if (chunkPc.cascadeData.z > 0.5)
    {
        return float4(0.02, 0.48, 1.0, 1.0);
    }

    const float3 cameraWorld = camera.shadowVoxelGridSize.xyz;
    const float3 viewDir = normalize(cameraWorld - input.inWorldPosition);
    const WaterSurfaceSample surface = waterSurface(
        input.inWorldPosition,
        input.inUv,
        input.inNormalTextureSlot,
        input.inFlowTextureSlot);
    const float3 normal = surface.normal;
    const float waterDebugMode = camera.dofConfig2.w;
    if (waterDebugMode > 0.5 && waterDebugMode < 1.5)
    {
        return float4((normal * 0.5) + 0.5, 1.0);
    }
    if (waterDebugMode > 7.5 && waterDebugMode < 8.5)
    {
        return float4((surface.flow * 0.5) + 0.5, surface.speed, 1.0);
    }

    const float3 reflectionDir = normalize(reflect(-viewDir, normal));
    const float nDotV = saturate(dot(normal, viewDir));
    const float rippleSlope = saturate(length(normal.xz) * 2.25);
    const float overheadView = pow(nDotV, 2.2);
    const float fresnelBase = 0.020;
    const float fresnel = fresnelBase + ((1.0 - fresnelBase) * pow(1.0 - nDotV, 5.0));
    const float fresnelReflect = fresnel;
    if (waterDebugMode > 5.5 && waterDebugMode < 6.5)
    {
        return float4(fresnelReflect.xxx, 1.0);
    }

    const float reflectionStrength = max(camera.waterConfig.z, 0.01);
    const float3 refractionDir = refract(-viewDir, normal, kAirToWaterEta);

    const SceneWaterSample sceneWater = sampleSceneWater(input.inPosition, input.inWorldPosition, normal);
    const float fallbackDepthHint = stableWaterNoise((input.inWorldPosition.xz * 0.010) + float2(17.0, -9.0));
    // Missing depth must mean "unknown/deep", not "invent a bright shallow".
    // The old 10..42 fallback made an entire cell flash turquoise whenever
    // normal-depth was absent or rejected at a silhouette.
    const float fallbackThickness = lerp(240.0, 360.0, fallbackDepthHint);
    const float sceneThickness = lerp(fallbackThickness, sceneWater.thickness, sceneWater.valid);
    const float shoreFactor = lerp(
        0.035 + (0.025 * fallbackDepthHint),
        1.0 - smoothstep(3.0, 34.0, sceneWater.thickness),
        sceneWater.valid);
    const float3 shallowWater = kWaterShallowTint;
    const float3 deepWater = kWaterDeepTint;
    float refractedHitDistance = 18.0;
    float rtDepthFactor = 0.0;
    float rtShallowFactor = shoreFactor;
    const float3 largeScaleColor = largeScaleWaterColor(input.inWorldPosition, shallowWater, deepWater);
    const float3 waterBody = lerp(lerp(deepWater, largeScaleColor, 0.35), shallowWater, shoreFactor * 0.45);

    const float3 skyReflection = proceduralWaterReflectionSky(reflectionDir);
    // The authored sky is HDR and is tonemapped after water. Feeding it into
    // the material at full radiance made even a ten-percent Fresnel lobe turn
    // the river white. A water reflection loses energy; screen/RT land hits
    // below are still allowed to replace this dimmed sky with scene colour.
    const float3 reflectedSkySurface = skyReflection * 0.52;
    float3 reflectionColor = reflectedSkySurface;
    const ScreenReflectionSample screenReflection = sampleScreenReflection(
        input.inWorldPosition,
        normal,
        reflectionDir,
        reflectedSkySurface);
#ifdef ODAI_RT_REFLECTIONS
    reflectionColor = traceWaterReflection(
        input.inWorldPosition + (normal * 0.08), reflectionDir, reflectedSkySurface);
#endif
    // The screen hit carries the real shaded material colour; the inline RT
    // query currently knows only that geometry was hit and uses a synthetic
    // terrain/building tint.  Prefer a confident screen hit in either variant.
    reflectionColor = lerp(reflectionColor, screenReflection.color, screenReflection.weight);
    // The planar target contains the actual authored WTHR sky/clouds and the
    // shaded terrain/statics seen by a camera mirrored across the selected
    // water plane. It therefore owns reflection whenever this patch belongs to
    // that plane; SSR remains useful as a local high-frequency correction.
    const float planarPlaneDistance = abs(input.inWorldPosition.y - camera.waterReflectionConfig.x);
    const float planarPlaneWeight = camera.waterReflectionConfig.y *
        (1.0 - smoothstep(0.50, 8.0, planarPlaneDistance));
    const float2 reflectionTexel = 1.0 / max(camera.contactShadowConfig.xy, float2(1.0, 1.0));
    const float2 planarUv = clamp(
        (input.inPosition.xy * reflectionTexel) +
            (normal.xz * reflectionTexel * lerp(3.0, 11.0, rippleSlope)),
        float2(0.001, 0.001),
        float2(0.999, 0.999));
    const float4 planarReflectionSample =
        planarReflectionTexture.SampleLevel(planarUv, 0.0);
    const float3 planarReflection = planarReflectionSample.rgb;
    const float planarHistoryConfidence = planarReflectionSample.a;
    reflectionColor = lerp(reflectionColor, planarReflection, planarPlaneWeight * 0.96);
    if (waterDebugMode > 4.5 && waterDebugMode < 5.5)
    {
        return float4(reflectionColor, 1.0);
    }
    if (waterDebugMode > 6.5 && waterDebugMode < 7.5)
    {
        return float4(
            planarHistoryConfidence,
            1.0 - planarHistoryConfidence,
            planarPlaneWeight,
            1.0);
    }

    float3 refractionColor = waterBody;
#ifdef ODAI_RT_REFLECTIONS
    if (dot(refractionDir, refractionDir) > 0.0001)
    {
        refractionColor = traceWaterRefraction(
            input.inWorldPosition - float3(0.0, 0.03, 0.0),
            refractionDir,
            deepWater,
            shallowWater,
            refractedHitDistance);
        rtDepthFactor = saturate(refractedHitDistance / 14.0);
        rtShallowFactor = 1.0 - rtDepthFactor;
    }
#else
    refractedHitDistance = sceneThickness;
    rtDepthFactor = saturate(refractedHitDistance / 34.0);
    rtShallowFactor = 1.0 - smoothstep(4.0, 30.0, refractedHitDistance);
    const float refractionDecay = max(camera.waterConfig.w, 0.01);
    const float3 attenuation = exp(-kWaterAbsorption * (refractedHitDistance * refractionDecay));
    const float3 fallbackBottomTint = lerp(float3(0.48, 0.42, 0.27), deepWater * 0.25, rtDepthFactor);
    const float3 bottomTint = lerp(fallbackBottomTint, sceneWater.bottomTint, sceneWater.valid);
    const float murkFactor = smoothstep(9.0, 62.0, refractedHitDistance * refractionDecay);
    const float caustics = waterCausticPattern(
        input.inWorldPosition - float3(0.0, min(refractedHitDistance, 32.0), 0.0),
        refractedHitDistance);
    const float3 transmittedColor =
        (bottomTint * (1.0 + (caustics * 0.12 * rtShallowFactor)) * attenuation) +
        (waterBody * (float3(1.0, 1.0, 1.0) - attenuation) * 0.36);
    const float3 volumeColor = lerp(shallowWater * 0.42, deepWater * 0.92, smoothstep(0.12, 1.0, rtDepthFactor));
    refractionColor = lerp(transmittedColor, volumeColor, saturate(murkFactor * 0.22));
#endif
    if (waterDebugMode > 1.5 && waterDebugMode < 2.5)
    {
        return float4(
            sceneWater.valid,
            saturate(refractedHitDistance / 120.0),
            sceneWater.contact,
            1.0);
    }

    const float2 baseScreenUv = waterScreenUv(input.inPosition);
    const ScreenRefractionSample screenRefraction = sampleAnimatedSceneRefraction(
        baseScreenUv,
        input.inWorldPosition,
        normal,
        refractedHitDistance,
        rtShallowFactor,
        sceneWater.valid,
        overheadView,
        sceneWater.contact,
        shallowWater,
        refractionColor);
    refractionColor = screenRefraction.color;
    if (waterDebugMode > 2.5 && waterDebugMode < 3.5)
    {
        return float4(screenRefraction.color, 1.0);
    }
    if (waterDebugMode > 3.5 && waterDebugMode < 4.5)
    {
        const float warpViz = saturate((screenRefraction.distortionPixels * length(normal.xz)) / 18.0);
        return float4(warpViz, screenRefraction.weight, 1.0 - warpViz, 1.0);
    }

    const float bottomMottle0 = stableWaterNoise((input.inWorldPosition.xz * 0.028) + float2(13.7, 4.1));
    const float bottomMottle1 = stableWaterNoise((input.inWorldPosition.zx * 0.061) + float2(-7.3, 18.9));
    const float bottomMottle = (bottomMottle0 * 0.62) + (bottomMottle1 * 0.38);
    const float3 overheadBottomTint = lerp(float3(0.34, 0.31, 0.22), float3(0.025, 0.070, 0.065), rtDepthFactor);
    const float screenRefractionPresence = screenRefraction.weight * (0.55 + (0.45 * rtShallowFactor));
    const float overheadClarity =
        overheadView *
        (0.18 + (0.14 * rtShallowFactor)) *
        lerp(1.0, 0.28, screenRefractionPresence);
    refractionColor = lerp(
        refractionColor,
        (refractionColor * (0.96 + (0.08 * bottomMottle))) + (overheadBottomTint * (0.025 + (0.020 * bottomMottle))),
        overheadClarity);

    const float3 sunDir = -normalize(camera.sunDirectionIntensity.xyz);
    const float3 halfVector = normalize(viewDir + sunDir);
    const float ndotl = saturate(dot(normal, sunDir));
    const float ndoth = saturate(dot(normal, halfVector));
    const float vdoth = saturate(dot(viewDir, halfVector));
    const float sunGlint = pow(saturate(dot(reflectionDir, sunDir)), 64.0);
    const float windNoise = stableWaterNoise((input.inWorldPosition.xz * 0.014) + (camera.skyConfig1.z * 0.010));
    const float broadSpecular = pow(ndoth, 10.0) * (0.105 + (0.22 * fresnelReflect));
    const float tightSpecular = pow(ndoth, lerp(58.0, 96.0, windNoise)) * (0.070 + (0.22 * fresnelReflect));
    const float directionalSpecular = (broadSpecular + tightSpecular) * (0.38 + (0.62 * ndotl)) * (0.55 + (0.45 * vdoth));
    const float3 sunScatterTint = shallowWater * lerp(0.025, 0.085, ndotl);
    const float grazingFactor = saturate(1.0 - nDotV);
    const float angleReflectionFactor = saturate(pow(grazingFactor, 1.45));
    const float bottomDepthFactor = saturate(refractedHitDistance / 14.0);
    // Reflection is an interface property. The previous formula increased it
    // with water DEPTH, so a deep river became a sky mirror even when viewed
    // from above—the opposite of the riverbed-readable Skyrim look. Depth now
    // affects only transmission/absorption; Schlick Fresnel and ripple slope
    // decide reflection.
    const float reflectionWeight = saturate(fresnelReflect * reflectionStrength);
    const float3 reflectionBoost = lerp(float3(0.96, 0.98, 0.96), float3(1.08, 1.11, 1.10), angleReflectionFactor);
    const float finalReflectionMix = reflectionWeight;
    // One optical split: transmitted scene below the surface versus reflected
    // sky/land above it. screenRefraction.color already contains the fallback
    // refraction in proportion to its confidence; mixing that fallback over
    // the result a second time erased both the real riverbed and reflections.
    float3 litColor = lerp(screenRefraction.color, reflectionColor * reflectionBoost, finalReflectionMix);
    const float contactFilmMask = sceneWater.valid * sceneWater.contact *
        (0.24 + (0.76 * stableWaterNoise((input.inWorldPosition.xz * 0.075) + float2(camera.skyConfig1.z * 0.012, 3.1))));
    const float energeticContact = saturate(max(
        (surface.speed - 0.35) * 2.2,
        sceneWater.depthEdge));
    const float foamMask = sceneWater.valid * sceneWater.contact * energeticContact *
        saturate((rippleSlope - 0.10) * 3.2) *
        (0.18 + (0.82 * stableWaterNoise((input.inWorldPosition.xz * 0.055) +
            (surface.flow * camera.skyConfig1.z * 0.018))));
    if (waterDebugMode > 8.5 && waterDebugMode < 9.5)
    {
        return float4(sceneWater.contact, sceneWater.depthEdge, foamMask, 1.0);
    }
    const float causticLift = waterCausticPattern(input.inWorldPosition - float3(0.0, refractedHitDistance, 0.0), refractedHitDistance);
    const float3 slopeLightDirection = normalize(float3(0.42, 0.58, 0.70));
    const float slopeLight = saturate((dot(normal, slopeLightDirection) * 0.5) + 0.5);
    const float slopeContrast = (slopeLight - 0.50) * rippleSlope;
    const float warpVisibility =
        screenRefraction.weight *
        saturate(screenRefraction.distortionPixels / 80.0) *
        saturate(length(normal.xz) * 3.0);
    litColor *= 0.98 + (0.16 * ndotl) + (0.07 * rippleSlope);
    litColor *= 1.0 + (slopeContrast * 0.38);
    litColor += lerp(float3(0.010, 0.028, 0.025), float3(0.055, 0.090, 0.070), slopeLight) *
        (rippleSlope * (0.012 + (0.025 * overheadView)));
    litColor += lerp(float3(0.008, 0.022, 0.020), float3(0.045, 0.080, 0.064), slopeLight) *
        (warpVisibility * (0.012 + (0.022 * overheadView)));
    litColor += shallowWater * (causticLift * (0.012 + (0.018 * ndotl)) * saturate(rtShallowFactor + 0.20));
    litColor += lerp(float3(0.38, 0.52, 0.47), float3(0.70, 0.78, 0.71), ndotl) * (foamMask * 0.055);
    litColor += float3(0.08, 0.17, 0.14) * (contactFilmMask * 0.018);
    litColor += camera.sunColorShadow.xyz * (sunGlint * (0.070 + (fresnelReflect * 0.18)));
    litColor += camera.sunColorShadow.xyz * directionalSpecular;
    litColor += sunScatterTint * 0.45;
    litColor += lerp(shallowWater * 0.022, deepWater * 0.012, bottomDepthFactor);

    // Screen refraction can look through a shallow river into a sunlit bank or
    // the bright sky at the edge of the copied opaque buffer. Passing that RGB
    // through almost unchanged made Skyrim's White River a flat white polygon:
    // the geometry and XCLW height were correct, but there was no remaining
    // optical cue that the surface was water. At overhead angles transmission,
    // not reflection, dominates; apply a modest wavelength-selective filter
    // there while leaving grazing sky reflections untouched.
    const float transmissionTintAmount =
        overheadView *
        (1.0 - angleReflectionFactor) *
        (1.0 - finalReflectionMix) *
        lerp(0.12, 0.22, rtShallowFactor);
    const float3 riverTransmission = lerp(
        float3(0.82, 0.84, 0.82),
        float3(0.78, 0.86, 0.82),
        rtShallowFactor);
    litColor *= lerp(float3(1.0, 1.0, 1.0), riverTransmission, transmissionTintAmount);
    litColor += waterBody * (0.012 * transmissionTintAmount);

    // Shore treatment is tied strictly to a reconstructed contact.  A shallow
    // fallback is useful for colour, but it cannot tell a bank from the middle
    // of a river and previously flooded whole patches with opaque white foam.
    const float shoreBand = sceneWater.valid * sceneWater.contact;
    const float shoreWashA = stableWaterNoise((input.inWorldPosition.xz * 0.085) + float2(camera.skyConfig1.z * 0.020, 0.0));
    const float shoreWashB = stableWaterNoise((input.inWorldPosition.zx * 0.150) - float2(0.0, camera.skyConfig1.z * 0.014));
    const float3 bankShallows = float3(0.050, 0.105, 0.080);
    const float3 shoreFoamColor = float3(0.58, 0.66, 0.61);
    litColor *= 1.0 - (shoreBand * 0.08);
    litColor = lerp(litColor, bankShallows, shoreBand * shoreBand * 0.10);
    const float foamCrest = saturate((shoreBand - 0.64) * 2.8) * saturate(shoreWashA - 0.40) * energeticContact;
    const float foamLace = saturate((shoreBand - 0.48) * 1.8) * saturate(shoreWashB - 0.68) * 0.22 * energeticContact;
    litColor = lerp(litColor, shoreFoamColor, saturate(foamMask + foamCrest + foamLace) * 0.22);

    // Auto-exposure is keyed by the terrain and can lift even a numerically
    // bounded river into pastel cyan. Compress broad overhead transmission
    // toward a cold, rock-fed neutral while retaining its normal/refraction
    // detail; grazing reflections remain essentially untouched.
    const float nonGrazingSurface = 1.0 - smoothstep(0.70, 0.98, angleReflectionFactor);
    const float overheadRiverGrade =
        (0.16 + (0.34 * overheadView)) *
        nonGrazingSurface *
        lerp(0.52, 0.32, finalReflectionMix);
    const float3 exposureStableRiver =
        (litColor * float3(0.72, 0.74, 0.69)) +
        lerp(float3(0.004, 0.012, 0.013), float3(0.012, 0.028, 0.022), rtShallowFactor);
    litColor = lerp(litColor, exposureStableRiver, overheadRiverGrade);

    // Keep the remaining HDR range finite before tonemapping. Grazing
    // reflections and sun glints retain a higher ceiling than transmission.
    const float waterLuminance = dot(litColor, float3(0.2126, 0.7152, 0.0722));
    const float waterLuminanceCeiling = lerp(0.34, 1.40, angleReflectionFactor);
    litColor *= min(1.0, waterLuminanceCeiling / max(waterLuminance, 0.001));

    const float3 cameraWorldPos = camera.shadowVoxelGridSize.xyz;
    const float waterViewDist = length(cameraWorldPos - input.inWorldPosition);
    litColor = applyAtmosphericFog(litColor, input.inWorldPosition, waterViewDist, normalize(camera.sunDirectionIntensity.xyz));
    return float4(litColor, 1.0);
}
