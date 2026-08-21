import camera_uniform;
import chunk_push_constants;

// Same bindless table the imported-geometry shaders use; the sky pipeline binds
// set 1 already, so cloud layers cost no new descriptor.
[[vk::binding(0, 1)]]
Sampler2D<float4> bindlessTextures[];

float saturate01(float value)
{
    return clamp(value, 0.0, 1.0);
}


float smoothStep(float edge0, float edge1, float x)
{
    const float t = saturate01((x - edge0) / max(edge1 - edge0, 1e-6));
    return t * t * (3.0 - (2.0 * t));
}

// TWO GENERATIONS, TWO PROJECTIONS, AND THEY ARE NOT INTERCHANGEABLE.
//
// Fallout and Oblivion cloud layers are SKY DOME MAPS: each texture is a
// fisheye of the whole sky, zenith at the centre of the image and horizon at
// the rim of an inscribed circle. Nevada Skies' WesternSky1.dds is unmistakable
// about it -- a bright cloud disc with sun rays radiating into the dark corners.
//
// That was first written as a plane projection, dir.xz/(dir.y+k), with wrap
// addressing. It maps the visible sky into roughly [-0.76, 0.76] of UV space,
// so it crosses zero and the sampler wraps -- which tiles the entire circular
// image and puts a hard seam where it crosses. The symptom is a vertical seam
// with the sky visibly repeating from it, and no amount of tuning the scale
// fixes it, because tiling a fisheye is meaningless. The right mapping sends
// the hemisphere onto the unit disc exactly once: angle from the zenith becomes
// radius, compass direction becomes angle, and every UV lands inside one copy.
//
// SKYRIM'S ARE THE OPPOSITE. Sky\SkyrimCloudsLower01.dds has no radial
// structure at all: it is a seamlessly tileable cloud field whose shape is in
// its alpha, meant to be repeated across a dome many times over. Drawn through
// the fisheye mapping it is one enormous smeared copy of a cloud field, which
// looks like a rendering bug rather than like the wrong projection.
//
// The dome projection for a tiling sheet is dir.xz / (dir.y + k). k is what
// keeps it finite: without it the horizon diverges into radial streaks, and at
// k = 0.35 the UV radius tops out around 2.9 sheets, which is a plausible
// perspective compression rather than an asymptote.
float3 applyWeatherClouds(float3 sky, float3 dir, float time)
{
    if (dir.y <= 0.0)
    {
        return sky;  // below the horizon; the ground blend owns this
    }

    // Angle from the zenith, normalised so 0 is straight up and 1 is the
    // horizon. acos rather than a cheaper approximation because the horizon is
    // where the gradient is steepest and where any error shows.
    const float radius = acos(clamp(dir.y, 0.0, 1.0)) * (2.0 / 3.14159265);
    const float2 compass = dir.xz;
    const float compassLength = max(length(compass), 1e-5);
    const float2 compassDir = compass / compassLength;

    // The dome-plane coordinate the tiling mapping repeats across, before its
    // per-layer tiling count.
    const float2 planeUv = dir.xz / (dir.y + 0.35);
    // Compass bearing, 0..1 around the sky, for the cylindrical mapping.
    const float bearing = atan2(compassDir.y, compassDir.x) * (0.5 / 3.14159265);

    float3 result = sky;
    for (int layer = 0; layer < 4; ++layer)
    {
        const float4 tint = camera.weatherCloudTint[layer];
        const float opacity = tint.w;
        if (opacity <= 0.001)
        {
            continue;
        }
        const float4 params = camera.weatherCloudParams[layer];
        const int slot = int(params.x + 0.5);
        if (slot < 0)
        {
            continue;
        }
        const float4 band = camera.weatherCloudBand[layer];
        const int mapping = int(band.z + 0.5);

        // The elevation window this layer occupies, feathered at both ends so a
        // deck fades out rather than ending on a line. A layer that covers the
        // whole sky takes no mask at all rather than a mask that rounds to one:
        // that is every Fallout and Oblivion layer, and their horizon fade is
        // already the one below.
        const float bandLow = band.x;
        const float bandHigh = band.y;
        //
        // Each end is feathered only where it is actually an end: a band that
        // reaches the zenith must not fade OUT at the zenith, which is what a
        // symmetric mask does and it reads as a hole in the sky directly
        // overhead.
        const float feather = max(0.12 * (bandHigh - bandLow), 0.02);
        const float fadeIn = (bandLow <= 0.0) ? 1.0 : smoothStep(bandLow, bandLow + feather, dir.y);
        const float fadeOut =
            (bandHigh >= 1.0) ? 1.0 : (1.0 - smoothStep(bandHigh - feather, bandHigh, dir.y));
        const float bandMask = fadeIn * fadeOut;

        float2 uv;
        float horizonFade;
        if (mapping == 2)
        {
            // Cylindrical: bearing across, elevation up. The band's own window
            // IS the v range, so the whole texture is spent on the slice of sky
            // the art was drawn for rather than stretched over the hemisphere.
            const float elevation =
                saturate01((dir.y - bandLow) / max(bandHigh - bandLow, 1e-4));
            // No v drift: a bank of cloud drifts around the sky, not up it.
            // The bindless sky sampler wraps so the bank can repeat around the
            // compass. V must NOT wrap: elevation=0 produces exactly 1.0 and
            // would jump from the texture's bottom row to its top row on the
            // skyline, leaving a one-pixel seam that temporal reconstruction
            // makes more obvious. Keep only V inside the texture; U still wraps.
            const float bankV = clamp(1.0 - elevation, 0.002, 0.998);
            uv = float2((bearing * params.z) + (time * params.y), bankV);
            horizonFade = 1.0;  // the band mask already owns both ends
        }
        else if (mapping == 1)
        {
            // A tiling sheet drifts by translating, which is what a cloud
            // field blowing across the sky actually does.
            uv = (planeUv * params.z) + float2(time * params.y, time * params.w);
            // Compressing hardest at the horizon, where it aliases into mush.
            horizonFade = 1.0 - smoothStep(0.94, 1.0, radius);
        }
        else
        {
            // A dome map cannot scroll by translating UV -- that would slide the
            // fisheye off its own centre. It ROTATES about the zenith, which is
            // what a cloud layer drifting overhead actually looks like.
            const float angle = time * params.y;
            const float sinA = sin(angle);
            const float cosA = cos(angle);
            const float2 rotated = float2(
                (compassDir.x * cosA) - (compassDir.y * sinA),
                (compassDir.x * sinA) + (compassDir.y * cosA));

            // params.z scales the dome: 1.0 puts the horizon exactly on the
            // inscribed circle. Staying at or under 1 keeps every sample inside
            // the texture, so wrap addressing never comes into play.
            uv = float2(0.5, 0.5) + (0.5 * radius * params.z) * rotated;
            // The dome map's own rim is a hard edge, and the projection
            // compresses hardest right there.
            horizonFade = 1.0 - smoothStep(0.86, 1.0, radius);
        }
        horizonFade *= bandMask;
        const float4 sampled = bindlessTextures[slot].Sample(uv);

        // FNV cloud textures carry the cloud SHAPE in alpha; rgb is shading
        // detail, not the final colour. PNAM's per-layer tint is the colour.
        //
        // Multiplying rgb by the tint squares the darkness and is how a heavy
        // overcast rendered as pure black: the tint is already sRGB 19,23,23
        // (linear 0.0075) and the texture rgb is dark too, so the product
        // landed below what 8-bit output can represent. Use the tint as the
        // base and let the texture only modulate around it.
        const float detail = dot(sampled.rgb, float3(0.2126, 0.7152, 0.0722));
        const float3 cloudColor = tint.xyz * lerp(0.70, 1.30, saturate01(detail));
        const float coverage = saturate01(sampled.a * opacity * horizonFade);
        result = lerp(result, cloudColor, coverage);
    }
    return result;
}

float3 proceduralSky(float3 direction, float3 sunDirection)
{
    const float3 dir = normalize(direction);
    const float3 toSun = -normalize(sunDirection);
    const float rayleighStrength = max(camera.skyConfig0.x, 0.01);
    const float mieStrength = max(camera.skyConfig0.y, 0.01);
    const float mieG = clamp(camera.skyConfig0.z, 0.0, 0.98);
    const float skyExposure = max(camera.skyConfig0.w, 0.01);

    const float sunAltitude = clamp(toSun.y, -1.0, 1.0);
    const float dayFactor = smoothStep(0.05, 0.65, sunAltitude);
    const float twilightFactor = 1.0 - dayFactor;
    const float horizonBand = saturate01(1.0 - (abs(sunAltitude) / 0.35));
    const float warmAmount = twilightFactor * pow(horizonBand, 1.2);
    const float pinkAmount = warmAmount * saturate01((0.10 - sunAltitude) / 0.30);

    const float horizonT = saturate01((dir.y * 0.5) + 0.5);
    const float skyT = pow(horizonT, 0.35);

    float3 horizonRayleigh = lerp(float3(0.54, 0.70, 1.00), float3(0.74, 0.44, 0.52), warmAmount);
    float3 horizonMie = lerp(float3(1.00, 0.74, 0.42), float3(1.18, 0.54, 0.30), warmAmount);
    horizonRayleigh = lerp(horizonRayleigh, float3(0.70, 0.36, 0.68), pinkAmount * 0.70);
    horizonMie = lerp(horizonMie, float3(1.08, 0.46, 0.72), pinkAmount * 0.85);

    const float zenithWarm = twilightFactor * 0.58;
    const float3 zenithRayleigh = lerp(float3(0.06, 0.24, 0.54), float3(0.16, 0.12, 0.30), zenithWarm);
    const float3 zenithMie = lerp(float3(0.22, 0.20, 0.15), float3(0.30, 0.18, 0.24), zenithWarm);

    float3 horizonColor =
        (horizonRayleigh * rayleighStrength) +
        (horizonMie * (mieStrength * 0.58));
    float3 zenithColor =
        (zenithRayleigh * rayleighStrength) +
        (zenithMie * (mieStrength * 0.25));

    // Authored sky from a WTHR record, blended over the procedural one. Fallout
    // stores three vertical stops rather than a two-end gradient -- Horizon at
    // the skyline, SkyLower just above it, SkyUpper at the zenith -- and the
    // narrow Horizon->SkyLower band near the ground is what gives these skies
    // their look. Collapsing it into a single lerp loses exactly the part worth
    // having.
    const float weatherWeight = saturate01(camera.weatherSkyUpper.w);
    const float upAmount = saturate01(dir.y);
    const float3 authoredLowBand =
        lerp(camera.weatherHorizon.xyz, camera.weatherSkyLower.xyz,
             smoothStep(0.0, 0.22, upAmount));
    const float3 authoredSky =
        lerp(authoredLowBand, camera.weatherSkyUpper.xyz, smoothStep(0.12, 0.75, upAmount));

    // The horizon and zenith stops move too, because everything downstream --
    // the ground blend below, and any caller reading these -- would otherwise
    // mix an authored sky against a procedural horizon and band at the skyline.
    horizonColor = lerp(horizonColor, camera.weatherHorizon.xyz, weatherWeight);
    zenithColor = lerp(zenithColor, camera.weatherSkyUpper.xyz, weatherWeight);

    // Falls through to the sun disk and phase terms below rather than returning:
    // an authored sky still needs a sun in it.
    float3 sky = lerp(lerp(horizonColor, zenithColor, skyT), authoredSky, weatherWeight);

    const float sunDotRaw = clamp(dot(dir, toSun), -1.0, 1.0);
    const float sunDot = max(sunDotRaw, 0.0);
    const float3 sunColor = camera.sunColorShadow.xyz;
    const float phaseRayleigh = 0.0596831 * (1.0 + (sunDot * sunDot));
    const float phaseMie = 0.0795775 * (1.0 - (mieG * mieG)) /
        max(0.001, pow(1.0 + (mieG * mieG) - (2.0 * mieG * sunDot), 1.5));
    const float phaseBoost = (phaseRayleigh * rayleighStrength) + (phaseMie * mieStrength * 1.4);

    // Analytic sun disk in angular space with altitude-dependent haze bloom.
    // Low sun: wider and vertically flattened by near-horizon atmospheric distortion.
    const float sunAngle = acos(sunDotRaw);
    const float diskSizeScale = clamp(camera.skyConfig2.x, 0.5, 6.0);
    const float hazeFalloffScale = clamp(camera.skyConfig2.y, 0.10, 1.20);
    const float lowSunHaze =
        (1.0 - smoothStep(0.08, 0.55, sunAltitude)) *
        smoothStep(-0.18, 0.12, sunAltitude);
    const float highSunCrisp = smoothStep(0.35, 0.85, sunAltitude);

    const float radiusDay = 0.0056;
    const float radiusTwilight = 0.0108;
    const float sunRadius = lerp(radiusTwilight, radiusDay, dayFactor) * diskSizeScale;
    const float sunAaBase = max(fwidth(sunAngle), 0.00010);
    const float sunEdgeWidth = sunAaBase * lerp(2.30, 0.85, highSunCrisp);
    const float3 horizonBasisUp = (abs(toSun.y) < 0.98) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 horizonRight = normalize(cross(horizonBasisUp, toSun));
    const float3 sunUp = normalize(cross(toSun, horizonRight));
    const float2 sunPlane = float2(dot(dir, horizonRight), dot(dir, sunUp));
    const float flattenAmount = (1.0 - highSunCrisp) * smoothStep(-0.08, 0.18, sunAltitude);
    const float horizontalRadius = sunRadius * lerp(1.0, 1.55, flattenAmount);
    const float verticalRadius = sunRadius * lerp(1.0, 0.42, flattenAmount);
    const float ellipseAngle = length(float2(
        sunPlane.x / max(horizontalRadius, 1e-5),
        sunPlane.y / max(verticalRadius, 1e-5))) * sunRadius;
    const float diskMask = 1.0 - smoothStep(sunRadius - sunEdgeWidth, sunRadius + sunEdgeWidth, ellipseAngle);
    const float centerT = saturate01(1.0 - (ellipseAngle / max(sunRadius, 1e-5)));
    const float limbDarkening = 0.52 + (0.48 * pow(centerT, lerp(0.50, 0.24, highSunCrisp)));

    const float diskIntensity = max(camera.skyConfig1.x / 1150.0, 0.01);
    const float disk = diskMask * limbDarkening * (6.8 * diskIntensity) * lerp(0.95, 1.15, highSunCrisp);

    const float haloIntensity = max(camera.skyConfig1.y / 22.0, 0.01);

    // Mid/near halo component.
    const float haloRadius = sunRadius * lerp(18.0, 6.0, highSunCrisp);
    const float haloFalloff = max(haloRadius * hazeFalloffScale * lerp(1.45, 0.55, highSunCrisp), 1e-5);
    const float haloCore = exp(-sunAngle / haloFalloff);
    const float haloMask = 1.0 - smoothStep(haloRadius * lerp(0.45, 0.72, highSunCrisp), haloRadius, sunAngle);
    const float halo = haloCore * haloMask * (1.2 * haloIntensity) * (0.35 + (1.35 * lowSunHaze));

    // Broad low-sun haze bloom component.
    const float hazeRadius = sunRadius * lerp(34.0, 10.0, highSunCrisp);
    const float hazeFalloff = max(hazeRadius * hazeFalloffScale * 0.85, 1e-5);
    const float hazeCore = exp(-sunAngle / hazeFalloff);
    const float hazeMask = 1.0 - smoothStep(hazeRadius * 0.55, hazeRadius, sunAngle);
    const float hazeBloom = hazeCore * hazeMask * (1.6 * haloIntensity * lowSunHaze);

    // WTHR's Sun Glare byte scales the halo and the low-sun bloom. The DISC is
    // left alone: a weather dims the sun by putting cloud in front of it, and
    // fading the disc itself instead reads as the sun burning out.
    const float sunGlareScale = camera.weatherHorizon.w;
    sky += sunColor * (disk + ((halo + hazeBloom) * sunGlareScale)) * (1.0 + phaseBoost);

    // After the sun, so a cloud layer covers the disk instead of the disk
    // burning through it.
    sky = applyWeatherClouds(sky, dir, camera.skyConfig1.z);

    // The distant world is faded toward WTHR's far-fog colour. The sky used to
    // arrive at the same skyline using the separate Horizon stop, while its
    // clouds faded independently; the result was a bright, flat strip behind
    // otherwise convincing aerial perspective. Make the last few degrees one
    // atmosphere by attenuating the completed sky (including clouds and sun)
    // toward the same far-fog colour. Horizon remains the dominant authored
    // stop rather than being replaced outright.
    const float horizonHaze =
        weatherWeight * (1.0 - smoothStep(0.0, 0.16, max(dir.y, 0.0)));
    const float3 atmosphericHorizon =
        lerp(horizonColor, camera.weatherFog.xyz, weatherWeight * 0.68);
    sky = lerp(sky, camera.weatherFog.xyz, horizonHaze * 0.68);

    const float3 groundColor = float3(0.05, 0.06, 0.07);
    const float belowHorizon = saturate01(-dir.y);
    // Begin below the SAME atmospheric colour used above. Darkening the old
    // horizon colour to 32% immediately below y=0 made open-water and mountain
    // gaps reveal a dark lower band unrelated to either sky or fog.
    const float3 horizonGroundColor =
        lerp(horizonColor * 0.32, atmosphericHorizon, weatherWeight);
    const float groundDepth = smoothStep(0.0, 0.28, belowHorizon);
    const float3 ground =
        lerp(horizonGroundColor, groundColor, pow(groundDepth, 0.72));
    const float skyBlendLow = lerp(-0.18, -0.10, weatherWeight);
    const float skyBlendHigh = lerp(0.02, 0.015, weatherWeight);
    const float skyWeight = smoothStep(skyBlendLow, skyBlendHigh, dir.y);
    const float3 color = lerp(ground, sky, skyWeight);
    return max(color * skyExposure, float3(0.0, 0.0, 0.0));
}

struct FragmentInput
{
    float2 inNdc : TEXCOORD0;
};

float4 main(FragmentInput input) : SV_Target0
{
    float projX = camera.proj[0][0];
    float projY = camera.proj[1][1];
    if (abs(projX) < 1e-6)
    {
        projX = (projX < 0.0) ? -1e-6 : 1e-6;
    }
    if (abs(projY) < 1e-6)
    {
        projY = (projY < 0.0) ? -1e-6 : 1e-6;
    }
    const float3 viewDir = normalize(float3(input.inNdc.x / projX, input.inNdc.y / projY, -1.0));
    const float3x3 invViewRotation = transpose((float3x3)camera.view);
    float3 worldDir = normalize(mul(invViewRotation, viewDir));
    // The planar pass reflects the camera across a horizontal plane. Sky is at
    // infinity, so only the ray direction changes; the plane height cancels.
    if (chunkPc.cascadeData.x > 0.5)
    {
        worldDir.y = -worldDir.y;
    }
    // Morrowind sky textures are authored for dedicated sky NIF meshes. Keep this
    // pass as the atmospheric base; the mesh cloud layer is drawn over sky pixels.
    return float4(proceduralSky(worldDir, camera.sunDirectionIntensity.xyz), 1.0);
}
